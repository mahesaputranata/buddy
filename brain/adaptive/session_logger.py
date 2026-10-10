"""
brain/adaptive/session_logger.py

Session Logger
==============
Records every Buddy interaction to a daily JSONL file.
Runs silently during every session — no user action required.

Format is rich enough for:
  - Rule improvement (intent + confidence + source)
  - Prompt improvement (what Buddy said vs what was expected)
  - Future fine-tuning (review.corrected_response field)
  - Personal profile learning (session_recap.py reads these)

Files saved to:
  logs/sessions/session_YYYY-MM-DD.jsonl
  One file per day, one JSON object per line.

Usage:
  # Wire into handle_request() in ConversationManager:
  from brain.adaptive.session_logger import SessionLogger
  logger = SessionLogger()
  logger.log(user_input, response, intent_result, context)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

TZ = ZoneInfo("Europe/Berlin")
LOG_DIR = Path("logs/sessions")


class SessionLogger:
    """
    Logs every Buddy interaction to a daily JSONL file.
    Silent — never crashes Buddy even if file write fails.
    """

    def __init__(self, log_dir: Path = LOG_DIR):
        self._log_dir = Path(log_dir)
        self._session_id = self._make_session_id()
        self._turn = 0

    def _make_session_id(self) -> str:
        """Unique ID for this Buddy session."""
        now = datetime.now(TZ)
        return now.strftime("%Y-%m-%d-%H%M%S")

    def _log_path(self) -> Path:
        """Returns today's log file path."""
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        return self._log_dir / f"session_{today}.jsonl"

    def log(
        self,
        user_input: str,
        response: str,
        intent_result: dict,
        context: Optional[dict] = None,
        outcome: Optional[dict] = None,
        shadow: Optional[dict] = None,
    ):
        """
        Logs a single interaction.

        Args:
            user_input:    what the user said
            response:      what Buddy replied
            intent_result: the dict from classify_intent + apply_guards
                           {"intent": ..., "confidence": ..., "source": ...}
            context:       optional dict with last_topic, pending state etc
            outcome:       optional dict with sql_delta, follow_up_needed etc

        Example in handle_request():
            logger.log(
                user_input=user_input,
                response=response,
                intent_result=intent_result,
                context={
                    "last_topic":     self.memory_manager.get_last_topic(),
                    "pending_before": pending_before,
                    "pending_after":  self.memory_manager.get_pending_topic(),
                },
                outcome={
                    "follow_up_needed": self.memory_manager.get_pending_topic() is not None,
                }
            )
        """
        try:
            self._turn += 1
            self._log_dir.mkdir(parents=True, exist_ok=True)

            entry = {
                "id":         str(uuid.uuid4())[:8],
                "ts":         datetime.now(TZ).isoformat(),
                "session_id": self._session_id,
                "turn":       self._turn,

                "input":    (user_input or "").strip(),
                "response": (response or "").strip(),

                "intent":     intent_result.get("intent", "unknown"),
                "confidence": round(float(intent_result.get("confidence", 0.0)), 3),
                "source":     intent_result.get("source", "unknown"),
                "tier":       self._infer_tier(intent_result.get("source", "")),

                "context": {
                    "last_topic":     (context or {}).get("last_topic"),
                    "pending_before": (context or {}).get("pending_before"),
                    "pending_after":  (context or {}).get("pending_after"),
                },

                "outcome": {
                    "follow_up_needed": (outcome or {}).get("follow_up_needed", False),
                    "sql_delta":        (outcome or {}).get("sql_delta"),
                    "adaptive_fired":   (outcome or {}).get("adaptive_fired", False),
                },

                # Review block — filled in manually or by session_recap.py
                # corrected_intent and corrected_response become fine-tune data
                "review": {
                    "flagged":            False,
                    "issue":              None,
                    "corrected_intent":   None,
                    "corrected_response": None,
                    "note":               None,
                },
            }

            # Shadow-mode prediction from the trained intent classifier (logged only,
            # never acted on). Absent while the model is still loading / disabled.
            if shadow:
                entry["shadow"] = shadow

            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            # Never crash Buddy because of logging
            print(f"[SessionLogger] Warning: could not log interaction: {e}")

    def _infer_tier(self, source: str) -> int:
        """
        Infers which classifier tier handled this intent.
        Tier 1 = rules, Tier 2 = evidence scorer, Tier 3 = LLM
        """
        if not source:
            return 0
        s = source.lower()
        if "rule" in s or "guard" in s or "cancel" in s or "system" in s:
            return 1
        if "tier2" in s or "evidence" in s:
            return 2
        if "llm" in s or "zero_shot" in s or "fallback" in s:
            return 3
        return 0

    def get_session_id(self) -> str:
        return self._session_id

    def get_turn_count(self) -> int:
        return self._turn

    def load_today(self) -> list[dict]:
        """
        Loads all interactions from today's session log.
        Used by session_recap.py at end of session.
        """
        path = self._log_path()
        if not path.exists():
            return []

        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[SessionLogger] Could not load today's log: {e}")

        return entries

    def load_session(self, session_id: str) -> list[dict]:
        """
        Loads all interactions from a specific session ID.
        Searches across all daily log files.
        """
        if not self._log_dir.exists():
            return []

        entries = []
        for log_file in sorted(self._log_dir.glob("session_*.jsonl")):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                if entry.get("session_id") == session_id:
                                    entries.append(entry)
                            except json.JSONDecodeError:
                                continue
            except Exception:
                continue

        return entries

    def load_date(self, date_str: str) -> list[dict]:
        """
        Loads all interactions from a specific date.
        date_str: "YYYY-MM-DD"
        """
        path = self._log_dir / f"session_{date_str}.jsonl"
        if not path.exists():
            return []

        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[SessionLogger] Could not load {date_str}: {e}")

        return entries

session_logger = SessionLogger()