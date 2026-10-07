"""
brain/adaptive/session_recap.py

Session Recap Engine
====================
Runs at the end of each Buddy session (triggered by Ctrl+C shutdown).

What it does:
  1. Loads today's session log from session_logger
  2. Shows you a plain-text summary of the session
  3. Asks if you want to save it
  4. If yes — runs LLM analysis to extract facts about you
  5. Shows extracted facts for your approval (one by one)
  6. Saves approved facts to logs/user_profile.json
  7. Next session Buddy loads the profile and knows you better

Usage — wire into chat() shutdown hook:
    import signal
    from brain.adaptive.session_recap import SessionRecap

    def _on_shutdown(sig, frame):
        recap = SessionRecap()
        recap.run_end_of_session()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_shutdown)

Manual usage (run from command line to review a past session):
    python brain/adaptive/session_recap.py 2026-04-10
"""

import json
import sys
import ollama
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

TZ = ZoneInfo("Europe/Berlin")

LOG_DIR      = Path("logs/sessions")
PROFILE_PATH = Path("logs/user_profile.json")
MODEL        = "llama3.2:latest"


class SessionRecap:
    """
    End-of-session analysis and personal profile updater.
    """

    def __init__(self):
        from brain.adaptive.session_logger import SessionLogger
        self._logger = SessionLogger()

    # ── Main entry point ─────────────────────────────────────────────────────

    def run_end_of_session(self, date_str: Optional[str] = None):
        """
        Full end-of-session flow.
        Called automatically on Ctrl+C shutdown.

        Args:
            date_str: "YYYY-MM-DD" — defaults to today
        """
        print("\n" + "="*60)
        print("  BUDDY SESSION RECAP")
        print("="*60)

        # Load session
        if date_str:
            entries = self._logger.load_date(date_str)
        else:
            entries = self._logger.load_today()

        if not entries:
            print("No interactions logged this session.")
            print("="*60 + "\n")
            return

        # Show summary
        self._print_session_summary(entries)

        # Ask to save
        print("\nSave this session? (y/n): ", end="", flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer not in ("y", "yes"):
            print("Session not saved. Goodbye.\n")
            return

        # Run LLM analysis
        print("\nAnalyzing session for personal insights...")
        facts = self._extract_facts(entries)

        if not facts:
            print("Nothing new to add to your profile this session.")
            print("Session log saved. Goodbye.\n")
            return

        # Review facts with user
        approved = self._review_facts(facts)

        if approved:
            self._update_profile(approved)
            print(f"\n{len(approved)} fact(s) saved to your profile.")
        else:
            print("\nNo facts approved. Profile unchanged.")

        print("Session log saved. Goodbye.\n")

    # ── Session summary ──────────────────────────────────────────────────────

    def _print_session_summary(self, entries: list[dict]):
        """Prints a plain-text summary of the session."""

        total = len(entries)
        session_id = entries[0].get("session_id", "unknown") if entries else "unknown"

        # Count by intent
        intent_counts: dict[str, int] = {}
        tier_counts = {1: 0, 2: 0, 3: 0}
        low_conf = []
        follow_ups = []

        for e in entries:
            intent = e.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

            tier = e.get("tier", 0)
            if tier in tier_counts:
                tier_counts[tier] += 1

            if e.get("confidence", 1.0) < 0.70:
                low_conf.append(e)

            if (e.get("outcome") or {}).get("follow_up_needed"):
                follow_ups.append(e)

        print(f"\nSession: {session_id}")
        print(f"Total interactions: {total}")
        print(f"\nIntent breakdown:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
            print(f"  {intent:<20} {count}")

        print(f"\nClassifier tiers used:")
        print(f"  Tier 1 (rules):    {tier_counts[1]}")
        print(f"  Tier 2 (evidence): {tier_counts[2]}")
        print(f"  Tier 3 (LLM):      {tier_counts[3]}")

        if low_conf:
            print(f"\nLow confidence routing ({len(low_conf)} interactions):")
            for e in low_conf[:3]:
                print(f"  [{e['intent']} {e['confidence']:.2f}] \"{e['input'][:50]}\"")
            if len(low_conf) > 3:
                print(f"  ... and {len(low_conf) - 3} more")

        if follow_ups:
            print(f"\nInteractions that needed follow-up: {len(follow_ups)}")

        # Show last 5 exchanges as conversation preview
        print(f"\nConversation preview (last {min(5, total)} turns):")
        for e in entries[-5:]:
            print(f"  You:   {e['input'][:60]}")
            print(f"  Buddy: {e['response'][:60]}")
            print()

    # ── LLM fact extraction ──────────────────────────────────────────────────

    def _extract_facts(self, entries: list[dict]) -> list[dict]:
        """
        Sends the session to llama3.2 and asks it to extract
        personal facts about the user.

        Returns list of fact dicts:
        [
            {
                "category": "work",
                "fact": "Works as a research assistant at a university",
                "confidence": "high",
                "source_turn": 3,
            },
            ...
        ]
        """

        # Build a compact conversation transcript
        transcript_lines = []
        for e in entries:
            user_text = (e.get("input") or "").strip()
            buddy_text = (e.get("response") or "").strip()
            if user_text:
                transcript_lines.append(f"[Turn {e.get('turn', '?')}]")
                transcript_lines.append(f"User: {user_text}")
                transcript_lines.append(f"Buddy:  {buddy_text}")
                transcript_lines.append("")

        transcript = "\n".join(transcript_lines)

        # Limit transcript length to avoid context overflow
        if len(transcript) > 6000:
            transcript = transcript[:6000] + "\n... [truncated]"

        prompt = f"""You are analyzing a conversation between the user and their personal AI assistant Buddy.
Extract personal facts about the user that would help Buddy know them better in future sessions.

CONVERSATION:
{transcript}

Extract facts in these categories:
- work: job, study, university, research topics, colleagues
- schedule: regular activities, routines, recurring events
- preferences: things he likes or dislikes, how he prefers to be reminded
- location: places he goes regularly, home, office, university
- personal: family, relationships, hobbies, health

Rules:
- Only extract facts that are clearly stated or strongly implied
- Do not invent or assume facts
- Each fact must be specific and useful for a personal assistant
- Skip trivial or obvious facts
- If nothing meaningful was revealed, return an empty array

Respond with ONLY a JSON array. No explanation outside the array.
Example format:
[
  {{
    "category": "work",
    "fact": "Works as a Hiwi student at a university",
    "confidence": "high",
    "source_turn": 2
  }}
]

If nothing useful was found, return: []"""

        try:
            response = ollama.generate(model=MODEL, prompt=prompt)
            raw = (response.get("response") or "").strip()

            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()

            facts = json.loads(raw)
            if not isinstance(facts, list):
                return []

            # Filter to only valid fact dicts
            valid = []
            for f in facts:
                if isinstance(f, dict) and f.get("fact") and f.get("category"):
                    valid.append(f)

            return valid

        except Exception as e:
            print(f"[SessionRecap] Fact extraction error: {e}")
            return []

    # ── Manual review ────────────────────────────────────────────────────────

    def _review_facts(self, facts: list[dict]) -> list[dict]:
        """
        Shows each extracted fact and asks for approval.
        Returns only the approved facts.
        """
        if not facts:
            return []

        print(f"\nBuddy found {len(facts)} potential fact(s) about you.")
        print("Review each one — press Enter to approve, 'n' to skip, 'e' to edit.\n")

        approved = []

        for i, fact in enumerate(facts, 1):
            category = fact.get("category", "general")
            text = fact.get("fact", "")
            conf = fact.get("confidence", "medium")

            print(f"[{i}/{len(facts)}] ({category}, {conf} confidence)")
            print(f"  \"{text}\"")
            print("  Approve? (Enter=yes / n=no / e=edit): ", end="", flush=True)

            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer == "n":
                print("  Skipped.\n")
                continue

            elif answer == "e":
                print(f"  Edit fact (current: \"{text}\"): ", end="", flush=True)
                try:
                    edited = input().strip()
                    if edited:
                        fact["fact"] = edited
                        print(f"  Saved as: \"{edited}\"\n")
                    else:
                        print("  No change made. Skipping.\n")
                        continue
                except (EOFError, KeyboardInterrupt):
                    continue

            else:
                print("  Approved.\n")

            fact["approved_ts"] = datetime.now(TZ).isoformat()
            approved.append(fact)

        return approved

    # ── Profile update ───────────────────────────────────────────────────────

    def _update_profile(self, new_facts: list[dict]):
        """
        Merges approved facts into user_profile.json.
        Deduplicates by fact text — same fact won't be added twice.
        """
        # Load existing profile
        profile = self._load_profile()

        # Ensure facts list exists
        if "facts" not in profile:
            profile["facts"] = []

        # Deduplicate — skip if same fact text already exists
        existing_texts = {
            f.get("fact", "").lower().strip()
            for f in profile["facts"]
        }

        added = 0
        for fact in new_facts:
            fact_text = fact.get("fact", "").lower().strip()
            if fact_text and fact_text not in existing_texts:
                profile["facts"].append(fact)
                existing_texts.add(fact_text)
                added += 1

        # Update metadata
        profile["last_updated"] = datetime.now(TZ).isoformat()
        profile["total_facts"]  = len(profile["facts"])

        # Save
        try:
            PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"[Profile] {added} new fact(s) added. "
                  f"Total: {profile['total_facts']} facts.")
        except Exception as e:
            print(f"[SessionRecap] Could not save profile: {e}")

    def _load_profile(self) -> dict:
        """Loads existing user_profile.json or returns empty profile."""
        if not PROFILE_PATH.exists():
            return {
                "name":         "User",
                "created":      datetime.now(TZ).isoformat(),
                "last_updated": None,
                "total_facts":  0,
                "facts":        [],
            }

        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"facts": []}

    # ── Profile reader (used by ConversationManager for context) ─────────────

    def load_profile_context(self) -> str:
        """
        Returns a TTS/prompt-ready summary of known facts about the user.
        Used by ConversationManager to inject personal context into responses.

        Example output:
          "The user works as a research assistant. They study terahertz signals.
           He prefers morning reminders."
        """
        profile = self._load_profile()
        facts = profile.get("facts", [])

        if not facts:
            return ""

        # Group by category
        by_category: dict[str, list[str]] = {}
        for f in facts:
            cat = f.get("category", "general")
            text = f.get("fact", "")
            if text:
                by_category.setdefault(cat, []).append(text)

        lines = []
        for cat, cat_facts in by_category.items():
            lines.extend(cat_facts)

        return " ".join(lines)


# ── Wiring instructions ──────────────────────────────────────────────────────
"""
HOW TO WIRE INTO chat() in main.py
====================================

Step 1 — Add imports at top of main.py:
    import signal
    import sys
    from brain.adaptive.session_logger import session_logger
    from brain.adaptive.session_recap import SessionRecap

Step 2 — Add shutdown hook BEFORE the voice loop in chat():
    def _on_shutdown(sig, frame):
        print()  # newline after ^C
        recap = SessionRecap()
        recap.run_end_of_session()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_shutdown)

Step 3 — Log every interaction in handle_request() in ConversationManager:
    # At the top of handle_request(), before routing:
    pending_before = self.memory_manager.get_pending_topic()
    
    # ... existing routing logic ...
    
    # At the very end, before return response:
    try:
        from brain.adaptive.session_logger import session_logger
        session_logger.log(
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
    except Exception:
        pass
    
    return response

That is the complete wiring. Three steps.
"""


# ── CLI usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    recap = SessionRecap()

    if len(sys.argv) > 1:
        # Review a specific date: python session_recap.py 2026-04-10
        date_arg = sys.argv[1]
        print(f"Reviewing session from {date_arg}...")
        entries = recap._logger.load_date(date_arg)
        if not entries:
            print(f"No session log found for {date_arg}")
            sys.exit(0)
        recap._print_session_summary(entries)

        print("\nExtract facts from this session? (y/n): ", end="", flush=True)
        if input().strip().lower() in ("y", "yes"):
            facts = recap._extract_facts(entries)
            if facts:
                approved = recap._review_facts(facts)
                if approved:
                    recap._update_profile(approved)
            else:
                print("No facts found in this session.")
    else:
        # Review today's session
        recap.run_end_of_session()