"""
job_watchdog.py — Daily job scanner for Buddy.

Flow:
  1. JobWatchdog runs once per day (triggered by Buddy's scheduler or voice command)
  2. It calls Friday.run_job_watchdog() with the candidate profile + target list
  3. Friday searches, scores, and saves a .md recap to LOCAL_RECAP_DIR
  4. On next morning startup, Buddy reads the latest recap aloud

Voice triggers (add to STRONG_RULES in main.py):
  "check job updates"
  "any new jobs"
  "Friday scan jobs"
  "show me the job recap"
  "what jobs came up today"

Integration in ConversationManager:
  - Add self.job_watchdog = JobWatchdog(friday=self.friday) in __init__
  - Add handle in handle_request() under a new "job_watchdog" intent
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, date
from pathlib import Path

from brain.settings import load_json, load_text
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from brain.friday.friday import Friday

# ── Config ────────────────────────────────────────────────────────────────────

# Where daily recap .md files are saved (edit to your preferred path)
LOCAL_RECAP_DIR = Path.home() / "buddy_friday" / "job_recaps"

# Domains Friday searches for
JOB_DOMAINS = [
    "robotics",
    "signal processing",
    "control systems",
    "embedded systems",
    "mechatronics",
    "autonomous systems",
    "drone / UAV",
    "motion planning",
    "sensor fusion",
    "SLAM",
    "ROS / ROS2",
    "computer vision (for robotics)",
    "motor control",
    "real-time systems (RTOS)",
]

# Target companies and the candidate profile are PRIVATE: see config/*.example.* and brain/settings.py
TARGET_COMPANIES = load_json("target_companies.json", ["Example Robotics GmbH", "Example Automation AG"])

CANDIDATE_PROFILE = load_text("candidate_profile.md", "Name: (your name)\nRoles sought: (your target roles)\n")

# What time to run the daily watchdog (24h format, local time)
WATCHDOG_HOUR = 7    # 7 AM
WATCHDOG_MINUTE = 0


# ── JobWatchdog class ─────────────────────────────────────────────────────────

class JobWatchdog:
    """
    Manages the daily job scanning loop.
    Runs in a background thread; calls Friday once per day.
    """

    def __init__(self, friday: "Friday", recap_dir: Path = LOCAL_RECAP_DIR):
        self.friday = friday
        self.recap_dir = recap_dir
        self.recap_dir.mkdir(parents=True, exist_ok=True)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_run_date: Optional[date] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background watchdog thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="JobWatchdog",
        )
        self._thread.start()
        print("[JobWatchdog] Started — will scan daily at "
              f"{WATCHDOG_HOUR:02d}:{WATCHDOG_MINUTE:02d}")

    def stop(self) -> None:
        self._stop_event.set()

    # ── Manual trigger (voice command or testing) ──────────────────────────

    def trigger_now(self) -> str:
        """
        Immediately dispatch a job scan regardless of time.
        Called when user says "check job updates" or "Friday scan jobs".
        """
        print("[JobWatchdog] Manual trigger fired")
        return self.friday.run_job_watchdog(
            recap_dir=self.recap_dir,
            domains=JOB_DOMAINS,
            companies=TARGET_COMPANIES,
            candidate_profile=CANDIDATE_PROFILE,
            date_label=datetime.now().strftime("%Y-%m-%d"),
        )

    # ── Morning briefing ──────────────────────────────────────────────────

    def get_morning_briefing(self) -> Optional[str]:
        """
        Returns a short spoken briefing from the latest recap file, or None
        if no recap exists for today.

        Called during Buddy's startup to inject into the morning greeting.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        recap_path = self.recap_dir / f"job_recap_{today}.md"

        if not recap_path.exists():
            # Also check yesterday's recap as fallback
            from datetime import timedelta
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            recap_path = self.recap_dir / f"job_recap_{yesterday}.md"
            if not recap_path.exists():
                return None

        return _parse_recap_for_tts(recap_path)

    def get_latest_recap_path(self) -> Optional[Path]:
        """Returns the most recent recap .md file path, or None."""
        files = sorted(self.recap_dir.glob("job_recap_*.md"), reverse=True)
        return files[0] if files else None

    def read_latest_recap(self) -> str:
        """
        Returns full text of the latest recap for display/voice.
        Called when user says "show me the job recap".
        """
        path = self.get_latest_recap_path()
        if not path:
            return ("I don't have a job recap yet. Say 'Friday scan jobs' and "
                    "I'll ask Friday to search for you.")
        return path.read_text(encoding="utf-8")

    # ── Background loop ───────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now()
            today = now.date()

            should_run = (
                now.hour == WATCHDOG_HOUR
                and now.minute == WATCHDOG_MINUTE
                and self._last_run_date != today
            )

            if should_run:
                print(f"[JobWatchdog] Scheduled run at {now.strftime('%H:%M')}")
                self._last_run_date = today
                try:
                    self.friday.run_job_watchdog(
                        recap_dir=self.recap_dir,
                        domains=JOB_DOMAINS,
                        companies=TARGET_COMPANIES,
                        candidate_profile=CANDIDATE_PROFILE,
                        date_label=today.strftime("%Y-%m-%d"),
                    )
                except Exception as e:
                    print(f"[JobWatchdog] Error during scheduled run: {e}")

            # Sleep 55 seconds between checks (fine-grained enough for minute accuracy)
            self._stop_event.wait(timeout=55)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_recap_for_tts(path: Path) -> str:
    """
    Extracts a short spoken summary from a recap .md file.
    Tries the ## Summary section first, then falls back to counting table rows.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return "I found a job recap but couldn't read it."

    # Try ## Summary section
    m = re.search(r"##\s*Summary\s*\n+(.*?)(?:\n##|\Z)", text, re.DOTALL)
    if m:
        summary = m.group(1).strip()
        # Keep it concise for TTS
        summary = " ".join(summary.split())
        if len(summary) > 400:
            summary = summary[:397] + "..."
        date_m = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
        date_str = date_m.group(0) if date_m else "today"
        return f"Job recap for {date_str}: {summary}"

    # Fallback: count rows in Strong Fit table
    strong_rows = len(re.findall(r"^\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|", text, re.MULTILINE)) - 2
    strong_rows = max(0, strong_rows)

    partial_m = re.search(r"## Partial Fit", text)
    partial_rows = 0
    if partial_m:
        after = text[partial_m.end():]
        partial_rows = max(0, len(re.findall(r"^\|[^|]+\|", after, re.MULTILINE)) - 2)

    return (
        f"Friday found {strong_rows} strong fit and {partial_rows} partial fit jobs today. "
        f"Say 'show me the job recap' for the full list."
    )


# ── Intent detection helpers (paste into main.py STRONG_RULES) ───────────────

JOB_WATCHDOG_PHRASES = [
    "check job updates",
    "any new jobs",
    "scan for jobs",
    "friday scan jobs",
    "job scan",
    "what jobs came up",
    "show me the job recap",
    "job recap",
    "any job openings",
    "check for new positions",
    "check for new openings",
    "job watchdog",
    "scan job listings",
]