"""
brain/adaptive/agent_core.py

Adaptive Agent Core
===================
The entry point for Buddy's proactive layer.
Runs as a background thread alongside the main voice loop in chat().

Responsibilities:
  1. On startup → run schedule_awareness.startup_scan() and speak results
  2. On a timer → re-check upcoming events and speak reminders
  3. Holiday conflict hook → called by memory_handler when saving new schedules

This is the ONLY file that chat() needs to know about.
Everything else in brain/adaptive/ is internal.

Wiring into chat() — add these two lines:
    from brain.adaptive.agent_core import AdaptiveAgent
    ...
    # After TTS is ready, before the voice loop:
    adaptive_agent = AdaptiveAgent(tts=tts, memory_manager=memory_manager)
    adaptive_agent.start()

That's it. One import, one start() call.
"""

import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

TZ = ZoneInfo("Europe/Berlin")

# How long to wait after startup before speaking the awareness report
# Gives Buddy time to finish the intro greeting first
STARTUP_DELAY_SECONDS = 8

# How often to re-check upcoming events (in seconds)
# Default: every 10 minutes
POLL_INTERVAL_SECONDS = 600

# How many minutes before an event to trigger a reminder
REMINDER_THRESHOLDS_MINUTES = [30, 60, 120]


class AdaptiveAgent:
    """
    Buddy's proactive awareness engine.
    Runs in a background daemon thread — dies cleanly when Buddy exits.

    Usage in chat():
        adaptive_agent = AdaptiveAgent(tts=tts, memory_manager=memory_manager)
        adaptive_agent.start()
    """

    def __init__(self, tts, memory_manager, startup_delay: int = STARTUP_DELAY_SECONDS):
        """
        Args:
            tts:            TTSWorker instance from chat() — used to speak
            memory_manager: MemoryManager instance — used to check pending state
        """
        self._tts = tts
        self._mm = memory_manager
        self._startup_delay = startup_delay 
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track which event IDs we've already reminded about
        # so we don't repeat the same reminder every poll cycle
        self._reminded: set[str] = set()

        # Track whether startup scan has been spoken
        self._startup_done = False

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self):
        """
        Starts the adaptive agent background thread.
        Safe to call multiple times — won't start a second thread.
        """
        if self._thread and self._thread.is_alive():
            print("[AdaptiveAgent] Already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="AdaptiveAgentThread",
            daemon=True,  # dies when main process exits
        )
        self._thread.start()
        print("[AdaptiveAgent] Started.")

    def stop(self):
        """Signals the background thread to stop cleanly."""
        self._stop_event.set()
        print("[AdaptiveAgent] Stop requested.")

    def is_running(self) -> bool:
        """Returns True if the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ── Holiday conflict check (called externally by memory_handler) ─────────

    def check_new_schedule(self, title: str, timestamp: str) -> Optional[str]:
        """
        Called by memory_handler.save_memory() immediately after a new
        schedule is saved. Returns a warning string if the date is a
        public holiday, else None.

        This is the REACTIVE holiday check — fires at save time.
        The PROACTIVE check (startup_scan) handles existing schedules.

        Args:
            title:     event title
            timestamp: "YYYY-MM-DD HH:MM"

        Returns:
            TTS-ready warning string if holiday conflict, else None

        Example in memory_handler.save_memory():
            warning = adaptive_agent.check_new_schedule(title, timestamp)
            if warning:
                response += f" {warning}"
        """
        try:
            from brain.adaptive.awareness.holiday_checker import HolidayChecker
            checker = HolidayChecker()
            result = checker.check_schedule({
                "title": title,
                "timestamp": timestamp,
            })
            return result.get("warning")
        except Exception as e:
            print(f"[AdaptiveAgent] check_new_schedule error: {e}")
            return None

    # ── Internal thread loop ─────────────────────────────────────────────────

    def _run(self):
        """
        Main background thread loop.
        1. Wait for startup delay (let greeting finish)
        2. Run startup scan and speak results
        3. Poll every POLL_INTERVAL_SECONDS for upcoming reminders
        """
        # Step 1 — wait for startup delay
        #print(f"[AdaptiveAgent] Waiting {STARTUP_DELAY_SECONDS}s before startup scan...")
        self._stop_event.wait(timeout=self._startup_delay)

        if self._stop_event.is_set():
            return

        # Step 2 — run startup scan
        self._run_startup_scan()

        # Step 3 — poll loop
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=POLL_INTERVAL_SECONDS)

            if self._stop_event.is_set():
                break

            self._run_poll_cycle()

        print("[AdaptiveAgent] Thread exiting.")

    def _is_buddy_busy(self) -> bool:
        """
        Returns True if Buddy is currently in a pending flow
        (e.g. waiting for user confirmation, mid-schedule save, etc.)
        We don't interrupt mid-conversation.
        """
        pending = self._mm.get_pending_topic()
        return bool(pending)

    def _speak(self, text: str):
        """
        Speaks text via TTS if Buddy is not currently busy.
        Sanitizes the text before speaking.
        """
        if not text or not text.strip():
            return

        if self._is_buddy_busy():
            print(f"[AdaptiveAgent] Buddy busy — queuing message: {text[:60]}...")
            # Simple retry after a short delay
            time.sleep(5)
            if self._is_buddy_busy():
                print("[AdaptiveAgent] Still busy — skipping message.")
                return

        try:
            # Reuse the sanitize_tts from main.py
            from main import sanitize_tts
            clean = sanitize_tts(text).strip()
        except ImportError:
            clean = text.strip()

        if clean:
            print(f"[AdaptiveAgent] Speaking: {clean[:80]}...")
            self._tts.speak(clean)

    # ── Startup scan ─────────────────────────────────────────────────────────

    def _run_startup_scan(self):
        """
        Runs the full awareness scan on startup and speaks the results.
        Speaks high-priority items immediately, low-priority items after a pause.
        """
        print("[AdaptiveAgent] Running startup scan...")

        try:
            from brain.adaptive.awareness.schedule_awareness import ScheduleAwareness
            awareness = ScheduleAwareness()
            report = awareness.startup_scan(
                hours_back=12,
                upcoming_days=1,
                conflict_days=14,
                holiday_heads_up_days=3,
            )
        except Exception as e:
            print(f"[AdaptiveAgent] startup_scan error: {e}")
            return

        self._startup_done = True

        if not report["has_items"]:
            print("[AdaptiveAgent] Startup scan: nothing to report.")
            return

        counts = report["counts"]
        print(
            f"[AdaptiveAgent] Startup scan complete: "
            f"{counts['conflicts']} conflicts, "
            f"{counts['past_checkins']} check-ins, "
            f"{counts['upcoming']} upcoming, "
            f"{counts['holiday_heads_up']} holiday heads-up"
        )

        # Speak high priority items first
        high_items = [i for i in report["items"] if i["priority"] == "high"]
        medium_items = [i for i in report["items"] if i["priority"] == "medium"]
        low_items = [i for i in report["items"] if i["priority"] == "low"]

        # Speak high priority immediately — max 2
        for item in high_items[:2]:
            self._speak(item["message"])
            time.sleep(1)

        # Brief pause before medium priority
        if medium_items:
            time.sleep(2)
            for item in medium_items[:1]:
                self._speak(item["message"])
                time.sleep(1)

        # Low priority — only if no high/medium items dominated
        if not high_items and not medium_items and low_items:
            for item in low_items[:1]:
                self._speak(item["message"])

        # Mark reminded events so poll cycle doesn't repeat them
        for item in report["items"]:
            event = item.get("event", {})
            event_id = self._event_id(event)
            if event_id:
                self._reminded.add(event_id)

    # ── Poll cycle ───────────────────────────────────────────────────────────

    def _run_poll_cycle(self):
        """
        Runs every POLL_INTERVAL_SECONDS.
        Checks for events coming up within the reminder thresholds
        that haven't been reminded about yet.
        """
        print("[AdaptiveAgent] Poll cycle running...")

        try:
            from brain.adaptive.awareness.schedule_awareness import ScheduleAwareness
            awareness = ScheduleAwareness()
            upcoming_items = awareness.scan_upcoming_warnings(
                warn_minutes=REMINDER_THRESHOLDS_MINUTES,
                days_ahead=1,
            )
        except Exception as e:
            print(f"[AdaptiveAgent] poll_cycle error: {e}")
            return

        for item in upcoming_items:
            event = item.get("event", {})
            event_id = self._event_id(event)

            # Skip if already reminded
            if event_id and event_id in self._reminded:
                continue

            # Only speak high priority in poll cycle
            # (medium/low were handled at startup)
            if item["priority"] == "high":
                self._speak(item["message"])
                time.sleep(1)

                if event_id:
                    self._reminded.add(event_id)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _event_id(self, event: dict) -> Optional[str]:
        """
        Builds a stable unique ID for an event for deduplication.
        Uses database ID if available, otherwise title+timestamp.
        """
        if not event:
            return None

        db_id = event.get("id")
        if db_id:
            return f"{event.get('source', 'local')}:{db_id}"

        # Fallback — title + timestamp
        title = (event.get("title") or "").strip().lower()
        ts = (event.get("timestamp") or "").strip()
        if title and ts:
            return f"derived:{title}|{ts}"

        return None


# ── Module-level singleton ───────────────────────────────────────────────────
# This gets populated in chat() when AdaptiveAgent is instantiated.
# memory_handler.save_memory() imports this to call check_new_schedule().
_agent_instance: Optional[AdaptiveAgent] = None


def get_agent() -> Optional[AdaptiveAgent]:
    """Returns the running AdaptiveAgent instance, or None if not started."""
    return _agent_instance


def init_agent(tts, memory_manager, startup_delay: int = STARTUP_DELAY_SECONDS):
    global _agent_instance
    _agent_instance = AdaptiveAgent(
        tts=tts,
        memory_manager=memory_manager,
        startup_delay=startup_delay,
    )
    _agent_instance.start()
    return _agent_instance


# ── dummy ──────────────────────────────────────────────────────
"""
COMMUNCIATION WITH MAIN

Step 1 — Add import at the top of main.py:
    from brain.adaptive.agent_core import init_agent, get_agent

Step 2 — Start the agent AFTER tts is ready, BEFORE the voice loop:
    # Existing code:
    tts = TTSWorker(vad=vad, spotify=conversation_manager.spotify)
    tts.start()

    adaptive_agent = init_agent(tts=tts, memory_manager=memory_manager)

Step 3 — Hook into memory_handler.save_memory() in main.py.
    Find the line where create_schedule() succeeds and returns a confirmation,
    for example:
        return f"Got it, I will remember for your {title} at {ts}."

    Change it to:
        base_response = f"Got it, I will remember for your {title} at {ts}."
        agent = get_agent()
        if agent:
            warning = agent.check_new_schedule(title, ts)
            if warning:
                return f"{base_response} {warning}"
        return base_response

"""


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== AdaptiveAgent — Self Test (no TTS) ===\n")

    # Mock TTS that just prints
    class MockTTS:
        def speak(self, text):
            print(f"[TTS] {text}")

    # Mock MemoryManager with no pending state
    class MockMM:
        def get_pending_topic(self):
            return None

    agent = AdaptiveAgent(tts=MockTTS(), memory_manager=MockMM())

    print("--- Testing check_new_schedule() ---")
    test_cases = [
        ("Hiwi work",      "2026-04-03 09:00"),
        ("Gym session",    "2026-04-07 08:00"),
        ("Team meeting",   "2026-05-01 10:00"),
        ("Regular errand", "2026-04-08 14:00"),
    ]
    for title, ts in test_cases:
        warning = agent.check_new_schedule(title, ts)
        if warning:
            print(f"  ⚠  [{title} @ {ts}] → {warning}")
        else:
            print(f"  ✓  [{title} @ {ts}] → no conflict")

    print("\n--- Running startup scan directly (no thread) ---")
    agent._run_startup_scan()

    print("\n--- Testing is_running() ---")
    print(f"  Before start: {agent.is_running()}")
    agent.start()
    print(f"  After start:  {agent.is_running()}")

    # Let it run briefly
    time.sleep(3)
    agent.stop()
    time.sleep(1)
    print(f"  After stop:   {agent.is_running()}")

    print("\n--- init_agent() singleton test ---")
    from brain.adaptive.agent_core import init_agent, get_agent
    instance = init_agent(tts=MockTTS(), memory_manager=MockMM())
    print(f"  get_agent() returns instance: {get_agent() is instance}")
    instance.stop()