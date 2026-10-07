"""
brain/adaptive/awareness/unified_schedule.py

Unified Schedule
================
Merges LocalScheduleSource and GCalSource into a single sorted view.
This is the single point of truth for all schedule data in Buddy.

The rest of the adaptive brain (schedule_awareness, morning_briefing,
agent_core) only talks to UnifiedSchedule — never directly to SQL or GCal.

Design principles:
  - GCal unavailable → falls back to local SQL silently, no crash
  - Deduplication → same event in both sources appears only once
  - Source tagging → every event knows where it came from
  - Holiday awareness → conflict detection built in at this layer

Usage:
    from brain.adaptive.awareness.unified_schedule import UnifiedSchedule

    schedule = UnifiedSchedule()

    # Get everything for today
    events = schedule.get_today()

    # Get upcoming week with holiday flags
    events = schedule.get_upcoming(days_ahead=7)

    # Get past events that need check-in
    events = schedule.get_past_unresolved()

    # Get full conflict report
    report = schedule.get_conflict_report()
"""

from datetime import date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Berlin")


class UnifiedSchedule:
    """
    Single interface for all of Buddy's schedule data.
    Merges local SQL + Google Calendar + holiday awareness.
    """

    def __init__(self):
        from brain.adaptive.awareness.local_schedule_source import LocalScheduleSource
        from brain.adaptive.awareness.gcal_source import GCalSource
        from brain.adaptive.awareness.holiday_checker import HolidayChecker

        self._local = LocalScheduleSource()
        self._gcal = GCalSource()
        self._holidays = HolidayChecker()

    # ── Deduplication ────────────────────────────────────────────────────────

    def _dedup_key(self, event: dict) -> str:
        """
        Builds a deduplication key from title + date + time.
        Events with the same key from different sources are treated as the same event.

        Logic: if you saved "Dentist at 9am Friday" in both Buddy and Google Calendar,
        it should appear once — not twice.
        """
        title = (event.get("title") or "").strip().lower()
        date_part = (event.get("date") or "").strip()
        time_part = (event.get("time") or "").strip()
        return f"{title}|{date_part}|{time_part}"

    def _merge(self, local: list[dict], gcal: list[dict]) -> list[dict]:
        """
        Merges two event lists, deduplicating by title+date+time.
        When a duplicate exists, prefers the GCal version (has gcal_link).
        Sorts result by timestamp ascending.
        """
        seen: dict[str, dict] = {}

        # Add local events first
        for event in local:
            key = self._dedup_key(event)
            seen[key] = event

        # GCal events overwrite local duplicates (GCal has more info)
        for event in gcal:
            key = self._dedup_key(event)
            seen[key] = event

        merged = list(seen.values())
        return sorted(merged, key=lambda e: e.get("timestamp") or "")

    # ── Holiday enrichment ───────────────────────────────────────────────────

    def _enrich_with_holiday(self, event: dict) -> dict:
        """
        Adds holiday conflict info to an event dict in place.
        Adds two fields:
          - "is_holiday": bool
          - "holiday_name": str | None
        """
        result = self._holidays.check_schedule(event)
        event["is_holiday"] = result["is_holiday"]
        event["holiday_name"] = result["holiday_name"]
        return event

    def _enrich_list(self, events: list[dict]) -> list[dict]:
        """Enriches a list of events with holiday info."""
        return [self._enrich_with_holiday(dict(e)) for e in events]

    # ── Core fetch methods ───────────────────────────────────────────────────

    def get_all(self, enrich: bool = True) -> list[dict]:
        """
        Returns all events from both sources merged and sorted.

        Args:
            enrich: if True, adds holiday conflict flags (default True)
        """
        local = self._local.get_all()
        gcal = self._gcal.get_upcoming(days_ahead=365) if self._gcal.is_available() else []
        merged = self._merge(local, gcal)
        return self._enrich_list(merged) if enrich else merged

    def get_by_date(self, date_str: str, enrich: bool = True) -> list[dict]:
        """
        Returns all events on a specific date from both sources.

        Args:
            date_str: "YYYY-MM-DD"
            enrich:   if True, adds holiday conflict flags
        """
        local = self._local.get_by_date(date_str)
        gcal = self._gcal.get_by_range(date_str, date_str) if self._gcal.is_available() else []
        merged = self._merge(local, gcal)
        return self._enrich_list(merged) if enrich else merged

    def get_by_range(
        self,
        start_date: str,
        end_date: str,
        enrich: bool = True,
    ) -> list[dict]:
        """
        Returns all events between start_date and end_date inclusive.

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"
            enrich:     if True, adds holiday conflict flags
        """
        local = self._local.get_by_range(start_date, end_date)
        gcal = self._gcal.get_by_range(start_date, end_date) if self._gcal.is_available() else []
        merged = self._merge(local, gcal)
        return self._enrich_list(merged) if enrich else merged

    def get_today(self, enrich: bool = True) -> list[dict]:
        """Returns all events for today from both sources."""
        today = date.today().strftime("%Y-%m-%d")
        return self.get_by_date(today, enrich=enrich)

    def get_tomorrow(self, enrich: bool = True) -> list[dict]:
        """Returns all events for tomorrow from both sources."""
        tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        return self.get_by_date(tomorrow, enrich=enrich)

    def get_upcoming(
        self,
        days_ahead: int = 7,
        enrich: bool = True,
    ) -> list[dict]:
        """
        Returns all events from today up to days_ahead days.

        Args:
            days_ahead: how many days forward to look (default 7)
            enrich:     if True, adds holiday conflict flags
        """
        today = date.today().strftime("%Y-%m-%d")
        end = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return self.get_by_range(today, end, enrich=enrich)

    def get_this_week(self, enrich: bool = True) -> list[dict]:
        """Returns all events for the current week (Monday to Sunday)."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        return self.get_by_range(
            monday.strftime("%Y-%m-%d"),
            sunday.strftime("%Y-%m-%d"),
            enrich=enrich,
        )

    def get_past_unresolved(
        self,
        hours_back: int = 24,
        enrich: bool = True,
    ) -> list[dict]:
        """
        Returns events from the last N hours that may need a check-in.
        Used by the watchdog to ask: "You had X at 9am — did that happen?"

        Args:
            hours_back: how far back to look (default 24)
            enrich:     if True, adds holiday conflict flags
        """
        local = self._local.get_past_unresolved(hours_back=hours_back)
        gcal = self._gcal.get_past_unresolved(hours_back=hours_back) if self._gcal.is_available() else []
        merged = self._merge(local, gcal)
        # Most recent first for check-in flow
        merged_sorted = sorted(
            merged,
            key=lambda e: e.get("timestamp") or "",
            reverse=True,
        )
        return self._enrich_list(merged_sorted) if enrich else merged_sorted

    # ── Holiday conflict report ──────────────────────────────────────────────

    def get_conflict_report(self, days_ahead: int = 30) -> dict:
        """
        Scans upcoming schedules and returns a full holiday conflict report.
        This is what agent_core calls on startup.

        Args:
            days_ahead: how many days ahead to scan (default 30)

        Returns:
            {
                "conflicts": [
                    {
                        "schedule": {...},
                        "holiday_name": "Karfreitag",
                        "warning": "Heads up — Hiwi work falls on Karfreitag...",
                        "date": "YYYY-MM-DD",
                    },
                    ...
                ],
                "upcoming_holidays": [
                    {"date": "YYYY-MM-DD", "name": "...", "days_until": int},
                    ...
                ],
                "total_conflicts": int,
                "has_conflicts": bool,
            }
        """
        upcoming_events = self.get_upcoming(days_ahead=days_ahead, enrich=True)

        conflicts = []
        for event in upcoming_events:
            if event.get("is_holiday"):
                title = event.get("title", "This event")
                holiday_name = event.get("holiday_name", "a public holiday")
                warning = (
                    f"Heads up — {title} falls on {holiday_name}, "
                    f"a public holiday in NRW. "
                    f"Most offices and shops will be closed."
                )
                conflicts.append({
                    "schedule": event,
                    "holiday_name": holiday_name,
                    "warning": warning,
                    "date": event.get("date"),
                })

        upcoming_holidays = self._holidays.is_upcoming_holiday(days_ahead=days_ahead)

        return {
            "conflicts":          conflicts,
            "upcoming_holidays":  upcoming_holidays,
            "total_conflicts":    len(conflicts),
            "has_conflicts":      len(conflicts) > 0,
        }

    # ── Summary helpers (for TTS output) ────────────────────────────────────

    def summarize_today(self) -> str:
        """
        Returns a TTS-ready summary of today's schedule.

        Examples:
            "You have 3 events today: Gym at 7am, Lecture at 10am,
             and Dentist at 3pm."
            "You have nothing scheduled for today."
            "You have Hiwi work at 9am today — but today is Karfreitag,
             a public holiday. Most offices will be closed."
        """
        events = self.get_today()

        if not events:
            return "You have nothing scheduled for today."

        # Check for holiday conflicts first
        holiday_warnings = [
            e for e in events if e.get("is_holiday")
        ]

        parts = []
        for e in events:
            time_str = e.get("time") or "all day"
            title = e.get("title", "something")
            parts.append(f"{title} at {time_str}")

        count = len(events)
        event_word = "event" if count == 1 else "events"

        if count == 1:
            summary = f"You have {parts[0]} today."
        elif count == 2:
            summary = f"You have {parts[0]} and {parts[1]} today."
        else:
            summary = f"You have {count} {event_word} today: " + \
                      ", ".join(parts[:-1]) + f", and {parts[-1]}."

        # Append holiday warnings
        for e in holiday_warnings:
            summary += f" By the way, {e['holiday_name']} is today — most offices will be closed."

        return summary

    def summarize_upcoming(self, days_ahead: int = 3) -> str:
        """
        Returns a TTS-ready summary of upcoming events.
        Used in the morning briefing.
        """
        events = self.get_upcoming(days_ahead=days_ahead)

        if not events:
            return f"You have nothing scheduled in the next {days_ahead} days."

        today_str = date.today().strftime("%Y-%m-%d")
        tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        lines = []
        for e in events:
            title = e.get("title", "something")
            time_str = e.get("time") or "all day"
            date_str = e.get("date") or ""
            holiday_str = f" — {e['holiday_name']} today" if e.get("is_holiday") else ""

            if date_str == today_str:
                day_label = "today"
            elif date_str == tomorrow_str:
                day_label = "tomorrow"
            else:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    day_label = dt.strftime("%A")  # e.g. "Wednesday"
                except Exception:
                    day_label = date_str

            lines.append(f"{title} {day_label} at {time_str}{holiday_str}")

        if len(lines) == 1:
            return f"Coming up: {lines[0]}."
        return "Coming up: " + ", ".join(lines[:-1]) + f", and {lines[-1]}."

    # ── Write through ────────────────────────────────────────────────────────

    def create_event(
        self,
        title: str,
        timestamp: str,
        description: Optional[str] = None,
        location: Optional[str] = None,
        save_to_gcal: bool = True,
    ) -> dict:
        """
        Creates an event in both local SQL and Google Calendar.
        Returns a result dict with status from both sources
        plus an immediate holiday conflict check.

        Args:
            title:        Event title
            timestamp:    "YYYY-MM-DD HH:MM"
            description:  Optional description
            location:     Optional location
            save_to_gcal: Whether to also save to Google Calendar (default True)

        Returns:
            {
                "saved_local":   bool,
                "saved_gcal":    bool,
                "holiday_check": {...},   ← from holiday_checker
                "warning":       str | None,  ← TTS-ready holiday warning
            }
        """
        from brain.core.sql_brain import create_schedule

        # Save to local SQL
        saved_local = False
        try:
            create_schedule(
                title,
                description or title,
                timestamp,
                location,
            )
            saved_local = True
        except Exception as e:
            print(f"[UnifiedSchedule] local save error: {e}")

        # Save to Google Calendar
        saved_gcal = False
        if save_to_gcal and self._gcal.is_available():
            result = self._gcal.create_event(
                title=title,
                timestamp=timestamp,
                description=description,
                location=location,
            )
            saved_gcal = result is not None

        # Immediate holiday conflict check
        fake_schedule = {"title": title, "timestamp": timestamp}
        holiday_check = self._holidays.check_schedule(fake_schedule)

        return {
            "saved_local":   saved_local,
            "saved_gcal":    saved_gcal,
            "holiday_check": holiday_check,
            "warning":       holiday_check.get("warning"),
        }

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Returns connection status of both sources.
        Useful for debugging and startup checks.
        """
        return {
            "local_available": True,  # always available
            "gcal_available":  self._gcal.is_available(),
            "local_count":     self._local.count(),
        }


# ── Convenience singleton ────────────────────────────────────────────────────
unified_schedule = UnifiedSchedule()


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import datetime
    from zoneinfo import ZoneInfo

    schedule = UnifiedSchedule()

    print("=== Unified Schedule — Self Test ===")
    status = schedule.status()
    print(f"Local SQL:       available=True, count={status['local_count']}")
    print(f"Google Calendar: available={status['gcal_available']}")

    print("\n--- Today ---")
    today_events = schedule.get_today()
    if today_events:
        for e in today_events:
            holiday_tag = f" ⚠ {e['holiday_name']}" if e.get("is_holiday") else ""
            print(f"  [{e['source']}] {e['time'] or 'all-day'}  {e['title']}{holiday_tag}")
    else:
        print("  (none)")

    print("\n--- Tomorrow ---")
    tomorrow_events = schedule.get_tomorrow()
    if tomorrow_events:
        for e in tomorrow_events:
            holiday_tag = f" ⚠ {e['holiday_name']}" if e.get("is_holiday") else ""
            print(f"  [{e['source']}] {e['time'] or 'all-day'}  {e['title']}{holiday_tag}")
    else:
        print("  (none)")

    print("\n--- Upcoming 7 days ---")
    upcoming = schedule.get_upcoming(days_ahead=7)
    if upcoming:
        for e in upcoming:
            holiday_tag = f" ⚠ {e['holiday_name']}" if e.get("is_holiday") else ""
            print(f"  [{e['source']}] {e['date']}  {e['time'] or 'all-day'}  {e['title']}{holiday_tag}")
    else:
        print("  (none)")

    print("\n--- Holiday conflict report (next 30 days) ---")
    report = schedule.get_conflict_report(days_ahead=30)
    print(f"  Total conflicts: {report['total_conflicts']}")
    if report["has_conflicts"]:
        for c in report["conflicts"]:
            print(f"  ⚠  {c['warning']}")
    if report["upcoming_holidays"]:
        print(f"\n  Upcoming holidays:")
        for h in report["upcoming_holidays"]:
            print(f"  {h['name']} in {h['days_until']} day(s) ({h['date']})")

    print("\n--- TTS summaries ---")
    print(f"  Today:    {schedule.summarize_today()}")
    print(f"  Upcoming: {schedule.summarize_upcoming(days_ahead=3)}")

    print("\n--- Past unresolved (last 24h) ---")
    past = schedule.get_past_unresolved(hours_back=24)
    if past:
        for e in past:
            print(f"  [{e['source']}] {e['timestamp']}  {e['title']}  ← needs check-in")
    else:
        print("  (none)")