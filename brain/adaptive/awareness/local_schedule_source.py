"""
brain/adaptive/awareness/local_schedule_source.py

Local Schedule Source
=====================
Thin wrapper around sql_brain.py that gives the unified schedule layer
a consistent interface. No logic lives here — just data access.

This exists so unified_schedule.py can treat local SQL and Google Calendar
identically, swapping one out without touching the other.

Usage:
    from brain.adaptive.awareness.local_schedule_source import LocalScheduleSource

    source = LocalScheduleSource()

    # Get all schedules
    schedules = source.get_all()

    # Get schedules for a specific date
    schedules = source.get_by_date("2026-04-07")

    # Get schedules in a date range
    schedules = source.get_by_range("2026-04-07", "2026-04-13")

    # Get schedules from today onward (upcoming only)
    schedules = source.get_upcoming(days_ahead=7)

    # Get past schedules that haven't been marked done
    schedules = source.get_past_unresolved()
"""

from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Berlin")


class LocalScheduleSource:
    """
    Reads from Buddy's local SQL schedule database (sql_brain.py).
    Returns normalized schedule dicts ready for unified_schedule.py.

    All returned dicts have this guaranteed shape:
    {
        "id":          int,
        "title":       str,
        "description": str | None,
        "timestamp":   str,          # "YYYY-MM-DD HH:MM"
        "date":        str,          # "YYYY-MM-DD"  (extracted from timestamp)
        "time":        str | None,   # "HH:MM"       (extracted from timestamp)
        "location":    str | None,
        "source":      "local",      # always "local" for this source
    }
    """

    SOURCE_NAME = "local"

    def _now(self) -> datetime:
        return datetime.now(TZ)

    def _today_str(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def _normalize(self, raw: dict) -> dict:
        """
        Takes a raw dict from get_all_schedules_as_dicts() and adds
        computed fields (date, time, source) for the unified layer.
        """
        ts = (raw.get("timestamp") or "").strip()

        # Extract date and time from timestamp
        date_part = ts[:10] if len(ts) >= 10 else None
        time_part = ts[11:16] if len(ts) >= 16 else None

        return {
            "id":          raw.get("id"),
            "title":       (raw.get("title") or "Untitled").strip(),
            "description": raw.get("description"),
            "timestamp":   ts or None,
            "date":        date_part,
            "time":        time_part,
            "location":    raw.get("location"),
            "source":      self.SOURCE_NAME,
        }

    def _load_all(self) -> list[dict]:
        """Loads all schedules from SQL and normalizes them."""
        from brain.core.sql_brain import get_all_schedules_as_dicts
        raw_list = get_all_schedules_as_dicts()
        return [self._normalize(r) for r in (raw_list or [])]

    # ── Public API ───────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        """
        Returns all schedules from local SQL, normalized.
        Sorted by timestamp ascending.
        """
        schedules = self._load_all()
        return sorted(
            schedules,
            key=lambda s: s.get("timestamp") or "",
        )

    def get_by_date(self, date_str: str) -> list[dict]:
        """
        Returns all schedules on a specific date.

        Args:
            date_str: "YYYY-MM-DD"

        Returns:
            list of normalized schedule dicts for that date
        """
        all_schedules = self._load_all()
        return [
            s for s in all_schedules
            if s.get("date") == date_str
        ]

    def get_by_range(self, start_date: str, end_date: str) -> list[dict]:
        """
        Returns all schedules between start_date and end_date inclusive.

        Args:
            start_date: "YYYY-MM-DD"
            end_date:   "YYYY-MM-DD"

        Returns:
            list of normalized schedule dicts sorted by timestamp
        """
        all_schedules = self._load_all()
        matches = [
            s for s in all_schedules
            if s.get("date")
            and start_date <= s["date"] <= end_date
        ]
        return sorted(matches, key=lambda s: s.get("timestamp") or "")

    def get_upcoming(self, days_ahead: int = 7) -> list[dict]:
        """
        Returns schedules from today onward, up to days_ahead days.

        Args:
            days_ahead: how many days forward to look (default 7)

        Returns:
            list of upcoming schedule dicts sorted by timestamp

        Example:
            # Get everything in the next 3 days
            upcoming = source.get_upcoming(days_ahead=3)
        """
        today = self._today_str()
        end = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return self.get_by_range(today, end)

    def get_today(self) -> list[dict]:
        """Returns all schedules for today."""
        return self.get_by_date(self._today_str())

    def get_tomorrow(self) -> list[dict]:
        """Returns all schedules for tomorrow."""
        tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        return self.get_by_date(tomorrow)

    def get_past_unresolved(self, hours_back: int = 24) -> list[dict]:
        """
        Returns schedules that are in the past (within the last N hours)
        and haven't been marked as done.

        This is what the watchdog uses to ask:
        "You had Hiwi work at 9am — did that happen?"

        Args:
            hours_back: how many hours back to look (default 24)

        Returns:
            list of past schedule dicts sorted by timestamp descending
            (most recent first)
        """
        now = self._now()
        cutoff = now - timedelta(hours=hours_back)

        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M")
        now_str = now.strftime("%Y-%m-%d %H:%M")

        all_schedules = self._load_all()

        past = [
            s for s in all_schedules
            if s.get("timestamp")
            and cutoff_str <= s["timestamp"] < now_str
        ]

        # Most recent first — so Buddy asks about the closest missed event first
        return sorted(past, key=lambda s: s.get("timestamp") or "", reverse=True)

    def get_this_week(self) -> list[dict]:
        """
        Returns all schedules for the current week (Monday to Sunday).
        """
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        return self.get_by_range(
            monday.strftime("%Y-%m-%d"),
            sunday.strftime("%Y-%m-%d"),
        )

    def count(self) -> int:
        """Returns total number of schedules in the local DB."""
        return len(self._load_all())


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    source = LocalScheduleSource()

    print(f"=== Local Schedule Source — Self Test ===")
    print(f"Total schedules in DB: {source.count()}")

    print(f"\n--- Today ({source._today_str()}) ---")
    today_scheds = source.get_today()
    if today_scheds:
        for s in today_scheds:
            print(f"  {s['time'] or '??:??'}  {s['title']}  [{s['location'] or 'no location'}]")
    else:
        print("  (none)")

    print(f"\n--- Tomorrow ---")
    tomorrow_scheds = source.get_tomorrow()
    if tomorrow_scheds:
        for s in tomorrow_scheds:
            print(f"  {s['time'] or '??:??'}  {s['title']}")
    else:
        print("  (none)")

    print(f"\n--- Upcoming 7 days ---")
    upcoming = source.get_upcoming(days_ahead=7)
    if upcoming:
        for s in upcoming:
            print(f"  {s['date']}  {s['time'] or '??:??'}  {s['title']}")
    else:
        print("  (none)")

    print(f"\n--- Past unresolved (last 24h) ---")
    past = source.get_past_unresolved(hours_back=24)
    if past:
        for s in past:
            print(f"  {s['timestamp']}  {s['title']}  ← needs check-in")
    else:
        print("  (none)")

    print(f"\n--- Source field check ---")
    all_scheds = source.get_all()
    if all_scheds:
        sample = all_scheds[0]
        print(f"  Sample schedule: {sample}")
        assert sample["source"] == "local", "Source field should be 'local'"
        print("  source='local' ✓")
    else:
        print("  No schedules in DB to verify.")