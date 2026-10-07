"""
brain/adaptive/awareness/schedule_awareness.py

Schedule Awareness — The Watchdog
===================================
Decides WHAT Buddy should say proactively about your schedule and WHEN.
This is the brain of the adaptive layer — it doesn't speak itself,
it returns structured awareness items that agent_core turns into speech.

Three awareness behaviors:
  1. PAST CHECK-IN   — "You had Hiwi work at 9am — did that happen?"
  2. UPCOMING WARN   — "You have a lecture at 10am in 2 hours"
  3. CONFLICT WARN   — "Your Hiwi shift on Friday is on Karfreitag — offices closed"

Usage:
    from brain.adaptive.awareness.schedule_awareness import ScheduleAwareness

    awareness = ScheduleAwareness()

    # Run full startup scan — returns everything Buddy should say
    report = awareness.startup_scan()

    for item in report["items"]:
        print(item["message"])   # TTS-ready string
        print(item["priority"])  # "high" | "medium" | "low"
        print(item["type"])      # "past_checkin" | "upcoming" | "conflict"
"""

from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional
from brain.projects.note_analyzer import scan_todo_deadlines
TZ = ZoneInfo("Europe/Berlin")


class ScheduleAwareness:
    """
    Watchdog that scans Buddy's unified schedule and produces
    structured awareness items for agent_core to speak.

    Designed to run once on Buddy startup.
    Later can run on a background timer too.
    """

    def __init__(self):
        from brain.adaptive.awareness.unified_schedule import UnifiedSchedule
        self._schedule = UnifiedSchedule()

    def _now(self) -> datetime:
        return datetime.now(TZ)

    def _today_str(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    def _parse_ts(self, ts: str) -> Optional[datetime]:
        """Safely parses a timestamp string into a datetime object."""
        if not ts:
            return None
        try:
            # Try full timestamp first
            return datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        except ValueError:
            try:
                # Date only — treat as start of day
                return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=TZ)
            except ValueError:
                return None

    # ── Awareness item builder ───────────────────────────────────────────────

    def _make_item(
        self,
        item_type: str,
        message: str,
        priority: str,
        event: Optional[dict] = None,
        data: Optional[dict] = None,
    ) -> dict:
        """
        Builds a standardized awareness item dict.

        item_type: "past_checkin" | "upcoming" | "conflict" | "holiday_heads_up"
        priority:  "high" | "medium" | "low"
        message:   TTS-ready string
        event:     the schedule event that triggered this (if any)
        data:      extra context for agent_core
        """
        return {
            "type":     item_type,
            "priority": priority,
            "message":  message,
            "event":    event or {},
            "data":     data or {},
        }

    # ── Past check-in scan ───────────────────────────────────────────────────

    def scan_past_checkins(self, hours_back: int = 12) -> list[dict]:
        """
        Finds events from the last N hours that Buddy should check in about.
        "You had Hiwi work at 9am — did that happen?"

        Only triggers if:
        - The event had a specific time (not all-day)
        - The event was in the past but within hours_back
        - The event wasn't already checked in (no status tracking yet —
          for now we just report it and let the user respond)

        Args:
            hours_back: how far back to look (default 12 hours)

        Returns:
            list of awareness items sorted by time descending
        """
        past_events = self._schedule.get_past_unresolved(hours_back=hours_back)
        items = []

        for event in past_events:
            title = event.get("title", "something")
            time_str = event.get("time")
            source = event.get("source", "local")

            # Skip all-day events — no specific time to check in about
            if not time_str:
                continue

            # Build TTS message
            message = (
                f"By the way, you had {title} scheduled at {time_str}. "
                f"Did that happen?"
            )

            items.append(self._make_item(
                item_type="past_checkin",
                priority="medium",
                message=message,
                event=event,
                data={"time_str": time_str, "source": source},
            ))

        return items

    # ── Upcoming event warnings ──────────────────────────────────────────────

    def scan_upcoming_warnings(
        self,
        warn_minutes: list[int] = None,
        days_ahead: int = 1,
    ) -> list[dict]:
        """
        Finds events coming up soon and builds timing warnings.
        "You have a lecture at 10am — that's in 90 minutes."

        Args:
            warn_minutes: list of minute thresholds to warn at
                          default [30, 60, 120] — warn if within 30min, 1hr, 2hr
            days_ahead:   how many days ahead to scan (default 1 — today + tomorrow)

        Returns:
            list of awareness items sorted by urgency
        """
        if warn_minutes is None:
            warn_minutes = [30, 60, 120]

        now = self._now()
        upcoming = self._schedule.get_upcoming(days_ahead=days_ahead)
        items = []

        for event in upcoming:
            ts = event.get("timestamp")
            if not ts or not event.get("time"):
                continue  # skip all-day events

            event_dt = self._parse_ts(ts)
            if not event_dt:
                continue

            # Only warn about future events
            if event_dt <= now:
                continue

            minutes_until = int((event_dt - now).total_seconds() / 60)

            # Check if it falls within any warning threshold
            triggered_threshold = None
            for threshold in sorted(warn_minutes):
                if minutes_until <= threshold:
                    triggered_threshold = threshold
                    break

            if triggered_threshold is None:
                continue

            title = event.get("title", "something")
            time_str = event.get("time")
            holiday_tag = ""
            if event.get("is_holiday"):
                holiday_tag = (
                    f" Note: {event['holiday_name']} is today — "
                    f"double-check if this is still happening."
                )

            # Choose phrasing based on urgency
            if minutes_until <= 30:
                timing = f"in {minutes_until} minutes"
                priority = "high"
            elif minutes_until <= 60:
                timing = "in about an hour"
                priority = "high"
            else:
                hours = round(minutes_until / 60, 1)
                timing = f"in about {int(hours)} hours"
                priority = "medium"

            message = f"Heads up — you have {title} at {time_str}, {timing}.{holiday_tag}"

            items.append(self._make_item(
                item_type="upcoming",
                priority=priority,
                message=message,
                event=event,
                data={
                    "minutes_until":       minutes_until,
                    "triggered_threshold": triggered_threshold,
                },
            ))

        # Sort by urgency — most imminent first
        return sorted(items, key=lambda x: x["data"].get("minutes_until", 9999))

    # ── Holiday conflict scan ────────────────────────────────────────────────

    def scan_holiday_conflicts(self, days_ahead: int = 14) -> list[dict]:
        """
        Finds schedules that conflict with public holidays in NRW.
        "Your Hiwi shift on Friday is on Karfreitag — offices will be closed."

        Args:
            days_ahead: how many days ahead to scan (default 14)

        Returns:
            list of awareness items, high priority
        """
        report = self._schedule.get_conflict_report(days_ahead=days_ahead)
        items = []

        for conflict in report["conflicts"]:
            event = conflict["schedule"]
            holiday_name = conflict["holiday_name"]
            title = event.get("title", "An event")
            date_str = conflict["date"]
            time_str = event.get("time")

            # Build natural date label
            today = self._today_str()
            tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

            if date_str == today:
                day_label = "today"
            elif date_str == tomorrow:
                day_label = "tomorrow"
            else:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    day_label = dt.strftime("on %A, %B %d")
                except Exception:
                    day_label = f"on {date_str}"

            time_label = f" at {time_str}" if time_str else ""

            message = (
                f"Just so you know — {title}{time_label} {day_label} "
                f"falls on {holiday_name}, a public holiday in NRW. "
                f"Most offices and shops will be closed."
            )

            items.append(self._make_item(
                item_type="conflict",
                priority="high",
                message=message,
                event=event,
                data={
                    "holiday_name": holiday_name,
                    "date":         date_str,
                },
            ))

        return items

    # ── Upcoming holiday heads-up ────────────────────────────────────────────

    def scan_upcoming_holidays(self, days_ahead: int = 3) -> list[dict]:
        """
        Warns about public holidays coming up soon even if no schedule
        conflicts with them — so the user knows in advance.
        "By the way, Karfreitag is in 2 days."

        Args:
            days_ahead: how many days ahead to look (default 3)

        Returns:
            list of awareness items, low priority
        """
        from brain.adaptive.awareness.holiday_checker import HolidayChecker
        checker = HolidayChecker()
        upcoming = checker.is_upcoming_holiday(days_ahead=days_ahead)
        items = []

        for holiday in upcoming:
            name = holiday["name"]
            days_until = holiday["days_until"]
            date_str = holiday["date"]

            if days_until == 1:
                timing = "tomorrow"
            elif days_until == 0:
                timing = "today"
            else:
                timing = f"in {days_until} days"

            message = f"Just a heads up — {name} is {timing} ({date_str})."

            # Elevate priority if a schedule conflict exists on that day
            conflicts_on_day = self._schedule.get_by_date(date_str)
            has_conflict = any(e.get("is_holiday") for e in conflicts_on_day)
            priority = "medium" if has_conflict else "low"

            items.append(self._make_item(
                item_type="holiday_heads_up",
                priority=priority,
                message=message,
                data={
                    "holiday_name": name,
                    "date":         date_str,
                    "days_until":   days_until,
                    "has_conflict": has_conflict,
                },
            ))

        return items

    # ── Full startup scan ────────────────────────────────────────────────────

    def startup_scan(
        self,
        hours_back: int = 12,
        upcoming_days: int = 1,
        conflict_days: int = 14,
        holiday_heads_up_days: int = 3,
    ) -> dict:
        """
        Runs all awareness checks and returns a prioritized report.
        This is the single method agent_core calls on Buddy startup.

        Args:
            hours_back:            how far back to look for past check-ins
            upcoming_days:         how many days ahead for timing warnings
            conflict_days:         how many days ahead for holiday conflicts
            holiday_heads_up_days: how many days ahead for holiday heads-up

        Returns:
            {
                "items":        list of awareness items sorted by priority,
                "has_items":    bool,
                "counts": {
                    "past_checkins":    int,
                    "upcoming":         int,
                    "conflicts":        int,
                    "holiday_heads_up": int,
                },
                "spoken_summary": str,  ← single TTS string if items exist
            }
        """
        # Run all scans
        past_items     = self.scan_past_checkins(hours_back=hours_back)
        upcoming_items = self.scan_upcoming_warnings(days_ahead=upcoming_days)
        conflict_items = self.scan_holiday_conflicts(days_ahead=conflict_days)
        holiday_items  = self.scan_upcoming_holidays(days_ahead=holiday_heads_up_days)

        # Merge all items
        todo_items = scan_todo_deadlines(days_ahead=3)
        all_items = conflict_items + past_items + upcoming_items + holiday_items + todo_items

        # Sort by priority: high → medium → low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_items.sort(key=lambda x: priority_order.get(x["priority"], 3))

        # Build spoken summary — agent_core can speak this directly
        spoken_summary = self._build_spoken_summary(all_items)

        return {
            "items":     all_items,
            "has_items": len(all_items) > 0,
            "counts": {
                "past_checkins":    len(past_items),
                "upcoming":         len(upcoming_items),
                "conflicts":        len(conflict_items),
                "holiday_heads_up": len(holiday_items),
                "todo_deadline":    len(todo_items),
            },
            "spoken_summary": spoken_summary,
        }

    # ── Spoken summary builder ───────────────────────────────────────────────

    def _build_spoken_summary(self, items: list[dict]) -> str:
        """
        Builds a single TTS-ready string from all awareness items.
        Limits to 3 most important items to avoid overwhelming the user.
        """
        if not items:
            return ""

        # Take top 3 by priority
        top = items[:3]
        messages = [item["message"] for item in top]

        if len(messages) == 1:
            return messages[0]

        # Join naturally
        return " Also, ".join(messages)


# ── Convenience singleton ────────────────────────────────────────────────────
schedule_awareness = ScheduleAwareness()


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    awareness = ScheduleAwareness()

    print("=== Schedule Awareness — Startup Scan ===\n")
    report = awareness.startup_scan(
        hours_back=12,
        upcoming_days=1,
        conflict_days=14,
        holiday_heads_up_days=3,
    )

    print(f"Total awareness items: {len(report['items'])}")
    print(f"Counts: {report['counts']}")
    print()

    if report["has_items"]:
        print("--- All items (sorted by priority) ---")
        for i, item in enumerate(report["items"], 1):
            print(f"  [{i}] [{item['priority'].upper()}] [{item['type']}]")
            print(f"       {item['message']}")
            print()

        print("--- Spoken summary (what agent_core will say) ---")
        print(f"  {report['spoken_summary']}")
    else:
        print("Nothing to report — no awareness items generated.")

    print("\n--- Individual scan tests ---")

    print("\nPast check-ins (last 12h):")
    past = awareness.scan_past_checkins(hours_back=12)
    print(f"  {len(past)} item(s)")
    for item in past:
        print(f"  → {item['message']}")

    print("\nUpcoming warnings (next 24h):")
    upcoming = awareness.scan_upcoming_warnings(days_ahead=1)
    print(f"  {len(upcoming)} item(s)")
    for item in upcoming:
        print(f"  → {item['message']}")

    print("\nHoliday conflicts (next 14 days):")
    conflicts = awareness.scan_holiday_conflicts(days_ahead=14)
    print(f"  {len(conflicts)} item(s)")
    for item in conflicts:
        print(f"  → {item['message']}")

    print("\nUpcoming holiday heads-up (next 3 days):")
    holidays = awareness.scan_upcoming_holidays(days_ahead=3)
    print(f"  {len(holidays)} item(s)")
    for item in holidays:
        print(f"  → {item['message']}")