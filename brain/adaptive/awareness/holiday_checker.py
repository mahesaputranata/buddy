"""
brain/adaptive/awareness/holiday_checker.py

German Public Holiday Checker — Nordrhein-Westfalen (NRW)
==========================================================
Covers:
  - All national (bundesweit) German public holidays
  - NRW-specific holidays on top

Usage:
    from brain.adaptive.awareness.holiday_checker import HolidayChecker

    checker = HolidayChecker()

    # Simple check
    is_holiday, name = checker.is_holiday("2026-04-03")
    # → (True, "Karfreitag")

    # Check a schedule dict directly
    result = checker.check_schedule({"title": "Hiwi work", "timestamp": "2026-04-03 09:00"})
    # → {"is_holiday": True, "holiday_name": "Karfreitag", "warning": "..."}

    # Get all holidays for a year
    holidays = checker.get_all_holidays(2026)

    # Check upcoming schedules for conflicts
    conflicts = checker.find_conflicts(schedules_list)
"""

from datetime import date, timedelta
from typing import Optional


class HolidayChecker:
    """
    Calculates German public holidays for NRW for any given year.
    All holidays are computed dynamically from Easter — no hardcoded year lists.
    """

    STATE = "NRW"  # Nordrhein-Westfalen

    def __init__(self):
        # Cache computed holidays per year so we don't recalculate on every call
        self._cache: dict[int, dict[str, str]] = {}

    # ── Core Easter calculation (Gauss algorithm) ────────────────────────────

    @staticmethod
    def _easter(year: int) -> date:
        """
        Computes Easter Sunday for a given year using the anonymous Gregorian algorithm.
        Works for years 1583-4099.
        """
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)

    # ── Holiday computation ──────────────────────────────────────────────────

    def _compute_holidays(self, year: int) -> dict[str, str]:
        """
        Returns a dict of {date_str: holiday_name} for the given year.
        Includes national holidays + NRW-specific holidays.
        """
        easter = self._easter(year)

        holidays = {}

        def add(d: date, name: str):
            holidays[d.strftime("%Y-%m-%d")] = name

        # ── National holidays (bundesweit) ───────────────────────────────────
        add(date(year, 1, 1),   "Neujahr")
        add(easter - timedelta(days=2), "Karfreitag")
        add(easter,                      "Ostersonntag")
        add(easter + timedelta(days=1),  "Ostermontag")
        add(date(year, 5, 1),   "Tag der Arbeit")
        add(easter + timedelta(days=39), "Christi Himmelfahrt")
        add(easter + timedelta(days=49), "Pfingstsonntag")
        add(easter + timedelta(days=50), "Pfingstmontag")
        add(date(year, 10, 3),  "Tag der Deutschen Einheit")
        add(date(year, 12, 25), "1. Weihnachtstag")
        add(date(year, 12, 26), "2. Weihnachtstag")

        # ── NRW-specific holidays ────────────────────────────────────────────
        # Fronleichnam — 60 days after Easter Sunday
        add(easter + timedelta(days=60), "Fronleichnam")

        # Allerheiligen — November 1st
        add(date(year, 11, 1), "Allerheiligen")

        return holidays

    def _get_holidays(self, year: int) -> dict[str, str]:
        """Returns holidays for year, using cache."""
        if year not in self._cache:
            self._cache[year] = self._compute_holidays(year)
        return self._cache[year]

    # ── Public API ───────────────────────────────────────────────────────────

    def is_holiday(self, date_str: str) -> tuple[bool, Optional[str]]:
        """
        Check if a given date is a public holiday in NRW.

        Args:
            date_str: Date in "YYYY-MM-DD" format

        Returns:
            (True, "Holiday Name") if it's a holiday
            (False, None) if it's not

        Example:
            is_holiday, name = checker.is_holiday("2026-04-03")
            # → (True, "Karfreitag")
        """
        if not date_str:
            return False, None

        try:
            # Accept both "YYYY-MM-DD" and "YYYY-MM-DD HH:MM"
            date_only = date_str.strip()[:10]
            year = int(date_only[:4])
            holidays = self._get_holidays(year)
            name = holidays.get(date_only)
            return (True, name) if name else (False, None)
        except Exception:
            return False, None

    def get_holiday_name(self, date_str: str) -> Optional[str]:
        """Returns holiday name if date is a holiday, else None."""
        _, name = self.is_holiday(date_str)
        return name

    def get_all_holidays(self, year: int) -> dict[str, str]:
        """
        Returns all NRW public holidays for a given year.

        Returns:
            {"YYYY-MM-DD": "Holiday Name", ...} sorted by date
        """
        holidays = self._get_holidays(year)
        return dict(sorted(holidays.items()))

    def check_schedule(self, schedule: dict) -> dict:
        """
        Checks a single schedule dict for holiday conflicts.

        Args:
            schedule: dict with at least "timestamp" or "date" field
                      e.g. {"title": "Hiwi work", "timestamp": "2026-04-03 09:00"}

        Returns:
            {
                "is_holiday": bool,
                "holiday_name": str | None,
                "warning": str | None,    ← ready to speak via TTS
                "date": str,
            }

        Example:
            result = checker.check_schedule({
                "title": "Hiwi work",
                "timestamp": "2026-04-03 09:00"
            })
            # → {
            #     "is_holiday": True,
            #     "holiday_name": "Karfreitag",
            #     "warning": "Heads up — Hiwi work falls on Karfreitag, a public holiday in NRW.",
            #     "date": "2026-04-03"
            #   }
        """
        # Extract date from schedule — try timestamp first, then date field
        timestamp = schedule.get("timestamp") or ""
        date_field = schedule.get("date") or ""
        date_str = (timestamp or date_field or "").strip()[:10]

        if not date_str:
            return {
                "is_holiday": False,
                "holiday_name": None,
                "warning": None,
                "date": None,
            }

        is_hol, holiday_name = self.is_holiday(date_str)

        warning = None
        if is_hol:
            title = (schedule.get("title") or "This event").strip()
            warning = (
                f"Heads up — {title} falls on {holiday_name}, "
                f"a public holiday in NRW. "
                f"Most offices and shops will be closed."
            )

        return {
            "is_holiday": is_hol,
            "holiday_name": holiday_name,
            "warning": warning,
            "date": date_str,
        }

    def find_conflicts(self, schedules: list[dict]) -> list[dict]:
        """
        Scans a list of schedule dicts and returns only those that fall on holidays.

        Args:
            schedules: list of schedule dicts from get_all_schedules_as_dicts()

        Returns:
            list of conflict dicts, each containing the original schedule
            plus holiday info:
            [
                {
                    "schedule": {...},        ← original schedule dict
                    "holiday_name": "...",
                    "warning": "...",
                    "date": "YYYY-MM-DD",
                },
                ...
            ]

        Example:
            all_schedules = get_all_schedules_as_dicts()
            conflicts = checker.find_conflicts(all_schedules)
            for c in conflicts:
                print(c["warning"])
        """
        conflicts = []
        for sched in (schedules or []):
            result = self.check_schedule(sched)
            if result["is_holiday"]:
                conflicts.append({
                    "schedule": sched,
                    "holiday_name": result["holiday_name"],
                    "warning": result["warning"],
                    "date": result["date"],
                })
        return conflicts

    def is_upcoming_holiday(
        self,
        days_ahead: int = 3,
        reference_date: Optional[date] = None,
    ) -> list[dict]:
        """
        Returns any public holidays coming up in the next N days.
        Useful for proactive morning warnings like:
        "By the way, Karfreitag is in 2 days."

        Args:
            days_ahead: how many days to look ahead (default 3)
            reference_date: date to count from (default today)

        Returns:
            list of {"date": "YYYY-MM-DD", "name": "...", "days_until": int}
        """
        ref = reference_date or date.today()
        upcoming = []

        for i in range(1, days_ahead + 1):
            check_date = ref + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            is_hol, name = self.is_holiday(date_str)
            if is_hol:
                upcoming.append({
                    "date": date_str,
                    "name": name,
                    "days_until": i,
                })

        return upcoming


# ── Convenience singleton ────────────────────────────────────────────────────
holiday_checker = HolidayChecker()


# ── Quick self-test (run directly: python holiday_checker.py) ────────────────

if __name__ == "__main__":
    checker = HolidayChecker()

    print("=== NRW Public Holidays 2026 ===")
    for date_str, name in checker.get_all_holidays(2026).items():
        print(f"  {date_str}  {name}")

    print("\n=== Holiday checks ===")
    test_dates = [
        ("2026-04-03", "Karfreitag"),
        ("2026-04-06", "Ostermontag"),
        ("2026-05-01", "Tag der Arbeit"),
        ("2026-06-04", "Fronleichnam — NRW specific"),
        ("2026-11-01", "Allerheiligen — NRW specific"),
        ("2026-04-07", "Normal Tuesday — not a holiday"),
        ("2026-12-25", "1. Weihnachtstag"),
    ]
    for date_str, label in test_dates:
        is_hol, name = checker.is_holiday(date_str)
        status = f"HOLIDAY: {name}" if is_hol else "not a holiday"
        print(f"  {date_str}  [{label}]  →  {status}")

    print("\n=== Schedule conflict check ===")
    fake_schedules = [
        {"title": "Hiwi work", "timestamp": "2026-04-03 09:00"},
        {"title": "Gym session", "timestamp": "2026-04-07 08:00"},
        {"title": "Team meeting", "timestamp": "2026-05-01 10:00"},
    ]
    conflicts = checker.find_conflicts(fake_schedules)
    if conflicts:
        for c in conflicts:
            print(f"  ⚠  {c['warning']}")
    else:
        print("  No conflicts found.")

    print("\n=== Upcoming holidays (next 7 days from today) ===")
    upcoming = checker.is_upcoming_holiday(days_ahead=7)
    if upcoming:
        for h in upcoming:
            print(f"  {h['name']} in {h['days_until']} day(s) ({h['date']})")
    else:
        print("  No holidays in the next 7 days.")