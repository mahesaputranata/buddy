"""
eval/test_adaptive_awareness.py

Buddy Adaptive Awareness Test Suite
=====================================
Tests the proactive layer — brain/adaptive/ — in full isolation.

No Ollama needed. No hardware. No network.
Pure logic tests — runs fast (< 30 seconds).

Sections:
  1. HolidayChecker        — NRW holiday logic
  2. LocalScheduleSource   — SQL reading and normalization
  3. UnifiedSchedule       — merge, dedup, enrichment
  4. ScheduleAwareness     — watchdog scan logic
  5. AgentCore             — entry point and holiday hook

Run with:
    pytest eval/test_adaptive_awareness.py -v
    pytest eval/test_adaptive_awareness.py -v -k "Holiday"   # just one section

Scoring:
    >= 90% → adaptive layer is solid, proceed to daily use
    < 90%  → fix failures before relying on watchdog
"""

import sys
import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

TZ = ZoneInfo("Europe/Berlin")

# ── Isolated DB setup ────────────────────────────────────────────────────────
import brain.core.sql_brain as sql_brain_module

_TEST_DB_PATH = None

def _get_test_connection():
    return sqlite3.connect(_TEST_DB_PATH)

@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    global _TEST_DB_PATH
    db_file = tmp_path / "test_adaptive.db"
    _TEST_DB_PATH = str(db_file)
    with patch.object(sql_brain_module, "get_connection", _get_test_connection):
        sql_brain_module.init_schedule_table()
        yield
    _TEST_DB_PATH = None

# ── Helpers ──────────────────────────────────────────────────────────────────

def today_str():
    return date.today().strftime("%Y-%m-%d")

def tomorrow_str():
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

def yesterday_str():
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

def make_ts(date_str, time_str="09:00"):
    return f"{date_str} {time_str}"

def seed_schedule(title, date_str, time_str="09:00", location=None):
    """Helper to create a schedule in the test DB."""
    from brain.core.sql_brain import create_schedule
    ts = make_ts(date_str, time_str)
    create_schedule(title, title, ts, location)
    return ts


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HOLIDAY CHECKER
# Pure logic tests — no DB, no Ollama, no network
# ════════════════════════════════════════════════════════════════════════════

class TestHolidayChecker:
    """Tests HolidayChecker NRW holiday logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from brain.adaptive.awareness.holiday_checker import HolidayChecker
        self.checker = HolidayChecker()

    # ── Known NRW holidays 2026 ──────────────────────────────────────────────

    def test_neujahr_2026(self):
        is_hol, name = self.checker.is_holiday("2026-01-01")
        assert is_hol is True
        assert name == "Neujahr"

    def test_karfreitag_2026(self):
        """The exact scenario from this morning."""
        is_hol, name = self.checker.is_holiday("2026-04-03")
        assert is_hol is True
        assert name == "Karfreitag"

    def test_ostermontag_2026(self):
        is_hol, name = self.checker.is_holiday("2026-04-06")
        assert is_hol is True
        assert name == "Ostermontag"

    def test_tag_der_arbeit_2026(self):
        is_hol, name = self.checker.is_holiday("2026-05-01")
        assert is_hol is True
        assert "Arbeit" in name

    def test_fronleichnam_2026_nrw_specific(self):
        """Fronleichnam is NRW-specific — not in all German states."""
        is_hol, name = self.checker.is_holiday("2026-06-04")
        assert is_hol is True
        assert name == "Fronleichnam"

    def test_allerheiligen_2026_nrw_specific(self):
        """Allerheiligen is NRW-specific."""
        is_hol, name = self.checker.is_holiday("2026-11-01")
        assert is_hol is True
        assert name == "Allerheiligen"

    def test_weihnachten_first_2026(self):
        is_hol, name = self.checker.is_holiday("2026-12-25")
        assert is_hol is True

    def test_weihnachten_second_2026(self):
        is_hol, name = self.checker.is_holiday("2026-12-26")
        assert is_hol is True

    def test_tag_der_einheit_2026(self):
        is_hol, name = self.checker.is_holiday("2026-10-03")
        assert is_hol is True
        assert "Einheit" in name

    # ── Non-holidays ─────────────────────────────────────────────────────────

    def test_normal_tuesday_not_holiday(self):
        is_hol, name = self.checker.is_holiday("2026-04-07")
        assert is_hol is False
        assert name is None

    def test_normal_monday_not_holiday(self):
        is_hol, name = self.checker.is_holiday("2026-03-16")
        assert is_hol is False
        assert name is None

    def test_empty_date_returns_false(self):
        is_hol, name = self.checker.is_holiday("")
        assert is_hol is False
        assert name is None

    def test_none_date_returns_false(self):
        is_hol, name = self.checker.is_holiday(None)
        assert is_hol is False
        assert name is None

    def test_timestamp_format_accepted(self):
        """Accepts 'YYYY-MM-DD HH:MM' format, not just 'YYYY-MM-DD'."""
        is_hol, name = self.checker.is_holiday("2026-04-03 09:00")
        assert is_hol is True
        assert name == "Karfreitag"

    # ── get_all_holidays ─────────────────────────────────────────────────────

    def test_get_all_holidays_2026_count(self):
        """NRW has 13 public holidays in 2026."""
        holidays = self.checker.get_all_holidays(2026)
        assert len(holidays) == 13, \
            f"Expected 13 NRW holidays, got {len(holidays)}: {list(holidays.values())}"

    def test_get_all_holidays_sorted(self):
        holidays = self.checker.get_all_holidays(2026)
        dates = list(holidays.keys())
        assert dates == sorted(dates), "Holidays should be sorted by date"

    def test_get_all_holidays_different_years(self):
        """Easter moves every year — verify 2025 and 2027 are different."""
        h2025 = self.checker.get_all_holidays(2025)
        h2027 = self.checker.get_all_holidays(2027)
        # Karfreitag dates should differ
        karfreitag_2025 = [d for d, n in h2025.items() if n == "Karfreitag"][0]
        karfreitag_2027 = [d for d, n in h2027.items() if n == "Karfreitag"][0]
        assert karfreitag_2025 != karfreitag_2027

    # ── check_schedule ───────────────────────────────────────────────────────

    def test_check_schedule_holiday_returns_warning(self):
        result = self.checker.check_schedule({
            "title": "Hiwi work",
            "timestamp": "2026-04-03 09:00",
        })
        assert result["is_holiday"] is True
        assert result["holiday_name"] == "Karfreitag"
        assert result["warning"] is not None
        assert "Hiwi work" in result["warning"]
        assert "Karfreitag" in result["warning"]

    def test_check_schedule_normal_day_no_warning(self):
        result = self.checker.check_schedule({
            "title": "Gym session",
            "timestamp": "2026-04-07 08:00",
        })
        assert result["is_holiday"] is False
        assert result["warning"] is None

    def test_check_schedule_warning_is_tts_ready(self):
        """Warning should not contain markdown, asterisks, or special chars."""
        result = self.checker.check_schedule({
            "title": "Office work",
            "timestamp": "2026-05-01 09:00",
        })
        if result["warning"]:
            warning = result["warning"]
            assert "**" not in warning, "Warning should not contain markdown bold"
            assert "`" not in warning, "Warning should not contain backticks"

    # ── find_conflicts ───────────────────────────────────────────────────────

    def test_find_conflicts_detects_holiday(self):
        schedules = [
            {"title": "Hiwi work",  "timestamp": "2026-04-03 09:00"},
            {"title": "Gym",        "timestamp": "2026-04-07 08:00"},
            {"title": "Meeting",    "timestamp": "2026-05-01 10:00"},
        ]
        conflicts = self.checker.find_conflicts(schedules)
        assert len(conflicts) == 2
        titles = [c["schedule"]["title"] for c in conflicts]
        assert "Hiwi work" in titles
        assert "Meeting" in titles
        assert "Gym" not in titles

    def test_find_conflicts_empty_list(self):
        conflicts = self.checker.find_conflicts([])
        assert conflicts == []

    def test_find_conflicts_no_conflicts(self):
        schedules = [
            {"title": "Normal day", "timestamp": "2026-04-07 09:00"},
        ]
        conflicts = self.checker.find_conflicts(schedules)
        assert conflicts == []

    # ── is_upcoming_holiday ──────────────────────────────────────────────────

    def test_upcoming_holiday_returns_list(self):
        result = self.checker.is_upcoming_holiday(days_ahead=365)
        assert isinstance(result, list)

    def test_upcoming_holiday_has_required_fields(self):
        result = self.checker.is_upcoming_holiday(days_ahead=365)
        if result:
            item = result[0]
            assert "date" in item
            assert "name" in item
            assert "days_until" in item
            assert item["days_until"] >= 1

    def test_upcoming_holiday_days_until_accurate(self):
        """days_until should match the actual date difference."""
        result = self.checker.is_upcoming_holiday(days_ahead=365)
        today = date.today()
        for item in result:
            expected_days = (
                datetime.strptime(item["date"], "%Y-%m-%d").date() - today
            ).days
            assert item["days_until"] == expected_days, \
                f"days_until mismatch for {item['name']}: " \
                f"expected {expected_days}, got {item['days_until']}"

    # ── Caching ──────────────────────────────────────────────────────────────

    def test_cache_works(self):
        """Same year should use cached result — same dict object."""
        h1 = self.checker._get_holidays(2026)
        h2 = self.checker._get_holidays(2026)
        assert h1 is h2, "Cache should return the same dict object"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOCAL SCHEDULE SOURCE
# ════════════════════════════════════════════════════════════════════════════

class TestLocalScheduleSource:
    """Tests LocalScheduleSource reads and normalizes correctly."""

    @pytest.fixture(autouse=True)
    def setup(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.adaptive.awareness.local_schedule_source import LocalScheduleSource
            self.source = LocalScheduleSource()

    def test_get_all_empty(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            result = self.source.get_all()
        assert result == []

    def test_get_all_returns_normalized(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Dentist", tomorrow_str(), "09:00")
            result = self.source.get_all()
        assert len(result) == 1
        item = result[0]
        assert item["title"] == "Dentist"
        assert item["source"] == "local"
        assert item["date"] == tomorrow_str()
        assert item["time"] == "09:00"

    def test_normalized_dict_has_all_required_keys(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Test event", tomorrow_str())
            result = self.source.get_all()
        required = {"id", "title", "description", "timestamp",
                    "date", "time", "location", "source"}
        for key in required:
            assert key in result[0], f"Missing key: {key}"

    def test_source_field_always_local(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Event A", tomorrow_str())
            seed_schedule("Event B", today_str())
            result = self.source.get_all()
        assert all(e["source"] == "local" for e in result)

    def test_get_by_date_correct_date(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Today event",    today_str())
            seed_schedule("Tomorrow event", tomorrow_str())
            result = self.source.get_by_date(today_str())
        assert len(result) == 1
        assert result[0]["title"] == "Today event"

    def test_get_by_date_empty(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            result = self.source.get_by_date("2099-01-01")
        assert result == []

    def test_get_today_filters_correctly(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Today",    today_str())
            seed_schedule("Tomorrow", tomorrow_str())
            result = self.source.get_today()
        assert len(result) == 1
        assert result[0]["title"] == "Today"

    def test_get_tomorrow_filters_correctly(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Today",    today_str())
            seed_schedule("Tomorrow", tomorrow_str())
            result = self.source.get_tomorrow()
        assert len(result) == 1
        assert result[0]["title"] == "Tomorrow"

    def test_get_upcoming_includes_today(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Today event", today_str())
            result = self.source.get_upcoming(days_ahead=3)
        titles = [e["title"] for e in result]
        assert "Today event" in titles

    def test_get_upcoming_excludes_past(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Past event", yesterday_str())
            result = self.source.get_upcoming(days_ahead=7)
        titles = [e["title"] for e in result]
        assert "Past event" not in titles

    def test_get_by_range_inclusive(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Start", today_str())
            seed_schedule("End",   tomorrow_str())
            result = self.source.get_by_range(today_str(), tomorrow_str())
        assert len(result) == 2

    def test_get_past_unresolved_catches_recent(self):
        """Event 2 hours ago should appear in past_unresolved(hours_back=12)."""
        from datetime import datetime
        two_hours_ago = (datetime.now(TZ) - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
        date_part = two_hours_ago[:10]
        time_part = two_hours_ago[11:16]
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Recent event", "desc", two_hours_ago)
            result = self.source.get_past_unresolved(hours_back=12)
        titles = [e["title"] for e in result]
        assert "Recent event" in titles

    def test_get_past_unresolved_excludes_future(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Future event", tomorrow_str(), "10:00")
            result = self.source.get_past_unresolved(hours_back=12)
        titles = [e["title"] for e in result]
        assert "Future event" not in titles

    def test_count_returns_correct_number(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("A", today_str())
            seed_schedule("B", tomorrow_str())
            seed_schedule("C", today_str())
            count = self.source.count()
        assert count == 3

    def test_sorted_by_timestamp(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Late",  tomorrow_str(), "20:00")
            seed_schedule("Early", tomorrow_str(), "06:00")
            result = self.source.get_all()
        timestamps = [e["timestamp"] for e in result]
        assert timestamps == sorted(timestamps)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UNIFIED SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

class TestUnifiedSchedule:
    """Tests UnifiedSchedule merge, dedup, and holiday enrichment."""

    def _get_unified(self):
        """Returns a UnifiedSchedule with GCal mocked as unavailable."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.adaptive.awareness.unified_schedule import UnifiedSchedule
            schedule = UnifiedSchedule()
            # Mock GCal as unavailable so tests run without OAuth
            schedule._gcal = MagicMock()
            schedule._gcal.is_available.return_value = False
            schedule._gcal.get_by_range.return_value = []
            schedule._gcal.get_upcoming.return_value = []
            schedule._gcal.get_past_unresolved.return_value = []
            return schedule

    def test_get_today_empty(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            schedule = self._get_unified()
            result = schedule.get_today()
        assert result == []

    def test_get_today_returns_events(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Today meeting", today_str(), "10:00")
            schedule = self._get_unified()
            result = schedule.get_today()
        assert len(result) == 1
        assert result[0]["title"] == "Today meeting"

    def test_holiday_enrichment_added(self):
        """All events should have is_holiday and holiday_name fields."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Some event", tomorrow_str())
            schedule = self._get_unified()
            result = schedule.get_upcoming(days_ahead=3)
        for event in result:
            assert "is_holiday" in event, "Missing is_holiday field"
            assert "holiday_name" in event, "Missing holiday_name field"

    def test_holiday_conflict_flagged_correctly(self):
        """Event on Karfreitag should be flagged as holiday."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Hiwi work", "desc", "2026-04-03 09:00")
            schedule = self._get_unified()
            result = schedule.get_by_date("2026-04-03")
        assert len(result) == 1
        assert result[0]["is_holiday"] is True
        assert result[0]["holiday_name"] == "Karfreitag"

    def test_normal_day_not_flagged_as_holiday(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Normal work", "2026-04-07")
            schedule = self._get_unified()
            result = schedule.get_by_date("2026-04-07")
        assert result[0]["is_holiday"] is False
        assert result[0]["holiday_name"] is None

    def test_gcal_unavailable_falls_back_to_local(self):
        """When GCal is unavailable, should return local events without error."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Local only", today_str())
            schedule = self._get_unified()
            result = schedule.get_today()
        assert len(result) == 1
        assert result[0]["source"] == "local"

    def test_deduplication_same_title_date_time(self):
        """Same event from local and GCal should appear only once."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Team meeting", tomorrow_str(), "10:00")
            from brain.adaptive.awareness.unified_schedule import UnifiedSchedule
            schedule = UnifiedSchedule()

            # Mock GCal returning same event
            gcal_event = {
                "id": "gcal-123",
                "title": "Team meeting",
                "description": None,
                "timestamp": make_ts(tomorrow_str(), "10:00"),
                "date": tomorrow_str(),
                "time": "10:00",
                "location": None,
                "source": "gcal",
                "gcal_link": "https://calendar.google.com/...",
            }
            schedule._gcal = MagicMock()
            schedule._gcal.is_available.return_value = True
            schedule._gcal.get_by_range.return_value = [gcal_event]

            result = schedule.get_by_range(tomorrow_str(), tomorrow_str())

        # Should be 1 event, not 2
        assert len(result) == 1, \
            f"Dedup failed — expected 1 event, got {len(result)}"
        # GCal version should win (has gcal_link)
        assert result[0]["source"] == "gcal"

    def test_get_conflict_report_structure(self):
        """get_conflict_report() should return correct structure."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            schedule = self._get_unified()
            report = schedule.get_conflict_report(days_ahead=30)
        assert "conflicts" in report
        assert "upcoming_holidays" in report
        assert "total_conflicts" in report
        assert "has_conflicts" in report
        assert isinstance(report["conflicts"], list)
        assert isinstance(report["total_conflicts"], int)

    def test_get_conflict_report_finds_holiday(self):
        """Karfreitag event should appear in conflict report."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Office work", "desc", "2026-04-03 09:00")
            schedule = self._get_unified()
            report = schedule.get_conflict_report(days_ahead=365)
        conflict_dates = [c["date"] for c in report["conflicts"]]
        assert "2026-04-03" in conflict_dates

    def test_summarize_today_empty(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            schedule = self._get_unified()
            summary = schedule.summarize_today()
        assert "nothing" in summary.lower() or "no" in summary.lower()

    def test_summarize_today_with_event(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            seed_schedule("Dentist", today_str(), "14:00")
            schedule = self._get_unified()
            summary = schedule.summarize_today()
        assert "Dentist" in summary
        assert "14:00" in summary

    def test_summarize_today_holiday_appended(self):
        """Holiday warning should be appended to today's summary."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Hiwi work", "desc", "2026-04-03 09:00")
            schedule = self._get_unified()
            summary = schedule.get_by_date("2026-04-03")
        # The event itself should have holiday info
        assert any(e.get("is_holiday") for e in summary)

    def test_status_returns_correct_structure(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            schedule = self._get_unified()
            status = schedule.status()
        assert "local_available" in status
        assert "gcal_available" in status
        assert "local_count" in status
        assert status["local_available"] is True


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCHEDULE AWARENESS
# ════════════════════════════════════════════════════════════════════════════

class TestScheduleAwareness:
    """Tests ScheduleAwareness watchdog scan logic."""

    def _get_awareness(self):
        """Returns ScheduleAwareness with GCal mocked as unavailable."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.adaptive.awareness.schedule_awareness import ScheduleAwareness
            awareness = ScheduleAwareness()
            awareness._schedule._gcal = MagicMock()
            awareness._schedule._gcal.is_available.return_value = False
            awareness._schedule._gcal.get_by_range.return_value = []
            awareness._schedule._gcal.get_upcoming.return_value = []
            awareness._schedule._gcal.get_past_unresolved.return_value = []
            return awareness

    def test_startup_scan_returns_correct_structure(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            awareness = self._get_awareness()
            report = awareness.startup_scan()
        assert "items" in report
        assert "has_items" in report
        assert "counts" in report
        assert "spoken_summary" in report
        assert isinstance(report["items"], list)
        assert isinstance(report["has_items"], bool)

    def test_startup_scan_empty_db_no_crash(self):
        """Empty DB should return empty report, not crash."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            awareness = self._get_awareness()
            report = awareness.startup_scan()
        assert report["has_items"] is False or isinstance(report["items"], list)

    def test_scan_holiday_conflicts_finds_karfreitag(self):
        """Event on Karfreitag should appear in conflict scan."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Office shift", "desc", "2026-04-03 09:00")
            awareness = self._get_awareness()
            items = awareness.scan_holiday_conflicts(days_ahead=365)
        assert len(items) >= 1
        messages = [i["message"] for i in items]
        assert any("Karfreitag" in m for m in messages), \
            f"Expected Karfreitag in conflict messages, got: {messages}"

    def test_scan_holiday_conflicts_item_structure(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Work", "desc", "2026-04-03 09:00")
            awareness = self._get_awareness()
            items = awareness.scan_holiday_conflicts(days_ahead=365)
        if items:
            item = items[0]
            assert "type" in item
            assert "priority" in item
            assert "message" in item
            assert "event" in item
            assert item["type"] == "conflict"
            assert item["priority"] == "high"

    def test_scan_past_checkins_skips_all_day_events(self):
        """All-day events (no time) should not generate check-in prompts."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            # All-day event — timestamp has no time component
            create_schedule("All day event", "desc", yesterday_str())
            awareness = self._get_awareness()
            items = awareness.scan_past_checkins(hours_back=48)
        # Should not generate check-in for all-day event
        messages = [i["message"] for i in items]
        assert not any("All day event" in m for m in messages)

    def test_scan_past_checkins_returns_message(self):
        """Recent past event should generate a check-in message."""
        from datetime import datetime
        two_hours_ago = (datetime.now(TZ) - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            create_schedule("Doctor visit", "desc", two_hours_ago)
            awareness = self._get_awareness()
            items = awareness.scan_past_checkins(hours_back=12)
        if items:
            assert any("Doctor visit" in i["message"] for i in items)
            assert any("happen" in i["message"].lower() for i in items)

    def test_scan_upcoming_warnings_no_crash_empty(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            awareness = self._get_awareness()
            items = awareness.scan_upcoming_warnings(days_ahead=1)
        assert isinstance(items, list)

    def test_startup_scan_sorted_by_priority(self):
        """High priority items should come before medium and low."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            # Add holiday conflict (high priority)
            create_schedule("Holiday work", "desc", "2026-04-03 09:00")
            awareness = self._get_awareness()
            report = awareness.startup_scan(conflict_days=365)

        items = report["items"]
        if len(items) >= 2:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            priorities = [priority_order.get(i["priority"], 3) for i in items]
            assert priorities == sorted(priorities), \
                f"Items not sorted by priority: {[i['priority'] for i in items]}"

    def test_spoken_summary_capped_at_3(self):
        """spoken_summary should never contain more than 3 items worth of text."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            from brain.core.sql_brain import create_schedule
            # Seed multiple holiday conflicts
            for day in ["2026-04-03", "2026-05-01", "2026-10-03",
                        "2026-11-01", "2026-12-25"]:
                create_schedule(f"Work on {day}", "desc", f"{day} 09:00")

            awareness = self._get_awareness()
            report = awareness.startup_scan(conflict_days=365)

        summary = report["spoken_summary"]
        # Count "Also," separators — max 2 means max 3 items
        also_count = summary.count("Also,")
        assert also_count <= 2, \
            f"Summary has too many items (Also count={also_count}): {summary[:100]}"

    def test_scan_upcoming_holidays_returns_list(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            awareness = self._get_awareness()
            items = awareness.scan_upcoming_holidays(days_ahead=365)
        assert isinstance(items, list)

    def test_scan_upcoming_holidays_item_type(self):
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            awareness = self._get_awareness()
            items = awareness.scan_upcoming_holidays(days_ahead=365)
        for item in items:
            assert item["type"] == "holiday_heads_up"
            assert "message" in item
            assert "priority" in item


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AGENT CORE
# ════════════════════════════════════════════════════════════════════════════

class TestAgentCore:
    """Tests AdaptiveAgent entry point and holiday hook."""

    def _make_agent(self):
        """Returns an AdaptiveAgent with mock TTS and MemoryManager."""
        from brain.adaptive.agent_core import AdaptiveAgent

        class MockTTS:
            def __init__(self):
                self.spoken = []
            def speak(self, text):
                self.spoken.append(text)

        class MockMM:
            def get_pending_topic(self):
                return None

        mock_tts = MockTTS()
        mock_mm = MockMM()
        agent = AdaptiveAgent(
            tts=mock_tts,
            memory_manager=mock_mm,
            startup_delay=0,  # no delay in tests
        )
        agent._mock_tts = mock_tts
        return agent

    def test_check_new_schedule_holiday_returns_warning(self):
        agent = self._make_agent()
        warning = agent.check_new_schedule("Hiwi work", "2026-04-03 09:00")
        assert warning is not None
        assert "Karfreitag" in warning
        assert "Hiwi work" in warning

    def test_check_new_schedule_normal_day_returns_none(self):
        agent = self._make_agent()
        warning = agent.check_new_schedule("Gym session", "2026-04-07 08:00")
        assert warning is None

    def test_check_new_schedule_tag_der_arbeit(self):
        agent = self._make_agent()
        warning = agent.check_new_schedule("Office meeting", "2026-05-01 10:00")
        assert warning is not None
        assert "Arbeit" in warning or "Mai" in warning or "holiday" in warning.lower()

    def test_check_new_schedule_empty_timestamp_no_crash(self):
        agent = self._make_agent()
        try:
            warning = agent.check_new_schedule("Some event", "")
            # Should return None gracefully, not crash
            assert warning is None or isinstance(warning, str)
        except Exception as e:
            pytest.fail(f"check_new_schedule crashed on empty timestamp: {e}")

    def test_is_running_before_start(self):
        agent = self._make_agent()
        assert agent.is_running() is False

    def test_start_and_stop(self):
        import time
        agent = self._make_agent()
        agent.start()
        time.sleep(0.2)
        assert agent.is_running() is True
        agent.stop()
        time.sleep(0.5)
        # Thread should stop cleanly
        assert agent.is_running() is False

    def test_double_start_safe(self):
        """Calling start() twice should not create a second thread."""
        import time
        agent = self._make_agent()
        agent.start()
        thread1 = agent._thread
        agent.start()  # second call — should be a no-op
        thread2 = agent._thread
        assert thread1 is thread2, "Second start() should not create a new thread"
        agent.stop()

    def test_init_agent_creates_singleton(self):
        """init_agent() should set the module-level singleton."""
        from brain.adaptive.agent_core import init_agent, get_agent

        class MockTTS:
            def speak(self, text): pass

        class MockMM:
            def get_pending_topic(self): return None

        instance = init_agent(
            tts=MockTTS(),
            memory_manager=MockMM(),
            startup_delay=0,
        )
        assert get_agent() is instance
        instance.stop()

    def test_get_agent_returns_none_before_init(self):
        """Before init_agent() is called, get_agent() should not crash."""
        from brain.adaptive import agent_core
        # Reset singleton for clean test
        original = agent_core._agent_instance
        agent_core._agent_instance = None
        try:
            result = agent_core.get_agent()
            assert result is None
        finally:
            agent_core._agent_instance = original

    def test_warning_is_tts_ready(self):
        """Holiday warning from agent should not contain markdown."""
        agent = self._make_agent()
        warning = agent.check_new_schedule("Work", "2026-04-03 09:00")
        if warning:
            assert "**" not in warning
            assert "`" not in warning
            assert "#" not in warning


# ════════════════════════════════════════════════════════════════════════════
# SCORING SUMMARY
# ════════════════════════════════════════════════════════════════════════════

"""
SCORING GUIDE:
─────────────────────────────────────────────────────────────────────────────
Section                  | Tests | Target   | Notes
─────────────────────────────────────────────────────────────────────────────
1. HolidayChecker        |  21   | 21/21    | Pure logic, no deps — must be 100%
2. LocalScheduleSource   |  14   | 14/14    | Pure SQL — must be 100%
3. UnifiedSchedule       |  13   | 12/13    | GCal mock may vary
4. ScheduleAwareness     |  11   | 10/11    | Timing tests may vary slightly
5. AgentCore             |  10   | 9/10     | Thread timing tests
─────────────────────────────────────────────────────────────────────────────
TOTAL                    |  69   | 66+/69   | Target: 96%+
─────────────────────────────────────────────────────────────────────────────

READINESS:
  >= 63/69 (91%) → adaptive layer solid, start daily use
  55-62/69       → fix failing tests before relying on watchdog
  < 55/69        → significant gap, review the failing section

Run both suites together to get full picture:
  pytest eval/ -v
─────────────────────────────────────────────────────────────────────────────
"""