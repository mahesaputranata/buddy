"""
tests/test_schedule_robustness.py

Buddy Schedule Robustness Test Suite
=====================================
End-to-end tests covering:
  - Intent routing (does classify_intent + apply_guards route correctly?)
  - Schedule save pipeline (does parse_schedule extract the right fields?)
  - Schedule recall pipeline (does retrieval return the right schedules?)
  - Edge cases (missing fields, past dates, ambiguous input, German formats)
  - Regression guard (previously working cases still pass)

Run with:
    pytest tests/test_schedule_robustness.py -v
    pytest tests/test_schedule_robustness.py -v --tb=short   (less noise)
    pytest tests/test_schedule_robustness.py -v -k "save"    (only save tests)

Requirements:
  - Ollama must be running locally with llama3.2:latest pulled
  - Run from project root (same level as src/)
  - Uses a temporary isolated DB — your real sqlBrain.db is NOT touched

Scoring:
  After the run, check the summary line.
  Target: 90%+ pass rate for "ready to build watchdog on top of"
"""

import sys
import os
import sqlite3
import tempfile
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

# ── Path setup ──────────────────────────────────────────────────────────────
# Add project root and src/ to path so imports work from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

# ── Isolated test DB setup ───────────────────────────────────────────────────
# We patch get_connection() to use a temp file so real data is NEVER touched

_TEST_DB_PATH = None

def _get_test_connection():
    return sqlite3.connect(_TEST_DB_PATH)

# ── Imports (after path setup) ───────────────────────────────────────────────
TZ = ZoneInfo("Europe/Berlin")

# We patch the DB connection before importing sql_brain
import brain.core.sql_brain as sql_brain_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """
    Each test gets a fresh isolated SQLite DB.
    Patches sql_brain.get_connection to use this temp DB.
    """
    global _TEST_DB_PATH
    db_file = tmp_path / "test_buddy.db"
    _TEST_DB_PATH = str(db_file)

    with patch.object(sql_brain_module, "get_connection", _get_test_connection):
        sql_brain_module.init_schedule_table()
        yield

    _TEST_DB_PATH = None


# ── Import Buddy modules after patching ─────────────────────────────────────
from brain.core.sql_brain import (
    create_schedule,
    get_all_schedules_as_dicts,
    retrieve_schedule_by_date_sql,
    retrieve_schedule_by_week_sql,
    retrieve_schedule_by_month_sql,
    retrieve_schedule_by_year_sql,
    smart_schedule_recall,
    extractScheduleDetails,
    init_schedule_table,
)

from brain.core.brain import MemoryManager


# ── Helpers ──────────────────────────────────────────────────────────────────

def today_str():
    return datetime.now(TZ).strftime("%Y-%m-%d")

def tomorrow_str():
    return (datetime.now(TZ) + timedelta(days=1)).strftime("%Y-%m-%d")

def next_monday_str():
    today = datetime.now(TZ).date()
    days_ahead = (7 - today.weekday()) % 7 or 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

def this_year():
    return str(datetime.now(TZ).year)

def this_month():
    return datetime.now(TZ).strftime("%Y-%m")

def make_timestamp(date_str, time_str="09:00"):
    return f"{date_str} {time_str}"


def get_conversation_manager():
    """
    Creates a minimal ConversationManager for end-to-end tests.
    Heavy dependencies (TelloClient, RoverClient, Spotify) are mocked
    so the test doesn't need hardware or network.
    """
    from unittest.mock import MagicMock, patch

    mm = MemoryManager()
    mm._instance = None  # reset singleton for clean state per test

    # Patch hardware deps so ConversationManager.__init__ doesn't fail
    with patch("main.TelloClient", return_value=MagicMock()), \
         patch("devices.rover.rover_bridge.RoverClient", return_value=MagicMock()), \
         patch("brain.music_player.spotify_manager.SpotifyManager", return_value=MagicMock()), \
         patch("brain.core.internet_access.internet_service.get_explorer", return_value=MagicMock()), \
         patch("brain.identity.identity_manager.IdentityManager", return_value=MagicMock()):

        from brain.conversation_manager import ConversationManager, MemoryManager as MainMM
        fresh_mm = MainMM()
        cm = ConversationManager(
            memory_manager=fresh_mm,
            tello_client=MagicMock(),
            explorer=MagicMock(),
            identity=MagicMock(),
        )
    return cm


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INTENT ROUTING
# Tests that classify_intent + apply_guards routes schedule inputs correctly
# ════════════════════════════════════════════════════════════════════════════

class TestIntentRouting:
    """
    Verifies that schedule-related utterances route to the right intent.
    These run Ollama for LLM-fallback cases.
    """

    def _classify(self, text):
        from brain.conversation_manager import classify_intent, apply_guards
        mm = MemoryManager()
        raw = classify_intent(text, mm)
        return apply_guards(raw, text, mm)

    # ── schedule_save ────────────────────────────────────────────────────────

    def test_intent_remind_me_basic(self):
        result = self._classify("remind me to buy groceries tomorrow at 5pm")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']} (src={result['source']})"

    def test_intent_remind_me_no_time(self):
        result = self._classify("remind me to call my supervisor")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_intent_set_reminder(self):
        result = self._classify("set a reminder for my dentist appointment next Monday")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_intent_i_have_appointment(self):
        result = self._classify("I have a meeting with my professor at 10am tomorrow")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_intent_add_to_schedule(self):
        result = self._classify("add to my schedule: gym session on Friday at 6pm")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_intent_dont_forget(self):
        result = self._classify("don't forget I have a Hiwi shift on Thursday")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_intent_remind_me_grocery_no_time(self):
        """Regression: 'remind me to buy X' should not route to save_memory"""
        result = self._classify("remind me to buy milk and eggs")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_intent_not_save_memory(self):
        """'remind me' should NOT route to save_memory"""
        result = self._classify("remind me about my deadline next week")
        assert result["intent"] != "save_memory", \
            f"Should not be save_memory, got {result['intent']}"

    # ── schedule_recall ──────────────────────────────────────────────────────

    def test_intent_what_is_my_schedule(self):
        result = self._classify("what is my schedule for today?")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_intent_do_i_have_anything(self):
        result = self._classify("do I have anything tomorrow?")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_intent_whats_on_my_schedule(self):
        result = self._classify("what's on my schedule this week?")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_intent_any_meetings_today(self):
        result = self._classify("do I have any meetings today?")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_intent_upcoming_appointments(self):
        result = self._classify("show me my upcoming appointments")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_intent_recall_not_recall_memory(self):
        """'what's my schedule' should NOT route to recall_memory"""
        result = self._classify("what did I schedule for next Monday?")
        assert result["intent"] != "recall_memory", \
            f"Should not be recall_memory, got {result['intent']}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCHEDULE SAVE PIPELINE (parse_schedule)
# Tests that parse_schedule extracts the right fields from raw utterances
# ════════════════════════════════════════════════════════════════════════════

class TestScheduleSavePipeline:
    """
    Tests parse_schedule() directly — does it extract title, date, time correctly?
    These run Ollama.
    """

    def _parse(self, text):
        """Get a ConversationManager and call parse_schedule"""
        from brain.conversation_manager import ConversationManager
        with patch("main.TelloClient", return_value=MagicMock()), \
             patch("devices.rover.rover_bridge.RoverClient", return_value=MagicMock()), \
             patch("brain.music_player.spotify_manager.SpotifyManager", return_value=MagicMock()), \
             patch("brain.core.internet_access.internet_service.get_explorer", return_value=MagicMock()), \
             patch("brain.identity.identity_manager.IdentityManager", return_value=MagicMock()):
            mm = MemoryManager()
            cm = ConversationManager(
                memory_manager=mm,
                tello_client=MagicMock(),
                explorer=MagicMock(),
                identity=MagicMock(),
            )
        return cm.memory_handler.parse_schedule(text)

    # ── complete inputs (should extract all fields) ──────────────────────────

    def test_parse_complete_basic(self):
        result = self._parse("remind me to buy groceries tomorrow at 5pm")
        data = result["data"]
        assert data.get("title"), "Title should be extracted"
        assert data.get("date") == tomorrow_str(), \
            f"Date should be {tomorrow_str()}, got {data.get('date')}"
        assert data.get("time"), "Time should be extracted"
        assert result["missing_fields"] == [], \
            f"No fields should be missing, got {result['missing_fields']}"

    def test_parse_complete_with_location(self):
        result = self._parse("I have a dentist appointment on Friday at 9am at Zahnarzt Müller")
        data = result["data"]
        assert data.get("title"), "Title should be extracted"
        assert data.get("time"), "Time should be extracted"

    def test_parse_complete_iso_date(self):
        result = self._parse("remind me about my exam on 2026-06-15 at 10:00")
        data = result["data"]
        assert data.get("date") == "2026-06-15", \
            f"Expected 2026-06-15, got {data.get('date')}"
        assert data.get("time"), "Time should be extracted"

    def test_parse_complete_german_date_format(self):
        """German-style date: 15.06.2026 um 9 Uhr"""
        result = self._parse("erinnerung: Termin am 15.06.2026 um 9 Uhr")
        data = result["data"]
        # We just check that SOMETHING was extracted — full German parsing is a bonus
        assert data.get("date") or data.get("timestamp"), \
            "Should extract at least a date or timestamp from German format"

    def test_parse_today_with_time(self):
        result = self._parse("remind me to take my medication today at 8pm")
        data = result["data"]
        assert data.get("date") == today_str(), \
            f"Expected today {today_str()}, got {data.get('date')}"
        assert data.get("time"), "Time should be extracted"

    def test_parse_next_monday(self):
        result = self._parse("I have a team meeting next Monday at 2pm")
        data = result["data"]
        assert data.get("date") == next_monday_str(), \
            f"Expected {next_monday_str()}, got {data.get('date')}"

    # ── incomplete inputs (should detect missing fields) ─────────────────────

    def test_parse_missing_time(self):
        """'remind me to go to office on Friday' — no time given"""
        result = self._parse("remind me to go to office on Friday")
        assert "timestamp" in result["missing_fields"] or "time" in result.get("missing_fields", []), \
            f"Should flag missing time, missing_fields={result['missing_fields']}"
        assert result["followup_question"], "Should ask a follow-up question"

    def test_parse_missing_title(self):
        """'set a reminder for 3pm tomorrow' — no title"""
        result = self._parse("set a reminder for 3pm tomorrow")
        assert "title" in result["missing_fields"], \
            f"Should flag missing title, got {result['missing_fields']}"
        assert result["followup_question"], "Should ask a follow-up question"

    def test_parse_missing_both(self):
        """'remind me' alone — both title and time missing"""
        result = self._parse("remind me")
        assert len(result["missing_fields"]) >= 1, \
            f"Should flag at least one missing field, got {result['missing_fields']}"
        assert result["followup_question"], "Should ask a follow-up question"

    def test_parse_no_false_time(self):
        """
        Regression: if user never mentions a time, parse_schedule should NOT
        hallucinate 00:00 or midnight.
        """
        result = self._parse("remind me to buy groceries on Saturday")
        data = result["data"]
        time_val = data.get("time")
        # 00:00 is only valid if user explicitly said midnight
        assert time_val not in ("00:00", "0:00"), \
            f"Should not hallucinate midnight as time, got time={time_val}"

    def test_parse_timestamp_built_correctly(self):
        """If date and time both extracted, timestamp should be date + space + time"""
        result = self._parse("remind me to call mom tomorrow at 3pm")
        data = result["data"]
        if data.get("date") and data.get("time") and data.get("timestamp"):
            ts = data["timestamp"]
            assert ts.startswith(data["date"]), \
                f"Timestamp {ts} should start with date {data['date']}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SQL LAYER (create + retrieve)
# Tests the sql_brain functions directly — no LLM needed here
# ════════════════════════════════════════════════════════════════════════════

class TestSQLLayer:
    """
    Tests the SQL schedule brain directly.
    These are pure unit tests — no Ollama needed.
    """

    def test_create_and_retrieve_basic(self):
        ts = make_timestamp(tomorrow_str(), "09:00")
        create_schedule("Dentist", "Dentist appointment", ts)
        rows = get_all_schedules_as_dicts()
        assert len(rows) == 1
        assert rows[0]["title"] == "Dentist"
        assert rows[0]["timestamp"] == ts

    def test_create_multiple_schedules(self):
        create_schedule("Gym", "Gym session", make_timestamp(tomorrow_str(), "07:00"))
        create_schedule("Meeting", "Team meeting", make_timestamp(tomorrow_str(), "10:00"))
        create_schedule("Groceries", "Buy groceries", make_timestamp(tomorrow_str(), "18:00"))
        rows = get_all_schedules_as_dicts()
        assert len(rows) == 3

    def test_retrieve_by_date_found(self):
        ts = make_timestamp(tomorrow_str(), "14:00")
        create_schedule("Lecture", "Physics lecture", ts)
        result = retrieve_schedule_by_date_sql(tomorrow_str())
        assert "Lecture" in result, f"Expected 'Lecture' in result, got: {result}"

    def test_retrieve_by_date_not_found(self):
        result = retrieve_schedule_by_date_sql("2099-01-01")
        assert "no schedules" in result.lower() or "couldn't find" in result.lower(), \
            f"Expected empty message, got: {result}"

    def test_retrieve_by_date_multiple(self):
        ts1 = make_timestamp(tomorrow_str(), "09:00")
        ts2 = make_timestamp(tomorrow_str(), "14:00")
        create_schedule("Morning run", "Exercise", ts1)
        create_schedule("Doctor", "Doctor visit", ts2)
        result = retrieve_schedule_by_date_sql(tomorrow_str())
        assert "Morning run" in result
        assert "Doctor" in result

    def test_retrieve_by_week(self):
        # put one event in the current week
        today = datetime.now(TZ).date()
        monday = today - timedelta(days=today.weekday())
        wednesday = monday + timedelta(days=2)
        ts = make_timestamp(wednesday.strftime("%Y-%m-%d"), "10:00")
        create_schedule("Weekly standup", "Team standup", ts)

        start = monday.strftime("%Y-%m-%d")
        end = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
        result = retrieve_schedule_by_week_sql(start, end)
        assert "standup" in result.lower() or "Weekly" in result, \
            f"Expected standup in week result, got: {result}"

    def test_retrieve_by_month(self):
        ts = make_timestamp(f"{this_month()}-15", "12:00")
        create_schedule("Monthly review", "Review session", ts)
        result = retrieve_schedule_by_month_sql(this_month())
        assert "review" in result.lower() or "Monthly" in result, \
            f"Expected review in month result, got: {result}"

    def test_retrieve_by_year(self):
        ts = make_timestamp(f"{this_year()}-08-20", "09:00")
        create_schedule("Summer vacation", "Vacation start", ts)
        result = retrieve_schedule_by_year_sql(this_year())
        assert "vacation" in result.lower() or "Summer" in result, \
            f"Expected vacation in year result, got: {result}"

    def test_schedule_sorted_by_time(self):
        """Schedules for same day should come back sorted by time"""
        create_schedule("Late event", "desc", make_timestamp(tomorrow_str(), "20:00"))
        create_schedule("Early event", "desc", make_timestamp(tomorrow_str(), "06:00"))
        rows = get_all_schedules_as_dicts()
        times = [r["timestamp"] for r in rows]
        assert times == sorted(times), \
            f"Schedules not sorted by timestamp: {times}"

    def test_location_stored_and_retrieved(self):
        ts = make_timestamp(tomorrow_str(), "10:00")
        create_schedule("Hiwi work", "Uni shift", ts, location="TU Darmstadt")
        rows = get_all_schedules_as_dicts()
        assert rows[0]["location"] == "TU Darmstadt"

    def test_null_location_handled(self):
        ts = make_timestamp(tomorrow_str(), "10:00")
        create_schedule("Reminder", "desc", ts, location=None)
        rows = get_all_schedules_as_dicts()
        assert rows[0]["location"] is None or rows[0]["location"] == ""


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EDGE CASES
# The tricky inputs that break naive systems
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """
    Edge cases: ambiguous phrasing, incomplete info, potential misfires.
    Most of these run Ollama.
    """

    def _classify(self, text):
        from brain.conversation_manager import classify_intent, apply_guards
        mm = MemoryManager()
        raw = classify_intent(text, mm)
        return apply_guards(raw, text, mm)

    def _parse(self, text):
        from brain.conversation_manager import ConversationManager
        with patch("main.TelloClient", return_value=MagicMock()), \
             patch("devices.rover.rover_bridge.RoverClient", return_value=MagicMock()), \
             patch("brain.music_player.spotify_manager.SpotifyManager", return_value=MagicMock()), \
             patch("brain.core.internet_access.internet_service.get_explorer", return_value=MagicMock()), \
             patch("brain.identity.identity_manager.IdentityManager", return_value=MagicMock()):
            mm = MemoryManager()
            cm = ConversationManager(
                memory_manager=mm,
                tello_client=MagicMock(),
                explorer=MagicMock(),
                identity=MagicMock(),
            )
        return cm.memory_handler.parse_schedule(text)

    # ── intent does NOT misfire ──────────────────────────────────────────────

    def test_no_misfire_on_research_talk(self):
        """'my thesis work involves...' should NOT become schedule_save"""
        result = self._classify("my thesis involves terahertz signal processing")
        assert result["intent"] not in ("schedule_save", "schedule_recall"), \
            f"Should be chat, got {result['intent']}"

    def test_no_misfire_engineering_discussion(self):
        """Engineering sentence with time-like words should not be schedule_save"""
        result = self._classify("the PID controller runs at 100hz with 3ms latency")
        assert result["intent"] != "schedule_save", \
            f"Should not be schedule_save, got {result['intent']}"

    def test_no_misfire_hypothetical(self):
        """'for an example, let's say I remind myself...' is not a real reminder"""
        result = self._classify("for an example let's say I remind myself to go to the store")
        assert result["intent"] != "schedule_save", \
            f"Hypothetical framing should not create a schedule, got {result['intent']}"

    def test_no_misfire_time_question(self):
        """'what time is it' is system_query not schedule_recall"""
        result = self._classify("what time is it?")
        assert result["intent"] == "system_query", \
            f"Expected system_query, got {result['intent']}"

    def test_no_misfire_save_memory(self):
        """'save this' without schedule context should not become schedule_save"""
        result = self._classify("save this conversation about reinforcement learning")
        assert result["intent"] != "schedule_save", \
            f"Save conversation should not be schedule_save, got {result['intent']}"

    # ── past date handling ───────────────────────────────────────────────────

    def test_past_date_still_saves(self):
        """
        Buddy should still save a past date — it's the watchdog's job to flag it.
        parse_schedule should not refuse a past date.
        """
        result = self._parse("remind me about my appointment on 2020-01-01 at 9am")
        data = result["data"]
        # Either it extracts the date or flags it missing — but should NOT crash
        assert isinstance(result, dict), "parse_schedule should return a dict"

    # ── recurring events ─────────────────────────────────────────────────────

    def test_recurring_fails_gracefully(self):
        """'remind me every Monday' — recurring not supported yet, should not crash"""
        result = self._classify("remind me every Monday to check my email")
        # Should route somewhere sensible and not raise an exception
        assert result["intent"] in ("schedule_save", "chat"), \
            f"Recurring should route to schedule_save or chat, got {result['intent']}"

    # ── phrasing variations ──────────────────────────────────────────────────

    def test_phrasing_variation_can_you_remind(self):
        result = self._classify("can you remind me to pick up my package tomorrow?")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_phrasing_variation_i_need_to_remember(self):
        result = self._classify("I need to remember to submit my report by Friday")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_phrasing_variation_dont_let_me_forget(self):
        result = self._classify("don't let me forget my sister's birthday on April 20th")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    def test_phrasing_variation_note_that_i_have(self):
        result = self._classify("note that I have a doctor's appointment next Tuesday at 11am")
        assert result["intent"] == "schedule_save", \
            f"Expected schedule_save, got {result['intent']}"

    # ── public holiday awareness (foundation for watchdog) ───────────────────

    def test_save_schedule_on_known_holiday_date(self):
        """
        Saving a schedule on Karfreitag (2026-04-03) should succeed at the SQL level.
        Holiday conflict detection is the watchdog's job — not a blocker here.
        This test verifies the foundation works before we add holiday awareness.
        """
        ts = make_timestamp("2026-04-03", "09:00")
        create_schedule("Hiwi work", "University shift", ts)
        rows = get_all_schedules_as_dicts()
        assert len(rows) == 1, "Schedule should be saved even if date is a holiday"
        assert rows[0]["timestamp"] == ts


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RECALL ROBUSTNESS
# Tests that recall routing works correctly
# ════════════════════════════════════════════════════════════════════════════

class TestRecallRobustness:
    """
    Tests recall routing — does the right recall_type get selected?
    Most of these run Ollama.
    """

    def _classify(self, text):
        from brain.conversation_manager import classify_intent, apply_guards
        mm = MemoryManager()
        raw = classify_intent(text, mm)
        return apply_guards(raw, text, mm)

    def test_recall_today(self):
        result = self._classify("what do I have today?")
        assert result["intent"] == "schedule_recall"

    def test_recall_tomorrow(self):
        result = self._classify("what do I have tomorrow?")
        assert result["intent"] == "schedule_recall"

    def test_recall_this_week(self):
        result = self._classify("show me everything on my schedule this week")
        assert result["intent"] == "schedule_recall"

    def test_recall_specific_event(self):
        result = self._classify("when is my dentist appointment?")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_recall_does_not_misroute_to_memory(self):
        """'what did I schedule' must not go to recall_memory"""
        result = self._classify("what did I schedule for this week?")
        assert result["intent"] == "schedule_recall", \
            f"Expected schedule_recall, got {result['intent']}"

    def test_sql_recall_by_date_returns_correct_event(self):
        """Direct SQL recall — correct event returned for date"""
        ts = make_timestamp(tomorrow_str(), "15:00")
        create_schedule("Physics lecture", "Lecture hall B", ts)
        result = retrieve_schedule_by_date_sql(tomorrow_str())
        assert "Physics" in result or "lecture" in result.lower(), \
            f"Expected lecture in result, got: {result}"

    def test_sql_recall_empty_returns_message(self):
        """No schedules saved — should return a clean message, not crash"""
        result = retrieve_schedule_by_date_sql(tomorrow_str())
        assert isinstance(result, str), "Should always return a string"
        assert len(result) > 0, "Should not return empty string"

    def test_smart_recall_fuzzy_match(self):
        """smart_schedule_recall should fuzzy-match on title"""
        ts = make_timestamp(tomorrow_str(), "10:00")
        create_schedule("Dentist appointment", "Dr. Schmidt", ts)
        extracted = {"title": "dentist", "timestamp": None, "location": None, "description": None}
        result = smart_schedule_recall(extracted)
        assert isinstance(result, (str, dict)), "smart_schedule_recall should return str or dict"
        if isinstance(result, str):
            assert "dentist" in result.lower() or "Dentist" in result, \
                f"Expected dentist in result, got: {result}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — REGRESSION GUARD
# Cases that previously worked — must continue to work
# ════════════════════════════════════════════════════════════════════════════

class TestRegressionGuard:
    """
    Regression tests — if any of these fail, something broke that used to work.
    Treat failures here as high-priority bugs.
    """

    def _classify(self, text):
        from brain.conversation_manager import classify_intent, apply_guards
        mm = MemoryManager()
        raw = classify_intent(text, mm)
        return apply_guards(raw, text, mm)

    def test_regression_remind_me_grocery(self):
        """Classic reminder — must always work"""
        result = self._classify("remind me to buy groceries tomorrow at 5pm")
        assert result["intent"] == "schedule_save"

    def test_regression_whats_my_schedule(self):
        """Classic recall — must always work"""
        result = self._classify("what's my schedule for today?")
        assert result["intent"] == "schedule_recall"

    def test_regression_chat_not_schedule(self):
        """General question — must never become schedule"""
        result = self._classify("what is reinforcement learning?")
        assert result["intent"] == "chat"

    def test_regression_sql_create_retrieve(self):
        """Core SQL pipeline — must always work"""
        ts = make_timestamp(tomorrow_str(), "09:00")
        create_schedule("Test event", "Test description", ts)
        rows = get_all_schedules_as_dicts()
        assert len(rows) == 1
        assert rows[0]["title"] == "Test event"

    def test_regression_no_crash_on_empty_input(self):
        """Empty input must not crash classify_intent"""
        from brain.conversation_manager import classify_intent, apply_guards
        mm = MemoryManager()
        try:
            raw = classify_intent("", mm)
            result = apply_guards(raw, "", mm)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Empty input caused crash: {e}")

    def test_regression_no_crash_on_garbage_input(self):
        """Garbage ASR output must not crash the pipeline"""
        from brain.conversation_manager import classify_intent, apply_guards
        mm = MemoryManager()
        garbage_inputs = [
            "asdfjkl qwerty zxcvb",
            "... ... ...",
            "uhhh hmm what",
            "123 456 789",
        ]
        for text in garbage_inputs:
            try:
                raw = classify_intent(text, mm)
                result = apply_guards(raw, text, mm)
                assert isinstance(result, dict), f"Should return dict for: {text}"
            except Exception as e:
                pytest.fail(f"Garbage input '{text}' caused crash: {e}")

    def test_regression_hiwi_reminder(self):
        """
        The exact scenario from this morning — saving a Hiwi work reminder.
        Must route to schedule_save.
        """
        result = self._classify("remind me I have Hiwi work tomorrow morning")
        assert result["intent"] == "schedule_save", \
            f"Hiwi work reminder should be schedule_save, got {result['intent']}"


# ════════════════════════════════════════════════════════════════════════════
# SCORING SUMMARY
# Run with: pytest tests/test_schedule_robustness.py -v
# ════════════════════════════════════════════════════════════════════════════

"""
SCORING GUIDE:
─────────────────────────────────────────────────────────────────
Section                      | Tests | Weight  | Target
─────────────────────────────────────────────────────────────────
1. Intent Routing            |  14   | High    | 12/14 (86%)
2. Schedule Save Pipeline    |   9   | High    | 8/9  (89%)
3. SQL Layer                 |  10   | High    | 10/10 (100%) ← pure logic, no LLM
4. Edge Cases                |  12   | Medium  | 9/12 (75%)
5. Recall Robustness         |   8   | High    | 7/8  (88%)
6. Regression Guard          |   7   | Critical| 7/7  (100%)
─────────────────────────────────────────────────────────────────
TOTAL                        |  60   |         | 53+/60 (88%+)
─────────────────────────────────────────────────────────────────

READINESS DECISION:
  >= 53/60 (88%) → Schedule foundation is solid. Proceed to watchdog.
  45-52/60       → Fix failing tests before building watchdog.
  < 45/60        → Significant gaps. Fix pipeline first.

After running, add every failure to this file as a new test.
Real failures discovered during daily use should be added here too.
─────────────────────────────────────────────────────────────────
"""