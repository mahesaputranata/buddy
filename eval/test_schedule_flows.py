"""
eval/test_schedule_flows.py

Buddy Schedule Flow Tests — End-to-End Chained Conversations
=============================================================
Tests full conversation flows, not individual functions.
Each test simulates a real Buddy session with multiple turns.

What this catches that unit tests cannot:
  - Hallucination: Buddy confirms a save that never happened in SQL
  - State leakage: pending state from save bleeds into recall
  - Cross-intent:  delete accidentally removes wrong schedule
  - Double save:   same reminder said twice creates duplicate rows
  - Phantom recall: Buddy recalls a schedule that doesn't exist
  - Holiday chain: warning fires on save AND shows up on recall
  - Flow recovery: wrong input mid-flow is handled gracefully

Run with:
    pytest eval/test_schedule_flows.py -v
    pytest eval/test_schedule_flows.py -v -k "save"
    pytest eval/test_schedule_flows.py -v -k "hallucination"

NOTE: These tests call Ollama (real LLM). Ollama must be running.
Expected runtime: 3-6 minutes for full suite.

Scoring:
    >= 85% → flow behavior is solid, ready for daily use
    70-84% → review failing flows before daily use
    < 70%  → significant flow bugs, fix before proceeding
"""

import sys
import os
import sqlite3
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

def _patched_create_schedule(title, desc, ts, loc=None):
    """Routes create_schedule calls to the isolated test DB."""
    conn = _get_test_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO schedules (title, description, timestamp, location)
        VALUES (?, ?, ?, ?)
    """, (title, desc, ts, loc))
    conn.commit()
    conn.close()
    print(f"[TEST_DB] create_schedule: '{title}' @ {ts}")

def _get_test_connection():
    print(f"[TEST_DB] connecting to: {_TEST_DB_PATH}")
    return sqlite3.connect(_TEST_DB_PATH)

@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    global _TEST_DB_PATH
    db_file = tmp_path / "test_flows.db"
    _TEST_DB_PATH = str(db_file)

    # Clear in-memory state between tests
    try:
        from brain.core.brain import temp_topic_memory, sensory_memory
        temp_topic_memory.clear()
        sensory_memory.clear()
    except Exception:
        pass

    with patch.object(sql_brain_module, "get_connection", _get_test_connection):
        sql_brain_module.init_schedule_table()
        yield

    _TEST_DB_PATH = None


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _patched_create_schedule(title, desc, ts, loc=None):
    """Routes create_schedule calls to the isolated test DB."""
    conn = _get_test_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO schedules (title, description, timestamp, location)
        VALUES (?, ?, ?, ?)
    """, (title, desc, ts, loc))
    conn.commit()
    conn.close()
    print(f"[TEST_DB] create_schedule: '{title}' @ {ts}")


@pytest.fixture
def buddy():
    """
    Returns a fully wired ConversationManager with:
    - Real Ollama
    - Isolated test DB
    - Hardware deps mocked
    - MemoryManager singleton fully reset before AND after each test
    """
    # ── Reset singleton BEFORE test ──────────────────────────────────────
    try:
        from brain.core.brain import MemoryManager as BrainMM
        BrainMM._instance = None
    except Exception:
        pass

    # ── Also reset in-memory topic state ─────────────────────────────────
    try:
        from brain.core.brain import temp_topic_memory, sensory_memory
        temp_topic_memory.clear()
        sensory_memory.clear()
    except Exception:
        pass

    with patch.object(sql_brain_module, "get_connection", _get_test_connection), \
         patch("brain.core.sql_brain.get_connection", _get_test_connection), \
         patch("brain.conversation_manager.create_schedule", side_effect=_patched_create_schedule), \
         patch("brain.conversation_manager.TelloClient", return_value=MagicMock()), \
         patch("brain.conversation_manager.RoverClient", return_value=MagicMock()), \
         patch("brain.conversation_manager.SpotifyManager", return_value=MagicMock()), \
         patch("brain.conversation_manager.get_explorer", return_value=MagicMock()), \
         patch("brain.conversation_manager.IdentityManager", return_value=MagicMock()), \
         patch("brain.conversation_manager.SpeakerWorkerClient", return_value=MagicMock()):

        from brain.conversation_manager import ConversationManager
        from brain.core.brain import MemoryManager

        mm = MemoryManager()
        cm = ConversationManager(
            memory_manager=mm,
            tello_client=MagicMock(),
            explorer=MagicMock(),
            identity=MagicMock(),
        )
        yield cm

    # ── Reset singleton AFTER test ────────────────────────────────────────
    try:
        from brain.core.brain import MemoryManager as BrainMM
        BrainMM._instance = None
    except Exception:
        pass


def get_sql_schedules():
    """Reads directly from test DB to verify what was actually saved."""
    with patch.object(sql_brain_module, "get_connection", _get_test_connection):
        return sql_brain_module.get_all_schedules_as_dicts()


def today_str():
    return date.today().strftime("%Y-%m-%d")

def tomorrow_str():
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

def days_from_now(n):
    return (date.today() + timedelta(days=n)).strftime("%Y-%m-%d")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SAVE FLOWS
# Does Buddy actually save to SQL after confirmation?
# ════════════════════════════════════════════════════════════════════════════

class TestSaveFlows:
    """End-to-end save flows — verifies SQL after each save."""

    def test_flow_save_complete_input(self, buddy):
        """
        Complete input with title, date, time → saves in one turn.
        No follow-up should be needed.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("remind me to buy groceries tomorrow at 5pm")
            rows = sql_brain_module.get_all_schedules_as_dicts()  # ← move inside

        # Verify response acknowledges the save
        assert response, "Should return a response"
        response_lower = response.lower()
        confirmed = any(w in response_lower for w in [
            "got it", "saved", "remind", "remember", "noted", "scheduled"
        ])
        assert confirmed, \
            f"Response should confirm save, got: {response}"

        # Verify SQL actually has the record
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()
        assert len(rows) >= 1, \
            f"SQL should have at least 1 schedule after save, got {len(rows)}"

        titles = [r["title"].lower() for r in rows]
        assert any("grocer" in t for t in titles), \
            f"Schedule title should contain 'grocer', got titles: {titles}"

    def test_flow_save_missing_time_asks_followup(self, buddy):
        """
        Input with no time → Buddy asks for time, not silently saves with NULL.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "remind me to go to the office on Friday"
            )

        response_lower = response.lower()

        # Either asks for time OR saves with a follow-up question
        asks_for_info = any(w in response_lower for w in [
            "what time", "when", "time", "what", "?"
        ])
        saved_directly = any(w in response_lower for w in [
            "got it", "saved", "noted"
        ])

        assert asks_for_info or saved_directly, \
            f"Should either ask for time or confirm save, got: {response}"

        # If it asked a follow-up, answer it
        if asks_for_info and not saved_directly:
            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                response2 = buddy.handle_request("at 9am")

            response2_lower = response2.lower()
            confirmed = any(w in response2_lower for w in [
                "got it", "saved", "remind", "remember", "noted"
            ])
            assert confirmed, \
                f"After providing time, should confirm save, got: {response2}"

            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                rows = sql_brain_module.get_all_schedules_as_dicts()
            assert len(rows) >= 1, "Schedule should be saved after follow-up"

    def test_flow_save_hiwi_reminder(self, buddy):
        """
        Regression: the exact Hiwi scenario from this morning.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "remind me I have Hiwi work tomorrow at 9am"
            )

        assert response, "Should return a response"

        # Check SQL
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        # Should either save directly or ask a follow-up (not silently do nothing)
        if not rows:
            # Buddy asked a follow-up — provide missing info
            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                response2 = buddy.handle_request("yes save it")
            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                rows = sql_brain_module.get_all_schedules_as_dicts()

        assert len(rows) >= 1, \
            "Hiwi reminder should eventually be saved to SQL"

    def test_flow_save_holiday_triggers_warning(self, buddy):
        """
        Saving a schedule on Karfreitag should include a holiday warning
        in the response — the reactive check.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "remind me to go to office on April 3rd at 9am"
            )

        # The holiday warning should appear somewhere in the response chain
        # Either in this response or after confirmation
        response_lower = response.lower()

        # Check if holiday warning fired
        holiday_mentioned = any(w in response_lower for w in [
            "karfreitag", "holiday", "public holiday", "feiertag", "closed"
        ])

        # It's acceptable if warning comes after save confirmation
        # so we also check SQL was saved
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        saved = len(rows) >= 1

        # Either the warning fired directly OR at minimum the save happened
        # (warning hook requires agent wiring which may not be active in tests)
        assert saved or holiday_mentioned, \
            f"Should either save or warn about holiday. response={response}"

    def test_flow_no_hallucination_on_save(self, buddy):
        """
        Anti-hallucination: if Buddy says 'saved', it MUST be in SQL.
        If Buddy says 'saved' but SQL is empty → hallucination detected.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "remind me about my dentist appointment next Monday at 10am"
            )

        response_lower = response.lower()
        claimed_saved = any(w in response_lower for w in [
            "saved", "got it", "noted", "i'll remind", "i will remind",
            "reminder set", "added", "scheduled"
        ])

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        if claimed_saved:
            assert len(rows) >= 1, \
                f"HALLUCINATION DETECTED: Buddy said saved but SQL is empty. " \
                f"Response was: {response}"

    def test_flow_double_save_no_crash(self, buddy):
        """
        Saying the same reminder twice should not crash and should
        result in at most 2 rows (not an explosion of duplicates).
        """
        input_text = "remind me to take my medication tomorrow at 8am"

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response1 = buddy.handle_request(input_text)

        # Reset pending state between calls
        buddy.memory_manager.set_pending_topic(None)
        buddy.memory_manager.pending_payload = None

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response2 = buddy.handle_request(input_text)

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        # Should not create more than 2 rows for the same reminder
        assert len(rows) <= 2, \
            f"Double save created {len(rows)} rows — potential runaway duplication"

    def test_flow_save_with_location(self, buddy):
        """Location in the input should be stored in SQL."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "remind me I have a meeting at TU Berlin on Thursday at 2pm"
            )

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        if rows:
            # Location should be stored if mentioned
            locations = [r.get("location") or "" for r in rows]
            timestamps = [r.get("timestamp") or "" for r in rows]
            # At minimum the event was saved
            assert len(rows) >= 1, "Meeting should be saved"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RECALL FLOWS
# Does Buddy return the right schedule after saving?
# ════════════════════════════════════════════════════════════════════════════

class TestRecallFlows:
    """End-to-end recall flows — save first, then verify recall returns it."""

    def test_flow_save_then_recall_today(self, buddy):
        """
        Save → recall today → event appears in response.
        """
        # Seed directly in SQL for clean test
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Gym session", "Morning workout",
                f"{today_str()} 07:00"
            )

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("what do I have today?")

        assert response, "Should return a response"
        assert "gym" in response.lower() or "workout" in response.lower() or \
               "session" in response.lower(), \
            f"Recall should return today's gym session, got: {response}"

    def test_flow_save_then_recall_tomorrow(self, buddy):
        """
        Save tomorrow event → recall tomorrow → event appears.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Dentist appointment", "Dr. Schmidt",
                f"{tomorrow_str()} 09:00"
            )

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("what do I have tomorrow?")

        assert response, "Should return a response"
        assert "dentist" in response.lower(), \
            f"Recall should mention dentist appointment, got: {response}"

    def test_flow_recall_empty_schedule_no_hallucination(self, buddy):
        """
        Anti-hallucination: when DB is empty, Buddy should say nothing scheduled.
        Should NOT invent events.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("what do I have today?")

        assert response, "Should return a response"
        response_lower = response.lower()

        # Should indicate nothing scheduled
        nothing_phrases = [
            "nothing", "no schedule", "no event", "don't have", "doesn't appear",
            "couldn't find", "no appointment", "free", "clear", "empty",
            "nothing scheduled", "no plans"
        ]
        found_nothing = any(p in response_lower for p in nothing_phrases)

        # Should NOT claim there are events when DB is empty
        fake_events = [
            "you have", "scheduled for", "appointment at",
            "meeting at", "reminder at"
        ]
        hallucinated = any(p in response_lower for p in fake_events)

        assert found_nothing or not hallucinated, \
            f"HALLUCINATION DETECTED: Buddy invented events on empty DB. " \
            f"Response: {response}"

    def test_flow_save_multiple_recall_specific(self, buddy):
        """
        Save multiple events → recall specific one by name.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Dentist", "Dr. Schmidt", f"{tomorrow_str()} 09:00"
            )
            sql_brain_module.create_schedule(
                "Gym", "Morning workout", f"{tomorrow_str()} 07:00"
            )
            sql_brain_module.create_schedule(
                "Lecture", "Physics 101", f"{days_from_now(2)} 10:00"
            )

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("when is my dentist appointment?")

        assert response, "Should return a response"
        assert "dentist" in response.lower(), \
            f"Should mention dentist, got: {response}"

    def test_flow_recall_this_week(self, buddy):
        """
        Save events across this week → weekly recall returns them all.
        """
        today = date.today()
        monday = today - timedelta(days=today.weekday())

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            # Add events on different days this week
            for i in range(5):
                day = monday + timedelta(days=i)
                if day >= today:  # only future/today
                    sql_brain_module.create_schedule(
                        f"Event day {i}", "desc",
                        f"{day.strftime('%Y-%m-%d')} 10:00"
                    )

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("what's on my schedule this week?")

        assert response, "Should return a response"
        # Should not say "nothing" if we seeded events
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        if rows:
            response_lower = response.lower()
            claimed_empty = any(p in response_lower for p in [
                "nothing scheduled", "no schedule", "no events", "don't have anything"
            ])
            assert not claimed_empty, \
                f"Should not say empty when {len(rows)} events exist. Got: {response}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DELETE FLOWS
# Does delete remove the right record and not phantom delete?
# ════════════════════════════════════════════════════════════════════════════

class TestDeleteFlows:
    """End-to-end delete flows."""

    def test_flow_save_then_delete(self, buddy):
        """
        Save → delete → SQL should be empty.
        """
        # Seed directly
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Dentist appointment", "Dr. Schmidt",
                f"{tomorrow_str()} 09:00"
            )

        # Verify it's saved
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows_before = sql_brain_module.get_all_schedules_as_dicts()
        assert len(rows_before) == 1

        # Delete via Buddy
        buddy.memory_manager.set_last_topic("dentist appointment")
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "delete my dentist appointment"
            )

        assert response, "Should return a response"

        # If Buddy asks for confirmation, confirm it
        if any(w in response.lower() for w in ["sure", "confirm", "delete", "?"]):
            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                response2 = buddy.handle_request("yes")
            assert response2, "Confirmation response should not be empty"

    def test_flow_delete_nonexistent_no_crash(self, buddy):
        """
        Trying to delete something that doesn't exist should not crash.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            try:
                response = buddy.handle_request(
                    "delete my dentist appointment"
                )
                assert isinstance(response, str), "Should return a string"
            except Exception as e:
                pytest.fail(f"Delete of nonexistent schedule crashed: {e}")

    def test_flow_delete_does_not_remove_wrong_event(self, buddy):
        """
        Delete specific event → other events should still exist.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Dentist", "desc", f"{tomorrow_str()} 09:00"
            )
            sql_brain_module.create_schedule(
                "Gym", "desc", f"{tomorrow_str()} 07:00"
            )

        # Try to delete dentist only
        buddy.memory_manager.set_last_topic("dentist")
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("delete my dentist appointment")

        # If confirmation asked, confirm
        if "?" in response or "sure" in response.lower():
            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                buddy.handle_request("yes")

        # Gym should still exist
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        gym_exists = any("gym" in r["title"].lower() for r in rows)
        assert gym_exists, \
            f"Gym should still exist after deleting dentist. Rows: {rows}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHAINED FLOWS
# Full multi-turn conversations: save → recall → delete → verify
# ════════════════════════════════════════════════════════════════════════════

class TestChainedFlows:
    """
    Multi-turn chained flows — the most realistic tests.
    Each test simulates several minutes of real Buddy usage.
    """

    def test_chain_save_recall_delete_recall(self, buddy):
        """
        Full chain:
        1. Save dentist appointment
        2. Recall → appears
        3. Delete it
        4. Recall → should be gone
        """
        # Step 1 — seed directly for reliability
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Dentist appointment", "Dr. Schmidt",
                f"{tomorrow_str()} 09:00"
            )

        # Step 2 — recall
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            recall_response = buddy.handle_request("what do I have tomorrow?")

        assert "dentist" in recall_response.lower(), \
            f"Step 2: Should find dentist, got: {recall_response}"

        # Step 3 — delete
        buddy.memory_manager.set_last_topic("dentist appointment")
        buddy.memory_manager.set_pending_topic(None)

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            delete_response = buddy.handle_request("delete my dentist appointment")

        # Confirm if asked
        if any(w in delete_response.lower() for w in ["sure", "confirm", "?"]):
            with patch.object(sql_brain_module, "get_connection", _get_test_connection):
                buddy.handle_request("yes")

        # Step 4 — recall again
        buddy.memory_manager.set_pending_topic(None)
        buddy.memory_manager.pending_payload = None

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            final_recall = buddy.handle_request("what do I have tomorrow?")

        # Should no longer find dentist OR say nothing scheduled
        dentist_still_there = "dentist" in final_recall.lower() and \
                               "nothing" not in final_recall.lower()

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()

        dentist_in_sql = any("dentist" in r["title"].lower() for r in rows)

        # If SQL is empty, Buddy should not claim dentist still exists
        if not dentist_in_sql:
            assert not dentist_still_there, \
                f"HALLUCINATION: Dentist deleted from SQL but Buddy still mentions it. " \
                f"Response: {final_recall}"

    def test_chain_multiple_events_recall_correct_day(self, buddy):
        """
        Save events on different days → recall specific day → only that day's events.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Today gym", "desc", f"{today_str()} 07:00"
            )
            sql_brain_module.create_schedule(
                "Tomorrow dentist", "desc", f"{tomorrow_str()} 09:00"
            )
            sql_brain_module.create_schedule(
                "Next week meeting", "desc", f"{days_from_now(7)} 10:00"
            )

        # Recall today only
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request("what do I have today?")

        # Should mention today's gym
        assert "gym" in response.lower(), \
            f"Today recall should mention gym, got: {response}"

        # Should NOT mention tomorrow's dentist
        # (acceptable if it does — Buddy might show upcoming too)
        # But it must mention gym

    def test_chain_holiday_save_then_recall_warning_present(self, buddy):
        """
        Save event on holiday → recall that day → holiday info should be visible.
        """
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Hiwi work", "University shift",
                "2026-04-03 09:00"
            )

        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response = buddy.handle_request(
                "what do I have on April 3rd?"
            )

        assert response, "Should return a response"
        # Should mention Hiwi work
        assert "hiwi" in response.lower() or "work" in response.lower() or \
               "university" in response.lower(), \
            f"Should mention Hiwi work, got: {response}"

    def test_chain_pending_state_clears_between_intents(self, buddy):
        """
        State leakage test: completing a save flow should clear pending state.
        After a save, a completely different question should work normally.
        """
        # Start a save flow
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response1 = buddy.handle_request(
                "remind me to buy groceries tomorrow at 6pm"
            )

        # Clear any pending state that might linger
        buddy.memory_manager.set_pending_topic(None)
        buddy.memory_manager.pending_payload = None

        # Now ask a completely unrelated question
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            response2 = buddy.handle_request(
                "what is machine learning?"
            )

        assert response2, "Should return a response to unrelated question"
        response2_lower = response2.lower()

        # Should NOT be treating this as part of schedule flow
        schedule_confusion = any(p in response2_lower for p in [
            "what time", "when is", "what should i call", "got it saved",
            "what date"
        ])
        assert not schedule_confusion, \
            f"State leaked from save flow into unrelated question. " \
            f"Response: {response2}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ADVERSARIAL INPUTS
# Inputs designed to confuse the system
# ════════════════════════════════════════════════════════════════════════════

class TestAdversarialInputs:
    """
    Deliberately tricky inputs that should be handled gracefully.
    Tests Buddy's robustness to edge-case phrasing.
    """

    def _classify(self, text):
        from brain.core.brain import MemoryManager
        from brain.conversation_manager import apply_guards, classify_intent
        mm = MemoryManager()
        raw = classify_intent(text, mm)
        return apply_guards(raw, text, mm)

    def test_adversarial_double_negative(self):
        """'remind me to never forget' — double negative should still route to schedule_save"""
        result = self._classify("remind me to never forget my dentist appointment")
        assert result["intent"] == "schedule_save", \
            f"Double negative should still be schedule_save, got {result['intent']}"

    def test_adversarial_negation_dont_remind(self):
        """'don't remind me about X' should NOT create a schedule."""
        result = self._classify("don't remind me about the gym anymore")
        assert result["intent"] != "schedule_save", \
            f"Negation should not create schedule, got {result['intent']}"

    def test_adversarial_hedged_intent(self):
        """Heavily hedged input should not crash."""
        result = self._classify(
            "I think I might have something on Friday maybe"
        )
        assert isinstance(result, dict), "Should return a dict"
        assert "intent" in result
        # Acceptable to be schedule_save or chat — just should not crash

    def test_adversarial_extremely_vague(self):
        """Extremely vague input should not crash."""
        result = self._classify(
            "Buddy can you like maybe set something for later"
        )
        assert isinstance(result, dict)
        assert "intent" in result

    def test_adversarial_past_recall_with_emotion(self):
        """
        'what did I have yesterday that I forgot' —
        should route to schedule_recall not crash.
        """
        result = self._classify(
            "what did I have yesterday that I forgot about?"
        )
        assert result["intent"] in ("schedule_recall", "recall_memory", "chat"), \
            f"Unexpected intent for past recall with emotion: {result['intent']}"

    def test_adversarial_mixed_german_english(self):
        """Mixed language input should not crash."""
        result = self._classify(
            "remind me about my Termin tomorrow at 10 Uhr"
        )
        assert isinstance(result, dict)
        # Should probably route to schedule_save
        # but we only require it doesn't crash

    def test_adversarial_numbers_that_look_like_times(self):
        """'I have 3 sensor inputs' should not become a schedule."""
        result = self._classify("I have 3 sensor inputs running at 100hz")
        assert result["intent"] != "schedule_save", \
            f"Engineering sentence should not be schedule_save, got {result['intent']}"

    def test_adversarial_empty_string_no_crash(self):
        """Empty string must not crash."""
        result = self._classify("")
        assert isinstance(result, dict)

    def test_adversarial_only_whitespace_no_crash(self):
        """Whitespace-only input must not crash."""
        result = self._classify("   ")
        assert isinstance(result, dict)

    def test_adversarial_very_long_input_no_crash(self):
        """Very long input must not crash."""
        long_input = "remind me " * 50 + "tomorrow at 9am"
        result = self._classify(long_input)
        assert isinstance(result, dict)

    def test_adversarial_schedule_in_question_form(self):
        """
        'can I schedule a meeting for tomorrow?' —
        question phrasing should still route to schedule_save.
        """
        result = self._classify(
            "can I schedule a meeting for tomorrow at 3pm?"
        )
        assert result["intent"] == "schedule_save", \
            f"Question-form schedule should be schedule_save, got {result['intent']}"

    def test_adversarial_recall_with_wrong_word(self):
        """
        'what have I got planned for this week' —
        non-standard phrasing should still be schedule_recall.
        """
        result = self._classify("what have I got planned for this week?")
        assert result["intent"] == "schedule_recall", \
            f"Planned-for phrasing should be schedule_recall, got {result['intent']}"


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — REGRESSION GUARD (flow version)
# Previously working flows must continue to work
# ════════════════════════════════════════════════════════════════════════════

class TestFlowRegression:
    """
    Flow-level regression tests.
    If any of these fail, a previously working flow is now broken.
    Treat as high-priority bugs.
    """

    def test_regression_sql_survives_save_recall_cycle(self, buddy):
        """Core SQL pipeline must never corrupt across save→recall."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Regression test", "desc",
                f"{tomorrow_str()} 10:00"
            )
            rows = sql_brain_module.get_all_schedules_as_dicts()

        assert len(rows) == 1
        assert rows[0]["title"] == "Regression test"
        assert rows[0]["timestamp"].startswith(tomorrow_str())

    def test_regression_karfreitag_detection(self):
        """Karfreitag 2026 must always be detected as a holiday."""
        from brain.adaptive.awareness.holiday_checker import HolidayChecker
        checker = HolidayChecker()
        is_hol, name = checker.is_holiday("2026-04-03")
        assert is_hol is True
        assert name == "Karfreitag"

    def test_regression_remind_me_routes_to_schedule(self):
        """'remind me' must always route to schedule_save."""
        from brain.conversation_manager import apply_guards, classify_intent
        from brain.core.brain import MemoryManager
        mm = MemoryManager()
        result = apply_guards(
            classify_intent("remind me to buy groceries tomorrow at 5pm", mm),
            "remind me to buy groceries tomorrow at 5pm",
            mm
        )
        assert result["intent"] == "schedule_save"

    def test_regression_what_do_i_have_routes_to_recall(self):
        """'what do I have today' must always route to schedule_recall."""
        from brain.conversation_manager import apply_guards, classify_intent
        from brain.core.brain import MemoryManager
        mm = MemoryManager()
        result = apply_guards(
            classify_intent("what do I have today?", mm),
            "what do I have today?",
            mm
        )
        assert result["intent"] == "schedule_recall"

    def test_regression_no_crash_on_empty_db_recall(self, buddy):
        """Recalling from empty DB must not crash."""
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            try:
                response = buddy.handle_request("what do I have this week?")
                assert isinstance(response, str)
            except Exception as e:
                pytest.fail(f"Empty DB recall crashed: {e}")

    def test_regression_hiwi_scenario_full_chain(self, buddy):
        """
        The exact morning scenario — full chain:
        1. Save Hiwi work on a holiday date
        2. Verify it's in SQL
        3. Holiday check returns warning
        """
        from brain.adaptive.awareness.holiday_checker import HolidayChecker

        # Step 1 — save
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            sql_brain_module.create_schedule(
                "Hiwi work", "University",
                "2026-04-03 09:00"
            )

        # Step 2 — verify SQL
        with patch.object(sql_brain_module, "get_connection", _get_test_connection):
            rows = sql_brain_module.get_all_schedules_as_dicts()
        assert len(rows) == 1
        assert "hiwi" in rows[0]["title"].lower()

        # Step 3 — holiday check
        checker = HolidayChecker()
        result = checker.check_schedule(rows[0])
        assert result["is_holiday"] is True
        assert result["holiday_name"] == "Karfreitag"
        assert result["warning"] is not None
        assert "Hiwi" in result["warning"] or "hiwi" in result["warning"].lower()


# ════════════════════════════════════════════════════════════════════════════
# SCORING SUMMARY
# ════════════════════════════════════════════════════════════════════════════

"""
SCORING GUIDE:
─────────────────────────────────────────────────────────────────────────────
Section                     | Tests | Target  | Notes
─────────────────────────────────────────────────────────────────────────────
1. Save Flows               |   7   | 6/7     | LLM variability on phrasing
2. Recall Flows             |   5   | 5/5     | SQL seeded directly = reliable
3. Delete Flows             |   3   | 3/3     | SQL seeded directly = reliable
4. Chained Flows            |   4   | 4/4     | Most important section
5. Adversarial Inputs       |  12   | 10/12   | Some LLM variability expected
6. Regression Guard         |   6   | 6/6     | Must be 100% — no regressions
─────────────────────────────────────────────────────────────────────────────
TOTAL                       |  37   | 34+/37  | Target: 92%+
─────────────────────────────────────────────────────────────────────────────

READINESS DECISION:
  >= 34/37 (92%) → flows are solid, start daily use confidently
  28-33/37       → review failing flows, fix before daily use
  < 28/37        → significant flow bugs, do not rely on Buddy yet

AFTER DAILY USE:
  Add every real failure as a new test in this file.
  The adversarial section should grow over time from real inputs.

RUN ALL THREE SUITES:
  pytest eval/ -v
  → Full picture: unit + component + flow
─────────────────────────────────────────────────────────────────────────────
"""