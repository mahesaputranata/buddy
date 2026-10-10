"""
test_device_routing.py
======================
Layer 2-3 tests: DeviceHandlerFSM._infer_device() and
RoverTaskManager._belongs_to_rover_context()

Tests whether utterances are correctly routed to the rover
BEFORE any command is parsed or sent to the Pi.

Run:
    python eval/test_device_routing.py
    python eval/test_device_routing.py --verbose
    python eval/test_device_routing.py --category infer_device
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Minimal mocks ────────────────────────────────────────────────────────────

class MockMemoryManager:
    def __init__(self):
        self._pending = None
        self._last_topic = None
        self.pending_payload = {}

    def get_pending_topic(self): return self._pending
    def set_pending_topic(self, v): self._pending = v
    def get_last_topic(self): return self._last_topic
    def set_last_topic(self, v): self._last_topic = v

class MockRoverBridge:
    """Simulates RoverClient — returns a canned OK response without hitting the Pi."""
    is_connected = True

    def send_command(self, cmd: dict) -> dict:
        return {
            "ok": True,
            "direction": cmd.get("direction", "stop"),
            "speed": cmd.get("speed_percent", 0),
            "battery": 85,
            "user_message": f"Mock OK: {cmd.get('action')}",
        }

    def emergency_stop(self): return {"ok": True}
    def disconnect_message(self): return "Mock disconnected."
    def set_telemetry_callback(self, cb): pass


# ── Import under test ────────────────────────────────────────────────────────

from brain.core.devices.device_handler import DeviceHandlerFSM
from devices.rover.rover_controller import RoverTaskManager


def make_fsm():
    mm = MockMemoryManager()
    fsm = DeviceHandlerFSM(
        mm=mm,
        tello_client=None,
        rover_client=MockRoverBridge(),
    )
    return fsm, mm


def make_task_manager(session_active=False):
    bridge = MockRoverBridge()
    tm = RoverTaskManager(rover_bridge=bridge, mm=MockMemoryManager())
    tm.session_active = session_active
    return tm


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# Format: (input_text, expected_result, description, category, extra_kwargs)
# ══════════════════════════════════════════════════════════════════════════════

# ── Category: infer_device ──────────────────────────────────────────────────
# Tests DeviceHandlerFSM._infer_device()
# Expected: "rover" | "tello" | None

INFER_DEVICE_CASES = [
    # Clear rover triggers
    ("rover move forward",          "rover",  "explicit 'rover' keyword"),
    ("control the rover",           "rover",  "rover keyword in command"),
    ("rover stop",                  "rover",  "rover stop"),
    ("lidar is detecting something","rover",  "lidar keyword → rover"),
    ("activate rover",              "rover",  "activate rover"),

    # Clear tello triggers
    ("tello take off",              "tello",  "explicit tello keyword"),
    ("fly the drone",               "tello",  "drone keyword"),
    ("connect to drone",            "tello",  "drone connect"),

    # KNOWN GAPS — these will likely FAIL with current code
    # These are what you need to fix before testing with the real Pi
    ("move forward",                "rover",  "GAP: no 'rover' word, session not active"),
    ("go forward",                  "rover",  "GAP: casual movement, no device word"),
    ("turn left",                   "rover",  "GAP: turn command without 'rover'"),
    ("follow me",                   "rover",  "GAP: follow me without 'rover' prefix"),
    ("stop the robot",              "rover",  "GAP: 'robot' not in _infer_device"),
    ("find the door",               "rover",  "GAP: vision mode without 'rover'"),
]

# ── Category: rover_context ─────────────────────────────────────────────────
# Tests RoverTaskManager._belongs_to_rover_context()
# Expected: True | False

ROVER_CONTEXT_CASES_NO_SESSION = [
    # Should return True (rover words present)
    ("rover move forward",      True,  "rover keyword present"),
    ("control the rover",       True,  "rover keyword present"),
    ("motor is spinning",       True,  "motor keyword"),
    ("lidar scan",              True,  "lidar keyword"),

    # Should return False (no rover words, no session)
    ("what time is it",         False, "system query, no session"),
    ("tell me about NLP",       False, "chat topic, no session"),
    ("how does a PID work",     False, "engineering chat, no session"),

    # KNOWN GAPS — these will FAIL without active session
    ("go forward",              False, "no session + no rover word → correctly False"),
    ("move backward",           False, "no session + no rover word → correctly False"),
    ("follow me",               False, "no session + no rover word → correctly False"),
]

ROVER_CONTEXT_CASES_WITH_SESSION = [
    # With session active, follow-up words should return True
    ("go forward",              True,  "session active + movement word"),
    ("turn left",               True,  "session active + turn word"),
    ("stop",                    True,  "session active + stop"),
    ("faster",                  True,  "session active + speed word"),
    ("slower",                  True,  "session active + speed word"),
    ("go backward",             True,  "session active + backward"),

    # KNOWN GAPS — these will FAIL even with session active
    ("follow me",               True,  "GAP: 'follow' not in followup_words list"),
    ("find the door",           True,  "GAP: 'find' not in followup_words list"),
    ("stop following",          True,  "GAP: only 'stop' would match, not 'stop following'"),
    ("go to sleep",             False, "non-rover phrase with session active"),
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_infer_device(verbose=False):
    print(f"\n{'─'*65}")
    print("  LAYER 2: _infer_device()  (DeviceHandlerFSM)")
    print(f"{'─'*65}")

    passed = failed = 0
    failures = []

    for text, expected, desc in INFER_DEVICE_CASES:
        fsm, _ = make_fsm()
        result = fsm._infer_device(text)

        if result == expected:
            passed += 1
            if verbose:
                print(f"  ✓ {desc}")
        else:
            failed += 1
            failures.append((text, expected, result, desc))

    score = passed / (passed + failed) * 100
    status = "✓" if score >= 85 else ("△" if score >= 60 else "✗")
    print(f"  {status} Score: {passed}/{passed+failed} ({score:.0f}%)")

    if failures:
        print(f"  Failures:")
        for text, exp, got, desc in failures:
            gap = "← FIX BEFORE PI" if "GAP" in desc else ""
            print(f"    ✗ {desc}")
            print(f"      Input:    {text!r}")
            print(f"      Expected: {exp!r:<8} Got: {got!r}  {gap}")

    return passed, failed


def run_rover_context(verbose=False):
    print(f"\n{'─'*65}")
    print("  LAYER 3: _belongs_to_rover_context()  (RoverTaskManager)")
    print(f"{'─'*65}")

    all_cases = [
        (c + (False,) for c in ROVER_CONTEXT_CASES_NO_SESSION),
        (c + (True,)  for c in ROVER_CONTEXT_CASES_WITH_SESSION),
    ]

    passed = failed = 0
    failures = []

    for session_flag, cases in [
        (False, ROVER_CONTEXT_CASES_NO_SESSION),
        (True,  ROVER_CONTEXT_CASES_WITH_SESSION),
    ]:
        label = "session=ACTIVE" if session_flag else "session=INACTIVE"
        print(f"\n  [{label}]")

        for text, expected, desc in cases:
            tm = make_task_manager(session_active=session_flag)
            result = tm._belongs_to_rover_context(tm._norm(text))

            if result == expected:
                passed += 1
                if verbose:
                    print(f"    ✓ {desc}")
            else:
                failed += 1
                failures.append((text, expected, result, desc, label))

    score = passed / (passed + failed) * 100
    status = "✓" if score >= 85 else ("△" if score >= 60 else "✗")
    print(f"\n  {status} Score: {passed}/{passed+failed} ({score:.0f}%)")

    if failures:
        print(f"  Failures:")
        for text, exp, got, desc, label in failures:
            gap = "← FIX BEFORE PI" if "GAP" in desc else ""
            print(f"    ✗ [{label}] {desc}")
            print(f"      Input:    {text!r}")
            print(f"      Expected: {exp!r:<8} Got: {got!r}  {gap}")

    return passed, failed


def run_all(verbose=False):
    print(f"\n{'═'*65}")
    print("  ROVER DEVICE ROUTING TESTS  (Layers 2-3)")
    print(f"{'═'*65}")

    p1, f1 = run_infer_device(verbose)
    p2, f2 = run_rover_context(verbose)

    total_p = p1 + p2
    total_f = f1 + f2
    total   = total_p + total_f
    score   = total_p / total * 100 if total else 0

    print(f"\n{'═'*65}")
    print(f"  OVERALL: {total_p}/{total} ({score:.0f}%)")
    print(f"{'═'*65}")

    print("\n  What to fix before testing with the Pi:")
    print("  1. _infer_device(): add 'robot' to rover keywords")
    print("  2. _infer_device(): when session active, skip device inference entirely")
    print("  3. _belongs_to_rover_context(): add 'follow', 'find' to followup_words")
    print("  4. _infer_device(): consider accepting None if rover session already open\n")

    return total_p, total_f, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", "-c", choices=["infer_device", "rover_context"], default=None)
    args = parser.parse_args()

    if args.category == "infer_device":
        run_infer_device(args.verbose)
    elif args.category == "rover_context":
        run_rover_context(args.verbose)
    else:
        run_all(args.verbose)