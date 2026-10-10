"""
test_rover_mock_server.py
=========================
Layer 5 tests: Full round-trip without the real Pi.

Spins up a local mock WebSocket server that mimics rover_server.py,
then fires real RoverClient.send_command() calls at it.

Tests:
  - Does the JSON payload arrive at the server correctly?
  - Does the server's ACK get parsed correctly by RoverClient?
  - Does the full pipeline produce the right natural reply?

Run:
    python eval/test_rover_mock_server.py
    python eval/test_rover_mock_server.py --verbose
    python eval/test_rover_mock_server.py --category payload
"""

import sys
import os
import asyncio
import json
import threading
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
#  Mock Pi Server  (mimics rover_server.py behaviour)
# ══════════════════════════════════════════════════════════════════════════════

MOCK_PORT = 18765   # use a different port so it never clashes with the real Pi

# Shared log — filled by the mock server, read by test assertions
_received_commands: list = []
_lock = threading.Lock()


async def _mock_handler(websocket):
    """Receives one command, logs it, sends a realistic ACK."""
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        cmd = json.loads(raw)

        with _lock:
            _received_commands.append(cmd)

        action = cmd.get("action", "")

        # Simulate rover_server.py ACK structure
        if action == "set_mode":
            mode = cmd.get("mode", "IDLE")
            ack = {
                "ok":        True,
                "mode":      mode,
                "status":    f"Vision mode {mode} started." if mode != "IDLE" else "Vision stopped.",
                "battery":   72,
                "timestamp": time.time(),
            }
        elif action == "stop":
            ack = {
                "ok":        True,
                "error":     None,
                "direction": "stop",
                "speed":     0,
                "mode":      "IDLE",
                "battery":   72,
                "timestamp": time.time(),
            }
        elif action in ("move", "turn"):
            ack = {
                "ok":        True,
                "error":     None,
                "direction": cmd.get("direction", "stop"),
                "speed_pct": cmd.get("speed_percent", 0),
                "speed":     cmd.get("speed_percent", 0),
                "battery":   72,
                "linear_x":  0.12,
                "angular_z": 0.0,
                "timestamp": time.time(),
            }
        elif action == "status":
            ack = {
                "ok":        True,
                "error":     None,
                "direction": "stop",
                "speed_pct": 0,
                "battery":   72,
                "timestamp": time.time(),
            }
        elif action == "set_speed":
            ack = {
                "ok":        True,
                "error":     None,
                "direction": cmd.get("direction", "stop"),
                "speed_pct": cmd.get("speed_percent", 0),
                "battery":   72,
                "timestamp": time.time(),
            }
        else:
            ack = {"ok": False, "error": f"Unknown action: {action}"}

        await websocket.send(json.dumps(ack))
    except Exception as e:
        print(f"[MOCK SERVER] Handler error: {e}")


def _run_mock_server():
    """Run the mock WebSocket server in a background thread."""
    import websockets

    async def _serve():
        async with websockets.serve(_mock_handler, "127.0.0.1", MOCK_PORT):
            await asyncio.Future()  # run forever

    asyncio.run(_serve())


def start_mock_server():
    t = threading.Thread(target=_run_mock_server, daemon=True, name="MockPiServer")
    t.start()
    time.sleep(0.5)   # give server time to bind
    return t


# ══════════════════════════════════════════════════════════════════════════════
#  Import under test
# ══════════════════════════════════════════════════════════════════════════════

from devices.rover.rover_bridge import RoverClient
from devices.rover.rover_controller import RoverTaskManager


def make_rover_client():
    return RoverClient(host="127.0.0.1", port=MOCK_PORT)


def make_task_manager():
    client = make_rover_client()
    time.sleep(0.3)   # allow async loop to start
    tm = RoverTaskManager(rover_bridge=client)
    tm.session_active = True
    return tm, client


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# Format: (utterance, check_fn, description, category)
#
# check_fn receives:
#   result       — dict returned by RoverTaskManager.handle()
#   received_cmd — the dict the mock server actually received
# ══════════════════════════════════════════════════════════════════════════════

def _ok(result, cmd):
    return result.get("handled") and (cmd is None or cmd.get("ok", True))

def _action_is(expected_action):
    def _check(result, cmd):
        if cmd is None: return False
        return cmd.get("action") == expected_action
    return _check

def _mode_is(expected_mode):
    def _check(result, cmd):
        if cmd is None: return False
        return cmd.get("action") == "set_mode" and cmd.get("mode") == expected_mode
    return _check

def _direction_is(expected_dir):
    def _check(result, cmd):
        if cmd is None: return False
        return cmd.get("direction") == expected_dir
    return _check

def _has_reply(result, cmd):
    return bool((result.get("reply") or "").strip())


ROUNDTRIP_CASES = [

    # ── Category: payload ──────────────────────────────────────────────────
    # Does the correct JSON arrive at the mock Pi server?

    (
        "rover move forward",
        _action_is("move"),
        "forward → action=move reaches server",
        "payload",
    ),
    (
        "rover turn left",
        _action_is("turn"),
        "left → action=turn reaches server",
        "payload",
    ),
    (
        "rover stop",
        _mode_is("IDLE"),
        "stop → set_mode IDLE reaches server",
        "payload",
    ),
    (
        "follow me",
        _mode_is("FOLLOW_ME"),
        "follow me → set_mode FOLLOW_ME reaches server",
        "payload",
    ),
    (
        "find door",
        _mode_is("FIND_DOOR"),
        "find door → set_mode FIND_DOOR reaches server",
        "payload",
    ),
    (
        "stop following",
        _mode_is("IDLE"),
        "stop following → set_mode IDLE reaches server",
        "payload",
    ),
    (
        "activate rover",
        _action_is("status"),
        "activate → status reaches server",
        "payload",
    ),

    # ── Category: ack_parsing ──────────────────────────────────────────────
    # Does RoverClient correctly parse the ACK from the Pi?

    (
        "rover move forward",
        lambda r, c: r.get("bridge_result", {}).get("ok") is True,
        "bridge_result.ok=True parsed correctly",
        "ack_parsing",
    ),
    (
        "rover move forward",
        lambda r, c: r.get("bridge_result", {}).get("battery") == 72,
        "battery field parsed from ACK",
        "ack_parsing",
    ),
    (
        "rover stop",
        lambda r, c: r.get("bridge_result", {}).get("ok") is True,
        "stop ACK parsed correctly",
        "ack_parsing",
    ),

    # ── Category: reply_quality ────────────────────────────────────────────
    # Does the final natural reply make sense?

    (
        "rover move forward",
        _has_reply,
        "forward command produces non-empty reply",
        "reply_quality",
    ),
    (
        "rover stop",
        _has_reply,
        "stop command produces non-empty reply",
        "reply_quality",
    ),
    (
        "follow me",
        _has_reply,
        "follow me produces non-empty reply",
        "reply_quality",
    ),
    (
        "rover move forward",
        lambda r, c: len((r.get("reply") or "")) < 300,
        "reply is concise (< 300 chars)",
        "reply_quality",
    ),

    # ── Category: error_handling ───────────────────────────────────────────
    # What happens when something unexpected occurs?

    (
        "tell me about NLP",
        lambda r, c: not r.get("handled"),
        "non-rover utterance → handled=False, no command sent",
        "error_handling",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = ["payload", "ack_parsing", "reply_quality", "error_handling"]

CATEGORY_LABELS = {
    "payload":       "Payload Correctness  (right JSON arrives at Pi)",
    "ack_parsing":   "ACK Parsing          (Pi response parsed correctly)",
    "reply_quality": "Reply Quality        (natural reply generated)",
    "error_handling":"Error Handling       (graceful failures)",
}


def run_tests(filter_category=None, verbose=False):
    print(f"\n{'═'*65}")
    print(f"  ROVER MOCK SERVER TESTS  (Layer 5: Full round-trip)")
    print(f"  Mock Pi server on ws://127.0.0.1:{MOCK_PORT}")
    print(f"{'═'*65}")

    # Start mock server once
    print("\n  [SETUP] Starting mock Pi server...")
    start_mock_server()
    print("  [SETUP] Mock server ready.\n")

    cases = ROUNDTRIP_CASES
    if filter_category:
        cases = [c for c in cases if c[3] == filter_category]
        if not cases:
            print(f"No cases for category: {filter_category}")
            return

    by_cat = {c: [] for c in CATEGORIES}
    for case in cases:
        by_cat[case[3]].append(case)

    total_p = total_f = 0
    all_failures = []

    for cat in CATEGORIES:
        cat_cases = by_cat[cat]
        if not cat_cases:
            continue

        cat_p = cat_f = 0
        cat_failures = []

        label = CATEGORY_LABELS.get(cat, cat)
        print(f"  Testing: {label}")

        for utterance, check_fn, desc, _ in cat_cases:
            # Fresh task manager per test so state doesn't bleed
            tm, client = make_task_manager()

            # Clear shared log before each test
            with _lock:
                _received_commands.clear()

            try:
                result = tm.handle(utterance)
            except Exception as e:
                result = {"handled": False, "reply": f"EXCEPTION: {e}"}

            # Get what the mock server received
            time.sleep(0.1)  # tiny wait for async round trip
            with _lock:
                received = _received_commands[-1] if _received_commands else None

            ok = check_fn(result, received)

            if ok:
                cat_p += 1
                total_p += 1
                if verbose:
                    print(f"    ✓ {desc}")
            else:
                cat_f += 1
                total_f += 1
                cat_failures.append({
                    "utterance": utterance,
                    "desc":      desc,
                    "result":    result,
                    "received":  received,
                })
                all_failures.append(cat_failures[-1])

        total_cat = cat_p + cat_f
        score = cat_p / total_cat * 100 if total_cat else 0
        status = "✓" if score >= 85 else ("△" if score >= 60 else "✗")
        print(f"  {status} Score: {cat_p}/{total_cat} ({score:.0f}%)\n")

        if cat_failures:
            for f in cat_failures:
                print(f"    ✗ {f['desc']}")
                print(f"      Utterance:      {f['utterance']!r}")
                print(f"      Server received:{f['received']}")
                reply_preview = (f['result'].get('reply') or '')[:80]
                print(f"      Reply:          {reply_preview!r}")
                print()

    total = total_p + total_f
    score = total_p / total * 100 if total else 0

    print(f"{'═'*65}")
    print(f"  OVERALL: {total_p}/{total} ({score:.0f}%)")
    print(f"{'═'*65}")

    if score >= 90:
        print("\n READY — safe to test with the real Pi.")
    elif score >= 70:
        print("\n MOSTLY READY — fix failures above before Pi testing.")
    else:
        print("\n NOT READY — significant gaps, fix Layer 4 first.")

    print()
    return total_p, total_f, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", "-c", choices=CATEGORIES, default=None)
    args = parser.parse_args()

    run_tests(filter_category=args.category, verbose=args.verbose)
