"""
test_command_builder.py
=======================
Layer 4 tests: RoverTaskManager._parse_command()

Tests that every utterance produces the EXACT JSON dict that
rover_server.py on the Pi expects to receive.

This is the most critical pre-Pi test — if the wrong JSON arrives,
the rover does the wrong thing (or crashes).

Run:
    python eval/test_command_builder.py
    python eval/test_command_builder.py --verbose
    python eval/test_command_builder.py --category movement
"""

import sys
import os
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from devices.rover.rover_controller import RoverTaskManager


class MockRoverBridge:
    is_connected = True
    def send_command(self, cmd): return {"ok": True, "battery": 85}
    def emergency_stop(self): return {"ok": True}
    def disconnect_message(self): return "Mock disconnected."
    def set_telemetry_callback(self, cb): pass


def make_tm(session_active=True, current_speed=0, current_direction="stop"):
    tm = RoverTaskManager(rover_bridge=MockRoverBridge())
    tm.session_active      = session_active
    tm.current_speed       = current_speed
    tm.current_direction   = current_direction
    return tm


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# Format: (input_text, expected_cmd_dict, description, category, setup_kwargs)
#
# expected_cmd_dict uses None to mean "don't care about this field"
# Use check_keys to only assert specific keys
# ══════════════════════════════════════════════════════════════════════════════

def fresh():    return {}
def with_speed(): return {"current_speed": 40, "current_direction": "forward"}

COMMAND_CASES = [

    # ══════════════════════════════════════════════════════
    # CATEGORY: movement
    # Basic directional movement commands
    # ══════════════════════════════════════════════════════

    (
        "rover move forward",
        {"action": "move", "direction": "forward"},
        "basic forward",
        "movement", fresh,
    ),
    (
        "go forward",
        {"action": "move", "direction": "forward"},
        "casual forward",
        "movement", fresh,
    ),
    (
        "rover move backward",
        {"action": "move", "direction": "backward"},
        "basic backward",
        "movement", fresh,
    ),
    (
        "go back",
        {"action": "move", "direction": "backward"},
        "casual go back",
        "movement", fresh,
    ),
    (
        "rover turn left",
        {"action": "turn", "direction": "left"},
        "turn left",
        "movement", fresh,
    ),
    (
        "rover turn right",
        {"action": "turn", "direction": "right"},
        "turn right",
        "movement", fresh,
    ),
    (
        "rover stop",
        {"action": "set_mode", "mode": "IDLE"},
        "stop command → IDLE mode",
        "movement", fresh,
    ),

    # ══════════════════════════════════════════════════════
    # CATEGORY: speed
    # Speed adjustment commands
    # ══════════════════════════════════════════════════════

    (
        "go faster",
        {"action": "set_speed"},
        "increase speed",
        "speed", with_speed,
    ),
    (
        "go slower",
        {"action": "set_speed"},
        "decrease speed",
        "speed", with_speed,
    ),
    (
        "full speed",
        {"action": "set_speed"},
        "max speed",
        "speed", with_speed,
    ),
    (
        "maximum speed",
        {"action": "set_speed"},
        "GAP: 'maximum speed' not handled, only 'full speed'/'max speed'",
        "speed", with_speed,
    ),

    # ══════════════════════════════════════════════════════
    # CATEGORY: vision_mode
    # Vision mode commands → should produce set_mode payloads
    # These are what rover_server.py looks for in the new mode branch
    # ══════════════════════════════════════════════════════

    (
        "follow me",
        {"action": "set_mode", "mode": "FOLLOW_ME"},
        "follow me → FOLLOW_ME",
        "vision_mode", fresh,
    ),
    (
        "start following",
        {"action": "set_mode", "mode": "FOLLOW_ME"},
        "start following → FOLLOW_ME",
        "vision_mode", fresh,
    ),
    (
        "rover follow me",
        {"action": "set_mode", "mode": "FOLLOW_ME"},
        "rover follow me → FOLLOW_ME",
        "vision_mode", fresh,
    ),
    (
        "find the door",
        {"action": "set_mode", "mode": "FIND_DOOR"},
        "GAP: 'find the door' has 'the' — does trigger match?",
        "vision_mode", fresh,
    ),
    (
        "find door",
        {"action": "set_mode", "mode": "FIND_DOOR"},
        "find door → FIND_DOOR",
        "vision_mode", fresh,
    ),
    (
        "search for a door",
        {"action": "set_mode", "mode": "FIND_DOOR"},
        "search for a door → FIND_DOOR",
        "vision_mode", fresh,
    ),
    (
        "stop following",
        {"action": "set_mode", "mode": "IDLE"},
        "stop following → IDLE",
        "vision_mode", fresh,
    ),
    (
        "exit follow mode",
        {"action": "set_mode", "mode": "IDLE"},
        "exit follow → IDLE",
        "vision_mode", fresh,
    ),
    (
        "stop vision",
        {"action": "set_mode", "mode": "IDLE"},
        "stop vision → IDLE",
        "vision_mode", fresh,
    ),
    (
        "manual mode",
        {"action": "set_mode", "mode": "IDLE"},
        "manual mode → IDLE",
        "vision_mode", fresh,
    ),

    # ══════════════════════════════════════════════════════
    # CATEGORY: activate
    # Connect/activate → status command
    # ══════════════════════════════════════════════════════

    (
        "activate rover",
        {"action": "status"},
        "activate → status",
        "activate", fresh,
    ),
    (
        "rover connect",
        {"action": "status"},
        "connect → status",
        "activate", fresh,
    ),
    (
        "wake up rover",
        {"action": "status"},
        "wake → status",
        "activate", fresh,
    ),

    # ══════════════════════════════════════════════════════
    # CATEGORY: gaps
    # Commands that SHOULD work but likely DON'T with current parser
    # These are what you need to fix before testing with the Pi
    # ══════════════════════════════════════════════════════

    (
        "reverse",
        {"action": "move", "direction": "backward"},
        "GAP: 'reverse' not in parser",
        "gaps", fresh,
    ),
    (
        "turn around",
        None,  # None = expect None (unhandled) — this is a known gap
        "GAP: 'turn around' not handled",
        "gaps", fresh,
    ),
    (
        "go straight",
        {"action": "move", "direction": "forward"},
        "GAP: 'straight' not in parser",
        "gaps", fresh,
    ),
    (
        "can you move forward please",
        {"action": "move", "direction": "forward"},
        "GAP: polite phrasing — filler stripper handles this?",
        "gaps", fresh,
    ),
    (
        "a little to the left",
        {"action": "turn", "direction": "left"},
        "GAP: 'a little to the left' — matches 'left'?",
        "gaps", fresh,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = ["movement", "speed", "vision_mode", "activate", "gaps"]

CATEGORY_LABELS = {
    "movement":    "Movement Commands    (forward/backward/left/right/stop)",
    "speed":       "Speed Commands       (faster/slower/full speed)",
    "vision_mode": "Vision Mode Commands (follow me / find door / stop following)",
    "activate":    "Activate/Connect     (activate/connect/wake)",
    "gaps":        "Known Gaps           (commands that likely break — FIX THESE)",
}


def _check_cmd(result, expected):
    """
    Returns True if result matches expected.
    expected=None means we expect _parse_command to return None.
    """
    if expected is None:
        return result is None

    if result is None:
        return False

    for k, v in expected.items():
        if v is None:
            continue  # don't-care
        if result.get(k) != v:
            return False
    return True


def run_tests(filter_category=None, verbose=False):
    cases = COMMAND_CASES
    if filter_category:
        cases = [c for c in cases if c[3] == filter_category]
        if not cases:
            print(f"No cases for category: {filter_category}")
            print(f"Available: {', '.join(CATEGORIES)}")
            return

    by_cat = {c: [] for c in CATEGORIES}
    for case in cases:
        by_cat[case[3]].append(case)

    total_p = total_f = 0
    all_failures = []
    cat_scores = {}

    print(f"\n{'═'*65}")
    print(f"  ROVER COMMAND BUILDER TESTS  (Layer 4: _parse_command)")
    print(f"{'═'*65}")

    for cat in CATEGORIES:
        cat_cases = by_cat[cat]
        if not cat_cases:
            continue

        cat_p = cat_f = 0
        cat_failures = []

        for text, expected_cmd, desc, _, setup_fn in cat_cases:
            kwargs = setup_fn()
            tm = make_tm(
                session_active   = True,
                current_speed    = kwargs.get("current_speed", 0),
                current_direction= kwargs.get("current_direction", "stop"),
            )

            normed = tm._norm(text)
            result = tm._parse_command(normed)

            ok = _check_cmd(result, expected_cmd)

            if ok:
                cat_p += 1
                total_p += 1
                if verbose:
                    print(f"  ✓ {desc}")
            else:
                cat_f += 1
                total_f += 1
                cat_failures.append({
                    "input":    text,
                    "normed":   normed,
                    "expected": expected_cmd,
                    "got":      result,
                    "desc":     desc,
                })
                all_failures.append(cat_failures[-1])

        total_cat = cat_p + cat_f
        score = cat_p / total_cat * 100 if total_cat else 0
        cat_scores[cat] = (cat_p, total_cat, score)

        status = "✓" if score >= 85 else ("△" if score >= 60 else "✗")
        label  = CATEGORY_LABELS.get(cat, cat)

        print(f"\n  {status} {label}")
        print(f"    Score: {cat_p}/{total_cat} ({score:.0f}%)")

        if cat_failures:
            print(f"    Failures:")
            for f in cat_failures:
                short = f['input'][:55] + "..." if len(f['input']) > 55 else f['input']
                gap = "← FIX BEFORE PI" if "GAP" in f['desc'] else ""
                print(f"      ✗ {f['desc']}")
                print(f"        Input:    {short!r}")
                print(f"        Expected: {f['expected']}")
                print(f"        Got:      {f['got']}  {gap}")

    total = total_p + total_f
    score = total_p / total * 100 if total else 0

    print(f"\n{'═'*65}")
    print(f"  OVERALL: {total_p}/{total} ({score:.0f}%)")
    print(f"{'═'*65}")

    # Actionable fix list
    print("\n  JSON payload mismatches (what the Pi will receive vs what it expects):")
    payload_issues = [f for f in all_failures if "GAP" in f["desc"]]
    if payload_issues:
        for f in payload_issues:
            print(f"    • {f['desc']}")
            print(f"      Pi receives: {f['got']}")
            print(f"      Pi expects:  {f['expected']}")
    else:
        print("    None — all tested payloads are correct.")

    print()
    return total_p, total_f, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", "-c", choices=CATEGORIES, default=None)
    args = parser.parse_args()

    run_tests(filter_category=args.category, verbose=args.verbose)