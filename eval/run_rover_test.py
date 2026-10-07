"""
run_rover_tests.py
==================
Master test runner — runs all three rover test layers in order
and gives a clear GO / NO-GO verdict for Pi testing.

Run:
    python eval/run_rover_tests.py
    python eval/run_rover_tests.py --verbose
    python eval/run_rover_tests.py --layer 2   (run only one layer)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import each test module
from eval.test_device_routing    import run_all  as run_layer_2
from eval.test_command_builder   import run_tests as run_layer_4
from eval.test_rover_mock_server import run_tests as run_layer_5


LAYER_LABELS = {
    2: "Layer 2-3  DeviceHandlerFSM routing + rover context",
    4: "Layer 4    _parse_command() → JSON payload",
    5: "Layer 5    Full round-trip vs mock Pi server",
}

# Minimum score to be considered "ready" for that layer
PASS_THRESHOLD = {
    2: 75,   # Layer 2-3: some known gaps are OK
    4: 85,   # Layer 4: payload correctness is critical
    5: 90,   # Layer 5: round-trip must be solid
}


def run_all_layers(verbose=False):
    results = {}

    print("\n" + "═" * 65)
    print("  ROVER PRE-PI TEST SUITE")
    print("  Run this before connecting to the real Raspberry Pi rover.")
    print("═" * 65)

    # ── Layer 2-3 ──
    print(f"\n\n{'━'*65}")
    print(f"  {LAYER_LABELS[2]}")
    print(f"{'━'*65}")
    p, f, score = run_layer_2(verbose=verbose)
    results[2] = (p, f, score)

    # ── Layer 4 ──
    print(f"\n\n{'━'*65}")
    print(f"  {LAYER_LABELS[4]}")
    print(f"{'━'*65}")
    p, f, score = run_layer_4(verbose=verbose)
    results[4] = (p, f, score)

    # ── Layer 5 ──
    print(f"\n\n{'━'*65}")
    print(f"  {LAYER_LABELS[5]}")
    print(f"{'━'*65}")
    p, f, score = run_layer_5(verbose=verbose)
    results[5] = (p, f, score)

    # ── Final verdict ──
    print(f"\n\n{'═'*65}")
    print("  FINAL VERDICT")
    print(f"{'═'*65}\n")

    all_pass = True
    for layer, (p, f, score) in results.items():
        threshold = PASS_THRESHOLD[layer]
        passed    = score >= threshold
        all_pass  = all_pass and passed
        icon      = "✅" if passed else "❌"
        label     = LAYER_LABELS[layer]
        print(f"  {icon}  {label}")
        print(f"       Score: {p}/{p+f} ({score:.0f}%)  — threshold {threshold}%")
        print()

    print(f"{'─'*65}")
    if all_pass:
        print("  ✅  ALL LAYERS PASS — you are ready to test with the real Pi.")
    else:
        print("  ❌  NOT READY — fix the failing layers before connecting to the Pi.")
        print()
        print("  Fix priority order:")

        if results[4][2] < PASS_THRESHOLD[4]:
            print("    1. Fix _parse_command() gaps in rover_controller.py")
            print("       → 'reverse', 'find the door', 'go straight' etc.")

        if results[2][2] < PASS_THRESHOLD[2]:
            print("    2. Fix _infer_device() in device_handler.py")
            print("       → add 'robot' keyword, handle session-active path")
            print("    3. Fix _belongs_to_rover_context() in rover_controller.py")
            print("       → add 'follow', 'find' to followup_words")

        if results[5][2] < PASS_THRESHOLD[5]:
            print("    4. Fix RoverClient round-trip issues")
            print("       → check WebSocket connection and ACK parsing")

    print(f"{'═'*65}\n")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--layer", "-l",
        type=int,
        choices=[2, 4, 5],
        default=None,
        help="Run only one layer (2, 4, or 5)",
    )
    args = parser.parse_args()

    if args.layer == 2:
        run_layer_2(verbose=args.verbose)
    elif args.layer == 4:
        run_layer_4(verbose=args.verbose)
    elif args.layer == 5:
        run_layer_5(verbose=args.verbose)
    else:
        run_all_layers(verbose=args.verbose)