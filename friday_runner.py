"""
friday_stressed_test_runner.py
──────────────────────────────
Minimal dummy test for the Friday stressed test pipeline.

Tests:
  1. detect_stressed_test_trigger() — does the right phrase fire the right (project, scope)?
  2. ConversationManager.handle_request() routing — does Buddy route to handle_stressed_test()?
  3. Friday dispatch — does friday.run_stressed_test() send or fail gracefully?

No TTS, no VAD, no audio, no MATLAB, no webcam.
Talks directly to handle_request() like buddy_alive_test_runner.py does.

Usage:
    python friday_stressed_test_runner.py
    python friday_stressed_test_runner.py --skip-dispatch   # skip the actual HTTP call to OpenClaw
    python friday_stressed_test_runner.py --verbose         # show full responses
"""

from __future__ import annotations

import sys
import time
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

# Format: (phrase, expected_project, expected_scope, should_trigger)
TRIGGER_DETECTION_CASES: list[tuple[str, Optional[str], Optional[str], bool]] = [
    # ── Should fire ───────────────────────────────────────────────────────────
    ("run stressed test on jastip",          "jastip", "full",   True),
    ("run stress test on jastip",            "jastip", "full",   True),
    ("do a stressed test on jastip",         "jastip", "full",   True),
    ("ask friday to do a stressed test",     "jastip", "full",   True),
    ("ask friday to run the tests",          "jastip", "full",   True),
    ("friday run stressed test",             "jastip", "full",   True),
    ("friday check jastip",                  "jastip", "full",   True),
    ("friday do the tests",                  "jastip", "full",   True),
    ("stressed test jastip",                 "jastip", "full",   True),
    ("stress test jastip",                   "jastip", "full",   True),
    ("run tests on jastip",                  "jastip", "full",   True),
    ("run the tests on jastip",              "jastip", "full",   True),
    ("check jastip",                         "jastip", "full",   True),
    ("test jastip",                          "jastip", "full",   True),
    # E2E scope
    ("run e2e on jastip",                    "jastip", "e2e",    True),
    ("run e2e tests",                        "jastip", "e2e",    True),
    ("run playwright",                       "jastip", "e2e",    True),
    ("check dead buttons",                   "jastip", "e2e",    True),
    # Stress scope
    ("run load test",                        "jastip", "stress", True),
    ("run k6",                               "jastip", "stress", True),
    ("load test jastip",                     "jastip", "stress", True),
    # Audit scope
    ("run audit",                            "jastip", "audit",  True),
    ("generate report",                      "jastip", "audit",  True),
    ("summarize test results",               "jastip", "audit",  True),
    ("friday summarize the results",         "jastip", "audit",  True),

    # ── Should NOT fire (false positive guard) ────────────────────────────────
    ("can you do a stressed test",           None, None, False),   # no project name, no friday
    ("how does a stress test work",          None, None, False),   # informational question
    ("what is jastip",                       None, None, False),   # no test intent
    ("tell me about jastip",                 None, None, False),   # chat
    ("remind me to run tests tomorrow",      None, None, False),   # schedule, not trigger
    ("what time is it",                      None, None, False),   # system query
]

# Phrases to feed through handle_request() to test end-to-end routing
HANDLE_REQUEST_CASES: list[tuple[str, list[str]]] = [
    # (phrase, list of strings that should appear in Buddy's response)
    (
        "friday check jastip",
        ["friday", "jastip"],           # Buddy should mention both
    ),
    (
        "ask friday to do a stressed test",
        ["friday", "test"],
    ),
    (
        "run e2e tests on jastip",
        ["playwright", "jastip"],
    ),
    (
        "run audit",
        ["audit", "friday"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TriggerResult:
    phrase: str
    expected_project: Optional[str]
    expected_scope: Optional[str]
    should_trigger: bool
    actual_result: Optional[tuple]
    passed: bool
    note: str = ""


@dataclass
class RoutingResult:
    phrase: str
    response: str
    response_time_ms: float
    expected_keywords: list[str]
    missing_keywords: list[str]
    passed: bool


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class FridayStressedTestRunner:

    def __init__(self, skip_dispatch: bool = False, verbose: bool = False):
        self.skip_dispatch = skip_dispatch
        self.verbose = verbose
        self.cm = None

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_conversation_manager(self):
        """
        Boots a real ConversationManager with Friday mocked out
        so we test routing without needing OpenClaw running.
        """
        print("[SETUP] Loading Buddy internals...")
        try:
            from brain.core.brain import MemoryManager
            from brain.identity.identity_manager import IdentityManager
            from brain.core.internet_access.internet_service import get_explorer
            from brain.conversation_manager import ConversationManager
        except Exception as e:
            print(f"[SETUP ERROR] Could not import Buddy internals: {e}")
            traceback.print_exc()
            sys.exit(1)

        try:
            identity = IdentityManager()
            explorer = get_explorer(enabled=False)
            mm = MemoryManager()
            self.cm = ConversationManager(
                mm,
                tello_client=None,
                explorer=explorer,
                identity=identity,
            )
            print("[SETUP] ConversationManager ready.")
        except Exception as e:
            print(f"[SETUP ERROR] ConversationManager failed: {e}")
            traceback.print_exc()
            sys.exit(1)

        # Mock Friday so dispatch doesn't need OpenClaw running
        if self.skip_dispatch:
            self._mock_friday()

    def _mock_friday(self):
        """Replace Friday's HTTP send with a fake that always succeeds."""
        if not self.cm:
            return

        friday = getattr(self.cm, "friday", None)
        if not friday:
            print("[SETUP] No friday instance found on ConversationManager — skipping mock.")
            return

        def fake_run_stressed_test(project_key, project_display, test_dir,
                                    frontend_url, backend_url, scope="full"):
            # Set the session state as the real method would
            from brain.friday.friday import FridayTaskMeta
            friday._active_task_meta = FridayTaskMeta(
                session_key=f"buddy:stressed_test:{project_key}",
                project_key=project_key,
                task_type="stressed_test",
                project_display=project_display,
                test_dir=test_dir,
            )
            friday.active_session_key = f"buddy:stressed_test:{project_key}"
            return "On it — I've passed that to Friday. I'll let you know when it's ready."

        friday.run_stressed_test = fake_run_stressed_test
        print("[SETUP] Friday.run_stressed_test mocked (no real HTTP call).")

    def _reset(self):
        """Reset memory state between tests."""
        if not self.cm:
            return
        try:
            self.cm.memory_manager.set_pending_topic(None)
            self.cm.memory_manager.pending_payload = None
            self.cm.memory_manager.set_last_topic(None)
            self.cm.memory_manager._stuck_count = 0
            self.cm.memory_manager._last_pending = None
        except Exception:
            pass

    # ── Layer 1: Trigger detection ───────────────────────────────────────────

    def run_trigger_detection(self) -> list[TriggerResult]:
        print(f"\n{'='*60}")
        print("LAYER 1 — detect_stressed_test_trigger() detection")
        print(f"{'='*60}")

        try:
            from brain.projects.project_handler import detect_stressed_test_trigger
        except ImportError:
            print("[ERROR] detect_stressed_test_trigger not found in project_handler.")
            print("        Make sure you've pasted it from project_handler_jastip_extension.py")
            return []

        results = []
        passed_count = 0

        for phrase, exp_proj, exp_scope, should_trigger in TRIGGER_DETECTION_CASES:
            actual = detect_stressed_test_trigger(phrase)

            if should_trigger:
                if actual is None:
                    passed = False
                    note = f"Expected ({exp_proj}, {exp_scope}) but got None"
                elif actual[0] != exp_proj or actual[1] != exp_scope:
                    passed = False
                    note = f"Expected ({exp_proj}, {exp_scope}) but got {actual}"
                else:
                    passed = True
                    note = ""
            else:
                # Should NOT trigger
                if actual is not None:
                    passed = False
                    note = f"False positive — should not have triggered, got {actual}"
                else:
                    passed = True
                    note = "Correctly did not trigger"

            status = "✓" if passed else "✗"
            if passed:
                passed_count += 1

            print(f"  {status} {phrase!r:50} → {str(actual):<25} {note}")

            results.append(TriggerResult(
                phrase=phrase,
                expected_project=exp_proj,
                expected_scope=exp_scope,
                should_trigger=should_trigger,
                actual_result=actual,
                passed=passed,
                note=note,
            ))

        total = len(results)
        print(f"\n  Result: {passed_count}/{total} passed")
        return results

    # ── Layer 2: handle_request() routing ────────────────────────────────────

    def run_routing_tests(self) -> list[RoutingResult]:
        print(f"\n{'='*60}")
        print("LAYER 2 — handle_request() routing")
        print(f"{'='*60}")

        if not self.cm:
            print("[SKIP] ConversationManager not initialised.")
            return []

        results = []
        passed_count = 0

        for phrase, expected_keywords in HANDLE_REQUEST_CASES:
            self._reset()

            print(f"\n  Input: {phrase!r}")

            t0 = time.perf_counter()
            try:
                response = self.cm.handle_request(phrase)
                response = (response or "").strip()
            except Exception as e:
                response = f"[ERROR: {e}]"
                traceback.print_exc()
            elapsed = (time.perf_counter() - t0) * 1000

            resp_lower = response.lower()
            missing = [kw for kw in expected_keywords if kw.lower() not in resp_lower]
            passed = len(missing) == 0

            if passed:
                passed_count += 1

            status = "✓" if passed else "✗"
            last_topic = self.cm.memory_manager.get_last_topic() or "—"
            friday_warm = getattr(getattr(self.cm, "friday", None), "is_warm", lambda: False)()

            print(f"  {status} Response ({elapsed:.0f}ms): {response[:120]!r}{'...' if len(response) > 120 else ''}")
            print(f"     last_topic={last_topic!r}  friday_warm={friday_warm}")

            if self.verbose:
                print(f"     Full response: {response}")

            if missing:
                print(f"     ✗ Missing keywords: {missing}")
            else:
                print(f"     ✓ All expected keywords found: {expected_keywords}")

            results.append(RoutingResult(
                phrase=phrase,
                response=response,
                response_time_ms=round(elapsed, 1),
                expected_keywords=expected_keywords,
                missing_keywords=missing,
                passed=passed,
            ))

        total = len(results)
        print(f"\n  Result: {passed_count}/{total} passed")
        return results

    # ── Layer 3: Friday dispatch (real or mocked) ─────────────────────────────

    def run_dispatch_test(self) -> dict:
        print(f"\n{'='*60}")
        print("LAYER 3 — Friday dispatch")
        print(f"{'='*60}")

        friday = getattr(self.cm, "friday", None) if self.cm else None

        if not friday:
            print("  [SKIP] No friday instance on ConversationManager.")
            return {"passed": False, "note": "No friday instance"}

        mode = "MOCKED" if self.skip_dispatch else "REAL HTTP"
        print(f"  Mode: {mode}")
        print(f"  Friday warm before test: {friday.is_warm()}")

        try:
            result = friday.run_stressed_test(
                project_key="jastip",
                project_display="JASTIP Marketplace",
                test_dir=__import__("brain.settings", fromlist=["setting"]).setting("jastip_test_dir", "external/jastip/stressed_test"),
                frontend_url="http://localhost:5173",
                backend_url="http://localhost:3000",
                scope="audit",
            )
            print(f"  Dispatch reply: {result!r}")
            print(f"  Friday warm after dispatch: {friday.is_warm()}")
            print(f"  Active session key: {friday.active_session_key!r}")

            dispatch_ok = "friday" in result.lower() or "on it" in result.lower() or "couldn't reach" in result.lower()

            # Test follow-up routing
            if friday.is_warm():
                followup_reply = friday.ask("what were the dead buttons?")
                print(f"  Follow-up reply: {followup_reply!r}")
                followup_ok = len(followup_reply) > 0
            else:
                followup_ok = False
                print("  [SKIP] Follow-up test skipped — Friday not warm after dispatch")

            passed = dispatch_ok
            print(f"\n  {'✓' if passed else '✗'} Dispatch {'OK' if dispatch_ok else 'FAILED'}")
            print(f"  {'✓' if followup_ok else '✗'} Follow-up routing {'OK' if followup_ok else 'FAILED'}")

            return {
                "passed": passed,
                "dispatch_reply": result,
                "session_key": friday.active_session_key,
                "followup_ok": followup_ok,
                "mode": mode,
            }

        except Exception as e:
            print(f"  [ERROR] Dispatch failed: {e}")
            traceback.print_exc()
            return {"passed": False, "error": str(e)}

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "="*60)
        print("FRIDAY STRESSED TEST PIPELINE — DUMMY RUNNER")
        print("="*60)
        print(f"Mode: {'skip-dispatch (mocked)' if self.skip_dispatch else 'real dispatch'}")
        print()

        # Layer 1 — pure function, no CM needed
        trigger_results = self.run_trigger_detection()

        # Layer 2 + 3 — need CM
        self._setup_conversation_manager()
        routing_results = self.run_routing_tests()
        dispatch_result = self.run_dispatch_test()

        # ── Final summary ─────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        t_pass = sum(1 for r in trigger_results if r.passed)
        t_total = len(trigger_results)
        t_false_pos = sum(1 for r in trigger_results if not r.passed and not r.should_trigger)
        t_missed = sum(1 for r in trigger_results if not r.passed and r.should_trigger)

        r_pass = sum(1 for r in routing_results if r.passed)
        r_total = len(routing_results)

        print(f"  Layer 1 — Trigger detection: {t_pass}/{t_total}")
        if t_false_pos:
            print(f"             False positives: {t_false_pos}  ← phrases that fired but shouldn't")
        if t_missed:
            print(f"             Missed triggers: {t_missed}  ← phrases that should fire but didn't")

        print(f"  Layer 2 — Routing:           {r_pass}/{r_total}")
        print(f"  Layer 3 — Dispatch:          {'✓ OK' if dispatch_result.get('passed') else '✗ FAILED'}")

        all_passed = (t_pass == t_total and r_pass == r_total and dispatch_result.get("passed"))
        print()
        if all_passed:
            print("  ✓ All layers green. Ready to test with real OpenClaw.")
        else:
            print("  ✗ Some layers failed. Fix before activating OpenClaw.")
            print()
            print("  What to fix:")
            for r in trigger_results:
                if not r.passed:
                    arrow = "should trigger but didn't" if r.should_trigger else "false positive"
                    print(f"    - {r.phrase!r} → {arrow}")
            for r in routing_results:
                if not r.passed:
                    print(f"    - handle_request({r.phrase!r}) missing keywords: {r.missing_keywords}")
            if not dispatch_result.get("passed"):
                print(f"    - Friday dispatch failed: {dispatch_result.get('error') or dispatch_result.get('dispatch_reply')}")

        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Friday Stressed Test Pipeline Runner")
    parser.add_argument(
        "--skip-dispatch",
        action="store_true",
        help="Mock Friday HTTP call — use this when OpenClaw is not running",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full responses (not just first 120 chars)",
    )
    args = parser.parse_args()

    runner = FridayStressedTestRunner(
        skip_dispatch=args.skip_dispatch,
        verbose=args.verbose,
    )
    runner.run()


if __name__ == "__main__":
    main()