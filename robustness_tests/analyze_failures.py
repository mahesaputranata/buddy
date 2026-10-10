"""
analyze_failures.py
===================
Statistical failure mode analysis for flow test results.

Reads the latest flow result JSON and produces:
- Failure mode frequency table
- Root cause clustering
- Calibration vs real bug separation
- Which single fix would resolve the most failures

Usage:
    python robustness_tests/analyze_failures.py
    python robustness_tests/analyze_failures.py --file results/flow_20260403.json
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_latest_flow_result(path: Path | None = None) -> dict:
    if path:
        with open(path) as f:
            return json.load(f)

    flow_files = sorted(RESULTS_DIR.glob("flow_*.json"), reverse=True)
    if not flow_files:
        raise FileNotFoundError("No flow result files found in results/")

    with open(flow_files[0]) as f:
        return json.load(f)


def classify_failure_mode(turn: dict) -> str:
    """
    Given a failing turn, classify its failure mode into one of:
    - calibration_pending     : pending state mismatch — likely wrong test expectation
    - calibration_intent      : intent mismatch — likely wrong test expectation
    - stuck_no_terminal       : flow never reached terminal state
    - wrong_intent_to_chat    : correct intent routed to chat instead
    - wrong_intent_crossfire  : one intent confused for another specific intent
    - python_error            : exception/crash
    - unknown                 : unclassified
    """
    intent_pass  = turn.get("intent_pass", True)
    pending_pass = turn.get("pending_pass", True)
    got_intent   = turn.get("got_intent", "")
    exp_intent   = turn.get("expected_intent", "")
    got_pending  = turn.get("got_pending")
    exp_pending  = turn.get("expected_pending")
    error        = turn.get("error")

    if error:
        return "python_error"

    if not intent_pass and not pending_pass:
        if got_intent == "chat" and exp_intent != "chat":
            return "wrong_intent_to_chat + pending_mismatch"
        return "both_intent_and_pending_wrong"

    if not pending_pass:
        # Pending mismatch — could be calibration or real bug
        if got_pending is not None and exp_pending is None:
            return "stuck_pending_not_cleared"
        if got_pending is None and exp_pending is not None:
            return "pending_cleared_too_early"
        if got_pending != exp_pending:
            return "wrong_pending_state"
        return "pending_mismatch_other"

    if not intent_pass:
        if got_intent == "chat" and exp_intent != "chat":
            return "wrong_intent_to_chat"
        if got_intent and exp_intent and got_intent != exp_intent:
            return f"intent_crossfire:{exp_intent}→{got_intent}"
        return "intent_mismatch_other"

    return "unknown"


def analyze(data: dict) -> None:
    flows = data.get("flows", [])
    metrics = data.get("metrics", {})
    run_ts = data.get("run_ts", "unknown")

    print(f"\n{'='*70}")
    print(f"  FAILURE MODE ANALYSIS")
    print(f"  Run: {run_ts}")
    print(f"  Total flows: {len(flows)} | "
          f"Completed: {metrics.get('completed_flows', 0)} | "
          f"Stuck: {metrics.get('stuck_flows', 0)}")
    print(f"{'='*70}\n")

    # ── Collect all failing turns ─────────────────────────────────────────
    all_turns = []
    failing_turns = []
    stuck_flows = []
    calibration_candidates = []
    real_bug_candidates = []

    for flow in flows:
        flow_id   = flow["id"]
        flow_type = flow["flow_type"]
        completed = flow["completed"]
        stuck     = flow.get("stuck", False)

        if stuck:
            stuck_flows.append(flow_id)

        for turn in flow.get("turns", []):
            turn["_flow_id"]   = flow_id
            turn["_flow_type"] = flow_type
            all_turns.append(turn)

            if not (turn.get("intent_pass", True) and turn.get("pending_pass", True)):
                mode = classify_failure_mode(turn)
                turn["_failure_mode"] = mode
                failing_turns.append(turn)

    # ── Failure mode frequency ────────────────────────────────────────────
    mode_counter = Counter(t["_failure_mode"] for t in failing_turns)

    print(f"  FAILURE MODE FREQUENCY  ({len(failing_turns)} failing turns total)\n")
    print(f"  {'MODE':<45} {'COUNT':>5}  {'%':>6}  VERDICT")
    print(f"  {'-'*45}  {'-'*5}  {'-'*6}  -------")

    CALIBRATION_MODES = {
        "stuck_pending_not_cleared",
        "pending_cleared_too_early",
        "wrong_pending_state",
        "pending_mismatch_other",
    }

    BUG_MODES = {
        "wrong_intent_to_chat",
        "wrong_intent_to_chat + pending_mismatch",
        "both_intent_and_pending_wrong",
        "python_error",
    }

    total_failing = len(failing_turns)
    calibration_count = 0
    bug_count = 0

    for mode, count in mode_counter.most_common():
        pct = count / total_failing * 100 if total_failing else 0

        if mode in CALIBRATION_MODES or "pending" in mode.lower():
            verdict = "likely calibration"
            calibration_count += count
        elif mode in BUG_MODES or "error" in mode.lower():
            verdict = "likely real bug"
            bug_count += count
        elif mode.startswith("intent_crossfire"):
            verdict = "needs investigation"
        else:
            verdict = "unknown"

        print(f"  {mode:<45} {count:>5}  {pct:>5.1f}%  {verdict}")

    print()

    # ── Pending state analysis ────────────────────────────────────────────
    stuck_pending_counter: dict[str, int] = Counter()
    for t in failing_turns:
        if t.get("got_pending") and not t.get("pending_pass"):
            stuck_pending_counter[t["got_pending"]] += 1

    if stuck_pending_counter:
        print(f"  MOST COMMON UNEXPECTED PENDING STATES:\n")
        for state, count in stuck_pending_counter.most_common(8):
            pct = count / total_failing * 100 if total_failing else 0
            print(f"    {state:<45} {count:>3}x  ({pct:.0f}% of failures)")
        print()

    # ── Flow type failure clustering ──────────────────────────────────────
    flow_type_failures: dict[str, list[str]] = defaultdict(list)
    for t in failing_turns:
        flow_type_failures[t["_flow_type"]].append(t.get("_failure_mode", "unknown"))

    print(f"  FAILURE CLUSTERING BY FLOW TYPE:\n")
    for ftype, modes in sorted(flow_type_failures.items()):
        dominant = Counter(modes).most_common(1)[0][0]
        print(f"    {ftype:<40} → {dominant}")
    print()

    # ── Root cause hypothesis ─────────────────────────────────────────────
    print(f"  ROOT CAUSE HYPOTHESIS:\n")

    dominant_mode = mode_counter.most_common(1)[0][0] if mode_counter else "none"
    dominant_count = mode_counter.most_common(1)[0][1] if mode_counter else 0
    dominant_pct = dominant_count / total_failing * 100 if total_failing else 0

    print(f"  Dominant failure mode: '{dominant_mode}'")
    print(f"  Appears in {dominant_count}/{total_failing} failing turns ({dominant_pct:.0f}%)")
    print()

    if dominant_pct >= 60:
        print(f"SINGLE ROOT CAUSE LIKELY — fixing '{dominant_mode}' could")
        print(f"resolve ~{dominant_pct:.0f}% of failures in one change.")
    elif dominant_pct >= 30:
        print(f"MIXED CAUSES — dominant mode explains ~{dominant_pct:.0f}% of failures.")
        print(f"Fix dominant mode first, then re-analyze.")
    else:
        print(f"DISTRIBUTED FAILURES — no single dominant cause.")
        print(f"Multiple independent fixes needed.")

    print()

    # ── Calibration vs real bug split ─────────────────────────────────────
    print(f"  CALIBRATION vs REAL BUG SPLIT:\n")
    if total_failing > 0:
        cal_pct = calibration_count / total_failing * 100
        bug_pct = bug_count / total_failing * 100
        unk_pct = 100 - cal_pct - bug_pct
        print(f"Likely calibration: {calibration_count:>3} turns  ({cal_pct:.0f}%)")
        print(f"Likely real bugs:    {bug_count:>3} turns  ({bug_pct:.0f}%)")
        print(f"Needs investigation: {int(unk_pct * total_failing / 100):>3} turns  ({unk_pct:.0f}%)")

    print()

    # ── Actionable fix priority ───────────────────────────────────────────
    print(f"  ACTIONABLE FIX PRIORITY:\n")

    # Find which expected_pending values appear most as mismatches
    exp_pending_mismatches: dict[str, int] = Counter()
    for t in failing_turns:
        if not t.get("pending_pass") and t.get("expected_pending"):
            exp_pending_mismatches[t["expected_pending"]] += 1

    if exp_pending_mismatches:
        print(f"  Test cases expecting these pending states most often fail:")
        for state, count in exp_pending_mismatches.most_common(5):
            print(f"    expected={state!r:<40} failed {count}x")
        print()
        print(f"  → If actual pending states are reasonable, update test expectations.")
        print(f"  → If actual pending states are wrong, fix the system flow for that state.")

    # Find which intents are most misclassified
    intent_mismatches = [(t["expected_intent"], t["got_intent"])
                         for t in failing_turns if not t.get("intent_pass")]
    if intent_mismatches:
        intent_counter = Counter(intent_mismatches)
        print(f"\n  Most common intent misclassifications in multi-turn context:")
        for (exp, got), count in intent_counter.most_common(5):
            print(f"    {exp:<22} → {got:<22}  {count}x")

    # ── Calibration table: actual vs expected pending per failing turn ───────
    print(f"  CALIBRATION TABLE — ACTUAL vs EXPECTED PENDING STATES:\n")
    print(f"  {'FLOW':<12} {'TURN':>4}  {'INPUT':<35} {'EXPECTED_PENDING':<35} {'ACTUAL_PENDING':<35}  ACTION")
    print(f"  {'-'*12}  {'-'*4}  {'-'*35}  {'-'*35}  {'-'*35}  ------")

    for t in sorted(failing_turns, key=lambda x: (x["_flow_id"], x.get("turn", 0))):
        if not t.get("pending_pass"):
            flow_id     = t["_flow_id"]
            turn_num    = t.get("turn", "?") + 1 if isinstance(t.get("turn"), int) else "?"
            input_text  = (t.get("input", "") or "")[:33]
            exp_pending = (t.get("expected_pending") or "None")[:33]
            got_pending = (t.get("got_pending") or "None")[:33]
            mode        = t.get("_failure_mode", "")

            if "cleared_too_early" in mode:
                action = "→ update expected to None"
            elif "not_cleared" in mode:
                action = "→ update expected to actual"
            elif "wrong_pending" in mode:
                action = "→ update expected to actual"
            else:
                action = "→ investigate"

            print(f"  {flow_id:<12}  {str(turn_num):>4}  {input_text:<35}  {exp_pending:<35}  {got_pending:<35}  {action}")

    print()

    # ── Ready-to-paste test case updates ─────────────────────────────────
    print(f"  READY-TO-PASTE TEST CASE UPDATES:\n")
    print(f"  Copy these expected_pending_after values into multi_turn.jsonl:\n")

    updates: dict[str, dict[int, str]] = defaultdict(dict)
    for t in failing_turns:
        if not t.get("pending_pass"):
            flow_id  = t["_flow_id"]
            turn_num = t.get("turn", 0)
            actual   = t.get("got_pending")  # None means cleared, string means set
            updates[flow_id][turn_num] = actual

    for flow_id, turn_updates in sorted(updates.items()):
        print(f"  {flow_id}:")
        for turn_num, actual_pending in sorted(turn_updates.items()):
            display = f'"{actual_pending}"' if actual_pending else "null"
            print(f"    Turn {turn_num + 1}: \"expected_pending_after\": {display}")
        print()

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Flow failure mode analyzer")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific flow result JSON file to analyze")
    args = parser.parse_args()

    path = Path(args.file) if args.file else None
    data = load_latest_flow_result(path)
    analyze(data)


if __name__ == "__main__":
    main()