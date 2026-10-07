"""
run_all.py
==========
Master robustness test orchestrator.

Runs all three phases in sequence and produces a unified weak-spot report.

Usage:
    python robustness_tests/run_all.py
    python robustness_tests/run_all.py --phase unit
    python robustness_tests/run_all.py --phase flow
    python robustness_tests/run_all.py --phase unit flow
    python robustness_tests/run_all.py --verbose
    python robustness_tests/run_all.py --skip-adversarial
"""

from __future__ import annotations

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _section(title: str, width: int = 62):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def run_unit_phase(verbose: bool) -> dict:
    _section("PHASE 1 — UNIT TESTS  (single-turn, white-box)")
    from robustness_tests.runner.unit_runner import UnitRunner
    runner  = UnitRunner(verbose=verbose)
    results = runner.run_all()
    metrics = runner.compute_metrics(results)
    runner.print_report(metrics)
    path    = runner.save_results(results, metrics)
    return {"metrics": metrics, "path": str(path), "results": results}


def run_flow_phase(verbose: bool) -> dict:
    _section("PHASE 2 — FLOW TESTS  (multi-turn, black-box)")
    from robustness_tests.runner.flow_runner import FlowRunner
    runner  = FlowRunner(verbose=verbose)
    results = runner.run_all()
    metrics = runner.compute_metrics(results)
    runner.print_report(metrics)
    path    = runner.save_results(results, metrics)
    return {"metrics": metrics, "path": str(path), "results": results}


def run_adversarial_phase(intent_filter=None) -> dict:
    _section("PHASE 3 — ADVERSARIAL  (LLM-generated variations)")
    print("  Step 3a: Generating adversarial cases...")
    from robustness_tests.runner.adversarial_generator import AdversarialGenerator
    gen  = AdversarialGenerator()
    gen.run(n_per_strategy=2, intent_filter=intent_filter, max_seeds=20)

    print("\n  Step 3b: Running adversarial cases through unit runner...")
    # After generation, run the generated cases
    adv_path = Path(__file__).parent / "test_cases" / "adversarial.jsonl"
    if not adv_path.exists() or adv_path.stat().st_size == 0:
        print("  [SKIP] No adversarial cases generated.")
        return {}

    from robustness_tests.runner.unit_runner import UnitRunner
    import importlib

    # Monkey-patch unit runner to load adversarial cases
    runner = UnitRunner(verbose=False)

    cases = []
    with open(adv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    adv_results = [runner.run_case(c) for c in cases]
    metrics     = runner.compute_metrics(adv_results)
    runner.print_report(metrics)
    path = runner.save_results(adv_results, metrics)

    return {"metrics": metrics, "path": str(path)}


def _color_f1(f1: float) -> str:
    if f1 >= 0.80: return "✅"
    if f1 >= 0.60: return "⚠️ "
    return "❌"


def print_final_report(unit: dict, flow: dict, adv: dict):
    """
    Unified weak-spot report that combines all three phases.
    This is the actionable output — tells you exactly what to fix.
    """
    _section("FINAL WEAK-SPOT REPORT", width=70)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"  Generated: {now}\n")

    # ── Phase 1 summary ──────────────────────────────────────────────────
    if unit:
        m = unit["metrics"]
        print(f"  PHASE 1 — Unit accuracy: {m['accuracy']*100:.1f}%  "
              f"({m['passed']}/{m['total']} passed)\n")

        weak = [(k, v) for k, v in m["per_intent"].items() if v["f1"] < 0.80]
        weak.sort(key=lambda x: x[1]["f1"])

        if weak:
            print(f"  ⚠️  WEAK INTENTS (F1 < 0.80):")
            for intent, v in weak:
                print(f"     {_color_f1(v['f1'])}  {intent:<22}  "
                      f"F1={v['f1']:.2f}  P={v['precision']:.2f}  R={v['recall']:.2f}")
        else:
            print(f"  ✅  All intents above F1=0.80 threshold.")

        # top confusions
        conf = m.get("confusion", {})
        pairs = []
        for exp, got_dict in conf.items():
            for got, count in got_dict.items():
                if got != exp and count > 0:
                    pairs.append((exp, got, count))
        pairs.sort(key=lambda x: -x[2])

        if pairs:
            print(f"\n  TOP CONFUSIONS:")
            for exp, got, count in pairs[:6]:
                print(f"     {exp:<22} → {got:<22}  {count}x")

    # ── Phase 2 summary ──────────────────────────────────────────────────
    if flow:
        m = flow["metrics"]
        print(f"\n  PHASE 2 — Flow completion: "
              f"{m['completion_rate']*100:.1f}%  "
              f"({m['completed_flows']}/{m['total_flows']} flows)\n")

        bad = [(k, v) for k, v in m["by_flow_type"].items()
               if v["completed"] / v["total"] < 0.80]
        bad.sort(key=lambda x: x[1]["completed"] / x[1]["total"])

        if bad:
            print(f"  ⚠️  WEAK FLOW TYPES:")
            for ftype, v in bad:
                rate = v["completed"] / v["total"]
                print(f"     ❌  {ftype:<40}  {v['completed']}/{v['total']} "
                      f"({rate*100:.0f}%)")

        if m["stuck_flows"] > 0:
            print(f"\n  ⚠️  STUCK FLOWS: {m['stuck_flows']} flows ended in a "
                  f"non-null pending state when none was expected.")
            print(f"     Check: pending state not cleared after flow terminal condition.")

    # ── Phase 3 summary ──────────────────────────────────────────────────
    if adv:
        m = adv.get("metrics", {})
        if m:
            print(f"\n  PHASE 3 — Adversarial accuracy: "
                  f"{m.get('accuracy', 0)*100:.1f}%")

            adv_weak = [(k, v) for k, v in m.get("per_intent", {}).items()
                        if v["f1"] < 0.70]
            adv_weak.sort(key=lambda x: x[1]["f1"])

            if adv_weak:
                print(f"  ⚠️  INTENTS THAT BREAK UNDER ADVERSARIAL INPUT:")
                for intent, v in adv_weak:
                    print(f"     ❌  {intent:<22}  F1={v['f1']:.2f}")
            else:
                print(f"  ✅  All intents robust to adversarial variations.")

    # ── Action list ───────────────────────────────────────────────────────
    print(f"\n  RECOMMENDED ACTIONS:")
    actions = []

    if unit:
        m = unit["metrics"]
        for intent, v in m["per_intent"].items():
            if v["f1"] < 0.60:
                actions.append(
                    f"  🔴  [{intent}] F1={v['f1']:.2f} — CRITICAL. "
                    f"Add more strong rules or review EVIDENCE list."
                )
            elif v["f1"] < 0.80:
                actions.append(
                    f"  🟡  [{intent}] F1={v['f1']:.2f} — "
                    f"Review ambiguous cases. Check post_validate_intent."
                )

    if flow and flow["metrics"]["stuck_flows"] > 0:
        actions.append(
            f"  🔴  Stuck pending states detected. "
            f"Check handle_pending() terminal conditions."
        )

    if not actions:
        actions.append("  ✅  No critical issues found. System looks healthy.")

    for a in actions:
        print(a)

    print(f"\n{'='*70}\n")


def save_combined_report(unit: dict, flow: dict, adv: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"combined_report_{ts}.json"

    payload = {
        "run_ts": ts,
        "unit":  unit.get("metrics") if unit else None,
        "flow":  flow.get("metrics") if flow else None,
        "adv":   adv.get("metrics")  if adv  else None,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  Combined report → {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Buddy robustness master runner")
    parser.add_argument("--phase",   nargs="+",
                        choices=["unit", "flow", "adversarial"],
                        default=["unit", "flow", "adversarial"],
                        help="Which phases to run (default: all)")
    parser.add_argument("--verbose",          action="store_true")
    parser.add_argument("--skip-adversarial", action="store_true",
                        help="Skip adversarial generation (faster)")
    parser.add_argument("--adversarial-intent", type=str, default=None,
                        help="Only generate adversarial for this intent")
    args = parser.parse_args()

    phases = set(args.phase)
    if args.skip_adversarial:
        phases.discard("adversarial")

    t_start = time.perf_counter()

    unit = {}
    flow = {}
    adv  = {}

    if "unit" in phases:
        unit = run_unit_phase(verbose=args.verbose)

    if "flow" in phases:
        flow = run_flow_phase(verbose=args.verbose)

    if "adversarial" in phases:
        adv = run_adversarial_phase(intent_filter=args.adversarial_intent)

    elapsed = time.perf_counter() - t_start

    print_final_report(unit, flow, adv)
    save_combined_report(unit, flow, adv)

    print(f"  Total runtime: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
