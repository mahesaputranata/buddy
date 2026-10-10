"""
unit_runner.py
==============
White-box single-turn intent classification tester.

Calls classify_intent() + apply_guards() directly — no LLM for the test itself,
only the system under test uses LLM internally.

Usage:
    python -m robustness_tests.runner.unit_runner
    python -m robustness_tests.runner.unit_runner --verbose
    python -m robustness_tests.runner.unit_runner --tags schedule
    python -m robustness_tests.runner.unit_runner --filter-id st_001,st_010
"""

from __future__ import annotations

import sys
import time
import json
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# ── Path bootstrap so we can import from the project root ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_CASES_DIR = Path(__file__).resolve().parents[1] / "test_cases"
RESULTS_DIR    = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Lazy import of the system under test ──────────────────────────────────
def _load_system():
    """
    Import classify_intent, apply_guards, and MemoryManager from the project.
    Wrapped in a function so import errors are caught cleanly.
    """
    try:
        from brain.conversation_manager import classify_intent, apply_guards
        from brain.core.brain import MemoryManager
        return classify_intent, apply_guards, MemoryManager
    except ImportError as e:
        print(f"[ERROR] Could not import system under test: {e}")
        print("        Make sure you run this from the project root.")
        sys.exit(1)


# ── Result dataclasses ─────────────────────────────────────────────────────
@dataclass
class TurnResult:
    test_id: str
    input_text: str
    expected_intent: str
    got_intent: str
    confidence: float
    source: str
    passed: bool
    latency_ms: float
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None


# ── Core runner ────────────────────────────────────────────────────────────
class UnitRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        classify_intent, apply_guards, MemoryManager = _load_system()
        self.classify_intent = classify_intent
        self.apply_guards     = apply_guards
        self.MemoryManager    = MemoryManager

    def _fresh_mm(self):
        """Each test gets a clean MemoryManager so state never bleeds."""
        return self.MemoryManager()

    def run_case(self, case: dict) -> TurnResult:
        test_id  = case["id"]
        text     = case["input"]
        expected = case["expected_intent"]
        tags     = case.get("tags", [])
        setup    = case.get("setup", {}) 

        mm = self._fresh_mm()

        if setup.get("web_topic"):
            web_topic = setup["web_topic"]
            web_query = setup.get("web_query", "")
            mock_results = [
                {"title": f"Mock result {i}", "url": f"https://example.com/{i}", 
                "snippet": f"Mock snippet {i}"}
                for i in range(1, 4)
            ]
            mm.set_web_context(
                web_topic, web_query, mock_results,
                summary="Mock summary", root_thread=web_query
            )
            mm.set_last_topic(web_topic)


        try:
            t0 = time.perf_counter()
            raw    = self.classify_intent(text, mm)
            result = self.apply_guards(raw, text, mm)
            latency_ms = (time.perf_counter() - t0) * 1000

            got    = result.get("intent", "unknown")
            conf   = float(result.get("confidence", 0.0))
            source = result.get("source", "unknown")
            passed = (got == expected)

        except Exception as e:
            latency_ms = 0.0
            got    = "ERROR"
            conf   = 0.0
            source = "exception"
            passed = False
            return TurnResult(
                test_id=test_id, input_text=text,
                expected_intent=expected, got_intent=got,
                confidence=conf, source=source,
                passed=passed, latency_ms=latency_ms,
                tags=tags, error=str(e)
            )

        return TurnResult(
            test_id=test_id, input_text=text,
            expected_intent=expected, got_intent=got,
            confidence=conf, source=source,
            passed=passed, latency_ms=latency_ms,
            tags=tags,
        )

    def run_all(
        self,
        tag_filter: Optional[str] = None,
        id_filter:  Optional[list[str]] = None,
    ) -> list[TurnResult]:
        cases_path = TEST_CASES_DIR / "single_turn.jsonl"
        cases = []
        with open(cases_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))

        # optional filters
        if tag_filter:
            cases = [c for c in cases if tag_filter in c.get("tags", [])]
        if id_filter:
            cases = [c for c in cases if c["id"] in id_filter]

        results: list[TurnResult] = []
        total = len(cases)

        print(f"\n{'='*60}")
        print(f"  UNIT RUNNER — {total} test cases")
        if tag_filter: print(f"  Tag filter: {tag_filter}")
        print(f"{'='*60}\n")

        for i, case in enumerate(cases, 1):
            r = self.run_case(case)
            results.append(r)

            status = "✅ PASS" if r.passed else "❌ FAIL"
            latency = f"{r.latency_ms:.0f}ms"

            if self.verbose or not r.passed:
                print(f"[{i:>3}/{total}] {status}  {r.test_id:<12} | "
                      f"expected={r.expected_intent:<16} got={r.got_intent:<16} "
                      f"conf={r.confidence:.2f}  src={r.source}  {latency}")
                if r.error:
                    print(f"          ERROR: {r.error}")
            else:
                dot = "." if r.passed else "F"
                print(dot, end="", flush=True)

        if not self.verbose:
            print()  # newline after dots

        return results

    def compute_metrics(self, results: list[TurnResult]) -> dict:
        """
        Computes per-intent precision, recall, F1.
        Also computes confusion matrix (expected → got counts).
        """
        intents = sorted(set(
            [r.expected_intent for r in results] +
            [r.got_intent for r in results if r.got_intent != "ERROR"]
        ))

        # confusion matrix: confusion[expected][got] = count
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in results:
            confusion[r.expected_intent][r.got_intent] += 1

        per_intent = {}
        for intent in intents:
            tp = confusion[intent].get(intent, 0)
            fp = sum(confusion[other].get(intent, 0)
                     for other in intents if other != intent)
            fn = sum(confusion[intent].get(other, 0)
                     for other in intents if other != intent)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)

            per_intent[intent] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
            }

        total   = len(results)
        passed  = sum(1 for r in results if r.passed)
        errors  = sum(1 for r in results if r.error)
        avg_lat = sum(r.latency_ms for r in results) / total if total else 0.0

        return {
            "total":      total,
            "passed":     passed,
            "failed":     total - passed - errors,
            "errors":     errors,
            "accuracy":   round(passed / total, 3) if total else 0.0,
            "avg_latency_ms": round(avg_lat, 1),
            "per_intent": per_intent,
            "confusion":  {k: dict(v) for k, v in confusion.items()},
        }

    def print_report(self, metrics: dict):
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total:     {metrics['total']}")
        print(f"  Passed:    {metrics['passed']}  ({metrics['accuracy']*100:.1f}%)")
        print(f"  Failed:    {metrics['failed']}")
        print(f"  Errors:    {metrics['errors']}")
        print(f"  Avg latency: {metrics['avg_latency_ms']}ms")
        print()

        print(f"  {'INTENT':<22} {'PREC':>6} {'REC':>6} {'F1':>6}  STATUS")
        print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  ------")

        for intent, m in sorted(metrics["per_intent"].items(),
                                 key=lambda kv: kv[1]["f1"]):
            f1 = m["f1"]
            status = "✅" if f1 >= 0.80 else ("⚠️ " if f1 >= 0.60 else "❌")
            print(f"  {intent:<22} {m['precision']:>6.2f}  {m['recall']:>6.2f}  "
                  f"{f1:>6.2f}  {status}")

        print()
        print(f"  TOP CONFUSIONS (expected → got, count ≥ 1):")
        conf = metrics["confusion"]
        pairs = []
        for expected, got_dict in conf.items():
            for got, count in got_dict.items():
                if got != expected and count > 0:
                    pairs.append((expected, got, count))
        pairs.sort(key=lambda x: -x[2])

        for expected, got, count in pairs[:10]:
            print(f"    {expected:<22} → {got:<22}  {count:>3}x")

        print(f"\n{'='*60}\n")

    def save_results(self, results: list[TurnResult], metrics: dict):
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"unit_{ts}.json"

        payload = {
            "run_ts": ts,
            "metrics": metrics,
            "results": [
                {
                    "id":       r.test_id,
                    "input":    r.input_text,
                    "expected": r.expected_intent,
                    "got":      r.got_intent,
                    "passed":   r.passed,
                    "conf":     round(r.confidence, 3),
                    "source":   r.source,
                    "latency_ms": round(r.latency_ms, 1),
                    "tags":     r.tags,
                    "error":    r.error,
                }
                for r in results
            ],
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Results saved → {out_path}")
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Buddy robustness unit runner")
    parser.add_argument("--verbose",   action="store_true", help="Print every result")
    parser.add_argument("--tags",      type=str, default=None, help="Filter by tag")
    parser.add_argument("--filter-id", type=str, default=None,
                        help="Comma-separated test IDs to run")
    args = parser.parse_args()

    id_filter = [x.strip() for x in args.filter_id.split(",")] \
        if args.filter_id else None

    runner  = UnitRunner(verbose=args.verbose)
    results = runner.run_all(tag_filter=args.tags, id_filter=id_filter)
    metrics = runner.compute_metrics(results)
    runner.print_report(metrics)
    runner.save_results(results, metrics)


if __name__ == "__main__":
    main()
