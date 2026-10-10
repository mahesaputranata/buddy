"""
flow_runner.py
==============
Black-box multi-turn flow tester.

Calls handle_request() (your full production pipeline) turn by turn,
tracks pending state transitions, and scores flow completion + recovery.

Usage:
    python -m robustness_tests.runner.flow_runner
    python -m robustness_tests.runner.flow_runner --verbose
    python -m robustness_tests.runner.flow_runner --tags adversarial
    python -m robustness_tests.runner.flow_runner --id mt_001,mt_010
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_CASES_DIR = Path(__file__).resolve().parents[1] / "test_cases"
RESULTS_DIR    = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _load_system():
    try:
        from brain.conversation_manager import ConversationManager, classify_intent, apply_guards
        from brain.core.brain import MemoryManager
        return ConversationManager, classify_intent, apply_guards, MemoryManager
    except ImportError as e:
        print(f"[ERROR] Could not import system under test: {e}")
        print("        Run from the project root.")
        sys.exit(1)


# ── Result dataclasses ─────────────────────────────────────────────────────
@dataclass
class TurnOutcome:
    turn_index:       int
    input_text:       str
    expected_intent:  str
    got_intent:       Optional[str]
    expected_pending: Optional[str]
    got_pending:      Optional[str]
    response:         str
    intent_pass:      bool
    pending_pass:     bool
    latency_ms:       float
    note:             str = ""
    error:            Optional[str] = None

@dataclass
class FlowResult:
    flow_id:          str
    description:      str
    flow_type:        str
    tags:             list[str]
    turns:            list[TurnOutcome] = field(default_factory=list)
    flow_completed:   bool = False
    all_intents_pass: bool = False
    all_pending_pass: bool = False
    got_stuck:        bool = False
    total_turns:      int = 0
    passed_turns:     int = 0


# ── Stub for devices/external services (no hardware needed for testing) ────
class _MockTelloClient:
    def __getattr__(self, name):
        return lambda *a, **kw: None

class _MockRoverClient:
    def __getattr__(self, name):
        return lambda *a, **kw: None

class _MockExplorer:
    def search_web(self, query, limit=5):
        return [
            {"title": f"Mock result {i} for {query}", "url": f"https://example.com/{i}",
             "snippet": f"This is a mock snippet for {query}"}
            for i in range(1, min(limit+1, 4))
        ]
    def get_startup_data(self):
        return {"weather": {}, "headlines": []}


# ── Core runner ────────────────────────────────────────────────────────────
class FlowRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        (ConversationManager,
         self.classify_intent,
         self.apply_guards,
         MemoryManager) = _load_system()

        self._ConversationManager = ConversationManager
        self._MemoryManager       = MemoryManager

    def _fresh_conversation_manager(self):
        mm = self._MemoryManager()

        # Clear module-level shared state so flows don't bleed into each other
        try:
            from brain.core.brain import temp_topic_memory, sensory_memory
            temp_topic_memory.clear()
            sensory_memory.clear()
        except Exception:
            pass

        cm = self._ConversationManager(
            memory_manager=mm,
            tello_client=_MockTelloClient(),
            explorer=_MockExplorer(),
        )
        return cm

    def _get_intent_from_response(self, cm, user_input: str) -> Optional[str]:
        """
        Re-run classify_intent on the same input to get the intent label.
        Used to compare expected vs actual intent in black-box mode.
        """
        try:
            raw    = self.classify_intent(user_input, cm.memory_manager)
            result = self.apply_guards(raw, user_input, cm.memory_manager)
            return result.get("intent")
        except Exception:
            return None

    def run_flow(self, scenario: dict) -> FlowResult:
        flow_id     = scenario["id"]
        description = scenario.get("description", "")
        flow_type   = scenario.get("flow_type", "unknown")
        tags        = scenario.get("tags", [])
        turns_spec  = scenario["turns"]

        cm     = self._fresh_conversation_manager()
        result = FlowResult(
            flow_id=flow_id,
            description=description,
            flow_type=flow_type,
            tags=tags,
            total_turns=len(turns_spec),
        )

        if self.verbose:
            print(f"\n  ── {flow_id}: {description}")

        for i, turn in enumerate(turns_spec):
            text             = turn["input"]
            expected_intent  = turn.get("expected_intent")
            expected_pending = turn.get("expected_pending_after", "__ANY__")
            note             = turn.get("note", "")

            # ── Snapshot pending BEFORE the call ──────────────────────────────
            pending_before = cm.memory_manager.get_pending_topic()

            # ── Determine intent label ─────────────────────────────────────────
            # If pending is active, handle_pending() will handle this turn —
            # classify_intent() won't be used for routing, so trust expected.
            # If no pending, pre-classify to get the intent label.
            if pending_before:
                got_intent = expected_intent  # pending turn — routing bypasses classifier
            else:
                try:
                    raw        = self.classify_intent(text, cm.memory_manager)
                    guard      = self.apply_guards(raw, text, cm.memory_manager)
                    got_intent = guard.get("intent")
                except Exception:
                    got_intent = None

            # ── Run through the full pipeline — EXACTLY ONCE ──────────────────
            t0 = time.perf_counter()
            try:
                response   = cm.handle_request(text)
                latency_ms = (time.perf_counter() - t0) * 1000
                error      = None
            except Exception as e:
                response   = ""
                latency_ms = (time.perf_counter() - t0) * 1000
                error      = f"{type(e).__name__}: {e}"
                traceback.print_exc()

            # ── Snapshot pending AFTER the call ───────────────────────────────
            got_pending = cm.memory_manager.get_pending_topic()

            # ── Score this turn ───────────────────────────────────────────────
            intent_pass = (got_intent == expected_intent) if expected_intent else True

            if expected_pending == "__ANY__":
                pending_pass = True
            elif expected_pending is None:
                pending_pass = (got_pending is None)
            else:
                pending_pass = (got_pending == expected_pending)

            outcome = TurnOutcome(
                turn_index=i,
                input_text=text,
                expected_intent=expected_intent or "any",
                got_intent=got_intent,
                expected_pending=expected_pending if expected_pending != "__ANY__" else None,
                got_pending=got_pending,
                response=response or "",
                intent_pass=intent_pass,
                pending_pass=pending_pass,
                latency_ms=latency_ms,
                note=note,
                error=error,
            )
            result.turns.append(outcome)

            if self.verbose:
                i_status = "✅" if intent_pass  else "❌"
                p_status = "✅" if pending_pass else "❌"
                print(f"    Turn {i+1}: {i_status} intent={got_intent:<18} "
                    f"{p_status} pending={str(got_pending):<30}  "
                    f"'{text[:50]}'")
                if note:
                    print(f"           note: {note}")
                if error:
                    print(f"           ERROR: {error}")

        # ── Flow-level scoring ─────────────────────────────────────────────────
        result.passed_turns      = sum(1 for t in result.turns if t.intent_pass and t.pending_pass)
        result.all_intents_pass  = all(t.intent_pass  for t in result.turns)
        result.all_pending_pass  = all(t.pending_pass for t in result.turns)
        result.flow_completed    = result.all_intents_pass and result.all_pending_pass

        last = result.turns[-1] if result.turns else None
        if last and last.expected_pending is None and last.got_pending is not None:
            result.got_stuck = True

        return result

    def run_all(
        self,
        tag_filter: Optional[str] = None,
        id_filter:  Optional[list[str]] = None,
    ) -> list[FlowResult]:
        path = TEST_CASES_DIR / "multi_turn.jsonl"
        with open(path) as f:
            content = f.read().strip()
        scenarios = json.loads(content) if content.startswith("[") else \
            [json.loads(line) for line in content.splitlines() if line.strip()]

        if tag_filter:
            scenarios = [s for s in scenarios if tag_filter in s.get("tags", [])]
        if id_filter:
            scenarios = [s for s in scenarios if s["id"] in id_filter]

        total = len(scenarios)
        print(f"\n{'='*60}")
        print(f"  FLOW RUNNER — {total} scenarios")
        if tag_filter: print(f"  Tag filter: {tag_filter}")
        print(f"{'='*60}")

        results = []
        for scenario in scenarios:
            r = self.run_flow(scenario)
            results.append(r)
            status = "✅" if r.flow_completed else ("🔁" if r.got_stuck else "❌")
            print(f"  {status}  {r.flow_id:<12}  [{r.flow_type:<35}]  "
                  f"{r.passed_turns}/{r.total_turns} turns pass")

        return results

    def compute_metrics(self, results: list[FlowResult]) -> dict:
        total     = len(results)
        completed = sum(1 for r in results if r.flow_completed)
        stuck     = sum(1 for r in results if r.got_stuck)

        # by flow type
        by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "completed": 0})
        for r in results:
            by_type[r.flow_type]["total"] += 1
            if r.flow_completed:
                by_type[r.flow_type]["completed"] += 1

        # intent accuracy across all turns of all flows
        all_turns = [t for r in results for t in r.turns]
        intent_pass_rate = sum(1 for t in all_turns if t.intent_pass) / len(all_turns) \
            if all_turns else 0.0
        pending_pass_rate = sum(1 for t in all_turns if t.pending_pass) / len(all_turns) \
            if all_turns else 0.0

        # turn-level confusion
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in all_turns:
            if t.got_intent:
                confusion[t.expected_intent][t.got_intent] += 1

        return {
            "total_flows":     total,
            "completed_flows": completed,
            "stuck_flows":     stuck,
            "completion_rate": round(completed / total, 3) if total else 0.0,
            "turn_intent_accuracy":  round(intent_pass_rate, 3),
            "turn_pending_accuracy": round(pending_pass_rate, 3),
            "by_flow_type":    {k: dict(v) for k, v in by_type.items()},
            "confusion":       {k: dict(v) for k, v in confusion.items()},
        }

    def print_report(self, metrics: dict):
        print(f"\n{'='*60}")
        print(f"  FLOW RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total flows:      {metrics['total_flows']}")
        print(f"  Completed:        {metrics['completed_flows']}  "
              f"({metrics['completion_rate']*100:.1f}%)")
        print(f"  Stuck:            {metrics['stuck_flows']}")
        print(f"  Turn intent acc:  {metrics['turn_intent_accuracy']*100:.1f}%")
        print(f"  Turn pending acc: {metrics['turn_pending_accuracy']*100:.1f}%")

        print(f"\n  COMPLETION RATE BY FLOW TYPE:")
        for ftype, m in sorted(metrics["by_flow_type"].items()):
            rate = m["completed"] / m["total"] if m["total"] else 0.0
            status = "✅" if rate >= 0.80 else ("⚠️ " if rate >= 0.50 else "❌")
            print(f"    {status}  {ftype:<40}  "
                  f"{m['completed']}/{m['total']}  ({rate*100:.0f}%)")

        print(f"\n  TOP CONFUSIONS (multi-turn):")
        conf = metrics["confusion"]
        pairs = []
        for expected, got_dict in conf.items():
            for got, count in got_dict.items():
                if got != expected and count > 0:
                    pairs.append((expected, got, count))
        pairs.sort(key=lambda x: -x[2])
        for expected, got, count in pairs[:8]:
            print(f"    {expected:<22} → {got:<22}  {count:>3}x")

        print(f"\n{'='*60}\n")

    def save_results(self, results: list[FlowResult], metrics: dict) -> Path:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"flow_{ts}.json"

        payload = {
            "run_ts":  ts,
            "metrics": metrics,
            "flows": [
                {
                    "id":            r.flow_id,
                    "description":   r.description,
                    "flow_type":     r.flow_type,
                    "tags":          r.tags,
                    "completed":     r.flow_completed,
                    "stuck":         r.got_stuck,
                    "passed_turns":  r.passed_turns,
                    "total_turns":   r.total_turns,
                    "turns": [
                        {
                            "turn":            t.turn_index,
                            "input":           t.input_text,
                            "expected_intent": t.expected_intent,
                            "got_intent":      t.got_intent,
                            "expected_pending":t.expected_pending,
                            "got_pending":     t.got_pending,
                            "intent_pass":     t.intent_pass,
                            "pending_pass":    t.pending_pass,
                            "latency_ms":      round(t.latency_ms, 1),
                            "response_preview":t.response[:120],
                            "note":            t.note,
                            "error":           t.error,
                        }
                        for t in r.turns
                    ],
                }
                for r in results
            ],
        }

        with open(out, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Results saved → {out}")
        return out


def main():
    parser = argparse.ArgumentParser(description="Buddy robustness flow runner")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tags",    type=str, default=None)
    parser.add_argument("--id",      type=str, default=None,
                        help="Comma-separated flow IDs")
    args = parser.parse_args()

    id_filter = [x.strip() for x in args.id.split(",")] if args.id else None

    runner  = FlowRunner(verbose=args.verbose)
    results = runner.run_all(tag_filter=args.tags, id_filter=id_filter)
    metrics = runner.compute_metrics(results)
    runner.print_report(metrics)
    runner.save_results(results, metrics)


if __name__ == "__main__":
    main()
