"""
adversarial_generator.py
========================
Uses your existing Ollama LLM to auto-expand manual seed test cases
into adversarial variations — rephrasing, intent-ambiguating, adding noise,
injecting domain terms, and creating mid-intent-switch scenarios.

The generated cases are saved to test_cases/adversarial.jsonl and can then
be run through unit_runner.py.

Usage:
    python -m robustness_tests.runner.adversarial_generator
    python -m robustness_tests.runner.adversarial_generator --intent schedule_save
    python -m robustness_tests.runner.adversarial_generator --n 5
    python -m robustness_tests.runner.adversarial_generator --dry-run
"""

from __future__ import annotations

import sys
import json
import argparse
import time
import re
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_CASES_DIR = Path(__file__).resolve().parents[1] / "test_cases"


# ── Adversarial variation strategies ──────────────────────────────────────
STRATEGIES = {
    "rephrase": (
        "Rephrase the input naturally — same intent, different words. "
        "Could be more casual, more formal, or use a different sentence structure."
    ),
    "add_noise": (
        "Add a filler phrase or off-topic word at the start or end — "
        "like a real user who mumbles or adds context. "
        "The core intent should still be detectable."
    ),
    "ambiguate": (
        "Make the intent slightly ambiguous — add a word or phrase that could "
        "point to a DIFFERENT intent, but the core intent is still the most likely one. "
        "Do NOT change the core intent."
    ),
    "engineering_inject": (
        "Add an engineering or technical term (pid, rover, lidar, sensor, mpc, uav, "
        "controller, ekf) somewhere natural in the sentence. "
        "The core intent should NOT change — this tests the engineering domain override."
    ),
    "broken_grammar": (
        "Simulate imperfect speech-to-text: remove some punctuation, lowercase everything, "
        "add a minor typo or dropped word. Keep the intent identifiable."
    ),
    "over_polite": (
        "Add excessive politeness: 'could you please', 'i was wondering if you could', "
        "'would it be possible to', etc. Intent should stay the same."
    ),
    "implicit": (
        "Make the intent more implicit — remove the explicit verb/keyword and rely on "
        "context. For example 'remind me' → 'dentist tomorrow 3pm' (still schedule_save). "
        "The intent should remain the same."
    ),
}

# ── Per-intent generation config ──────────────────────────────────────────
INTENT_STRATEGIES = {
    "schedule_save":    ["rephrase", "add_noise", "engineering_inject", "implicit"],
    "schedule_recall":  ["rephrase", "ambiguate", "engineering_inject"],
    "save_memory":      ["rephrase", "ambiguate", "over_polite"],
    "recall_memory":    ["rephrase", "ambiguate", "broken_grammar"],
    "delete_memory":    ["rephrase", "ambiguate", "over_polite"],
    "web_search":       ["rephrase", "add_noise", "broken_grammar"],
    "web_followup":     ["rephrase", "implicit"],
    "device_control":   ["rephrase", "add_noise", "engineering_inject"],
    "system_query":     ["rephrase", "broken_grammar", "over_polite"],
    "chat":             ["rephrase", "add_noise", "ambiguate"],
}


class AdversarialGenerator:
    def __init__(self, model: str = "llama3.2:latest", dry_run: bool = False):
        self.model   = model
        self.dry_run = dry_run
        self._counter = 0

        if not dry_run:
            try:
                import ollama
                self.ollama = ollama
            except ImportError:
                print("[ERROR] ollama not importable.")
                sys.exit(1)

    def _next_id(self, base_id: str, strategy: str) -> str:
        self._counter += 1
        return f"adv_{base_id}_{strategy}_{self._counter:03d}"

    def _generate_variation(
        self,
        original: str,
        intent: str,
        strategy: str,
        strategy_desc: str,
    ) -> Optional[str]:
        """
        Calls LLM to produce one adversarial variation.
        Returns the variation string, or None if generation fails.
        """
        prompt = f"""You are a test data generator for a voice assistant intent classifier.

Original utterance: "{original}"
Intended intent classification: {intent}

Your task: {strategy_desc}

RULES:
- Return ONLY the new utterance, nothing else.
- Do NOT add quotes or explanation.
- Do NOT include the original utterance.
- Maximum 20 words.
- The intent must still be {intent} after your modification.
- Make it sound like a real person saying it."""

        try:
            response = self.ollama.generate(model=self.model, prompt=prompt)
            raw = (response.get("response") or "").strip()

            # basic sanity: reject empty, too long, or returned JSON
            if not raw or len(raw.split()) > 30 or raw.startswith("{"):
                return None

            # strip accidental quotes
            raw = re.sub(r'^["\']|["\']$', '', raw).strip()

            return raw if raw else None

        except Exception as e:
            print(f"    [LLM error] {e}")
            return None

    def expand_seed(
        self,
        case: dict,
        n_per_strategy: int = 2,
        intent_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Takes one seed test case and returns a list of adversarial variations.
        """
        seed_id     = case["id"]
        original    = case["input"]
        intent      = case["expected_intent"]
        seed_tags   = case.get("tags", [])

        if intent_filter and intent != intent_filter:
            return []

        strategies = INTENT_STRATEGIES.get(intent, ["rephrase"])
        new_cases  = []

        for strategy in strategies:
            desc = STRATEGIES.get(strategy, "Rephrase naturally.")

            for attempt in range(n_per_strategy):
                if self.dry_run:
                    variation = f"[DRY RUN] {strategy} of: {original}"
                else:
                    variation = self._generate_variation(
                        original, intent, strategy, desc
                    )
                    time.sleep(0.1)  # light rate limiting

                if not variation:
                    continue

                new_id = self._next_id(seed_id, strategy)
                new_cases.append({
                    "id":              new_id,
                    "input":           variation,
                    "expected_intent": intent,
                    "tags":            seed_tags + ["adversarial", strategy],
                    "seed_id":         seed_id,
                    "strategy":        strategy,
                    "original":        original,
                })

        return new_cases

    def run(
        self,
        n_per_strategy: int = 2,
        intent_filter:  Optional[str] = None,
        max_seeds:      Optional[int] = None,
    ) -> list[dict]:
        seeds_path = TEST_CASES_DIR / "single_turn.jsonl"
        seeds = []
        with open(seeds_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seeds.append(json.loads(line))

        if intent_filter:
            seeds = [s for s in seeds if s["expected_intent"] == intent_filter]
        if max_seeds:
            seeds = seeds[:max_seeds]

        total = len(seeds)
        print(f"\n{'='*60}")
        print(f"  ADVERSARIAL GENERATOR")
        print(f"  Seeds:          {total}")
        print(f"  N per strategy: {n_per_strategy}")
        print(f"  Intent filter:  {intent_filter or 'all'}")
        if self.dry_run:
            print(f"  Mode:           DRY RUN (no LLM calls)")
        print(f"{'='*60}\n")

        all_cases: list[dict] = []

        for i, seed in enumerate(seeds, 1):
            print(f"  [{i:>3}/{total}] Expanding {seed['id']} "
                  f"({seed['expected_intent']})...", end="", flush=True)

            cases = self.expand_seed(
                seed,
                n_per_strategy=n_per_strategy,
                intent_filter=intent_filter,
            )
            all_cases.extend(cases)
            print(f" → {len(cases)} variations")

        # Save to adversarial.jsonl
        out_path = TEST_CASES_DIR / "adversarial.jsonl"
        with open(out_path, "w") as f:
            for c in all_cases:
                f.write(json.dumps(c) + "\n")

        print(f"\n  Generated: {len(all_cases)} adversarial cases")
        print(f"  Saved to:  {out_path}")
        print(f"\n  Run them with:")
        print(f"    python -m robustness_tests.runner.unit_runner "
              f"--cases adversarial\n")

        return all_cases


def main():
    parser = argparse.ArgumentParser(description="Adversarial test case generator")
    parser.add_argument("--intent",  type=str, default=None,
                        help="Only expand seeds for this intent")
    parser.add_argument("--n",       type=int, default=2,
                        help="Variations per strategy per seed (default: 2)")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Max seed cases to expand (for quick runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, show structure only")
    args = parser.parse_args()

    gen = AdversarialGenerator(dry_run=args.dry_run)
    gen.run(
        n_per_strategy=args.n,
        intent_filter=args.intent,
        max_seeds=args.max_seeds,
    )


if __name__ == "__main__":
    main()
