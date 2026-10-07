"""
augment_sparse_intents.py — generates synthetic paraphrase examples for the
sparsest intent classes in the training set, seeded from your own real
examples of that class (not generic invented phrasing).

Only touches classes with fewer than --min-count real training examples.
Synthetic rows are tagged "synthetic": true and written to a SEPARATE file
(intent_train_augmented.jsonl) alongside the original real rows (tagged
"synthetic": false) — the original intent_train.jsonl is left untouched, so
you can train with or without augmentation and compare.

Synthetic data is train-only, by design — it never touches val/test, since
eval has to reflect real usage, not the paraphrase generator's style.

Usage:
    python augment_sparse_intents.py --backend openai --train training_data/intent_train.jsonl --out training_data/intent_train_augmented.jsonl --min-count 30 --target 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MODELS = {
    "ollama": "llama3.2:latest",
    "openai": "gpt-4o-mini",
}

_INPUT_RE = re.compile(r"\[INPUT: (.*)\]\s*$", re.DOTALL)

SYSTEM_PROMPT = """You generate realistic paraphrases of things a specific person says to \
their personal voice assistant "Buddy" — NOT generic textbook phrasing. You'll be given \
several REAL things this person actually said for one intent category. Generate new \
utterances that:
- Preserve the exact same intent/meaning as the seeds
- Sound like the SAME PERSON — same register, same habit of trailing off or adding asides,
  same technical vocabulary if the seeds show it (rover, PID, Kalman filter, etc.), same
  mix of formal/casual phrasing you see in the seeds
- Vary sentence structure, directness, and length — some short and blunt, some longer and
  hedged ("can you maybe...", "actually could you...")
- Vary any concrete details (event names, dates, times, topics) rather than reusing the
  seeds' exact details verbatim
- Do NOT invent a wildly different scenario that technically fits the intent but doesn't
  sound like this person's real usage patterns

Respond with ONLY a JSON object: {"utterances": ["...", "...", ...]} — no other text.
"""


def _chat_openai(model: str, system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.9,  # want variety here, unlike the labeling script
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


def _chat_ollama(model: str, system: str, user: str) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options={"temperature": 0.9},
        format="json",
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


_BACKENDS = {"ollama": _chat_ollama, "openai": _chat_openai}


def extract_input_text(formatted_text: str) -> str:
    m = _INPUT_RE.search(formatted_text)
    return m.group(1).strip() if m else formatted_text


def generate_paraphrases(seeds: list[str], n: int, backend: str, model: str) -> list[str]:
    seed_block = "\n".join(f"- {s}" for s in seeds)
    user = f"REAL examples for this intent:\n{seed_block}\n\nGenerate {n} new utterances."
    try:
        raw = _BACKENDS[backend](model, SYSTEM_PROMPT, user)
        parsed = json.loads(raw)
        utterances = parsed.get("utterances", [])
        return [u.strip() for u in utterances if isinstance(u, str) and u.strip()]
    except Exception as e:
        print(f"  [ERROR] generation failed: {e}")
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="Path to intent_train.jsonl (real data)")
    ap.add_argument("--out", required=True, help="Output path for the augmented train file")
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-count", type=int, default=30,
                     help="Only augment classes with fewer than this many real train examples")
    ap.add_argument("--target", type=int, default=40,
                     help="Bring each targeted class up to roughly this many real+synthetic examples")
    ap.add_argument("--batch-size", type=int, default=15, help="Utterances requested per API call")
    ap.add_argument("--exclude-labels", default="",
                     help="Comma-separated labels to skip even if under min-count (e.g. classes known to "
                          "produce unreliable synthetic data, or with real seeds too context-dependent to paraphrase)")
    args = ap.parse_args()
    excluded_labels = {l.strip() for l in args.exclude_labels.split(",") if l.strip()}

    model = args.model or DEFAULT_MODELS[args.backend]
    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] --backend openai requires OPENAI_API_KEY to be set.")
        sys.exit(1)

    with open(args.train, encoding="utf-8") as f:
        real_rows = [json.loads(line) for line in f if line.strip()]

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in real_rows:
        by_label[r["label"]].append(r)

    counts = Counter(r["label"] for r in real_rows)
    print(f"[LOAD] {len(real_rows)} real training examples across {len(counts)} labels")

    targets = {label: c for label, c in counts.items() if c < args.min_count and label not in excluded_labels}
    if excluded_labels:
        print(f"[TARGET] explicitly excluded: {sorted(excluded_labels)}")
    print(f"[TARGET] augmenting {len(targets)} sparse classes: {targets}")

    synthetic_rows = []
    for label, current_count in targets.items():
        seeds_all = [extract_input_text(r["text"]) for r in by_label[label]]
        if not seeds_all:
            print(f"\n[{label}] 0 real seed examples — can't paraphrase from nothing, skipping")
            continue
        needed = max(0, args.target - current_count)
        print(f"\n[{label}] {current_count} real -> target {args.target} (need {needed} synthetic)")

        generated: list[str] = []
        while len(generated) < needed:
            batch_n = min(args.batch_size, needed - len(generated))
            new = generate_paraphrases(seeds_all, batch_n, args.backend, model)
            # de-dup against seeds and against what we've already generated
            existing = set(s.lower() for s in seeds_all) | set(u.lower() for u in generated)
            new = [u for u in new if u.lower() not in existing]
            if not new:
                print(f"  [WARN] generation stalled at {len(generated)}/{needed} — stopping this class")
                break
            generated.extend(new)
            print(f"  generated {len(generated)}/{needed}")

        for u in generated:
            synthetic_rows.append({
                "text": f"[TOPIC: NONE] [PENDING: NONE] [PREV: NONE] [INPUT: {u}]",
                "label": label,
                "id": None,
                "session_id": None,
                "turn": None,
                "synthetic": True,
            })

    # Write augmented file: all real rows (tagged) + new synthetic rows
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in real_rows:
            r["synthetic"] = False
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in synthetic_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[DONE] wrote {out_path}")
    print(f"  real: {len(real_rows)}  synthetic: {len(synthetic_rows)}  total: {len(real_rows) + len(synthetic_rows)}")
    final_counts = Counter(r["label"] for r in real_rows) + Counter(r["label"] for r in synthetic_rows)
    for label in sorted(final_counts):
        marker = " (augmented)" if label in targets else ""
        print(f"    {label:20} {final_counts[label]:5}{marker}")


if __name__ == "__main__":
    main()
