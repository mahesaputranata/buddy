"""
prepare_training_data.py — converts a labeled merged_sessions_*.json into a
train/val/test dataset ready for fine-tuning the intent classifier.

GROUND TRUTH LABEL: corrected_intent if the turn was flagged, otherwise the
original intent. Turns whose resolved label is "unknown" are EXCLUDED — a
non-answer isn't something to train toward. Synthetic test-harness inputs
(e.g. "__SIMULATE_IDLE_65_SECONDS__") are excluded too.

INPUT FORMAT: each example concatenates context + input into one string,
since pending_before / the prior turn was the single biggest driver of
correct labels in this dataset (a bare "yes" or a date means nothing without
knowing what question it's answering):

    [TOPIC: <last_topic BEFORE the turn>] [PENDING: <pending_before>] [PREV: <prev input> -> <prev response>] [INPUT: <input>]

Built by brain/intent_model/formatting.py, the same function the live shadow
classifier uses. TOPIC is the previous turn's logged last_topic: the logger
records last_topic after the turn runs, so the current turn's own value leaks
the outcome (e.g. a web search sets "web:...").

DEDUP: exact duplicate (formatted_text, label) pairs are collapsed to one —
repeated stress-test turns would otherwise bias class weights without adding
signal.

SPLIT: done at the SESSION level, not per-example. Near-duplicate turns
within the same session must not end up split across train/eval or eval
accuracy would be inflated by leakage. This means rare labels concentrated
in one or two sessions may end up entirely in train with none in val/test —
a known tradeoff, worth checking the printed label distribution per split.

MOJIBAKE CLEANUP: the raw log has U+FFFD replacement characters where the
original source data lost bytes before this ever reached us (an encoding
bug upstream, not something introduced here). Best-effort heuristic cleanup
is applied — contractions ("wasn<0xFFFD>t" -> "wasn't") and standalone
" <0xFFFD> " -> " — " (em dash), based on the patterns actually observed in
this dataset. This is a guess, not a recovery of ground truth — flagged
here so it's not mistaken for certainty.

Usage:
    python prepare_training_data.py --in merged_sessions_full.json --out-dir training_data
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Shared with the shadow classifier so training and serving format text identically.
from brain.intent_model.formatting import build_input_text  # noqa: E402

_SYNTHETIC_INPUT_RE = re.compile(r"^__[A-Z0-9_]+__$")


def build_prev_index(data: list[dict]) -> dict[str, dict]:
    by_session: dict[str, list[dict]] = {}
    for e in data:
        by_session.setdefault(e.get("session_id"), []).append(e)
    for turns in by_session.values():
        turns.sort(key=lambda e: e.get("turn", 0))
    prev_of: dict[str, dict] = {}
    for turns in by_session.values():
        for i in range(1, len(turns)):
            prev_of[turns[i]["id"]] = turns[i - 1]
    return prev_of


def resolve_label(entry: dict) -> str | None:
    review = entry.get("review") or {}
    if review.get("flagged") and review.get("corrected_intent"):
        label = review["corrected_intent"]
    else:
        label = entry.get("intent")
    if not label or label == "unknown":
        return None
    return label


def format_input(entry: dict, prev_entry: dict | None) -> str:
    ctx = entry.get("context") or {}
    # The session logger writes last_topic AFTER the turn ran, so this turn's own
    # logged last_topic leaks its outcome (a web search sets "web:..." itself).
    # The state BEFORE this turn is what the previous turn logged. Using the
    # current turn's value here was a train/serve skew bug.
    topic_before = (prev_entry.get("context") or {}).get("last_topic") if prev_entry is not None else None
    return build_input_text(
        entry.get("input", ""),
        last_topic=topic_before,
        pending_before=ctx.get("pending_before"),
        prev_input=prev_entry.get("input", "") if prev_entry is not None else None,
        prev_response=prev_entry.get("response", "") if prev_entry is not None else None,
    )


def split_sessions(
    session_ids: list[str], train_frac: float, val_frac: float, seed: int,
) -> dict[str, str]:
    """Assigns each session_id to 'train'/'val'/'test'. Splitting by session,
    not by example, to avoid near-duplicate turns leaking across splits."""
    ids = sorted(set(session_ids))  # sort first for determinism, then shuffle with seed
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    assignment = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            assignment[sid] = "train"
        elif i < n_train + n_val:
            assignment[sid] = "val"
        else:
            assignment[sid] = "test"
    return assignment


def guarantee_label_coverage(
    examples: list[dict], assignment: dict[str, str], seed: int,
) -> dict[str, str]:
    """
    Pure random session assignment can leave a rare label with ZERO examples
    in val or test purely by chance (all its sessions happened to land in
    train) — that produces a meaningless support=0 row in the eval report,
    not a real measure of anything. For every label that has at least 2
    distinct sessions available, force at least one session into val and one
    into test (moving them out of train) so every label gets SOME real eval
    coverage. Labels confined to a single session can't be fixed this way —
    that's a real data-scarcity limit, not a splitting bug.
    """
    assignment = dict(assignment)
    sessions_by_label: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        sessions_by_label[ex["label"]].add(ex["session_id"])

    rng = random.Random(seed + 1)
    rescued = []
    for label, sessions in sessions_by_label.items():
        sessions = sorted(sessions)
        for target_split in ("val", "test"):
            if any(assignment[s] == target_split for s in sessions):
                continue  # already has coverage there
            candidates = [s for s in sessions if assignment[s] == "train"]
            if not candidates:
                continue  # only option left is to steal from the OTHER eval split — skip, not worth it
            rng.shuffle(candidates)
            chosen = candidates[0]
            assignment[chosen] = target_split
            rescued.append((label, target_split, chosen))

    if rescued:
        print(f"[COVERAGE] rescued {len(rescued)} session(s) to guarantee eval coverage:")
        for label, split_name, sid in rescued:
            print(f"    {label:20} -> moved session {sid} into {split_name}")

    return assignment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overrides", default=None,
                     help="label_overrides.json: explicit per-turn label corrections (id -> label, or null to drop)")
    args = ap.parse_args()

    overrides: dict = {}
    if args.overrides:
        with open(args.overrides, encoding="utf-8") as f:
            overrides = json.load(f).get("overrides", {})
        print(f"[OVERRIDES] loaded {len(overrides)} from {args.overrides}")

    with open(args.in_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[LOAD] {len(data)} total turns")

    prev_of = build_prev_index(data)

    examples = []
    excluded_unknown = 0
    excluded_synthetic = 0
    applied_overrides = 0
    for e in data:
        if _SYNTHETIC_INPUT_RE.match((e.get("input") or "").strip()):
            excluded_synthetic += 1
            continue
        label = resolve_label(e)
        if e["id"] in overrides:
            label = overrides[e["id"]].get("label")  # None => explicitly dropped
            applied_overrides += 1
        if label is None:
            excluded_unknown += 1
            continue
        text = format_input(e, prev_of.get(e["id"]))
        examples.append({
            "text": text,
            "label": label,
            "id": e["id"],
            "session_id": e["session_id"],
            "turn": e["turn"],
        })

    print(f"[FILTER] excluded {excluded_unknown} unresolved 'unknown' turns, "
          f"{excluded_synthetic} synthetic test-harness turns")
    if overrides:
        print(f"[OVERRIDES] applied to {applied_overrides} turns")
    print(f"[FILTER] {len(examples)} candidate examples")

    # Dedup exact (text, label) pairs
    seen = set()
    deduped = []
    for ex in examples:
        key = (ex["text"], ex["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    print(f"[DEDUP] {len(examples) - len(deduped)} exact duplicates removed -> {len(deduped)} examples")
    examples = deduped

    # Session-level split
    session_ids = [ex["session_id"] for ex in examples]
    split_of = split_sessions(session_ids, args.train_frac, args.val_frac, args.seed)
    split_of = guarantee_label_coverage(examples, split_of, args.seed)

    splits: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        splits[split_of[ex["session_id"]]].append(ex)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        path = out_dir / f"intent_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WRITE] {path} ({len(rows)} examples)")

    all_labels = sorted({ex["label"] for ex in examples})
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels, "label2id": {l: i for i, l in enumerate(all_labels)}}, f, indent=2)
    print(f"[WRITE] {labels_path} ({len(all_labels)} labels)")

    print("\n[SUMMARY] label distribution per split:")
    for split_name in ["train", "val", "test"]:
        rows = splits.get(split_name, [])
        counts = Counter(r["label"] for r in rows)
        print(f"  {split_name} ({len(rows)} total):")
        for label in all_labels:
            c = counts.get(label, 0)
            flag = "  <-- ZERO in this split" if c == 0 else ""
            print(f"    {label:20} {c:5}{flag}")


if __name__ == "__main__":
    main()
