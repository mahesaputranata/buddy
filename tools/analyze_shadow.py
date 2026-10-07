"""
analyze_shadow.py — reads the session logs written while shadow mode was on and
compares the trained classifier's prediction (entry["shadow"]) with what the old
classifier decided (entry["intent"]).

It answers the cutover questions:
  - How often do they agree, and for which intents do they disagree?
  - At what shadow confidence is the model trustworthy enough to act on?
  - On turns the old system punted on (intent "unknown", i.e. pre_classify),
    what would the model have said, and how sure is it?
  - Which high-confidence disagreements should I look at by hand?
  - If turns have been labeled with label_intents.py (review.corrected_intent):
    how many old-system mistakes would the model have fixed, and how many
    correct turns would it have broken?

Usage:
    python analyze_shadow.py                       # logs/sessions, all dates
    python analyze_shadow.py --since 2026-10-04 --min-conf 0.85 --show 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_entries(log_dir: Path, since: str | None) -> list[dict]:
    entries = []
    for f in sorted(log_dir.glob("session_*.jsonl")):
        if since and f.stem.replace("session_", "") < since:
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("shadow"):
                    entries.append(e)
    return entries


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/sessions")
    ap.add_argument("--since", default=None, help="only log files from this date on, YYYY-MM-DD")
    ap.add_argument("--min-conf", type=float, default=0.8, help="'confident' threshold for the disagreement list")
    ap.add_argument("--show", type=int, default=20, help="how many disagreements to print")
    args = ap.parse_args()

    entries = load_entries(Path(args.logs), args.since)
    if not entries:
        print(f"No entries with a 'shadow' field found in {args.logs}. "
              "Run Buddy with the shadow hook in place first.")
        return

    model_labels = {lab for e in entries for lab, _ in e["shadow"].get("top3", [])}
    lat = sorted(e["shadow"].get("latency_ms", 0) for e in entries)
    print(f"[COVERAGE] {len(entries)} shadowed turns, {len(model_labels)} model labels")
    print(f"[LATENCY]  median {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * 0.95)]:.0f} ms")

    known = [e for e in entries if e.get("intent") not in (None, "unknown")]
    unknown = [e for e in entries if e.get("intent") in (None, "unknown")]
    out_of_space = [e for e in known if e["intent"] not in model_labels]
    comparable = [e for e in known if e["intent"] in model_labels]

    print(f"\n[TURNS] old system gave a real intent: {len(known)}  |  old system punted (unknown): {len(unknown)}")
    if out_of_space:
        c = Counter(e["intent"] for e in out_of_space)
        print(f"        {len(out_of_space)} old intents the model cannot predict (not in its label space): {dict(c)}")

    # ── agreement ────────────────────────────────────────────────────────────
    agree = sum(e["shadow"]["label"] == e["intent"] for e in comparable)
    print(f"\n[AGREEMENT] model == old classifier on {agree}/{len(comparable)} comparable turns ({pct(agree, len(comparable))})")

    by_old = defaultdict(lambda: [0, 0])
    confusions = Counter()
    for e in comparable:
        by_old[e["intent"]][1] += 1
        if e["shadow"]["label"] == e["intent"]:
            by_old[e["intent"]][0] += 1
        else:
            confusions[(e["intent"], e["shadow"]["label"])] += 1
    print(f"  {'old intent':18} {'n':>5}  agree")
    for intent, (a, n) in sorted(by_old.items(), key=lambda kv: -kv[1][1]):
        print(f"  {intent:18} {n:5}  {pct(a, n)}")
    if confusions:
        print("  most common disagreements (old -> model):")
        for (o, m), n in confusions.most_common(8):
            print(f"    {o:16} -> {m:16} {n}")

    # ── confidence calibration: where can the model be trusted? ──────────────
    print("\n[CALIBRATION] agreement with old classifier by model confidence")
    buckets = [(0.95, 1.01), (0.9, 0.95), (0.8, 0.9), (0.6, 0.8), (0.0, 0.6)]
    for lo, hi in buckets:
        sel = [e for e in comparable if lo <= e["shadow"]["confidence"] < hi]
        a = sum(e["shadow"]["label"] == e["intent"] for e in sel)
        print(f"  conf {lo:.2f}-{min(hi, 1.0):.2f}  n={len(sel):4}  agree {pct(a, len(sel))}")

    # ── turns the old system punted on ───────────────────────────────────────
    if unknown:
        c = Counter(e["shadow"]["label"] for e in unknown)
        confident = sum(e["shadow"]["confidence"] >= args.min_conf for e in unknown)
        print(f"\n[PUNTED TURNS] {len(unknown)} turns where the old system said 'unknown'; "
              f"model is >= {args.min_conf} confident on {confident} ({pct(confident, len(unknown))})")
        for lab, n in c.most_common():
            print(f"  would have said {lab:16} {n}")

    # ── disagreements to eyeball ─────────────────────────────────────────────
    hot = [e for e in comparable
           if e["shadow"]["label"] != e["intent"] and e["shadow"]["confidence"] >= args.min_conf]
    hot.sort(key=lambda e: -e["shadow"]["confidence"])
    print(f"\n[REVIEW THESE] {len(hot)} confident disagreements (model conf >= {args.min_conf}); first {min(args.show, len(hot))}:")
    for e in hot[: args.show]:
        print(f"  old={e['intent']:14} ({e.get('source')})  model={e['shadow']['label']:14} {e['shadow']['confidence']:.2f}  | {e['input'][:80]!r}")

    # ── fixes vs breaks, when turns have been labeled ────────────────────────
    labeled = [e for e in entries if (e.get("review") or {}).get("flagged") and (e["review"].get("corrected_intent"))]
    if labeled:
        fixed = sum(e["shadow"]["label"] == e["review"]["corrected_intent"] for e in labeled)
        broke = sum(e["shadow"]["label"] != e["intent"] for e in comparable
                    if not (e.get("review") or {}).get("flagged"))
        unflagged = sum(1 for e in comparable if not (e.get("review") or {}).get("flagged"))
        print(f"\n[FIXES vs BREAKS] (uses review.corrected_intent from label_intents.py)")
        print(f"  old system was wrong on {len(labeled)} labeled turns; the model gets {fixed} right ({pct(fixed, len(labeled))})")
        print(f"  old system assumed right on {unflagged} turns; the model would change {broke} of them ({pct(broke, unflagged)})")


if __name__ == "__main__":
    main()
