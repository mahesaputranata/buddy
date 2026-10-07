"""
label_intents.py — LLM-assisted labeling pass over Buddy's merged_sessions.json

Fills in the EXISTING `review` block (flagged, issue, corrected_intent,
corrected_response, note) on each turn. Nothing else in the entry is touched —
input/response/intent/confidence/source/tier/context/outcome are left exactly
as they are, since those are the historical record of what Buddy actually did.

Judgment is made using only what Buddy actually had available at the time:
the previous turn (often the question a "yes"/"no" reply is answering) and
the recorded last_topic / pending_before state — not hindsight.

Safe to re-run / resume: entries that already have review.corrected_intent
set (from a previous run) are skipped unless --relabel is passed. Progress
is checkpointed to --out every --checkpoint-every entries, so an interrupted
run loses at most one checkpoint's worth of work.

Usage:
    # Dry run on a handful of low-confidence turns first
    python label_intents.py --in merged_sessions.json --out merged_sessions_labeled.json --max-confidence 0.5 --limit 20

    # Full pass over everything still unlabeled, prioritizing low confidence first
    python label_intents.py --in merged_sessions.json --out merged_sessions_labeled.json --max-confidence 0.5

    # Resume / continue into the rest of the dataset
    python label_intents.py --in merged_sessions_labeled.json --out merged_sessions_labeled.json

Backends:
    --backend ollama (default) — requires `pip install ollama` and a local Ollama
        server running (same model family already used elsewhere in Buddy —
        brain/router/confirmation_router.py, brain/projects/project_handler.py).
    --backend openai — requires `pip install openai` and an OPENAI_API_KEY env var.
        Example:
        python label_intents.py --backend openai --model gpt-4o-mini --in ... --out ... --max-confidence 0.5 --limit 15
"""

from __future__ import annotations

import argparse
import os
import json
import random
import re
import sys
import time
from pathlib import Path

DEFAULT_MODELS = {
    "ollama": "llama3.2:latest",
    "openai": "gpt-4o-mini",
}

# Kept in sync with brain/core/intent/zero_shot_classifier.py's
# INTENT_DESCRIPTIONS, extended with the intents that classifier didn't cover
# (schedule_delete, friday_task, job_watchdog, unknown). Duplicated here
# rather than imported so this script has zero dependency on Buddy's package
# layout and can run standalone.
ALLOWED_INTENTS = {
    "chat": "general conversation, question, explanation, curiosity, opinion, technical discussion",
    "recall_memory": "user wants to retrieve or look up something previously saved",
    "save_memory": "user explicitly wants to save/store the current conversation or topic right now",
    "delete_memory": "user wants to delete, remove, erase, forget, or wipe something from memory",
    "schedule_save": "user wants to set a reminder, schedule an appointment, or add an event",
    "schedule_recall": "user asks about their schedule, upcoming events, or reminders",
    "schedule_delete": "user wants to delete, cancel, or remove a saved reminder or schedule item",
    "web_search": "user wants to search the web / find current news or information online",
    "web_followup": "user is following up on a previous web search result (open/summarize/compare)",
    "device_control": "user is giving a direct command to control a drone, rover, or robot device",
    "system_query": "user asks about current time, date, battery, weather, location, or CPU load",
    "friday_task": "user is delegating a background task to Friday, the coding/QA agent",
    "job_watchdog": "user wants Friday's job-search scan run or read",
    "unknown": "genuinely cannot tell what the user wants from this turn alone",
}

ISSUE_CATEGORIES = [
    "wrong_intent",           # classifier picked the wrong intent entirely
    "confirmation_bug",       # yes/no or pending-state resolved against the wrong question
    "stale_context",          # response pulled in the wrong last_topic / blended topics
    "hallucinated_response",  # response invented content not grounded in a real tool result
    "tool_failure_ignored",   # a tool failed but the response didn't acknowledge it
    "wrong_response_text",    # intent was right, but the reply itself is bad for another reason
]

# Rough guide for what a reply is USUALLY continuing when a given pending_before
# state is active. Not authoritative — the model should still use judgment (an
# explicit cancel/pivot in the input overrides this) — but it stops the model
# from re-classifying a slot-fill answer ("12th March, 2026.") as if it were a
# fresh, standalone utterance.
PENDING_STATE_HINTS = """
Guide to what a reply is usually continuing, based on pending_before (override this
if the input clearly cancels or pivots away instead, e.g. "never mind", "actually let's
talk about X" — those are usually `chat`, not the flow's intent):
- Awaiting_Schedule_Details, Awaiting_Schedule_Recall_Disambiguation -> schedule_save / schedule_recall
- Awaiting_Delete_Confirm, Awaiting_Delete_TopicName, Awaiting_Temporal_Delete_Confirm -> delete_memory
- Awaiting_Schedule_Delete_Confirm, Awaiting_Schedule_Delete_Pick -> schedule_delete
- Awaiting_Topic, Awaiting_Topic_Content, Awaiting_Topic_Add_Info, Awaiting_ConversationSave_Confirm, Awaiting_ConversationSave_TopicPick -> save_memory
- Awaiting_General_Recall, Awaiting_Temporal_Recall, Awaiting_Specific_Recall_Menu -> recall_memory
- Awaiting_Rover_Sequence_Confirmation, Awaiting_DeviceControl_Action, Awaiting_LLM_Tool_Confirm -> device_control
- Awaiting_Friday_Confirm -> friday_task
- Awaiting_Project_Resume, Awaiting_Note_Confirm -> chat
A bare "yes"/"no"/date/name/number answered right after one of these states is a
SLOT-FILL ANSWER to the question Buddy just asked, not a new topic to classify from
scratch — judge it as a continuation of that flow's intent unless it's an explicit cancel.

Exception: a literal, unambiguous device command ("connect the rover", "move forward",
"land the drone") arriving during a save_memory/recall_memory/schedule flow (e.g.
Awaiting_Topic_Content) is an IMPLICIT PIVOT to device_control, even with no cancel word —
a real command should not get silently absorbed as note content or recall text. This is
the mirror case of the cancel exception above: strong, unambiguous command phrasing
overrides flow continuation just as much as an explicit "never mind" does.
"""

SOURCE_TIER_NOTE = """
Note on source/tier: source="pre_classify" with tier=0 and intent="unknown" often means
this turn was caught by a PRE-classification guard (e.g. an interrupt/cancel detector)
that runs before real intent classification and never needed to classify it — that is
CORRECT, EXPECTED behavior, not a bug. In that case, judge the REPLY on its own merits
(is "Got it — cancelled, what would you like to do?" a reasonable thing to say to this
input, given the context?) rather than assuming "unknown" itself must be wrong. Only
flag it if the reply itself doesn't make sense for the input.

IMPORTANT — do not confuse two different fields: source="unknown" is a LOGGING gap (the
system didn't record which code path classified this turn), completely separate from
intent="unknown" (the system failed to classify it). A turn can have source="unknown" while
intent already holds a perfectly correct value like "device_control" or "web_search" — in
that case there is NO intent bug. Do not flag corrected_intent as the same value the intent
field already had and call it "wrong_intent" — if the intent field's value is already
correct, flagged must be false, full stop, regardless of what source or confidence say.
Confidence being 0.0 while the intent value itself is correct is not something you can fix
by relabeling — it's not your job to grade the confidence NUMBER, only the intent CATEGORY.

Also: input like "__SIMULATE_IDLE_65_SECONDS__" or any other __ALL_CAPS__ token is synthetic
test-harness scaffolding injected by an automated test run, not real user speech. Do not
interpret it as a command or classify it as anything — set flagged=false and leave a note
that this is test scaffolding, not a real utterance.
"""

SYSTEM_PROMPT = """You are auditing logs from a personal voice assistant named Buddy, \
to build a labeled dataset for training a new intent classifier and for fine-tuning \
Buddy's response model.

For each turn you'll see: the prior turn (if any — many replies like "yes"/"no"/"never \
mind" only make sense relative to whatever Buddy just asked), Buddy's belief state going \
into this turn (last_topic, pending_before), the user's input, what Buddy's system \
classified the intent as, and Buddy's actual reply.

Judge ONLY using what Buddy had available at the time — do not use hindsight knowledge \
the transcript doesn't contain. Decide whether the intent classification was correct and \
whether the reply was appropriate given that context.
""" + PENDING_STATE_HINTS + SOURCE_TIER_NOTE + """

Allowed intents (corrected_intent must be exactly one of these):
""" + "\n".join(f"- {k}: {v}" for k, v in ALLOWED_INTENTS.items()) + """

If the original intent and reply were both fine: flagged=false, corrected_intent=null, \
issue=null, note=null.

If flagged=true, corrected_intent MUST be filled with exactly one of the allowed intents \
above — never leave corrected_intent null when flagged is true. If you cannot decide on a \
specific correct intent, set flagged=false instead and explain the ambiguity in note.

Worked examples (these are real past mistakes another auditor made — do not repeat them):
- A bare date/time/name given while pending_before is an "Awaiting_Schedule_Details"-style
  state is answering "when is X?" — but check what STARTED that flow, not just the state's
  generic name. If the prior turns show the flow actually began as a DELETE request (e.g.
  "can you delete my schedule about X?") that got misrouted into a create-flow, the correct
  intent for the date reply is still schedule_delete — the state name lags the true intent,
  it doesn't override it. Do not default to schedule_save just because the state is named
  "Awaiting_Schedule_Details".
- source="pre_classify", intent="unknown", input="Never mind.", reply="Got it — cancelled.
  What would you like to do?" -> the reply is a reasonable cancellation ack -> flagged=false
  (or if you do flag it for the intent label itself, corrected_intent="chat", never null).
- A statement DESCRIBING a plan or methodology ("I will record/log/observe/measure/track
  the rover's speed...", "the way I plan to test this is...") is `chat` — ongoing technical
  discussion — even though it mentions a device. It is NOT `device_control` unless the user
  is directly commanding the device to do something right now (move, connect, stop, land).

CONSISTENCY: if the prior turn (or a turn just before it in this same session) is a near-
duplicate of the current one — same session, same pending_before, same or paraphrased input,
answering the same open question — and you already know what it should be labeled, give this
turn the SAME corrected_intent. Do not contradict a judgment you would make one turn away for
essentially the same content.

If something was wrong: flagged=true, corrected_intent=<the correct intent from the list \
above>, issue=<exactly one of: """ + ", ".join(ISSUE_CATEGORIES) + """>, note=<one sentence \
explaining specifically what went wrong>.

Only fill corrected_response if you can write a genuinely better reply in Buddy's voice \
(short, spoken-style, one to two sentences) — otherwise leave it null. Do not fill it just \
because the intent was wrong; only when the actual reply text itself needs fixing.

Respond with ONLY a single JSON object, no prose before or after, exactly this shape:
{"flagged": true/false, "corrected_intent": "<intent>"|null, "issue": "<issue>"|null, "corrected_response": "<text>"|null, "note": "<text>"|null}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_user_message(entry: dict, prev_entry: dict | None) -> str:
    parts = []

    if prev_entry is not None:
        parts.append(
            "PRIOR TURN (same session):\n"
            f'  User said: "{prev_entry.get("input", "")}"\n'
            f'  Buddy replied: "{prev_entry.get("response", "")}"\n'
        )
    else:
        parts.append("PRIOR TURN: (none — this is the first turn of the session)\n")

    ctx = entry.get("context", {}) or {}
    parts.append(
        "BUDDY'S STATE GOING INTO THIS TURN:\n"
        f'  last_topic: {ctx.get("last_topic")!r}\n'
        f'  pending_before: {ctx.get("pending_before")!r}\n'
    )

    parts.append(
        "THIS TURN:\n"
        f'  User said: "{entry.get("input", "")}"\n'
        f'  Buddy classified this as: intent={entry.get("intent")!r}, '
        f'confidence={entry.get("confidence")}, source={entry.get("source")!r}\n'
        f'  Buddy replied: "{entry.get("response", "")}"\n'
    )

    parts.append("Audit this turn now. Respond with only the JSON object.")
    return "\n".join(parts)


def _chat_ollama(model: str, system: str, user: str) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
        format="json",
    )
    return (resp.get("message", {}) or {}).get("content", "").strip()


def _chat_openai(model: str, system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


_BACKENDS = {
    "ollama": _chat_ollama,
    "openai": _chat_openai,
}


def _call_once(
    entry: dict, prev_entry: dict | None, backend: str, model: str, retry_note: str = "",
) -> dict | None:
    user_msg = build_user_message(entry, prev_entry)
    if retry_note:
        user_msg += f"\n\n{retry_note}"

    try:
        raw = _BACKENDS[backend](model, SYSTEM_PROMPT, user_msg)
    except Exception as e:
        print(f"  [ERROR] model call failed: {e}")
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw)
        if not m:
            print(f"  [ERROR] unparseable output: {raw[:200]!r}")
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            print(f"  [ERROR] unparseable JSON block: {raw[:200]!r}")
            return None

    corrected_intent = parsed.get("corrected_intent")
    if corrected_intent is not None and corrected_intent not in ALLOWED_INTENTS:
        print(f"  [WARN] model returned unknown intent {corrected_intent!r} — dropping")
        corrected_intent = None

    issue = parsed.get("issue")
    note = parsed.get("note")
    if issue is not None and issue not in ISSUE_CATEGORIES:
        print(f"  [WARN] model returned unknown issue {issue!r} — keeping as note only")
        note = f"{note or ''} (raw issue: {issue})".strip()
        issue = None

    flagged = bool(parsed.get("flagged", False))
    corrected_response = parsed.get("corrected_response")

    # Normalize: an unflagged turn shouldn't carry correction fields.
    if not flagged:
        corrected_intent, issue = None, None

    # Guard: "wrong_intent" where corrected_intent == the original intent is a
    # contradiction (nothing about the intent actually needs correcting — this
    # is the source="unknown"/intent="unknown" confusion). If there's a real
    # corrected_response alongside it, that's actually a response-text issue,
    # not an intent issue; otherwise there's no bug here at all.
    if flagged and corrected_intent is not None and corrected_intent == entry.get("intent"):
        if issue == "wrong_intent":
            if corrected_response:
                issue = "wrong_response_text"
            else:
                flagged, corrected_intent, issue, note = False, None, None, None

    # An unflagged turn shouldn't carry a note either — a stray note on a
    # "nothing wrong here" entry reads as unresolved commentary later.
    if not flagged:
        note = None

    return {
        "flagged": flagged,
        "corrected_intent": corrected_intent,
        "issue": issue,
        "corrected_response": corrected_response,
        "note": note,
    }


def call_model(entry: dict, prev_entry: dict | None, backend: str, model: str) -> dict | None:
    """
    Calls the model, and if it returns the self-contradictory flagged=true
    with corrected_intent=null, retries once with an explicit nudge. If it's
    still invalid after the retry, returns None rather than writing a broken
    label — the caller treats that as "needs manual review", not a labeled turn.
    """
    result = _call_once(entry, prev_entry, backend, model)
    if result is None:
        return None

    if result["flagged"] and result["corrected_intent"] is None:
        print("  [RETRY] flagged=true but corrected_intent=null — retrying with a nudge")
        result = _call_once(
            entry, prev_entry, backend, model,
            retry_note=(
                "Your previous answer set flagged=true but corrected_intent=null — that's "
                "invalid. If you're confident something is wrong, you MUST name the correct "
                "intent from the allowed list. If you can't, set flagged=false instead."
            ),
        )
        if result is not None and result["flagged"] and result["corrected_intent"] is None:
            print("  [SKIP] still invalid after retry — leaving unlabeled for manual review")
            return None

    return result


_SYNTHETIC_INPUT_RE = re.compile(r"^__[A-Z0-9_]+__$")


def is_eligible(entry: dict, max_confidence: float | None, relabel: bool) -> bool:
    # Test-harness scaffolding (e.g. "__SIMULATE_IDLE_65_SECONDS__") isn't real
    # user speech — never label it, don't waste a model call on it.
    if _SYNTHETIC_INPUT_RE.match((entry.get("input") or "").strip()):
        return False

    review = entry.get("review") or {}
    already_labeled = review.get("corrected_intent") is not None or review.get("flagged") is True
    if already_labeled and not relabel:
        return False
    if max_confidence is None:
        return True
    conf = entry.get("confidence")
    return entry.get("intent") == "unknown" or (conf is not None and conf <= max_confidence)


def find_inconsistent_pending_flows(data: list[dict]) -> list[tuple[dict, dict]]:
    """
    Flags pairs of turns in the same session, sharing the same non-null
    pending_before (i.e. the flow never resolved between them — a strong
    signal they're answering the same open question), that were both labeled
    but with DIFFERENT corrected_intent. These are almost certainly the model
    contradicting itself on near-duplicate content and should be manually
    reviewed rather than trusted as-is.
    """
    by_session: dict[str, list[dict]] = {}
    for e in data:
        by_session.setdefault(e.get("session_id"), []).append(e)
    for turns in by_session.values():
        turns.sort(key=lambda e: e.get("turn", 0))

    pairs = []
    for turns in by_session.values():
        for i in range(len(turns)):
            a = turns[i]
            a_pending = (a.get("context") or {}).get("pending_before")
            a_ci = (a.get("review") or {}).get("corrected_intent")
            if not a_pending or not a_ci:
                continue
            for j in range(i + 1, len(turns)):
                b = turns[j]
                b_pending = (b.get("context") or {}).get("pending_before")
                if b_pending != a_pending:
                    break  # flow moved on — stop looking further in this session
                # pending_before can stay stuck across many turns of genuinely
                # different content (a separate, real bug in Buddy's own state
                # tracking — not a labeling issue). Only treat two turns as a
                # contradiction if they're actually near-duplicate INPUT TEXT,
                # not just turns that happen to share a stale pending_before.
                if not _texts_similar(a.get("input", ""), b.get("input", "")):
                    continue
                b_ci = (b.get("review") or {}).get("corrected_intent")
                if b_ci and b_ci != a_ci:
                    pairs.append((a, b))
    return pairs


def _texts_similar(a: str, b: str, threshold: float = 0.5) -> bool:
    """Jaccard similarity over lowercased word tokens — cheap near-duplicate check."""
    wa = set(re.findall(r"[a-z0-9']+", a.lower()))
    wb = set(re.findall(r"[a-z0-9']+", b.lower()))
    if not wa or not wb:
        return False
    overlap = len(wa & wb) / len(wa | wb)
    return overlap >= threshold


def report_inconsistencies(data: list[dict]) -> None:
    pairs = find_inconsistent_pending_flows(data)
    if not pairs:
        print("[CONSISTENCY] no contradictory pairs found.")
        return
    print(f"[CONSISTENCY] {len(pairs)} contradictory pair(s) — same session, same "
          f"unresolved pending_before, different corrected_intent:")
    for a, b in pairs:
        pending = (a.get("context") or {}).get("pending_before")
        print(f"  session={a['session_id']} pending_before={pending!r}")
        print(f"    turn {a['turn']} ({a['id']}): {a['input']!r} -> {a['review']['corrected_intent']!r}")
        print(f"    turn {b['turn']} ({b['id']}): {b['input']!r} -> {b['review']['corrected_intent']!r}")


def build_prev_index(data: list[dict]) -> dict[str, dict]:
    """Maps entry id -> previous entry in the same session (by turn order)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="Path to merged_sessions.json")
    ap.add_argument("--out", dest="out_path", required=True, help="Output path (can equal --in to overwrite)")
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--model", default=None, help=f"Defaults per backend: {DEFAULT_MODELS}")
    ap.add_argument("--limit", type=int, default=None, help="Max number of turns to label this run")
    ap.add_argument("--max-confidence", type=float, default=None,
                     help="Only label turns with confidence <= this (plus all 'unknown'-intent turns). Omit to label everything eligible.")
    ap.add_argument("--sample", type=int, default=None, help="Randomly sample this many eligible turns instead of taking them in order")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relabel", action="store_true", help="Reprocess turns that already have a review label")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--check-only", action="store_true",
                     help="Skip labeling entirely — just load --in and report contradictory pending-flow labels")
    args = ap.parse_args()

    if args.check_only:
        with open(args.in_path, encoding="utf-8") as f:
            data = json.load(f)
        report_inconsistencies(data)
        return

    model = args.model or DEFAULT_MODELS[args.backend]

    if args.backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] --backend openai requires the OPENAI_API_KEY environment variable to be set.")
        sys.exit(1)

    print(f"[CONFIG] backend={args.backend} model={model}")

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    print(f"[LOAD] {in_path}")
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[LOAD] {len(data)} total turns")

    prev_of = build_prev_index(data)

    eligible = [e for e in data if is_eligible(e, args.max_confidence, args.relabel)]
    print(f"[FILTER] {len(eligible)} turns eligible for labeling "
          f"(max_confidence={args.max_confidence}, relabel={args.relabel})")

    if args.sample:
        random.seed(args.seed)
        eligible = random.sample(eligible, min(args.sample, len(eligible)))
        print(f"[SAMPLE] labeling {len(eligible)} randomly sampled turns (seed={args.seed})")

    if args.limit:
        eligible = eligible[: args.limit]

    if not eligible:
        print("[DONE] nothing to label.")
        return

    id_to_entry = {e["id"]: e for e in data}

    flagged_count = 0
    issue_counts: dict[str, int] = {}
    error_count = 0
    start = time.time()

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    try:
        for i, entry in enumerate(eligible, 1):
            prev_entry = prev_of.get(entry["id"])
            print(f"[{i}/{len(eligible)}] {entry['id']} | {entry.get('input', '')[:60]!r}")

            result = call_model(entry, prev_entry, args.backend, model)
            if result is None:
                error_count += 1
                continue

            id_to_entry[entry["id"]]["review"] = result

            if result["flagged"]:
                flagged_count += 1
                issue_counts[result["issue"]] = issue_counts.get(result["issue"], 0) + 1
                print(f"  -> FLAGGED: {entry.get('intent')} -> {result['corrected_intent']} "
                      f"[{result['issue']}] {result['note']}")
            else:
                print(f"  -> ok ({entry.get('intent')})")

            if i % args.checkpoint_every == 0:
                save()
                print(f"  [checkpoint] saved progress to {out_path}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] saving progress before exit...")

    save()
    elapsed = time.time() - start
    print(f"\n[DONE] wrote {out_path}")
    print(f"  processed: {len(eligible) - error_count}/{len(eligible)}  errors: {error_count}")
    print(f"  flagged:   {flagged_count} ({flagged_count / max(1, len(eligible) - error_count) * 100:.1f}%)")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"    {issue}: {count}")
    print(f"  elapsed: {elapsed / 60:.1f} min")

    print()
    report_inconsistencies(data)


if __name__ == "__main__":
    sys.exit(main())
