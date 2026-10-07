# Engineering log

Short case studies of problems found by running the system and measuring it. Each one is: symptom -> evidence -> change ->
result, including where the first idea was wrong.

## 1. Confirmations answered by the LLM instead of the state machine
**Symptom.** Short replies like "yes" could bypass the pending-state logic, so a confirmed action did not run.
**Change.** Every confirm flow now goes through `handle_confirm_state` + `resolve_confirmation`
(CONFIRM / DECLINE / REDIRECT / UNCLEAR); the project-notes flow, which had its own ad-hoc yes/no check, was moved onto it.
**Status: improved, not closed.** `mid_conversation_save` (the final "yes") and `delete_memory_flow` fail in **every** recorded
continuous run (0/9, see [EVALUATION.md](EVALUATION.md)). The delete flow asks for confirmation *before* checking that the topic
exists; fixing that order is the next item.

## 2. "Never mind" cancelling flows it should not
**Symptom.** The phrase "never mind" (and similar) anywhere in a sentence cancelled the active flow.
**Change.** Boundary-aware detection: a cancel phrase only counts at the start of a clause (after a pause, dash, comma, period,
or a lead-in word), keeping the device-stop and length guards.
**Result.** 13/13 hand-written unit cases; replayed over the real interaction log it fires on fewer turns than the substring
rule (132 -> 124), i.e. it removed false triggers without needing a new model.

## 3. A clarifying question that made things worse (reverted)
**Idea.** When the intent classifier is unsure, ask a clarifying question instead of guessing.
**Measurement.** In the log, the fallback `chat` guess was right 370/477 times (78%); the clarifier turned correct guesses into
extra questions, and the stress suite dropped to 8/19. **Reverted**, with a comment explaining why so it is not re-added by accident.

## 4. "Save this" hijacked by a robot that had nothing to save
**Symptom.** With the rover idle, "save this" was routed to rover telemetry saving.
**Change.** `score_save_target` only prefers the rover if telemetry exists; otherwise the conversation is the save target.
Covered by 8 AST-extraction unit tests (`eval/test_save_target.py`), which test the real function without importing the stack.

## 5. A trustworthy dataset for the intent classifier
The goal was to replace a regex cascade with a small fine-tuned classifier (DistilBERT), run first in **shadow mode**
(predictions logged next to the live decision, never acted on).
- **Labelling.** An LLM relabelled the logged turns; its output was checked: contradictions between near-identical inputs were
  flagged (the first check reported 820 false contradictions, all from a stuck `pending_before` field, fixed with a text-similarity
  gate), and human overrides live in `label_overrides.json`.
- **Train/serve skew.** The logger wrote `last_topic` *after* the turn, so training used a feature that would not exist at
  inference. Fixed by using the previous turn's topic and sharing one input formatter between training and serving.
- **Splits and selection.** Session-level splits with label-coverage guarantees, class-weighted loss, macro-F1 for model selection,
  labels with <10 training rows excluded, augmentation only for sparse labels.
- **Not published here:** the data (real conversations), model weights, and a metrics summary. The scripts are in `tools/`.

## 6. Context bleed: answering a new question with an old topic
**Symptom.** A question about reinforcement vs. imitation learning was answered about Nav2/MPPI.
**First (wrong) belief.** My crude detector reported 2-6 bleed turns per run as a standing weakness; a stricter detector showed
0-1 in an earlier run and 4-5 now, so the earlier claim was wrong and was corrected before any fix was built.
**Method.** Add bleed detection to the harness, then `--trace-prompts` to capture every prompt.
**Finding.** Four independent sources, all visible in the traced prompts:
1. a *global* recent-exchange buffer (not tied to any topic) injected into every answer prompt;
2. a stale belief-state block (old entities, old `Thread:` line, an expired tool result);
3. topic memory poisoned by storing a bare "yes" and the model's improvised reply to it;
4. a template instruction telling the 3B model to "connect the ideas smoothly".
**Dead end.** "The relevance threshold (cosine 0.32) is too low" was falsified by replaying the traced turns: foreign buffers
scored up to 0.43 because every scenario is about robotics, so no cosine threshold separates them; the gate is therefore a
follow-up rule first and a (0.45) similarity check second.
**Change.** Four guards behind one switch (`BUDDY_CONTEXT_GUARDS=0` disables): relevance-gated buffer, belief-state reset on
topic change, bare acknowledgements not stored, reworded template. 56 offline tests in `eval/test_context_guards.py`.
**Result, honestly.** Suite pass rate before: 79% (4 runs); after: 84%, 84%, 79%. Within noise; no guards-off control. See
[EVALUATION.md](EVALUATION.md). The mechanism is proven by the trace; the benefit is not yet measured.

## 7. Smaller things
- Removed a dead zero-shot model that was still loaded at startup.
- Keys that had been hardcoded were moved to environment variables, and the news-provider code that passed the key itself to
  `os.getenv` (so it never worked) now passes the variable name.
