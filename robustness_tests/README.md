# Buddy Robustness Testing Framework

A 3-phase industry-standard testing framework for your AI assistant.

---

## Structure

```
robustness_tests/
├── test_cases/
│   ├── single_turn.jsonl       ← 50 manual single-turn intent tests
│   ├── multi_turn.jsonl        ← 20 multi-turn flow scenarios
│   └── adversarial.jsonl       ← LLM-generated (auto-created on first run)
│
├── runner/
│   ├── unit_runner.py          ← Phase 1: white-box classify_intent() tests
│   ├── flow_runner.py          ← Phase 2: black-box handle_request() flow tests
│   └── adversarial_generator.py← Phase 3: LLM expands seeds into edge cases
│
├── results/                    ← All JSON result files saved here
│   ├── unit_YYYYMMDD_HHMMSS.json
│   ├── flow_YYYYMMDD_HHMMSS.json
│   └── combined_report_YYYYMMDD_HHMMSS.json
│
└── run_all.py                  ← Master orchestrator
```

---

## Quick Start

Run all three phases (from your project root):

```bash
python robustness_tests/run_all.py
```

Run only specific phases:

```bash
python robustness_tests/run_all.py --phase unit
python robustness_tests/run_all.py --phase unit flow
python robustness_tests/run_all.py --phase unit flow adversarial
```

Run faster (skip LLM adversarial generation):

```bash
python robustness_tests/run_all.py --skip-adversarial
```

---

## Phase 1 — Unit Tests (White-box)

Tests `classify_intent()` + `apply_guards()` directly on single utterances.

```bash
python -m robustness_tests.runner.unit_runner
python -m robustness_tests.runner.unit_runner --verbose
python -m robustness_tests.runner.unit_runner --tags schedule
python -m robustness_tests.runner.unit_runner --filter-id st_001,st_010
```

**Output:**
- Per-intent Precision / Recall / F1
- Confusion matrix (which intents are being confused with each other)
- Average classification latency

---

## Phase 2 — Flow Tests (Black-box)

Runs complete multi-turn scenarios through `handle_request()`. Tracks:
- Intent accuracy per turn
- Pending state transitions (did the FSM move correctly?)
- Flow completion (did the scenario reach its terminal state?)
- Stuck detection (pending state never cleared = bug)

```bash
python -m robustness_tests.runner.flow_runner
python -m robustness_tests.runner.flow_runner --verbose
python -m robustness_tests.runner.flow_runner --tags adversarial
python -m robustness_tests.runner.flow_runner --id mt_001,mt_010
```

**Flow types tested:**
| Flow Type | What It Tests |
|---|---|
| `schedule_save_completion` | Full schedule save with missing fields |
| `mid_flow_interrupt_soft` | Soft query (weather) during active flow |
| `mid_flow_cancel` | nevermind/cancel during active flow |
| `save_memory_full` | Full save → topic → content → confirm |
| `recall_memory_general_to_specific` | General → pick topic → specific |
| `delete_memory_confirm` | Delete with confirmation |
| `web_search_to_followup` | Search → open result → summarize |
| `hard_intent_switch` | Complete intent pivot mid-pending |
| `three_way_switch` | Three different intents in three turns |
| `stuck_state_recovery` | Repeated unclear answers → stuck detection |

---

## Phase 3 — Adversarial Generation

Uses your local Ollama LLM to expand manual seeds into adversarial variations.

```bash
python -m robustness_tests.runner.adversarial_generator
python -m robustness_tests.runner.adversarial_generator --intent schedule_save
python -m robustness_tests.runner.adversarial_generator --n 3
python -m robustness_tests.runner.adversarial_generator --dry-run
```

**Strategies applied per seed:**

| Strategy | What It Does |
|---|---|
| `rephrase` | Same intent, different words |
| `add_noise` | Filler words, real-user mumbling |
| `ambiguate` | Adds words that could point to wrong intent |
| `engineering_inject` | Adds "pid", "rover", "lidar" etc — tests domain override |
| `broken_grammar` | Simulates bad STT output |
| `over_polite` | "could you please...", "i was wondering..." |
| `implicit` | Removes explicit keywords — intent must be inferred |

---

## Reading the Weak-Spot Report

```
WEAK INTENTS (F1 < 0.80):
   ❌  delete_memory          F1=0.58  P=0.61  R=0.55
   ⚠️  recall_memory          F1=0.70  P=0.73  R=0.68

TOP CONFUSIONS:
   delete_memory          → chat                   23x
   recall_memory          → schedule_recall         18x

RECOMMENDED ACTIONS:
   🔴  [delete_memory] F1=0.58 — CRITICAL. Add more strong rules or review EVIDENCE list.
   🟡  [recall_memory] F1=0.70 — Review ambiguous cases. Check post_validate_intent.
```

**How to act on the results:**
- **F1 < 0.60 (🔴)**: Add explicit strong rules in `STRONG_RULES` or `DELETE_STRONG`
- **F1 0.60–0.79 (⚠️)**: Review `post_validate_intent` and `_evidence_score_classify` for that intent
- **Confusion A → B**: The patterns for B are too broad — add guards in `apply_guards()`
- **Stuck flows**: Check `handle_pending()` — a terminal condition is missing
- **Low adversarial but high unit**: Your rules are too narrow — intent breaks on rephrasing

---

## Adding Your Own Test Cases

**Single-turn** (`test_cases/single_turn.jsonl`):
```json
{"id":"st_051","input":"your utterance here","expected_intent":"schedule_save","tags":["schedule","your_tag"]}
```

**Multi-turn** (`test_cases/multi_turn.jsonl`):
```json
{
  "id": "mt_021",
  "description": "What this tests",
  "flow_type": "your_flow_type",
  "tags": ["your_tags"],
  "turns": [
    {
      "input": "first utterance",
      "expected_intent": "schedule_save",
      "expected_pending_after": "Awaiting_Schedule_Details",
      "note": "optional explanation"
    },
    {
      "input": "second utterance",
      "expected_intent": "schedule_save",
      "expected_pending_after": null
    }
  ]
}
```

**Available intents:**
`schedule_save`, `schedule_recall`, `save_memory`, `recall_memory`, `delete_memory`,
`web_search`, `web_followup`, `device_control`, `system_query`, `chat`

**Available pending states (from your codebase):**
`Awaiting_Schedule_Details`, `Awaiting_Topic`, `Awaiting_Topic_Content`,
`Awaiting_Delete_Confirm`, `Awaiting_Delete_TopicName`, `Awaiting_General_Recall`,
`Awaiting_Temporal_Recall`, `Awaiting_ConversationSave_Confirm`,
`Awaiting_ConversationSave_TopicPick`, `Awaiting_Vision_Source_Confirm`, etc.

---

## Using Your Existing Uncertain Intent Logs

Your system already writes to `logs/intent_uncertain.jsonl`. Convert real
uncertain cases into test cases with:

```python
import json

with open("logs/intent_uncertain.jsonl") as f:
    for i, line in enumerate(f, 51):  # start IDs after st_050
        entry = json.loads(line)
        case = {
            "id": f"st_{i:03d}",
            "input": entry["input"],
            "expected_intent": entry["intent"],  # fill this in manually
            "tags": ["from_log", entry["source"]]
        }
        print(json.dumps(case))
```

This turns your production failure logs directly into test cases — exactly
what industry teams do to build regression test suites.
