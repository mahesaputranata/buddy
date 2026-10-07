# Evaluation

Buddy is evaluated with a conversational scenario runner, not with unit tests alone. A 3B local model is non-deterministic,
so the point of the harness is to make failures *visible and repeatable enough to fix*, and to be honest about how much a
single number can be trusted.

## How it works

`buddy_alive_test_runner.py` plays multi-turn scenarios (`scenarios/*.json`) through `ConversationManager.handle_request()`
with real web/camera/robot calls switched off, then scores each turn:

- **Behaviour checks per turn**: `must_contain_one_of`, `must_not_contain`, expected intent/route, latency, tone markers.
- **Alive score** (0-1) per turn and scenario; **pass** = no fatal check failed.
- **Context-bleed detection**: terms that belong to an *earlier* scenario (derived from the scenario inputs, plus terms learned
  from earlier replies) showing up in a later answer are flagged. This catches the failure where Buddy answers a new question
  with an old topic.
- **`--isolate`**: wipes all session memory between scenarios. Continuous mode (default) is how Buddy is really used and is what
  exposes bleed; isolated mode separates "this scenario is broken" from "an earlier topic leaked into it".
- **`--trace-prompts`**: records the exact prompts sent to the LLM each turn and says which prompt section a leaked term came from.

```bash
python run_eval.py unit                       # offline checks, no LLM
python run_eval.py alive --output out.json    # needs Ollama + llama3.2
python run_eval.py alive --isolate
python run_eval.py alive --trace-prompts
python run_eval.py results                    # table below, from results/stress_runs.json
```

## Results (19-scenario core suite)

| run | date | mode | passed | alive | bleed turns | note |
|---|---|---|---|---|---|---|
| new | 2026-10-04 | continuous | 8/19 (42%) | 0.83 | - | first run of this series: before the interrupt-gate, confirmation and save-target fixes, with the (later reverted) clarifier active |
| new2 | 2026-10-04 | continuous | 15/19 (79%) | 0.93 | - | intermediate fixes (interrupt gate, clarifier revert, confirmation handling) |
| new3 | 2026-10-04 | continuous | 15/19 (79%) | 0.91 | - | intermediate fixes (same series as new2) |
| new4 | 2026-10-04 | continuous | 15/19 (79%) | 0.93 | - | intermediate fixes (same series as new2) |
| new5 | 2026-10-04 | continuous | 15/19 (79%) | 0.96 | 1 | after the save-target fix; first run with context-bleed detection |
| isolated | 2026-10-04 | isolated | 16/19 (84%) | 0.96 | 1 | --isolate: memory wiped between scenarios (diagnostic, not comparable to continuous runs) |
| traced | 2026-10-04 | continuous | 14/19 (74%) | 0.89 | 3 | --trace-prompts on; used to find the context-bleed mechanism |
| new6_attempt1 | 2026-10-04 | continuous | 16/19 (84%) | 0.94 | 1 | after the four context guards were added (flag state not recorded in the report) |
| new6_attempt2 | 2026-10-04 | continuous | 16/19 (84%) | 0.94 | 0 | same, repeat run |
| new6_attempt3 | 2026-10-04 | continuous | 15/19 (79%) | 0.95 | 1 | same, repeat run |

Raw numbers: [`results/stress_runs.json`](../results/stress_runs.json) (per-run failed scenarios and category scores).

**What these numbers do and do not show**

- The first run passed **8/19 (42%)**. After the fixes described in the [engineering log](ENGINEERING_LOG.md), runs of the same
  suite pass **14-16/19 (74-84%)**. That improvement is real: it is consistent over 8 later continuous runs.
- Between runs with the same code the pass count still moves by about one scenario (e.g. 16 -> 16 -> 15 on the last three
  runs). Treat differences of one scenario as noise.
- The three runs after the context guards were added (84%, 84%, 79%) are **not** evidence that the guards work: there is no
  guards-off control set, and the difference to the previous four runs (79% each) is within the noise above. What the guards do
  have is a mechanism: the prompt trace showed the bleed, and the guards remove its sources. The runner now records the guard
  flag in the report so a proper A/B (`BUDDY_CONTEXT_GUARDS=0` vs default, several runs each) can be done.
- The scenarios and checks were written by the same person who wrote the system; keyword checks are shallow. This is a
  regression harness, not a benchmark.
- The rover hardware was offline, so rover-control outcomes are not meaningful.

Scenarios that did not pass in every continuous run:

| scenario | passed / runs |
|---|---|
| `mid_conversation_save` | 0/9 |
| `delete_memory_flow` | 0/9 |
| `rover_explain_repair` | 3/9 |
| `physical_troubleshoot_vision` | 5/9 |
| `vision_suppression_abstract` | 6/9 |
| `mid_conversation_pivot_then_return` | 7/9 |
| `frustration_empathy` | 7/9 |
| `session_seed_weather_followup` | 8/9 |
| `schedule_pivot_rover` | 8/9 |
| `memory_save_recall_flow` | 8/9 |
| `mid_conversation_save_then_recall` | 8/9 |
| `implicit_continuation` | 8/9 |
| `self_correction_cancel` | 8/9 |
| `technical_deep_dive` | 8/9 |
