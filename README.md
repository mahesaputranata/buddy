# Buddy

A voice-first personal assistant that runs on a **local 3B model**, built as a system you can measure: an ear (VAD, speaker
verification, STT), a layered decision pipeline (interrupts, confirmations, intent routing, memory, belief state), and a mouth
(TTS with music ducking). It delegates long tasks to an agent (Friday) and has clients for a rover and a Tello drone.

<!-- TODO: demo video link here -->

The interesting part is not that it talks, it is that **every failure I fixed was found by a harness I built to catch it**.

| | |
|---|---|
| Evaluation | 11 scenario suites, a runner with context-bleed detection, `--isolate`, and prompt tracing |
| First run of the 19-scenario core suite | **8/19 (42%)** |
| After the fixes (continuous runs) | **14-16/19 (74-84%)**, about one scenario of run-to-run noise |
| Honest caveat | author-written keyword checks, 3B sampling noise, rover offline; see [EVALUATION.md](docs/EVALUATION.md) |

## Start here

- [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md): seven problems, how they were found, and the dead ends
- [docs/EVALUATION.md](docs/EVALUATION.md): how the harness works and what the numbers do and do not show
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): ear -> brain -> mouth, with a layer diagram
- [docs/SPEAKER_RECOGNITION.md](docs/SPEAKER_RECOGNITION.md): VAD + ECAPA speaker verification lab

## What is in the code

- **Layered request handling** (`brain/conversation_manager.py`): interrupt -> pending-state FSM -> intent routing -> memory-task
  fusion -> RAG prompt. Deterministic code decides before the LLM does.
- **Belief state, episodic memory router, memory-task fusion** (`brain/belief_state.py`, `brain/memory/`).
- **Delegation to Friday** (`brain/friday/`): every task carries an instruction not to read or report secrets. A prompt-level
  constraint, not a sandbox.
- **Intent-classifier pipeline** (`tools/`, `brain/intent_model/`): LLM-assisted labelling with consistency checks, leakage-free
  input formatting, class-weighted DistilBERT fine-tuning, shadow-mode serving.
- **Voice front-end** (`ear/`, `mouth/`, `speaker_recognition/`): adaptive-noise-floor VAD with pre-roll and end-of-turn grace,
  self-mute while speaking, speaker verification in an isolated subprocess (JSON-lines protocol), Spotify volume ducking.
- **Devices** (`devices/`): rover websocket client with a mock server for hardware-free tests, Tello client.

## Quick start (text mode, no hardware)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.2                                      # Ollama must be running

python run_eval.py unit        # offline checks, no LLM
python text_chat.py            # talk to it in the terminal
python run_eval.py alive       # the 19-scenario conversational suite
python run_eval.py results     # committed results
```

Machine-specific values (device paths, default city) go in `config.example.yaml` -> `config.yaml`; secrets only in environment
variables. Nothing personal is read from the repo. The voice stack needs `requirements-voice.txt` plus a microphone.

## What's real, what's still rough

**Real**
- The pipeline runs end to end on my machine (voice in/out, device clients); the evaluation results are from real runs.
- The harness, the guards and the offline tests are in this repo and runnable.

**Rough (known)**
- `brain/conversation_manager.py` is ~8.5k lines that grew organically. Splitting it into handlers is the first refactor I would do.
- Some modules still contain commented-out experiments and mixed-encoding comments.
- The text mode and `alive` suite were exercised on my Windows machine; they have not been verified on a clean Linux/macOS install.
- The context guards are justified by a prompt trace, not yet by a controlled A/B.
- Speaker verification has no published error rates and no anti-spoofing.
- The intent-model weights, training data (real conversations) and metrics are not included.
- A 3B model hallucinates. The harness reduces the visible failure modes; it does not remove the cause.

## License

MIT, see [LICENSE](LICENSE).
