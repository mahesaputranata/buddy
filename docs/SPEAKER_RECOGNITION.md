# Speaker recognition lab

Goal: Buddy should react to its owner's voice only. The lab (`speaker_recognition/`) is where the signal path was prototyped
before parts of it were moved into the main app (`ear/push_to_talk.py`, `ear/voice_pipeline.py`).

## Pipeline

```
mic (16 kHz mono, 20 ms frames)
  -> VAD: RMS energy in dB vs. (noise floor + margin)
  -> segmenter: start/end frame counts, 200 ms pre-roll, min/max utterance length, end-of-turn silence, cooldown
  -> worker thread (never run recognition inside the audio callback)
       -> ECAPA-TDNN embedding (SpeechBrain, VoxCeleb-pretrained)
       -> cosine similarity to the enrolled speaker's centroid
       -> ACCEPT / REJECT at a threshold
  -> only if ACCEPT: Whisper STT -> transcript
```

- **VAD** (`realtime_vad_sr.py`, and the production version `VADRecorder` in `ear/push_to_talk.py`): the noise floor is an
  exponential moving average of non-speech frame energy (`0.9 * floor + 0.1 * e`), clamped; speech is `e > floor + margin`.
  Pre-roll keeps the first syllable that VAD would otherwise clip. The recorder mutes itself while Buddy speaks and applies a
  cooldown after unmuting so it does not transcribe Buddy's own voice.
- **Embeddings and enrolment**: `ExtractingEmbeddings/` (extractor wrapper, centroid builder). `Similarities/` holds the
  comparison functions (cosine, Euclidean).
- **Subprocess protocol** (`speaker_worker.py`, client in `ear/voice_pipeline.py`): the speaker model runs in its own Python
  process (separate virtualenv, heavy dependencies) and talks JSON lines over stdin/stdout:
  - worker prints `{"status": "ready"}` once the model is loaded
  - request: `{"wav_path": "<file>"}` -> response: `{"ok": bool, "score": float, ...}` or `{"ok": false, "error": "..."}`
  - `__quit__` shuts the worker down
  This keeps the heavy model stack out of the main process and makes a crash in it non-fatal.
- `DESIGN_NOTES.tex.txt` is the original design recap written while building the real-time version.

## What is not here, and what is not claimed

- No audio, enrolment recordings, impostor data, centroids or model weights are included (biometric / licensed data).
  The model downloads from the SpeechBrain hub on first run.
- The accept threshold (0.2 in `verify_wav.py` and the real-time script, 0.4 in the one-shot gate script) was set by hand for
  one speaker in one room. There is **no threshold-selection experiment in this repo** (the lab's `Thresholding/` folder was an
  empty stub and is left out), **no equal-error-rate or false-accept figures**, no anti-spoofing (replay / synthetic voice), and
  enrolment is single-speaker. Treat it as a convenience gate, not a security boundary. Measuring EER on held-out impostor data
  is the obvious next step.

To reproduce: record enrolment audio, set `BUDDY_SPEAKER_DATA_DIR` / `BUDDY_SPEAKER_PROFILE_DIR` (see `speaker_recognition/paths.py`),
run `ExtractingEmbeddings/build_speaker_profile.py`, then `realtime_vad_sr.py`.
