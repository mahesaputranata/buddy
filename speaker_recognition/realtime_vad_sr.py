from __future__ import annotations

from pathlib import Path
import datetime
import csv
import time
import threading
import queue
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf

from ExtractingEmbeddings.types.ecapa_extractor import ECAPAEmbeddingExtractor
from Similarities.similarities import Similarities
from faster_whisper import WhisperModel


# -------------------------
# CONFIG
# -------------------------
SR = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)

# Debug flags (toggle these)
DBG_VAD_EVENTS = True        # start/end/finalize/cooldown
DBG_VAD_LEVELS = True        # periodic energy/noise/threshold print
DBG_QUEUE = True             # queue size / drops
DBG_WORKER_TIMINGS = True    # SR + STT timing

DBG_LEVEL_EVERY_N_FRAMES = 25  # print levels about every 25 frames (~0.5s)

COOLDOWN_MS = 200
COOLDOWN_FRAMES = int(COOLDOWN_MS / FRAME_MS)

# VAD parameters
# MARGIN_DB = 3.0
START_FRAMES = 3
END_FRAMES = 6
# NOISE_CLAMP_DB = -80.0
NOISE_CLAMP_DB = -65.00
MARGIN_DB = 6.0
IGNORE_BELOW_DB = -90.0
EPS = 1e-12

# Segmenting
PRE_ROLL_MS = 200
MIN_UTTERANCE_MS = 800
MAX_UTTERANCE_MS = 12000

# Speaker recognition
THRESHOLD = 0.2

from paths import PROFILE_DIR
LOG_DIR = Path("mic_logs")
AUDIO_LOG_DIR = LOG_DIR / "audio"
LOG_DIR.mkdir(exist_ok=True)
AUDIO_LOG_DIR.mkdir(parents=True, exist_ok=True)

END_OF_TURN_SILENCE_SEC = 1.5
END_OF_TURN_SILENCE_FRAMES = int((END_OF_TURN_SILENCE_SEC * 1000) / FRAME_MS)

# -------------------------
# STT
# -------------------------
class WhisperSTT:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_wav(self, wav_path: str) -> str:
        segments, info = self.model.transcribe(wav_path, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments).strip()


# -------------------------
# UTIL
# -------------------------
def rms_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x * x) + EPS)
    return 20.0 * np.log10(rms + EPS)

def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def init_log_file() -> Path:
    ts = now_ts()
    log_path = LOG_DIR / f"mic_test_log_{ts}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "vad_ms", "score", "decision", "text", "audio_file"])
    return log_path

def append_log(log_path: Path, vad_ms: int, score: float, decision: str, text: str, audio_file: str):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now_ts(), vad_ms, f"{score:.6f}", decision, text, audio_file])


# -------------------------
# WORKER (SR + STT + log)
# -------------------------
def sr_worker(
    utt_q: "queue.Queue[tuple[np.ndarray, int]]",
    stop_evt: threading.Event,
    extractor: ECAPAEmbeddingExtractor,
    centroid: np.ndarray,
    log_path: Path,
):
    stt = WhisperSTT(model_size="small", device="cpu")

    while not stop_evt.is_set():
        try:
            audio, sr = utt_q.get(timeout=0.2)
        except queue.Empty:
            continue

        out_wav = None
        try:
            vad_ms = int(len(audio) / sr * 1000)

            # Save wav
            audio_ts = now_ts()
            out_wav = AUDIO_LOG_DIR / f"utt_{audio_ts}.wav"
            sf.write(out_wav, audio, sr)

            # SR timing
            t0 = time.time()
            emb = extractor.encode_file(out_wav)
            score = float(Similarities.comparison(emb, centroid, mode="cosine"))
            decision = "ACCEPT" if score >= THRESHOLD else "REJECT"
            t1 = time.time()

            # STT timing (only if ACCEPT)
            text = ""
            t2 = t1
            if decision == "ACCEPT":
                text = stt.transcribe_wav(str(out_wav))
                t2 = time.time()

            # Print summary
            if DBG_WORKER_TIMINGS:
                sr_ms = int((t1 - t0) * 1000)
                stt_ms = int((t2 - t1) * 1000) if decision == "ACCEPT" else 0
                print(f"\n[UTT] {vad_ms} ms | score={score:.4f} | {decision} | SR={sr_ms}ms | STT={stt_ms}ms | {out_wav}")
            else:
                print(f"\n[UTT] {vad_ms} ms | score={score:.4f} | {decision} | {out_wav}")

            if decision == "ACCEPT":
                print(f"[STT] {text}")

            # Log
            append_log(log_path, vad_ms, score, decision, text, str(out_wav))

        except Exception as e:
            print(f"\n[WORKER-ERROR] {repr(e)} (wav={out_wav})")
        finally:
            utt_q.task_done()


# -------------------------
# MAIN (stream + VAD)
# -------------------------
def main():
    print("[INIT] Loading centroid...")
    centroid = np.load(PROFILE_DIR / "speaker_centroid.npy")

    print("[INIT] Initializing ECAPA extractor...")
    extractor = ECAPAEmbeddingExtractor()  # keep OUTSIDE callback

    log_path = init_log_file()
    print(f"[LOG] Writing results to {log_path}")
    print("Press 'q' + Enter to quit.\n")

    utt_q: "queue.Queue[tuple[np.ndarray, int]]" = queue.Queue(maxsize=8)
    stop_evt = threading.Event()

    worker = threading.Thread(
        target=sr_worker,
        args=(utt_q, stop_evt, extractor, centroid, log_path),
        daemon=True,
    )
    worker.start()

    # Keyboard listener
    def keyboard_listener():
        while not stop_evt.is_set():
            key = sys.stdin.readline().strip()
            if key.lower() == "q":
                stop_evt.set()
                break

    threading.Thread(target=keyboard_listener, daemon=True).start()

    # VAD state
    in_speech = False
    post_speech = False          # we're in "waiting to finalize" mode
    speech_count = 0
    silence_count = 0            # counts consecutive non-speech frames (general)
    post_silence = 0             # counts silence frames since we *ended* speech
    noise_db = -60.0

    cooldown = 0
    frame_counter = 0

    # Buffers
    pre_roll_frames = max(1, int(PRE_ROLL_MS / FRAME_MS))
    pre_roll: list[np.ndarray] = []

    utt_frames: list[np.ndarray] = []
    utt_samples = 0

    min_utt_samples = int(SR * (MIN_UTTERANCE_MS / 1000.0))
    max_utt_samples = int(SR * (MAX_UTTERANCE_MS / 1000.0))

    def finalize_utterance(reason: str):
        nonlocal utt_frames, utt_samples, cooldown

        if utt_samples < min_utt_samples:
            if DBG_VAD_EVENTS:
                ms = int(utt_samples / SR * 1000)
                print(f"\n[VAD] finalize drop (too short) {ms}ms reason={reason}")
            utt_frames = []
            utt_samples = 0
            cooldown = COOLDOWN_FRAMES
            return

        audio = np.concatenate(utt_frames, axis=0).astype(np.float32)
        ms = int(len(audio) / SR * 1000)

        if DBG_VAD_EVENTS:
            print(f"\n[VAD] finalize OK {ms}ms reason={reason} (queue={utt_q.qsize()}/{utt_q.maxsize})")

        try:
            utt_q.put_nowait((audio, SR))
            if DBG_QUEUE:
                print(f"[QUEUE] pushed utterance, qsize={utt_q.qsize()}")
        except queue.Full:
            print("\n[WARN] Worker queue full; dropping utterance.")
        finally:
            utt_frames = []
            utt_samples = 0
            cooldown = COOLDOWN_FRAMES

    def audio_callback(indata, frames, time_info, status):
        nonlocal in_speech, speech_count, silence_count, noise_db
        nonlocal pre_roll, utt_frames, utt_samples
        nonlocal cooldown, frame_counter
        nonlocal in_speech, post_speech, speech_count, silence_count, post_silence, noise_db

        if stop_evt.is_set():
            raise sd.CallbackStop()

        # cooldown: ignore triggers briefly after an utterance ends
        if cooldown > 0:
            cooldown -= 1
            frame_counter += 1
            return

        x = indata[:, 0].astype(np.float32)

        # Update pre-roll always
        pre_roll.append(x.copy())
        if len(pre_roll) > pre_roll_frames:
            pre_roll.pop(0)

        e_db = rms_db(x)

        # Clamp noise floor
        noise_db = max(noise_db, NOISE_CLAMP_DB)
        threshold = noise_db + MARGIN_DB

        # Decide speech / silence (no early return; treat very low as silence)
        if e_db < IGNORE_BELOW_DB:
            is_speech = False
        else:
            is_speech = e_db > threshold

        # Update noise only when confident non-speech and not in speech
        if (not in_speech) and (not is_speech):
            # noise_db = 0.95 * noise_db + 0.05 * e_db
            noise_db = 0.90 * noise_db + 0.10 * e_db


        # Periodic levels debug
        if DBG_VAD_LEVELS and (frame_counter % DBG_LEVEL_EVERY_N_FRAMES == 0):
            print(f"[LVL] e={e_db:6.1f}dB  noise={noise_db:6.1f}dB  thr={threshold:6.1f}dB  speech={is_speech}  in_speech={in_speech}")

        frame_counter += 1

        # Counters
        if is_speech:
            speech_count += 1
            silence_count = 0
        else:
            silence_count += 1
            speech_count = 0

        # Counters
        if is_speech:
            speech_count += 1
            silence_count = 0
        else:
            silence_count += 1
            speech_count = 0

        # If we were waiting-to-finalize and speech returns, treat it as continuation
        if post_speech and is_speech:
            post_speech = False
            post_silence = 0
            in_speech = True

            # add a bit of pre-roll before continuing (helps not clip restarts)
            utt_frames.extend(pre_roll)
            utt_samples += sum(len(f) for f in pre_roll)

            if DBG_VAD_EVENTS:
                print("\n[VAD] RESUME (continuation)")

        # Start speech (fresh)
        if (not in_speech) and (not post_speech) and (speech_count >= START_FRAMES):
            in_speech = True
            utt_frames = list(pre_roll)  # seed with pre-roll
            utt_samples = sum(len(f) for f in utt_frames)
            if DBG_VAD_EVENTS:
                print(f"\n[VAD] START (seed={int(utt_samples/SR*1000)}ms)")

        # Collect while speaking
        if in_speech:
            utt_frames.append(x.copy())
            utt_samples += len(x)

            if utt_samples >= max_utt_samples:
                in_speech = False
                post_speech = False
                post_silence = 0
                finalize_utterance(reason="max_len")
                return

            # Detect end-of-speech, but DON'T finalize yet
            if silence_count >= END_FRAMES:
                in_speech = False
                post_speech = True
                post_silence = 0
                if DBG_VAD_EVENTS:
                    print("\n[VAD] END -> waiting for end-of-turn timeout")
                return

        # If we ended speech earlier, wait until "end-of-turn" silence is long enough
        if post_speech:
            if is_speech:
                # handled above (RESUME)
                pass
            else:
                post_silence += 1
                if post_silence >= END_OF_TURN_SILENCE_FRAMES:
                    post_speech = False
                    post_silence = 0
                    if DBG_VAD_EVENTS:
                        print(f"\n[VAD] FINALIZE after {END_OF_TURN_SILENCE_SEC:.1f}s silence")
                    finalize_utterance(reason="end_of_turn_timeout")
                    return

    # Start audio stream
    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=FRAME_SAMPLES,
        callback=audio_callback,
        dtype="float32",
    ):
        while not stop_evt.is_set():
            time.sleep(0.1)

    stop_evt.set()
    print("\n[STOP] Shutting down... waiting for worker queue to drain.")
    try:
        utt_q.join()
    except Exception:
        pass
    print("[DONE] Results saved:", log_path)


if __name__ == "__main__":
    main()
