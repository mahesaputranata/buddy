# flow_a.py

from pathlib import Path
import time

from BuddyEar_Architecture.helpers import (
    record_wav,
    ecapa_score,
    whisper_transcribe,
    DEFAULT_SR,
)

def flow_guard_first(
    SHORT_WIN,
    THRESHOLD,
    CMD_WIN,
    CENTROID,
    client,
    SR=DEFAULT_SR,
):
    print("\n=== FLOW A: Guard first (short ECAPA, then Whisper if accepted) ===")

    short_path = Path("temp_short.wav")

    # 1) Record short sample
    t0 = time.perf_counter()
    record_wav(SHORT_WIN, short_path, SR)
    t_rec_short_end = time.perf_counter()

    # 2) ECAPA on short segment
    t_ecapa_start = time.perf_counter()
    score = ecapa_score(short_path, CENTROID)
    t_ecapa_end = time.perf_counter()

    print(f"[ECAPA] score={score:.4f}")

    if score < THRESHOLD:
        total = t_ecapa_end - t0
        print(f"[DECISION] REJECT (not the enrolled speaker). Total latency ≈ {total:.3f}s")
        return {
            "decision": "REJECT",
            "score": score,
            "latency_total": total,
            "text": None,
        }

    print("[DECISION] ACCEPT (enrolled speaker). Recording full command...")

    # 3) Full command recording
    cmd_path = Path("temp_cmd.wav")
    t_cmd_rec_start = time.perf_counter()
    record_wav(CMD_WIN, cmd_path, SR)
    t_cmd_rec_end = time.perf_counter()

    # 4) Whisper (+ fallback)
    t_whisper_start = time.perf_counter()
    text = whisper_transcribe(cmd_path, client)
    t_whisper_end = time.perf_counter()

    total_latency = t_whisper_end - t0

    print(f"[WHISPER/BACKEND TEXT] {text}")
    print(f"[FLOW A] Total latency: {total_latency:.3f}s")

    return {
        "decision": "ACCEPT",
        "score": score,
        "text": text,
        "latency_total": total_latency,
        "breakdown": {
            "rec_short": t_rec_short_end - t0,
            "ecapa": t_ecapa_end - t_ecapa_start,
            "rec_cmd": t_cmd_rec_end - t_cmd_rec_start,
            "stt": t_whisper_end - t_whisper_start,
        },
    }
