# flow_b.py

from pathlib import Path
import time

from BuddyEar_Architecture.helpers import (
    ecapa_score,
    record_wav,
    whisper_transcribe,
    DEFAULT_SR,
)
from BuddyEar_Architecture.silence_detector import record_until_silence

def flow_parallel_full(
    CMD_WIN: float,
    THRESHOLD: float,
    CENTROID,
    client,
    SR: int = DEFAULT_SR,
):
    """
    FLOW B: Record full command once, then run ECAPA + STT (Whisper or fallback)
    on the same audio in parallel (conceptually).

    Latency ≈ record_time + max(ecapa_time, stt_time)
    """
    print("\n=== FLOW B: Parallel ECAPA + STT (Whisper + fallback) ===")

    cmd_path = Path("temp_cmd.wav")

    # 1) Record command window
    t0 = time.perf_counter()
    # record_wav(CMD_WIN, cmd_path, SR)
    record_until_silence(cmd_path)

    t_rec_end = time.perf_counter()
    rec_time = t_rec_end - t0

    # 2) ECAPA
    t_ecapa_start = time.perf_counter()
    score = ecapa_score(cmd_path, CENTROID)
    t_ecapa_end = time.perf_counter()
    ecapa_time = t_ecapa_end - t_ecapa_start

    # 3) STT backend (Whisper first, fallback if needed)
    t_stt_start = time.perf_counter()
    text = whisper_transcribe(cmd_path, client)
    t_stt_end = time.perf_counter()
    stt_time = t_stt_end - t_stt_start

    print(f"[ECAPA] score={score:.4f}")

    # 4) Decision logic based on ECAPA
    if score >= THRESHOLD:
        decision = "ACCEPT"
        print("[DECISION] ACCEPT (enrolled speaker)")
        print(f"[STT TEXT] {text}")
    else:
        decision = "REJECT"
        print("[DECISION] REJECT (not the enrolled speaker) -> ignoring text")
        text = None

    # 5) Approximate parallel latency:
    total_parallel = rec_time + max(ecapa_time, stt_time)

    print(
        f"[FLOW B] Latency ≈ {total_parallel:.3f}s "
        f"(rec={rec_time:.3f}, ecapa={ecapa_time:.3f}, stt={stt_time:.3f})"
    )

    return {
        "decision": decision,
        "score": score,
        "text": text,
        "latency_total": total_parallel,
        "breakdown": {
            "rec": rec_time,
            "ecapa": ecapa_time,
            "stt": stt_time,
        },
    }
