import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from pathlib import Path

SR = 16000
FRAME_MS = 30          # 30 ms frames
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)
SILENCE_THRESHOLD = 0.01  # tune this
SILENCE_TAIL_SEC = 0.8    # stop after 0.8s of continuous silence
MAX_DURATION_SEC = 8.0    # absolute max safety bound

def record_until_silence(out_path: Path):
    print("[REC] Starting record_until_silence, speak your command...")
    frames = []
    silence_run = 0.0
    start_time = time.time()

    def callback(indata, frames_count, time_info, status):
        nonlocal silence_run
        # indata shape: (FRAME_SAMPLES, channels)
        mono = indata[:, 0]
        rms = np.sqrt(np.mean(mono**2))

        frames.append(mono.copy())

        if rms < SILENCE_THRESHOLD:
            silence_run += FRAME_MS / 1000.0
        else:
            silence_run = 0.0

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=FRAME_SAMPLES,
        callback=callback
    ):
        while True:
            time.sleep(0.05)

            elapsed = time.time() - start_time
            if elapsed > MAX_DURATION_SEC:
                print("[REC] Hit max duration, stopping.")
                break

            if silence_run >= SILENCE_TAIL_SEC:
                print("[REC] Detected trailing silence, stopping.")
                break

    if not frames:
        print("[REC] No audio captured.")
        return

    audio = np.concatenate(frames)
    sf.write(out_path, audio, SR)
    print(f"[REC] Saved command to {out_path}, length={len(audio)/SR:.2f}s")
