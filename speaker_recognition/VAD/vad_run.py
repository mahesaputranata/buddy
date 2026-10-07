import sounddevice as sd
import numpy as np
import threading
import sys
import time


SR = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)

MARGIN_DB = 3.0
START_FRAMES = 3
END_FRAMES = 6

NOISE_CLAMP_DB = -80.0   
IGNORE_BELOW_DB = -90.0 
EPS = 1e-12


running = True
in_speech = False
speech_count = 0
silence_count = 0

noise_db = -60.0


def rms_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x * x) + EPS)
    return 20.0 * np.log10(rms + EPS)


def keyboard_listener():
    global running
    while running:
        key = sys.stdin.readline().strip()
        if key.lower() == "q":
            running = False
            break


def audio_callback(indata, frames, time_info, status):
    global noise_db, speech_count, silence_count, in_speech

    if not running:
        return

    x = indata[:, 0].astype(np.float32)
    e_db = rms_db(x)

    # Ignore hard digital silence
    if e_db < IGNORE_BELOW_DB:
        return

    # Clamp noise floor
    noise_db = max(noise_db, NOISE_CLAMP_DB)
    threshold = noise_db + MARGIN_DB

    is_speech = e_db > threshold

    # Update noise ONLY when confident non-speech
    if not in_speech and not is_speech:
        noise_db = 0.95 * noise_db + 0.05 * e_db

    # Hangover logic
    if is_speech:
        speech_count += 1
        silence_count = 0
    else:
        silence_count += 1
        speech_count = 0

    # State transitions
    if not in_speech and speech_count >= START_FRAMES:
        in_speech = True
        print("detected")

    if in_speech and silence_count >= END_FRAMES:
        in_speech = False


def main():
    print("Real-time VAD running")
    print("Press 'q' + Enter to quit\n")

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=FRAME_SAMPLES,
        callback=audio_callback,
    ):
        while running:
            time.sleep(0.1)

    print("Stopped.")


if __name__ == "__main__":
    main()
