# speaker_gate_once.py
import json
import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import traceback

from ExtractingEmbeddings.types.ecapa_extractor import ECAPAEmbeddingExtractor
from Similarities.similarities import Similarities

SR = 16000
DURATION = 4.0
THRESHOLD = 0.4

from paths import PROFILE_DIR
LOG_DIR = Path("mic_logs")
AUDIO_LOG_DIR = LOG_DIR / "audio"
LOG_DIR.mkdir(exist_ok=True)
AUDIO_LOG_DIR.mkdir(parents=True, exist_ok=True)

def record_once(out_path: Path, duration: float = DURATION, sr: int = SR):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.squeeze()
    sf.write(out_path, audio, sr)

def main():
    try:
        centroid = np.load(PROFILE_DIR / "speaker_centroid.npy")
        extractor = ECAPAEmbeddingExtractor()

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_wav = AUDIO_LOG_DIR / f"mic_{ts}.wav"

        record_once(out_wav)

        emb = extractor.encode_file(out_wav)
        score = float(Similarities.comparison(emb, centroid, mode="cosine"))
        ok = score >= THRESHOLD

        # One-line JSON output for Buddy
        print(json.dumps({
            "ok": ok,
            "score": score,
            "wav": str(out_wav),
            "sr": SR,
            "duration": DURATION
        }))

    except Exception as e:
        # Buddy can detect errors too
        print(json.dumps({
            "ok": False,
            "error": repr(e),
            "trace": traceback.format_exc()[:2000]
        }))

if __name__ == "__main__":
    main()