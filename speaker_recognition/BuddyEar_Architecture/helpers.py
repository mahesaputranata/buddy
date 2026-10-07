# helpers.py

import datetime
from pathlib import Path
import time

import sounddevice as sd
import soundfile as sf
import numpy as np
from openai import OpenAI, RateLimitError, APIError  # add RateLimitError, APIError
import speech_recognition as sr 

from ExtractingEmbeddings.types.ecapa_extractor import ECAPAEmbeddingExtractor
from Similarities.similarities import Similarities

# Global extractor (loads once)
extractor = ECAPAEmbeddingExtractor()

DEFAULT_SR = 16000


def build_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_audio_path(log_dir: Path, prefix="mic"):
    ts = build_timestamp()
    return log_dir / f"{prefix}_{ts}.wav"


def record_wav(duration: float, out_path: Path, sr: int = DEFAULT_SR):
    print(f"[REC] Recording {duration:.1f}s ...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.squeeze()
    sf.write(out_path, audio, sr)
    print(f"[REC] Saved to {out_path}")


def ecapa_score(wav_path: Path, centroid: np.ndarray) -> float:
    emb = extractor.encode_file(wav_path)
    score = Similarities.comparison(emb, centroid, mode="cosine")
    return score


# ---------- NEW: Local fallback STT ----------

def local_fallback_transcribe(wav_path: Path) -> str:
    """
    Free fallback STT using SpeechRecognition + Google Web API.
    No key needed, but requires internet.
    """
    print("[FALLBACK] Using local SpeechRecognition (recognize_google) ...")
    r = sr.Recognizer()

    with sr.AudioFile(str(wav_path)) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio, language="en-US")
        print(f"[FALLBACK] Transcribed text: {text}")
        return text
    except sr.UnknownValueError:
        print("[FALLBACK] Could not understand audio.")
        return "<UNINTELLIGIBLE>"
    except sr.RequestError as e:
        print(f"[FALLBACK][ERROR] Google STT request failed: {e}")
        return "<FALLBACK_ERROR>"


# ---------- MODIFIED: Whisper with fallback ----------

def whisper_transcribe(wav_path: Path, client: OpenAI) -> str:
    """
    Whisper API call (non-streaming) with automatic fallback to free STT
    if quota/network/API errors occur.
    """
    import traceback

    t0 = time.perf_counter()
    try:
        with open(wav_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        t1 = time.perf_counter()
        print(f"[WHISPER] Transcribe took {t1 - t0:.3f}s")
        return resp.text

    except RateLimitError as e:
        # No credit / quota → fall back
        t1 = time.perf_counter()
        print(f"[WHISPER][QUOTA] No credit / quota after {t1 - t0:.3f}s")
        print("  -> Falling back to free local STT.")
        return local_fallback_transcribe(wav_path)

    except APIError as e:
        # General API-side errors → fall back
        t1 = time.perf_counter()
        print(f"[WHISPER][API ERROR] after {t1 - t0:.3f}s: {repr(e)}")
        print("  -> Falling back to free local STT.")
        return local_fallback_transcribe(wav_path)

    except Exception as e:
        # Unknown error → log and fall back too
        t1 = time.perf_counter()
        print(f"[WHISPER][ERROR] after {t1 - t0:.3f}s: {repr(e)}")
        traceback.print_exc()
        print("  -> Falling back to free local STT.")
        return local_fallback_transcribe(wav_path)
