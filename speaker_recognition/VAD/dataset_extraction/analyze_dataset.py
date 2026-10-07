import json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


DATASET_ROOT = (__import__("paths").DATA_DIR / "wav_dataset")  
SR_TARGET = 16000
FRAME_MS = 20
HOP_MS = 10
EPS = 1e-12

ACTIVE_DB = 3.0      # how far above noise a frame must be to count as "speech-active"
MARGIN_CLAMP = (3.0, 12.0)  # allow a bit more range


# If your folder names differ, edit these mappings:
SPEECH_HINTS = {"speech", "my_voice"}
NONSPEECH_HINTS = {"nonspeech", "environment_voice", "noise", "silence"}

FRAME_SAMPLES = int(SR_TARGET * FRAME_MS / 1000)
HOP_SAMPLES = int(SR_TARGET * HOP_MS / 1000)

def rms_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x * x) + EPS)
    return 20.0 * np.log10(rms + EPS)

def zcr(x: np.ndarray) -> float:
    # zero-crossing rate per frame
    x = x.astype(np.float32)
    return np.mean(x[:-1] * x[1:] < 0)

def iter_frames(x: np.ndarray, frame_len: int, hop: int):
    for start in range(0, len(x) - frame_len + 1, hop):
        yield x[start:start + frame_len]

def classify_path(p: Path) -> str:
    parts = {s.lower() for s in p.parts}
    if parts & SPEECH_HINTS:
        return "speech"
    if parts & NONSPEECH_HINTS:
        return "nonspeech"
    # fallback: if unknown, keep as "unknown"
    return "unknown"

def read_mono_resampled(path: Path):
    # NOTE: conversion script already makes 16k mono wav.
    # This checks + enforces mono.
    audio, sr = sf.read(str(path), always_2d=True)
    if sr != SR_TARGET:
        raise ValueError(f"{path} has sr={sr}, expected {SR_TARGET}. Reconvert with ffmpeg settings.")
    audio = audio[:, 0]  # mono
    return audio.astype(np.float32), sr

def collect_features():
    wavs = list(DATASET_ROOT.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found under {DATASET_ROOT.resolve()}")

    by_class = {
        "speech": {"rms_db": [], "zcr": []},
        "nonspeech": {"rms_db": [], "zcr": []},
        "unknown": {"rms_db": [], "zcr": []},
    }

    for w in wavs:
        cls = classify_path(w)
        x, sr = read_mono_resampled(w)

        # normalize? For VAD thresholding, usually DON'T normalize globally.
        # Keep raw levels so energy threshold reflects reality.

        for frame in iter_frames(x, FRAME_SAMPLES, HOP_SAMPLES):
            by_class[cls]["rms_db"].append(rms_db(frame))
            by_class[cls]["zcr"].append(zcr(frame))

    # convert lists to arrays
    for cls in by_class:
        for k in by_class[cls]:
            by_class[cls][k] = np.array(by_class[cls][k], dtype=np.float32)

    return by_class

def summarize(arr: np.ndarray, min_db: float = -90.0):
    """
    Summarize RMS energy while ignoring digital-silence frames
    caused by mic gating / DSP.
    """
    valid = arr[arr > min_db]

    if valid.size == 0:
        return {
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
            "note": "no frames above min_db"
        }

    return {
        "min": float(np.min(valid)),
        "p05": float(np.percentile(valid, 5)),
        "p50": float(np.percentile(valid, 50)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "frames_used": int(valid.size),
        "frames_total": int(arr.size),
    }

MIN_DB = -90.0  # same as summarize()

def valid_frames(arr, min_db=MIN_DB):
    return arr[arr > min_db]

def main():
    feats = collect_features()

    # Basic summaries
    print("\n=== RMS(dB) summaries ===")
    for cls in ["speech", "nonspeech", "unknown"]:
        if feats[cls]["rms_db"].size == 0:
            continue
        s = summarize(feats[cls]["rms_db"], min_db=-90.0)
        print(f"\n[{cls}]")
        for k, v in s.items():
            print(f"  {k:>4}: {v:7.2f}")

    # Plot distributions (speech vs nonspeech if available)
    plt.figure()

    if feats["speech"]["rms_db"].size:
        plt.hist(
            valid_frames(feats["speech"]["rms_db"]),
            bins=120,
            alpha=0.6,
            label="speech (valid)"
        )

    if feats["nonspeech"]["rms_db"].size:
        plt.hist(
            valid_frames(feats["nonspeech"]["rms_db"]),
            bins=120,
            alpha=0.6,
            label="nonspeech (valid)"
        )

    plt.xlabel("Frame RMS Energy (dB)")
    plt.ylabel("Frame count")
    plt.title("Energy distribution (valid frames only)")
    plt.legend()
    plt.show()



    params = {}

    if feats["speech"]["rms_db"].size and feats["nonspeech"]["rms_db"].size:

        speech_valid = valid_frames(feats["speech"]["rms_db"])
        noise_valid  = valid_frames(feats["nonspeech"]["rms_db"])

        noise_p95 = float(np.percentile(noise_valid, 95))

        # Keep only speech frames that are clearly above high-end noise
        speech_active = speech_valid[speech_valid > (noise_p95 + ACTIVE_DB)]


        if speech_active.size < 1000:
            # fallback if too few active frames (dataset too quiet / too short)
            speech_ref = float(np.percentile(speech_valid, 50))  # median of speech-valid
        else:
            speech_ref = float(np.percentile(speech_active, 20)) # "quiet speech" among active speech

        raw_gap = speech_ref - noise_p95

        # Now margin is chosen relative to noise. If overlap remains, start conservative.
        if raw_gap > 0:
            margin_db = float(np.clip(raw_gap * 0.6, MARGIN_CLAMP[0], MARGIN_CLAMP[1]))
        else:
            margin_db = 8.0

        plt.figure()

        plt.hist(
            noise_valid,
            bins=120,
            alpha=0.6,
            label="nonspeech (valid)"
        )

        plt.hist(
            speech_active,
            bins=120,
            alpha=0.6,
            label="speech (active)"
        )

        plt.axvline(noise_p95, color="k", linestyle="--", label="noise p95")
        plt.axvline(noise_p95 + margin_db, color="r", linestyle="--", label="threshold")

        plt.xlabel("Frame RMS Energy (dB)")
        plt.ylabel("Frame count")
        plt.title("VAD energy separation")
        plt.legend()
        plt.show()

        params = {
            "sr": SR_TARGET,
            "frame_ms": FRAME_MS,
            "hop_ms": HOP_MS,
            "suggested_margin_db": margin_db,
            "nonspeech_rms_p95_db": noise_p95,
            "speech_ref_db": speech_ref,
            "speech_active_frames": int(speech_active.size),
            "speech_valid_frames": int(speech_valid.size),
            "estimated_gap_db": raw_gap,
            "start_frames": 3,
            "end_frames": 12
        }


        print("\n=== Suggested VAD params ===")
        for k, v in params.items():
            print(f"{k}: {v}")

        out = Path("vad_params.json")
        out.write_text(json.dumps(params, indent=2))
        print(f"\nSaved: {out.resolve()}")
    else:
        print("\nNot enough labeled data to suggest margin (need both speech and nonspeech).")
        print("Make sure your folder names include 'speech'/'nonspeech' or adjust SPEECH_HINTS/NONSPEECH_HINTS.")

if __name__ == "__main__":
    main()
