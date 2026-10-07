from faster_whisper import WhisperModel

# load once (important)
# "small" is a good balance. try "base" if your CPU is weak.
_model = WhisperModel("small", device="cpu", compute_type="int8")  # int8 = faster CPU

def transcribe_wav(wav_path: str) -> str:
    segments, info = _model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=False,   # you already do push-to-talk
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text