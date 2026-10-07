from TTS.api import TTS

tts = TTS("tts_models/en/vctk/vits")

candidates = ["p225", "p226", "p228", "p229", "p230", "p232", "p234", "p270"]
text = "Good afternoon, sir. Here is your briefing."

for spk in candidates:
    if spk in tts.speakers:
        out = f"voice_{spk}.wav"
        tts.tts_to_file(text=text, speaker=spk, file_path=out)
        print("Wrote", out)