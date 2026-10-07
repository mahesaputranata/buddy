# # verify_wav.py
# import json, sys, traceback
# from pathlib import Path
# import numpy as np

# from ExtractingEmbeddings.types.ecapa_extractor import ECAPAEmbeddingExtractor
# from Similarities.similarities import Similarities

# THRESHOLD = 0.2

# def main():
#     try:
#         if len(sys.argv) < 2:
#             print(json.dumps({"ok": False, "error": "Missing wav path"}))
#             return

#         wav_path = Path(sys.argv[1])
#         if not wav_path.exists():
#             print(json.dumps({"ok": False, "error": f"Wav not found: {wav_path}"}))
#             return

#         centroid = np.load(PROFILE_DIR / "speaker_centroid.npy")
#         extractor = ECAPAEmbeddingExtractor()

#         emb = extractor.encode_file(wav_path)
#         score = float(Similarities.comparison(emb, centroid, mode="cosine"))
#         ok = score >= THRESHOLD

#         print(json.dumps({"ok": ok, "score": score, "wav": str(wav_path)}))

#     except Exception as e:
#         print(json.dumps({
#             "ok": False,
#             "error": repr(e),
#             "trace": traceback.format_exc()[:2000],
#         }))

# if __name__ == "__main__":
#     main()

# speaker_verifier.py
from pathlib import Path
import sys
import numpy as np

from paths import PROFILE_DIR
from ExtractingEmbeddings.types.ecapa_extractor import ECAPAEmbeddingExtractor
from Similarities.similarities import Similarities


class SpeakerVerifier:
    def __init__(
        self,
        profile_dir=None,
        threshold=0.2,
    ):
        self.threshold = threshold
        self.profile_dir = Path(profile_dir or PROFILE_DIR)

        print("[SPK] Loading centroid...", file=sys.stderr, flush=True)
        self.centroid = np.load(self.profile_dir / "speaker_centroid.npy")

        print("[SPK] Loading ECAPA extractor...", file=sys.stderr, flush=True)
        self.extractor = ECAPAEmbeddingExtractor()

        print("[SPK] Speaker verifier ready.", file=sys.stderr, flush=True)

    def verify(self, wav_path):
        wav_path = Path(wav_path)

        if not wav_path.exists():
            return {
                "ok": False,
                "error": f"Wav not found: {wav_path}",
            }

        emb = self.extractor.encode_file(wav_path)
        score = float(Similarities.comparison(emb, self.centroid, mode="cosine"))
        ok = score >= self.threshold

        return {
            "ok": ok,
            "score": score,
            "wav": str(wav_path),
        }