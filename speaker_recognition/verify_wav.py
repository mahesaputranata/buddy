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