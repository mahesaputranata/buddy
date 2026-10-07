# ecapa_extractor.py
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  

import torchaudio
import numpy as np
from typing import Optional
from typing import Union

from pathlib import Path
import torch
# from speechbrain.pretrained import SpeakerRecognition
from speechbrain.inference.speaker import SpeakerRecognition

from speechbrain.utils.fetching import LocalStrategy

class ECAPAEmbeddingExtractor:
    """
    Wrapper around SpeechBrain's ECAPA-TDNN pretrained model.
    Usage:
        extractor = ECAPAEmbeddingExtractor()
        emb = extractor.encode_file("some_chunk.wav")  # -> np.ndarray (D,)
    """

    # def __init__(self, device: Optional[str]=None):
    #     if device is None:
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #     self.device = device

    #     # Load pretrained ECAPA model (downloaded automatically on first run)
    #     # You can change savedir if you want a different cache location.
    #     self.model = SpeakerRecognition.from_hparams(
    #         source="speechbrain/spkrec-ecapa-voxceleb",
    #         savedir="pretrained_models/spkrec-ecapa-voxceleb",
    #         run_opts={"device": self.device},
    #     )

    #     # For safety, define the expected sample rate
    #     self.target_sr = 16000

    def __init__(self, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # savedir = Path(__file__).resolve().parent / "pretrained_models" / "spkrec-ecapa-voxceleb"
        from paths import MODEL_DIR
        savedir = MODEL_DIR
        savedir.mkdir(parents=True, exist_ok=True)

        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,  # key line
        )

        self.target_sr = 16000

    def _load_wav(self, wav_path: Path) -> torch.Tensor:
        """Load a wav file and return a mono waveform tensor on the correct device."""
        wav_path = Path(wav_path)
        waveform, sr = torchaudio.load(str(wav_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Shape: (batch=1, time)
        waveform = waveform.to(self.device)
        return waveform

    def encode_file(self, wav_path: Union[str, Path]) -> np.ndarray:
        wav = self._load_wav(Path(wav_path))

        # encode_batch expects (batch, time)
        with torch.no_grad():
            emb = self.model.encode_batch(wav)  # shape: (1, 1, D) or (1, D)

        emb = emb.squeeze().cpu().numpy()  # -> (D,)
        return emb
