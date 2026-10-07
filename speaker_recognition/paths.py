"""Where the speaker-recognition lab keeps its private artefacts (none are in the repo)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_DIR = Path(os.environ.get("BUDDY_SPEAKER_PROFILE_DIR", HERE / "profile"))      # speaker_centroid.npy etc.
MODEL_DIR = Path(os.environ.get("BUDDY_SPEAKER_MODEL_DIR", HERE / "pretrained_models" / "spkrec-ecapa-voxceleb"))
DATA_DIR = Path(os.environ.get("BUDDY_SPEAKER_DATA_DIR", HERE / "data"))               # raw / enrolment audio
