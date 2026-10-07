from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from brain.settings import setting

# ── Config: set via config.yaml / BUDDY_FACE_PYTHON / BUDDY_FACE_WORKER_SCRIPT ──
FACE_RECOGNITION_PYTHON = Path(setting("face_python", sys.executable))
WORKER_SCRIPT_PATH = Path(setting("face_worker_script", "recognizer_worker.py"))
WORKER_TIMEOUT_S = 15
# ──────────────────────────────────────────────────────────────────────────────


class RecognitionStatus(Enum):
    AUTHORIZED = auto()
    UNKNOWN    = auto()
    NO_FACE    = auto()
    ERROR      = auto()


@dataclass
class RecognitionResult:
    status:     RecognitionStatus
    confidence: float
    face_count: int
    message:    str = ""


def recognize_face(image_path: str) -> RecognitionResult:
    """
    Run face recognition by calling the isolated worker subprocess.
    Returns RecognitionResult — never raises.
    """

    if not FACE_RECOGNITION_PYTHON.exists():
        print(f"[FACE_RECOGNITION] Python not found: {FACE_RECOGNITION_PYTHON}")
        return RecognitionResult(RecognitionStatus.ERROR, 0.0, 0,
                                 "Face recognition environment not found.")

    if not WORKER_SCRIPT_PATH.exists():
        print(f"[FACE_RECOGNITION] Worker not found: {WORKER_SCRIPT_PATH}")
        return RecognitionResult(RecognitionStatus.ERROR, 0.0, 0,
                                 "Worker script not found.")

    try:
        proc = subprocess.run(
            [str(FACE_RECOGNITION_PYTHON), str(WORKER_SCRIPT_PATH), str(image_path)],
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
        )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if stderr:
            print(f"[FACE_RECOGNITION] stderr: {stderr[:200]}")

        if not stdout:
            return RecognitionResult(RecognitionStatus.ERROR, 0.0, 0,
                                     "No response from worker.")

        data = json.loads(stdout)

        status_map = {
            "AUTHORIZED": RecognitionStatus.AUTHORIZED,
            "UNKNOWN":    RecognitionStatus.UNKNOWN,
            "NO_FACE":    RecognitionStatus.NO_FACE,
            "ERROR":      RecognitionStatus.ERROR,
        }

        status = status_map.get(data.get("status", "ERROR"), RecognitionStatus.ERROR)
        result = RecognitionResult(
            status=status,
            confidence=float(data.get("confidence", 0.0)),
            face_count=int(data.get("face_count", 0)),
            message=data.get("message", ""),
        )

        print(f"[FACE_RECOGNITION] {result.status.name} "
              f"conf={result.confidence:.2f} faces={result.face_count}")
        return result

    except subprocess.TimeoutExpired:
        print(f"[FACE_RECOGNITION] Timed out after {WORKER_TIMEOUT_S}s")
        return RecognitionResult(RecognitionStatus.ERROR, 0.0, 0,
                                 "Recognition timed out.")

    except json.JSONDecodeError as e:
        print(f"[FACE_RECOGNITION] JSON parse error: {e}")
        return RecognitionResult(RecognitionStatus.ERROR, 0.0, 0,
                                 "Could not parse result.")

    except Exception as e:
        print(f"[FACE_RECOGNITION] Error: {e}")
        return RecognitionResult(RecognitionStatus.ERROR, 0.0, 0, str(e))