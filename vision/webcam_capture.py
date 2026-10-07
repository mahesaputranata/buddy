"""
vision/webcam_capture.py

Captures frames from the webcam using OpenCV.

Two modes
---------
single_frame : one snapshot — used on first vision trigger
burst_frames : N frames over a short interval — used when user asks
               for detailed observation on follow-up

Both return a list of absolute file paths saved to WEBCAM_SAVE_DIR,
ready to pass directly into run_vision_pipeline().
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Tuneable constants ─────────────────────────────────────────────────────────
WEBCAM_SAVE_DIR      = "webcam_captures"
WEBCAM_INDEX         = 0       # camera index (0 = default webcam)
BURST_FRAME_COUNT    = 3       # frames to capture in burst mode
BURST_INTERVAL_S     = 0.4     # seconds between burst frames
WARMUP_FRAMES        = 3       # frames to discard before capture (exposure settle)
WEBCAM_EXT           = ".png"
# ──────────────────────────────────────────────────────────────────────────────


def _timestamp_filename(prefix: str = "webcam") -> str:
    now = datetime.now()
    return now.strftime(f"{prefix}_%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}{WEBCAM_EXT}"


def _save_frame(frame, out_dir: Path) -> Optional[str]:
    """Save a single OpenCV frame to disk. Returns absolute path or None."""
    try:
        import cv2
        filename = _timestamp_filename()
        out_path = (out_dir / filename).resolve()
        cv2.imwrite(str(out_path), frame)
        return str(out_path)
    except Exception as e:
        print(f"[WEBCAM] Failed to save frame: {e}")
        return None


def _open_camera():
    """Open camera and return cap, or None if unavailable."""
    try:
        import cv2
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            return None

        # Warm up — discard first N frames so exposure settles
        for _ in range(WARMUP_FRAMES):
            cap.read()

        return cap
    except Exception as e:
        print(f"[WEBCAM] Failed to open camera: {e}")
        return None


def capture_single_frame(out_dir: str = WEBCAM_SAVE_DIR) -> Optional[str]:
    """
    Capture one frame from the webcam.

    Returns
    -------
    Absolute path to saved image, or None if capture failed.
    """
    import cv2

    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_camera()
    if cap is None:
        print("[WEBCAM] Camera not available for single capture.")
        return None

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WEBCAM] Failed to read frame.")
            return None

        path = _save_frame(frame, out_path_dir)
        print(f"[WEBCAM] Single frame saved: {path}")
        return path

    finally:
        cap.release()


def capture_burst_frames(
    out_dir: str = WEBCAM_SAVE_DIR,
    count: int = BURST_FRAME_COUNT,
    interval_s: float = BURST_INTERVAL_S,
) -> List[str]:
    """
    Capture N frames over a short interval from the webcam.

    Returns
    -------
    List of absolute paths (newest-first), empty list if capture failed.
    """
    import cv2

    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_camera()
    if cap is None:
        print("[WEBCAM] Camera not available for burst capture.")
        return []

    paths = []
    try:
        for i in range(count):
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[WEBCAM] Failed to read frame {i+1}/{count}")
                continue

            path = _save_frame(frame, out_path_dir)
            if path:
                paths.append(path)
                print(f"[WEBCAM] Burst frame {i+1}/{count} saved: {path}")

            if i < count - 1:
                time.sleep(interval_s)

    finally:
        cap.release()

    # Return newest-first (consistent with screenshot_selector convention)
    return list(reversed(paths))


def is_webcam_available() -> bool:
    """Non-destructive check — opens and immediately closes the camera."""
    try:
        import cv2
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        available = cap.isOpened()
        cap.release()
        return available
    except Exception:
        return False


# ── Follow-up detection ────────────────────────────────────────────────────────
_BURST_FOLLOWUP_TRIGGERS = [
    "look more carefully", "look again", "observe this",
    "watch what i", "watch carefully", "look closely",
    "take a closer look", "look in detail", "more detail",
    "watch me do", "can you see what i",
]

def wants_burst_mode(user_text: str) -> bool:
    """
    Returns True if the follow-up utterance suggests the user wants
    a more detailed multi-frame observation.
    """
    t = (user_text or "").lower()
    return any(trigger in t for trigger in _BURST_FOLLOWUP_TRIGGERS)