"""
vision/vision_source_classifier.py

Classifies whether a vision trigger phrase is asking to:
  - ACCESS_WEBCAM  : look at the user / real world through camera
  - USE_SCREENSHOT : analyze what's on the screen
  - AMBIGUOUS      : unclear, ask the user

Pipeline
--------
1. Keyword rules  (zero cost, handles obvious cases instantly)
2. Sentence-transformers cosine similarity (handles ambiguous cases)
3. Fallback matrix with availability checks

Thresholds
----------
KEYWORD_WEBCAM_SCORE   : score added per webcam keyword hit
KEYWORD_SCREEN_SCORE   : score added per screenshot keyword hit
SIMILARITY_THRESHOLD   : minimum cosine score to trust classification
AMBIGUITY_BAND         : if |webcam_score - screen_score| < this → ambiguous
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import cv2

# ── Tuneable constants ─────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD  = 0.35   # minimum score to trust either class
AMBIGUITY_BAND        = 0.10   # if scores are this close → ambiguous
WEBCAM_SAVE_DIR       = "webcam_captures"
# ──────────────────────────────────────────────────────────────────────────────


class VisionSource(Enum):
    NONE = "none"  
    ACCESS_WEBCAM  = auto()
    USE_SCREENSHOT = auto()
    AMBIGUOUS      = auto()


@dataclass
class ClassificationResult:
    source: VisionSource
    confidence: float          # 0.0 – 1.0
    webcam_score: float
    screen_score: float
    method: str                # "keyword" | "similarity" | "fallback"
    message: str = ""          # populated only for AMBIGUOUS + fallback cases


# ── Reference sentences for cosine similarity ─────────────────────────────────
_WEBCAM_REFERENCE = (
    "look at me see me through the camera watch me observe me "
    "what am I doing what do I look like face person real world "
    "can you see me look at my face watch what I am doing"
)

_SCREEN_REFERENCE = (
    "look at my screen what is on my screen analyze this screenshot "
    "what is this error check this code UI display window desktop "
    "what do you see on the screen read this text on screen"
)

# ── Keyword rules ──────────────────────────────────────────────────────────────
_WEBCAM_KEYWORDS = [
    "look at me", "can you see me", "see me", "watch me",
    "what am i doing", "what do i look like", "look at my face",
    "observe me", "are you watching", "look at my hands",
    "what am i holding", "can you watch", "look at this object",
    "what is this thing", "what do i have",
]

_SCREEN_KEYWORDS = [
    "look at my screen", "what's on my screen", "what is on my screen",
    "analyze screenshot", "analyze my screen", "check this error",
    "what error", "read this", "debug this", "check this code",
    "what do you see on screen", "my display", "on my monitor",
    "in the screenshot", "on my desktop",
]

# ── Lazy-loaded sentence transformer ──────────────────────────────────────────
_model = None

def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[VISION_CLASSIFIER] sentence-transformers model loaded")
        except ImportError:
            print("[VISION_CLASSIFIER] sentence-transformers not available — keyword-only mode")
            _model = False   # mark as unavailable, don't retry
    return _model if _model is not False else None


def _cosine_similarity(a, b) -> float:
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _keyword_scores(text: str) -> tuple[float, float]:
    import re
    # strip punctuation before matching
    t = re.sub(r"[?.!,]", "", text.lower())
    webcam = sum(0.25 for kw in _WEBCAM_KEYWORDS if kw in t)
    screen  = sum(0.25 for kw in _SCREEN_KEYWORDS  if kw in t)
    return min(webcam, 1.0), min(screen, 1.0)


def _similarity_scores(text: str) -> tuple[float, float]:
    """
    Returns (webcam_score, screen_score) via cosine similarity.
    Returns (0, 0) if model unavailable.
    """
    model = _get_model()
    if model is None:
        return 0.0, 0.0

    embeddings = model.encode([text, _WEBCAM_REFERENCE, _SCREEN_REFERENCE])
    webcam = _cosine_similarity(embeddings[0], embeddings[1])
    screen  = _cosine_similarity(embeddings[0], embeddings[2])
    return webcam, screen


def _is_webcam_available() -> bool:
    """Quick OpenCV check — tries to open camera index 0."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        available = cap.isOpened()
        cap.release()
        return available
    except Exception:
        return False


# ── Main classifier ────────────────────────────────────────────────────────────

def classify_vision_source(user_text: str) -> ClassificationResult:
    """
    Classify whether the user wants webcam or screenshot vision.

    Returns a ClassificationResult with source, scores, and an optional
    message for AMBIGUOUS cases that should be spoken to the user.
    """
    t = (user_text or "").strip().lower()

    # ── Stage 1: Keyword rules ─────────────────────────────────────────────
    kw_webcam, kw_screen = _keyword_scores(t)

    if kw_webcam >= 0.50 and kw_webcam > kw_screen + AMBIGUITY_BAND:
        return ClassificationResult(
            source=VisionSource.ACCESS_WEBCAM,
            confidence=min(kw_webcam, 1.0),
            webcam_score=kw_webcam,
            screen_score=kw_screen,
            method="keyword",
        )

    if kw_screen >= 0.50 and kw_screen > kw_webcam + AMBIGUITY_BAND:
        return ClassificationResult(
            source=VisionSource.USE_SCREENSHOT,
            confidence=min(kw_screen, 1.0),
            webcam_score=kw_webcam,
            screen_score=kw_screen,
            method="keyword",
        )

    # ── Stage 2: Cosine similarity ─────────────────────────────────────────
    sim_webcam, sim_screen = _similarity_scores(t)

    # blend keyword + similarity scores (keyword gets slight priority)
    webcam_score = (kw_webcam * 0.4) + (sim_webcam * 0.6)
    screen_score  = (kw_screen  * 0.4) + (sim_screen  * 0.6)

    print(f"[VISION_CLASSIFIER] webcam={webcam_score:.2f} screen={screen_score:.2f} "
          f"(kw: {kw_webcam:.2f}/{kw_screen:.2f} "
          f"sim: {sim_webcam:.2f}/{sim_screen:.2f})")

    gap = abs(webcam_score - screen_score)

    # ── Stage 3: Availability + fallback matrix ────────────────────────────

    # Clear webcam winner
    if webcam_score >= SIMILARITY_THRESHOLD and webcam_score > screen_score + AMBIGUITY_BAND:
        webcam_available = _is_webcam_available()

        if webcam_available:
            return ClassificationResult(
                source=VisionSource.ACCESS_WEBCAM,
                confidence=webcam_score,
                webcam_score=webcam_score,
                screen_score=screen_score,
                method="similarity",
            )
        else:
            # High webcam confidence but no webcam
            return ClassificationResult(
                source=VisionSource.AMBIGUOUS,
                confidence=webcam_score,
                webcam_score=webcam_score,
                screen_score=screen_score,
                method="similarity",
                message=(
                    "I think you want me to look at you through the camera, "
                    "but I don't have webcam access right now. "
                    "Do you mean you want me to look at your screen instead?"
                ),
            )

    # Clear screenshot winner
    if screen_score >= SIMILARITY_THRESHOLD and screen_score > webcam_score + AMBIGUITY_BAND:
        return ClassificationResult(
            source=VisionSource.USE_SCREENSHOT,
            confidence=screen_score,
            webcam_score=webcam_score,
            screen_score=screen_score,
            method="similarity",
        )

    # Ambiguous — scores too close or both too low
    webcam_available = _is_webcam_available()

    if not webcam_available:
        # Webcam not available — nudge toward screenshot
        return ClassificationResult(
            source=VisionSource.AMBIGUOUS,
            confidence=max(webcam_score, screen_score),
            webcam_score=webcam_score,
            screen_score=screen_score,
            method="fallback",
            message=(
                "I'm not sure if you want me to look at your screen or through the camera. "
                "Since I don't have webcam access, "
                "did you mean to take a screenshot with Windows Shift S first?"
            ),
        )

    return ClassificationResult(
        source=VisionSource.AMBIGUOUS,
        confidence=max(webcam_score, screen_score),
        webcam_score=webcam_score,
        screen_score=screen_score,
        method="fallback",
        message=(
            "Do you want me to look at your screen, "
            "or look at you through the camera?"
        ),
    )