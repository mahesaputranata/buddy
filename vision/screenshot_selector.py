"""
vision/screenshot_selector.py

Scans the screenshots/ folder and decides which image(s) to analyze
based on recency and burst-clustering logic.

Outcomes
--------
SINGLE  – one fresh screenshot → analyze immediately
BURST   – 2-5 screenshots within BURST_WINDOW_S of each other → ask user
STALE   – latest screenshot is older than FRESHNESS_WINDOW_S → confirm first
EMPTY   – no screenshots found → tell user to take one
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List

# ── Tuneable constants ─────────────────────────────────────────────────────────
BURST_WINDOW_S     = 5    # seconds: screenshots this close together = one burst
FRESHNESS_WINDOW_S = 90   # seconds: older than this → confirm before analyzing
MAX_BURST_IMAGES   = 5    # cap on images combined into one multi-image prompt
SCREENSHOT_DIR     = "screenshots"
SCREENSHOT_EXT     = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
# ──────────────────────────────────────────────────────────────────────────────


class SelectionOutcome(Enum):
    SINGLE = auto()
    BURST  = auto()
    STALE  = auto()
    EMPTY  = auto()


@dataclass
class ScreenshotSelection:
    outcome: SelectionOutcome
    paths: List[str] = field(default_factory=list)   # selected file path(s)
    age_seconds: float = 0.0                          # age of the newest file
    burst_count: int = 0                              # how many files in burst
    # Human-readable message for TTS / logging
    message: str = ""


def _scan_folder(folder: str) -> List[tuple[float, str]]:
    """
    Return list of (mtime, abs_path) for all screenshot files in *folder*,
    sorted newest-first.
    Falls back gracefully if the folder doesn't exist yet.
    """
    p = Path(folder)
    if not p.exists():
        return []

    entries = []
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in SCREENSHOT_EXT:
            try:
                mtime = f.stat().st_mtime
                entries.append((mtime, str(f.resolve())))
            except OSError:
                pass

    entries.sort(key=lambda x: x[0], reverse=True)   # newest first
    return entries


def select_screenshots(
    folder: str = SCREENSHOT_DIR,
    now: float | None = None,
) -> ScreenshotSelection:
    """
    Main entry point.  Returns a ScreenshotSelection describing what was found
    and what the system should do next.

    Parameters
    ----------
    folder : directory to scan (default: SCREENSHOT_DIR constant)
    now    : override current time (useful for unit tests)
    """
    now = now or time.time()
    entries = _scan_folder(folder)

    # ── EMPTY ──────────────────────────────────────────────────────────────────
    if not entries:
        return ScreenshotSelection(
            outcome=SelectionOutcome.EMPTY,
            message=(
                "I don't have any screenshots to analyze. "
                "Take one with Windows Shift S first, then ask me again."
            ),
        )

    newest_mtime, newest_path = entries[0]
    age = now - newest_mtime

    # ── STALE ──────────────────────────────────────────────────────────────────
    if age > FRESHNESS_WINDOW_S:
        age_str = _fmt_age(age)
        return ScreenshotSelection(
            outcome=SelectionOutcome.STALE,
            paths=[newest_path],
            age_seconds=age,
            message=(
                f"The most recent screenshot I have is from {age_str} ago. "
                "Is that the one you want me to analyze?"
            ),
        )

    # ── Collect burst candidates (within BURST_WINDOW_S of the newest) ────────
    burst: List[str] = []
    for mtime, path in entries:
        if newest_mtime - mtime <= BURST_WINDOW_S:
            burst.append(path)
        else:
            break   # list is sorted newest-first; once outside window, stop

    burst = burst[:MAX_BURST_IMAGES]   # hard cap

    # ── BURST ──────────────────────────────────────────────────────────────────
    if len(burst) > 1:
        count = len(burst)
        return ScreenshotSelection(
            outcome=SelectionOutcome.BURST,
            paths=burst,                # newest-first order
            age_seconds=age,
            burst_count=count,
            message=(
                f"I see {count} recent screenshots taken within {BURST_WINDOW_S} seconds "
                f"of each other. Should I analyze all {count} together, "
                "or just the latest one?"
            ),
        )

    # ── SINGLE ─────────────────────────────────────────────────────────────────
    return ScreenshotSelection(
        outcome=SelectionOutcome.SINGLE,
        paths=[newest_path],
        age_seconds=age,
        message="",   # no prompt needed — analyze immediately
    )


def _fmt_age(seconds: float) -> str:
    """Return a human-friendly age string, e.g. '2 minutes', '45 seconds'."""
    s = int(seconds)
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''}"
    m = s // 60
    return f"{m} minute{'s' if m != 1 else ''}"