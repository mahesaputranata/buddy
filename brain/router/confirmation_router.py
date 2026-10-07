from __future__ import annotations
import re
from enum import Enum
import ollama

CONFIRM_MODEL = "llama3.2:latest"


class ConfirmResult(Enum):
    CONFIRM = "confirm"    # user agrees, wants Buddy to proceed
    DECLINE = "decline"    # user disagrees, wants Buddy to not proceed
    REDIRECT = "redirect"  # user ignored the question and asked/said something else
    UNCLEAR = "unclear"    # genuinely can't tell — filler, silence, confusion


# ── Tier 1: cheap deterministic short-circuit ──────────────────────────
_YES_ANCHORS = {
    "yes", "y", "yeah", "yep", "sure", "ok", "okay", "alright",
    "affirmative", "correct", "do it", "go ahead",
}
_NO_ANCHORS = {
    "no", "n", "nope", "nah", "negative", "don't", "dont",
    "stop", "cancel", "not needed", "no need", "no thanks",
}

def _fast_match(text: str) -> ConfirmResult | None:
    t = re.sub(r"[.!?,]+$", "", (text or "").strip().lower())
    if t in _YES_ANCHORS:
        return ConfirmResult.CONFIRM
    if t in _NO_ANCHORS:
        return ConfirmResult.DECLINE
    return None


# ── Tier 2: semantic resolution via LLM ─────────────────────────────────
def resolve_confirmation(user_input: str, question_context: str,
                          model: str = CONFIRM_MODEL) -> ConfirmResult:
    """
    Decide what user_input means RELATIVE TO the question Buddy just asked.
    Falls back to the fast keyword tier first; only calls the LLM for
    natural-language replies that don't match a clean anchor.
    """
    fast = _fast_match(user_input)
    if fast is not None:
        return fast

    system = (
        "You are resolving a yes/no confirmation for a voice assistant. "
        "Buddy just asked the user a question. Decide what the user's reply "
        "means, RELATIVE TO THAT SPECIFIC QUESTION — not in general.\n\n"
        "Respond with exactly one word:\n"
        "CONFIRM  - the user agrees / wants Buddy to proceed\n"
        "DECLINE  - the user disagrees / wants Buddy to NOT proceed\n"
        "REDIRECT - the user ignored the question and said/asked something else entirely\n"
        "UNCLEAR  - you genuinely cannot tell what they mean\n"
    )
    user = f'Buddy asked: "{question_context}"\nUser replied: "{user_input}"\n\nAnswer:'

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0},
        )
        raw = (resp.get("message", {}) or {}).get("content", "").strip().upper()
    except Exception as e:
        print(f"[ConfirmRouter] LLM error: {e} -> UNCLEAR")
        return ConfirmResult.UNCLEAR

    for result in ConfirmResult:
        if result.name in raw:
            print(f"[ConfirmRouter] '{user_input[:40]}' -> {result.name}")
            return result

    print(f"[ConfirmRouter] unparseable LLM output {raw!r} -> UNCLEAR")
    return ConfirmResult.UNCLEAR

class VisionSourceChoice(Enum):
    SCREEN = "screen"
    CAMERA = "camera"
    DECLINE = "decline"
    UNCLEAR = "unclear"


def resolve_vision_source_choice(user_input: str, model: str = CONFIRM_MODEL) -> VisionSourceChoice:
    """
    Resolves a 3-way vision-source confirmation: screen vs camera vs decline.
    Same tiered approach as resolve_confirmation — cheap keyword tier first,
    LLM only for ambiguous natural language.
    """
    t = (user_input or "").strip().lower()

    # ── Tier 1: cheap keyword anchors ──
    if any(w in t for w in ["screen", "monitor", "display", "desktop"]):
        return VisionSourceChoice.SCREEN
    if any(w in t for w in ["camera", "webcam", "cam"]):
        return VisionSourceChoice.CAMERA

    fast_decline = _fast_match(user_input)
    if fast_decline == ConfirmResult.DECLINE:
        return VisionSourceChoice.DECLINE

    # ── Tier 2: LLM for natural language that doesn't hit the anchors ──
    system = (
        "Buddy just asked the user whether it should look at their screen, "
        "or through the camera at them. Decide what the user's reply means.\n\n"
        "Respond with exactly one word:\n"
        "SCREEN  - user wants Buddy to look at the screen\n"
        "CAMERA  - user wants Buddy to look through the camera\n"
        "DECLINE - user doesn't want Buddy to look at anything\n"
        "UNCLEAR - you genuinely cannot tell\n"
    )
    user = f'User replied: "{user_input}"\n\nAnswer:'

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0},
        )
        raw = (resp.get("message", {}) or {}).get("content", "").strip().upper()
    except Exception as e:
        print(f"[VisionSourceRouter] LLM error: {e} -> UNCLEAR")
        return VisionSourceChoice.UNCLEAR

    for choice in VisionSourceChoice:
        if choice.name in raw:
            print(f"[VisionSourceRouter] '{user_input[:40]}' -> {choice.name}")
            return choice

    print(f"[VisionSourceRouter] unparseable LLM output {raw!r} -> UNCLEAR")
    return VisionSourceChoice.UNCLEAR