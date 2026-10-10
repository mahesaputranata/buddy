import os

from openai import OpenAI
import base64
from pathlib import Path
from typing import Union, List, Optional


def image_to_data_url(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    mime = f"image/{ext}"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _to_path_list(image_path: Union[str, List[str]]) -> List[str]:
    if isinstance(image_path, (str, Path)):
        return [str(image_path)]
    return [str(p) for p in image_path]


def _build_prompt(
    user_question: str,
    path_count: int,
    authorized_user: bool,
    is_webcam: bool,
    context_hint: Optional[str] = None,   # ← NEW: injected by vision awareness scorer
) -> str:
    """
    Builds the vision prompt based on context.

    context_hint: when provided (from vision_awareness scorer), shapes the
    prompt toward the current situation so GPT-4o knows what to look for.
    It also controls the response style — Buddy should speak naturally,
    as if it just glanced at the scene, using phrases like
    "Based on what I can see..." or "Looking at your setup..."
    """

    # ── Context-aware path (proactive / awareness-triggered) ──────────────
    # This fires when vision_awareness.py decided to grab a frame proactively.
    # The hint tells GPT-4o the exact situation — rover off, troubleshooting, etc.
    if context_hint and is_webcam:
        base = (
            "You are Buddy, a personal AI assistant. "
            "You just proactively looked at the user's webcam because the context called for it. "
            "Do NOT say 'I used the camera' or 'I grabbed a frame'. "
            "Instead, speak naturally as if you just glanced at their workspace. "
            "Use phrases like: 'Based on what I can see...', 'Looking at your setup...', "
            "'From what I'm seeing...', 'I can see that...'. "
            "Be concise and directly useful — focus on what's relevant to the situation.\n\n"
            f"SITUATION CONTEXT: {context_hint}\n\n"
        )
        if authorized_user:
            base += (
                "The person in view is the verified owner — respond personally and directly.\n\n"
            )
        base += f"User said: {user_question or 'What do you see?'}"
        return base

    # ── Webcam + authorized user ───────────────────────────────────────────
    if is_webcam and authorized_user:
        if path_count == 1:
            return (
                "You are Buddy, a personal AI assistant. "
                "The person in this image has been verified as your owner — "
                "respond directly and personally, as if speaking to them. "
                "Do not describe them as 'a person' or 'someone' — "
                "acknowledge them as the user you serve. "
                "Describe what you observe naturally and conversationally.\n\n"
                f"User question: {user_question or 'What do you see?'}"
            )
        else:
            return (
                f"You are Buddy, a personal AI assistant analyzing {path_count} "
                "webcam frames taken in sequence of your verified owner. "
                "Respond directly and personally. "
                "Describe what you observe across the frames — "
                "note any movement, activity, or changes between them.\n\n"
                f"User question: {user_question or 'What do you see?'}"
            )

    # ── Webcam + unknown / no face ─────────────────────────────────────────
    if is_webcam and not authorized_user:
        return (
            "You are Buddy's vision sensor. "
            "Analyze what is visible in this webcam frame and describe it. "
            "Be concise and direct.\n\n"
            f"User question: {user_question or 'What do you see?'}"
        )

    # ── Screenshot single ──────────────────────────────────────────────────
    if path_count == 1:
        return (
            "You are the vision module for an AI personal assistant. "
            "The user is an engineer with a control systems background. "
            "Analyze this screenshot and respond based on what is actually visible — "
            "skip any section that is not relevant to the content.\n\n"
            "If the screenshot contains:\n"
            "- Code or a terminal: focus on errors, logic issues, and what the code does\n"
            "- A UI or app: describe the state and any visible problems\n"
            "- A chart or diagram: describe what it shows and key values\n"
            "- Text or a document: summarize the key points\n\n"
            "Be concise and direct. No filler. Speak to an engineer.\n\n"
            f"User question: {user_question or 'What do you see?'}"
        )

    # ── Screenshot burst ───────────────────────────────────────────────────
    return (
        "You are the vision module for an AI personal assistant. "
        "The user is an engineer with a control systems background. "
        f"You are analyzing {path_count} screenshots taken within seconds of each other — "
        "treat them as a deliberate sequence.\n\n"
        "Focus on:\n"
        "- What changed between the screenshots\n"
        "- Whether the sequence shows a process, a problem evolving, or a before/after comparison\n"
        "- Any consistent issues visible across all screenshots\n\n"
        "Be concise and direct. No filler. Speak to an engineer.\n\n"
        f"User question: {user_question or 'What do you see across these screenshots?'}"
    )


def run_vision_pipeline(
    image_path: Union[str, List[str]],
    user_question: str = "",
    conversation_manager=None,
    authorized_user: bool = False,
    is_webcam: bool = False,
    context_hint: Optional[str] = None,    # ← NEW: from vision_awareness scorer
) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return "I can't look at images right now: vision needs an OpenAI API key (OPENAI_API_KEY)."
    paths = _to_path_list(image_path)

    prompt = _build_prompt(
        user_question=user_question,
        path_count=len(paths),
        authorized_user=authorized_user,
        is_webcam=is_webcam,
        context_hint=context_hint,
    )

    content = [{"type": "input_text", "text": prompt}]
    for p in paths:
        content.append({
            "type": "input_image",
            "image_url": image_to_data_url(p),
            "detail": "auto",
        })

    # Key comes from the OPENAI_API_KEY env var. Built at call time so Buddy still
    # starts without it (only vision needs it). Never hard-code the key here.
    resp = OpenAI().responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": content}],
    )
    return (resp.output_text or "").strip()