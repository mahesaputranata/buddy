"""
brain/intent_model/formatting.py

Single source of truth for the text the intent classifier sees. Used by BOTH
prepare_training_data.py (training) and shadow_classifier.py (serving) so the two
can never drift apart — train/serve skew here silently ruins accuracy.

All context fields are the state BEFORE the current turn ran:
  - last_topic:     what mm.get_last_topic() returned before handle_request did anything.
                    NOTE: session_logger writes last_topic AFTER the turn ran (35% of turns
                    change it themselves, e.g. a web search sets "web:..."), so when building
                    training data use the PREVIOUS turn's logged last_topic, never the
                    current turn's.
  - pending_before: pending topic at the start of the turn (logged correctly already).
  - prev_*:         previous user input / Buddy response in the same session.

Format:
    [TOPIC: <last_topic|NONE>] [PENDING: <pending|NONE>] [PREV: <input> -> <response[:200]>] [INPUT: <input>]
"""

import re


def clean_mojibake(text: str) -> str:
    """Best-effort cleanup of U+FFFD characters that an upstream encoding bug left in the logs."""
    if not text:
        return text
    text = re.sub(r"(\w)�(t|s|re|ll|ve|d|m)\b", r"\1'\2", text)  # wasn?t -> wasn't
    text = re.sub(r"\s�\s", " — ", text)                         # " ? " -> " — "
    return text.replace("�", "")                                  # no confident guess: drop


def build_input_text(
    user_input: str,
    last_topic=None,
    pending_before=None,
    prev_input=None,
    prev_response=None,
) -> str:
    topic = last_topic or "NONE"
    pending = pending_before or "NONE"

    if prev_input is not None or prev_response is not None:
        prev_part = (
            f"[PREV: {clean_mojibake(prev_input or '')} -> "
            f"{clean_mojibake(prev_response or '')[:200]}]"
        )
    else:
        prev_part = "[PREV: NONE]"

    return f"[TOPIC: {topic}] [PENDING: {pending}] {prev_part} [INPUT: {clean_mojibake(user_input or '')}]"
