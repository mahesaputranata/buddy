"""
brain/utils/context_guards.py

Guards against "context bleed": Buddy answering a new question with an OLD topic's context.
Found with the alive stress test's --trace-prompts: a question about RL-vs-imitation-learning
was answered about Nav2/MPPI because the prompt carried the previous scenario's exchanges
(a global recent-exchange buffer), a stale belief block, and a poisoned memory entry, plus a
template line telling a 3B model to "connect the ideas smoothly".

One switch for everything:   BUDDY_CONTEXT_GUARDS=0   turns all four guards off (A/B testing).

The four guards:
  1. recent_context_is_relevant / select_recent_context
        the "Last Interaction Context" buffer is only shown when the message is a real
        follow-up or is similar to the buffer.                      (used by brain.generate_llm_response)
  2. belief-state resets                                             (see brain/belief_state.py)
  3. should_store_exchange / note_last_reply
        a bare "yes"/"ok" that answers nothing is not stored as topic memory, so Buddy's
        improvised reply to it can't poison the topic.              (used by brain.store_short_term_memory)
  4. reworded answer-template instructions                           (see brain/core/prompts.py)

Kept dependency-free (no torch / chroma) so it is trivially unit-testable.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional, Sequence

ENV_SWITCH = "BUDDY_CONTEXT_GUARDS"

# Minimum cosine similarity between the user's message and the recent-exchange buffer for the
# buffer to be shown to the LLM when the message is NOT itself a follow-up. Calibrated on the
# traced stress run (see the calibration notes in the commit/summary).
RELEVANCE_THRESHOLD = 0.45  # foreign buffers scored up to 0.43 (all topics are robotics); real follow-ups pass via is_followup_like instead


def guards_enabled() -> bool:
    """Read at call time so it can be flipped per process without code changes."""
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in ("0", "false", "off", "no")


# ─────────────────────────────────────────────────────────────────────────────
# Guard 1 — only show the recent-exchange buffer when it is relevant
# ─────────────────────────────────────────────────────────────────────────────

try:  # reuse the project's follow-up patterns (pure regex module)
    from brain.conversation.followup_detector import _SHORT_FOLLOWUP_RE, _OPINION_QUESTION_RE
except Exception:  # pragma: no cover - keeps this module importable on its own
    _SHORT_FOLLOWUP_RE = re.compile(r"^\s*(?:yes|yeah|yep|sure|no|nope|okay|ok|go\s+on|continue|tell\s+me\s+more)\b", re.I)
    _OPINION_QUESTION_RE = re.compile(r"^\s*(?:in\s+your\s+opinion|what\s+do\s+you\s+think|should\s+i)\b", re.I)

# "is it X", "how does that work" — anaphoric questions (NOT i/we/you: "can you explain X" is a fresh ask)
_PRONOUN_QUESTION_RE = re.compile(
    r"^\s*(?:(?:what|how|why|where|when|who|which)\s+)?"
    r"(?:(?:is|are|was|were|can|could|would|should|do|does|did|will)\s+)?"
    r"(?:it|this|that|they|them|those|these)\b",
    re.IGNORECASE,
)
_ANAPHORA = {"it", "that", "this", "them", "those", "these", "again", "more", "why", "how"}
_BACKREF_RE = re.compile(
    r"\b(?:back\s+to|as\s+i\s+(?:said|mentioned)|you\s+(?:said|mentioned)|earlier|that\s+one|"
    r"the\s+same|like\s+before|again)\b",
    re.IGNORECASE,
)


def is_followup_like(text: str) -> bool:
    """
    True if the message plainly depends on what was just said ("yes", "how does it handle
    obstacles though", "back to what we were discussing"). A long, self-contained question is
    NOT a follow-up even if it starts with "can you explain".
    """
    text = (text or "").strip()
    if not text:
        return False
    words = re.sub(r"[^a-z' ]", " ", text.lower()).split()
    n = len(text.split())
    # <=3 words only counts if it is an acknowledgement or has a pronoun to resolve;
    # "activate the rover" is a fresh command, not a follow-up.
    if n <= 3 and (is_bare_ack(text) or any(w in _ANAPHORA for w in words) or _SHORT_FOLLOWUP_RE.match(text)):
        return True
    if n <= 3:
        return False
    if _PRONOUN_QUESTION_RE.match(text) or _BACKREF_RE.search(text):
        return True
    if n <= 7 and (_SHORT_FOLLOWUP_RE.match(text) or _OPINION_QUESTION_RE.match(text)):
        return True
    return False


_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")


def _token_overlap(a: str, b: str) -> float:
    """Fallback similarity when no embedding function is available (0..1)."""
    ta, tb = set(_TOKEN_RE.findall(a.lower())), set(_TOKEN_RE.findall(b.lower()))
    return len(ta & tb) / max(len(ta), 1)


def recent_context_is_relevant(
    user_text: str,
    exchanges: Sequence[str],
    similarity_fn: Optional[Callable[[str, str], float]] = None,
    threshold: float = RELEVANCE_THRESHOLD,
) -> bool:
    if not exchanges:
        return False
    if is_followup_like(user_text):
        return True
    try:
        # best single exchange: one relevant exchange must not be diluted by two unrelated ones
        sims = [similarity_fn(user_text, ex) if similarity_fn else _token_overlap(user_text, ex)
                for ex in exchanges]
        return max(sims) >= threshold
    except Exception:
        return True  # if scoring breaks, fail OPEN: old behaviour, never silently lose context


def select_recent_context(
    user_text: str,
    exchanges: Sequence[str],
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> list:
    """The buffer to hand the LLM: all of it when guards are off or it is relevant, else none."""
    exchanges = list(exchanges)
    if not exchanges or not guards_enabled():
        return exchanges
    return exchanges if recent_context_is_relevant(user_text, exchanges, similarity_fn) else []


# ─────────────────────────────────────────────────────────────────────────────
# Guard 3 — don't store a bare acknowledgement that answers nothing
# ─────────────────────────────────────────────────────────────────────────────

_ACK_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "right", "no", "nope", "nah",
    "thanks", "thank", "you", "go", "ahead", "please", "do", "it", "fine", "cool", "great", "sounds", "good",
}

_last_reply = ""


def note_last_reply(text: str) -> None:
    """Called with each reply Buddy sends, so the NEXT turn knows whether a question was open."""
    global _last_reply
    _last_reply = text or ""


def last_reply_asked_question() -> bool:
    return "?" in _last_reply[-240:]


def is_bare_ack(text: str) -> bool:
    words = re.sub(r"[^a-z' ]", " ", (text or "").lower()).split()
    return 0 < len(words) <= 3 and all(w in _ACK_WORDS for w in words)


def should_store_exchange(user_input: str) -> bool:
    """
    A bare "yes" is only meaningful as an answer to a question. If Buddy's last reply asked
    nothing (e.g. "Got it - saved ... to long-term memory."), the "yes" answers nothing and
    Buddy's improvised reply to it ("let's explore DDPG baselines...") must not be stored as
    topic memory, where it would later be retrieved and continued by the LLM.
    """
    if not guards_enabled():
        return True
    return not (is_bare_ack(user_input) and not last_reply_asked_question())
