"""
vision/vision_awareness.py

Vision Awareness Scorer — decides whether Buddy should use the webcam
proactively, ask permission, or stay blind.

Thresholds:
    score >= 0.70  →  SILENT  (grab frame, inject naturally, no mention of camera)
    0.40 <= s < 0.70 →  ASK    (offer to look)
    score < 0.40   →  IGNORE  (do nothing)

Usage:
    from vision.vision_awareness import score_vision_need, VisionDecision

    result = score_vision_need(user_text, context)
    if result.decision == VisionDecision.SILENT:
        # grab frame, pass context_hint to run_vision_pipeline
    elif result.decision == VisionDecision.ASK:
        # offer camera to user
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VisionDecision(Enum):
    SILENT = "silent"   # grab silently, inject naturally
    ASK    = "ask"      # offer camera first
    IGNORE = "ignore"   # don't use camera


@dataclass
class VisionAwarenessResult:
    decision:     VisionDecision
    score:        float
    context_hint: str               # injected into run_vision_pipeline prompt
    reason:       str               # debug — why this score
    signals:      list[str] = field(default_factory=list)


# ── Thresholds ────────────────────────────────────────────────────────────────
SILENT_THRESHOLD = 0.70
ASK_THRESHOLD    = 0.40


# ── Signal definitions ────────────────────────────────────────────────────────

# Strong physical pointing — user is almost certainly describing something
# visible in front of them. Silent grab is justified.
_PHYSICAL_POINTING = [
    r"\b(look at|looking at|see this|see here|here is|here's|right here)\b",
    r"\b(on the table|on my desk|in front of me|next to me|beside me)\b",
    r"\b(my setup|my workspace|my station|my equipment)\b",
    r"\bthis (thing|device|setup|unit|board|cable|module|rover|drone|robot)\b",
    r"\b(can you see|take a look|have a look|glance at)\b",
    r"\blook at this\b",
    r"\bthis thing here\b",
    r"\bright here\b",
]

# Troubleshooting physical cause — user is explaining why something
# failed with a physical reason. High relevance for rover/device flows.
_PHYSICAL_TROUBLESHOOT = [
    r"\b(haven'?t (started|turned|powered|plugged|connected|switched))\b",
    r"\b(not (on|running|connected|powered|plugged in|turned on))\b",
    r"\b(it'?s (off|dead|unplugged|disconnected|not on))\b",
    r"\b(forgot to|didn'?t (turn|start|plug|connect|power))\b",
    r"\b(cable|wire|power|battery|charger)\b.{0,30}\b(off|dead|missing|not|loose)\b",
    r"\b(physically|hardware|the actual|the real)\b",
    r"\b(not booting|won'?t boot|keeps (failing|crashing)|boot(ing)? issue)\b",
    r"\b(keep(s)? trying|tried (again|multiple)|attempted multiple)\b",
    r"\bhaven'?t (yet\s+)?(started|turned on|powered|booted)\b",
    r"\bnot (yet\s+)?(started|on|running|powered|booted)\b",
    r"\bjust because\b.{0,30}\b(haven'?t|not yet|didn'?t)\b",
]

# Demonstrative + spatial — "it's like this", "this one here", pointing language
# without explicit visual triggers
_DEMONSTRATIVE_SPATIAL = [
    r"\b(like this|this way|this one|that one|those ones)\b",
    r"\b(over here|over there|right there|just there)\b",
    r"\b(in this|at this|from this)\b.{0,20}\b(angle|position|spot|place|side)\b",
]

# Situational physical context — medium confidence, enough to ask
_SITUATIONAL_PHYSICAL = [
    r"\b(my (rover|robot|drone|device|board|sensor|camera|hardware))\b",
    r"\b(the (rover|robot|drone|device|board|sensor))\b.{0,20}\b(is|was|has|have)\b",
    r"\b(just (placed|put|set|left|moved|built|assembled))\b",
    r"\b(wiring|circuit|connection|port|pin|plug)\b",
]

# Context boosters — these alone don't trigger but boost score
# when combined with above signals
_CONTEXT_BOOSTERS = [
    r"\b(because|since|the reason|that'?s why|which is why)\b",
    r"\b(explain|showing|demonstrating|pointing)\b",
    r"\b(actually|literally|physically|in person|irl)\b",
]

# Hard suppressors — these strongly suggest NO camera needed
_SUPPRESSORS = [
    r"\b(abstract|theoretical|in theory|hypothetically|imagine|conceptually)\b",
    r"\b(search|look up|google|find online|web|internet)\b",
    r"\b(remember|recall|memory|schedule|calendar|reminder)\b",
    r"\b(what time|what date|weather|battery|cpu|location)\b",
    r"\b(tell me about|explain to me|how does|what is|what are)\b.{0,30}\?$",
]


def _match_patterns(text: str, patterns: list[str]) -> list[str]:
    """Returns list of matched pattern strings."""
    matched = []
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            matched.append(p)
    return matched


def _build_context_hint(
    user_text: str,
    last_topic: str,
    pending_state: str,
    failure_context: Optional[str],
) -> str:
    """
    Builds a natural language hint injected into the vision prompt
    so GPT-4o knows what to look for.
    """
    parts = []

    # Device/rover context
    if last_topic and "rover" in last_topic.lower():
        parts.append("The user is working with a rover robot")
        if failure_context:
            parts.append(f"There was a connection issue: {failure_context}")
        parts.append(
            "Look at the physical setup — is the rover powered on? "
            "Are there any visible issues like loose cables, "
            "missing connections, or the device being off?"
        )

    elif last_topic and "tello" in last_topic.lower():
        parts.append("The user is working with a Tello drone")
        parts.append(
            "Look at the physical setup — check if the drone is powered, "
            "any visible damage, or environmental issues."
        )

    # Troubleshooting context
    elif pending_state and "explain" in pending_state.lower():
        parts.append(
            "The user is explaining a physical issue. "
            "Describe what you see on the desk or workspace. "
            "Note any devices, their power state, and anything relevant."
        )

    # Generic physical context
    else:
        parts.append(
            "The user is referring to something physical in their environment. "
            "Describe what you see — focus on any devices, setups, or objects "
            "that seem relevant to what they're describing."
        )

    return " ".join(parts)


def score_vision_need(
    user_text:      str,
    last_topic:     str = "",
    pending_state:  str = "",
    recent_turns:   list[str] | None = None,
    failure_context: Optional[str] = None,
) -> VisionAwarenessResult:
    """
    Main scorer. Returns a VisionAwarenessResult with decision + context hint.

    Args:
        user_text:       The user's current utterance
        last_topic:      mm.get_last_topic() — e.g. "rover:session"
        pending_state:   mm.get_pending_topic() — e.g. "Awaiting_RoverContext_Explain"
        recent_turns:    Last 2-3 turns from sensory memory
        failure_context: Error message from a failed device operation, if any
    """
    text = (user_text or "").strip()
    score = 0.0
    signals: list[str] = []

    # ── Hard suppressor check — exit early ───────────────────────────────
    suppressor_hits = _match_patterns(text, _SUPPRESSORS)
    if suppressor_hits:
        # Only suppress if no strong physical pointing overrides it
        pointing_hits = _match_patterns(text, _PHYSICAL_POINTING)
        if not pointing_hits:
            return VisionAwarenessResult(
                decision=VisionDecision.IGNORE,
                score=0.0,
                context_hint="",
                reason="suppressed by abstract/non-physical signals",
                signals=suppressor_hits,
            )

    # ── Signal scoring ────────────────────────────────────────────────────

    # Physical pointing — strongest signal (+0.45 per hit, cap at 0.55)
    pointing = _match_patterns(text, _PHYSICAL_POINTING)
    if pointing:
        score += min(0.55 * len(pointing), 0.65)
        signals.extend([f"pointing:{p}" for p in pointing])

    if re.search(r"\bright here\b|\bover here\b|\bjust here\b", text, re.IGNORECASE):
        score += 0.12
        signals.append("deictic:here_present")

    # Physical troubleshoot — strong signal (+0.35 per hit, cap at 0.45)
    troubleshoot = _match_patterns(text, _PHYSICAL_TROUBLESHOOT)
    if troubleshoot:
        score += min(0.35 * len(troubleshoot), 0.45)
        signals.extend([f"troubleshoot:{p}" for p in troubleshoot])

    # Demonstrative spatial — medium signal (+0.20 per hit, cap at 0.25)
    demonstrative = _match_patterns(text, _DEMONSTRATIVE_SPATIAL)
    if demonstrative:
        score += min(0.20 * len(demonstrative), 0.25)
        signals.extend([f"demonstrative:{p}" for p in demonstrative])

    # Situational physical — medium signal (+0.18 per hit, cap at 0.22)
    situational = _match_patterns(text, _SITUATIONAL_PHYSICAL)
    if situational:
        score += min(0.18 * len(situational), 0.22)
        signals.extend([f"situational:{p}" for p in situational])

    # Context boosters — only add if base score is already non-zero
    if score > 0.10:
        boosters = _match_patterns(text, _CONTEXT_BOOSTERS)
        if boosters:
            score += min(0.08 * len(boosters), 0.15)
            signals.extend([f"booster:{p}" for p in boosters])

    # ── Context-based boosts ──────────────────────────────────────────────

    # Boost if we're in a device/rover context — physical explanation is likely
    if last_topic:
        lt = last_topic.lower()
        if "rover" in lt or "tello" in lt or "drone" in lt or "robot" in lt:
            score += 0.20
            signals.append("context:device_topic_active")

    # Boost if there was a recent failure — user explaining physical cause
    if failure_context:
        score += 0.25
        signals.append("context:failure_just_occurred")

    # Boost if pending state is a repair/explain state
    if pending_state and any(k in pending_state.lower() for k in
                              ("explain", "repair", "context", "reason")):
        score += 0.18
        signals.append("context:pending_explain_state")

    # Boost if recent turns contain device failures or physical references
    if recent_turns:
        recent_str = " ".join(recent_turns[-2:]).lower()
        if any(k in recent_str for k in
               ("connection failed", "lost contact", "couldn't reach",
                "not powered", "turned off", "not started")):
            score += 0.15
            signals.append("context:recent_failure_in_memory")

    # ── Cap and threshold ─────────────────────────────────────────────────
    score = round(min(score, 1.0), 3)

    context_hint = _build_context_hint(
        user_text, last_topic, pending_state, failure_context
    )

    if score >= SILENT_THRESHOLD:
        decision = VisionDecision.SILENT
        reason = f"score={score:.2f} — physical signals strong enough for silent grab"
    elif score >= ASK_THRESHOLD:
        decision = VisionDecision.ASK
        reason = f"score={score:.2f} — physical context present, asking permission"
    else:
        decision = VisionDecision.IGNORE
        context_hint = ""
        reason = f"score={score:.2f} — insufficient physical signal"

    return VisionAwarenessResult(
        decision=VisionDecision(decision.value),
        score=score,
        context_hint=context_hint,
        reason=reason,
        signals=signals,
    )


# ── Offer text helpers ────────────────────────────────────────────────────────

_ASK_PHRASES_ROVER = [
    "Want me to take a look at your setup?",
    "Should I check what I can see on your end?",
    "Want me to have a quick look at the rover?",
]

_ASK_PHRASES_GENERIC = [
    "Want me to take a look?",
    "Should I check what's in front of you?",
    "Want me to see what you're working with?",
]

import random

def get_camera_offer_text(last_topic: str = "") -> str:
    """Returns a natural offer phrase based on context."""
    lt = (last_topic or "").lower()
    if "rover" in lt or "tello" in lt or "drone" in lt or "robot" in lt:
        return random.choice(_ASK_PHRASES_ROVER)
    return random.choice(_ASK_PHRASES_GENERIC)


# ── Silent injection prefix ───────────────────────────────────────────────────

_SILENT_PREFIXES = [
    "Based on what I can see, ",
    "Looking at your setup, ",
    "From what I'm seeing, ",
    "I can see that ",
    "Taking a look at your end — ",
]

def get_silent_prefix() -> str:
    """Returns a natural prefix for silent vision injection."""
    return random.choice(_SILENT_PREFIXES)