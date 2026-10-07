from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import re
import time

from brain.utils.context_guards import guards_enabled

# a tool result older than this many turns is no longer 'the last tool' as far as the LLM is told
_TOOL_RESULT_TTL_TURNS = 3


class Domain(str, Enum):
    TECHNICAL    = "technical"
    CASUAL       = "casual"
    VISION       = "vision"
    DEVICE       = "device"
    WEB          = "web"
    MEMORY_OPS   = "memory_ops"
    SYSTEM       = "system"
    UNKNOWN      = "unknown"


class Phase(str, Enum):
    EXPLORING    = "exploring"    # first turn on a topic
    DEEPENING    = "deepening"    # follow-up questions on same topic
    PIVOTING     = "pivoting"     # user is switching topic
    RETURNING    = "returning"    # user coming back to old topic
    CONFIRMING   = "confirming"   # yes/no/confirm response expected
    CLOSING      = "closing"      # wrapping up / saving


@dataclass
class ToolResult:
    tool:       str              # "webcam", "web_search", "recall", "schedule"
    success:    bool
    summary:    str = ""         # short human-readable description
    raw:        dict = field(default_factory=dict)
    turn:       int  = 0         # belief turn when it was reported (for expiry in the prompt block)


@dataclass
class BeliefState:
    # ── Core state ────────────────────────────────────────────────────────
    domain:           Domain = Domain.UNKNOWN
    phase:            Phase  = Phase.EXPLORING
    
    # ── Active entities (topic words with weight) ─────────────────────────
    # e.g. {"lidar": 0.9, "pointnet": 0.8, "voxel": 0.3}
    active_entities:  dict[str, float] = field(default_factory=dict)
    
    # ── Last tool result ──────────────────────────────────────────────────
    last_tool_result: Optional[ToolResult] = None
    
    # ── Open tasks (what is unresolved in this conversation) ─────────────
    open_tasks:       list[str] = field(default_factory=list)
    
    # ── Thread history (compressed, not raw) ─────────────────────────────
    thread_summary:   str = ""
    
    # ── Confidence in the current belief ─────────────────────────────────
    confidence:       float = 0.5
    
    # ── Timestamp of last update ──────────────────────────────────────────
    updated_at:       float = field(default_factory=time.time)

    # ── Turn counter (incremented once per update) ────────────────────────
    turn:             int = 0

    def to_prompt_block(self) -> str:
        """
        Serialize belief state into a compact string for LLM prompts.
        Keeps it short — this is injected into every classify call.
        """
        parts = [
            f"Domain: {self.domain.value}",
            f"Phase: {self.phase.value}",
        ]
        
        if self.active_entities:
            top = sorted(
                self.active_entities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:4]
            entities_str = ", ".join(f"{k}({v:.1f})" for k, v in top)
            parts.append(f"Active entities: {entities_str}")
        
        tool_is_fresh = (
            self.last_tool_result is not None
            and (not guards_enabled()
                 or self.turn - self.last_tool_result.turn <= _TOOL_RESULT_TTL_TURNS)
        )
        if tool_is_fresh:
            status = "succeeded" if self.last_tool_result.success else "FAILED"
            parts.append(
                f"Last tool: {self.last_tool_result.tool} → {status}"
                + (f" ({self.last_tool_result.summary})" 
                   if self.last_tool_result.summary else "")
            )
        
        if self.open_tasks:
            parts.append(f"Open questions: {'; '.join(self.open_tasks[:2])}")
        
        if self.thread_summary:
            parts.append(f"Thread: {self.thread_summary[:120]}")
        
        return "\n".join(parts)

    def is_vision_active(self) -> bool:
        return (
            self.domain == Domain.VISION
            or (
                self.last_tool_result is not None
                and self.last_tool_result.tool == "webcam"
            )
        )

    def is_technical(self) -> bool:
        return self.domain == Domain.TECHNICAL

    def webcam_failed(self) -> bool:
        return (
            self.last_tool_result is not None
            and self.last_tool_result.tool == "webcam"
            and not self.last_tool_result.success
        )
    

# ── Domain detection signals ──────────────────────────────────────────────────

_TECHNICAL_SIGNALS = {
    "lidar", "pointnet", "voxel", "cnn", "neural", "network", "pipeline",
    "classification", "preprocessing", "slam", "ekf", "pid", "mpc", "lqr",
    "ros", "gazebo", "sensor", "actuator", "controller", "trajectory",
    "sampling", "hierarchical", "occupancy", "grid", "point cloud",
    "convolutional", "architecture", "training", "inference", "model",
    "signal", "processing", "filter", "kalman", "encoder", "decoder",
    "robotic", "rover", "drone", "arm", "3d", "depth", "stereo",
}

_VISION_SIGNALS = {
    "look at", "see", "surroundings", "camera", "webcam", "what you saw",
    "what did you see", "around me", "my setup", "my workspace",
    "what i have", "equipment around", "what's around",
}

_CASUAL_SIGNALS = {
    "news", "weather", "time", "date", "today", "what happened",
    "tell me about", "how are you", "what do you think",
    "political", "economy", "sport",
}

_RETURN_SIGNALS = {
    "back to", "go back", "return to", "discuss back", "back again",
    "continue with", "resume", "back regarding", "let's go back",
    "i want to discuss back",
}

_PIVOT_SIGNALS = {
    "now i want", "let's talk about", "shift to", "different topic",
    "change topic", "actually", "forget that", "moving on",
    "new topic", "something else",
}

_CONFIRMING_SIGNALS = {
    "yes", "no", "yeah", "nope", "sure", "okay", "ok",
    "confirm", "cancel", "save it", "don't save",
}


class BeliefStateManager:
    """
    Maintains and updates BeliefState across conversation turns.
    Lives on ConversationManager as self.belief.
    """

    def __init__(self):
        self.state = BeliefState()
        self._thread_topic: Optional[str] = None   # normalized topic the current thread belongs to

    @staticmethod
    def _norm_topic(topic) -> Optional[str]:
        """'vision_awareness_1791117086' and 'vision_awareness' are the same thread."""
        t = re.sub(r"[\d_]+$", "", (topic or "").strip().lower())
        return t or None

    def _reset_thread(self):
        """Topic changed: entities/open questions/summary describe the OLD thread."""
        self.state.active_entities.clear()
        self.state.open_tasks.clear()
        self.state.thread_summary = ""

    def update(
        self,
        utterance: str,
        memory_manager,           # your existing MemoryManager
        last_tool_result: Optional[ToolResult] = None,
    ) -> BeliefState:
        """
        Call this at the START of handle_request, before any classification.
        Updates belief state in place and returns it.
        """
        text = (utterance or "").lower().strip()
        self.state.turn += 1
        prev_domain = self.state.domain
        prev_phase  = self.state.phase

        # ── Update tool result if provided ────────────────────────────────
        if last_tool_result is not None:
            last_tool_result.turn = self.state.turn
            self.state.last_tool_result = last_tool_result

        # ── Detect domain ─────────────────────────────────────────────────
        new_domain = self._detect_domain(text, memory_manager)
        
        # ── Detect phase ──────────────────────────────────────────────────
        new_phase = self._detect_phase(
            text, prev_domain, new_domain, prev_phase, memory_manager
        )

        # ── Context guard 2: don't carry the old thread into a new one ────
        # Reset on an explicit pivot or when the topic changed/cleared. No reset on
        # None -> topic: that is the same conversation getting its first topic.
        if guards_enabled():
            cur_topic = self._norm_topic(memory_manager.get_last_topic())
            topic_changed = self._thread_topic is not None and cur_topic != self._thread_topic
            if new_phase == Phase.PIVOTING or topic_changed:
                self._reset_thread()
            self._thread_topic = cur_topic

        # ── Update active entities ────────────────────────────────────────
        self._update_entities(text, new_domain, new_phase)

        # ── Update thread summary ─────────────────────────────────────────
        self._update_thread_summary(memory_manager)

        # ── Update open tasks ─────────────────────────────────────────────
        self._update_open_tasks(text, new_phase)

        # ── Commit ────────────────────────────────────────────────────────
        self.state.domain     = new_domain
        self.state.phase      = new_phase
        self.state.updated_at = time.time()
        self.state.confidence = self._compute_confidence(text, new_domain)

        return self.state

    def report_tool_result(
        self,
        tool: str,
        success: bool,
        summary: str = "",
        raw: dict = None,
    ):
        """
        Call this whenever a tool completes — webcam capture, web search,
        memory recall, schedule creation. Keeps belief state current.
        """
        self.state.last_tool_result = ToolResult(
            tool=tool,
            success=success,
            summary=summary,
            raw=raw or {},
            turn=self.state.turn,
        )

    def _detect_domain(self, text: str, mm) -> Domain:
        last_topic = (mm.get_last_topic() or "").lower()

        # Explicit vision — either active topic or vision words in text
        if last_topic.startswith("vision") or any(
            s in text for s in _VISION_SIGNALS
        ):
            return Domain.VISION

        # Active web session
        if last_topic.startswith("web:"):
            return Domain.WEB

        # Device context
        if last_topic.startswith("rover:") or any(
            w in text for w in ["rover", "drone", "tello", "take off", 
                                 "land", "connect", "arm"]
        ):
            return Domain.DEVICE

        # Technical — score by signal count
        tech_hits = sum(1 for s in _TECHNICAL_SIGNALS if s in text)
        if tech_hits >= 2:
            return Domain.TECHNICAL
        
        # Also inherit technical domain if we're mid-thread
        if self.state.domain == Domain.TECHNICAL and tech_hits >= 1:
            return Domain.TECHNICAL

        # Casual
        casual_hits = sum(1 for s in _CASUAL_SIGNALS if s in text)
        if casual_hits >= 1:
            return Domain.CASUAL

        # Memory operations
        if any(w in text for w in [
            "save", "store", "remember", "recall", "delete", "forget",
            "what did we", "what have we"
        ]):
            return Domain.MEMORY_OPS

        # System
        if any(w in text for w in [
            "time", "date", "battery", "weather", "location", "cpu"
        ]):
            return Domain.SYSTEM

        # Inherit previous if no strong signal
        # (guard 2: only for short messages - a long self-contained question carries its own signal)
        if self.state.domain != Domain.UNKNOWN and (
            not guards_enabled() or len(text.split()) <= 6
        ):
            return self.state.domain

        return Domain.UNKNOWN

    def _detect_phase(
        self,
        text: str,
        prev_domain: Domain,
        new_domain: Domain,
        prev_phase: Phase,
        mm,
    ) -> Phase:

        # Explicit return signal
        if any(s in text for s in _RETURN_SIGNALS):
            return Phase.RETURNING

        # Explicit pivot signal
        if any(s in text for s in _PIVOT_SIGNALS):
            return Phase.PIVOTING

        # Confirming — short yes/no when pending state exists
        if any(s in text for s in _CONFIRMING_SIGNALS) and len(text.split()) <= 4:
            if mm.get_pending_topic():
                return Phase.CONFIRMING

        # Domain switch = pivoting
        if (prev_domain != Domain.UNKNOWN 
                and new_domain != Domain.UNKNOWN
                and prev_domain != new_domain):
            return Phase.PIVOTING

        # Same domain, already exploring = deepening
        if prev_domain == new_domain and prev_phase in (
            Phase.EXPLORING, Phase.DEEPENING
        ):
            return Phase.DEEPENING

        return Phase.EXPLORING

    def _update_entities(
        self, text: str, domain: Domain, phase: Phase
    ):
        """
        Extract meaningful terms from utterance and update entity weights.
        Decay existing entities slightly, boost mentioned ones.
        """
        STOPWORDS = {
            "the", "a", "an", "of", "in", "on", "at", "for", "to", "from",
            "about", "what", "how", "why", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "would", "should",
            "i", "you", "we", "my", "your", "our", "it", "this", "that",
            "and", "or", "but", "so", "with", "than", "more", "less",
        }

        # Decay existing entities
        for k in list(self.state.active_entities.keys()):
            self.state.active_entities[k] *= 0.85
            if self.state.active_entities[k] < 0.1:
                del self.state.active_entities[k]

        # Extract new terms (3+ chars, not stopwords)
        words = re.findall(r"[a-z0-9+]+", text)
        meaningful = [
            w for w in words 
            if len(w) >= 3 and w not in STOPWORDS
        ]

        # Boost mentioned terms
        for word in meaningful:
            current = self.state.active_entities.get(word, 0.0)
            # Technical terms get higher boost
            boost = 0.4 if word in _TECHNICAL_SIGNALS else 0.25
            self.state.active_entities[word] = min(1.0, current + boost)

        # Cap at 12 entities
        if len(self.state.active_entities) > 12:
            sorted_e = sorted(
                self.state.active_entities.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.state.active_entities = dict(sorted_e[:12])

    def _update_thread_summary(self, mm):
        """
        Keep a short compressed summary of the active thread.
        Uses top entity names — no LLM call needed.
        """
        last_topic = (mm.get_last_topic() or "").strip()
        if not last_topic:
            if guards_enabled():
                self.state.thread_summary = ""   # no topic -> no "Thread:" line from the previous one
            return

        top_entities = sorted(
            self.state.active_entities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        entity_str = ", ".join(k for k, _ in top_entities)
        if entity_str:
            self.state.thread_summary = (
                f"Discussing {last_topic}: {entity_str}"
            )
        else:
            self.state.thread_summary = f"Discussing {last_topic}"

    def _update_open_tasks(self, text: str, phase: Phase):
        """
        Track unresolved questions as open tasks.
        Simple heuristic — question marks and "how/what/why" starters.
        """
        if "?" in text and phase in (Phase.EXPLORING, Phase.DEEPENING):
            # Extract the question as a short task
            q = text.strip().rstrip("?")[:60]
            if q not in self.state.open_tasks:
                self.state.open_tasks.append(q)
                # Keep only last 3
                self.state.open_tasks = self.state.open_tasks[-3:]

        # Clear open tasks when confirming or closing
        if phase in (Phase.CONFIRMING, Phase.CLOSING):
            self.state.open_tasks.clear()

    def _compute_confidence(self, text: str, domain: Domain) -> float:
        if domain == Domain.UNKNOWN:
            return 0.3
        # More words = more signal = more confident
        word_count = len(text.split())
        base = 0.5
        length_bonus = min(0.3, word_count * 0.02)
        domain_bonus = 0.2 if domain != Domain.UNKNOWN else 0.0
        return min(1.0, base + length_bonus + domain_bonus)