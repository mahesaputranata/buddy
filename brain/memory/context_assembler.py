from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from brain.core.brain import embedding_model
            _embedding_model = embedding_model
        except Exception as e:
            print(f"[ContextAssembler] WARNING: could not load embedding_model: {e}")
            _embedding_model = None
    return _embedding_model


# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum cosine similarity for a group to be included
SIMILARITY_THRESHOLD = 0.32

# Maximum number of groups to include in one context block
MAX_GROUPS = 3

# Maximum characters taken from any single group
# (keeps the oldest groups from flooding the context window)
MAX_CHARS_PER_GROUP = 1200

# Groups whose keys start with these prefixes are treated differently
SESSION_PREFIXES = ("session:", )
WEB_PREFIXES     = ("web:",     )
VISION_PREFIXES  = ("vision",   )
ROVER_PREFIXES   = ("rover:",   )

# Topics that are too generic to be useful as context anchors
GENERIC_TOPICS = {
    "current_conversation", "general", "chat",
    "unknown", "none", "null", "",
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GroupMatch:
    topic:    str
    score:    float
    text:     str              # trimmed context text
    is_current: bool = False   # whether this is the active last_topic


@dataclass
class AssembledContext:
    groups:       list[GroupMatch] = field(default_factory=list)
    merged_text:  str = ""
    topic_labels: list[str] = field(default_factory=list)  # for debug / logging

    @property
    def is_multi_group(self) -> bool:
        return len(self.groups) > 1

    @property
    def found_something(self) -> bool:
        return bool(self.merged_text.strip())


# ── Core assembler ────────────────────────────────────────────────────────────

class ContextAssembler:
    """
    Scores every group in temp_topic_memory against the incoming user input
    and returns the top-k most relevant groups merged into one context string.

    Usage:
        assembler = ContextAssembler(temp_topic_memory)
        ctx = assembler.build(user_input, current_topic="robotic arm")
        short_term_str = ctx.merged_text
    """

    def __init__(
        self,
        topic_memory: dict,                     # temp_topic_memory dict
        threshold:    float = SIMILARITY_THRESHOLD,
        max_groups:   int   = MAX_GROUPS,
        max_chars:    int   = MAX_CHARS_PER_GROUP,
    ):
        self.topic_memory = topic_memory
        self.threshold    = threshold
        self.max_groups   = max_groups
        self.max_chars    = max_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        user_input:     str,
        current_topic:  str,
        include_session: bool = False,
    ) -> AssembledContext:
        """
        Main entry point.

        Args:
            user_input:      The raw user utterance.
            current_topic:   The active last_topic from memory_manager.
            include_session: Whether to include session:* seed entries.

        Returns:
            AssembledContext with .merged_text ready to feed into RagContext.
        """
        current_topic = (current_topic or "").strip().lower()
        user_input    = (user_input    or "").strip()

        if not user_input:
            return self._fallback(current_topic)

        model = _get_embedding_model()
        if model is None:
            # No embedding model available — fall back to keyword scoring
            return self._keyword_build(user_input, current_topic, include_session)

        return self._embedding_build(user_input, current_topic, include_session, model)

    # ── Embedding-based scoring ───────────────────────────────────────────────

    def _embedding_build(
        self,
        user_input:     str,
        current_topic:  str,
        include_session: bool,
        model,
    ) -> AssembledContext:
        try:
            from sentence_transformers import util as st_util
            import numpy as np

            emb_input = model.encode(user_input, convert_to_numpy=True)
        except Exception as e:
            print(f"[ContextAssembler] embedding encode failed: {e}")
            return self._keyword_build(user_input, current_topic, include_session)

        scored: list[GroupMatch] = []

        for topic, exchanges in self.topic_memory.items():
            if not exchanges:
                continue

            # Filter by prefix rules
            if self._should_skip(topic, include_session):
                continue

            context_text = self._join_exchanges(exchanges)
            if not context_text.strip():
                continue

            try:
                emb_ctx = model.encode(context_text, convert_to_numpy=True)
                score   = float(st_util.cos_sim(emb_input, emb_ctx))
            except Exception:
                score = 0.0

            is_current = (topic == current_topic)

            # Always include the current topic — boost its score if needed
            if is_current:
                score = max(score, self.threshold + 0.25)

            if score >= self.threshold:
                trimmed = self._trim(context_text)
                scored.append(GroupMatch(
                    topic=topic,
                    score=score,
                    text=trimmed,
                    is_current=is_current,
                ))

        return self._assemble(scored, current_topic)

    # ── Keyword fallback scoring ──────────────────────────────────────────────

    def _keyword_build(
        self,
        user_input:     str,
        current_topic:  str,
        include_session: bool,
    ) -> AssembledContext:
        """
        Pure keyword overlap scoring — used when sentence-transformers
        is unavailable. Less accurate but always works.
        """
        input_words = self._tokenize(user_input)
        scored: list[GroupMatch] = []

        for topic, exchanges in self.topic_memory.items():
            if not exchanges:
                continue
            if self._should_skip(topic, include_session):
                continue

            context_text = self._join_exchanges(exchanges)
            topic_words  = self._tokenize(topic + " " + context_text[:300])

            overlap = len(input_words & topic_words)
            total   = max(len(topic_words), 1)
            score   = overlap / total

            is_current = (topic == current_topic)
            if is_current:
                score = max(score, self.threshold + 0.01)

            if score >= self.threshold or is_current:
                trimmed = self._trim(context_text)
                scored.append(GroupMatch(
                    topic=topic,
                    score=score,
                    text=trimmed,
                    is_current=is_current,
                ))

        return self._assemble(scored, current_topic)

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _assemble(
        self,
        scored:        list[GroupMatch],
        current_topic: str,
    ) -> AssembledContext:
        if not scored:
            return self._fallback(current_topic)

        scored.sort(key=lambda g: (not g.is_current, -g.score))
        top = scored[:self.max_groups]

        # ── Cap background scores below primary so LLM sees clear priority ──
        primary_score = next((g.score for g in top if g.is_current), 0.0)
        for g in top:
            if not g.is_current and g.score >= primary_score:
                g.score = primary_score - 0.05

        blocks = []
        labels = []

        for g in top:
            if g.is_current:
                role = "PRIMARY TOPIC"
                instruction = ""
            else:
                role = "BACKGROUND CONTEXT"
                instruction = "\n(Reference only — do not answer as if this is the current topic)"
                if len(g.text) > 600:
                    g.text = "..." + g.text[-600:]

            label = f"[{role}: {g.topic} | similarity: {g.score:.2f}]{instruction}"
            blocks.append(f"{label}\n{g.text}")
            labels.append(g.topic)

        if len(top) > 1:
            primary = next((g for g in top if g.is_current), None)
            if primary:
                print(
                    f"[ContextAssembler] multi-group active: "
                    f"current='{primary.topic}' + "
                    f"related={[g.topic for g in top if not g.is_current]}"
                )

        merged = "\n\n---\n\n".join(blocks)

        return AssembledContext(
            groups=top,
            merged_text=merged,
            topic_labels=labels,
        )

    def _fallback(self, current_topic: str) -> AssembledContext:
        """Returns only the current topic, no scoring."""
        exchanges = self.topic_memory.get(current_topic, [])
        if not exchanges:
            return AssembledContext()

        text = self._trim(self._join_exchanges(exchanges))
        group = GroupMatch(
            topic=current_topic,
            score=1.0,
            text=text,
            is_current=True,
        )
        return AssembledContext(
            groups=[group],
            merged_text=text,
            topic_labels=[current_topic],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _should_skip(self, topic: str, include_session: bool) -> bool:
        """Returns True if this topic should be excluded from scoring."""
        t = (topic or "").strip().lower()

        if t in GENERIC_TOPICS:
            return True

        if re.match(r"^topic_\d+$", t):
            return True

        if not include_session and t.startswith(SESSION_PREFIXES):
            return True

        # Web, vision, rover topics have their own dedicated flows —
        # don't bleed them into general context assembly
        if t.startswith(WEB_PREFIXES):
            return True
        if t.startswith(VISION_PREFIXES):
            return True
        if t.startswith(ROVER_PREFIXES):
            return True

        return False

    def _join_exchanges(self, exchanges: list) -> str:
        """Joins a list of exchange strings into one block."""
        if not exchanges:
            return ""
        return "\n".join(str(e) for e in exchanges if e)

    def _trim(self, text: str) -> str:
        """
        Keeps the text within max_chars.
        Takes the MOST RECENT content (tail of string)
        so older stale context doesn't crowd the window.
        """
        if not text:
            return ""
        if len(text) <= self.max_chars:
            return text
        # Take the tail — most recent exchanges
        return "..." + text[-self.max_chars:]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple tokenizer for keyword scoring fallback."""
        STOP = {
            "the", "a", "an", "of", "in", "on", "at", "for", "to",
            "from", "about", "what", "how", "is", "are", "was",
            "were", "i", "my", "me", "you", "your", "we", "our",
            "this", "that", "it", "and", "or", "but", "so",
            "with", "have", "has", "had", "do", "does", "did",
            "can", "could", "would", "should", "will",
        }
        words = re.findall(r"[a-z0-9]{3,}", text.lower())
        return {w for w in words if w not in STOP}


# ── Module-level convenience function ─────────────────────────────────────────
def retrieve_relevant_groups(
    user_input:     str,
    current_topic:  str,
    topic_memory:   dict,
    top_k:          int   = MAX_GROUPS,
    threshold:      float = SIMILARITY_THRESHOLD,
) -> str:
    """
    Convenience wrapper around ContextAssembler.

    Returns a merged context string ready to pass into RagContext.short_term.

    Example:
        short_term_str = retrieve_relevant_groups(
            user_input    = raw_input,
            current_topic = last_topic,
            topic_memory  = temp_topic_memory,
        )
    """
    assembler = ContextAssembler(
        topic_memory = topic_memory,
        threshold    = threshold,
        max_groups   = top_k,
    )
    result = assembler.build(user_input, current_topic)
    return result.merged_text
