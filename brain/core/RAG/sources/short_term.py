# brain/retrieval/sources/short_term.py
from __future__ import annotations
from typing import List
from brain.core.RAG.types import ContextItem, RetrievalOptions
from brain.core.RAG.sources.base import RetrievalSource

class ShortTermSource(RetrievalSource):
    name = "short_term"

    def __init__(self, temp_topic_memory: dict):
        self.temp_topic_memory = temp_topic_memory

    def retrieve(self, query: str, options: RetrievalOptions) -> List[ContextItem]:
        topic = (options.topic_hint or "").strip().lower()
        if not topic:
            return []

        convo = self.temp_topic_memory.get(topic, [])
        if not convo:
            return []

        # Keep it small: last N turns only (tune)
        tail = convo[-8:]
        text = "\n".join(tail)

        return [ContextItem(
            source="short_term",
            title=f"Short-term context: {topic}",
            text=text,
            score=1.0,
            meta={"topic": topic, "turns": len(tail)},
        )]