# brain/retrieval/sources/web.py
from __future__ import annotations
from typing import List
from brain.core.RAG.types import ContextItem, RetrievalOptions
from brain.core.RAG.sources.base import RetrievalSource

class WebContextSource(RetrievalSource):
    name = "web"

    def __init__(self, memory_manager):
        self.mm = memory_manager

    def retrieve(self, query: str, options: RetrievalOptions) -> List[ContextItem]:
        if not options.use_web_context:
            return []

        # Prefer topic_hint if it's a web topic
        topic = (options.topic_hint or "").strip()
        wctx = None
        if topic.startswith("web:"):
            wctx = self.mm.get_web_context(topic)

        # fallback: last recent web context
        if wctx is None:
            wctx = self.mm.get_most_recent_web_context() if hasattr(self.mm, "get_most_recent_web_context") else None

        if not wctx:
            return []

        results = wctx.get("results", [])[: options.per_source_k]
        items: List[ContextItem] = []
        for i, r in enumerate(results, start=1):
            items.append(ContextItem(
                source="web",
                title=f"[{i}] {r.get('title','(untitled)')}",
                text=(r.get("snippet") or "").strip(),
                url=r.get("url"),
                score=0.6,
                meta={"rank": i, "query": wctx.get("query", "")},
            ))
        return items