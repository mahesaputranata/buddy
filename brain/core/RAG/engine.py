# brain/retrieval/engine.py
from __future__ import annotations
from typing import List
from brain.core.RAG.types import RetrievalResult, RetrievalOptions, ContextItem
from brain.core.RAG.rankers.simple import SimpleRanker

class RetrievalEngine:
    def __init__(self, sources: List, ranker=None):
        self.sources = sources
        self.ranker = ranker or SimpleRanker()

    def retrieve(self, query: str, options: RetrievalOptions) -> RetrievalResult:
        include = set(options.include_sources) if options.include_sources else None

        items: List[ContextItem] = []
        for src in self.sources:
            if include and getattr(src, "name", None) not in include:
                continue
            got = src.retrieve(query, options) or []
            # cap per source
            if options.per_source_k and len(got) > options.per_source_k:
                got = got[: options.per_source_k]
            items.extend(got)

        # score + sort
        items = self.ranker.score(query, items)

        # filter + top_k
        if options.min_score > 0:
            items = [it for it in items if it.score >= options.min_score]

        if options.top_k and len(items) > options.top_k:
            items = items[: options.top_k]

        return RetrievalResult(query=query, items=items)