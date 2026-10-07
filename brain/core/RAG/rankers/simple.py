# brain/retrieval/rankers/simple.py
from __future__ import annotations
from typing import List
from brain.core.RAG.types import ContextItem

class SimpleRanker:
    def score(self, query: str, items: List[ContextItem]) -> List[ContextItem]:
        q = (query or "").lower().strip()
        if not q:
            return items

        q_words = set([w for w in q.split() if len(w) >= 3])

        def overlap(text: str) -> float:
            t_words = set([w for w in (text or "").lower().split() if len(w) >= 3])
            if not q_words:
                return 0.0
            return len(q_words & t_words) / max(1, len(q_words))

        for it in items:
            base = it.score
            it.score = base + 0.4 * overlap(it.title + " " + it.text)

        items.sort(key=lambda x: x.score, reverse=True)
        return items