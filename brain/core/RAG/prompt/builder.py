# brain/retrieval/prompt/builder.py
from __future__ import annotations
from typing import List
from brain.core.RAG.types import ContextItem

class PromptBuilder:
    def build(self, user_input: str, context_items: List[ContextItem], belief_block: str = "") -> str:
        ctx_blocks = []
        for it in context_items:
            header = f"[{it.source.upper()}] {it.title}"
            if it.url:
                header += f" ({it.url})"
            body = (it.text or "").strip()
            if body:
                ctx_blocks.append(header + "\n" + body)

        ctx_text = "\n\n".join(ctx_blocks).strip()

        parts = ["You are Buddy, a helpful assistant."]
        if belief_block:
            parts.append(f"[Current situational context]\n{belief_block}")
        if ctx_text:
            parts.append(f"Context:\n{ctx_text}")
        parts.append(f"User: {user_input}\nBuddy:")

        return "\n\n".join(parts)