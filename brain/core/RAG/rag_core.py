from dataclasses import dataclass
from typing import Dict, Any, List, Optional


def safe_join(parts: List[str], sep: str = "\n\n") -> str:
    return sep.join([p for p in parts if (p or "").strip()])


def shorten(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "..."


@dataclass
class RagContext:
    topic: str
    short_term: str = ""
    web_block: str = ""
    mode: str = "General Conversation"
    system_context: str = "" 
    belief_block: str = ""


class RagCore:
    def __init__(self, llm_fn):
        # llm_fn(prompt:str, mode:str) -> str
        self.llm_fn = llm_fn
        # llm_fn only gets the real user message if it accepts `user_text` (generate_llm_response does)
        try:
            import inspect
            self._llm_takes_user_text = "user_text" in inspect.signature(llm_fn).parameters
        except (TypeError, ValueError):
            self._llm_takes_user_text = False

    def _build_system_prompt(self, ctx: RagContext) -> str:
        identity_block = f"{ctx.system_context}\n\n" if ctx.system_context else ""
        base = f"{identity_block}You are Buddy, a helpful conversational assistant."

        if ctx.mode == "Web Navigation":
            return (
                base
                + " The user is interacting with saved web search results."
                + " Use the web context to answer."
                + " If the user refers to a result number, treat it as that saved result."
                + " Help with actions like summarize, open, compare, or explain a specific result."
                + " Stay grounded in the saved web context."
                + " Be concise and do not drift into broad discussion unless the user asks for it."
                + " Ask a short clarifying question only if the action is ambiguous."
            )

        if ctx.mode == "Web Discussion":
            return (
                base
                + " The user is now discussing the topic behind a recent web search, not merely navigating results."
                + " Use the web context as background knowledge, but prioritize answering the user's latest message directly."
                + " Do not repeat the saved summary unless it is necessary."
                + " If the user asks for elaboration, expand the previous answer instead of restarting from the beginning."
                + " Interpret short follow-ups like 'yes', 'please elaborate', 'tell me more', or 'go on' using the recent conversation context."
                + " If the user asks for analysis, implications, causes, opinions, or possible responses, answer conversationally and thoughtfully."
                + " Only mention specific result numbers if the user explicitly refers to them."
                + " Avoid falling back to menu-like instructions unless the user is clearly asking about saved results."
            )

        return (
            base
            + " Continue the conversation naturally."
            + " Prioritize the user's latest question."
            + " Avoid repeating earlier answers unless needed."
        )

    def build_prompt(self, *, user_input: str, ctx: RagContext) -> str:
        sys = self._build_system_prompt(ctx)

        parts = [sys]

        if ctx.belief_block:
            parts.append(f"[Current situational context]\n{ctx.belief_block}")

        if ctx.topic:
            parts.append(f"[Topic]\n{ctx.topic}")

        if ctx.web_block:
            parts.append(f"[Web context]\n{shorten(ctx.web_block, 2200)}")

        if ctx.short_term:
            parts.append(f"[Conversation so far]\n{shorten(ctx.short_term, 3000)}")

        parts.append(f"User: {user_input}")
        parts.append("Buddy:")

        return safe_join(parts)

    def answer(self, *, user_input: str, ctx: RagContext) -> str:
        prompt = self.build_prompt(user_input=user_input, ctx=ctx)
        if self._llm_takes_user_text:
            return self.llm_fn(prompt, ctx.mode, user_text=user_input)
        return self.llm_fn(prompt, ctx.mode)