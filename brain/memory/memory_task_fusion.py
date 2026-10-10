from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemoryTaskFusionDecision:
    kind: str
    final_intent: str
    topic: str

    use_memory: bool
    run_task: bool
    require_confirmation: bool

    memory_route: Optional[Any] = None
    reason: str = ""


@dataclass
class NullMemoryRoute:
    kind: str = "none"
    topic: str = ""
    score: float = 0.0
    selected_episodes: list = None
    reason: str = "No memory route available."

    def __post_init__(self):
        if self.selected_episodes is None:
            self.selected_episodes = []


class MemoryTaskFusion:
    """
    Arbiter between task intent and episodic memory routing.

    This is not another classifier.
    It receives:
      - existing task intent
      - episodic memory route
      - last topic / belief state

    Then it decides the final execution path.
    """

    TASK_INTENTS = {
        "web_search",
        "web_followup",
        "schedule_save",
        "schedule_delete",
        "schedule_recall",
        "save_memory",
        "recall_memory",
        "delete_memory",
        "device_control",
        "system_query",
        "friday_task",
    }

    def decide(
        self,
        user_input: str,
        intent_result: dict,
        memory_route: Optional[Any],
        last_topic: str,
        belief: Optional[Any] = None,
    ) -> MemoryTaskFusionDecision:
        text = (user_input or "").strip().lower()
        intent = (intent_result or {}).get("intent", "chat")
        conf = float((intent_result or {}).get("confidence", 0.0) or 0.0)
        last_topic = (last_topic or "").strip().lower()

        if memory_route is None:
            memory_route = NullMemoryRoute(topic=last_topic)

        memory_kind = getattr(memory_route, "kind", "none")
        memory_topic = (getattr(memory_route, "topic", "") or last_topic).strip().lower()
        memory_score = float(getattr(memory_route, "score", 0.0) or 0.0)

        memory_recall_like = self.looks_like_memory_recall_query(text)
        memory_reminder_like = self.looks_like_memory_reminder_query(text)
        repeat_prior_action = self.looks_like_repeat_prior_action(text)
        web_navigation = self.looks_like_web_navigation_followup(text)
        live_rover_command = self.looks_like_live_rover_command(text)

        # 1. Active web navigation should not become a fresh web search.
        if last_topic.startswith("web:") and web_navigation:
            return MemoryTaskFusionDecision(
                kind="task_only",
                final_intent="web_followup",
                topic=last_topic,
                use_memory=False,
                run_task=True,
                require_confirmation=False,
                memory_route=memory_route,
                reason="Active web topic plus web navigation follow-up.",
            )

        # 2. Active rover direct commands should stay device control.
        if last_topic.startswith("rover:") and live_rover_command:
            return MemoryTaskFusionDecision(
                kind="task_only",
                final_intent="device_control",
                topic=last_topic,
                use_memory=False,
                run_task=True,
                require_confirmation=False,
                memory_route=memory_route,
                reason="Direct live rover command in active rover session.",
            )

        # 3. "Remind me where/what/how..." is memory recall, not scheduling.
        if memory_reminder_like and memory_kind == "episodic_recall":
            return MemoryTaskFusionDecision(
                kind="episodic_chat",
                final_intent="chat",
                topic=memory_topic,
                use_memory=True,
                run_task=False,
                require_confirmation=False,
                memory_route=memory_route,
                reason="Memory-reminder phrasing matched episodic recall.",
            )

        # 4. Repeating previous physical/device action needs confirmation.
        if repeat_prior_action and memory_kind == "episodic_recall":
            return MemoryTaskFusionDecision(
                kind="task_with_memory_confirm",
                final_intent="device_control" if "rover" in memory_topic else intent,
                topic=memory_topic,
                use_memory=True,
                run_task=False,
                require_confirmation=True,
                memory_route=memory_route,
                reason="User refers to repeating a prior action; confirmation required before execution.",
            )

        # 5. Recall-like query + episodic match should use episodic chat.
        if memory_recall_like and memory_kind == "episodic_recall":
            return MemoryTaskFusionDecision(
                kind="episodic_chat",
                final_intent="chat",
                topic=memory_topic,
                use_memory=True,
                run_task=False,
                require_confirmation=False,
                memory_route=memory_route,
                reason="Recall-like query matched an older episodic memory.",
            )

        # 6. Strong explicit task intents still win when not memory-like.
        if intent in self.TASK_INTENTS and intent != "system_query":
            return MemoryTaskFusionDecision(
                kind="task_only",
                final_intent=intent,
                topic=last_topic,
                use_memory=False,
                run_task=True,
                require_confirmation=False,
                memory_route=memory_route,
                reason=f"Task intent '{intent}' dominates; no memory override applied.",
            )

        # 7. Normal chat can use current or recalled memory.
        if intent in {"chat", "system_query"} and memory_kind in {"continue_current", "episodic_recall"}:
            return MemoryTaskFusionDecision(
                kind="episodic_chat",
                final_intent="chat",
                topic=memory_topic,
                use_memory=True,
                run_task=False,
                require_confirmation=False,
                memory_route=memory_route,
                reason=f"Chat uses memory route '{memory_kind}'.",
            )

        # 8. New topic chat.
        if memory_kind == "new_topic":
            return MemoryTaskFusionDecision(
                kind="new_topic_chat",
                final_intent="chat",
                topic=memory_topic,
                use_memory=False,
                run_task=False,
                require_confirmation=False,
                memory_route=memory_route,
                reason="Memory router selected a new topic.",
            )

        # 9. Default passthrough.
        return MemoryTaskFusionDecision(
            kind="task_only" if intent in self.TASK_INTENTS else "new_topic_chat",
            final_intent=intent,
            topic=last_topic,
            use_memory=False,
            run_task=intent in self.TASK_INTENTS,
            require_confirmation=False,
            memory_route=memory_route,
            reason="Default fusion passthrough.",
        )

    @staticmethod
    def looks_like_memory_recall_query(text: str) -> bool:
        t = (text or "").lower().strip()

        markers = [
            "what did we",
            "what was",
            "where did we",
            "where should",
            "how did we",
            "what do we",
            "from earlier",
            "from before",
            "earlier about",
            "before about",
            "again",
            "previous",
            "last time",
            "continue the",
            "continue that",
            "we discussed",
            "did we decide",
            "what did i say",
            "what did you say",
        ]

        return any(m in t for m in markers)

    @staticmethod
    def looks_like_memory_reminder_query(text: str) -> bool:
        t = (text or "").lower().strip()

        prefixes = [
            "remind me what",
            "remind me where",
            "remind me how",
            "remind me why",
            "remind me which",
        ]

        return any(t.startswith(p) for p in prefixes)

    @staticmethod
    def looks_like_repeat_prior_action(text: str) -> bool:
        t = (text or "").lower().strip()

        has_repeat = any(p in t for p in [
            "repeat",
            "do the same",
            "same movement",
            "same action",
            "do that again",
            "replay",
            "run the same",
        ])

        has_past_ref = any(p in t for p in [
            "from earlier",
            "from before",
            "earlier",
            "before",
            "previous",
            "last time",
        ])

        return has_repeat and has_past_ref

    @staticmethod
    def looks_like_live_rover_command(text: str) -> bool:
        t = (text or "").lower().strip()
        words = set(re.findall(r"[a-z0-9]+", t))

        commands = {
            "forward", "backward", "left", "right", "stop",
            "faster", "slower", "turn", "move", "go",
            "reverse", "halt",
        }

        return len(words) <= 4 and bool(words & commands)

    @staticmethod
    def looks_like_web_navigation_followup(text: str) -> bool:
        t = (text or "").lower().strip()

        if re.match(
            r"^(open|click|show|read)\s+(the\s+)?"
            r"(first|second|third|fourth|fifth|\d+)(\s+result)?",
            t,
        ):
            return True

        return any(p in t for p in [
            "open first result",
            "open second result",
            "click first result",
            "click second result",
            "show result",
            "read result",
        ])