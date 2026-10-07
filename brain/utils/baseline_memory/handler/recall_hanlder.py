from dataclasses import dataclass
import re
from typing import Optional

@dataclass
class GeneralDecision:
    action: str                    
    topic: Optional[str] = None
    next_state: Optional[str] = None
    prompt: Optional[str] = None  


class GeneralRequestHandler:
    OPEN_PAT = re.compile(r"\b(open|continue|show|tell me about)\b", re.I)
    SUM_PAT  = re.compile(r"\b(summarize|summary|recap)\b", re.I)
    DEL_PAT  = re.compile(r"\b(forget|delete|remove)\b", re.I)
    ADD_PAT  = re.compile(r"\b(add|save|remember)\b", re.I)
    LIST_PAT = re.compile(r"\b(what did we talk about|list topics|show topics|topics)\b", re.I)

    @staticmethod
    def decide(user_text: str, last_topics: list[str], pending_state: Optional[str] = None) -> GeneralDecision:
        text = (user_text or "").strip()

        # If we previously asked a follow-up after summarizing, interpret short replies
        if pending_state == "awaiting_post_summary_action":
            if GeneralRequestHandler.DEL_PAT.search(text):
                topic = GeneralRequestHandler._resolve_topic(text, last_topics)
                return GeneralDecision(action="delete", topic=topic)
            if GeneralRequestHandler.ADD_PAT.search(text):
                return GeneralDecision(action="add")
            if GeneralRequestHandler.OPEN_PAT.search(text):
                topic = GeneralRequestHandler._resolve_topic(text, last_topics)
                return GeneralDecision(action="open", topic=topic)

            # If user says just "yes"/"no", ask a crisp question again
            if re.fullmatch(r"(yes|yeah|yep|sure)", text, re.I):
                return GeneralDecision(
                    action="unknown",
                    next_state="awaiting_post_summary_action",
                    prompt="Would you like me to save something new, or forget one of the topics? You can say, add, or forget followed by a topic name."
                )
            if re.fullmatch(r"(no|nope|nah)", text, re.I):
                return GeneralDecision(action="unknown", next_state=None, prompt="Okay. What would you like to do next?")

            return GeneralDecision(
                action="unknown",
                next_state="awaiting_post_summary_action",
                prompt="After that recap, you can say add, forget followed by a topic name, or open followed by a topic name."
            )

        # Normal parsing
        if GeneralRequestHandler.LIST_PAT.search(text):
            return GeneralDecision(action="list")

        if GeneralRequestHandler.SUM_PAT.search(text):
            topic = GeneralRequestHandler._resolve_topic(text, last_topics)
            if not topic:
                return GeneralDecision(action="unknown", prompt="Which topic should I summarize? Please say the topic name.")
            return GeneralDecision(action="summarize", topic=topic, next_state="awaiting_post_summary_action")

        if GeneralRequestHandler.DEL_PAT.search(text):
            topic = GeneralRequestHandler._resolve_topic(text, last_topics)
            if not topic:
                return GeneralDecision(action="unknown", prompt="Which topic should I forget? Please say the topic name.")
            return GeneralDecision(action="delete", topic=topic)

        if GeneralRequestHandler.OPEN_PAT.search(text):
            topic = GeneralRequestHandler._resolve_topic(text, last_topics)
            if not topic:
                return GeneralDecision(action="unknown", prompt="Which topic should I open? Please say the topic name.")
            return GeneralDecision(action="open", topic=topic)

        if GeneralRequestHandler.ADD_PAT.search(text):
            return GeneralDecision(action="add")

        # If user just says a topic name, treat as open
        topic = GeneralRequestHandler._resolve_topic(text, last_topics)
        if topic:
            return GeneralDecision(action="open", topic=topic)

        return GeneralDecision(action="unknown", prompt="I can open, summarize, or forget a topic. Please say what you’d like, followed by the topic name.")

    @staticmethod
    def _resolve_topic(text: str, topics: list[str]) -> Optional[str]:
        """Resolve topic name from user text without requiring numbering."""
        t = text.lower()

        # Exact/substring match first
        for topic in topics:
            if topic.lower() in t:
                return topic

        # Loose match: user might just say partial words
        # Simple heuristic: pick the topic with most overlapping words
        words = set(re.findall(r"[a-z0-9]+", t))
        best = None
        best_score = 0
        for topic in topics:
            tw = set(re.findall(r"[a-z0-9]+", topic.lower()))
            score = len(words & tw)
            if score > best_score:
                best_score = score
                best = topic
        return best if best_score > 0 else None
