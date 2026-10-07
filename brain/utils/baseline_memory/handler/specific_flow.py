import re
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

# Types for injected functions (so this stays modular / testable)
RetrieveFn = Callable[[str], Tuple[Optional[str], Optional[str], Optional[str]]]
SummarizeFn = Callable[[str, str, str], str]  # (topic, stored_memory, discussion_date) -> summary

def parse_stored_doc(doc: str) -> dict:
    """
    Parses a stored ChromaDB document string into its components.
    Returns: {"topic": str, "date": str, "summary": str}
    """
    topic, date, summary = "", "", ""
    lines = doc.split("\n")
    
    summary_lines = []
    in_summary = False
    
    for line in lines:
        if line.startswith("Topic:"):
            topic = line.replace("Topic:", "").strip()
        elif line.startswith("Date:"):
            date = line.replace("Date:", "").strip()
        elif line.startswith("Summary:"):
            in_summary = True
        elif in_summary:
            summary_lines.append(line)
    
    return {
        "topic": topic,
        "date": date,
        "summary": "\n".join(summary_lines).strip()
    }


@dataclass
class SpecificDecision:
    action: str                      
    prompt: Optional[str] = None


class SpecificRequestHandler:
    # SHOW_PAT = re.compile(r"\b(show everything|show all|full details|everything|all of it)\b", re.I)
    # ADD_PAT  = re.compile(r"\b(add information|add info|add new|remember this|save this|add)\b", re.I)
    # MOVE_PAT = re.compile(r"\b(move on|continue|next|done|stop|cancel)\b", re.I)

    SHOW_PAT = re.compile(r"\b(show|open|details?|full|everything)\b", re.I)
    ADD_PAT  = re.compile(r"\b(add|append|update|note|save|remember)\b", re.I)
    MOVE_PAT = re.compile(r"\b(move on|next|back|exit|stop|cancel|continue)\b", re.I)
    YES_PAT  = re.compile(r"^(yes|yep|yeah|ok|okay|sure)(\s+please|\s+pls)?$", re.I)
    MORE_PAT = re.compile(r"\b(more|elaborate|explain|tell me more)\b", re.I)


    @staticmethod
    def decide(user_text: str) -> SpecificDecision:
        text = (user_text or "").strip()

        if SpecificRequestHandler.SHOW_PAT.search(text):
            return SpecificDecision(action="show_all")
        if SpecificRequestHandler.YES_PAT.search(text):
            return SpecificDecision(action="show_all")
        if SpecificRequestHandler.MORE_PAT.search(text):
            return SpecificDecision(action="show_all")
        if SpecificRequestHandler.ADD_PAT.search(text):
            return SpecificDecision(action="add")
        if SpecificRequestHandler.MOVE_PAT.search(text):
            return SpecificDecision(action="move_on")

        return SpecificDecision(
            action="unknown",
            prompt="I didn’t catch that. You can say “show everything,” “add information,” or “move on.”"
        )


class SpecificFlow:

    PENDING_MENU = "Awaiting_Specific_Recall_Menu"
    PENDING_ADD  = "Awaiting_Topic_Add_Info"

    def __init__(self, memory_manager, retrieve_fn: RetrieveFn, summarize_fn: SummarizeFn):
        self.mm = memory_manager
        self.retrieve = retrieve_fn
        self.summarize = summarize_fn

    # ---------- Entry point: start specific ----------
    def start(self, topic_name: Optional[str]) -> str:
        if not topic_name:
            return "I can recall our past discussions. Which topic would you like me to summarize?"

        stored_memory, discussion_date, _topic_id = self.retrieve(topic_name)
        print(f"[DBG] start() retrieved stored_memory=\n{repr(stored_memory)}")
        print(f"[DBG] start() discussion_date={discussion_date}") 
        if not stored_memory:
            return f"I couldn't find any past conversations about {topic_name}."

        self.mm.set_last_topic(topic_name)
        self.mm.set_pending_topic(self.PENDING_MENU)
        self.mm.pending_payload = {
            "topic": topic_name,
            "stored_memory": stored_memory,      # ← cache it here
            "discussion_date": discussion_date,
        }

        # ← detect here instead of inside summarize_topic
        looks_like_raw_conversation = (
            "User:" in stored_memory or "Buddy:" in stored_memory
        )

        if looks_like_raw_conversation:
            # Raw STM exchange → summarize via LLM for the menu prompt
            topic_summary = self.summarize(topic_name, stored_memory, discussion_date or "Date Unknown")
        else:
            # Already a finished LTM summary → skip LLM, just show the menu prompt
            display_topic = re.sub(r"^about\s+", "", topic_name, flags=re.I).strip()
            topic_summary = (
                f"I found our conversation about {display_topic} "
                f"from {discussion_date or 'a previous session'}. "
                "Would you like me to show the full details, add new information, or move on?"
            )

        return (
            f"{topic_summary}\n\n"
            'If you want the full details, say "show everything." '
            'If you want to add new information, say "add information." '
            'If you want to move on, say "move on."'
        )

    # ---------- Follow-up handler: menu state ----------
    def handle_menu_turn(self, user_text: str) -> str:
        payload = getattr(self.mm, "pending_payload", None) or {}
        topic = payload.get("topic")

        if not topic:
            # state is inconsistent; reset safely
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None
            return "I lost track of the topic. Please tell me which topic you want to open."

        decision = SpecificRequestHandler.decide(user_text)

        if decision.action == "show_all":
            full_memory = payload.get("stored_memory")
            print(f"[DBG] show_all full_memory from payload=\n{repr(full_memory)}")
            if not full_memory:
                full_memory, _date, _ = self.retrieve(topic)

            if not full_memory:
                return f"I couldn't load the full details for {topic}."

            # Parse out just the summary body — don't dump raw document headers
            parsed = parse_stored_doc(full_memory)
            print(f"[DBG] parsed topic={repr(parsed['topic'])}")
            print(f"[DBG] parsed date={repr(parsed['date'])}")
            print(f"[DBG] parsed summary=\n{repr(parsed['summary'])}")
            display_topic = re.sub(r"^about\s+", "", topic, flags=re.I).strip()

            return (
                f"Here's everything I have about {display_topic}:\n\n"
                f"Topic: {parsed['topic']}\n"
                f"Date: {parsed['date']}\n"
                f"Summary:\n{parsed['summary']}\n\n"
                'You can say "add information" to save more, or say "move on."'
            )

        if decision.action == "add":
            self.mm.set_pending_topic("Awaiting_Topic_Add_Info")   # keep your existing string
            self.mm.pending_payload = {"topic": topic}
            self.mm.set_last_topic(topic)
            return f"Okay. Tell me the new information for **{topic}**. When you're finished, say “done.”"

        if decision.action == "move_on":
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None
            self.mm.set_last_topic(None) # for clearing current topic after retreating recalling memory
            return "Okay. What would you like to do next?"

        # unknown
        return decision.prompt or "Sorry, I didn’t catch that."

    # ---------- Optional: handle add-info state ----------
    # You can keep your existing add/save logic elsewhere, but here's a clean stub.
    def handle_add_info_turn(self, user_text: str) -> str:
        payload = getattr(self.mm, "pending_payload", None) or {}
        topic = payload.get("topic")

        if not topic:
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None
            return "Which topic should I add this information to?"

        text = (user_text or "").strip()
        if re.fullmatch(r"(done|finish|finished|stop)", text, re.I):
            # end add mode, return to menu
            self.mm.set_pending_topic(self.PENDING_MENU)
            self.mm.pending_payload = {"topic": topic}
            return (
                f"Got it. I’m ready if you want to “show everything,” add more information, or “move on.”"
            )

        # Here you would call your save routine for (topic, text)
        # For now we just acknowledge.
        return (
            f"Okay, I’ll add that under **{topic}**. You can keep going, or say “done.”"
        )
