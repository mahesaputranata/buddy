import chromadb
import datetime
from collections import deque, defaultdict
import ollama
from sentence_transformers import SentenceTransformer, util
import numpy as np
from fuzzywuzzy import fuzz
from brain.core.helpers import get_date
from datetime import datetime, timedelta
import re
import uuid
from brain.core.prompts import detect_similarity_llm, generate_response, detectMemoryRequest, recallMemoryAccess, extractTopicName, storeSummaryLongTermMemory
from brain.core.prompts import newInfoRelevantCheck, saveMemoryAccess, deleteMemoryAccess, extractTime, saveScheduleCategory, extract_specific_request, recallScheduleCategory
import dateparser
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import spacy
from brain.utils.recall_temporal_helper import has_temporal_cue
from brain.primary_task.save_task.save_type_classifier import SaveTypeClassifier
from brain.conversation.followup_detector import is_short_followup
from brain.utils.context_guards import select_recent_context, should_store_exchange
import time

from dataclasses import dataclass
from typing import Optional, Tuple

from pathlib import Path


TZ = ZoneInfo("Europe/Berlin")


chroma_client = chromadb.PersistentClient(path="memory_db")
chat_collection = chroma_client.get_or_create_collection(name="chat_memory")

SENSORY_MEMORY_LIMIT = 3
sensory_memory = deque(maxlen=SENSORY_MEMORY_LIMIT)

topic_frequency = defaultdict(int)
temp_topic_memory = defaultdict(list)
long_term_candidates = set()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

memory_ready = False
memory_timeout = 0

class handleGeneral:

    @staticmethod
    def extract_all_topic(formatted_topics: str) -> list[str]:
        if not formatted_topics:
            return []
        lines = [l.strip() for l in formatted_topics.splitlines() if l.strip()]
        return [l[2:].strip() for l in lines if l.startswith("- ")]

    @staticmethod
    def extract_all_contents(topics: list[str]) -> list[dict]:
        return [{"topic": t, "last_discussed": None, "summary": None} for t in topics]


def get_time():
    return datetime.now(TZ)

def initialize_memory():
    global last_topic, pending_topic

    if last_topic is None:
        last_topic = None

    if pending_topic is None:
        pending_topic = None


# temporary normalize input (latrer will be centralized becauyse in here there are many normalized logic)
def normalize_topic(topic: str) -> str:
    topic = (topic or "").strip()
    return topic.lower() if topic else None

class MemoryManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance.pending_topic = None
            cls._instance.last_topic = None
            cls._instance.pending_ids = []
            cls._instance.pending_action = None
            cls._instance.pending_payload = None
            cls._instance.pending_update_payload = None
            cls._last_topics_shown = []
            cls._instance.flow_stack = []
            cls._instance.last_detour = False

            cls._instance._web_contexts = {}
            cls._instance.last_web_topic = None

            # evolve phase 1 version:
            cls._instance.active_web_session_id = None

            # ── vision ────────────────────────────────────────────
            cls._instance.pending_vision = None
            cls._instance.active_vision_source = None  # "webcam" | "screenshot" | None
            cls._instance.last_saved_topic = None

        return cls._instance

    def reset_session_state(self):
        """
        Called once per process boot (see ConversationManager.__init__).
        Clears everything that's scoped to a single conversational session,
        so a fresh boot never inherits stale pending state, topics, or
        flow stacks from a previous run.
        """
        self.pending_topic = None
        self.pending_payload = None
        self.pending_update_payload = None

        self.last_topic = None
        self.last_saved_topic = None

        self.pending_ids = []
        self.pending_action = None

        self.flow_stack = []
        self.last_detour = False

        self.pending_vision = None
        self.active_vision_source = None

        # Web session pointer always resets — don't silently resume
        # "still discussing X" from a previous boot without the user asking.
        self.active_web_session_id = None
        self.last_web_topic = None
        self._web_contexts = {}

        self._last_topics_shown = []

        print("[MemoryManager] session state reset")

    def set_last_saved_topic(self, topic: str | None):
        t = (topic or "").strip()
        self.last_saved_topic = t or None

    def get_last_saved_topic(self) -> str | None:
        return getattr(self, "last_saved_topic", None)

    def build_resume_hint(self, snap: dict) -> str:
        ft = snap.get("frame_type")
        topic = snap.get("last_topic")

        if ft == "topic_context" and topic:
            return f"By the way we paused **{topic}** — say “resume” to continue."
        return "By the way we paused your previous task — say “resume” to continue."


    # evolve phase 1 version
    def _make_web_session_id(self, topic: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", (topic or "").lower()).strip("_")
        return f"web_{int(time.time())}_{slug[:40]}"

    def set_web_context(self, topic, query, results, summary=None, activate=True, root_thread=None):
        topic = (topic or "").strip()
        if not topic:
            return

        now = time.time()
        old = self._web_contexts.get(topic) or {}

        session_id = old.get("session_id") or self._make_web_session_id(topic)

        # Preserve existing root_thread if not explicitly provided
        resolved_root_thread = (
            (root_thread or "").strip()
            or (old.get("root_thread") or "").strip()
            or ""
        )

        self._web_contexts[topic] = {
            "session_id": session_id,
            "topic_key": topic,
            "query": query,
            "results": results or [],
            "summary": (summary or "").strip(),
            "created_at": old.get("created_at", now),
            "last_used_at": now,
            "active_result": old.get("active_result"),
            "root_thread": resolved_root_thread,
        }

        if activate:
            self.last_web_topic = topic
            self.active_web_session_id = session_id

    def touch_web_context(self, topic: str):
        ctx = self._web_contexts.get(topic)
        if ctx:
            ctx["last_used_at"] = time.time()

    def get_web_context(self, topic: str):
        topic = (topic or "").strip()
        ctx = self._web_contexts.get(topic)
        if ctx:
            self.touch_web_context(topic)
        return ctx

    def get_latest_web_topic(self) -> str | None:
        if self.last_web_topic and self.last_web_topic in self._web_contexts:
            return self.last_web_topic

        if not self._web_contexts:
            return None

        # pick newest by created_at
        return max(self._web_contexts.items(), key=lambda kv: kv[1].get("created_at", 0))[0]

    def pop_web_context(self, topic: str):
        topic = (topic or "").strip()
        if self.last_web_topic == topic:
            self.last_web_topic = None
        ctx = self._web_contexts.get(topic)
        if ctx and ctx.get("session_id") == self.active_web_session_id:
            self.active_web_session_id = None
        return self._web_contexts.pop(topic, None)

    def clear_web_contexts(self):
        self._web_contexts.clear()
        self.last_web_topic = None
        self.active_web_session_id = None

    def clear_pending(self):
        self.pending_topic = None
        self.pending_payload = None

    def set_last_topics_shown(self, topics: list[str]):
        self._last_topics_shown = topics or []

    def get_last_topics_shown(self) -> list[str]:
        return getattr(self, "_last_topics_shown", []) or []

    def set_pending_topic(self, topic):
        self.pending_topic = topic

    def get_pending_topic(self):
        return self.pending_topic

    def prune_web_contexts(self, *, keep_last_n: int = 8, ttl_seconds: int = 60 * 60):

        now = time.time()

        # 1) TTL prune
        dead = []
        for k, v in self._web_contexts.items():
            last_used = v.get("last_used_at", v.get("created_at", 0))
            if now - last_used > ttl_seconds:
                dead.append(k)
        for k in dead:
            self._web_contexts.pop(k, None)

        if self.active_web_session_id:
            still_exists = any(
                ctx.get("session_id") == self.active_web_session_id
                for ctx in self._web_contexts.values()
            )
            if not still_exists:
                self.active_web_session_id = None

        # 2) keep last N by created_at
        if len(self._web_contexts) > keep_last_n:
            items = sorted(self._web_contexts.items(), key=lambda kv: kv[1].get("created_at", 0), reverse=True)
            keep = set([k for k, _ in items[:keep_last_n]])
            for k in list(self._web_contexts.keys()):
                if k not in keep:
                    self._web_contexts.pop(k, None)


        # refresh pointer
        self.last_web_topic = self.get_latest_web_topic()

    def get_active_web_topic(self) -> str | None:
        sid = getattr(self, "active_web_session_id", None)
        if not sid:
            return None

        for topic, ctx in self._web_contexts.items():
            if ctx.get("session_id") == sid:
                self.touch_web_context(topic)
                return topic
        return None

    def get_active_web_context(self):
        topic = self.get_active_web_topic()
        if not topic:
            return None
        return self._web_contexts.get(topic)

    def deactivate_web_session(self):
        self.active_web_session_id = None

    def set_last_topic(self, topic):

        t = (topic or "").strip()
        if not t:
            self.last_topic = None
            return

        if t.lower().startswith("web:"):
            self.last_topic = t.lower()
            return

        self.last_topic = normalize_topic(t)

    def get_last_topic(self):
        return self.last_topic

    def set_pending_ids(self, ids):
        self.pending_ids = ids

    def get_pending_ids(self):
        return self.pending_ids

    def clear_pending_ids(self):
        self.pending_ids = []

    def set_pending_action(self, action, payload=None):
        self.pending_action = action
        self.pending_payload = payload

    def get_pending_action(self):
        return self.pending_action, self.pending_payload

    def clear_pending_action(self):
        self.pending_action = None
        self.pending_payload = None

    def merge_pending_payload(self, patch: dict):
        if self.pending_payload is None:
            self.pending_payload = {}
        self.pending_payload.update(patch)

    def push_flow(self, reason: str = "pause_switch"):
        """Save current pending state so we can resume it later."""
        if not self.pending_topic and not self.pending_payload:
            return False

        snapshot = {
            "pending_topic": self.pending_topic,
            "pending_payload": self.pending_payload,
            "pending_update_payload": self.pending_update_payload,
            "last_topic": self.last_topic,
            "paused_reason": reason,
            "frame_type": "flow_state",
        }
        self.flow_stack.append(snapshot)

        # clear current flow
        self.pending_topic = None
        self.pending_payload = None
        self.pending_update_payload = None
        return True

    def push_context(self, topic: str, reason: str = "pause_topic"):
        if not topic:
            return False
        snapshot = {
            "pending_topic": None,
            "pending_payload": None,
            "pending_update_payload": None,
            "last_topic": topic,
            "paused_reason": reason,
            "frame_type": "topic_context",
        }
        self.flow_stack.append(snapshot)
        return True


    def pop_flow(self):
        if not self.flow_stack:
            return None
        snap = self.flow_stack.pop()
        self.pending_topic = snap.get("pending_topic")
        self.pending_payload = snap.get("pending_payload")
        self.pending_update_payload = snap.get("pending_update_payload")
        self.last_topic = snap.get("last_topic")
        return snap


    def has_paused_flow(self):
        return bool(getattr(self, "flow_stack", []))

    def clear_all_flows(self):
        self.clear_pending()
        self.flow_stack = []
        self.last_detour = False

    def set_active_vision_source(self, source: str | None):
        """Track whether current vision session is 'webcam' or 'screenshot'."""
        self.active_vision_source = source

    def get_active_vision_source(self) -> str | None:
        return getattr(self, "active_vision_source", None)

    def clear_pending_vision(self):
        self.pending_vision = None
        self.active_vision_source = None

    def set_pending_vision(self, img_path, img_hash: str = ""):
        # normalize to always store a list of paths internally
        if isinstance(img_path, (str, Path)):
            paths = [str(img_path)]
        else:
            paths = [str(p) for p in img_path]

        self.pending_vision = {"paths": paths, "hash": img_hash}

    def is_burst_vision(self) -> bool:
        pv = self.get_pending_vision()
        if not pv:
            return False
        return len(pv.get("paths", [])) > 1

    def pop_pending_vision(self):
        v = getattr(self, "pending_vision", None)
        self.pending_vision = None
        return v

    def has_pending_vision(self) -> bool:
        return getattr(self, "pending_vision", None) is not None

    def get_pending_vision(self):
        return getattr(self, "pending_vision", None)

    def clear_pending_vision(self):
        self.pending_vision = None


memory_manager = MemoryManager()


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def extract_specificCase_condition(user_input):

    lower = user_input.lower()
    if "more" in lower or "elaborate" in lower or "explain" in lower:
        return "1"
    elif "add" in lower or "update" in lower or "include" in lower:
        return "2"
    elif "move on" in lower or "different topic" in lower or "nevermind" in lower:
        return "3"

    prompt = extract_specific_request(user_input)
    response = ollama.generate(model="llama3.2:latest", prompt=prompt)
    specific_case = response['response'].strip()

    if specific_case in ["1", "2", "3"]:
        return specific_case
    return "1"


def get_text_embedding(text):
    """Generates an embedding for a given text input."""
    return embedding_model.encode(text, convert_to_numpy=True)


def compute_similarity_llm(user_input, last_topic, short_term_context):

    print({short_term_context})

    detectSimilarity = detect_similarity_llm(user_input, last_topic, short_term_context)
    response = ollama.generate(model="llama3.2:latest", prompt=detectSimilarity)
    valid_responses = ["yes", "no"]
    decision = response['response'].strip().lower()

    if decision not in valid_responses:
        print("[ERROR] Invalid response format. Defaulting to 'no'.")
        return False

    print(f"[DEBUG] LAST TOPIC: {last_topic}")
    print(f"[DEBUG] CONTEXT: {short_term_context if short_term_context else 'None'}")
    print(f"[DEBUG] LLM Decision Raw Output: {response['response']}")
    print(f"[DEBUG] LLM Parsed Decision: {decision}")

    return decision == "yes"

def compute_similarity_hybrid(
    user_input: str,
    last_topic: str,
    short_term_context: str = "",
    *,
    cosine_high: float = 0.70,   # above this → definitely same topic
    cosine_low: float = 0.35,    # below this → definitely different topic
) -> bool:
    """
    Three-zone hybrid:
      HIGH zone  → cosine alone says yes (fast path)
      LOW zone   → cosine alone says no  (fast path)
      MIDDLE zone → LLM decides          (slow path, only when needed)
    """

    # Zone 1: cosine fast path
    emb_input = embedding_model.encode(user_input, convert_to_numpy=True)
    emb_topic = embedding_model.encode(last_topic, convert_to_numpy=True)
    score = float(util.cos_sim(emb_input, emb_topic))

    print(f"[DBG] cosine_sim='{score:.3f}' input='{user_input[:40]}' topic='{last_topic}'")

    if score >= cosine_high:
        print(f"[DBG] similarity → YES (cosine high {score:.3f})")
        return True

    if score < cosine_low:
        print(f"[DBG] similarity → NO (cosine low {score:.3f})")
        return False

    # Zone 2: uncertain — escalate to LLM
    print(f"[DBG] similarity → UNCERTAIN ({score:.3f}), escalating to LLM")
    return compute_similarity_llm(user_input, last_topic, short_term_context)

# ---------------------------
# TOPIC IDENTIFICATION
# ---------------------------

def compute_similarity_embedding(text1: str, text2: str, threshold=0.5) -> bool:
    emb1 = embedding_model.encode(text1, convert_to_numpy=True)
    emb2 = embedding_model.encode(text2, convert_to_numpy=True)
    score = float(util.cos_sim(emb1, emb2))
    print(f"[DEBUG] Cosine similarity: {score:.3f}")
    return score > threshold

def retrieve_all_topics_from_memory():
    results = chat_collection.get(include=["documents"])
    topics = set()

    if "documents" in results:
        for doc in results["documents"]:
            if "Topic:" in doc:
                topic_name = doc.split("\n")[0].replace("Topic:", "").strip()
                topics.add(topic_name)

    print(f"[DEBUG] Retrieved topics from memory: {topics}")
    return list(topics)


def identify_topic(user_input):
    global last_topic

    if last_topic:
        if compute_similarity_llm(user_input, last_topic, retrieve_short_term_memory()):
            return last_topic

    all_topics = retrieve_all_topics_from_memory()

    if not all_topics:
        print("[DEBUG] No topics found in long-term memory. Checking short-term...")
        return None

    best_match = None
    best_score = 0

    for topic in all_topics:
        if compute_similarity_llm(user_input, topic, None):
            best_match = topic
            break

    return best_match if best_match else None


def confirm_memory_storage(user_input, last_topic, date=None, conversation_history=None):
    if date is None:
        date = get_date()

    memory_manager.set_pending_topic("Awaiting_Update_Confirm")
    memory_manager.pending_update_payload = {
        "topic": last_topic,
        "date": date,
        "user_input": user_input,
        "conversation_history": conversation_history,
    }

    # Extract the new topic from user_input to answer in the right context
    new_topic = extract_topic_name(user_input)
    answer = generate_llm_response(user_input, new_topic)
    return answer + "\n\n(Also, do you want me to save our previous topic?)"

def _embed_similarity(a: str, b: str) -> float:
    ea, eb = embedding_model.encode([a, b], convert_to_tensor=True)
    return float(util.cos_sim(ea, eb))


def generate_llm_response(user_input, topic, system_context: str = "", override_sensory=None, user_text=None):
    # user_text: the user's actual message. `user_input` can be a whole RAG prompt (RagCore passes
    # its assembled prompt through here), which is useless for judging relevance.
    if override_sensory is not None:
        context = override_sensory
    else:
        # Guard 1 (BUDDY_CONTEXT_GUARDS): the global recent-exchange buffer is not tied to any topic,
        # so it still holds the PREVIOUS conversation. Only show it when this message is a follow-up
        # or is similar to it; otherwise a 3B model happily continues the old subject.
        context = select_recent_context(user_text or user_input, list(sensory_memory), _embed_similarity)
    short_term_context = "\n".join(context) if len(context) > 0 else "No relevant past discussion."

    generateResponse = generate_response(user_input, topic, short_term_context, system_context)

    response = ollama.generate(model="llama3.2:latest", prompt=generateResponse)
    return response['response'].strip()

# ---------------------------
# MEMORY DETECTION
# ---------------------------

# for saving in a middle conversation
def _fuzzy_match_topic(query: str, topic_list: list[str]) -> str | None:
    if not query or not topic_list:
        return None

    q = query.lower().strip()

    # exact match first
    if q in topic_list:
        return q

    # substring match
    for t in topic_list:
        if q in t or t in q:
            return t

    # word overlap
    q_words = set(q.split())
    best, best_score = None, 0
    for t in topic_list:
        t_words = set(t.lower().split())
        overlap = len(q_words & t_words)
        if overlap > best_score:
            best_score = overlap
            best = t

    return best if best_score >= 1 else None

def resolve_save_target(
    user_input: str,
    memory_manager,
    temp_topic_memory: dict,
) -> dict:
    """
    Figures out WHAT the user wants to save.
    Returns: {
        "topic": str | None,
        "history": list | None, 
        "confidence": float,
        "needs_clarification": bool,
        "candidates": list  # if multiple matches
    }
    """
    text = (user_input or "").lower().strip()

    # Case 1: User named the topic explicitly
    # "save our conversation about reinforcement learning"
    topic_name = extract_topic_name(user_input)
    if topic_name:
        # RAG step: search short term memory
        matched = _fuzzy_match_topic(topic_name, list(temp_topic_memory.keys()))
        if matched:
            return {
                "topic": matched,
                "history": temp_topic_memory[matched],
                "confidence": 0.9,
                "needs_clarification": False,
                "candidates": [],
            }
        # named but not found
        return {
            "topic": topic_name,
            "history": None,
            "confidence": 0.0,
            "needs_clarification": False,
            "candidates": [],
        }

    # Case 2: User said "this" / "our conversation" without naming topic
    # → check last active topic first
    last_topic = memory_manager.get_last_topic()
    if last_topic and not last_topic.startswith("web:"):
        history = temp_topic_memory.get(last_topic, [])
        if history:
            return {
                "topic": last_topic,
                "history": history,
                "confidence": 0.85,
                "needs_clarification": False,
                "candidates": [],
            }

    # Case 3: Multiple topics in short term — ask user
    available = [t for t, h in temp_topic_memory.items() if h]
    if len(available) == 1:
        t = available[0]
        return {
            "topic": t,
            "history": temp_topic_memory[t],
            "confidence": 0.8,
            "needs_clarification": False,
            "candidates": [],
        }

    if len(available) > 1:
        return {
            "topic": None,
            "history": None,
            "confidence": 0.5,
            "needs_clarification": True,
            "candidates": available,
        }

    # Case 4: Nothing in short term memory at all
    return {
        "topic": None,
        "history": None,
        "confidence": 0.0,
        "needs_clarification": False,
        "candidates": [],
    }

def consolidate_to_long_term(
    self,
    topic: str,
    history: list[str],
    date: str,
) -> str:
    """
    Summarizes short term history then stores to long term.
    This is the Memory Consolidation step.
    """
    if not history:
        return f"No conversation history found for {topic}."

    # RAG: format retrieved chunks as context
    raw_history = "\n".join(history)

    # LLM: summarize into dense form before storing
    summary_prompt = f"""
You are summarizing a conversation for long-term memory storage.
Topic: {topic}
Date: {date}

Conversation:
{raw_history}

Write a dense, factual summary of the key points discussed.
Focus on facts, decisions, and important details.
Do not include filler or small talk.
Summary:
"""
    response = ollama.generate(
        model="llama3.2:latest",
        prompt=summary_prompt
    )
    summary = (response.get("response") or "").strip()

    # Store summary (not raw history) to long term
    store_long_term_memory(
        topic,
        date,
        conversation_history=summary,  # ← summarized, not raw
    )

    return summary

def delete_memory_from_db(topic):
    try:
        print(f"[DEBUG] Attempting to delete topic: {topic}")

        stored_memory, discussion_date, topic_id = retrieve_long_term_memory(topic)

        if not topic_id:
            # ── Fuzzy fallback — exact match failed, try closest stored topic ──
            print(f"[DEBUG] Exact match failed for '{topic}', trying fuzzy match")
            resolved_topic = _fuzzy_resolve_topic(topic)

            if resolved_topic:
                print(f"[DEBUG] Fuzzy matched '{topic}' -> '{resolved_topic}'")
                stored_memory, discussion_date, topic_id = retrieve_long_term_memory(resolved_topic)
                if topic_id:
                    topic = resolved_topic  # use the real stored name in the response

        if not topic_id:
            return f"I couldn't find any past conversations about **{topic}** to delete."

        chat_collection.delete(ids=[topic_id])
        print(f"[DEBUG] Successfully deleted topic: {topic}")

        return f"I've deleted all stored memory about **{topic}**."

    except Exception as e:
        print(f"[ERROR] Failed to delete memory: {e}")
        return "I encountered an error while trying to delete the memory. Please try again."


def _fuzzy_resolve_topic(topic: str, min_score: int = 75) -> str | None:
    """
    Given a topic string that failed exact lookup, find the closest
    actually-stored topic name. Reuses the same enumeration pattern
    as handle_recall's general branch.
    """
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        return None

    all_memory, _, _ = retrieve_long_term_memory(None)
    if not all_memory:
        return None

    all_topics = handleGeneral.extract_all_topic(all_memory)
    if not all_topics:
        return None

    normalized = topic.strip().lower().replace("_", " ")

    best_match, best_score = None, 0
    for t in all_topics:
        score = fuzz.token_sort_ratio(normalized, t.lower())
        if score > best_score:
            best_score, best_match = score, t

    if best_match and best_score >= min_score:
        return best_match

    return None

def delete_memory_by_ids(ids):
    try:
        print(f"[DEBUG] Deleting memory by IDs: {ids}")
        chat_collection.delete(ids=ids)
        return f"I've deleted all memory entries from that time period."
    except Exception as e:
        print(f"[ERROR] Failed to delete by IDs: {e}")
        return "I encountered an error while trying to delete the memory."


def detect_memory_request(user_input):
    global memory_ready, pending_topic

    explicit_triggers = [
        "buddy please remember this",
        "store this in your memory",
        "save this conversation",
        "remember this for me",
    ]

    for phrase in explicit_triggers:
        if fuzz.partial_ratio(phrase, user_input.lower()) > 85:
            memory_ready = True
            pending_topic = "Awaiting_Topic"
            print("[DEBUG] 📌 Explicit memory request detected. Awaiting topic name.")
            return True

    memoryRequestTriggered = detectMemoryRequest(user_input)
    response = ollama.generate(model="llama3.2:latest", prompt=memoryRequestTriggered)
    decision = response['response'].strip().lower()

    print(f"[DEBUG] LLM Memory Detection Decision: {decision}")

    if decision.startswith("yes"):
        memory_ready = True
        pending_topic = "Awaiting_Topic"
        print("[DEBUG] LLM-based implicit memory request detected. Awaiting topic name.")
        return True

    return False


def save_memory_request(user_input: str, classifier: SaveTypeClassifier = None) -> str:
    text = (user_input or "").lower().strip()

    # ── EARLY GUARD: "save our X conversation/discussion" ────────────────
    # Catches topic-specific saves like "save our voxel grid pipeline conversation"
    # before the classifier even runs
    SAVE_CONV_RE = re.compile(
        r"\b(save|store|keep|remember)\s+(our|my|this|the)\s+"
        r"(.+?)\s+"
        r"(conversation|discussion|chat|talk|session)\b",
        re.IGNORECASE
    )
    if SAVE_CONV_RE.search(user_input):
        return "conversation_save"

    SAVE_CONVERSATION_DIRECT = [
        "save our previous conversation",
        "save our conversation",
        "save what we discussed",
        "save our previous discussion",
        "save our voxel",       # ← also add topic-prefix shortcuts
        "save our lidar",
        "save our discussion",
        "save our talk",
    ]
    if any(p in text for p in SAVE_CONVERSATION_DIRECT):
        return "conversation_save"

    SAVE_CONVERSATION_DIRECT = [
        "save our previous conversation",
        "save our conversation",
        "save what we discussed",
        "save our previous discussion",
    ]
    if any(p in user_input.lower() for p in SAVE_CONVERSATION_DIRECT):
        return "conversation_save"

    # ── TIER 1: Hard rules (zero ambiguity, no embedding needed) ──
    # Most specific first to avoid clashes

    CONV_SAVE_EXACT = [
        "save our conversation about",
        "save our discussion about",
        "store our conversation about",
        "keep our conversation about",
        "save what we talked about",
        "save our talk about",
    ]
    if any(p in text for p in CONV_SAVE_EXACT):
        return "conversation_save"

    GENERAL_EXACT = [
        "take a note", "take note", "note this down",
        "save this", "store this", "bookmark this",
        "keep this in mind", "write this down",
        "jot this down", "log this",
    ]
    if any(p in text for p in GENERAL_EXACT):
        return "general"

    PREAMBLE_EXACT = [
        "i want to tell you something important",
        "i want to talk something important",
        "i have something important to tell",
        "i need you to remember something",
        "i have something to share",
    ]
    if any(p in text for p in PREAMBLE_EXACT):
        return "preamble"

    SPECIFIC_EXACT = [
        "save about", "remember about", "note about",
        "save regarding", "note regarding",
        "remember the details about",
    ]
    if any(p in text for p in SPECIFIC_EXACT):
        return "specific"

    # ── TIER 2: Semantic embedding classifier (ambiguous cases) ──
    if classifier is not None:
        save_type, conf = classifier.classify(user_input)
        print(f"[SaveMemReq] embedding → {save_type} conf={conf:.3f}")

        if save_type != "unknown":
            # preamble and general both go to Awaiting_Topic
            # so merge them here for simplicity
            if save_type == "preamble":
                return "general"
            return save_type

    # ── TIER 3: LLM fallback (last resort) ──
    saveMemoryRequest = saveMemoryAccess(user_input)
    resp = ollama.generate(
        model="llama3.2:latest",
        prompt=saveMemoryRequest
    ).get("response", "").strip()

    try:
        data = json.loads(resp)
        t    = (data.get("type") or "unknown").lower().strip()
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        if conf < 0.60:
            return "unknown"
        if t in ("general", "specific", "conversation_save"):
            return t
        return "unknown"
    except Exception:
        return "unknown"


def extract_scheduleType_recall(user_input: str):
    text = (user_input or "").lower()

    # Simple helpful rules (add more later)
    if any(p in text for p in ["what's my schedule", "what is my schedule", "upcoming", "next meeting", "next appointment"]):
        return "general"

    if any(p in text for p in ["tomorrow", "today", "next week", "next month", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
        return "temporal"

    # LLM fallback with strict JSON
    recallScheduleCtg = recallScheduleCategory(user_input)
    resp = ollama.generate(model="llama3.2:latest", prompt=recallScheduleCtg).get("response","").strip()
    try:
        data = json.loads(resp)
        t = (data.get("type") or "unknown").lower().strip()
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        if conf < 0.60:
            return "unknown"

        if t in ("general", "specific", "temporal"):
            return t
        return "unknown"
    except Exception:
        return "unknown"


def extract_scheduleType_save(user_input: str):
    text = (user_input or "").lower()

    # --- Simple, SAFE rule hints (optional, not required) ---
    if any(p in text for p in ["remind me", "set a reminder", "add to calendar", "add to my calendar"]):
        # still might be missing time → general
        return "general"

    # --- LLM fallback (STRICT JSON) ---
    prompt = saveScheduleCategory(user_input)

    resp = ollama.generate(model="llama3.2:latest", prompt=prompt).get("response", "").strip()

    # extract topic
    prompt = extractTopicName(text)
    response = ollama.generate(model="llama3.2:latest", prompt=prompt)
    topic = (response.get("response") or "").strip()

    try:
        data = json.loads(resp)
        typ = (data.get("type") or "unknown").lower().strip()
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))

        if conf < 0.60:
            return "unknown"

        if topic:
            return "specific"

        if typ in ("general", "specific"):
            return typ
        return "unknown"

    except Exception:
        return "unknown"


def clean_llm_label(text: str) -> str:
    text = (text or "").strip().lower()
    # remove markdown bold/italics/backticks and surrounding quotes
    text = text.strip("`'\"")
    text = text.replace("**", "").replace("__", "").replace("*", "")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


DELETE_VERBS = r"(?:delete|remove|forget|erase|wipe)"
ABOUT_WORDS  = r"(?:about|regarding|re:\s*|related to|on)"

GENERAL_TRIGGERS = [
    "delete something", "delete a memory", "remove a memory",
    "can you forget", "erase", "wipe memory", "delete old stuff",
    "clear memory", "clear my memory", "forget everything", "wipe everything"
]

TEMPORAL_PHRASES = [
    "yesterday", "this morning", "last night", "earlier today",
    "two days ago", "a week ago"
]

# lightweight date-ish patterns (expand as needed)
DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",        # 2025-12-15
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # 12/15/2025
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
]

def _has_delete_intent(t: str) -> bool:
    return re.search(rf"\b{DELETE_VERBS}\b", t, flags=re.I) is not None


def delete_memory_request(user_input: str):
    print("[DEBUG] Delete Memory Request")
    t = (user_input or "").strip()
    tl = t.lower()

    # 0) If no delete intent at all, don’t bother
    if not _has_delete_intent(t):
        print("[DEBUG] No delete intent detected.")
        return "none"

    # 1) Specific-topic delete (highest value): detect and extract early
    topic = extract_topic_name(t)
    if topic:
        # optional: reuse your existing cleaner from extract_topic_name
        topic = topic.strip(" .!?\"'`")
        if topic:
            memory_manager.set_last_topic(topic)
        print(f"[DEBUG] Rule-based specific delete detected. topic={topic!r}")
        return "specific"

    # 2) Temporal delete
    if any(p in tl for p in TEMPORAL_PHRASES) or any(re.search(pat, tl) for pat in DATE_PATTERNS):
        print("[DEBUG] Rule-based temporal delete detected.")
        return "temporal"

    # 3) General delete
    if any(trigger in tl for trigger in GENERAL_TRIGGERS):
        print("[DEBUG] Rule-based general delete detected.")
        return "general"

    # 4) LLM fallback (only when rules couldn’t decide)
    llm_prompt = deleteMemoryAccess(user_input)
    response = ollama.generate(model="llama3.2:latest", prompt=llm_prompt)
    decision_raw = (response.get("response") or "")
    decision = clean_llm_label(decision_raw)

    print(f"[DEBUG] 🤖 LLM Delete Detection Decision Raw: {decision_raw!r}")
    print(f"[DEBUG] 🤖 LLM Delete Detection Decision Clean: {decision!r}")

    if decision == "general":
        return "general"
    if decision.startswith("specific:"):
        topic = decision.split(":", 1)[1].strip()
        if topic:
            memory_manager.set_last_topic(topic)
        return "specific"
    if decision == "temporal":
        return "temporal"
    if decision == "none":
        return "none"

    print("[DEBUG] ⚠️ Unrecognized format. Defaulting to general.")
    return "general"


########## RECALL MEMORY REQUEST ############################

@dataclass
class RecallRoute:
    recall_type: str          # "general" | "specific" | "temporal" | "schedule"
    confidence: float         # 0.0 - 1.0
    reason: str               # debug-friendly explanation


# --- helpers ---
_TIME_REGEXES = [
    r"\b(today|tomorrow|yesterday|tonight)\b",
    r"\b(this|next|last)\s+(week|month|year)\b",
    r"\b(in\s+\d+\s+(minutes?|hours?|days?|weeks?|months?))\b",
    r"\b(\d{1,2}:\d{2})\b",                       # 14:30
    r"\b(\d{1,2}\s*(am|pm))\b",                   # 2pm
    r"\b(on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b",  # Dec 19
]

def _has_time_reference(text: str) -> bool:
    return any(re.search(rx, text, re.I) for rx in _TIME_REGEXES)

def _contains_any(text: str, phrases: list[str]) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)

def _normalize(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def temporal_strength(user_input: str) -> int:
    """
    Returns 0..6 based on how confident we are that the user gave a time reference.
    """
    score = 0
    if has_temporal_cue(user_input):
        score += 2

    tr = extract_time_reference(user_input)
    if tr:
        # Strongest signal: we actually parsed something usable
        score += 4
    return score

def recall_memory_request(
    user_input: str,
    routed_intent: Optional[str] = None,
    extract_topic_fn=None,                 # pass your extract_topic_name
) -> RecallRoute:
    """
    Deterministic router: schedule vs memory, and memory -> general/specific/temporal.
    Returns recall_type + confidence + reason.
    """

    text = _normalize(user_input).lower()

    # Strong cues (phrase lists)
    schedule_phrases = [
        "schedule", "calendar", "agenda", "reminder", "appointment", "meeting",
        "invite", "invitation", "availability", "free time",
        "what do i have", "do i have", "what's on", "upcoming",
    ]

    memory_general_phrases = [
        "what did we talk about", "what have we been discussing", "what were we talking about",
        "recall", "remember", "our discussion", "previous conversation", "last time",
        "did you save", "notes", "in your memory", "from before",
    ]

    memory_specific_action_phrases = [
        "summarize", "summary", "recap",
        "open", "show", "tell me about", "pull up",
        "forget", "delete", "remove",
        "add", "save", "remember this",
    ]


    # If your upstream router already knows schedule, trust it.
    if routed_intent == "schedule_recall":
        return RecallRoute("schedule", 0.95, "Upstream routed_intent == schedule_recall")

    # --- scoring ---
    schedule_score = 0
    memory_score = 0
    specific_score = 0
    temporal_score = 0


    # Temporal cues (strong + stable)
    temporal_score += temporal_strength(user_input)


    # Schedule cues
    if _contains_any(text, schedule_phrases):
        schedule_score += 3

    # Memory cues
    if _contains_any(text, memory_general_phrases):
        memory_score += 3

    # Specific action words (often implies specific memory task)
    if _contains_any(text, memory_specific_action_phrases):
        memory_score += 1
        specific_score += 2

    # Explicit topic extraction (best signal for "specific")
    topic_name = None
    if extract_topic_fn is not None:
        try:
            topic_name = extract_topic_fn(user_input)
        except Exception:
            topic_name = None

    if topic_name:
        specific_score += 4
        memory_score += 2


    # If temporal + memory-ish language => temporal memory
    if temporal_score and (memory_score > 0):
        temporal_score += 1

    # Disambiguation: if schedule & memory both present, schedule usually wins unless a topic is named
    if schedule_score and memory_score:
        if topic_name:
            # user is likely recalling a specific meeting-related discussion, not querying calendar
            memory_score += 1
        else:
            return RecallRoute("schedule", 0.75, "Both schedule and memory cues; no explicit topic, default schedule")

    # Decide among memory types
    if memory_score == 0 and schedule_score == 0:
        # default safe
        return RecallRoute("general", 0.40, "No strong cues; default to general memory recall")

    if schedule_score > memory_score:
        conf = min(0.60 + 0.10 * schedule_score, 0.95)
        return RecallRoute("schedule", conf, f"schedule_score={schedule_score} > memory_score={memory_score}")

    # memory path
    if topic_name or specific_score >= 4:
        # specific wins over temporal unless user explicitly included time
        if temporal_score >= 4:
            return RecallRoute("temporal", 0.80, "Explicit topic + strong time reference => temporal memory")
        return RecallRoute("specific", 0.85, "Explicit topic or strong specific action cues")

    if temporal_score >= 4:
        return RecallRoute("temporal", 0.75, "Strong time reference with memory cues")

    return RecallRoute("general", 0.75, "Memory cues without explicit topic => general recall")

STOP_TOPICS = {"it", "this", "that", "something", "anything", "topic", "memory"}

# Optional: common filler that often leaks into topic names
LEADING_FILLER = re.compile(r"^(?:about|regarding)\s+", re.I)

# Optional: normalize “I would like to …” style preambles
PREAMBLE = re.compile(
    r"^\s*(?:i\s+(?:would\s+like\s+to|want\s+to|need\s+to)\s+)?",
    re.I
)


_nlp = None
_spacy_load_attempted = False
_spacy_load_failed = False

def _get_nlp():
    """Lazy-load spaCy model. Caches failure state to avoid retrying."""
    global _nlp, _spacy_load_attempted, _spacy_load_failed

    if _spacy_load_failed:
        return None
    if _nlp is not None:
        return _nlp
    if _spacy_load_attempted:
        return None

    _spacy_load_attempted = True
    try:
        _nlp = spacy.load("en_core_web_sm")
        print("[TopicValidator] spaCy model loaded successfully")
        return _nlp
    except Exception as e:
        print(f"[TopicValidator] spaCy load failed: {e} — falling back to regex")
        _spacy_load_failed = True
        return None


# Lexicon: words commonly used as verbs that spaCy sometimes mis-tags as NOUN
# when parsing fragments. Keep this list focused on verbs that appear in
# clause-fragments masquerading as topics.
KNOWN_VERB_FORMS = {
    # being / having
    "is", "are", "was", "were", "has", "have", "had", "be", "been", "being",
    # doing
    "do", "does", "did", "done", "doing",
    # state changes
    "got", "gets", "getting", "went", "goes", "going", "came", "comes",
    "made", "makes", "making",
    # common action verbs in fragments
    "hurts", "burns", "stops", "starts", "works", "fails", "fell", "broke",
    "breaks", "ran", "runs", "running", "moved", "moves",
    # speaking / thinking
    "said", "says", "told", "thinks", "thought", "felt", "feels",
    # negative state
    "lost", "missed", "failed",
}

# Discourse markers that signal "not a topic" even if spaCy tags differently
DISCOURSE_MARKERS = {
    "oh", "ah", "uh", "um", "hmm", "okay", "ok", "well",
    "yeah", "yes", "no", "nope", "nah", "hell", "damn",
    "wow", "ugh", "ouch", "geez", "yikes",
}


def _is_valid_topic_structure(s: str) -> bool:
    """
    Validates that a candidate string has the structural properties of a topic:
    - Not an interjection or discourse marker
    - Not a clause/sentence (no finite verb)
    - Contains at least one substantive noun
    - Doesn't dangle on a preposition or adverb
    
    Uses spaCy when available, falls back to regex on failure.
    """
    if not s or not s.strip():
        return False

    nlp = _get_nlp()
    if nlp is not None:
        return _is_valid_topic_structure_spacy(s, nlp)
    return _is_valid_topic_structure_regex(s)


def _is_valid_topic_structure_spacy(s: str, nlp) -> bool:
    """
    Primary path: spaCy POS analysis + lexical verb double-check.
    
    Layered defense:
      1. spaCy POS tags (structural)
      2. Known verb lexicon (catches spaCy mis-tags)
      3. Discourse marker list (catches words spaCy may not flag as INTJ)
    """
    try:
        doc = nlp(s)
    except Exception as e:
        print(f"[TopicValidator] spaCy parse failed on {s!r}: {e}")
        return _is_valid_topic_structure_regex(s)

    # Filter out punctuation for structural reasoning
    content_tokens = [t for t in doc if t.pos_ != "PUNCT"]
    if not content_tokens:
        return False

    first = content_tokens[0]
    last = content_tokens[-1]

    # ── Rule 1: Reject if first token is an interjection ──
    if first.pos_ == "INTJ":
        return False

    # ── Rule 2: Reject if first token is a known discourse marker ──
    # (catches cases where spaCy doesn't tag as INTJ but the word is one)
    if first.text.lower() in DISCOURSE_MARKERS:
        return False

    # ── Rule 3: Reject if any token is a finite verb or auxiliary ──
    # Exception: VBG (gerund) and VBN (past participle) can act as nouns
    # Examples: "machine learning" (VBG=noun-like), "battery mounting" (VBG=noun-like)
    has_finite_verb = any(
        t.pos_ in ("VERB", "AUX") and t.tag_ not in ("VBG", "VBN")
        for t in content_tokens
    )
    if has_finite_verb:
        return False

    # ── Rule 4: Lexical verb double-check ──
    # spaCy sometimes mis-tags fragment verbs as NOUN (e.g., "hurts" → NN).
    # Cross-reference against known verb lexicon to catch these.
    has_lexical_verb = any(
        t.text.lower() in KNOWN_VERB_FORMS
        for t in content_tokens
    )
    if has_lexical_verb:
        return False

    # ── Rule 5: Reject disjunctive coordinating conjunctions ──
    # "but" and "or" in the middle of a phrase signal clause-fragment, not topic.
    # ("tear hurts but more" — caught here even though spaCy mis-tagged hurts)
    has_disjunctive_conj = any(
        t.pos_ == "CCONJ" and t.text.lower() in ("but", "or", "yet")
        for t in content_tokens
    )
    if has_disjunctive_conj:
        return False

    # ── Rule 6: Reject if last content token is an adverb ──
    # Topics don't typically end on adverbs ("hurts but more" → ends on "more")
    if last.pos_ == "ADV":
        return False

    # ── Rule 7: Reject if last content token dangles ──
    # Prepositions, particles, determiners, possessives at the end signal incomplete phrases
    if last.pos_ in ("ADP", "PART"):
        return False
    if last.pos_ == "DET":
        return False
    if last.pos_ in ("PRON",):
        return False

    if last.tag_ in ("PRP$", "WP$"):  # my, your, his, her, our, their, whose
        return False

    if last.text.lower() in {
        "my", "your", "his", "her", "our", "their", "its",
        "this", "that", "these", "those", "me", "you", "him",
        "them", "us", "it"
    }:
        return False

    # ── Rule 8: Must contain at least one noun-like token ──
    # NOUN, PROPN, or VBG (gerund acting as noun)
    has_noun_like = any(
        t.pos_ in ("NOUN", "PROPN") or
        (t.pos_ == "VERB" and t.tag_ == "VBG")
        for t in content_tokens
    )
    if not has_noun_like:
        return False


    RELATIONAL_FILLERS = {
        "about", "regarding", "concerning", "around",
        "toward", "towards", "related"
    }

    DANGLING_REFERENTS = {
        "my", "your", "his", "her", "our", "their", "its",
        "this", "that", "these", "those",
        "me", "you", "him", "them", "us", "it"
    }

    RELATIONAL_FILLERS = {
        "about", "regarding", "concerning", "around",
        "toward", "towards"
    }

    PRONOUN_DANGLERS = {
        "my", "your", "his", "her", "our", "their", "its",
        "this", "that", "these", "those",
        "me", "you", "him", "them", "us", "it"
    }

    WEAK_TOPIC_HEADS = {
        "more", "something", "anything", "everything", "nothing",
        "thing", "stuff"
    }

    DOMAIN_NOUN_HINTS = {
        "rover", "battery", "motor", "controller", "raspberry", "pi",
        "converter", "cable", "wiring", "power", "supply",
        "publication", "paper", "study", "spectroscopy", "thz",
        "model", "dataset", "experiment", "electronics", "connection",
        "communication", "server"
    }

    tokens_lower = [t.text.lower() for t in content_tokens]

    # Reject topics that end with pronoun / possessive / vague reference.
    if last.pos_ == "PRON":
        return False

    if last.tag_ in ("PRP$", "WP$"):
        return False

    if last.text.lower() in PRONOUN_DANGLERS:
        return False

    # Reject dangling relational fragments:
    # "more regarding my", "about this", "related to that"
    for i, tok in enumerate(tokens_lower):
        if tok in RELATIONAL_FILLERS:
            after = content_tokens[i + 1:]
            if not after:
                return False

            has_strong_after = any(
                t.pos_ in ("NOUN", "PROPN") and t.text.lower() not in PRONOUN_DANGLERS
                for t in after
            )

            if not has_strong_after:
                return False

    # Reject weak vague topic heads unless a known domain noun is present.
    if tokens_lower[0] in WEAK_TOPIC_HEADS:
        if not any(tok in DOMAIN_NOUN_HINTS for tok in tokens_lower):
            return False

    return True


def _is_valid_topic_structure_regex(s: str) -> bool:
    """
    Fallback path: regex heuristics. Lower precision than spaCy but always works.
    Used only if spaCy fails to load.
    """
    s_lower = s.lower().strip()
    if not s_lower:
        return False

    # Reject leading interjections / discourse markers
    if re.match(
        r"^(?:oh+|ah+|uh+|um+|hmm+|wow|ugh|ouch|hell|damn|"
        r"yeah|yes|no|nope|nah|okay|ok|well)\b",
        s_lower
    ):
        return False

    # Reject if contains common finite verb forms
    if re.search(
        r"\b(?:is|are|was|were|has|have|had|hurts?|got|gets|"
        r"i'?m|you'?re|it'?s|that'?s|works?|fails?|broke)\b",
        s_lower
    ):
        return False

    # Reject disjunctive conjunctions in the middle
    if re.search(r"\b(?:but|or|yet)\b", s_lower):
        return False

    # Reject if dangles on preposition/conjunction/determiner
    if re.search(
        r"\b(?:and|or|but|with|to|of|in|on|for|from|about|regarding|related|"
        r"the|a|an|than|my|your|his|her|our|their|its|this|that|these|those|"
        r"me|you|him|them|us|it)\s*$",
        s_lower
    ):
        return False

    # Must have at least one substantive token (length >= 4, non-pronoun)
    pronouns = {"this", "that", "these", "those", "mine", "ours",
                "your", "their", "them"}
    tokens = re.findall(r"[a-z]+", s_lower)
    substantive = [t for t in tokens if len(t) >= 4 and t not in pronouns]
    if not substantive:
        return False

    return True


# ─────────────────────────────────────────────────────────────────
# END TOPIC STRUCTURAL VALIDATION
# ─────────────────────────────────────────────────────────────────

def _clean_topic(s: str) -> str | None:
    """
    Cleans and validates a candidate topic string.
    Returns the cleaned topic if valid, or None if it should be rejected.

    Pipeline:
      Phase 0: basic whitespace/punctuation cleanup
      Phase 1: strip leading filler ("about", "regarding")
      Phase 2: stop-topics check
      Phase 3: cheap surface rejection
      Phase 4: structural validation
      Phase 5: final sanity checks
    """
    if not s:
        return None

    # Phase 0: basic cleanup
    s = s.strip().strip(" .!?\"'`")
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    # Optional: normalize smart quotes / weird apostrophes
    s = (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("“", '"')
         .replace("”", '"')
    ).strip()

    if not s:
        return None

    # Phase 1: strip leading filler
    # Example: "about the rover wiring" -> "the rover wiring"
    s = LEADING_FILLER.sub("", s).strip()

    if not s:
        return None

    s_lower = s.lower().strip()

    # Phase 2: stop-topics check
    if s_lower in STOP_TOPICS:
        return None

    # Phase 3: cheap surface rejection
    # Reject fallback/internal topic names.
    if re.fullmatch(r"topic_\d+", s_lower):
        return None

    # Reject very short vague phrases.
    if s_lower in {
        "no need", "feel free", "never mind", "nevermind",
        "forget it", "okay down", "ok down",
        "not yet top", "it is not yet top",
        "oh hell no", "hell no",
    }:
        return None

    # Reject phrases ending in dangling possessive/pronoun/filler.
    if re.search(
        r"\b(?:my|your|his|her|our|their|its|this|that|these|those|"
        r"me|you|him|them|us|it|the|a|an|of|to|for|with|about|"
        r"regarding|concerning|related|toward|towards|because|due)\s*$",
        s_lower,
    ):
        return None

    # Reject obvious clause fragments / discourse reactions before spaCy.
    if re.match(
        r"^(?:oh|ah|uh|um|hmm|okay|ok|yeah|yes|no|nope|nah|hell|damn|ugh|ouch)\b",
        s_lower,
    ):
        return None

    # Phase 4: structural validation
    if not _is_valid_topic_structure(s):
        return None

    # Phase 5: final sanity checks
    tokens = re.findall(r"[a-zA-Z0-9]+", s)
    if not tokens:
        return None

    # One-word topics are allowed only if they are not vague.
    if len(tokens) == 1 and tokens[0].lower() in {
        "thing", "stuff", "issue", "problem", "debugging",
        "help", "work", "project", "topic",
    }:
        return None

    # Avoid overly long sentence-like topics.
    if len(tokens) > 6:
        return None

    return s


def _strip_time_phrases(s: str) -> str:
    s = re.sub(r"\b(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b", "", s, flags=re.I).strip()
    s = re.sub(r"\b(yesterday|today|tomorrow|last week|this week|next week)\b", "", s, flags=re.I).strip()
    s = re.sub(r"\bat\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b", "", s, flags=re.I).strip()
    return s


_BROAD_CATEGORY_STOPLIST = {
    "robotics", "programming", "engineering", "technology",
    "electronics", "science", "computing", "software",
    "hardware", "development", "research", "design",
    "system", "systems", "discussion", "topic", "question",
    "project", "projects", "general", "stuff",
}

# Sentence-shape patterns — if input matches these, it's a sentence
# expressing intent or opinion, NOT a topic label.
_SENTENCE_SHAPE_RE = re.compile(
    r"^\s*(?:i|you|we|they|he|she|it)\s+"
    r"(?:am|'m|is|are|was|were|have|has|had|"
    r"will|would|could|should|can|may|might|"
    r"want|plan|think|need|like|prefer|hope|"
    r"open|consider|build|make|use|try|"
    r"don'?t|do\s+not)\b",
    re.IGNORECASE,
)

_FINITE_VERB_RE = re.compile(
    r"\b(?:am|is|are|was|were|have|has|had|will|"
    r"would|could|should|may|might|"
    r"want(?:s|ed)?|plan(?:s|ned)?|think(?:s|ing)?|need(?:s|ed)?|"
    r"open(?:ed|ing)?|consider(?:s|ed|ing)?)\b",
    re.IGNORECASE,
)

# new extract topic
def extract_topic_name(user_input: str):
    """
    Suggests a topic name from user input, or returns None.
 
    Contract: NEVER invents a placeholder. NEVER returns a
    topic_NNNN string. NEVER returns broad category words.
    If unsure, returns None and lets the caller keep the current topic.
    """
    print("[DBG]: Entering extracting topic name")

    text = (user_input or "").strip()
    if not text:
        return None

    text = re.sub(r"\s+", " ", text).strip()
    lower_original = text.lower().strip()

    # ── NEW GUARD 1: Short followups never produce a topic ──────────
    # This is the shared contract. extract is just an advisor — when
    # the utterance is a followup, we return None and let the routing
    # layer keep the current topic.
    if is_short_followup(text):
        print(f"[DEBUG] extract_topic_name: short followup, returning None")
        return None

    # ── Existing: hard-reject conversational fragments ─────────────
    if lower_original in {
        "no need", "never mind", "nevermind", "forget it",
        "oh hell no", "hell no", "okay", "ok", "thanks", "thank you",
    }:
        return None

    # ── 1) Recall question handling (unchanged) ─────────────────────
    recall_patterns = [
        r"\b(?:do we|did we|have we)\s+(?:discuss|talk about|mention)\s+"
        r"(?:anything\s+)?(?:about|regarding)?\s*(.+?)\??$",
        r"\bwhat\s+(?:did|have)\s+we\s+(?:discussed?|talked about|said)\s+"
        r"(?:about|regarding)\s+(.+?)\??$",
    ]

    for p in recall_patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            cleaned = _clean_topic(m.group(1))
            if cleaned:
                cleaned = _strip_time_phrases(cleaned)
                if cleaned:
                    print(f"[DEBUG] Recall topic extract: {cleaned}")
                    return cleaned

    # Pure question without clear technical noun -> no topic
    if "?" in text:
        noun_match = re.search(
            r"\b((?:[\w]+\s+){0,3}(?:signal|controller|system|"
            r"laser|diode|sensor|algorithm|method|technique|"
            r"battery|mounting|converter|rover|raspberry\s+pi|"
            r"server|wiring|paper|publication|model|dataset)s?)\b",
            text,
            re.I,
        )
        if not noun_match:
            return None

    # ── 2) Strip safe command prefixes ──────────────────────────────
    COMMAND_PREFIXES = [
        r"^\s*(summarize|summary|recap)\s+",
        r"^\s*(tell me about)\s+",
    ]

    text_cmd_stripped = text
    for rx in COMMAND_PREFIXES:
        text_cmd_stripped = re.sub(rx, "", text_cmd_stripped, flags=re.I).strip()

    lower = text_cmd_stripped.lower().strip()

    # ── 3) Stop prefixes (unchanged) ────────────────────────────────
    STOP_PREFIXES = (
        "can you help", "could you help",
        "can you explain", "could you explain",
    )
    if lower.startswith(STOP_PREFIXES):
        return None

    # ── 4) Strong explicit patterns (unchanged) ─────────────────────
    patterns = [
        r"(?:topic|subject)\s*[:\-]\s*(.+)$",
        r"(?:call it|name it)\s+(.+)$",
        r"\b(?:issue|problem|question|concern|topic|discussion)\s+"
        r"(?:about|regarding|with)\s+(.+)$",
        r"\bdiscuss(?:\s+about)?\s+(.+)$",
        r"\btalk\s+about\s+(.+)$",
        r"\bregarding\s+(.+)$",
        r"^\s*about\s+(.+)$",
        r"(?:about|regarding|discussing|discussed\s+about)\s+"
        r"([a-z][a-z\s]{2,40}?)(?:\s*[?.,]|$)",
        r"\bi(?:'m| am)\s+(?:interested in|curious about)\s+(.+)$",
    ]

    text_for_match = PREAMBLE.sub("", text_cmd_stripped).strip()

    for p in patterns:
        m = re.search(p, text_for_match, flags=re.I)
        if m:
            extracted = m.group(1)
            cleaned = _clean_topic(extracted)
            if cleaned:
                cleaned = _strip_time_phrases(cleaned)
            if cleaned:
                print(f"[DEBUG] Rule-based topic extract: {cleaned}")
                return cleaned

    # ── 5) NEW GUARD: short title-like branch, BUT with sentence rejection ──
    # The old code accepted "I open to any suggestion" as a 5-word title.
    # Now we reject sentence shapes and finite-verb-bearing fragments.
    if 1 <= len(text_cmd_stripped.split()) <= 6 and len(text_cmd_stripped) <= 60:
        # Reject sentence shapes
        if _SENTENCE_SHAPE_RE.match(text_cmd_stripped):
            print(f"[DEBUG] extract_topic_name: sentence shape, no topic")
            return None

        # Reject anything with a finite verb
        if _FINITE_VERB_RE.search(text_cmd_stripped):
            print(f"[DEBUG] extract_topic_name: finite verb, no topic")
            return None

        # Reject question-shaped short input
        if any(w in lower for w in [
            "please", "can you", "could you",
            "what is", "how do", "help me",
            "is it", "are you", "do you",
        ]):
            return None

        candidate = _clean_topic(text_cmd_stripped)
        if candidate:
            candidate = _strip_time_phrases(candidate)
        if candidate:
            return candidate

    # ── 6) LLM fallback — TIGHTER guards ────────────────────────────
    prompt = extractTopicName(text)
    response = ollama.generate(model="llama3.2:latest", prompt=prompt)
    topic = (response.get("response") or "").strip()

    topic = _clean_topic(topic)
    if not topic or topic.lower() == "none":
        return None

    # Reject single-word broad categories — LLM hallucinations
    topic_lower = topic.lower().strip()
    if topic_lower in _BROAD_CATEGORY_STOPLIST:
        print(f"[DEBUG] LLM topic too broad ({topic}); falling back to None")
        return None

    # Reject single-word topics when input is longer — overgeneralization
    if len(topic.split()) == 1 and len(text.split()) >= 3:
        # BUT: allow single-word if it's a strong content noun
        STRONG_SINGLE_WORDS = {
            "rover", "drone", "tello", "lqr", "pid",
            "spotify", "raspberry", "arduino", "esp32",
        }
        if topic_lower not in STRONG_SINGLE_WORDS:
            print(f"[DEBUG] LLM topic too short ({topic}); falling back to None")
            return None

    if not _topic_is_grounded(topic, text):
        print(f"[DEBUG] LLM topic not grounded ({topic!r}); falling back to None")
        return None

    print(f"[DEBUG] Extracted Topic Name (LLM): {topic}")
    return topic


# ─────────────────────────────────────────────────────────────────
# Tighter grounding check
# ─────────────────────────────────────────────────────────────────
def _topic_is_grounded(topic: str, source_text: str) -> bool:
    """
    Tightened: rejects when overlap is only on broad category words.
    """
    topic_tokens = {
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", topic)
        if len(t) >= 3
    }
    source_tokens = {
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", source_text)
        if len(t) >= 3
    }

    if not topic_tokens:
        return False

    overlap = topic_tokens & source_tokens

    # NEW: overlap on broad categories doesn't count
    meaningful_overlap = overlap - _BROAD_CATEGORY_STOPLIST
    if not meaningful_overlap:
        return False

    return len(overlap) / len(topic_tokens) >= 0.5


WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "a": 1,          # optional: "a day ago"
    "an": 1,         # optional
    "two": 2,
    "couple": 2,     # optional: "a couple days ago"
    "three": 3,
    "few": 3,        # optional: "a few days ago"
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

def normalize_number_words(text: str) -> str:
    """
    Replace simple number words with digits to unify downstream regex rules.
    Conservative by design (only whole-word matches).
    """
    if not text:
        return text
    out = text
    # Handle "a couple of days ago" / "a few days ago"
    out = re.sub(r"\ba\s+couple\s+of\b", "2", out)
    out = re.sub(r"\ba\s+few\s+of\b", "3", out)

    # Replace standalone number words
    for w, n in WORD_NUMBERS.items():
        out = re.sub(rf"\b{re.escape(w)}\b", str(n), out)
    return out


def _now():
    return datetime.now(TZ)

def _ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _week_range(base: datetime, offset_weeks: int, week_start: int = 0):
    """
    week_start: 0=Monday, 6=Sunday
    Returns [start_date, end_date] inclusive in YYYY-MM-DD.
    """
    base_date = base.date()
    # Monday-based weekday: Monday=0..Sunday=6
    wd = base_date.weekday()
    delta_to_start = (wd - week_start) % 7
    start = base_date - timedelta(days=delta_to_start) + timedelta(weeks=offset_weeks)
    end = start + timedelta(days=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

_WEEKDAY_NAMES = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

SAVE_INTENT_SIGNALS = ["remind", "reminder", "schedule", "meeting", "appointment", "book", "set up"]

def extract_time_reference(user_input: str, prefer_future: bool = None):
    text = (user_input or "").lower().strip()
    now = _now()
    today = now.date()

    text = normalize_number_words(text)
    text = re.sub(r"\s+", " ", text)

    # ... sections 1 and 2 unchanged ...

    # --- NEW: bare weekday name handling, added before the dateparser fallback ---
    weekday_match = re.search(_WEEKDAY_NAMES, text)
    if weekday_match:
        target_day = weekday_match.group(1)
        weekday_index = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"].index(target_day)
        days_ahead = (weekday_index - today.weekday()) % 7

        # decide direction: explicit save-intent language → always future
        wants_future = prefer_future
        if wants_future is None:
            wants_future = any(s in text for s in SAVE_INTENT_SIGNALS)

        if wants_future:
            if days_ahead == 0:
                days_ahead = 7  # "Friday" said on a Friday means next Friday, not today
            target_date = today + timedelta(days=days_ahead)
        else:
            days_back = (today.weekday() - weekday_index) % 7
            if days_back == 0:
                days_back = 7
            target_date = today - timedelta(days=days_back)

        return {"type": "date", "value": target_date.strftime("%Y-%m-%d")}

    # --- 3) Fallback: parse a likely time phrase snippet (existing dateparser path) ---
    snippet_match = re.search(
        r"\b(?:on|at|in|from|around)?\s*"
        r"(today|tomorrow|yesterday|last week|this week|next week|last month|next month|last year|next year|"
        r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
        text
    )
    snippet = snippet_match.group(1) if snippet_match else text

    settings = {
        "RELATIVE_BASE": now.replace(tzinfo=None),
        "PREFER_DATES_FROM": "future" if any(s in text for s in SAVE_INTENT_SIGNALS) else "past",
        "DATE_ORDER": "DMY",
    }

    parsed = dateparser.parse(snippet, settings=settings)
    if parsed:
        return {"type": "date", "value": parsed.strftime("%Y-%m-%d")}

    return None

def store_short_term_memory(user_input, bot_response, topic=None):
    # Guard 3 (BUDDY_CONTEXT_GUARDS): a bare "yes" that answers no question is not memory; Buddy's
    # improvised reply to it would otherwise be stored under the topic and continued later.
    if not should_store_exchange(user_input):
        print("[DEBUG] Skipped storing bare acknowledgement (no open question)")
        return

    # Prefer explicit topic, fallback to memory_manager
    topic = (topic or memory_manager.get_last_topic() or "").strip().lower()

    # Strip wrapper phrases at storage time — defensive catch-all
    STRIP_PATTERNS = [
        r"^it(?:'?s| is) about\s+",
        r"^this is about\s+",
        r"^about\s+",
        r"^it(?:'?s| is)\s+",
        r"^the topic is\s+",
        r"^topic[:\s]+",
        r"^talk(?:ing)? about\s+",
        r"^discuss(?:ing)? about\s+",
        r"^regarding\s+",
    ]
    for pattern in STRIP_PATTERNS:
        topic = re.sub(pattern, "", topic, flags=re.I).strip()

    # Reject if topic is still too long or nonsensical after stripping
    if topic and len(topic.split()) > 6:
        topic = " ".join(topic.split()[:4])

    if topic:
        if topic not in temp_topic_memory:
            temp_topic_memory[topic] = []

        temp_topic_memory[topic].append(f"User: {user_input}\nBuddy: {bot_response}")
        print(f"[DEBUG] Stored in short-term memory under topic: {topic}")
    else:
        sensory_memory.append(f"User: {user_input}\nBuddy: {bot_response}")
        print(f"[DEBUG] Stored in general sensory memory: {len(sensory_memory)} recent interactions")


def retrieve_short_term_memory(user_input=None):
    topic = memory_manager.get_last_topic()
    if not topic or len(temp_topic_memory[topic]) == 0:
        return None
    return "\n".join(temp_topic_memory[topic])


# ---------------------------
# LONG-TERM MEMORY
# ---------------------------


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

def _extract_raw_summary(text: str) -> str:
    """
    If text is a stored document (has Topic:/Date:/Summary: headers),
    extract only the summary body. Otherwise return as-is.
    """
    if "Summary:" in text:
        parts = text.split("Summary:", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return text

def _safe_date_str(date) -> str:
    """Always returns a plain YYYY-MM-DD string, no matter what comes in."""
    if date is None:
        return datetime.datetime.now().strftime("%Y-%m-%d")
    if isinstance(date, str):
        return date.strip() or datetime.datetime.now().strftime("%Y-%m-%d")
    # Handle datetime objects (with or without timezone)
    if isinstance(date, datetime.datetime):
        return date.strftime("%Y-%m-%d")
    # Fallback for anything weird (ZoneInfo object, etc.)
    try:
        return str(date)[:10]  # take first 10 chars of string representation
    except Exception:
        return datetime.datetime.now().strftime("%Y-%m-%d")


def store_long_term_memory(topic, date=None, conversation_history=None):
    global temp_topic_memory
    date = _safe_date_str(date)
    topic = (topic or "").strip().lower()
    if not topic:
        print("[DEBUG] ❌ Empty topic. Nothing to save!")
        return

    if date is None:
        date = get_date()

    # Use provided snapshot if available
    if conversation_history is None:
        if topic not in temp_topic_memory or not temp_topic_memory[topic]:
            print(f"[DEBUG] No stored short-term memory for '{topic}'. Nothing to save!")
            return
        conversation_history = "\n".join(temp_topic_memory[topic])

    print(f"\n[Updating Long-Term Memory] Topic: {topic}\n{conversation_history}\n")

    existing_memory, existing_date, topic_id = retrieve_long_term_memory(topic)
    print(f"[DBG] store_ltm existing_memory={repr(existing_memory)}")
    print(f"[DBG] store_ltm existing_date={existing_date}")
    print(f"[DBG] store_ltm topic_id={topic_id}")

    # updated_summary = (
    #     f"{existing_memory}\n\n[NEW INFO - {date}]\n{conversation_history}"
    #     if existing_memory else conversation_history
    # )
    existing_summary_only = _extract_raw_summary(existing_memory) if existing_memory else None
    updated_summary = (
        f"{existing_summary_only}\n\n[NEW INFO - {date}]\n{conversation_history}"
        if existing_summary_only else conversation_history
    )

    print(f"[DBG] store_ltm updated_summary preview={repr(updated_summary[:300]) if existing_memory else 'None - using raw history'}")


    prompt = storeSummaryLongTermMemory(topic, date, updated_summary)
    response = ollama.generate(model="llama3.2:latest", prompt=prompt)
    summary = response['response'].strip()

    doc = f"Topic: {topic}\nDate: {date}\nSummary:\n{summary}"

    embedding = embedding_model.encode(doc, convert_to_numpy=True).tolist()

    if topic_id:
        chat_collection.update(
            ids=[topic_id],
            documents=[doc],
            embeddings=[embedding],
            metadatas=[{"id": topic_id, "topic": topic, "timestamp": date}]
        )
        print(f"[✅] Updated existing topic '{topic}' in long-term memory.")
    else:
        unique_id = str(uuid.uuid4())
        chat_collection.add(
            documents=[doc],
            embeddings=[embedding],
            metadatas=[{"id": unique_id, "topic": topic, "timestamp": date}],
            ids=[unique_id]
        )
        print(f"[✅] Created new topic '{topic}' in long-term memory.")


def update_long_term_memory(topic, new_info):
    try:
        print(f"[DEBUG] Adding new information to topic: {topic}")

        existing_memory, _, topic_id = retrieve_long_term_memory(topic)

        updated_memory = f"{existing_memory}\n\n[New Info Added]:\n{new_info}"

        chat_collection.update(
            ids=[topic_id],
            documents=[updated_memory],
            metadatas=[{"topic": topic}]
        )

        print(f"Successfully updated topic: {topic}")

    except Exception as e:
        print(f"[ERROR] Failed to update memory: {e}")

def check_relevant_newInfo(topic, new_info):
    """Checks if new user input is related to the existing topic."""

    newInfoRelevantChecked = newInfoRelevantCheck(topic, new_info)

    response = ollama.generate(model="llama3.2:latest", prompt=newInfoRelevantChecked)
    decision = response['response'].strip().lower()

    valid_responses = ["yes", "no"]
    if decision not in valid_responses:
        print("[ERROR] Invalid LLM response. Defaulting to 'no'.")
        return "no"

    return decision


def retrieve_long_term_memory(user_input):
    try:
        if user_input is None:
            stored_topics = retrieve_all_topics_from_memory()
            print(f"[DEBUG] General recall requested. Returning list of past topics: {stored_topics}")

            if not stored_topics:
                return None, "I couldn't find any past discussions in my memory.", None

            formatted_topics = '\n'.join([f"- {t}" for t in stored_topics])
            return formatted_topics, None, None


        topic = extract_topic_name(user_input) or memory_manager.get_last_topic()
        if not topic:
            return None, "I couldn't find any related past conversations.", None

        # Use semantic embedding for query
        query_embedding = embedding_model.encode(topic, convert_to_numpy=True).tolist()

        results = chat_collection.query(
            query_embeddings=[query_embedding],  # ← semantic search
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        if not results.get("documents"):
            return None, f"I couldn't find past conversations about {topic}.", None

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]  # lower = more similar in ChromaDB

        # Filter by semantic distance threshold instead of fuzzy ratio
        DISTANCE_THRESHOLD = 1.2  # tune this; ChromaDB uses L2 by default

        filtered = [
            (doc, meta) for doc, meta, dist
            in zip(docs, metas, distances)
            if dist < DISTANCE_THRESHOLD
        ]

        if not filtered:
            return None, f"I couldn't find past conversations about {topic}.", None

        best_doc, best_meta = filtered[0]
        doc_topic_line = [l for l in best_doc.split("\n") if l.startswith("Topic:")]
        if doc_topic_line:
            doc_topic = doc_topic_line[0].replace("Topic:", "").strip().lower()
            if doc_topic != topic.lower():
                print(f"[DBG] retrieve_ltm: topic mismatch '{doc_topic}' vs '{topic}' — returning None")
                return None, None, None

        # return best_doc, best_meta.get("timestamp"), best_meta.get("id")
        timestamp = best_meta.get("timestamp")
        return best_doc, _safe_date_str(timestamp), best_meta.get("id")

    except Exception as e:
        print(f"[ERROR] Failed to Retrieve Long-Term Memory: {e}")
        return None, "I encountered an error while retrieving the memory.", None

def retrieve_memory_by_date(target_date):
    try:
        print(f"[DEBUG] Searching memory for date: {target_date}")

        results = chat_collection.get(include=["documents", "metadatas"])

        matches = []
        ids_to_delete = []

        for doc, metadata in zip(results["documents"], results["metadatas"]):
            memory_date = metadata.get("timestamp")
            memory_id = metadata.get("id", None)

            if memory_date == target_date:
                matches.append(doc)
                if memory_id:
                    ids_to_delete.append(memory_id)

        if not matches:
            return None, "I couldn’t find anything saved on that date.", None

        full_text = "\n\n".join(matches)
        return full_text, None, ids_to_delete

    except Exception as e:
        print(f"[ERROR] Date-based memory search failed: {e}")
        return None, "There was an error searching by date.", None

def retrieve_memory_by_month(target_month):
    try:
        print(f"[DEBUG] Searching memory for month: {target_month}")

        results = chat_collection.get(include=["documents", "metadatas"])

        matches = []
        ids_to_delete = []

        for doc, metadata in zip(results["documents"], results["metadatas"]):
            memory_date = metadata.get("timestamp", "")
            memory_id = metadata.get("id")

            if memory_date.startswith(target_month):
                matches.append(doc)
                if memory_id:
                    ids_to_delete.append(memory_id)

        if not matches:
            return None, f"I couldn’t find any memories from **{target_month}**.", None

        full_text = "\n\n".join(matches)
        return full_text, None, ids_to_delete

    except Exception as e:
        print(f"[ERROR] Failed to retrieve memory by month: {e}")
        return None, "An error occurred while searching for memories.", None

def retrieve_memory_by_year(target_year):
    try:
        print(f"[DEBUG] Searching memory for year: {target_year}")

        results = chat_collection.get(include=["documents", "metadatas"])

        matches = []
        ids_to_delete = []

        for doc, metadata in zip(results["documents"], results["metadatas"]):
            memory_date = metadata.get("timestamp", "")
            memory_id = metadata.get("id")

            if memory_date.startswith(target_year):
                matches.append(doc)
                if memory_id:
                    ids_to_delete.append(memory_id)

        if not matches:
            return None, f"I couldn’t find any memories from **{target_year}**.", None

        full_text = "\n\n".join(matches)
        return full_text, None, ids_to_delete

    except Exception as e:
        print(f"[ERROR] Failed to retrieve memory by year: {e}")
        return None, "An error occurred while searching for memories.", None


def retrieve_memory_by_week(start_date, end_date):
    try:
        print(f"[DEBUG] Searching memory between {start_date} and {end_date}")

        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        results = chat_collection.get(include=["documents", "metadatas"])

        matches = []
        ids_to_delete = []

        for doc, metadata in zip(results["documents"], results["metadatas"]):
            timestamp = metadata.get("timestamp", "")
            memory_id = metadata.get("id")

            try:
                memory_date = datetime.strptime(timestamp, "%Y-%m-%d").date()
                if start <= memory_date <= end:
                    matches.append(doc)
                    if memory_id:
                        ids_to_delete.append(memory_id)
            except:
                continue

        if not matches:
            return None, "I couldn’t find any memories from that week.", None

        full_text = "\n\n".join(matches)
        return full_text, None, ids_to_delete

    except Exception as e:
        print(f"[ERROR] Failed to retrieve memory by week: {e}")
        return None, "Something went wrong while retrieving your memory.", None


#DEBUG LONG-TERM MEMORY
def view_all_long_term_memory():
    """Retrieves and prints all stored topics and summaries from ChromaDB."""
    try:
        results = chat_collection.get(include=["documents"])

        if "documents" in results and results["documents"]:
            print("\n**Long-Term Memory Contents:**\n")
            for idx, doc in enumerate(results["documents"], 1):
                print(f"{idx}. {doc}\n")
            print("End of stored memories.\n")
        else:
            print("No long-term memory stored yet.")

    except Exception as e:
        print(f"[ERROR] Failed to retrieve long-term memory: {e}")

def store_memory_in_db(topic, summary):
    """Stores a new topic into ChromaDB with a unique ID."""
    try:
        unique_id = str(uuid.uuid4())

        chat_collection.add(
            documents=[summary],
            metadatas=[{"id": unique_id, "topic": topic}],
            ids=[unique_id]
        )

        print(f"[DEBUG] Successfully stored topic: {topic} with ID {unique_id}")
        return f"Memory about **{topic}** has been stored."

    except Exception as e:
        print(f"[ERROR] Failed to store memory: {e}")
        return "I encountered an error while saving the memory."

def delete_all_memory():
    """Deletes all stored memory from ChromaDB, even those without IDs."""
    try:
        results = chat_collection.get(include=["documents", "metadatas"])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            print("No stored memories found. Nothing to delete.")
            return

        # Collect all available IDs
        ids_to_delete = []
        topics_to_delete = []

        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            topic = metadata.get("topic", "[No Topic]")

            # If ID exists, use it; otherwise, delete by document text
            doc_id = metadata.get("id")
            if doc_id:
                ids_to_delete.append(doc_id)
            else:
                topics_to_delete.append(doc)

        if ids_to_delete:
            chat_collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} entries using IDs.")

        if topics_to_delete:
            for doc in topics_to_delete:
                chat_collection.delete(where_document={"$contains": doc})
            print(f"Deleted {len(topics_to_delete)} old entries without IDs.")

        print("All memory has been successfully erased.")

    except Exception as e:
        print(f"[ERROR] Failed to delete all memories: {e}")


def debug_chroma_db():
    """Prints the current stored documents, metadata, and IDs from ChromaDB to understand its structure."""
    try:
        # ✅ Corrected query: Only request valid fields
        results = chat_collection.get(include=["documents", "metadatas"])  # Removed "ids"

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        print("\n[📌] **ChromaDB Debugging: Stored Data Overview**")
        if not documents:
            print("[ℹ️] No stored documents found.")
        else:
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}

                doc_id = metadata.get("id", "[No ID]")  # Extract ID from metadata instead
                topic = metadata.get("topic", "[No Topic]")

                print(f"\n🔹 Document ID: {doc_id}")
                print(f"📄 Content:\n{doc}")
                print(f"🗂️ Metadata: {metadata}")
                print(f"📝 Topic: {topic}")

        print("\nEnd of database debug report.")

    except Exception as e:
        print(f"[ERROR] Failed to fetch stored data: {e}")

def update_existing_db_entries():
    """Assigns unique IDs to old database entries that are missing IDs."""
    try:
        results = chat_collection.get(include=["documents", "metadatas"])

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            print("No stored documents found.")
            return

        updates = []

        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}

            if "id" not in metadata:
                new_id = str(uuid.uuid4())
                metadata["id"] = new_id
                updates.append((doc, metadata, new_id))

        for doc, metadata, new_id in updates:
            chat_collection.add(
                documents=[doc],
                metadatas=[metadata],
                ids=[new_id]
            )

        print(f"Updated {len(updates)} entries with missing IDs.")

    except Exception as e:
        print(f"[ERROR] Failed to update database: {e}")


# [DUMMY TEST]

# Purpose:
#   1. Validate _clean_topic() candidate-level topic filtering.
#   2. Validate extract_topic_name() full-input topic extraction.
#

#
# Notes:
#   - _clean_topic() tests check whether a candidate topic is structurally valid.
#   - extract_topic_name() tests check whether real user utterances produce sane topics.
#   - If extract_topic_name() calls the LLM fallback, bad/ungrounded topics should be exposed here.

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TOPIC VALIDATOR + EXTRACTOR TEST SUITE")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # 1) _clean_topic() TESTS
    # ─────────────────────────────────────────────────────────────

    clean_topic_cases = [
        # (input, expected_output)

        # Should reject — interjections / emotional reactions
        ("oh hell no",                  None),
        ("hell no",                     None),
        ("okay, down",                  None),
        ("yeah whatever",               None),
        ("ugh this sucks",              None),
        ("wow okay",                    None),

        # Should reject — clause fragments / sentences, not topics
        ("tear hurts but more",         None),
        ("my hand just got bleeding",   None),
        ("is not yet top",              None),
        ("i'm still debugging",         None),
        ("the rover is broken",         None),
        ("battery is not the issue",    None),
        ("it finally works",            None),
        ("i already solved it",         None),

        # Should reject — danglers / relational fragments
        ("more regarding my",           None),
        ("more about my",               None),
        ("regarding my",                None),
        ("about my",                    None),
        ("related to my",               None),
        ("related to this",             None),
        ("about this",                  None),
        ("concerning that",             None),
        ("about the",                   None),
        ("with the",                    None),
        ("because of my",               None),
        ("issue with my",               None),

        # Should pass — relational phrases with real noun phrase after filler
        ("issue regarding rover wiring",           "issue regarding rover wiring"),
        ("problem about battery mounting",         "problem about battery mounting"),
        ("related to rover wiring",                "related to rover wiring"),
        ("issue with power distribution",          "issue with power distribution"),
        ("problem concerning motor controller",    "problem concerning motor controller"),

        # Should pass — real topics
        ("rover motor controller",      "rover motor controller"),
        ("battery mounting",            "battery mounting"),
        ("buck converter",              "buck converter"),
        ("debugging electronics",       "debugging electronics"),
        ("electrical connection",       "electrical connection"),
        ("the cable issue",             "the cable issue"),
        ("rover server",                "rover server"),
        ("raspberry pi",                "raspberry pi"),
        ("the communication issue",     "the communication issue"),
        ("publication paper",           "publication paper"),
        ("machine learning",            "machine learning"),
        ("feedback loop control",       "feedback loop control"),
        ("thz spectroscopy",            "thz spectroscopy"),
        ("3d printer",                  "3d printer"),
    ]

    print("\n" + "-" * 70)
    print("_clean_topic() TESTS")
    print("-" * 70)

    clean_passed = 0
    clean_failed = 0

    for input_str, expected in clean_topic_cases:
        result = _clean_topic(input_str)

        if result == expected:
            status = "✓"
            clean_passed += 1
        else:
            status = "✗"
            clean_failed += 1

        print(
            f"{status} {input_str!r:48} → {repr(result):36} "
            f"(expected {expected!r})"
        )

    print("-" * 70)
    print(
        f"_clean_topic Results: {clean_passed} passed, "
        f"{clean_failed} failed out of {len(clean_topic_cases)} total"
    )

    # ─────────────────────────────────────────────────────────────
    # 2) extract_topic_name() TESTS
    # ─────────────────────────────────────────────────────────────

    extract_topic_cases = [
        # (input, expected_output)

        # Should reject — emotional / conversational fragments
        ("Oh hell no.",                                      None),
        ("Okay, down...",                                   None),
        ("Never mind.",                                     None),
        ("Forget it.",                                      None),
        ("No need.",                                        None),
        ("No need, no need to save anything here.",          None),
        ("Sometimes I want to give up doing this.",          None),
        ("Okay, I'm good, I don't need assistance right now.", None),
        ("Thank you very much for today. I think it's enough.", None),

        # Should reject — status fragments that should not become topics
        ("I'm still debugging.",                            None),
        ("It is not yet top.",                              None),
        ("Okay now the blood stopped which seems good.",    None),
        ("It finally works.",                               None),
        ("I already solved it.",                            None),

        # Should reject — safety / body events as long-term technical topics
        ("My hand just got bleeding.",                      None),
        ("I'm cleaning the cut right now.",                 None),
        ("Wait wait wait, I just smell some burn here.",    None),

        # Should extract — explicit topic markers
        ("topic: rover motor controller",                   "rover motor controller"),
        ("subject: battery mounting",                       "battery mounting"),
        ("call it Power Distribution Issue",                "Power Distribution Issue"),
        ("name it rover wiring problem",                    "rover wiring problem"),

        # Should extract — discuss/talk/about/regarding
        ("Let's discuss rover motor controller.",            "rover motor controller"),
        ("I want to discuss about battery mounting.",        "battery mounting"),
        ("Can we talk about the rover server?",              "the rover server"),
        ("This is regarding buck converter.",                "buck converter"),
        ("About raspberry pi.",                             "raspberry pi"),

        # Should extract — issue/problem/question pattern
        ("I have an issue regarding the power distribution of my rover.",
            "the power distribution of my rover"),
        ("I have a problem about battery mounting.",
            "battery mounting"),
        ("There is a problem with the rover motor controller.",
            "the rover motor controller"),
        ("The issue with power distribution is still there.",
            "power distribution"),

        # Should extract — research context
        ("Let's switch from the rover to my THz spectroscopy paper.",
            "THz spectroscopy paper"),
        ("I want to brainstorm my third study for publication.",
            "my third study for publication"),
        ("My goal is to get at least one publication paper.",
            "publication paper"),
        ("Help me turn that into a concrete research goal.",
            "concrete research goal"),

        # Should handle recall-style questions
        ("Do we discuss something about battery?",
            "battery"),
        ("What did we discuss about rover power?",
            "rover power"),
        ("Have we talked about battery mounting?",
            "battery mounting"),
        ("What have we said regarding the motor controller?",
            "the motor controller"),

        # Should reject pure questions without a good topic
        ("What happened here?",                             None),
        ("Can you help me?",                                None),
        ("Could you explain this?",                         None),
        ("What should I do?",                               None),

        # ASR/noisy hardware phrases — expected behavior depends on your alias layer.
        # If you do not yet normalize aliases in extract_topic_name(), keep the noisy phrase.
        # If you add alias normalization, change expected output to "raspberry pi" / "rover".
        ("I'm waiting for my respiratory pipe to boot.",
            "respiratory pipe"),
        ("The reservoir pipe cable is very sensitive.",
            "reservoir pipe cable"),
        ("I'm debugging my rubber power supply.",
            "rubber power supply"),
        ("Can you start the roober?",
            None),

        # Should reject tool/action commands as topics
        ("Start the rover.",                                None),
        ("Let's start the rover please.",                   None),
        ("Rover move forward.",                             None),
        ("Make the rover move forward.",                    None),
        ("Open MATLAB.",                                    None),
        ("Delete battery from memory.",                     None),

        # Important physical-action case:
        # This must NOT become delete-memory topic.
        # Whether it extracts "battery" as a topic is your design choice.
        # I recommend None because it is an action plan, not a topic label.
        ("What I would like to do is remove the battery from the rover.",
            None),
        ("So currently I want to remove the battery mounting.",
            None),

        # Should not invent THz / reinforcement learning / laser diode
        ("No need because I already know what I'm aiming for.",
            None),
        ("Feel free if you want to see it.",
            None),
        ("Just see it if you want.",
            None),
    ]

    print("\n" + "-" * 70)
    print("extract_topic_name() TESTS")
    print("-" * 70)

    extract_passed = 0
    extract_failed = 0

    for input_str, expected in extract_topic_cases:
        try:
            result = extract_topic_name(input_str)
        except Exception as e:
            result = f"[ERROR: {type(e).__name__}: {e}]"

        if result == expected:
            status = "✓"
            extract_passed += 1
        else:
            status = "✗"
            extract_failed += 1

        print(
            f"{status} {input_str!r:64} → {repr(result):38} "
            f"(expected {expected!r})"
        )

    print("-" * 70)
    print(
        f"extract_topic_name Results: {extract_passed} passed, "
        f"{extract_failed} failed out of {len(extract_topic_cases)} total"
    )

    # ─────────────────────────────────────────────────────────────
    # Final report
    # ─────────────────────────────────────────────────────────────

    total_passed = clean_passed + extract_passed
    total_failed = clean_failed + extract_failed
    total_cases = len(clean_topic_cases) + len(extract_topic_cases)

    print("\n" + "=" * 70)
    print("FINAL TOPIC SYSTEM TEST RESULTS")
    print("=" * 70)
    print(f"_clean_topic:        {clean_passed}/{len(clean_topic_cases)} passed")
    print(f"extract_topic_name: {extract_passed}/{len(extract_topic_cases)} passed")
    print("-" * 70)
    print(f"TOTAL:              {total_passed}/{total_cases} passed")
    print(f"FAILED:             {total_failed}")
    print("=" * 70)