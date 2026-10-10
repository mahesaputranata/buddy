from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import ollama
import psutil
import requests

from brain.adaptive.session_logger import session_logger
from brain.adaptive.session_recap import SessionRecap
from brain.core.brain import (
    extract_topic_name, store_short_term_memory, generate_llm_response, memory_manager, MemoryManager,
    compute_similarity_hybrid, resolve_save_target, recall_memory_request, retrieve_long_term_memory,
    sensory_memory, compute_similarity_llm, temp_topic_memory, delete_memory_from_db,
    extract_scheduleType_recall, _clean_topic, save_memory_request, delete_memory_request,
    extract_time_reference, retrieve_memory_by_date, delete_memory_by_ids, retrieve_memory_by_month,
    retrieve_memory_by_year, retrieve_memory_by_week, store_long_term_memory,
)
from brain.core.helpers import get_date
from brain.core.prompts import (
    build_summary_prompt, llmBasedIntentDetection, schedule_extract_with_followup_prompt,
)
from brain.core.sql_brain import (
    create_schedule, get_all_schedules_as_dicts, safe_json_parse, smart_schedule_recall,
    extractScheduleDetails, retrieve_schedule_by_week_sql, init_schedule_table, retrieve_schedule_by_date_sql,
    retrieve_schedule_by_year_sql, retrieve_schedule_by_month_sql, delete_schedule_by_title,
    delete_schedules_by_ids,
)
from brain.utils.extract_name_context import build_schedule_title
from brain.core.devices.device_handler import DeviceHandlerFSM
from devices.tello_client import TelloClient
from devices.rover.rover_bridge import RoverClient
from brain.utils.text_sanitizer import sanitize_for_tts
from brain.utils.context_guards import note_last_reply
from brain.settings import setting
from brain.core.internet_access.internet_service import get_explorer
from brain.adaptive.vision.vision_awareness import (
    get_camera_offer_text, score_vision_need, VisionDecision, get_silent_prefix,
)
from brain.router.confirmation_router import (
    resolve_vision_source_choice, VisionSourceChoice, resolve_confirmation, ConfirmResult,
)
from brain.intent_model.shadow_classifier import get_shadow
from vision.webcam_capture import (
    capture_single_frame, WEBCAM_SAVE_DIR, capture_burst_frames, wants_burst_mode,
)
from vision.run_vision_pipeline import run_vision_pipeline
from vision.face_recognition_module.recognizer import recognize_face, RecognitionStatus
from brain.friday.job_watchdog import JobWatchdog, JOB_WATCHDOG_PHRASES
from brain.router.briefing import (
    route_via_llm, filter_tool_call, make_dispatch_table, BUDDY_TOOL_TO_INTENT, BUDDY_TOOL_SCHEMAS,
    _tool_call_name, _tool_call_args, _passes_grounding,
)
from brain.memory.episodic_memory_router import EpisodicMemoryRouter, MemoryRoute
from brain.memory.memory_task_fusion import MemoryTaskFusion
from devices.rover.simulation_manager import SimulationManager
from brain.conversation.followup_detector import (
    is_short_followup, is_explicit_topic_shift, _is_explicit_web_exit,
)
from brain.friday.friday import Friday, FridayConfig
from brain.memory.context_assembler import ContextAssembler
from brain.adaptive.agent_core import init_agent, get_agent
from brain.belief_state import BeliefStateManager, Domain, Phase
from brain.utils.baseline_memory.handler.specific_flow import SpecificFlow
from brain.utils.baseline_memory.recall_baseline_helper import (
    handleGeneral, compiler_general, resolve_topic_from_utterance, _tts_join,
)
from brain.utils.schedule.recall_helper import handle_general_schedule
from brain.utils.interupt_detector import is_interrupt, is_resume
from brain.projects.project_handler import ProjectHandler, detect_stressed_test_trigger
from brain.music_player.spotify_manager import (
    SpotifyManager, MusicWatcher, detect_music_intent, handle_music_command,
)
from ear.push_to_talk import VOICE_QUEUE, TextHotkey, VADRecorder
from ear.voice_pipeline import SpeakerWorkerClient
from ear.stt_faster_whisper import transcribe_wav
from mouth.debug.mouth_v1 import TTSWorker
from vision.clipboard_hotkey import ClipboardHotkey
from vision.clipboard_capture import grab_clipboard_image
from vision.screenshot_selector import select_screenshots, SelectionOutcome
from vision.vision_source_classifier import classify_vision_source, VisionSource
from brain.core.config.features import USE_RAG, RAG_WEB_CONTEXT, RAG_TOP_K, RAG_PER_SOURCE_K
from brain.core.RAG.engine import RetrievalEngine
from brain.core.RAG.prompt.builder import PromptBuilder
from brain.core.RAG.types import RetrievalOptions
from brain.core.RAG.sources.short_term import ShortTermSource
from brain.core.RAG.sources.web import WebContextSource
from brain.core.RAG.rag_core import RagCore, RagContext
from brain.core.RAG.web_actions import parse_web_followup
from brain.core.RAG.web_followup_helper import WebFollowupHelper
from brain.core.RAG.web_page_fetcher import WebPageFetcher
from brain.primary_task.save_task.save_type_classifier import SaveTypeClassifier
from brain.identity.identity_manager import IdentityManager
from brain.utils.confirmation import is_yes


MODEL_NAME = "llama3.2:latest"
TZ = ZoneInfo("Europe/Berlin")


# Define Payload Dataclasses
@dataclass
class RecallPayload:
    topics: List[str] = field(default_factory=list)
    ref_value: str = ""
    matching_ids: List[str] = field(default_factory=list)
    memory_text: str = ""

@dataclass
class SchedulePayload:
    title: Optional[str] = None
    timestamp: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    raw_text: str = ""
    time_confirmed_by_user: bool = False

@dataclass
class ConversationSavePayload:
    topic: str = ""
    history: List[str] = field(default_factory=list)
    date: str = ""

@dataclass
class TopicPayload:
    topic: str = ""
    draft: str = ""

@dataclass
class VisionPayload:
    user_question: str = ""
    burst_paths: List[str] = field(default_factory=list)

@dataclass
class CandidatesPayload:
    candidates: List[str] = field(default_factory=list)


# Rover Stuff
ROVER_SAVE_SIGNALS = ["log", "telemetry", "data", "trial", "run", "response",
                       "reading", "sample", "csv", "movement", "motor", "wheel"]
CONVERSATION_SAVE_SIGNALS = ["conversation", "discussion", "topic", "notes",
                             "chat", "what we talked", "this idea", "our conversation"]

def _rover_has_telemetry(rover_manager) -> bool:
    """
    True only if save_last_log() could actually write something. Mirrors that method's
    own guard (verdict with samples AND buffered session samples), so the router never
    picks "save the rover log" when the handler would answer "I don't have any recent
    telemetry to save yet." (e.g. rover offline, or a topic that merely mentions "rover").
    """
    verdict = getattr(rover_manager, "_last_telemetry_verdict", None)
    return bool(getattr(verdict, "samples", None)) and bool(getattr(rover_manager, "_session_samples", None))


def score_save_target(user_input: str, mm, rover_manager) -> dict:
    text = (user_input or "").lower()
    last_topic = (mm.get_last_topic() or "").strip()

    rover_hits = sum(1 for w in ROVER_SAVE_SIGNALS if w in text)
    convo_hits = sum(1 for w in CONVERSATION_SAVE_SIGNALS if w in text)

    # The rover context only makes "save it" mean "save the rover log" if there is a log.
    has_telemetry = _rover_has_telemetry(rover_manager)
    session_prior = 0.35 if has_telemetry and (
        last_topic.startswith("rover:") or getattr(rover_manager, "session_active", False)
    ) else 0.0

    rover_score = min(1.0, rover_hits * 0.3 + session_prior)
    convo_score = min(1.0, convo_hits * 0.3)

    # Nothing to save on the rover side and no explicit rover-log request ("save the rover
    # telemetry log" still scores >= 0.5 and gets the honest "no telemetry" reply):
    # treat it as an ordinary save of the conversation instead of hijacking it.
    if not has_telemetry and rover_score < 0.5:
        return {"target": "conversation", "score": convo_score, "reason": "no_rover_telemetry_to_save"}

    if rover_score >= 0.5 and rover_score - convo_score >= 0.2:
        return {"target": "rover_log", "score": rover_score, "reason": f"rover_hits={rover_hits} prior={session_prior}"}
    if convo_score >= 0.5 and convo_score - rover_score >= 0.2:
        return {"target": "conversation", "score": convo_score, "reason": f"convo_hits={convo_hits}"}
    if session_prior > 0 and rover_hits == 0 and convo_hits == 0:
        return {"target": "rover_log", "score": session_prior, "reason": "bare_save_in_rover_context"}
    return {"target": "ambiguous", "score": max(rover_score, convo_score), "reason": "insufficient_signal"}


_TIMEISH_TITLE_RE = re.compile(
    r"^(at\s+)?(\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?|noon|midnight)\b",
    re.IGNORECASE
)

def looks_like_time_only_title(title: str) -> bool:
    if not isinstance(title, str):
        return False
    t = title.strip().lower()
    return bool(_TIMEISH_TITLE_RE.match(t))


_TIME_RE = re.compile(
    r"""
    \b(
        # 24h or 12h with minutes: 8:30, 08:30, 8:30pm, 8:30 pm
        \d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?
        |
        # EU style with dot minutes: 8.30, 08.30, 8.30pm
        \d{1,2}\.\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?
        |
        # hour only: 8, 12, 8pm, 12am, 8 pm
        \d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)?
        |
        # keywords
        noon|midnight
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _open_url_in_browser(url: str) -> bool:
    try:
        return webbrowser.open(url)
    except Exception:
        return False

def user_mentioned_time(text: str) -> bool:
    return bool(_TIME_RE.search(text or ""))


VAGUE_TOPICS = {"this", "that", "it", "yes", "no", "ok", "okay",
                "none", "something", "nothing", "stuff", "thing"}

DELETE_ACTION_PATTERNS = [
        "delete our",
        "delete my",
        "delete the",
        "remove our",
        "remove my",
        "remove the",
        "erase our",
        "erase my",
        "forget about",
        "forget our",
        "forget my",
        "wipe our",
        "wipe my",
        "clear our",
        "clear my",
        "remove about",
        "delete about",
        "erase about",
        "forget everything",
        "forget everything we",
        "erase everything",
        "delete everything",
]

MEMORY_TARGET_WORDS = [
        "conversation", "discussion", "memory", "topic",
        "talk", "notes", "chat", "history", "about",
        "rover", "drone", "project", "tello",
]


SAVE_CONVERSATION_PATTERNS = [
    "remember about this conversation",
    "remember what we discussed about",
    "remember what we talked about regarding",
    "remember this conversation",
    "remember our conversation",
    "save this conversation",
    "save our conversation",
    "save our previous conversation",    # ← add
    "save our last conversation",        # ← add
    "save our recent conversation",      # ← add
    "store this conversation",
    "keep this conversation",
    "remember what we talked about",
    "remember what we discussed",
    "save what we discussed",
    "save what we talked about",
    "can you remember this",
    "please remember this",
    "don't forget this conversation",
    "dont forget this conversation",
    "remember about this",
    "remember this discussion",
    "save our previous discussion",      # ← add
    "save this discussion",              # ← add
    "save this entire discussion",
    "save this whole discussion",
    "save this entire conversation",
    "save this whole conversation",
    "save everything we discussed",
    "save all of this",
    "save this session",
    "save this whole session",
    "i want to come back to this",     # implies save
    "save this so i can come back",
    "save this entire discussion",
    "save this whole discussion",
    "save this entire conversation",
    "save this whole conversation",
    "save everything we discussed",
    "save all of this",
    "save this session",
    "save this whole session",
    "i want to come back to this",
    "save this so i can come back",
]
# USER MENTIONED EVENT TITLE
_EVENT_WORDS = {
    "meeting","appointment","call","class","deadline","interview",
    "flight","dentist","doctor","lecture","exam","workout","gym",
    "shopping","groceries","pickup","pick up","deliver","delivery"
}

_WITH_NAME_RE = re.compile(r"\bwith\s+([a-z]+)\b", re.IGNORECASE)
_WITH_STOPWORDS = {"it","this","that","them","him","her","me","you","us","everyone","someone","anyone"}

def user_mentioned_event_title(text: str) -> bool:
    t = (text or "").strip().lower()
    if any(w in t for w in _EVENT_WORDS):
        return True

    m = _WITH_NAME_RE.search(t)
    if m:
        who = m.group(1).lower()
        if who not in _WITH_STOPWORDS:
            return True

    return False

### USER MENTIONED DATE ###
_DATE_WORDS = {
    "today","tomorrow","yesterday",
    "this week","next week","last week",
    "this month","next month","last month",
    "this year","next year","last year",
    "weekend","this weekend","next weekend",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday"
}

_RELATIVE_RE = re.compile(r"\bin\s+\d+\s+(day|days|week|weeks|month|months|year|years)\b", re.IGNORECASE)
_ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_DMY_SLASH_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DMY_DOT_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
_MONTHNAME_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.IGNORECASE)

def user_mentioned_date(text: str) -> bool:
    t = (text or "").lower()
    if any(w in t for w in _DATE_WORDS):
        return True
    if _RELATIVE_RE.search(t):
        return True
    if _ISO_RE.search(t) or _DMY_SLASH_RE.search(t) or _DMY_DOT_RE.search(t):
        return True
    if _MONTHNAME_RE.search(t):
        return True
    return False


###### USER WANT VISION ##########
def wants_vision(user_text: str) -> bool:
    t = re.sub(r"[?.!,]", "", (user_text or "").lower().strip())

    RECALL_OVERRIDE_SIGNALS = [
        "summary", "summarize", "memory", "saved", "note", "notes",
        "you wrote", "you saved", "you said", "what did you", "what did we",
        "recall", "previous conversation", "past conversation",
    ]
    if any(s in t for s in RECALL_OVERRIDE_SIGNALS):
        return False

    triggers = [
        # screenshot triggers
        "analyze screenshot", "analyze image", "look at this", "what do you see",
        "read this", "what error", "explain this", "what is this", "summarize this",
        "check this", "debug this",
        "do you see what i see", "look at my screen", "can you see my screen",
        "buddy look", "look at this screen", "what's on my screen",
        "what is on my screen", "can you see this", "see what i see",
        "take a look at this", "take a look at my screen",
        "analyze what's on my screen", "analyze my screen",
        "what do you think about this", "what am i looking at",
        # webcam triggers
        "look at me", "can you see me", "see me", "watch me",
        "what am i doing", "what do i look like", "look at my face",
        "observe me", "are you watching", "look at my hands",
        "what am i holding", "can you watch", "look at this object",
        "what is this thing", "what do i have",
    ]
    return any(k in t for k in triggers)

def _extract_vision_topic(analysis: str, img_path: str) -> str:
    """
    Generate a short human-friendly topic label from the vision analysis.
    Falls back to a hash-based key if LLM fails.
    """
    try:
        label_prompt = (
            "Given this image analysis, generate a short 3-5 word topic label "
            "that describes what the image is about. "
            "Return ONLY the label, no punctuation, no extra words.\n\n"
            f"Analysis: {analysis[:300]}"
        )
        response = ollama.generate(model="llama3.2:latest", prompt=label_prompt)
        label = (response.get("response") or "").strip()[:50]

        # sanitize — remove quotes, punctuation, lowercase
        label = re.sub(r"[\"'.,!?]", "", label).strip().lower()

        if label and len(label.split()) <= 6:
            return f"vision: {label}"
    except Exception:
        pass

    # fallback to hash-based key
    img_name = Path(img_path).stem
    return f"vision_{img_name[-8:]}"

# ─── Friday task helpers ──────────────────────────────────────────────────────
def format_tool_confirm_question(tool_name: str) -> str:
    return f"Should I go ahead and {(tool_name or 'that').replace('_', ' ')}?"

def format_friday_confirm_question(task: str, reason: str) -> str:
    return f"Should I ask Friday to {task}? This looks sensitive because {reason}."

def format_vision_burst_confirm(burst_paths: list) -> str:
    return (
        f"Should I analyze all {len(burst_paths)} screenshots together, "
        "or just the latest one?"
    )
def extract_friday_task(user_input: str) -> str:
    """Strip the Friday trigger phrase and return just the task."""
    text = (user_input or "").strip()
    STRIP_PREFIXES = [
        r"^friday[,\s]+please\s+",
        r"^friday[,\s]+",
        r"^hey friday[,\s]+",
        r"^ask friday to\s+",
        r"^ask friday\s+to\s+",
        r"^tell friday to\s+",
        r"^tell friday\s+to\s+",
        r"^friday please\s+",
    ]
    for pattern in STRIP_PREFIXES:
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if cleaned.lower() != text.lower():
            text = cleaned
            break

    # ── Project alias substitution ─────────────────────────────────────────
    # Spoken aliases → real project names (avoids saying sensitive names to STT)
    PROJECT_ALIASES = {
        r"\bdummy project\b": r"JASTIP (demo project)",
        r"\bthe dummy\b": r"JASTIP (demo project)",
        r"\bmy dummy\b": r"JASTIP (demo project)",
    }
    for pattern, replacement in PROJECT_ALIASES.items():
        text = re.sub(pattern, lambda m: replacement, text, flags=re.IGNORECASE)

    return text


def score_friday_task_risk(task: str) -> tuple[float, str]:
    """
    Score how risky a Friday task is.
    Returns (risk_score 0.0-1.0, reason string).
    High risk (>= 0.7) triggers confirmation before dispatch.
    """
    text = (task or "").lower()

    HIGH_RISK_SIGNALS = [
        ("delete", "potentially destructive — deletes data"),
        ("remove", "potentially destructive — removes data"),
        ("drop", "potentially destructive — drops data"),
        ("wipe", "potentially destructive — wipes data"),
        ("push", "touches deployment pipeline"),
        ("deploy", "touches deployment pipeline"),
        ("merge", "touches deployment pipeline"),
        ("production", "touches production environment"),
        ("git push", "touches git remote"),
        ("force push", "touches git remote with force"),
    ]

    MEDIUM_RISK_SIGNALS = [
        ("update", "modifies existing code"),
        ("change", "modifies existing code"),
        ("modify", "modifies existing code"),
        ("rewrite", "rewrites existing code"),
        ("run", "executes a command"),
        ("execute", "executes a command"),
        ("stress test", "runs live test against production"),
    ]

    LOW_RISK_SIGNALS = [
        "analyze", "check", "audit", "report", "read",
        "review", "summarize", "plan", "draft", "list",
        "implement", "add", "build", "create", "fix",
        "install", "generate",
    ]

    for signal, reason in HIGH_RISK_SIGNALS:
        if signal in text:
            return 0.9, reason

    for signal, reason in MEDIUM_RISK_SIGNALS:
        if signal in text:
            return 0.5, reason

    for signal in LOW_RISK_SIGNALS:
        if signal in text:
            return 0.1, "low risk task"

    return 0.2, "unrecognized task pattern"

def _handle_vision_request(user_text: str, conversation_manager, tts) -> str:
    mm = conversation_manager.memory_manager

    # ── Stage 1: Classify source ───────────────────────────────────────────
    classification = classify_vision_source(user_text)
    print(f"[VISION] source={classification.source.name} "
          f"conf={classification.confidence:.2f} "
          f"method={classification.method}")

    # ── Stage 2: AMBIGUOUS ─────────────────────────────────────────────────
    if classification.source == VisionSource.AMBIGUOUS:
        mm.set_pending_topic("Awaiting_Vision_Source_Confirm")
        mm.pending_payload = {"user_question": user_text}
        return classification.message

    # ── Stage 3: WEBCAM path ───────────────────────────────────────────────
    if classification.source == VisionSource.ACCESS_WEBCAM:
        if wants_burst_mode(user_text):
            paths = capture_burst_frames(out_dir=WEBCAM_SAVE_DIR)
            if not paths:
                return "I tried to access the webcam for a detailed look but couldn't capture any frames."

            recognition = recognize_face(paths[0])
            if recognition.status == RecognitionStatus.UNKNOWN:
                return recognition.message

            # inject identity context if authorized
            identity_context = "The person in this image is the authorized user. " \
                if recognition.status == RecognitionStatus.AUTHORIZED else ""
            augmented_question = identity_context + user_text

            result = run_vision_pipeline(
                image_path=paths,
                user_question=augmented_question,
                conversation_manager=conversation_manager,
                authorized_user=(recognition.status == RecognitionStatus.AUTHORIZED),
                is_webcam=True,
            )
            topic_key = _extract_vision_topic(result, paths[0])

        else:

            path = capture_single_frame(out_dir=WEBCAM_SAVE_DIR)
            if not path:
                conversation_manager.belief.report_tool_result("webcam", success=False,
                                                                summary="no frame captured")
                return "I couldn't access the webcam right now."
            else:
                conversation_manager.belief.report_tool_result("webcam", success=True,
                                                                summary="frame captured")

            recognition = recognize_face(path)
            if recognition.status == RecognitionStatus.UNKNOWN:
                return recognition.message

            # inject identity context if authorized
            identity_context = "The person in this image is the authorized user. " \
                if recognition.status == RecognitionStatus.AUTHORIZED else ""
            augmented_question = identity_context + user_text

            result = run_vision_pipeline(
                image_path=path,
                user_question=augmented_question,
                conversation_manager=conversation_manager,
                authorized_user=(recognition.status == RecognitionStatus.AUTHORIZED),
                is_webcam=True,
            )
            topic_key = _extract_vision_topic(result, path)

        store_short_term_memory(user_text, result, topic=topic_key)
        mm.set_last_topic(topic_key)
        mm.set_active_vision_source("webcam")
        return result

    # ── Stage 4: SCREENSHOT path ───────────────────────────────────────────
    grabbed = grab_clipboard_image(out_dir="screenshots")
    if grabbed:
        img_path, h = grabbed
        print(f"[VISION] Fresh clipboard grab: {img_path}")
        result = run_vision_pipeline(
            image_path=img_path,
            user_question=user_text,
            conversation_manager=conversation_manager,
        )
        topic_key = _extract_vision_topic(result, img_path)
        store_short_term_memory(user_text, result, topic=topic_key)
        mm.set_last_topic(topic_key)
        mm.set_active_vision_source("screenshot")
        print(f"[VISION] screenshot topic_key={topic_key!r}")
        return result

    # No fresh clipboard — fall back to folder scan
    selection = select_screenshots()

    if selection.outcome == SelectionOutcome.EMPTY:
        return selection.message

    if selection.outcome == SelectionOutcome.STALE:
        mm.set_pending_vision(selection.paths[0], img_hash="")
        mm.set_pending_topic("Awaiting_Vision_Confirm")
        return selection.message

    if selection.outcome == SelectionOutcome.SINGLE:
        result = run_vision_pipeline(
            image_path=selection.paths[0],
            user_question=user_text,
            conversation_manager=conversation_manager,
        )
        topic_key = _extract_vision_topic(result, selection.paths[0])
        store_short_term_memory(user_text, result, topic=topic_key)
        mm.set_last_topic(topic_key)
        mm.set_active_vision_source("screenshot")
        print(f"[VISION] screenshot topic_key={topic_key!r}")
        return result

    if selection.outcome == SelectionOutcome.BURST:
        mm.set_pending_topic("Awaiting_Vision_Burst_Confirm")
        mm.pending_payload = {
            "burst_paths": selection.paths,
            "user_question": user_text,
        }
        return selection.message

    return "Something went wrong with the screenshot selection. Please try again."

# helper
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    # strip trailing punctuation first
    t = re.sub(r"[.!?,]+$", "", t).strip()
    # normalize common filler suffixes
    t = re.sub(r"[,\s]+(please|do it|go ahead|save it|store it|confirm|sure|now)\.?$", "", t).strip()
    # normalize "yes/yeah/yep + filler" → "yes"
    t = re.sub(
        r"^(yes|yeah|yep)[,\s]+(please|sure|okay|ok|do it|go ahead|save it|store it|absolutely)\.?$",
        "yes", t
    )

    if t in {
        # bare confirmations
        "yes", "y", "yeah", "yep", "sure", "ok", "okay", "alright",
        # action confirmations
        "do it", "go ahead", "save it", "store it", "confirm",
        "please", "absolutely", "correct", "affirmative",
        # natural language yes
        "that's right", "thats right", "that's correct", "thats correct",
        "sounds good", "looks good", "perfect", "great",
        # compound yes
        "yes go ahead", "yeah go ahead", "yes please",
        "yes do it", "yes sure", "ok go ahead", "okay go ahead",
        "yes save it", "yes store it", "yes save", "yes store",
        "go ahead and save", "go ahead save",
        "go ahead and store", "go ahead store",
        "yeah sure", "yeah ok", "yeah okay",
    }:
        return True

    # catch "yes + anything" patterns
    if re.match(r"^(yes|yeah|yep|sure|ok|okay)\b", t) and len(t.split()) <= 4:
        # exclude if it contains negation
        if not re.search(r"\b(no|not|don'?t|never|cancel|stop)\b", t):
            return True

    return False


def is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    # strip trailing punctuation first
    t = re.sub(r"[.!?,]+$", "", t).strip()
    # normalize common filler suffixes
    t = re.sub(r"[,\s]+(please|thanks|thank you|for now|at all)\.?$", "", t).strip()
    # normalize "no/nope/nah + filler" → "no"
    t = re.sub(
        r"^(no|nope|nah)[,\s]+(thanks|thank you|not now|later|don'?t|please)\.?$",
        "no", t
    )

    if t in {
        # bare rejections
        "no", "n", "nope", "nah", "nah",
        # soft rejections
        "not now", "later", "not yet", "not today",
        # commands
        "cancel", "skip", "stop", "abort",
        # negations
        "don't", "dont", "do not",
        "negative", "never mind", "nevermind",
        "forget it", "forget that",
        # specific save/store rejections
        "don't save", "dont save", "no save",
        "don't store", "dont store", "no store",
        "don't save it", "dont save it",
        "don't store it", "dont store it",
        # polite rejections
        "no thanks", "no thank you",
        "no need", "not needed", "not necessary",
        # casual rejections
        "that's fine", "thats fine", "i'm good", "im good",
        "all good", "we're good", "were good",
        # compound no
        "no don't save", "no dont save",
        "no don't save it", "no dont save it",
        "no don't store it", "no dont store it",
        "please don't save", "please dont save",
    }:
        return True

    # catch "no + negation verb" patterns
    if re.match(r"^no[,\s]+(don'?t|stop|never|please don'?t|do not)\b", t):
        return True

    # catch "don't/please don't + save/store" patterns
    if re.match(r"^(don'?t|please don'?t|stop)\s+(save|store)\b", t):
        return True

    # catch "no + action" short phrases like "no don't save it"
    if re.match(r"^no\b", t) and re.search(r"\b(save|store|don'?t|never|cancel)\b", t):
        if len(t.split()) <= 5:
            return True

    return False



def looks_like_vision_question(text: str) -> bool:
    t = (text or "").lower()
    triggers = ["this", "here", "in the screenshot", "on my screen", "what do you see", "what is this"]
    return any(x in t for x in triggers)

def looks_like_domain_number_usage(text: str) -> bool:
    t = (text or "").lower()

    return bool(re.search(
        r"\b\d+\s+(sensor|sensors|controller|controllers|loop|loops|state|states|input|inputs|"
        r"method|methods|camera|cameras|lidar|constraint|constraints|uav|uavs|drone|drones|"
        r"axis|axes|motor|motors|model|models|gain|gains)\b",
        t
    ))

def _meaningful_terms(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "of", "in", "on", "at", "for", "to", "from",
        "about", "regarding", "current", "latest", "news", "situation",
        "search", "web", "result", "article", "topic", "please", "show",
        "open", "summarize", "summary", "compare", "and", "or", "with"
    }

    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if len(w) >= 3 and w not in stop]


def _score_session_match(text: str, topic: str, query: str) -> int:
    score = 0

    # normalize topic by stripping leading "web:"
    topic_core = topic[4:] if topic.startswith("web:") else topic

    # tokenize lightly
    topic_terms = _meaningful_terms(topic_core)
    query_terms = _meaningful_terms(query)

    # exact substring matches
    if topic_core and topic_core in text:
        score += 3
    if query and query in text:
        score += 3

    # overlapping meaningful terms
    overlap = set(topic_terms) | set(query_terms)
    overlap_count = sum(1 for w in overlap if re.search(rf"\b{re.escape(w)}\b", text))
    score += overlap_count

    return score


def _matches_known_web_session(text: str, mm) -> bool:
    web_contexts = getattr(mm, "_web_contexts", {}) or {}
    for topic, ctx in web_contexts.items():
        q = (ctx.get("query") or "").strip().lower()
        topic_l = (topic or "").strip().lower()
        if _score_session_match(text, topic_l, q) >= 2:
            return True
    return False


def resolve_web_session_reference(text: str, mm) -> str | None:
    t = (text or "").strip().lower()
    if not t:
        return None

    web_contexts = getattr(mm, "_web_contexts", {}) or {}
    if not web_contexts:
        return None

    # 1) direct match against known topic/query words
    best_topic = None
    best_score = 0

    for topic, ctx in web_contexts.items():
        q = (ctx.get("query") or "").strip().lower()
        topic_l = (topic or "").strip().lower()

        score = _score_session_match(t, topic_l, q)
        if score > best_score:
            best_score = score
            best_topic = topic

    # require a meaningful match
    if best_score >= 2:
        return best_topic

    # 2) fallback for explicitly "previous/earlier/last search" phrasing
    if re.search(r"\b(previous|earlier|last)\s+(search|web search|article|result|web results?)\b", t):
        return mm.get_latest_web_topic()

    if re.search(r"\b(go back|return to|back to|reopen)\b", t):
        return mm.get_latest_web_topic()

    return None

def score_web_callback(text: str, mm) -> float:
    t = (text or "").strip().lower()
    if not t:
        return 0.0

    web_contexts = getattr(mm, "_web_contexts", {}) or {}
    if not web_contexts:
        return 0.0

    score = 0.0

    # strong callback cues
    if re.search(r"\b(go back|return to|back to|reopen|open again|bring back)\b", t):
        score += 0.45

    if re.search(r"\b(previous|earlier|last)\s+(search|web search|article|result|news|web results?)\b", t):
        score += 0.40

    # explicit web/article/result framing
    if re.search(r"\b(article|result|search|web search|web result|news article|source)\b", t):
        score += 0.20

    # explicit follow-up actions help, but do not dominate
    if re.search(r"\b(open|show|read|summarize|summary|compare|explain)\b", t):
        score += 0.15

    # "the first one / second one" is weak callback evidence by itself
    if re.search(r"\bthe\s+(first|second|third|fourth|fifth)\s+one\b", t):
        score += 0.10

    # mention of a known web topic/query gives strong evidence
    known_match = _matches_known_web_session(t, mm)
    if known_match:
        score += 0.35

    # suppress obvious non-callback technical/domain number usage
    if looks_like_domain_number_usage(t):
        score -= 0.50

    # weak generic pronouns should not create callback by themselves
    if re.fullmatch(r"(it|this|that|those|these)", t):
        score -= 0.25

    return max(0.0, min(score, 1.0))
###################################

def user_mentioned_time(text: str) -> bool:
    return bool(_TIME_RE.search(text or ""))

def get_time():
    return datetime.now(TZ)

def get_time_str():
    return get_time().strftime("%H:%M:%S")


def get_battery_status():
    battery = psutil.sensors_battery()
    if battery:
        percent = battery.percent
        charging = "charging" if battery.power_plugged else "not charging"
        return f"Battery at {percent}% and {charging}."
    return "Battery status not available."

def get_cpu_usage():
    return f"Current CPU usage is {psutil.cpu_percent(interval=1)}%."

def get_weather_from_windows():
    try:
        powershell_command = 'powershell -Command "& {Get-Weather -ErrorAction Stop}"'
        weather_output = subprocess.check_output(
            powershell_command,
            shell=True,
            text=True,
            stderr=subprocess.STDOUT  # capture stderr too
        )
        output = weather_output.strip()

        # Guard: if output looks like an error, return clean fallback
        if not output or len(output) > 200 or "not recognized" in output.lower() or "error" in output.lower():
            return "Weather unavailable."

        return f"Current weather: {output}"
    except Exception:
        return "Weather unavailable."


def get_location_from_windows():
    """
    Approximate location based on public IP.
    Returns a short human-readable string.
    Note: city-level accuracy; can be wrong with VPN/campus networks.
    """
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        r.raise_for_status()
        data = r.json()

        city = data.get("city")
        region = data.get("region")
        country = data.get("country")
        loc = data.get("loc")  # "lat,lon"

        parts = [p for p in [city, region, country] if p]
        place = ", ".join(parts) if parts else "Unknown location"
        if loc:
            return f"{place} (approx. coords: {loc})"
        return place
    except Exception as e:
        return f"Location unavailable ({e})"

def open_application(app_name):
    apps = {
        "browser": "chrome" if os.name == "nt" else "google-chrome",
        "file explorer": "explorer" if os.name == "nt" else "nautilus",
        "notepad": "notepad" if os.name == "nt" else "gedit",
        "calculator": "calc" if os.name == "nt" else "gnome-calculator",
    }
    if app_name in apps:
        os.system(apps[app_name])
        return f"Opening {app_name}..."
    return f"Sorry, I don't know how to open {app_name}."

def preprocess_input(user_input):
    user_input = user_input.strip().lower()

    # Remove common filler words
    filler_words = ["please", "could you", "can you", "would you mind"]
    for word in filler_words:
        user_input = user_input.replace(word, "").strip()

    return user_input

#Pre-coded detection

def rule_based_intent_detection(user_input):
    keywords = {
        "recall_memory": [ "what did we talk about", "remind me", "recall our talk", "did we talk about", "can you recall"],
        "save_memory": ["save this", "store this", "note this down", "this is important", "keep this in mind", "bookmark this"],
        "delete_memory": ["forget", "erase", "remove", "delete"],
        "time": ["what time", "current time"],
        "date": ["today's date", "what date"],
        "battery": ["battery status", "check battery"],
        "cpu": ["cpu usage", "processor usage"],
        "weather": ["what’s the weather", "weather", "sky condition"],
        "location": ["where am i", "my location"]
    }

    user_input = user_input.lower()

    for intent, triggers in keywords.items():
        if any(phrase in user_input for phrase in triggers):
            return intent

    return None


#HYBRID APPROACH INTENT DETECTION
ALLOWED_INTENTS = [
    "schedule_save",    # saving a reminder/event/appointment
    "schedule_recall",  # asking about existing schedule
    "schedule_delete",
    "save_memory",      # saving a topic/note
    "recall_memory",    # recalling a saved topic
    "delete_memory",    # deleting saved memory
    "web_search",       # new search request
    "web_followup",     # followup on previous search results
    "device_control",   # drone/robot control
    "system_query",     # time/date/battery/weather/location
    "chat",             # general conversation
    "friday_task",
    "job_watchdog",
]

STRONG_RULES: list[tuple[list[str], str]] = [
    # ── Job watchdog — must come before friday_task ──
    (JOB_WATCHDOG_PHRASES, "job_watchdog"),

    # Friday task dispatch — must be FIRST
    (["friday,", "friday please", "hey friday",
    "ask friday to", "ask friday,", "tell friday to",
    "tell friday,", "friday -", "friday:",
    "can you ask friday", "could you ask friday",
    "can you tell friday", "please ask friday",
    "friday can you", "friday could you",
    ], "friday_task"),

    # Schedule save — the patterns your log missed
    (["remind me", "reminder for", "remind me to", "remind me that",
      "set a reminder", "add to my schedule", "schedule a",
      "i need to remember to", "don't forget i have", "note that i have",
      "can you note", "save that i need"], "schedule_save"),

    # Schedule Recall
    (["what is my schedule", "what's my schedule",
    "show my schedule", "what do i have scheduled",
    "what are my plans for",
    "do i have any", "any meetings", "any appointments",  # ← add these
    "can you remember something about my schedule",       # ← add this
    "what's on my schedule"], "schedule_recall"),

    (["what do i have", "what have i got",
    "what did i schedule", "when is my",
    "do i have anything"], "schedule_recall"),

    (["i have a meeting", "i have an appointment",
    "don't let me forget", "dont let me forget",
    "note that i have"], "schedule_save"),

    # Recall Memory
    (["recall our", "recall my", "recall the",
    "remember our previous", "remember our conversation",
    "what did we talk", "what did we discuss",
    "previous conversation", "past conversation",
    "do you remember our", "can you remember our",
    "what have we talked"], "recall_memory"),

    # Save memory
    (["take a note", "save this", "store this", "keep this in mind",
      "remember this", "save our conversation", "remember our conversation",
      "note this down"], "save_memory"),

    # Delete
    (["delete from memory", "remove from memory", "erase from memory",
    "wipe my memory", "forget about", "delete our", "erase our",
    "erase everything about", "erase all about",  # ← add
    "delete everything about", "remove everything about",  # ← add
    ], "delete_memory"),

    # Delte Schedule
    (["delete my schedule", "delete my reminder", "remove my schedule",
    "remove my reminder", "cancel my schedule", "cancel my reminder",
    "delete the schedule", "remove the schedule"], "schedule_delete"),


    # Web search — explicit command
    (["search for", "search the", "google", "look up",
      "can you search"], "web_search"),

    # Device
    (["take off", "takeoff", "land the drone", "hover",
      "open camera", "start stream"], "device_control"),

    # Project
    (
        [
            "open terahertz", "open thz", "open the terahertz",
            "open my terahertz", "terahertz project", "thz project",
        ],
        "open_project",   # sentinel intent — never actually routed, just for safety
    ),
]


# Simple keyword evidence (not exact phrases)
EVIDENCE = {
    "cpu": ["cpu", "processor", "usage", "load", "cores", "ram", "memory usage"],
    "location": ["where am i", "my location", "where are we", "near me", "location"],
    "recall_memory": ["recall", "remember", "what did we", "what did i", "earlier", "previous", "last time", "saved", "notes", "did you save", "read me", "read it back", "read that back", "show me the summary", "what did you write", "what did you save", "what's in the summary"],
    "delete_memory": ["delete", "remove", "forget", "erase", "wipe", "clear"],
    "save_memory": ["save", "remember this", "store", "note", "take a note", "keep this in mind", "bookmark"],
    "schedule_save": ["remind", "reminder", "schedule", "appointment", "meeting", "calendar"],
    "schedule_recall": ["what's my schedule", "what are my reminders", "upcoming", "next meeting", "calendar"],
}
EVIDENCE.update({
    "weather": ["weather", "forecast", "rain", "temperature"],
    "time": ["time", "clock"],
    "date": ["date", "day", "today", "tomorrow"],
    "battery": ["battery", "charging", "power level"],
    "device_control": [
        "tello", "connect", "disconnect",        # keep tello as it implies action
        "start", "stop", "run", "launch",
        "execute", "stream", "video",
        "take off", "takeoff", "land", "hover",
        "follow me", "track", "scan",
    ],
    "camera_request": [
        "look at me", "can you see me", "see me", "open camera", "turn on camera",
        "start camera", "camera feed", "stream", "video feed"
    ],
    "flight_request": [
        "takeoff", "take off", "land", "hover", "fly", "go up", "go down"
    ],
    "tello_device": ["tello", "drone", "djitellopy"],
    "rover_device": ["rover", "lidar", "robot", "gazebo"]
})
DELETE_STRONG = [
    "delete from memory",
    "remove from memory",
    "erase from memory",
    "forget this from memory",
    "wipe my memory",
]

# Add to STRONG_RULES at module level, before the curiosity check
SYSTEM_QUERY_EXACT = [
    "what time is it", "what's the time", "current time",
    "what is today's date", "what's today's date", "today's date", "what date is it",
    "what is the weather", "what's the weather", "weather today",
    "battery status", "battery level", "check battery",
    "where am i", "my location", "what is my location",
    "cpu usage", "processor usage",
]

# At module level, near your other constants
CURIOSITY_CHAT_RE = re.compile(
    r"^(tell me about|explain|how does|how do|what is|what are|"
    r"i want to learn|i am curious|i'm curious|curious about|"
    r"i think|in my opinion|i believe|i feel like)\b",
    re.IGNORECASE
)

PROJECT_DISCUSSION_RE = re.compile(
    r"^my (project|work|research|thesis|assignment)\b",
    re.IGNORECASE
)

POSSESSION_RE = re.compile(
    r"^i (am using|use|have|have some|have a|have been using|'m using)\b",
    re.IGNORECASE
)

SENSOR_STATEMENT_RE = re.compile(
    r"^(sensor|lidar|camera|imu|gps|actuator|motor)\b.{0,60}"
    r"\b(i have|i am using|i use|we have|we use|there is|there are)\b",
    re.IGNORECASE
)

WORLD_EVENT_RE = re.compile(
    r"\b(political|politics|situation in|what happened in|"
    r"election|government|president|war|conflict|crisis|"
    r"news in|news about|economy|minister|parliament)\b",
    re.IGNORECASE
)
NEWS_SEARCH_RE = re.compile(
    r"\b(latest|breaking|recent|today'?s?)\s+(news|headlines|updates)\b",
    re.IGNORECASE
)

SYSTEM_QUERY_SIGNALS = {
    "time", "date", "today's date", "battery",
    "weather", "location", "where am i", "cpu"
}

DEVICE_STRONG = [
    "tello", "drone", "take off", "takeoff", "land", "hover",
    "open camera", "turn on camera", "camera feed", "start stream",
    "follow me", "track me", "scan the room", "hit rover", "activate rover",
    "move the rover forward", "move the rover backward",
    "turn the rover left", "turn the rover right",
    "rover go forward", "rover go back",
    "rover stop", "move the rover", "turn the rover"
]


SAVE_START_PATTERNS = [
    "i want to talk something important",
    "i want to tell you something important",
    "i have something important",
    "something i want you to remember",
    "take a note",
    "can you take a note",
    "please take a note",
    "note this",
    "keep this in mind",
    "remember this",
    "save this",
    "store this",
    "bookmark this",
]

# Search Task
INTERNET_TRIGGERS = (
    "search", "google", "look up", "find", "latest", "news", "today", "yesterday",
    "price", "release date", "update", "documentation", "docs", "github", "paper"
)

WEB_COMMAND_RE = re.compile(r"^\s*(buddy\s*)?(search|google|look up|find)\b", re.I)
WEB_NEEDS_RE = re.compile(r"\b(latest|current|today|this week|yesterday|breaking)\b", re.I)

## WEB

# helper
def _mentions_strong_web_ref(t: str) -> bool:
    return bool(re.search(
        r"\b(this one|that one|the article|the result|that result|this result|the link|the first one|the second one)\b",
        t
    ))

def _mentions_weak_pronoun(t: str) -> bool:
    return bool(re.search(r"\b(it|this|that)\b", t))

def looks_like_reflective_chat(text: str) -> bool:
    t = (text or "").strip().lower()

    patterns = [
        r"\bwhat do you think\b",
        r"\bwhat's your take\b",
        r"\bhow do you see it\b",
        r"\bit seems\b",
        r"\bthis seems\b",
        r"\bthat seems\b",
        r"\bi think\b",
        r"\bin my opinion\b",
        r"\bseems to be\b",
        r"\bsounds like\b",
    ]

    return any(re.search(p, t) for p in patterns)

# Scoring Method
def score_web_followup(text: str, mm) -> float:
    score = 0.0
    last_topic = (mm.get_last_topic() or "").strip().lower()
    norm = normalize_web_followup(text)

    action = parse_web_followup(norm)
    idx = parse_index_utterance(norm)

    if last_topic.startswith("web:"):
        score += 0.25

    if action:
        score += 0.50

    if idx is not None:
        score += 0.20

    if re.search(r"\b(result|article|source|link|open|summary|summarize|compare|read|show)\b", norm):
        score += 0.15

    if _mentions_strong_web_ref(norm):
        score += 0.10

    # weak pronouns should NOT push it over the threshold
    if _mentions_weak_pronoun(norm):
        score += 0.02

    if looks_like_reflective_chat(norm):
        score -= 0.25

    return min(score, 1.0)

######


def should_use_internet(text: str) -> bool:
    t = (text or "").strip()
    if WEB_COMMAND_RE.search(t):
        return True
    # only use this if question-like
    if "?" in t and WEB_NEEDS_RE.search(t):
        return True
    return False

NUM_WORDS = {
  "one":1,"first":1,"1":1,
  "two":2,"second":2,"2":2,
  "three":3,"third":3,"3":3,
  "four":4,"fourth":4,"4":4,
  "five":5,"fifth":5,"5":5,
}


_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}

def word_to_num(t: str) -> Optional[int]:
    t = (t or "").strip().lower()
    if t.isdigit():
        return int(t)
    return _WORD_NUM.get(t)


def parse_index_utterance(text: str) -> Optional[int]:
    t = normalize_web_followup(text)

    if re.search(r"\b(open|show|read|go to|visit|summarize|summary|tell|explain|compare|vs|versus)\b", t):
        return None

    if re.fullmatch(r"#?\d+", t):
        n = int(re.sub(r"\D", "", t))
        return n if n > 0 else None

    m = re.fullmatch(r"(?:number|result)\s+(\d+)", t)
    if m:
        n = int(m.group(1))
        return n if n > 0 else None

    if re.fullmatch(r"\d+", t):
        return int(t)

    return None


def extract_topic_thread(wctx: dict | None) -> str:
    if not wctx:
        return ""
    # Prefer root thread if available (persists across chained searches)
    root = (wctx.get("root_thread") or "").strip()
    if root:
        return " ".join(root.split()[:4])
    query = (wctx.get("query") or "").strip()
    core = re.sub(r"^(search|find|look up|google|can you search|what is|tell me about)\s+", "", query, flags=re.I).strip()
    return " ".join(core.split()[:4])

def extract_search_query(text: str) -> str:
    t = (text or "").strip()

    # command style
    t2 = re.sub(r"^(buddy\s*)?(search|google|look up|find)\s+(for\s+)?", "", t, flags=re.I).strip()
    if t2 != t:
        return t2

    # question style
    m = re.search(r"\b(latest|current|today|breaking)\b(.*)$", t, re.I)
    if m:
        rest = m.group(2).strip(" :,-")
        return (rest or t).strip()

    return t


def looks_like_web_followup(text: str) -> bool:
    t = normalize_web_followup(text)
    if not t:
        return False

    if re.search(r"\b(open|summarize|read|show|compare)\b", t):
        return True

    if re.search(r"\b(result\s+)?\d+\b", t):
        return True

    if re.search(r"\b(it|this|that|the article|the result|that one|this one)\b", t):
        return True

    return False

# intents that should NOT be hard-vetoed because they're low risk
LOW_RISK = {"chat", "recall_memory", "schedule_recall"}

def _has_any(text: str, phrases: list[str]) -> bool:
    return any(p in text for p in phrases)

def normalize_intent(user_input: str, raw_intent: str) -> str:
    text = (user_input or "").lower().strip()
    intent = (raw_intent or "").lower().strip()

    if intent not in ALLOWED_INTENTS:
        return "chat"

    # Low-risk intents: be permissive (don't kill them with strict guards)
    if intent in LOW_RISK:
        return intent

    # Tool-ish intents: require some evidence, but keep it loose
    if intent in ("cpu", "location"):
        if _has_any(text, EVIDENCE[intent]):
            return intent
        return "chat"

    # State-changing: require delete/save evidence
    if intent == "delete_memory":
        if _has_any(text, EVIDENCE["delete_memory"]):
            return intent
        return "chat"

    if intent == "save_memory":
        # You already have SAVE_START_PATTERNS rule override upstream,
        # so here just prevent wild misfires.
        if _has_any(text, EVIDENCE["save_memory"]):
            return intent
        # still allow if user is clearly asking to remember/store something without exact phrase
        if "remember" in text or "save" in text or "note" in text:
            return intent
        return "chat"

    if intent == "schedule_save":
        # allow if there's schedule/reminder keywords OR a time signal
        time_signal = re.search(
            r"\b(\d{1,2}:\d{2}\b|\d{1,2}\s?(am|pm)\b|at\s?\d{1,2}(:\d{2})?\b|"
            r"today\b|tomorrow\b|next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
            r"\d{4}-\d{2}-\d{2}\b)\b",
            text
        )

        if _has_any(text, EVIDENCE["schedule_save"]) or time_signal:
            return intent
        return "chat"

    if intent == "schedule_recall":
        return intent


    return intent

# define threshold
STRONG_RULE_CONF = 0.99
WEAK_RULE_CONF   = 0.80
LLM_DEFAULT_CONF = 0.65

# normalizing llm output for intent detection
def safe_parse_intent(raw: str):
    if not raw:
        return None
    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def normalize_llm_output(data, user_text):
    if not isinstance(data, dict):
        return {"intent": "unknown", "confidence": 0.0, "source": "llm_invalid_json"}


    raw_intent = data.get("intent", "chat")
    raw_conf   = data.get("confidence", LLM_DEFAULT_CONF)

    try:
        conf = float(raw_conf)
    except Exception:
        conf = LLM_DEFAULT_CONF

    conf = max(0.0, min(1.0, conf))

    intent = normalize_intent(user_text, raw_intent)
    return {"intent": intent, "confidence": conf, "source": "llm"}

# temprary help for the limitation of llama 3.2
ENGINEERING_TERMS = [
    "pid", "mpc", "lqr", "px4", "controller", "baseline",
    "control system", "sensor", "actuator", "feedback",
    "lidar", "imu", "ekf", "trajectory", "waypoint",
    "laser", "diode", "optical", "signal", "transmission",
    # REMOVED: "rover", "uav", "drone", "autopilot", "quadrotor"
    "autopilot", "quadrotor"   # keep these since they're unlikely to be commands
]
def post_validate_intent(result, user_input, user_text: str):
    lower = user_input.lower()

    intent = result.get("intent", "chat")
    conf   = float(result.get("confidence", 0.0))
    text   = (user_text or "").lower()

    # ── NEW: Delete signal beats recall when both are present ──────────────
    # "can you delete/remove/forget our previous conversation about X"
    # The LLM or weak rules may have returned recall_memory because it saw
    # "previous conversation", but delete is the user's real intent.
    _DELETE_SIGNALS = ["delete", "remove", "forget", "erase", "wipe", "clear"]
    _RECALL_SIGNALS = ["previous conversation", "past conversation", "previous chat",
                       "what did we", "recall", "remember when"]
    _MEMORY_OBJECTS = [
        "conversation", "discussion", "memory", "topic",
        "talk", "notes", "chat", "history",
        "rover", "drone", "project", "tello",  # ← add here too
    ]


    has_delete_signal = any(w in text for w in _DELETE_SIGNALS)
    has_recall_signal = any(p in text for p in _RECALL_SIGNALS)
    has_memory_object = any(w in text for w in _MEMORY_OBJECTS)


    # for handling saving
    SAVE_THIS_SIGNALS = [
        "this conversation", "this discussion", "this chat",
        "our conversation", "what we discussed", "what we talked about",
        "this topic", "about this", "remember this",
    ]
    RECALL_PAST_SIGNALS = [
        "previous conversation", "past conversation", "earlier conversation",
        "last time", "what did we", "we talked about", "we discussed",
        "from before", "old conversation",
    ]

    is_save_this = any(p in text for p in SAVE_THIS_SIGNALS)
    is_recall_past = any(p in text for p in RECALL_PAST_SIGNALS)

    if intent == "recall_memory" and has_delete_signal and has_memory_object:
        # User said something like "delete our previous conversation about X"
        # Recall cues tripped recall, but delete is clearly the intent.
        return {
            "intent": "delete_memory",
            "confidence": min(conf + 0.15, STRONG_RULE_CONF),
            "source": "delete_beats_recall_override",
        }

    # If any engineering term is present, these intents are almost certainly wrong
    if any(term in lower for term in ENGINEERING_TERMS):
        ENGINEERING_UNSAFE_INTENTS = {
            "weather",
            "schedule_recall",
            "schedule_save",
            "delete_memory",
            "device_control",
            "location",
            "battery",
        }
        if intent in ENGINEERING_UNSAFE_INTENTS:
            print(f"[DBG] post_validate: overriding {intent} → chat (engineering context)")
            # ← FIXED: returns dict not tuple, save_memory and recall_memory still allowed
            return {
                "intent": "chat",
                "confidence": min(conf, 0.75),
                "source": "engineering_domain_override",
            }

    SESSION_RESUME_PATTERNS = [
        r"\bwe('?re| are) (previously|just) (talking|discussing|on)\b",
        r"\bnow we('?re| are) on the same page\b",
        r"\bwe can continue\b",
        r"\bback to (what|where) we were\b",
        r"\blet('?s| us) continue\b",
        r"\bas i (was|were) saying\b",
        r"\banyway[,\s]+(back to|continuing)\b",
    ]

    if any(re.search(p, text) for p in SESSION_RESUME_PATTERNS):
        if intent == "recall_memory":
            print(f"[DBG] post_validate: session resume detected → chat")
            return {
                "intent": "chat",
                "confidence": min(conf, 0.75),
                "source": "session_resume_override",
            }

    # --- helpers ---
    def has_any(key):
        return _has_any(text, EVIDENCE.get(key, []))

    time_signal = re.search(
        r"\b("
        r"\d{1,2}:\d{2}"
        r"|\d{1,2}\s?(am|pm)"
        r"|at\s?\d{1,2}(:\d{2})?"
        r"|\d{4}-\d{2}-\d{2}"
        r"|today|tomorrow"
        r"|next\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
        r"|in\s+\d+\s+(minutes|minute|hours|hour|days|day)"
        r")\b",
        text
    )

    reminder_language        = has_any("schedule_save")
    save_language            = has_any("save_memory")
    delete_language          = has_any("delete_memory")
    recall_language          = has_any("recall_memory")
    schedule_recall_language = has_any("schedule_recall")
    device_language          = has_any("device_control")
    camera_language          = has_any("camera_request")
    flight_language          = has_any("flight_request")

    # 0) Handle Unknown
    if intent == "unknown":
        if recall_language:
            return {"intent": "recall_memory",  "confidence": 0.6, "source": "unknown_fallback_recall"}
        if schedule_recall_language:
            return {"intent": "schedule_recall", "confidence": 0.6, "source": "unknown_fallback_schedule_recall"}
        if save_language:
            return {"intent": "save_memory",     "confidence": 0.6, "source": "unknown_fallback_save"}
        if delete_language:
            return {"intent": "delete_memory",   "confidence": 0.6, "source": "unknown_fallback_delete"}
        if reminder_language or time_signal:
            return {"intent": "schedule_save",   "confidence": 0.6, "source": "unknown_fallback_schedule_save"}
        if device_language or camera_language or flight_language:
            return {"intent": "device_control",  "confidence": 0.6, "source": "unknown_fallback_device"}
        return {"intent": "chat", "confidence": 0.4, "source": "unknown_fallback_chat"}

    # 1) Hard fallback on very low confidence
    if conf < 0.55:
        return {"intent": "chat", "confidence": conf, "source": "low_conf"}

    # 2) Evidence gating for state-changing intents
    if intent == "delete_memory" and not delete_language:
        return {"intent": "chat", "confidence": min(conf, 0.4), "source": "delete_no_evidence"}

    if intent == "save_memory" and is_save_this and not is_recall_past:
        if "don't forget" in text or "dont forget" in text:
            if time_signal or reminder_language:
                return {"intent": "schedule_save", "confidence": min(conf, 0.8), "source": "reroute_dont_forget"}
        return {"intent": "chat", "confidence": min(conf, 0.45), "source": "save_no_evidence"}

    # 3) Schedule save gating + conflict resolution
    if intent == "schedule_save":
        if not (time_signal or reminder_language):
            return {"intent": "chat", "confidence": min(conf, 0.4), "source": "schedule_no_time"}

    if intent == "save_memory" and (time_signal or reminder_language):
        if ("save this" not in text and "store this" not in text and "take a note" not in text):
            return {"intent": "schedule_save", "confidence": min(conf, 0.8), "source": "reroute_to_schedule"}

    # 4) Schedule recall vs memory recall conflict
    elif intent == "recall_memory" and is_save_this and not is_recall_past:
        # LLM misrouted — this is actually a save request
        return {
            "intent": "save_memory",
            "confidence": min(conf, 0.85),
            "source": "save_this_override",
        }

    # 5) System query conflict (time/date) vs schedule
    if intent in ("time", "date") and ("meeting" in text or "appointment" in text or "reminder" in text):
        if "what time" in text and (time_signal or "tomorrow" in text or "next" in text):
            return {"intent": "schedule_recall", "confidence": min(conf, 0.8), "source": "reroute_time_to_schedule"}

    # 6) Device Control
    ACTION_EVIDENCE = [
        "connect", "disconnect", "start", "stop", "run", "launch", "execute",
        "stream", "take off", "takeoff", "land", "hover", "follow me",
        "track", "scan", "move forward", "turn left", "turn right",
        "go forward", "go back", "move the",
    ]
    DEVICE_NAMES = ["rover", "drone", "robot", "tello"]

    has_action_evidence = any(w in text for w in ACTION_EVIDENCE)
    has_device_name = any(w in text for w in DEVICE_NAMES)

    if intent == "device_control":
        if not has_device_name:
            return {"intent": "chat", "confidence": min(conf, 0.4),
                    "source": "device_no_device_name"}
        if not has_action_evidence and not flight_language and not camera_language:
            return {"intent": "chat", "confidence": min(conf, 0.5),
                    "source": "device_no_action_evidence"}
    # 7) Camera and Flight in Chat
    if intent == "chat" and (camera_language or flight_language) and conf >= 0.55:
        return {"intent": "device_control", "confidence": min(conf, 0.8), "source": "reroute_chat_to_device"}

    # 8) Downgrade chat confidence slightly
    if intent == "chat":
        conf = min(conf, 0.75)

    return {"intent": intent, "confidence": conf, "source": result.get("source", "llm")}


# Confindence Gate (New Approach, for make the handle intent is more strict and more robust)
ACTION_INTENTS = {
    "save_memory",
    "delete_memory",
    "schedule_save",
    "device_control",
}

QUESTION_INTENTS = {
    "recall_memory",
    "schedule_recall",
    "chat",
}

CONF_HIGH  = 0.85   # route directly, trust the intent
CONF_LOW   = 0.65   # below this, RAG takes over completely

def route_with_confidence_gate(
    intent_result: dict,
    user_input: str,
    short_term_context: str = "",
) -> dict:
    """
    Takes the raw intent result from detect_intent() and returns
    a routing decision dict:

    {
        "action":  "direct" | "rag_first" | "rag_only",
        "intent":  <original intent string>,
        "confidence": <float>,
        "context": <short_term_context string for RAG>,
        "reason":  <debug string>,
    }

    - "direct"   → high confidence or action intent, skip RAG
    - "rag_first" → medium confidence, try RAG, fall back to handler
    - "rag_only"  → low confidence, ignore intent, use RAG/chat
    """

    intent = intent_result.get("intent", "chat")
    conf   = float(intent_result.get("confidence", 0.0))
    source = intent_result.get("source", "unknown")

    # --- Rule 1: Action intents always go direct ---
    # Reason: mutating state (save/delete/device) should never
    # be guessed by RAG. If confidence is low on these, we still
    # route direct but let the handler ask for clarification.
    if intent in ACTION_INTENTS:
        return {
            "action":     "direct",
            "intent":     intent,
            "confidence": conf,
            "context":    "",
            "reason":     f"action_intent_always_direct (src={source})",
        }

    # --- Rule 2: High confidence → direct route ---
    if conf >= CONF_HIGH:
        return {
            "action":     "direct",
            "intent":     intent,
            "confidence": conf,
            "context":    "",
            "reason":     f"high_conf={conf:.2f} (src={source})",
        }

    # --- Rule 3: Medium confidence → RAG first, handler fallback ---
    if conf >= CONF_LOW:
        return {
            "action":     "rag_first",
            "intent":     intent,
            "confidence": conf,
            "context":    short_term_context,
            "reason":     f"medium_conf={conf:.2f} (src={source})",
        }

    # --- Rule 4: Low confidence → RAG only ---
    return {
        "action":     "rag_only",
        "intent":     "chat",
        "confidence": conf,
        "context":    short_term_context,
        "reason":     f"low_conf={conf:.2f} (src={source})",
    }

CANCEL_PHRASES = {
    "nevermind", "never mind", "forget it", "cancel that",
    "actually no", "no wait", "ignore that", "scratch that",
    "nvm", "nm",
}

def _matches_device_strong(text: str) -> bool:
    for phrase in DEVICE_STRONG:
        # use word boundary matching instead of bare substring
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def detect_intent(user_input: str):
    text = (user_input or "").lower().replace("'", "'").strip()

    _CANCEL_EXACT = {
        "nevermind", "never mind", "forget it", "cancel that",
        "actually no", "no wait", "ignore that", "scratch that",
        "nvm", "nm", "forget it never mind", "never mind forget it",
        "nevermind forget it", "forget what i said",
        "actually forget that", "no nevermind", "no never mind",
    }
    _CANCEL_STARTS = [
        "never mind", "nevermind", "forget it",
        "actually forget what", "no wait",
    ]
    _MEMORY_ANCHOR = [
        "memory", "conversation", "topic", "history",
        "notes", "chat", "discussion",
    ]

    is_cancel = (
        text in _CANCEL_EXACT
        or any(text.startswith(p) for p in _CANCEL_STARTS)
    )
    has_memory_anchor = any(w in text for w in _MEMORY_ANCHOR)

    # Only treat as cancel if there's no explicit memory target
    # e.g. "forget it" → cancel, but "forget my conversation about X" → delete
    if is_cancel and not has_memory_anchor:
        return {"intent": "chat", "confidence": 0.99, "source": "rule_cancel"}

    # 1) Strong rule-based overrides — ORDER MATTERS
    # ── DELETE must be checked BEFORE recall, because
    #    "delete our previous conversation" contains recall cues
    #    but the user's clear intent is deletion.

    has_delete_action   = any(p in text for p in DELETE_ACTION_PATTERNS)
    NEGATED_FORGET = [
        "don't forget", "dont forget", "do not forget",
        "never forget", "please don't forget", "please dont forget",
    ]
    if any(p in text for p in NEGATED_FORGET):
        has_delete_action = False
    has_memory_target   = any(w in text for w in MEMORY_TARGET_WORDS)
    has_delete_strong   = any(p in text for p in DELETE_STRONG)

    if has_delete_strong or (has_delete_action and has_memory_target):
        return {"intent": "delete_memory", "confidence": STRONG_RULE_CONF, "source": "rule_strong"}

    if _matches_device_strong(text):
        return {"intent": "device_control", "confidence": STRONG_RULE_CONF,
                "source": "rule_strong"}

    if any(p in text for p in SAVE_START_PATTERNS):
        return {"intent": "save_memory", "confidence": STRONG_RULE_CONF, "source": "rule_strong"}

    if any(p in text for p in SAVE_CONVERSATION_PATTERNS):
        return {"intent": "save_memory", "confidence": STRONG_RULE_CONF, "source": "rule_strong"}

    # ── Recall strong rules — only fire when there is NO delete signal
    has_weak_delete = any(p in text for p in ["delete", "remove", "forget", "erase", "wipe", "clear"])
    has_weak_save = any(p in text for p in ["save", "store", "keep", "remember this"])
    if not has_weak_delete and not has_weak_save:
        if any(p in text for p in [
            "what did we talk about",
            "previous conversation",
            "previous chat",
            "past conversation",
            "recall our talk",
            "can you recall",
            "remember when",
        ]):
            return {"intent": "recall_memory", "confidence": STRONG_RULE_CONF, "source": "rule_strong"}

    # 2) Weak rules
    rule_intent = rule_based_intent_detection(text)
    if rule_intent:
        normalized = normalize_intent(text, rule_intent)
        return {"intent": normalized, "confidence": WEAK_RULE_CONF, "source": "rule_weak"}

    # 3) LLM-based detection
    prompt = llmBasedIntentDetection(text)
    response = ollama.generate(model="llama3.2:latest", prompt=prompt)
    raw = (response.get("response") or "").strip()
    parsed = safe_parse_intent(raw)
    normalized = normalize_llm_output(parsed, text)
    final = post_validate_intent(normalized, user_input, text)

    print(f"[DEBUG] LLM raw: {raw[:200]}")
    print(f"[DEBUG] Routed: {final}")

    return final


def normalize_schedule_dict(d: dict) -> dict:
    if not isinstance(d, dict):
        return {}

    BAD = {"none", "null", "unknown", "n/a", ""}
    out = dict(d)

    # normalize strings -> None
    for k, v in list(out.items()):
        if isinstance(v, str):
            vv = v.strip()
            out[k] = None if vv.lower() in BAD else vv

    # normalize timestamp first (so it can overwrite date/time)
    ts = out.get("timestamp")
    if isinstance(ts, str):
        ts = ts.strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2})[ T](\d{1,2}:\d{2})", ts)
        if m:
            date = m.group(1)
            time = m.group(2)
            if len(time.split(":")[0]) == 1:
                time = "0" + time  # "8:00" -> "08:00"

            out["date"] = date
            out["time"] = time
            out["timestamp"] = f"{date} {time}"

    # build timestamp if date+time exist but timestamp missing
    if out.get("date") and out.get("time") and not out.get("timestamp"):
        time = out["time"]
        if isinstance(time, str) and len(time.split(":")[0]) == 1:
            time = "0" + time
            out["time"] = time
        out["timestamp"] = f"{out['date']} {out['time']}"

    return out


## Normalize Web Followup

# helper
_WORD_TO_DIGIT = {
    "one": "1", "first": "1",
    "two": "2", "second": "2",
    "three": "3", "third": "3",
    "four": "4", "fourth": "4",
    "five": "5", "fifth": "5",
    "six": "6", "sixth": "6",
    "seven": "7", "seventh": "7",
    "eight": "8", "eighth": "8",
    "nine": "9", "ninth": "9",
    "ten": "10", "tenth": "10",
}

def normalize_number_words(text: str) -> str:
    t = (text or "").strip().lower()
    for word, digit in _WORD_TO_DIGIT.items():
        t = re.sub(rf"\b{re.escape(word)}\b", digit, t)
    return t

# function normalize web followup
def normalize_web_followup(text: str) -> str:
    t = (text or "").strip().lower()

    t = re.sub(
        r"\b(open|summarize|summary|show|read|compare)\s+(one|first)\b",
        lambda m: m.group(1) + " 1", t
    )
    t = re.sub(
        r"\b(open|summarize|summary|show|read|compare)\s+(two|second)\b",
        lambda m: m.group(1) + " 2", t
    )
    t = re.sub(
        r"\b(open|summarize|summary|show|read|compare)\s+(three|third)\b",
        lambda m: m.group(1) + " 3", t
    )

    t = re.sub(
        r"\b(?:make|give|do)\s+(?:a\s+)?summary\s+(?:for\s+)?(?:number\s+|result\s+)?(\d+)\b",
        r"summarize \1",
        t
    )
    t = re.sub(
        r"\bsummary\s+(?:for\s+|of\s+)?(?:number\s+|result\s+)?(\d+)\b",
        r"summarize \1",
        t
    )
    t = re.sub(
        r"\bsummarize\s+(?:number\s+|result\s+)?(\d+)\b",
        r"summarize \1",
        t
    )
    t = re.sub(
        r"\b(?:open|show|read|visit|go to)\s+(?:number\s+|result\s+)?(\d+)\b",
        r"open \1",
        t
    )
    t = re.sub(
        r"\bcompare\s+(?:number\s+|result\s+)?(\d+)\s*(?:and|vs|versus)\s*(?:number\s+|result\s+)?(\d+)\b",
        r"compare \1 and \2",
        t
    )
    # ── NEW: handle "open the second one", "open the first one" ──
    t = re.sub(
        r"\b(open|show|read|visit|go to)\s+the\s+(first|second|third|fourth|fifth)\s+one\b",
        lambda m: m.group(1) + " " + str({"first":"1","second":"2","third":"3",
                                          "fourth":"4","fifth":"5"}[m.group(2)]),
        t
    )

    # ── NEW: handle "make a summary for result one/two/three" ──
    t = re.sub(
        r"\b(?:make|give|do)\s+(?:a\s+)?summary\s+(?:for\s+)?(?:result\s+)?"
        r"(one|two|three|four|five|first|second|third|fourth|fifth)\b",
        lambda m: "summarize " + str({
            "one":"1","first":"1","two":"2","second":"2",
            "three":"3","third":"3","four":"4","fourth":"4",
            "five":"5","fifth":"5"
        }[m.group(1)]),
        t
    )

    # ── NEW: handle "make a summary for result 1" (digit version) ──
    t = re.sub(
        r"\b(?:make|give|do)\s+(?:a\s+)?summary\s+(?:for\s+)?(?:result\s+)?(\d+)\b",
        r"summarize \1",
        t
    )

    return t

##############################

# Intent Handler
class IntentHandler:
    def __init__(self):
        pass

    def handle_intent(self, intent):
        system_outputs = {
            "time": get_time_str(),
            "date": get_date(),
            "battery": get_battery_status(),
            "cpu": get_cpu_usage(),
            "weather": get_weather_from_windows(),
            "location": get_location_from_windows(),
        }

        return system_outputs.get(intent, None)

def get_eligible_intents(mm) -> list[str]:
    last_topic = (mm.get_last_topic() or "").lower()
    base = [
        "chat", "system_query", "save_memory", "recall_memory",
        "delete_memory", "schedule_save", "schedule_recall",
        "device_control", "web_search"
    ]
    # web_followup only when there's an active web session
    if last_topic.startswith("web:"):
        base.append("web_followup")
    return base


def _tool_router_context(mm, belief_state=None) -> str:
    last_topic = (mm.get_last_topic() or "").strip()
    pending = (mm.get_pending_topic() or "").strip()
    try:
        active_web = (mm.get_active_web_topic() or "").strip()
    except Exception:
        active_web = ""

    recent_turns = list(sensory_memory)[-3:] if sensory_memory else []
    context = []
    if last_topic:
        context.append(f"last_topic={last_topic}")
    if active_web:
        context.append(f"active_web_topic={active_web}")
    if pending:
        context.append(f"pending_state={pending}")
    if recent_turns:
        context.append("recent_turns:\n" + "\n".join(recent_turns))
    if belief_state:
        try:
            context.append("belief_state:\n" + belief_state.to_prompt_block())
        except Exception:
            pass
    return "\n".join(context) if context else "No prior context."

def _native_tool_classify_intent(user_input: str, mm, belief_state=None) -> dict | None:
    """
    Phase 2 routing layer: use Ollama native tool-calling to choose a capability.
    Existing handler methods still execute the selected intent.
    """
    system = (
        "You are Buddy's routing layer. Choose exactly one tool when the user asks Buddy "
        "to perform a capability. If the user is only chatting, asking for an explanation, "
        "or no tool fits, answer in plain text without a tool call.\n\n"
        "Use web_followup for requests about opening, showing, reading, summarizing, or "
        "comparing a previous web result/link/source/article, including terse utterances "
        "like 'open 2', 'summarize the first result', or 'compare result 1 and result 2'. "
        "Use search_web only for a new web search. Use system_query only for local "
        "time/date/battery/weather/location/CPU. Device discussion or learning questions "
        "are plain chat; direct rover/drone actions are tools."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{_tool_router_context(mm, belief_state)}"},
        {"role": "user", "content": user_input},
    ]

    try:
        response = ollama.chat(
            model="llama3.2:latest",
            messages=messages,
            tools=BUDDY_TOOL_SCHEMAS,
            options={"temperature": 0},
        )
    except Exception as e:
        print(f"[ToolRouter] error: {e}")
        return None

    message = getattr(response, "message", None)
    tool_calls = getattr(message, "tool_calls", None) if message is not None else None
    if not tool_calls:
        print("[ToolRouter] no tool call -> chat")
        return {"intent": "chat", "confidence": 0.72, "source": "ollama_tool_no_call"}

    call = tool_calls[0]
    name = _tool_call_name(call)
    args = _tool_call_args(call)
    if name not in BUDDY_TOOL_TO_INTENT:
        return None

    if not _passes_grounding(name, user_input, mm):   # ← same function, reused
        print(f"[ToolRouter] rejected ungrounded {name} for: {user_input[:50]!r}")
        return None  # falls through naturally to this function's own chat fallback

    intent = BUDDY_TOOL_TO_INTENT[name]
    print(f"[ToolRouter] tool={name} intent={intent} args={args}")
    return {
        "intent": intent,
        "confidence": 0.86,
        "source": f"ollama_tool:{name}",
        "tool_name": name,
        "tool_args": args,
    }

def classify_intent(user_input: str, mm, conversation_manager=None) -> dict:
    text = (user_input or "").strip().lower()
    text = text.replace("—", " ").replace("–", " ")

    # ── Context priming (currently missing) ──
    last_topic = (mm.get_last_topic() or "").lower()
    pending = (mm.get_pending_topic() or "").lower()

    # If we're already in a rover/device context, short-circuit
    # before rules even run
    if last_topic.startswith("rover:") or "rover" in last_topic:
        IMPLICIT_DEVICE = re.compile(
            r"^(turn it|connect|disconnect|wake|start it|"
            r"stop it|is it|check it|power)\b", re.I
        )
        if IMPLICIT_DEVICE.match(text):
            return {"intent": "device_control", "confidence": 0.95,
                    "source": "context_rover_continuation"}

        SAVE_SIGNAL_RE = re.compile(r"\b(save|store|export)\b", re.I)
        if SAVE_SIGNAL_RE.search(text):
            rover_manager = getattr(getattr(conversation_manager, "device_handler", None), "rover_manager", None)
            if rover_manager is not None:
                target = score_save_target(user_input, mm, rover_manager)
                if target["target"] == "rover_log":
                    return {"intent": "device_control", "confidence": 0.90,
                            "source": "rover_save_target", "save_target": "rover_log"}
                if target["target"] == "ambiguous":
                    return {"intent": "device_control", "confidence": 0.60,
                            "source": "rover_save_ambiguous", "save_target": "ambiguous"}

    # ── CANCEL GUARD — must be FIRST, before any other check ──
    _CANCEL_EXACT = {
        "nevermind", "never mind", "forget it", "cancel that",
        "actually no", "no wait", "ignore that", "scratch that",
        "nvm", "nm", "forget it never mind", "never mind forget it",
        "nevermind forget it", "forget what i said",
        "actually forget that", "no nevermind", "no never mind",
    }
    _CANCEL_STARTS = [
        "never mind", "nevermind", "forget it",
        "actually forget what", "no wait",
    ]
    _MEMORY_ANCHOR = [
        "memory", "conversation", "topic", "history",
        "notes", "chat", "discussion",
    ]

    is_cancel = (
        text in _CANCEL_EXACT
        or any(text.startswith(p) for p in _CANCEL_STARTS)
    )
    has_memory_anchor = any(w in text for w in _MEMORY_ANCHOR)

    if is_cancel and not has_memory_anchor:
        return {"intent": "chat", "confidence": 0.99, "source": "rule_cancel"}


    # ── DELETE guard (must run before recall rules) ──
    has_delete_action = any(p in text for p in DELETE_ACTION_PATTERNS)
    has_memory_target = any(w in text for w in MEMORY_TARGET_WORDS)
    has_delete_strong = any(p in text for p in DELETE_STRONG)

    if has_delete_strong or (has_delete_action and has_memory_target):
        return {"intent": "delete_memory", "confidence": STRONG_RULE_CONF,
                "source": "rule_strong_delete"}

    HYPOTHETICAL_PREFIXES = [
        "for an example", "for example", "let's say", "lets say",
        "imagine if", "what if", "suppose", "hypothetically",
        "if i wanted to", "if i asked you to", "if i ask you to",
    ]
    if any(text.startswith(p) for p in HYPOTHETICAL_PREFIXES):
        return {"intent": "chat", "confidence": 0.90, "source": "rule_hypothetical_prefix"}

    # Add this BEFORE the STRONG_RULES loop, after the cancel guard:
    NEGATION_REMINDER_RE = re.compile(
        r"^(don'?t|do not|stop|no longer|never)\s+(remind|schedule|add|set)",
        re.IGNORECASE
    )
    if NEGATION_REMINDER_RE.match(text):
        return {"intent": "chat", "confidence": 0.90, "source": "rule_negation_reminder"}

    # Step 0: Exact system query matching
    if any(text == q or text.startswith(q) for q in SYSTEM_QUERY_EXACT):
        return {"intent": "system_query", "confidence": 0.99, "source": "rule_system_exact"}

    # ── Step 1: Strong rules (zero LLM cost, highest precision) ──
    has_weak_save = any(p in text for p in [
        "save", "store", "keep", "remember this"
    ])
    has_weak_delete = any(p in text for p in [    # ← move it up here
        "delete", "remove", "forget", "erase", "wipe", "clear"
    ])

    RECALL_OVERRIDE_PATTERNS = [
        r"remind me what (we|i) (discussed|talked|covered)",
        r"remind me (about|of) (what|our|the) (we|i|previous)",
        r"can you remind me what",
        r"remind me of (our|my|the) (previous|past|earlier)",
    ]
    if any(re.search(p, text) for p in RECALL_OVERRIDE_PATTERNS):
        return {"intent": "recall_memory", "confidence": 0.99,
                "source": "rule_recall_override"}

    for patterns, intent in STRONG_RULES:
        if intent == "recall_memory" and has_weak_save:
            continue
        if intent == "recall_memory" and has_weak_delete:
            continue
        if any(p in text for p in patterns):
            return {"intent": intent, "confidence": 0.99, "source": "rule_strong"}


    if any(p in text for p in SAVE_CONVERSATION_PATTERNS):
        return {"intent": "save_memory", "confidence": 0.99, "source": "rule_save_conversation"}

    if not has_weak_delete and not has_weak_save:
        if any(p in text for p in [
            "what did we talk about",
            "previous conversation",
            "previous chat",
            "past conversation",
            "recall our talk",
            "can you recall",
            "remember when",
        ]):
            return {"intent": "recall_memory", "confidence": 0.99, "source": "rule_strong"}

    # ── ADD THIS: Save start patterns ────────────────────────────
    if any(p in text for p in SAVE_START_PATTERNS):
        return {"intent": "save_memory", "confidence": 0.99, "source": "rule_save_start"}


    # ── Step 2: Web followup detection (replaces score_web_followup) ──
    # Only fires if there IS an active web session AND the utterance
    # is clearly a navigation command, not a new search or reminder

    # early guard for web followup
    TECHNICAL_NUMBER_PATTERNS = [
        r"\d+\s*(thz|ghz|hz|mhz)",          # frequencies
        r"\d+\s*(dof|degrees?|°)",           # angles/DOF
        r"\d+\s*(%|percent)",                # percentages
        r"\d+\s*(samples?|classes?|epoch)",  # ML numbers
        r"\b(0\.\d+|1\.\d+|2\.\d+)\b",      # decimal values
    ]

    if any(re.search(p, text, re.I) for p in TECHNICAL_NUMBER_PATTERNS):
        pass  # don't return web_followup

    active_web = mm.get_active_web_topic()

    if active_web and _is_explicit_web_exit(text, mm, temp_topic_memory):
        print(f"[CLASSIFY] explicit web exit (target matches non-web topic)")
        # Clear web session state — adjust to match your MemoryManager API
        mm.active_web_session_id = None
        if (getattr(mm, "last_topic", "") or "").startswith("web:"):
            mm.last_topic = None  # or set to a sensible non-web topic if you have one
        active_web = None  # refresh local so we skip web_followup logic below

    eligible = get_eligible_intents(mm)

    if active_web and "web_followup" in eligible:
        norm = normalize_web_followup(text)
        action = parse_web_followup(norm)
        bare_idx = parse_index_utterance(norm)

        if action is not None or bare_idx is not None:
            # Guard 1: technical domain numbers are NOT result indices
            TECHNICAL_NUMBER_RE = re.compile(
                r"\d+\s*(-\s*)?(dof|degrees?|°|thz|ghz|hz|mhz|ms|rpm|"
                r"percent|%|samples?|classes?|epochs?|layers?|bits?|"
                r"axis|axes|joint|link|bar|psi|nm|kg|cm|mm)\b"
                r"|\b\d+\.\d+\s*(thz|ghz|hz)\b"
                r"|\b(6-dof|3-dof|7-dof)\b",
                re.IGNORECASE
            )
            if TECHNICAL_NUMBER_RE.search(text):
                print(f"[DBG] web_followup suppressed — technical number: {text[:60]}")
                # Fall through to rest of classify_intent

            else:
                # Guard 2: schedule phrases
                schedule_guard = any(p in text for p in [
                    "remind", "schedule", "appointment", "remember to",
                    "birthday", "meeting", "tomorrow", "today at"
                ])
                # Guard 3: comparison/opinion phrases
                comparison_guard = any(p in text for p in [
                    "is it better", "is this better", "which is better",
                    "is it worse", "do you think", "what do you think",
                    "is it good", "is it bad", "how good", "how bad",
                ])

                if not schedule_guard and not comparison_guard:
                    return {
                        "intent": "web_followup",
                        "confidence": 0.95,
                        "source": "rule_web_followup"
                    }

    if WORLD_EVENT_RE.search(text):
        return {"intent": "web_search", "confidence": 0.92, "source": "rule_world_event"}

    if NEWS_SEARCH_RE.search(text):
        return {"intent": "web_search", "confidence": 0.92, "source": "rule_news_search"}

    if POSSESSION_RE.match(text):
        return {"intent": "chat", "confidence": 0.90, "source": "rule_possession_statement"}

    if SENSOR_STATEMENT_RE.match(text):
        return {"intent": "chat", "confidence": 0.90, "source": "rule_sensor_statement"}

    if PROJECT_DISCUSSION_RE.match(text):
        return {"intent": "chat", "confidence": 0.90, "source": "rule_project_discussion"}

    if CURIOSITY_CHAT_RE.match(text):
        is_system = any(s in text for s in SYSTEM_QUERY_SIGNALS)
        if not is_system and not re.search(r"\b(search|look up|google|latest news)\b", text):
            return {"intent": "chat", "confidence": 0.90, "source": "rule_curiosity_chat"}

    HYPOTHETICAL_RE = re.compile(
        r"^(for an example|let'?s say|imagine if|what if|suppose|"
        r"for instance|hypothetically|in theory|if i (want|wanted|could|ask))\b",
        re.IGNORECASE
    )

    # Add this before the zero-shot call in classify_intent
    if HYPOTHETICAL_RE.match(text):
        return {"intent": "chat", "confidence": 0.90,
                "source": "rule_hypothetical"}

    DECLARATIVE_RE = re.compile(
        r"^(i (plan|want|am|have|use|will|need)|"
        r"my (rover|robot)|"
        r"the (rover|robot) (has|is|will|can)|"
        r"(record|log|observe|measure|track)\s+the)",
        re.IGNORECASE
    )

    # Add this BEFORE the DEVICE_STRONG check in classify_intent:
    if DECLARATIVE_RE.match(text):
        return {"intent": "chat", "confidence": 0.90, "source": "rule_declarative_not_command"}

    # ── Step 2.5: Device strong check (before zero-shot, catches "activate rover" etc.) ──
    DEVICE_VERB_RE = re.compile(
        r"\b(connect|turn on|turn off|power on|power off|start|stop|enable|disable|"
        r"wake|wake up|boot|shutdown|shut down|restart|initialize|init)\b"
        r".{0,20}\b(rover|robot|drone|tello)\b"
        r"|"
        r"\b(rover|robot|drone|tello)\b"
        r".{0,20}\b(on|off|connect|disconnect|start|stop|ready|up|down)\b",
        re.IGNORECASE
    )

    if DEVICE_VERB_RE.search(text):
        print(f"[CANARY] rule_device_verb fired for: {text[:50]!r}")
        return {"intent": "device_control", "confidence": STRONG_RULE_CONF,
                "source": "rule_device_verb"}

    if _matches_device_strong(text):
        return {"intent": "device_control", "confidence": STRONG_RULE_CONF,
                "source": "rule_device_strong"}

    # ── Bare "save it" / "save this" — default to save_topic_memory ──────────
    BARE_SAVE_RE = re.compile(
        r"^(please\s+)?(can you\s+)?(save|store)\s*(it|this|that)?\.?$",
        re.IGNORECASE
    )
    if BARE_SAVE_RE.match(text.strip()):
        has_time_signal = user_mentioned_time(text) or user_mentioned_date(text)
        last_topic = (mm.get_last_topic() or "").strip()
        if not has_time_signal and last_topic and not last_topic.startswith(("web:", "session:", "_pending_")):
            return {"intent": "save_memory", "confidence": 0.95, "source": "rule_bare_save_active_topic"}

    # ── Step 3: Context-aware LLM classifier ──
    tier2_result = _evidence_score_classify(text, mm)
    if tier2_result is not None:
        return tier2_result

    belief = getattr(conversation_manager, 'belief', None)
    belief_state = belief.state if belief else None
    tool_result = _native_tool_classify_intent(user_input, mm, belief_state=belief_state)
    if tool_result is not None:
        return tool_result

    return {
        "intent": "chat",
        "confidence": 0.50,
        "source": "tool_router_fallback_chat",
    }

## aditional function for classify intent ##
def _evidence_score_classify(text: str, mm) -> dict | None:
    """
    Tier 2 classifier — keyword evidence scoring.
    Fast, no LLM, deterministic.
    
    Returns a result dict if confidence is high enough to trust.
    Returns None to signal "I don't know, escalate to LLM".
    
    This catches the common natural language inputs that rules miss
    but where keyword evidence is strong enough to be confident.
    """

    scores: dict[str, float] = {intent: 0.0 for intent in ALLOWED_INTENTS}

    VISION_FOLLOWUP_SIGNALS = [
        "what you saw", "what did you see", "based on what you saw",
        "what do you see", "what can you see", "from what you saw",
        "you saw earlier", "what was around", "what equipment",
        "what setup do you", "surroundings",
    ]
    if any(s in text for s in VISION_FOLLOWUP_SIGNALS):
        return {"intent": "chat", "confidence": 0.90,
                "source": "vision_followup_guard"}


    # At the start of _evidence_score_classify:
    DOMAIN_DISCUSSION_SIGNALS = [
        "pid", "speed", "encoder", "motor", "sensor", "controller",
        "rover", "robot", "lidar", "camera", "data", "performance",
        "observe", "measure", "record", "log the", "logging"
    ]
    in_domain_discussion = sum(1 for w in DOMAIN_DISCUSSION_SIGNALS if w in text) >= 2

    if in_domain_discussion:
        # Suppress schedule scoring entirely in technical domain context
        scores["schedule_save"] = 0.0
        scores["schedule_recall"] = 0.0

    # ── Schedule save signals ────────────────────────────────────────────
    SAVE_STRONG = [
        "remind me", "set a reminder", "don't forget", "dont forget",
        "i have a meeting", "i have an appointment", "i have a",
        "note that i have", "can you remind", "please remind",
        "don't let me forget", "dont let me forget",
        "i need to remember", "remember to",
    ]
    SAVE_WEAK = [
        "tomorrow", "next week", "on monday", "on tuesday", "on wednesday",
        "on thursday", "on friday", "on saturday", "on sunday",
        "at 9", "at 10", "at 11", "at noon", "pm", "am",
        "appointment", "meeting", "dentist", "doctor", "lecture",
        "gym", "groceries", "pickup", "deadline",
    ]

    for p in SAVE_STRONG:
        if p in text:
            scores["schedule_save"] += 0.45
    for p in SAVE_WEAK:
        if p in text:
            scores["schedule_save"] += 0.12

    # ── Schedule recall signals ──────────────────────────────────────────
    RECALL_STRONG = [
        "what do i have", "what have i got", "what's on my schedule",
        "what is on my schedule", "do i have anything", "do i have any",
        "show my schedule", "what did i schedule", "when is my",
        "what are my plans", "what have i planned", "any appointments",
        "any meetings", "upcoming events", "what's coming up",
    ]
    RECALL_WEAK = [
        "today", "tomorrow", "this week", "next week", "schedule",
        "calendar", "agenda", "upcoming", "planned",
    ]
    for p in RECALL_STRONG:
        if p in text:
            scores["schedule_recall"] += 0.50
    for p in RECALL_WEAK:
        if p in text:
            scores["schedule_recall"] += 0.10

    # ── Chat signals — prevent false positives ───────────────────────────
    CHAT_STRONG = [
        "tell me about", "explain", "how does", "how do",
        "what is", "what are", "i want to learn", "i'm curious",
        "i am curious", "curious about", "i think", "in my opinion",
        "i believe", "i feel like", "what do you think",
    ]
    for p in CHAT_STRONG:
        if text.startswith(p):
            scores["chat"] += 0.55

    # ── Context boosting ─────────────────────────────────────────────────
    # If last topic was a web search, boost web_followup
    last_topic = (mm.get_last_topic() or "").lower()
    if last_topic.startswith("web:"):
        scores["web_followup"] += 0.20

    # If last topic was rover/device, boost device_control
    if last_topic.startswith("rover:") or "rover" in last_topic:
        scores["device_control"] += 0.20

    # ── Find best scoring intent ─────────────────────────────────────────
    best_intent = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_intent]

    # ── Abstain thresholds ───────────────────────────────────────────────
    # Only return a result if we're confident enough
    # Otherwise return None to escalate to LLM
    ABSTAIN_THRESHOLD = 0.45   # below this → LLM decides
    CONFIDENT_THRESHOLD = 0.70 # above this → high confidence result

    if best_score < ABSTAIN_THRESHOLD:
        print(f"[Tier2] Abstaining (best={best_intent} score={best_score:.2f}) → escalating to LLM")
        return None

    # Sanity check — second best should not be too close
    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    gap = best_score - second_score

    if gap < 0.15 and best_score < CONFIDENT_THRESHOLD:
        # Ambiguous — two intents scored similarly, let LLM decide
        print(f"[Tier2] Ambiguous (best={best_intent} score={best_score:.2f} gap={gap:.2f}) → LLM")
        return None

    conf = min(0.90, 0.55 + best_score * 0.50)
    print(f"[Tier2] {best_intent} score={best_score:.2f} conf={conf:.2f}")

    return {
        "intent":     best_intent,
        "confidence": conf,
        "source":     "tier2_evidence",
    }

def _llm_classify_intent(text: str, user_input: str, mm, belief_state=None) -> dict:
    """
    Context-aware LLM intent classifier.
    Hybrid: existing context block (recency + pending) 
            + belief state block (domain + phase + tool results)
    """

    # ── Existing context block (keep exactly as is) ──────────────────────
    last_topic = (mm.get_last_topic() or "").strip()
    pending    = (mm.get_pending_topic() or "").strip()

    recent_turns = list(sensory_memory)[-2:] if sensory_memory else []
    recent_str = "\n".join(recent_turns) if recent_turns else "None"

    context_parts = []
    if last_topic:
        context_parts.append(f"Last topic discussed: {last_topic}")
    if pending:
        context_parts.append(f"Buddy was waiting for: {pending}")
    if recent_turns:
        context_parts.append(f"Recent conversation:\n{recent_str}")

    context_block = "\n".join(context_parts) if context_parts else "No prior context."

    # ── NEW: belief state block (adds semantic layer) ────────────────────
    belief_block = belief_state.to_prompt_block() if belief_state else ""

    # ── Build prompt: belief first, then context, then message ──────────
    prompt = f"""You are an intent classifier for a personal voice assistant called Buddy.
Classify the user message into exactly one intent.

{f"BELIEF STATE (semantic context):{chr(10)}{belief_block}{chr(10)}" if belief_block else ""}
CONVERSATION CONTEXT (recent history):
{context_block}

USER MESSAGE: "{user_input}"

AVAILABLE INTENTS:
- schedule_save:   user wants to set a reminder, save an event, or be reminded of something
- schedule_recall: user asks about their existing schedule or upcoming events
- save_memory:     user wants to save a note or topic for later reference
- recall_memory:   user wants to recall a previously saved topic or conversation
- delete_memory:   user wants to delete something from memory
- web_search:      user explicitly wants to search the internet for current information
- web_followup:    user is following up on a previous web search result
- device_control:  user is controlling a drone or robot
- system_query:    ONLY for: time, date, battery level, local weather, CPU usage, GPS location
- chat:            general questions, explanations, opinions, curiosity, learning requests

CLASSIFICATION RULES:
- "remind me", "don't forget", "I have X on Y" → schedule_save
- "what do I have", "what's on my schedule", "do I have anything" → schedule_recall
- "tell me about X", "explain X", "how does X work" → chat
- "search for", "look up", "latest news" → web_search
- If context shows last_topic is a web search and user asks follow-up → web_followup
- Short fragments like "yes", "no", "ok" with pending context → chat
- Engineering/technical discussion without scheduling language → chat
- domain=vision + words like "setup", "equipment", "what you saw" → chat (NOT schedule_save)
- domain=technical + question about surroundings → vision request, classify as chat
- last_tool=webcam FAILED + followup about what was seen → chat, acknowledge failure
- phase=confirming + "yes"/"no" → confirmation, classify as chat

Respond with ONLY valid JSON, nothing else. No explanation, no markdown.
{{"intent": "<intent_name>", "confidence": <0.0-1.0>}}"""

    try:
        response = ollama.generate(model="llama3.2:latest", prompt=prompt)
        raw = (response.get("response") or "").strip()

        print(f"[LLMClassify] raw={raw[:100]}")

        parsed = safe_parse_intent(raw)
        if not parsed or not isinstance(parsed, dict):
            return {"intent": "chat", "confidence": 0.5,
                    "source": "llm_classify_parse_fail"}

        raw_intent = parsed.get("intent", "chat")
        raw_conf   = parsed.get("confidence", 0.65)

        try:
            conf = float(raw_conf)
        except Exception:
            conf = 0.65

        conf = max(0.0, min(1.0, conf))

        if raw_intent not in ALLOWED_INTENTS:
            return {"intent": "chat", "confidence": 0.5,
                    "source": "llm_classify_invalid_intent"}

        normalized = normalize_intent(user_input, raw_intent)

        print(f"[LLMClassify] intent={normalized} conf={conf:.2f}")
        return {"intent": normalized, "confidence": conf, "source": "llm_classify"}

    except Exception as e:
        print(f"[LLMClassify] error: {e}")
        return {"intent": "chat", "confidence": 0.4, "source": "llm_classify_error"}


#############################################


def _log_uncertain_intent(text: str, result: dict, signals: dict = None):
    """
    Logs any intent classification that was uncertain or used fallback.
    These logs become your real-world test cases after daily use.
    Only logs when confidence < 0.70 OR source is fallback/zero-shot.
    Silent — never crashes Buddy even if file write fails.
    """
    conf = result.get("confidence", 1.0)
    source = result.get("source", "")

    # Only log uncertain cases — high confidence rule hits are trustworthy
    uncertain = (
        conf < 0.70
        or source in ("fallback", "zero_shot", "guard_low_conf")
        or "low_conf" in source
    )

    if not uncertain:
        return

    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_entry = {
            "ts":     datetime.now(TZ).isoformat(),
            "input":  text,
            "intent": result.get("intent"),
            "conf":   round(conf, 3),
            "source": source,
            "signals": signals or {},
            "label": None,
        }

        log_path = log_dir / "intent_uncertain.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception:
        pass

CONF_THRESHOLDS = {
    "schedule_save":    0.40,   # lower — classifier scores these around 0.45-0.55
    "schedule_recall":  0.35,   # lower — classifier scores these around 0.35-0.45
    "save_memory":      0.45,
    "recall_memory":    0.40,
    "delete_memory":    0.55,   # keep high — dangerous if wrong
    "web_search":       0.50,
    "device_control":   0.55,   # keep high — dangerous if wrong
    "system_query":     0.50,
    "chat":             0.25,   # fallback, always allow
}

def apply_guards(result: dict, user_input: str, mm) -> dict:
    intent = result["intent"]
    text = user_input.lower().strip()

    text_normalized = text
    STT_NORMALIZATIONS = [
        (r"\bwanna\b", "want to"),
        (r"\bgonna\b", "going to"),
        (r"\bgotta\b", "got to"),
        (r"\bkinda\b", "kind of"),
        (r"\bsorta\b", "sort of"),
        # collapse "want to start to X" → "want to X"
        (r"\b(want to|going to|need to|like to)\s+(start to|begin to|try to)\s+", r"\1 "),
    ]
    for pattern, replacement in STT_NORMALIZATIONS:
        text_normalized = re.sub(pattern, replacement, text_normalized)

    # Guard 0
    FLIGHT_COMMANDS = ["take off", "takeoff", "land", "hover", "fly", "go up", "go down"]
    UNSAFE_IN_ENGINEERING = {"schedule_save", "schedule_recall", "delete_memory", "device_control"}

    if any(t in text for t in ENGINEERING_TERMS) and intent in UNSAFE_IN_ENGINEERING:
        is_direct_flight_command = any(cmd in text for cmd in FLIGHT_COMMANDS)
        is_strong_device_rule = result.get("source") in ("rule_device_strong", "rule_strong")

        if intent == "device_control" and (is_direct_flight_command or is_strong_device_rule):
            pass
        elif intent == "device_control":
            pass  # let Block C handle it with full discussion signal logic
        else:
            return {"intent": "chat", "confidence": 0.75, "source": "guard_engineering"}


    WORLD_EVENT_SIGNALS = [
        "political", "politics", "situation in", "news in",
        "election", "government", "president", "war", "conflict",
        "economy", "crisis", "minister", "parliament",
    ]

    SYSTEM_QUERY_SAFE_WORDS = [
        "time", "date", "battery", "weather",
        "cpu", "location", "where am i", "clock"
    ]

    if intent == "system_query":
        # Guard A: no safe words present → definitely not a real system query
        if not any(w in text for w in SYSTEM_QUERY_SAFE_WORDS):
            return {
                "intent": "chat",
                "confidence": 0.75,
                "source": "guard_system_query_no_evidence"
            }
        # Guard B: world event signals present → route to web_search instead
        if any(w in text for w in WORLD_EVENT_SIGNALS):
            return {
                "intent": "web_search",
                "confidence": 0.85,
                "source": "guard_world_event_to_web"
            }

    if intent == "system_query":
        if "what time" in text and any(w in text for w in
                ["meeting", "appointment", "dentist", "doctor", "class"]):
            return {"intent": "schedule_recall", "confidence": 0.88,
                    "source": "guard_time_query_is_schedule"}


    if intent == "device_control":

        # Allow through: direct flight commands or strong rule source
        is_direct_flight = any(cmd in text for cmd in FLIGHT_COMMANDS)
        is_strong_rule = result.get("source") == "rule_strong"

        BYPASS_SOURCES = {"rule_strong"}
        is_strong_rule = result.get("source") in BYPASS_SOURCES

        # rule_device_verb explicitly falls through to discussion guards
        is_verb_match = result.get("source") == "rule_device_verb"

        if is_direct_flight or (is_strong_rule and not is_verb_match):
            pass  # genuine device command — skip all discussion guards

        else:
            # Discussion-shape patterns — these catch natural phrasings that
            # substring lists miss
            DISCUSSION_SHAPE_PATTERNS = [
                # "I'm confused / not sure / wondering" + anything
                r"\b(i'?m|i am)\s+(confused|not sure|unsure|wondering|stuck|"
                r"thinking|considering|curious|interested|trying to (figure|decide|understand))\b",
                # "where/how should I" — never a command, always a question
                r"\b(where|how|when|why)\s+(should|do|can|would)\s+i\b",
                # "I want/wanna/need to + build/make/learn/understand/discuss/talk"
                # (after normalization, "wanna" is already "want to")
                r"\b(i\s+(want|need|would like|hope|plan)\s+to)\s+"
                r"(build|make|create|design|learn|understand|discuss|talk|"
                r"figure out|decide|explore|consider|try)\b",
                # "I'd like / I would like" + discussion verbs
                r"\bi'?d\s+like\s+to\s+(build|make|learn|understand|discuss|talk|know)\b",
                # "in your opinion", "what do you think"
                r"\b(in your opinion|what do you think|your thoughts|your take)\b",
                # state-of-mind openers
                r"\b(state of (confusion|mind)|not sure (where|how|what|if))\b",
            ]

            matched_shapes = [
                p for p in DISCUSSION_SHAPE_PATTERNS
                if re.search(p, text_normalized, re.I)
            ]

            if matched_shapes:
                print(f"[GUARD] discussion shape matched: {matched_shapes[0][:50]}")
                return {
                    "intent": "chat",
                    "confidence": 0.85,
                    "source": "guard_discussion_shape",
                }

            QUESTION_FORM_SIGNALS = [
                "what kind of", "what type of", "what is the best",
                "what controller", "what algorithm", "what architecture",
                "what specific", "what parts", "which controller", "which is better",
                "can i use", "can i print", "can i make",
                "is it possible", "is it worth",
                "would work for", "would you recommend",
                "how does", "how should", "how can i",
            ]

            # Everything that should stay in chat domain
            DEVICE_DISCUSSION_SIGNALS = [
                # Discussion openers
                "discuss", "talk about", "tell me about", "what is",
                "explain", "curious about", "learn about", "interested in",
                "regarding", "what do you think about",
                # Build/make descriptions (not commands)
                "i want to make", "i want it to", "i want the",
                "i would like to make", "i would like to build",
                "i want to build", "able to", "capable of",
                "should be able", "my robotic arm", "the arm should",
                # Purchase consideration
                "considering buying", "thinking about buying",
                "thinking of buying",
                # Topic pivot signals
                "by the way", "by the way i am",
                # How-to questions (never device commands)
                "how do i", "how to", "how can i", "how should i",
                "implement", "should i use", "should i also",
                "what is the best", "what algorithm",
                # Return/resume signals (always chat)
                "back to", "going back", "return to",
                "let me go back", "okay back to",
                # Delete signals (should be delete_memory not device)
                "delete", "remove", "erase", "forget", "wipe", "clear",
            ]

            matched_discussion = [s for s in DEVICE_DISCUSSION_SIGNALS if s in text]
            matched_question = [s for s in QUESTION_FORM_SIGNALS if s in text]
            print(f"[GUARD_DEBUG] text={text[:60]!r} discussion={matched_discussion} question={matched_question}")

            if any(s in text for s in QUESTION_FORM_SIGNALS):
                return {"intent": "chat", "confidence": 0.85,
                        "source": "guard_question_not_device"}

            if any(s in text for s in DEVICE_DISCUSSION_SIGNALS):
                # Special case: delete signals need memory object to confirm
                DELETE_SIGNALS = ["delete", "remove", "erase", "forget", "wipe", "clear"]
                MEMORY_OBJECTS = ["conversation", "discussion", "memory",
                                "topic", "chat", "history", "about"]

                if any(s in text for s in DELETE_SIGNALS):
                    if any(w in text for w in MEMORY_OBJECTS):
                        return {"intent": "delete_memory", "confidence": 0.85,
                                "source": "guard_delete_beats_device"}
                    # delete signal but no memory object — still chat not device
                    return {"intent": "chat", "confidence": 0.80,
                            "source": "guard_discussion_not_device"}

                # All other discussion/how-to/return signals
                confidence = 0.90 if any(s in text for s in [
                    "back to", "going back", "return to", "let me go back", "okay back to"
                ]) else 0.80

                return {"intent": "chat", "confidence": confidence,
                        "source": "guard_discussion_not_device"}


    # Guard 1: Never let web_followup beat schedule_save
    SCHEDULE_SIGNALS = [
        "remind", "birthday", "appointment", "meeting",
        "tomorrow", "today at", "next week", "at 8", "at 9",
        "don't forget", "remember to", "i need to buy",
        "i have to", "i need to"
    ]
    if intent == "web_followup" and any(s in text for s in SCHEDULE_SIGNALS):
        return {"intent": "schedule_save", "confidence": 0.90, "source": "guard_schedule_beats_web"}

    # Guard 2: "can you remember something" = save, not recall
    AMBIGUOUS_REMEMBER = ["can you remember", "remember something", "save something"]
    FUTURE_TASK_SIGNALS = ["today", "tomorrow", "i have", "i need", "i must", "i should"]
    if intent == "recall_memory" and any(p in text for p in AMBIGUOUS_REMEMBER):
        if any(s in text for s in FUTURE_TASK_SIGNALS):
            return {"intent": "schedule_save", "confidence": 0.85, "source": "guard_remember_future"}

    # Guard 4: Low confidence → chat

    min_conf = CONF_THRESHOLDS.get(intent, 0.40)
    if result["confidence"] < min_conf:
        return {"intent": "chat", "confidence": result["confidence"], "source": "guard_low_conf"}

    # Guard 5: Block save_memory when there's no actual save signal
    SAVE_EVIDENCE = [
        "save", "store", "note", "remember this", "keep this",
        "bookmark", "record", "write down", "don't forget this",
        "take a note", "jot"
    ]
    CURIOSITY_SIGNALS = [
        "curious", "interested in", "tell me more", "more information",
        "more details", "what about", "how does", "explain",
        "elaborate", "expand on", "aspect of", "i want to know",
        "i'd like to know", "i would like"
    ]
    if intent == "save_memory":
        has_save_evidence = any(s in text for s in SAVE_EVIDENCE)
        looks_like_curiosity = any(s in text for s in CURIOSITY_SIGNALS)
        if not has_save_evidence and looks_like_curiosity:
            # Reroute to chat — user is asking a question, not saving
            return {"intent": "chat", "confidence": 0.70, "source": "guard_curiosity_not_save"}

    return result

def _find_specific_topic_in_query(user_input: str, topics: list[str]) -> Optional[str]:
    """
    Given a user question and a list of known topic names, check if the
    question is asking about one specific topic rather than all topics.
 
    Returns the matched topic name, or None if the question is genuinely
    asking for the full list (e.g. "what have we talked about?").
 
    Matching strategy (in priority order):
    1. Exact substring match — topic name appears literally in the question
    2. Word overlap — majority of topic words appear in the question
    3. None if no confident match
    """
    if not user_input or not topics:
        return None

    text = user_input.lower().strip()

    # ── Filter out generic "what did we talk about" questions ────────────
    # These are asking for the full list, not a specific topic
    GENERAL_LIST_SIGNALS = [
        r"^what (have|did) (we|i|you) (talk|discuss|cover|go over)",
        r"^what (topics|things|subjects) (have|did) (we|i)",
        r"^(show|list|give) (me )?(all|our|the) (topics|conversations|memories)",
        r"^(remind me|what are) (all|our|the) (topics|things)",
    ]
    for pattern in GENERAL_LIST_SIGNALS:
        if re.search(pattern, text):
            return None  # genuinely asking for the full list

    # ── Check if question contains a specific topic reference ─────────────
    # Signals that the user wants ONE specific thing
    SPECIFIC_SIGNALS = [
        r"\babout (the |my |our |that )?\w",   # "about the rover", "about quantum"
        r"\bregarding\b",
        r"\brelated to\b",
        r"\bconcerning\b",
        r"(just |we just |i just )(saved|talked|discussed|added)",
        r"\blast (thing|topic|note|save)\b",
    ]
    looks_specific = any(re.search(p, text) for p in SPECIFIC_SIGNALS)

    # ── Strategy 1: Exact substring match ────────────────────────────────
    for topic in topics:
        topic_lower = topic.lower().strip()
        if topic_lower and topic_lower in text:
            return topic

    # ── Strategy 2: Word overlap ──────────────────────────────────────────
    # Only attempt if the question looks like it's asking about something specific
    if looks_specific:
        # tokenize user input (meaningful words only)
        stop_words = {
            "the", "a", "an", "of", "in", "on", "at", "for", "to", "from",
            "about", "what", "did", "we", "i", "you", "just", "save", "saved",
            "talk", "talked", "discuss", "discussed", "our", "my", "can", "do",
            "have", "has", "is", "are", "was", "were", "tell", "me", "with",
            "that", "this", "it", "them", "and", "or", "but", "so", "which"
        }
        input_words = set(
            w for w in re.findall(r"[a-z0-9]+", text)
            if len(w) >= 3 and w not in stop_words
        )

        if not input_words:
            return None

        best_topic = None
        best_overlap = 0

        for topic in topics:
            topic_words = set(
                w for w in re.findall(r"[a-z0-9]+", topic.lower())
                if len(w) >= 3 and w not in stop_words
            )
            if not topic_words:
                continue

            overlap = len(input_words & topic_words)
            # require at least half the topic words to match
            coverage = overlap / len(topic_words)

            if overlap >= 1 and len(topic_words) <= 4:
                # Short topics (1-4 words): one word match is sufficient
                best_overlap = overlap
                best_topic = topic
                break

            if overlap > best_overlap and coverage >= 0.5:
                best_overlap = overlap
                best_topic = topic

        if best_topic:
            return best_topic

    return None

def _extract_queried_subject(user_input: str) -> Optional[str]:
    """
    Extract what the user was asking about from a recall question.
    Used to produce clean "we haven't discussed X" responses.
 
    Examples:
      "What did we discuss about quantum computing?" → "quantum computing"
      "Did we talk about machine learning?" → "machine learning"
      "What did we save about the rover?" → "the rover"
    """
    text = (user_input or "").strip()

    # Pattern: "about X", "regarding X", "on X"
    m = re.search(
        r"\b(?:about|regarding|on|concerning|related to)\s+(.+?)(?:\s*\??\s*$)",
        text, re.IGNORECASE
    )
    if m:
        subject = m.group(1).strip().rstrip("?.!")
        # reject if too vague
        VAGUE = {"it", "this", "that", "those", "these", "everything", "anything", "something"}
        if subject.lower() not in VAGUE and len(subject.split()) <= 5:
            return subject

    # Pattern: "did we talk about X", "what did we discuss on X"
    m = re.search(
        r"\b(?:talk|discuss|cover|save|store|note)\s+(?:about\s+)?(.+?)(?:\s*\??\s*$)",
        text, re.IGNORECASE
    )
    if m:
        subject = m.group(1).strip().rstrip("?.!")
        VAGUE = {"it", "this", "that", "those", "anything", "everything"}
        if subject.lower() not in VAGUE and len(subject.split()) <= 5:
            return subject

    return None

# Memory Handler
class MemoryHandler:

    def __init__(self, memory_manager, conversation_manager):
        self.memory_manager = memory_manager
        self.conversation_manager = conversation_manager
        self.specific_flow = SpecificFlow(
            memory_manager=self.memory_manager,
            retrieve_fn=retrieve_long_term_memory,
            summarize_fn=self.summarize_topic,
        )
        self.save_classifier = SaveTypeClassifier(
            ollama_client=ollama,
            model="llama3.2:latest",
        )

    def consolidate_to_long_term(self, topic: str, history: list, date: str) -> str:
        print(f"[DBG] consolidate history type={type(history)} len={len(history)}")
        print(f"[DBG] consolidate history preview={repr(history[:2]) if history else 'EMPTY'}")

        if not history:
            return f"No conversation history found for {topic}."

        raw_history = "\n".join(history) if isinstance(history, list) else history
        print(f"[DBG] raw_history preview={repr(raw_history[:300])}")

        # summarize_topic now returns a dict (extract just the summary text)
        result = self.summarize_topic(topic, raw_history, date)

        if isinstance(result, dict):
            summary_text = result.get("summary") or ""
        else:
            # fallback for legacy plain string return
            summary_text = result or ""

        # If LLM produced nothing, store raw history as fallback
        if not summary_text.strip():
            print(f"[DBG] consolidate: LLM summary empty, storing raw history as fallback")
            summary_text = raw_history

        store_long_term_memory(topic, date, conversation_history=summary_text)
        print(f"[DEBUG] Consolidated and saved to LTM: {topic}")

        return summary_text

    def _schedule_followup(self, missing_fields: list[str], data: dict) -> str:
        title = (data.get("title") or "this event").strip()

        # guard: avoid asking "What time is at 8 am?"
        if title.lower().startswith(("at ", "on ", "tomorrow", "next ")):
            title = "this event"

        if "title" in missing_fields and "timestamp" in missing_fields:
            return "Okay. What should I call it, and when is it?"
        if "title" in missing_fields:
            return "Okay. What should I call this event?"
        if "timestamp" in missing_fields:
            if data.get("date") and not data.get("time"):
                return f"Got it. What time is {title}?"
            if data.get("time") and not data.get("date"):
                return f"Got it. What day is {title}?"
            return f"Got it. When is {title}?"
        return "Okay. What should I remind you about, and when?"

    def handle_recall(self, user_input, routed_intent: str | None = None):

        PRE_SPECIFIC_PATTERNS = [
            r"\bwhat did we (just )?(save|store|note|add) (about|on|for) (.+?)[\?\.]*$",
            r"\bwhat did we (discuss|talk about|cover) (about|on|regarding) (.+?)[\?\.]*$",
            r"\bwhat (have we|did we) (saved|stored|noted) (about|on) (.+?)[\?\.]*$",
            r"\bdo we have (anything|something|notes) (about|on|regarding) (.+?)[\?\.]*$",
            r"\bdid we (discuss|talk about|cover) (.+?)[\?\.]*$",
        ]

        for pattern in PRE_SPECIFIC_PATTERNS:
            m = re.search(pattern, user_input.lower().strip())
            if m:
                subject_words = [w for w in re.findall(r"[a-z]{3,}", subject.lower())
                                if w not in {"the", "our", "this", "that"}]

                for topic in topics:
                    if any(w in topic.lower() for w in subject_words):
                        print(f"[DEBUG] pre-check direct match: '{subject}' → '{topic}'")
                        return self.specific_flow.start(topic)

                # Extract the subject from the last capture group
                subject = m.group(m.lastindex).strip().rstrip("?.,!")
                if subject and len(subject.split()) <= 5:
                    # Try to match against stored topics
                    stored_memory, msg, _ = retrieve_long_term_memory(None)
                    if stored_memory is not None:
                        topics = handleGeneral.extract_all_topic(stored_memory)
                        specific_match = _find_specific_topic_in_query(user_input, list(topics))
                        if specific_match:
                            print(f"[DEBUG] pre-check: specific match → {specific_match!r}")
                            return self.specific_flow.start(specific_match)
                        # No match — give clean "not found" response
                        return f"We haven't discussed {subject} in any saved conversations."


        if routed_intent == "schedule_recall":
            recall_type = "schedule"
        elif routed_intent == "recall_memory":
            route = recall_memory_request(
                user_input,
                routed_intent=routed_intent,
                extract_topic_fn=extract_topic_name,
            )
            print(f"[DEBUG] recall_memory_request -> {route.recall_type} (conf={route.confidence:.2f}) reason={route.reason}")
            recall_type = route.recall_type

        else:
            route = recall_memory_request(
                user_input,
                routed_intent=routed_intent,
                extract_topic_fn=extract_topic_name,
            )
            print(f"[DEBUG] recall_memory_request -> {route.recall_type} (conf={route.confidence:.2f}) reason={route.reason}")
            recall_type = route.recall_type


        RECENCY_SIGNALS = [
            "previously", "just now", "just wrote", "just saved",
            "you just", "a moment ago", "earlier just",
        ]

        if recall_type == "general":
            text = user_input.lower()
            if any(s in text for s in RECENCY_SIGNALS):
                last_saved = self.memory_manager.get_last_saved_topic()
                if last_saved:
                    print(f"[DEBUG] recall: recency reference → last_saved_topic={last_saved!r}")
                    self.memory_manager.set_pending_topic(None)
                    self.memory_manager.pending_payload = None
                    return self.specific_flow.start(last_saved)

            stored_memory, msg, _ = retrieve_long_term_memory(None)
            stored_memory, msg, _ = retrieve_long_term_memory(None)
            if stored_memory is None:
                return msg or "I couldn't find any past discussions."

            topics = handleGeneral.extract_all_topic(stored_memory)

            specific_match = _find_specific_topic_in_query(user_input, topics)

            if specific_match:
                # User asked about something specific — answer directly
                # without dumping the whole list
                print(f"[DEBUG] recall: specific match found → {specific_match!r}")
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                # Route to the specific recall flow for this topic
                return self.specific_flow.start(specific_match)

            # ── NEW: Check if user asked about something that doesn't exist ───────
            # Extract what they were asking about for a clean "not found" response
            queried_subject = _extract_queried_subject(user_input)
            if queried_subject:
                # Check if it resembles any known topic (loose match)
                text = queried_subject.lower()
                any_loose_match = any(
                    any(w in topic.lower() for w in re.findall(r"[a-z]{3,}", text))
                    for topic in topics
                )
                if not any_loose_match:
                    # Clean "not found" using the QUERIED topic, not last_topic
                    print(f"[DEBUG] recall: queried subject '{queried_subject}' not in any topic")
                    return f"We haven't discussed {queried_subject} in any saved conversations."

            # ── ORIGINAL: Genuinely general question — dump the list ─────────────
            self.memory_manager.set_last_topics_shown(topics)
            self.memory_manager.set_pending_topic("Awaiting_General_Recall")
            self.memory_manager.pending_payload = {"topics": topics}

            return compiler_general(topics) + " Which topic would you like to open?"


        elif recall_type == "temporal":
            time_reference = extract_time_reference(user_input)
            print("[DEBUG] Time Reference: ", time_reference)

            if not time_reference:
                return "I couldn’t extract a valid time from your request."

            ref_type = time_reference["type"]

            if ref_type == "date":
                ref_value = time_reference["value"]
                memory_text, topics, matching_ids = retrieve_memory_by_date(ref_value)

            elif ref_type == "month":
                ref_value = time_reference["value"]
                memory_text, topics, matching_ids = retrieve_memory_by_month(ref_value)

            elif ref_type == "year":
                ref_value = time_reference["value"]
                memory_text, topics, matching_ids = retrieve_memory_by_year(ref_value)

            elif ref_type == "week":
                start = time_reference["start"]
                end = time_reference["end"]
                ref_value = f"{start} to {end}"
                memory_text, topics, matching_ids = retrieve_memory_by_week(start, end)

            else:
                return "I recognized a time reference, but I’m not sure how to search for it."

            if not matching_ids:
                return f"I couldn’t find anything saved from {ref_value}."

            self.memory_manager.set_pending_topic("Awaiting_Temporal_Recall")
            self.memory_manager.pending_payload = RecallPayload(
                ref_value=ref_value,
                topics=topics or [],
                matching_ids=matching_ids,
                memory_text=memory_text,
            )

            if topics:
                topics_phrase = _tts_join(topics)
                return (
                    f"I found saved discussions from {ref_value} about {topics_phrase}. "
                    "Which topic would you like to open?"
                )

            return (
                f"I found the following saved memory from {ref_value}. "
                "If you want to open one topic, please say its name.\n\n"
                f"{memory_text}"
            )


        elif recall_type =="schedule":
            schedule_type = extract_scheduleType_recall(user_input)

            if ":" in schedule_type:
                schedule_type, _ = schedule_type.split(":", 1)
                schedule_type = schedule_type.strip()

            schedule_type = schedule_type.lower().strip()
            if schedule_type not in ["general", "specific", "temporal"]:
                print("[DEBUG] ⚠️ Unexpected schedule type from LLM. Defaulting to general.")
                schedule_type = "general"

            print("[DEBUG] Schedule Type: ", schedule_type)


            if schedule_type == "general":
                print("[DEBUG] Fetching all stored topics [GENERAL MODE]...")
                stored_memory = get_all_schedules_as_dicts()

                if not stored_memory:
                    print("[DEBUG] No Schedule inside SQL")
                    return "It looks like you don't have any schedules saved yet."

                # IMPORTANT: use a timezone-aware datetime for "now"
                current_dt = get_time()  # ideally returns datetime with tzinfo=Europe/Berlin
                if isinstance(current_dt, str):
                    # if your get_time() returns a string, fix it to return datetime instead
                    current_dt = datetime.now(TZ)

                return handle_general_schedule(ollama, stored_memory, current_dt)


            elif schedule_type == "specific":
                extracted = extractScheduleDetails(user_input)
                finalResult = smart_schedule_recall(extracted)

                if isinstance(finalResult, dict) and finalResult.get("status") == "need_clarification":
                    self.memory_manager.set_pending_topic("Awaiting_Schedule_Recall_Disambiguation")
                    self.memory_manager.pending_payload = {"candidates": finalResult["candidates"]}
                    return finalResult["question"]

                if isinstance(finalResult, str):
                    return finalResult

                return "I couldn't understand your schedule. Could you rephrase it?"


            elif schedule_type == "temporal":
                time_reference = extract_time_reference(user_input)
                print("[DEBUG] Time Reference: ", time_reference)

                if not time_reference:
                    return "I couldn’t extract a valid time from your request."

                ref_type = time_reference["type"]

                if ref_type == "date":
                    date = time_reference["value"]          # "YYYY-MM-DD"
                    return retrieve_schedule_by_date_sql(date)

                if ref_type == "week":
                    start = time_reference["start"]
                    end   = time_reference["end"]
                    return retrieve_schedule_by_week_sql(start, end)

                if ref_type == "month":
                    month = time_reference["value"]
                    return retrieve_schedule_by_month_sql(month)

                if ref_type == "year":
                    year = time_reference["value"]
                    return retrieve_schedule_by_year_sql(year)

                return "I recognized a time reference, but I’m not sure how to search for it."


            else:
                return "I need more info to save this schedule. What should I remind you about and when?"

        elif recall_type == "specific":
            topic_name = extract_topic_name(user_input)
            return self.specific_flow.start(topic_name)


    def summarize_topic(self, topic_name, stored_memory, discussion_date):
        if not stored_memory.strip():
            return {
                "summary": None,
                "menu_prompt": (
                    f"There is no previous conversation about {topic_name}. "
                    "Would you like to discuss it now, add new information, or move on?"
                )
            }

        display_topic = re.sub(r"^about\s+", "", topic_name, flags=re.I).strip()
        looks_like_raw_conversation = (
            "User:" in stored_memory or "Buddy:" in stored_memory
        )

        if looks_like_raw_conversation:
            print("[DBG] Preparing for summarizing, detect like a raw conversation")
            formatted_memory = "\n".join(
                f"- {line}" for line in stored_memory.split("\n") if line.strip()
            )
            summary_prompt = build_summary_prompt(topic_name, formatted_memory, discussion_date)
            response = ollama.generate(model="llama3.2:latest", prompt=summary_prompt)
            display_summary = response['response'].strip()
        else:
            print("[DBG] Prepare for the output, it seems it is like a finsihed summarization")
            # Already a finished LTM summary — display directly
            display_summary = stored_memory.strip()

        return {
            "summary": display_summary,
            "menu_prompt": (
                f"I found our conversation about {display_topic} "
                f"from {discussion_date}. "
                "Would you like me to elaborate, add new information, or move on?"
            )
        }

    ##Schedule Function

    def finalize_schedule_creation(self, user_input):
        return


    def parse_schedule(self, user_input: str) -> dict:
        today = get_date()
        now = get_time_str()

        prompt = schedule_extract_with_followup_prompt(user_input, today, now)
        raw = (ollama.generate(model="llama3.2:latest", prompt=prompt).get("response") or "").strip()

        print("[DEBUG] raw output: ", raw)


        obj = safe_json_parse(raw)
        if not isinstance(obj, dict):
            return {
                "data": {},
                "missing_fields": ["title", "timestamp"],
                "followup_question": "Okay — what should I remind you about, and when?",
                "confidence": 0.0,
                "raw": raw[:200],
            }

        data = obj.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        # ── ADD THE GUARD HERE — data exists now ──────────────────────────────
        if data.get("date") and not _ISO_DATE_RE.match(str(data.get("date", ""))):
            print(f"[DBG] parse_schedule: non-ISO date '{data['date']}' → trying deterministic fallback")
            tr = extract_time_reference(user_input)
            if tr and tr.get("type") == "date":
                data["date"] = tr["value"]
                print(f"[DBG] parse_schedule: recovered date via extract_time_reference → {data['date']}")
            else:
                data["date"] = None
                data["timestamp"] = None
        # ─────────────────────────────────────────────────────────────────────

        # GUARD FOR DATE
        if not data.get("date"):
            tr = extract_time_reference(user_input)
            if tr and tr.get("type") == "date":
                data["date"] = tr["value"]

        # Normalize time to HH:MM format after date re-parsing
        if data.get("time") and data.get("date") and not data.get("timestamp"):
            raw_time = str(data["time"]).strip().lower()
            # Handle "9am", "9pm", "9:00am" etc
            time_match = re.match(
                r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", raw_time, re.IGNORECASE
            )
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2) or 0)
                meridiem = (time_match.group(3) or "").lower()
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                data["time"] = f"{hour:02d}:{minute:02d}"

        # GUARD FOR TIME PARSED
        if not user_mentioned_time(user_input):
            data["time"] = None
            data["timestamp"] = None


        text = (user_input or "").lower()
        user_meant_midnight = any(p in text for p in ["00:00", "0:00", "midnight", "12 am", "12:00 am", "twelve am"])

        if data.get("time") in ("00:00", "0:00", "12:00 am", "12:00 a.m.") and not user_meant_midnight:
            data["time"] = None
            data["timestamp"] = None


        BAD_STRINGS = {"none", "null", "unknown", "n/a"}

        # normalize strings
        for k, v in list(data.items()):
            if isinstance(v, str):
                vv = v.strip()
                if vv == "" or vv.lower() in BAD_STRINGS:
                    data[k] = None
                else:
                    data[k] = vv

        if looks_like_time_only_title(data.get("title") or ""):
            data["title"] = None
        if looks_like_time_only_title(data.get("description") or ""):
            data["description"] = None

        data = normalize_schedule_dict(data)


        # build timestamp if date+time exist
        if data.get("date") and data.get("time") and not data.get("timestamp"):
            data["timestamp"] = f"{data['date']} {data['time']}"

        # normalize timestamp to minutes
        ts = data.get("timestamp")
        if isinstance(ts, str):
            m = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})(:\d{2})?$", ts)
            if m:
                data["timestamp"] = m.group(1)

        # recompute missing
        missing = []
        if not data.get("title"): missing.append("title")
        if not data.get("timestamp"): missing.append("timestamp")

        # ignore LLM followup, generate deterministically
        followup = self._schedule_followup(missing, data) if missing else ""

        conf = obj.get("confidence", 0.0)
        try: conf = float(conf)
        except: conf = 0.0
        conf = max(0.0, min(1.0, conf))

        return {
            "data": data,
            "missing_fields": missing,
            "followup_question": followup,
            "confidence": conf,
        }

    def _looks_like_mid_conversation(self, user_input: str) -> bool:
        """
        Returns True if this utterance looks like it's part of an ongoing
        conversation rather than a genuine save request.
        """
        text = (user_input or "").lower().strip()

        # Has explicit save signal → not mid-conversation
        EXPLICIT_SAVE = [
            "save", "store", "remember this", "take a note",
            "note this", "keep this", "bookmark"
        ]
        if any(s in text for s in EXPLICIT_SAVE):
            return False

        # Looks like a statement continuing a technical discussion
        CONTINUATION_SIGNALS = [
            "i have", "i also have", "i think", "i want",
            "i would like", "i plan to", "i'm using", "i am using",
            "which", "and also", "as well", "in addition",
            "regarding", "about", "for the",
        ]
        if any(s in text for s in CONTINUATION_SIGNALS):
            return True

        # Active topic exists → probably continuing it
        last_topic = (self.memory_manager.get_last_topic() or "").strip()
        if last_topic:
            return True

        return False

    def save_from_router(self, topic: str, raw_text: str) -> str:
        """
        Entry point for router-originated save calls. Router already decided
        this IS a save — skip SaveTypeClassifier re-detection entirely so we
        never fall back into the legacy Awaiting_Topic slot-filling FSM.
        """
        topic = (topic or "").strip()
        raw_text = (raw_text or "").strip()

        if not raw_text:
            raw_text = topic

        if not topic:
            extracted = extract_topic_name(raw_text)
            topic = self.conversation_manager._clean_extracted_topic(extracted) if extracted else None

        if not topic:
            # genuinely nothing to name it — ask once, store draft for follow-up
            self.memory_manager.set_pending_topic("Awaiting_Topic")
            self.memory_manager.pending_payload = TopicPayload(draft=raw_text)
            return "Got it — what should I call this topic?"

        # Both topic and content present — save immediately, no further asking
        self.memory_manager.set_last_topic(topic)
        store_long_term_memory(topic, get_date(), conversation_history=raw_text)
        self.memory_manager.set_pending_topic(None)
        self.memory_manager.pending_payload = None
        return f"Got it — saved that under {topic} to long-term memory."


    def save_memory(self, user_input: str, routed_intent: str | None = None):
        pending = self.memory_manager.get_pending_topic()

        if routed_intent == "save_memory" and pending is None:
            last_topic = (self.memory_manager.get_last_topic() or "").strip()
            if last_topic and not last_topic.startswith(("web:", "session:", "_pending_")):
                existing = temp_topic_memory.get(last_topic, [])
                if existing:
                    print(f"[DEBUG] save_memory: router-dispatched with active topic '{last_topic}' — saving directly")
                    return self.save_from_router(last_topic, user_input)

        print(f"[DBG save_memory] input={user_input!r} routed={routed_intent!r} pending={pending!r}")
        import traceback
        traceback.print_stack(limit=4)

        def is_blank(v):
            if v is None:
                return True
            if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "unknown", "n/a"):
                return True
            return False

        def finalize_schedule_dict(d: dict) -> dict:
            d = normalize_schedule_dict(d)

            # Only build timestamp if we trust the time
            time_user = bool(d.get("_time_user"))  # default False
            if d.get("date") and d.get("time") and time_user:
                d["timestamp"] = f"{d['date']} {d['time']}"

            # If time is not trusted, force timestamp to None
            if not time_user:
                d["time"] = None
                d["timestamp"] = None

            return normalize_schedule_dict(d)


        if pending == "Awaiting_Schedule_Details":
            partial = (self.memory_manager.pending_payload or {}).get("schedule") or {}

            HARD_EXIT_SIGNALS = [
                r"\b(what is|explain|how does|tell me about)\b",  # questions
                r"\b(search|look up|find|google)\b",              # web search
                r"\b(ekf|slam|mpc|thz|lidar|ros)\b",              # technical domain
                r"\b(actually|forget|never mind|skip)\b",          # explicit cancel
            ]

            if any(re.search(p, user_input.lower()) for p in HARD_EXIT_SIGNALS):
                # User moved on — abandon the schedule flow
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                self.conversation_manager.handle_request(user_input)

            merged = dict(partial)
            merged["_raw_text"] = (merged.get("_raw_text", "") + " " + (user_input or "")).strip()

            # update user-time flag EARLY
            merged["_time_user"] = bool(merged.get("_time_user")) or user_mentioned_time(user_input)

            # parse using accumulated context
            parsed = self.parse_schedule(merged["_raw_text"]) or {}
            new_data = parsed.get("data") or {}

            for k, v in new_data.items():
                if is_blank(v):
                    continue

                # protect fields that user didn't re-mention
                if k == "title" and partial.get("title") and not user_mentioned_event_title(user_input):
                    continue
                if k == "date" and partial.get("date") and not user_mentioned_date(user_input):
                    continue

                # HARD GATE: don't accept time/timestamp unless user mentioned time at least once
                if k in ("time", "timestamp") and not merged["_time_user"]:
                    continue

                merged[k] = v

            merged = normalize_schedule_dict(merged)

            if is_blank(merged.get("title")):
                merged["title"] = build_schedule_title(merged["_raw_text"], merged)

            merged = finalize_schedule_dict(merged)

            missing = []
            if is_blank(merged.get("title")): missing.append("title")
            if is_blank(merged.get("timestamp")): missing.append("timestamp")

            if not missing:

                title = merged["title"]
                desc  = merged.get("description") or title
                ts    = merged["timestamp"]
                loc   = merged.get("location") or None

                create_schedule(title, desc, ts, loc)

                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None

                # TTS-friendly output (no markdown)
                base_response = f"Got it, I will remember for your {title} at {ts}."
                agent = get_agent()
                if agent:
                    warning = agent.check_new_schedule(title, ts)
                    if warning:
                        return f"{base_response} {warning}"
                if loc:
                    return base_response
                return base_response

            # Still missing → persist and ask only what’s missing
            self.memory_manager.pending_payload = {"schedule": merged}
            return self._schedule_followup(missing, merged)

        if pending == "Awaiting_Topic":
            raw = user_input.strip().lower().rstrip(".,!?")
            if not raw or raw in VAGUE_TOPICS:
                return "That's a bit vague — what should I call this topic?"
            topic_name = extract_topic_name(user_input)

            VAGUE = {"this", "that", "it", "yes", "no", "ok", "okay", "none", "something"}
            if not topic_name or (isinstance(topic_name, str) and topic_name.strip().lower() in VAGUE):
                return "That's a bit vague — what should I call this topic?"

            if topic_name:
                # strip common wrappers: "it is about X", "about X", "it's X"
                for pattern in [
                    r"^it(?:'?s| is) about\s+",
                    r"^this is about\s+",
                    r"^about\s+",
                    r"^it(?:'?s| is)\s+",
                    r"^the topic is\s+",
                    r"^topic[:\s]+",
                ]:
                    topic_name = re.sub(pattern, "", topic_name, flags=re.I).strip()

            if not topic_name:
                # fallback: try stripping directly from user_input
                raw = re.sub(
                    r"^(it(?:'?s| is) about|about|the topic is|topic[:\s]+)\s*",
                    "", user_input, flags=re.I
                ).strip()
                topic_name = raw if len(raw.split()) <= 6 else None

            if not topic_name:
                return "What should I name this topic? "

            self.memory_manager.set_last_topic(topic_name)

            payload = self.memory_manager.pending_payload or {}
            if isinstance(payload, TopicPayload):
                draft = payload.draft
            else:
                draft = (payload or {}).get("draft", "")

            self.memory_manager.set_pending_topic("Awaiting_Topic_Content")
            self.memory_manager.pending_payload = TopicPayload(topic=topic_name, draft=draft)

            return (
                f"Got it — I'll save this under {topic_name}. "
                + (f"I have this so far: {draft}. " if draft else "")
                + "Want to add anything, or just say save to store it now?"
            )

        if pending == "Awaiting_Topic_Content":
            payload = self.memory_manager.pending_payload
            if isinstance(payload, TopicPayload):
                topic = payload.topic or self.memory_manager.get_last_topic()
                draft = payload.draft
            else:
                topic = (payload or {}).get("topic") or self.memory_manager.get_last_topic()
                draft = (payload or {}).get("draft", "")

            # ── Define triggers FIRST ────────────────────────────────────────────
            SAVE_TRIGGERS = {
                "save", "store", "yes", "yep", "yeah", "ok", "okay", "sure",
                "save it", "store it", "save now", "store now",
                "save that", "store that",
                "save it now", "store it now",
                "story it now", "story now", "store it", "stored it",
                "looks good", "good", "done", "done adding", "finish",
                "that's it", "thats it", "that's all", "thats all",
                "go ahead", "go ahead and save", "just save", "just save it",
                "yes save", "yes save it", "yes store it", "please save",
                "please save it", "please store it",
                "okay now save it", "ok now save it", "now save it",
                "okay save it", "ok save it", "save it please",
                "okay save", "ok save", "now save",
                "go ahead and save it",
                "go ahead and save",
                "go ahead save it",
                "go ahead save",
                "please go ahead and save",
                "save it now please",
                "yes go ahead and save",
            }

            PREAMBLE_TRIGGERS = {
                "i want to add something",
                "i want to add more",
                "i have something to add",
                "let me add something",
                "i'd like to add",
                "i would like to add",
                "i want to say something",
                "i have more to add",
                "add something",
                "keep adding",
                "continue adding",
                "i want to continue",
            }

            # ── Now normalize and debug ──────────────────────────────────────────
            t = (user_input or "").strip().lower()
            t = re.sub(r"[.!?,]+$", "", t).strip()
            print(f"[DBG SAVE] t={t!r}")
            print(f"[DBG SAVE] t in SAVE_TRIGGERS: {t in SAVE_TRIGGERS}")

            # ── Preamble guard ───────────────────────────────────────────────────
            if t in PREAMBLE_TRIGGERS or any(t.startswith(p) for p in PREAMBLE_TRIGGERS):
                return "Go ahead — what would you like to add?"

            is_save_trigger = (
                t in SAVE_TRIGGERS
                or is_yes(user_input)
                or bool(re.match(
                    r"^(please\s+)?(save|store)\s*(it|that|now|this)?\s*(now|please)?$",
                    t, re.IGNORECASE
                ))
                or bool(re.match(
                    r"^(okay|ok|alright)[,\s]+(now\s+)?(save|store)\s*(it|that|now|this)?\.?$",
                    t, re.IGNORECASE
                ))
                # ── ADD THIS ──
                or bool(re.search(
                    r"\b(save|store)\s*(it|that|this|now)?\s*$",
                    t, re.IGNORECASE
                ))
            )

            if is_save_trigger:
                store_long_term_memory(
                    topic,
                    get_date(),
                    conversation_history=draft,
                )
                self.memory_manager.set_last_saved_topic(topic)
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                print(f"[DEBUG] ✅ LTM saved: {topic!r}")
                return f"Got it — saved our notes on {topic} to long-term memory."

            updated = (draft + "\n\nUser note: " + user_input.strip()).strip()
            self.memory_manager.pending_payload = TopicPayload(topic=topic, draft=updated)
            return (
                f"Got it — added that. Current notes:\n\n{updated}\n\n"
                "Say save to store it, or keep adding."
            )

        if routed_intent == "schedule_save":
            parsed = self.parse_schedule(user_input) or {}
            data = parsed.get("data") or {}
            data["_raw_text"] = (data.get("_raw_text", "") + " " + (user_input or "")).strip()

            if data.get("title") and not user_mentioned_event_title(user_input):
                data["title"] = None

            # Mark time as user-provided if user mentioned a time
            # This prevents finalize_schedule_dict from clearing a valid timestamp
            data["_time_user"] = user_mentioned_time(user_input)

            data = finalize_schedule_dict(data)

            # If complete → save now
            if data.get("title") and data.get("timestamp"):
                title = data["title"]
                desc  = data.get("description") or title
                ts    = data["timestamp"]
                loc   = data.get("location") or None

                create_schedule(title, desc, ts, loc)

                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None

                if loc:
                    return f"Saved. {title}. At {ts}. Location: {loc}."
                return f"Saved. {title}. At {ts}."

            # Otherwise → store partial and ask follow-up
            missing = []
            if not data.get("title"):
                missing.append("title")
            if not data.get("timestamp"):
                missing.append("timestamp")

            self.memory_manager.set_pending_topic("Awaiting_Schedule_Details")
            self.memory_manager.pending_payload = {"schedule": data}

            # Prefer deterministic follow-up (avoid trusting LLM string)
            return self._schedule_followup(missing, data)

        # In save_memory(), extend AMBIGUOUS_SAVE_PHRASES:
        AMBIGUOUS_SAVE_PHRASES = [
            "save this topic",
            "save this conversation",
            "save this discussion",
            "store this topic",
            "store this conversation",
            "can you save this topic",
            "can you save this conversation",
            "please save this topic",
            "please save this conversation",
            "save about this topic",
            "save about this",
            "can you save about this",
            "please save about this",
            "store about this",
            "remember about this conversation",
            "remember this conversation",
            "remember our conversation",
            "remember what we discussed",
            "remember what we talked about",
            "don't forget this conversation",
            "dont forget this conversation",
            "keep this conversation",
            "store this discussion",
            "remember this discussion",
            "can you remember this",
            "please remember this",
            "remember about this",
        ]
        # ← Separate check for bare "save this" / "can you save this"
        # Only ambiguous if there's nothing after it
        BARE_SAVE_RE = re.compile(
            r"^(please\s+)?(can you\s+)?(save|store)(\s+about)?\s+this[.\s]*$",
            re.IGNORECASE
        )

        text = (user_input or "").lower().strip()
        is_ambiguous = (
            any(p in text for p in AMBIGUOUS_SAVE_PHRASES)
            or bool(BARE_SAVE_RE.match(text))
        )

        if is_ambiguous:
            last_topic = (self.memory_manager.get_last_topic() or "").strip().lower()

            # Case 1: active topic with STM content → go straight to confirm
            if last_topic and not last_topic.startswith("web:"):
                existing = temp_topic_memory.get(last_topic, [])
                if existing:
                    self.memory_manager.set_pending_topic("Awaiting_ConversationSave_Confirm")
                    self.memory_manager.pending_payload = ConversationSavePayload(
                        topic=last_topic,
                        history=existing,
                        date=get_date(),
                    )
                    return (
                        f"I found our conversation about {last_topic} "
                        f"with {len(existing)} exchanges. "
                        "Should I summarize and save it to long-term memory?"
                    )

            # Case 2: last_topic exists but no STM content
            if last_topic and not last_topic.startswith("web:"):
                self.memory_manager.set_pending_topic("Awaiting_Topic")
                self.memory_manager.pending_payload = TopicPayload()
                return (
                    f"I know we were discussing {last_topic} but I don't have "
                    f"enough context saved yet. What would you like me to remember about it?"
                )

            # Case 3: no last_topic at all → full general flow
            self.memory_manager.set_pending_topic("Awaiting_Topic")
            self.memory_manager.pending_payload = TopicPayload()
            return "What topic should I save? What were we discussing?"

        # 1) No pending: classify request
        save_type = save_memory_request(
            user_input,
            classifier=self.save_classifier,
        )
        print("[DEBUG] SAVE_TYPE:", save_type)

        if save_type == "general":
            # Start note-taking flow
            self.memory_manager.set_pending_topic("Awaiting_Topic")
            self.memory_manager.pending_payload = TopicPayload()
            return "Okay — what should I call this topic?"

        if save_type == "specific":
            topic_name = extract_topic_name(user_input)

            # Reject vague topics that are just pronouns or save commands
            if not topic_name or topic_name.strip().lower() in VAGUE_TOPICS:
                # Fall through to general flow instead
                self.memory_manager.set_pending_topic("Awaiting_Topic")
                self.memory_manager.pending_payload = TopicPayload()
                return "Okay — what should I call this topic?"

            self.memory_manager.set_last_topic(topic_name)
            self.memory_manager.set_pending_topic("Awaiting_Topic_Content")
            self.memory_manager.pending_payload = TopicPayload(topic=topic_name, draft="")
            return f"Got it — topic is {topic_name}. What should I remember about it?"


        if save_type == "conversation_save":
            result = resolve_save_target(
                user_input,
                self.memory_manager,
                temp_topic_memory,
            )

            # Topic not found in short term
            if result["history"] is None and not result["needs_clarification"]:
                topic = result["topic"] or "that topic"
                return (
                    f"I don't have a recent conversation about "
                    f"{topic} in this session. "
                    "We may not have discussed it yet."
                )

            # Multiple topics — ask user to pick
            if result["needs_clarification"]:
                candidates = result["candidates"]
                topics_phrase = _tts_join(candidates)

                self.memory_manager.set_pending_topic("Awaiting_ConversationSave_TopicPick")
                self.memory_manager.pending_payload = CandidatesPayload(candidates=candidates)

                return (
                    f"I have conversations about {topics_phrase} "
                    "in memory. Which one should I save?"
                )

            # Found it — confirm with user before saving
            topic = result["topic"]
            history = result["history"]

            self.memory_manager.set_pending_topic("Awaiting_ConversationSave_Confirm")
            self.memory_manager.pending_payload = ConversationSavePayload(
                topic=topic,
                history=history,
                date=get_date(),
            )

            return (
                f"I found our conversation about {topic} "
                f"with {len(history)} exchanges. "
                "Should I summarize and save it to long-term memory?"
            )

        if save_type == "unknown" or save_type is None:
            # NEW: don't interrupt an ongoing conversation
            if self._looks_like_mid_conversation(user_input):
                return None  # let it fall through to chat handler

            self.memory_manager.set_pending_topic("Awaiting_Topic")
            self.memory_manager.pending_payload = TopicPayload(draft=user_input)
            return "Sure, I'm listening. What should I call this topic?"

        # Fallback
        return None

    def execute_delete_from_router(self, target: str) -> str:
        target = (target or "").strip().lower()
        if not target:
            return "I lost track of what to delete."
        deletion_response = delete_memory_from_db(target)
        if "I've deleted all stored memory about" in deletion_response:
            if (self.memory_manager.get_last_topic() or "").strip().lower() == target:
                self.memory_manager.set_last_topic(None)
            temp_topic_memory.pop(target, None)
        return deletion_response

    def delete_memory_from_router(self, target: str) -> str:
        target = (target or "").strip()
        if not target:
            return "What should I delete? Please specify a topic."
        self.memory_manager.set_last_topic(target.lower())
        self.memory_manager.set_pending_topic("Awaiting_Delete_Confirm")
        return f"Are you sure you want to delete {target} from long-term memory?"

    def delete_memory(self, user_input):
        pending_state = self.memory_manager.get_pending_topic()

        if pending_state == "Awaiting_Delete_Confirm":
            topic_to_delete = (self.memory_manager.get_last_topic() or "").strip().lower()

            def _on_confirm():
                if not topic_to_delete:
                    return "I couldn't identify which topic to delete. Please specify it again."

                deletion_response = delete_memory_from_db(topic_to_delete)

                if "I've deleted all stored memory about" in deletion_response:
                    if (self.memory_manager.get_last_topic() or "").strip().lower() == topic_to_delete:
                        self.memory_manager.set_last_topic(None)
                    temp_topic_memory.pop(topic_to_delete, None)

                self.memory_manager.set_pending_topic(None)
                return deletion_response

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                current = self.memory_manager.get_last_topic() or ""
                if any(p in current for p in DELETE_ACTION_PATTERNS + ["don't forget", "forget about"]):
                    self.memory_manager.set_last_topic(None)
                return "Okay, I won't delete anything."

            return self.handle_confirm_state(
                user_input,
                question_context=f"Just to confirm — delete everything about {topic_to_delete or 'that'}?",
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_Delete_Confirm",
            )

        if pending_state == "Awaiting_Delete_TopicName":
            topic_name = extract_topic_name(user_input)
            if topic_name:
                self.memory_manager.set_last_topic(topic_name)
                self.memory_manager.set_pending_topic("Awaiting_Delete_Confirm")
                return f"Are you sure you want to delete **{topic_name}** from long-term memory?"
            else:
                return "I couldn’t extract a topic from that. Could you rephrase or be more specific?"

        if pending_state == "Awaiting_Temporal_Delete_Confirm":

            def _on_confirm():
                ids_to_delete = self.memory_manager.get_ids_to_delete()
                if ids_to_delete:
                    deletion_response = delete_memory_by_ids(ids_to_delete)
                    self.memory_manager.set_pending_topic(None)
                    return deletion_response
                self.memory_manager.set_pending_topic(None)
                return "I couldn't locate the memory IDs to delete."

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                return "Okay, I won't delete anything."

            return self.conversation_manager.handle_confirm_state(
                user_input,
                question_context="Do you want to delete this memory?",
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_Temporal_Delete_Confirm",
            )

        delete_type = delete_memory_request(user_input)
        print("[DEBUG] Delete Type Detected:", delete_type)

        if delete_type == "specific":
            topic_name = extract_topic_name(user_input)

            if not topic_name:
                # Fallback 1: regex extraction for "delete/remove/forget ... about X"
                m = re.search(
                    r"\b(?:delete|remove|forget|erase|wipe)\s+"
                    r"(?:our\s+|my\s+|the\s+)?(?:previous\s+|past\s+|old\s+)?"
                    r"(?:conversation|discussion|memory|topic|notes|chat|history)?\s*"
                    r"(?:about|on|for|of|regarding|related to)?\s+"
                    r"(.+?)(?:\s*[?.]?\s*$)",
                    user_input, re.IGNORECASE
                )
                if m:
                    candidate = m.group(1).strip().rstrip("?.!").strip()
                    # reject if too vague
                    if candidate and len(candidate.split()) <= 6 and candidate.lower() not in (
                        "it", "this", "that", "those", "these", "everything", "all"
                    ):
                        topic_name = candidate

            if not topic_name:
                # Fallback 2: check last_topic
                topic_name = (self.memory_manager.get_last_topic() or "").strip().lower()
                if topic_name and not topic_name.startswith("web:"):
                    self.memory_manager.set_pending_topic("Awaiting_Delete_Confirm")
                    return f"Are you sure you want to delete {topic_name} from long-term memory?"

            if not topic_name:
                return "I couldn't extract the topic you want to delete. Please try again."

            self.memory_manager.set_last_topic(topic_name)
            self.memory_manager.set_pending_topic("Awaiting_Delete_Confirm")
            return f"Are you sure you want to delete {topic_name} from long-term memory?"

        elif delete_type == "general":
            self.memory_manager.set_pending_topic("Awaiting_Delete_TopicName")
            return "Which topic would you like me to delete from memory?"


        elif delete_type == "temporal":
            time_reference = extract_time_reference(user_input)
            print("[DEBUG] Time Reference: ", time_reference)

            if not time_reference:
                return "I couldn’t extract a valid time from your request."

            ref_type = time_reference["type"]

            if ref_type == "date":
                ref_value = time_reference["value"]
                print("[DEBUG] Retrieve memory based on date")
                memory_text, _, matching_ids = retrieve_memory_by_date(ref_value)

            elif ref_type == "month":
                ref_value = time_reference["value"]
                print("[DEBUG] Retrieve memory based on month")
                memory_text, _, matching_ids = retrieve_memory_by_month(ref_value)

            elif ref_type == "year":
                ref_value = time_reference["value"]
                print("[DEBUG] Retrieve memory based on year")
                memory_text, _, matching_ids = retrieve_memory_by_year(ref_value)

            elif ref_type == "week":
                start = time_reference["start"]
                end = time_reference["end"]
                print("[DEBUG] Retrieve memory based on week")
                memory_text, _, matching_ids = retrieve_memory_by_week(start, end)
                ref_value = f"{start} to {end}"

            else:
                return "I recognized a time reference, but I’m not sure how to search for it."

            if not matching_ids:
                return f"I couldn’t find anything saved from **{ref_value}**."

            self.memory_manager.set_last_topic(f"__TEMP_DATE__{time_reference}")
            self.memory_manager.set_pending_topic("Awaiting_Temporal_Delete_Confirm")
            self.memory_manager.temp_ids_to_delete = matching_ids

            return (
                f"I found the following memory from **{time_reference}**:\n\n"
                f"{memory_text}\n\n"
                "Would you like to delete all of this?"
            )


        return None

    def delete_schedule(self, user_input: str):
        pending_state = self.memory_manager.get_pending_topic()

        # ── Confirmation pending ───────────────────────────────────────────
        if pending_state == "Awaiting_Schedule_Delete_Confirm":
            payload = self.memory_manager.pending_payload or {}
            ids_to_delete = payload.get("ids", [])
            title = payload.get("title", "that schedule")

            def _on_confirm():
                result = delete_schedules_by_ids(ids_to_delete)
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return f"Done. {result}"

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "Okay, I won't delete anything."

            return self.conversation_manager.handle_confirm_state(
                user_input,
                question_context=f"Just to confirm — delete '{title}'?",
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_Schedule_Delete_Confirm",
            )

        # ── Disambiguation pending (multiple matches found) ────────────────
        if pending_state == "Awaiting_Schedule_Delete_Pick":
            payload = self.memory_manager.pending_payload or {}
            candidates = payload.get("candidates", [])

            choice = (user_input or "").strip()
            idx = None
            if choice.isdigit():
                idx = int(choice) - 1
            else:
                # try to match by title word
                for i, c in enumerate(candidates):
                    if choice.lower() in (c.get("title") or "").lower():
                        idx = i
                        break

            if idx is None or idx >= len(candidates):
                opts = ", ".join(
                    f"{i+1}) {c['title']} at {c['timestamp']}"
                    for i, c in enumerate(candidates)
                )
                return f"Please pick one: {opts}"

            chosen = candidates[idx]
            self.memory_manager.set_pending_topic("Awaiting_Schedule_Delete_Confirm")
            self.memory_manager.pending_payload = {
                "ids": [chosen["id"]],
                "title": chosen["title"],
            }
            return (
                f"Are you sure you want to delete "
                f"'{chosen['title']}' at {chosen['timestamp']}?"
            )

        # ── Fresh delete request ───────────────────────────────────────────
        text = (user_input or "").lower()

        # Try to extract a title from the utterance
        m = re.search(
            r"\b(?:delete|remove|cancel)\s+"
            r"(?:my\s+|the\s+)?(?:schedule|reminder|appointment)?\s*"
            r"(?:for\s+|about\s+|on\s+)?(.+?)(?:\s*[?.!]?\s*$)",
            user_input, re.IGNORECASE
        )
        title_query = m.group(1).strip() if m else None

        # Fallback: check last_topic for a schedule context
        if not title_query:
            last = (self.memory_manager.get_last_topic() or "").strip()
            if last and not last.startswith("web:") and not last.startswith("rover:"):
                title_query = last

        if not title_query:
            return "Which schedule would you like me to delete? Please give me the title or event name."

        matches = delete_schedule_by_title(title_query)

        if not matches:
            return f"I couldn't find any schedule matching '{title_query}'."

        if len(matches) == 1:
            s = matches[0]
            self.memory_manager.set_pending_topic("Awaiting_Schedule_Delete_Confirm")
            self.memory_manager.pending_payload = {
                "ids": [s["id"]],
                "title": s["title"],
            }
            return (
                f"Are you sure you want to delete "
                f"'{s['title']}' at {s['timestamp']}?"
            )

        # Multiple matches — ask user to pick
        self.memory_manager.set_pending_topic("Awaiting_Schedule_Delete_Pick")
        self.memory_manager.pending_payload = {"candidates": matches}
        opts = " ".join(
            f"{i+1}) {s['title']} at {s['timestamp']}."
            for i, s in enumerate(matches[:5])
        )
        return f"I found multiple matching schedules. Which one? {opts}"


def distill_search_query(raw_query: str, topic_thread: str = "") -> str:
    """
    Converts a conversational utterance into a concise search query.
    Falls back to raw_query if LLM fails.
    """
    context = f"Topic context: {topic_thread}\n" if topic_thread else ""

    prompt = (
        f"{context}"
        f"User said: \"{raw_query}\"\n\n"
        "Convert this into a short, effective web search query (3-7 words max).\n"
        "Remove conversational filler. Focus on the core information need.\n"
        "Return ONLY the search query, nothing else."
    )

    try:
        response = ollama.generate(model="llama3.2:latest", prompt=prompt)
        result = (response.get("response") or "").strip().strip('"').strip("'")
        # Sanity check: reject if too long or looks like a sentence
        if result and len(result.split()) <= 8 and "?" not in result:
            return result
    except Exception:
        pass

    return raw_query  # fallback


def _pivot_match(text, signals):
    for s in signals:
        if re.search(r'\b' + re.escape(s) + r'\b', text):
            return True
    return False

# Conversation Manager
class ConversationManager:
    def __init__(self, memory_manager, tello_client, explorer=None, identity=None):
        init_schedule_table()  # a fresh clone has an empty DB; text mode and the test runner skip chat()
        self.identity = identity
        self.friday = Friday(notify_fn=lambda msg: self.tts.speak(msg) if hasattr(self, 'tts') and self.tts else print(f"[Friday] {msg}"))
        self.job_watchdog = JobWatchdog(friday=self.friday)
        self.job_watchdog.start()
        self.memory_manager = memory_manager
        if hasattr(self.memory_manager, "reset_session_state"):
            self.memory_manager.reset_session_state()
        self.memory_task_fusion = MemoryTaskFusion()
        self.intent_handler = IntentHandler()
        self.memory_handler = MemoryHandler(self.memory_manager, self)
        # adding new params which the device handler

        # device deps
        self.tello_client = TelloClient(agent_py_path=setting("tello_agent_py", "tello_agent.py"),
                                        python_exe=setting("tello_python") or None)
        self.rover_client = RoverClient(port=8765)

        # device handler FSM
        self.device_handler = DeviceHandlerFSM(
            mm=self.memory_manager,
            tello_client=self.tello_client,
            rover_client=self.rover_client,
            memory_handler=self.memory_handler,
            conversation_manager=self,
        )

        # simulation
        self.sim_manager = SimulationManager(
            sim_binary=setting("rover_sim_binary", "rover_sim"),
            analyze_script=setting("rover_analyze_script", "analyze_run.py"),
            results_root=setting("rover_results_dir", "rover_results"),
        )

        self.vision_cache = {}  # topic -> vision_summary
        self.explorer = explorer

        # RAG DEMO 01
        self.rag = RagCore(llm_fn=generate_llm_response)
        # ---- RAG (safe init) ----
        self.use_rag = USE_RAG
        self.rag_opts = RetrievalOptions(
            top_k=RAG_TOP_K,
            per_source_k=RAG_PER_SOURCE_K,
            use_web_context=RAG_WEB_CONTEXT,
        )

        self.prompt_builder = PromptBuilder()

        self.retriever = RetrievalEngine([
            ShortTermSource(temp_topic_memory),        # your existing short-term store
            WebContextSource(self.memory_manager),     # uses mm._web_contexts
            # later: LongTermSource(self.memory_manager) etc.
        ])

        self.web_page_fetcher = WebPageFetcher(timeout_s=10, max_chars=12000)

        self.web_helper = WebFollowupHelper(llm_fn=generate_llm_response)


        self.project_handler = ProjectHandler(
            memory_manager=self.memory_manager,
            ollama=ollama,
            model=MODEL_NAME,
            friday=self.friday,
        )

        self.belief = BeliefStateManager()

        self.spotify = SpotifyManager(memory_manager=self.memory_manager)
        self.friday = Friday(
            notify_fn=self._handle_friday_notification,
            config=FridayConfig()
        )
        self.friday.start_watching()

        # ── Presence detection ─────────────────────────────────────────────
        self.last_voice_input_time: float = time.time()
        self.away_threshold_seconds: int = 120
        self._pending_friday_notifications: list = []

        self.episodic_router = EpisodicMemoryRouter(
            topic_memory=temp_topic_memory,
            similarity_fn=compute_similarity_hybrid,
            topic_extractor=extract_topic_name,
            topic_cleaner=self._clean_extracted_topic,
        )

    def report_last_saved_location(self) -> str:
        rover_mgr = self.device_handler.rover_manager
        candidates = []
        if rover_mgr._last_saved_log_path:
            candidates.append(("rover telemetry log", rover_mgr._last_saved_log_path, rover_mgr._last_saved_log_ts or 0))
        if self.sim_manager.last_run_dir:
            candidates.append(("simulation run", self.sim_manager.last_run_dir, self.sim_manager._last_run_ts or 0))

        if not candidates:
            return "I haven't saved anything yet this session."
        candidates.sort(key=lambda c: c[2], reverse=True)
        kind, path, _ = candidates[0]
        if len(candidates) == 1:
            return f"The {kind} is saved at {path}."
        return f"The most recent save was the {kind}, at {path}."

    def handle_confirm_state(
        self,
        user_input: str,
        question_context: str,
        on_confirm,
        on_decline,
        state_name: str,
        max_unclear: int = 2,
    ) -> str:
        """
        Centralized resolver for every Awaiting_*_Confirm state.
        Replaces per-handler is_yes()/is_no() branching.
        """
        result = resolve_confirmation(user_input, question_context)

        if result == ConfirmResult.CONFIRM:
            self.memory_manager._confirm_retry_count = 0
            return on_confirm()

        if result == ConfirmResult.DECLINE:
            self.memory_manager._confirm_retry_count = 0
            return on_decline()

        if result == ConfirmResult.REDIRECT:
            # User moved on entirely — don't force them through the yes/no gate.
            self.memory_manager._confirm_retry_count = 0
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None
            return self.handle_request(user_input)

        # UNCLEAR — vary the reprompt, and hard-stop the loop
        count = getattr(self.memory_manager, "_confirm_retry_count", 0) + 1
        self.memory_manager._confirm_retry_count = count

        if count > max_unclear:
            self.memory_manager._confirm_retry_count = 0
            print(f"[ConfirmState] {state_name}: exceeded retries -> defaulting to decline")
            return on_decline()

        UNCLEAR_REPROMPTS = [
            "Sorry, was that a yes or a no?",
            "Not sure I caught that — should I go ahead, or skip it?",
        ]

        idx = min(count - 1, len(UNCLEAR_REPROMPTS) - 1)
        return UNCLEAR_REPROMPTS[idx]

    def set_tts(self, tts):
        self.tts = tts

    def is_user_present(self) -> bool:
        return (time.time() - self.last_voice_input_time) < self.away_threshold_seconds

    def _extract_paths_from_friday_message(self, msg: str) -> List[Path]:

        candidates = []

        # Markdown links: [label](path)
        for match in re.findall(r'\[[^\]]+\]\(([^)]+)\)', msg):
            candidates.append(match.strip())

        # Backtick paths: `some/path/file.md`
        for match in re.findall(r'`([^`]+\.[a-zA-Z0-9]{1,8})`', msg):
            candidates.append(match.strip())

        # Windows absolute paths
        for match in re.findall(r'[A-Za-z]:\\[^\n\r:*?"<>|]+\.[a-zA-Z0-9]{1,8}', msg):
            candidates.append(match.strip())

        # Relative or Unix-style paths
        for match in re.findall(
            r'(?:\.{1,2}[\\/])?(?:[\w .-]+[\\/])+[\w .-]+\.[a-zA-Z0-9]{1,8}',
            msg
        ):
            candidates.append(match.strip())

        # Loose filename with extension, including names like "summary in progress.md"
        for match in re.findall(r'[\w][\w ._-]{1,80}\.(?:md|txt|json|log|html|csv|py|js|ts|tsx|jsx|mjs|css|env|yaml|yml)', msg, re.I):
            candidates.append(match.strip())

        paths = []
        seen = set()

        for raw in candidates:
            raw = raw.strip().strip('.,;:"\'()[]{}')
            raw = raw.replace("/", os.sep)

            # Ignore URLs
            if raw.startswith(("http:", "https:")):
                continue

            p = Path(raw)

            # If relative, try resolving from common roots
            possible_paths = [p]

            if not p.is_absolute():
                possible_paths.extend([
                    Path.cwd() / p,
                    getattr(self, "project_dir", Path.cwd()) / p if hasattr(self, "project_dir") else Path.cwd() / p,
                    Path.home() / "buddy_friday" / p,
                    Path.home() / "buddy_friday" / "results" / p,
                ])

            for candidate in possible_paths:
                try:
                    candidate = candidate.expanduser().resolve()
                except Exception:
                    continue

                key = str(candidate).lower()
                if key not in seen and candidate.exists() and candidate.is_file():
                    seen.add(key)
                    paths.append(candidate)

        return paths

    def _score_friday_file_for_opening(self, path: Path, msg: str = "") -> Dict[str, object]:
        """
        Decide whether Buddy should auto-open a file from Friday's response.

        Returns:
        {
            "score": int,
            "action": "auto_open" | "ask_first" | "ignore",
            "reason": str,
            "kind": str,
        }
        """

        name = path.name.lower()
        suffix = path.suffix.lower()
        text = f"{msg.lower()} {name}"

        score = 0
        kind = "file"

        summary_keywords = [
            "summary", "report", "audit", "result", "results",
            "progress", "review", "findings", "verdict", "qa"
        ]

        code_keywords = [
            "code", "script", "component", "function", "class",
            "bug", "fix", "patch", "implementation"
        ]

        summary_exts = {".md", ".txt", ".log", ".html", ".csv"}
        structured_result_exts = {".json"}
        code_exts = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs",
            ".css", ".scss", ".html", ".java", ".go", ".rs",
            ".php", ".rb", ".cs", ".cpp", ".c", ".h"
        }

        config_or_sensitive_exts = {
            ".env", ".pem", ".key", ".crt", ".p12", ".pfx",
            ".sqlite", ".db"
        }

        if any(k in text for k in summary_keywords):
            score += 50
            kind = "summary"

        if suffix in summary_exts:
            score += 30

        if suffix in structured_result_exts:
            score += 15
            kind = "result data"

        if any(k in text for k in code_keywords):
            score += 15

        if suffix in code_exts:
            score -= 20
            kind = "code"

        if suffix in config_or_sensitive_exts:
            score -= 100
            kind = "sensitive/config"

        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 5:
                score -= 30
            if size_mb > 25:
                score -= 100
        except Exception:
            pass

        if score >= 60 and kind in {"summary", "result data", "file"}:
            action = "auto_open"
        elif score >= 10:
            action = "ask_first"
        else:
            action = "ignore"

        return {
            "score": score,
            "action": action,
            "reason": f"{kind} file, score {score}",
            "kind": kind,
        }

    def _open_file_for_user(self, path: Path) -> bool:

        try:
            if os.name == "nt":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            return True
        except Exception as e:
            print(f"[Friday] Could not open file {path}: {e}")
            return False

    def _clean_friday_message(self, msg: str) -> str:
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', msg)
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)
        clean = re.sub(r'`(.+?)`', r'\1', clean)
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
        clean = re.sub(r'#+\s*', '', clean)
        clean = clean.replace('→', 'passed').strip()
        return clean

    def _handle_friday_notification(self, msg: str) -> None:
        clean = self._clean_friday_message(msg)

        friday_paths = self._extract_paths_from_friday_message(msg)

        auto_opened = []
        ask_first = []

        for path in friday_paths:
            decision = self._score_friday_file_for_opening(path, msg)

            if decision["action"] == "auto_open":
                opened = self._open_file_for_user(path)
                if opened:
                    auto_opened.append((path, decision))

            elif decision["action"] == "ask_first":
                ask_first.append((path, decision))

        if not hasattr(self, 'tts') or self.tts is None:
            print(f"[Friday] {msg}")

            if auto_opened:
                for path, decision in auto_opened:
                    print(f"[Friday] Opened {decision['kind']}: {path}")

            if ask_first:
                print("[Friday] Files that may need confirmation before opening:")
                for path, decision in ask_first:
                    print(f" - {path} ({decision['reason']})")

            return

        extra_line = ""

        if auto_opened:
            opened_names = ", ".join(path.name for path, _ in auto_opened)

            # More natural message for summary-like files
            if any(decision["kind"] == "summary" for _, decision in auto_opened):
                extra_line = f" I opened the summary Friday mentioned: {opened_names}."
            else:
                extra_line = f" I opened the file Friday mentioned: {opened_names}."

        elif ask_first:
            first_path, first_decision = ask_first[0]

            if first_decision["kind"] == "code":
                extra_line = (
                    f" Friday also mentioned a code file, {first_path.name}. "
                    f"Do you want me to open it?"
                )
            else:
                extra_line = (
                    f" Friday also mentioned {first_path.name}. "
                    f"Do you want me to open it?"
                )

            # Store this so your voice-command handler can later open it
            self._pending_friday_file_to_open = str(first_path)

        if self.is_user_present():
            self.tts.speak(f"Friday just finished. {clean}{extra_line}")
            print(f"[Friday] Notified immediately — user present")

            if auto_opened:
                for path, decision in auto_opened:
                    print(f"[Friday] Auto-opened {decision['kind']}: {path}")

            if ask_first:
                for path, decision in ask_first:
                    print(f"[Friday] Waiting for permission to open {decision['kind']}: {path}")

        else:
            self._pending_friday_notifications.append({
                "msg": clean,
                "finished_at": time.time(),
                "paths": [str(p) for p in friday_paths],
                "auto_opened": [str(p) for p, _ in auto_opened],
                "ask_first": [str(p) for p, _ in ask_first],
            })
            print(f"[Friday] User away — notification queued ({len(self._pending_friday_notifications)} pending)")

    # Phase 01 Idea -> implement to make the system smarter to decide an action
    def _device_intent_fusion(
        self,
        user_input: str,
        intent_result: dict,
        belief_state,
    ) -> dict:

        source = intent_result.get("source", "")

        ABSOLUTE_BYPASS = [
            "take off", "takeoff", "hover at",
            "land the drone", "land the rover",
            "emergency stop",
        ]
        text = user_input.lower().strip()
        if any(cmd in text for cmd in ABSOLUTE_BYPASS):
            print(f"[FUSION] absolute bypass: {text[:40]!r}")
            return intent_result

        # Only bypass rule_strong (STRONG_RULES list exact phrases)
        # Everything else goes through scoring
        source = intent_result.get("source", "")
        if source == "rule_strong":
            print(f"[FUSION] bypassed — rule_strong")
            return intent_result

        # ── Compute the 6 signals ─────────────────────────────────────────────
        signals = self._extract_fusion_signals(user_input, intent_result, belief_state)
        device_score = self._compute_fusion_score(signals)

        print(f"[FUSION] signals={signals}")
        print(f"[FUSION] device_score={device_score:.3f}")

        _log_uncertain_intent(user_input, intent_result, signals)

        # ── Decision ──────────────────────────────────────────────────────────
        COMMAND_THRESHOLD = 0.62
        MARGIN            = 0.15
        discussion_score  = 1.0 - device_score

        if device_score >= COMMAND_THRESHOLD and (device_score - discussion_score) >= MARGIN:
            print(f"[FUSION] → device_control confirmed (score={device_score:.3f})")
            return intent_result

        print(f"[FUSION] → demoted to chat (device={device_score:.3f} discussion={discussion_score:.3f})")
        return {
            "intent":     "chat",
            "confidence": max(0.75, intent_result.get("confidence", 0.75)),
            "source":     "fusion_demoted_to_chat",
        }

    TOOL_REQUIRED_ARGS = {
        "move_rover":      {"direction": ("forward", "backward", "left", "right")},
        "connect_rover":   {},
        "save_memory":     {"topic": None},
        "recall_memory":   {},
        "delete_memory":   {"topic": None},
        "search_web":      {"query": None},
        "web_followup":    {},
        "schedule_save":   {},
        "system_query":    {},
    }

    def _validate_tool_args(self, tool_name: str, args: dict) -> bool:
        schema = self.TOOL_REQUIRED_ARGS.get(tool_name)
        if schema is None:
            return False
        for key, allowed_values in schema.items():
            value = (args or {}).get(key)
            if value is None:
                return False
            if allowed_values is not None and value not in allowed_values:
                return False
        return True


    def _extract_fusion_signals(
            self,
            user_input: str,
            intent_result: dict,
            belief_state,
        ) -> dict:
            from brain.core.brain import compute_similarity_hybrid

            text       = user_input.lower().strip()
            last_topic = (self.memory_manager.get_last_topic() or "").lower()

            # ── S1-S6 unchanged — keep exactly as they are ──────────────────────
            S1 = float(intent_result.get("confidence", 0.5))
            source = intent_result.get("source", "")
            print(f"[FUSION_DEBUG] raw source string = {source!r}")
            if source.startswith("rule_device"):
                S2 = 1.0
            elif source.startswith("rule_"):
                S2 = 0.80
            elif source == "tier2_evidence":
                S2 = 0.60
            else:
                S2 = 0.40

            from brain.belief_state import Domain, Phase
            if belief_state.domain == Domain.DEVICE:
                S3 = 1.0
            elif belief_state.domain == Domain.TECHNICAL:
                S3 = 0.30
            elif belief_state.domain == Domain.UNKNOWN:
                S3 = 0.50
            else:
                S3 = 0.20

            PHASE_MAP = {
                Phase.EXPLORING:  0.50,
                Phase.DEEPENING:  0.25,
                Phase.PIVOTING:   0.70,
                Phase.RETURNING:  0.40,
                Phase.CONFIRMING: 0.20,
                Phase.CLOSING:    0.30,
            }
            S4 = PHASE_MAP.get(belief_state.phase, 0.50)

            context_str = " ".join(self._get_top_k_context(last_topic, top_k=5))
            if not context_str:
                S5 = 0.50
            else:
                is_continuation = compute_similarity_hybrid(user_input, last_topic, context_str)
                S5 = 0.20 if is_continuation else 0.80

            active_entities = belief_state.active_entities or {}
            if active_entities:
                matched = sum(1 for entity in active_entities if entity in text)
                overlap_ratio = matched / len(active_entities)
            else:
                overlap_ratio = 0.0
            S6 = 1.0 - overlap_ratio

            # ── S7: Tool call specificity — NEW ──────────────────────────────────
            if source.startswith("ollama_tool:"):
                tool_name = source.split("ollama_tool:", 1)[1]
                args = intent_result.get("tool_args", {})
                print(f"[FUSION_DEBUG] tool_name={tool_name!r} args_from_intent_result={args!r} full_intent_result_keys={list(intent_result.keys())}")
                S7 = 1.0 if self._validate_tool_args(tool_name, args) else 0.3
            else:
                S7 = 0.3

            return {
                "S1_classifier_conf":   S1,
                "S2_rule_strength":     S2,
                "S3_domain_match":      S3,
                "S4_phase":             S4,
                "S5_context_sim":       S5,
                "S6_entity_overlap":    S6,
                "S7_tool_specificity":  S7,
            }


    def _compute_fusion_score(self, signals: dict) -> float:
            WEIGHTS = {
                "S1_classifier_conf":  0.15,
                "S2_rule_strength":    0.25,
                "S3_domain_match":     0.10,
                "S4_phase":            0.15,
                "S5_context_sim":      0.08,
                "S6_entity_overlap":   0.07,
                "S7_tool_specificity": 0.30,   # NEW
            }

            score = sum(WEIGHTS[k] * signals[k] for k in WEIGHTS)

            if signals["S4_phase"] <= 0.25:
                score = min(score, 0.64)

            return max(0.0, min(1.0, score))


    def _get_top_k_context(self, topic: str, top_k: int = 5) -> list[str]:
        """
        Retrieves the top_k most recent exchanges for a topic.
        Acts as the growth guard — caps how much context enters fusion scoring.
        
        Phase 3 upgrade path:
            Replace with cluster centroid retrieval when session data
            exceeds ~500 groups across long-term memory.
        """
        from brain.core.brain import temp_topic_memory

        if not topic:
            return []

        exchanges = temp_topic_memory.get(topic, [])

        # Guard: cap at top_k most recent — prevents O(n) blowup
        # as session grows. Real data stays intact in temp_topic_memory.
        if len(exchanges) > top_k:
            print(f"[FUSION] context capped: {len(exchanges)} → {top_k} exchanges")
            return exchanges[-top_k:]

        return exchanges

    def _llm(self, user_input: str, topic: str | None = None, override_sensory: list | None = None) -> str:
        # Build relevance context from topic and recent exchanges
        recent_context = ""
        if topic:
            exchanges = temp_topic_memory.get(topic, [])
            if exchanges:
                recent_context = "\n".join(exchanges[-2:])

        system_ctx = ""
        if self.identity:
            system_ctx = self.identity.build_system_context(
                current_topic=topic or "",
                recent_context=recent_context + " " + user_input,
            )
        return generate_llm_response(user_input, topic, system_context=system_ctx, override_sensory=override_sensory)


    def _make_ctx(self, **kwargs) -> RagContext:
        topic = kwargs.get("topic", "")
        short_term = kwargs.get("short_term", "")

        system_ctx = ""
        if self.identity:
            system_ctx = self.identity.build_system_context(
                current_topic=topic,
                recent_context=short_term,
            )
        belief_block = ""
        try:
            belief_block = self.belief.state.to_prompt_block()
        except Exception:
            pass

        kwargs.setdefault("system_context", system_ctx)
        kwargs.setdefault("belief_block", belief_block)
        return RagContext(**kwargs)

    def _clean_extracted_topic(self, raw_topic: str) -> str | None:
        if not raw_topic:
            return None

        t = raw_topic.strip().lower()

        action_starts = (
            "then", "can you", "do you", "could you", "please",
            "i want", "i would", "let me", "you recall", "recall",
            "remember", "tell me", "show me"
        )
        if any(t.startswith(p) for p in action_starts):
            return None

        # ── CHANGED: only reject if too long AND can't be compressed ────────
        if len(t.split()) > 6:
            stop_words = {"which", "kind", "of", "that", "might", "be", "used",
                        "what", "how", "is", "the", "a", "an", "for", "to"}
            words = [w for w in t.split() if w not in stop_words]
            compressed = " ".join(words[:4])
            return compressed if len(compressed) > 3 else None

        # ── CHANGED: allow short single-word topics ──────────────────────────
        MEANINGLESS = {"this", "that", "it", "yes", "no", "ok", "okay", "none"}
        if t in MEANINGLESS:
            return None

        return raw_topic.strip()


    # helper for RAG
    def _resolve_rag_context(self, user_input: str) -> tuple[str | None, str]:
        last_topic = self.memory_manager.get_last_topic()

        if not last_topic:
            return None, ""

        # ── Case 1: Active web topic ──
        if last_topic.startswith("web:"):
            wctx = self._get_web_ctx(last_topic)
            web_query = (wctx.get("query") or "").strip() if wctx else ""

            still_on_topic = self._is_still_on_web_topic(
                user_input, last_topic, web_query, wctx
            )

            if not still_on_topic:
                print(f"[DBG] _resolve_rag_context: topic switch from '{last_topic}' -> clearing")
                self.memory_manager.active_web_session_id = None

                raw_topic = extract_topic_name(user_input)
                new_topic = self._clean_extracted_topic(raw_topic)
                if new_topic:
                    self.memory_manager.set_last_topic(new_topic)
                    raw = temp_topic_memory.get(new_topic, [])
                    return new_topic, "\n".join(raw)

                return None, ""

            # Still on web topic — return web context block
            web_block = self._format_web_ctx_for_prompt(wctx, max_items=3)
            return last_topic, web_block

        # ── Case 2: Normal topic ──
        raw = temp_topic_memory.get(last_topic, [])
        short_term_str = "\n".join(raw) if raw else ""
        return last_topic, short_term_str


    def _is_still_on_web_topic(
        self,
        user_input: str,
        last_topic: str,
        web_query: str,
        wctx: dict | None,
    ) -> bool:
        text = (user_input or "").strip().lower()

        # ── Tier 0: Active web session + pure pronoun/opinion followup ──────────
        # "What do you think about it?", "What's your take?", "Interesting."
        # These are almost always web discussion continuations when a web session is active
        IMPLICIT_FOLLOWUP_RE = re.compile(
            r"^(what (do you think|are your thoughts|is your (take|opinion|view))|"
            r"(interesting|fascinating|really|wow|that's|thats|sounds|seems)|"
            r"(tell me more|go on|continue|and\?|so\?)|"
            r"(do you agree|would you say|is that|are they))\b",
            re.IGNORECASE
        )
        PURE_PRONOUN_RE = re.compile(
            r"^(what|how|why|is|are|do|does|can|could|would|should)\b.{0,40}"
            r"\b(it|this|that|those|these|them)\b",
            re.IGNORECASE
        )

        if last_topic.startswith("web:") and wctx:
            if IMPLICIT_FOLLOWUP_RE.match(text) or PURE_PRONOUN_RE.match(text):
                print(f"[DBG] _is_still_on_web_topic: implicit followup → staying on topic")
                return True


        # ── Tier 1: Hard exits — these always signal a topic switch ──
        HARD_EXIT_PATTERNS = [
            r"\b(search|look up|find|google)\b",          # new search
            r"\b(remind me|set a reminder|schedule)\b",   # schedule intent
            r"\b(save|store|remember this)\b",            # save intent
            r"\b(recall|what did we|previous conversation)\b",  # memory recall
            r"\b(what time|battery|weather|where am i)\b", # system queries
        ]
        for pattern in HARD_EXIT_PATTERNS:
            if re.search(pattern, text):
                return False

        # ── Tier 2: Hard stays — these always mean still on topic ──
        # User is asking a discussion question about the active web topic
        DISCUSSION_SIGNALS = [
            r"\bi('?m| am) (more |still )?(curious|interested|wondering)\b",
            r"\btell me more\b",
            r"\bmore (information|details|info) (on|about)\b",
            r"\bwhat (about|do you think|does that mean)\b",
            r"\bcan you (explain|elaborate|expand)\b",
            r"\bhow (does|do|is|are)\b",
            r"\bwhy (is|are|does|do)\b",
        ]
        if any(re.search(p, text) for p in DISCUSSION_SIGNALS):
            # Check if the topic words from the web session appear in the question
            if web_query:
                query_words = [w for w in re.findall(r"[a-z]{4,}", web_query.lower())
                            if w not in {"with", "that", "this", "from", "about", "what"}]
                if any(w in text for w in query_words):
                    return True

        # Named entity from search results mentioned
        if wctx:
            results = wctx.get("results") or []
            for r in results:
                title = (r.get("title") or "").lower()
                # extract meaningful words from result title (3+ chars, not stopwords)
                words = [w for w in re.findall(r"[a-z]{3,}", title)
                        if w not in {"the", "and", "for", "are", "from", "this", "that"}]
                if any(w in text for w in words):
                    return True

        # Web navigation commands always stay on topic
        norm = normalize_web_followup(text)
        if parse_web_followup(norm) is not None:
            return True
        if parse_index_utterance(norm) is not None:
            return True

        # ── Tier 3: Semantic check via LLM (only for ambiguous cases) ──
        # Build a compact reference string from what we know about the session
        reference_parts = [web_query]
        if wctx:
            results = wctx.get("results") or []
            for r in results[:3]:
                t = (r.get("title") or "").strip()
                if t:
                    reference_parts.append(t)
        reference = " | ".join(p for p in reference_parts if p)

        return compute_similarity_hybrid(user_input, reference, "")

    def llm_chat_with_rag(self, user_input: str, topic_hint: str | None):
        if not self.use_rag:
            return generate_llm_response(user_input, None)

        opts = self.rag_opts
        opts.topic_hint = topic_hint  # set per call

        rr = self.retriever.retrieve(user_input, opts)
        try:
            belief_block = self.belief.state.to_prompt_block()
        except Exception:
            belief_block = ""
        prompt = self.prompt_builder.build(user_input, rr.items, belief_block=belief_block)

        return generate_llm_response(prompt, "General Conversation")
    ###############

    def _get_web_ctx(self, topic: str):
        try:
            return self.memory_manager.get_web_context(topic)
        except Exception:
            return getattr(self.memory_manager, "_web_contexts", {}).get(topic)

    def _format_web_ctx_for_prompt(self, wctx: dict, max_items: int = 5) -> str:
        if not wctx:
            return ""

        q = wctx.get("query", "")
        results = wctx.get("results", []) or []
        summary = wctx.get("summary", "")

        lines = []
        if q:
            lines.append(f"Query: {q}")
        if summary:
            lines.append(f"Saved summary:\n{summary}")

        # keep it compact (title + url only)
        if results:
            lines.append("Sources:")
            for i, r in enumerate(results[:max_items], start=1):
                title = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                if title or url:
                    lines.append(f"[{i}] {title} ({url})")

        return "\n".join(lines).strip()

    def _format_search_results_brief(self, results):
        lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title","(untitled)").strip()
            snippet = (r.get("snippet") or "").strip()
            url = r.get("url","").strip()
            # keep snippets short
            if snippet and len(snippet) > 200:
                snippet = snippet[:197].rstrip() + "..."
            lines.append({
                "num": i,
                "title": title,
                "snippet": snippet,
                "url": url
            })
        return lines

    def _enrich_query(self, raw_query: str, topic_thread: str, user_input: str) -> str:
        """
        Decides whether and how to enrich the search query with topic context.
        
        Rules:
        1. If raw_query already contains topic words → no enrichment needed
        2. If raw_query is a follow-up question → prepend topic thread
        3. If raw_query starts a clearly new topic → no enrichment
        """
        if not topic_thread:
            return raw_query

        text = raw_query.lower()
        thread_words = [w for w in topic_thread.lower().split() if len(w) > 3]

        # Rule 1: already contains topic context
        if any(w in text for w in thread_words):
            return raw_query

        # Rule 2: looks like a follow-up question — enrich it
        FOLLOWUP_SIGNALS = [
            r"^(are there|is there|what about|any|how about)",
            r"^(does|do|can|could|would|should)",
            r"\brelat(ed|ion|ionship)\b",
            r"\baside from\b",
            r"\binstead of\b",
            r"\bcompared to\b",
            r"\bsimilar to\b",
            r"\bbetter than\b",
            r"\balternative\b",
            r"\bother (method|approach|model|technique|way)\b",
        ]
        is_followup = any(re.search(p, text) for p in FOLLOWUP_SIGNALS)

        if is_followup:
            return f"{raw_query} {topic_thread}"

        # Rule 3: new topic signal — don't enrich
        NEW_TOPIC_SIGNALS = [
            r"^(search|find|look up|tell me about|what is|explain)\b",
            r"^(latest|current|recent|new)\b",
        ]
        is_new_topic = any(re.search(p, text) for p in NEW_TOPIC_SIGNALS)

        if is_new_topic:
            return raw_query

        # Default: enrich conservatively
        return f"{raw_query} {topic_thread}"


    def handle_internet_request(self, user_input: str, max_results: int = 5) -> str:
        raw_query = extract_search_query(user_input) or user_input.strip()

        active_web = self.memory_manager.get_active_web_topic()
        wctx = self._get_web_ctx(active_web) if active_web else None
        topic_thread = extract_topic_thread(wctx)

        if len(raw_query.split()) <= 6:
            enriched_query = raw_query
        else:
            distilled = distill_search_query(raw_query, topic_thread)
            enriched_query = self._enrich_query(distilled, topic_thread, user_input)

        print(f"[DBG] raw_query={raw_query!r}")
        print(f"[DBG] topic_thread={topic_thread!r}")
        print(f"[DBG] enriched_query={enriched_query!r}")

        if not enriched_query:
            return "What would you like me to search for?"

        if not self.explorer:
            return "Internet access isn't configured right now."

        try:
            results = self.explorer.search_web(query=enriched_query, limit=max_results)
            print(f"[DBG] search_web returned type={type(results).__name__} value={results!r}")
        except Exception as e:
            print("[ERR] search_web crashed")
            traceback.print_exc()
            self.belief.report_tool_result(
                "web_search",
                success=False,
                summary=f"Exception during search: {type(e).__name__}: {e}"
            )
            return f'Search failed for "{enriched_query}" due to an internal error.'

        if not results:
            print(f"[DBG] No results. query={enriched_query!r}, results={results!r}")
            self.belief.report_tool_result(
                "web_search",
                success=False,
                summary=f"No results for '{enriched_query[:40]}'"
            )
            return f'I couldn\'t find results for "{enriched_query}".'

        self.belief.report_tool_result(
            "web_search",
            success=True,
            summary=f"{len(results)} results for '{enriched_query[:40]}'"
        )

        topic_key = f"web:{enriched_query.lower()[:120].strip()}"
        formatted = self._format_search_results_brief(results)

        try:
            overview = self.web_helper.llm_summarize_search_results(enriched_query, results[:3])
        except Exception:
            print("[ERR] llm_summarize_search_results crashed")
            traceback.print_exc()
            overview = f"I found {len(results)} results for {enriched_query!r}."

        speech = f"{overview} "

        self.memory_manager.set_web_context(
            topic_key,
            enriched_query,
            results,
            summary=speech,
            root_thread=topic_thread or enriched_query
        )

        self.memory_manager.set_last_topic(topic_key)

        print("\n[WEB] Full results:")
        for item in formatted:
            snippet = f" — {item['snippet']}" if item['snippet'] else ""
            print(f"[{item['num']}] {item['title']}{snippet}\n    {item['url']}\n")

        return speech

    def answer_from_web_followup(self, user_input: str) -> str:

        last = (self.memory_manager.get_last_topic() or "").strip()
        web_topic = self.memory_manager.get_active_web_topic()

        # fallback
        if not web_topic and last.startswith("web:"):
            web_topic = last

        if not web_topic:
            items = getattr(self.memory_manager, "_web_contexts", {}).items()
            if items:
                web_topic = max(items, key=lambda kv: kv[1].get("created_at", 0))[0]

        if not web_topic:
            return "I don’t have a recent web search saved yet. Tell me what you want to search for."

        wctx = self.memory_manager.get_web_context(web_topic)
        if not wctx:
            return "I found a reference to your last web search, but I couldn’t load its saved context."

        results = wctx.get("results") or []
        if not results:
            return "I found your last web search, but no results were saved with it."

        query = (wctx.get("query") or "").strip()
        action = parse_web_followup(user_input)

        if action:
            act, n = action
            print("[DBG] act result: ", act)

            if n is None:
                n = wctx.get("active_result") or 1

            idx = n - 1
            if idx < 0 or idx >= len(results):
                return f"I couldn’t find result {n}. I currently have results 1 through {len(results)} saved."

            result = results[idx]
            title = (result.get("title") or "(untitled)").strip()
            url = (result.get("url") or "").strip()
            source = self.web_helper.extract_source_name(url)

            if act == "open":
                wctx["active_result"] = n
                self.memory_manager.set_web_context(
                    web_topic,
                    query,
                    results,
                    summary=wctx.get("summary", ""),
                    root_thread=wctx.get("root_thread", "")
                )

                if url:
                    ok = _open_url_in_browser(url)
                    if ok:
                        return f"Opening result {n} from {source}, '{title}'."
                    return f"I found result {n} from {source}, but I couldn’t open it automatically."

                return f"I found result {n}, '{title}', but there isn’t a saved link for it."

            if act == "summarize":
                payload = self.web_helper.build_web_result_payload(
                    result, n, user_input, query=query
                )

                page_text = self.web_page_fetcher.fetch_page_text(url)
                if page_text:
                    llm_text = self.web_helper.llm_summarize_web_result(payload, page_text=page_text)
                else:
                    llm_text = self.web_helper.llm_summarize_web_result(payload)

                if url:
                    _open_url_in_browser(url)

                return (
                    f"Here’s a quick summary of result {n} from {source}, '{title}':\n"
                    f"{llm_text}\n\n"
                    f"I also opened the site in your browser."
                )

        t = (user_input or "").lower().strip()
        m = re.search(r"\bcompare\s*(?:#?(\d+))\s*(?:and|vs|v|versus)\s*(?:#?(\d+))\b", t)
        if m:
            i1, i2 = int(m.group(1)) - 1, int(m.group(2)) - 1
            if i1 < 0 or i1 >= len(results) or i2 < 0 or i2 >= len(results):
                return f"Please choose valid result numbers between 1 and {len(results)}."

            a = results[i1]
            b = results[i2]

            pa = self.web_helper.build_web_result_payload(a, i1 + 1, user_input, query=query)
            pb = self.web_helper.build_web_result_payload(b, i2 + 1, user_input, query=query)

            llm_text = self.web_helper.llm_compare_web_results(pa, pb, user_input)

            return (
                f"Here’s a comparison of results {i1 + 1} and {i2 + 1}:\n"
                f"{llm_text}\n\n"
                f"You can say 'open {i1 + 1}' or 'open {i2 + 1}' next."
            )

        return (
            "I still have the results from your last web search.\n"
            "You can say things like:\n"
            "- 'open one'\n"
            "- 'summarize two'\n"
            "- 'compare one and two'"
        )

    def _handle_vision_awareness_confirm(self, user_input: str) -> str:
        payload = self.memory_manager.pending_payload or {}
        context_hint = payload.get("context_hint", "")
        original_question = payload.get("pending_vision_question", user_input)

        def _on_confirm():
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None
            try:
                path = capture_single_frame(out_dir=WEBCAM_SAVE_DIR)
                if not path:
                    return "I couldn't access the webcam. " + self._llm(original_question, None)

                recognition = recognize_face(path)
                authorized = (recognition.status == RecognitionStatus.AUTHORIZED)

                vision_result = run_vision_pipeline(
                    image_path=path,
                    user_question=original_question,
                    conversation_manager=self,
                    authorized_user=authorized,
                    is_webcam=True,
                    context_hint=context_hint,
                )

                prefix = get_silent_prefix()
                llm_reply = self._llm(original_question, self.memory_manager.get_last_topic())
                combined = f"{prefix}{vision_result}\n\n{llm_reply}".strip()

                topic_key = f"vision_awareness_{int(time.time())}"
                store_short_term_memory(original_question, combined, topic=topic_key)
                self.memory_manager.set_last_topic(topic_key)
                return combined
            except Exception as e:
                print(f"[VisionAwarenessConfirm] error: {e}")
                return self._llm(original_question, None)

        def _on_decline():
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None
            return self.handle_request(original_question)

        return self.handle_confirm_state(
            user_input,
            question_context="Want me to take a quick look through the camera?",
            on_confirm=_on_confirm,
            on_decline=_on_decline,
            state_name="Awaiting_VisionAwareness_Confirm",
        )

    def handle_pending(self, pending_state: str, user_input: str):
        if not pending_state:
            return None

        if pending_state == "Awaiting_SaveTarget_Confirm":
            payload = self.memory_manager.pending_payload or {}
            raw_text = payload.get("raw_text", user_input)
            text = (user_input or "").lower()

            ROVER_CHOICE = ["log", "telemetry", "rover", "data"]
            CONVO_CHOICE = ["conversation", "discussion", "chat", "topic", "talk"]

            if any(w in text for w in ROVER_CHOICE):
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                self.memory_manager._confirm_retry_count = 0
                return self.device_handler.rover_manager.save_last_log()

            if any(w in text for w in CONVO_CHOICE):
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                self.memory_manager._confirm_retry_count = 0
                return self.memory_handler.save_memory(raw_text)

            count = getattr(self.memory_manager, "_confirm_retry_count", 0) + 1
            self.memory_manager._confirm_retry_count = count
            if count > 2:
                self.memory_manager._confirm_retry_count = 0
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "Okay, I won't save anything for now."

            return "Sorry — the rover log, or our conversation?"

        if pending_state == "Awaiting_LLM_Tool_Confirm":
            payload = self.memory_manager.pending_payload or {}
            tool_name = payload.get("tool_name")
            args = payload.get("tool_args", {})

            def _on_confirm():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                dispatch = make_dispatch_table(self)
                fn = dispatch.get(tool_name)
                if fn:
                    return fn(args)
                return "Something went wrong, it seems I lost track of that action"

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "Okay, I wont do that."

            return self.handle_confirm_state(
                user_input,
                question_context=format_tool_confirm_question(tool_name),
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_LLM_Tool_Confirm"
            )

        # ── Friday confirmation handler ────────────────────────────────────────
        if pending_state == "Awaiting_Friday_Confirm":
            payload = self.memory_manager.pending_payload or {}
            task = payload.get("task", "")
            reason = payload.get("reason", "sensitive task")

            def _on_confirm():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return self.friday._dispatch_async(task, session_label="buddy_voice_confirmed")

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "Got it, I won't send that to Friday."

            return self.handle_confirm_state(
                user_input,
                question_context=format_friday_confirm_question(task, reason),
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_Friday_Confirm",
            )

        if pending_state == "Awaiting_Rover_Sequence_Confirmation":
            return self._handle_rover_sequence_confirmation(user_input)

        # --- Device control flow ---
        if pending_state == "Awaiting_ConversationSave_Confirm":
            payload = self.memory_manager.pending_payload

            if isinstance(payload, ConversationSavePayload):
                topic   = payload.topic
                history = payload.history
                date    = payload.date or get_date()
            else:
                # fallback for any legacy dict still in flight
                topic   = (payload or {}).get("topic")
                history = (payload or {}).get("history", [])
                date    = (payload or {}).get("date") or get_date()

            def _on_confirm():
                summary = self.memory_handler.consolidate_to_long_term(topic, history, date)
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None

                response = f"Done. I summarized and saved our conversation about {topic} to long-term memory."
                store_short_term_memory(
                    "system: conversation summarized and saved",
                    f"Summary saved: {summary}",
                    topic=topic,
                )
                return response

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "Okay, I won't save it."

            PIVOT_SIGNALS = [
                "now let", "let us talk", "talk about", "discuss",
                "what about", "by the way", "can you remind",
                "actually", "forget", "never mind", "skip",
            ]
            if any(s in user_input.lower() for s in PIVOT_SIGNALS):
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                # Don't save — user moved on — re-route
                return self.handle_request(user_input)

            return self.handle_confirm_state(
                user_input,
                question_context=f"Should I save our conversation about {topic}?",
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_LLM_Tool_Confirm",
            )

        if pending_state == "Awaiting_Project_Resume":
            return self.project_handler.handle_resume_response(
                user_input,
                conversation_manager=self,
            )

        if pending_state == "Awaiting_Note_Confirm":
            return self.project_handler._handle_note_confirm(
                user_input, conversation_manager=self
            )


        if pending_state == "Awaiting_ConversationSave_TopicPick":
            p = self.memory_manager.pending_payload
            if isinstance(payload, CandidatesPayload):
                candidates = payload.candidates
            else:
                candidates = (payload or {}).get("candidates", [])
            chosen = resolve_topic_from_utterance(user_input, candidates)

            if not chosen:
                topics_phrase = _tts_join(candidates)
                return f"Which topic — {topics_phrase}?"

            history = temp_topic_memory.get(chosen, [])

            self.memory_manager.set_pending_topic("Awaiting_ConversationSave_Confirm")
            self.memory_manager.pending_payload = ConversationSavePayload(
                topic=chosen,
                history=history,
                date=get_date(),
            )
            return (
                f"Got it — our conversation about {chosen}. "
                "Should I summarize and save it?"
            )

        if pending_state == "Awaiting_Vision_Source_Confirm":
            payload = self.memory_manager.pending_payload
            if isinstance(payload, VisionPayload):
                user_question = payload.user_question
            else:
                user_question = (payload or {}).get("user_question", "")

            choice = resolve_vision_source_choice(user_input)

            if choice == VisionSourceChoice.SCREEN:
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                self.memory_manager._confirm_retry_count = 0

                grabbed = grab_clipboard_image(out_dir="screenshots")
                if grabbed:
                    img_path, h = grabbed
                    result = run_vision_pipeline(
                        image_path=img_path,
                        user_question=user_question,
                        conversation_manager=self,
                    )
                    topic_key = _extract_vision_topic(result, img_path)
                    store_short_term_memory(user_question, result, topic=topic_key)
                    self.memory_manager.set_last_topic(topic_key)
                    return result
                selection = select_screenshots()
                if selection.outcome == SelectionOutcome.SINGLE:
                    result = run_vision_pipeline(
                        image_path=selection.paths[0],
                        user_question=user_question,
                        conversation_manager=self,
                    )
                    topic_key = _extract_vision_topic(result, selection.paths[0])
                    store_short_term_memory(user_question, result, topic=topic_key)
                    self.memory_manager.set_last_topic(topic_key)
                    return result
                return "I couldn't find a screenshot to analyze. Take one with Windows Shift S."

            if choice == VisionSourceChoice.CAMERA:
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                self.memory_manager._confirm_retry_count = 0

                path = capture_single_frame(out_dir=WEBCAM_SAVE_DIR)
                if not path:
                    return "I couldn't access the webcam. Please check your camera connection."

                recognition = recognize_face(path)
                if recognition.status == RecognitionStatus.UNKNOWN:
                    return recognition.message

                result = run_vision_pipeline(image_path=path, user_question=user_question,
                                            conversation_manager=self)
                topic_key = _extract_vision_topic(result, path)
                store_short_term_memory(user_question, result, topic=topic_key)
                self.memory_manager.set_last_topic(topic_key)
                return result

            if choice == VisionSourceChoice.DECLINE:
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                self.memory_manager._confirm_retry_count = 0
                return self.handle_request(user_question)

            # UNCLEAR — bounded retry, same safety pattern as handle_confirm_state
            count = getattr(self.memory_manager, "_confirm_retry_count", 0) + 1
            self.memory_manager._confirm_retry_count = count

            if count > 2:
                self.memory_manager._confirm_retry_count = 0
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return self.handle_request(user_question)

            return "Sorry — screen, camera, or should I skip it?"

        if pending_state == "Awaiting_Vision_Burst_Confirm":
            payload = self.memory_manager.pending_payload
            if isinstance(payload, VisionPayload):
                burst_paths = payload.burst_paths
                user_question = payload.user_question
            else:
                burst_paths = (payload or {}).get("burst_paths", [])
                user_question = (payload or {}).get("user_question", "")

            def _on_confirm():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                result = run_vision_pipeline(
                    image_path=burst_paths,
                    user_question=user_question,
                    conversation_manager=self,
                )
                img_name = Path(burst_paths[0]).stem
                topic_key = f"vision_{img_name[-8:]}"
                store_short_term_memory(user_question, result, topic=topic_key)
                self.memory_manager.set_last_topic(topic_key)
                return result

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                result = run_vision_pipeline(
                    image_path=burst_paths[0],
                    user_question=user_question,
                    conversation_manager=self,
                )
                img_name = Path(burst_paths[0]).stem
                topic_key = f"vision_{img_name[-8:]}"
                store_short_term_memory(user_question, result, topic=topic_key)
                self.memory_manager.set_last_topic(topic_key)
                return result

            return self.handle_confirm_state(
                user_input,
                question_context=format_vision_burst_confirm(burst_paths),
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_Vision_Burst_Confirm",
            )

        if pending_state in ():
            ROVER_PIVOT_SIGNALS = [
                "actually", "wait", "what controller", "what algorithm",
                "by the way", "how should", "what about",
                "can the rover",
                "on steep", "incline", "terrain", "capability",
            ]
            if _pivot_match(user_input.lower(), ROVER_PIVOT_SIGNALS):
                print(f"[ROVER_PIVOT] fired for: {user_input[:50]!r}")
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.set_last_topic(None)  # ← add this
                return self.handle_request(user_input)
            return self.device_handler.handle(user_input)

        # ── NEW: General vision awareness confirm (non-device contexts) ──────────
        if pending_state == "Awaiting_VisionAwareness_Confirm":
            return self._handle_vision_awareness_confirm(user_input)

        if pending_state in ("Awaiting_Device_Name", "Awaiting_Rover_Correction_Confirm"):
            return self.device_handler.handle(user_input)

        if pending_state == "Awaiting_DeviceControl_Confirm":
            text_lower = user_input.lower()
            print(f"[PIVOT_DEBUG] input={text_lower[:60]!r}")
            DEVICE_PIVOT_SIGNALS = [
                "now i want", "let us talk", "talk about",
                "what kind of", "what controller", "what algorithm",
                "what altitude", "what sensors", "what about",
                "should i use", "how should i", "how should the",
                "what happens when", "by the way",
                "actually wait", "before i start",
                "battery drain", "what is the battery",
                "can the rover", "is it possible to",
            ]

            for s in DEVICE_PIVOT_SIGNALS:
                pattern = r'\b' + re.escape(s) + r'\b'
                if re.search(pattern, text_lower):
                    print(f"[PIVOT_DEBUG] matched: {s!r}")

            if _pivot_match(text_lower, DEVICE_PIVOT_SIGNALS):
                print(f"[PIVOT_DEBUG] → pivot fired, clearing pending")
                self.memory_manager.set_pending_topic(None)
                return self.handle_request(user_input)

            print(f"[PIVOT_MISS] no match → device_handler")
            return self.device_handler.handle(user_input)

        if pending_state in ["Awaiting_Topic", "Awaiting_Topic_Content"]:

            PIVOT_RE = re.compile(
                r"^(what|how|why|actually|wait|remind|search|forget|"
                r"never mind|skip|cancel|explain|tell me|look up|"
                r"good|and|now|okay|so|back to|going deeper)\b",  # ← add these
                re.IGNORECASE
            )
            RECALL_SIGNALS = [
                "what did i save", "what did we", "do i have",
                "what was", "remind me about", "did we discuss"
            ]

            is_pivot = (
                PIVOT_RE.match(user_input.strip())
                and not any(s in user_input.lower() for s in
                            ["save", "store", "note", "remember this", "keep"])
            )
            is_recall = any(s in user_input.lower() for s in RECALL_SIGNALS)

            if is_pivot or is_recall:
                print(f"[DEBUG] Awaiting_Topic: pivot detected → re-routing")
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return self.handle_request(user_input)

            result = self.memory_handler.save_memory(user_input)
            if result is None:
                return self.start_new_conversation(user_input)
            return result

        # --- Delete flows (delegated to handler) ---
        if pending_state in ["Awaiting_Delete_TopicName", "Awaiting_Delete_Confirm", "Awaiting_Delete_Topic"]:
            return self.memory_handler.delete_memory(user_input)

        # --- Schedule flows (delegated to handler) ---
        if pending_state in ["Awaiting_Schedule_Details"]:

            # ── Pivot guard — user switched topic mid-schedule flow ──────────
            PIVOT_SIGNALS = [
                r"\bactually\b.{0,15}\b(how|what|why|explain|tell me|can you)\b",
                r"\b(forget|skip|cancel|drop) (the |that |this )?(schedule|reminder|meeting|appointment)\b",
                r"\b(connect|activate|start|launch)\b.{0,20}\b(rover|drone|robot|tello)\b",
            ]
            if any(re.search(p, user_input.lower()) for p in PIVOT_SIGNALS):
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return self.handle_request(user_input)  # re-route cleanly

            return self.memory_handler.save_memory(user_input, routed_intent="schedule_save")

        if pending_state == "Awaiting_Temporal_Recall":
            p = self.memory_manager.pending_payload
            if isinstance(p, RecallPayload):
                topics = p.topics
                ref_value = p.ref_value
            else:
                topics = (p or {}).get("topics", [])
                ref_value = (p or {}).get("ref_value", "that time period")

            chosen = resolve_topic_from_utterance(user_input, topics) if topics else None
            if not chosen:
                return f"Which topic from {ref_value} should I open? Please say the topic name."

            # clear pending then route to specific
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None
            return self.memory_handler.specific_flow.start(chosen)

        if pending_state == "Awaiting_General_Recall":
            REJECTION_SIGNALS = ["none", "none of them", "not these", "never mind",
                                "forget it", "not any of those"]
            if any(s in user_input.lower() for s in REJECTION_SIGNALS):
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "Okay, what would you like to do instead?"

            p = self.memory_manager.pending_payload
            if isinstance(p, RecallPayload):
                topics = p.topics
            else:
                topics = (p or {}).get("topics", [])
            text = (user_input or "").strip()

            # resolve which topic user meant
            chosen = resolve_topic_from_utterance(text, topics)

            if not chosen:
                return "Sorry, which topic should I open? Please say the topic name."

            # clear general pending and route to specific
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None

            return self.memory_handler.specific_flow.start(chosen)


        # --- Topic naming flow ---
        if pending_state == "Awaiting_Topic":
            text = user_input.strip()

            # unwrap common answers in topic-naming mode
            m = re.search(r"\b(it('?s)?|it is|this|the topic)\s+(is\s+)?about\s+(.+)$", text, re.I)
            if m:
                text = m.group(4).strip()

            m = re.search(r"\b(topic|subject|title)\s*[:\-]\s*(.+)$", text, re.I)
            if m:
                text = m.group(2).strip()

            topic_name = extract_topic_name(text) or _clean_topic(text)

            if topic_name:
                self.memory_manager.set_last_topic(topic_name)
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                print(f"[DEBUG] New Topic Name Set: {topic_name}")
                return self.continue_conversation(f"Let's discuss about {topic_name}.", topic_name)

            return "I'm not sure what to name the topic. Could you clarify?"


        if pending_state == "Awaiting_Schedule_Recall_Disambiguation":
            payload = self.memory_manager.pending_payload
            if isinstance(payload, CandidatesPayload):
                candidates = payload.candidates
            else:
                candidates = (payload or {}).get("candidates", [])
            choice = user_input.strip()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(candidates):
                    s = candidates[idx]
                    self.memory_manager.set_pending_topic(None)
                    self.memory_manager.pending_payload = None
                    return f"You meant: **{s['title']}** at **{s['timestamp']}** in **{s.get('location') or '—'}**."
            return "Please reply with a number (1-5)."

        if pending_state in ["Awaiting_Schedule_Delete_Confirm",
                            "Awaiting_Schedule_Delete_Pick"]:
            return self.memory_handler.delete_schedule(user_input)

        # --- Add info to an existing topic ---
        if pending_state == "Awaiting_Topic_Add_Info":
            p = self.memory_manager.pending_payload
            if isinstance(p, TopicPayload):
                topic = p.topic
            else:
                topic = (p or {}).get("topic")

            if not topic:
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return "I lost track of the topic. Please try again."

            if user_input.strip().lower() == "done":
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_payload = None
                return f"Got it — finished updating **{topic}**. What next?"

            self.memory_manager.set_last_topic(topic)
            reply = self.continue_conversation(user_input, topic)

            # stay in add-info mode
            self.memory_manager.set_pending_topic("Awaiting_Topic_Add_Info")
            self.memory_manager.pending_payload = TopicPayload(topic=topic)

            return reply

        # --- Specific recall menu ---
        if pending_state == SpecificFlow.PENDING_MENU:
            return self.memory_handler.specific_flow.handle_menu_turn(user_input)


        # --- Update confirm flow ---
        if pending_state == "Awaiting_Update_Confirm":
            update_payload = getattr(self.memory_manager, "pending_update_payload", None)

            def _on_confirm():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_update_payload = None
                self.memory_manager._pending_new_topic = None

                if not update_payload:
                    return "I lost track of what we were saving. Please try again."

                topic = update_payload["topic"]
                store_long_term_memory(
                    topic,
                    update_payload["date"],
                    conversation_history=update_payload["conversation_history"],
                )
                self.memory_manager.set_last_saved_topic(topic)
                return f"Saved our discussion about {topic}."

            def _on_decline():
                self.memory_manager.set_pending_topic(None)
                self.memory_manager.pending_update_payload = None
                topic = (update_payload or {}).get("topic", "that topic")

                new_topic = getattr(self.memory_manager, "_pending_new_topic", None)
                if new_topic:
                    self.memory_manager.set_last_topic(new_topic)
                    self.memory_manager._pending_new_topic = None
                else:
                    self.memory_manager.set_last_topic(None)

                return f"Okay — I won't save {topic}."

            return self.handle_confirm_state(
                user_input,
                question_context=f"Should I save our discussion about {(update_payload or {}).get('topic', 'this topic')}?",
                on_confirm=_on_confirm,
                on_decline=_on_decline,
                state_name="Awaiting_Update_Confirm",
            )

        # --- Unknown pending state safety fallback ---
        print(f"[DEBUG] Awaiting_Update_Confirm: neither yes/no — re-routing as new question")
        self.memory_manager.set_pending_topic(None)
        self.memory_manager.pending_update_payload = None
        self.memory_manager.set_last_topic(None)
        return self.handle_request(user_input)

    def build_resume_prompt(self, snap=None):
        snap = snap or {}

        # Prefer restored state, but allow snap fallbacks
        st = self.memory_manager.get_pending_topic() or snap.get("pending_topic")
        payload = (self.memory_manager.pending_payload or {}) or (snap.get("pending_payload") or {})

        frame_type = snap.get("frame_type")
        paused_reason = snap.get("paused_reason")

        # --- Topic context resume ---
        if frame_type == "topic_context":
            topic = snap.get("last_topic") or self.memory_manager.get_last_topic() or "that topic"
            return f"Back to **{topic}** — what would you like to discuss or add next?"

        # --- FSM flow resume (specific known states) ---
        if st == "Awaiting_Topic_Add_Info":
            topic = payload.get("topic") or self.memory_manager.get_last_topic() or "that topic"
            return f"Back to updating **{topic}** — tell me what to add, or say “done”."

        if st == SpecificFlow.PENDING_MENU:
            topic = payload.get("topic") or self.memory_manager.get_last_topic() or "that topic"
            return f"Back to **{topic}** — say “show everything”, “add information”, or “move on”."

        if st == "Awaiting_General_Recall":
            return "Back to your topic list — which topic would you like to open?"

        # --- Generic flow-state resume (nice fallback) ---
        if frame_type == "flow_state" and st:
            return "Back to your previous task — what should we do next?"

        return "Back to what we were doing — what would you like to do next?"

    def _vision_awareness_check(self, user_input: str, intent: str) -> str | None:

        # early guard


        print(f"[VACheck] called with intent={intent!r} input={user_input[:50]!r}")
        AWARENESS_ELIGIBLE = {"chat", "device_control"}
        if intent not in AWARENESS_ELIGIBLE:
            print(f"[VACheck] skipped — intent not eligible")
            return None

        if wants_vision(user_input):
            print(f"[VACheck] skipped — wants_vision triggered")
            return None

        FACE_IDIOMS = [
            "made the face", "make the face", "face of", "facing the",
            "in the face", "face it", "face this", "face that",
            "face my", "face your", "face the fact",
        ]
        if any(p in user_input.lower() for p in FACE_IDIOMS):
            print(f"[VACheck] skipped — face idiom detected")
            return None

        last_topic = self.memory_manager.get_last_topic() or ""
        pending = self.memory_manager.get_pending_topic() or ""

        # Skip vision check when in device error recovery state
        if pending in ():
            print(f"[VACheck] skipped — rover context pending")
            return None
        if last_topic.startswith("vision_awareness_"):
            print(f"[VACheck] skipped — recent vision grab")
            return None


        recent = list(sensory_memory)[-2:] if sensory_memory else []

        failure_context = ""
        pending_payload = self.memory_manager.pending_payload or {}
        if isinstance(pending_payload, dict):
            failure_context = pending_payload.get("failure_context", "")

        THEORETICAL_Q_RE = re.compile(
            r"^(what'?s?|how|why|explain|what is|what are|compare|"
            r"which is|would you|should i|is it|does it|can it|"
            r"remind me|remind|schedule|take a note)\b",
            re.IGNORECASE
        )

        if THEORETICAL_Q_RE.match(user_input.strip()):
            print(f"[VACheck] skipped — theoretical/action question")
            return None

        awareness = score_vision_need(
            user_text=user_input,
            last_topic=self.memory_manager.get_last_topic() or "",
            pending_state=self.memory_manager.get_pending_topic() or "",
            recent_turns=recent,
            failure_context=failure_context,
        )

        print(f"[VisionAwareness] score={awareness.score:.2f} "
            f"decision={awareness.decision.value}")

        if awareness.decision == VisionDecision.SILENT:
            path = capture_single_frame(out_dir=WEBCAM_SAVE_DIR)
            if not path:
                # Webcam unavailable — degrade to ASK
                offer = get_camera_offer_text(self.memory_manager.get_last_topic() or "")
                self.memory_manager.set_pending_topic("Awaiting_VisionAwareness_Confirm")
                self.memory_manager.pending_payload = {
                    "pending_vision_question": user_input,
                    "context_hint": awareness.context_hint,
                    "original_intent": intent,
                }
                return offer

            recognition = recognize_face(path)
            authorized = (recognition.status == RecognitionStatus.AUTHORIZED)

            vision_result = run_vision_pipeline(
                image_path=path,
                user_question=user_input,
                conversation_manager=self,
                authorized_user=authorized,
                is_webcam=True,
                context_hint=awareness.context_hint,
            )

            if not vision_result:
                return None

            prefix = get_silent_prefix()
            llm_reply = self._llm(user_input, self.memory_manager.get_last_topic())
            combined = f"{prefix}{vision_result}\n\n{llm_reply}".strip()

            topic_key = f"vision_awareness_{int(time.time())}"
            store_short_term_memory(user_input, combined, topic=topic_key)
            self.memory_manager.set_last_topic(topic_key)
            return combined

        if awareness.decision == VisionDecision.ASK:

            offer = get_camera_offer_text(self.memory_manager.get_last_topic() or "")

            self.memory_manager.set_pending_topic("Awaiting_VisionAwareness_Confirm")
            self.memory_manager.pending_payload = {
                "pending_vision_question": user_input,
                "context_hint": awareness.context_hint,
                "original_intent": intent,
            }
            return offer

        return None

    def _resolve_return_to_topic(self, user_input: str) -> str | None:
        text = (user_input or "").lower()

        RETURN_SIGNALS = [
            "back to", "return to", "go back to", "discuss back",
            "back again", "continue with", "resume", "back regarding",
            "i want to discuss back", "let's go back",
            "back to our", "back to the", "return to the",
        ]
        if not any(s in text for s in RETURN_SIGNALS):
            return None

        for topic, exchanges in temp_topic_memory.items():
            if not re.match(r"^topic_\d+$", topic):
                continue
            context = " ".join(exchanges)
            # check if q-learning or reinforcement words appear in the content
            input_words = set(re.findall(r"[a-z]{4,}", user_input.lower()))
            context_words = set(re.findall(r"[a-z]{4,}", context.lower()))
            if len(input_words & context_words) >= 2:
                return topic

        existing_topics = list(temp_topic_memory.keys())
        if not existing_topics:
            return None

        best = None
        best_score = 0

        for topic in existing_topics:
            if topic.startswith("session:") or topic.startswith("web:"):
                continue
            if topic == "current_conversation":
                continue

            topic_words = set(re.findall(r"[a-z]{3,}", topic.lower()))
            input_words = set(re.findall(r"[a-z]{3,}", text))
            overlap = len(topic_words & input_words)

            # fuzzy fallback for STT corruptions
            if overlap == 0:
                try:
                    from fuzzywuzzy import fuzz
                    score = fuzz.partial_ratio(topic.lower(), text) / 100.0
                    if score >= 0.7:
                        overlap = 1
                except ImportError:
                    pass

            if overlap > best_score:
                best_score = overlap
                best = topic

        return best if best_score >= 1 else None

    def _check_proactive_link(self, user_input: str, last_response: str) -> str | None:
        """
        After answering, check if the current topic has semantic overlap
        with any existing topic group that wasn't retrieved.
        """
        current_topic = self.memory_manager.get_last_topic() or ""

        # Only fire when a new topic just started
        if not current_topic or current_topic.startswith(("web:", "session:", "vision")):
            return None

        # Score all OTHER groups against the current input
        suggestions = []
        for topic, exchanges in temp_topic_memory.items():
            if topic == current_topic:
                continue
            if topic.startswith(("web:", "session:", "vision")):
                continue

            context = " ".join(exchanges)
            is_related = compute_similarity_hybrid(user_input, topic, context)
            if is_related:
                suggestions.append(topic)

        if not suggestions:
            return None

        # Build a natural suggestion
        topic_list = " and ".join(suggestions[:2])
        return f"By the way, this connects to what we discussed about {topic_list}. Want me to link them?"

    def _should_check_episodic_memory(self, user_input: str, intent: str, last_topic: str) -> bool:
        """
        Decide whether to run EpisodicMemoryRouter before MemoryTaskFusion.
        This is only a cost-saving gate, not the final decision.
        """
        text = (user_input or "").lower().strip()
        intent = (intent or "").strip()
        last_topic = (last_topic or "").strip()

        memory_markers = [
            "before",
            "earlier",
            "again",
            "previous",
            "last time",
            "what did we",
            "what was",
            "where should",
            "where did",
            "how did we",
            "remind me what",
            "remind me where",
            "remind me how",
            "continue",
            "same",
            "repeat",
            "from before",
            "from earlier",
        ]

        if any(marker in text for marker in memory_markers):
            return True

        if intent in {"chat", "system_query", "recall_memory"}:
            return True

        if last_topic:
            return True

        return False

    # update for device control
    def _extract_rover_sequence_from_memory_route(self, memory_route) -> list[str]:
        """
        Extract normalized rover commands from recalled episodic memory.

        Input memory may contain:
            User: go forward. Assistant: Moving forward.
            User: turn left. Assistant: Turning left.
            User: stop. Assistant: Stopping rover.

        Output:
            ["go forward", "turn left", "stop"]
        """
        if not memory_route or not getattr(memory_route, "selected_episodes", None):
            return []

        raw_blocks = []

        for match in memory_route.selected_episodes:
            ep = getattr(match, "episode", None)
            if not ep:
                continue

            raw = getattr(ep, "raw_text", "") or ""
            summary = getattr(ep, "summary", "") or ""

            raw_blocks.append(raw)
            raw_blocks.append(summary)

        joined = "\n".join(raw_blocks).lower()

        command_variants = [
            ("go forward", [
                "go forward",
                "move forward",
                "forward",
                "moving forward",
            ]),
            ("go backward", [
                "go backward",
                "move backward",
                "backward",
                "reverse",
                "moving backward",
            ]),
            ("turn left", [
                "turn left",
                "left",
                "turning left",
            ]),
            ("turn right", [
                "turn right",
                "right",
                "turning right",
            ]),
            ("faster", [
                "faster",
                "increase speed",
                "speed up",
                "increasing rover speed",
            ]),
            ("slower", [
                "slower",
                "decrease speed",
                "slow down",
                "reducing rover speed",
            ]),
            ("stop", [
                "stop",
                "halt",
                "stopping",
                "stopping rover",
            ]),
        ]

        sequence = []

        # Prefer scanning line-by-line to preserve command order.
        lines = [line.strip() for line in joined.splitlines() if line.strip()]

        for line in lines:
            # Prefer user command lines, but allow assistant lines as fallback.
            for canonical, variants in command_variants:
                if any(v in line for v in variants):
                    if not sequence or sequence[-1] != canonical:
                        sequence.append(canonical)
                    break

        # Fallback: scan whole text by earliest occurrence.
        if not sequence:
            found = []

            for canonical, variants in command_variants:
                positions = [
                    joined.find(v)
                    for v in variants
                    if joined.find(v) != -1
                ]

                if positions:
                    found.append((min(positions), canonical))

            found.sort(key=lambda x: x[0])
            sequence = [cmd for _, cmd in found]

        # Remove duplicate commands while preserving order.
        # If later you need true repeated cycles, remove this dedupe.
        cleaned = []
        for cmd in sequence:
            if cmd not in cleaned:
                cleaned.append(cmd)

        return cleaned

    def _format_rover_sequence(self, sequence: list[str]) -> str:
        if not sequence:
            return "the earlier rover movement sequence"

        return " → ".join(sequence)

    def _format_rover_confirm_question(self, sequence: list[str]) -> str:
        sequence_text = self._format_rover_sequence(sequence)
        return f"I found the earlier rover sequence: {sequence_text}. Should I run it?"

    def _make_rover_sequence_confirmation_response(self, sequence: list[str]) -> str:
        return self._format_rover_confirm_question(sequence)

    def _execute_rover_sequence(self, sequence: list[str]) -> str:
        if not sequence:
            return "I could not recover a valid rover sequence, so I will not execute anything."

        results = []

        for command in sequence:
            try:
                # Try plain command first if your existing rover flow accepts it.
                result = self.device_handler.handle(f"rover {command}")

                result_text = str(result or "").strip()
                if not result_text:
                    result_text = "sent"

                results.append(f"{command}: {result_text}")

            except Exception as e:
                results.append(f"{command}: failed ({e})")
                break

        return (
            "Rover sequence completed:\n"
            + "\n".join(f"- {item}" for item in results)
        )

    def _handle_rover_sequence_confirmation(self, user_input: str) -> str:
        payload = getattr(self.memory_manager, "pending_payload", None) or {}
        sequence = payload.get("sequence", [])
        topic = payload.get("topic", "rover:session")

        def _on_confirm():
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None
            self.memory_manager.set_last_topic(topic)
            return self._execute_rover_sequence(sequence)
        def _on_decline():
            self.memory_manager.set_pending_topic(None)
            self.memory_manager.pending_payload = None
            self.memory_manager.set_last_topic(topic)
            return "Got it, I just cancelled. I will not run the rover sequence"

        sequence_text = self._format_rover_sequence(sequence)
        question_context = self._format_rover_confirm_question(sequence)

        return self.handle_confirm_state(
            user_input,
            question_context=question_context,
            on_confirm=_on_confirm,
            on_decline=_on_decline,
            state_name="Awaiting_Rover_Sequence_Confirmation",
        )

    def handle_request(self, user_input: str) -> str:
        # ── Deferred Friday notifications — speak when user returns ───────────
        if self._pending_friday_notifications:
            notifications = self._pending_friday_notifications.copy()
            self._pending_friday_notifications.clear()
            for notif in notifications:
                mins_ago = max(1, int((time.time() - notif["finished_at"]) / 60))
                time_str = f"{mins_ago} minute{'s' if mins_ago != 1 else ''} ago"
                self.tts.speak(
                    f"Welcome back. While you were away, Friday finished {time_str}. "
                    f"{notif['msg']}"
                )
                time.sleep(0.3)

        pending = self.memory_manager.get_pending_topic()
        print(f"[ENTRY] input={user_input[:40]!r} pending={pending!r}")

        belief = self.belief.update(user_input, self.memory_manager)
        print(f"[BELIEF] domain={belief.domain.value} phase={belief.phase.value} "
            f"conf={belief.confidence:.2f} webcam_failed={belief.webcam_failed()}")

        pending_before = self.memory_manager.get_pending_topic()
        user_input = (user_input or "").strip()
        if not user_input:
            return ""

        # ── Shadow-mode intent classifier — LOGGED ONLY, never acted on ───────────
        # Runs before any layer touches state, so it sees the same pre-turn context it
        # was trained on, and covers turns that the interrupt/FSM layers short-circuit.
        shadow_pred = None
        try:
            _prev = getattr(self, "_shadow_prev_turn", None)
            shadow_pred = get_shadow().predict(
                user_input,
                last_topic=self.memory_manager.get_last_topic(),
                pending_before=pending_before,
                prev_input=_prev[0] if _prev else None,
                prev_response=_prev[1] if _prev else None,
            )
        except Exception:
            shadow_pred = None

        def _respond(response: str, intent_res: dict = None) -> str:
            response = sanitize_for_tts(response)
            try:
                session_logger.log(
                    user_input=user_input,
                    response=response,
                    intent_result=intent_res or {"intent": "unknown",
                                                "confidence": 0.0,
                                                "source": "pre_classify"},
                    context={
                        "last_topic":     self.memory_manager.get_last_topic(),
                        "pending_before": pending_before,
                        "pending_after":  self.memory_manager.get_pending_topic(),
                    },
                    outcome={
                        "follow_up_needed": self.memory_manager.get_pending_topic() is not None,
                    },
                    shadow=shadow_pred,
                )
            except Exception:
                pass
            # previous turn for the next shadow prediction (same text the logger stores)
            self._shadow_prev_turn = (user_input, (response or "").strip())
            note_last_reply(response)  # context guard 3: next turn needs to know if a question was open
            return response

        # Safety Purposes
        STOP_ROVER_PHRASES = re.compile(
            r"\b(stop|halt|freeze)\b.{0,15}\b(rover)\b|\b(rover)\b.{0,15}\b(stop|halt|freeze)\b",
            re.IGNORECASE
        )
        if self.device_handler.rover_manager.session_active and STOP_ROVER_PHRASES.search(user_input):
            return _respond(self.device_handler.rover_manager.emergency_stop_and_report(), None)

        # ── Layer 1: Interrupt / Cancel ───────────────────────────────────────
        if is_interrupt(user_input):
            self.memory_manager.clear_all_flows()
            return _respond("Got it — cancelled. What would you like to do?")

        if is_resume(user_input):
            if self.memory_manager.has_paused_flow():
                snap = self.memory_manager.pop_flow()
                return _respond(self.build_resume_prompt(snap))
            return _respond("Nothing paused right now.")

        # ── Layer 2: FSM Pending State ─────────────────────────────────────────
        pending = self.memory_manager.get_pending_topic()
        if pending:
            result = self.handle_pending(pending, user_input)
            if result is not None:
                return _respond(result)

        # ── Layer 2.5: Stuck State Detection ──────────────────────────────────
        stuck_count = getattr(self.memory_manager, '_stuck_count', 0)
        if pending and pending == getattr(self.memory_manager, '_last_pending', None):
            stuck_count += 1
        else:
            stuck_count = 0
        self.memory_manager._stuck_count = stuck_count
        self.memory_manager._last_pending = pending

        if stuck_count >= 2:
            self.memory_manager.clear_all_flows()
            self.memory_manager._stuck_count = 0
            return _respond("I'm having trouble with that. Let's start over — what do you need?")

        # ── Layer 3: Music Detection ───────────────────────────────────────────
        music = detect_music_intent(user_input)
        if music:
            intent, params = music
            return _respond(handle_music_command(intent, params, self.spotify))

        stressed_test = detect_stressed_test_trigger(user_input)
        if stressed_test:
            project_key, scope = stressed_test
            return _respond(self.project_handler.handle_stressed_test(project_key, scope))

        if (self.friday.is_warm()
                and last_topic.startswith("friday:")
                and intent == "chat"):
            return _respond(self.friday.ask(user_input))

        # ── Layer 4: Project Trigger (guarded — won't hijack memory operations)
        proj_key = self.project_handler.detect_project_trigger(user_input)
        if proj_key:
            text = user_input.lower()
            MEMORY_ACTION_SIGNALS = [
                "delete", "remove", "erase", "forget", "wipe",
                "save", "store", "remember",
                "recall", "what did we",
            ]
            if not any(s in text for s in MEMORY_ACTION_SIGNALS):
                return _respond(self.project_handler.handle_open_project(proj_key))

        # Layer 4.5
        if self.memory_manager.get_pending_topic() is None:
            routed = route_via_llm(user_input, self.memory_manager, self.belief.state,
                                    self.episodic_router, tello_client=self.tello_client,
                                    rover_client=self.rover_client)
            if routed and routed.get("tool_name"):
                decision = filter_tool_call(routed["tool_name"], routed["tool_args"])
                if decision == "allow":
                    dispatch = make_dispatch_table(self)
                    fn = dispatch.get(routed["tool_name"])
                    if fn:
                        result = fn(routed["tool_args"])
                        if result:
                            return _respond(result, routed)

                elif decision == "confirm":
                    self.memory_manager.set_pending_topic("Awaiting_LLM_Tool_Confirm")
                    self.memory_manager.pending_payload = {"tool_name": routed["tool_name"], "tool_args": routed["tool_args"]}
                    question = format_tool_confirm_question(routed["tool_name"])
                    return _respond(question, routed)

        # ── Layer 5: Intent Classification + Guards ────────────────────────────
        raw_intent = classify_intent(user_input, self.memory_manager,
                            conversation_manager=self)
        intent_result = apply_guards(raw_intent, user_input, self.memory_manager)

        if intent_result.get("source") == "rover_save_target":
            return _respond(self.device_handler.rover_manager.save_last_log(), intent_result)

        if intent_result.get("source") == "rover_save_ambiguous":
            self.memory_manager.set_pending_topic("Awaiting_SaveTarget_Confirm")
            self.memory_manager.pending_payload = {"raw_text": user_input}
            return _respond("Should I save the rover telemetry log, or our conversation?", intent_result)

        # NOTE: turns with source == "tool_router_fallback_chat" (no rule/tier2/tool-router
        # match) deliberately keep falling through as plain "chat". A blanket "I'm not sure
        # what you'd like me to do" here was tried and reverted: in the logged data that
        # guess was RIGHT for 370 of 477 turns (78%), and the clarifier dropped the alive
        # stress test from ~15/19 to 8/19. Revisit only with the trained intent model
        # deciding per turn (ask only when it is NOT confident the turn is chat).

        # ── Fusion gate — wires all signals together for device_control ───────
        if intent_result["intent"] == "device_control":
            intent_result = self._device_intent_fusion(
                user_input,
                intent_result,
                self.belief.state,
            )

        _log_uncertain_intent(user_input, intent_result)
        intent = intent_result["intent"]
        print(f"[INTENT] {intent} conf={intent_result['confidence']:.2f} src={intent_result['source']}")

        # test add fusion logic
        last_topic = self.memory_manager.get_last_topic() or ""

        memory_route = None
        if self._should_check_episodic_memory(user_input, intent, last_topic):
            memory_route = EpisodicMemoryRouter(
                topic_memory=temp_topic_memory,
                similarity_fn=compute_similarity_hybrid,
                topic_extractor=extract_topic_name,
                topic_cleaner=self._clean_extracted_topic,
            ).route(
                user_input=user_input,
                last_topic=last_topic,
            )

        fusion = self.memory_task_fusion.decide(
            user_input=user_input,
            intent_result=intent_result,
            memory_route=memory_route,
            last_topic=last_topic,
            belief=self.belief.state,
        )

        print(
            f"[MEM_TASK_FUSION] kind={fusion.kind!r} "
            f"intent={fusion.final_intent!r} topic={fusion.topic!r} "
            f"use_memory={fusion.use_memory} run_task={fusion.run_task} "
            f"confirm={fusion.require_confirmation} reason={fusion.reason}"
        )

        # ── Layer 5.5: Memory/Task Fusion Execution ───────────────────────────
        if fusion.kind == "episodic_chat":
            self.memory_manager.set_last_topic(fusion.topic)

            short_term_str = ""
            if fusion.memory_route:
                short_term_str = self.build_context_from_memory_route(fusion.memory_route)

            response = self.rag.answer(
                user_input=user_input,
                ctx=self._make_ctx(
                    topic=fusion.topic,
                    short_term=short_term_str,
                    mode="Episodic Memory Recall",
                ),
            )

            store_short_term_memory(user_input, response, topic=fusion.topic)
            return _respond(response, intent_result)

        if fusion.kind == "task_with_memory_confirm":
            self.memory_manager.set_last_topic(fusion.topic)

            short_term_str = ""
            if fusion.memory_route:
                short_term_str = self.build_context_from_memory_route(fusion.memory_route)

            if fusion.topic.startswith("rover:"):
                rover_sequence = self._extract_rover_sequence_from_memory_route(
                    fusion.memory_route
                )

                self.memory_manager.set_pending_topic("Awaiting_Rover_Sequence_Confirmation")
                self.memory_manager.pending_payload = {
                    "topic": fusion.topic,
                    "sequence": rover_sequence,
                    "source": "episodic_memory",
                    "raw_user_input": user_input,
                    "short_term": short_term_str,
                }

                response = self._make_rover_sequence_confirmation_response(rover_sequence)

                store_short_term_memory(user_input, response, topic=fusion.topic)
                return _respond(response, intent_result)

            # Generic non-rover fallback.
            response = self.rag.answer(
                user_input=(
                    user_input
                    + "\n\nImportant: The user is referring to a previous physical/device action. "
                    "Do not execute anything directly. Summarize the remembered action and ask for confirmation."
                ),
                ctx=self._make_ctx(
                    topic=fusion.topic,
                    short_term=short_term_str,
                    mode="Task With Episodic Memory Confirmation",
                ),
            )

            store_short_term_memory(user_input, response, topic=fusion.topic)
            return _respond(response, intent_result)

        # For normal task execution, let fusion override the final intent.
        intent = fusion.final_intent

        vision_aware_reply = self._vision_awareness_check(user_input, intent)
        if vision_aware_reply is not None:
            return _respond(vision_aware_reply, intent_result)

        def _handle_job_watchdog_intent(conversation_manager, user_input: str) -> str:
            """
            Routes job watchdog voice commands.
            Paste the body of this function inside handle_request() under:
                if intent == "job_watchdog":
            """
            text = (user_input or "").lower()

            # User wants to read the existing recap
            READ_SIGNALS = [
                "show me", "read", "what jobs", "tell me", "recap",
                "list", "open", "display", "what came up",
            ]
            if any(s in text for s in READ_SIGNALS):
                recap_text = conversation_manager.job_watchdog.read_latest_recap()

                # If it's a full markdown file, give a spoken summary + offer to open the file
                if recap_text.startswith("#"):
                    from brain.friday.job_watchdog import _parse_recap_for_tts
                    path = conversation_manager.job_watchdog.get_latest_recap_path()
                    spoken = _parse_recap_for_tts(path) if path else "No recap found."
                    if path:
                        # Open the .md file so user can read it
                        import os, subprocess, sys
                        try:
                            if os.name == "nt":
                                os.startfile(str(path))
                            elif sys.platform == "darwin":
                                subprocess.Popen(["open", str(path)])
                            else:
                                subprocess.Popen(["xdg-open", str(path)])
                            return f"{spoken} I also opened the full recap file for you."
                        except Exception:
                            pass
                    return spoken
                return recap_text

            # User wants to trigger a new scan right now
            SCAN_SIGNALS = [
                "scan", "check", "search", "find", "scan for", "new scan",
                "run watchdog", "friday scan",
            ]
            if any(s in text for s in SCAN_SIGNALS):
                return conversation_manager.job_watchdog.trigger_now()

            # Ambiguous — offer both options
            return (
                "I can either scan for new jobs now, or read you the latest recap. "
                "Which would you like?"
            )

        if intent == "job_watchdog":
            return _respond(
                _handle_job_watchdog_intent(self, user_input),
                intent_result
            )

        # ── Friday Task Dispatch ───────────────────────────────────────────────
        if intent == "friday_task":
            task = extract_friday_task(user_input)
            risk, reason = score_friday_task_risk(task)
            print(f"[FRIDAY] task={task!r} risk={risk:.2f} reason={reason}")

            if risk >= 0.7:
                self.memory_manager.set_pending_topic("Awaiting_Friday_Confirm")
                self.memory_manager.pending_payload = {"task": task, "risk": risk, "reason": reason}
                return _respond(format_friday_confirm_question(task, reason), intent_result)

            response = self.friday._dispatch_async(task, session_label="buddy_voice")
            return _respond(response, intent_result)


        # ── Layer 6: Route to Skill ────────────────────────────────────────────
        if intent == "web_search":
            return _respond(self.handle_internet_request(user_input), intent_result)

        if intent == "web_followup":
            norm = normalize_web_followup(user_input)
            return _respond(self.answer_from_web_followup(norm), intent_result)

        if intent == "schedule_save":
            return _respond(
                self.memory_handler.save_memory(user_input, routed_intent="schedule_save"),
                intent_result
            )

        if intent == "schedule_delete":
            return _respond(
                self.memory_handler.delete_schedule(user_input),
                intent_result
            )

        if intent == "schedule_recall":
            return _respond(
                self.memory_handler.handle_recall(user_input, routed_intent="schedule_recall"),
                intent_result
            )

        if intent == "save_memory":
            result = self.memory_handler.save_memory(user_input, routed_intent="save_memory")

            if result is None:
                last_topic = self.memory_manager.get_last_topic() or ""

                if last_topic and not last_topic.startswith("session:"):
                    response = self.rag.answer(
                        user_input=user_input,
                        ctx=RagContext(
                            topic=last_topic,
                            short_term="\n".join(temp_topic_memory.get(last_topic, [])),
                            mode="Save Memory Fallback",
                        ),
                    )
                    store_short_term_memory(user_input, response, topic=last_topic)
                    return _respond(response, intent_result)

                return _respond(self.start_new_conversation(user_input), intent_result)

            return _respond(result, intent_result)

        if intent == "recall_memory":
            return _respond(
                self.memory_handler.handle_recall(user_input, routed_intent="recall_memory"),
                intent_result
            )

        if intent == "delete_memory":
            return _respond(self.memory_handler.delete_memory(user_input), intent_result)

        if intent == "schedule_delete":
            return _respond(self.memory_handler.delete_schedule(user_input), intent_result)

        if intent == "device_control":
            result = self.device_handler.handle(user_input)
            if any(s in result.lower() for s in ["couldn't reach", "connection failed", "couldn't connect"]):
                current = self.memory_manager.get_last_topic() or ""
                if current.startswith(("rover:", "tello:")):
                    self.memory_manager.set_last_topic(None)
                    print(f"[CLEANUP] cleared {current!r} after failed device call")
            return _respond(result, intent_result)

        if intent == "system_query":
            system_output = self.intent_handler.handle_intent(
                self._resolve_system_query(user_input)
            )
            if system_output:
                return _respond(system_output, intent_result)

        # check vision context continuation
        last_topic = self.memory_manager.get_last_topic() or ""
        if last_topic.startswith("vision_awareness_"):
            return _respond(
                self.continue_conversation(user_input, last_topic),
                intent_result
            )
        # ── END vision context continuation ──────────────────────────────────

        # ── Layer 7: Chat / RAG Fallback ──────────────────────────────────────
        # resolved_topic, short_term_ctx = self._resolve_rag_context(user_input)
        # if resolved_topic and short_term_ctx:
        #     return _respond(
        #         self.continue_conversation(user_input, resolved_topic),
        #         intent_result
        #     )

        #return _respond(self.start_new_conversation(user_input), intent_result)

        # new type
        # ── Layer 7: Chat / RAG Fallback ──────────────────────────────────────────
        # Layer 7 in handle_request — replace the existing block with this:
        last_topic = self.memory_manager.get_last_topic() or ""

        # ── Phase-aware return resolution ─────────────────────────────────────
        # Must run BEFORE continue_conversation so a stale last_topic
        # doesn't hijack an explicit "back to X" request
        if self.belief.state.phase.value == "returning":
            existing = self._resolve_return_to_topic(user_input)
            if existing:
                self.memory_manager.set_last_topic(existing)
                return _respond(
                    self.continue_conversation(user_input, existing),
                    intent_result
                )

        if last_topic and not last_topic.startswith("session:"):
            return _respond(
                self.continue_conversation(user_input, last_topic),
                intent_result
            )

        best_seed = _find_best_session_seed(user_input, self.memory_manager)
        if best_seed:
            return _respond(
                self.continue_conversation(user_input, best_seed),
                intent_result
            )

        existing = self._resolve_return_to_topic(user_input)
        if existing:
            self.memory_manager.set_last_topic(existing)
            return _respond(
                self.continue_conversation(user_input, existing),
                intent_result
            )

        response = _respond(self.start_new_conversation(user_input), intent_result)

        proactive = self._check_proactive_link(user_input, response)
        if proactive:
            response = response + "\n\n" + proactive

        return response

    def _resolve_system_query(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["time", "clock"]): return "time"
        if any(w in t for w in ["date", "today", "day"]): return "date"
        if any(w in t for w in ["battery", "charging"]): return "battery"
        if any(w in t for w in ["weather", "forecast", "rain"]): return "weather"
        if any(w in t for w in ["location", "where am i"]): return "location"
        if any(w in t for w in ["cpu", "processor"]): return "cpu"
        return "chat"


    def build_context_from_memory_route(self, route: MemoryRoute, max_chars: int = 1800) -> str:
        """
        Converts selected episodic memories into a prompt-ready context block.
        This is intentionally separate from the router.
        """
        if not route.selected_episodes:
            return ""

        blocks = []

        for idx, match in enumerate(route.selected_episodes, start=1):
            ep = match.episode

            if route.kind == "continue_current":
                role = "CURRENT TOPIC MEMORY"
            elif route.kind == "episodic_recall":
                role = "RECALLED EPISODIC MEMORY"
            else:
                role = "RELATED MEMORY"

            text = ep.raw_text.strip()

            if len(text) > max_chars:
                text = "..." + text[-max_chars:]

            block = f"""
    [{role} #{idx}]
    Topic: {ep.topic}
    Episode ID: {ep.episode_id}
    Similarity: {match.score:.2f}
    Summary: {ep.summary}
    Keywords: {", ".join(ep.keywords)}

    {text}
    """.strip()

            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    def _register_topic_in_graph(self, child: str, parent: Optional[str]):
        """
        Records that `child` topic was born while `parent` was active.
        Called from start_new_conversation.
        """
        if not hasattr(self.memory_manager, "_topic_graph"):
            self.memory_manager._topic_graph = {}

        graph = self.memory_manager._topic_graph
        child = (child or "").strip().lower()
        parent = (parent or "").strip().lower() if parent else None

        if not child:
            return

        # Skip session/web/vision/rover as parents — they're not real topics
        if parent and parent.startswith(("session:", "web:", "vision", "rover:")):
            parent = None

        # Add child node
        if child not in graph:
            graph[child] = {
                "parent": parent,
                "children": [],
                "created_at": time.time(),
            }
        elif parent and not graph[child].get("parent"):
            # Backfill parent if we now know it
            graph[child]["parent"] = parent

        # Add to parent's children list
        if parent:
            if parent not in graph:
                graph[parent] = {
                    "parent": None,
                    "children": [],
                    "created_at": time.time(),
                }
            if child not in graph[parent]["children"]:
                graph[parent]["children"].append(child)
                print(f"[GRAPH] {child!r} ← child of ← {parent!r}")

    def continue_conversation(self, user_input, last_topic):
        last_topic = (last_topic or "").strip().lower()
        raw_input = (user_input or "").strip()
        print("[DEBUG] Passing continue_conversation")

        # ---- ROVER TOPIC FLOW ----
        # When rover:session is the active topic, follow-up utterances like
        # "go faster", "turn left", "stop" arrive here via the chat/fallback
        # path in handle_request. We must re-route them to device_handler
        # instead of letting them fall into general RAG.
        #
        # Topic switch detection: if the user says something that is clearly
        # not a rover command (e.g. "what's the weather"), _belongs_to_rover_context
        # inside RoverTaskManager will return handled=False, which causes
        # _route_rover in DeviceHandlerFSM to call _close_session and return
        # handled=False — DeviceHandlerFSM then clears the FSM state and
        # ConversationManager falls through to start_new_conversation naturally.
        if last_topic.startswith("rover:"):
            print(f"[DEBUG] ROVER topic flow → re-routing to device_handler")
            return self.device_handler.handle(raw_input)

        # ---- WEB TOPIC FLOW ----
        if last_topic.startswith("web:"):
            wctx = self._get_web_ctx(last_topic)
            web_block = self._format_web_ctx_for_prompt(wctx, max_items=5)

            short_term_context = temp_topic_memory.get(last_topic, [])
            short_term_str = "\n".join(short_term_context) if short_term_context else ""

            raw_input = (user_input or "").strip()
            norm_input = normalize_web_followup(raw_input)

            parsed_action = parse_web_followup(norm_input)
            bare_idx = parse_index_utterance(norm_input)

            is_web_navigation = parsed_action is not None or bare_idx is not None

            if not is_web_navigation:
                web_query = ""
                if wctx:
                    web_query = (wctx.get("query") or "").strip()

                same_topic = compute_similarity_llm(
                    raw_input,
                    web_query or last_topic,
                    short_term_str + ("\n" + web_block if web_block else "")
                )

                if not same_topic:
                    new_topic = extract_topic_name(raw_input)
                    if new_topic:
                        self.memory_manager.set_last_topic(new_topic)
                        return self.continue_conversation(raw_input, new_topic)
                    return self.start_new_conversation(raw_input)

            mode = "Web Navigation" if is_web_navigation else "Web Discussion"
            effective_input = norm_input if is_web_navigation else raw_input

            print(f"[DEBUG] WEB mode={mode}")
            print(f"[DEBUG] WEB parsed_action={parsed_action}, bare_idx={bare_idx}")
            print(f"[DEBUG] WEB effective_input={effective_input!r}")

            ctx = self._make_ctx(
                topic=last_topic,
                short_term=short_term_str,
                web_block=web_block,
                mode=mode,
            )

            response = self.rag.answer(user_input=effective_input, ctx=ctx)
            store_short_term_memory(raw_input, response, topic=last_topic)
            return response

        # ---- VISION TOPIC FLOW ----
        if last_topic.startswith("vision"):
            short_term_context = temp_topic_memory.get(last_topic, [])
            short_term_str = "\n".join(short_term_context) if short_term_context else ""

            if not short_term_str:
                return self.start_new_conversation(raw_input)

            # ── NEW: failure-state guard ──────────────────────────────────
            VISION_FAILURE_SIGNALS = [
                "color bars", "color-bars", "colour bars",
                "can't see", "cannot see", "no view",
                "not your bench", "just a test pattern",
                "color test pattern", "colour test pattern",
            ]
            vision_failed = any(s in short_term_str.lower()
                                for s in VISION_FAILURE_SIGNALS)

            if vision_failed:
                # deterministic response — no LLM variance possible
                return ("I still can't get a clear view of your setup — "
                        "the camera feed is showing a test pattern instead of your bench. "
                        "Can you check the camera is pointing the right way?")
            # ── END failure guard ─────────────────────────────────────────

            ctx = self._make_ctx(
                topic=last_topic,
                short_term=short_term_str,
                mode="Vision Discussion",
            )

            response = self.rag.answer(user_input=raw_input, ctx=ctx)
            store_short_term_memory(raw_input, response, topic=last_topic)
            return response

        # ---- SESSION SEED FLOW ----
        if last_topic.startswith("session:"):
            short_term_context = temp_topic_memory.get(last_topic, [])
            short_term_str = "\n".join(short_term_context) if short_term_context else ""

            if not short_term_str:
                return self.start_new_conversation(raw_input)

            ctx = self._make_ctx(
                topic=last_topic,
                short_term=short_term_str,
                mode="Session Briefing Discussion",
            )

            response = self.rag.answer(user_input=raw_input, ctx=ctx)

            # Promote to a real topic so future turns don't loop back to session seed
            raw_topic = extract_topic_name(raw_input)
            clean = self._clean_extracted_topic(raw_topic)
            if clean and not clean.startswith("session:"):
                self.memory_manager.set_last_topic(clean)
                store_short_term_memory(raw_input, response, topic=clean)
                print(f"[DEBUG] Promoted session seed → {clean!r}")
            else:
                store_short_term_memory(raw_input, response, topic=last_topic)

            return response

        # ---- NORMAL TOPIC FLOW ----
        # ── STEP 1: Short followup bypass ───────────────────────────────
        # If it's a followup, we DON'T run drift detection at all.
        # Just continue the current topic. This single check fixes
        # "I open to any suggestion", "mean can you elaborate?",
        # "Is it hard to implement from scratch?", etc.
        if is_short_followup(raw_input) and last_topic and last_topic != "current_conversation":
            print(f"[CONT] short followup → staying on {last_topic!r}")
            return self._answer_in_topic(raw_input, last_topic)

        # ── STEP 2: Explicit topic shift ────────────────────────────────
        # User said "let's talk about X" or "different question" —
        # honor that intent regardless of similarity scores.
        if is_explicit_topic_shift(raw_input):
            print(f"[CONT] explicit topic shift detected")
            raw_topic = extract_topic_name(raw_input)
            clean = self._clean_extracted_topic(raw_topic)
            if clean and clean != last_topic:
                print(f"[CONT] explicit shift: {last_topic!r} → {clean!r}")
                self.memory_manager.set_last_topic(clean)
                return self.start_new_conversation(raw_input)
            # User signaled shift but didn't name one — keep current
            # rather than invent a topic_NNNN
            return self._answer_in_topic(raw_input, last_topic)

        # ── STEP 3: No current topic — try to extract one ───────────────
        if not last_topic or last_topic == "current_conversation":
            raw_topic = extract_topic_name(raw_input)
            clean = self._clean_extracted_topic(raw_topic)
            if clean:
                print(f"[CONT] new topic from empty state: {clean!r}")
                self.memory_manager.set_last_topic(clean)
                return self.start_new_conversation(raw_input)
            # No extractable topic — answer without setting a placeholder
            return self._answer_topicless(raw_input)

        # ── STEP 4: Drift detection — but against EXCHANGES, not labels ─
        # The old code compared raw_input cosine to last_topic (a 2-word
        # string). That always scored low for follow-ups. We now compare
        # against the actual recent conversation.
        recent_exchanges = temp_topic_memory.get(last_topic, [])
        recent_context = "\n".join(recent_exchanges[-3:]) if recent_exchanges else ""

        # additional
        # Run the assembler against ALL topic memory
        assembler = ContextAssembler(temp_topic_memory)
        assembled = assembler.build(user_input=raw_input, current_topic=last_topic)

        # If the current topic shows up in the assembler's results with decent score,
        # stay on it — even if cosine vs recent_context is low.
        current_in_assembled = any(g.is_current for g in assembled.groups)

        if current_in_assembled and not recent_context:
            # No recent history but the assembler still relates this to current topic
            return self._answer_in_topic(raw_input, last_topic)

        if recent_context:
            is_continuation = compute_similarity_hybrid(raw_input, last_topic, recent_context)

            # Override: if assembler thinks current topic is highly relevant,
            # treat as continuation even if cosine missed
            if not is_continuation and current_in_assembled:
                primary = next((g for g in assembled.groups if g.is_current), None)
                if primary and primary.score >= 0.50:
                    print(f"[CONT] cosine said drift but assembler kept current (score={primary.score:.2f})")
                    is_continuation = True

            if is_continuation:
                return self._answer_in_topic(raw_input, last_topic)
        ####


        if not recent_context:
            # No history yet — assume continuation
            print(f"[CONT] no exchange history, staying on {last_topic!r}")
            return self._answer_in_topic(raw_input, last_topic)

        is_continuation = compute_similarity_hybrid(
            raw_input,
            last_topic,
            recent_context,
        )

        if is_continuation:
            print(f"[CONT] similarity says continuation, staying on {last_topic!r}")
            return self._answer_in_topic(raw_input, last_topic)

        # ── STEP 5: Similarity says drift — but be conservative ─────────
        # Only drift if we can extract a REAL topic. Otherwise keep current.
        # This is what stops topic_NNNN from being created.
        raw_topic = extract_topic_name(raw_input)
        clean = self._clean_extracted_topic(raw_topic)

        if not clean:
            # Similarity flagged drift, but we can't name what we drifted to.
            # That means the utterance is conversational, not a new topic.
            # Stay put.
            print(f"[CONT] drift signal but no extractable topic, staying on {last_topic!r}")
            return self._answer_in_topic(raw_input, last_topic)

        if clean == last_topic:
            print(f"[CONT] extracted topic matches current, staying")
            return self._answer_in_topic(raw_input, last_topic)

        # ── STEP 6: Check episodic memory for an existing match ─────────
        # Before creating a new topic, see if this is a return to an
        # older topic we already have.
        router = EpisodicMemoryRouter(
            topic_memory=temp_topic_memory,
            current_threshold=0.38,
            recall_threshold=0.34,
            max_episode_matches=5,
            similarity_fn=compute_similarity_hybrid,
            topic_extractor=extract_topic_name,
            topic_cleaner=self._clean_extracted_topic,
        )
        route = router.route(user_input=raw_input, last_topic=last_topic)
        print(f"[CONT] router: kind={route.kind!r} topic={route.topic!r} score={route.score:.2f}")

        if route.kind == "episodic_recall" and route.topic and not re.match(r"^topic_\d+$", route.topic):
            print(f"[CONT] episodic recall: → {route.topic!r}")
            self.memory_manager.set_last_topic(route.topic)
            short_term_str = self.build_context_from_memory_route(route)
            response = self.rag.answer(
                user_input=raw_input,
                ctx=self._make_ctx(
                    topic=route.topic,
                    short_term=short_term_str,
                    mode="Episodic Memory Recall",
                ),
            )
            store_short_term_memory(raw_input, response, topic=route.topic)
            return response

        # ── STEP 7: Genuine new topic ───────────────────────────────────
        print(f"[CONT] genuine drift: {last_topic!r} → {clean!r}")
        self.memory_manager.set_last_topic(clean)
        return self.start_new_conversation(raw_input, prefetched_topic=clean)


    def _answer_in_topic(self, user_input: str, topic: str) -> str:
        # ── Layer 1: Graph-based context (explicit parent-child links) ──────
        # This is high-confidence but only works for topics already in the graph
        router = EpisodicMemoryRouter(
            topic_memory=temp_topic_memory,
            topic_graph=getattr(self.memory_manager, "_topic_graph", {}),
            similarity_fn=compute_similarity_hybrid,
            topic_extractor=extract_topic_name,
            topic_cleaner=self._clean_extracted_topic,
        )
        related = router.get_related_context(topic, max_episodes_per_topic=3)

        # ── Layer 2: Semantic context (catches what the graph missed) ───────
        # This catches topics that ARE related but weren't explicitly linked
        # in the graph — e.g. "3d printer" semantically connects to "robotic arm"
        # even if no parent-child relationship was recorded
        assembler = ContextAssembler(temp_topic_memory)
        assembled = assembler.build(
            user_input=user_input,
            current_topic=topic,
        )

        # ── Merge both layers, deduplicating by topic ───────────────────────
        blocks = []
        seen_topics = set()

        # Graph layer first (most explicit relationships)
        if related["current"]:
            blocks.append(
                f"[CURRENT DISCUSSION: {topic}]\n"
                + "\n".join(ep.raw_text for ep in related["current"])
            )
            seen_topics.add(topic.lower())

        if related["parent"]:
            parent_name = related["parent"][0].topic
            if parent_name.lower() not in seen_topics:
                blocks.append(
                    f"[PARENT TOPIC: {parent_name}]\n"
                    + "\n".join(ep.raw_text for ep in related["parent"])
                )
                seen_topics.add(parent_name.lower())

        if related["siblings"]:
            sibling_blocks = []
            for ep in related["siblings"]:
                if ep.topic.lower() not in seen_topics:
                    sibling_blocks.append(f"({ep.topic}): {ep.raw_text}")
                    seen_topics.add(ep.topic.lower())
            if sibling_blocks:
                blocks.append("[RELATED SUBTOPICS]\n" + "\n\n".join(sibling_blocks))

        # Semantic layer second (broader net for topics the graph didn't catch)
        semantic_extras = []
        for group in assembled.groups:
            if group.is_current:
                continue  # already covered by graph's "current"
            if group.topic.lower() in seen_topics:
                continue  # already covered by graph's parent/siblings
            semantic_extras.append(
                f"({group.topic} | similarity {group.score:.2f}): {group.text}"
            )
            seen_topics.add(group.topic.lower())

        if semantic_extras:
            blocks.append(
                "[SEMANTICALLY RELATED TOPICS]\n"
                "(These are topics from earlier in our conversation that "
                "may be relevant to the current discussion)\n"
                + "\n\n".join(semantic_extras)
            )

        short_term_str = "\n\n---\n\n".join(blocks)

        # ── Optional: log what got assembled for debugging ──────────────────
        if blocks:
            print(f"[CONTEXT] assembled {len(blocks)} block(s) for topic={topic!r}")
            print(f"[CONTEXT] topics included: {sorted(seen_topics)}")

        response = self.rag.answer(
            user_input=user_input,
            ctx=self._make_ctx(
                topic=topic,
                short_term=short_term_str,
                mode="General Conversation",
            ),
        )
        store_short_term_memory(user_input, response, topic=topic)
        return response


    def _answer_topicless(self, user_input: str) -> str:
        """
        Answer when there's no topic yet AND we couldn't extract one.
        Critically: we DON'T create a topic_NNNN here. We just answer
        and let the next turn establish a topic when possible.
        """
        response = self.rag.answer(
            user_input=user_input,
            ctx=RagContext(
                topic="",
                short_term="",
                mode="General Conversation",
            ),
        )
        # Store under a synthetic key that NEVER becomes last_topic
        # This is internal-only — the user-visible last_topic stays unset
        # until a real topic emerges.
        synthetic_key = f"_pending_{int(time.time())}"
        store_short_term_memory(user_input, response, topic=synthetic_key)
        # Do NOT call self.memory_manager.set_last_topic(synthetic_key)
        return response

    def start_new_conversation(self, user_input, prefetched_topic: str | None = None):
        print(f"[NEW_CONV] starting (prefetched={prefetched_topic!r})")
        previous_topic = self.memory_manager.get_last_topic()

        if prefetched_topic:
            clean = prefetched_topic   # trust the caller
        else:
            raw_topic = extract_topic_name(user_input)
            clean = self._clean_extracted_topic(raw_topic)

            if not clean:
                existing = self._resolve_return_to_topic(user_input)
                if existing and not re.match(r"^topic_\d+$", existing):
                    clean = existing

        self.memory_manager._pre_switch_sensory = list(sensory_memory)

        if clean:
            self._register_topic_in_graph(clean, previous_topic)
            response = self._llm(user_input, clean, override_sensory=[])
            self.memory_manager.set_last_topic(clean)
            store_short_term_memory(user_input, response, topic=clean)
            return response

        response = self._llm(user_input, None, override_sensory=[])
        synthetic_key = f"_pending_{int(time.time())}"
        store_short_term_memory(user_input, response, topic=synthetic_key)
        return response

def check_exit_intent(user_input):
    """Check if the user wants to end the conversation."""
    exit_keywords = ["bye", "exit", "goodbye", "see you", "stop", "quit"]

    if any(keyword in user_input.lower() for keyword in exit_keywords):
        return True

    return False

def greeting_for(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h < 12: return "Good morning"
    if 12 <= h < 17: return "Good afternoon"
    if 17 <= h < 22: return "Good evening"
    return "Good night"

def _fmt_c(x: Any) -> str:
    try:
        return f"{float(x):.1f}".rstrip("0").rstrip(".")
    except Exception:
        return "?"

def _safe_get_weather_line_deterministic(weather: Dict[str, Any]) -> str:
    cond = str(weather.get("condition", "Unknown"))
    temp = _fmt_c(weather.get("temp_c", "?"))
    high = _fmt_c(weather.get("high_c", "?"))
    low = _fmt_c(weather.get("low_c", "?"))

    # Match the strict format used by the LLM
    if "?" in (temp, low, high):
        return f"{cond}, {temp}°C now."
    return f"{cond}, {temp}°C now. Range: {low}–{high}°C."


def _safe_headlines_block(headlines: List[str], limit: int = 3) -> str:
    headlines = (headlines or [])[:limit]
    if not headlines:
        return "- No headlines available."
    return "\n".join([f"- {h}" for h in headlines])


def _llm_polish_weather_line(ollama, model: str, weather: Dict[str, Any]) -> Optional[str]:

    cond = str(weather.get("condition", "Unknown")).strip()
    temp = _fmt_c(weather.get("temp_c", "?"))
    high = _fmt_c(weather.get("high_c", "?"))
    low = _fmt_c(weather.get("low_c", "?"))

    # If we don't have solid numbers, don't bother calling LLM.
    if "?" in (temp, low, high):
        return None

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You format ONE weather status line for a voice assistant.\n"
                    "You MUST follow the format exactly.\n"
                    "Output ONE line only. No quotes. No greetings. No extra words.\n"
                    "Use ONLY the provided values.\n"
                    "FORMAT:\n"
                    "<CONDITION>, <TEMP>°C now. Range: <LOW>–<HIGH>°C.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Fill the format using these values (copy exactly):\n"
                    f"CONDITION: {cond}\n"
                    f"TEMP: {temp}\n"
                    f"LOW: {low}\n"
                    f"HIGH: {high}\n"
                ),
            },
        ]

        resp = ollama.chat(model=model, messages=messages)
        out = (resp.get("message", {}) or {}).get("content", "").strip()

        # Hard rejects
        if not out:
            return None
        if "\n" in out:
            return None
        if any(ch in out for ch in ['"', "’", "“", "”", "!", "?", ":"]):
            return None

        # Strict pattern
        pattern = r"^[^,]+,\s*-?\d+(\.\d+)?°C now\. Range:\s*-?\d+(\.\d+)?–-?\d+(\.\d+)?°C\.$"
        if not re.match(pattern, out):
            return None

        # Ensure it used EXACT condition string (prevents the model changing "Clear" -> "Sunny")
        if not out.startswith(cond + ","):
            return None

        return out

    except Exception:
        return None


def _normalize_headline(text: str) -> str:
    """Light cleanup for headline inputs before sending to the LLM."""
    if not text:
        return ""
    t = " ".join(text.split())  # collapse whitespace
    t = re.sub(r"^\s*[-*•]\s+", "", t)
    t = re.sub(r"^\s*\d+[\.\)]\s+", "", t)
    return t.strip()


def _normalize_story(text: str) -> str:
    """Cleanup for model output."""
    if not text:
        return ""
    t = text.strip()
    t = t.strip("“”\"'")  # remove surrounding quotes
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = " ".join(t.split())  # force single paragraph
    return t.strip()


def _trim_to_sentence_boundary(text: str, max_chars: int) -> str:
    """
    Trim to <= max_chars, preferring to end at a sentence boundary.
    Falls back to word boundary trimming.
    """
    t = " ".join((text or "").split()).strip()
    if len(t) <= max_chars:
        return t

    window = t[:max_chars].rstrip()

    # Find last sentence terminator within the window.
    # Includes common English terminators; add others if needed.
    ends = [m.end() for m in re.finditer(r"[.!?]", window)]
    if ends:
        cut = ends[-1]
        return window[:cut].strip()

    # Fallback: cut at last whitespace
    if " " in window:
        return window.rsplit(" ", 1)[0].strip()

    return window.strip()


def _looks_truncated(text: str) -> bool:
    """
    Heuristic: reject outputs that look like they trail off.
    """
    if not text:
        return True
    t = text.strip()
    if t.endswith(("…", "...", "…\"", "...\"", "…'", "...'")):
        return True
    # Common trailing fragments
    if re.search(r"\b(which|that|as|while|amid)\s+(will|would|could|may|might|is|are)\s*$", t, re.I):
        return True
    # Ends with conjunction/preposition-ish
    if re.search(r"\b(and|but|or|because|with|to|of|in|on|for)\s*$", t, re.I):
        return True
    return False


def _llm_shorten_headlines(
    ollama,
    model: str,
    headlines: List[str],
    limit: int = 3,
    max_chars: int = 280,
) -> Optional[str]:
    """
    Returns a short, connected, reporter-style news brief (1 paragraph),
    based only on the given headlines.
    """
    try:
        items = [h for h in (headlines or []) if h and h.strip()]
        items = [_normalize_headline(h) for h in items]
        items = [h for h in items if h][:limit]
        if not items:
            return None

        joined = "\n".join(f"- {h}" for h in items)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a straight-news reporter.\n"
                    "Write one connected news brief based ONLY on the provided headlines.\n"
                    "Rules:\n"
                    "- Use ONLY facts present in the headlines; do not add new details.\n"
                    "- Do NOT introduce yourself.\n"
                    "- Do NOT use bullet points or numbered lists.\n"
                    "- Write 1–2 complete sentences (do not trail off).\n"
                    "- Do not end with an ellipsis.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Turn these headlines into one smooth news brief (single paragraph):\n"
                    f"{joined}\n\n"
                    f"Keep it under {max_chars} characters."
                ),
            },
        ]

        resp = ollama.chat(model=model, messages=messages)

        content = ""
        if isinstance(resp, dict):
            content = ((resp.get("message") or {}).get("content") or "").strip()
        else:
            content = getattr(getattr(resp, "message", None), "content", "") or ""

        out = _normalize_story(content)
        if not out:
            return None

        # Reject bullet-y output
        if any(line.lstrip().startswith(("-", "*", "•")) for line in out.splitlines()):
            return None

        # If too long, trim cleanly (NO ellipsis)
        if len(out) > max_chars:
            out = _trim_to_sentence_boundary(out, max_chars)

        # Reject if it still looks like it trails off
        if _looks_truncated(out):
            return None

        return out or None

    except Exception:
        return None

#####################


WEATHER_PREFIXES = [
    "Weather status:",
    "Current atmospheric conditions:",
    "Outside conditions report:",
    "Meteorological update:",
]

NEWS_PREFIXES = [
    "Here are the latest developments:",
    "Today's key updates:",
    "Current headline summary:",
    "Latest intelligence briefing:",
]

CLOSING_LINES = [
    "Standing by.",
    "Awaiting your instruction.",
    "Ready when you are.",
]


NewsInput = Union[str, List[str]]

def _safe_news_block(news: NewsInput, limit: int = 3) -> str:
    if isinstance(news, str):
        story = " ".join(news.split()).strip()
        return story if story else "(No news updates.)"

    # list fallback
    return _safe_headlines_block(news or [], limit=limit)


def build_jarvis_briefing_template(
    startup_data: Dict[str, Any],
    *,
    weather_line: str,
    headlines: NewsInput,  # <-- changed: now accepts str OR list
    personality_level: int = 1,
) -> str:

    dt = datetime.now()
    greet = greeting_for(dt)
    dt_str = dt.strftime("%A, %B %d, %Y — %H:%M")

    news_block = _safe_news_block(headlines, limit=3)

    if personality_level <= 0:
        weather_prefix = "Weather status:"
        news_prefix = "Updated news:"
        closing = "Standing by."
    else:
        weather_prefix = random.choice(WEATHER_PREFIXES)
        news_prefix = random.choice(NEWS_PREFIXES)
        closing = random.choice(CLOSING_LINES)

    return (
        f"{greet} sir. As of {dt_str}.\n"
        f"{weather_prefix} {weather_line}\n"
        f"{news_prefix}\n"
        f"{news_block}\n"
        # f"{closing}"
    )

def introduce_bot_hybrid(
    *,
    startup_data: Dict[str, Any],
    ollama=None,
    model: str = "",
    use_llm: bool = True,
    polish_headlines: bool = True,
    headlines_limit: int = 3,
    debug: bool = False,
) -> str:
    startup_data = startup_data or {}
    weather = startup_data.get("weather") or {}
    headlines = startup_data.get("headlines") or []

    # Clamp early for stability (list fallback still uses this)
    headlines = list(headlines)[:headlines_limit]

    weather_line = _safe_get_weather_line_deterministic(weather)

    used_llm_weather = False
    used_llm_headlines = False  # <-- start False; set True only if we actually use it

    if use_llm and ollama and model:
        polished_weather = _llm_polish_weather_line(ollama, model, weather)
        if polished_weather:
            weather_line = polished_weather
            used_llm_weather = True

        if polish_headlines and headlines:
            story = _llm_shorten_headlines(ollama, model, headlines, limit=headlines_limit)

            # Only overwrite if valid
            if story and story.strip():
                headlines = story
                used_llm_headlines = True

    if debug:
        print(f"[INTRO] LLM weather used={used_llm_weather} | LLM headlines used={used_llm_headlines}")
        if used_llm_headlines:
            print("[INTRO] News mode=storyline")
        else:
            print("[INTRO] News mode=bullets")

    return build_jarvis_briefing_template(
        startup_data,
        weather_line=weather_line,
        headlines=headlines,
    )


def sanitize_tts(text: str) -> str:
    text = (text or "")

    # normalize unicode punctuation -> ascii
    text = (text
        .replace("\u2019", "'")   # ’ -> '
        .replace("\u2018", "'")   # ‘ -> '
        .replace("\u201c", '"')   # “ -> "
        .replace("\u201d", '"')   # ” -> "
        .replace("\u2014", "-")   # — -> -
        .replace("\u2013", "-")   # – -> -
        .replace("\u00a0", " ")   # NBSP -> space
        .replace("\u200b", "")    # zero-width space -> remove
    )

    text = text.strip()

    # remove markdown-ish junk
    text = text.replace("**", "").replace("`", "")
    text = text.replace("•", "-")

    # remove wrapping quotes (common LLM habit)
    text = text.strip('"\'“”')

    # avoid leading punctuation that can make SAPI skip
    text = re.sub(r'^[\s\-\.\,\:\;\!\?\(\)\[\]\{\}]+', '', text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


TEXT_QUEUE = queue.Queue()

def _on_shutdown(sig, frame):
    print()
    recap = SessionRecap()
    recap.run_end_of_session()
    sys.exit(0)

def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# dummy helper -> TODO: NEED TO BE PLACED IN A PROPER PLACE ONE IT IS WORK
# ─────────────────────────────────────────────────────────────────────────────
# SESSION CONTEXT SEEDING
# Called once after full intro speaks. Splits the intro into topic buckets
# and pre-populates temp_topic_memory so the first real turn already has context.
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_KEYS = ["session:weather", "session:news", "session:schedule"]

def _seed_session_context(intro_text: str, memory_manager) -> None:
    if not intro_text:
        return

    lines = [l.strip() for l in intro_text.splitlines() if l.strip()]

    # Strip greeting/header lines — define BEFORE use
    WEATHER_SIGNALS = [
        "°c", "°f", "weather", "temperature", "forecast",
        "rain", "cloudy", "sunny", "clear", "overcast", "wind",
        "atmospheric", "meteorological", "conditions",
    ]
    NEWS_SIGNALS = [
        "headline", "news", "update", "development", "briefing",
        "latest", "reported", "according to", "intelligence",
    ]
    SCHEDULE_SIGNALS = [
        "meeting", "appointment", "reminder", "schedule", "calendar",
        "dentist", "doctor", "deadline", "lecture", "class", "at ",
        "today you have", "don't forget", "coming up",
    ]

    content_lines = [
        l for l in lines
        if not re.match(r"^good (morning|afternoon|evening|night)", l, re.I)
        and not re.match(r"^systems online", l, re.I)
    ]

    weather_lines  = []
    news_lines     = []
    schedule_lines = []

    # ONE loop, uses content_lines
    for line in content_lines:
        ll = line.lower()
        if any(s in ll for s in WEATHER_SIGNALS):
            weather_lines.append(line)
        elif any(s in ll for s in SCHEDULE_SIGNALS):
            schedule_lines.append(line)
        elif any(s in ll for s in NEWS_SIGNALS):
            news_lines.append(line)
        else:
            news_lines.append(line)

    # store and set last_topic — unchanged
    if weather_lines:
        weather_text = " ".join(weather_lines)
        store_short_term_memory("greeting", weather_text, topic="session:weather")
        print(f"[SessionSeed] session:weather → {weather_text[:80]!r}")

    if news_lines:
        news_text = " ".join(news_lines)
        store_short_term_memory("greeting", news_text, topic="session:news")
        print(f"[SessionSeed] session:news → {news_text[:80]!r}")

    if schedule_lines:
        schedule_text = " ".join(schedule_lines)
        store_short_term_memory("greeting", schedule_text, topic="session:schedule")
        print(f"[SessionSeed] session:schedule → {schedule_text[:80]!r}")

    if weather_lines:
        memory_manager.set_last_topic("session:weather")
    elif news_lines:
        memory_manager.set_last_topic("session:news")

    print(f"[SessionSeed] last_topic set to: {memory_manager.get_last_topic()!r}")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION SEED MATCHER
# Used in handle_request() as the fallback before start_new_conversation().
# Scores each session:* bucket against the user's input and returns the
# best matching one, or None if nothing is close enough.
# ─────────────────────────────────────────────────────────────────────────────

_SEED_MATCH_THRESHOLD = 0.30  # minimum cosine score to consider a seed relevant

def _find_best_session_seed(user_input: str, memory_manager) -> str | None:
    """
    Checks all session:* keys in temp_topic_memory and returns the one
    most semantically similar to user_input.

    Uses compute_similarity_hybrid for efficient three-zone scoring:
      - cosine HIGH  → fast yes
      - cosine LOW   → fast no
      - MIDDLE zone  → LLM decides

    Returns the best matching topic key, or None if no seed is relevant.
    """
    if not user_input:
        return None

    # Collect all session seeds that have content
    candidates: list[tuple[str, str]] = []  # (topic_key, context_text)
    for key in _SESSION_KEYS:
        entries = temp_topic_memory.get(key, [])
        if entries:
            # join all stored lines for this seed into one comparison string
            context_text = " ".join(entries)
            candidates.append((key, context_text))

    if not candidates:
        print(f"[SessionSeed] no session seeds found in temp_topic_memory")
        return None

    best_key   = None
    best_score = -1.0

    for key, context_text in candidates:
        # Use the raw cosine score for ranking (hybrid handles routing internally)
        try:
            from brain.core.brain import embedding_model
            from sentence_transformers import util as st_util
            import numpy as np

            emb_input   = embedding_model.encode(user_input,    convert_to_numpy=True)
            emb_context = embedding_model.encode(context_text,  convert_to_numpy=True)
            score = float(st_util.cos_sim(emb_input, emb_context))

            print(f"[SessionSeed] {key} cosine={score:.3f}")

            if score > best_score:
                best_score = score
                best_key   = key

        except Exception as e:
            print(f"[SessionSeed] scoring error for {key}: {e}")
            continue

    # Threshold gate — don't force a seed match if nothing is actually relevant
    if best_score < _SEED_MATCH_THRESHOLD:
        print(f"[SessionSeed] best score {best_score:.3f} below threshold — no seed match")
        return None

    # Final confirmation via hybrid (handles the middle zone properly)
    best_context = temp_topic_memory.get(best_key, [])
    best_ctx_str = " ".join(best_context)

    is_match = compute_similarity_hybrid(user_input, best_key, best_ctx_str)
    if not is_match:
        print(f"[SessionSeed] hybrid rejected {best_key} — falling through")
        return None

    print(f"[SessionSeed] matched → {best_key} (cosine={best_score:.3f})")
    return best_key


def _get_morning_job_line(conversation_manager) -> str:
    """Returns a job briefing line for the morning greeting, or empty string."""
    try:
        briefing = conversation_manager.job_watchdog.get_morning_briefing()
        return briefing or ""
    except Exception:
        return ""


def chat(tello_client=None):

    init_schedule_table()
    print("Starting Buddy...\n")

    identity = IdentityManager()

    time.sleep(2)

    explorer = get_explorer(enabled=True) #TODO add a fallback Qanother api for fetching news
    try:
        startup_data = explorer.get_startup_data()
    except Exception as e:
        print(f"[STARTUP] Weather fetch failed: {e} — continuing without startup data")
        startup_data = None

    # fallback if news fail
    news_diagnostic = getattr(explorer.news_provider, "last_diagnostic", None)
    if news_diagnostic and startup_data is not None:
        print(news_diagnostic.as_debug_summary())
        startup_data["news_warning"] = news_diagnostic.as_buddy_explanation()

    memory_manager = MemoryManager()
    conversation_manager = ConversationManager(
        memory_manager,
        tello_client=tello_client,
        explorer=explorer,
        identity=identity,
    )

    # music feature is disabled for now; re-enable by uncommenting
    # music_watcher = MusicWatcher(
    #     spotify=conversation_manager.spotify,
    #     identity=identity,
    # )
    # music_watcher.start()

    vad = VADRecorder(
        VOICE_QUEUE,
        margin_db=8.0,
        end_of_turn_silence_sec=2.0,
        cooldown_ms=350,
        start_frames=4,
        min_utterance_ms=500,
        max_utterance_ms=60000,
    )
    vad.start()

    tts = TTSWorker(vad=vad, spotify=conversation_manager.spotify)
    conversation_manager.set_tts(tts)
    tts.start()

    full_intro = identity.should_do_full_intro()
    adaptive_agent = init_agent(
        tts=tts,
        memory_manager=memory_manager,
        startup_delay=15 if full_intro else 3,
    )
    text_hotkey = TextHotkey()
    text_hotkey.start()

    clipboard_hotkey = ClipboardHotkey()
    clipboard_hotkey.start()


    print("[INFO] Voice: VAD active. Speak normally. Text: Ctrl+Alt+T. Exit: Ctrl+C")

    bot_intro = ""
    if full_intro:
        bot_intro = introduce_bot_hybrid(
            startup_data=startup_data,
            ollama=ollama,
            model=MODEL_NAME,
            use_llm=True,
            polish_headlines=True,
        )
        job_line = _get_morning_job_line(conversation_manager)
        if job_line:
            bot_intro = bot_intro + "\n\n" + job_line
        _seed_session_context(bot_intro, memory_manager)
        print(f"Buddy: {bot_intro}\n")
        tts.speak(sanitize_tts(bot_intro))
    else:

        gap = identity.hours_since_last_interaction()
        mins = int((gap or 0) * 60)
        short_greeting = f"Systems online. Last session {mins} minutes ago."
        _seed_session_context(short_greeting, memory_manager)

        print(f"Buddy: {short_greeting}\n")


    speaker_client = SpeakerWorkerClient()
    speaker_client.start()

    signal.signal(signal.SIGINT, _on_shutdown)

    while True:

        # ---- VOICE ----
        if not VOICE_QUEUE.empty():
            wav_path = VOICE_QUEUE.get()
            print(f"[VOICE] wav: {wav_path}")

            result = speaker_client.verify_wav(wav_path)
            if not result.get("ok"):
                print(f"[GATE] REJECT score={result.get('score')}")
                continue

            user_text = (transcribe_wav(wav_path) or "").strip()
            if not user_text:
                print("[ASR] empty")
                continue

            print("You (voice):", user_text)
            identity.touch()
            conversation_manager.last_voice_input_time = time.time()

            # Mute VAD immediately — closes the window during LLM processing
            vad.mute()

            bot_reply = ""
            spoken = ""
            try:
                if identity.is_wake_phrase(user_text):
                    bot_reply = identity.wake_response()
                elif wants_vision(user_text):
                    bot_reply = _handle_vision_request(user_text, conversation_manager, tts)
                else:
                    bot_reply = conversation_manager.handle_request(user_text)

                bot_reply = (bot_reply or "").strip()
                spoken = sanitize_tts(bot_reply).strip()

            except Exception as e:
                print(f"[VOICE] handle_request error: {e}")
                traceback.print_exc()
                bot_reply = "Sorry, something went wrong."
                spoken = sanitize_tts(bot_reply).strip()

            finally:
                # If nothing to speak, TTS won't unmute — must do it here
                if not spoken:
                    print("[DBG] empty reply — unmuting VAD manually")
                    vad.unmute()

            print("Buddy:", bot_reply)
            print("[DBG] TTS len:", len(spoken))
            print("[DBG] TTS repr:", repr(spoken))

            if spoken:
                print(f"[DBG] about to call tts.speak(), spoken length={len(spoken)}")
                clean = sanitize_for_tts(spoken)
                tts.speak(spoken)
                print(f"[DBG] tts.speak() returned")  # if this never prints, speak() is blocking


                def _force_unmute():
                    time.sleep(15)
                    if getattr(vad, '_muted', False):
                        print("[WATCHDOG] Force unmuting VAD after timeout")
                        vad.unmute()
                threading.Thread(target=_force_unmute, daemon=True).start()

            continue


        # ---- TEXT ----
        if not TEXT_QUEUE.empty():
            cmd = TEXT_QUEUE.get()
            if cmd == "__OPEN_TEXT__":
                user_text = input("\n[TEXT MODE] Type your message (blank cancels): ").strip()
                if not user_text:
                    continue

                print("You (text):", user_text)

                bot_reply = ""
                try:
                    pv = conversation_manager.memory_manager.get_pending_vision()
                    if pv and wants_vision(user_text):
                        pv = conversation_manager.memory_manager.pop_pending_vision()
                        bot_reply = run_vision_pipeline(
                            image_path=pv["paths"],
                            user_question=user_text,
                            conversation_manager=conversation_manager,
                        )
                    else:
                        bot_reply = conversation_manager.handle_request(user_text)
                except Exception as e:
                    print(f"[TEXT] handle_request error: {e}")
                    bot_reply = "Sorry, something went wrong."

                bot_reply = (bot_reply or "").strip()
                print("Buddy:", bot_reply)

                spoken = sanitize_tts(bot_reply).strip()
                if spoken:
                    clean = sanitize_for_tts(spoken)
                    tts.speak(spoken)

            continue

        time.sleep(0.03)
