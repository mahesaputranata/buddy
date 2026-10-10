"""
project_handler.py
──────────────────
Project-aware launcher and note summarizer for Buddy.

Responsibilities:
  1. Detect trigger phrases like "open terahertz project"
  2. Check MATLAB running state vs. project path
  3. Read + summarize the latest daily note
  4. Open MATLAB at the correct working directory
  5. Set the conversation topic and wait for user confirmation
     before handing off to the existing conversation flow.

Drop-in usage inside ConversationManager.handle_request():
    from brain.projects.project_handler import ProjectHandler
    self.project_handler = ProjectHandler(memory_manager, ollama)
    # then in handle_request, before the classify_intent call:
    proj = self.project_handler.detect_project_trigger(user_input)
    if proj:
        return self.project_handler.handle_open_project(proj)
"""

from __future__ import annotations

import os
import re
import subprocess
import psutil

from datetime import datetime
from pathlib import Path
from typing import Optional
from brain.settings import setting
from zoneinfo import ZoneInfo

import ollama as _ollama_module   # rename to avoid shadowing the param
from brain.projects.note_analyzer import NoteAnalyzer
from brain.projects.note_finder_fallback import (
    find_latest_note_smart, open_file_explorer, NOTE_CONFIRM_STATE,
)
from brain.projects.note_analyzer import NoteAnalyzer

import re
from difflib import SequenceMatcher
from typing import Optional


# Stressed Test Task
JASTIP_PROJECT_ENTRY = {
    "jastip": {
        "display_name": "JASTIP Marketplace",
 
        # Your Vite frontend
        "frontend_url": "http://localhost:5173",
 
        # Your Hono backend
        "backend_url": "http://localhost:3000",
 
        # The stressed_test directory
        "test_dir": setting("jastip_test_dir", "external/jastip/stressed_test"),
 
        # Topic name for Buddy's memory
        "topic_name": "jastip project",
 
        # No MATLAB or notes folder for this project
        "matlab_folder": None,
        "matlab_exe": None,
        "notes_folder": None,
        "note_pattern": None,
    }
}
 
# ── 2. Stressed test trigger phrases ──────────────────────────────────────────
# Add this list near your existing _TRIGGER_MAP
 
STRESSED_TEST_TRIGGERS: list[tuple[list[str], str, str]] = [
    # ── E2E first ────────────────────────────────────────────────────────────
    (
        ["run e2e on jastip", "run e2e tests", "playwright jastip",
         "run playwright", "check dead buttons", "check the flows"],
        "jastip", "e2e",
    ),
    # ── Stress/load ──────────────────────────────────────────────────────────
    (
        ["run load test", "run k6", "stress test only",
         "load test jastip", "load test"],   # ← explicit here
        "jastip", "stress",
    ),
    # ── Audit ────────────────────────────────────────────────────────────────
    (
        ["run audit", "generate report", "get the report",
         "what does friday say", "friday summarize the results",
         "summarize test results"],
        "jastip", "audit",
    ),
    # ── Full (last — catches everything else) ────────────────────────────────
    (
        ["run stressed test on jastip", "run stress test on jastip",
         "do a stressed test on jastip", "ask friday to do a stressed test",
         "ask friday to run the tests", "friday run stressed test",
         "friday check jastip", "friday do the tests",
         "stressed test jastip", "stress test jastip",
         "run tests on jastip", "run the tests on jastip",
         "check jastip", "test jastip"],
        "jastip", "full",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT REGISTRY
# Add more projects here as needed.
# ──────────────────────────────────────────────────────────────────────────────

PROJECTS: dict[str, dict] = {
    "terahertz": {
        "display_name": "THz TDS Material Classification",
        "matlab_folder": setting("thz_project_folder", ""),
        "notes_folder": setting("thz_notes_folder", ""),
        "note_pattern": re.compile(
            r"^Daily_Note_(\d{2})_(\d{2})_(\d{2})",
            re.IGNORECASE,
        ),
        "topic_name": "terahertz project",
        "matlab_exe": setting("matlab_exe", "matlab"),
    },

    "jastip": {
        "display_name": "JASTIP Marketplace",
        "frontend_url": "http://localhost:5173",
        "backend_url": "http://localhost:3000",
        "test_dir": setting("jastip_test_dir", "external/jastip/stressed_test"),
        "topic_name": "jastip project",

        # not used for stressed test
        "matlab_folder": None,
        "matlab_exe": None,
        "notes_folder": None,
        "note_pattern": None,
        "protocols_folder": None,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# TRIGGER DETECTION
# ──────────────────────────────────────────────────────────────────────────────

# Maps trigger keywords → project registry key
_TRIGGER_MAP: list[tuple[list[str], str]] = [
    (
        [
            "terahertz project", "thz project", "thz tds project",
            "open terahertz", "open thz", "open the terahertz",
            "open my terahertz", "open my thz",
            "terahertz work", "thz work",
            "material classification project",
            "open terahertz tds",
        ],
        "terahertz",
    ),
    # Comming Soon for MPC Project and Rover Project
]

_PROJECT_ACTION_WORDS = [
    "open", "load", "start", "launch", "continue", "resume",
    "work on", "discuss", "check", "inspect", "summarize",
]

_PROJECT_CONTEXT_WORDS = [
    "project", "work", "folder", "notes", "note", "checkpoint",
    "daily note", "update",
]

_PROJECT_ALIASES: dict[str, list[str]] = {
    "terahertz": [
        "terahertz",
        "thz",
        "thz tds",
        "terahertz tds",
        "tera hertz",
        "terra hertz",
        "thera hertz",
        "terrahertz",
        "therahertz",
        "terahz",
        "thz project",
        "terahertz project",
        "material classification",
        "material classification project",
    ],
}

def _normalize_project_text(text: str) -> str:
    """
    Normalize voice/text input for project trigger matching.
    Handles punctuation, spacing, and common ASR variants.
    """
    t = (text or "").lower().strip()

    # normalize punctuation to spaces
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # common ASR / typo corrections
    replacements = {
        "therahertz": "terahertz",
        "terrahertz": "terahertz",
        "tera hertz": "terahertz",
        "terra hertz": "terahertz",
        "thera hertz": "terahertz",
        "teraheartz": "terahertz",
        "terahz": "terahertz",
        "t h z": "thz",
    }

    for wrong, right in replacements.items():
        t = t.replace(wrong, right)

    return t


def _has_action_or_context(t: str) -> bool:
    """
    Prevents random mentions like 'terahertz is interesting'
    from always opening the project.
    """
    return (
        any(w in t for w in _PROJECT_ACTION_WORDS)
        or any(w in t for w in _PROJECT_CONTEXT_WORDS)
    )


def _fuzzy_contains_project_alias(t: str, aliases: list[str], threshold: float = 0.82) -> bool:
    """
    Fuzzy fallback for voice transcription errors.
    Example: 'therahertz' / 'terrahertz' / slightly broken variants.
    """
    words = t.split()

    # Check single words and small 2-word windows
    windows = words[:]
    windows += [" ".join(words[i:i+2]) for i in range(len(words) - 1)]

    for w in windows:
        for alias in aliases:
            ratio = SequenceMatcher(None, w, alias).ratio()
            if ratio >= threshold:
                return True

    return False


def detect_project_trigger(user_input: str) -> Optional[str]:
    """
    Returns project key if the user's utterance refers to a known project.
    Robust against common voice transcription variants.
    """
    t = _normalize_project_text(user_input)

    # 1. Exact phrase map, keep your old behavior
    for phrases, project_key in _TRIGGER_MAP:
        normalized_phrases = [_normalize_project_text(p) for p in phrases]
        if any(p in t for p in normalized_phrases):
            print(f"[PROJECT_TRIGGER] exact match -> {project_key} | input={t!r}")
            return project_key

    # 2. Alias-based matching
    for project_key, aliases in _PROJECT_ALIASES.items():
        aliases_norm = [_normalize_project_text(a) for a in aliases]

        alias_hit = any(a in t for a in aliases_norm)
        fuzzy_hit = _fuzzy_contains_project_alias(t, aliases_norm)

        if (alias_hit or fuzzy_hit) and _has_action_or_context(t):
            mode = "alias" if alias_hit else "fuzzy"
            print(f"[PROJECT_TRIGGER] {mode} match -> {project_key} | input={t!r}")
            return project_key

    return None

# ──────────────────────────────────────────────────────────────────────────────
# RESUME TRIGGER DETECTION
# Detects "let's work on it", "let's continue", "let's discuss" etc.
# Used by ConversationManager.handle_pending to know when to open the topic.
# ──────────────────────────────────────────────────────────────────────────────

_RESUME_PHRASES = [
    "let's continue", "lets continue",
    "let's work", "lets work",
    "let's discuss", "lets discuss",
    "continue", "yes", "yeah", "yep", "sure", "ok", "okay",
    "open it", "open the topic", "start working",
    "let's go", "lets go", "go ahead",
    "start", "begin", "proceed",
    "yes let's", "yes lets",
]

def is_project_resume(user_input: str) -> bool:
    """
    Returns True if the user's response means 'yes, open the project topic'.
    """
    t = (user_input or "").lower().strip()
    # strip punctuation
    t = re.sub(r"[.,!?]$", "", t).strip()
    return any(t == p or t.startswith(p) for p in _RESUME_PHRASES)


# ──────────────────────────────────────────────────────────────────────────────
# DAILY NOTE PARSING
# ──────────────────────────────────────────────────────────────────────────────

def _parse_note_date(filename: str, pattern: re.Pattern) -> Optional[datetime]:
    """
    Parse date from filename like Daily_Note_13_03_26.txt
    Returns a datetime or None.
    """
    m = pattern.match(Path(filename).stem)
    if not m:
        return None
    try:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 2000 + yy
        return datetime(year, mm, dd)
    except ValueError:
        return None


def find_latest_note(notes_folder: str, pattern: re.Pattern) -> Optional[Path]:
    """
    Scans the notes folder and returns the Path of the most recent daily note
    based on the date encoded in the filename.
    """
    folder = Path(notes_folder)
    if not folder.exists():
        print(f"[PROJECT] Notes folder not found: {notes_folder}")
        return None

    candidates: list[tuple[datetime, Path]] = []
    for f in folder.iterdir():
        if f.suffix.lower() != ".txt":
            continue
        dt = _parse_note_date(f.name, pattern)
        if dt:
            candidates.append((dt, f))

    if not candidates:
        return None

    # Return path of the note with the latest date
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def read_note_sections(note_path: Path) -> dict[str, str]:
    """
    Parses a daily note file into named sections.
    Returns a dict with keys: 'date', 'objectives', 'methodology', 'notes'
    Values are raw text strings (may be long).
    """
    try:
        raw = note_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[PROJECT] Failed to read note: {e}")
        return {}

    sections: dict[str, str] = {"date": "", "objectives": "", "methodology": "", "notes": ""}

    # Extract header date
    date_match = re.search(r"\[Daily Note\]\s*-\s*(.+)", raw)
    if date_match:
        sections["date"] = date_match.group(1).strip()

    # Extract objectives block (between "Objectives" and next numbered section)
    obj_match = re.search(
        r"\d+\.\s*\(Objectives?\)\s*[-–>]*\s*(.+?)(?=\n\d+\.\s*\(|\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    if obj_match:
        sections["objectives"] = obj_match.group(1).strip()

    # Extract methodology block
    meth_match = re.search(
        r"\d+\.\s*\(Methodology\)\s*[-–>]*\s*(.+?)(?=\nnote\s*:|^\s*note\s*\d*\s*:|\Z)",
        raw, re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    if meth_match:
        sections["methodology"] = meth_match.group(1).strip()

    # Extract all note: / note N: blocks at the bottom
    note_blocks = re.findall(
        r"note\s*\d*\s*:\s*(.+?)(?=note\s*\d*\s*:|\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    if note_blocks:
        sections["notes"] = "\n\n".join(b.strip() for b in note_blocks)

    return sections


# ──────────────────────────────────────────────────────────────────────────────
# TTS DATE NORMALIZER
# Converts raw date strings from note headers into natural spoken English.
# Handles the formats that appear in your daily note filenames and headers.
#
# Examples:
#   "13/03/26"          → "March 13th, 2026"
#   "13/03/2026"        → "March 13th, 2026"
#   "2026-03-13"        → "March 13th, 2026"
#   "Daily_Note_13_03_26" → "March 13th, 2026"  (stem fallback)
#   "unknown"           → "unknown"             (passthrough)
# ──────────────────────────────────────────────────────────────────────────────

_MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def _ordinal(n: int) -> str:
    """Return the ordinal string for a day number, e.g. 13 → '13th'."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"

def normalize_date_for_tts(raw: str) -> str:
    """
    Converts a raw date string into a natural spoken-English date.
    Returns the original string unchanged if it cannot be parsed,
    so it is always safe to call.
    """
    s = (raw or "").strip()
    if not s or s.lower() == "unknown":
        return s

    day = month = year = None

    # Format: DD/MM/YY or DD/MM/YYYY
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000

    # Format: YYYY-MM-DD  (ISO)
    if day is None:
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Format: Daily_Note_DD_MM_YY (filename stem fallback)
    if day is None:
        m = re.search(r"Daily_Note_(\d{2})_(\d{2})_(\d{2,4})", s, re.IGNORECASE)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 2000

    # Validate and format
    if day and month and year and 1 <= month <= 12 and 1 <= day <= 31:
        return f"{_MONTHS[month]} {_ordinal(day)}, {year}"

    # Passthrough — unknown format, return as-is (better than crashing)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# LLM SUMMARIZER
# ──────────────────────────────────────────────────────────────────────────────

_SUMMARY_SYSTEM = (
    "You are a personal AI assistant giving a spoken briefing to your user about their own project notes. "
    "The user is a researcher working on machine learning with THz TDS spectroscopy. "
    "You are speaking DIRECTLY TO THEM, so always use second-person language: "
    "'you', 'your', 'you were', 'you explored' — never 'the researcher' or third-person. "
    "Your summary will be spoken aloud, so:\n"
    "- Use plain natural language, no markdown, no bullet points.\n"
    "- Be concise: 4–6 sentences maximum.\n"
    "- Cover: what the objective was, what methodology or approach was used, "
    "and what the key takeaway or next step is from the notes section.\n"
    "- Do not start with 'Sure' or 'Here is a summary'. Start directly with the content.\n"
    "- Example of correct tone: 'Yesterday you were working on a bandpass filter approach "
    "before windowing. Your objective was to replicate the Bochum paper pipeline...'"
)

def summarize_note(
    sections: dict[str, str],
    ollama,
    model: str = "llama3.2:latest",
) -> str:
    """
    Uses the LLM to produce a spoken-friendly summary of the daily note.
    Falls back to a deterministic summary if LLM fails.
    """
    if not sections:
        return "I couldn't read the latest project note."

    date_str = sections.get("date", "unknown date")
    objectives = (sections.get("objectives") or "")[:600]
    methodology = (sections.get("methodology") or "")[:800]
    notes = (sections.get("notes") or "")[:600]

    user_prompt = (
        f"Daily note date: {date_str}\n\n"
        f"OBJECTIVES:\n{objectives}\n\n"
        f"METHODOLOGY APPROACH:\n{methodology[:300]}\n\n"  # keep it short for LLM
        f"YOUR NOTES:\n{notes}\n\n"
        "Summarize this as a spoken briefing directly to the user. "
        "Use 'you' and 'your' throughout — never 'the researcher'."
    )

    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        out = ""
        if isinstance(resp, dict):
            out = ((resp.get("message") or {}).get("content") or "").strip()
        else:
            out = getattr(getattr(resp, "message", None), "content", "") or ""

        out = out.strip()
        if out:
            return out
    except Exception as e:
        print(f"[PROJECT] LLM summarize failed: {e}")

    # ── Deterministic fallback ──
    parts = []
    if date_str:
        parts.append(f"Latest note from {date_str}.")
    if objectives:
        # first sentence only
        first = objectives.split(".")[0].strip()
        if first:
            parts.append(f"Objective: {first}.")
    if notes:
        first_note = notes.split("\n")[0].strip()
        if first_note:
            parts.append(f"Key note: {first_note}.")
    return " ".join(parts) if parts else "I found the latest note but couldn't summarize it."


# ──────────────────────────────────────────────────────────────────────────────
# MATLAB PROCESS INSPECTOR
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_path(p: str) -> str:
    """Lowercase + normalize separators for comparison."""
    return os.path.normpath(p or "").lower()


def _get_matlab_processes() -> list[psutil.Process]:
    """Returns all running MATLAB processes."""
    result = []
    for proc in psutil.process_iter(["name", "pid"]):
        try:
            if "matlab" in (proc.info.get("name") or "").lower():
                result.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def check_matlab_state(project_folder: str) -> str:
    """
    Checks the state of MATLAB relative to the project folder.

    Returns one of:
        "not_running"     — MATLAB is not open at all
        "wrong_folder"    — MATLAB is open but NOT in the project folder
        "correct_folder"  — MATLAB is already open in the project folder
        "unknown"         — MATLAB is open but we can't determine its folder
    """
    procs = _get_matlab_processes()
    if not procs:
        return "not_running"

    target = _normalize_path(project_folder)

    for proc in procs:
        # Method 1: check cwd (may need elevated privileges on Windows)
        try:
            cwd = _normalize_path(proc.cwd())
            if cwd.startswith(target):
                return "correct_folder"
            else:
                # cwd is readable but points elsewhere
                pass
        except (psutil.AccessDenied, psutil.NoSuchProcess, NotImplementedError):
            pass

        # Method 2: scan open file handles for .m files in our project folder
        try:
            for f in proc.open_files():
                fpath = _normalize_path(f.path)
                if target in fpath and fpath.endswith(".m"):
                    return "correct_folder"
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        # Method 3: check command line arguments
        try:
            cmdline = " ".join(proc.cmdline()).lower()
            if target in cmdline:
                return "correct_folder"
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    # MATLAB is running but we couldn't confirm the folder
    # Default to "wrong_folder" so we open the project just in case,
    # unless cwd was readable and pointed elsewhere → already handled above.
    return "unknown"


def open_matlab_at_folder(matlab_exe: str, project_folder: str) -> bool:
    """
    Launches MATLAB with the project folder as the working directory.
    Returns True if the subprocess launched without error.
    """
    try:
        subprocess.Popen(
            [matlab_exe],
            cwd=project_folder,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        print(f"[PROJECT] MATLAB launched at {project_folder}")
        return True
    except Exception as e:
        print(f"[PROJECT] Failed to launch MATLAB: {e}")
        return False

# Stressed Test
def detect_stressed_test_trigger(user_input: str) -> tuple[str, str] | None:
    """
    Returns (project_key, scope) if the user's utterance is a stressed test request.
    Returns None otherwise.
 
    Examples:
      "buddy ask friday to do a stressed test" → ("jastip", "full")
      "run e2e tests on jastip"               → ("jastip", "e2e")
      "friday check jastip"                   → ("jastip", "full")
    """
    t = _normalize_project_text(user_input)
 
    for phrases, project_key, scope in STRESSED_TEST_TRIGGERS:
        normalized = [_normalize_project_text(p) for p in phrases]
        if any(p in t for p in normalized):
            print(f"[STRESSED_TEST_TRIGGER] matched → project={project_key} scope={scope}")
            return project_key, scope
 
    # Also catch "ask friday" + any project alias
    if "friday" in t or "ask friday" in t:
        for project_key, aliases in _PROJECT_ALIASES.items():
            aliases_norm = [_normalize_project_text(a) for a in aliases]
            if any(a in t for a in aliases_norm):
                # Determine scope from context
                if "e2e" in t or "playwright" in t or "flow" in t:
                    scope = "e2e"
                elif "load" in t or "k6" in t or "stress" in t:
                    scope = "stress"
                elif "audit" in t or "report" in t or "summary" in t:
                    scope = "audit"
                else:
                    scope = "full"
                print(f"[STRESSED_TEST_TRIGGER] friday+alias → project={project_key} scope={scope}")
                return project_key, scope
 
    return None

# ──────────────────────────────────────────────────────────────────────────────
# PROJECT HANDLER — main class
# ──────────────────────────────────────────────────────────────────────────────
class StressedTestMixin:
    """
    Mixin for stressed test support.
    Requires self.friday to be a Friday instance.
    """

    def handle_stressed_test(self, project_key: str, scope: str = "full") -> str:
        from brain.projects.project_handler import PROJECTS

        cfg = PROJECTS.get(project_key)
        if not cfg:
            return f"I don't have '{project_key}' registered as a project."

        if not self.friday:
            return (
                "Friday isn't connected. "
                "Make sure FRIDAY_URL and FRIDAY_HOOK_TOKEN are set."
            )

        display_name = cfg["display_name"]
        test_dir = cfg.get("test_dir", "")
        frontend_url = cfg.get("frontend_url", "http://localhost:5173")
        backend_url = cfg.get("backend_url", "http://localhost:3000")

        if not test_dir:
            return f"No test directory configured for {display_name}."

        scope_labels = {
            "full": "the full test suite (Playwright + k6 load test + audit report)",
            "e2e": "the Playwright E2E tests only",
            "stress": "the k6 load test only",
            "audit": "the audit report from existing results",
        }
        scope_label = scope_labels.get(scope, "the tests")

        self.mm.set_last_topic(f"friday:jastip_test:{scope}")

        from brain.core.brain import store_short_term_memory
        store_short_term_memory(
            f"stressed test {scope}",
            f"Running {scope_label} for {display_name}. Friday is on it.",
            topic=f"friday:jastip_test:{scope}",
        )

        reply = self.friday.run_stressed_test(
            project_key=project_key,
            project_display=display_name,
            test_dir=test_dir,
            frontend_url=frontend_url,
            backend_url=backend_url,
            scope=scope,
        )

        return (
            f"Running {scope_label} for {display_name}. "
            f"{reply} "
            f"I'll speak the results when Friday is done. "
            f"You can ask me follow-up questions about the results after."
        )

class ProjectHandler(StressedTestMixin):
    """
    Plug into ConversationManager like so:

        self.project_handler = ProjectHandler(
            memory_manager=self.memory_manager,
            ollama=ollama,
            model="llama3.2:latest",
        )

    Then in handle_request(), BEFORE classify_intent():

        proj_key = self.project_handler.detect_project_trigger(user_input)
        if proj_key:
            return self.project_handler.handle_open_project(proj_key)

    And in handle_pending(), add:

        if pending_state == "Awaiting_Project_Resume":
            return self.project_handler.handle_resume_response(
                user_input, conversation_manager=self
            )
    """

    PENDING_STATE = "Awaiting_Project_Resume"

    def __init__(self, memory_manager, ollama, model: str = "llama3.2:latest", friday=None):
        self.mm = memory_manager
        self.ollama = ollama
        self.model = model
        self.friday = friday

    def handle_stressed_test(self, project_key: str, scope: str = "full") -> str:
        from brain.projects.project_handler import PROJECTS
        cfg = PROJECTS.get(project_key)
        if not cfg:
            return f"I don't have '{project_key}' registered as a project."

        if not self.friday:
            return "Friday isn't connected. Make sure FRIDAY_URL and FRIDAY_HOOK_TOKEN are set."

        display_name = cfg["display_name"]
        test_dir     = cfg.get("test_dir", "")
        frontend_url = cfg.get("frontend_url", "http://localhost:5173")
        backend_url  = cfg.get("backend_url", "http://localhost:3000")

        if not test_dir:
            return f"No test directory configured for {display_name}."

        scope_labels = {
            "full":   "the full test suite (Playwright + k6 load test + audit report)",
            "e2e":    "the Playwright E2E tests only",
            "stress": "the k6 load test only",
            "audit":  "the audit report from existing results",
        }
        scope_label = scope_labels.get(scope, "the tests")

        self.mm.set_last_topic(f"friday:jastip_test:{scope}")

        from brain.core.brain import store_short_term_memory
        store_short_term_memory(
            f"stressed test {scope}",
            f"Running {scope_label} for {display_name}. Friday is on it.",
            topic=f"friday:jastip_test:{scope}",
        )

        reply = self.friday.run_stressed_test(
            project_key=project_key,
            project_display=display_name,
            test_dir=test_dir,
            frontend_url=frontend_url,
            backend_url=backend_url,
            scope=scope,
        )

        return (
            f"Running {scope_label} for {display_name}. "
            f"{reply} "
            f"I'll let you know when Friday is done."
        )
    
    # public entry points ──────────────────────────────────────────────────────
    def _handle_note_confirm(
        self,
        user_input: str,
        conversation_manager,
    ) -> str:
        """
        Called from handle_resume_response() when pending_state
        == "Awaiting_Note_Confirm".

        User is confirming or rejecting the note file Buddy found.
        yes → proceed with the candidate path
        no  → open File Explorer, clear pending
        """
        from brain.projects.note_finder_fallback import open_file_explorer

        payload = self.mm.pending_payload or {}
        notes_folder   = payload.get("notes_folder", "")
        note_candidate = payload.get("note_candidate")   # str path
        note_spoken    = payload.get("note_spoken", "that file")
        project_key    = payload.get("project_key", "")
        topic_name     = payload.get("topic_name", "")
        note_date      = payload.get("note_date", "")

        def _on_confirm() -> str:
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None

            # Resume the open-project flow with the confirmed note path
            if note_candidate:
                from pathlib import Path
                from brain.projects.project_handler import (
                    read_note_sections, summarize_note,
                    normalize_date_for_tts,
                )
                from brain.projects.note_analyzer import NoteAnalyzer
                from brain.core.brain import store_short_term_memory

                note_path = Path(note_candidate)
                sections  = read_note_sections(note_path)
                summary   = summarize_note(sections, self.ollama, self.model)

                cfg = PROJECTS.get(project_key, {})
                analyzer = NoteAnalyzer(
                    project_key=project_key,
                    ollama=self.ollama,
                    model=self.model,
                )
                todos, _ = analyzer.run(
                    note_path=note_path,
                    protocols_folder=cfg.get("protocols_folder"),
                )
                todo_briefing = analyzer.build_session_briefing(todos)

                store_short_term_memory(
                    f"Opening {cfg.get('display_name', project_key)}",
                    summary,
                    topic=topic_name,
                )
                self.mm.set_last_topic(topic_name)

                spoken_date = normalize_date_for_tts(note_date)
                response = (
                    f"Got it. Here is your update from {spoken_date}: "
                    f"{summary}"
                )
                if todo_briefing:
                    response += f" {todo_briefing}"
                return response

            return "Okay, let me know what you would like to work on."

        def _on_decline() -> str:
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None

            if notes_folder:
                open_file_explorer(notes_folder)
                return (
                    "No problem — I have opened the notes folder so you can "
                    "find the right file. Just let me know the date when you are ready."
                )

            return "Okay, I will skip the note for now. What would you like to work on?"

        question_context = f"I found a file called {note_spoken} — is that the right one?"
        return conversation_manager.handle_confirm_state(
            user_input,
            question_context,
            _on_confirm,
            _on_decline,
            state_name=NOTE_CONFIRM_STATE,
        )


    def detect_project_trigger(self, user_input: str) -> Optional[str]:
        return detect_project_trigger(user_input)

    def handle_open_project(self, project_key: str) -> str:
        cfg = PROJECTS.get(project_key)
        if not cfg:
            return f"I don't have a project registered under '{project_key}'."

        display_name = cfg["display_name"]
        notes_folder = cfg["notes_folder"]
        note_pattern = cfg["note_pattern"]
        matlab_folder = cfg["matlab_folder"]
        matlab_exe    = cfg["matlab_exe"]
        topic_name    = cfg["topic_name"]

        # ── Step 1: MATLAB state check (must be first — matlab_line used below) ──
        matlab_status = check_matlab_state(matlab_folder)
        print(f"[PROJECT] MATLAB state: {matlab_status}")

        if matlab_status == "correct_folder":
            matlab_line = (
                "It looks like MATLAB is already open on the terahertz project, "
                "so I left it as is."
            )
        elif matlab_status == "not_running":
            opened = open_matlab_at_folder(matlab_exe, matlab_folder)
            matlab_line = (
                "I've opened MATLAB at the project folder."
                if opened
                else "I tried to open MATLAB but something went wrong. "
                    "You may need to open it manually."
            )
        elif matlab_status == "wrong_folder":
            opened = open_matlab_at_folder(matlab_exe, matlab_folder)
            matlab_line = (
                "MATLAB was open on a different folder, "
                "so I opened a new instance at the terahertz project."
                if opened
                else "MATLAB is open but on a different folder. "
                    "Please navigate to the terahertz project manually."
            )
        else:
            opened = open_matlab_at_folder(matlab_exe, matlab_folder)
            matlab_line = (
                "MATLAB is running — I've also opened the terahertz project folder "
                "in a new instance just in case."
                if opened
                else "MATLAB seems to be running. Please make sure you're in the project folder."
            )

        # ── Step 2: Read latest note (smart finder with fallback) ────────────────
        note_result = find_latest_note_smart(notes_folder, note_pattern)
        note_path   = note_result["path"]

        # Case A: confidence too low — open File Explorer immediately
        if note_result["open_explorer"]:
            open_file_explorer(notes_folder)
            self.mm.set_pending_topic(self.PENDING_STATE)
            self.mm.pending_payload = {
                "project_key":  project_key,
                "topic_name":   topic_name,
                "note_date":    "",
                "notes_folder": notes_folder,
            }
            return (
                f"Opening the {display_name} project. {matlab_line} "
                f"I couldn't find your latest daily note — "
                f"I've opened the notes folder so you can point me to it. "
                f"Would you like to continue without the note summary for now?"
            )

        # Case B: borderline confidence — ask user to confirm
        if note_result["ask_user"]:
            spoken_name = note_result["spoken_name"]
            self.mm.set_pending_topic(NOTE_CONFIRM_STATE)
            self.mm.pending_payload = {
                "project_key":    project_key,
                "topic_name":     topic_name,
                "note_date":      str(note_result["file_date"]),
                "notes_folder":   notes_folder,
                "note_candidate": str(note_path),
                "note_spoken":    spoken_name,
            }
            return (
                f"Opening the {display_name} project. {matlab_line} "
                f"I found a file called {spoken_name} — is that the right one?"
            )

        # Case C: confident match — proceed directly
        if note_path:
            sections  = read_note_sections(note_path)
            summary   = summarize_note(sections, self.ollama, self.model)
            note_date = sections.get("date") or note_path.stem

            analyzer = NoteAnalyzer(
                project_key=project_key,
                ollama=self.ollama,
                model=self.model,
            )
            todos, short_term_ctx = analyzer.run(
                note_path=note_path,
                protocols_folder=cfg.get("protocols_folder"),
            )
            todo_briefing = analyzer.build_session_briefing(todos)
        else:
            summary       = "I couldn't find any daily notes for this project."
            note_date     = "unknown"
            todo_briefing = ""

        if self.friday and note_path and Path(notes_folder).exists():
            try:
                friday_reply = self.friday.analyze_folder(
                    folder_path=notes_folder,
                    instruction=(
                        "Use friday-local-project-inspector. "
                        f"Buddy already identified this latest note: {note_path}. "
                        "Inspect the notes folder read-only and summarize the checkpoint from that note. "
                        "Focus on objective, methodology, current problem, uncertainty, and next step. "
                        "Return structured output for Buddy. "
                        "Do not modify, create, delete, move, or rename any files."
                    )
                )
                print(f"[PROJECT][FRIDAY] {friday_reply}")
            except Exception as e:
                print(f"[PROJECT][FRIDAY] Failed to dispatch Friday task: {e}")

        # ── Step 3: Store context in STM ─────────────────────────────────────────
        from brain.core.brain import store_short_term_memory
        store_short_term_memory(
            f"Opening {display_name}",
            summary,
            topic=topic_name,
        )

        # ── Step 4: Set pending state ─────────────────────────────────────────────
        self.mm.set_pending_topic(self.PENDING_STATE)
        self.mm.pending_payload = {
            "project_key": project_key,
            "topic_name":  topic_name,
            "note_date":   note_date,
        }

        # ── Step 5: Build spoken response ─────────────────────────────────────────
        spoken_date = normalize_date_for_tts(note_date)
        response = (
            f"Opening the {display_name} project. "
            f"{matlab_line} "
            f"Here is your latest update from {spoken_date}: "
            f"{summary} "
        )
        if todo_briefing:
            response += f"{todo_briefing} "
        response += "Would you like to continue working on it?"

        return response

    def handle_resume_response(self, user_input, conversation_manager):
        payload = self.mm.pending_payload or {}
        topic_name = payload.get("topic_name", "terahertz project")
        note_date = payload.get("note_date", "")

        # 1. Note confirm takes absolute priority
        if self.mm.get_pending_topic() == NOTE_CONFIRM_STATE:
            return self._handle_note_confirm(user_input, conversation_manager)

        # 2. Explicit yes
        if is_project_resume(user_input):
            return self._accept_project_resume(topic_name, note_date, conversation_manager)

        # 3. Explicit no (hard keywords)
        if any(w in (user_input or "").lower() for w in
            ["no", "not now", "later", "cancel", "skip"]):
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None
            return "Okay, I'll leave it for now. Let me know when you want to pick it up."

        # 4. Soft decline (catches "I'm not ready", "not yet", etc.)
        SOFT_DECLINE_RE = re.compile(
            r"\b(not ready|not now|not yet|maybe later|i'?m not|i am not)\b",
            re.IGNORECASE
        )
        if SOFT_DECLINE_RE.search(user_input):
            self.mm.set_pending_topic(None)
            self.mm.pending_payload = None
            return "Okay, I'll leave it for now. Let me know when you want to pick it up."

        # 5. Substantive input → implicit yes
        SUBSTANTIVE_RE = re.compile(
            r"^(what|how|why|explain|tell me|i'?m|i am|i have|i want|"
            r"the |my |can you|could you|is there|are there|"
            r"let'?s|actually|open|search|remind|take a note)\b",
            re.IGNORECASE
        )
        if bool(SUBSTANTIVE_RE.match(user_input.strip())) or len(user_input.strip().split()) > 5:
            print(f"[DEBUG] Awaiting_Project_Resume: substantive input → implicit yes")
            self._accept_project_resume(topic_name, note_date, conversation_manager, silent=True)
            return conversation_manager.handle_request(user_input)

        # 6. Genuinely ambiguous short input → re-prompt once
        return (
            "Just say something like 'let's continue' or 'let's discuss' "
            "to open the project topic, or 'not now' to skip."
        )
    
    def _accept_project_resume(
        self,
        topic_name: str,
        note_date: str,
        conversation_manager,
        silent: bool = False,
    ) -> str:
        """Clears pending state and opens the project as active topic."""
        self.mm.set_pending_topic(None)
        self.mm.pending_payload = None
        self.mm.set_last_topic(topic_name)
        print(f"[PROJECT] Topic opened: {topic_name!r}")
        
        if silent:
            # Called before re-routing a substantive question
            # No spoken response needed — the question answer replaces it
            return ""
        
        spoken_date = normalize_date_for_tts(note_date)
        return (
            f"Great. I've opened the {topic_name} as our active topic. "
            f"We were last working on the note from {spoken_date}. "
            "What would you like to work on or discuss?"
        )