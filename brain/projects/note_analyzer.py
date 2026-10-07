"""
brain/projects/note_analyzer.py
────────────────────────────────
NoteAnalyzer — reads THz daily notes and meeting protocol PDFs,
extracts TODOs, scores them, and writes to the todos SQL table.

Two source types:
  1. Daily note  (.txt / .md)  — lab notebook prose, LLM extraction
  2. Meeting protocol (.pdf)   — structured action items table, rule-based

Called from ProjectHandler.handle_open_project() after find_latest_note().
Also adds a scan_todo_deadlines() shim so ScheduleAwareness can pick up
open TODOs automatically without any changes to agent_core.py.

Dependencies:
  pip install pdfplumber          ← primary PDF parser
  pdfminer.six / pypdf            ← fallback (usually already installed)

Drop-in usage inside ProjectHandler.handle_open_project():
    from brain.projects.note_analyzer import NoteAnalyzer
    analyzer = NoteAnalyzer(project_key="terahertz", ollama=ollama)
    todos = analyzer.run(
        note_path=note_path,          # Path | None
        protocol_folder=protocol_folder,  # str path to 1_meeting_protocols
    )
    # todos is a list of dicts — already saved to SQL, also returned for logging
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# PDF BACKEND — pdfplumber preferred, pypdf fallback
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pdfplumber
    _PDF_BACKEND = "pdfplumber"
except ImportError:
    pdfplumber = None
    try:
        from pypdf import PdfReader as _PdfReader
        _PDF_BACKEND = "pypdf"
    except ImportError:
        _PdfReader = None
        _PDF_BACKEND = "none"

print(f"[NoteAnalyzer] PDF backend: {_PDF_BACKEND}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Maps Category column text → priority integer (1=critical … 4=low)
CATEGORY_PRIORITY: dict[str, int] = {
    "organisation":  2,
    "organization":  2,
    "experimental":  2,
    "documentation": 3,
    "low priority":  4,
    "low":           4,
}

# Maps Deadline column text → days from today (None = no hard date)
DEADLINE_DAYS: dict[str, Optional[int]] = {
    "today":        0,
    "this week":    7,
    "next meeting": None,
    "tbd":          None,
    "":             None,
}

# Signals that push a TODO to long-term SQL storage
HIGH_URGENCY_WORDS = {
    "must", "critical", "urgent", "before", "by",
    "deadline", "asap", "fix", "broken", "crash",
    "important", "required", "validate", "confirm",
}

# LLM prompt for extracting TODOs from lab notebook prose
_TODO_EXTRACTION_PROMPT = """\
You are reading a lab notebook entry for a THz TDS material classification research project.
Extract every unfinished task, open question, planned next step, or concern mentioned.

Rules:
- Return ONLY a JSON array of objects, nothing else, no markdown fences.
- Each object has exactly these keys:
    "content"  : string — the task or question, one sentence, plain English
    "category" : string — one of: experimental | documentation | organisation | low_priority
    "urgency"  : string — one of: high | medium | low
    "deadline_hint" : string — any date or timing mentioned near this item, or ""
- Skip anything already marked as done or resolved.
- Skip general observations with no action needed.
- Maximum 10 items.

Example output:
[
  {"content": "Test model with single real shot signal", "category": "experimental", "urgency": "high", "deadline_hint": "next meeting"},
  {"content": "Investigate why Shot 2 is a dead trace", "category": "experimental", "urgency": "medium", "deadline_hint": ""}
]

Lab notebook text:
"""


# ─────────────────────────────────────────────────────────────────────────────
# MEETING PROTOCOL FINDER
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_meeting_protocol(protocols_folder: str) -> Optional[Path]:
    """
    Scans the meeting protocols folder for date-named subfolders.
    Each subfolder is named Meeting_Protocol_YYYY-MM-DD and contains
    Meeting_Protocol_YYYY-MM-DD.pdf.

    Returns the Path to the most recent PDF, or None if not found.
    """
    folder = Path(protocols_folder)
    if not folder.exists():
        print(f"[NoteAnalyzer] Protocols folder not found: {protocols_folder}")
        return None

    # Pattern: Meeting_Protocol_YYYY-MM-DD
    folder_pattern = re.compile(
        r"^Meeting_Protocol_(\d{4})-(\d{2})-(\d{2})$",
        re.IGNORECASE,
    )

    candidates: list[tuple[date, Path]] = []

    for item in folder.iterdir():
        if not item.is_dir():
            continue
        m = folder_pattern.match(item.name)
        if not m:
            continue
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue

        # Look for the PDF inside
        pdf_path = item / f"{item.name}.pdf"
        if pdf_path.exists():
            candidates.append((d, pdf_path))

    if not candidates:
        print(f"[NoteAnalyzer] No meeting protocol PDFs found in {protocols_folder}")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    latest = candidates[0][1]
    print(f"[NoteAnalyzer] Latest meeting protocol: {latest.name}")
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# PDF TABLE PARSER — pdfplumber primary
# ─────────────────────────────────────────────────────────────────────────────

def _parse_table_pdfplumber(pdf_path: Path) -> list[dict]:
    """
    Extracts action item rows from the PDF using pdfplumber.

    Your meeting protocol PDF uses merged cells — pdfplumber splits each
    merged cell across multiple columns filled with None.  The real data
    always lands at fixed column indices:
        0  = row number
        3  = Category
        6  = Action Item
        9  = Deadline
        12 = Progress (☒ / ☐)

    Rows for a single action item are also split across multiple pdfplumber
    rows (number on one row, content on another).  We accumulate values and
    flush when a progress symbol is found, which always marks the end of a
    logical action item row.
    """
    # Fixed column indices — derived from actual PDF structure
    COL = {
        "number":   0,
        "category": 3,
        "content":  6,
        "deadline": 9,
        "progress": 12,
    }
    PROGRESS_SYMBOLS = {"☒", "☐", "✓", "✗", "x", "X"}

    all_rows = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Detect action items table by header row
                header_flat = " ".join(
                    str(c or "").strip() for c in table[0]
                ).lower()
                if not any(
                    kw in header_flat
                    for kw in ["category", "action", "deadline", "progress"]
                ):
                    continue

                # Accumulator — carries forward last seen non-None value
                # per column as we step through pdfplumber rows
                current: dict[str, str | None] = {k: None for k in COL}

                for raw_row in table[2:]:   # skip header rows (row 0 + row 1)
                    if not raw_row:
                        continue

                    def _cell(idx: int) -> str | None:
                        if idx >= len(raw_row):
                            return None
                        v = raw_row[idx]
                        s = str(v).strip() if v is not None else None
                        return s if s else None

                    # Pick up any non-None values into accumulator
                    for key, idx in COL.items():
                        val = _cell(idx)
                        if val:
                            current[key] = val

                    # Flush when progress symbol found — end of logical row
                    prog = current.get("progress") or ""
                    if prog in PROGRESS_SYMBOLS and current.get("content"):
                        all_rows.append({k: (v or "") for k, v in current.items()})
                        current = {k: None for k in COL}

    return all_rows


def _parse_table_pypdf(pdf_path: Path) -> list[dict]:
    """
    Fallback PDF parser using pypdf (text only, no table structure).
    Extracts lines that look like action item rows by pattern matching.

    Row pattern: number  category  content  deadline  progress_symbol
    e.g. "8  Experimental  For 200ps: use best hyperparameter...  Next meeting  ☐"
    """
    rows = []
    reader = _PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # Find the Action Items section
    section_match = re.search(
        r"action items?.*?(?=\n\d+\.|$)",
        full_text,
        re.IGNORECASE | re.DOTALL,
    )
    if section_match:
        section_text = full_text[section_match.start():]
    else:
        section_text = full_text

    # Match lines: starts with number, ends with ☒ or ☐
    line_pattern = re.compile(
        r"(\d+)\s+"                           # number
        r"(Organisation|Experimental|Documentation|Low Priority)\s+"  # category
        r"(.+?)\s+"                           # content
        r"(Today|This week|Next meeting|TBD)\s*"  # deadline
        r"([☒☐✓✗]?)",                        # progress
        re.IGNORECASE,
    )

    for m in line_pattern.finditer(section_text):
        rows.append({
            "number":   m.group(1).strip(),
            "category": m.group(2).strip(),
            "content":  m.group(3).strip(),
            "deadline": m.group(4).strip(),
            "progress": m.group(5).strip(),
        })

    return rows


def extract_action_items_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extracts action item rows from the meeting protocol PDF.
    Uses pdfplumber if available, falls back to pypdf text parsing.

    Returns list of raw row dicts — not yet scored or converted to TODOs.
    """
    if _PDF_BACKEND == "none":
        print("[NoteAnalyzer] No PDF backend available — skipping protocol parse.")
        return []

    try:
        if _PDF_BACKEND == "pdfplumber":
            rows = _parse_table_pdfplumber(pdf_path)
        else:
            rows = _parse_table_pypdf(pdf_path)

        print(f"[NoteAnalyzer] Extracted {len(rows)} raw rows from protocol PDF.")
        return rows

    except Exception as e:
        print(f"[NoteAnalyzer] PDF extraction failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# CHECKBOX DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _is_done(progress_str: str) -> bool:
    """
    Returns True if the progress cell indicates the item is completed.
    Handles: ☒  ✓  x  X  [x]  done  completed
    """
    s = progress_str.strip()
    return s in {"☒", "✓", "✗", "x", "X", "[x]", "[X]", "done", "completed", "1"}


# ─────────────────────────────────────────────────────────────────────────────
# DEADLINE RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_deadline(deadline_str: str) -> Optional[str]:
    """
    Converts a deadline string to an ISO date string (YYYY-MM-DD) or None.

    Handles:
        "Today"        → today's date
        "This week"    → end of current week (Sunday)
        "Next meeting" → None (no hard date)
        "TBD"          → None
        "2026-04-15"   → as-is
        "15/04/26"     → parsed
    """
    s = deadline_str.strip().lower()

    # Keyword mapping
    today = date.today()

    if s in ("today", "today ☒", "today ☐"):
        return today.isoformat()

    if s in ("this week", "this week ☒", "this week ☐"):
        # End of current week = Sunday
        days_until_sunday = 6 - today.weekday()
        return (today + timedelta(days=days_until_sunday)).isoformat()

    if s in ("next meeting", "tbd", "", "—", "-"):
        return None

    # Try ISO format
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", deadline_str.strip())
    if m:
        return deadline_str.strip()

    # Try DD/MM/YY or DD/MM/YYYY
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", deadline_str.strip())
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# TODO PERSISTENCE SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_todo_persistence(
    content: str,
    deadline: Optional[str],
    recurrence: int,
    urgency: str = "medium",
) -> str:
    """
    Decides whether a TODO should go to long-term SQL storage or
    just short-term session memory.

    Scoring (pure rule-based, no LLM):
        deadline present     → +3
        high urgency word    → +2
        urgency == "high"    → +2
        recurrence > 1       → +2
        urgency == "medium"  → +1

    Returns: "long_term" if score >= 3 else "short_term"
    """
    score = 0

    if deadline:
        score += 3

    words = set(content.lower().split())
    if words & HIGH_URGENCY_WORDS:
        score += 2

    if urgency == "high":
        score += 2
    elif urgency == "medium":
        score += 1

    if recurrence > 1:
        score += 2

    return "long_term" if score >= 3 else "short_term"


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL ROWS → TODO DICTS
# ─────────────────────────────────────────────────────────────────────────────

def protocol_rows_to_todos(
    rows: list[dict],
    project: str,
    source_file: str,
) -> list[dict]:
    """
    Converts raw PDF table rows into structured TODO dicts.
    Skips rows marked as done.

    Returns list of todo dicts ready for SQL insertion or short-term use.
    Each dict has keys matching the todos SQL table schema.
    """
    todos = []
    now = datetime.now().isoformat(timespec="seconds")

    for row in rows:
        # Skip completed items
        if _is_done(row.get("progress", "")):
            continue

        content = row.get("content", "").strip()
        if not content:
            continue

        category_raw = row.get("category", "").strip().lower()
        deadline_raw = row.get("deadline", "").strip()

        priority = CATEGORY_PRIORITY.get(category_raw, 3)
        deadline = _resolve_deadline(deadline_raw)

        # Urgency from category
        urgency = "high" if priority <= 2 else ("medium" if priority == 3 else "low")

        persistence = score_todo_persistence(
            content=content,
            deadline=deadline,
            recurrence=1,
            urgency=urgency,
        )

        todos.append({
            "id":          str(uuid.uuid4()),
            "project":     project,
            "source_file": source_file,
            "source_type": "meeting_protocol",
            "content":     content,
            "priority":    priority,
            "deadline":    deadline,
            "status":      "open",
            "first_seen":  now,
            "last_seen":   now,
            "recurrence":  1,
            "persistence": persistence,   # "long_term" | "short_term"
            "urgency":     urgency,
        })

    return todos


# ─────────────────────────────────────────────────────────────────────────────
# DAILY NOTE → TODO DICTS (LLM extraction)
# ─────────────────────────────────────────────────────────────────────────────

def extract_todos_from_note_llm(
    sections: dict[str, str],
    ollama,
    model: str,
    project: str,
    source_file: str,
) -> list[dict]:
    """
    Uses a focused LLM prompt to extract TODOs from lab notebook prose.
    Returns structured TODO dicts (same schema as protocol_rows_to_todos).

    Falls back to empty list if LLM fails or returns malformed JSON.
    """
    # Build compact input — combine sections
    text_parts = []
    for section in ["objectives", "methodology", "notes"]:
        val = (sections.get(section) or "").strip()
        if val:
            text_parts.append(f"[{section.upper()}]\n{val[:800]}")

    if not text_parts:
        return []

    combined = "\n\n".join(text_parts)
    prompt = _TODO_EXTRACTION_PROMPT + combined

    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = ""
        if isinstance(resp, dict):
            raw = ((resp.get("message") or {}).get("content") or "").strip()
        else:
            raw = getattr(getattr(resp, "message", None), "content", "") or ""

        raw = raw.strip()

        # Strip markdown fences if LLM included them
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("LLM did not return a JSON array")

    except Exception as e:
        print(f"[NoteAnalyzer] LLM TODO extraction failed: {e}")
        return []

    now = datetime.now().isoformat(timespec="seconds")
    todos = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        content = (item.get("content") or "").strip()
        if not content:
            continue

        category_raw = (item.get("category") or "experimental").lower()
        urgency = (item.get("urgency") or "medium").lower()
        deadline_hint = (item.get("deadline_hint") or "").strip()

        priority = CATEGORY_PRIORITY.get(category_raw, 3)
        deadline = _resolve_deadline(deadline_hint)

        persistence = score_todo_persistence(
            content=content,
            deadline=deadline,
            recurrence=1,
            urgency=urgency,
        )

        todos.append({
            "id":          str(uuid.uuid4()),
            "project":     project,
            "source_file": source_file,
            "source_type": "daily_note",
            "content":     content,
            "priority":    priority,
            "deadline":    deadline,
            "status":      "open",
            "first_seen":  now,
            "last_seen":   now,
            "recurrence":  1,
            "persistence": persistence,
            "urgency":     urgency,
        })

    print(f"[NoteAnalyzer] LLM extracted {len(todos)} TODOs from daily note.")
    return todos


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def _todo_key(content: str) -> str:
    """Normalised key for deduplication — lowercase, no punctuation."""
    return re.sub(r"[^a-z0-9 ]", "", content.lower().strip())


def merge_todos(
    protocol_todos: list[dict],
    note_todos: list[dict],
) -> list[dict]:
    """
    Merges two TODO lists.
    Meeting protocol items take priority on content collision.
    Daily note items that don't overlap are added after.

    Deduplication is fuzzy — if two items share 60%+ of words, 
    they're considered the same task.
    """
    merged = list(protocol_todos)  # protocol is ground truth
    protocol_keys = [set(_todo_key(t["content"]).split()) for t in protocol_todos]

    for note_todo in note_todos:
        note_words = set(_todo_key(note_todo["content"]).split())
        if not note_words:
            continue

        # Check overlap with each protocol item
        is_duplicate = False
        for p_words in protocol_keys:
            if not p_words:
                continue
            overlap = len(note_words & p_words) / max(len(note_words), len(p_words))
            if overlap >= 0.60:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(note_todo)

    # Sort: long_term first, then by priority
    merged.sort(key=lambda t: (
        0 if t["persistence"] == "long_term" else 1,
        t["priority"],
    ))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SQL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _get_db_connection():
    """
    Resolves sqlBrain.db relative to Buddy root.
    Walks up from this file until it finds a parent containing src/db/.
    Falls back to cwd-relative path (same as sql_brain.py) if not found.
    """
    import sqlite3

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "src" / "db" / "sqlBrain.db"
        if candidate.parent.exists():
            os.makedirs(str(candidate.parent), exist_ok=True)
            return sqlite3.connect(str(candidate))

    # Fallback — cwd-relative, matches sql_brain.py behaviour
    fallback = "src/db/sqlBrain.db"
    os.makedirs(os.path.dirname(fallback), exist_ok=True)
    return sqlite3.connect(fallback)


def save_todos_to_sql(todos: list[dict]) -> tuple[int, int]:
    """
    Saves long_term TODOs to the todos SQL table.
    Skips short_term TODOs (session-only context).

    For each item:
      - If same content + project already exists as 'open' → bump recurrence
      - Otherwise → insert new row

    Returns (inserted, bumped) counts.
    """
    conn = _get_db_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    inserted = 0
    bumped = 0

    for todo in todos:
        if todo.get("persistence") != "long_term":
            continue

        # Check for existing open item with same content
        cur.execute("""
            SELECT id, recurrence FROM todos
            WHERE status = 'open'
              AND project = ?
              AND LOWER(TRIM(content)) = LOWER(TRIM(?))
        """, (todo["project"], todo["content"]))
        existing = cur.fetchone()

        if existing:
            # Bump recurrence and update last_seen
            cur.execute("""
                UPDATE todos
                SET recurrence = recurrence + 1,
                    last_seen  = ?,
                    priority   = MIN(priority, ?)
                WHERE id = ?
            """, (now, todo["priority"], existing[0]))
            bumped += 1
            print(f"[NoteAnalyzer] Bumped recurrence: '{todo['content'][:60]}'")
        else:
            cur.execute("""
                INSERT INTO todos
                    (id, project, source_file, content, priority,
                     deadline, status, first_seen, last_seen, recurrence)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, 1)
            """, (
                todo["id"],
                todo["project"],
                todo["source_file"],
                todo["content"],
                todo["priority"],
                todo.get("deadline"),
                now,
                now,
            ))
            inserted += 1
            print(f"[NoteAnalyzer] Inserted TODO: '{todo['content'][:60]}'")

    conn.commit()
    conn.close()
    return inserted, bumped


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULE AWARENESS SHIM
# ─────────────────────────────────────────────────────────────────────────────

def scan_todo_deadlines(days_ahead: int = 3) -> list[dict]:
    """
    Reads open TODOs with deadlines from SQL and returns awareness items
    in the exact format ScheduleAwareness._make_item() produces.

    Drop this into schedule_awareness.py:

        from brain.projects.note_analyzer import scan_todo_deadlines

        # Inside startup_scan():
        todo_items = scan_todo_deadlines(days_ahead=3)
        all_items = conflict_items + past_items + upcoming_items + holiday_items + todo_items

        # Update counts dict:
        "todo_deadlines": len(todo_items),
    """
    conn = _get_db_connection()
    cur = conn.cursor()

    cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()

    cur.execute("""
        SELECT id, project, content, priority, deadline, recurrence
        FROM todos
        WHERE status   = 'open'
          AND deadline IS NOT NULL
          AND deadline <= ?
        ORDER BY deadline ASC, priority ASC
    """, (cutoff,))

    rows = cur.fetchall()
    conn.close()

    items = []
    today = date.today()

    for row in rows:
        todo_id, project, content, priority, deadline_str, recurrence = row

        try:
            deadline_date = date.fromisoformat(deadline_str)
        except ValueError:
            continue

        days_until = (deadline_date - today).days

        # Build natural timing string
        if days_until < 0:
            timing = f"{abs(days_until)} day{'s' if abs(days_until) != 1 else ''} overdue"
            aware_priority = "high"
        elif days_until == 0:
            timing = "today"
            aware_priority = "high"
        elif days_until == 1:
            timing = "tomorrow"
            aware_priority = "high"
        else:
            timing = f"in {days_until} days"
            aware_priority = "medium" if days_until <= 7 else "low"

        # Escalate if it recurs across multiple notes
        if recurrence > 1 and aware_priority != "high":
            aware_priority = "high"

        project_label = project.replace("_", " ").title() if project else "project"
        message = (
            f"Reminder — {project_label} TODO due {timing}: {content}."
        )

        items.append({
            "type":     "todo_deadline",
            "priority": aware_priority,
            "message":  message,
            "event":    {
                "id":        todo_id,
                "title":     content,
                "timestamp": f"{deadline_str} 00:00",
                "source":    "todos",
            },
            "data": {
                "days_until":  days_until,
                "project":     project,
                "recurrence":  recurrence,
            },
        })

    return items


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class NoteAnalyzer:
    """
    Orchestrates full note analysis for a project session.

    Usage inside ProjectHandler.handle_open_project():

        from brain.projects.note_analyzer import NoteAnalyzer

        analyzer = NoteAnalyzer(
            project_key="terahertz",
            ollama=self.ollama,
            model=self.model,
        )
        todos, short_term_context = analyzer.run(
            note_path=note_path,
            protocols_folder=cfg["protocols_folder"],
        )
        # todos         → already saved to SQL, list for logging
        # short_term_context → list of strings for session memory
    """

    def __init__(self, project_key: str, ollama, model: str = "llama3.2:latest"):
        self.project_key = project_key
        self.ollama = ollama
        self.model = model

    def run(
        self,
        note_path: Optional[Path],
        protocols_folder: Optional[str],
    ) -> tuple[list[dict], list[str]]:
        """
        Full analysis pipeline. Returns:
            todos             — all extracted TODOs (long_term already saved to SQL)
            short_term_context — plain strings for session STM (short_term items)
        """
        protocol_todos: list[dict] = []
        note_todos:     list[dict] = []

        # ── Step 1: Meeting protocol PDF ─────────────────────────────────────
        if protocols_folder:
            pdf_path = find_latest_meeting_protocol(protocols_folder)
            if pdf_path:
                raw_rows = extract_action_items_from_pdf(pdf_path)
                protocol_todos = protocol_rows_to_todos(
                    rows=raw_rows,
                    project=self.project_key,
                    source_file=pdf_path.name,
                )
                print(f"[NoteAnalyzer] Protocol TODOs: {len(protocol_todos)} open items.")
            else:
                print("[NoteAnalyzer] No meeting protocol found — skipping.")

        # ── Step 2: Daily note LLM extraction ────────────────────────────────
        if note_path and note_path.exists():
            from brain.projects.project_handler import read_note_sections
            sections = read_note_sections(note_path)
            if sections:
                note_todos = extract_todos_from_note_llm(
                    sections=sections,
                    ollama=self.ollama,
                    model=self.model,
                    project=self.project_key,
                    source_file=note_path.name,
                )
            else:
                print("[NoteAnalyzer] Daily note empty or unreadable — skipping LLM pass.")
        else:
            print("[NoteAnalyzer] No daily note path — skipping LLM extraction.")

        # ── Step 3: Merge (protocol takes priority) ───────────────────────────
        merged = merge_todos(protocol_todos, note_todos)
        print(f"[NoteAnalyzer] Merged TODO list: {len(merged)} items.")

        # ── Step 4: Save long_term items to SQL ───────────────────────────────
        long_term = [t for t in merged if t["persistence"] == "long_term"]
        short_term = [t for t in merged if t["persistence"] == "short_term"]

        if long_term:
            inserted, bumped = save_todos_to_sql(long_term)
            print(f"[NoteAnalyzer] SQL → {inserted} inserted, {bumped} bumped.")
        else:
            print("[NoteAnalyzer] No long-term TODOs to save.")

        # ── Step 5: Build short-term context strings ──────────────────────────
        short_term_context = [t["content"] for t in short_term]

        return merged, short_term_context

    def build_session_briefing(self, todos: list[dict]) -> str:
        """
        Builds a short TTS-ready briefing of open TODOs for the session start.
        Keeps it to 3 items max to avoid overwhelming.

        Example:
            "You have 5 open action items for the terahertz project.
             The top priorities are: test model with single real shot signal,
             document hyperparameter setup clearly, and run transmission dataset analysis."
        """
        open_todos = [t for t in todos if t.get("status", "open") == "open"]
        if not open_todos:
            return ""

        count = len(open_todos)
        project_label = self.project_key.replace("_", " ")

        top = open_todos[:3]
        top_contents = [t["content"].rstrip(".") for t in top]

        if len(top_contents) == 1:
            items_str = top_contents[0]
        elif len(top_contents) == 2:
            items_str = f"{top_contents[0]} and {top_contents[1]}"
        else:
            items_str = (
                f"{top_contents[0]}, {top_contents[1]}, "
                f"and {top_contents[2]}"
            )

        more = f" Plus {count - 3} more." if count > 3 else ""

        return (
            f"You have {count} open action item{'s' if count != 1 else ''} "
            f"for the {project_label} project. "
            f"Top priorities: {items_str}.{more}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== NoteAnalyzer — Self Test ===\n")

    # ── Test 1: PDF table extraction ─────────────────────────────────────────
    print("--- Test 1: PDF backend ---")
    print(f"  Backend: {_PDF_BACKEND}")

    if len(sys.argv) > 1:
        pdf_test = Path(sys.argv[1])
        if pdf_test.exists() and pdf_test.suffix.lower() == ".pdf":
            print(f"  Testing with: {pdf_test.name}")
            rows = extract_action_items_from_pdf(pdf_test)
            print(f"  Extracted {len(rows)} rows:")
            for r in rows:
                status = "☒ done" if _is_done(r["progress"]) else "☐ open"
                print(f"    [{status}] [{r['category']}] {r['content'][:70]} — {r['deadline']}")

            todos = protocol_rows_to_todos(rows, "terahertz", pdf_test.name)
            open_todos = [t for t in todos if t["status"] == "open"]
            print(f"\n  Open TODOs after filtering: {len(open_todos)}")
            for t in open_todos:
                print(f"    [p{t['priority']}] [{t['persistence']}] "
                      f"[{t['deadline'] or 'no deadline'}] {t['content'][:70]}")

    # ── Test 2: Deadline resolver ─────────────────────────────────────────────
    print("\n--- Test 2: Deadline resolver ---")
    test_deadlines = [
        "Today", "This week", "Next meeting", "TBD",
        "2026-04-20", "15/04/26", "",
    ]
    for d in test_deadlines:
        resolved = _resolve_deadline(d)
        print(f"  '{d}' → {resolved}")

    # ── Test 3: Persistence scorer ────────────────────────────────────────────
    print("\n--- Test 3: Persistence scorer ---")
    test_cases = [
        ("Test model with single real shot signal",   "2026-04-20", 1, "high"),
        ("Investigate dead Shot 2",                   None,          1, "medium"),
        ("Organise Sciebo folder",                    "2026-04-14", 1, "high"),
        ("Consider alternative augmentation",         None,          1, "low"),
        ("Fix critical crash in bandpass pipeline",   None,          3, "high"),
    ]
    for content, deadline, recurrence, urgency in test_cases:
        result = score_todo_persistence(content, deadline, recurrence, urgency)
        print(f"  [{result}] {content[:60]}")

    # ── Test 4: scan_todo_deadlines shim ──────────────────────────────────────
    print("\n--- Test 4: scan_todo_deadlines (reads from SQL) ---")
    try:
        items = scan_todo_deadlines(days_ahead=7)
        if items:
            for item in items:
                print(f"  [{item['priority'].upper()}] {item['message']}")
        else:
            print("  No deadline TODOs in database (expected if DB is empty).")
    except Exception as e:
        print(f"  Skipped — DB not accessible from this path: {e}")
        print("  This is normal when running outside the Buddy root directory.")

    print("\n=== Self test complete ===")