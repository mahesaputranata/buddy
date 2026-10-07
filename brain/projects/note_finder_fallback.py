"""
note_finder_fallback.py
───────────────────────
Drop-in replacement for find_latest_note() in project_handler.py.

Adds a two-stage fallback:
  Stage 1 — strict pattern match (existing logic, unchanged)
  Stage 2 — confidence scorer over all .txt/.md files in the folder
             → high confidence  (>= 0.75) : return candidate, caller asks user
             → low confidence   (< 0.75)  : open File Explorer, return None

New FSM state: "Awaiting_Note_Confirm"
  Buddy asks: "I found a file called Daily Note from March 13th — is that right?"
  yes → proceed with that path
  no  → open File Explorer, clear pending

How to integrate into project_handler.py:
─────────────────────────────────────────
1. Copy ALL functions below into project_handler.py
   (replace the existing find_latest_note() with find_latest_note_smart())

2. In handle_open_project(), replace:
       note_path = find_latest_note(notes_folder, note_pattern)

   with:
       note_result = find_latest_note_smart(notes_folder, note_pattern)
       note_path   = note_result["path"]       # None if not found or low conf
       note_conf   = note_result["confidence"] # float 0.0–1.0
       note_query  = note_result["ask_user"]   # bool — should Buddy confirm?
       note_spoken = note_result["spoken_name"] # e.g. "Daily Note from March 13th"

       if note_result["open_explorer"]:
           _open_file_explorer(notes_folder)
           # Build spoken response without note summary, skip NoteAnalyzer
           ...

       elif note_query:
           # Store candidate in pending payload, ask user to confirm
           self.mm.pending_payload["note_candidate"] = str(note_path)
           self.mm.pending_payload["note_spoken"]    = note_spoken
           self.mm.set_pending_topic("Awaiting_Note_Confirm")
           return (
               f"I found a file called {note_spoken} — is that the right one?"
           )
           # handle_resume_response() handles the yes/no

3. In handle_resume_response(), add before is_project_resume() check:
       if self.mm.get_pending_topic() == "Awaiting_Note_Confirm":
           return self._handle_note_confirm(user_input, conversation_manager)

4. Add _handle_note_confirm() method to ProjectHandler (included below).
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

CONF_AUTO_ACCEPT  = 1.00   # strict pattern match — no need to ask
CONF_ASK_USER     = 0.75   # good enough to present, but confirm with user
CONF_OPEN_EXPLORER = 0.74  # below this — open File Explorer immediately


# ─────────────────────────────────────────────────────────────────────────────
# MONTH NAMES (for spoken output)
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"

def _date_to_spoken(d: date) -> str:
    """Converts a date to a natural spoken string. e.g. 'March 13th, 2026'"""
    return f"{_MONTHS[d.month]} {_ordinal(d.day)}, {d.year}"


# ─────────────────────────────────────────────────────────────────────────────
# NUMBER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_numbers(stem: str) -> list[int]:
    """
    Extracts all 1-4 digit number sequences from a filename stem.
    '13_03_26' → [13, 3, 26]
    'Daily_Note_01_04_2026' → [1, 4, 2026]
    """
    return [int(n) for n in re.findall(r'\d{1,4}', stem)]


# ─────────────────────────────────────────────────────────────────────────────
# DATE INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def _try_as_date(numbers: list[int]) -> tuple[Optional[date], float]:
    """
    Tries to interpret a list of integers as a date.
    Tests every consecutive triple using plausible orderings.

    Returns (best_date, confidence) or (None, 0.0).

    Confidence is based on:
      - Which ordering matched (DD MM YY = 0.90, YYYY MM DD = 0.80)
      - How recent the date is (last 2 years = full score, older = 50% penalty)
    """
    if len(numbers) < 3:
        return None, 0.0

    today = date.today()
    best_date: Optional[date] = None
    best_conf = 0.0

    for i in range(len(numbers) - 2):
        a, b, c = numbers[i], numbers[i + 1], numbers[i + 2]

        # (day, month, year, base_confidence) orderings to try
        candidates = []

        # DD MM YY  — your primary format
        if c < 100:
            candidates.append((a, b, 2000 + c, 0.90))

        # DD MM YYYY
        if 2000 <= c <= 2100:
            candidates.append((a, b, c, 0.85))

        # YYYY MM DD  (ISO-ish)
        if 2000 <= a <= 2100:
            candidates.append((c, b, a, 0.80))

        for dd, mm, yyyy, base_conf in candidates:
            try:
                d = date(yyyy, int(mm), int(dd))
            except (ValueError, TypeError):
                continue

            # Recency scoring
            age_days = abs((today - d).days)
            if age_days <= 30:
                recency = 1.00
            elif age_days <= 365:
                recency = 0.90
            elif age_days <= 730:
                recency = 0.70
            else:
                recency = 0.50   # older than 2 years — unlikely to be "latest"

            conf = base_conf * recency

            if conf > best_conf:
                best_conf = conf
                best_date = d

    return best_date, best_conf


# ─────────────────────────────────────────────────────────────────────────────
# NAME BONUSES
# ─────────────────────────────────────────────────────────────────────────────

def _name_bonus(stem: str) -> float:
    """
    Small confidence bonus for filenames that look like daily notes.
    'Daily_Note_...' → +0.05
    'Note_...' or 'Daily_...' → +0.03
    """
    s = stem.lower()
    if "daily" in s and "note" in s:
        return 0.05
    if "daily" in s or "note" in s:
        return 0.03
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STRICT PATTERN MATCHER (existing logic, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _strict_match(filename: str, pattern: re.Pattern) -> Optional[date]:
    """
    Tries the strict project-defined regex pattern against the filename.
    Returns a date if matched, None otherwise.
    """
    m = pattern.match(Path(filename).stem)
    if not m:
        return None
    try:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(2000 + yy, mm, dd)
    except (ValueError, IndexError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FINDER
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_note_smart(
    notes_folder: str,
    pattern: re.Pattern,
) -> dict:
    """
    Smart note finder with two-stage fallback.

    Returns a result dict:
    {
        "path":         Path | None,   # path to the best candidate
        "confidence":   float,         # 0.0 – 1.00
        "file_date":    date | None,   # parsed date from filename
        "spoken_name":  str,           # e.g. "Daily Note from March 13th, 2026"
        "ask_user":     bool,          # True if Buddy should confirm with user
        "open_explorer": bool,         # True if Buddy should open File Explorer
        "all_candidates": list[dict],  # all scored candidates (for stepping)
    }
    """
    folder = Path(notes_folder)
    empty_result = {
        "path":           None,
        "confidence":     0.0,
        "file_date":      None,
        "spoken_name":    "an unknown file",
        "ask_user":       False,
        "open_explorer":  True,
        "all_candidates": [],
    }

    if not folder.exists():
        print(f"[NoteFinder] Folder not found: {notes_folder}")
        return empty_result

    # ── Score every .txt and .md file in the folder ──────────────────────────
    scored: list[dict] = []

    for f in folder.iterdir():
        if f.suffix.lower() not in (".txt", ".md"):
            continue

        stem = f.stem
        strict_date = _strict_match(f.name, pattern)

        if strict_date:
            # Strict pattern match — highest possible confidence
            confidence = CONF_AUTO_ACCEPT
            file_date  = strict_date
        else:
            # Fallback — number extraction + date interpretation
            numbers    = _extract_numbers(stem)
            file_date, confidence = _try_as_date(numbers)
            if file_date:
                confidence = min(confidence + _name_bonus(stem), 1.0)

        if file_date is None:
            continue   # no date found in this filename at all

        scored.append({
            "path":       f,
            "confidence": round(confidence, 3),
            "file_date":  file_date,
            "stem":       stem,
        })

    if not scored:
        print(f"[NoteFinder] No dateable files found in {notes_folder}")
        return empty_result

    # Sort: most recent date first, then highest confidence
    scored.sort(key=lambda x: (x["file_date"], x["confidence"]), reverse=True)

    best = scored[0]
    spoken_name = f"Daily Note from {_date_to_spoken(best['file_date'])}"

    print(
        f"[NoteFinder] Best candidate: {best['path'].name} "
        f"(conf={best['confidence']:.2f}, date={best['file_date']})"
    )

    if best["confidence"] >= CONF_ASK_USER:
        # Good enough to present — ask user only if not an exact match
        ask_user = best["confidence"] < CONF_AUTO_ACCEPT
        return {
            "path":           best["path"],
            "confidence":     best["confidence"],
            "file_date":      best["file_date"],
            "spoken_name":    spoken_name,
            "ask_user":       ask_user,
            "open_explorer":  False,
            "all_candidates": scored,
        }

    # Confidence too low — open File Explorer
    print(
        f"[NoteFinder] Confidence {best['confidence']:.2f} below threshold "
        f"— will open File Explorer."
    )
    return {
        "path":           None,
        "confidence":     best["confidence"],
        "file_date":      best["file_date"],
        "spoken_name":    spoken_name,
        "ask_user":       False,
        "open_explorer":  True,
        "all_candidates": scored,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FILE EXPLORER LAUNCHER
# ─────────────────────────────────────────────────────────────────────────────

def open_file_explorer(folder_path: str) -> bool:
    """
    Opens File Explorer at the given folder path.
    Works on Windows (explorer), macOS (open), Linux (xdg-open).
    Returns True if launched without error.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[NoteFinder] Cannot open explorer — folder not found: {folder_path}")
        return False

    try:
        import platform
        system = platform.system()
        if system == "Windows":
            os.startfile(str(folder))
        elif system == "Darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
        print(f"[NoteFinder] File Explorer opened at: {folder_path}")
        return True
    except Exception as e:
        print(f"[NoteFinder] Failed to open File Explorer: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# FSM HANDLER — add as method to ProjectHandler
# ─────────────────────────────────────────────────────────────────────────────

NOTE_CONFIRM_STATE = "Awaiting_Note_Confirm"

# Paste this method into the ProjectHandler class:
_NOTE_CONFIRM_METHOD = '''
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
        from brain.adaptive.agent_core import is_yes, is_no  # reuse existing helpers

        payload = self.mm.pending_payload or {}
        notes_folder   = payload.get("notes_folder", "")
        note_candidate = payload.get("note_candidate")   # str path
        note_spoken    = payload.get("note_spoken", "that file")
        project_key    = payload.get("project_key", "")
        topic_name     = payload.get("topic_name", "")
        note_date      = payload.get("note_date", "")

        if is_yes(user_input):
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

        # No or unclear — open File Explorer
        self.mm.set_pending_topic(None)
        self.mm.pending_payload = None

        if notes_folder:
            open_file_explorer(notes_folder)
            return (
                "No problem — I have opened the notes folder so you can "
                "find the right file. Just let me know the date when you are ready."
            )

        return "Okay, I will skip the note for now. What would you like to work on?"
'''


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED handle_open_project SNIPPET
# Replace the "Step 1: Read latest note" block in ProjectHandler
# ─────────────────────────────────────────────────────────────────────────────

_HANDLE_OPEN_PROJECT_SNIPPET = '''
        # ── Step 1: Read latest note (smart finder with fallback) ──────────
        from brain.projects.note_finder_fallback import (
            find_latest_note_smart, open_file_explorer, NOTE_CONFIRM_STATE,
        )
        from brain.projects.note_analyzer import NoteAnalyzer

        note_result = find_latest_note_smart(notes_folder, note_pattern)
        note_path   = note_result["path"]

        # Case A: open File Explorer — confidence too low to guess
        if note_result["open_explorer"]:
            open_file_explorer(notes_folder)
            self.mm.set_pending_topic(self.PENDING_STATE)
            self.mm.pending_payload = {
                "project_key":   project_key,
                "topic_name":    topic_name,
                "note_date":     "",
                "notes_folder":  notes_folder,
            }
            return (
                f"Opening the {display_name} project. {matlab_line} "
                f"I couldn\'t find your latest daily note — "
                f"I\'ve opened the notes folder so you can point me to it. "
                f"Would you like to continue without the note summary for now?"
            )

        # Case B: found with borderline confidence — ask user to confirm
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

        # Case C: confident match — proceed directly (existing happy path)
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
            summary       = "I couldn\'t find any daily notes for this project."
            note_date     = "unknown"
            todo_briefing = ""
'''


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED handle_resume_response SNIPPET
# Add this check at the very top of handle_resume_response()
# ─────────────────────────────────────────────────────────────────────────────

_RESUME_RESPONSE_SNIPPET = '''
        # ── Note confirmation intercept (must be first) ───────────────────
        if self.mm.get_pending_topic() == NOTE_CONFIRM_STATE:
            return self._handle_note_confirm(user_input, conversation_manager)
'''


# ─────────────────────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, shutil

    print("=== Note Finder Fallback — Self Test ===\n")

    # Build a temp folder with realistic filenames
    tmp = Path(tempfile.mkdtemp())
    test_files = [
        "Daily_Note_13_03_26.txt",      # strict match — should be 1.00
        "Daily_Note_01_04_26.txt",      # strict match — most recent
        "Note_2026_02_15.txt",          # ISO-ish — fallback
        "THz_update_09_02_26.md",       # different prefix
        "random_no_date.txt",           # no date — should be skipped
        "old_note_12_06_23.txt",        # old date — low recency
    ]
    for fname in test_files:
        (tmp / fname).write_text("test content")

    # Use the same pattern as project_handler.py
    pattern = re.compile(r"^Daily_Note_(\d{2})_(\d{2})_(\d{2})", re.IGNORECASE)

    print("--- All candidates scored ---")
    result = find_latest_note_smart(str(tmp), pattern)

    print(f"\nAll candidates:")
    for c in result["all_candidates"]:
        print(f"  {c['path'].name:<40} conf={c['confidence']:.2f}  date={c['file_date']}")

    print(f"\nBest result:")
    print(f"  path:          {result['path']}")
    print(f"  confidence:    {result['confidence']:.2f}")
    print(f"  file_date:     {result['file_date']}")
    print(f"  spoken_name:   {result['spoken_name']}")
    print(f"  ask_user:      {result['ask_user']}")
    print(f"  open_explorer: {result['open_explorer']}")

    # Test with empty folder
    print("\n--- Empty folder test ---")
    empty = Path(tempfile.mkdtemp())
    result2 = find_latest_note_smart(str(empty), pattern)
    print(f"  open_explorer: {result2['open_explorer']} (expected True)")
    print(f"  path:          {result2['path']} (expected None)")

    # Test low-confidence folder (no files matching strict pattern)
    print("\n--- Low confidence folder (no strict matches) ---")
    low_conf_dir = Path(tempfile.mkdtemp())
    (low_conf_dir / "random_text_file.txt").write_text("content")
    (low_conf_dir / "another_file.md").write_text("content")
    result3 = find_latest_note_smart(str(low_conf_dir), pattern)
    print(f"  open_explorer: {result3['open_explorer']} (expected True — no dates)")

    # Cleanup
    shutil.rmtree(tmp)
    shutil.rmtree(empty)
    shutil.rmtree(low_conf_dir)

    print("\n=== Self test complete ===")
