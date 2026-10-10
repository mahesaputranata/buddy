# sql_brain.py
import sqlite3
import os
from brain.core.prompts import extract_schedule_details
import json
import ollama
import difflib
from datetime import datetime, timedelta
import re
from brain.utils.cleaning_date import clean_text, format_ts_speech
import uuid

def safe_json_parse(raw: str):
    """Parse JSON safely. Returns dict/list on success, else None."""
    if raw is None:
        return None
    raw = raw.strip()

    # Fast path: strict JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    if not m:
        return None

    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def delete_schedule_by_id(schedule_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def delete_schedule_by_title(title: str) -> list[dict]:
    """Returns list of matching schedules for confirmation before deleting."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, timestamp, location 
        FROM schedules 
        WHERE LOWER(title) LIKE LOWER(?)
    """, (f"%{title}%",))
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1],
             "timestamp": r[2], "location": r[3]} for r in rows]

def delete_schedules_by_ids(ids: list[int]) -> str:
    if not ids:
        return "No schedules to delete."
    conn = get_connection()
    cur = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    cur.execute(f"DELETE FROM schedules WHERE id IN ({placeholders})", ids)
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return f"Deleted {affected} schedule{'s' if affected != 1 else ''}."


def get_connection():
    db_path = "src/db/sqlBrain.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def init_schedule_table():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            timestamp TEXT,
            location TEXT
        )
    """)
    conn.commit()
    conn.close()

def create_schedule(title, desc, timestamp, location=None):
    print("[DEBUG] Creating Schedule..")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO schedules (title, description, timestamp, location)
        VALUES (?, ?, ?, ?)
    """, (title, desc, timestamp, location))
    print("[DEBUG] Schedule Created")
    conn.commit()
    conn.close()

def get_all_schedules():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM schedules ORDER BY timestamp ASC")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_all_schedules_as_dicts():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, description, timestamp, location
        FROM schedules
        ORDER BY timestamp ASC
    """)
    rows = cur.fetchall()
    conn.close()

    schedules = []
    for row in rows:
        schedules.append({
            "id": row[0],  # keep as int
            "title": clean_text(row[1]) or "Untitled event",
            "description": clean_text(row[2]),
            "timestamp": clean_text(row[3]),
            "location": clean_text(row[4]),
        })
    return schedules


def validate_schedule(extracted: dict):
    required = ["timestamp", "title"]
    missing = [f for f in required if not extracted.get(f)]
    return (len(missing) == 0), missing


def retrieve_schedule_by_date_sql(date_yyyy_mm_dd: str):
    schedules = get_all_schedules_as_dicts()
    matches = []

    for s in schedules:
        ts = clean_text(s.get("timestamp"))
        if ts and ts.startswith(date_yyyy_mm_dd):
            matches.append(s)

    if not matches:
        return f"You have no schedules on {date_yyyy_mm_dd}."

    matches.sort(key=lambda s: s["timestamp"])

    if len(matches) == 1:
        s = matches[0]
        when = format_ts_speech(s["timestamp"])
        loc = clean_text(s.get("location"))
        if loc:
            return f"On {date_yyyy_mm_dd}, you have: {s['title']}, at {when}, at {loc}."
        return f"On {date_yyyy_mm_dd}, you have: {s['title']}, at {when}."

    parts = []
    for m in matches:
        when = format_ts_speech(m["timestamp"])
        loc = clean_text(m.get("location"))
        if loc:
            parts.append(f"{m['title']}, at {when}, at {loc}.")
        else:
            parts.append(f"{m['title']}, at {when}.")

    return f"Here are your schedules on {date_yyyy_mm_dd}. " + " ".join(parts)


##SQL MANAGEMENT BRAIN SYSTEM

def retrieve_schedule_by_week_sql(start_date: str, end_date: str):
    print(f"[DEBUG] Fetching schedules between {start_date} and {end_date}")

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date, "%Y-%m-%d")

        schedules = get_all_schedules_as_dicts()
        matches = []

        for sched in schedules:
            ts = clean_text(sched.get("timestamp"))
            if not ts:
                continue

            try:
                sched_time = datetime.strptime(ts, "%Y-%m-%d %H:%M")
                if start <= sched_time <= end:
                    # sanitize fields once
                    sched["title"] = clean_text(sched.get("title")) or "Untitled event"
                    sched["location"] = clean_text(sched.get("location"))
                    sched["timestamp"] = ts
                    matches.append(sched)
            except Exception:
                continue

        if not matches:
            return "You have no schedules for that week."

        # Sort by timestamp ascending (nice for speech)
        matches.sort(key=lambda s: s["timestamp"])

        # Build TTS-friendly response
        if len(matches) == 1:
            s = matches[0]
            when = format_ts_speech(s["timestamp"])
            loc = s["location"]
            if loc:
                return f"You have one schedule that week: {s['title']}, on {when}, at {loc}."
            return f"You have one schedule that week: {s['title']}, on {when}."

        parts = []
        for s in matches:
            when = format_ts_speech(s["timestamp"])
            loc = s["location"]
            if loc:
                parts.append(f"{s['title']}, on {when}, at {loc}.")
            else:
                parts.append(f"{s['title']}, on {when}.")

        return "Here are your schedules for that week. " + " ".join(parts)

    except Exception as e:
        print("[ERROR] Failed to retrieve weekly schedule:", e)
        return "Something went wrong while retrieving your schedule for that week."


def retrieve_schedule_by_month_sql(month_yyyy_mm: str):
    """
    month_yyyy_mm: "YYYY-MM"
    matches schedules whose timestamp begins with "YYYY-MM-"
    """
    schedules = get_all_schedules_as_dicts()
    matches = []

    prefix = month_yyyy_mm + "-"  # "YYYY-MM-"
    for s in schedules:
        ts = s.get("timestamp")  # "YYYY-MM-DD HH:MM"
        if ts and ts.startswith(prefix):
            matches.append(s)

    if not matches:
        return f"I couldn't find any schedules in **{month_yyyy_mm}**."

    # sort just in case
    def _dt(sched):
        try:
            return datetime.strptime(sched["timestamp"], "%Y-%m-%d %H:%M")
        except:
            return datetime.max

    matches.sort(key=_dt)

    lines = []
    for m in matches:
        loc = f" in **{m['location']}**" if m.get("location") else ""
        lines.append(f"- **{m['title']}** at **{m['timestamp']}**{loc}")

    return f"Here’s your schedule for **{month_yyyy_mm}**:\n\n" + "\n".join(lines)

def retrieve_schedule_by_year_sql(year_yyyy: str):
    """
    year_yyyy: "YYYY"
    matches schedules whose timestamp begins with "YYYY-"
    """
    schedules = get_all_schedules_as_dicts()
    matches = []

    prefix = year_yyyy + "-"  # "YYYY-"
    for s in schedules:
        ts = s.get("timestamp")
        if ts and ts.startswith(prefix):
            matches.append(s)

    if not matches:
        return f"I couldn't find any schedules in **{year_yyyy}**."

    def _dt(sched):
        try:
            return datetime.strptime(sched["timestamp"], "%Y-%m-%d %H:%M")
        except:
            return datetime.max

    matches.sort(key=_dt)

    lines = []
    for m in matches:
        loc = f" in **{m['location']}**" if m.get("location") else ""
        lines.append(f"- **{m['title']}** at **{m['timestamp']}**{loc}")

    return f"Here’s your schedule for **{year_yyyy}**:\n\n" + "\n".join(lines)

def extractScheduleDetails(user_input):
    extracting = extract_schedule_details(user_input)
    response = ollama.generate(model="llama3.2:latest", prompt=extracting)
    extracted = safe_json_parse(response['response'])
    print("[DEBUG] Extracted in JSON Format: ", extracted)

    return extracted

def smart_schedule_recall(extracted):
    print("[DEBUG] Smart Schedule Recall Start...")

    all_schedules = get_all_schedules_as_dicts()

    if not all_schedules:
        return "Sorry, it seems you don't have any schedule yet"

    print("[DEBUG] Searching Schedule Start...")

    # declare a temporary array
    match_scores = []
    for sched in all_schedules:
        score = 0
        matched_fields = []

        for field in ["title", "description", "location", "timestamp"]:
            query_val = extracted.get(field)
            db_val = sched.get(field)

            if not query_val or not db_val:
                continue

            if field == "timestamp":
                if query_val == db_val:
                    score += 2
                    matched_fields.append(field)

            else:
                ratio = difflib.SequenceMatcher(None, query_val.lower(), db_val.lower()).ratio()
                if ratio > 0.6:
                    score += 1
                    matched_fields.append(field)


        match_scores.append({"score": score, "matched_fields": matched_fields, "schedule": sched})

    match_scores = sorted(match_scores, key=lambda x: x["score"], reverse=True)


    top_matches = [m for m in match_scores if m["score"] > 0]
    if not top_matches:
        return "I couldn't find any schedule that matches what you said. Could you rephrase?"

    if len(top_matches) == 1:
        print("[DEBUG] Only one similarity detected")
        s = top_matches[0]["schedule"]
        return f"You have a schedule: **{s['title']}** at **{s['timestamp']}** in **{s['location']}**."

    if top_matches and top_matches[0]["score"] >= 3:
        s = top_matches[0]["schedule"]
        return f"I found your exact schedule: **{s['title']}** at **{s['timestamp']}** in **{s['location']}**."

    print("[DEBUG] Many Similarities Detected")
    for i, m in enumerate(top_matches[:5]):
        print(f"[{i+1}] {m['schedule']}")

    descriptions = {}
    for match in top_matches:
        desc = match["schedule"].get("description", "unknown").lower()
        descriptions[desc] = descriptions.get(desc, 0) + 1

    common_desc = sorted(descriptions.items(), key=lambda x: x[1], reverse=True)[0][0]

    print(f"It looks like you have multiple schedules related to **{common_desc}**.\n")
    # return candidates to Buddy for pending-state disambiguation
    top = top_matches[:5]
    options = []
    for i, m in enumerate(top, 1):
        s = m["schedule"]
        options.append(f"{i}) {s['title']} @ {s['timestamp']} ({s.get('location') or '—'})")

    return {
        "status": "need_clarification",
        "candidates": [m["schedule"] for m in top],
        "question": "I found multiple matching schedules. Which one did you mean?\n" + "\n".join(options) + "\nReply with 1-5."
    }


def create_todo(project: str, content: str, priority: int = 3,
                deadline: str = None, source_file: str = "voice") -> str:
    """Insert a new TODO. Returns the new id."""
    conn = get_connection()
    cur = conn.cursor()
    todo_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO todos (id, project, source_file, content, priority,
                           deadline, status, first_seen, last_seen, recurrence)
        VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, 1)
    """, (todo_id, project, source_file, content, priority, deadline, now, now))
    conn.commit()
    conn.close()
    return todo_id


def get_open_todos(project: str = None) -> list[dict]:
    """Return all open TODOs, optionally filtered by project."""
    conn = get_connection()
    cur = conn.cursor()
    if project:
        cur.execute("""
            SELECT id, project, content, priority, deadline, recurrence
            FROM todos WHERE status = 'open' AND project = ?
            ORDER BY priority ASC, deadline ASC NULLS LAST
        """, (project,))
    else:
        cur.execute("""
            SELECT id, project, content, priority, deadline, recurrence
            FROM todos WHERE status = 'open'
            ORDER BY priority ASC, deadline ASC NULLS LAST
        """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "project": r[1], "content": r[2],
         "priority": r[3], "deadline": r[4], "recurrence": r[5]}
        for r in rows
    ]


def mark_todo_done(todo_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE todos SET status = 'done' WHERE id = ?", (todo_id,))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def bump_todo_recurrence(content: str, project: str) -> bool:
    """
    Called by NoteAnalyzer when the same TODO appears in a new note.
    Increments recurrence and updates last_seen.
    Returns True if a matching open TODO was found and updated.
    """
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        UPDATE todos
        SET recurrence = recurrence + 1, last_seen = ?
        WHERE status = 'open'
          AND project = ?
          AND LOWER(content) = LOWER(?)
    """, (now, project, content))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok


# ── user_profile ──────────────────────────────────────────────────────────────

def get_profile(key: str) -> str | None:
    """Retrieve a single profile value by key."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM user_profile WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def set_profile(key: str, value: str, confidence: float = 1.0) -> None:
    """Upsert a profile value."""
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO user_profile (key, value, confidence, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value        = excluded.value,
            confidence   = excluded.confidence,
            last_updated = excluded.last_updated
    """, (key, value, confidence, now))
    conn.commit()
    conn.close()


# ── reminders ────────────────────────────────────────────────────────────────

def create_reminder(content: str, project: str = None,
                    trigger_type: str = "session_start",
                    trigger_value: str = "any",
                    priority: int = 2) -> str:
    """Insert a new reminder. Returns the new id."""
    conn = get_connection()
    cur = conn.cursor()
    reminder_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    cur.execute("""
        INSERT INTO reminders (id, content, project, trigger_type,
                               trigger_value, priority, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
    """, (reminder_id, content, project, trigger_type, trigger_value, priority, now))
    conn.commit()
    conn.close()
    return reminder_id


def get_active_reminders(trigger_type: str = None,
                         trigger_value: str = None) -> list[dict]:
    """Return active reminders, optionally filtered by trigger."""
    conn = get_connection()
    cur = conn.cursor()
    query = "SELECT id, content, project, trigger_type, trigger_value, priority FROM reminders WHERE status = 'active'"
    params = []
    if trigger_type:
        query += " AND trigger_type = ?"
        params.append(trigger_type)
    if trigger_value:
        query += " AND trigger_value = ?"
        params.append(trigger_value)
    query += " ORDER BY priority ASC"
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "content": r[1], "project": r[2],
         "trigger_type": r[3], "trigger_value": r[4], "priority": r[5]}
        for r in rows
    ]


def dismiss_reminder(reminder_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE reminders SET status = 'dismissed' WHERE id = ?", (reminder_id,))
    ok = cur.rowcount > 0
    conn.commit()
    conn.close()
    return ok

