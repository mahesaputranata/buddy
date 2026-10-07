import re

EVENT_WORDS = [
    "meeting", "call", "appointment", "interview", "deadline", "exam",
    "class", "lecture", "flight", "train", "shopping", "groceries",
    "dentist", "doctor"
]

def guess_event_type(text: str) -> str | None:
    t = text.lower()
    for w in EVENT_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return w.title()
    return None

def extract_with_person(text: str) -> str | None:
    # "meeting with zaki" / "call with prof yamen"
    m = re.search(r"\bwith\s+([A-Z][a-zA-Z0-9_-]{1,})", text)
    if m:
        return m.group(1)
    return None

def build_schedule_title(user_text: str, extracted: dict | None = None) -> str:
    extracted = extracted or {}
    event_type = guess_event_type(user_text) or "Reminder"
    person = extract_with_person(user_text)
    loc = (extracted.get("location") or "").strip()

    # Prefer: Meeting with Zaki
    if person:
        return f"{event_type} with {person}"

    # If no person but location exists: Meeting at UDE
    if event_type != "Reminder" and loc:
        return f"{event_type} at {loc}"

    # Otherwise: just Meeting / Reminder
    return event_type
