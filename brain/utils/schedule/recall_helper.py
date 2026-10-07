import re
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Berlin")

def parse_timestamp(ts: str) -> datetime | None:
    """
    Expecting 'YYYY-MM-DD HH:MM' (like '2025-12-19 10:00').
    Handles duplicate date prefix: '2026-04-05 2026-04-05 15:00'
    """
    if not ts:
        return None
    
    ts = ts.strip()

    # ── Fix: deduplicate date prefix ─────────────────────────────────────
    # Catches '2026-04-05 2026-04-05 15:00' → '2026-04-05 15:00'
    dupe = re.match(
        r"^(\d{4}-\d{2}-\d{2})\s+\1\s+(\d{1,2}:\d{2})$", ts
    )
    if dupe:
        ts = f"{dupe.group(1)} {dupe.group(2)}"

    # ── Primary format ────────────────────────────────────────────────────
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        return dt.replace(tzinfo=TZ)
    except ValueError:
        pass

    # ── Fallback formats ──────────────────────────────────────────────────
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=TZ)
        except ValueError:
            continue

    return None

def build_general_template(stored_memory: list[dict], now: datetime) -> tuple[str, list[str]]:
    """
    Returns (template_text, required_facts_list).

    template_text: one-paragraph deterministic summary (source of truth).
    required_facts_list: tokens we require in any LLM rephrase (fact-preservation guard).
    """

    def fmt_event_sentence(e: dict) -> str:
        dt: datetime = e["dt"]
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")

        title = e["title"]
        loc = f" at {e['location']}" if e["location"] else ""
        desc = f" ({e['description']})" if e["description"] else ""

        return f"{date_str} at {time_str}: {title}{loc}{desc}"

    # 1) Normalize rows -> events
    events: list[dict] = []
    for item in stored_memory:
        dt = parse_timestamp(item.get("timestamp", ""))  # your existing parser
        if dt is None:
            # If timestamp is required in your DB, you can skip this.
            # Keeping it safe in case of dirty rows.
            continue

        # Ensure tz-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)

        events.append({
            "id": item.get("id"),
            "title": (item.get("title") or "Untitled").strip(),
            "description": (item.get("description") or "").strip(),
            "location": (item.get("location") or "").strip(),
            "dt": dt,
        })

    # 2) Split deterministically
    upcoming = [e for e in events if e["dt"] >= now]
    passed = [e for e in events if e["dt"] < now]

    upcoming.sort(key=lambda e: e["dt"])
    passed.sort(key=lambda e: e["dt"], reverse=True)

    # 3) Build one-paragraph template
    parts: list[str] = []
    parts.append(f"As of {now.strftime('%Y-%m-%d %H:%M')} (Europe/Berlin), here’s your schedule.")

    if upcoming:
        upcoming_txt = "; ".join(fmt_event_sentence(e) for e in upcoming)
        parts.append(f"Upcoming: {upcoming_txt}.")
    else:
        parts.append("Upcoming: none.")

    # Limit passed events to avoid long rambles
    if passed:
        passed_txt = "; ".join(fmt_event_sentence(e) for e in passed[:3])
        parts.append(f"Recently passed: {passed_txt}.")
    else:
        parts.append("Recently passed: none.")

    template = " ".join(parts).strip()

    # 4) Required facts for guard
    required: list[str] = []
    # Require at least these anchors so LLM can’t “forget” key info:
    required.append(now.strftime("%Y-%m-%d"))  # date anchor (optional)

    if upcoming:
        for e in upcoming:
            required.append(e["dt"].strftime("%Y-%m-%d"))
            required.append(e["dt"].strftime("%H:%M"))
            required.append(e["title"])
            if e["location"]:
                required.append(e["location"])
    else:
        required.append("Upcoming: none")

    return template, required

def rewrite_prompt_for_llm(template_text: str) -> str:
    return f"""
Rewrite the following schedule summary to sound like you speak.

Rules:
- Keep ALL dates, times, event titles, and locations exactly the same.
- Do not add or remove events.
- Output must be ONE paragraph (no line breaks).
- Do not use bullet points, numbering, markdown, or JSON.
- Return plain text only.

Text to rewrite:
{template_text}
""".strip()

def tts_sanitize_light(text: str) -> str:
    t = text.strip()
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"[*_`]+", "", t)

    # kill bullets if the model still outputs them
    t = re.sub(r"(?m)^\s*[-•*]\s+", "", t)

    # collapse ALL whitespace into single spaces (=> one paragraph)
    t = re.sub(r"\s+", " ", t).strip()

    return t


def preserves_facts(text: str, required: list[str]) -> bool:
    low = text.lower()
    # allow slight variation: compare lowercased required tokens
    return all(r.lower() in low for r in required if r.strip())


def handle_general_schedule(ollama, stored_memory: list[dict], current_dt: datetime) -> str:
    template_text, required = build_general_template(stored_memory, current_dt)

    prompt = rewrite_prompt_for_llm(template_text)
    try:
        response = ollama.generate(model="llama3.2:latest", prompt=prompt)
        llm_out = (response.get("response") or "").strip()
    except Exception:
        llm_out = ""

    llm_out = tts_sanitize_light(llm_out)

    # Guard: must preserve required facts, otherwise fallback
    if not llm_out or not preserves_facts(llm_out, required):
        return template_text

    return llm_out
