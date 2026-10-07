from datetime import datetime

def clean_text(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in ("none", "null"):
        return ""
    return x

def format_ts_speech(ts: str) -> str:
    """
    Input:  'YYYY-MM-DD HH:MM'
    Output: 'Wednesday, Dec 24 at 12:00 PM'
    """
    ts = clean_text(ts)
    if not ts:
        return ""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        s = dt.strftime("%A, %b %d at %I:%M %p")
        return s.replace(" 0", " ")
    except Exception:
        return ts  # fallback
