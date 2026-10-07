# time_context.py
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = "Europe/Berlin"
# prompting for know what day is now
def now_context():
    now = datetime.now(ZoneInfo(TZ))
    return {
        "tz": TZ,
        "now_iso": now.isoformat(timespec="seconds"),
        "today": now.date().isoformat(),
        "weekday": now.strftime("%A"),
    }

def with_time_context(prompt: str) -> str:
    ctx = now_context()
    return (
        prompt.strip()
        + "\n\n"
        + f"[TIME_CONTEXT]\n"
          f"- timezone: {ctx['tz']}\n"
          f"- now: {ctx['now_iso']}\n"
          f"- today: {ctx['today']} ({ctx['weekday']})\n"
          f"[/TIME_CONTEXT]\n"
          f"Rules: When user says relative dates (today/tomorrow/next week), convert using TIME_CONTEXT.\n"
    )
