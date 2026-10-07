"""
brain/adaptive/awareness/gcal_source.py

Google Calendar Source
======================
Read/write integration with Google Calendar API.
Gives the unified schedule layer the same interface as LocalScheduleSource.

SETUP (one time only):
----------------------
1. Go to https://console.cloud.google.com
2. Create a new project (e.g. "Buddy")
3. Enable "Google Calendar API"
4. Go to APIs & Services → Credentials
5. Create OAuth 2.0 Client ID → Desktop App
6. Download the JSON → save as:
   brain/adaptive/calendar/credentials.json
7. First time you run Buddy, a browser window opens for login
8. After login, token.json is saved automatically — never login again

Required packages:
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

Usage:
    from brain.adaptive.awareness.gcal_source import GCalSource

    source = GCalSource()

    # Check if connected
    if not source.is_available():
        print("Google Calendar not connected yet")

    # Get today's events
    events = source.get_today()

    # Get upcoming 7 days
    events = source.get_upcoming(days_ahead=7)

    # Create an event (from Buddy voice command)
    source.create_event(
        title="Dentist appointment",
        timestamp="2026-04-10 09:00",
        description="Dr. Schmidt",
        location="Zahnarzt Müller, Essen"
    )
"""

import os
import json
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
from pathlib import Path

TZ = ZoneInfo("Europe/Berlin")

# ── Credential paths ─────────────────────────────────────────────────────────
# Place your downloaded credentials.json here
_ADAPTIVE_DIR = Path(__file__).parent.parent  # brain/adaptive/
CREDENTIALS_PATH = _ADAPTIVE_DIR / "calendar" / "credentials.json"
TOKEN_PATH = _ADAPTIVE_DIR / "calendar" / "token.json"

# ── Google Calendar API scopes ───────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ── Your primary calendar ID ─────────────────────────────────────────────────
# "primary" means your main Google Calendar — change if you want a specific one
CALENDAR_ID = "primary"


class GCalSource:
    """
    Google Calendar read/write source.
    Falls back gracefully if credentials are not set up yet —
    Buddy will still work with local SQL only.

    All returned dicts have the same guaranteed shape as LocalScheduleSource:
    {
        "id":          str,          # Google event ID
        "title":       str,
        "description": str | None,
        "timestamp":   str,          # "YYYY-MM-DD HH:MM"
        "date":        str,          # "YYYY-MM-DD"
        "time":        str | None,   # "HH:MM"
        "location":    str | None,
        "source":      "gcal",       # always "gcal" for this source
        "gcal_link":   str | None,   # HTML link to open in browser
    }
    """

    SOURCE_NAME = "gcal"

    def __init__(self):
        self._service = None
        self._available = None  # None = not checked yet

    # ── Auth and connection ──────────────────────────────────────────────────

    def _get_service(self):
        """
        Returns an authenticated Google Calendar service object.
        Handles token refresh automatically.
        Raises GCalNotAvailableError if credentials are missing.
        """
        if self._service is not None:
            return self._service

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            raise GCalNotAvailableError(
                "Google API packages not installed. "
                "Run: pip install google-auth google-auth-oauthlib "
                "google-auth-httplib2 google-api-python-client"
            )

        if not CREDENTIALS_PATH.exists():
            raise GCalNotAvailableError(
                f"credentials.json not found at {CREDENTIALS_PATH}. "
                "Download it from Google Cloud Console and place it there."
            )

        creds = None

        # Load existing token if available
        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        # Refresh or do first-time OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # First time — opens browser for login
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CREDENTIALS_PATH), SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save token for future runs
            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_PATH.write_text(creds.to_json())
            print(f"[GCal] Token saved to {TOKEN_PATH}")

        self._service = build("calendar", "v3", credentials=creds)
        self._available = True
        print("[GCal] Connected to Google Calendar.")
        return self._service

    def is_available(self) -> bool:
        """
        Returns True if Google Calendar is configured and reachable.
        Safe to call without crashing — returns False if not set up.
        """
        if self._available is True:
            return True
        if self._available is False:
            return False

        try:
            self._get_service()
            self._available = True
            return True
        except (GCalNotAvailableError, Exception):
            self._available = False
            return False

    # ── Normalization ────────────────────────────────────────────────────────

    def _normalize_event(self, event: dict) -> dict:
        """
        Converts a raw Google Calendar event dict into Buddy's standard shape.
        Handles both timed events and all-day events.
        """
        title = event.get("summary", "Untitled").strip()
        description = (event.get("description") or "").strip() or None
        location = (event.get("location") or "").strip() or None
        gcal_link = event.get("htmlLink")
        event_id = event.get("id", "")

        # Google Calendar has two formats:
        # Timed event: {"dateTime": "2026-04-10T09:00:00+02:00"}
        # All-day event: {"date": "2026-04-10"}
        start = event.get("start", {})
        dt_str = start.get("dateTime") or start.get("date") or ""

        date_part = None
        time_part = None
        timestamp = None

        if "T" in dt_str:
            # Timed event — parse and convert to Berlin timezone
            try:
                dt = datetime.fromisoformat(dt_str)
                dt_berlin = dt.astimezone(TZ)
                date_part = dt_berlin.strftime("%Y-%m-%d")
                time_part = dt_berlin.strftime("%H:%M")
                timestamp = f"{date_part} {time_part}"
            except Exception:
                date_part = dt_str[:10] if len(dt_str) >= 10 else None
                timestamp = date_part
        else:
            # All-day event
            date_part = dt_str[:10] if len(dt_str) >= 10 else None
            timestamp = date_part

        return {
            "id":          event_id,
            "title":       title,
            "description": description,
            "timestamp":   timestamp,
            "date":        date_part,
            "time":        time_part,
            "location":    location,
            "source":      self.SOURCE_NAME,
            "gcal_link":   gcal_link,
        }

    # ── Timestamp helpers ────────────────────────────────────────────────────

    def _to_rfc3339(self, date_str: str) -> str:
        """Converts "YYYY-MM-DD" to RFC3339 start-of-day in Berlin timezone."""
        return f"{date_str}T00:00:00+01:00"

    def _end_of_day_rfc3339(self, date_str: str) -> str:
        """Converts "YYYY-MM-DD" to RFC3339 end-of-day in Berlin timezone."""
        return f"{date_str}T23:59:59+01:00"

    def _now(self) -> datetime:
        return datetime.now(TZ)

    def _today_str(self) -> str:
        return self._now().strftime("%Y-%m-%d")

    # ── Read API ─────────────────────────────────────────────────────────────

    def get_by_range(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 50,
    ) -> list[dict]:
        """
        Returns Google Calendar events between start_date and end_date inclusive.

        Args:
            start_date:  "YYYY-MM-DD"
            end_date:    "YYYY-MM-DD"
            max_results: max number of events to fetch (default 50)

        Returns:
            list of normalized event dicts sorted by timestamp
        """
        if not self.is_available():
            return []

        try:
            service = self._get_service()
            time_min = self._to_rfc3339(start_date)
            time_max = self._end_of_day_rfc3339(end_date)

            result = service.events().list(
                calendarId=CALENDAR_ID,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = result.get("items", [])
            return [self._normalize_event(e) for e in events]

        except Exception as e:
            print(f"[GCal] get_by_range error: {e}")
            return []

    def get_today(self) -> list[dict]:
        """Returns all Google Calendar events for today."""
        today = self._today_str()
        return self.get_by_range(today, today)

    def get_tomorrow(self) -> list[dict]:
        """Returns all Google Calendar events for tomorrow."""
        tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        return self.get_by_range(tomorrow, tomorrow)

    def get_upcoming(self, days_ahead: int = 7) -> list[dict]:
        """
        Returns Google Calendar events from today up to days_ahead days.

        Args:
            days_ahead: how many days forward to look (default 7)
        """
        today = self._today_str()
        end = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return self.get_by_range(today, end)

    def get_past_unresolved(self, hours_back: int = 24) -> list[dict]:
        """
        Returns Google Calendar events from the last N hours.
        Used by the watchdog for check-in prompts.
        """
        if not self.is_available():
            return []

        now = self._now()
        cutoff = now - timedelta(hours=hours_back)

        start_date = cutoff.strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        all_events = self.get_by_range(start_date, end_date)

        now_str = now.strftime("%Y-%m-%d %H:%M")
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M")

        past = [
            e for e in all_events
            if e.get("timestamp")
            and cutoff_str <= e["timestamp"] < now_str
        ]

        return sorted(past, key=lambda e: e.get("timestamp") or "", reverse=True)

    # ── Write API ────────────────────────────────────────────────────────────

    def create_event(
        self,
        title: str,
        timestamp: str,
        description: Optional[str] = None,
        location: Optional[str] = None,
        duration_minutes: int = 60,
    ) -> Optional[dict]:
        """
        Creates a new event in Google Calendar.

        Args:
            title:            Event title
            timestamp:        "YYYY-MM-DD HH:MM" — Buddy's standard format
            description:      Optional description
            location:         Optional location string
            duration_minutes: Event duration in minutes (default 60)

        Returns:
            Normalized event dict if successful, None if failed

        Example:
            event = source.create_event(
                title="Dentist appointment",
                timestamp="2026-04-10 09:00",
                description="Dr. Schmidt",
                location="Zahnarzt Müller, Essen"
            )
        """
        if not self.is_available():
            print("[GCal] Not available — event not created in Google Calendar.")
            return None

        try:
            service = self._get_service()

            # Parse timestamp
            dt_start = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            dt_start = dt_start.replace(tzinfo=TZ)
            dt_end = dt_start + timedelta(minutes=duration_minutes)

            event_body = {
                "summary": title,
                "start": {
                    "dateTime": dt_start.isoformat(),
                    "timeZone": "Europe/Berlin",
                },
                "end": {
                    "dateTime": dt_end.isoformat(),
                    "timeZone": "Europe/Berlin",
                },
            }

            if description:
                event_body["description"] = description
            if location:
                event_body["location"] = location

            created = service.events().insert(
                calendarId=CALENDAR_ID,
                body=event_body,
            ).execute()

            print(f"[GCal] Event created: {title} at {timestamp}")
            return self._normalize_event(created)

        except Exception as e:
            print(f"[GCal] create_event error: {e}")
            return None

    def delete_event(self, event_id: str) -> bool:
        """
        Deletes an event from Google Calendar by its ID.

        Args:
            event_id: Google Calendar event ID (from the "id" field)

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.is_available():
            return False

        try:
            service = self._get_service()
            service.events().delete(
                calendarId=CALENDAR_ID,
                eventId=event_id,
            ).execute()
            print(f"[GCal] Event deleted: {event_id}")
            return True
        except Exception as e:
            print(f"[GCal] delete_event error: {e}")
            return False

    def get_this_week(self) -> list[dict]:
        """Returns all Google Calendar events for the current week."""
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        return self.get_by_range(
            monday.strftime("%Y-%m-%d"),
            sunday.strftime("%Y-%m-%d"),
        )


# ── Custom exception ─────────────────────────────────────────────────────────

class GCalNotAvailableError(Exception):
    """Raised when Google Calendar credentials are missing or invalid."""
    pass


# ── Convenience singleton ────────────────────────────────────────────────────
# Import this for simple one-liner usage:
#   from brain.adaptive.awareness.gcal_source import gcal_source
#   events = gcal_source.get_today()

gcal_source = GCalSource()


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    source = GCalSource()

    print("=== Google Calendar Source — Self Test ===")
    print(f"credentials.json exists: {CREDENTIALS_PATH.exists()}")
    print(f"token.json exists:        {TOKEN_PATH.exists()}")
    print(f"is_available():           {source.is_available()}")

    if not source.is_available():
        print("\n[!] Google Calendar not connected yet.")
        print("    Setup steps:")
        print(f"    1. Place credentials.json at: {CREDENTIALS_PATH}")
        print("    2. Run this file again — browser will open for login")
        print("    3. After login, token.json is saved automatically")
    else:
        print(f"\n--- Today ({source._today_str()}) ---")
        today_events = source.get_today()
        if today_events:
            for e in today_events:
                loc = f" @ {e['location']}" if e.get("location") else ""
                print(f"  {e['time'] or 'all-day'}  {e['title']}{loc}  [source={e['source']}]")
        else:
            print("  (no events today)")

        print(f"\n--- Upcoming 7 days ---")
        upcoming = source.get_upcoming(days_ahead=7)
        if upcoming:
            for e in upcoming:
                print(f"  {e['date']}  {e['time'] or 'all-day'}  {e['title']}")
        else:
            print("  (no upcoming events)")

        print(f"\n--- Past unresolved (last 24h) ---")
        past = source.get_past_unresolved(hours_back=24)
        if past:
            for e in past:
                print(f"  {e['timestamp']}  {e['title']}  ← needs check-in")
        else:
            print("  (none)")

        print("\n--- Shape check ---")
        all_events = source.get_upcoming(days_ahead=30)
        if all_events:
            sample = all_events[0]
            required_keys = {"id", "title", "description", "timestamp",
                             "date", "time", "location", "source", "gcal_link"}
            missing = required_keys - set(sample.keys())
            if missing:
                print(f"  [!] Missing keys: {missing}")
            else:
                print(f"  All required keys present ✓")
                print(f"  source='{sample['source']}' ✓")
        else:
            print("  No events to verify shape.")