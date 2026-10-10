from __future__ import annotations
import json
import re
import random
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Berlin")
from brain.settings import CONFIG_DIR
PROFILE_PATH    = CONFIG_DIR / ("user_profile.json" if (CONFIG_DIR / "user_profile.json").exists() else "user_profile.example.json")
SESSION_PATH    = CONFIG_DIR / "session_state.json"
INTRO_GAP_HOURS = 5


class IdentityManager:
    def __init__(self):
        self._profile = self._load_json(PROFILE_PATH)
        self._session = self._load_json(SESSION_PATH)

    # ── loaders ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_session(self):
        try:
            SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
            SESSION_PATH.write_text(
                json.dumps(self._session, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"[IdentityManager] Could not save session: {e}")

    # ── public API ───────────────────────────────────────────────────────

    @property
    def buddy_name(self) -> str:
        return self._profile.get("buddy", {}).get("name", "Buddy")

    @property
    def user_name(self) -> str:
        return self._profile.get("user", {}).get("name", "sir")

    @property
    def wake_phrases(self) -> list[str]:
        return self._profile.get("buddy", {}).get("wake_phrases", [])

    @property
    def preferred_address(self) -> str:
        return self._profile.get("user", {}).get("preferred_address", "sir")

    def build_system_context(self, current_topic: str = "", recent_context: str = "") -> str:
        """
        Builds system context filtered by relevance to the current conversation.
        
        When current_topic or recent_context is provided, only include background
        items and preferences that share keywords with the active discussion.
        Always include personality, address rules, and communication style.
        """
        buddy = self._profile.get("buddy", {})
        user = self._profile.get("user", {})

        personality = buddy.get("personality", "")
        style = user.get("communication_style", "")
        uname = user.get("name", "the user")
        address = user.get("preferred_address", "sir")

        # Get all background and preferences
        all_background = user.get("background", [])
        all_prefs = user.get("preferences", [])

        # ── NEW: relevance filtering ──
        relevance_text = (current_topic + " " + recent_context).lower()

        if relevance_text.strip():
            # Filter to items that share at least one meaningful keyword
            background_items = self._filter_by_relevance(all_background, relevance_text)
            pref_items = self._filter_by_relevance(all_prefs, relevance_text)
        else:
            # No conversation context yet — include a small default subset
            background_items = all_background[:2]   # just the top 2
            pref_items = all_prefs[:2]

        background = "\n".join(f"- {b}" for b in background_items) if background_items else "- (no specific background relevant)"
        prefs = "\n".join(f"- {p}" for p in pref_items) if pref_items else "- (no specific preferences relevant)"

        return (
            f"{personality}\n\n"
            f"You are talking to {uname}.\n"
            f"About {uname}:\n{background}\n\n"
            f"Preferences:\n{prefs}\n\n"
            f"Communication style: {style}\n\n"
            f"IMPORTANT ADDRESS RULES:\n"
            f"- Always refer to the user as '{address}', never by their real name '{uname}'.\n"
            f"- Do NOT say '{uname}' at any point in your responses.\n"
            f"- Do NOT use the name more than once per response under any circumstance.\n"
            f"- '{address}' should feel natural, not forced — use it sparingly, "
            f"only at the start or end of a sentence when it flows naturally.\n\n"
            f"CRITICAL CONTEXT RULE:\n"
            f"- Only reference the user's background or past projects if directly relevant.\n"
            f"- Do NOT inject unrelated past work (like research projects) into responses "
            f"about new topics.\n"
            f"- If the user is asking about something new, treat it as new."
        ).strip()


    def _filter_by_relevance(self, items: list[str], context: str, max_items: int = 3) -> list[str]:
        """
        Returns items that share at least one meaningful keyword with the context.
        Falls back to empty list if nothing matches — better to say nothing than
        to inject irrelevant background.
        """
        STOP = {
            "the", "a", "an", "of", "in", "on", "at", "for", "to", "from",
            "and", "or", "but", "with", "is", "are", "was", "were", "be",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "this", "that", "these", "those",
            "my", "your", "his", "her", "its", "our", "their", "i", "you",
            "he", "she", "we", "they", "it",
        }

        def tokens(text: str) -> set[str]:
            words = re.findall(r"[a-z0-9]{3,}", text.lower())
            return {w for w in words if w not in STOP}

        context_tokens = tokens(context)
        if not context_tokens:
            return []

        scored = []
        for item in items:
            item_tokens = tokens(item)
            overlap = context_tokens & item_tokens
            if overlap:
                scored.append((len(overlap), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:max_items]]

    def hours_since_last_interaction(self) -> float | None:
        """Returns hours since last interaction, or None if never."""
        last = self._session.get("last_interaction")
        if not last:
            return None
        try:
            last_dt = datetime.fromisoformat(last).astimezone(TZ)
            now     = datetime.now(TZ)
            return (now - last_dt).total_seconds() / 3600
        except Exception:
            return None

    def should_do_full_intro(self) -> bool:
        """True if gap > INTRO_GAP_HOURS or never interacted before."""
        gap = self.hours_since_last_interaction()
        if gap is None:
            return True  # first ever launch
        return gap >= INTRO_GAP_HOURS

    def touch(self):
        """Call this every time user sends a message."""
        self._session["last_interaction"] = datetime.now(TZ).isoformat()
        self._save_session()

    # ── wake phrase detection (hybrid) ───────────────────────────────────

    # Regex layer:
    _WAKE_RE = re.compile(
        r"^("
        # leading "hey/yo" optional
        r"(hey|yo)\s+buddy"
        r"|"
        # "buddy <check-in phrase>"
        r"buddy[\s,]*(are you|you)?\s*"
        r"(up|there|awake|online|listening|ready|around|with me|alive|active|running)?"
        r"|"
        # trailing buddy: "are you there buddy"
        r"(are you|you)\s+(there|up|awake|online|listening|ready|around)\s+buddy"
        r"|"
        # bare name only
        r"buddy"
        r")$",
        re.IGNORECASE,
    )

    def is_wake_phrase(self, text: str) -> bool:
        """
        Hybrid wake detection:
          Layer 1 — exact match against wake_phrases in user_profile.json
          Layer 2 — regex for common Jarvis-style check-in patterns
        Both layers strip trailing punctuation first.
        """
        t = re.sub(r"[,!?.]+$", "", (text or "").strip().lower()).strip()

        # Layer 1: exact match from profile
        if any(t == phrase.lower() for phrase in self.wake_phrases):
            return True

        # Layer 2: regex check-in patterns
        return bool(self._WAKE_RE.match(t))

    def wake_response(self) -> str:
        """
        Jarvis-style wake responses — personality varies by how long
        the user has been away. No comma before 'sir' to avoid TTS pause.
        """
        gap     = self.hours_since_last_interaction()
        address = self.preferred_address

        # ── First launch ever, or returning after a long absence ──────────
        # Full presence, a touch of wit — like flipping the switch back on.
        if gap is None or gap >= INTRO_GAP_HOURS:
            responses = [
                f"For you always {address}.",
                f"Online and standing by {address}.",
                f"Always at your service {address}.",
                f"Systems online. Ready when you are {address}.",
                f"At your disposal {address}.",
                f"Right here {address}. Fully operational.",
                f"Good to have you back {address}. What are we working on?",
            ]
            return random.choice(responses)

        # ── Returning within the same working session (1–5 hours) ─────────
        # Dry acknowledgment of the gap, slight personality.
        if gap >= 1:
            h = int(gap)
            hour_str = f"{h} hour{'s' if h != 1 else ''}"
            responses = [
                f"Back already {address}? It's only been {hour_str}.",
                f"Still here {address}. {hour_str} of standby.",
                f"Right where you left me {address}.",
                f"Picking up where we left off {address}.",
                f"Present {address}. {hour_str} and I haven't moved.",
            ]
            return random.choice(responses)

        # ── Same session, short gap (under an hour) ───────────────────────
        # Minimal and sharp — Jarvis never over-explains.
        responses = [
            f"Still here {address}.",
            f"Always {address}.",
            f"Listening {address}.",
            f"Never left {address}.",
            f"Right here {address}.",
            f"What do you need {address}?",
        ]
        return random.choice(responses)