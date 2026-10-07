"""
spotify_manager.py
──────────────────
Spotify integration for Buddy.

Features:
  1. Mood-based playback  — maps active conversation topic → playlist
  2. Inactivity playlist  — auto-plays after 5 min of silence
  3. Volume ducking       — lowers volume while Buddy speaks / user talks,
                            restores afterward
  4. Direct controls      — play, pause, resume, skip, volume, now playing

Dependencies:
    pip install spotipy

Setup (one-time):
  1. Go to https://developer.spotify.com/dashboard
  2. Create an app → get CLIENT_ID and CLIENT_SECRET
  3. Add http://localhost:8888/callback to Redirect URIs in the app settings
  4. Fill in SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET below
  5. Fill in MOOD_PLAYLISTS with your own Spotify playlist URIs
  6. First run will open a browser for OAuth — after that it auto-refreshes

Usage in ConversationManager:
    from brain.music.spotify_manager import SpotifyManager, MusicWatcher

    self.spotify = SpotifyManager(memory_manager=self.memory_manager)
    self.music_watcher = MusicWatcher(
        spotify=self.spotify,
        identity=self.identity,
    )
    self.music_watcher.start()
"""

from __future__ import annotations
 
import threading
import time
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — fill these in before first run
# ──────────────────────────────────────────────────────────────────────────────

import os as _os
from brain.settings import CONFIG_DIR as _CONFIG_DIR
SPOTIFY_CLIENT_ID     = _os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = _os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"

# Cache path for OAuth token — stored locally so you only auth once
SPOTIFY_CACHE_PATH    = str(_CONFIG_DIR / ".spotify_token_cache")
SPOTIFY_OUTPUT_DEVICE_HINT = "Realtek"   # matches "Realtek(R) Audio" or similar
# ──────────────────────────────────────────────────────────────────────────────
# MOOD PLAYLIST MAP
# Fill in your own Spotify playlist URIs for each mood.
# Format: "spotify:playlist:XXXXXXXXXXXXXXXXXXXXXX"
# Find a URI by right-clicking a playlist in Spotify → Share → Copy Spotify URI
#
# Mood categories:
#   focus    → deep work, technical tasks, research
#   chill    → light conversation, casual browsing
#   morning  → startup / inactivity playlist (played after 5 min silence)
#   energize → upbeat, after a break
#   default  → fallback when no mood is detected
# ──────────────────────────────────────────────────────────────────────────────

MOOD_PLAYLISTS: dict[str, str] = {
    "focus":    "https://open.spotify.com/playlist/37i9dQZF1DX7EF8wVxBVhG?si=32fb8878a3fd4b22",
    "chill":    "https://open.spotify.com/playlist/513SNdQCspFx7bTakZFU60?si=acb245af596242cd&pt=9c4e6f17babfe6b1c159dfdf6a7525f0",
    "morning":  "https://open.spotify.com/playlist/4ebfRaqEiAMzWzzPlo4mPS?si=31b4c1ff594d4fe0",
    "energize": "https://open.spotify.com/playlist/4ebfRaqEiAMzWzzPlo4mPS?si=7927bfef405f417f",
    "default":  "https://open.spotify.com/playlist/4ebfRaqEiAMzWzzPlo4mPS?si=1c47e26b1bc14dc1",
}

# ──────────────────────────────────────────────────────────────────────────────
# MOOD KEYWORD MAP
# Maps conversation topic keywords → mood category.
# Add/adjust keywords to match your actual topics.
# ──────────────────────────────────────────────────────────────────────────────
 
MOOD_KEYWORD_MAP: list[tuple[list[str], str]] = [
    # Technical / research → focus music
    (
        [
            "terahertz", "thz", "matlab", "machine learning", "neural network",
            "signal processing", "fft", "bandpass", "spectroscopy", "dataset",
            "training", "classification", "controller", "pid", "algorithm",
            "code", "debug", "error", "implementation", "architecture",
        ],
        "focus",
    ),
    # Light / casual → chill
    (
        [
            "weather", "news", "chat", "movie", "music", "food",
            "weekend", "plan", "schedule", "reminder", "general",
        ],
        "chill",
    ),
    # Physical / active → energize
    (
        [
            "workout", "gym", "exercise", "run", "sport", "energy",
        ],
        "energize",
    ),
]
 
# Volume levels
DUCK_VOLUME    = 15   # % during speech
NORMAL_VOLUME  = 60   # % restored after speech
INACTIVITY_VOLUME = 45  # % for background inactivity playlist
 
# Inactivity threshold in seconds
INACTIVITY_THRESHOLD_S = 5 * 60   # 5 minutes
WATCHER_POLL_INTERVAL_S = 30       # check every 30 seconds
 
 
# ──────────────────────────────────────────────────────────────────────────────
# SPOTIFY MANAGER
# ──────────────────────────────────────────────────────────────────────────────
 
class SpotifyManager:
    """
    Thin wrapper around spotipy that adds mood routing,
    volume ducking, and inactivity playback.
    """
 
    def __init__(self, memory_manager=None):
        self.mm = memory_manager
        self._sp = None                  
        self._device_id: Optional[str] = None
        self._pre_duck_volume: int = NORMAL_VOLUME
        self._current_mood: Optional[str] = None
        self._lock = threading.Lock()
 
    # ── Spotify client (lazy init) ────────────────────────────────────────────
 
    def _get_client(self):
        """
        Returns the spotipy client, initializing it on first call.
        First call opens a browser for OAuth — subsequent calls use
        the cached token.
        """
        if self._sp is not None:
            return self._sp
 
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth
 
            scope = (
                "user-modify-playback-state "
                "user-read-playback-state "
                "user-read-currently-playing "
                "streaming"
            )
 
            print(f"[SPOTIFY] Using redirect URI: {SPOTIFY_REDIRECT_URI!r}")

            auth = SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=SPOTIFY_CACHE_PATH,
                open_browser=True,
            )
 
            # self._sp = spotipy.Spotify(auth_manager=auth)
            self._sp = spotipy.Spotify(auth_manager=auth, requests_timeout=3)
            print("[SPOTIFY] Client initialized.")
            return self._sp
 
        except ImportError:
            print("[SPOTIFY] spotipy not installed. Run: pip install spotipy")
            return None
        except Exception as e:
            print(f"[SPOTIFY] Init failed: {e}")
            return None
 
    # ── Device resolution ─────────────────────────────────────────────────────
 
    def _get_device_id(self) -> Optional[str]:
        """
        Returns the Spotify device ID for music output.
 
        Priority:
          1. Device whose name contains SPOTIFY_OUTPUT_DEVICE_HINT (Realtek)
          2. Currently active device (fallback)
          3. First available device (last resort)
 
        Result is cached for the session — call _reset_device_cache() to refresh
        if the device list changes (e.g. Spotify reconnects).
        """
        if self._device_id:
            return self._device_id
 
        sp = self._get_client()
        if not sp:
            return None
 
        try:
            devices = sp.devices().get("devices", [])
            if not devices:
                print("[SPOTIFY] No active devices found. Open Spotify on your PC first.")
                return None
 
            print(f"[SPOTIFY] Available devices: {[d['name'] for d in devices]}")
 
            # Priority 1: match by SPOTIFY_OUTPUT_DEVICE_HINT
            if SPOTIFY_OUTPUT_DEVICE_HINT:
                hint = SPOTIFY_OUTPUT_DEVICE_HINT.lower()
                for d in devices:
                    if hint in (d.get("name") or "").lower():
                        self._device_id = d["id"]
                        print(f"[SPOTIFY] Matched output device: {d['name']!r}")
                        return self._device_id
 
                print(f"[SPOTIFY] Hint {SPOTIFY_OUTPUT_DEVICE_HINT!r} not found in device list.")
 
            # Priority 2: active device
            for d in devices:
                if d.get("is_active"):
                    self._device_id = d["id"]
                    print(f"[SPOTIFY] Using active device: {d['name']!r}")
                    return self._device_id
 
            # Priority 3: first available
            self._device_id = devices[0]["id"]
            print(f"[SPOTIFY] Using first available device: {devices[0]['name']!r}")
            return self._device_id
 
        except Exception as e:
            import traceback
            print(f"[SPOTIFY] Device resolution failed: {e}")
            traceback.print_exc()
            return None
 
    def _reset_device_cache(self) -> None:
        """Call this to force re-resolution of the Spotify device on next playback."""
        self._device_id = None
 
    # ── Mood detection ────────────────────────────────────────────────────────
 
    def detect_mood(self) -> str:
        """
        Reads the active conversation topic from MemoryManager and
        maps it to a mood category using MOOD_KEYWORD_MAP.
        Falls back to 'default' if no match.
        """
        if not self.mm:
            return "default"
 
        topic = (self.mm.get_last_topic() or "").lower().strip()
        if not topic:
            return "default"
 
        # Strip "web:" prefix for web search topics
        if topic.startswith("web:"):
            topic = topic[4:].strip()
 
        for keywords, mood in MOOD_KEYWORD_MAP:
            if any(kw in topic for kw in keywords):
                print(f"[SPOTIFY] Mood detected: {mood!r} (topic={topic!r})")
                return mood
 
        return "default"
 
    # ── Playback controls ─────────────────────────────────────────────────────
 
    def play_mood(self, mood: Optional[str] = None, force: bool = False) -> str:
        """
        Plays the playlist for the given mood (or auto-detects from topic).
        Skips if the same mood is already playing, unless force=True.
        Returns a spoken response string.
        """
        mood = mood or self.detect_mood()
        playlist_uri = MOOD_PLAYLISTS.get(mood) or MOOD_PLAYLISTS.get("default")
 
        if not playlist_uri or "YOUR_" in playlist_uri:
            return f"I don't have a playlist configured for {mood} mode yet."
 
        # Skip if already on the same mood (avoid restart mid-track)
        if not force and self._current_mood == mood:
            return f"Already playing {mood} music."
 
        sp = self._get_client()
        device_id = self._get_device_id()
        if not sp or not device_id:
            return self._fallback_youtube(mood)
 
        try:
            sp.start_playback(
                device_id=device_id,
                context_uri=playlist_uri,
            )
            sp.shuffle(True, device_id=device_id)
            self._set_volume(NORMAL_VOLUME)
            self._current_mood = mood
            print(f"[SPOTIFY] Playing mood={mood!r} playlist={playlist_uri}")
 
            mood_phrases = {
                "focus":    "Playing focus music. Let's get to work.",
                "chill":    "Putting on something chill.",
                "morning":  "Good morning playlist starting now.",
                "energize": "Time to energize. Putting on something upbeat.",
                "default":  "Music is on.",
            }
            return mood_phrases.get(mood, "Music is on.")
 
        except Exception as e:
            print(f"[SPOTIFY] Playback failed: {e}")
            return self._fallback_youtube(mood)
 
    def play_inactivity(self) -> None:
        """
        Plays the morning/inactivity playlist at a softer volume.
        Called automatically by MusicWatcher after 5 min of silence.
        Silent — no spoken response since user isn't there.
        """
        sp = self._get_client()
        device_id = self._get_device_id()
        if not sp or not device_id:
            return
 
        playlist_uri = MOOD_PLAYLISTS.get("morning") or MOOD_PLAYLISTS.get("default")
        if not playlist_uri or "YOUR_" in playlist_uri:
            return
 
        try:
            # Only start if nothing is currently playing
            playback = sp.current_playback()
            if playback and playback.get("is_playing"):
                return  # already playing, leave it
 
            sp.start_playback(device_id=device_id, context_uri=playlist_uri)
            sp.shuffle(True, device_id=device_id)
            self._set_volume(INACTIVITY_VOLUME)
            self._current_mood = "morning"
            print("[SPOTIFY] Inactivity playlist started.")
        except Exception as e:
            print(f"[SPOTIFY] Inactivity playback failed: {e}")
 
    def pause(self) -> str:
        sp = self._get_client()
        device_id = self._get_device_id()
        if not sp or not device_id:
            return "Spotify isn't available right now."
        try:
            sp.pause_playback(device_id=device_id)
            return "Music paused."
        except Exception as e:
            print(f"[SPOTIFY] Pause failed: {e}")
            return "Couldn't pause the music."
 
    def resume(self) -> str:
        sp = self._get_client()
        device_id = self._get_device_id()
        if not sp or not device_id:
            return "Spotify isn't available right now."
        try:
            sp.start_playback(device_id=device_id)
            return "Resuming."
        except Exception as e:
            print(f"[SPOTIFY] Resume failed: {e}")
            return "Couldn't resume the music."
 
    def skip(self) -> str:
        sp = self._get_client()
        device_id = self._get_device_id()
        if not sp or not device_id:
            return "Spotify isn't available right now."
        try:
            sp.next_track(device_id=device_id)
            return "Skipping."
        except Exception as e:
            print(f"[SPOTIFY] Skip failed: {e}")
            return "Couldn't skip the track."
 
    def set_volume(self, percent: int) -> str:
        """Public volume setter with spoken response."""
        percent = max(0, min(100, percent))
        sp = self._get_client()
        device_id = self._get_device_id()
        if not sp or not device_id:
            return "Spotify isn't available right now."
        try:
            self._set_volume(percent)
            return f"Volume set to {percent}."
        except Exception as e:
            print(f"[SPOTIFY] Volume failed: {e}")
            return "Couldn't adjust the volume."
 
    def now_playing(self) -> str:
        """Returns a spoken description of the current track."""
        sp = self._get_client()
        if not sp:
            return "Spotify isn't available right now."
        try:
            playback = sp.current_playback()
            if not playback or not playback.get("is_playing"):
                return "Nothing is playing right now."
 
            item = playback.get("item") or {}
            track = item.get("name", "Unknown track")
            artists = ", ".join(
                a["name"] for a in item.get("artists", [])
            ) or "Unknown artist"
            return f"Currently playing {track} by {artists}."
        except Exception as e:
            print(f"[SPOTIFY] Now playing failed: {e}")
            return "I couldn't check what's playing."
 
    # ── Volume ducking ────────────────────────────────────────────────────────
 
    def duck(self) -> None:
        """
        Lower volume for speech. Call when VAD fires or TTS starts.
        Saves current volume so restore() can bring it back.
        """
        sp = self._get_client()
        if not sp:
            return
        try:
            playback = sp.current_playback()
            if playback and playback.get("is_playing"):
                current_vol = (playback.get("device") or {}).get("volume_percent", NORMAL_VOLUME)
                self._pre_duck_volume = current_vol
                self._set_volume(DUCK_VOLUME)
                print(f"[SPOTIFY] Ducked: {current_vol}% → {DUCK_VOLUME}%")
        except Exception as e:
            print(f"[SPOTIFY] Duck failed: {e}")
 
    def restore(self) -> None:
        """
        Restore volume after speech ends.
        Call when VAD unmutes or TTS finishes.
        """
        sp = self._get_client()
        if not sp:
            return
        try:
            playback = sp.current_playback()
            if playback and playback.get("is_playing"):
                self._set_volume(self._pre_duck_volume)
                print(f"[SPOTIFY] Restored: {DUCK_VOLUME}% → {self._pre_duck_volume}%")
        except Exception as e:
            print(f"[SPOTIFY] Restore failed: {e}")
 
    # ── Internal helpers ──────────────────────────────────────────────────────
 
    def _set_volume(self, percent: int) -> None:
        sp = self._get_client()
        device_id = self._get_device_id()
        if sp and device_id:
            sp.volume(percent, device_id=device_id)
 
    def _fallback_youtube(self, mood: str) -> str:
        """
        Opens a YouTube search in the browser as fallback when
        Spotify is unavailable or not configured.
        """
        import webbrowser
        queries = {
            "focus":    "lofi hip hop focus music",
            "chill":    "chill playlist music",
            "morning":  "morning playlist uplifting",
            "energize": "upbeat energy workout music",
            "default":  "background music playlist",
        }
        query = queries.get(mood, "background music").replace(" ", "+")
        url = f"https://www.youtube.com/results?search_query={query}"
        webbrowser.open(url)
        print(f"[SPOTIFY] Fallback: opened YouTube for mood={mood!r}")
        return f"Spotify isn't available, so I opened a {mood} playlist on YouTube."
 
 
# ──────────────────────────────────────────────────────────────────────────────
# MUSIC WATCHER — background inactivity thread
# ──────────────────────────────────────────────────────────────────────────────
 
class MusicWatcher(threading.Thread):
    """
    Background daemon thread that watches for inactivity and
    auto-plays the morning playlist after INACTIVITY_THRESHOLD_S seconds.
 
    Usage:
        watcher = MusicWatcher(spotify=spotify_manager, identity=identity_manager)
        watcher.start()
        # To stop cleanly on exit:
        watcher.stop()
    """
 
    def __init__(self, spotify: SpotifyManager, identity):
        super().__init__(daemon=True, name="MusicWatcher")
        self.spotify = spotify
        self.identity = identity
        self._stop_event = threading.Event()
 
    def run(self):
        print("[MUSIC_WATCHER] Started.")
        while not self._stop_event.is_set():
            try:
                self._check()
            except Exception as e:
                print(f"[MUSIC_WATCHER] Error: {e}")
            self._stop_event.wait(WATCHER_POLL_INTERVAL_S)
 
    def stop(self):
        self._stop_event.set()
 
    def _check(self):
        gap = self.identity.hours_since_last_interaction()
        if gap is None:
            return
 
        gap_seconds = gap * 3600
        if gap_seconds >= INACTIVITY_THRESHOLD_S:
            print(f"[MUSIC_WATCHER] Inactivity {gap_seconds:.0f}s >= threshold. Triggering playlist.")
            self.spotify.play_inactivity()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION — voice command routing
# ──────────────────────────────────────────────────────────────────────────────
 
import re
 
# ── Pre-strip filler prefixes before pattern matching ────────────────────────
# Removes "can you", "could you", "buddy", "please" etc. so patterns stay clean
_FILLER_PREFIX_RE = re.compile(
    r"^(buddy[\s,]*|hey buddy[\s,]*)?"
    r"(can you|could you|please|would you|i want you to|i'd like you to|i need you to)[\s,]*",
    re.IGNORECASE,
)
 
# ── Intent patterns — ordered from most specific to most general ──────────────
# ORDER MATTERS: play_query must come after play (generic) would steal matches,
# but here play_query is checked separately after stripping filler.
 
_MUSIC_INTENTS: list[tuple[re.Pattern, str]] = [
 
    # ── now_playing — FIRST, before play patterns steal "what's playing" ──
    (re.compile(
        r"\b(what('?s| is) (playing|this( song)?|on)|now playing"
        r"|current (song|track|music)|what song (is this|am i hearing)"
        r"|name of (this|the) (song|track))\b",
        re.I), "now_playing"),
 
    # ── pause ──
    (re.compile(
        r"\b(pause|stop the music|stop music|turn off (the )?music"
        r"|mute (the )?music|silence (the )?music)\b",
        re.I), "pause"),
 
    # ── resume ──
    (re.compile(
        r"\b(resume|unpause|play again|continue (the )?(music|song|playlist)"
        r"|bring (the )?music back)\b",
        re.I), "resume"),
 
    # ── skip ──
    (re.compile(
        r"\b(skip( this)?( song| track)?|next (song|track|one)"
        r"|change (the )?(song|track)|i don'?t like this (song|one)"
        r"|play (the )?next( one)?)\b",
        re.I), "skip"),
 
    # ── volume — specific number ──
    (re.compile(
        r"\b(set (the )?(volume|music) to|volume (to|at)|at volume)\s*\d",
        re.I), "volume"),
 
    # ── volume — directional ──
    (re.compile(
        r"\b(turn (it|the (music|volume))?\s*(up|down)"
        r"|make it (louder|quieter|softer|a bit (louder|quieter|softer))"
        r"|a (bit|little) (louder|quieter|softer|lower|higher)"
        r"|volume (up|down|higher|lower)"
        r"|louder|quieter|softer)\b",
        re.I), "volume"),
 
    # ── play generic — no specific query ──
    (re.compile(
        r"\b(play|start|put on)\s*(some\s*)?"
        r"(music|something|a (song|playlist|track)|anything)\b",
        re.I), "play"),
 
    # ── play with mood keyword directly ──
    # e.g. "put on something for focus", "play focus music"
    (re.compile(
        r"\b(play|put on|start)\s*(something\s*)?(for\s*)?"
        r"(?P<mood>focus|chill|chilling|relaxing|energizing|energize|morning|work|studying|study)\b",
        re.I), "play_mood"),
 
    # ── play with explicit query ──
    # e.g. "play lo-fi hip hop", "play Kendrick Lamar"
    (re.compile(r"\bplay\s+(?P<query>.+)", re.I), "play_query"),
 
    # ── I want to listen / I feel like listening ──
    (re.compile(
        r"\b(i (want|feel like|would like) (to )?(listen|hear)"
        r"|put (on )?some music"
        r"|can (we|i) (have|get) (some )?(music|a song))\b",
        re.I), "play"),
]
 
_VOLUME_AMOUNT_RE = re.compile(r"\b(\d{1,3})\s*(%|percent)?\b")
_VOLUME_DIR_RE    = re.compile(
    r"\b(up|louder|higher|down|lower|quieter|softer)\b", re.I
)
 
# Words that mean "play generic" even if they follow "play"
_GENERIC_QUERY_WORDS = {
    "something", "music", "a song", "a playlist", "a track",
    "anything", "some music", "some songs",
}
 
# Mood keyword → mood category for play_mood intent
_MOOD_WORD_MAP = {
    "focus": "focus", "work": "focus", "study": "focus", "studying": "focus",
    "chill": "chill", "chilling": "chill", "relax": "chill", "relaxing": "chill",
    "energize": "energize", "energizing": "energize",
    "morning": "morning",
}
 
 
def detect_music_intent(text: str) -> Optional[tuple[str, dict]]:
    """
    Returns (intent, params) if the text is a music command, else None.
    Designed to be called from handle_request() BEFORE classify_intent().
 
    Handles natural phrasing with filler prefix stripping:
        "can you play some music"        → ("play", {})
        "put on something for focus"     → ("play_mood", {"mood": "focus"})
        "play lo-fi hip hop"             → ("play_query", {"query": "lo-fi hip hop"})
        "skip this song"                 → ("skip", {})
        "make it a bit louder"           → ("volume", {"direction": "up"})
        "set the volume to 40"           → ("volume", {"percent": 40})
        "what's playing"                 → ("now_playing", {})
        "i feel like listening to music" → ("play", {})
    """
    if not text:
        return None
 
    # Strip filler prefix for cleaner matching
    stripped = _FILLER_PREFIX_RE.sub("", text.strip()).strip()
 
    for pattern, intent in _MUSIC_INTENTS:
        m = pattern.search(stripped)
        if not m:
            continue
 
        params: dict = {}
 
        # ── play_mood: named mood word in utterance ──
        if intent == "play_mood":
            raw_mood = (m.group("mood") or "").strip().lower()
            params["mood"] = _MOOD_WORD_MAP.get(raw_mood, "default")
            return intent, params
 
        # ── play_query: extract query, downgrade to play if generic ──
        if intent == "play_query":
            query = m.group("query").strip()
            if query.lower() in _GENERIC_QUERY_WORDS:
                return "play", {}
            params["query"] = query
            return intent, params
 
        # ── volume: extract percent or direction ──
        if intent == "volume":
            vm = _VOLUME_AMOUNT_RE.search(stripped)
            if vm:
                val = int(vm.group(1))
                # Sanity check: ignore numbers that look like years/times
                if 0 <= val <= 100:
                    params["percent"] = val
            else:
                dm = _VOLUME_DIR_RE.search(stripped)
                if dm:
                    direction = dm.group(1).lower()
                    params["direction"] = (
                        "up" if direction in {"up", "louder", "higher"} else "down"
                    )
            return intent, params
 
        return intent, params
 
    return None
 
 
def handle_music_command(
    intent: str,
    params: dict,
    spotify: SpotifyManager,
    current_volume: int = NORMAL_VOLUME,
) -> str:
    """
    Executes a music command and returns a spoken response.
    Pass current_volume to support relative up/down adjustments.
    """
    if intent == "play":
        return spotify.play_mood()
 
    if intent == "play_mood":
        mood = params.get("mood", "default")
        return spotify.play_mood(mood=mood, force=True)
 
    if intent == "play_query":
        # Try to map query to a mood first, fall back to default
        query = (params.get("query") or "").lower()
        mood = "default"
        for keywords, m in MOOD_KEYWORD_MAP:
            if any(kw in query for kw in keywords):
                mood = m
                break
        return spotify.play_mood(mood=mood, force=True)
 
    if intent == "pause":
        return spotify.pause()
 
    if intent == "resume":
        return spotify.resume()
 
    if intent == "skip":
        return spotify.skip()
 
    if intent == "volume":
        if "percent" in params:
            return spotify.set_volume(params["percent"])
        if "direction" in params:
            step = 15
            new_vol = (
                min(100, current_volume + step)
                if params["direction"] == "up"
                else max(0, current_volume - step)
            )
            return spotify.set_volume(new_vol)
        return "What volume would you like?"
 
    if intent == "now_playing":
        return spotify.now_playing()
 
    return "I didn't understand that music command."