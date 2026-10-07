"""
rover_controller.py  —  Windows PC side

New in this version:
  - TelemetryCache: thread-safe dict updated by async WebSocket push
  - AlertWatcher: background thread, polls cache, fires TTS on critical events
  - RoverPersonality: reads live cache for richer prompt context (0ms overhead)
  - RoverTaskManager: feeds bridge telemetry into cache after every command
"""

import re
import time
import threading
import traceback
from pathlib import Path

from fuzzywuzzy import fuzz

from brain.core.brain import generate_llm_response, store_short_term_memory
from devices.rover.rover_telemetry import parse_line, RoverTelemetryWindow, check_summary, plot_rover_log, plot_estimated_trajectory

ROVER_TOPIC = "rover:session"

ROVER_LOG_DIR = Path(__file__).resolve().parent / "rover_logs"
ROVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

ROVER_VOCAB = [
    "rover", "forward", "backward", "left", "right", "stop",
    "faster", "slower", "follow", "find", "door", "move", "turn",
]

_COMMAND_SHAPE_RE = re.compile(
    r"\b(go|move|turn|stop|come|follow|find|connect|disconnect|activate|"
    r"start|drive|faster|slower|left|right|forward|backward|reverse)\b",
    re.IGNORECASE
)

_FILLER_PREFIX_RE = re.compile(
    r"^(okay|ok|alright|alright then|great|good|it'?s good( now)?|"
    r"nice|perfect|cool|got it|sounds good|that'?s good)[,\s]+",
    re.IGNORECASE,
)

WATCHDOG_TIMEOUT_S = 2.0 


# ─────────────────────────────────────────────────────────────────────────────
#  TelemetryCache
#  Updated by your existing WebSocket push handler — just call .update().
#  Read by RoverPersonality and AlertWatcher — zero network cost.
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryCache:
    """
    Thread-safe store for the latest rover telemetry.

    Your RoverBridge WebSocket listener should call:
        telemetry_cache.update(battery=72, speed=40, direction="forward")

    on every incoming push frame. No locks needed for reads since Python
    dict assignment is atomic for simple types.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "battery":   None,   # int 0-100
            "speed":     None,   # int 0-100
            "direction": None,   # str "forward" | "backward" | "left" | "right" | "stop"
            "updated_at": 0.0,   # float timestamp of last push
        }

        

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k in self._data:
                    self._data[k] = v
            self._data["updated_at"] = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def is_stale(self, max_age_sec: float = 10.0) -> bool:
        return (time.time() - self._data["updated_at"]) > max_age_sec


# ─────────────────────────────────────────────────────────────────────────────
#  AlertWatcher
#  Background thread. Polls TelemetryCache, fires TTS on threshold crossings.
#  Completely decoupled from the command path — zero latency impact.
# ─────────────────────────────────────────────────────────────────────────────

class AlertWatcher:
    """
    Monitors telemetry and speaks up proactively when something matters.
    Does NOT use an LLM call — messages are pre-written for instant TTS.

    Usage:
        watcher = AlertWatcher(cache=telemetry_cache, tts=tts_worker)
        watcher.start()   # call once when rover session opens
        watcher.stop()    # call when session closes
    """

    POLL_INTERVAL_SEC  = 3.0
    BATTERY_WARN       = 20    # % — warn once when crossing below
    BATTERY_CRITICAL   = 10    # % — warn again, more urgently
    SPEED_JUMP_THRESH  = 30    # % delta in one poll cycle

    def __init__(self, cache: TelemetryCache, tts=None):
        self.cache = cache
        self.tts   = tts        # TTSWorker — call tts.speak(text)

        self._thread  = None
        self._running = False

        # Track what we've already warned about so we don't repeat
        self._warned_battery_low      = False
        self._warned_battery_critical = False
        self._last_speed              = None

    def start(self):
        self._running = True
        self._warned_battery_low      = False
        self._warned_battery_critical = False
        self._last_speed              = None

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="RoverAlertWatcher",
        )
        self._thread.start()
        print("[ROVER] AlertWatcher started")

    def stop(self):
        self._running = False
        print("[ROVER] AlertWatcher stopped")

    def _loop(self):
        while self._running:
            try:
                self._check()
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL_SEC)

    def _check(self):
        snap = self.cache.snapshot()

        if self.cache.is_stale(max_age_sec=15.0):
            return  # no fresh data, don't alert

        battery = snap.get("battery")
        speed   = snap.get("speed")

        # ── Battery alerts ────────────────────────────────────────────────
        if battery is not None:
            if battery <= self.BATTERY_CRITICAL and not self._warned_battery_critical:
                self._warned_battery_critical = True
                self._speak(
                    f"Heads up — rover battery is critically low at {battery} percent. "
                    "Consider stopping soon."
                )
            elif battery <= self.BATTERY_WARN and not self._warned_battery_low:
                self._warned_battery_low = True
                self._speak(
                    f"Battery's getting low, sitting at {battery} percent."
                )

        # ── Sudden speed change alert ─────────────────────────────────────
        # Flags unexpected jumps (e.g. motor fault causing speed spike)
        # not user-commanded speed changes (those come through _parse_command).
        if speed is not None and self._last_speed is not None:
            delta = abs(speed - self._last_speed)
            if delta >= self.SPEED_JUMP_THRESH:
                self._speak(
                    f"Speed jumped by {delta} percent unexpectedly — "
                    "checking if that was intentional."
                )
        self._last_speed = speed

    def _speak(self, text: str):
        print(f"[ROVER ALERT] {text}")
        if self.tts:
            try:
                self.tts.speak(text)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  RoverPersonality
# ─────────────────────────────────────────────────────────────────────────────

class RoverPersonality:
    """
    Converts rigid FSM reply strings into natural co-pilot sentences.
    Reads TelemetryCache for richer context at zero extra latency —
    the cache is already populated by the async WebSocket push.
    """

    _SYSTEM = (
        "You are Buddy, a friendly and casual co-pilot AI for a ground rover robot. "
        "Your job is to take a raw status message and rephrase it as a short, natural "
        "sentence — like a helpful teammate talking to the operator.\n\n"
        "Rules:\n"
        "- 1-2 sentences MAXIMUM. Never longer.\n"
        "- Sound conversational, never robotic or template-y.\n"
        "- React naturally: low battery = mild concern, error = calm alert, "
        "movement = light acknowledgement.\n"
        "- Never say 'I have processed your command' or similar corporate-speak.\n"
        "- If the rover stopped unexpectedly, sound genuinely attentive.\n"
        "- NEVER ask a question. Statements only — questions break operator focus.\n"
        "- NEVER end with filler sign-offs like 'All good!', 'Roger that!', "
        "'Got it!', 'Copy that!' — vary your endings naturally.\n"
        "- For a status/activate/connect command, reply as a simple ready "
        "acknowledgement — do NOT imply the rover is already moving.\n"
        "- If battery is below 20%, weave a brief casual note about it into the reply.\n"
        "- Output the rephrased sentence ONLY — no preamble, no labels."
    )

    def __init__(self, mm=None, cache: TelemetryCache = None):
        self.mm    = mm
        self.cache = cache

    def personalize(self, raw_reply: str, rover_state: dict) -> str:
        try:
            # Merge live cache into rover_state for richer prompt
            enriched = self._enrich_state(rover_state)
            prompt   = self._build_prompt(raw_reply, enriched)
            reply    = generate_llm_response(prompt, topic=ROVER_TOPIC)
            reply    = (reply or "").strip()
            return reply if reply else raw_reply
        except Exception:
            return raw_reply

    def _enrich_state(self, rover_state: dict) -> dict:
        """
        Merge the latest TelemetryCache snapshot into rover_state.
        Cache values fill in gaps; command-path values take priority.
        """
        if not self.cache or self.cache.is_stale(max_age_sec=10.0):
            return rover_state

        snap = self.cache.snapshot()
        merged = dict(rover_state)

        # Only fill in if not already set by the command path
        if merged.get("battery") is None and snap.get("battery") is not None:
            merged["battery"] = snap["battery"]
        if merged.get("speed_percent") is None and snap.get("speed") is not None:
            merged["speed_percent"] = snap["speed"]
        if merged.get("direction") in (None, "stop") and snap.get("direction") not in (None, "stop"):
            merged["direction"] = snap["direction"]

        return merged

    # def _recent_history(self, n: int = 4) -> str:
    #     if not self.mm:
    #         return ""
    #     try:
    #         messages = self.mm.get_recent_messages(n=n)
    #         if not messages:
    #             return ""
    #         lines = []
    #         for turn in messages:
    #             role    = turn.get("role", "user").capitalize()
    #             content = (turn.get("content") or "").strip()
    #             if content:
    #                 lines.append(f"{role}: {content}")
    #         return "\n".join(lines)
    #     except Exception:
    #         return ""

    def _recent_history(self, n: int = 4) -> str:
        try:
            from brain.core.brain import temp_topic_memory
            
            messages = temp_topic_memory.get(ROVER_TOPIC, [])
            if not messages:
                return ""
            
            # Each entry is already formatted as "User: X\nBuddy: Y"
            # Take the last n exchanges and join them
            recent = messages[-n:]
            return "\n".join(recent)
        except Exception:
            return ""

    def _state_summary(self, rover_state: dict) -> str:
        parts = []
        if rover_state.get("direction"):
            parts.append(f"direction={rover_state['direction']}")
        if rover_state.get("speed_percent") is not None:
            parts.append(f"speed={rover_state['speed_percent']}%")
        if rover_state.get("battery") is not None:
            b = rover_state["battery"]
            warn = " (LOW)" if b <= 20 else ""
            parts.append(f"battery={b}%{warn}")
        if rover_state.get("session_active") is not None:
            parts.append(f"session={'active' if rover_state['session_active'] else 'inactive'}")
        if rover_state.get("error"):
            parts.append(f"error='{rover_state['error']}'")
        if rover_state.get("mode"):
            parts.append(f"mode={rover_state['mode']}")
        return ", ".join(parts) if parts else "state unknown"

    def _build_prompt(self, raw_reply: str, rover_state: dict) -> str:
        history = self._recent_history(n=4)
        state   = self._state_summary(rover_state)
        history_block = (
            f"Recent rover session:\n{history}"
            if history
            else "Recent rover session: (session just started)"
        )
        return (
            f"{self._SYSTEM}\n\n"
            f"Rover state: {state}\n"
            f"{history_block}\n\n"
            f"Raw status: \"{raw_reply}\"\n\n"
            f"Rephrase this as a natural co-pilot sentence:"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  RoverTaskManager
# ─────────────────────────────────────────────────────────────────────────────

class RoverTaskManager:
    SESSION_TIMEOUT_SEC = 60
    SPEED_STEP    = 10
    DEFAULT_SPEED = 40
    MAX_SPEED     = 100
    MIN_SPEED     = 0

    def __init__(self, rover_bridge, mm=None, tts=None):
        self.rover_bridge        = rover_bridge
        self.mm                  = mm
        self.tts                 = tts   # passed in so AlertWatcher can speak

        self.session_active      = False
        self.current_direction   = "stop"
        self.current_speed       = 0
        self.last_updated        = 0.0
        self._last_was_connected = True
        self.wheels_off_mode = False
        self._last_telemetry_verdict = None
        self._last_saved_log_path: str | None = None
        self._last_saved_log_ts = time.time()

        self._session_samples: list[dict] = [] # array for store past log

        # Shared cache — bridge WebSocket listener calls cache.update()
        self.cache       = TelemetryCache()
        self.personality = RoverPersonality(mm=mm, cache=self.cache)
        self.watcher     = AlertWatcher(cache=self.cache, tts=tts)
        self._command_history: list[dict] = []
        self._current_segment: dict | None = None
        # Wire the bridge to the cache if it supports a telemetry callback
        if hasattr(rover_bridge, "set_telemetry_callback"):
            rover_bridge.set_telemetry_callback(self.cache.update)

    def plot_last_trajectory_estimate(self, open_file_fn) -> str:
        self._close_current_segment()
        if not self._command_history:
            return "I don't have a recent command history to estimate a path from."
        from .rover_telemetry import plot_estimated_trajectory
        png_path = str(ROVER_LOG_DIR / f"trajectory_estimate_{int(time.time())}.png")
        try:
            plot_estimated_trajectory(self._command_history, png_path)
            open_file_fn(png_path)
            reply = "Here's the estimated path based on the commands I've issued — this shows what the rover was told to do, not what it physically did."
        except Exception as e:
            reply = f"I couldn't generate the trajectory estimate: {e}"
        self._store_turn("plot the estimated direction", reply)
        return reply

    def _process_telemetry(self, raw_lines: list):
        window = RoverTelemetryWindow()
        for line in raw_lines:
            window.feed(parse_line(line))
        samples = window.get_samples()
        self._session_samples.extend(samples)
        summary = window.close()
        verdict = check_summary(summary, wheels_off=self.wheels_off_mode, samples=samples)
        self._last_telemetry_verdict = verdict   
        return verdict
    
    def _close_current_segment(self, end_ts: float | None = None):
        if self._current_segment is None:
            return
        end_ts = end_ts if end_ts is not None else time.time()
        raw_duration = max(0.0, end_ts - self._current_segment["start_ts"])
        duration = min(raw_duration, WATCHDOG_TIMEOUT_S)   # ← cap it
        self._command_history.append({
            "action": self._current_segment["action"],
            "direction": self._current_segment["direction"],
            "speed_percent": self._current_segment["speed_percent"],
            "duration_s": duration,
        })
        self._current_segment = None

    def _open_new_segment(self, cmd: dict):
        now = time.time()
        self._close_current_segment(end_ts=now)
        self._current_segment = {
            "action": cmd.get("action", ""),
            "direction": cmd.get("direction", "stop"),
            "speed_percent": cmd.get("speed_percent", 0),
            "start_ts": now,
        }
        

    def emergency_stop_and_report(self) -> str:
        self._emergency_stop_silent()
        try:
            result = self.rover_bridge.send_command(self._make_mode_command("IDLE"))
            ok = result.get("ok", False)
        except Exception:
            ok = False
        self._store_turn("stop", "Rover stopped." if ok else "Stop sent, but I couldn't confirm the rover received it.")
        return "Rover stopped." if ok else "Stop sent, but I couldn't confirm the rover received it."
        
    def _fuzzy_rover_match(self, text: str) -> tuple[float, str | None, str | None]:
        """Scans words for phonetic closeness to rover vocabulary.
        Returns (confidence 0.0-1.0, matched_canonical_word, original_word)."""
        words = re.findall(r"[a-z]+", text.lower())
        best = (0.0, None, None)
        for w in words:
            if len(w) < 3:
                continue
            for v in ROVER_VOCAB:
                ratio = fuzz.ratio(w, v) / 100.0
                if ratio > best[0]:
                    best = (ratio, v, w)
        return best

    def _score_rover_relevance(self, text: str) -> dict:
        if self._belongs_to_rover_context(text):
            return {"score": 1.0, "corrected_text": text, "reason": "exact_match"}

        if not _COMMAND_SHAPE_RE.search(text):
            return {"score": 0.0, "corrected_text": text, "canonical": None, "reason": "no_command_shape"}

        fuzzy_score, canonical, orig_word = self._fuzzy_rover_match(text)
        
        # Continuity: mid-movement sessions lean toward "still talking to the rover"
        continuity_bonus = 0.10 if (self.session_active and self.current_direction != "stop") else 0.0

        # Recency: very fresh session = more benefit of the doubt
        recency_bonus = 0.05 if (self.session_active and (time.time() - self.last_updated) < 15) else 0.0

        score = min(1.0, fuzzy_score + continuity_bonus + recency_bonus)
        corrected_text = text.replace(orig_word, canonical) if (canonical and fuzzy_score >= 0.55) else text

        return {
            "score": score,
            "canonical": canonical,
            "original_word": orig_word,
            "corrected_text": corrected_text,
            "reason": f"fuzzy={fuzzy_score:.2f} continuity={continuity_bonus:.2f} recency={recency_bonus:.2f}",
        }

    def save_last_log(self) -> str:
        if not self._last_telemetry_verdict or not self._last_telemetry_verdict.samples:
            return "I don't have any recent telemetry to save yet."
        from .rover_telemetry import export_samples_csv
        import time as _time
        
        if not self._session_samples:
            return "I don't have any recent telementry to save yet."
        
        filepath = ROVER_LOG_DIR / f"rover_log_{int(_time.time())}.csv"
        n = export_samples_csv(self._session_samples, str(filepath))
        self._last_saved_log_path = str(filepath)
        reply = f"Saved {n} telemetry samples to {filepath.name}."
        self._store_turn("save this log", reply)
        return reply
    
    def report_last_log_result(self) -> str:
        if not self._last_telemetry_verdict:
            return "I don't have a recent rover response to report on."
        from .rover_telemetry import compute_step_response_metrics
        m = compute_step_response_metrics(self._last_telemetry_verdict.samples)
        if m.get("error"):
            reply = f"I couldn't compute step-response metrics: {m['error']}."
        elif not m["reached_setpoint"]:
            reply = (
                f"The rover never reached the commanded setpoint of {m['setpoint']} m/s — "
                f"steady-state error was {m['steady_state_error']:.2f} m/s. "
                + ("This matches wheels-off testing." if self.wheels_off_mode else "That's worth investigating — could be a stall or encoder issue.")
            )
        else:
            reply = (
                f"Rise time {m['rise_time_s']:.2f}s, settled by {m['settle_time_s']:.2f}s, "
                f"overshoot {m['overshoot_pct']:.1f}%."
            )
        self._store_turn("show me the result", reply)
        return reply
    
    def plot_last_log(self, open_file_fn) -> str:
        if not self._last_saved_log_path:
            return "I haven't saved a log yet this session — say 'save this log' first."
        from .rover_telemetry import plot_rover_log
        png_path = self._last_saved_log_path.replace(".csv", ".png")
        try:
            plot_rover_log(self._last_saved_log_path, png_path)
            open_file_fn(png_path)
            reply = "Here's the plot of the rover's response — opened it for you."
        except Exception as e:
            reply = f"I couldn't generate the plot: {e}"
        self._store_turn("can you plot the result", reply)
        return reply

    def handle(self, user_text: str) -> dict:
        # first layer check
        print(f"[ROVER_DEBUG] handle() called with: {user_text!r}")

        cleaned  = _FILLER_PREFIX_RE.sub("", (user_text or "")).strip()
        text     = self._norm(cleaned)
        raw_text = user_text

        # ── Explicit exit — check first, skip scoring entirely ─────────────
        EXIT_PHRASES = ("disconnect the rover", "disconnect from rover", "exit rover mode",
                        "done with the rover", "stop the rover session", "end rover session",
                        "close the rover")
        
        if text.strip() in ("disconnect", "exit") or any(p in text for p in EXIT_PHRASES):
            self._emergency_stop_silent()
            self._close_session()
            return {"handled": True, "reply": "Rover session ended.", "session_active": False}
        
        if any(p in text for p in EXIT_PHRASES):
            self._emergency_stop_silent()
            self._close_session()
            return {"handled": True, "reply": "Rover session ended.", "session_active": False}

        # ── Confidence scoring (single source of truth for is_rover) ───────
        relevance = self._score_rover_relevance(text)
        score = relevance["score"]
        print(f"[ROVER_SCORE] score={score:.2f} reason={relevance['reason']}")

        HIGH_CONF, LOW_CONF = 0.75, 0.45

        if score >= HIGH_CONF:
            is_rover = True
            text = relevance["corrected_text"]   # use corrected wording downstream
        elif score >= LOW_CONF:
            self._expire_if_needed()
            return {
                "handled": "ambiguous",
                "reply": f"Did you mean tell the rover to {relevance['canonical']}?",
                "session_active": self.session_active,
                "pending_correction": relevance,
            }
        else:
            is_rover = False

        if is_rover:
            self.last_updated = time.time()
        else:
            self._expire_if_needed()
            # Soft release — do NOT force-close on unrelated small talk.
            # Session lives on; only the topic/routing falls through to normal chat.
            return {"handled": False, "reply": "", "session_active": self.session_active}

        # second layer check
        if not is_rover:
            if self.session_active:
                self._close_session()
            self.session_active = False
            return {"handled": False, "reply": "", "session_active": False}

        cmd = self._parse_command(text)  # ← parse FIRST
        print(f"[ROVER_DEBUG] _parse_command → {cmd}") 

        # Only treat as disconnect if mid-session AND not a reconnect attempt
        if not self.rover_bridge.is_connected and self.session_active:
            if cmd is None or cmd.get("action") != "status":
                return self._handle_disconnect(raw_text)
        if cmd is None:
            return {
                "handled":        True,
                "reply":          "I understood this is about the rover, but I couldn't parse the exact command.",
                "session_active": self.session_active,
            }

        if not self.session_active:
            self._open_session()
        self.session_active = True

        try:
            self._open_new_segment(cmd)
            bridge_result = self.rover_bridge.send_command(cmd)
        except Exception as e:
            reply = self._compose_error_reply(str(e), traceback.format_exc())
            self._store_turn(raw_text, reply)
            self.session_active = False
            self._close_session()
            return {
                "handled": True,
                "reply": reply,
                "session_active": False,
                "connection_failed": True,          
                "error": str(e),                    
            }
        
        if not bridge_result.get("ok", True):
            if cmd.get("action") != "status":
                self._emergency_stop_silent()
            error_msg = bridge_result.get("error", "unknown error")
            reply = self._compose_error_reply(
                error_msg,
                bridge_result.get("traceback", ""),
            )
            self._store_turn(raw_text, reply)
            self.session_active = False
            self._close_session()
            return {
                "handled": True,
                "reply": reply,
                "session_active": False,
                "connection_failed": True,         
                "error": error_msg,                 
            }
            
        self._last_was_connected = True

        # Feed bridge telemetry into cache so personality + watcher stay current
        self.cache.update(
            battery=bridge_result.get("battery"),
            speed=bridge_result.get("speed") or self.current_speed,
            direction=self.current_direction,
        )
        raw_lines = bridge_result.get("raw_telemetry") or []
        telemetry_verdict = self._process_telemetry(raw_lines) if raw_lines else None

        if cmd.get("action") in ("move", "turn"):
            time.sleep(1.5)  # let the ESP32 actually respond before checking
            followup = self.rover_bridge.send_command(self._make_command("status", self.current_direction, self.current_speed))
            raw_lines = followup.get("raw_telemetry") or []
        else:
            raw_lines = bridge_result.get("raw_telemetry") or []

        telemetry_verdict = self._process_telemetry(raw_lines) if raw_lines else None

        raw_reply   = bridge_result.get("user_message") or self._build_reply(cmd)
        rover_state = {
            "direction":      self.current_direction,
            "speed_percent":  self.current_speed,
            "session_active": self.session_active,
            "battery":        bridge_result.get("battery"),
            "mode":           cmd.get("mode"),        # ← add this
            "error":          None,
            "telemetry_verdict": telemetry_verdict,
        }
        natural_reply = self.personality.personalize(raw_reply, rover_state)
        self._store_turn(raw_text, natural_reply)

        if cmd["action"] == "stop":
            self._close_session()

        return {
            "handled":        True,
            "reply":          natural_reply,
            "command":        cmd,
            "bridge_result":  bridge_result,
            "session_active": self.session_active,
            "telemetry": {
                "battery":    bridge_result.get("battery"),
                "cpu_temp_c": bridge_result.get("cpu_temp_c"),
                "speed":      bridge_result.get("speed"),
                "direction":  bridge_result.get("direction"),
            },
        }

    # ── Session lifecycle ─────────────────────────────────────────────────

    def _open_session(self):
        if self.mm:
            try:
                self.mm.set_last_topic(ROVER_TOPIC)
                print(f"[ROVER] Session opened → topic={ROVER_TOPIC}")
            except Exception:
                pass
        self.watcher.start()

    def _close_session(self):
        self.watcher.stop()
        if self.mm:
            try:
                if (self.mm.get_last_topic() or "").strip() == ROVER_TOPIC:
                    self.mm.set_last_topic(None)
                    print(f"[ROVER] Session closed → topic cleared")
            except Exception:
                pass
        self.session_active = False

    def _store_turn(self, user_text: str, reply: str):
        try:
            store_short_term_memory(user_text, reply, topic=ROVER_TOPIC)
        except Exception:
            pass

    # ── Disconnect / error ────────────────────────────────────────────────

    def _handle_disconnect(self, user_text: str = "") -> dict:
        self._emergency_stop_silent()
        msg = self.rover_bridge.disconnect_message()
        rover_state = {"direction": "stop", "speed_percent": 0,
                       "session_active": False, "error": "connection lost"}
        natural_msg = self.personality.personalize(msg, rover_state)
        self._store_turn(user_text, natural_msg)
        self._close_session()
        return {"handled": True, "reply": natural_msg, "session_active": False}

    def _emergency_stop_silent(self):
        try:
            self.rover_bridge.emergency_stop()
        except Exception:
            pass
        self.current_direction = "stop"
        self.current_speed     = 0

    def _compose_error_reply(self, error: str, tb: str = "") -> str:
        raw = (
            f"Lost connection with the rover. Reason: {error}. "
            "Rover has been stopped as a precaution."
        )
        rover_state = {"direction": "stop", "speed_percent": 0,
                       "session_active": False, "error": error}
        return self.personality.personalize(raw, rover_state)

    # ── Context / parsing ─────────────────────────────────────────────────

    def _belongs_to_rover_context(self, text: str) -> bool:
        rover_words = ["rover", "motor", "robot", "lidar"]
        followup_words = ["forward", "backward", "back", "left", "right",
                        "stop", "faster", "slower", "speed", "follow", "find"]
        # ── ADD: context continuation words when session was recently active ──
        explanation_words = ["booting", "boot", "starting", "powering", "connecting",
                            "trying", "not working", "failing", "can't connect",
                            "won't connect", "no connection"]
        
        if any(w in text for w in rover_words):
            return True
        if self.session_active and any(w in text for w in followup_words):
            return True
        # ── ADD: if we recently had a session, treat explanations as rover context ──
        if any(w in text for w in explanation_words):
            return True
        return False

    def _parse_command(self, text: str):
        """
        Extended command parser — handles both movement and vision mode commands.
    
        New commands added:
            "follow me" / "start following" / "follow"  → set_mode FOLLOW_ME
            "find door" / "search for door" / "find a door" → set_mode FIND_DOOR
            "stop following" / "exit follow"             → set_mode IDLE
            "stop"  → set_mode IDLE + stop motors (existing stop extended)
        """
    
        # ── Vision mode commands ───────────────────────────────────────────────
        follow_triggers = ("follow me", "start following", "start follow", "follow mode")
        if any(p in text for p in follow_triggers) or (
            "follow" in text and "stop" not in text and "face" not in text
        ):
            self.current_direction  = "stop"
            self.current_speed      = 0
            self._current_mode      = "FOLLOW_ME"
            return self._make_mode_command("FOLLOW_ME")
    
        # In the door_triggers tuple — add "find the door"
        door_triggers = ("find door", "find a door", "search for door",
                        "search door", "look for door", "locate door",
                        "find the door")  
        if any(p in text for p in door_triggers):
            self.current_direction  = "stop"
            self.current_speed      = 0
            self._current_mode      = "FIND_DOOR"
            return self._make_mode_command("FIND_DOOR")
    
        stop_vision_triggers = ("stop following", "exit follow", "cancel follow",
                                "stop vision", "manual mode")
        if any(p in text for p in stop_vision_triggers):
            self.current_direction  = "stop"
            self.current_speed      = 0
            self._current_mode      = "IDLE"
            return self._make_mode_command("IDLE")
    
        # ── Original movement commands (unchanged) ─────────────────────────────
        if any(w in text for w in ("activate", "start", "wake", "online", "connect")):
            self.current_direction = "stop"
            self.current_speed     = 0
            return self._make_command("status", "stop", 0)
    
        if "stop" in text:
            self.current_direction  = "stop"
            self.current_speed      = 0
            self._current_mode      = "IDLE"
            # Send both stop movement AND set_mode IDLE so vision loop also stops
            # We send set_mode IDLE — rover_server.py stop handler covers motors
            return self._make_mode_command("IDLE")
    
        if "faster" in text:
            self.current_speed = min(self.current_speed + self.SPEED_STEP, self.MAX_SPEED)
            if self.current_speed == 0:
                self.current_speed = self.DEFAULT_SPEED
            return self._make_command("set_speed", self.current_direction, self.current_speed)
    
        if "slower" in text:
            self.current_speed = max(self.current_speed - self.SPEED_STEP, self.MIN_SPEED)
            return self._make_command("set_speed", self.current_direction, self.current_speed)
    
        if any(p in text for p in ("full capacity", "full speed", "max speed")):
            self.current_speed = self.MAX_SPEED
            return self._make_command("set_speed", self.current_direction, self.current_speed)
    
        # In the backward check — add "reverse"
        if "backward" in text or "go back" in text or "reverse" in text or re.search(r"\bback\b", text):
            self.current_direction = "backward"
            if self.current_speed == 0:
                self.current_speed = self.DEFAULT_SPEED
            return self._make_command("move", "backward", self.current_speed)

        # In the forward check — add "straight"
        if "forward" in text or "straight" in text:   # ← added straight
            self.current_direction = "forward"
            if self.current_speed == 0:
                self.current_speed = self.DEFAULT_SPEED
            return self._make_command("move", "forward", self.current_speed)
    
        # In the full speed check — add "maximum speed"
        if any(p in text for p in ("full capacity", "full speed", "max speed", "maximum speed")):   # ← added
            self.current_speed = self.MAX_SPEED
            return self._make_command("set_speed", self.current_direction, self.current_speed)

        if "left" in text:
            self.current_direction = "left"
            if self.current_speed == 0:
                self.current_speed = self.DEFAULT_SPEED
            return self._make_command("turn", "left", self.current_speed)
    
        if "right" in text:
            self.current_direction = "right"
            if self.current_speed == 0:
                self.current_speed = self.DEFAULT_SPEED
            return self._make_command("turn", "right", self.current_speed)
    
        FRUSTRATION_WORDS = ["failing", "not working", "broken", "won't work", 
                            "keeps failing", "problem", "trouble", "booting",
                            "not booting", "won't boot", "won't start"]
        if any(w in text for w in FRUSTRATION_WORDS):
            return self._make_command("status", "stop", 0)

        return None
    
    
    # ── New helper — add this to RoverTaskManager ──────────────────────────────────
    def _make_mode_command(self, mode: str) -> dict:
        """
        Build a set_mode command dict.
        rover_server.py handles this in the new mode branch of handle_client().
        """
        return {
            "device":    "rover",
            "action":    "set_mode",
            "mode":      mode,
            "timestamp": time.time(),
        }

    def _make_command(self, action: str, direction: str, speed: int) -> dict:
        return {
            "device":        "rover",
            "action":        action,
            "direction":     direction,
            "speed_percent": speed,
            "timestamp":     time.time(),
        }
        
    
    # ── Extended _build_reply — replace existing one ──────────────────────────────
    def _build_reply(self, cmd: dict) -> str:
        action    = cmd["action"]
        direction = cmd.get("direction", "stop")
        speed     = cmd.get("speed_percent", 0)
        mode      = cmd.get("mode", "")
    
        # New mode replies
        if action == "set_mode":
            if mode == "FOLLOW_ME":
                return "Rover is now in follow mode. I will track you."
            if mode == "FIND_DOOR":
                return "Rover is searching for a door."
            if mode == "IDLE":
                return "Rover vision stopped. Back to manual control."
    
        # Original movement replies
        if action == "status":    return "Rover is online and ready for commands."
        if action == "stop":      return "Rover stopped."
        if action == "move":      return f"Rover moving {direction} at {speed} percent speed."
        if action == "turn":      return f"Rover turning {direction} at {speed} percent speed."
        if action == "set_speed": return f"Rover speed adjusted to {speed} percent."
        return "Rover command sent."

    def _expire_if_needed(self):
        if not self.session_active:
            return
        if (time.time() - self.last_updated) > self.SESSION_TIMEOUT_SEC:
            print(f"[ROVER] Session expired after {self.SESSION_TIMEOUT_SEC}s idle")
            self._close_session()

    def _norm(self, text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text
    

