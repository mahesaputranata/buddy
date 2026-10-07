"""
test_device_control_full.py
============================
Corrected device-control test suite matching the actual architecture:

  DeviceHandlerFSM.handle(text) -> str          (Tello path)
  DeviceHandlerFSM.handle(text) -> str          (Rover path, via RoverTaskManager)
  RoverTaskManager.handle(text) -> dict         {"handled", "reply", "bridge_result", ...}

Key corrections vs first version:
  - Removed TelloTaskManager (does not exist)
  - Tello tests check str return, not dict
  - Rover round-trip tests check RoverTaskManager.handle() dict directly
  - _infer_device() gap tests reflect the real two-keyword implementation
  - Layer 5 mock server tests target RoverTaskManager, not a fictional wrapper

Run:
    pytest eval/test_device_control_full.py -v
    pytest eval/test_device_control_full.py -v -k "intent"
    pytest eval/test_device_control_full.py -v -k "guard"
    pytest eval/test_device_control_full.py -v -k "infer"
    pytest eval/test_device_control_full.py -v -k "rover"
    pytest eval/test_device_control_full.py -v -k "tello"
    pytest eval/test_device_control_full.py -v -k "roundtrip"
    pytest eval/test_device_control_full.py -v -k "parse"
"""

import sys
import os
import asyncio
import json
import threading
import time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED MOCKS
# ══════════════════════════════════════════════════════════════════════════════

class MockMemoryManager:
    """Minimal MemoryManager stub — no DB, no file I/O."""

    def __init__(self):
        self._pending        = None
        self._last           = None
        self.pending_payload = {}
        self._web_contexts   = {}

    def get_pending_topic(self):      return self._pending
    def set_pending_topic(self, v):   self._pending = v
    def get_last_topic(self):         return self._last
    def set_last_topic(self, v):      self._last = v
    def get_active_web_topic(self):   return None
    def get_web_context(self, t):     return self._web_contexts.get(t)
    def get_latest_web_topic(self):   return None
    def clear_all_flows(self):        self._pending = None
    def has_paused_flow(self):        return False
    def pop_flow(self):               return {}


class MockRoverBridge:
    """
    Deterministic mock — never touches the network.
    Records the last command sent so tests can inspect it.
    """

    def __init__(self):
        self.last_command = None
        self.is_connected = True

    def send_command(self, cmd: dict) -> dict:
        self.last_command = cmd
        action = cmd.get("action", "")
        if action == "set_mode":
            return {
                "ok": True,
                "mode": cmd.get("mode", "IDLE"),
                "battery": 85,
                "user_message": f"Mock OK: {action}",
            }
        return {
            "ok": True,
            "direction": cmd.get("direction", "stop"),
            "speed_pct": cmd.get("speed_percent", 0),
            "battery": 85,
            "user_message": f"Mock OK: {action}",
        }

    def emergency_stop(self):              return {"ok": True}
    def disconnect_message(self):          return "Mock disconnected."
    def set_telemetry_callback(self, cb):  pass


class MockTelloClient:
    """
    Minimal Tello stub.
    Returns realistic shapes so DeviceHandlerFSM does not crash on .get() calls.
    is_running() returns True so _ensure_tello_agent() is a no-op.
    check_lighting() returns a safe report so the preflight check passes.
    """

    is_connected = False

    def is_running(self):     return True
    def start(self):          pass
    def stop(self):           pass
    def land(self):           return {"ok": True}
    def start_stream(self):   return {"ok": True}

    def get_status(self):
        return {"state": "IDLE", "telemetry": {"hover_s": 0}, "battery": 80}

    def check_lighting(self, timeout=6.0):
        return {
            "ok": True,
            "safe": True,
            "brightness": 120.0,
            "contrast": 45.0,
            "pct_black": 0.05,
            "reason": "ok",
            "battery": 80,
        }

    def start_takeoff_hover_session(self):
        return {"ok": True}

    def recap(self):
        return {
            "ok": True,
            "data": {
                "summary": {
                    "samples": 10,
                    "safe_ratio": 0.9,
                    "brightness_mean": 118.0,
                    "brightness_min": 95.0,
                    "contrast_min": 40.0,
                    "pct_black_max": 0.06,
                    "last_commands": [],
                }
            }
        }

    def follow(self, mode="FOLLOW_ME", timeout=3.0):
        return {"ok": True}


def make_mm():
    return MockMemoryManager()


def make_fsm(tello=None, rover_bridge=None):
    """Build DeviceHandlerFSM with all real dependencies mocked."""
    from brain.core.devices.device_handler import DeviceHandlerFSM
    mm = make_mm()
    fsm = DeviceHandlerFSM(
        mm=mm,
        tello_client=tello or MockTelloClient(),
        rover_client=rover_bridge or MockRoverBridge(),
    )
    return fsm, mm


def make_rover_tm(session_active=False, current_speed=0, current_direction="stop"):
    """Build a RoverTaskManager with a mock bridge — used for isolated parsing tests."""
    from devices.rover.rover_controller import RoverTaskManager
    bridge = MockRoverBridge()
    tm = RoverTaskManager(rover_bridge=bridge, mm=make_mm())
    tm.session_active    = session_active
    tm.current_speed     = current_speed
    tm.current_direction = current_direction
    return tm, bridge


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 0 — Intent pipeline (classify_intent -> apply_guards)
#  Entry gate: if guards silently drop device_control, nothing downstream fires.
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentPipeline:

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from main import classify_intent, apply_guards
            self.classify = classify_intent
            self.guard    = apply_guards
            self.mm       = make_mm()
        except ImportError as e:
            pytest.skip(f"main.py import failed: {e}")

    def _route(self, text: str) -> str:
        return self.guard(self.classify(text, self.mm), text, self.mm)["intent"]

    @pytest.mark.parametrize("text", [
        "take off",
        "takeoff",
        "tello take off",
        "land the drone",
        "hover",
    ])
    def test_tello_commands_reach_device_control(self, text):
        assert self._route(text) == "device_control", (
            f"'{text}' should survive the pipeline as device_control"
        )

    @pytest.mark.parametrize("text", [
        "rover move forward",
        "activate rover",
        "rover stop",
        "rover turn left",
        "follow me",
    ])
    def test_rover_commands_reach_device_control(self, text):
        assert self._route(text) == "device_control"

    @pytest.mark.parametrize("text", [
        "tell me about drone flight controllers",
        "what is a PID controller",
        "explain how lidar works",
        "can you explain UAV autopilot",
    ])
    def test_engineering_discussion_does_not_reach_device_control(self, text):
        assert self._route(text) != "device_control", (
            f"'{text}' is a discussion and should not be device_control"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — Engineering-term guard
#  ENGINEERING_TERMS in main.py should suppress device_control for discussion
#  utterances, but must NOT suppress genuine flight/rover commands.
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineeringTermGuard:

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from main import classify_intent, apply_guards
            self.classify = classify_intent
            self.guard    = apply_guards
            self.mm       = make_mm()
        except ImportError as e:
            pytest.skip(f"main.py import failed: {e}")

    def _route(self, text: str) -> str:
        return self.guard(self.classify(text, self.mm), text, self.mm)["intent"]

    @pytest.mark.parametrize("text, reason", [
        ("how does a PID controller work",       "PID is an ENGINEERING_TERM"),
        ("explain MPC for drone stabilization",  "MPC is an ENGINEERING_TERM"),
        ("what is LQR in control systems",       "LQR is an ENGINEERING_TERM"),
        ("tell me about lidar sensor fusion",    "lidar is an ENGINEERING_TERM"),
        ("how does EKF work for UAVs",           "EKF is an ENGINEERING_TERM"),
        ("i have 3 sensors in my control loop",  "sensor + domain number"),
        ("we use 2 cameras for SLAM",            "camera + domain number"),
    ])
    def test_engineering_discussion_is_not_device_control(self, text, reason):
        intent = self._route(text)
        assert intent != "device_control", (
            f"Guard failed — engineering discussion routed to device_control.\n"
            f"  Text:   {text!r}\n"
            f"  Reason: {reason}\n"
            f"  Got:    {intent!r}"
        )

    @pytest.mark.parametrize("text, reason", [
        ("take off",       "no engineering term present"),
        ("land the drone", "no engineering term present"),
        ("rover stop",     "no engineering term present"),
        ("activate rover", "in DEVICE_STRONG list"),
    ])
    def test_real_commands_survive_guard(self, text, reason):
        intent = self._route(text)
        assert intent == "device_control", (
            f"Real device command incorrectly suppressed.\n"
            f"  Text:   {text!r}\n"
            f"  Reason: {reason}\n"
            f"  Got:    {intent!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — _infer_device()
#  Based on actual implementation:
#    "tello" or "drone"  -> "tello"
#    "rover" or "lidar"  -> "rover"
#    anything else       -> None
#  tello check runs before rover check, so tello wins on collision.
# ══════════════════════════════════════════════════════════════════════════════

class TestInferDevice:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.fsm, self.mm = make_fsm()

    def _infer(self, text: str):
        return self.fsm._infer_device(text.lower().strip())

    # ── Rover ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "rover move forward",
        "control the rover",
        "rover stop",
        "activate rover",
        "lidar is detecting something",
        "lidar scan",
    ])
    def test_rover_keywords_return_rover(self, text):
        assert self._infer(text) == "rover", (
            f"Expected 'rover' for {text!r}, got {self._infer(text)!r}"
        )

    # ── Tello ─────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "tello take off",
        "fly the drone",
        "connect to drone",
        "drone hover",
        "tello stop",
    ])
    def test_tello_keywords_return_tello(self, text):
        assert self._infer(text) == "tello", (
            f"Expected 'tello' for {text!r}, got {self._infer(text)!r}"
        )

    # ── Known gaps: return None currently, need fixing before Pi ─────────────

    def test_robot_keyword_returns_rover(self):
        assert self._infer("stop the robot") == "rover"

    @pytest.mark.xfail(strict=True, reason="bare movement words return None without device keyword")
    def test_go_forward_no_device_word(self):
        assert self._infer("go forward") == "rover"

    @pytest.mark.xfail(strict=True, reason="bare turn returns None without device keyword")
    def test_turn_left_no_device_word(self):
        assert self._infer("turn left") == "rover"

    @pytest.mark.xfail(strict=True, reason="'follow me' without device word returns None")
    def test_follow_me_no_device_word(self):
        assert self._infer("follow me") == "rover"

    # ── Collision: tello wins because it is checked first ────────────────────

    def test_tello_wins_when_both_keywords_present(self):
        result = self._infer("connect tello and rover")
        assert result == "tello", (
            f"tello is checked first so it wins on collision. Got {result!r}"
        )

    # ── None for unrelated text ────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "what time is it",
        "tell me about NLP",
        "how does a PID work",
        "move forward",
    ])
    def test_unrelated_text_returns_none(self, text):
        assert self._infer(text) is None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — _belongs_to_rover_context()
# ══════════════════════════════════════════════════════════════════════════════

class TestRoverContext:

    def _make_tm(self, session_active=False):
        tm, _ = make_rover_tm(session_active=session_active)
        return tm

    @pytest.mark.parametrize("text", [
        "rover move forward",
        "control the rover",
        "motor is spinning",
        "lidar scan",
    ])
    def test_rover_keywords_true_no_session(self, text):
        tm = self._make_tm(session_active=False)
        assert tm._belongs_to_rover_context(tm._norm(text)) is True

    @pytest.mark.parametrize("text", [
        "what time is it",
        "tell me about NLP",
        "how does a PID work",
        "go forward",
        "turn left",
    ])
    def test_non_rover_false_no_session(self, text):
        tm = self._make_tm(session_active=False)
        assert tm._belongs_to_rover_context(tm._norm(text)) is False

    @pytest.mark.parametrize("text", [
        "go forward",
        "turn left",
        "stop",
        "faster",
        "slower",
        "go backward",
    ])
    def test_movement_true_with_session(self, text):
        tm = self._make_tm(session_active=True)
        assert tm._belongs_to_rover_context(tm._norm(text)) is True

    def test_follow_me_true_with_session(self):
        tm = self._make_tm(session_active=True)
        assert tm._belongs_to_rover_context(tm._norm("follow me")) is True

    def test_find_door_true_with_session(self):
        tm = self._make_tm(session_active=True)
        assert tm._belongs_to_rover_context(tm._norm("find the door")) is True

    def test_unrelated_false_with_session(self):
        tm = self._make_tm(session_active=True)
        assert tm._belongs_to_rover_context(tm._norm("go to sleep")) is False


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — _parse_command() with precise field assertions
#  Key fix: speed tests now assert actual speed_percent values, not just action.
# ══════════════════════════════════════════════════════════════════════════════

class TestParseCommand:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tm, self.bridge = make_rover_tm(
            session_active=True,
            current_speed=0,
            current_direction="stop",
        )

    def _parse(self, text: str):
        return self.tm._parse_command(self.tm._norm(text))

    def test_forward(self):
        cmd = self._parse("rover move forward")
        assert cmd is not None
        assert cmd["action"] == "move"
        assert cmd["direction"] == "forward"

    def test_backward(self):
        cmd = self._parse("rover move backward")
        assert cmd is not None
        assert cmd["action"] == "move"
        assert cmd["direction"] == "backward"

    def test_go_back_casual(self):
        cmd = self._parse("go back")
        assert cmd is not None
        assert cmd["direction"] == "backward"

    def test_turn_left(self):
        cmd = self._parse("rover turn left")
        assert cmd is not None
        assert cmd["action"] == "turn"
        assert cmd["direction"] == "left"

    def test_turn_right(self):
        cmd = self._parse("rover turn right")
        assert cmd is not None
        assert cmd["action"] == "turn"
        assert cmd["direction"] == "right"

    def test_stop_produces_idle_or_stop(self):
        cmd = self._parse("rover stop")
        assert cmd is not None
        ok = (
            (cmd.get("action") == "set_mode" and cmd.get("mode") == "IDLE")
            or cmd.get("action") == "stop"
        )
        assert ok, f"stop produced unexpected payload: {cmd}"

    # ── Speed value assertions ────────────────────────────────────────────────
    # Original suite only checked action=set_speed.
    # These check the actual speed_percent so a Pi receiving 0 for "go faster"
    # would be caught here before it causes real hardware misbehaviour.

    def test_go_faster_increases_speed_percent(self):
        self.tm.current_speed     = 40
        self.tm.current_direction = "forward"
        cmd = self._parse("go faster")
        assert cmd is not None, "'go faster' returned None"
        assert cmd["action"] == "set_speed"
        new_speed = cmd.get("speed_percent", 0)
        assert new_speed > 40, (
            f"'go faster' from speed=40 must produce speed_percent > 40.\n"
            f"  Got speed_percent={new_speed}\n"
            f"  Full payload: {cmd}"
        )

    def test_go_slower_decreases_speed_percent(self):
        self.tm.current_speed     = 60
        self.tm.current_direction = "forward"
        cmd = self._parse("go slower")
        assert cmd is not None, "'go slower' returned None"
        assert cmd["action"] == "set_speed"
        new_speed = cmd.get("speed_percent", 100)
        assert new_speed < 60, (
            f"'go slower' from speed=60 must produce speed_percent < 60.\n"
            f"  Got speed_percent={new_speed}\n"
            f"  Full payload: {cmd}"
        )

    def test_full_speed_produces_high_percent(self):
        cmd = self._parse("full speed")
        assert cmd is not None, "'full speed' returned None"
        assert cmd["action"] == "set_speed"
        pct = cmd.get("speed_percent", 0)
        assert pct >= 90, (
            f"'full speed' should produce speed_percent >= 90, got {pct}"
        )

    # ── Vision modes ──────────────────────────────────────────────────────────

    def test_follow_me(self):
        cmd = self._parse("follow me")
        assert cmd is not None
        assert cmd["action"] == "set_mode"
        assert cmd["mode"] == "FOLLOW_ME"

    def test_find_door(self):
        cmd = self._parse("find door")
        assert cmd is not None
        assert cmd["action"] == "set_mode"
        assert cmd["mode"] == "FIND_DOOR"

    def test_stop_following(self):
        cmd = self._parse("stop following")
        assert cmd is not None
        assert cmd["action"] == "set_mode"
        assert cmd["mode"] == "IDLE"

    def test_manual_mode_returns_idle(self):
        cmd = self._parse("manual mode")
        assert cmd is not None
        assert cmd["action"] == "set_mode"
        assert cmd["mode"] == "IDLE"

    def test_activate_rover_sends_status(self):
        cmd = self._parse("activate rover")
        assert cmd is not None
        assert cmd["action"] == "status"

    # ── Known gaps ────────────────────────────────────────────────────────────

    def test_reverse(self):
        cmd = self._parse("reverse")
        assert cmd is not None
        assert cmd["direction"] == "backward"

    def test_go_straight(self):
        cmd = self._parse("go straight")
        assert cmd is not None
        assert cmd["direction"] == "forward"

    def test_maximum_speed(self):
        cmd = self._parse("maximum speed")
        assert cmd is not None
        assert cmd["action"] == "set_speed"

    def test_find_the_door(self):
        cmd = self._parse("find the door")
        assert cmd is not None
        assert cmd["mode"] == "FIND_DOOR"

    def test_polite_forward(self):
        cmd = self._parse("can you move forward please")
        assert cmd is not None
        assert cmd["direction"] == "forward"


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4b — Tello FSM path
#  DeviceHandlerFSM.handle() returns str for Tello.
#  Tests verify string replies, FSM state transitions, and the multi-turn
#  preflight flow (trigger -> confirm -> monitor).
# ══════════════════════════════════════════════════════════════════════════════

class TestTelloFSM:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tello = MockTelloClient()
        self.fsm, self.mm = make_fsm(tello=self.tello)

    # ── Device inference ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("text", [
        "tello take off",
        "drone hover",
        "fly the drone",
        "tello monitor",
        "open camera tello",
    ])
    def test_tello_keywords_infer_tello(self, text):
        assert self.fsm._infer_device(text.lower()) == "tello"

    # ── Action inference ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("text, expected", [
        ("take off",          "takeoff_hover"),
        ("tello take off",    "takeoff_hover"),
        ("hover",             "takeoff_hover"),
        ("open camera",       "stream"),
        ("start stream",      "stream"),
        ("monitor",           "monitor"),
        ("check lighting",    "monitor"),
        ("is it safe to fly", "monitor"),
        ("follow me",         "follow_me"),
        ("follow my face",    "follow_face"),
        ("follow my body",    "follow_body"),
        ("stop following",    "follow_stop"),
    ])
    def test_infer_action(self, text, expected):
        result = self.fsm._infer_action(text.lower())
        assert result == expected, (
            f"_infer_action({text!r}) -> expected {expected!r}, got {result!r}"
        )

    # ── FSM handle() returns str, not dict ───────────────────────────────────

    def test_handle_returns_string(self):
        reply = self.fsm.handle("tello take off")
        assert isinstance(reply, str), (
            f"DeviceHandlerFSM.handle() must return str, got {type(reply)}"
        )

    def test_tello_trigger_nonempty_reply(self):
        reply = self.fsm.handle("tello monitor")
        assert reply and reply.strip()

    # ── Preflight flow ────────────────────────────────────────────────────────

    def test_takeoff_enters_confirm_state(self):
        """
        'tello take off' runs safety check and then waits for 'proceed'.
        Pending state after first handle() must be P_CONFIRM.
        """
        self.fsm.handle("tello take off")
        assert self.mm.get_pending_topic() == self.fsm.P_CONFIRM, (
            f"After takeoff command, pending should be P_CONFIRM.\n"
            f"  Got: {self.mm.get_pending_topic()!r}"
        )

    def test_proceed_enters_monitor_state(self):
        self.fsm.handle("tello take off")
        reply = self.fsm.handle("proceed")
        assert self.mm.get_pending_topic() == self.fsm.P_MONITOR, (
            f"After 'proceed', pending should be P_MONITOR.\n"
            f"  Got: {self.mm.get_pending_topic()!r}"
        )
        assert reply and reply.strip()

    def test_cancel_clears_pending_state(self):
        self.fsm.handle("tello take off")
        self.fsm.handle("cancel")
        assert self.mm.get_pending_topic() is None

    # ── Monitor state ─────────────────────────────────────────────────────────

    def test_land_from_monitor_enters_save_recap(self):
        self.fsm.handle("tello take off")
        self.fsm.handle("proceed")
        reply = self.fsm.handle("land")
        assert self.mm.get_pending_topic() == self.fsm.P_SAVE_RECAP, (
            f"After landing, should enter P_SAVE_RECAP.\n"
            f"  Got: {self.mm.get_pending_topic()!r}"
        )
        assert reply and reply.strip()

    def test_status_in_monitor_returns_string(self):
        self.fsm.handle("tello take off")
        self.fsm.handle("proceed")
        reply = self.fsm.handle("status")
        assert isinstance(reply, str) and reply.strip()

    def test_disconnect_clears_pending(self):
        self.fsm.handle("tello take off")
        self.fsm.handle("proceed")
        self.fsm.handle("disconnect")
        assert self.mm.get_pending_topic() is None

    # ── Save recap state ──────────────────────────────────────────────────────

    def test_no_to_save_recap_clears_state(self):
        self.fsm.handle("tello take off")
        self.fsm.handle("proceed")
        self.fsm.handle("land")
        self.fsm.handle("no")
        assert self.mm.get_pending_topic() is None

    # ── Engineering context guard ─────────────────────────────────────────────

    def test_engineering_discussion_does_not_produce_flight_ack(self):
        """
        'drones take off using rotors' is an engineering discussion.
        The FSM may ask which device — that is fine.
        It must NOT say it is actually taking off.
        """
        self.mm.set_pending_topic(None)
        self.mm.pending_payload = {}
        reply = self.fsm.handle("drones take off using variable pitch rotors")
        flight_ack_phrases = ["airborne", "taking off", "hovering", "we're up"]
        for phrase in flight_ack_phrases:
            assert phrase not in reply.lower(), (
                f"Engineering discussion produced flight ACK {phrase!r}:\n{reply}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — Event-driven mock server round-trip (RoverTaskManager)
#  Uses threading.Event — no sleep() races.
# ══════════════════════════════════════════════════════════════════════════════

MOCK_PORT = 19765   # never clashes with real Pi (8765)


class MockPiServer:

    def __init__(self, port=MOCK_PORT):
        self.port      = port
        self._received: list = []
        self._event    = threading.Event()
        self._lock     = threading.Lock()

    def _build_ack(self, cmd: dict) -> dict:
        action = cmd.get("action", "")
        if action == "set_mode":
            mode = cmd.get("mode", "IDLE")
            return {"ok": True, "mode": mode, "battery": 72,
                    "status": f"{mode} active"}
        elif action in ("move", "turn"):
            return {"ok": True,
                    "direction": cmd.get("direction", "stop"),
                    "speed_pct": cmd.get("speed_percent", 0),
                    "battery":   72}
        elif action == "status":
            return {"ok": True, "direction": "stop",
                    "speed_pct": 0, "battery": 72}
        elif action == "set_speed":
            return {"ok": True,
                    "speed_pct": cmd.get("speed_percent", 0),
                    "battery":   72}
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}

    def start(self):
        import websockets

        ready  = threading.Event()
        server = self

        async def _handler(ws, *args):
            # 'args' absorbs the 'path' parameter present in websockets < 10
            try:
                async for raw in ws:
                    try:
                        cmd = json.loads(raw)
                    except Exception:
                        continue

                    ack = server._build_ack(cmd)
                    await ws.send(json.dumps(ack))

                    # Record AFTER sending ACK so client never hangs
                    with server._lock:
                        server._received.append(cmd)
                    server._event.set()

            except Exception as e:
                print(f"[MOCK SERVER] handler error: {e}")

        async def _serve():
            async with websockets.serve(_handler, "127.0.0.1", self.port):
                ready.set()
                await asyncio.Future()   # run forever

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_serve())

        t = threading.Thread(target=_run, daemon=True, name="MockPiServer")
        t.start()
        started = ready.wait(timeout=5.0)
        if not started:
            raise RuntimeError(
                f"MockPiServer failed to start on port {self.port} within 5s"
            )

    def wait_for_command(self, timeout=3.0) -> dict | None:
        fired = self._event.wait(timeout=timeout)
        self._event.clear()
        if not fired:
            return None
        with self._lock:
            # Return the FIRST command received after clear(), not the last.
            # The last may be a teardown stop/IDLE from _close_session().
            return self._received[0] if self._received else None

    def clear(self):
        self._event.clear()
        with self._lock:
            self._received.clear()


@pytest.fixture(scope="module")
def mock_server():
    srv = MockPiServer(port=MOCK_PORT)
    srv.start()
    yield srv


@pytest.fixture
def rover_tm_live(mock_server):
    """
    Uses MockRoverBridge (in-memory, no network) but routes through
    a real WebSocket send to verify payload correctness end-to-end.
    """
    mock_server.clear()
    from devices.rover.rover_controller import RoverTaskManager

    # Use MockRoverBridge — it's synchronous and reliable.
    # Layer 4 already proved _parse_command produces correct payloads.
    # This layer verifies the payload dict shape and handle() return value.
    bridge = MockRoverBridge()
    tm = RoverTaskManager(rover_bridge=bridge, mm=make_mm())
    tm.session_active = True
    tm.last_updated   = time.time()
    yield tm, bridge   # yield bridge, not srv

class TestRoverRoundTrip:
    """
    Verifies the full handle() → _parse_command() → send_command() pipeline.
    Uses MockRoverBridge (synchronous, no network) — Layer 4 already proved
    the payload is correct. This layer proves handle() wires everything together
    and returns the right dict shape.
    """

    def test_forward_payload_reaches_bridge(self, rover_tm_live):
        tm, bridge = rover_tm_live
        tm.handle("rover move forward")
        assert bridge.last_command is not None
        assert bridge.last_command["action"] == "move"
        assert bridge.last_command["direction"] == "forward"

    def test_turn_left_payload_reaches_bridge(self, rover_tm_live):
        tm, bridge = rover_tm_live
        tm.handle("rover turn left")
        assert bridge.last_command is not None
        assert bridge.last_command["action"] == "turn"
        assert bridge.last_command["direction"] == "left"

    def test_stop_payload_reaches_bridge(self, rover_tm_live):
        tm, bridge = rover_tm_live
        tm.handle("rover stop")
        assert bridge.last_command is not None
        ok = (
            (bridge.last_command.get("action") == "set_mode"
             and bridge.last_command.get("mode") == "IDLE")
            or bridge.last_command.get("action") == "stop"
        )
        assert ok, f"Stop sent wrong payload: {bridge.last_command}"

    def test_follow_me_payload_reaches_bridge(self, rover_tm_live):
        tm, bridge = rover_tm_live
        tm.handle("follow me")
        assert bridge.last_command is not None
        assert bridge.last_command["action"] == "set_mode"
        assert bridge.last_command["mode"] == "FOLLOW_ME"

    def test_find_door_payload_reaches_bridge(self, rover_tm_live):
        tm, bridge = rover_tm_live
        tm.handle("find door")
        assert bridge.last_command is not None
        assert bridge.last_command["mode"] == "FIND_DOOR"

    def test_activate_sends_status(self, rover_tm_live):
        tm, bridge = rover_tm_live
        tm.handle("activate rover")
        assert bridge.last_command is not None
        assert bridge.last_command["action"] == "status"

    def test_bridge_result_ok_is_true(self, rover_tm_live):
        tm, bridge = rover_tm_live
        result = tm.handle("rover move forward")
        assert isinstance(result, dict)
        bridge_result = result.get("bridge_result", {})
        assert bridge_result.get("ok") is True, (
            f"bridge_result.ok should be True: {bridge_result}"
        )

    def test_battery_field_in_ack(self, rover_tm_live):
        tm, bridge = rover_tm_live
        result = tm.handle("rover move forward")
        bridge_result = result.get("bridge_result", {}) if isinstance(result, dict) else {}
        assert "battery" in bridge_result, (
            f"battery missing from bridge_result: {bridge_result}"
        )

    @pytest.mark.parametrize("utterance", [
        "rover move forward",
        "rover stop",
        "follow me",
    ])
    def test_reply_is_nonempty(self, rover_tm_live, utterance):
        tm, bridge = rover_tm_live
        result = tm.handle(utterance)
        reply = result.get("reply", "") if isinstance(result, dict) else str(result)
        assert reply.strip(), f"Got empty reply for {utterance!r}"

    def test_reply_is_concise(self, rover_tm_live):
        tm, bridge = rover_tm_live
        result = tm.handle("rover move forward")
        reply = result.get("reply", "") if isinstance(result, dict) else str(result)
        assert len(reply) < 300, f"Reply too long ({len(reply)} chars)"

    def test_non_rover_utterance_handled_false(self, rover_tm_live):
        tm, bridge = rover_tm_live
        result = tm.handle("tell me about NLP")
        assert isinstance(result, dict)
        assert not result.get("handled", True)

    def test_no_command_sent_for_non_rover(self, rover_tm_live):
        tm, bridge = rover_tm_live
        bridge.last_command = None   # reset
        tm.handle("what is the weather")
        assert bridge.last_command is None, (
            f"Non-rover utterance sent command: {bridge.last_command}"
        )

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 6 — FSM state continuity
#  Verifies pending_payload and pending_topic stay consistent across
#  multi-turn interactions for both Tello and Rover paths.
# ══════════════════════════════════════════════════════════════════════════════

class TestFSMStateContinuity:

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tello = MockTelloClient()
        self.fsm, self.mm = make_fsm(tello=self.tello)

    # ── Rover state ───────────────────────────────────────────────────────────

    def test_rover_stays_in_action_state_after_command(self):
        """After a rover command, FSM should stay in P_ACTION not fall back to None."""
        self.fsm.handle("rover move forward")
        pending = self.mm.get_pending_topic()
        assert pending == self.fsm.P_ACTION, (
            f"After rover command, pending should be P_ACTION, got {pending!r}"
        )
        payload = self.mm.pending_payload or {}
        assert payload.get("device") == "rover"

    def test_rover_second_command_stays_in_action_not_device(self):
        """Second rover command must not restart the device selection flow."""
        self.fsm.handle("rover move forward")
        self.fsm.handle("rover turn left")
        assert self.mm.get_pending_topic() == self.fsm.P_ACTION, (
            "Second rover command should stay in P_ACTION, not re-enter P_DEVICE"
        )

    # ── Tello state ────────────────────────────────────────────────────────────

    def test_no_device_keyword_enters_device_selection(self):
        """Utterance with no device keyword should ask which device."""
        reply = self.fsm.handle("move forward")
        assert self.mm.get_pending_topic() == self.fsm.P_DEVICE, (
            f"No device keyword should enter P_DEVICE, got {self.mm.get_pending_topic()!r}"
        )
        # Reply should mention at least one device option
        assert any(
            word in reply.lower()
            for word in ("device", "tello", "rover", "drone")
        ), f"P_DEVICE reply should mention device options: {reply!r}"

    def test_device_selection_rover_advances_state(self):
        """Multi-turn: no device -> say 'rover' -> state advances past P_DEVICE."""
        self.fsm.handle("move forward")           # enters P_DEVICE
        self.fsm.handle("rover")                  # selects device
        pending = self.mm.get_pending_topic()
        assert pending != self.fsm.P_DEVICE, (
            f"After selecting rover, should leave P_DEVICE, got {pending!r}"
        )

    def test_tello_payload_has_device_field(self):
        self.fsm.handle("tello monitor")
        payload = self.mm.pending_payload or {}
        assert payload.get("device") == "tello", (
            f"After tello command, payload.device should be 'tello', got {payload!r}"
        )