# rover_bridge.py
"""
rover_client.py  –  Windows PC side
Async WebSocket client that talks to rover_server.py on the Raspberry Pi.
Used by RoverBridge (the sync shim that RoverTaskManager calls).
"""

import asyncio
import json
import threading
import time
import traceback
from typing import Optional

from .rover_discovery import RoverConnectionDiscovery


# ──────────────────────────────────────────────
#  Low-level async client
# ──────────────────────────────────────────────
class _AsyncRoverClient:
    """Runs inside a dedicated background thread with its own event loop."""

    RECONNECT_DELAY_SEC = 3.0
    CONNECT_TIMEOUT_SEC = 5.0
    SEND_TIMEOUT_SEC    = 4.0

    def __init__(self, host: str, port: int):
        self.uri  = f"ws://{host}:{port}"
        self._ws  = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        self._stop_flag = False

        # last known telemetry from rover
        self.last_telemetry: dict = {}
        # last disconnection reason (for user-facing message)
        self.last_disconnect_reason: str = ""

    # ── lifecycle ──────────────────────────────
    def start(self):
        """Spin up the background thread + event loop."""
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_loop, daemon=True, name="RoverWSClient")
        t.start()

    def stop(self):
        self._stop_flag = True
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ── connection ─────────────────────────────
    async def _connect(self):
        import websockets  # lazy: the app still starts when websockets is not installed
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.uri),
                timeout=self.CONNECT_TIMEOUT_SEC,
            )
            self._connected = True
            self.last_disconnect_reason = ""
            print(f"[ROVER] Connected to {self.uri}")
        except Exception as e:
            self._connected = False
            self.last_disconnect_reason = f"Connection failed: {e}"
            self._ws = None

    # ── send / receive ─────────────────────────
    async def _send_async(self, command: dict) -> dict:
        """
        Attempt to send a command and wait for one ack.
        Returns a telemetry dict with at minimum {"ok": bool, "error": str|None}.
        """
        if not self._connected or self._ws is None:
            await self._connect()

        if not self._connected:
            return {
                "ok": False,
                "error": self.last_disconnect_reason or "Not connected to rover.",
            }

        try:
            payload = json.dumps(command)
            await asyncio.wait_for(self._ws.send(payload), timeout=self.SEND_TIMEOUT_SEC)

            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.SEND_TIMEOUT_SEC)
            ack  = json.loads(raw)

            # cache telemetry for status queries
            self.last_telemetry = ack
            return ack

        except Exception as e:
            tb = traceback.format_exc()
            reason = f"{type(e).__name__}: {e}"
            self.last_disconnect_reason = reason
            self._connected = False
            self._ws = None

            print(f"[ROVER] Send error:\n{tb}")
            return {
                "ok": False,
                "error": reason,
                "traceback": tb,
            }

    def send_command(self, command: dict) -> dict:
        """
        Blocking call from the sync world (RoverBridge).
        Submits coroutine to the background loop and waits.
        """
        if self._loop is None or not self._loop.is_running():
            return {"ok": False, "error": "Rover client loop is not running."}

        future = asyncio.run_coroutine_threadsafe(
            self._send_async(command), self._loop
        )
        try:
            return future.result(timeout=self.SEND_TIMEOUT_SEC + 1)
        except Exception as e:
            tb = traceback.format_exc()
            return {"ok": False, "error": str(e), "traceback": tb}

    # ── helpers ────────────────────────────────
    @property
    def is_connected(self) -> bool:
        return self._connected


# ──────────────────────────────────────────────
#  Sync bridge (used by RoverTaskManager)
# ──────────────────────────────────────────────
class RoverClient:
    """
    Thin synchronous wrapper around _AsyncRoverClient.
    This is what ConversationManager and RoverTaskManager call.
    """

    def __init__(
        self,
        host: str = "ubuntu.local",
        port: int = 8765,
        discovery: Optional[RoverConnectionDiscovery] = None,
        enable_discovery: bool = True,
    ):
        self.host = host
        self.port = port
        self._discovery = discovery or RoverConnectionDiscovery.from_env(hostname=host)
        self._enable_discovery = enable_discovery
        self._resolved_host: Optional[str] = None
        self._client = _AsyncRoverClient(host=host, port=port)
        self._client.start()

    # ── command ────────────────────────────────
    def send_command(self, command: dict) -> dict:
        """
        Send a command dict to the rover.
        Returns:
            {
                "ok": bool,
                "direction": str,
                "speed": int,
                "battery": int,          # 0-100
                "cpu_temp_c": float,
                "uptime_s": float,
                "error": str | None,
                "user_message": str,     # human-readable result
            }
        """
        discovery_error = self._prepare_connection(skip_cache=False)
        if discovery_error:
            result = {"ok": False, "error": discovery_error}
            result["user_message"] = self._build_user_message(command, result)
            return result

        result = self._client.send_command(command)
        if self._should_retry_after_discovery(result):
            retry_error = self._prepare_connection(skip_cache=True)
            if not retry_error:
                result = self._client.send_command(command)

        if result.get("ok") and self._resolved_host and self._enable_discovery:
            self._discovery.cache_success(self._resolved_host)

        result["user_message"] = self._build_user_message(command, result)
        return result

    def emergency_stop(self) -> dict:
        import traceback
        print("[DEBUG] emergency_stop called from:")
        traceback.print_stack(limit=6)
        stop_cmd = {
            "action": "stop",
            "direction": "stop",
            "speed_percent": 0,
        }
        return self.send_command(stop_cmd)

    # ── status ─────────────────────────────────
    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    @property
    def last_telemetry(self) -> dict:
        return self._client.last_telemetry

    @property
    def last_disconnect_reason(self) -> str:
        return self._client.last_disconnect_reason

    def disconnect_message(self) -> str:
        """
        Compose a user-facing message when the rover link drops.
        Includes the traceback snippet if available.
        """
        reason = self._client.last_disconnect_reason or "unknown reason"
        lines = [
            "I think I lost the connection with the rover.",
            f"Reason: {reason}",
        ]
        return " ".join(lines)

    # ── private ────────────────────────────────
    def _prepare_connection(self, skip_cache: bool) -> Optional[str]:
        if self._resolved_host and not skip_cache:
            return None

        if not self._enable_discovery:
            if self._resolved_host is None:
                self._resolved_host = self.host
            return None

        discovered_host = self._discovery.discover(port=self.port, skip_cache=skip_cache)
        if not discovered_host:
            return (
                "Rover discovery failed after checking cached IP, mDNS hostname, "
                "and local subnet scan."
            )

        if discovered_host != self._resolved_host:
            self._replace_client(discovered_host)
        return None

    def _replace_client(self, host: str) -> None:
        self._client.stop()
        self._resolved_host = host
        self._client = _AsyncRoverClient(host=host, port=self.port)
        self._client.start()

    def _should_retry_after_discovery(self, result: dict) -> bool:
        if result.get("ok") or not self._enable_discovery:
            return False
        err = result.get("error", "")
        return "Connection failed" in err or "Not connected" in err

    def _build_user_message(self, command: dict, result: dict) -> str:
        if not result.get("ok"):
            err = result.get("error", "unknown error")
            # Distinguish a failed connect from a mid-session drop
            if "Connection failed" in err or "Not connected" in err or "Rover discovery failed" in err:
                return (
                    "I can't reach the rover right now — "
                    "I checked the saved IP, mDNS hostname, and local subnet scan. "
                    "Make sure the Pi is on and rover_server.py is running."
                )
            tb  = result.get("traceback", "")
            msg = f"Rover command failed — {err}."
            if tb:
                last_line = [l.strip() for l in tb.strip().splitlines() if l.strip()][-1]
                msg += f" ({last_line})"
            return msg

        action    = command.get("action", "")
        direction = command.get("direction", "")
        speed     = command.get("speed_percent", 0)
        battery   = result.get("battery")
        bat_str   = f" Battery at {battery}%." if battery is not None else ""

        if action == "stop":
            return f"Rover stopped.{bat_str}"
        if action == "move":
            return f"Rover moving {direction} at {speed}%.{bat_str}"
        if action == "turn":
            return f"Rover turning {direction} at {speed}%.{bat_str}"
        if action == "set_speed":
            return f"Rover speed set to {speed}%.{bat_str}"
        return f"Rover command sent.{bat_str}"
