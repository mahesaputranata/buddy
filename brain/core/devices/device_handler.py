import re
import time

from brain.utils.confirmation import is_no, is_yes
from devices.rover.rover_controller import RoverTaskManager


class DeviceHandlerFSM:
    P_CONFIRM = "Awaiting_DeviceControl_Confirm"
    P_DEVICE  = "Awaiting_Device_Name"
    P_ROVER_CORRECTION = "Awaiting_Rover_Correction_Confirm"

    def __init__(self, mm, tello_client, rover_client=None, memory_handler=None, conversation_manager=None):
        self.mm = mm
        self.tello = tello_client
        self.memory_handler = memory_handler
        self.cm = conversation_manager
        self.rover_manager = RoverTaskManager(rover_bridge=rover_client, mm=self.mm)

    def handle(self, user_text: str) -> str:
        pending = self.mm.get_pending_topic()
        if pending == self.P_CONFIRM:
            return self._handle_confirm(user_text)
        if pending == self.P_DEVICE:
            return self._resume_device_pick(user_text)
        if pending == self.P_ROVER_CORRECTION:
            return self._handle_rover_correction_confirm(user_text)
        return self._start(user_text)

    def _resume_device_pick(self, user_text: str) -> str:
        device = self._infer_device(user_text)
        if not device:
            return "Sorry, I still didn't catch that — Tello or Rover?"
        self._clear_pending()
        payload = self.mm.pending_payload or {}
        payload["device"] = device
        payload["raw_text"] = payload.get("raw_text", "") + " " + user_text
        if device == "rover":
            return self._route_rover(payload)
        action = payload.get("action") or self._infer_action(user_text)
        if not action:
            return "Okay - what should I do with Tello? You can say: monitor, open camera, status, take off, land, or disconnect."
        return self._route_tello(action, payload)

    def _start(self, user_text: str) -> str:
        device = self._infer_device(user_text)
        action = self._infer_action(user_text)

        if not device:
            active_topic = (self.mm.get_last_topic() or "").strip().lower()
            if active_topic == "rover:session" or self.rover_manager.session_active:
                device = "rover"
                print(f"[DHF] no device named, defaulting to rover (active session)")

        if not device:
            self.mm.set_pending_topic(self.P_DEVICE)
            self.mm.pending_payload = {}
            return "Which device do you want to control? (Tello drone / Rover)"
        
        payload = {
            "device": device,
            "action": action,
            "raw_text": user_text,
        }
        self.mm.pending_payload = payload

        if device == "rover":
            return self._route_rover(payload)

        if not action:
            self._clear_pending()
            return "Okay - what should I do with Tello? You can say: monitor, open camera, status, take off, land, or disconnect."

        return self._route_tello(action, payload)

    def _route_rover(self, payload: dict) -> str:
        user_text = payload.get("raw_text", "")
        result = self.rover_manager.handle(user_text)
        handled = result.get("handled", False)

        if handled == "ambiguous":
            self.mm.set_pending_topic("Awaiting_Rover_Correction_Confirm")
            self.mm.pending_payload = {
                "raw_text": user_text,
                "correction": result.get("pending_correction", {}),
            }
            return result["reply"]

        if not handled:
            self._clear_pending()
            if (self.mm.get_last_topic() or "").strip() == "rover:session":
                self.mm.set_last_topic(None)
            return self.cm.handle_request(user_text) if self.cm else "I think this is no longer about rover control."

        self._clear_pending()
        self.mm.set_last_topic("rover:session")

        if result.get("connection_failed"):
            error_msg = result.get("error", "connection failed")
            return (
                "Hey, looks like I couldn't reach the rover right now. "
                f"Error: {error_msg}. Power it up, then say 'activate the rover' when you're ready."
            )

        return result.get("reply", "Rover command sent.")

    def _route_tello(self, action: str, payload: dict) -> str:
        if action in ("status", "land", "disconnect", "stop"):
            return self._route_tello_direct(action)

        ok, start_err = self._ensure_tello_agent()
        if not ok:
            self._clear_pending()
            return start_err

        if action == "stream":
            try:
                self.tello.start_stream()
            except Exception as e:
                self._clear_pending()
                return f"I couldn't start the Tello stream: {e}"
            self._clear_pending()
            return "Camera stream started."

        data, err = self._safe_lighting_check(timeout=6.0)
        if err:
            self._clear_pending()
            return err

        safe_report = data or {}
        payload["safety"] = {"last": safe_report}

        if action == "monitor":
            self._clear_pending()
            return self._format_lighting_report(safe_report)

        if action in ("takeoff_hover", "recovery"):
            if not safe_report.get("safe", False):
                self._clear_pending()
                return (
                    self._format_lighting_report(safe_report)
                    + "\n\nIt doesn't look safe to fly right now. Improve lighting and try again."
                )

            self.mm.set_pending_topic(self.P_CONFIRM)
            self.mm.pending_payload = payload
            battery = safe_report.get("battery")
            reason = safe_report.get("reason") or "ok"
            return (
                "Preflight check complete. "
                f"Battery: {battery if battery is not None else 'unknown'}%. "
                f"Lighting: safe ({reason}). "
                f"{self._format_lighting_report(safe_report)}\n\n"
                "Say proceed to take off and hover, or cancel to abort."
            )

        if action in ("follow_me", "follow_face", "follow_body", "follow_stop"):
            if action == "follow_stop":
                self._clear_pending()
                return self._apply_follow_mode(action)

            status = self.tello.get_status() or {}
            if status.get("state") != "HOVER":
                payload["desired_follow_mode"] = action
                payload["action"] = "takeoff_hover"
                self.mm.set_pending_topic(self.P_CONFIRM)
                self.mm.pending_payload = payload
                return (
                    "To start following, I need to take off and hover first. "
                    "Say proceed to take off, or cancel to abort."
                )

            self._clear_pending()
            return self._apply_follow_mode(action)

        self._clear_pending()
        return "I'm not sure which Tello action that is."

    def _route_tello_direct(self, action: str) -> str:
        if action == "status":
            self._clear_pending()
            return self._format_status(self.tello.get_status())

        if action == "land":
            return self._land_and_recap()

        if action in ("disconnect", "stop"):
            try:
                self.tello.stop()
            except Exception:
                pass
            self._clear_pending()
            return "Alright. I've disconnected from Tello."

        self._clear_pending()
        return "I'm not sure which Tello action that is."

    def _handle_rover_correction_confirm(self, user_text: str) -> str:
        payload = self.mm.pending_payload or {}
        correction = payload.get("correction", {})

        if is_yes(user_text):
            self._clear_pending()
            corrected = correction.get("corrected_text", payload.get("raw_text", ""))
            return self._route_rover({"raw_text": corrected})   # re-enter, now force-parsed

        if is_no(user_text):
            self._clear_pending()
            if (self.mm.get_last_topic() or "").strip() == "rover:session":
                self.mm.set_last_topic(None)
            return self.cm.handle_request(payload.get("raw_text", user_text)) if self.cm else "Okay, never mind."

        return "Sorry, was that a yes or no?"

    def _handle_confirm(self, user_text: str) -> str:
        t = (user_text or "").strip().lower()

        if "land" in t:
            return self._land_and_recap()

        if is_no(user_text):
            self._clear_pending()
            return "Okay - cancelled."

        if not (is_yes(user_text) or t == "proceed"):
            return "Please say proceed to continue or cancel to abort."

        payload = self.mm.pending_payload or {}
        action = payload.get("action")

        if action in ("takeoff_hover", "takeoff_hover_land"):
            desired = payload.get("desired_follow_mode")
            res = self.tello.start_takeoff_hover_session()
            if isinstance(res, dict) and not res.get("ok", True):
                self._clear_pending()
                return f"I couldn't take off: {res.get('error', 'unknown error')}."

            self._clear_pending()
            if desired:
                msg = self._apply_follow_mode(desired)
                return "We're airborne and hovering.\n" + msg
            return "We're airborne and hovering."

        self._clear_pending()
        return "I'm not sure what I was confirming."

    def _land_and_recap(self) -> str:
        try:
            self.tello.land()
        except Exception:
            pass

        try:
            rec = self.tello.recap()
        except Exception:
            rec = None

        if isinstance(rec, dict):
            summary = rec.get("data", {}).get("summary") or rec
        elif rec is None:
            summary = {"note": "No recap available (recap() failed)."}
        else:
            summary = {"raw": rec}

        self._clear_pending()
        return self.polish_tello_recap(summary if isinstance(summary, dict) else {"raw": summary})

    def _apply_follow_mode(self, action: str) -> str:
        mode_map = {
            "follow_me": "FOLLOW_ME",
            "follow_face": "FOLLOW_MY_FACE",
            "follow_body": "FOLLOW_MY_BODY",
            "follow_stop": "STOP",
        }
        mode = mode_map.get(action)
        if not mode:
            return "Unknown follow mode."
        try:
            r = self.tello.follow(mode=mode, timeout=3.0)
        except Exception as e:
            return f"I couldn't set follow mode due to an error: {e}"
        if isinstance(r, dict) and not r.get("ok", True):
            return f"Follow mode failed: {r.get('error', 'unknown error')}"
        return f"Follow mode set to **{mode}**."

    def _ensure_tello_agent(self) -> tuple[bool, str | None]:
        if self.tello.is_running():
            return True, None
        try:
            self.tello.start()
            return True, None
        except Exception as e:
            msg = str(e)
            if "WinError 10048" in msg:
                return False, (
                    "I can't start the Tello agent because the network port is already in use. "
                    "It looks like another Tello session is still running. "
                    "Please close the other session and try again."
                )
            return False, f"I couldn't start the Tello agent. Error: {e}"

    def _is_not_ready_report(self, report: dict) -> bool:
        if not isinstance(report, dict):
            return True
        reason = (report.get("reason") or "").lower()
        return (
            report.get("ok") is False
            or report.get("brightness") is None
            or report.get("safe") is None
            or "not_ready" in reason
            or "no_valid_frames" in reason
        )

    def _safe_lighting_check(self, timeout=6.0, retries=3, delay=0.8):
        last_report = None
        for attempt in range(1, retries + 1):
            try:
                report = self.tello.check_lighting(timeout=timeout)
                last_report = report
                if report is None:
                    if attempt < retries:
                        time.sleep(delay)
                        continue
                    return None, (
                        "I couldn't read the lighting yet (no data returned). "
                        "If you just started the Tello agent, it may still be warming up."
                    )
                if self._is_not_ready_report(report):
                    if attempt < retries:
                        time.sleep(delay)
                        continue
                    return report, (
                        "I tried to read the lighting 3 times, but I'm still not getting valid frames. "
                        "This could mean the stream is not started, the connection is unstable, "
                        "or the drone/decoder is stuck."
                    )
                return report, None
            except TimeoutError:
                return None, (
                    "I couldn't reach the Tello agent in time. "
                    "Please make sure it's running and you're connected to the Tello Wi-Fi."
                )
            except Exception as e:
                return None, f"Tello check failed: {e}"
        return last_report, "Lighting check failed unexpectedly."

    def polish_tello_recap(self, summary: dict) -> str:
        if not isinstance(summary, dict):
            return f"Tello flight recap:\n{summary}"

        samples = summary.get("samples")
        safe_ratio = summary.get("safe_ratio")
        b_mean = summary.get("brightness_mean")
        b_min = summary.get("brightness_min")
        c_min = summary.get("contrast_min")
        pb_max = summary.get("pct_black_max")
        cmds = summary.get("last_commands") or []
        takeoff = next((c for c in cmds if c.get("cmd") == "takeoff"), None)
        land = next((c for c in cmds if c.get("cmd") == "land"), None)
        status = next((c for c in reversed(cmds) if c.get("cmd") == "status"), None)

        lines = ["Tello flight recap"]
        if samples is not None:
            lines.append(f"- Telemetry samples: {samples}")
        if safe_ratio is not None:
            lines.append(f"- Lighting safe ratio: {safe_ratio:.2f}" if isinstance(safe_ratio, (int, float)) else f"- Lighting safe ratio: {safe_ratio}")
        if b_mean is not None:
            lines.append(f"- Brightness (mean): {b_mean:.1f}")
        if b_min is not None:
            lines.append(f"- Brightness (min): {b_min:.1f}")
        if c_min is not None:
            lines.append(f"- Contrast (min): {c_min:.1f}")
        if pb_max is not None:
            lines.append(f"- Max dark area (% black): {pb_max:.1f}")
        if takeoff:
            lines.append(f"- Takeoff at t_ms={takeoff.get('t_ms')}")
        if status:
            lines.append(f"- Last status at t_ms={status.get('t_ms')} (state={status.get('state')})")
        if land:
            lines.append(f"- Land at t_ms={land.get('t_ms')}")

        trace = ", ".join([c.get("cmd", "?") for c in cmds][-8:])
        if trace:
            lines.append(f"- Command trace: {trace}")

        return "\n".join(lines)

    def _infer_device(self, text: str):
        t = (text or "").lower()
        if "tello" in t or "drone" in t:
            return "tello"
        if "rover" in t or "lidar" in t or "robot" in t:
            return "rover"
        return None

    def _infer_action(self, text: str):
        t = (text or "").lower()
        if any(k in t for k in ("brightness", "lighting", "safe to fly", "is it safe", "monitor")):
            return "monitor"
        if any(k in t for k in ("open camera", "camera", "stream", "video")):
            return "stream"
        if any(k in t for k in ("takeoff", "take off", "hover", "fly")):
            return "takeoff_hover"
        if "land" in t:
            return "land"
        if "status" in t or "battery" in t:
            return "status"
        if "disconnect" in t:
            return "disconnect"
        if re.search(r"\bstop\b", t):
            return "stop"
        if "follow" in t:
            if "face" in t:
                return "follow_face"
            if "body" in t:
                return "follow_body"
            if any(k in t for k in ("stop", "cancel", "disable", "off")):
                return "follow_stop"
            return "follow_me"
        if "recovery" in t:
            return "recovery"
        return None

    def _to_float(self, v, default=None):
        try:
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    def _format_lighting_report(self, r: dict) -> str:
        reason = (r.get("reason") or "").lower()
        not_ready = (
            r.get("ok") is False
            or r.get("safe") is None
            or r.get("brightness") is None
            or "not_ready" in reason
            or "no_valid_frames" in reason
        )

        if not_ready:
            return (
                "I tried to check the lighting, but I can't get a valid camera frame yet. "
                f"Reason: {r.get('reason') or 'not ready'}. "
                "If you just started the Tello agent, give it a moment or try opening the camera stream first."
            )

        b = self._to_float(r.get("brightness"), default=0.0)
        c = self._to_float(r.get("contrast"), default=0.0)
        pb = self._to_float(r.get("pct_black"), default=0.0)
        if 0.0 <= pb <= 1.0:
            pb *= 100.0

        return (
            "I've checked the lighting. "
            f"Overall, it looks {'safe' if r.get('safe') else 'unsafe'}. "
            f"Reason: {r.get('reason') or 'no reason provided'}. "
            f"Brightness is about {b:.1f}. "
            f"Contrast is around {c:.1f}. "
            f"Dark areas are roughly {pb:.1f} percent."
        )

    def _format_status(self, s: dict) -> str:
        s = s or {}
        state = s.get("state", "unknown")
        battery = s.get("battery", "unknown")
        tele = s.get("telemetry", {})
        hover_s = tele.get("hover_s", 0)
        return (
            f"Tello is {state}. "
            f"Battery at {battery}%. "
            f"Hover time: {hover_s} seconds."
        )

    def _clear_pending(self) -> None:
        self.mm.set_pending_topic(None)
        self.mm.pending_payload = None
