import subprocess, threading, json, os, sys
import time, itertools


class TelloClient:
    def __init__(self, agent_py_path: str, python_exe: str | None = None):
        self.agent_py_path = os.path.abspath(agent_py_path)
        self.python_exe = python_exe or sys.executable
        self.p = None
        self.latest = {}
        self.events = []
        self._reader_thread = None

        self._id = itertools.count(1)

        self._lock = threading.Lock()
        self._responses = {}     
        self._raw_lines = []    
    
    def is_running(self):
        return self.p is not None and self.p.poll() is None


    def start(self, ready_timeout=10.0):
        if self.is_running():
            return True


        # reset state (don't replace lock!)
        with self._lock:
            self._responses.clear()
            self._raw_lines.clear()
            self.events.clear()
            self.latest = {}

        self._ready = threading.Event()
        self._startup_error = None

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.p = subprocess.Popen(
            [self.python_exe, self.agent_py_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags
        )

        print("[TelloClient] started pid=", self.p.pid)

        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

        if not self._ready.wait(timeout=ready_timeout):
            raise TimeoutError("Tello agent did not become ready_for_commands in time.")

        if self._startup_error:
            raise RuntimeError(self._startup_error)

        return True

    def _reader(self):
        try:
            if not self.p or not self.p.stdout:
                # Nothing to read; unblock start() if waiting
                if hasattr(self, "_ready") and not self._ready.is_set():
                    self._startup_error = self._startup_error or "Agent stdout is not available."
                    self._ready.set()
                return

            for line in self.p.stdout:
                line_raw = line.rstrip("\n")

                # store tail for debugging (thread-safe)
                with self._lock:
                    self._raw_lines.append(line_raw)
                    if len(self._raw_lines) > 500:
                        self._raw_lines = self._raw_lines[-500:]

                # print("[AGENT RAW]", line_raw)

                s = line_raw.strip()
                if not s:
                    continue

                # try parse JSON messages
                try:
                    msg = json.loads(s)
                    # debug: surface key events
                    if msg.get("type") == "event" and msg.get("name") in (
                        "safety_land_on_exit", "auto_land_timeout", "landed", "takeoff", "ended", "debug_land_source"
                    ):
                        print("[AGENT EVENT]", msg)

                except Exception:
                    # ignore non-JSON log lines
                    continue

                # fail-fast on agent-reported error
                if msg.get("type") == "error":
                    rid = msg.get("req_id", msg.get("id"))
                    if rid is not None:
                        with self._lock:
                            self._responses[rid] = {"type":"resp", "req_id": rid, "ok": False, "error": msg.get("message")}
                    else:
                        # no rid = can't map it; store as latest
                        pass
                    with self._lock:
                        self.events.append(msg)
                    continue


                # ready gate
                if msg.get("type") == "event" and msg.get("name") == "ready_for_commands":
                    if hasattr(self, "_ready"):
                        self._ready.set()

                # response store
                if msg.get("type") == "resp":
                    rid = msg.get("req_id", msg.get("id"))
                    if rid is not None:
                        with self._lock:
                            self._responses[rid] = msg

                # latest/event history (not strictly necessary to lock, but safe)
                with self._lock:
                    self.latest = msg
                    self.events.append(msg)

        except Exception as e:
            # If reader dies unexpectedly, unblock start() with error
            self._startup_error = self._startup_error or f"Reader thread crashed: {e}"
            try:
                if hasattr(self, "_ready") and not self._ready.is_set():
                    self._ready.set()
            except Exception:
                pass

        finally:
            # If the agent exits before we ever got ready_for_commands, unblock start()
            try:
                if hasattr(self, "_ready") and not self._ready.is_set():
                    exit_code = None
                    try:
                        exit_code = None if not self.p else self.p.poll()
                    except Exception:
                        pass

                    self._startup_error = self._startup_error or f"Tello agent exited early. exit_code={exit_code}"
                    self._ready.set()
            except Exception:
                pass


    def send(self, cmd: str):
        if not self.p or self.p.poll() is not None:
            raise RuntimeError("Tello agent is not running.")
        self.p.stdin.write(cmd.strip() + "\n")
        self.p.stdin.flush()

    def request(self, cmd: str, payload=None, timeout=3.0):
        if not self.is_running():
            raise RuntimeError("Tello agent is not running. Call ensure_started() first.")
        # ensure agent is running
        # if not self.is_running():
        #     self.start()
        #     time.sleep(0.2)  # small warmup so agent prints "ready"

        req_id = next(self._id)
        payload_str = "" if payload is None else json.dumps(payload)

        # debug
        if not self.p or self.p.poll() is not None:
            raise RuntimeError(f"Agent not running. exit_code={None if not self.p else self.p.poll()}")

        # always send 3 parts so agent parser stays simple
        self.send(f"{cmd}|{req_id}|{payload_str}")

        print(f"[CLIENT] SEND {cmd} payload={payload}")

        return self._wait_resp(req_id, timeout)

    def _wait_resp(self, req_id: int, timeout: float):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                msg = self._responses.pop(req_id, None)
            if msg is not None:
                print(f"[CLIENT] RECV resp={msg}")
                return msg
            time.sleep(0.02)

        # timeout → show what the agent printed
        exit_code = None if not self.p else self.p.poll()
        with self._lock:
            tail = "\n".join(self._raw_lines[-30:])


        raise TimeoutError(
            f"No response for req_id={req_id} within {timeout}s. exit_code={exit_code}\n"
            f"--- agent stdout tail ---\n{tail}\n"
            f"--- end ---"
        )


    def stop(self):
        if not self.p or self.p.poll() is not None:
            return

        try:
            self.send("quit|0|")
        except Exception:
            pass

        try:
            self.p.wait(timeout=2)
        except Exception:
            pass

        if self.p.poll() is None:
            try:
                self.p.terminate()
                self.p.wait(timeout=2)
            except Exception:
                pass

        if self.p.poll() is None:
            try:
                self.p.kill()
            except Exception:
                pass

        # close pipes (best effort)
        try:
            if self.p.stdin: self.p.stdin.close()
        except Exception:
            pass
        try:
            if self.p.stdout: self.p.stdout.close()
        except Exception:
            pass

        self.p = None



    def ensure_started(self):
        # starts the agent if not running
        if not self.p or self.p.poll() is not None:
            self.start()

    def check_lighting(self, timeout=3.0) -> dict:
        """
        Returns: dict with keys safe, reason, brightness, contrast, pct_black, battery, state
        """
        self.ensure_started()
        resp = self.request("lighting", timeout=timeout)
        if not resp.get("ok", False):
            return {
                "safe": None,
                "reason": resp.get("error", "lighting_failed"),
                "brightness": None,
                "contrast": None,
                "pct_black": None,
                "battery": None,
                "state": None,
            }

        data = resp.get("data", {}) or {}
        # normalize fields used by DeviceHandlerFSM
        return {
            "safe": bool(data.get("safe", False)),
            "reason": data.get("reason", "unknown"),
            "brightness": data.get("brightness"),
            "contrast": data.get("contrast"),
            "pct_black": data.get("pct_black"),
            "battery": data.get("battery"),
            "state": data.get("state"),
        }

    def get_status(self, timeout=3.0) -> dict:
        self.ensure_started()
        resp = self.request("status", timeout=timeout)
        if resp.get("ok"):
            data = resp.get("data", {}) or {}
            if isinstance(data, dict) and "telemetry" in data and isinstance(data["telemetry"], dict):
                return data["telemetry"]   # <-- flatten
            return data
        return {"error": resp.get("error", "status_failed")}


    def preflight(self, timeout=10.0) -> dict:
        self.ensure_started()
        return self.request("preflight", timeout=timeout)
    

    def start_takeoff_hover_session(self, timeout=10.0) -> dict:
        """
        Preflight -> takeoff. Do NOT auto-land.
        """
        self.ensure_started()

        pf = self.preflight(timeout=10.0)
        if isinstance(pf, dict) and not pf.get("ok", False):
            return {"ok": False, "error": pf.get("error", "preflight_failed")}

        r1 = self.takeoff(timeout=timeout)
        if not r1.get("ok", False):
            return {"ok": False, "error": r1.get("error", "takeoff_failed")}

        return {"ok": True, "result": "airborne"}


    def takeoff(self, timeout=10.0) -> dict:
        self.ensure_started()
        return self.request("takeoff", timeout=timeout)
    
    def follow(self, mode="FOLLOW_ME", timeout=3.0) -> dict:
        self.ensure_started()
        return self.request("follow", payload={"mode": mode}, timeout=timeout)

    def land(self, timeout=10.0) -> dict:
        self.ensure_started()
        return self.request("land", timeout=timeout)

    def recap(self, timeout=3.0) -> dict:
        self.ensure_started()
        return self.request("recap", timeout=timeout)
