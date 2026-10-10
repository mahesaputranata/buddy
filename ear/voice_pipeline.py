import subprocess
import json
import sys
import threading
import time
from pathlib import Path

from brain.settings import setting

SPEAKER_PY = setting("speaker_python", sys.executable)
WORKER_SCRIPT = setting("speaker_worker_script",
                        str(Path(__file__).resolve().parent.parent / "speaker_recognition" / "speaker_worker.py"))

class SpeakerWorkerClient:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()

    def start(self):
        if self.proc is not None:
            return

        self.proc = subprocess.Popen(
            [SPEAKER_PY, WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        time.sleep(2.0)  # give worker time to initialize

        retcode = self.proc.poll()
        print(f"[DEBUG] worker retcode after 2s: {retcode}")

        if retcode is not None:
            out, err = self.proc.communicate()
            raise RuntimeError(
                f"Speaker worker exited early.\n"
                f"RETURNCODE: {retcode}\n"
                f"STDOUT:\n{out}\n"
                f"STDERR:\n{err}"
            )

        # If still alive, try to read first stdout line
        ready_line = self.proc.stdout.readline().strip()
        print(f"[DEBUG] worker first stdout line: {repr(ready_line)}")

        if not ready_line:
            raise RuntimeError("Speaker worker alive but produced no ready line.")

        ready_msg = json.loads(ready_line)

        if ready_msg.get("status") != "ready":
            raise RuntimeError(f"Unexpected speaker worker startup response: {ready_msg}")

        print("[SPK] Worker connected.")

    def verify_wav(self, wav_path: str) -> dict:
        if self.proc is None:
            raise RuntimeError("Speaker worker not started")

        with self.lock:
            req = {"wav_path": wav_path}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()

            line = self.proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("Speaker worker returned empty response")

            return json.loads(line)

    def stop(self):
        if self.proc is None:
            return

        try:
            with self.lock:
                self.proc.stdin.write("__quit__\n")
                self.proc.stdin.flush()
        except Exception:
            pass

        try:
            self.proc.terminate()
        except Exception:
            pass

        self.proc = None