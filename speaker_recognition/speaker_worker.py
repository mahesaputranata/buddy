# speaker_worker.py
import json
import sys
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("[WK] process started", file=sys.stderr, flush=True)

from verify_wav import SpeakerVerifier

print("[WK] imported SpeakerVerifier", file=sys.stderr, flush=True)


def main():
    try:
        print("[WK] before constructing verifier", file=sys.stderr, flush=True)
        verifier = SpeakerVerifier()
        print("[WK] verifier constructed", file=sys.stderr, flush=True)

        print(json.dumps({"status": "ready"}), flush=True)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line == "__quit__":
                print(json.dumps({"status": "bye"}), flush=True)
                break

            try:
                req = json.loads(line)
                wav_path = req.get("wav_path")

                if not wav_path:
                    print(json.dumps({
                        "ok": False,
                        "error": "Missing wav_path"
                    }), flush=True)
                    continue

                result = verifier.verify(wav_path)
                print(json.dumps(result), flush=True)

            except Exception as e:
                print(json.dumps({
                    "ok": False,
                    "error": repr(e),
                    "trace": traceback.format_exc()[:2000],
                }), flush=True)

    except Exception as e:
        print("[WK] startup exception", file=sys.stderr, flush=True)
        print(json.dumps({
            "status": "startup_error",
            "error": repr(e),
            "trace": traceback.format_exc()[:3000],
        }), flush=True)


if __name__ == "__main__":
    main()