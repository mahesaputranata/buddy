"""
mouth_v1.py
───────────
TTSWorker with:
  - Forced audio output to a specific device (JBL Tune Beam 2)
  - Spotify volume ducking during speech
  - VAD mute/unmute during speech

Dependencies:
    pip install sounddevice soundfile numpy

How it works:
  pyttsx3 renders speech → temp WAV file on disk
  sounddevice plays that WAV through the named output device
  This bypasses Windows default audio routing entirely.
"""

import os
import re
import queue
import tempfile
import threading
import time
import traceback
from typing import Optional

import sounddevice as sd
import soundfile as sf
import pyttsx3

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

TTS_DRIVER   = "sapi5"
TTS_RATE     = 140
TTS_VOLUME   = 1.0
PREFERRED_VOICE_KEYWORDS = ("david", "male")
MAX_CHUNK_CHARS = 1000
DEBUG_TTS    = True

# Partial name match against sounddevice device list (case-insensitive).
# Run list_audio_devices() once to confirm the exact name.
# Set to None to use system default output.
# TTS_OUTPUT_DEVICE_HINT = "JBL Tune Beam"   # matches "JBL Tune Beam 2"
TTS_OUTPUT_DEVICE_HINT = "Headphones (2- JBL Tune Beam 2)"


# ──────────────────────────────────────────────────────────────────────────────
# DEVICE UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def list_audio_devices() -> None:
    """
    Prints all available audio devices with their indices.
    Call this once from a standalone script to confirm device names:

        from mouth.debug.mouth_v1 import list_audio_devices
        list_audio_devices()
    """
    print("\n[AUDIO DEVICES]")
    for i, d in enumerate(sd.query_devices()):
        tags = []
        if d["max_output_channels"] > 0:
            tags.append("OUTPUT")
        if d["max_input_channels"] > 0:
            tags.append("INPUT")
        print(f"  [{i:2d}] {d['name']}  ({', '.join(tags)})")
    print()


def resolve_output_device(hint: Optional[str]) -> Optional[int]:
    """
    Finds the sounddevice index for an output device whose name contains hint.
    Returns None if not found — caller then uses system default.
    """
    if not hint:
        return None

    hint_lower = hint.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and hint_lower in d["name"].lower():
            if DEBUG_TTS:
                print(f"[TTS] Output device matched: [{i}] {d['name']!r}")
            return i

    print(f"[TTS] WARNING: Output device {hint!r} not found. Using system default.")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKER
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, buf = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(s), max_chars):
                chunks.append(s[i:i + max_chars].strip())
            continue
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# TTS WORKER
# ──────────────────────────────────────────────────────────────────────────────

class TTSWorker:
    def __init__(self, vad=None, spotify=None):
        self.vad = vad
        self.spotify = spotify
        self.q = queue.Queue()
        self._output_device: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._is_speaking = False

    def start(self):
        self._thread.start()

    def speak(self, text: str):
        text = (text or "").strip()
        if text:
            self.q.put(text)

    def stop(self):
        pass  # no persistent engine to stop

    def _duck(self):
        if self.spotify is not None:
            threading.Thread(target=self.__safe_duck, daemon=True).start()

    def __safe_duck(self):
        try:
            self.spotify.duck()
        except Exception as e:
            if DEBUG_TTS:
                print(f"[TTS] duck failed (non-fatal): {e}")

    def _restore(self):
        if self.spotify is not None:
            threading.Thread(target=self.__safe_restore, daemon=True).start()

    def __safe_restore(self):
        try:
            self.spotify.restore()
        except Exception as e:
            if DEBUG_TTS:
                print(f"[TTS] restore failed (non-fatal): {e}")

    def _render_to_wav(self, text: str) -> Optional[str]:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="buddy_tts_"
        )
        tmp_path = tmp.name
        tmp.close()

        try:
            # Fresh engine every time — avoids sapi5 hang on reuse
            eng = pyttsx3.init(driverName=TTS_DRIVER)
            eng.setProperty("rate", TTS_RATE)
            eng.setProperty("volume", TTS_VOLUME)

            voices = eng.getProperty("voices") or []
            chosen = None
            for v in voices:
                name = (getattr(v, "name", "") or "").lower()
                if any(k in name for k in PREFERRED_VOICE_KEYWORDS):
                    chosen = v.id
                    break
            if chosen is None and voices:
                chosen = voices[0].id
            if chosen:
                eng.setProperty("voice", chosen)

            eng.save_to_file(text, tmp_path)
            eng.runAndWait()

            try:
                eng.stop()
            except Exception:
                pass
            del eng

            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
                return tmp_path

            if DEBUG_TTS:
                print(f"[TTS] WAV file empty or missing: {tmp_path}")
            return None

        except Exception as e:
            print(f"[TTS] _render_to_wav failed: {e}")
            traceback.print_exc()
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None

    def _play_wav(self, wav_path: str) -> None:
        try:
            self._is_speaking = True
            data, samplerate = sf.read(wav_path, dtype="float32")
            sd.play(data, samplerate=samplerate, device=self._output_device)
            print(f"[TTS] sd.wait() starting...")
            sd.wait()
            print(f"[TTS] sd.wait() done")
        except Exception as e:
            print(f"[TTS] _play_wav failed: {e}")
            traceback.print_exc()
        finally:
            self._is_speaking = False
            try:
                os.remove(wav_path)
            except Exception:
                pass

    def is_speaking(self) -> bool:
        return self._is_speaking

    def _run(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception as e:
            if DEBUG_TTS:
                print("[TTS] COM init failed (pythoncom):", repr(e))

        if DEBUG_TTS:
            print("[TTS] worker thread starting...")

        # Resolve output device once at startup
        self._output_device = resolve_output_device(TTS_OUTPUT_DEVICE_HINT)
        if DEBUG_TTS:
            print(f"[TTS] output device index: {self._output_device} "
                  f"({'system default' if self._output_device is None else 'JBL Tune Beam 2'})")

        while True:
            text = self.q.get()
            print(f"[TTS] dequeued: {text[:60]!r}")
            try:
                chunks = _chunk_text(text, MAX_CHUNK_CHARS)
                if not chunks:
                    continue

                if self.vad is not None:
                    self.vad.mute()
                self._duck()

                try:
                    for chunk in chunks:
                        wav_path = self._render_to_wav(chunk)
                        if wav_path:
                            if DEBUG_TTS:
                                print(f"[TTS] playing chunk ({len(chunk)} chars) "
                                      f"→ device={self._output_device}")
                            self._play_wav(wav_path)
                        else:
                            print(f"[TTS] render failed, skipping chunk: {chunk[:60]!r}")

                finally:
                    try:
                        self._restore()
                    except Exception as e:
                        print(f"[TTS] restore error in finally: {e}")
                    try:
                        if self.vad is not None:
                            self.vad.unmute()
                    except Exception as e:
                        print(f"[TTS] unmute error in finally: {e}")

            except Exception as e:
                print("[TTS ERROR]", repr(e))
                traceback.print_exc()
                self._restore()
                if self.vad is not None:
                    self.vad.unmute()
            finally:
                self.q.task_done()

# import time, re, traceback, threading, queue
# import pyttsx3

# TTS_DRIVER = "sapi5"
# TTS_RATE = 140
# TTS_VOLUME = 1.0
# PREFERRED_VOICE_KEYWORDS = ("david", "male")
# MAX_CHUNK_CHARS = 1000
# DEBUG_TTS = True


# def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS):
#     text = re.sub(r"\s+", " ", (text or "").strip())
#     if not text:
#         return []
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks, buf = [], ""
#     for s in sentences:
#         s = s.strip()
#         if not s:
#             continue
#         if len(s) > max_chars:
#             if buf:
#                 chunks.append(buf)
#                 buf = ""
#             for i in range(0, len(s), max_chars):
#                 chunks.append(s[i:i + max_chars].strip())
#             continue
#         if not buf:
#             buf = s
#         elif len(buf) + 1 + len(s) <= max_chars:
#             buf = f"{buf} {s}"
#         else:
#             chunks.append(buf)
#             buf = s
#     if buf:
#         chunks.append(buf)
#     return chunks


# class TTSWorker:
#     def __init__(self, vad=None, spotify=None):
#         self.vad = vad
#         self.spotify = spotify      # optional SpotifyManager — None = no ducking
#         self.q = queue.Queue()
#         self._engine = None
#         self._lock = threading.Lock()
#         self._thread = threading.Thread(target=self._run, daemon=True)

#     def start(self):
#         self._thread.start()

#     def speak(self, text: str):
#         text = (text or "").strip()
#         if text:
#             self.q.put(text)

#     def stop(self):
#         with self._lock:
#             try:
#                 if self._engine is not None:
#                     self._engine.stop()
#             except Exception:
#                 pass

#     # ── duck / restore helpers ────────────────────────────────────────────────
#     # Wrapped in try/except so a Spotify hiccup never breaks TTS

#     def _duck(self):
#         if self.spotify is not None:
#             try:
#                 self.spotify.duck()
#             except Exception as e:
#                 if DEBUG_TTS:
#                     print(f"[TTS] spotify.duck() failed (non-fatal): {e}")

#     def _restore(self):
#         if self.spotify is not None:
#             try:
#                 self.spotify.restore()
#             except Exception as e:
#                 if DEBUG_TTS:
#                     print(f"[TTS] spotify.restore() failed (non-fatal): {e}")

#     # ── engine init ───────────────────────────────────────────────────────────

#     def _init_engine(self):
#         if DEBUG_TTS:
#             print("[TTS] init engine...")

#         eng = pyttsx3.init(driverName=TTS_DRIVER)
#         eng.setProperty("rate", TTS_RATE)
#         eng.setProperty("volume", TTS_VOLUME)

#         voices = eng.getProperty("voices") or []
#         if DEBUG_TTS:
#             print(f"[TTS] voices: {len(voices)}")

#         chosen = None
#         for v in voices:
#             name = (getattr(v, "name", "") or "").lower()
#             if any(k in name for k in PREFERRED_VOICE_KEYWORDS):
#                 chosen = v.id
#                 break
#         if chosen is None and voices:
#             chosen = voices[0].id

#         if chosen:
#             eng.setProperty("voice", chosen)
#             if DEBUG_TTS:
#                 print(f"[TTS] voice set: {chosen}")

#         if DEBUG_TTS:
#             print("[TTS] engine ready")

#         return eng

#     def _rebuild_engine(self):
#         with self._lock:
#             try:
#                 if self._engine is not None:
#                     try:
#                         self._engine.stop()
#                     except Exception:
#                         pass
#             finally:
#                 self._engine = None
#             self._engine = self._init_engine()

#     # ── worker loop ───────────────────────────────────────────────────────────

#     def _run(self):
#         try:
#             import pythoncom
#             pythoncom.CoInitialize()
#         except Exception as e:
#             if DEBUG_TTS:
#                 print("[TTS] COM init failed (pythoncom). Install pywin32?", repr(e))

#         if DEBUG_TTS:
#             print("[TTS] worker thread starting...")
#         self._rebuild_engine()

#         while True:
#             text = self.q.get()
#             try:
#                 chunks = _chunk_text(text, MAX_CHUNK_CHARS)

#                 # ── mute mic + duck Spotify before speaking ───────────────
#                 if self.vad is not None:
#                     self.vad.mute()
#                 self._duck()

#                 try:
#                     # 1) First attempt
#                     with self._lock:
#                         for ch in chunks:
#                             self._engine.say(ch)

#                         t0 = time.time()
#                         self._engine.runAndWait()
#                         elapsed = time.time() - t0

#                     if DEBUG_TTS:
#                         print(f"[TTS] runAndWait elapsed={elapsed:.3f}s")

#                     # 2) If returned too fast → silent failure → rebuild + retry
#                     min_elapsed = 0.20 if len(text) >= 20 else 0.05
#                     if elapsed < min_elapsed:
#                         if DEBUG_TTS:
#                             print("[TTS] too fast -> rebuild+retry once")

#                         self._rebuild_engine()

#                         with self._lock:
#                             for ch in chunks:
#                                 self._engine.say(ch)

#                             t0 = time.time()
#                             self._engine.runAndWait()
#                             elapsed2 = time.time() - t0

#                         if DEBUG_TTS:
#                             print(f"[TTS] retry elapsed={elapsed2:.3f}s")

#                 finally:
#                     # ── always restore Spotify + unmute mic after speaking ─
#                     self._restore()
#                     if self.vad is not None:
#                         self.vad.unmute()

#             except Exception as e:
#                 print("[TTS ERROR]", repr(e))
#                 traceback.print_exc()
#                 # restore in case of crash mid-speech
#                 self._restore()
#                 if self.vad is not None:
#                     self.vad.unmute()
#                 self._rebuild_engine()
#             finally:
#                 self.q.task_done()


# import time, re, traceback, threading, queue
# import pyttsx3

# TTS_DRIVER = "sapi5"
# TTS_RATE = 140
# TTS_VOLUME = 1.0
# PREFERRED_VOICE_KEYWORDS = ("david", "male")
# MAX_CHUNK_CHARS = 1000
# DEBUG_TTS = True


# def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS):
#     text = re.sub(r"\s+", " ", (text or "").strip())
#     if not text:
#         return []
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     chunks, buf = [], ""
#     for s in sentences:
#         s = s.strip()
#         if not s:
#             continue
#         if len(s) > max_chars:
#             if buf:
#                 chunks.append(buf)
#                 buf = ""
#             for i in range(0, len(s), max_chars):
#                 chunks.append(s[i:i + max_chars].strip())
#             continue
#         if not buf:
#             buf = s
#         elif len(buf) + 1 + len(s) <= max_chars:
#             buf = f"{buf} {s}"
#         else:
#             chunks.append(buf)
#             buf = s
#     if buf:
#         chunks.append(buf)
#     return chunks


# class TTSWorker:
#     def __init__(self, vad = None):
#         self.vad = vad
#         self.q = queue.Queue()
#         self._engine = None
#         self._lock = threading.Lock()
#         self._thread = threading.Thread(target=self._run, daemon=True)

#     def start(self):
#         self._thread.start()

#     def speak(self, text: str):
#         text = (text or "").strip()
#         if text:
#             self.q.put(text)

#     def stop(self):
#         # optional: stop current speech
#         with self._lock:
#             try:
#                 if self._engine is not None:
#                     self._engine.stop()
#             except Exception:
#                 pass

#     def _init_engine(self):
#         if DEBUG_TTS:
#             print("[TTS] init engine...")

#         eng = pyttsx3.init(driverName=TTS_DRIVER)
#         eng.setProperty("rate", TTS_RATE)
#         eng.setProperty("volume", TTS_VOLUME)

#         voices = eng.getProperty("voices") or []
#         if DEBUG_TTS:
#             print(f"[TTS] voices: {len(voices)}")

#         chosen = None
#         for v in voices:
#             name = (getattr(v, "name", "") or "").lower()
#             if any(k in name for k in PREFERRED_VOICE_KEYWORDS):
#                 chosen = v.id
#                 break
#         if chosen is None and voices:
#             chosen = voices[0].id

#         if chosen:
#             eng.setProperty("voice", chosen)
#             if DEBUG_TTS:
#                 print(f"[TTS] voice set: {chosen}")

#         if DEBUG_TTS:
#             print("[TTS] engine ready")

#         return eng

#     def _rebuild_engine(self):
#         with self._lock:
#             try:
#                 if self._engine is not None:
#                     try:
#                         self._engine.stop()
#                     except Exception:
#                         pass
#             finally:
#                 self._engine = None
#             self._engine = self._init_engine()

#     def _run(self):
#         try:
#             import pythoncom
#             pythoncom.CoInitialize()
#         except Exception as e:
#             if DEBUG_TTS:
#                 print("[TTS] COM init failed (pythoncom). Install pywin32?", repr(e))

#         if DEBUG_TTS:
#             print("[TTS] worker thread starting...")
#         self._rebuild_engine()

#         while True:
#             text = self.q.get()
#             try:
#                 chunks = _chunk_text(text, MAX_CHUNK_CHARS)

#                 if self.vad is not None:
#                     self.vad.mute()

#                 try:
#                     # 1) First attempt
#                     with self._lock:
#                         for ch in chunks:
#                             self._engine.say(ch)

#                         t0 = time.time()
#                         self._engine.runAndWait()
#                         elapsed = time.time() - t0

#                     if DEBUG_TTS:
#                         print(f"[TTS] runAndWait elapsed={elapsed:.3f}s")

#                     # 2) If it returned "too fast", treat as likely silent failure
#                     min_elapsed = 0.20 if len(text) >= 20 else 0.05
#                     if elapsed < min_elapsed:
#                         if DEBUG_TTS:
#                             print("[TTS] too fast -> rebuild+retry once")

#                         self._rebuild_engine()

#                         with self._lock:
#                             for ch in chunks:
#                                 self._engine.say(ch)

#                             t0 = time.time()
#                             self._engine.runAndWait()
#                             elapsed2 = time.time() - t0

#                         if DEBUG_TTS:
#                             print(f"[TTS] retry elapsed={elapsed2:.3f}s")

#                 finally:
#                     if self.vad is not None:
#                         self.vad.unmute()

#             except Exception as e:
#                 print("[TTS ERROR]", repr(e))
#                 traceback.print_exc()
#                 self._rebuild_engine()
#             finally:
#                 self.q.task_done()


