import queue
import threading
from pathlib import Path
import datetime

import sounddevice as sd
import soundfile as sf
from pynput import keyboard
import numpy as np

VOICE_QUEUE = queue.Queue()
TEXT_QUEUE = queue.Queue()

class TextHotkey:
    def __init__(self):
        self._ctrl = False
        self._alt = False
        self._t = False
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )

    def start(self):
        self.listener.start()
        print("[TEXT] Hotkey ready (Ctrl+Alt+T to type).")

    def _on_press(self, key):
        try:
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._ctrl = True
            if key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                self._alt = True
            if hasattr(key, "char") and key.char and key.char.lower() == "t":
                self._t = True

            if self._ctrl and self._alt and self._t:
                TEXT_QUEUE.put("__OPEN_TEXT__")
        except Exception:
            pass

    def _on_release(self, key):
        try:
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._ctrl = False
            if key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
                self._alt = False
            if hasattr(key, "char") and key.char and key.char.lower() == "t":
                self._t = False
        except Exception:
            pass

class HoldToTalk:
    def __init__(self, sr=16000, channels=1, out_dir="mic_logs/audio"):
        self.sr = sr
        self.channels = channels
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._frames = []
        self._lock = threading.Lock()
        self._stream = None

        # hotkey state: Ctrl + Space (old approach)
        self._ctrl = False
        self._space = False

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def start(self):
        self.listener.start()
        print("[PTT] Hold-to-talk ready (hold Ctrl+Space).")


    def _start_recording(self):
        with self._lock:
            if self._recording:
                return
            self._recording = True
            self._frames = []

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        print("[PTT] REC START")

    def _stop_recording(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            pass

        with self._lock:
            frames = self._frames
            self._frames = []

        if not frames:
            print("[PTT] No audio captured.")
            return

        audio_np = np.concatenate(frames, axis=0)

        # optional safety: skip super short clips (accidental taps)
        if audio_np.shape[0] < int(0.25 * self.sr):
            print("[PTT] Too short, skipping.")
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wav_path = self.out_dir / f"ptt_{ts}.wav"
        sf.write(wav_path, audio_np, self.sr)
        print(f"[PTT] REC STOP -> {wav_path}")

        VOICE_QUEUE.put(str(wav_path))

    def _on_press(self, key):
        try:
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._ctrl = True
            if key == keyboard.Key.space:
                self._space = True

            if self._ctrl and self._space:
                self._start_recording()
        except Exception:
            pass

    def _on_release(self, key):
        try:
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._ctrl = False
            if key == keyboard.Key.space:
                self._space = False

            # Stop only when the combo is no longer held
            if not (self._ctrl and self._space):
                self._stop_recording()
        except Exception:
            pass


class VADRecorder:
    def __init__(
        self,
        voice_queue,
        sr=16000,
        channels=1,
        out_dir="mic_logs/audio",
        frame_ms=20,
        start_frames=3,
        end_frames=6,
        margin_db=6.0,
        noise_clamp_db=-65.0,
        ignore_below_db=-90.0,
        pre_roll_ms=200,
        min_utterance_ms=800,
        max_utterance_ms=12000,
        end_of_turn_silence_sec=1.5,
        cooldown_ms=200,
        debug=True,
    ):
        self.voice_queue = voice_queue
        self.sr = sr
        self.channels = channels
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.frame_ms = frame_ms
        self.frame_samples = int(sr * frame_ms / 1000)

        self.start_frames = start_frames
        self.end_frames = end_frames
        self.margin_db = margin_db
        self.noise_clamp_db = noise_clamp_db
        self.ignore_below_db = ignore_below_db
        self.pre_roll_ms = pre_roll_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.end_of_turn_silence_sec = end_of_turn_silence_sec
        self.cooldown_ms = cooldown_ms
        self.debug = debug

        self.cooldown_frames = int(cooldown_ms / frame_ms)
        self.end_of_turn_silence_frames = int((end_of_turn_silence_sec * 1000) / frame_ms)
        self.pre_roll_frames = max(1, int(pre_roll_ms / frame_ms))
        self.min_utt_samples = int(sr * (min_utterance_ms / 1000.0))
        self.max_utt_samples = int(sr * (max_utterance_ms / 1000.0))

        self.eps = 1e-12
        self._lock = threading.Lock()
        self._stream = None
        self._running = False

        self.muted = False

        self._reset_state()


    def mute(self):
        self.muted = True
        # Don't call _reset_state() — it wipes noise_db causing false triggers on unmute
        self.in_speech = False
        self.post_speech = False
        self.speech_count = 0
        self.silence_count = 0
        self.post_silence = 0
        self.utt_frames = []
        self.utt_samples = 0
        self.pre_roll = []
        if self.debug:
            print("[VAD] muted")

    def unmute(self):
        self.in_speech = False
        self.post_speech = False
        self.speech_count = 0
        self.silence_count = 0
        self.post_silence = 0
        self.utt_frames = []
        self.utt_samples = 0
        self.pre_roll = []
        self.noise_db = -45.0
        # self.cooldown = max(self.cooldown_frames, int(3000 / self.frame_ms))  # 3s — covers full TTS playback
        self.cooldown = max(self.cooldown_frames, int(600 / self.frame_ms))
        self.muted = False
        if self.debug:
            print(f"[VAD] unmuted — cooldown={self.cooldown} frames")

    def _reset_state(self):
        self.in_speech = False
        self.post_speech = False
        self.speech_count = 0
        self.silence_count = 0
        self.post_silence = 0
        self.noise_db = -60.0
        self.cooldown = 0
        self.frame_counter = 0

        self.pre_roll = []
        self.utt_frames = []
        self.utt_samples = 0

    def _rms_db(self, x: np.ndarray) -> float:
        rms = np.sqrt(np.mean(x * x) + self.eps)
        return 20.0 * np.log10(rms + self.eps)

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            blocksize=self.frame_samples,
            callback=self._audio_callback,
            dtype="float32",
        )
        self._stream.start()
        print("[VAD] Started.")

    def stop(self):
        with self._lock:
            self._running = False

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            pass

        print("[VAD] Stopped.")

    def _finalize_utterance(self, reason: str):
        if self.utt_samples < self.min_utt_samples:
            if self.debug:
                ms = int(self.utt_samples / self.sr * 1000)
                print(f"[VAD] drop too short: {ms}ms reason={reason}")
            self.utt_frames = []
            self.utt_samples = 0
            self.cooldown = self.cooldown_frames
            return

        audio = np.concatenate(self.utt_frames, axis=0).astype(np.float32)
        ms = int(len(audio) / self.sr * 1000)

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wav_path = self.out_dir / f"vad_{ts}.wav"
        sf.write(wav_path, audio, self.sr)

        if self.debug:
            print(f"[VAD] finalize {ms}ms reason={reason} -> {wav_path}")

        self.voice_queue.put(str(wav_path))

        self.utt_frames = []
        self.utt_samples = 0
        self.cooldown = self.cooldown_frames

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._running:
            return

        if self.muted:
            return

        x = indata[:, 0].astype(np.float32)

        if self.cooldown > 0:
            self.cooldown -= 1
            self.pre_roll = []
            self.speech_count = 0
            return

        self.pre_roll.append(x.copy())
        if len(self.pre_roll) > self.pre_roll_frames:
            self.pre_roll.pop(0)

        e_db = self._rms_db(x)

        self.noise_db = max(self.noise_db, self.noise_clamp_db)
        threshold = self.noise_db + self.margin_db

        if e_db < self.ignore_below_db:
            is_speech = False
        else:
            is_speech = e_db > threshold

        if (not self.in_speech) and (not is_speech):
            self.noise_db = 0.90 * self.noise_db + 0.10 * e_db

        if is_speech:
            self.speech_count += 1
            self.silence_count = 0
        else:
            self.silence_count += 1
            self.speech_count = 0

        if self.post_speech and is_speech:
            self.post_speech = False
            self.post_silence = 0
            self.in_speech = True
            if self.debug:
                print("[VAD] resume")

        if (not self.in_speech) and (not self.post_speech) and (self.speech_count >= self.start_frames):
            self.in_speech = True
            self.utt_frames = list(self.pre_roll)
            self.utt_samples = sum(len(f) for f in self.utt_frames)
            if self.debug:
                print(f"[VAD] start seed={int(self.utt_samples/self.sr*1000)}ms")

        if self.in_speech:
            self.utt_frames.append(x.copy())
            self.utt_samples += len(x)

            if self.utt_samples >= self.max_utt_samples:
                self.in_speech = False
                self.post_speech = False
                self.post_silence = 0
                self._finalize_utterance("max_len")
                return

            if self.silence_count >= self.end_frames:
                self.in_speech = False
                self.post_speech = True
                self.post_silence = 0
                if self.debug:
                    print("[VAD] end -> wait end-of-turn")
                return

        if self.post_speech and not is_speech:
            self.post_silence += 1
            if self.post_silence >= self.end_of_turn_silence_frames:
                self.post_speech = False
                self.post_silence = 0
                self._finalize_utterance("end_of_turn_timeout")
