import queue
from pynput import keyboard

CLIP_QUEUE = queue.Queue()

class ClipboardHotkey:
    def __init__(self):
        self.pressed = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )

    def start(self):
        self.listener.start()
        print("[CLIP] Hotkey ready: Ctrl+H")

    def _trigger(self):
        CLIP_QUEUE.put("CAPTURE")
        print("[CLIP] trigger")

    def _on_press(self, key):
        self.pressed.add(key)

        ctrl_down = (
            keyboard.Key.ctrl_l in self.pressed or
            keyboard.Key.ctrl_r in self.pressed
        )

        h_down = (
            keyboard.KeyCode.from_char('h') in self.pressed or
            keyboard.KeyCode.from_char('H') in self.pressed or
            any(getattr(k, "vk", None) == 72 for k in self.pressed if isinstance(k, keyboard.KeyCode))
        )

        if ctrl_down and h_down:
            if "__FIRED__" not in self.pressed:
                self.pressed.add("__FIRED__")
                self._trigger()

    def _on_release(self, key):
        self.pressed.discard(key)
        self.pressed.discard("__FIRED__")