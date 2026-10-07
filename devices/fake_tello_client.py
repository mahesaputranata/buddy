import time

class FakeTelloClient:
    def __init__(self):
        self.connected = False
        self.streaming = False
        self.flying = False

    def connect(self):
        self.connected = True
        return {"ok": True, "msg": "FAKE: connected"}

    def disconnect(self):
        self.connected = False
        self.streaming = False
        self.flying = False
        return {"ok": True, "msg": "FAKE: disconnected"}

    def get_status(self):
        return {
            "ok": True,
            "connected": self.connected,
            "streaming": self.streaming,
            "flying": self.flying,
            "brightness_mean": 72.3,
            "contrast_std": 18.1,
            "light_safe": True,
            "light_reason": "lighting_ok",
        }

    def start_monitor(self):
        if not self.connected:
            return {"ok": False, "msg": "FAKE: not connected"}
        self.streaming = True
        return {"ok": True, "msg": "FAKE: monitor started"}

    def stop_monitor(self):
        self.streaming = False
        return {"ok": True, "msg": "FAKE: monitor stopped"}

    def takeoff(self):
        if not self.connected:
            return {"ok": False, "msg": "FAKE: not connected"}
        self.flying = True
        return {"ok": True, "msg": "FAKE: takeoff"}

    def hover(self, seconds=3):
        if not self.flying:
            return {"ok": False, "msg": "FAKE: not flying"}
        time.sleep(0.1)
        return {"ok": True, "msg": f"FAKE: hover {seconds}s"}

    def land(self):
        self.flying = False
        return {"ok": True, "msg": "FAKE: landed"}
