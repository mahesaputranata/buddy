# device_control/device_router.py

from __future__ import annotations

from typing import Any, Dict

from .rover_bridge import RoverClient
from .rover_controller import RoverTaskManager


class DeviceRouter:
    def __init__(self) -> None:
        self.rover_bridge = RoverClient()
        self.rover_manager = RoverTaskManager(self.rover_bridge)

    def handle_device_control(self, text: str) -> Dict[str, Any]:
        """
        Route device-control-related text.
        For now only rover is implemented.
        """
        result = self.rover_manager.handle(text)
        return result