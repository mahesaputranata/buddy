"""Voice entry point (needs a microphone, speakers and the full stack). No hardware? Use text_chat.py."""
from brain.conversation_manager import chat
from brain.settings import setting
from devices.tello_client import TelloClient

if __name__ == "__main__":
    agent = setting("tello_agent_py")
    tello_client = TelloClient(agent_py_path=agent, python_exe=setting("tello_python") or None) if agent else None

    try:
        chat(tello_client=tello_client)
    finally:
        if tello_client:
            print("[DEBUG] stopping client")
            tello_client.stop()
