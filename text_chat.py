"""
Text-only Buddy: type in, text out. No microphone, speakers, camera or robots.

Needs Ollama running locally with the llama3.2 model (`ollama pull llama3.2`).
Real-world calls (web, calendar, camera) are switched off, like in the evaluation runner.

    python text_chat.py
"""
import unittest.mock as mock


def main() -> None:
    mock.patch("vision.webcam_capture.capture_single_frame", return_value=None).start()  # no camera

    from brain.conversation_manager import ConversationManager
    from brain.core.brain import MemoryManager
    from brain.core.internet_access.internet_service import get_explorer
    from brain.identity.identity_manager import IdentityManager

    manager = ConversationManager(
        MemoryManager(),
        tello_client=None,
        explorer=get_explorer(enabled=False),
        identity=IdentityManager(),
    )
    print("Buddy (text mode). Type 'quit' to exit.\n")
    while True:
        try:
            text = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            print(f"buddy > {manager.handle_request(text)}\n")


if __name__ == "__main__":
    main()
