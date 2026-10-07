import re
 
# Internal identifiers that must never reach the user
_TOPIC_ID_RE = re.compile(r"\btopic_\d+\b", re.IGNORECASE)
_INTERNAL_TOKEN_RE = re.compile(
    r"\b(?:current_conversation|session:\w+|web:[^\s]+|vision_awareness_\d+)\b",
    re.IGNORECASE,
)
 
 
def sanitize_for_tts(text: str) -> str:
    """
    Strips internal identifiers from text before it reaches TTS.
    This is the last line of defense — upstream code should already
    avoid generating these, but if anything leaks, we catch it here.
    """
    if not text:
        return text
 
    # Replace topic_NNNN with a natural pronoun
    text = _TOPIC_ID_RE.sub("this", text)
 
    # Replace other internal tokens with neutral phrasing
    text = _INTERNAL_TOKEN_RE.sub("our discussion", text)
 
    # Clean up the awkward phrasing that results
    text = re.sub(r"\bhelp you with this\?", "help you?", text)
    text = re.sub(r"\bsomething new on this\b", "something new", text)
    text = re.sub(r"\babout this\?\s*Please", "about? Please", text)
 
    return text