import re
 
# Utterances that are NEVER topic introductions and should ALWAYS
# continue the current topic (if one exists).
_SHORT_FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    # Acknowledgments / fillers
    r"yes|yeah|yep|yup|sure|right|exactly|true|correct|"
    r"no|nope|nah|"
    r"okay|ok|alright|"
    # "I + opinion verb" (open to suggestion, agree, see, guess)
    r"i\s+(?:open|am\s+open|agree|see|guess|think\s+so|don't\s+know|"
    r"do\s+not\s+know|still\s+don't|still\s+do\s+not)"
    r"|"
    # Direct elaboration requests
    r"(?:can|could|would|will)\s+you\s+(?:elaborate|explain|clarify|"
    r"expand|continue|tell\s+me\s+more|go\s+deeper|repeat)"
    r"|"
    # "mean" leading — "mean can you elaborate", "I mean..."
    r"mean(?:\s+can\s+you)?|i\s+mean|what\s+(?:i\s+)?mean"
    r"|"
    # "tell me more", "go on", "continue"
    r"tell\s+me\s+more|go\s+on|continue|keep\s+going|"
    # "what about that", "what do you think"
    r"what\s+(?:do\s+you|about\s+(?:that|it|this))|"
    r"what'?s?\s+your\s+(?:take|opinion|view|thought)|"
    # Cancellation / dismissal
    r"never\s*mind|nevermind|forget\s+it|skip\s+(?:it|that)|"
    r"no\s+need|"
    # Apology / didn't hear
    r"sorry|pardon|"
    r"(?:i'?m\s+sorry|sorry)\s+(?:what|i)\s+(?:did(?:n'?t)?|did\s+not)\s+hear"
    r")\b",
    re.IGNORECASE,
)
 
# Pure pronoun-driven questions: "is it X", "can it Y", "does that work"
_PRONOUN_QUESTION_RE = re.compile(
    r"^\s*(?:what|how|why|where|when|who|which|"
    r"is|are|was|were|can|could|would|should|do|does|did|will)"
    r"\s+(?:it|this|that|they|them|those|these|"
    r"i|we|you)\s+",
    re.IGNORECASE,
)
 
# "in your opinion" style questions — always continuations
_OPINION_QUESTION_RE = re.compile(
    r"^\s*(?:in\s+your\s+opinion|"
    r"what\s+(?:do|would)\s+you\s+(?:think|suggest|recommend)|"
    r"do\s+you\s+(?:think|recommend|suggest|believe)|"
    r"should\s+i)\b",
    re.IGNORECASE,
)
 
 
def is_short_followup(text: str) -> bool:
    """
    Returns True if this utterance is a follow-up that should NOT
    trigger topic extraction or drift detection.
 
    This is the SHARED contract between extract_topic_name,
    continue_conversation's drift check, and EpisodicMemoryRouter.
    All three should call this first and bail if it returns True.
    """
    if not text:
        return False
 
    text = text.strip()
 
    # Very short utterances (≤3 words) are almost always followups
    word_count = len(text.split())
    if word_count <= 3:
        return True
 
    if _SHORT_FOLLOWUP_RE.match(text):
        return True
 
    if _PRONOUN_QUESTION_RE.match(text):
        return True
 
    if _OPINION_QUESTION_RE.match(text):
        return True
 
    return False
 
 
def is_explicit_topic_shift(text: str) -> bool:
    """
    Returns True only if the user explicitly signals they want
    to change topics. This is the ONLY thing that should override
    topic stickiness.
 
    Conservative on purpose — false positives here cause drift.
    """
    if not text:
        return False
 
    text_lower = text.lower().strip()
 
    EXPLICIT_SHIFTS = [
        r"\b(?:now\s+)?let'?s?\s+(?:talk|discuss)\s+about\b",
        r"\bchang(?:e|ing)\s+(?:the\s+)?(?:topic|subject)\b",
        r"\bdifferent\s+(?:topic|subject|question)\b",
        r"\bnew\s+(?:topic|subject|question)\b",
        r"\bswitching\s+(?:topics?|subjects?|gears)\b",
        r"\bmoving\s+on\s+to\b",
        r"\banyway\s*,?\s+(?:what|how|tell|let)",
        r"\bby\s+the\s+way\b",
        r"\bunrelated\s+(?:question|topic)\b",
        r"\bon\s+(?:a\s+)?different\s+note\b",
    ]
 
    return any(re.search(p, text_lower) for p in EXPLICIT_SHIFTS)

def _is_explicit_web_exit(text: str, mm, temp_topic_memory: dict) -> bool:
    """
    Returns True only if the user is explicitly leaving the web session
    to return to a previously-discussed non-web topic.
    """
    text = text.lower().strip()
    
    # Must contain a "return" signal
    return_signals = ["back to", "return to", "go back to", 
                      "let's discuss", "let's talk about"]
    if not any(s in text for s in return_signals):
        return False
    
    # Extract what comes after the return signal
    target = None
    for sig in return_signals:
        idx = text.find(sig)
        if idx >= 0:
            target = text[idx + len(sig):].strip()
            break
    
    if not target:
        return False
    
    # ── Guard 1: If target points to web-session internals, NOT an exit ──
    WEB_INTERNAL_WORDS = [
        "result", "article", "search", "source", "link",
        "news", "news 1", "news 2", "the first", "the second",
        "result 1", "result 2", "number", "one", "two", "three"
    ]
    if any(w in target[:30] for w in WEB_INTERNAL_WORDS):
        return False
    
    # ── Guard 2: Check if target matches an existing non-web topic ──
    # Extract meaningful words from target (first ~5 words)
    target_words = set(re.findall(r"[a-z]{3,}", target[:60].lower()))
    target_words -= {"the", "our", "this", "that", "discussion", "discussions",
                     "topic", "conversation", "about", "regarding"}
    
    if not target_words:
        return False
    
    # Look for matching non-web topic in memory
    for topic in temp_topic_memory.keys():
        if topic.startswith(("web:", "session:", "vision", "_pending_", "rover:")):
            continue
        topic_words = set(re.findall(r"[a-z]{3,}", topic.lower()))
        if target_words & topic_words:
            return True
    
    return False