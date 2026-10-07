import re

INTERRUPT_PATTERNS = [
    r"\bnever mind\b",
    r"\bnevermind\b",
    r"\bi changed my mind\b",
    r"\bchange my mind\b",
    r"\bforget it\b",
    r"\bscratch that\b",
    r"\bcancel\b",
    r"\bstop\b",
    r"\bignore that\b",
    r"\bno thanks\b",
    r"\bactually no\b",          # ← ADD
    r"\bactually forget\b",      # ← ADD
    r"\bactually never mind\b",  # ← ADD
    r"\bwait no\b",              # ← ADD (natural speech)
    r"\bhold on\b",              # ← ADD (natural speech)
]

PAUSE_PATTERNS = [
    r"\bwait\b",
    r"\bwait a moment\b",
    r"\bhold on\b",
    r"\bhold up\b",
    r"\bbefore that\b",
    r"\bquick question\b",
    r"\bcan you first\b",
    r"\bone more thing\b",
]

RESUME_PATTERNS = [
    r"\bresume\b",
    r"\bcontinue previous\b",
    r"\bback to that\b",        # "back to that" — no topic specified
    r"\breturn to that\b",      # "return to that" — no topic specified
    r"\bpick up where\b",
    r"\bcontinue where\b",
    r"\bunpause\b",
]


# An interrupt phrase only counts where a person actually says it AS a cancel:
#   - at the very start            "never mind, what time is it"
#   - after a pause/self-correction "remind me about the — actually no forget it"
#   - after a lead-in word          "ok stop", "please cancel that", "actually never mind"
# and not when it is buried mid-clause as ordinary vocabulary:
#   "I told him never mind and moved on", "how do I stop overshoot in my PID loop".
# (A plain word-count cap was tried first and rejected: it missed mid-sentence
# self-corrections and long genuine cancels, and still fired on short questions.)
_BOUNDARY_CHARS = tuple("—–-,;:.!?…")
_LEAD_WORDS = {
    "actually", "wait", "no", "oh", "ok", "okay", "hmm", "sorry", "please",
    "um", "uh", "so", "but", "and", "yeah", "well", "just",
}


def _starts_a_clause(prefix: str) -> bool:
    prefix = prefix.rstrip()
    if not prefix or prefix.endswith(_BOUNDARY_CHARS):
        return True
    words = re.findall(r"[a-z']+", prefix)
    return bool(words) and words[-1] in _LEAD_WORDS


def is_interrupt(text: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    # Guard: device-specific stop is a command, not a conversation cancel
    if re.search(r"\bstop\s+(the\s+)?(rover|drone|tello|robot|arm)\b", t):
        return False

    # Guard: design requirement with stop is not an interrupt
    if re.search(r"\bstop\b.{0,30}\b(when|if|battery|percent|threshold)\b", t):
        return False

    for pattern in INTERRUPT_PATTERNS:
        for m in re.finditer(pattern, t):
            if _starts_a_clause(t[: m.start()]):
                return True
    return False

def is_pause(text: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return any(re.search(p, t) for p in PAUSE_PATTERNS)

def looks_like_new_task(text: str) -> bool:
    t = (text or "").lower()
    t = re.sub(r"\s+", " ", t).strip()

    # --- Core task keywords ---
    # TASK_KEYS = [
    #     # memory
    #     "recall", "remember", "what did we talk", "past conversation",
    #     "save", "note", "take a note", "write down", "store this",
    #     "delete", "forget", "remove",

    #     # schedule
    #     "remind", "reminder", "schedule", "meeting", "appointment",

    #     # system queries
    #     "time", "date", "weather", "location",
    # ]
    TASK_KEYS = [
        # memory
        "recall", "remember", "past conversation", "what did we talk",
        "save", "note", "take a note", "store", "delete", "forget", "remove",
        # schedule
        "remind", "schedule", "meeting", "appointment",
        # system
        "time", "date", "weather", "location", "cpu", "battery",
        # device
        "tello", "drone", "device", "connect", "disconnect",
        "camera", "stream", "monitor", "take off", "takeoff", "land", "hover", "follow me",
    ]

    # --- Device / robot / camera tasks ---
    DEVICE_KEYS = [
        # explicit device names
        "tello", "drone", "rover", "robot",

        # camera / perception
        "camera", "open camera", "turn on camera", "stream",
        "video", "see", "look", "look at me",

        # motion / flight
        "take off", "takeoff", "land", "hover",
        "fly", "follow me", "track", "scan",

        # execution verbs
        "start", "stop", "run", "launch", "execute",
    ]

    # --- Guard against false positives ---
    CHAT_ONLY_PHRASES = [
        "talk about",
        "discuss",
        "explain",
        "what is",
        "tell me about",
    ]

    # If it looks like explanation / chat, not a task
    if any(p in t for p in CHAT_ONLY_PHRASES):
        return False

    return any(k in t for k in TASK_KEYS) or any(k in t for k in DEVICE_KEYS)


# def looks_like_new_task(text: str) -> bool:
#     t = (text or "").lower()
#     t = re.sub(r"\s+", " ", t).strip()

#     # keywords that clearly indicate a task request
#     TASK_KEYS = [
#         # memory
#         "recall", "remember", "what did we talk", "past conversation",
#         "save", "note", "take a note", "write down", "store this",
#         "delete", "forget", "remove",
#         # schedule
#         "remind", "reminder", "schedule", "meeting", "appointment",
#         # system
#         "time", "date", "weather", "location",
#     ]
#     return any(k in t for k in TASK_KEYS)

def is_resume(text: str) -> bool:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)

    return any(re.search(p, t) for p in RESUME_PATTERNS)
