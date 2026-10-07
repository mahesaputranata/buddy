import re

def is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"[.!?,]+$", "", t).strip()
    t = re.sub(r"[,\s]+(please|do it|go ahead|save it|store it|confirm|sure|now)\.?$", "", t).strip()
    t = re.sub(
        r"^(yes|yeah|yep)[,\s]+(please|sure|okay|ok|do it|go ahead|save it|store it|absolutely)\.?$",
        "yes",
        t,
    )

    if t in {
        "yes", "y", "yeah", "yep", "sure", "ok", "okay", "alright",
        "do it", "go ahead", "save it", "store it", "confirm",
        "please", "absolutely", "correct", "affirmative",
        "that's right", "thats right", "that's correct", "thats correct",
        "sounds good", "looks good", "perfect", "great",
        "yes go ahead", "yeah go ahead", "yes please",
        "yes do it", "yes sure", "ok go ahead", "okay go ahead",
        "yes save it", "yes store it", "yes save", "yes store",
        "go ahead and save", "go ahead save",
        "go ahead and store", "go ahead store",
        "yeah sure", "yeah ok", "yeah okay",
    }:
        return True

    if re.match(r"^(yes|yeah|yep|sure|ok|okay)\b", t) and len(t.split()) <= 4:
        if not re.search(r"\b(no|not|don'?t|never|cancel|stop)\b", t):
            return True

    return False


def is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"[.!?,]+$", "", t).strip()
    t = re.sub(r"[,\s]+(please|thanks|thank you|for now|at all)\.?$", "", t).strip()
    t = re.sub(
        r"^(no|nope|nah)[,\s]+(thanks|thank you|not now|later|don'?t|please)\.?$",
        "no",
        t,
    )

    if t in {
        "no", "n", "nope", "nah",
        "not now", "later", "not yet", "not today",
        "cancel", "skip", "stop", "abort",
        "don't", "dont", "do not",
        "negative", "never mind", "nevermind",
        "forget it", "forget that",
        "don't save", "dont save", "no save",
        "don't store", "dont store", "no store",
        "don't save it", "dont save it",
        "don't store it", "dont store it",
        "no thanks", "no thank you",
        "no need", "not needed", "not necessary",
        "that's fine", "thats fine", "i'm good", "im good",
        "all good", "we're good", "were good",
        "no don't save", "no dont save",
        "no don't save it", "no dont save it",
        "no don't store it", "no dont store it",
        "please don't save", "please dont save",
    }:
        return True

    if re.match(r"^(no|nope|nah)\b", t) and len(t.split()) <= 5:
        return True

    return False
