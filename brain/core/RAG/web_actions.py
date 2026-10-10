import re
from typing import Optional, Tuple

_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}

_OPEN_SYNS = ("open", "show", "read", "go to", "visit")
_SUMMARY_SYNS = ("summarize", "summary", "tell me about", "explain", "key points", "what's in", "whats in")
_COMPARE_SYNS = ("compare", "versus", "vs")


def word_to_num(t: str) -> Optional[int]:
    t = (t or "").strip().lower()
    if t.isdigit():
        return int(t)
    return _WORD_NUM.get(t)


def _extract_any_number_phrase(text: str) -> Optional[int]:
    """
    Matches: "1", "#1", "one", "number one", "result two", "open result 3"
    """
    t = (text or "").strip().lower()

    m = re.search(r"\b#?(\d+)\b", t)
    if m:
        return int(m.group(1))

    # 'number one' / 'result two'
    m = re.search(r"\b(?:number|result)\s+([a-z]+)\b", t)
    if m:
        return word_to_num(m.group(1))

    # bare word number anywhere
    for w, n in _WORD_NUM.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            return n

    return None


def looks_like_web_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    n = _extract_any_number_phrase(t)
    if n is not None:
        return bool(re.search(r"\b(result|open|show|read|summarize|summary|compare|vs|versus)\b", t))


def _has_any_syn(t: str, syns: tuple[str, ...]) -> bool:
    # safer than "k in t" for short words; still supports multiword syns
    for k in syns:
        if " " in k:
            if k in t:
                return True
        else:
            if re.search(rf"\b{re.escape(k)}\b", t):
                return True
    return False

def _mentions_pronoun_ref(t: str) -> bool:
    # “it / this / that / the article / the result / the first one”
    return bool(re.search(r"\b(it|this|that|the result|the article|that one|this one)\b", t))

def parse_web_followup(text: str) -> Optional[Tuple[str, Optional[int]]]:

    t = (text or "").strip().lower()
    if not t:
        return None

    # detect compare here or let your compare regex handle it later
    if _has_any_syn(t, _COMPARE_SYNS):
        return None

    n = _extract_any_number_phrase(t)  # may be None

    # action detection (prefer explicit actions)
    if _has_any_syn(t, _OPEN_SYNS):
        return ("open", n)  # n can be None -> implicit “open it”
    if _has_any_syn(t, _SUMMARY_SYNS):
        return ("summarize", n)  # n can be None -> implicit “summarize it”

    # If no action words but user gave a number -> default summarize that number
    if n and n > 0:
        return ("summarize", n)

    return None

def parse_index_utterance(text: str) -> Optional[int]:
    """
    Only returns an index if the utterance is basically *just* selecting a number,
    with no action words (open/summarize/compare/etc).
    """
    t = (text or "").strip().lower()

    # If it contains action-ish words, it's not a bare index selection.
    if re.search(r"\b(open|show|read|go to|visit|summarize|summary|tell|explain|compare|vs|versus)\b", t):
        return None

    # Accept common bare forms
    # "one" / "2" / "#3" / "number one" / "result two"
    if re.fullmatch(r"#?\d+", t):
        n = int(re.sub(r"\D", "", t))
        return n if n > 0 else None

    m = re.fullmatch(r"(?:number|result)\s+([a-z]+|\d+)", t)
    if m:
        n = word_to_num(m.group(1))
        return n if n and n > 0 else None

    if re.fullmatch(r"[a-z]+", t):
        n = word_to_num(t)
        return n if n and n > 0 else None

    return None