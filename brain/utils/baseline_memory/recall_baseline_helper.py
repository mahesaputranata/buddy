import re
from brain.core.brain import extract_topic_name
from difflib import SequenceMatcher

class handleGeneral:

    @staticmethod
    def extract_all_topic(formatted_topics: str) -> list[str]:
        if not formatted_topics:
            return []
        lines = [l.strip() for l in formatted_topics.splitlines() if l.strip()]
        return [l[2:].strip() for l in lines if l.startswith("- ")]
    
    @staticmethod
    def extract_all_contents(topics: list[str]) -> list[dict]:
        return [{"topic": t, "last_discussed": None, "summary": None} for t in topics]
    

def _tts_join(items: list[str]) -> str:
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def _sanitize_for_tts(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[_*`]+", "", text)   
    text = re.sub(r"\s+", " ", text)
    return text


def compiler_general(topics: list[str]) -> str:
    if not topics:
        return "I couldn't find any past discussions in my memory. Let's start something new here."

    clean_topics = [_sanitize_for_tts(t) for t in topics]
    topics_phrase = _tts_join(clean_topics)

    return (
        f"As I remember, we’ve talked about {topics_phrase}. "
        "If you want to continue, say the topic name "
        "If you want a recap, say summarize followed by the topic name. "
        "If you want me to remove a topic, say forget followed by the topic name."
    )

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def resolve_topic_from_utterance(user_text: str, topics: list[str]) -> str | None:
    t = (user_text or "").strip()
    tl = t.lower()

    if not topics:
        return None

    # 1) Try explicit extraction (your rules + LLM fallback)
    extracted = extract_topic_name(t)
    if extracted:
        el = extracted.lower()

        # exact/substring against known topics (prefer canonical topic from list)
        for known in topics:
            kl = known.lower()
            if el == kl or el in kl or kl in el:
                return known

        # fuzzy match extracted -> known topics
        best = None
        best_score = 0.0
        for known in topics:
            score = _similar(extracted, known)
            if score > best_score:
                best_score = score
                best = known
        if best and best_score >= 0.72:
            return best

    # 2) Direct substring match from raw utterance
    for known in topics:
        if known.lower() in tl:
            return known

    # 3) Word overlap fallback
    words = set(re.findall(r"[a-z0-9]+", tl))
    best = None
    best_score = 0
    for known in topics:
        tw = set(re.findall(r"[a-z0-9]+", known.lower()))
        score = len(words & tw)
        if score > best_score:
            best_score = score
            best = known
    if best and best_score > 0:
        return best

    # 4) Last resort fuzzy on full utterance (optional)
    best = None
    best_score = 0.0
    for known in topics:
        score = _similar(t, known)
        if score > best_score:
            best_score = score
            best = known
    if best and best_score >= 0.70:
        return best

    return None