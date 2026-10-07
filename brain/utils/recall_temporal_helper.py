import re

TEMPORAL_REGEXES = [
    # Relative offsets
    r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b",
    r"\b(?:yesterday|today|tomorrow)\b",
    r"\b(?:last|this|next)\s+(?:week|month|year)\b",

    # Weekday references
    r"\b(?:last|this|next)\s+(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",

    # “on <date>” formats
    r"\b\d{4}-\d{2}-\d{2}\b",                 # 2025-12-20
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",   # 20/12/2025, 20.12.25, 20-12-2025

    # Month names (English)
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)\b",

    # Ranges
    r"\b(?:between|from)\b.*\b(?:and|to)\b",
]

# Optional: question forms that imply time even if no explicit date
TEMPORAL_QUESTION_FORMS = [
    r"\bwhen did we\b",
    r"\bwhat did we\b.*\b(last time|previously|before)\b",
    r"\bfrom (?:before|earlier)\b",
]

def has_temporal_cue(text: str) -> bool:
    t = (text or "").lower()
    for rx in TEMPORAL_REGEXES:
        if re.search(rx, t):
            return True
    for rx in TEMPORAL_QUESTION_FORMS:
        if re.search(rx, t):
            return True
    return False


