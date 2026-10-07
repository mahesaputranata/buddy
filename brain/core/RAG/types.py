from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Literal, TypeAlias

SourceName: TypeAlias = Literal[
    "short_term",
    "long_term",
    "web",
    "schedule",
    "notes",
]


@dataclass
class ContextItem:
    source: SourceName
    title: str
    text: str
    url: Optional[str] = None
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    items: List[ContextItem] = field(default_factory=list)


@dataclass
class RetrievalOptions:
    top_k: int = 6
    per_source_k: int = 4
    include_sources: Optional[List[SourceName]] = None
    min_score: float = 0.0
    topic_hint: Optional[str] = None
    use_web_context: bool = True