from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_PATH = Path(__file__).with_name("semantic_episode_embeddings.json")

_embedding_model = None
_embedding_model_failed = False


@dataclass
class Episode:
    """
    Structured view of one remembered exchange.

    Created from the existing temp_topic_memory shape:
        topic -> list[str]
    """
    episode_id: str
    topic: str
    raw_text: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    importance: float = 0.5
    timestamp: str = ""
    parent_topic: Optional[str] = None
    sibling_topics: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None


@dataclass
class EpisodeMatch:
    episode: Episode
    score: float
    reason: str = ""


@dataclass
class MemoryRoute:
    """Final decision from the episodic memory router."""
    kind: str
    topic: str
    score: float = 0.0
    selected_episodes: list[EpisodeMatch] = field(default_factory=list)
    reason: str = ""

    @property
    def has_memory(self) -> bool:
        return bool(self.selected_episodes)

    @property
    def is_recall(self) -> bool:
        return self.kind == "episodic_recall"

    @property
    def is_continue(self) -> bool:
        return self.kind == "continue_current"

    @property
    def is_new_topic(self) -> bool:
        return self.kind == "new_topic"


class EpisodicMemoryRouter:
    """
    Routes memory context by semantic similarity.

    Public interface intentionally matches the old rule-based router:
        route(user_input, last_topic) -> MemoryRoute

    Internally this builds one Episode per stored exchange, embeds every
    episode, persists those vectors in a small JSON cache, and brute-force
    cosine-scores the current utterance against all stored episodes.
    """

    def __init__(
        self,
        topic_memory: dict,
        topic_graph: Optional[dict] = None,
        current_threshold: float = 0.38,
        recall_threshold: float = 0.34,
        max_episode_matches: int = 5,
        similarity_fn: Optional[Callable[[str, str, str], bool]] = None,
        topic_extractor: Optional[Callable[[str], str]] = None,
        topic_cleaner: Optional[Callable[[str], str]] = None,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embedding_cache_path: Optional[str | Path] = None,
    ):
        self.topic_memory = topic_memory or {}
        self.topic_graph = topic_graph or {}
        self.current_threshold = current_threshold
        self.recall_threshold = recall_threshold
        self.max_episode_matches = max_episode_matches
        self.similarity_fn = similarity_fn
        self.topic_extractor = topic_extractor
        self.topic_cleaner = topic_cleaner
        self.embedding_model_name = embedding_model_name
        self.embedding_cache_path = Path(embedding_cache_path or EMBEDDING_CACHE_PATH)
        self._embedding_cache = self._load_embedding_cache()

        self.episodes: list[Episode] = self._build_episodes_from_topic_memory(
            self.topic_memory
        )
        self._ensure_episode_embeddings()

    def route(self, user_input: str, last_topic: str) -> MemoryRoute:
        user_input = (user_input or "").strip()
        last_topic = self._norm_topic(last_topic)

        if not user_input:
            return MemoryRoute(
                kind="empty",
                topic=last_topic,
                reason="Empty user input.",
            )

        current_matches = self._score_episodes(
            user_input=user_input,
            restrict_topic=last_topic,
        )
        recall_matches = self._score_episodes(
            user_input=user_input,
            exclude_topic=last_topic,
        )

        best_current = current_matches[0] if current_matches else None
        best_recall = recall_matches[0] if recall_matches else None

        if best_current and best_current.score >= self.current_threshold:
            return MemoryRoute(
                kind="continue_current",
                topic=last_topic,
                score=best_current.score,
                selected_episodes=current_matches[:self.max_episode_matches],
                reason=(
                    "Current topic selected by embedding cosine similarity "
                    f"({best_current.score:.2f})."
                ),
            )

        if best_recall and best_recall.score >= self.recall_threshold:
            recalled_topic = self._norm_topic(best_recall.episode.topic)
            return MemoryRoute(
                kind="episodic_recall",
                topic=recalled_topic,
                score=best_recall.score,
                selected_episodes=recall_matches[:self.max_episode_matches],
                reason=(
                    f"Older episode '{best_recall.episode.episode_id}' matched "
                    f"by embedding cosine similarity ({best_recall.score:.2f})."
                ),
            )

        new_topic = self._extract_new_topic(user_input)
        return MemoryRoute(
            kind="new_topic",
            topic=new_topic,
            score=0.0,
            selected_episodes=[],
            reason="No semantic memory match passed threshold.",
        )

    def get_related_context(
        self,
        current_topic: str,
        max_episodes_per_topic: int = 3,
    ) -> dict[str, list[Episode]]:
        """
        Returns episodes from the current topic, its parent, and its siblings.

        This keeps the graph-facing helper intact for ConversationManager.
        """
        current_topic = self._norm_topic(current_topic)
        result = {"current": [], "parent": [], "siblings": []}

        if not current_topic:
            return result

        current_episodes = [
            ep for ep in self.episodes
            if self._norm_topic(ep.topic) == current_topic
        ]
        result["current"] = current_episodes[-max_episodes_per_topic:]

        node = self.topic_graph.get(current_topic, {})
        parent = node.get("parent")

        if parent:
            parent_norm = self._norm_topic(parent)
            parent_episodes = [
                ep for ep in self.episodes
                if self._norm_topic(ep.topic) == parent_norm
            ]
            result["parent"] = parent_episodes[-max_episodes_per_topic:]

            parent_node = self.topic_graph.get(parent_norm, {})
            sibling_topics = [
                self._norm_topic(topic)
                for topic in parent_node.get("children", [])
                if self._norm_topic(topic) != current_topic
            ]

            for sibling in sibling_topics:
                sibling_episodes = [
                    ep for ep in self.episodes
                    if self._norm_topic(ep.topic) == sibling
                ]
                if sibling_episodes:
                    result["siblings"].append(sibling_episodes[-1])

        return result

    def _score_episodes(
        self,
        user_input: str,
        exclude_topic: Optional[str] = None,
        restrict_topic: Optional[str] = None,
        parent_boost_topic: Optional[str] = None,
    ) -> list[EpisodeMatch]:
        exclude_topic = self._norm_topic(exclude_topic)
        restrict_topic = self._norm_topic(restrict_topic)

        query_embedding = self._embed_text(user_input)
        scored: list[EpisodeMatch] = []

        for episode in self.episodes:
            episode_topic = self._norm_topic(episode.topic)

            if self._should_skip_topic_for_recall(episode_topic):
                continue
            if exclude_topic and episode_topic == exclude_topic:
                continue
            if restrict_topic and episode_topic != restrict_topic:
                continue
            if not episode.embedding or query_embedding is None:
                continue

            score = self._cosine_similarity(query_embedding, episode.embedding)
            score += min(max(episode.importance, 0.0), 1.0) * 0.03

            if parent_boost_topic and episode.parent_topic == self._norm_topic(parent_boost_topic):
                score += 0.06

            if score > 0:
                scored.append(EpisodeMatch(
                    episode=episode,
                    score=round(score, 4),
                    reason="embedding_cosine",
                ))

        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:self.max_episode_matches]

    def _build_episodes_from_topic_memory(self, topic_memory: dict) -> list[Episode]:
        episodes: list[Episode] = []

        for topic, exchanges in topic_memory.items():
            topic_norm = self._norm_topic(topic)

            if self._should_skip_topic_for_storage(topic_norm):
                continue
            if not exchanges:
                continue

            graph_node = self.topic_graph.get(topic_norm, {})
            parent = graph_node.get("parent")

            for idx, exchange in enumerate(exchanges):
                raw_text = str(exchange or "").strip()
                if not raw_text:
                    continue

                episode = Episode(
                    episode_id=f"{topic_norm}:{idx}",
                    topic=topic_norm,
                    raw_text=raw_text,
                    summary=self._cheap_summary(raw_text),
                    keywords=self._extract_keywords(topic_norm + " " + raw_text),
                    importance=self._estimate_importance(raw_text),
                    timestamp="",
                    parent_topic=self._norm_topic(parent),
                )
                episodes.append(episode)

        return episodes

    def _ensure_episode_embeddings(self) -> None:
        changed = False

        for episode in self.episodes:
            cache_key = self._episode_cache_key(episode)
            cached = self._embedding_cache.get(cache_key)

            if isinstance(cached, list) and cached:
                episode.embedding = [float(value) for value in cached]
                continue

            embedding = self._embed_text(self._episode_retrieval_text(episode))
            if embedding is None:
                continue

            episode.embedding = embedding
            self._embedding_cache[cache_key] = embedding
            changed = True

        if changed:
            self._save_embedding_cache()

    def _embed_text(self, text: str) -> Optional[list[float]]:
        text = (text or "").strip()
        if not text:
            return None

        model = self._get_embedding_model()
        if model is None:
            return self._hashed_semantic_fallback(text)

        try:
            vector = model.encode(text, convert_to_numpy=False)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            return [float(value) for value in vector]
        except Exception as exc:
            print(f"[EpisodicMemoryRouter] embedding encode failed: {exc}")
            return self._hashed_semantic_fallback(text)

    def _get_embedding_model(self):
        global _embedding_model, _embedding_model_failed

        if _embedding_model is not None:
            return _embedding_model
        if _embedding_model_failed:
            return None

        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(self.embedding_model_name)
            return _embedding_model
        except Exception as exc:
            _embedding_model_failed = True
            print(
                "[EpisodicMemoryRouter] sentence-transformers unavailable; "
                f"using hashed fallback embeddings: {exc}"
            )
            return None

    def _episode_retrieval_text(self, episode: Episode) -> str:
        return "\n".join([
            f"Topic: {episode.topic}",
            f"Summary: {episode.summary}",
            f"Keywords: {' '.join(episode.keywords)}",
            episode.raw_text,
        ]).strip()

    def _load_embedding_cache(self) -> dict[str, list[float]]:
        try:
            if self.embedding_cache_path.exists():
                with self.embedding_cache_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            print(f"[EpisodicMemoryRouter] embedding cache read failed: {exc}")
        return {}

    def _save_embedding_cache(self) -> None:
        try:
            self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.embedding_cache_path.open("w", encoding="utf-8") as handle:
                json.dump(self._embedding_cache, handle)
        except Exception as exc:
            print(f"[EpisodicMemoryRouter] embedding cache write failed: {exc}")

    @staticmethod
    def _episode_cache_key(episode: Episode) -> str:
        payload = "\n".join([episode.topic, episode.raw_text])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{episode.episode_id}:{digest}"

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        size = min(len(left), len(right))
        if size == 0:
            return 0.0

        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0

        for idx in range(size):
            l_value = float(left[idx])
            r_value = float(right[idx])
            dot += l_value * r_value
            left_norm += l_value * l_value
            right_norm += r_value * r_value

        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0

        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))

    @staticmethod
    def _hashed_semantic_fallback(text: str, dims: int = 384) -> list[float]:
        """
        Dependency-safe fallback for environments without sentence-transformers.

        This is intentionally not the target semantic path, but it keeps tests
        and basic routing alive until the local model is installed.
        """
        vector = [0.0] * dims
        tokens = re.findall(r"[a-z0-9]{3,}", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return vector

        return [value / norm for value in vector]

    def _extract_new_topic(self, user_input: str) -> str:
        raw_topic = ""

        if self.topic_extractor is not None:
            try:
                raw_topic = self.topic_extractor(user_input) or ""
            except Exception as exc:
                print(f"[EpisodicMemoryRouter] topic_extractor failed: {exc}")

        if self.topic_cleaner is not None:
            try:
                cleaned = self.topic_cleaner(raw_topic) or ""
                if cleaned:
                    return self._norm_topic(cleaned)
            except Exception as exc:
                print(f"[EpisodicMemoryRouter] topic_cleaner failed: {exc}")

        if raw_topic:
            return self._norm_topic(raw_topic)

        return f"topic_{int(time.time()) % 10000}"

    def _should_skip_topic_for_storage(self, topic: str) -> bool:
        topic = self._norm_topic(topic)

        if not topic:
            return True
        if topic in {"unknown", "none", "null"}:
            return True

        return False

    def _should_skip_topic_for_recall(self, topic: str) -> bool:
        topic = self._norm_topic(topic)

        if not topic:
            return True
        if topic in {"unknown", "none", "null", "general", "chat"}:
            return True
        if topic.startswith("session:"):
            return True

        return False

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 12) -> list[str]:
        words = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        stop = {
            "the", "and", "for", "with", "that", "this", "you", "your",
            "what", "how", "why", "when", "where", "from", "about",
        }
        counts: dict[str, int] = {}

        for word in words:
            if word in stop:
                continue
            counts[word] = counts.get(word, 0) + 1

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in ranked[:max_keywords]]

    @staticmethod
    def _cheap_summary(text: str, max_chars: int = 280) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @staticmethod
    def _estimate_importance(text: str) -> float:
        lowered = (text or "").lower()
        markers = {
            "remember", "important", "project", "error", "bug", "fix",
            "decision", "decided", "goal", "plan", "architecture",
            "implementation", "memory", "router", "context",
        }
        hits = sum(1 for marker in markers if marker in lowered)

        if hits >= 4:
            return 0.9
        if hits >= 2:
            return 0.7
        if hits == 1:
            return 0.6
        return 0.5

    @staticmethod
    def _norm_topic(topic: Optional[str]) -> str:
        return (topic or "").strip().lower()
