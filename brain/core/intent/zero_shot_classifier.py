"""
brain/intent/zero_shot_classifier.py
=====================================
Zero-shot intent classifier using facebook/bart-large-mnli.

Replaces the llama3.2 LLM call in classify_intent() Step 3.

Key differences from LLM approach:
    - Deterministic: same input → same output every run
    - Fast: ~50ms vs ~2000ms for LLM
    - No hallucination: outputs probability distribution, not single confident pick
    - Model-agnostic: works regardless of which LLM you use for responses

Install dependencies:
    pip install transformers torch
    pip install transformers[torch]   (alternative)

First run will download ~1.5GB model to HuggingFace cache (~/.cache/huggingface).
Subsequent runs load from cache instantly.

Usage:
    from brain.intent.zero_shot_classifier import ZeroShotIntentClassifier

    classifier = ZeroShotIntentClassifier()  # load once at startup
    result = classifier.classify("nevermind forget it", ALLOWED_INTENTS)
    # → {"intent": "chat", "confidence": 0.82, "source": "zero_shot"}
"""

from __future__ import annotations
import time
from typing import Optional


# ── Intent label descriptions ─────────────────────────────────────────────────
# The zero-shot classifier works by comparing the input text against these
# human-readable descriptions. Better descriptions = better accuracy.
# These are tuned for Buddy's specific intents.

INTENT_DESCRIPTIONS = {
    "chat": (
        "general conversation, question, explanation, curiosity, opinion, "
        "learning request, technical discussion, how something works"
    ),
    "recall_memory": (
        "user wants to retrieve or look up something previously saved, "
        "asking what was stored or noted before, reviewing past notes"
    ),
    "save_memory": (
        "user explicitly wants to save or store the current conversation "
        "or current topic right now for future reference"
    ),
    "delete_memory": (
        "user wants to delete, remove, erase, forget, or wipe something "
        "from memory or storage"
    ),
    "schedule_save": (
        "user wants to set a reminder, schedule an appointment, add an event, "
        "or be reminded of something at a specific time"
    ),
    "schedule_recall": (
        "user asks about their schedule, upcoming events, reminders, "
        "or calendar appointments"
    ),
    "web_search": (
        "user explicitly wants to search the web, look something up, "
        "find current news or information online"
    ),
    "web_followup": (
        "user is following up on a previous web search result, "
        "wants to open, summarize, or compare search results"
    ),
    "device_control": (
        "user is giving a direct command to control a drone, rover, "
        "or robot device"
    ),
    "system_query": (
        "user is asking about the current time, today's date, battery level, "
        "local weather, or device location"
    ),
}


class ZeroShotIntentClassifier:
    """
    Wraps HuggingFace zero-shot classification pipeline.
    
    Load once at startup, reuse for every classification call.
    Thread-safe for single-threaded VAD loop (Buddy's architecture).
    """

    # Model options — ordered by quality vs speed tradeoff:
    #   facebook/bart-large-mnli     ~1.5GB  best accuracy  ~50ms on CPU
    #   cross-encoder/nli-MiniLM2-L6-H768  ~90MB   faster         ~20ms on CPU
    #   typeform/distilbart-mnli-12-3       ~600MB  middle ground
    DEFAULT_MODEL = "facebook/bart-large-mnli"
    FAST_MODEL    = "cross-encoder/nli-MiniLM2-L6-H768"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int = -1,          # -1 = CPU, 0 = first GPU
        multi_label: bool = False,  # False = softmax (one winner), True = sigmoid
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.multi_label = multi_label
        self.verbose = verbose
        self._pipeline = None
        self._load_time_s: Optional[float] = None

    def _ensure_loaded(self):
        """Lazy load — only downloads/loads model on first classify() call."""
        if self._pipeline is not None:
            return

        if self.verbose:
            print(f"[ZeroShot] Loading model: {self.model_name}")
            print(f"[ZeroShot] First run downloads ~1.5GB to HuggingFace cache...")

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )

        t0 = time.time()
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )
        self._load_time_s = time.time() - t0

        if self.verbose:
            print(f"[ZeroShot] Model loaded in {self._load_time_s:.1f}s")

    def classify(
        self,
        text: str,
        allowed_intents: list[str],
        low_conf_threshold: float = 0.40,
        fallback_intent: str = "chat",
    ) -> dict:
        """
        Classify text into one of the allowed intents.

        Args:
            text:               The user utterance (already lowercased is fine)
            allowed_intents:    List of intent strings to consider
            low_conf_threshold: Below this confidence → fallback to chat
            fallback_intent:    Intent to return when confidence is too low

        Returns:
            {"intent": str, "confidence": float, "source": "zero_shot"}
        """
        if not text or not text.strip():
            return {"intent": fallback_intent, "confidence": 0.0, "source": "zero_shot_empty"}

        self._ensure_loaded()

        # Build candidate labels from descriptions for intents we care about
        candidate_labels = []
        label_to_intent = {}

        for intent in allowed_intents:
            if intent in INTENT_DESCRIPTIONS:
                desc = INTENT_DESCRIPTIONS[intent]
            else:
                # Fallback: use intent name itself as description
                desc = intent.replace("_", " ")

            candidate_labels.append(desc)
            label_to_intent[desc] = intent

        if not candidate_labels:
            return {"intent": fallback_intent, "confidence": 0.5, "source": "zero_shot_no_labels"}

        t0 = time.time()
        try:
            result = self._pipeline(
                text,
                candidate_labels=candidate_labels,
                multi_label=self.multi_label,
            )
        except Exception as e:
            print(f"[ZeroShot] Classification error: {e}")
            return {"intent": fallback_intent, "confidence": 0.5, "source": "zero_shot_error"}

        elapsed_ms = (time.time() - t0) * 1000

        # result["labels"] is sorted by score descending
        best_label = result["labels"][0]
        best_score = result["scores"][0]
        best_intent = label_to_intent.get(best_label, fallback_intent)

        if self.verbose:
            print(f"[ZeroShot] {elapsed_ms:.0f}ms | "
                  f"top={best_intent} ({best_score:.2f}) | "
                  f"input={text[:60]!r}")

        # Low confidence → fallback to chat
        # This is the KEY difference from LLM: the classifier KNOWS when it's uncertain
        # "nevermind" → scores ~0.25 for delete_memory, 0.68 for chat → chat wins
        # LLM would output conf=1.00 for delete_memory (doesn't know what it doesn't know)
        if best_score < low_conf_threshold:
            print(f"[ZeroShot] Low confidence ({best_score:.2f}) → fallback to {fallback_intent}")
            return {
                "intent": fallback_intent,
                "confidence": best_score,
                "source": "zero_shot_low_conf",
            }

        return {
            "intent": best_intent,
            "confidence": best_score,
            "source": "zero_shot",
        }

    def classify_with_distribution(
        self,
        text: str,
        allowed_intents: list[str],
    ) -> dict:
        """
        Returns full probability distribution across all intents.
        Useful for debugging and threshold tuning.
        
        Returns:
            {
                "top_intent": str,
                "top_confidence": float,
                "distribution": {"intent": score, ...},
                "source": "zero_shot"
            }
        """
        if not text or not text.strip():
            return {"top_intent": "chat", "top_confidence": 0.0, "distribution": {}}

        self._ensure_loaded()

        candidate_labels = []
        label_to_intent = {}
        for intent in allowed_intents:
            desc = INTENT_DESCRIPTIONS.get(intent, intent.replace("_", " "))
            candidate_labels.append(desc)
            label_to_intent[desc] = intent

        result = self._pipeline(text, candidate_labels=candidate_labels)

        distribution = {
            label_to_intent.get(label, label): round(score, 3)
            for label, score in zip(result["labels"], result["scores"])
        }

        top_intent = label_to_intent.get(result["labels"][0], "chat")
        top_conf = result["scores"][0]

        return {
            "top_intent": top_intent,
            "top_confidence": top_conf,
            "distribution": distribution,
            "source": "zero_shot",
        }

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this in your main file instead of creating a new instance each call.
# Lazy-loaded on first use.

_classifier_instance: Optional[ZeroShotIntentClassifier] = None


def get_classifier(
    model_name: str = ZeroShotIntentClassifier.DEFAULT_MODEL,
    use_fast: bool = False,
) -> ZeroShotIntentClassifier:
    """
    Returns the module-level singleton classifier.
    Creates it on first call, reuses on subsequent calls.
    
    Args:
        model_name: Override model (default: facebook/bart-large-mnli)
        use_fast:   Use smaller/faster model instead of default
    """
    global _classifier_instance
    if _classifier_instance is None:
        model = ZeroShotIntentClassifier.FAST_MODEL if use_fast else model_name
        _classifier_instance = ZeroShotIntentClassifier(model_name=model)
    return _classifier_instance


def zero_shot_classify(
    text: str,
    allowed_intents: list[str],
    low_conf_threshold: float = 0.40,
) -> dict:
    """
    Convenience function — uses the singleton classifier.
    This is the function you call from classify_intent().
    
    Returns:
        {"intent": str, "confidence": float, "source": "zero_shot"}
    """
    classifier = get_classifier()
    return classifier.classify(text, allowed_intents, low_conf_threshold)