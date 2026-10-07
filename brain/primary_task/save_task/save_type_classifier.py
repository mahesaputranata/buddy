from numpy import dot
from numpy.linalg import norm
import numpy as np

SAVE_INTENT_EXAMPLES = {
    "conversation_save": [
        "save our conversation about this",
        "store what we discussed",
        "keep our talk about reinforcement learning",
        "save our discussion from this session",
        "can you save what we talked about",
    ],
    "general": [
        "take a note",
        "jot this down",
        "save this",
        "keep this in mind",
        "this is important remember it",
        "log this for me",
        "don't let me forget this",
        "write this down",
    ],
    "preamble": [
        "i want to tell you something important",
        "i have something important",
        "i need you to remember something",
        "there is something i want to share",
        "i want to talk about something",
    ],
    "specific": [
        "save about the LQR controller",
        "remember the details about project alpha",
        "note about our meeting with John",
        "save regarding the drone setup",
    ],
}

class SaveTypeClassifier:
    def __init__(self, ollama_client, model: str = "llama3.2:latest"):
        self.ollama = ollama_client
        self.model = model
        self._cache = {}          # input → embedding cache
        self._example_cache = {}  # precomputed example embeddings

        # precompute on init so first call isn't slow
        self._precompute_examples()

    def _embed(self, text: str) -> np.ndarray:
        text = (text or "").strip().lower()
        if text in self._cache:
            return self._cache[text]
        
        resp = self.ollama.embeddings(model=self.model, prompt=text)
        vec = np.array(resp["embedding"], dtype=np.float32)
        self._cache[text] = vec
        return vec

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = norm(a) * norm(b)
        if denom == 0:
            return 0.0
        return float(dot(a, b) / denom)

    def _precompute_examples(self):
        for intent, examples in SAVE_INTENT_EXAMPLES.items():
            vecs = [self._embed(ex) for ex in examples]
            self._example_cache[intent] = vecs
        print("[SaveClassifier] Example embeddings precomputed.")

    def classify(self, user_input: str) -> tuple[str, float]:
        """
        Returns (save_type, confidence)
        save_type: "conversation_save" | "general" | "preamble" | "specific" | "unknown"
        """
        vec = self._embed(user_input)

        scores = {}
        for intent, example_vecs in self._example_cache.items():
            # score = max similarity against any example
            # (nearest neighbor approach)
            sims = [self._cosine(vec, ev) for ev in example_vecs]
            scores[intent] = max(sims)

        best_intent = max(scores, key=scores.get)
        best_score  = scores[best_intent]

        print(f"[SaveClassifier] scores={scores}")
        print(f"[SaveClassifier] best={best_intent} score={best_score:.3f}")

        # confidence threshold
        if best_score < 0.70:
            return "unknown", best_score

        return best_intent, best_score