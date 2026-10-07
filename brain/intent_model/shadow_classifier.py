"""
brain/intent_model/shadow_classifier.py

Shadow-mode wrapper around the fine-tuned intent classifier.

"Shadow" means: it runs on every real turn and its prediction is written to the
session log (entry["shadow"]) next to what the old classifier decided — but nothing
in Buddy ever acts on it. That gives a live accuracy read on real traffic before any
cutover decision. See analyze_shadow.py for reading the results.

Safety properties (this sits on the voice loop's hot path):
  - predict() never raises; any failure returns None.
  - The model loads in a background thread on first use. Until it is ready,
    predict() returns None, so startup and the first turns are never delayed.
  - Disable entirely with env BUDDY_SHADOW_INTENT=0.

Model location, in order:
  1. env BUDDY_INTENT_MODEL_DIR
  2. <Buddy root>/models/intent_classifier/final   <- the DEPLOYED model (put a model here to promote it)
  3. ./logs/sessions/intent_classifier_model/final  (training output; legacy fallback)

If a VERSION.txt sits next to config.json (or one folder up) its first line is logged as
shadow["model_version"], so logs from different model versions can be told apart later.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from brain.intent_model.formatting import build_input_text

ENV_DISABLE = "BUDDY_SHADOW_INTENT"
ENV_MODEL_DIR = "BUDDY_INTENT_MODEL_DIR"
MAX_LENGTH = 256  # must match training

_BUDDY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIRS = [
    _BUDDY_ROOT / "models" / "intent_classifier" / "final",
    Path("logs/sessions/intent_classifier_model/final"),
]
_SYNTHETIC_INPUT_RE = re.compile(r"^__[A-Z0-9_]+__$")


class ShadowIntentClassifier:
    def __init__(self):
        self.enabled = os.environ.get(ENV_DISABLE, "1") != "0"
        self.state = "idle"  # idle | loading | ready | failed
        self.model_dir: Optional[Path] = None
        self.model_version = "unversioned"
        self._tok = None
        self._model = None
        self._torch = None
        self._id2label: dict[int, str] = {}

    # ── loading ──────────────────────────────────────────────────────────────

    def _find_model_dir(self) -> Optional[Path]:
        candidates = []
        env = os.environ.get(ENV_MODEL_DIR)
        if env:
            candidates.append(Path(env))
        candidates.extend(_DEFAULT_DIRS)
        for c in candidates:
            if (c / "config.json").exists():
                return c
        return None

    def _load(self):
        try:
            model_dir = self._find_model_dir()
            if model_dir is None:
                raise FileNotFoundError(
                    f"no model found (set {ENV_MODEL_DIR} or put it at {_DEFAULT_DIRS[0]})"
                )
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            t0 = time.perf_counter()
            self._tok = AutoTokenizer.from_pretrained(str(model_dir))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            self._model.eval()
            self._torch = torch
            self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}
            self.model_dir = model_dir
            for vf in (model_dir / "VERSION.txt", model_dir.parent / "VERSION.txt"):
                if vf.exists():
                    # utf-8-sig: tolerate the BOM that Windows editors/PowerShell add
                    self.model_version = vf.read_text(encoding="utf-8-sig").strip().splitlines()[0][:80]
                    break
            self.state = "ready"
            print(f"[ShadowIntent] ready in {time.perf_counter() - t0:.1f}s "
                  f"({len(self._id2label)} labels, version {self.model_version!r}) from {model_dir}")
        except Exception as e:
            self.state = "failed"
            print(f"[ShadowIntent] disabled — could not load model: {e}")

    def _ensure_loading(self):
        if self.state == "idle":
            self.state = "loading"
            threading.Thread(target=self._load, name="shadow-intent-load", daemon=True).start()

    # ── inference ────────────────────────────────────────────────────────────

    @property
    def labels(self) -> list[str]:
        return [self._id2label[i] for i in sorted(self._id2label)]

    def predict(
        self,
        user_input: str,
        last_topic=None,
        pending_before=None,
        prev_input=None,
        prev_response=None,
    ) -> Optional[dict]:
        """
        Returns {"label", "confidence", "top3": [[label, p], ...], "latency_ms"} or None.
        Context args must be the state BEFORE this turn ran (see formatting.py).
        """
        if not self.enabled:
            return None
        try:
            if _SYNTHETIC_INPUT_RE.match((user_input or "").strip()):
                return None  # test-harness scaffolding, not speech
            self._ensure_loading()
            if self.state != "ready":
                return None

            text = build_input_text(user_input, last_topic, pending_before, prev_input, prev_response)
            t0 = time.perf_counter()
            enc = self._tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with self._torch.no_grad():
                probs = self._torch.softmax(self._model(**enc).logits[0], dim=-1)
            k = min(3, probs.shape[0])
            top_p, top_i = self._torch.topk(probs, k)
            top3 = [[self._id2label[int(i)], round(float(p), 4)] for p, i in zip(top_p, top_i)]
            return {
                "label": top3[0][0],
                "confidence": top3[0][1],
                "top3": top3,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "model_version": self.model_version,
            }
        except Exception as e:
            print(f"[ShadowIntent] predict failed: {e}")
            return None


_shadow = ShadowIntentClassifier()


def get_shadow() -> ShadowIntentClassifier:
    return _shadow
