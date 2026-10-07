"""
train_intent_classifier.py — fine-tunes a small local encoder classifier on
the labeled intent dataset.

This is the "reflex" tier meant to replace the semantic-judgment portion of
classify_intent()'s regex cascade — fast, local, no LLM call needed at
inference time. Classes with too few examples to learn or evaluate reliably
are excluded from the classifier's label space entirely (see --min-class-
count) rather than force-included; that traffic should keep routing through
the existing rule-based/LLM fallback tiers, which don't need training data.

Model selection uses macro-F1 on val, not accuracy — with "chat" as ~55% of
the data, a model that always guessed "chat" would still look decent on raw
accuracy. Macro-F1 treats every class equally regardless of frequency.

Requires (not yet in myenv — install before running):
    pip install transformers datasets torch scikit-learn accelerate

Usage:
    python train_intent_classifier.py \\
        --train training_data/intent_train_augmented.jsonl \\
        --val training_data/intent_val.jsonl \\
        --test training_data/intent_test.jsonl \\
        --out-dir intent_classifier_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_label_space(train_rows: list[dict], min_class_count: int) -> tuple[list[str], dict[str, int]]:
    from collections import Counter
    counts = Counter(r["label"] for r in train_rows)
    excluded = {label: c for label, c in counts.items() if c < min_class_count}
    included = sorted(label for label in counts if label not in excluded)

    print(f"[LABELS] {len(included)} classes included, {len(excluded)} excluded (< {min_class_count} examples)")
    if excluded:
        print(f"[LABELS] excluded: {excluded}")
        print("[LABELS] traffic for these intents should keep routing through the "
              "existing rule-based/LLM fallback tiers, not this classifier.")

    label2id = {label: i for i, label in enumerate(included)}
    return included, label2id


def filter_and_encode(rows: list[dict], label2id: dict[str, int], split_name: str) -> list[dict]:
    kept, dropped = [], 0
    for r in rows:
        if r["label"] not in label2id:
            dropped += 1
            continue
        kept.append({"text": r["text"], "label": label2id[r["label"]]})
    if dropped:
        print(f"[{split_name}] dropped {dropped} rows with excluded labels")
    print(f"[{split_name}] {len(kept)} rows")
    return kept


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(rows: list[dict], num_labels: int) -> torch.Tensor:
    from collections import Counter
    counts = Counter(r["label"] for r in rows)
    total = sum(counts.values())
    weights = torch.ones(num_labels)
    for label_id in range(num_labels):
        c = counts.get(label_id, 1)
        weights[label_id] = total / (num_labels * c)
    return weights


def make_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }
    return compute_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="distilbert-base-uncased")
    ap.add_argument("--min-class-count", type=int, default=10,
                     help="Classes with fewer train examples than this are excluded from the classifier entirely")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    train_rows_raw = load_jsonl(args.train)
    val_rows_raw = load_jsonl(args.val)
    test_rows_raw = load_jsonl(args.test)

    labels, label2id = build_label_space(train_rows_raw, args.min_class_count)
    id2label = {i: l for l, i in label2id.items()}

    train_rows = filter_and_encode(train_rows_raw, label2id, "train")
    val_rows = filter_and_encode(val_rows_raw, label2id, "val")
    test_rows = filter_and_encode(test_rows_raw, label2id, "test")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # No fixed padding here: DataCollatorWithPadding pads each batch only to its longest
    # example. Same results (padding is masked), several times faster on CPU.
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True)
    val_ds = Dataset.from_list(val_rows).map(tokenize, batched=True)
    test_ds = Dataset.from_list(test_rows).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )

    class_weights = compute_class_weights(train_rows, len(labels))
    print(f"[WEIGHTS] {[round(w, 2) for w in class_weights.tolist()]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    training_args = TrainingArguments(
        output_dir=str(Path(args.out_dir) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        seed=args.seed,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(id2label),
        class_weights=class_weights,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n[TRAIN] starting...")
    trainer.train()

    print("\n[VAL] final metrics on validation set:")
    val_metrics = trainer.evaluate(val_ds)
    print(val_metrics)

    print("\n[TEST] held-out test set — never touched during training:")
    test_pred = trainer.predict(test_ds)
    test_preds = np.argmax(test_pred.predictions, axis=-1)
    test_labels = test_pred.label_ids
    print(classification_report(
        test_labels, test_preds,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    ))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\n[DONE] model saved to {out_dir / 'final'}")
    print(f"[DONE] label map saved to {out_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
