"""
Simple benchmarking script for the resume classifier models.

Usage:
  python benchmark_models.py --data test.csv --sample 500
  python benchmark_models.py --data test.csv --hybrid-only

Assumptions:
- The dataset has columns: `cleaned_resume` (text) and `label`.
- Hybrid artifacts are in ./artifacts (created by hybrid_training.py).
"""

import argparse
import random
import time
from typing import Iterable, Tuple

import pandas as pd
import torch
from transformers import pipeline

from app.hybrid_inference import HybridPredictor


MODEL_ID = "madrazaldi/resume-classifier-distilbert-enhanced"


def load_data(path: str, sample: int | None) -> Tuple[Iterable[str], Iterable[str]]:
    df = pd.read_csv(path)
    # Support both legacy "label" column and original "Category" column
    if "label" in df.columns:
        label_col = "label"
    elif "Category" in df.columns:
        label_col = "Category"
    else:
        raise ValueError("Dataset must contain 'cleaned_resume' and 'label' or 'Category' columns.")
    if "cleaned_resume" not in df.columns:
        raise ValueError("Dataset must contain 'cleaned_resume' column.")

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
    return df["cleaned_resume"].tolist(), df[label_col].tolist()


def benchmark_transformer(texts, labels) -> float:
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, device=device)
    start = time.time()
    # Batch call with truncation/padding to avoid 512+ token errors
    outputs = clf(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        batch_size=8,
    )
    preds = [o["label"] for o in outputs]
    elapsed = time.time() - start
    correct = sum(p == y for p, y in zip(preds, labels))
    acc = correct / len(labels)
    print(f"[Transformer] Accuracy: {acc:.4f} on {len(labels)} samples (time: {elapsed:.1f}s)")
    return acc


def benchmark_hybrid(texts, labels) -> float:
    predictor = HybridPredictor(artifacts_dir="./artifacts")
    if not predictor.available:
        print("[Hybrid] Artifacts not found; skipping hybrid benchmark.")
        return -1.0
    start = time.time()
    preds = []
    for t in texts:
        result = predictor.predict(t)
        preds.append(result[0] if result else None)
    elapsed = time.time() - start
    correct = sum(p == y for p, y in zip(preds, labels))
    acc = correct / len(labels)
    print(f"[Hybrid] Accuracy: {acc:.4f} on {len(labels)} samples (time: {elapsed:.1f}s)")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Benchmark resume classifier models.")
    parser.add_argument("--data", default="test.csv", help="Path to CSV with cleaned_resume,label columns")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for quick runs")
    parser.add_argument("--hybrid-only", action="store_true", help="Skip transformer benchmark")
    args = parser.parse_args()

    texts, labels = load_data(args.data, args.sample)
    print(f"Loaded {len(texts)} samples from {args.data}")

    if not args.hybrid_only:
        benchmark_transformer(texts, labels)
    benchmark_hybrid(texts, labels)


if __name__ == "__main__":
    main()
