"""
Hybrid TF-IDF + Transformer training script for the resume classifier.

This implements Task 1:
- Fit TF-IDF + TruncatedSVD on the training split.
- Train a hybrid classifier that concatenates DistilBERT/BERT pooled output
  with a small MLP over the reduced TF-IDF features.
- Evaluate against TF-IDF-only and Transformer-only baselines.

Outputs (saved under ./artifacts by default):
- tfidf_vectorizer.pkl
- tfidf_svd.pkl
- label_encoder.pkl
- hybrid_config.json
- hybrid_model.pt
- metrics.csv (per run)
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data classes and helpers
# --------------------------------------------------------------------------- #
@dataclass
class HybridConfig:
    transformer_name: str = "distilbert-base-uncased"
    max_length: int = 256
    tfidf_max_features: int = 20000
    svd_components: int = 128
    tfidf_hidden: int = 256
    dropout: float = 0.2
    lr_transformer: float = 2e-5
    lr_heads: float = 5e-4
    batch_size: int = 8
    epochs: int = 3
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    output_dir: str = "./artifacts"


class HybridResumeDataset(Dataset):
    """
    Yields tokenized text tensors + reduced TF-IDF features + labels.
    """

    def __init__(self, encodings, tfidf_vectors, labels):
        self.encodings = encodings
        self.tfidf_vectors = tfidf_vectors
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["tfidf_features"] = torch.tensor(self.tfidf_vectors[idx], dtype=torch.float32)
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class HybridClassifier(nn.Module):
    """
    Transformer CLS pooled output -> concat with TF-IDF MLP -> classifier.
    """

    def __init__(self, transformer_name: str, tfidf_dim: int, tfidf_hidden: int, num_labels: int, dropout: float):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden = self.transformer.config.hidden_size

        self.tfidf_mlp = nn.Sequential(
            nn.Linear(tfidf_dim, tfidf_hidden),
            nn.BatchNorm1d(tfidf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tfidf_hidden, tfidf_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden + tfidf_hidden, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, tfidf_features, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)

        tfidf_repr = self.tfidf_mlp(tfidf_features)
        concat = torch.cat([pooled, tfidf_repr], dim=1)
        logits = self.classifier(concat)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------------- #
# TF-IDF + SVD utilities
# --------------------------------------------------------------------------- #
def fit_tfidf_svd(train_texts: List[str], val_texts: List[str], test_texts: List[str], cfg: HybridConfig):
    vectorizer = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    svd = TruncatedSVD(n_components=cfg.svd_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train)
    X_val_reduced = svd.transform(X_val)
    X_test_reduced = svd.transform(X_test)

    return vectorizer, svd, X_train_reduced, X_val_reduced, X_test_reduced


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def tfidf_baseline(train_vectors, train_labels, val_vectors, val_labels, label_encoder) -> dict:
    model = LogisticRegression(max_iter=200)
    model.fit(train_vectors, train_labels)
    val_pred = model.predict(val_vectors)
    return {
        "accuracy": accuracy_score(val_labels, val_pred),
        "f1_macro": f1_score(val_labels, val_pred, average="macro"),
        "report": classification_report(val_labels, val_pred, target_names=label_encoder.classes_, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #
def train_hybrid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    cfg: HybridConfig,
):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "hybrid_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer_name)

    # TF-IDF + SVD
    vectorizer, svd, X_train_tfidf, X_val_tfidf, X_test_tfidf = fit_tfidf_svd(
        train_df["cleaned_resume"].tolist(), val_df["cleaned_resume"].tolist(), test_df["cleaned_resume"].tolist(), cfg
    )
    save_pickle(vectorizer, os.path.join(cfg.output_dir, "tfidf_vectorizer.pkl"))
    save_pickle(svd, os.path.join(cfg.output_dir, "tfidf_svd.pkl"))
    save_pickle(label_encoder, os.path.join(cfg.output_dir, "label_encoder.pkl"))

    train_enc = tokenizer(
        train_df["cleaned_resume"].tolist(), truncation=True, padding=True, max_length=cfg.max_length
    )
    val_enc = tokenizer(val_df["cleaned_resume"].tolist(), truncation=True, padding=True, max_length=cfg.max_length)
    test_enc = tokenizer(test_df["cleaned_resume"].tolist(), truncation=True, padding=True, max_length=cfg.max_length)

    train_dataset = HybridResumeDataset(train_enc, X_train_tfidf, train_df["label"].tolist())
    val_dataset = HybridResumeDataset(val_enc, X_val_tfidf, val_df["label"].tolist())
    test_dataset = HybridResumeDataset(test_enc, X_test_tfidf, test_df["label"].tolist())

    model = HybridClassifier(
        transformer_name=cfg.transformer_name,
        tfidf_dim=cfg.svd_components,
        tfidf_hidden=cfg.tfidf_hidden,
        num_labels=len(label_encoder.classes_),
        dropout=cfg.dropout,
    ).to(DEVICE)

    # Optimizer with separate lrs
    no_decay = ["bias", "LayerNorm.weight"]
    transformer_params, head_params = [], []
    for name, param in model.named_parameters():
        target = transformer_params if name.startswith("transformer") else head_params
        target.append(param)

    optimizer = AdamW(
        [
            {"params": transformer_params, "lr": cfg.lr_transformer},
            {"params": head_params, "lr": cfg.lr_heads},
        ],
        weight_decay=0.01,
    )

    total_steps = (len(train_dataset) // cfg.batch_size + 1) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={epoch_loss/len(train_loader):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "hybrid_model.pt"))

    test_acc, test_f1 = evaluate(model, test_loader)
    print(f"Test accuracy: {test_acc:.4f} | Test F1 (macro): {test_f1:.4f}")

    # Save metrics
    metrics_row = {
        "val_f1": best_val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "config": cfg.transformer_name,
    }
    metrics_path = os.path.join(cfg.output_dir, "metrics.csv")
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch["labels"].numpy()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch)["logits"]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train hybrid TF-IDF + Transformer classifier")
    parser.add_argument("--train", default="train.csv", help="Path to train split")
    parser.add_argument("--val", default="validation.csv", help="Path to validation split")
    parser.add_argument("--test", default="test.csv", help="Path to test split")
    parser.add_argument("--transformer", default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--output_dir", default="./artifacts", help="Where to save artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    for df in [train_df, val_df, test_df]:
        df["cleaned_resume"] = df["cleaned_resume"].fillna("")
        df["Category"] = df["Category"].fillna("Unknown")

    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["Category"])
    val_df["label"] = le.transform(val_df["Category"])
    test_df["label"] = le.transform(test_df["Category"])

    cfg = HybridConfig(
        transformer_name=args.transformer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    # TF-IDF-only baseline
    print(">>> Training TF-IDF + Logistic Regression baseline...")
    vectorizer = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_df["cleaned_resume"])
    X_val = vectorizer.transform(val_df["cleaned_resume"])
    baseline_metrics = tfidf_baseline(X_train, train_df["label"], X_val, val_df["label"], le)
    print(f"Baseline TF-IDF F1 (macro): {baseline_metrics['f1_macro']:.4f}")

    # Hybrid
    print(">>> Training Hybrid TF-IDF + Transformer model...")
    cfg.output_dir = args.output_dir
    cfg.transformer_name = args.transformer
    train_hybrid(train_df, val_df, test_df, le, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
