"""
Lightweight inference utilities for the hybrid TF-IDF + Transformer model.
"""

import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


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


class HybridPredictor:
    """
    Wraps vectorizer + SVD + label encoder + hybrid model for inference.
    """

    def __init__(self, artifacts_dir: str = "./artifacts"):
        self.artifacts_dir = artifacts_dir
        self.available = False
        self._load()

    def _load(self):
        try:
            cfg_path = os.path.join(self.artifacts_dir, "hybrid_config.json")
            with open(cfg_path) as f:
                cfg_dict = json.load(f)
            self.cfg = HybridConfig(**cfg_dict)
            self.vectorizer = load_pickle(os.path.join(self.artifacts_dir, "tfidf_vectorizer.pkl"))
            self.svd = load_pickle(os.path.join(self.artifacts_dir, "tfidf_svd.pkl"))
            self.label_encoder = load_pickle(os.path.join(self.artifacts_dir, "label_encoder.pkl"))
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.transformer_name)

            self.model = HybridClassifier(
                transformer_name=self.cfg.transformer_name,
                tfidf_dim=self.cfg.svd_components,
                tfidf_hidden=self.cfg.tfidf_hidden,
                num_labels=len(self.label_encoder.classes_),
                dropout=self.cfg.dropout,
            )
            state_dict = torch.load(os.path.join(self.artifacts_dir, "hybrid_model.pt"), map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.to(DEVICE)
            self.model.eval()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            print(f"[HybridPredictor] Skipping hybrid load: {exc}")
            self.available = False

    @torch.no_grad()
    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        if not self.available:
            return None
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.cfg.max_length,
        )
        tfidf = self.vectorizer.transform([text])
        tfidf_reduced = self.svd.transform(tfidf)
        batch = {
            "input_ids": tokens["input_ids"].to(DEVICE),
            "attention_mask": tokens["attention_mask"].to(DEVICE),
            "tfidf_features": torch.tensor(tfidf_reduced, dtype=torch.float32).to(DEVICE),
        }
        logits = self.model(**batch)["logits"]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return self.label_encoder.inverse_transform([pred_idx])[0], float(probs[pred_idx])
