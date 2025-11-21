"""
Lightweight inference utilities for the hybrid TF-IDF + Transformer model.
"""

import json
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch

from hybrid_training import HybridClassifier, HybridConfig  # reuse definitions
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


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
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.cfg.max_length)
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
