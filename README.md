---
title: Resume Classifier
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# End-to-End Resume Classifier

This project now covers:
- **Task 1:** Hybrid TF-IDF + SVD features concatenated with a Transformer head.
- **Task 2:** Deep-learning OCR (TrOCR) → Transformer classifier.
- **Baseline transformer:** The original DistilBERT model from the Hugging Face Hub (kept for compatibility/fallback).

## Quickstart (local, no Docker)

1) **Clone + venv**
```
git clone https://github.com/madrazaldi/resume-bert-project.git
cd resume-bert-project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2) **Install deps**
```
pip install -r app/requirements.txt
pip install matplotlib seaborn nltk  # for preprocessing plots
```

3) **Dataset**
- Download `Resume.csv` (Kaggle resume dataset) and place it in the repo root.

4) **Preprocess (creates splits)**
```
python eda_and_preprocessing.py
```
Outputs: `cleaned_resumes.csv`, `train.csv`, `validation.csv`, `test.csv` (+ EDA PNGs).

5) **Train Hybrid (Task 1)**
```
python hybrid_training.py --train train.csv --val validation.csv --test test.csv \
  --transformer distilbert-base-uncased --output_dir artifacts
```
Outputs in `artifacts/`: `tfidf_vectorizer.pkl`, `tfidf_svd.pkl`, `label_encoder.pkl`, `hybrid_model.pt`, `hybrid_config.json`, `metrics.csv`.

6) **Run the app (serves Task 1 + Task 2)**
```
python -m uvicorn app.api:app --reload
```
Open `http://localhost:8000`:
- **Text mode** → calls `/predict/hybrid` (uses hybrid if `artifacts/` exists, else baseline transformer from HF).
- **Image mode** → calls `/predict/ocr` (TrOCR OCR → classifier; prefers hybrid, falls back to baseline).
- Baseline endpoint remains at `/predict`.

## Docker (optional, app only)
If you prefer a container to serve the already-trained models:
```
docker-compose build --no-cache
docker-compose up
```
The container will load `artifacts/` if present; otherwise it uses the baseline from Hugging Face Hub.

## API reference
- `POST /predict` — transformer-only baseline (HF Hub).
- `POST /predict/hybrid` — hybrid (TF-IDF + Transformer) if artifacts exist; otherwise baseline.
- `POST /predict/ocr` — image upload (PNG/JPG); TrOCR OCR → classifier; uses hybrid if available.

## Legacy sweep (optional)
`resume_classifier_full_sweep.py` runs the original transformer-only experiments (configs 1–6, including the DistilBERT “enhanced” baseline). Useful for comparison but not required for the hybrid/OCR pipelines.
