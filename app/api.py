from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline
import torch

# Local modules live in the same directory when copied into the container
from hybrid_inference import HybridPredictor
from ocr_pipeline import OcrEngine

# --- App Setup ---
app = FastAPI(
    title="Resume Classification API",
    description="An API to classify resume text into one of 24 professional categories.",
    version="1.0"
)

# --- Model Loading ---
# This is a global variable that will hold the loaded model.
# The model is loaded once when the application starts.
classifier = None
hybrid_predictor = None
ocr_engine = None

@app.on_event("startup")
def load_model():
    """
    Load text classifier (transformer), optional hybrid head, and OCR engine.
    """
    global classifier, hybrid_predictor, ocr_engine
    print("Loading model from Hugging Face Hub...")
    
    # Set the device to CPU. For inference, a GPU is often not necessary and
    # this makes the application more portable.
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device set to use {device}")

    # Your model's ID on the Hugging Face Hub
    model_id = "madrazaldi/resume-classifier-distilbert-enhanced"
    
    classifier = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device
    )
    print("✅ Base transformer model loaded successfully!")

    # Optional hybrid branch (TF-IDF + Transformer) for Task 1
    hybrid_predictor = HybridPredictor(artifacts_dir="./artifacts")
    if hybrid_predictor.available:
        print("✅ Hybrid TF-IDF + Transformer model loaded.")
    else:
        print("⚠️  Hybrid artifacts not found; falling back to transformer-only.")

    # OCR (TrOCR) for Task 2
    ocr_engine = OcrEngine()
    if ocr_engine.available:
        print("✅ TrOCR OCR model loaded.")
    else:
        print("⚠️  OCR model unavailable; /predict/ocr will be disabled.")

# --- API Endpoints ---
class ResumeRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float
    used_hybrid: bool = False


class OcrPredictionResponse(BaseModel):
    text: str
    category: str
    confidence: float
    used_hybrid: bool = False

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ResumeRequest):
    """
    Transformer-only classification (original behavior).
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        prediction = classifier(request.text)[0]
        return PredictionResponse(
            category=prediction['label'],
            confidence=prediction['score'],
            used_hybrid=False
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")


@app.post("/predict/hybrid", response_model=PredictionResponse)
def predict_hybrid(request: ResumeRequest):
    """
    Uses TF-IDF + Transformer hybrid if available, otherwise defaults to transformer.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        if hybrid_predictor and hybrid_predictor.available:
            result = hybrid_predictor.predict(request.text)
            if result:
                label, score = result
                return PredictionResponse(category=label, confidence=score, used_hybrid=True)
        # fallback
        prediction = classifier(request.text)[0]
        return PredictionResponse(
            category=prediction["label"],
            confidence=prediction["score"],
            used_hybrid=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Hybrid prediction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")


@app.post("/predict/ocr", response_model=OcrPredictionResponse)
async def predict_ocr(file: UploadFile = File(...)):
    """
    Accepts an image, runs deep-learning OCR (TrOCR), then classifies the extracted text.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")
    if not ocr_engine or not ocr_engine.available:
        raise HTTPException(status_code=503, detail="OCR model is unavailable on this server.")

    try:
        content = await file.read()
        text = ocr_engine.image_to_text(content)
        if not text:
            raise HTTPException(status_code=400, detail="OCR failed to extract text.")

        if hybrid_predictor and hybrid_predictor.available:
            result = hybrid_predictor.predict(text)
            if result:
                label, score = result
                return OcrPredictionResponse(text=text, category=label, confidence=score, used_hybrid=True)

        prediction = classifier(text)[0]
        return OcrPredictionResponse(text=text, category=prediction["label"], confidence=prediction["score"], used_hybrid=False)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"OCR prediction error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")

# --- Static File Serving ---
# Resolve static directory relative to this file so it works both locally and in Docker
from pathlib import Path  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory not found at {STATIC_DIR}")

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
