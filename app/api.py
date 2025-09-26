from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline
import torch

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

@app.on_event("startup")
def load_model():
    """
    Load the model from the Hugging Face Hub on application startup.
    This is a long-running operation and should only be done once.
    """
    global classifier
    print("Loading model from Hugging Face Hub...")
    
    # Set the device to CPU. For inference, a GPU is often not necessary and
    # this makes the application more portable.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")

    # Your model's ID on the Hugging Face Hub
    model_id = "madrazaldi/resume-classifier-distilbert-enhanced"
    
    classifier = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=device
    )
    print("✅ Model loaded successfully!")

# --- API Endpoints ---
class ResumeRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ResumeRequest):
    """
    Accepts resume text and returns the predicted category.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please wait.")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        prediction = classifier(request.text)[0]
        return PredictionResponse(
            category=prediction['label'],
            confidence=prediction['score']
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the request.")

# --- Static File Serving ---
# Mount the 'static' directory to serve the index.html file.
# The path is now relative to the location of this script inside the container.
app.mount("/", StaticFiles(directory="static", html=True), name="static")

