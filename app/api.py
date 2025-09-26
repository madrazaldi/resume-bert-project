from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# --- Configuration ---
# The model ID from the Hugging Face Hub for our fine-tuned model.
MODEL_ID = "madrazaldi/resume-classifier-distilbert-enhanced"

# --- Pydantic Model for Request Body ---
class ResumeRequest(BaseModel):
    text: str

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Load Model and Create Pipeline ---
# We load the model and tokenizer once when the application starts.
# This is a critical optimization to avoid reloading the model on every request.
print("Loading model from Hugging Face Hub...")
try:
    # Use a pipeline for simplified inference
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        # Use CPU for deployment as it's more portable and sufficient for single predictions.
        # If you have a GPU server, you could change this to device=0.
        device=-1 
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

# --- API Endpoints ---
@app.post("/predict")
async def predict(request: ResumeRequest):
    """
    Receives resume text, runs it through the classification model,
    and returns the predicted category.
    """
    if classifier is None:
        return {"error": "Model is not available. Please check the server logs."}, 500

    if not request.text or not request.text.strip():
        return {"error": "Resume text cannot be empty."}, 400

    try:
        # The pipeline handles tokenization and inference for us.
        prediction = classifier(request.text)
        
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'ENGINEERING', 'score': 0.99...}]
        # We just need the label from the first result.
        predicted_category = prediction[0]['label']
        
        return {"predicted_category": predicted_category}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "An unexpected error occurred during prediction."}, 500

# --- Serve Static Frontend ---
# This mounts the 'static' directory, making files inside it accessible from the browser.
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main index.html file for the user interface.
    """
    # Construct the full path to the index.html file
    index_path = os.path.join("app/static", "index.html")
    try:
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend file not found</h1>", status_code=404)

