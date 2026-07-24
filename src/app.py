import sys
import os
from contextlib import asynccontextmanager

# Add parent directory of 'src' to python path to resolve absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
from src import config
from src.preprocess import clean_text
from src.predict import predict_spam

# Define request schema
class EmailRequest(BaseModel):
    text: str

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Check model path
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {config.MODEL_SAVE_PATH}")
        # We do not raise error immediately to allow startup, but will return 503 on request
    else:
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
        print("Model loaded successfully.")
    yield
    # Clean up model resources on shutdown
    model = None

# Initialize FastAPI app with modern lifespan handler
app = FastAPI(
    title="Email Spam Classifier API",
    description="REST API and UI for email spam classification using TensorFlow",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to support cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    global model
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": config.MODEL_SAVE_PATH
    }

@app.post("/predict")
async def get_prediction(request: EmailRequest):
    global model
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is unavailable. Ensure it has been trained and saved to outputs/."
        )
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        label, confidence, prob = predict_spam(model, request.text)
        return {
            "text": request.text,
            "label": label,
            "confidence": float(confidence),
            "spam_probability": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend template index.html not found.")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    reload_flag = os.environ.get("RELOAD", "false").lower() in ["true", "1"]
    uvicorn.run("src.app:app", host="0.0.0.0", port=port, reload=reload_flag)

