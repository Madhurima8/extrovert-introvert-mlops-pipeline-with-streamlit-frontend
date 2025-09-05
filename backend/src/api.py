from typing import Any
from fastapi import FastAPI
from src.model_inference import predict_with_metrics
from src.schemas import PersonalityFeatures

# Initialize FastAPI app
app = FastAPI(title="Extrovert vs Introvert Classifier", version="1.0.0")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(payload: PersonalityFeatures):
    return predict_with_metrics(payload.dict())
