from typing import Any
from fastapi import FastAPI
from src.model_inference import predict_with_metrics

app = FastAPI(title="Extrovert vs Introvert Classifier", version="1.0.0")

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict[str, Any]):
    """
    Send JSON with keys matching your CSV feature columns.
    Example:
    {
      "Social_Media_Usage": "Often",
      "Likes_Parties": "Yes",
      "Prefers_Solo_Activities": "No"
    }
    """
    return predict_with_metrics(payload)
