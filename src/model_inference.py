from pathlib import Path
import joblib, json, time
import pandas as pd
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "classifier.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
INFER_LOG = ROOT / "logs" / "inference_log.jsonl"
INFER_LOG.parent.mkdir(parents=True, exist_ok=True)

class ModelInference:
    
    def __init__(self):
        """Load the trained model and metadata from disk.
        """
        self.pipeline = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.model_name = meta["model_name"]
        self.feature_columns = meta["feature_columns"]
        self.label_mapping = meta["label_mapping"]

    def payload_to_df(self, payload: dict[str, any]) -> pd.DataFrame:
        """
        Validate and convert input payload (dict) into a DataFrame with the expected feature columns.
        """
         # Step 1: Checks missing features
        missing_cols = [col for col in self.feature_columns if col not in payload]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Step 2: Build row dict with only expected features
        row = {col: payload[col] for col in self.feature_columns}

        # Step 3: Create 1-row DataFrame in expected column order
        df = pd.DataFrame([row], columns=self.feature_columns)

        return df

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Predict the class labels for the input DataFrame X.
        Logs inference time and input shape to INFER_LOG.
        """
        X = self.payload_to_df(features)
        if hasattr(self.pipeline[-1], "predict_proba"):
            proba = float(self.pipeline.predict_proba(X)[0, 1])  # Probability of the 2nd class in the first input
            pred = int(proba >= 0.5) #if probability of the 2nd class is >= 0.5, predict 1 else 0
        else:
            pred = int(self.pipeline.predict(X)[0])
            proba = None

        inv_label_mapping = {v: k for k, v in self.label_mapping.items()} # inv_label_mapping = {0: "introvert", 1: "extrovert"}
        return {
            "model_name": self.model_name,
            "predicted_label": inv_label_mapping.get(pred, str(pred)),
            "predicted_class": pred,
            "predicted_proba": proba
        }
    
def log_inference(log_entry: dict[str, Any]):
    """
    Log inference input data along with timestamp to INFER_LOG.
    """
    with open(INFER_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

_singleton = None
def get_predictor() -> ModelInference:
    global _singleton
    if _singleton is None:
        _singleton = ModelInference()
    return _singleton

def predict_with_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Wrapper function to get a singleton ModelInference instance and make predictions.
    Also logs inference time and input shape to INFER_LOG.
    """
    t0 = time.time()
    success, err = True, None
    try:
        result = get_predictor().predict(payload)
        return result
    except Exception as e:
        success, err = False, str(e)
        raise
    finally:
        latency_ms = (time.time() - t0) * 1000.0
        log_inference({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "success": success,
            "error": err,
            "latency_ms": round(latency_ms, 2),
            "payload_keys": sorted(list(payload.keys()))
        })