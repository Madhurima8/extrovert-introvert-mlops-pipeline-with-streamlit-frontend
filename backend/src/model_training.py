from pathlib import Path
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time, json, joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "personality_dataset.csv"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
TRAIN_LOG = LOG_DIR / "train_eval_metrics.jsonl"
META_PATH = MODEL_DIR / "model_meta.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

#checks if xgboost is installed or not
HAS_XGB = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass


def select_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically find the target column in the DataFrame. Looks for columns containing both "introvert" and "extrovert",
    or common names like "personality", "type", "label", "target".
    """
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.lower().unique()           
        if any("introvert" in v for v in vals) and any("extrovert" in v for v in vals):
            return col
    common_names = ("personality", "type", "label", "target")
    lower_cols = {c.lower(): c for c in df.columns}
    for name in common_names:
        if name in lower_cols:
            return lower_cols[name]
    raise ValueError("Could not find target column. Please specify it manually.")

def binarize_labels(y: pd.Series, positive_hint: str = "extrovert") -> Tuple[np.ndarray, dict]:
    """
    Convert string labels into binary: 1 for the positive class, 0 for others.
    Uses 'positive_hint' to guess which label is positive.
    """
    
    y_str = y.astype(str).str.lower()
    unique_labels = y_str.unique().tolist()
    positive_label = None

    for label in unique_labels:
        if positive_hint in label:
            positive_label = label
            break

    # Fallback: if positive hint is not found in any column, uses last label alphabetically
    if positive_label is None:
        positive_label = sorted(unique_labels)[-1]
    
    # Maps positive to 1 and others to 0
    label_mapping = {}
    for label in unique_labels:
        if label == positive_label:
            label_mapping[label] = 1
        else:
            label_mapping[label] = 0

    # Apply the mapping to the label series
    y_binary = y_str.map(label_mapping).values
    return y_binary, label_mapping

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline for the features in X.
    Categorical features are imputed with the most frequent value and one-hot encoded.
    Numerical features are imputed with the median and standardized.
    """
    # Step 1: Separates column types
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Step 2: Defines preprocessing for categorical columns
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),   
        ("encoder", OneHotEncoder(handle_unknown="ignore"))     
    ])

    # Step 3: Defines preprocessing for numerical columns
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),          
        ("scaler", StandardScaler())                            
    ])

    # Step 4: Combines both pipelines into a column transformer
    preprocessor = ColumnTransformer([
        ("categorical", cat_pipeline, cat_cols),
        ("numerical", num_pipeline, num_cols)
    ],
    remainder="drop",                          # Drops any unused columns
    verbose_feature_names_out=False            # Keep output feature names clean
    )

    return preprocessor


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Computes classification metrics given true labels and predictions.
    If y_proba is provided, computes ROC AUC as well.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else None,
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Data file not found at {DATA}. Please ensure the dataset is available.")
    df = pd.read_csv(DATA)
    df.columns = [str(c).strip() for c in df.columns]
    target_col = select_target_column(df)
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    X_eval, X_infer, y_eval_raw, y_infer_raw = train_test_split(
        X_test, y_test_raw, test_size=0.5, stratify=y_test_raw, random_state=42
    )
    y_train, label_map = binarize_labels(pd.Series(y_train_raw))
    y_eval = pd.Series(y_eval_raw).astype(str).str.lower().map(label_map).values
    pre = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=300, random_state=42)
    }

    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
        
    best_name, best_f1, best_pipeline = None, -1, None
    logs = []
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", pre),
            ("model", model)
        ])
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_train = pipeline.predict(X_train)
        y_pred_eval = pipeline.predict(X_eval)
        y_proba_train = pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline[-1], "predict_proba") else None
        y_proba_eval = pipeline.predict_proba(X_eval)[:, 1] if hasattr(pipeline[-1], "predict_proba") else None
        metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
        metrics_eval = compute_metrics(y_eval, y_pred_eval, y_proba_eval)

        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": "extrovert_vs_introvert",
            "target_col": target_col,
            "label_mapping": label_map,
            "model": name,
            "train_time_sec": round(train_time, 4),
            "metrics_train": metrics_train,
            "metrics_eval": metrics_eval,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_eval)),    
        }
        logs.append(log_entry)

        #with open(TRAIN_LOG, "a", encoding="utf-8") as f:
            # f.write(json.dumps(log_entry) + "\n")
        
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

        if metrics_eval["f1"] > best_f1:
            best_name, best_f1, best_pipeline = name, metrics_eval["f1"], pipeline

    with open(TRAIN_LOG.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    joblib.dump(best_pipeline, MODEL_DIR / "classifier.joblib")

    meta_data = {
        "model_name": best_name,
        "feature_columns": X.columns.tolist(),
        "target_column": target_col,
        "label_mapping": label_map, 
        "classes_order": [0, 1],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    print(f"Best model: {best_name} (F1={best_f1:.3f})")

if __name__ == "__main__":
    main()

