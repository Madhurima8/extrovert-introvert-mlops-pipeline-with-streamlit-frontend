# Extrovert vs Introvert Personality Classifier - MLOps Pipeline

This repository implements a complete MLOps pipeline for a binary classification task using the [Extrovert vs. Introvert Behavior Dataset](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data). The goal is to predict whether a person is an extrovert or introvert based on behavioral traits.

---

## 📁 Folder Structure

```bash
extrovert-mlops-pipeline/
├── data/                          # Raw input dataset (.csv)
├── logs/                          # Training and inference logs
│   ├── train_eval_metrics.json    # Per-model training logs
│   ├── train_eval_metrics_summary.csv
│   └── inference_log.jsonl        # Inference requests logging
├── models/                        # Trained model and metadata
│   ├── classifier.joblib          # Final model
│   └── model_meta.json            # Metadata like label map, input features
├── src/                           # Source code
│   ├── api.py                     # FastAPI app definition
│   ├── model_training.py          # Training script (multi-model)
│   ├── model_inference.py         # Inference logic and logging
│   ├── schemas.py                 # Pydantic input validation schema
│   └── simulate_batch_inference.py # Batch testing script
├── Dockerfile                     # Docker build configuration
├── requirements.txt               # Python dependencies
└── README.md                      # You're here :)
```

---

## 🔍 Task Overview

* **Goal**: Classify user personality as `extrovert` or `introvert`.
* **Dataset**: Cleaned version of `personality_dataset.csv` (https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data) with behavioral traits as features.
* **Problem Type**: Binary Classification
* **Model Selection**: Trained and evaluated multiple models:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
  * (Optionally) XGBoost (if available)
* **Metrics Logged**: Accuracy, Precision, Recall, F1-score, ROC AUC

---

## ✅ Steps to Run the Project

### 1. 📦 Install dependencies

```bash
# Create virtual env
conda create -n extrovert-mlops python=3.11 -y
conda activate extrovert-mlops

# Install dependencies
pip install -r requirements.txt
```

### 2. 🏋️‍♀️ Train the Model

```bash
python src/model_training.py
```

This:

* Loads the dataset
* Preprocesses numeric & categorical features
* Trains multiple models and logs evaluation metrics
* Saves the best model as `classifier.joblib`
* Stores metadata like label mapping and features

### 3. 🚀 Launch the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

Visit Swagger UI at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Test with sample payload like:

```json
{
  "Time_spent_Alone": "Often",
  "Stage_fear": "Yes",
  "Social_event_attendance": "Sometimes",
  "Going_outside": "Rarely",
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": "Small",
  "Post_frequency": "Rarely"
}
```

### 4. 🧪 Simulate Batch Inference

```bash
python src/simulate_batch_inference.py
```

Runs inference on 5 random samples and prints prediction output.

---

## 🐳 Docker Usage

### Build Image

```bash
docker build -t extrovert-mlops .
```

### Run Container

```bash
docker run -p 8000:8000 extrovert-mlops
```

Then access: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Metrics Logged

* During **training**:

  * Accuracy, Precision, Recall, F1, ROC AUC (train & eval splits)
  * `logs/train_eval_metrics.json`
* During **inference**:

  * Timestamp, latency, model name, input payload keys, prediction
  * `logs/inference_log.jsonl`

---

## 🔁 Future Extensions

If developed into a production-grade system, we could expand it with:

### 🔍 Monitoring

* Track model drift / data distribution changes
* Visualize prediction confidence / input volume

### 🔄 Auto-Retraining

* Trigger training when:

  * Accuracy drops
  * Model age exceeds threshold
  * More labeled data is available
* Schedule via cron jobs or CI/CD pipeline

### 🔐 Security

* Add input validation for malformed inputs
* Use authentication for API endpoints

---

## 👩‍💻 Author

**Madhurima Khamroy** — Extrovert vs. Introvert MLOps Pipeline

---

## 📎 Resources

* [Dataset (Kaggle)](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data)
* [FastAPI Docs](https://fastapi.tiangolo.com/)
* [Docker Docs](https://docs.docker.com/)

---

> For any queries, suggestions or improvements, feel free to contribute or reach out!
