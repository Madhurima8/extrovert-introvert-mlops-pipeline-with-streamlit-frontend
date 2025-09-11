# Extrovert vs Introvert Personality Classifier - MLOps Pipeline

This repository implements a complete MLOps pipeline for a binary classification task using the [Extrovert vs. Introvert Behavior Dataset](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data). The goal is to predict whether a person is an extrovert or introvert based on behavioral traits.

---

## 📁 Folder Structure

```bash
extrovert-mlops-pipeline/
├── backend/
│   ├── data/                          # Raw input dataset (.csv)
│   ├── logs/                          # Training and inference logs
│   ├── models/                        # Trained model and metadata
│   ├── src/                           # Backend source code
│   │   ├── api.py                     # FastAPI app definition
│   │   ├── model_training.py          # Training script
│   │   ├── model_inference.py         # Inference logic and logging
│   │   ├── schemas.py                 # Pydantic input schema
│   │   └── simulate_batch_inference.py # Batch test script
│   ├── Dockerfile                     # Backend Docker build
│   └── requirements.txt               # Backend Python dependencies
│
├── frontend/
│   ├── streamlit_app.py               # Streamlit frontend app
│   ├── Dockerfile                     # Frontend Docker build
│   └── requirements.txt               # Frontend dependencies
│
├── docker-compose.yml                 # Multi-container orchestration
└── README.md                          # You're here :)

```

---

## 🔍 Task Overview

* **Goal**: Classify user personality as `extrovert` or `introvert`.
* **Dataset**: Cleaned version of `personality_dataset.csv` with behavioral traits as features.
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

Visit Swagger UI at: http://127.0.0.1:8080/docs

Test with sample payload like:

```json
{
  "Time_spent_Alone": 11.0,
  "Stage_fear": "Yes",
  "Social_event_attendance": 0.0,
  "Going_outside": 2.0,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 4.0,
  "Post_frequency": 2.0
}
```
### 4. 💻 Run Streamlit Frontend (optional)

```bash
cd frontend
streamlit run streamlit_app.py
```
Open http://localhost:8501 to use the UI.

### 🧪 Simulate Batch Inference

```bash
python src/simulate_batch_inference.py
```

Runs inference on 5 random samples and prints prediction output.

---

## 🐳 Docker Usage

### 🔀 Docker Compose (Backend + Frontend)

```bash
docker-compose up --build
```
FastAPI backend: http://localhost:8080/docs
Streamlit frontend: http://localhost:8501

### 🐳 Backend Only

```bash
cd backend
docker build -t extrovert-backend .
docker run -p 8080:8080 extrovert-backend

```

Then access: http://localhost:8080/docs

---

## 📊 Metrics Logged

* During **training**:

  * Accuracy, Precision, Recall, F1, ROC AUC (train & eval splits)
  * `logs/train_eval_metrics.json`
  * `logs/train_eval_metrics_summary.csv`(for better readability and comparison of the models'performance)
* During **inference**:

  * Timestamp, latency, model name, input payload keys, prediction
  * `logs/inference_log.jsonl`

---

## 👩‍💻 Author

**Madhurima Khamroy** — Extrovert vs. Introvert MLOps Pipeline

