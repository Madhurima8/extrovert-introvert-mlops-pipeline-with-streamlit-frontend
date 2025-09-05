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
docker run -p 8080:8080 extrovert-mlops
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

## 🔁 Future Extensions

If developed into a production-grade system, we could expand it with:

### 🔍 Monitoring

To ensure model performance and reliability in production, consider implementing the following:

- **Input Distribution Monitoring**
  - Track changes in input feature distributions over time to detect data drift.
  - Tools: [Evidently](https://evidentlyai.com/), Prometheus + Grafana, custom Python scripts.

- **Prediction Logging**
  - Log inputs, outputs, latency, and model confidence scores for every inference.
  - Store logs in structured format (e.g., JSONL or database like PostgreSQL).
  - Tools: Python logging, ELK Stack (Elasticsearch, Logstash, Kibana).

- **Performance Dashboard**
  - Visualize key metrics like:
    - Prediction volume per hour/day
    - Inference latency
    - Confidence score distributions
    - Error trends (when true labels are available)
  - Tools: Grafana, Streamlit, Power BI, Metabase.

- **Alerting**
  - Trigger alerts on anomalies in:
    - Prediction confidence drift
    - Latency spikes or system failures
    - Unexpected input patterns
  - Tools: Prometheus Alertmanager, Sentry, custom Slack/email alerts.

---

### 🔄 Auto-Retraining

To keep the model up-to-date and resilient to drift or stale data, implement automated retraining workflows:

- **Retraining Triggers**
  - Conditions to initiate retraining:
    - Drop in validation performance (e.g., F1-score or AUC)
    - New labeled data becomes available
    - Model age exceeds a defined time window
  - Tools: [Evidently](https://evidentlyai.com/), custom performance monitors, Prometheus alerts.

- **Data Pipeline Integration**
  - Continuously collect and store inference data + feedback.
  - Version incoming datasets and maintain training history.
  - Tools: Airflow, DVC, Prefect, or plain cron + scripts.

- **Scheduled Retraining Jobs**
  - Use job schedulers or CI/CD pipelines to:
    - Run `model_training.py` automatically (e.g., weekly)
    - Compare new vs. old model metrics
    - Redeploy if new model is better
  - Tools: GitHub Actions, GitLab CI, Jenkins, cron jobs, Docker.

- **Model Versioning & Deployment**
  - Track model versions with performance metrics and training metadata.
  - Serve new models using versioned APIs or containerized services.
  - Roll back if new model underperforms.
  - Tools: MLflow, Weights & Biases, DVC, Docker Hub, Kubernetes.



### 🔐 Security

* Add input validation for malformed inputs
* Use authentication for API endpoints

---

## 👩‍💻 Author

**Madhurima Khamroy** — Extrovert vs. Introvert MLOps Pipeline

