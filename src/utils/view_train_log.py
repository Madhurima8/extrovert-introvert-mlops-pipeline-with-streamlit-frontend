import json
from pathlib import Path
import pandas as pd

# Define the root and log path
ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = ROOT / "logs" / "train_eval_metrics.json"

# Load the JSON log
with open(LOG_PATH, "r", encoding="utf-8") as f:
    logs = json.load(f)

# Prepare rows for DataFrame
rows = []
for entry in logs:
    row = {
        "Model": entry["model"],
        "Train Time (s)": entry["train_time_sec"],
        
        # Training metrics
        "Train Acc": entry["metrics_train"]["accuracy"],
        "Train Precision": entry["metrics_train"]["precision"],
        "Train Recall": entry["metrics_train"]["recall"],
        "Train F1": entry["metrics_train"]["f1"],
        "Train ROC AUC": entry["metrics_train"]["roc_auc"],
        
        # Eval metrics
        "Eval Acc": entry["metrics_eval"]["accuracy"],
        "Eval Precision": entry["metrics_eval"]["precision"],
        "Eval Recall": entry["metrics_eval"]["recall"],
        "Eval F1": entry["metrics_eval"]["f1"],
        "Eval ROC AUC": entry["metrics_eval"]["roc_auc"],
    }
    rows.append(row)

# Create a DataFrame
df = pd.DataFrame(rows)

# Sort by evaluation F1 score (or any other metric if preferred)
df = df.sort_values(by="Eval F1", ascending=False)

# Display as a table
print("\n=== Training Metrics Summary ===\n")
print(df.to_string(index=False))

# Optional: Save to CSV for reference
output_csv = ROOT / "logs" / "train_eval_metrics_summary.csv"
df.to_csv(output_csv, index=False)
print(f"\nCSV summary saved to: {output_csv}")
