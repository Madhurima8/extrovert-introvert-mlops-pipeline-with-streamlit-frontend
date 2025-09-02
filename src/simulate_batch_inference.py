from src.model_inference import predict_with_metrics
import pandas as pd

df = pd.read_csv("data/personality_dataset.csv").dropna()
target_col = "Personality"
X = df.drop(columns=[target_col])
y = df[target_col]

# Simulate 5 inference calls
for i, row in X.sample(5, random_state=42).iterrows():
    features = row.to_dict()
    output = predict_with_metrics(features)
    print(f"Input: {features}")
    print("Predicted Label:", output["predicted_label"])
    print("Predicted Class (0/1):", output["predicted_class"])
    print("Probability (if available):", output.get("predicted_proba"))
