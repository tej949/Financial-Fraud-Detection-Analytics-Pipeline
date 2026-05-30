"""
Predict with the best trained model on 5 sample transactions
using top feature importances

Author: Emmanuel Daniel Chonza
"""

import pandas as pd
import numpy as np
import joblib
import json
import pickle
import os

# ----------------------- Config ----------------------
MODEL_PATH = "/Users/emmanueldanielchonza/Documents/DS-final-project/best_model_RandomForest.joblib"
FEATURES_PKL_PATH = "/Users/emmanueldanielchonza/Documents/DS-final-project/top_n_feature_importances.pkl"
FEATURES_JSON_PATH = "/Users/emmanueldanielchonza/Documents/DS-final-project/top_n_feature_importances.json"
DATA_PATH = "/Users/emmanueldanielchonza/Documents/DS-final-project/sampledcreditcard.csv"

id2label = {0: "not_fraud", 1: "fraud"}
label2id = {"not_fraud": 0, "fraud": 1}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------- Load model ----------------------
print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# -------------------- Load top features ----------------------
if os.path.exists(FEATURES_PKL_PATH):
    print(f"Loading top features from: {FEATURES_PKL_PATH}")
    with open(FEATURES_PKL_PATH, "rb") as f:
        top_features_series = pickle.load(f)
    if isinstance(top_features_series, pd.Series):
        top_features = top_features_series.index.tolist()
    elif isinstance(top_features_series, list):
        top_features = top_features_series
    else:
        raise ValueError("Unrecognized format in top_n_feature_importances.pkl")
elif os.path.exists(FEATURES_JSON_PATH):
    print(f"Loading top features from: {FEATURES_JSON_PATH}")
    with open(FEATURES_JSON_PATH, "r") as f:
        top_features = json.load(f)
else:
    raise FileNotFoundError("No top feature importance file found (.pkl or .json)")

print(f"Top features ({len(top_features)}): {top_features}")

# -------------------- Load data ----------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# -------------------- Sample 5 transactions ----------------------
print("Sampling 5 transactions: 3 fraud & 2 not_fraud...")
fraud_samples = df[df['Class'] == 1].sample(3, random_state=RANDOM_STATE)
not_fraud_samples = df[df['Class'] == 0].sample(2, random_state=RANDOM_STATE)

df_samples = pd.concat([fraud_samples, not_fraud_samples]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nSample transactions:\n", df_samples[["Time", "Amount", "Class"]])

# -------------------- Prepare features ----------------------
# Ensure columns match model training order
trained_feature_order = list(model.feature_names_in_)
X_samples = df_samples[trained_feature_order]

# X_samples = df_samples[top_features]
y_true = df_samples['Class'].map(id2label)

# -------------------- Predict ----------------------
print("\nMaking predictions...")
y_pred = model.predict(X_samples)
y_pred_labels = [id2label[y] for y in y_pred]

# -------------------- Output ----------------------
print("\n📊 Predictions:")
result_df = df_samples.copy()
result_df["Predicted"] = y_pred_labels
result_df["Actual"] = y_true

result_df.to_csv("sample_predictions.csv", index=False)
print("Results saved to sample_predictions.csv")


print(result_df[["Time", "Amount", "Actual", "Predicted"]])
