"""
Credit Card Fraud Detection Pipeline (All Features)
Author: Emmanuel Daniel Chonza

This pipeline trains and selects the best model
using *all features* (no feature selection).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================
# Config
# ======================
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Class"

# ======================
# Data Loading
# ======================
print("Loading data...")
df = pd.read_csv("/Users/emmanueldanielchonza/Documents/DS-final-project/sampledcreditcard.csv")

print(f"Dataset shape: {df.shape}")
print(df[TARGET_COL].value_counts())

# ======================
# Data Preprocessing
# ======================

# Convert 'Time' (seconds) → seconds in day, hour, PartOfDay
df['SecondsInDay'] = df['Time'] % 86400
df['Hour'] = (df['SecondsInDay'] // 3600).astype(int)

def get_part_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

df['PartOfDay'] = df['Hour'].apply(get_part_of_day)

# Ensure 'Amount' is numeric
df['Amount'] = df['Amount'].astype(float)

# ======================
# Feature Engineering
# ======================

# Separate features & target
X = df.drop([TARGET_COL, 'SecondsInDay', 'Hour'], axis=1)
y = df[TARGET_COL]

# One-hot encode PartOfDay
encoder = OneHotEncoder(sparse_output=False, drop='first')
part_of_day_encoded = encoder.fit_transform(X[['PartOfDay']])
part_of_day_cols = encoder.get_feature_names_out(['PartOfDay'])
X_encoded = pd.concat([
    X.drop(['Time', 'PartOfDay'], axis=1).reset_index(drop=True),
    pd.DataFrame(part_of_day_encoded, columns=part_of_day_cols)
], axis=1)

# ======================
# Scaling
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# ======================
# SMOTE
# ======================
print("Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {np.bincount(y_train_sm)}")

# ======================
# Model Training & Evaluation
# ======================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    results[name] = {
        "model": model,
        "f1_score": report['1']['f1-score'],
        "roc_auc": auc
    }
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {auc:.4f}")
    print(f"F1-Score (Fraud): {report['1']['f1-score']:.4f}")

# ======================
# Select Best Model (by F1-score of fraud class)
# ======================
best_model_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_model_name]['model']
print(f"\nBest model: {best_model_name} with F1-score: {results[best_model_name]['f1_score']:.4f}")

# ======================
# Save Artifacts
# ======================
joblib.dump(best_model, f"best_model_all_features_{best_model_name}.joblib")
joblib.dump(scaler, "scaler_all_features.joblib")
joblib.dump(encoder, "partofday_encoder_all_features.joblib")

print(f"Saved best model as best_model_all_features_{best_model_name}.joblib")
print("Saved scaler as scaler_all_features.joblib")
print("Saved PartOfDay encoder as partofday_encoder_all_features.joblib")
