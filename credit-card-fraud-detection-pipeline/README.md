# Credit Card Fraud Detection Pipeline

## Project Overview
This project develops a robust machine learning pipeline for credit card fraud detection, leveraging a highly imbalanced dataset. Detecting fraudulent transactions is crucial for financial institutions to minimize monetary losses and protect customer assets. The pipeline trains multiple models, performs feature selection, addresses data imbalance, and selects the best-performing model for deployment.

## Key Challenges Addressed
- Handling class imbalance between fraud and non-fraud transactions

- Selecting the most predictive features for model efficiency and interpretability

- Preventing data leakage to ensure realistic model evaluation

- Fine-tuning and comparing multiple classification models for optimal fraud detection

## 📊 Dataset
- The dataset (sampled_creditcard.csv) is a sampled subset derived from the public Kaggle Credit Card Fraud Detection dataset. It contains:

- Anonymized features: V1, V2, ..., V28

- Transaction Amount

- Target variable Class:

  - `0` — Non-fraudulent transaction

  - `1` — Fraudulent transaction (minority class)

**Note:** The dataset is highly imbalanced with very few fraud cases, posing a challenge for traditional classification.

## Pipeline Workflow
The pipeline proceeds through the following stages:

1. Data Loading & Preprocessing
- Loads data from CSV.
- Engineers additional time-based features (Hour, PartOfDay).
- Standardizes feature values using StandardScaler.

2. Train-Test Split
- Stratified split to preserve class distribution.

3. Imbalance Handling
- Applies SMOTE (Synthetic Minority Oversampling Technique) only on training data to generate synthetic minority class samples, avoiding data leakage.

4. Feature Selection
- Fits a Random Forest classifier on training data to compute feature importances.
- Selects top 19 features (configurable via TOP_N_FEATURES).

5. Model Training & Evaluation
- Four classifiers are trained and evaluated on the selected features:

## Model Performance on Fraud Class

| Model                | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC AUC |
|----------------------|-------------------|-----------------|------------------|---------|
| Logistic Regression  | 0.04              | 0.50            | 0.07             | 0.5555  |
| Random Forest        | 1.00              | 0.75            | 0.86             | 0.8648  |
| Decision Tree        | 0.33              | 0.50            | 0.40             | 0.7494  |
| XGBoost              | 0.38              | 0.75            | 0.50             | 0.7771  |


- Metrics are focused on the fraud class (minority) since detecting frauds correctly is the priority.
- F1-Score balances precision and recall, critical due to imbalanced classes.
- ROC AUC measures overall discriminative ability.

6. Model Selection & Saving
- The Random Forest model achieved the best F1-Score (0.86) and ROC AUC (0.86) on the test set.
- This model, along with the scaler, encoder, and selected features list, is saved using joblib for later deployment.

## Results Summary
- Random Forest outperforms others by a significant margin in both fraud detection F1-score and ROC AUC, demonstrating better precision and recall trade-offs.
- Logistic Regression struggles with low precision despite decent recall.
- Decision Tree and XGBoost show moderate performance but lag behind Random Forest.
- The model comparison underscores the importance of tree-based ensemble methods in handling class imbalance and complex feature interactions in fraud detection.

## Usage & Requirements
- Python 3.12 environment recommended.
- Required libraries are listed in requirements.txt.

## Install dependencies via:
- bash: `pip install -r requirements.txt`
Running the Pipeline
- bash: `python fraud-detection-main.py`

## The script will:
- Load and preprocess data
- Train models with SMOTE-augmented training data
- Evaluate models and print classification reports
- Save the best model (best_model_RandomForest.joblib), scaler, encoder, and feature list
- Save a feature importance plot (feature_importances.png)
- Save top feature importances in pickle and JSON formats
- Run inference predictions

## Inference: Predicting Fraud on New Transactions
- Once the best model is trained and saved (best_model_RandomForest.joblib) along with the top feature importances (top_n_feature_importances.pkl or .json), you can use the inference script to predict fraud or not fraud on unseen transaction samples.

## Files Used
best_model_RandomForest.joblib — trained Random Forest model.

top_n_feature_importances.pkl — the 19 most important features used during training.

sampledcreditcard.csv — dataset containing transaction records.

## How to Run Inference
Run the prediction script:

bash:
``
python predictions.py
``

## The script will:
- Load the trained model and top feature names.
- Load the dataset.
- Randomly sample 5 transactions: 3 fraudulent and 2 non-fraudulent.
- Use the model to predict each transaction’s label.
- Print a summary table showing predictions:

### 📊 Sample Predictions
|   Time   | Amount  |   Actual    |  Predicted  |
|----------|---------|-------------|-------------|
|  41870.0 |   1.00  | fraud       | fraud       |
| 135855.0 |   2.18  | not_fraud   | not_fraud   |
|  85181.0 |   2.00  | fraud       | fraud       |
|  41505.0 | 364.19  | fraud       | fraud       |
|  36754.0 |  83.51  | not_fraud   | not_fraud   |


## Usage & Requirements
- Python 3.12 environment recommended.
- Required libraries are listed in requirements.txt.

## Install dependencies:
bash:
``
pip install -r requirements.txt
``
Run the training pipeline:
bash:
``
python fraud-detection-main.py
``

🎯 Streamlit Prediction App
This project also includes an interactive web application built with Streamlit to make predictions on new credit card transactions using the best-trained Random Forest model.

📄 App Overview
The app provides an intuitive UI where users can manually input values for the 19 most important features, and the model predicts whether the transaction is fraudulent or not fraudulent, along with the prediction probabilities.

🔷 Features Used
The prediction is based on the following 19 features (in order):

``
V17, V14, V10, V16, V12, V18, V11, V2, V7, V3, V9, V20, V1, V15, V22, V5, V21, V26, V13
``
## ️ How to Run the App
1. Install the dependencies (if not already):

bash:
``
pip install -r requirements.txt
``
2. Run the Streamlit app:

bash:
``
streamlit run fraud-app.py
``
3. Open the link that appears in the terminal in your browser (e.g., http://localhost:8501/).

## Using the App
- The app displays a form with input fields for each of the 19 features in the correct order.
- Enter the numerical values for each feature. Default value for each field is 0.0000.
- After filling in the fields, click the "🔍 Predict" button.

## The app will display:
- The predicted class: FRAUD or NOT_FRAUD.
- The prediction probabilities for each class.
- The input data you submitted, displayed in JSON format for easy reference.

## Example Output

 Prediction: FRAUD

📊 Probabilities:
{
  "not_fraud": 0.2150,
  "fraud": 0.7850
}

🗒️ Input Data:
{
  "V17": 12.0000,
  "V14": 78.0000,
  "V10": 0.1000,
  ...
}

## Notes
- The model expects inputs in the exact order shown above; the app automatically ensures this.
- If you encounter errors about feature names or order, make sure your App.py and trained model are in sync with the specified feature list.


## Contact & Contribution
- Developed by Emmanuel Daniel Chonza
- For questions or collaborations, please reach out.

**Note:** This pipeline is designed for experimentation and prototyping. Further tuning, cross-validation, and deployment considerations are recommended before production use.