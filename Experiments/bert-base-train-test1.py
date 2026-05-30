"""
BERT-based Binary Text Classification Pipeline for Fraud Detection
with Classification Report and Confusion Matrix Visualization

Author: Emmanuel Daniel Chonza
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import set_seed
import os
import yaml
import evaluate
from functools import cache  # Python 3.9+
from imblearn.over_sampling import RandomOverSampler
import random
from transformers import set_seed


# Setup seed for reproducibility
RANDOM_STATE = 42
set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# -------------------------------------- Setup requirements — environment & API keys ----------------------------
def load_api_keys():
    with open("huggingface_credentials.yml", "r") as f:
        huggingface_keys = yaml.safe_load(f)
    hf_key = huggingface_keys.get("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        raise ValueError("HUGGINGFACE_API_KEY missing in huggingface_credentials.yml")

    os.environ["HUGGINGFACE_API_KEY"] = hf_key
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    return hf_key

HUGGINGFACE_API_KEY = load_api_keys()

# --------------------------------------------- Config ------------------------------------------
MODEL_NAME = "bert-base-uncased"
RANDOM_STATE = 42
MAX_LEN = 128
BATCH_SIZE = 128
EPOCHS = 2
OUTPUT_DIR = "./bert_fraud_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------- Load Data ------------------------------------------
print("Loading data...")
df = pd.read_csv("/Users/emmanueldanielchonza/Documents/DS-final-project/sampledcreditcard.csv")

label2id = {"not_fraud": 0, "fraud": 1}
id2label = {0: "not_fraud", 1: "fraud"}

if 'text' in df.columns and 'label' in df.columns:
    print("Found 'text' and 'label' columns in dataset.")
else:
    print("No 'text' and 'label' columns found — generating them from tabular features.")
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column for the target variable.")
    feature_cols = [col for col in df.columns if col != 'Class']
    df['text'] = df[feature_cols].astype(str).agg(' '.join, axis=1)
    df['label'] = df['Class']
    df = df[['text', 'label']]

df['label'] = df['label'].map(id2label)

print("\nSample data:\n", df.sample(3))
print("\nLabel distribution:\n", df['label'].value_counts())

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=RANDOM_STATE)

# Perform Random OverSampler Technique
ros = RandomOverSampler(random_state=RANDOM_STATE)

X_train = train_df['text'].values.reshape(-1, 1)
y_train = train_df['label'].map(label2id).values

X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

train_df = pd.DataFrame({
    'text': X_train_resampled.flatten(),
    'label': y_train_resampled
})

train_df['label'] = train_df['label'].map(id2label)

# Optional: oversample minority class
# Optional: oversample minority class
fraud_df = train_df[train_df['label'] == 'fraud']
not_fraud_df = train_df[train_df['label'] == 'not_fraud']

# Uncomment below to apply oversampling
# train_df = pd.concat([not_fraud_df, fraud_df.sample(len(not_fraud_df), replace=True)])
# train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Print training labels
print("\nTraining label distribution after optional oversampling:")
print(train_df['label'].value_counts())

# --------------------------------------------- Tokenization ------------------------------------------------------
@cache
def load_tokenizer():
    print("Loading tokenizer...")
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_API_KEY)

tokenizer = load_tokenizer()

def tokenize_batch(texts):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )

class FraudDataset(TorchDataset):
    def __init__(self, df):
        labels_mapped = df['label'].map(label2id).values
        encodings = tokenize_batch(df['text'].tolist())
        self.encodings = encodings
        self.labels = torch.tensor(labels_mapped)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = FraudDataset(train_df)
test_dataset = FraudDataset(test_df)

# ----------------------------------------------- Weighted Model ----------------------------------------------------
print("\nLoading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    use_auth_token=HUGGINGFACE_API_KEY
)

# Compute weights inversely proportional to class frequency
class_counts = train_df['label'].value_counts()
class_fractions = class_counts / class_counts.sum()
# Flip so that fraud=1 weight corresponds to index 1
weights = torch.tensor(
    [class_fractions['fraud'], class_fractions['not_fraud']],
    dtype=torch.float32
).flip(0).to(device)

print(f"\nClass weights: {weights.cpu().numpy()}")


class WeightedModel(torch.nn.Module):
    def __init__(self, base_model, weights):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = CrossEntropyLoss(weight=weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return logits

model = WeightedModel(base_model, weights)

print("\nWeighted model ready")

# ---------------------------------------------- Metrics ------------------------------------------------
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = metric.compute(predictions=preds, references=labels, average='binary')
    return {"f1": f1['f1']}

# -------------------------------------------- Trainer ---------------------------------------------
# Create the training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=RANDOM_STATE,
    logging_steps=50,
    logging_dir=f"{OUTPUT_DIR}/logs"
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    # tokenizer=tokenizer,  # deprecated: remove
)

# ---------------------------------------------- Train ----------------------------------------------
print("Starting training...")
trainer.train()

# --------------------------------------------- Evaluate --------------------------------------------
results = trainer.evaluate()
print("\nFinal Evaluation Metrics:", results)

# -------------------------------------------- Predict ---------------------------------------------
print("\nPredicting on test set...")
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# ----------------------------------------- Classification Report ------------------------------------
print("\nClassification Report:")
report = classification_report(true_labels, pred_labels, target_names=list(label2id.keys()))
print(report)

# ----------------------------------------- Confusion Matrix ----------------------------------------
print("\nPlotting Confusion Matrix...")
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label2id.keys()))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - BERT Fraud Detection")
plt.show()

# ---------------------------------------- Save Model & Tokenizer -----------------------------------
model.base_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nBest model & tokenizer saved in {OUTPUT_DIR}")