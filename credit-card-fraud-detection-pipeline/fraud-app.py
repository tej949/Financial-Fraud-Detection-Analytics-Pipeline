import streamlit as st
import pandas as pd
import joblib

# Title
st.title("💳 Credit Card Fraud Detection App")
st.markdown(
    """
This application lets you input a single transaction (**with 19 key features**)  
and predicts whether it is **fraudulent** or **not fraudulent** using the best trained **Random Forest model**.
"""
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model_RandomForest.joblib")

# Define the correct feature order
features_order = ['V17', 'V14', 'V10', 'V16', 'V12', 'V18', 'V11', 'V2', 'V7', 'V3', 'V9', 'V20', 'V1', 'V15', 'V22', 'V5', 'V21', 'V26', 'V13']

model = load_model()

# Create inputs
st.subheader("📝 Input Transaction Data")

input_data = {}
for feat in features_order:
    val = st.number_input(f"{feat}", value=0.0000, format="%.4f")
    input_data[feat] = val

# Submit button
if st.button("🔍 Predict"):
    st.subheader("Prediction Results")

    # Build DataFrame
    X_input = pd.DataFrame([input_data])
    X_input = X_input[features_order]  # ensure correct order

    try:
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        label_map = {0: "not_fraud", 1: "fraud"}
        result = label_map[pred]

        st.markdown(f"### **Prediction:** `{result.upper()}`")
        st.markdown(f"### 📊 **Probabilities:**")
        st.json({
            "not_fraud": round(proba[0], 4),
            "fraud": round(proba[1], 4)
        })

        st.markdown("### 🗒️ Input Data:")
        st.json(X_input.to_dict(orient="records")[0])

    except Exception as e:
        st.error(f"An error occurred: {e}")
