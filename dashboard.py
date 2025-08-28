import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date
from scipy.sparse import hstack

# === Load all models and encoders ===
category_model = joblib.load("voting_ensemble_task_classifier.joblib")
category_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
category_label_encoder = joblib.load("task_label_encoder.joblib")

priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

user_model = joblib.load("user_assignment_xgb (1).pkl")
user_vectorizer = joblib.load("user_assignment_tfidf (1).pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder (1).pkl")
user_scaler = joblib.load("user_assignment_scaler (1).pkl")
user_feature_names = joblib.load("user_assignment_feature_names (1).pkl")

# === Load dataset (for dropdown) ===
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# === Helper function: Days left ===
def calculate_days_left(deadline_str):
    try:
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
        return max((deadline - datetime.now()).days, 0)
    except:
        return 0

# === Prediction function ===
def predict_all(task_description, deadline):
    # --- Category prediction ---
    X_cat = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(X_cat)[0]
    category_name = category_label_encoder.inverse_transform([category_pred])[0]

    # --- Priority prediction ---
    X_pri = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(X_pri)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # --- Base numeric features ---
    sample_row = df.sample(1).iloc[0]  # For filling context-dependent features
    base_numeric = {
        'category_encoded': category_pred,
        'priority_encoded': priority_pred,
        'deadline_days': calculate_days_left(deadline),
        'has_keyword_urgent': int("urgent" in task_description.lower()),
        'task_length': len(task_description.split()),
        'user_current_load': sample_row['user_current_load'],
        'user_workload': sample_row['user_workload'],
        'past_behavior_score': sample_row['past_behavior_score'],
        'is_weekend_deadline': 1 if datetime.strptime(deadline, "%Y-%m-%d").weekday() >= 5 else 0
    }
    numeric_features = pd.DataFrame([base_numeric])

    # --- Ensure all training columns exist ---
    for col in user_feature_names:
        if col not in numeric_features.columns:
            numeric_features[col] = 0  # Fill missing features with 0

    numeric_features = numeric_features[user_feature_names]  # Reorder

    # --- Scale numeric ---
    numeric_scaled = user_scaler.transform(numeric_features)

    # --- Combine text + numeric ---
    X_text_user = user_vectorizer.transform([task_description])
    X_user_final = hstack([X_text_user, numeric_scaled])

    # --- User prediction ---
    user_pred = user_model.predict(X_user_final)[0]
    assigned_user = user_label_encoder.inverse_transform([user_pred])[0]

    # --- Days left ---
    days_left = calculate_days_left(deadline)

    return category_name, priority_name, assigned_user, days_left

# === Streamlit UI ===
st.set_page_config(page_title="AI Task Management System", layout="wide")
st.title(" 📌 AI-Powered Task Management System")

# --- Task input ---
task_description = st.selectbox("Select a Task from Dataset", df['task_description_clean'].unique())

# Deadline should start from today
deadline = st.date_input("Deadline (YYYY-MM-DD)", min_value=date.today())
deadline_str = deadline.strftime("%Y-%m-%d")

# --- Predict button ---
if st.button("Assign Task"):
    category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline_str)

    st.subheader("Prediction Results:")
    st.write(f"**Task Category:** {category_name}")
    st.write(f"**Task Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Days Left until Deadline:** {days_left}")
