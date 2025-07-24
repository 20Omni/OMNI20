import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# === Load models & data ===
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# Load classification models
category_model = joblib.load("voting_ensemble_task_classifier.joblib")
category_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
category_label_encoder = joblib.load("task_label_encoder.joblib")

priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

# Load user assignment model
user_model = joblib.load("user_assignment_xgb (1).pkl")
user_tfidf = joblib.load("user_assignment_tfidf (1).pkl")
user_scaler = joblib.load("user_assignment_scaler (1).pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder (1).pkl")
user_feature_names = joblib.load("user_assignment_feature_names (1).pkl")

# === Prediction function ===
def predict_all(task_description, deadline):
    # ---- Category prediction ----
    cat_tfidf = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(cat_tfidf)[0]
    category_name = category_label_encoder.inverse_transform([category_pred])[0]

    # ---- Priority prediction ----
    prio_tfidf = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(prio_tfidf)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # ---- Deadline calculation ----
    today = datetime.now().date()
    deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
    days_left = max((deadline_date - today).days, 0)

    # ---- Build numeric features (aligned with training) ----
    numeric_features = pd.DataFrame([{
        'category_encoded': category_pred,
        'priority_encoded': priority_pred,
        'deadline_days': days_left,
        'has_keyword_urgent': int('urgent' in task_description.lower()),
        'task_length': len(task_description.split())
    }])

    # Ensure all features exist (from training)
    for col in user_feature_names:
        if col not in numeric_features.columns:
            numeric_features[col] = 0
    numeric_features = numeric_features[user_feature_names]  # Reorder to match scaler

    # ---- Combine TF-IDF + numeric ----
    text_tfidf = user_vectorizer.transform([task_description])
    numeric_scaled = user_scaler.transform(numeric_features)

    from scipy.sparse import hstack
    user_input = hstack([text_tfidf, numeric_scaled])

    # ---- Predict user ----
    user_pred = user_model.predict(user_input)[0]
    assigned_user = user_label_encoder.inverse_transform([user_pred])[0]

    return category_name, priority_name, assigned_user, days_left

# === Streamlit UI ===
st.title("AI Task Management Dashboard")

# Task selection (only from dataset)
task_description = st.selectbox("Select a Task", df['task_description_clean'].unique())

# Deadline input
deadline = st.date_input("Select Deadline", datetime.now()).strftime("%Y-%m-%d")

# Prediction
if st.button("Assign Task"):
    category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline)

    st.subheader("Prediction Results")
    st.write(f"**Category:** {category_name}")
    st.write(f"**Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Days Left for Deadline:** {days_left} days")
