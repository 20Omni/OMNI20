import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# === Load models and data ===
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# Task classification
task_model = joblib.load("voting_ensemble_task_classifier.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
task_label_encoder = joblib.load("task_label_encoder.joblib")

# Priority prediction
priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

# User assignment
user_model = joblib.load("user_assignment_xgb.pkl")
user_vectorizer = joblib.load("user_assignment_tfidf.pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder.pkl")
user_scaler = joblib.load("user_assignment_scaler.pkl")
feature_names = joblib.load("user_assignment_feature_names.pkl")

st.title("AI Task Management Dashboard")
st.write("Auto‑classify tasks, predict priority, and assign to the best user.")

# === Task selection or manual entry ===
mode = st.radio("Choose input mode", ["Select from dataset", "Write manually"])
if mode == "Select from dataset":
    task_description = st.selectbox("Select a Task from Dataset", df['task_description_clean'].unique())
    deadline = st.date_input("Deadline", datetime.now().date())
else:
    task_description = st.text_area("Enter Task Description")
    deadline = st.date_input("Deadline", datetime.now().date())

# === Predict category, priority, and user ===
def predict_all(task_description, deadline):
    # Category prediction
    task_vec = task_vectorizer.transform([task_description])
    category_pred = task_model.predict(task_vec)[0]
    category_name = task_label_encoder.inverse_transform([category_pred])[0]

    # Priority prediction
    priority_vec = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(priority_vec)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # Deadline days
    days_left = (deadline - datetime.now().date()).days

    # Numeric features for user model
    task_length = len(task_description.split())
    has_urgent = int("urgent" in task_description.lower() or "immediate" in task_description.lower())
    is_weekend_deadline = int(deadline.weekday() >= 5)

    numeric_features = pd.DataFrame([{
        'category_encoded': category_pred,
        'priority_encoded': priority_pred,
        'deadline_days': days_left,
        'has_keyword_urgent': has_urgent,
        'task_length': task_length,
        'user_current_load': 0,      # Dummy, model adjusts internally
        'user_workload': 0.5,        # Placeholder
        'past_behavior_score': 0.5,  # Placeholder
        'is_weekend_deadline': is_weekend_deadline
    }])

    # Scale & transform
    numeric_scaled = user_scaler.transform(numeric_features[feature_names])
    task_tfidf = user_vectorizer.transform([task_description])
    from scipy.sparse import hstack
    user_input = hstack([task_tfidf, numeric_scaled])

    # User prediction
    user_pred = user_model.predict(user_input)[0]
    assigned_user = user_label_encoder.inverse_transform([user_pred])[0]

    return category_name, priority_name, assigned_user, days_left

if st.button("Predict Task Assignment"):
    if task_description.strip():
        category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline)
        st.success(f"**Predicted Category:** {category_name}")
        st.success(f"**Predicted Priority:** {priority_name}")
        st.success(f"**Assigned User:** {assigned_user}")
        st.info(f"**Days Left for Deadline:** {days_left}")
    else:
        st.error("Please enter or select a task description.")
