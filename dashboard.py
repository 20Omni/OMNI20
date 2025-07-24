# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy.sparse import hstack

# ==== Load dataset (for dropdown and feature lookup) ====
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# ==== Load Models ====
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

# ==== Helper: Days left ====
def calculate_days_left(deadline):
    deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
    return (deadline_date - datetime.now()).days

# ==== Prediction ====
def predict_all(task_description, deadline):
    # --- Category ---
    X_cat = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(X_cat)[0]
    category_name = category_label_encoder.inverse_transform([category_pred])[0]

    # --- Priority ---
    X_pri = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(X_pri)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # --- User Assignment ---
    sample_row = df.sample(1).iloc[0]
    numeric_features = pd.DataFrame([{
        'category_encoded': category_pred,
        'priority_encoded': priority_pred,
        'deadline_days': calculate_days_left(deadline),
        'has_keyword_urgent': int("urgent" in task_description.lower()),
        'task_length': len(task_description.split()),
        'user_current_load': sample_row['user_current_load'],
        'user_workload': sample_row['user_workload'],
        'past_behavior_score': sample_row['past_behavior_score'],
        'is_weekend_deadline': 1 if datetime.strptime(deadline, "%Y-%m-%d").weekday() >= 5 else 0
    }])
    numeric_scaled = user_scaler.transform(numeric_features[user_feature_names])
    X_text_user = user_vectorizer.transform([task_description])
    X_user_final = hstack([X_text_user, numeric_scaled])
    user_pred = user_model.predict(X_user_final)[0]
    assigned_user = user_label_encoder.inverse_transform([user_pred])[0]

    # --- Days Left ---
    days_left = calculate_days_left(deadline)

    return category_name, priority_name, assigned_user, days_left

# ==== Streamlit App ====
st.title("AI Task Management Dashboard")

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose Input Mode", ["Manual Task Entry", "Select from Dataset"])

if input_mode == "Manual Task Entry":
    task_description = st.text_area("Enter Task Description")
else:
    task_description = st.selectbox("Select a Task from Dataset", df['task_description_clean'].dropna().unique())

deadline = st.date_input("Select Deadline")
deadline_str = deadline.strftime("%Y-%m-%d")

if st.button("Assign Task"):
    if task_description.strip() == "":
        st.warning("Please enter or select a task description.")
    else:
        category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline_str)
        st.success(f"**Task:** {task_description}")
        st.write(f"**Predicted Category:** {category_name}")
        st.write(f"**Predicted Priority:** {priority_name}")
        st.write(f"**Assigned User:** {assigned_user}")
        st.write(f"**Deadline:** {deadline_str} ({days_left} days left)")
