import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# === Load dataset ===
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# === Load Models ===
category_model = joblib.load("voting_ensemble_task_classifier.joblib")
category_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
category_label_encoder = joblib.load("task_label_encoder.joblib")

priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

user_model = joblib.load("user_assignment_xgb.pkl")
user_vectorizer = joblib.load("user_assignment_tfidf.pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder.pkl")
scaler = joblib.load("user_assignment_scaler.pkl")
feature_names = joblib.load("user_assignment_feature_names.pkl")

# === Helper: Days left ===
def calculate_days_left(deadline):
    return (deadline - datetime.now().date()).days

# === Predict function ===
def predict_all(task_description, deadline):
    # ---- CATEGORY PREDICTION ----
    task_tfidf = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(task_tfidf)[0]
    category_name = category_label_encoder.inverse_transform([category_pred])[0]

    # ---- PRIORITY PREDICTION ----
    priority_tfidf = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(priority_tfidf)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # ---- USER ASSIGNMENT ----
    # Extract numeric features from dataset (mean values for unseen task)
    category_encoded = df[df['category'] == category_name]['category_encoded'].mode()[0]
    priority_encoded = df[df['Priority'] == priority_name]['priority_encoded'].mode()[0]
    deadline_days = calculate_days_left(deadline)
    has_keyword_urgent = int("urgent" in task_description.lower())
    task_length = len(task_description.split())
    user_current_load = df['user_current_load'].mean()
    user_workload = df['user_workload'].mean()
    past_behavior_score = df['past_behavior_score'].mean()
    is_weekend_deadline = int(deadline.weekday() >= 5)

    numeric_features = pd.DataFrame([[
        category_encoded, priority_encoded, deadline_days, has_keyword_urgent, task_length,
        user_current_load, user_workload, past_behavior_score, is_weekend_deadline
    ]], columns=feature_names)

    numeric_scaled = scaler.transform(numeric_features)
    user_tfidf = user_vectorizer.transform([task_description])
    user_input = np.hstack([user_tfidf.toarray(), numeric_scaled])
    assigned_user = user_model.predict(user_input)[0]
    assigned_user_name = user_label_encoder.inverse_transform([assigned_user])[0]

    return category_name, priority_name, assigned_user_name, deadline_days

# === Streamlit UI ===
st.title("AI Task Management Dashboard")

input_mode = st.radio("Select Input Mode", ["Manual Entry", "Choose from Dataset"])

if input_mode == "Manual Entry":
    task_description = st.text_area("Enter Task Description")
else:
    task_description = st.selectbox("Select a Task from Dataset", df['task_description_clean'].unique())

deadline = st.date_input("Set Deadline")

if st.button("Predict & Assign Task"):
    if task_description.strip() == "":
        st.warning("Please enter or select a task.")
    else:
        category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline)
        st.subheader("Prediction Results")
        st.write(f"**Category:** {category_name}")
        st.write(f"**Predicted Priority:** {priority_name}")
        st.write(f"**Assigned User:** {assigned_user}")
        st.write(f"**Days Left for Deadline:** {days_left}")
