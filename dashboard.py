import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from scipy.sparse import hstack

# === Load Models ===
task_model = joblib.load("voting_ensemble_task_classifier.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
task_label_encoder = joblib.load("task_label_encoder.joblib")

priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

user_model = joblib.load("user_assignment_xgb.pkl")
user_vectorizer = joblib.load("user_assignment_tfidf.pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder.pkl")
user_scaler = joblib.load("user_assignment_scaler.pkl")
user_feature_names = joblib.load("user_assignment_feature_names.pkl")

# === Load Dataset ===
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# === Utility: Calculate days left ===
def calculate_days_left(deadline):
    today = datetime.today().date()
    return (deadline - today).days

# === Predict Function ===
def predict_all(task_description, deadline):
    # ---- CATEGORY ----
    task_tfidf = task_vectorizer.transform([task_description])
    category_pred = task_model.predict(task_tfidf)[0]
    category_name = task_label_encoder.inverse_transform([category_pred])[0]

    # ---- PRIORITY ----
    priority_tfidf = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(priority_tfidf)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # ---- USER ASSIGNMENT ----
    deadline_days = calculate_days_left(deadline)
    has_keyword_urgent = int(any(word in task_description.lower() for word in ["urgent", "immediate", "asap"]))
    task_length = len(task_description.split())
    is_weekend_deadline = int(deadline.weekday() >= 5)

    # Get all users
    all_users = user_label_encoder.classes_
    user_scores = {}

    for user in all_users:
        # Get user stats from dataset
        user_data = df[df['assigned_user'] == user]
        user_current_load = len(user_data)
        user_workload = user_data['user_workload'].mean() if not user_data.empty else df['user_workload'].mean()
        past_behavior_score = user_data['past_behavior_score'].mean() if not user_data.empty else df['past_behavior_score'].mean()

        # Numeric features
        numeric_features = pd.DataFrame([[
            category_pred, priority_pred, deadline_days,
            has_keyword_urgent, task_length, user_current_load,
            user_workload, past_behavior_score, is_weekend_deadline
        ]], columns=user_feature_names)

        numeric_scaled = user_scaler.transform(numeric_features)
        text_tfidf = user_vectorizer.transform([task_description])
        user_input = hstack([text_tfidf, numeric_scaled])

        # Predict probability
        prob = user_model.predict_proba(user_input)[0]
        user_index = np.where(all_users == user)[0][0]
        user_scores[user] = prob[user_index]

    # Assign to user with highest probability
    assigned_user = max(user_scores, key=user_scores.get)
    return category_name, priority_name, assigned_user, deadline_days

# === Streamlit UI ===
st.title("AI Task Assignment Dashboard")

option = st.radio("Choose Input Method", ["Manual Entry", "Select from Dataset"])

if option == "Manual Entry":
    task_description = st.text_area("Task Description")
    deadline = st.date_input("Deadline")
else:
    task_description = st.selectbox("Select a Task", df['task_description_clean'].unique())
    deadline = st.date_input("Deadline", datetime.today())

if st.button("Assign Task"):
    if task_description.strip():
        category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline)
        st.success(f"**Category:** {category_name}")
        st.success(f"**Priority:** {priority_name}")
        st.success(f"**Assigned User:** {assigned_user}")
        st.info(f"**Days left for deadline:** {days_left}")
    else:
        st.warning("Please enter or select a task description.")

