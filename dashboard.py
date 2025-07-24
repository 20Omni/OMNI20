import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy.sparse import hstack

# === Load all models and encoders ===
try:
    category_model = joblib.load("voting_ensemble_task_classifier.joblib")
    category_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
    category_label_encoder = joblib.load("task_label_encoder.joblib")

    priority_model = joblib.load("priority_xgboost.pkl")
    priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
    priority_label_encoder = joblib.load("priority_label_encoder.pkl")

    
    user_model = joblib.load("user_assignment_xgb (1).pkl")
    user_vectorizer = joblib.load("user_assignment_tfidf (1).pkl")
    user_scaler = joblib.load("user_assignment_scaler (1).pkl")
    user_label_encoder = joblib.load("user_assignment_label_encoder (1).pkl")
    user_feature_names = joblib.load("user_assignment_feature_names (1).pkl")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# === Load dataset ===
df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# === Preprocessing helper ===
def preprocess_deadline(deadline_str):
    try:
        deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d")
        days_left = (deadline_date - datetime.today()).days
        return max(days_left, 0)
    except:
        return 0

# === Prediction helper ===
def predict_all(task_description, deadline):
    # ---- Category Prediction ----
    task_tfidf = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(task_tfidf)[0]
    category_name = category_label_encoder.inverse_transform([category_pred])[0]

    # ---- Priority Prediction ----
    priority_tfidf = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(priority_tfidf)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # ---- User Assignment ----
    days_left = preprocess_deadline(deadline)

    # Get matching row features if available
    task_row = df[df['Task Description'] == task_description].iloc[0]
    numeric_features = pd.DataFrame([{
        'category_encoded': task_row['category_encoded'],
        'priority_encoded': task_row['priority_encoded'],
        'deadline_days': days_left,
        'has_keyword_urgent': task_row['has_keyword_urgent'],
        'task_length': task_row['task_length'],
        'user_current_load': task_row['user_current_load'],
        'user_workload': task_row['user_workload'],
        'past_behavior_score': task_row['past_behavior_score'],
        'is_weekend_deadline': task_row['is_weekend_deadline'],
        'user_category_affinity': task_row['user_category_affinity'],
        'user_priority_affinity': task_row['user_priority_affinity']
    }], columns=user_feature_names)

    # Transform text
    text_tfidf = user_vectorizer.transform([task_description])
    numeric_scaled = user_scaler.transform(numeric_features)
    user_input = hstack([text_tfidf, numeric_scaled])

    user_pred = user_model.predict(user_input)[0]
    assigned_user = user_label_encoder.inverse_transform([user_pred])[0]

    return category_name, priority_name, assigned_user, days_left

# === Streamlit UI ===
st.set_page_config(page_title="AI Task Assignment Dashboard", layout="wide")
st.title("📌 AI Task Management System")
st.write("Select a task from the dataset to view predictions:")

# Dropdown to select a task
task_description = st.selectbox("Select a Task from Dataset", df['Task Description'].unique())
deadline = st.date_input("Select Deadline")

if st.button("Predict"):
    category_name, priority_name, assigned_user, days_left = predict_all(task_description, str(deadline))

    st.subheader("Prediction Results")
    st.write(f"**Task:** {task_description}")
    st.write(f"**Predicted Category:** {category_name}")
    st.write(f"**Predicted Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Days Left for Deadline:** {days_left}")
