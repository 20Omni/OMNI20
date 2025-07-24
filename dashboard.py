import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
from scipy.sparse import hstack
import numpy as np

# === Load all models ===
# Task classification (Voting Ensemble)
task_model = joblib.load("voting_ensemble_task_classifier.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
task_label_encoder = joblib.load("task_label_encoder.joblib")

# Priority prediction (XGBoost)
priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

# User assignment (XGBoost)
user_model = joblib.load("user_assignment_xgb.pkl")
user_vectorizer = joblib.load("user_assignment_tfidf.pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder.pkl")
scaler = joblib.load("user_assignment_scaler.pkl")

# Dataset for dropdowns and category encodings
df = pd.read_csv("final_task_dataset_balanced.csv")

# === Prediction function ===
def predict_all(task_desc, deadline, priority_encoded, user_load, past_behavior, workload):
    # === Category Prediction ===
    cat_vec = task_vectorizer.transform([task_desc])
    category_pred = task_label_encoder.inverse_transform(task_model.predict(cat_vec))[0]
    
    # === Priority Prediction ===
    pri_vec = priority_vectorizer.transform([task_desc])
    priority_pred = priority_label_encoder.inverse_transform(priority_model.predict(pri_vec))[0]

    # === User Assignment ===
    days_left = (deadline - datetime.today().date()).days
    numeric_features = pd.DataFrame([{
        'category_encoded': df[df['category'] == category_pred]['category_encoded'].mean(),
        'priority_encoded': df[df['priority'] == priority_pred]['priority_encoded'].mean(),
        'deadline_days': days_left,
        'has_keyword_urgent': int("urgent" in task_desc.lower()),
        'task_length': len(task_desc.split()),
        'user_current_load': user_load,
        'user_workload': workload,
        'past_behavior_score': past_behavior,
        'is_weekend_deadline': 1 if deadline.weekday() >= 5 else 0,
        'is_completed': 0
    }])

    numeric_scaled = scaler.transform(numeric_features)
    user_tfidf = user_vectorizer.transform([task_desc])
    user_input = hstack([user_tfidf, numeric_scaled])

    # Debug: check shapes
    st.write(f"DEBUG: User TFIDF shape: {user_tfidf.shape}, Numeric shape: {numeric_scaled.shape}, Combined: {user_input.shape}")

    user_pred = user_label_encoder.inverse_transform(user_model.predict(user_input))[0]

    return category_pred, priority_pred, user_pred, days_left

# === Streamlit UI ===
st.title("AI-Powered Task Management Dashboard")

mode = st.radio("Choose Input Mode:", ["Manual Input", "Select from Dataset"])

if mode == "Manual Input":
    task_description = st.text_area("Task Description:")
    deadline = st.date_input("Deadline:", min_value=datetime.today().date())
else:
    task_choice = st.selectbox("Select Task:", df['task_description'].unique())
    selected_task = df[df['task_description'] == task_choice].iloc[0]
    task_description = selected_task['task_description']
    deadline = st.date_input("Deadline:", min_value=datetime.today().date())

# User-specific fields
priority = st.selectbox("Priority:", ["Low", "Medium", "High"])
user_load = st.slider("Current User Load:", 0, 50, 5)
past_behavior = st.slider("Past Behavior Score:", 0.0, 1.0, 0.5)
workload = st.slider("User Workload:", 0.0, 1.0, 0.5)

if st.button("Assign Task"):
    category_name, priority_name, assigned_user, days_left = predict_all(
        task_description, deadline, priority, user_load, past_behavior, workload
    )

    st.subheader("Predictions:")
    st.write(f"**Category:** {category_name}")
    st.write(f"**Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Days Left Until Deadline:** {days_left}")

    # Check if assigned user has experience in this category
    user_category_experience = df[(df['assigned_user'] == assigned_user) & (df['category'] == category_name)]
    if not user_category_experience.empty:
        st.success(f"Assigned user **{assigned_user}** has experience with tasks in this category.")
    else:
        st.warning(f"Assigned user **{assigned_user}** has no prior tasks in this category.")

