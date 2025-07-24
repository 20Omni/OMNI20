

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# === Load dataset ===
df = pd.read_csv("final_task_dataset_balanced.csv")

# === Load all models & encoders ===
# Category
category_model = joblib.load("voting_ensemble_task_classifier.joblib")
category_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
category_encoder = joblib.load("task_label_encoder.joblib")



# Priority
priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_encoder = joblib.load("priority_label_encoder.pkl")

# User Assignment
user_model = joblib.load("user_assignment_xgb.pkl")
user_vectorizer = joblib.load("user_assignment_tfidf.pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder.pkl")
scaler = joblib.load("user_assignment_scaler.pkl")


# === Helper function to preprocess ===
def preprocess_features(task_description, deadline, priority, user_load=0, past_behavior=0.5, workload=0.5):
    deadline_days = (datetime.strptime(deadline, "%Y-%m-%d") - datetime.now()).days
    task_length = len(task_description.split())
    has_keyword_urgent = int("urgent" in task_description.lower())
    is_weekend_deadline = int(datetime.strptime(deadline, "%Y-%m-%d").weekday() >= 5)
    return {
        "task_description_clean": task_description.lower(),
        "deadline_days": deadline_days,
        "task_length": task_length,
        "priority_encoded": priority_encoder.transform([priority])[0],
        "user_current_load": user_load,
        "past_behavior_score": past_behavior,
        "workload": workload,
        "has_keyword_urgent": has_keyword_urgent,
        "is_weekend_deadline": is_weekend_deadline
    }

# === Prediction function ===
def predict_all(task_description, deadline, priority, user_load, past_behavior, workload):
    features = preprocess_features(task_description, deadline, priority, user_load, past_behavior, workload)
    features_df = pd.DataFrame([features])

    # Category Prediction
    category_tfidf = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(category_tfidf)[0]
    category_name = category_encoder.inverse_transform([category_pred])[0]

    # Priority Prediction
    priority_tfidf = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(priority_tfidf)[0]
    priority_name = priority_encoder.inverse_transform([priority_pred])[0]

    # User Assignment
    user_tfidf = user_vectorizer.transform([task_description])
    numeric_feats = features_df[['deadline_days', 'task_length', 'priority_encoded',
                                 'user_current_load', 'past_behavior_score',
                                 'workload', 'has_keyword_urgent',
                                 'is_weekend_deadline']].values
    user_input = np.hstack([user_tfidf.toarray(), numeric_feats])
    user_pred = user_model.predict(user_input)[0]
    assigned_user = user_encoder.inverse_transform([user_pred])[0]

    return category_name, priority_name, assigned_user, features['deadline_days']

# === Streamlit UI ===
st.title("🧠 AI-Powered Task Management Dashboard")

# Option: Manual entry or Select existing
mode = st.radio("Select input mode:", ["Enter Task Manually", "Choose from Existing Tasks"])

if mode == "Enter Task Manually":
    task_description = st.text_area("Task Description")
    deadline = st.date_input("Deadline", min_value=datetime.today()).strftime("%Y-%m-%d")
    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Urgent"])
else:
    selected_task = st.selectbox("Select a task", df['task_description'].tolist())
    row = df[df['task_description'] == selected_task].iloc[0]
    task_description = row['task_description']
    deadline = st.date_input("Deadline", datetime.strptime(row['deadline'], "%Y-%m-%d")).strftime("%Y-%m-%d")
    priority = row['priority']

# Extra inputs
user_load = st.slider("Current User Load", 0, 10, 0)
past_behavior = st.slider("Past Behavior Score", 0.0, 1.0, 0.5)
workload = st.slider("Workload Score", 0.0, 1.0, 0.5)

if st.button("Predict"):
    category_name, priority_name, assigned_user, days_left = predict_all(
        task_description, deadline, priority, user_load, past_behavior, workload
    )

    st.subheader("Predictions")
    st.write(f"**Category:** {category_name}")
    st.write(f"**Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Deadline:** {deadline} ({days_left} days left)")

    # Check if assigned user has worked in this category before
    user_worked = df[(df['assigned_user'] == assigned_user) & (df['category'] == category_name)]
    if not user_worked.empty:
        st.success(f"✅ Assigned user **{assigned_user}** has experience in **{category_name}**.")
    else:
        st.warning(f"⚠️ Assigned user **{assigned_user}** has no prior tasks in **{category_name}**.")



