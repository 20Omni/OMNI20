import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# === Load models & preprocessing ===
category_model = joblib.load("voting_ensemble_task_classifier.joblib")
category_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
category_label_encoder = joblib.load("task_label_encoder.joblib")

priority_model = joblib.load("priority_xgboos.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

user_model = joblib.load("user_assignment_xgb (1).pkl")
user_tfidf = joblib.load("user_assignment_tfidf (1).pkl")
user_scaler = joblib.load("user_assignment_scaler (1).pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder (1).pkl")
user_feature_names = joblib.load("user_assignment_feature_names (1).pkl")

df = pd.read_csv("nlp_cleaned_task_dataset.csv")

# === Prediction function ===
def predict_all(task_description, deadline):
    # Category prediction
    task_vec = category_vectorizer.transform([task_description])
    category_pred = category_model.predict(task_vec)[0]
    category_name = category_label_encoder.inverse_transform([category_pred])[0]

    # Priority prediction
    priority_vec = priority_vectorizer.transform([task_description])
    priority_pred = priority_model.predict(priority_vec)[0]
    priority_name = priority_label_encoder.inverse_transform([priority_pred])[0]

    # Build numeric features for user assignment
    today = datetime.now().date()
    deadline_date = datetime.strptime(deadline, "%Y-%m-%d").date()
    days_left = max((deadline_date - today).days, 0)

    priority_encoded = df[df['Priority'] == priority_name]['priority_encoded'].mode()[0]
    category_encoded = df[df['category'] == category_name]['category_encoded'].mode()[0]

    numeric_features = pd.DataFrame([{
        'category_encoded': category_encoded,
        'priority_encoded': priority_encoded,
        'deadline_days': days_left,
        'has_keyword_urgent': int('urgent' in task_description.lower()),
        'task_length': len(task_description.split())
    }])

    # Ensure feature order
    numeric_features = numeric_features.reindex(columns=[col for col in user_feature_names if col in numeric_features.columns], fill_value=0)
    numeric_scaled = user_scaler.transform(numeric_features)

    # Text for user assignment
    user_text_vec = user_tfidf.transform([task_description])

    from scipy.sparse import hstack
    user_input = hstack([user_text_vec, numeric_scaled])

    user_pred = user_model.predict(user_input)[0]
    assigned_user = user_label_encoder.inverse_transform([user_pred])[0]

    return category_name, priority_name, assigned_user, days_left

# === Streamlit UI ===
st.title("AI-Powered Task Management Dashboard")

# Select task from dataset only
task_description = st.selectbox("Select a Task from Dataset", df['task_description_clean'].unique())
deadline = st.date_input("Select Deadline", datetime.now()).strftime("%Y-%m-%d")

if st.button("Assign Task"):
    category_name, priority_name, assigned_user, days_left = predict_all(task_description, deadline)
    st.subheader("Prediction Results")
    st.write(f"**Category:** {category_name}")
    st.write(f"**Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Days Left for Deadline:** {days_left}")



