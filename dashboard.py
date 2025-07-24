import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, date
from scipy.sparse import hstack

# === Load all models ===
task_model = joblib.load("voting_ensemble_task_classifier.joblib")
task_vectorizer = joblib.load("task_tfidf_vectorizer.joblib")
task_label_encoder = joblib.load("task_label_encoder.joblib")

priority_model = joblib.load("priority_xgboost.pkl")
priority_vectorizer = joblib.load("priority_tfidf_vectorizer.pkl")
priority_label_encoder = joblib.load("priority_label_encoder.pkl")

user_model = joblib.load("user_assignment_xgb.pkl")
user_vectorizer = joblib.load("user_assignment_tfidf.pkl")
user_label_encoder = joblib.load("user_assignment_label_encoder.pkl")
scaler = joblib.load("user_assignment_scaler.pkl")
feature_names = joblib.load("user_assignment_feature_names.pkl")  # NEW: enforce feature order

# === Load dataset (for dropdown & feature mapping) ===
df = pd.read_csv("final_task_dataset_balanced.csv")

# === Prediction Function ===
def predict_all(task_desc, deadline, priority_encoded, user_load, past_behavior, workload):
    # --- Category Prediction ---
    cat_vec = task_vectorizer.transform([task_desc])
    category_pred = task_label_encoder.inverse_transform(task_model.predict(cat_vec))[0]
    
    # --- Priority Prediction ---
    pri_vec = priority_vectorizer.transform([task_desc])
    priority_pred = priority_label_encoder.inverse_transform(priority_model.predict(pri_vec))[0]

    # --- User Assignment ---
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

    # Align features with training order
    numeric_features = numeric_features.reindex(columns=feature_names, fill_value=0)
    numeric_scaled = scaler.transform(numeric_features)

    # Combine TF-IDF & numeric
    user_tfidf = user_vectorizer.transform([task_desc])
    user_input = hstack([user_tfidf, numeric_scaled])

    user_pred = user_label_encoder.inverse_transform(user_model.predict(user_input))[0]

    return category_pred, priority_pred, user_pred, days_left

# === Streamlit UI ===
st.title("AI-Powered Task Management Dashboard")

# Task input: Manual or Dropdown
input_mode = st.radio("Select Input Mode:", ["Manual Input", "Select from Dataset"])
if input_mode == "Manual Input":
    task_description = st.text_area("Task Description")
else:
    task_description = st.selectbox("Select a Task", df['task_description_clean'].unique())

# Deadline input
deadline = st.date_input("Select Deadline", min_value=date.today())

# Numeric inputs for user assignment
priority = st.selectbox("Priority Level (for model context)", sorted(df['priority_encoded'].unique()))
user_load = st.slider("Current User Load", 0, 50, 5)
past_behavior = st.slider("Past Behavior Score", 0.0, 1.0, 0.5)
workload = st.slider("User Workload Score", 0.0, 1.0, 0.5)

if st.button("Predict"):
    category_name, priority_name, assigned_user, days_left = predict_all(
        task_description, deadline, priority, user_load, past_behavior, workload
    )

    st.subheader("Prediction Results:")
    st.write(f"**Category:** {category_name}")
    st.write(f"**Predicted Priority:** {priority_name}")
    st.write(f"**Assigned User:** {assigned_user}")
    st.write(f"**Days Left Until Deadline:** {days_left}")

    # Extra check: has this user worked in this category?
    worked_before = df[(df['assigned_user'] == assigned_user) & (df['category'] == category_name)].shape[0] > 0
    st.write(f"**Has this user worked in this category before?** {'Yes' if worked_before else 'No'}")
