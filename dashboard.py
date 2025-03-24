#!/usr/bin/env python
# coding: utf-8

# In[1]:


# predictive_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="centered")

st.title("⚙️ Predictive Maintenance: AI vs Threshold Alerts")
st.markdown("""
This demo compares traditional **threshold-based alerts** with an **AI model** for detecting machine failure.
""")

# -----------------------------
# Step 1: Simulate Data
# -----------------------------
def generate_data():
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'machine_id': np.random.choice(['M1', 'M2', 'M3'], size=n_samples),
        'vibration': np.random.normal(5, 1, n_samples),
        'temperature': np.random.normal(70, 5, n_samples),
        'current': np.random.normal(10, 2, n_samples),
    })

    # Label failure based on soft conditions
    data['failure'] = ((data['vibration'] > 6.0) & 
                       (data['temperature'] > 72) & 
                       (data['current'] > 11)).astype(int)

    # Inject borderline failure cases
    borderline = pd.DataFrame({
        'timestamp': pd.date_range('2024-03-15', periods=20, freq='H'),
        'machine_id': ['M1']*20,
        'vibration': np.random.normal(6.2, 0.1, 20),
        'temperature': np.random.normal(74, 0.5, 20),
        'current': np.random.normal(11.5, 0.2, 20),
        'failure': 1
    })

    data = pd.concat([data, borderline], ignore_index=True)

    # Inject noise
    noise_level = 0.3
    data['vibration'] += np.random.normal(0, noise_level, len(data))
    data['temperature'] += np.random.normal(0, noise_level * 2, len(data))
    data['current'] += np.random.normal(0, noise_level * 1.5, len(data))

    return data.sample(frac=1, random_state=42).reset_index(drop=True)

df = generate_data()

# -----------------------------
# Step 2: Train AI Model
# -----------------------------
X = df[['vibration', 'temperature', 'current']]
y = df['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Store predictions
df['ai_prediction'] = model.predict(X)
df['threshold_alert'] = df.apply(lambda row: 1 if (row['vibration'] > 6.5 or row['temperature'] > 75 or row['current'] > 12) else 0, axis=1)

# -----------------------------
# Step 3: Input Sliders
# -----------------------------
st.subheader("🔍 Test a Sample Reading")

vib = st.slider("Vibration Level", 3.0, 10.0, 5.0)
temp = st.slider("Motor Temperature (°C)", 60.0, 90.0, 70.0)
curr = st.slider("Current Draw (Amps)", 6.0, 16.0, 10.0)

input_df = pd.DataFrame([[vib, temp, curr]], columns=['vibration', 'temperature', 'current'])

# AI Prediction
ai_pred = model.predict(input_df)[0]
ai_proba = model.predict_proba(input_df)[0][1]

# Threshold Prediction
threshold_pred = 1 if (vib > 6.5 or temp > 75 or curr > 12) else 0

# -----------------------------
# Step 4: Output Comparison
# -----------------------------
st.subheader("📊 Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**AI Prediction**")
    if ai_pred == 1:
        st.error(f"⚠️ AI says: Failure likely ({ai_proba:.2%} risk)")
    else:
        st.success(f"✅ AI says: Machine is fine ({1 - ai_proba:.2%} confidence)")

with col2:
    st.markdown("**Threshold Alert**")
    if threshold_pred == 1:
        st.warning("⚠️ Threshold crossed – manual alert triggered")
    else:
        st.success("✅ No threshold breached")

# -----------------------------
# Step 5: Show Dataset Insights
# -----------------------------
st.subheader("🧠 Model Performance on Full Dataset")

col3, col4 = st.columns(2)

# AI Confusion Matrix
cm_ai = confusion_matrix(df['failure'], df['ai_prediction'])
cm_ai_df = pd.DataFrame(
    cm_ai,
    index=["Actual: No Failure", "Actual: Failure"],
    columns=["Predicted: No Failure", "Predicted: Failure"]
)

# Threshold Confusion Matrix
cm_thresh = confusion_matrix(df['failure'], df['threshold_alert'])
cm_thresh_df = pd.DataFrame(
    cm_thresh,
    index=["Actual: No Failure", "Actual: Failure"],
    columns=["Predicted: No Failure", "Predicted: Failure"]
)

with col3:
    st.markdown("### 🤖 AI Confusion Matrix")
    st.table(cm_ai_df)

    fig_ai, ax_ai = plt.subplots()
    sns.heatmap(cm_ai_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_ai)
    ax_ai.set_title("AI Prediction Heatmap")
    st.pyplot(fig_ai)

    # AI Metrics
    st.markdown("**AI Performance Metrics**")
    st.write(f"Precision: {precision_score(df['failure'], df['ai_prediction']):.2f}")
    st.write(f"Recall: {recall_score(df['failure'], df['ai_prediction']):.2f}")
    st.write(f"F1-Score: {f1_score(df['failure'], df['ai_prediction']):.2f}")

with col4:
    st.markdown("### 🧾 Threshold Confusion Matrix")
    st.table(cm_thresh_df)

    fig_thresh, ax_thresh = plt.subplots()
    sns.heatmap(cm_thresh_df, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax_thresh)
    ax_thresh.set_title("Threshold Alert Heatmap")
    st.pyplot(fig_thresh)

# Optional: Show data
with st.expander("📈 Show Sample Data"):
    st.dataframe(df[['vibration', 'temperature', 'current', 'failure', 'threshold_alert', 'ai_prediction']].head(20))
