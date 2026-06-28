# ==========================================================
# Credit Card Fraud Detection Dashboard
# Author: Gulam Zohra
# ==========================================================

import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Load Dataset
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")

df = load_data()

# ==========================================================
# Load Model & Scaler
# ==========================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_detection_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Dataset",
            "Visualization",
            "Prediction",
            "Performance",
            "About"
        ],
        icons=[
            "house",
            "table",
            "bar-chart",
            "robot",
            "graph-up",
            "info-circle"
        ],
        menu_icon="shield-check",
        default_index=0
    )

# ==========================================================
# HOME PAGE
# ==========================================================

if selected == "Home":

    st.title("💳 Credit Card Fraud Detection System")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
This application detects fraudulent credit card transactions using Machine Learning.

### Features
- Data Analysis Dashboard
- Interactive Visualizations
- Fraud Prediction
- Model Performance
- Professional UI

The final deployed model is **Random Forest Classifier**.
""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", f"{len(df):,}")

    with col2:
        st.metric("Fraud Cases", f"{sum(df['Class']==1):,}")

    with col3:
        st.metric("Accuracy", "99.95%")


        # ==========================================================
# DATASET PAGE
# ==========================================================

if selected == "Dataset":

    st.title("📊 Dataset Overview")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", f"{len(df):,}")

    with col2:
        st.metric("Fraud Transactions", f"{df['Class'].sum():,}")

    with col3:
        fraud_percent = (df["Class"].sum() / len(df)) * 100
        st.metric("Fraud Percentage", f"{fraud_percent:.3f}%")

    st.markdown("---")

    st.subheader("Dataset Shape")

    st.write(f"Rows : {df.shape[0]:,}")
    st.write(f"Columns : {df.shape[1]}")

    st.markdown("---")

    st.subheader("First 10 Records")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("Column Names")

    st.write(list(df.columns))

    # ==========================================================
# VISUALIZATION PAGE
# ==========================================================

if selected == "Visualization":

    st.title("📈 Data Visualization")

    st.markdown("---")

    # Class Distribution
    st.subheader("Class Distribution")

    class_counts = df["Class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]

    class_counts["Class"] = class_counts["Class"].replace({
        0: "Legitimate",
        1: "Fraud"
    })

    fig = px.bar(
        class_counts,
        x="Class",
        y="Count",
        color="Class",
        text="Count",
        title="Fraud vs Legitimate Transactions"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Transaction Amount
    st.subheader("Transaction Amount Distribution")

    fig = px.histogram(
        df,
        x="Amount",
        nbins=100,
        title="Distribution of Transaction Amount"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Time Distribution
    st.subheader("Transaction Time Distribution")

    fig = px.histogram(
        df,
        x="Time",
        nbins=100,
        title="Transaction Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation Heatmap

    st.subheader("Correlation Heatmap")

    corr = df.corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
# ==========================================================
# PREDICTION PAGE
# ==========================================================

if selected == "Prediction":

    st.title("🤖 Credit Card Fraud Prediction")

    st.write("Predict transactions manually or test using real fraud/legitimate examples.")

    st.markdown("---")

    # ==========================
    # Manual Prediction
    # ==========================

    st.subheader("Manual Prediction")

    transaction_index = st.number_input(
        "Enter Transaction Index",
        min_value=0,
        max_value=len(df)-1,
        value=0,
        step=1
    )

    if st.button("Predict"):

        sample = df.drop("Class", axis=1).iloc[[transaction_index]]

        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)[0]

        probability = model.predict_proba(sample_scaled)[0][1]

        st.markdown("## Prediction Result")

        if prediction == 1:

            st.error("🚨 Fraudulent Transaction")

        else:

            st.success("✅ Legitimate Transaction")

        st.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
        )

        if probability < 0.30:

            st.success("🟢 Risk Level : LOW")
            st.info("Recommendation : Approve Transaction")

        elif probability < 0.70:

            st.warning("🟡 Risk Level : MEDIUM")
            st.info("Recommendation : Manual Review")

        else:

            st.error("🔴 Risk Level : HIGH")
            st.error("Recommendation : Block Transaction")

        st.subheader("Transaction Details")

        st.dataframe(sample)

    st.markdown("---")

    # ==========================
    # Demo Prediction
    # ==========================

    st.subheader("Quick Demo")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("🟢 Random Legitimate Example"):

            sample = df[df["Class"] == 0].sample(1)

            st.session_state["sample"] = sample

    with col2:

        if st.button("🔴 Random Fraud Example"):

            sample = df[df["Class"] == 1].sample(1)

            st.session_state["sample"] = sample

    if "sample" in st.session_state:

        sample = st.session_state["sample"]

        X = sample.drop("Class", axis=1)

        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]

        probability = model.predict_proba(X_scaled)[0][1]

        st.markdown("## Demo Prediction")

        if prediction == 1:

            st.error("🚨 Fraudulent Transaction")

        else:

            st.success("✅ Legitimate Transaction")

        st.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
        )

        if probability < 0.30:

            st.success("🟢 Risk Level : LOW")

        elif probability < 0.70:

            st.warning("🟡 Risk Level : MEDIUM")

        else:

            st.error("🔴 Risk Level : HIGH")

        st.subheader("Transaction Details")

        st.dataframe(sample)

        # ==========================================================
# ==========================================================
# PERFORMANCE PAGE
# ==========================================================

if selected == "Performance":

    st.title("📈 Model Performance Dashboard")
    st.markdown("### Performance of Machine Learning Models")
    st.markdown("---")

    # ==========================================================
    # Performance Table
    # ==========================================================

    performance = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Tuned Random Forest"
        ],

        "Accuracy": [
            0.9741,
            0.9995,
            0.9992
        ],

        "Precision": [
            0.0578,
            0.8710,
            0.7477
        ],

        "Recall": [
            0.9184,
            0.8265,
            0.8163
        ],

        "F1 Score": [
            0.1088,
            0.8482,
            0.7805
        ],

        "ROC AUC": [
            0.9708,
            0.9754,
            0.9678
        ]

    })

    st.subheader("📋 Performance Comparison")

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # Best Model
    # ==========================================================

    st.subheader(" Best Model")

    st.success("""
Random Forest achieved the best overall performance.

✔ Accuracy : 99.95%

✔ Precision : 87.10%

✔ Recall : 82.65%

✔ F1 Score : 84.82%

✔ ROC-AUC : 97.54%
""")

    st.markdown("---")

    # ==========================================================
    # Accuracy
    # ==========================================================

    st.subheader("📊 Accuracy Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # Precision
    # ==========================================================

    st.subheader("📊 Precision Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="Precision",
        color="Model",
        text="Precision"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # Recall
    # ==========================================================

    st.subheader("📊 Recall Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="Recall",
        color="Model",
        text="Recall"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # F1 Score
    # ==========================================================

    st.subheader("📊 F1 Score Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="F1 Score",
        color="Model",
        text="F1 Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # ROC AUC
    # ==========================================================

    st.subheader("📊 ROC AUC Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="ROC AUC",
        color="Model",
        text="ROC AUC"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==========================================================
    # Conclusion
    # ==========================================================

    st.subheader(" Conclusion")

    st.info("""
Random Forest was selected as the final model because it achieved the best balance between:

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

The model can successfully identify fraudulent credit card transactions while maintaining excellent overall performance.
""")
    
    # ==========================================================
# ABOUT PAGE
# ==========================================================

if selected == "About":

    st.title("About This Project")

    st.markdown("---")

    st.header(" Project Overview")

    st.write("""
This project is a **Credit Card Fraud Detection System** developed using
Machine Learning techniques.

The goal of this project is to identify fraudulent credit card transactions
with high accuracy while minimizing false alarms.

The system helps financial institutions detect suspicious transactions
and reduce financial losses caused by fraud.
""")

    st.markdown("---")

    st.header("Project Objectives")

    st.write("""
✔ Detect fraudulent credit card transactions.

✔ Compare multiple Machine Learning models.

✔ Select the best performing model.

✔ Visualize transaction patterns.

✔ Build an interactive Streamlit Dashboard.

✔ Perform real-time fraud prediction.
""")

    st.markdown("---")

    st.header(" Dataset Information")

    st.write("""
Dataset Name:
**Credit Card Fraud Detection Dataset**

Source:
Kaggle

Total Transactions:
284,807

Fraud Transactions:
492

Legitimate Transactions:
284,315

Features:
30 Input Features + 1 Target Variable
""")

    st.markdown("---")

    st.header("🤖 Machine Learning Models")

    st.write("""
• Logistic Regression

• Random Forest

• Tuned Random Forest
""")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    st.write("""
• Python

• Pandas

• NumPy

• Matplotlib

• Seaborn

• Scikit-Learn

• SMOTE

• Joblib

• Streamlit

• Plotly
""")

    st.markdown("---")

    st.header(" Best Model")

    st.success("""
Random Forest was selected as the final model.

Accuracy : 99.95%

Precision : 87.10%

Recall : 82.65%

F1 Score : 84.82%

ROC-AUC : 97.54%
""")

    st.markdown("---")

    st.header(" Developed By")

    st.write("""
**Name:** Gulam Zhhra

BS Computer Science

Data Science Intern

Machine Learning & Data Analytics Enthusiast
""")

    st.markdown("---")

