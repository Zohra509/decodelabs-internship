# 💳 Credit Card Fraud Detection System

A Machine Learning-based Credit Card Fraud Detection System that identifies fraudulent transactions with high accuracy using multiple classification algorithms. The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and an interactive Streamlit dashboard for real-time predictions.

---

## 📌 Project Overview

Credit card fraud has become one of the major challenges in the financial industry. This project aims to detect fraudulent transactions using machine learning techniques while minimizing false positives.

The project compares multiple machine learning models and selects the best-performing model based on evaluation metrics.

---

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Preprocessing
- ⚖️ Handling Imbalanced Data using SMOTE
- 🤖 Logistic Regression Model
- 🌲 Random Forest Classifier
- 🎯 Hyperparameter Tuning (RandomizedSearchCV)
- 📈 Model Performance Comparison
- 📉 ROC Curve & Precision-Recall Curve
- 📊 Feature Importance Visualization
- 💾 Model Saving using Joblib
- 🌐 Interactive Streamlit Dashboard
- 🔍 Real-Time Fraud Prediction

---

## 📂 Dataset

**Dataset:** Credit Card Fraud Detection Dataset

- Total Transactions: **284,807**
- Fraud Transactions: **492**
- Legitimate Transactions: **284,315**

The dataset contains anonymized features (`V1`–`V28`) generated using PCA along with:

- Time
- Amount
- Class (Target Variable)

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- SMOTE (Imbalanced-learn)
- Joblib
- Streamlit

---

## 🤖 Machine Learning Models

The following models were implemented:

1. Logistic Regression
2. Random Forest Classifier
3. Tuned Random Forest (RandomizedSearchCV)

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 97.41% | 5.78% | 91.84% | 10.88% | 97.08% |
| Random Forest | **99.95%** | **87.10%** | **82.65%** | **84.82%** | **97.54%** |
| Tuned Random Forest | 99.92% | 74.77% | 81.63% | 78.05% | 96.78% |

✅ **Best Model:** Random Forest Classifier

---

## 📈 Dashboard

The Streamlit dashboard includes:

- 🏠 Home
- 📂 Dataset Information
- 📊 Data Visualization
- 🤖 Fraud Prediction
- 📈 Model Performance
- ℹ️ About Project

---

## 📁 Project Structure

```
fraud_detection_task_2/
│
├── app/
│   └── app.py
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── dataset/
│   └── creditcard.csv
│
├── notebook/
│   └── fraud_detection.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/fraud_detection_task_2.git
```

Navigate to the project

```bash
cd fraud_detection_task_2
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## 📸 Screenshots

You can add screenshots of:

- Home Dashboard
- Dataset Page
- Visualization Page
- Prediction Page
- Performance Page

---

## 🎯 Future Improvements

- Deep Learning Models
- XGBoost & LightGBM
- Live Transaction Prediction
- Database Integration
- REST API
- Cloud Deployment
- User Authentication

---

## 👩‍💻 Author

**Gulam Zohra**

BS Computer Science

Machine Learning & Data Science Enthusiast

---

## ⭐ Acknowledgements

- Kaggle
- Scikit-learn
- Streamlit
- Imbalanced-learn

---


## 📂 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

**Dataset Link:**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Statistics:

- Total Transactions: **284,807**
- Legitimate Transactions: **284,315**
- Fraudulent Transactions: **492**
- Features: **30 Input Features + 1 Target Variable**

> **Note:** The dataset is not included in this repository due to file size limitations and Kaggle's distribution policy. Please download the dataset from the Kaggle link above and place the `creditcard.csv` file in the project's dataset folder before running the application.
