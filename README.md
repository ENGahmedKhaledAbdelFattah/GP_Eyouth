# 🔮 IBM Telco Customer Churn Prediction & Analytics Portal

[![Streamlit App](https://static.streamlit.io/badge_indicator.svg)](https://gpeyouth-65ffzybxszyjnnvmd7tdpk.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Supervised%20%26%20Unsupervised-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()

This repository hosts an end-to-end Machine Learning pipeline and an interactive, bilingual (English & Arabic) analytics portal designed to predict customer churn in the telecom industry. Built for our graduation project, it uses the **IBM Telco Customer Churn** dataset.

🔗 **Live Application Link:** [https://gpeyouth-65ffzybxszyjnnvmd7tdpk.streamlit.io/](https://gpeyouth-65ffzybxszyjnnvmd7tdpk.streamlit.app/)

---

## 🚀 Key Features

*   **🌐 Bilingual Interface:** Toggle instantly between English and Arabic (`العربية`) with localized UI layouts (supports RTL rendering).
*   **📊 Executive Overview:** KPI cards summarizing total customers, average churn rate, and monthly recurring revenue alongside interactive correlation heatmaps.
*   **📈 Demographic & Service Insights:** Deep-dive analysis of demographics, contract types, billing models, and churn rate distributions.
*   **🧩 Customer Segmentation (K-Means):** Unsupervised customer grouping into 3 target profiles (Loyal High Value, Mid Value Developing, New Low Value) projected onto a 2D PCA space.
*   **🎯 Model Performance & Decision Curves:** Comparative analysis of SVM, Logistic Regression, K-Means classifier, and XGBoost using ROC and Precision-Recall curves.
*   **🧠 Real-Time Churn Predictor:** A smart interface where business users can input any customer profile to generate immediate churn risk probabilities.

---

## 🛠️ Tech Stack & Libraries

*   **Frontend & UI:** Streamlit, Custom HTML/CSS (Glassmorphism, HSL colors).
*   **Data Manipulation:** Pandas, NumPy.
*   **Visualization:** Matplotlib, Seaborn.
*   **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost.
*   **Dimensionality Reduction & Clustering:** PCA, K-Means.
*   **Imbalance Handling:** SMOTE (imbalanced-learn).
*   **Model Serialization:** Joblib.

---

## 📂 Project Structure

```bash
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw Dataset (7,043 rows)
├── churn (1).ipynb                       # Clean end-to-end ML notebook
├── app.py                                # Streamlit dashboard code (Bilingual)
├── requirements.txt                      # Cloud deployment dependencies
├── .gitignore                            # Excluded cache and temp directories
├── encoder.joblib                        # Fitted One-Hot Encoder
├── scaler.joblib                         # Fitted StandardScaler (continuous variables)
├── scaler_cluster.joblib                 # Fitted StandardScaler (K-Means features)
├── kmeans_final.joblib                   # Fitted K-Means model (K=3)
├── xgb_model.joblib                      # Pre-trained XGBoost Classifier (recall: 90.64%)
├── feature_names.json                    # Feature mapping dictionary
└── README.md                             # Project Documentation (this file)
```

---

## 🖥️ Local Execution

To run the application locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ENGahmedKhaledAbdelFattah/GP_Eyouth.git
   cd GP_Eyouth
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser.

---

## 🧠 Preprocessing & Modeling Pipeline

The predictive engine uses **XGBoost + SMOTE** which achieves the best performance:
*   **Imbalance Correction:** SMOTE balances training classes (4,139 samples per class).
*   **Recall Priority:** Catching **90.64%** of churners 30 days in advance.
*   **Unsupervised segments:** Seamlessly maps K-Means clusters on the fly for new input customer profiles.
