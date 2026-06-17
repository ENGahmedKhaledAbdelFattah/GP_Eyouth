import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Customer Churn Portal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_dataset():
    if os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce').fillna(0)
        return df
    return None

# Load preprocessed variables and models
def load_artifacts():
    try:
        encoder = joblib.load("encoder.joblib")
        scaler = joblib.load("scaler.joblib")
        scaler_cluster = joblib.load("scaler_cluster.joblib")
        kmeans_final = joblib.load("kmeans_final.joblib")
        xgb_model = joblib.load("xgb_model.joblib")
        with open("feature_names.json", "r") as f:
            feature_config = json.load(f)
        return encoder, scaler, scaler_cluster, kmeans_final, xgb_model, feature_config
    except:
        return None, None, None, None, None, None

df_raw = load_dataset()
encoder, scaler, scaler_cluster, kmeans_final, xgb_model, feature_config = load_artifacts()

# Sidebar menu
st.sidebar.markdown("<h2 style='text-align: center; color: #1a73e8;'>🔮 Churn Portal</h2>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Executive Overview", 
        "📊 Demographic & Service Insights", 
        "🧩 Unsupervised Segmentation", 
        "🎯 Model Performance & Curves", 
        "🔮 Churn Risk Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Business Value:** Outlining demographics, segments, and predictions to mitigate churn risk."
)

if df_raw is None:
    st.error("⚠️ Dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` not found in workspace.")
else:
    # Pre-calculate global metrics
    total_customers = df_raw.shape[0]
    churn_rate = (df_raw['Churn'] == 'Yes').mean()
    avg_tenure = df_raw['tenure'].mean()
    avg_monthly = df_raw['MonthlyCharges'].mean()

    # Executive Overview
    if menu == "🏠 Executive Overview":
        st.markdown("# 🏠 Churn Executive Overview")
        st.write("Understand target distribution, metrics, and correlations.")

        # KPI Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_customers:,}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ea4335;">{churn_rate:.2%}</div><div class="metric-label">Average Churn Rate</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_tenure:.1f} mo</div><div class="metric-label">Avg Tenure</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">${avg_monthly:.2f}</div><div class="metric-label">Avg Monthly Charge</div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Target Label Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            churn_counts = df_raw['Churn'].value_counts()
            ax.pie(churn_counts, labels=['No Churn', 'Churned'], autopct='%1.1f%%', colors=['steelblue', 'salmon'], startangle=90)
            st.pyplot(fig)
            st.caption("The dataset has a 73/27 class imbalance, which is handled using SMOTE resampling during training.")

        with c2:
            st.markdown("#### Numerical Feature Correlation Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            numerical_data = df_raw[['tenure', 'MonthlyCharges', 'TotalCharges']]
            sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
            st.caption("Strong correlation (0.83) exists between tenure and TotalCharges, indicating multicollinearity.")

    # Demographic & Service Insights
    elif menu == "📊 Demographic & Service Insights":
        st.markdown("# 📊 Demographic & Service Insights")
        
        tab1, tab2, tab3 = st.tabs(["Demographics", "Billing & Service Type", "Charges Distribution"])
        
        with tab1:
            st.markdown("#### Demographics vs Churn Rate")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            gender_churn = df_raw.groupby('gender')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='gender', y='Churn', data=gender_churn, ax=axes[0,0], palette='pastel')
            axes[0,0].set_title('Churn Rate by Gender')
            axes[0,0].set_ylabel('Churn Rate')
            
            senior_churn = df_raw.groupby('SeniorCitizen')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='SeniorCitizen', y='Churn', data=senior_churn, ax=axes[0,1], palette='pastel')
            axes[0,1].set_title('Churn Rate by Senior Citizen')
            axes[0,1].set_ylabel('Churn Rate')
            
            partner_churn = df_raw.groupby('Partner')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='Partner', y='Churn', data=partner_churn, ax=axes[1,0], palette='pastel')
            axes[1,0].set_title('Churn Rate by Partner Status')
            axes[1,0].set_ylabel('Churn Rate')
            
            dependents_churn = df_raw.groupby('Dependents')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='Dependents', y='Churn', data=dependents_churn, ax=axes[1,1], palette='pastel')
            axes[1,1].set_title('Churn Rate by Dependents Status')
            axes[1,1].set_ylabel('Churn Rate')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with tab2:
            st.markdown("#### Contracts & Services vs Churn")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            sns.countplot(x='Contract', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[0,0])
            axes[0,0].set_title("Contract Type")
            
            sns.countplot(x='InternetService', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[0,1])
            axes[0,1].set_title("Internet Service Type")
            
            sns.countplot(x='TechSupport', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[1,0])
            axes[1,0].set_title("Tech Support Subscription")
            
            sns.countplot(x='PaymentMethod', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[1,1])
            axes[1,1].set_title("Payment Method choice")
            axes[1,1].tick_params(axis='x', rotation=20)
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.markdown("#### Continuous Charges Distributions density")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'No']['MonthlyCharges'], label='No Churn', shade=True, color='steelblue', ax=axes[0])
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'Yes']['MonthlyCharges'], label='Churned', shade=True, color='salmon', ax=axes[0])
            axes[0].set_title("Monthly Charges Density")
            axes[0].legend()
            
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'No']['TotalCharges'], label='No Churn', shade=True, color='steelblue', ax=axes[1])
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'Yes']['TotalCharges'], label='Churned', shade=True, color='salmon', ax=axes[1])
            axes[1].set_title("Total Charges Density")
            axes[1].legend()
            
            st.pyplot(fig)

    # Unsupervised Customer Segmentation
    elif menu == "🧩 Unsupervised Segmentation":
        st.markdown("# 🧩 Customer Segmentation via K-Means")
        st.write("Visualizing customer segments computed using the K-Means clustering algorithm.")

        if kmeans_final is None or scaler_cluster is None:
            st.warning("⚠️ Clustering artifacts not found. Run the notebook first to train the K-Means model.")
        else:
            # We reconstruct df_cluster and K-Means PCA components on startup to avoid re-clustering lags
            df_cleaned = df_raw.copy().drop(columns=['customerID', 'gender', 'Churn'])
            df_cleaned[feature_config['binary_columns']] = df_cleaned[feature_config['binary_columns']].replace({'Yes': 1, 'No': 0})
            df_cleaned['Contract'] = df_cleaned['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
            
            # Encoded nominals
            nom_df = pd.DataFrame(encoder.transform(df_raw[['MultipleLines', 'InternetService', 'PaymentMethod']]), 
                                  columns=encoder.get_feature_names_out(['MultipleLines', 'InternetService', 'PaymentMethod']))
            df_cleaned = pd.concat([df_cleaned.drop(columns=['MultipleLines', 'InternetService', 'PaymentMethod']), nom_df], axis=1)
            
            # SpendingTier / TenureGroup
            df_cleaned['SpendingTier'] = pd.cut(df_raw['TotalCharges'], bins=[0, 1000, 4000, 10000], labels=[0, 1, 2], include_lowest=True).astype(int)
            df_cleaned['TenureGroup'] = pd.cut(df_raw['tenure'], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
            
            df_cluster_inp = df_cleaned[feature_config['df_cluster_cols']]
            df_scaled = scaler_cluster.transform(df_cluster_inp)
            clusters = kmeans_final.predict(df_scaled)
            
            # PCA 2D
            pca = PCA(n_components=2)
            df_pca = pca.fit_transform(df_scaled)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### K-Means Customer Segments (PCA 2D)")
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='Set1', alpha=0.5, s=12)
                legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", labels=['Loyal High Value', 'Mid Value Developing', 'New Low Value'])
                ax.add_artist(legend1)
                st.pyplot(fig)
                
            with c2:
                st.markdown("#### Actual Churn Labels on PCA Space")
                fig, ax = plt.subplots(figsize=(6, 5))
                churn_labels = df_raw['Churn'].map({'Yes': 1, 'No': 0}).values
                scatter2 = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=churn_labels, cmap='coolwarm', alpha=0.5, s=12)
                legend2 = ax.legend(*scatter2.legend_elements(), title="Churn", labels=['No Churn', 'Churned'])
                ax.add_artist(legend2)
                st.pyplot(fig)

    # Model Performance & Curves
    elif menu == "🎯 Model Performance & Curves":
        st.markdown("# 🎯 Model Performance & Decision Curves")
        st.write("Compare standard classification curves and importances.")

        if xgb_model is None:
            st.warning("⚠️ Pre-trained model config not found. Run the notebook first.")
        else:
            # We reconstruct test scores quickly to draw the exact curves dynamically on the app!
            df_prep = df_raw.copy().drop(columns=['customerID', 'gender', 'Churn'])
            df_prep[feature_config['binary_columns']] = df_prep[feature_config['binary_columns']].replace({'Yes': 1, 'No': 0})
            df_prep['Contract'] = df_prep['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
            
            nom_df = pd.DataFrame(encoder.transform(df_raw[['MultipleLines', 'InternetService', 'PaymentMethod']]), 
                                  columns=encoder.get_feature_names_out(['MultipleLines', 'InternetService', 'PaymentMethod']))
            df_prep = pd.concat([df_prep.drop(columns=['MultipleLines', 'InternetService', 'PaymentMethod']), nom_df], axis=1)
            
            df_prep['SpendingTier'] = pd.cut(df_raw['TotalCharges'], bins=[0, 1000, 4000, 10000], labels=[0, 1, 2], include_lowest=True).astype(int)
            df_prep['TenureGroup'] = pd.cut(df_raw['tenure'], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
            
            # Predict Cluster
            df_clust_scale = scaler_cluster.transform(df_prep[feature_config['df_cluster_cols']])
            df_prep['Cluster'] = kmeans_final.predict(df_clust_scale)
            
            # Coerce security columns (replace mixed string/numbers)
            # The logic that makes it work:
            for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                df_prep[col] = df_prep[col].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
                
            # Ensure all columns have numeric dtype matching the notebook training data
            df_prep = df_prep.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Align features
            df_features = df_prep[feature_config['X_train_cols']]
            
            # Split
            X_tr, X_te, y_tr, y_te = train_test_split(df_features, df_raw['Churn'].map({'Yes': 1, 'No': 0}), test_size=0.2, random_state=42, stratify=df_raw['Churn'].map({'Yes': 1, 'No': 0}))
            
            # Scale
            X_tr_sc = X_tr.copy()
            X_te_sc = X_te.copy()
            X_tr_sc[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_tr[['tenure', 'MonthlyCharges', 'TotalCharges']])
            X_te_sc[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_te[['tenure', 'MonthlyCharges', 'TotalCharges']])
            
            # Fit Logistic Regression and SVM quickly to plot exact comparisons
            @st.cache_resource
            def get_other_models():
                lr = LogisticRegression(C=10, random_state=42, max_iter=1000)
                lr.fit(X_tr_sc, y_tr)
                svm = SVC(C=10, kernel='rbf', probability=True, random_state=42)
                svm.fit(X_tr_sc, y_tr)
                return lr, svm
                
            lr, svm = get_other_models()
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ROC Curve Comparison")
                fig, ax = plt.subplots(figsize=(6, 5))
                
                # XGBoost
                xgb_proba = xgb_model.predict_proba(X_te_sc)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, xgb_proba)
                ax.plot(fpr, tpr, label=f"XGBoost (AUC={roc_auc_score(y_te, xgb_proba):.3f})")
                
                # SVM
                svm_proba = svm.predict_proba(X_te_sc)[:, 1]
                fpr_s, tpr_s, _ = roc_curve(y_te, svm_proba)
                ax.plot(fpr_s, tpr_s, label=f"SVM Tuned (AUC={roc_auc_score(y_te, svm_proba):.3f})")
                
                # LR
                lr_proba = lr.predict_proba(X_te_sc)[:, 1]
                fpr_l, tpr_l, _ = roc_curve(y_te, lr_proba)
                ax.plot(fpr_l, tpr_l, label=f"LR Tuned (AUC={roc_auc_score(y_te, lr_proba):.3f})")
                
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)
                
            with c2:
                st.markdown("#### Precision-Recall Curve Comparison")
                fig, ax = plt.subplots(figsize=(6, 5))
                
                p, r, _ = precision_recall_curve(y_te, xgb_proba)
                ax.plot(r, p, label="XGBoost")
                
                p_s, r_s, _ = precision_recall_curve(y_te, svm_proba)
                ax.plot(r_s, p_s, label="SVM Tuned")
                
                p_l, r_l, _ = precision_recall_curve(y_te, lr_proba)
                ax.plot(r_l, p_l, label="LR Tuned")
                
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.legend()
                st.pyplot(fig)
                
            st.markdown("#### XGBoost Feature Importance")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            importances = xgb_model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': feature_config['X_train_cols'], 'Importance': importances}).sort_values(by='Importance', ascending=False)
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='viridis', ax=ax2)
            st.pyplot(fig2)

    # Churn Risk Predictor
    elif menu == "🔮 Churn Risk Predictor":
        st.markdown("# 🧠 Customer Churn Risk Predictor")
        
        if xgb_model is None:
            st.warning("⚠️ Model artifacts not found. Please run all code cells in the Jupyter notebook first.")
        else:
            st.write("Enter details of a single customer profile to predict their risk of churn in real-time.")
            
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.subheader("Demographics")
                    senior = st.selectbox("Senior Citizen?", ["No", "Yes"])
                    partner = st.selectbox("Has Partner?", ["No", "Yes"])
                    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
                    gender = st.selectbox("Gender", ["Female", "Male"]) # Display only
                    
                with c2:
                    st.subheader("Contract & Charges")
                    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                    paperless = st.selectbox("Paperless Billing?", ["No", "Yes"])
                    payment = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check", 
                        "Bank transfer (automatic)", "Credit card (automatic)"
                    ])
                    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0)
                    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * monthly_charges)

                with c3:
                    st.subheader("Services")
                    phone = st.selectbox("Phone Service?", ["No", "Yes"])
                    multiple_lines = st.selectbox("Multiple Lines?", ["No phone service", "No", "Yes"] if phone == "No" else ["No", "Yes"])
                    internet = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
                    
                    if internet != "No":
                        security = st.selectbox("Online Security?", ["No", "Yes"])
                        backup = st.selectbox("Online Backup?", ["No", "Yes"])
                        protection = st.selectbox("Device Protection?", ["No", "Yes"])
                        support = st.selectbox("Tech Support?", ["No", "Yes"])
                        tv = st.selectbox("Streaming TV?", ["No", "Yes"])
                        movies = st.selectbox("Streaming Movies?", ["No", "Yes"])
                    else:
                        security, backup, protection, support, tv, movies = "No internet service", "No internet service", "No internet service", "No internet service", "No internet service", "No internet service"

                submit_btn = st.form_submit_button("Predict Churn Risk")

            if submit_btn:
                # 1. Map values
                inputs = {
                    'SeniorCitizen': 1 if senior == "Yes" else 0,
                    'Partner': 1 if partner == "Yes" else 0,
                    'Dependents': 1 if dependents == "Yes" else 0,
                    'tenure': tenure,
                    'PhoneService': 1 if phone == "Yes" else 0,
                    'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
                    'PaperlessBilling': 1 if paperless == "Yes" else 0,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet,
                    'PaymentMethod': payment
                }
                
                # Preprocess nominal features
                df_nominal = pd.DataFrame([[inputs['MultipleLines'], inputs['InternetService'], inputs['PaymentMethod']]], 
                                          columns=['MultipleLines', 'InternetService', 'PaymentMethod'])
                encoded_features = encoder.transform(df_nominal)
                encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['MultipleLines', 'InternetService', 'PaymentMethod']))
                
                # SpendingTier and TenureGroup
                spending_tier = 0 if total_charges <= 1000 else (1 if total_charges <= 4000 else 2)
                tenure_group = 0 if tenure <= 12 else (1 if tenure <= 24 else (2 if tenure <= 48 else 3))
                
                # df_cluster input features
                df_cluster_input = pd.DataFrame([[
                    inputs['SeniorCitizen'], inputs['Partner'], inputs['Dependents'], inputs['tenure'], inputs['PhoneService'],
                    inputs['Contract'], inputs['PaperlessBilling'], inputs['MonthlyCharges'], inputs['TotalCharges'],
                    encoded_df.get('MultipleLines_No phone service', [0])[0], encoded_df.get('MultipleLines_Yes', [0])[0],
                    encoded_df.get('InternetService_Fiber optic', [0])[0], encoded_df.get('InternetService_No', [0])[0],
                    encoded_df.get('PaymentMethod_Credit card (automatic)', [0])[0], encoded_df.get('PaymentMethod_Electronic check', [0])[0],
                    encoded_df.get('PaymentMethod_Mailed check', [0])[0],
                    spending_tier, tenure_group
                ]], columns=feature_config['df_cluster_cols'])
                
                # Predict Cluster
                df_cluster_scaled = scaler_cluster.transform(df_cluster_input)
                predicted_cluster = kmeans_final.predict(df_cluster_scaled)[0]
                
                # Map OnlineSecurity etc. (Yes -> 1, No / No internet service -> 0)
                sec_val = 1 if security == "Yes" else 0
                bk_val = 1 if backup == "Yes" else 0
                prot_val = 1 if protection == "Yes" else 0
                sup_val = 1 if support == "Yes" else 0
                tv_val = 1 if tv == "Yes" else 0
                mov_val = 1 if movies == "Yes" else 0
                
                # Combine final features
                df_final_input = pd.DataFrame([[
                    inputs['SeniorCitizen'], inputs['Partner'], inputs['Dependents'], inputs['tenure'], inputs['PhoneService'],
                    sec_val, bk_val, prot_val, sup_val, tv_val, mov_val,
                    inputs['Contract'], inputs['PaperlessBilling'], inputs['MonthlyCharges'], inputs['TotalCharges'],
                    df_cluster_input['MultipleLines_No phone service'][0], df_cluster_input['MultipleLines_Yes'][0],
                    df_cluster_input['InternetService_Fiber optic'][0], df_cluster_input['InternetService_No'][0],
                    df_cluster_input['PaymentMethod_Credit card (automatic)'][0], df_cluster_input['PaymentMethod_Electronic check'][0],
                    df_cluster_input['PaymentMethod_Mailed check'][0],
                    spending_tier, tenure_group,
                    predicted_cluster
                ]], columns=feature_config['X_train_cols'])
                
                # Scale continuous features
                df_final_input[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df_final_input[['tenure', 'MonthlyCharges', 'TotalCharges']])
                
                # Run prediction
                proba = xgb_model.predict_proba(df_final_input)[0][1]
                
                st.markdown("---")
                st.subheader("🔍 Prediction Results")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write(f"**Customer Cluster:** Segment {predicted_cluster}")
                    if predicted_cluster == 0:
                        st.success("Segment Type: Loyal / High Value Customer")
                    elif predicted_cluster == 1:
                        st.info("Segment Type: Mid Value / Developing Customer")
                    else:
                        st.warning("Segment Type: New / Low Value Customer")
                        
                with col_res2:
                    st.write(f"**Churn Probability:** `{proba:.2%}`")
                    if proba >= 0.6:
                        st.error("🚨 **High Risk:** High churn probability detected. Urgent retention action recommended.")
                    elif proba >= 0.3:
                        st.warning("⚠️ **Medium Risk:** Moderate risk. Keep monitoring user satisfaction.")
                    else:
                        st.success("✅ **Low Risk:** Customer is stable and unlikely to churn.")
                
                st.progress(float(proba))
