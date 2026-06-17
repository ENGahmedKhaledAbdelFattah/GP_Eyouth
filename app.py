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
    page_title="Customer Churn Portal | بوابة توقع مغادرة العملاء",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bilingual Translation Dictionary
T = {
    "en": {
        "title": "🔮 Telecom Customer Churn Analytics Portal",
        "subtitle": "An end-to-end Machine Learning approach for forecasting and understanding customer retention risk.",
        "sidebar_header": "🔮 Churn Portal",
        "sidebar_info": "💡 **Business Value:** Outlining demographics, segments, and predictions to mitigate churn risk.",
        "lang_selector": "🌐 Language / اللغة",
        "menu_label": "Navigation Menu",
        "menu_opts": [
            "🏠 Executive Overview", 
            "📊 Demographic & Service Insights", 
            "🧩 Unsupervised Segmentation", 
            "🎯 Model Performance & Curves", 
            "🔮 Churn Risk Predictor"
        ],
        "kpi_total": "Total Customers",
        "kpi_churn": "Average Churn Rate",
        "kpi_tenure": "Avg Tenure (Months)",
        "kpi_charge": "Avg Monthly Charge",
        "pie_title": "Target Label Distribution (Churn vs No Churn)",
        "pie_caption": "The dataset has a 73/27 class imbalance, handled using SMOTE resampling during training.",
        "heatmap_title": "Numerical Feature Correlation Heatmap",
        "heatmap_caption": "Strong correlation (0.83) exists between tenure and TotalCharges, indicating multicollinearity.",
        "tab_demo": "Demographics",
        "tab_billing": "Billing & Service Type",
        "tab_charges": "Charges Distribution",
        "demo_title": "Demographics vs Churn Rate",
        "billing_title": "Contracts & Services vs Churn",
        "charges_title": "Continuous Charges Distributions Density",
        "clust_title": "K-Means Customer Segments (PCA 2D)",
        "actual_clust_title": "Actual Churn Labels on PCA Space",
        "clust_explanation": "Visualizing customer segments computed using the K-Means clustering algorithm.",
        "model_title": "🎯 Model Performance & Decision Curves",
        "model_subtitle": "Compare standard classification curves and feature importances.",
        "roc_title": "ROC Curve Comparison",
        "pr_title": "Precision-Recall Curve Comparison",
        "feat_imp_title": "XGBoost Feature Importance (Top 10)",
        "predictor_title": "🧠 Customer Churn Risk Predictor",
        "predictor_subtitle": "Enter details of a single customer profile to predict their risk of churn in real-time.",
        "sec_demo": "Demographics",
        "sec_contract": "Contract & Charges",
        "sec_services": "Services Subscribed",
        "lbl_senior": "Senior Citizen?",
        "lbl_partner": "Has Partner?",
        "lbl_dependents": "Has Dependents?",
        "lbl_gender": "Gender",
        "lbl_tenure": "Tenure (Months)",
        "lbl_contract": "Contract Type",
        "lbl_paperless": "Paperless Billing?",
        "lbl_payment": "Payment Method",
        "lbl_monthly": "Monthly Charges ($)",
        "lbl_total": "Total Charges ($)",
        "lbl_phone": "Phone Service?",
        "lbl_multiple": "Multiple Lines?",
        "lbl_internet": "Internet Service Type",
        "lbl_security": "Online Security?",
        "lbl_backup": "Online Backup?",
        "lbl_protection": "Device Protection?",
        "lbl_support": "Tech Support?",
        "lbl_tv": "Streaming TV?",
        "lbl_movies": "Streaming Movies?",
        "btn_predict": "Predict Churn Risk",
        "res_header": "🔍 Prediction Results",
        "res_cluster": "Customer Cluster:",
        "res_prob": "Churn Probability:",
        "risk_high": "🚨 High Risk: High churn probability detected. Urgent retention action recommended.",
        "risk_med": "⚠️ Medium Risk: Moderate risk. Keep monitoring user satisfaction.",
        "risk_low": "✅ Low Risk: Customer is stable and unlikely to churn.",
        "segment_loyal": "Segment Type: Loyal / High Value Customer",
        "segment_developing": "Segment Type: Mid Value / Developing Customer",
        "segment_new": "Segment Type: New / Low Value Customer",
        "no_data": "⚠️ Dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` not found in workspace.",
        "no_models": "⚠️ Pre-trained model config not found. Run the notebook first."
    },
    "ar": {
        "title": "🔮 بوابة تحليلات توقعات مغادرة العملاء للاتصالات",
        "subtitle": "نهج تعلّم الآلة الشامل للتنبؤ بمخاطر الاحتفاظ بالعملاء وفهمها.",
        "sidebar_header": "🔮 بوابة المغادرة",
        "sidebar_info": "💡 **القيمة التجارية:** تحديد الخصائص الديموغرافية والشرائح والتوقعات للحد من مخاطر مغادرة العملاء.",
        "lang_selector": "🌐 Language / اللغة",
        "menu_label": "قائمة التنقل",
        "menu_opts": [
            "🏠 نظرة عامة تنفيذية", 
            "📊 رؤى الديموغرافيا والخدمات", 
            "🧩 تقسيم العملاء (K-Means)", 
            "🎯 أداء النموذج ومنحنيات القرار", 
            "🔮 متنبئ مخاطر المغادرة"
        ],
        "kpi_total": "إجمالي العملاء",
        "kpi_churn": "متوسط معدل المغادرة",
        "kpi_tenure": "متوسط فترة الاشتراك (شهور)",
        "kpi_charge": "متوسط الرسوم الشهرية",
        "pie_title": "توزيع حالات المغادرة (مغادرة مقابل بقاء)",
        "pie_caption": "تحتوي مجموعة البيانات على عدم توازن فئوي بنسبة 73/27، تم التعامل معه باستخدام إعادة أخذ العينات SMOTE أثناء التدريب.",
        "heatmap_title": "خريطة الارتباط للميزات العددية",
        "heatmap_caption": "يوجد ارتباط قوي (0.83) بين فترة الاشتراك وإجمالي الرسوم، مما يشير إلى وجود علاقة خطية متعددة.",
        "tab_demo": "البيانات الديموغرافية",
        "tab_billing": "الفواتير ونوع الخدمة",
        "tab_charges": "توزيع الرسوم",
        "demo_title": "البيانات الديموغرافية مقابل معدل مغادرة العملاء",
        "billing_title": "العقود والخدمات مقابل مغادرة العملاء",
        "charges_title": "كثافة توزيع الرسوم المستمرة",
        "clust_title": "شرائح عملاء K-Means (PCA 2D)",
        "actual_clust_title": "حالات المغادرة الفعلية على فضاء PCA",
        "clust_explanation": "تصوير شرائح العملاء المحسوبة باستخدام خوارزمية تقسيم العملاء K-Means.",
        "model_title": "🎯 أداء النموذج ومنحنيات القرار",
        "model_subtitle": "مقارنة منحنيات التصنيف القياسية وأهمية الميزات.",
        "roc_title": "مقارنة منحنيات ROC",
        "pr_title": "مقارنة منحنيات Precision-Recall",
        "feat_imp_title": "أهمية ميزات XGBoost (أعلى 10 ميزات)",
        "predictor_title": "🧠 متنبئ مخاطر مغادرة العملاء",
        "predictor_subtitle": "أدخل تفاصيل ملف تعريف عميل واحد للتنبؤ بخطر مغادرته في الوقت الفعلي.",
        "sec_demo": "الديموغرافيا",
        "sec_contract": "العقود والرسوم",
        "sec_services": "الخدمات المشترك بها",
        "lbl_senior": "مواطن مسن؟",
        "lbl_partner": "لديه شريك؟",
        "lbl_dependents": "يعيل أشخاصًا؟",
        "lbl_gender": "الجنس",
        "lbl_tenure": "فترة الاشتراك (بالشهور)",
        "lbl_contract": "نوع العقد",
        "lbl_paperless": "الفواتير الإلكترونية؟",
        "lbl_payment": "طريقة الدفع",
        "lbl_monthly": "الرسوم الشهرية ($)",
        "lbl_total": "إجمالي الرسوم ($)",
        "lbl_phone": "خدمة هاتفية؟",
        "lbl_multiple": "خطوط متعددة؟",
        "lbl_internet": "نوع خدمة الإنترنت",
        "lbl_security": "الأمان عبر الإنترنت؟",
        "lbl_backup": "النسخ الاحتياطي عبر الإنترنت؟",
        "lbl_protection": "حماية الأجهزة؟",
        "lbl_support": "الدعم الفني؟",
        "lbl_tv": "بث التلفزيون؟",
        "lbl_movies": "بث الأفلام؟",
        "btn_predict": "توقع خطر المغادرة",
        "res_header": "🔍 نتائج التنبؤ",
        "res_cluster": "شريحة العميل:",
        "res_prob": "احتمال المغادرة:",
        "risk_high": "🚨 خطر مرتفع: تم الكشف عن احتمال مغادرة مرتفع. يوصى بإجراءات فورية للاحتفاظ بالعميل.",
        "risk_med": "⚠️ خطر متوسط: خطر معتدل. استمر في مراقبة رضا العميل.",
        "risk_low": "✅ خطر منخفض: العميل مستقر ومن غير المرجح أن يغادر.",
        "segment_loyal": "نوع الشريحة: عميل وفي ذو قيمة عالية",
        "segment_developing": "نوع الشريحة: عميل متوسط القيمة قيد التطوير",
        "segment_new": "نوع الشريحة: عميل جديد ذو قيمة منخفضة",
        "no_data": "⚠️ ملف البيانات `WA_Fn-UseC_-Telco-Customer-Churn.csv` غير موجود في بيئة العمل.",
        "no_models": "⚠️ ملفات النماذج المدربة غير موجودة. يرجى تشغيل دفتر Jupyter أولاً لحفظ النماذج."
    }
}

# Custom styling for premium look & RTL support
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', 'Cairo', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
        border: 1px solid #e9ecef;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(26, 115, 232, 0.2);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #155cb0;
        box-shadow: 0 6px 12px rgba(26, 115, 232, 0.3);
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

# Language Toggle in Sidebar
st.sidebar.markdown("<div style='text-align: center;'>🌍</div>", unsafe_allow_html=True)
lang = st.sidebar.selectbox("🌐 Select Language / اختر اللغة", ["English", "العربية"])
lang_code = "en" if lang == "English" else "ar"
text = T[lang_code]

# Apply RTL layout style for Arabic
if lang_code == "ar":
    st.markdown("""
    <style>
        .block-container {
            direction: rtl;
            text-align: right;
        }
        .stMarkdown, .stSubheader, .stTitle, .stHeader, .stCaption, .stFormSubmitButton, .stTextInput, .stNumberInput, .stSelectbox, .stSlider {
            direction: rtl;
            text-align: right !important;
        }
        .metric-card {
            direction: rtl;
        }
        .stTabs {
            direction: rtl;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation menu using translated labels
st.sidebar.markdown(f"<h2 style='text-align: center; color: #1a73e8;'>{text['sidebar_header']}</h2>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    text["menu_label"],
    text["menu_opts"]
)

st.sidebar.markdown("---")
st.sidebar.info(text["sidebar_info"])

if df_raw is None:
    st.error(text["no_data"])
else:
    # Pre-calculate global metrics
    total_customers = df_raw.shape[0]
    churn_rate = (df_raw['Churn'] == 'Yes').mean()
    avg_tenure = df_raw['tenure'].mean()
    avg_monthly = df_raw['MonthlyCharges'].mean()

    # Executive Overview Page
    if menu == text["menu_opts"][0]: # Overview
        st.markdown(f"# {text['title']}")
        st.write(text["subtitle"])

        # KPI Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_customers:,}</div><div class="metric-label">{text["kpi_total"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ea4335;">{churn_rate:.2%}</div><div class="metric-label">{text["kpi_churn"]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_tenure:.1f} mo</div><div class="metric-label">{text["kpi_tenure"]}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">${avg_monthly:.2f}</div><div class="metric-label">{text["kpi_charge"]}</div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### {text['pie_title']}")
            fig, ax = plt.subplots(figsize=(6, 4))
            churn_counts = df_raw['Churn'].value_counts()
            ax.pie(churn_counts, labels=['No Churn', 'Churned'], autopct='%1.1f%%', colors=['#3a86c8', '#f17b6a'], startangle=90)
            st.pyplot(fig)
            st.caption(text["pie_caption"])

        with c2:
            st.markdown(f"#### {text['heatmap_title']}")
            fig, ax = plt.subplots(figsize=(6, 4))
            numerical_data = df_raw[['tenure', 'MonthlyCharges', 'TotalCharges']]
            sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
            st.caption(text["heatmap_caption"])

    # Demographics & Service Insights
    elif menu == text["menu_opts"][1]: # Insights
        st.markdown(f"# 📊 {text['menu_opts'][1]}")
        
        tab1, tab2, tab3 = st.tabs([text["tab_demo"], text["tab_billing"], text["tab_charges"]])
        
        with tab1:
            st.markdown(f"#### {text['demo_title']}")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            gender_churn = df_raw.groupby('gender')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='gender', y='Churn', data=gender_churn, ax=axes[0,0], palette='pastel')
            axes[0,0].set_title('Churn Rate by Gender / معدل المغادرة حسب الجنس')
            axes[0,0].set_ylabel('Churn Rate')
            
            senior_churn = df_raw.groupby('SeniorCitizen')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='SeniorCitizen', y='Churn', data=senior_churn, ax=axes[0,1], palette='pastel')
            axes[0,1].set_title('Churn Rate by Senior Citizen / معدل المغادرة لكبار السن')
            axes[0,1].set_ylabel('Churn Rate')
            
            partner_churn = df_raw.groupby('Partner')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='Partner', y='Churn', data=partner_churn, ax=axes[1,0], palette='pastel')
            axes[1,0].set_title('Churn Rate by Partner / معدل المغادرة للمرتبطين')
            axes[1,0].set_ylabel('Churn Rate')
            
            dependents_churn = df_raw.groupby('Dependents')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            sns.barplot(x='Dependents', y='Churn', data=dependents_churn, ax=axes[1,1], palette='pastel')
            axes[1,1].set_title('Churn Rate by Dependents / معدل المغادرة للمستقلين')
            axes[1,1].set_ylabel('Churn Rate')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        with tab2:
            st.markdown(f"#### {text['billing_title']}")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            sns.countplot(x='Contract', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[0,0])
            axes[0,0].set_title("Contract Type / نوع العقد")
            
            sns.countplot(x='InternetService', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[0,1])
            axes[0,1].set_title("Internet Service / نوع خدمة الإنترنت")
            
            sns.countplot(x='TechSupport', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[1,0])
            axes[1,0].set_title("Tech Support / الدعم الفني")
            
            sns.countplot(x='PaymentMethod', hue='Churn', data=df_raw, palette='coolwarm', ax=axes[1,1])
            axes[1,1].set_title("Payment Method / طريقة الدفع")
            axes[1,1].tick_params(axis='x', rotation=20)
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab3:
            st.markdown(f"#### {text['charges_title']}")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'No']['MonthlyCharges'], label='No Churn', fill=True, color='steelblue', ax=axes[0])
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'Yes']['MonthlyCharges'], label='Churned', fill=True, color='salmon', ax=axes[0])
            axes[0].set_title("Monthly Charges Density / توزيع الرسوم الشهرية")
            axes[0].legend()
            
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'No']['TotalCharges'], label='No Churn', fill=True, color='steelblue', ax=axes[1])
            sns.kdeplot(data=df_raw[df_raw['Churn'] == 'Yes']['TotalCharges'], label='Churned', fill=True, color='salmon', ax=axes[1])
            axes[1].set_title("Total Charges Density / توزيع إجمالي الرسوم")
            axes[1].legend()
            
            st.pyplot(fig)

    # Customer Segmentation
    elif menu == text["menu_opts"][2]: # Segmentation
        st.markdown(f"# 🧩 {text['menu_opts'][2]}")
        st.write(text["clust_explanation"])

        if kmeans_final is None or scaler_cluster is None:
            st.warning(text["no_models"])
        else:
            # Reconstruct and scale clusters dynamically
            df_cleaned = df_raw.copy().drop(columns=['customerID', 'gender', 'Churn'])
            df_cleaned[feature_config['binary_columns']] = df_cleaned[feature_config['binary_columns']].replace({'Yes': 1, 'No': 0})
            df_cleaned['Contract'] = df_cleaned['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
            
            nom_df = pd.DataFrame(encoder.transform(df_raw[['MultipleLines', 'InternetService', 'PaymentMethod']]), 
                                  columns=encoder.get_feature_names_out(['MultipleLines', 'InternetService', 'PaymentMethod']))
            df_cleaned = pd.concat([df_cleaned.drop(columns=['MultipleLines', 'InternetService', 'PaymentMethod']), nom_df], axis=1)
            
            df_cleaned['SpendingTier'] = pd.cut(df_raw['TotalCharges'], bins=[0, 1000, 4000, 10000], labels=[0, 1, 2], include_lowest=True).astype(int)
            df_cleaned['TenureGroup'] = pd.cut(df_raw['tenure'], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
            
            df_cluster_inp = df_cleaned[feature_config['df_cluster_cols']]
            df_scaled = scaler_cluster.transform(df_cluster_inp)
            clusters = kmeans_final.predict(df_scaled)
            
            # Reduce with PCA
            pca = PCA(n_components=2)
            df_pca = pca.fit_transform(df_scaled)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {text['clust_title']}")
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='Set1', alpha=0.5, s=12)
                legend_labels = ['Loyal High Value / عملاء أوفياء', 'Mid Value Developing / عملاء متوسطين', 'New Low Value / عملاء جدد']
                legend1 = ax.legend(scatter.legend_elements()[0], legend_labels, title="Cluster / الشريحة")
                ax.add_artist(legend1)
                st.pyplot(fig)
                
            with c2:
                st.markdown(f"#### {text['actual_clust_title']}")
                fig, ax = plt.subplots(figsize=(6, 5))
                churn_labels = df_raw['Churn'].map({'Yes': 1, 'No': 0}).values
                scatter2 = ax.scatter(df_pca[:, 0], df_pca[:, 1], c=churn_labels, cmap='coolwarm', alpha=0.5, s=12)
                legend2 = ax.legend(scatter2.legend_elements()[0], ['No Churn / لم يغادر', 'Churned / غادر'], title="Churn / المغادرة")
                ax.add_artist(legend2)
                st.pyplot(fig)

    # Model Performance
    elif menu == text["menu_opts"][3]: # Model performance
        st.markdown(f"# {text['model_title']}")
        st.write(text["model_subtitle"])

        if xgb_model is None:
            st.warning(text["no_models"])
        else:
            # Reconstruct and scale X_test for curves evaluation
            df_prep = df_raw.copy().drop(columns=['customerID', 'gender', 'Churn'])
            df_prep[feature_config['binary_columns']] = df_prep[feature_config['binary_columns']].replace({'Yes': 1, 'No': 0})
            df_prep['Contract'] = df_prep['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
            
            nom_df = pd.DataFrame(encoder.transform(df_raw[['MultipleLines', 'InternetService', 'PaymentMethod']]), 
                                  columns=encoder.get_feature_names_out(['MultipleLines', 'InternetService', 'PaymentMethod']))
            df_prep = pd.concat([df_prep.drop(columns=['MultipleLines', 'InternetService', 'PaymentMethod']), nom_df], axis=1)
            
            df_prep['SpendingTier'] = pd.cut(df_raw['TotalCharges'], bins=[0, 1000, 4000, 10000], labels=[0, 1, 2], include_lowest=True).astype(int)
            df_prep['TenureGroup'] = pd.cut(df_raw['tenure'], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
            
            # Predict clusters
            df_clust_scale = scaler_cluster.transform(df_prep[feature_config['df_cluster_cols']])
            df_prep['Cluster'] = kmeans_final.predict(df_clust_scale)
            
            for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
                df_prep[col] = df_prep[col].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0)
                
            df_prep = df_prep.apply(pd.to_numeric, errors='coerce').fillna(0)
            df_features = df_prep[feature_config['X_train_cols']]
            
            X_tr, X_te, y_tr, y_te = train_test_split(df_features, df_raw['Churn'].map({'Yes': 1, 'No': 0}), test_size=0.2, random_state=42, stratify=df_raw['Churn'].map({'Yes': 1, 'No': 0}))
            
            X_tr_sc = X_tr.copy()
            X_te_sc = X_te.copy()
            X_tr_sc[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_tr[['tenure', 'MonthlyCharges', 'TotalCharges']])
            X_te_sc[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_te[['tenure', 'MonthlyCharges', 'TotalCharges']])
            
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
                st.markdown(f"#### {text['roc_title']}")
                fig, ax = plt.subplots(figsize=(6, 5))
                
                xgb_proba = xgb_model.predict_proba(X_te_sc)[:, 1]
                fpr, tpr, _ = roc_curve(y_te, xgb_proba)
                ax.plot(fpr, tpr, label=f"XGBoost (AUC={roc_auc_score(y_te, xgb_proba):.3f})")
                
                svm_proba = svm.predict_proba(X_te_sc)[:, 1]
                fpr_s, tpr_s, _ = roc_curve(y_te, svm_proba)
                ax.plot(fpr_s, tpr_s, label=f"SVM (AUC={roc_auc_score(y_te, svm_proba):.3f})")
                
                lr_proba = lr.predict_proba(X_te_sc)[:, 1]
                fpr_l, tpr_l, _ = roc_curve(y_te, lr_proba)
                ax.plot(fpr_l, tpr_l, label=f"Logistic Regression (AUC={roc_auc_score(y_te, lr_proba):.3f})")
                
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend()
                st.pyplot(fig)
                
            with c2:
                st.markdown(f"#### {text['pr_title']}")
                fig, ax = plt.subplots(figsize=(6, 5))
                
                p, r, _ = precision_recall_curve(y_te, xgb_proba)
                ax.plot(r, p, label="XGBoost")
                
                p_s, r_s, _ = precision_recall_curve(y_te, svm_proba)
                ax.plot(r_s, p_s, label="SVM")
                
                p_l, r_l, _ = precision_recall_curve(y_te, lr_proba)
                ax.plot(r_l, p_l, label="Logistic Regression")
                
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.legend()
                st.pyplot(fig)
                
            st.markdown(f"#### {text['feat_imp_title']}")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            importances = xgb_model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': feature_config['X_train_cols'], 'Importance': importances}).sort_values(by='Importance', ascending=False)
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='viridis', ax=ax2)
            st.pyplot(fig2)

    # Churn Risk Predictor
    elif menu == text["menu_opts"][4]: # Predictor
        st.markdown(f"# {text['predictor_title']}")
        st.write(text["predictor_subtitle"])
        
        if xgb_model is None:
            st.warning(text["no_models"])
        else:
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.subheader(text["sec_demo"])
                    # Translate input values to align with English/Arabic forms
                    senior_opt = ["No", "Yes"] if lang_code == "en" else ["لا", "نعم"]
                    senior = st.selectbox(text["lbl_senior"], senior_opt)
                    
                    partner_opt = ["No", "Yes"] if lang_code == "en" else ["لا", "نعم"]
                    partner = st.selectbox(text["lbl_partner"], partner_opt)
                    
                    dep_opt = ["No", "Yes"] if lang_code == "en" else ["لا", "نعم"]
                    dependents = st.selectbox(text["lbl_dependents"], dep_opt)
                    
                    gender_opt = ["Female", "Male"] if lang_code == "en" else ["أنثى", "ذكر"]
                    gender = st.selectbox(text["lbl_gender"], gender_opt)
                    
                with c2:
                    st.subheader(text["sec_contract"])
                    tenure = st.slider(text["lbl_tenure"], min_value=0, max_value=72, value=12)
                    
                    contract_opt = ["Month-to-month", "One year", "Two year"] if lang_code == "en" else ["شهري", "سنة واحدة", "سنتين"]
                    contract = st.selectbox(text["lbl_contract"], contract_opt)
                    
                    paperless_opt = ["No", "Yes"] if lang_code == "en" else ["لا", "نعم"]
                    paperless = st.selectbox(text["lbl_paperless"], paperless_opt)
                    
                    payment_opt = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"] if lang_code == "en" else ["شيك إلكتروني", "شيك ورقي بريدي", "تحويل بنكي تلقائي", "بطاقة ائتمان تلقائية"]
                    payment = st.selectbox(text["lbl_payment"], payment_opt)
                    
                    monthly_charges = st.number_input(text["lbl_monthly"], min_value=10.0, max_value=150.0, value=70.0)
                    total_charges = st.number_input(text["lbl_total"], min_value=0.0, max_value=10000.0, value=tenure * monthly_charges)

                with c3:
                    st.subheader(text["sec_services"])
                    phone_opt = ["No", "Yes"] if lang_code == "en" else ["لا", "نعم"]
                    phone = st.selectbox(text["lbl_phone"], phone_opt)
                    
                    ml_opt = ["No phone service", "No", "Yes"] if lang_code == "en" else ["لا توجد خدمة هاتف", "لا", "نعم"]
                    multiple_lines = st.selectbox(text["lbl_multiple"], ml_opt if phone == ("No" if lang_code == "en" else "لا") else ml_opt[1:])
                    
                    internet_opt = ["DSL", "Fiber optic", "No"] if lang_code == "en" else ["DSL", "ألياف ضوئية", "لا توجد"]
                    internet = st.selectbox(text["lbl_internet"], internet_opt)
                    
                    yn_opt = ["No", "Yes"] if lang_code == "en" else ["لا", "نعم"]
                    no_internet_val = "No internet service" if lang_code == "en" else "No internet service"
                    
                    if internet != ("No" if lang_code == "en" else "لا توجد"):
                        security = st.selectbox(text["lbl_security"], yn_opt)
                        backup = st.selectbox(text["lbl_backup"], yn_opt)
                        protection = st.selectbox(text["lbl_protection"], yn_opt)
                        support = st.selectbox(text["lbl_support"], yn_opt)
                        tv = st.selectbox(text["lbl_tv"], yn_opt)
                        movies = st.selectbox(text["lbl_movies"], yn_opt)
                    else:
                        security, backup, protection, support, tv, movies = no_internet_val, no_internet_val, no_internet_val, no_internet_val, no_internet_val, no_internet_val

                submit_btn = st.form_submit_button(text["btn_predict"])

            if submit_btn:
                # 1. Map values back to English labels for modeling
                map_yes_no = {"Yes": 1, "No": 0, "نعم": 1, "لا": 0}
                
                inputs = {
                    'SeniorCitizen': map_yes_no[senior],
                    'Partner': map_yes_no[partner],
                    'Dependents': map_yes_no[dependents],
                    'tenure': tenure,
                    'PhoneService': map_yes_no[phone],
                    'Contract': 0 if contract in ["Month-to-month", "شهري"] else (1 if contract in ["One year", "سنة واحدة"] else 2),
                    'PaperlessBilling': map_yes_no[paperless],
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges,
                    'MultipleLines': "No phone service" if multiple_lines in ["No phone service", "لا توجد خدمة هاتف"] else ("Yes" if multiple_lines in ["Yes", "نعم"] else "No"),
                    'InternetService': "DSL" if internet == "DSL" else ("Fiber optic" if internet in ["Fiber optic", "ألياف ضوئية"] else "No"),
                    'PaymentMethod': "Electronic check" if payment in ["Electronic check", "شيك إلكتروني"] else (
                                     "Mailed check" if payment in ["Mailed check", "شيك ورقي بريدي"] else (
                                     "Bank transfer (automatic)" if payment in ["Bank transfer (automatic)", "تحويل بنكي تلقائي"] else "Credit card (automatic)")
                                     )
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
                sec_val = 1 if security in ["Yes", "نعم"] else 0
                bk_val = 1 if backup in ["Yes", "نعم"] else 0
                prot_val = 1 if protection in ["Yes", "نعم"] else 0
                sup_val = 1 if support in ["Yes", "نعم"] else 0
                tv_val = 1 if tv in ["Yes", "نعم"] else 0
                mov_val = 1 if movies in ["Yes", "نعم"] else 0
                
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
                st.subheader(text["res_header"])
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.write(f"**{text['res_cluster']}** `{predicted_cluster}`")
                    if predicted_cluster == 0:
                        st.success(text["segment_loyal"])
                    elif predicted_cluster == 1:
                        st.info(text["segment_developing"])
                    else:
                        st.warning(text["segment_new"])
                        
                with col_res2:
                    st.write(f"**{text['res_prob']}** `{proba:.2%}`")
                    if proba >= 0.6:
                        st.error(text["risk_high"])
                    elif proba >= 0.3:
                        st.warning(text["risk_med"])
                    else:
                        st.success(text["risk_low"])
                
                st.progress(float(proba))
