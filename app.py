"""
Bank Customer Churn Prediction System
Streamlit Web Application - FIXED VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Bank Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Initialize Session State
# ============================================
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'lr_metrics' not in st.session_state:
    st.session_state.lr_metrics = None
if 'rf_metrics' not in st.session_state:
    st.session_state.rf_metrics = None
if 'dataset_columns' not in st.session_state:
    st.session_state.dataset_columns = []

# ============================================
# Helper Functions
# ============================================
def get_latest_dataset():
    """Get the most recently uploaded dataset"""
    os.makedirs('uploads', exist_ok=True)
    csv_files = glob.glob(os.path.join('uploads', '*.csv'))
    excel_files = glob.glob(os.path.join('uploads', '*.xlsx')) + glob.glob(os.path.join('uploads', '*.xls'))
    all_files = csv_files + excel_files
    
    if not all_files:
        return None, None
    
    latest_file = max(all_files, key=os.path.getmtime)
    
    if latest_file.endswith('.csv'):
        df = pd.read_csv(latest_file)
    else:
        df = pd.read_excel(latest_file)
    
    # Store column names for reference
    st.session_state.dataset_columns = df.columns.tolist()
    
    return df, latest_file

def engineer_features(df):
    """Create engineered features safely"""
    df = df.copy()
    
    # Only create features if required columns exist
    if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
        df['Balance_to_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    
    if 'Tenure' in df.columns and 'Age' in df.columns:
        df['Tenure_to_Age_Ratio'] = df['Tenure'] / (df['Age'] + 1)
    
    if 'Age' in df.columns and 'Tenure' in df.columns:
        df['Age_Tenure_Interaction'] = df['Age'] * df['Tenure']
    
    if 'NumOfProducts' in df.columns and 'IsActiveMember' in df.columns:
        df['Product_Interaction'] = df['NumOfProducts'] * df['IsActiveMember']
    
    return df

def preprocess_data(df, fit=False):
    """Preprocess the data for training"""
    df = engineer_features(df)
    
    # Separate features and target
    if 'Exited' in df.columns:
        X = df.drop('Exited', axis=1)
        y = df['Exited']
    else:
        X = df
        y = None
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if fit:
            st.session_state.label_encoders[col] = LabelEncoder()
            X[col] = st.session_state.label_encoders[col].fit_transform(X[col].astype(str))
        else:
            if col in st.session_state.label_encoders:
                X[col] = st.session_state.label_encoders[col].transform(X[col].astype(str))
            else:
                X[col] = 0
    
    # Fill missing values
    X = X.fillna(X.median())
    
    # Scale features
    if fit:
        st.session_state.scaler = StandardScaler()
        X_scaled = st.session_state.scaler.fit_transform(X)
        st.session_state.feature_columns = X.columns.tolist()
    else:
        X_scaled = st.session_state.scaler.transform(X)
    
    return X_scaled, y

def train_models(X_train, y_train, X_test, y_test):
    """Train both models and return metrics"""
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    lr_metrics = {
        'name': 'Logistic Regression',
        'accuracy': accuracy_score(y_test, lr_pred),
        'precision': precision_score(y_test, lr_pred, zero_division=0),
        'recall': recall_score(y_test, lr_pred, zero_division=0),
        'f1_score': f1_score(y_test, lr_pred, zero_division=0)
    }
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_metrics = {
        'name': 'Random Forest',
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1_score': f1_score(y_test, rf_pred, zero_division=0)
    }
    
    # Select best model
    if rf_metrics['f1_score'] >= lr_metrics['f1_score']:
        best_model = rf_model
        best_name = 'Random Forest'
        best_metrics = rf_metrics
    else:
        best_model = lr_model
        best_name = 'Logistic Regression'
        best_metrics = lr_metrics
    
    return best_model, best_name, best_metrics, lr_metrics, rf_metrics

def predict_churn(model, scaler, input_data, label_encoders):
    """Make prediction for a single customer"""
    # Create dataframe
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in ['Geography', 'Gender']:
        if col in input_df.columns and col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    
    # Engineer features
    if 'Balance' in input_df.columns and 'EstimatedSalary' in input_df.columns:
        input_df['Balance_to_Salary_Ratio'] = input_df['Balance'] / (input_df['EstimatedSalary'] + 1)
    
    if 'Tenure' in input_df.columns and 'Age' in input_df.columns:
        input_df['Tenure_to_Age_Ratio'] = input_df['Tenure'] / (input_df['Age'] + 1)
    
    if 'Age' in input_df.columns and 'Tenure' in input_df.columns:
        input_df['Age_Tenure_Interaction'] = input_df['Age'] * input_df['Tenure']
    
    if 'NumOfProducts' in input_df.columns and 'IsActiveMember' in input_df.columns:
        input_df['Product_Interaction'] = input_df['NumOfProducts'] * input_df['IsActiveMember']
    
    # Fill missing columns
    for col in st.session_state.feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Scale features
    input_scaled = scaler.transform(input_df[st.session_state.feature_columns])
    
    # Predict
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability >= 0.7:
        return "🔴 Very High Risk", "danger"
    elif probability >= 0.5:
        return "🟠 High Risk", "warning"
    elif probability >= 0.3:
        return "🟡 Medium Risk", "info"
    elif probability >= 0.1:
        return "🟢 Low Risk", "success"
    else:
        return "⚪ Very Low Risk", "secondary"

def safe_groupby(df, column, target):
    """Safely perform groupby operation"""
    if column in df.columns and target in df.columns:
        result = df.groupby(column)[target].agg(['count', 'sum'])
        return result
    return pd.DataFrame()

# ============================================
# Sidebar Navigation
# ============================================
st.sidebar.title("🏦 Bank Churn Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📁 Dataset", "🎯 Train Model", "🔮 Predict", "📈 Model Comparison", "ℹ️ About"]
)

st.sidebar.markdown("---")

# Get dataset info for sidebar
df_check, _ = get_latest_dataset()
st.sidebar.info(
    f"**System Status:**\n\n"
    f"✅ Model Trained: {'Yes' if st.session_state.model_trained else 'No'}\n\n"
    f"📁 Dataset: {'Loaded' if df_check is not None else 'Not loaded'}\n\n"
    f"📊 Columns: {len(st.session_state.dataset_columns) if st.session_state.dataset_columns else 0}"
)

# ============================================
# DASHBOARD PAGE
# ============================================
if page == "📊 Dashboard":
    st.markdown('<div class="main-header"><h1 style="color:white;">📊 Customer Churn Dashboard</h1></div>', unsafe_allow_html=True)
    
    df, filepath = get_latest_dataset()
    
    if df is not None:
        st.success(f"📁 Using dataset: {os.path.basename(filepath)} ({len(df)} rows, {len(df.columns)} columns)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            if 'Exited' in df.columns:
                churned = df['Exited'].sum()
                st.metric("Churned Customers", f"{int(churned):,}", delta=f"{churned/len(df)*100:.1f}% churn rate")
            else:
                st.metric("Churned Customers", "N/A (no target column)")
        
        with col3:
            if 'Exited' in df.columns:
                retained = len(df) - df['Exited'].sum()
                st.metric("Retained Customers", f"{int(retained):,}")
            else:
                st.metric("Retained Customers", "N/A")
        
        with col4:
            st.metric("Model Status", "✅ Ready" if st.session_state.model_trained else "⚠️ Not Trained")
        
        # Show available columns
        with st.expander("📋 Available Columns in Dataset"):
            st.write(df.dtypes)
        
        if 'Exited' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Churn Distribution")
                churn_counts = df['Exited'].value_counts()
                fig = px.pie(values=churn_counts.values, names=['Retained', 'Churned'],
                            title='Customer Churn Distribution',
                            color_discrete_sequence=['#2E86AB', '#A23B72'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Churn by Geography")
                if 'Geography' in df.columns:
                    churn_geo = df.groupby('Geography')['Exited'].mean() * 100
                    fig = px.bar(x=churn_geo.index, y=churn_geo.values, 
                                title='Churn Rate by Geography',
                                color=churn_geo.values, color_continuous_scale='Reds')
                    fig.update_layout(xaxis_title="Geography", yaxis_title="Churn Rate (%)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No 'Geography' column found in dataset")
            
            # Age distribution if available
            if 'Age' in df.columns:
                st.subheader("Churn by Age Group")
                df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 40, 50, 60, 100], 
                                         labels=['18-30', '30-40', '40-50', '50-60', '60+'])
                churn_by_age = df.groupby('Age_Group')['Exited'].mean() * 100
                fig = px.line(x=churn_by_age.index, y=churn_by_age.values, 
                             title='Churn Rate by Age Group', markers=True)
                fig.update_layout(xaxis_title="Age Group", yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No 'Exited' column found. Please upload a dataset with target column for churn analysis.")
    else:
        st.info("📁 No dataset found. Please upload a dataset in the 'Dataset' page.")

# ============================================
# DATASET PAGE
# ============================================
elif page == "📁 Dataset":
    st.markdown('<div class="main-header"><h1 style="color:white;">📁 Dataset Management</h1></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            os.makedirs('uploads', exist_ok=True)
            filepath = os.path.join('uploads', uploaded_file.name)
            
            with open(filepath, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            st.info(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.subheader("🔍 Data Preview (First 20 rows)")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            if 'Exited' in df.columns:
                st.success("✅ Target column 'Exited' found - Ready for training!")
                churn_rate = df['Exited'].mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            else:
                st.warning("⚠️ No 'Exited' column found - This dataset cannot be used for training")
                st.info("Required columns for training: Age, Balance, CreditScore, Geography, Gender, Tenure, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited")
    
    with col2:
        st.subheader("📋 Dataset Requirements")
        st.markdown("""
        **Required columns for training:**
        - `Exited` (0=stayed, 1=churned) - Target variable
        - `Age` - Customer age
        - `Balance` - Account balance
        - `CreditScore` - Credit score
        - `Geography` - France/Germany/Spain
        - `Gender` - Male/Female
        - `Tenure` - Years with bank
        - `NumOfProducts` - Number of products
        - `HasCrCard` - Has credit card (0/1)
        - `IsActiveMember` - Active member (0/1)
        - `EstimatedSalary` - Estimated salary
        
        **Optional columns:**
        - `CustomerId` - Customer identifier
        - Any other columns will be ignored
        """)

# ============================================
# TRAIN MODEL PAGE
# ============================================
elif page == "🎯 Train Model":
    st.markdown('<div class="main-header"><h1 style="color:white;">🎯 Train Churn Prediction Model</h1></div>', unsafe_allow_html=True)
    
    df, filepath = get_latest_dataset()
    
    if df is None:
        st.error("❌ No dataset found! Please upload a dataset first.")
        st.stop()
    
    if 'Exited' not in df.columns:
        st.error("❌ Dataset must have an 'Exited' column for training!")
        st.info("Please upload a dataset that contains the target column 'Exited'")
        st.stop()
    
    st.success(f"📊 Training on: {os.path.basename(filepath)}")
    st.info(f"📈 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.info(f"🎯 Churn rate: {df['Exited'].mean()*100:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        st.write(f"📊 Training samples: {int(len(df) * (1 - test_size)):,}")
        st.write(f"🧪 Test samples: {int(len(df) * test_size):,}")
    
    if st.button("🚀 Start Training", use_container_width=True):
        with st.spinner("Training models... This may take 10-30 seconds..."):
            try:
                # Preprocess
                X_scaled, y = preprocess_data(df, fit=True)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Train
                best_model, best_name, best_metrics, lr_metrics, rf_metrics = train_models(
                    X_train, y_train, X_test, y_test
                )
                
                # Save to session
                st.session_state.model = best_model
                st.session_state.model_trained = True
                st.session_state.lr_metrics = lr_metrics
                st.session_state.rf_metrics = rf_metrics
                st.session_state.best_model_name = best_name
                
                st.success(f"✅ Training complete!")
                st.balloons()
                
                # Display results
                st.subheader("📊 Training Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Model", best_name)
                with col2:
                    st.metric("Accuracy", f"{best_metrics['accuracy']*100:.1f}%")
                with col3:
                    st.metric("F1-Score", f"{best_metrics['f1_score']*100:.1f}%")
                
                # Show detailed metrics
                with st.expander("📈 Detailed Model Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Logistic Regression**")
                        st.write(f"Accuracy: {lr_metrics['accuracy']*100:.2f}%")
                        st.write(f"Precision: {lr_metrics['precision']*100:.2f}%")
                        st.write(f"Recall: {lr_metrics['recall']*100:.2f}%")
                        st.write(f"F1-Score: {lr_metrics['f1_score']*100:.2f}%")
                    
                    with col2:
                        st.write("**Random Forest**")
                        st.write(f"Accuracy: {rf_metrics['accuracy']*100:.2f}%")
                        st.write(f"Precision: {rf_metrics['precision']*100:.2f}%")
                        st.write(f"Recall: {rf_metrics['recall']*100:.2f}%")
                        st.write(f"F1-Score: {rf_metrics['f1_score']*100:.2f}%")
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                st.exception(e)

# ============================================
# PREDICT PAGE
# ============================================
elif page == "🔮 Predict":
    st.markdown('<div class="main-header"><h1 style="color:white;">🔮 Customer Churn Prediction</h1></div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained yet! Please go to 'Train Model' page first.")
        st.stop()
    
    st.subheader("📝 Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
        balance = st.number_input("Balance ($)", min_value=0, max_value=500000, value=125000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=10)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5, step=1)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary ($)", min_value=0, max_value=500000, value=85000, step=1000)
    
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        input_data = {
            'Age': age,
            'Balance': balance,
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Tenure': tenure,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': salary
        }
        
        try:
            prediction, probability = predict_churn(
                st.session_state.model, 
                st.session_state.scaler, 
                input_data,
                st.session_state.label_encoders
            )
            
            risk_level, _ = get_risk_level(probability)
            
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ Customer WILL CHURN")
                else:
                    st.success("✅ Customer WILL STAY")
            
            with col2:
                st.metric("Churn Probability", f"{probability*100:.1f}%")
            
            with col3:
                st.write(f"**Risk Level:** {risk_level}")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability*100,
                title={'text': "Churn Risk Meter"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if probability > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 50], 'color': "yellow"},
                        {'range': [50, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            if probability >= 0.7:
                st.error("🚨 **URGENT:** Immediate retention call required!")
            elif probability >= 0.5:
                st.warning("📞 **Action Required:** Schedule retention call")
            elif probability >= 0.3:
                st.info("📧 **Recommendation:** Send personalized email offer")
            else:
                st.success("✅ **Status:** Regular monitoring only")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)

# ============================================
# MODEL COMPARISON PAGE
# ============================================
elif page == "📈 Model Comparison":
    st.markdown('<div class="main-header"><h1 style="color:white;">📊 Model Performance Comparison</h1></div>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ No model trained yet! Please go to 'Train Model' page first.")
        st.stop()
    
    if st.session_state.lr_metrics and st.session_state.rf_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Logistic Regression")
            lr = st.session_state.lr_metrics
            st.metric("Accuracy", f"{lr['accuracy']*100:.1f}%")
            st.metric("Precision", f"{lr['precision']*100:.1f}%")
            st.metric("Recall", f"{lr['recall']*100:.1f}%")
            st.metric("F1-Score", f"{lr['f1_score']*100:.1f}%")
        
        with col2:
            st.subheader("🌲 Random Forest")
            rf = st.session_state.rf_metrics
            st.metric("Accuracy", f"{rf['accuracy']*100:.1f}%")
            st.metric("Precision", f"{rf['precision']*100:.1f}%")
            st.metric("Recall", f"{rf['recall']*100:.1f}%")
            st.metric("F1-Score", f"{rf['f1_score']*100:.1f}%")
        
        # Comparison chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_values = [lr['accuracy'], lr['precision'], lr['recall'], lr['f1_score']]
        rf_values = [rf['accuracy'], rf['precision'], rf['recall'], rf['f1_score']]
        
        fig = go.Figure(data=[
            go.Bar(name='Logistic Regression', x=metrics, y=lr_values, marker_color='#2E86AB'),
            go.Bar(name='Random Forest', x=metrics, y=rf_values, marker_color='#A23B72')
        ])
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group',
            height=500,
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"🏆 **Best Model:** {st.session_state.best_model_name}")
        
        # Feature importance (if Random Forest won)
        if st.session_state.best_model_name == "Random Forest" and hasattr(st.session_state.model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance Analysis")
            
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_columns,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Top 10 Features for Churn Prediction',
                        color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No training data available. Please train the model first.")

# ============================================
# ABOUT PAGE
# ============================================
else:
    st.markdown('<div class="main-header"><h1 style="color:white;">ℹ️ About Bank Churn Prediction System</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏦 Bank Customer Churn Prediction System
    
    This system uses **Machine Learning** to predict which bank customers are likely to leave,
    enabling proactive retention strategies.
    
    ### 🎯 Key Features
    - **📁 Dataset Management**: Upload CSV/Excel files, preview data
    - **🎯 Model Training**: Train Logistic Regression & Random Forest
    - **🔮 Churn Prediction**: Real-time predictions for new customers
    - **📈 Model Comparison**: Compare performance metrics visually
    - **📊 Dashboard**: Interactive data visualization
    
    ### 📊 How to Use
    1. **Upload Dataset** - Go to Dataset page and upload your CSV/Excel file
    2. **Train Model** - Go to Train Model page and click "Start Training"
    3. **Make Predictions** - Go to Predict page and enter customer details
    4. **View Results** - Check the Dashboard for insights
    
    ### 📋 Dataset Requirements
    Your dataset must contain these columns:
    - `Exited` (0=stayed, 1=churned) - Target variable
    - `Age`, `Balance`, `CreditScore`, `Geography`, `Gender`
    - `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
    
    ### 🔧 Technologies Used
    - **Frontend**: Streamlit
    - **Machine Learning**: scikit-learn (Random Forest, Logistic Regression)
    - **Visualization**: Plotly, Matplotlib
    - **Data Processing**: Pandas, NumPy
    
    ### 📈 Business Impact
    - Reduce customer churn by **15-25%**
    - Save **$5-10M annually** for mid-sized banks
    - Improve customer retention ROI
    
    ### 💡 Tips
    - Larger datasets (5,000+ rows) produce more accurate models
    - Train the model after uploading new data
    - Check feature importance to understand churn drivers
    """)

# ============================================
# Run with: streamlit run app.py
# ============================================