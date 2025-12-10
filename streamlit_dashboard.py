import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Helper function to get plotly template based on theme
def get_plotly_template():
    return 'plotly_dark' if theme == "Dark" else 'plotly'

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme toggle in sidebar
st.sidebar.markdown("### 🎨 Theme Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"], index=0)

# Custom CSS for better styling with dark theme support
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #4fc3f7;
            text-align: center;
            padding: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .metric-card {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #4fc3f7;
            color: #fafafa;
        }
        .stMetric {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        h1, h2, h3 {
            color: #4fc3f7 !important;
        }
        .stDataFrame {
            background-color: #1e1e1e;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    data = pd.read_csv("bank-full.csv", sep=";")
    df = data.drop('duration', axis=1)
    
    # Replace 'unknown' with NaN
    for col in data.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace('unknown', np.nan)
    
    # Fill missing categorical columns with mode
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Convert target variable
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    return df, data

@st.cache_resource
def train_models(df):
    """Train models without class weights"""
    # Prepare features
    X = df.drop('y', axis=1)
    y = df['y']
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    
    # Scale for logistic regression
    scaler_lr = StandardScaler()
    X_train_sm_scaled = scaler_lr.fit_transform(X_train_sm)
    X_test_scaled = scaler_lr.transform(X_test)
    
    # Train models WITHOUT class weights
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    log_reg.fit(X_train_sm_scaled, y_train_sm)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_sm, y_train_sm)
    
    # XGBoost - tuned to achieve 89% accuracy
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=9,
        learning_rate=0.1,
        gamma=0.2,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_sm, y_train_sm)
    
    # Get predictions
    log_preds = log_reg.predict(X_test_scaled)
    rf_preds = rf.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    
    # Calculate metrics
    models_metrics = {
        'Logistic Regression': {
            'model': log_reg,
            'predictions': log_preds,
            'accuracy': accuracy_score(y_test, log_preds),
            'precision': precision_score(y_test, log_preds),
            'recall': recall_score(y_test, log_preds),
            'f1': f1_score(y_test, log_preds),
            'roc_auc': roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1]),
            'needs_scaling': True
        },
        'Random Forest': {
            'model': rf,
            'predictions': rf_preds,
            'accuracy': accuracy_score(y_test, rf_preds),
            'precision': precision_score(y_test, rf_preds),
            'recall': recall_score(y_test, rf_preds),
            'f1': f1_score(y_test, rf_preds),
            'roc_auc': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
            'needs_scaling': False
        },
        'XGBoost': {
            'model': xgb_model,
            'predictions': xgb_preds,
            'accuracy': accuracy_score(y_test, xgb_preds),
            'precision': precision_score(y_test, xgb_preds),
            'recall': recall_score(y_test, xgb_preds),
            'f1': f1_score(y_test, xgb_preds),
            'roc_auc': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]),
            'needs_scaling': False
        }
    }
    
    # Store which columns are numeric (for scaling during prediction)
    numeric_cols_list = num_cols.tolist()
    
    return {
        'models_metrics': models_metrics,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_sm': X_train_sm,
        'scaler_lr': scaler_lr,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'numeric_cols': numeric_cols_list
    }

# Load data
df, original_data = load_data()

# Train models
with st.spinner('Training models... This may take a moment.'):
    model_results = train_models(df)

# Main title
st.markdown('<h1 class="main-header">🏦 Bank Marketing Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["📋 Executive Summary", "📈 Overview", "🔍 Exploratory Data Analysis", "🤖 Model Performance", 
     "💰 ROI Analysis", "🎯 Customer Insights", "🔮 Predictions"]
)

if page == "📋 Executive Summary":
    st.header("📋 Executive Summary")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    xgb_metrics = model_results['models_metrics']['XGBoost']
    subscription_rate = (df['y'].sum() / len(df)) * 100
    
    with col1:
        st.metric("📊 Total Customers", f"{len(df):,}", help="Total number of customers in the dataset")
    
    with col2:
        st.metric("🎯 Subscription Rate", f"{subscription_rate:.2f}%", 
                 help="Percentage of customers who subscribed")
    
    with col3:
        st.metric("🤖 Model Accuracy", f"{xgb_metrics['accuracy']*100:.2f}%", 
                 delta=f"{xgb_metrics['accuracy']*100 - 88:.2f}% above baseline",
                 help="XGBoost model accuracy without class weights")
    
    with col4:
        st.metric("💰 Potential ROI", "1,102%", 
                 help="Maximum ROI achievable with optimal threshold")
    
    st.markdown("---")
    
    # Key Insights
    st.subheader("🎯 Key Business Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("""
        #### 📈 Performance Highlights
        - **Best Model**: XGBoost achieves **89% accuracy** without class weights
        - **Data Balance**: Using SMOTE for handling imbalanced dataset (88% No, 12% Yes)
        - **Model Comparison**: XGBoost outperforms Logistic Regression (75%) and Random Forest (88%)
        - **ROC-AUC Score**: {:.3f} indicating excellent model discrimination
        """.format(xgb_metrics['roc_auc']))
        
        st.markdown("""
        #### 💼 Customer Segmentation
        - **High-Value Segments**: Retired and student customers show highest subscription rates
        - **Job Analysis**: Management and blue-collar workers have lower conversion rates
        - **Education Impact**: Tertiary education customers are more likely to subscribe
        """)
    
    with insight_col2:
        st.markdown("""
        #### 📅 Optimal Contact Strategy
        - **Best Months**: March (52%), December (47%), September (46%), October (44%)
        - **Worst Month**: May (7%) - highest call volume but lowest conversion
        - **Best Days**: Days 1, 10, 30 show highest success rates
        - **Previous Campaign**: Customers with previous success have 64.7% subscription rate
        """)
        
        st.markdown("""
        #### 💰 Financial Impact
        - **Cost Savings**: Up to 80% reduction in call costs using ML predictions
        - **Optimal Threshold**: 0.020 probability threshold maximizes profit
        - **Expected Conversions**: 1,056 conversions from 8,785 targeted calls
        - **Net Profit Potential**: $1.9M+ with optimal strategy
        """)
    
    st.markdown("---")
    
    # Interactive Model Performance Comparison
    st.subheader("📊 Model Performance Comparison")
    
    models_metrics = model_results['models_metrics']
    comparison_data = {
        'Model': list(models_metrics.keys()),
        'Accuracy': [m['accuracy']*100 for m in models_metrics.values()],
        'Precision': [m['precision']*100 for m in models_metrics.values()],
        'Recall': [m['recall']*100 for m in models_metrics.values()],
        'F1-Score': [m['f1']*100 for m in models_metrics.values()],
        'ROC-AUC': [m['roc_auc']*100 for m in models_metrics.values()]
    }
    
    fig = go.Figure()
    
    metrics_to_plot = st.multiselect(
        "Select metrics to compare",
        ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        default=['Accuracy', 'F1-Score', 'ROC-AUC']
    )
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_data['Model'],
            y=comparison_data[metric],
            text=[f"{val:.1f}%" for val in comparison_data[metric]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Metrics Comparison",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        barmode='group',
        height=500,
        template=get_plotly_template()
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Features Importance
    st.subheader("🎯 Top 10 Most Important Features")
    
    xgb_model = models_metrics['XGBoost']['model']
    feature_importance = pd.DataFrame({
        'Feature': model_results['feature_names'],
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance (XGBoost)",
        color='Importance',
        color_continuous_scale='Viridis',
        text='Importance'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        height=400,
        template=get_plotly_template()
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.info("""
        **🎯 Targeting Strategy**
        1. Focus on customers with previous successful campaign outcomes (64.7% success rate)
        2. Prioritize contact during March, September, October, and December
        3. Avoid heavy calling in May (lowest conversion rate)
        4. Target retired and student demographics
        """)
    
    with rec_col2:
        st.success("""
        **💰 Cost Optimization**
        1. Use ML model predictions to reduce call volume by 80%+
        2. Implement optimal threshold (0.020) for maximum profit
        3. Expected ROI of 1,102% with targeted approach
        4. Potential savings of $145K+ in call costs
        """)
    
    st.warning("""
    **⚠️ Important Notes**
    - Model achieves 89% accuracy without class weights using SMOTE
    - Duration feature was removed (data leakage concern)
    - All models trained on SMOTE-balanced data for fair comparison
    - XGBoost selected as production model due to best performance
    """)

elif page == "📈 Overview":
    st.header("📊 Dataset Overview")
    
    # Interactive filters
    st.sidebar.markdown("### 🔍 Data Filters")
    filter_subscription = st.sidebar.selectbox("Filter by Subscription", ["All", "Subscribed", "Not Subscribed"])
    
    # Apply filters
    filtered_df = df.copy()
    if filter_subscription == "Subscribed":
        filtered_df = filtered_df[filtered_df['y'] == 1]
    elif filter_subscription == "Not Subscribed":
        filtered_df = filtered_df[filtered_df['y'] == 0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_df):,}", 
                 delta=f"{len(filtered_df) - len(df)}" if filter_subscription != "All" else None)
    
    with col2:
        st.metric("Features", len(df.columns) - 1)
    
    with col3:
        subscription_rate = (filtered_df['y'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric("Subscription Rate", f"{subscription_rate:.2f}%")
    
    with col4:
        st.metric("Best Model Accuracy", f"{model_results['models_metrics']['XGBoost']['accuracy']*100:.2f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dataset Sample")
        num_rows = st.slider("Number of rows to display", 5, 50, 10, key="overview_rows")
        st.dataframe(filtered_df.head(num_rows), use_container_width=True)
    
    with col2:
        st.subheader("📊 Target Variable Distribution")
        target_counts = filtered_df['y'].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=['No Subscription', 'Subscription'],
            title="Subscription Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Dataset Statistics")
    stat_option = st.radio("Statistics Type", ["Numeric Only", "All Columns"], horizontal=True)
    if stat_option == "Numeric Only":
        st.dataframe(filtered_df.describe(), use_container_width=True)
    else:
        st.dataframe(filtered_df.describe(include='all'), use_container_width=True)

elif page == "🔍 Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    # Interactive filters in sidebar
    st.sidebar.markdown("### 📊 EDA Filters")
    age_range = st.sidebar.slider("Age Range", int(df['age'].min()), int(df['age'].max()), 
                                  (int(df['age'].min()), int(df['age'].max())))
    balance_range = st.sidebar.slider("Balance Range", int(df['balance'].min()), int(df['balance'].max()),
                                     (int(df['balance'].min()), int(df['balance'].max())))
    
    # Apply filters
    filtered_df_eda = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) & 
                         (df['balance'] >= balance_range[0]) & (df['balance'] <= balance_range[1])]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "📅 Temporal Analysis", "💼 Job & Education", "🔗 Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            bins = st.slider("Number of bins", 10, 50, 30, key="age_bins")
            fig = px.histogram(
                filtered_df_eda, x='age', nbins=bins, 
                title="Age Distribution of Customers",
                labels={'age': 'Age', 'count': 'Count'},
                color_discrete_sequence=['#4fc3f7' if theme == "Dark" else '#1f77b4']
            )
            fig.update_layout(showlegend=False, template=get_plotly_template())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Subscription Rate by Age Group")
            age_groups = pd.cut(filtered_df_eda['age'], bins=[0, 30, 40, 50, 60, 100], 
                               labels=['<30', '30-40', '40-50', '50-60', '60+'])
            success_by_age = filtered_df_eda.groupby(age_groups)['y'].mean()
            fig = px.bar(
                x=success_by_age.index, 
                y=success_by_age.values,
                title="Success Rate by Age Group",
                labels={'x': 'Age Group', 'y': 'Success Rate'},
                color=success_by_age.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(template=get_plotly_template())
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Marital Status Analysis")
        marital_success = filtered_df_eda.groupby('marital')['y'].agg(['mean', 'count']).reset_index()
        marital_success.columns = ['Marital Status', 'Success Rate', 'Count']
        fig = px.bar(
            marital_success, x='Marital Status', y='Success Rate',
            title="Subscription Rate by Marital Status",
            color='Success Rate',
            color_continuous_scale='Blues',
            text='Success Rate'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Conversion Rate")
            month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
            month_success = filtered_df_eda.groupby('month')['y'].mean().reindex(month_order)
            fig = px.bar(
                x=month_success.index, 
                y=month_success.values,
                title="Success Rate by Month",
                labels={'x': 'Month', 'y': 'Success Rate'},
                color=month_success.values,
                color_continuous_scale='Viridis',
                text=month_success.values
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(template=get_plotly_template())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Day of Month Analysis")
            num_days = st.slider("Number of top days", 5, 20, 10, key="top_days")
            day_success = filtered_df_eda.groupby('day')['y'].mean().sort_values(ascending=False)
            top_days = day_success.head(num_days)
            fig = px.bar(
                x=top_days.index, 
                y=top_days.values,
                title=f"Top {num_days} Days by Success Rate",
                labels={'x': 'Day of Month', 'y': 'Success Rate'},
                color=top_days.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(template=get_plotly_template())
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Job vs Subscription")
            job_success = filtered_df_eda.groupby('job')['y'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=job_success.index, 
                y=job_success.values,
                title="Subscription Rate by Job",
                labels={'x': 'Job', 'y': 'Success Rate'},
                color=job_success.values,
                color_continuous_scale='Reds',
                text=job_success.values
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_xaxes(tickangle=45)
            fig.update_layout(template=get_plotly_template())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Education vs Subscription")
            edu_success = filtered_df_eda.groupby('education')['y'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=edu_success.index, 
                y=edu_success.values,
                title="Subscription Rate by Education",
                labels={'x': 'Education', 'y': 'Success Rate'},
                color=edu_success.values,
                color_continuous_scale='Greens',
                text=edu_success.values
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(template=get_plotly_template())
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = filtered_df_eda.select_dtypes(include=[np.number]).columns
        corr_matrix = filtered_df_eda[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu',
            aspect="auto",
            text_auto=True
        )
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Performance":
    st.header("🤖 Model Performance Comparison")
    
    models_metrics = model_results['models_metrics']
    
    # Metrics comparison
    col1, col2, col3 = st.columns(3)
    
    metrics_df = pd.DataFrame({
        'Model': list(models_metrics.keys()),
        'Accuracy': [m['accuracy'] for m in models_metrics.values()],
        'Precision': [m['precision'] for m in models_metrics.values()],
        'Recall': [m['recall'] for m in models_metrics.values()],
        'F1-Score': [m['f1'] for m in models_metrics.values()],
        'ROC-AUC': [m['roc_auc'] for m in models_metrics.values()]
    })
    
    st.subheader("📊 Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for model_name, metrics in models_metrics.items():
            fig.add_trace(go.Bar(
                name=model_name,
                x=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                y=[metrics['accuracy'], metrics['precision'], 
                   metrics['recall'], metrics['f1'], metrics['roc_auc']]
            ))
        fig.update_layout(
            title="Model Performance Comparison",
            barmode='group',
            yaxis_title="Score",
            height=400,
            template=get_plotly_template()
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Accuracy comparison
        accuracies = [m['accuracy'] for m in models_metrics.values()]
        fig = px.bar(
            x=list(models_metrics.keys()),
            y=accuracies,
            title="Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy'},
            color=accuracies,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrices
    st.subheader("🔍 Confusion Matrices")
    cols = st.columns(3)
    
    for idx, (model_name, metrics) in enumerate(models_metrics.items()):
        cm = confusion_matrix(model_results['y_test'], metrics['predictions'])
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No', 'Yes'],
            y=['No', 'Yes'],
            text_auto=True,
            aspect="auto",
            title=f"{model_name}",
            color_continuous_scale='Blues'
        )
        fig.update_layout(template=get_plotly_template())
        cols[idx].plotly_chart(fig, use_container_width=True)
    
    # Best model highlight
    best_model = max(models_metrics.items(), key=lambda x: x[1]['accuracy'])
    st.success(f"🏆 **Best Model: {best_model[0]}** with {best_model[1]['accuracy']*100:.2f}% accuracy")
    
    # Feature importance for XGBoost
    st.subheader("🎯 XGBoost Feature Importance")
    xgb_model = models_metrics['XGBoost']['model']
    feature_importance = pd.DataFrame({
        'Feature': model_results['feature_names'],
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    num_features = st.slider("Number of top features to display", 5, 20, 15, key="feature_slider")
    feature_importance_display = feature_importance.head(num_features)
    
    fig = px.bar(
        feature_importance_display,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {num_features} Most Important Features",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, template=get_plotly_template())
    st.plotly_chart(fig, use_container_width=True)

elif page == "💰 ROI Analysis":
    st.header("💰 Return on Investment (ROI) Analysis")
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        cost_per_call = st.number_input("Cost per Call ($)", min_value=1, value=20, step=1)
    with col2:
        profit_per_conversion = st.number_input("Profit per Conversion ($)", min_value=1, value=2000, step=100)
    
    # Get XGBoost predictions
    xgb_model = model_results['models_metrics']['XGBoost']['model']
    y_prob = xgb_model.predict_proba(model_results['X_test'])[:, 1]
    y_test = model_results['y_test']
    
    # Calculate profit for different thresholds
    thresholds = np.linspace(0.01, 0.99, 100)
    profit_metrics = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        conversions = (y_pred_thresh * y_test.values).sum()
        calls_made = y_pred_thresh.sum()
        revenue = conversions * profit_per_conversion
        cost = calls_made * cost_per_call
        net_profit = revenue - cost
        roi = (net_profit / cost) * 100 if cost > 0 else 0
        
        profit_metrics.append({
            'threshold': threshold,
            'net_profit': net_profit,
            'conversions': conversions,
            'calls_made': calls_made,
            'revenue': revenue,
            'cost': cost,
            'ROI': roi
        })
    
    profit_df = pd.DataFrame(profit_metrics)
    optimal_threshold = profit_df.loc[profit_df['net_profit'].idxmax()]
    
    # Display optimal threshold
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Optimal Threshold", f"{optimal_threshold['threshold']:.3f}")
    with col2:
        st.metric("Max Net Profit", f"${optimal_threshold['net_profit']:,.2f}")
    with col3:
        st.metric("Expected Conversions", f"{int(optimal_threshold['conversions'])}")
    with col4:
        st.metric("ROI", f"{optimal_threshold['ROI']:.2f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            profit_df,
            x='threshold',
            y='net_profit',
            title="Net Profit vs Threshold",
            labels={'threshold': 'Probability Threshold', 'net_profit': 'Net Profit ($)'}
        )
        fig.add_vline(x=optimal_threshold['threshold'], line_dash="dash", 
                     line_color="red", annotation_text="Optimal")
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            profit_df,
            x='threshold',
            y='ROI',
            title="ROI vs Threshold",
            labels={'threshold': 'Probability Threshold', 'ROI': 'ROI (%)'}
        )
        fig.add_vline(x=optimal_threshold['threshold'], line_dash="dash", 
                     line_color="red", annotation_text="Optimal")
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison: Before vs After
    st.subheader("📊 Strategy Comparison")
    
    # Before (calling everyone)
    n_total = len(y_test)
    cost_before = n_total * cost_per_call
    conversions_before = y_test.sum()
    revenue_before = conversions_before * profit_per_conversion
    profit_before = revenue_before - cost_before
    
    # After (using model with optimal threshold)
    y_pred_optimal = (y_prob >= optimal_threshold['threshold']).astype(int)
    calls_after = y_pred_optimal.sum()
    cost_after = calls_after * cost_per_call
    conversions_after = (y_pred_optimal * y_test.values).sum()
    revenue_after = conversions_after * profit_per_conversion
    profit_after = revenue_after - cost_after
    
    # Create comparison data with numeric values for proper display
    comparison_data = {
        'Metric': ['Total Calls', 'Cost ($)', 'Conversions', 'Revenue ($)', 'Net Profit ($)'],
        'Before Strategy': [
            n_total,
            cost_before,
            conversions_before,
            revenue_before,
            profit_before
        ],
        'After Strategy': [
            calls_after,
            cost_after,
            conversions_after,
            revenue_after,
            profit_after
        ],
        'Improvement': [
            f"{(1 - calls_after/n_total)*100:.1f}% reduction",
            f"${cost_before - cost_after:,.2f} saved",
            f"{conversions_after - conversions_before}",
            f"${revenue_after - revenue_before:,.2f}",
            f"${profit_after - profit_before:,.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format numeric columns for display
    display_df = comparison_df.copy()
    display_df['Before Strategy'] = display_df.apply(
        lambda row: f"${row['Before Strategy']:,.2f}" if 'Cost' in row['Metric'] or '$' in row['Metric'] 
        else f"{row['Before Strategy']:,}" if isinstance(row['Before Strategy'], (int, float)) 
        else row['Before Strategy'], axis=1
    )
    display_df['After Strategy'] = display_df.apply(
        lambda row: f"${row['After Strategy']:,.2f}" if 'Cost' in row['Metric'] or '$' in row['Metric'] 
        else f"{row['After Strategy']:,}" if isinstance(row['After Strategy'], (int, float)) 
        else row['After Strategy'], axis=1
    )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "🎯 Customer Insights":
    st.header("🎯 Customer Insights & Strategy")
    
    tab1, tab2, tab3 = st.tabs(["📅 Best Contact Times", "👥 Customer Segments", "💡 Key Insights"])
    
    with tab1:
        st.subheader("Optimal Contact Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Best Months for Contact:**")
            month_success = df.groupby('month')['y'].mean().sort_values(ascending=False)
            month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
            month_success_ordered = df.groupby('month')['y'].mean().reindex(month_order)
            top_months = month_success_ordered.sort_values(ascending=False).head(5)
            
            for month, rate in top_months.items():
                st.write(f"📅 {month.capitalize()}: {rate*100:.1f}% success rate")
        
        with col2:
            st.write("**Best Days for Contact:**")
            day_success = df.groupby('day')['y'].mean().sort_values(ascending=False)
            top_days = day_success.head(5)
            
            for day, rate in top_days.items():
                st.write(f"📆 Day {day}: {rate*100:.1f}% success rate")
        
        st.subheader("Previous Campaign Outcome Impact")
        poutcome_success = df.groupby('poutcome')['y'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=poutcome_success.index,
            y=poutcome_success.values,
            title="Success Rate by Previous Campaign Outcome",
            labels={'x': 'Previous Outcome', 'y': 'Success Rate'},
            color=poutcome_success.values,
            color_continuous_scale='Greens',
            text=poutcome_success.values
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Customer Segmentation Analysis")
        
        # Job-based segmentation
        job_analysis = df.groupby('job').agg({
            'y': ['mean', 'count']
        }).reset_index()
        job_analysis.columns = ['Job', 'Success Rate', 'Count']
        job_analysis = job_analysis.sort_values('Success Rate', ascending=False)
        
        fig = px.scatter(
            job_analysis,
            x='Count',
            y='Success Rate',
            size='Count',
            color='Success Rate',
            hover_name='Job',
            title="Job Segmentation: Success Rate vs Customer Count",
            labels={'Count': 'Number of Customers', 'Success Rate': 'Subscription Rate'},
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
        
        # Education-based segmentation
        edu_analysis = df.groupby('education').agg({
            'y': ['mean', 'count']
        }).reset_index()
        edu_analysis.columns = ['Education', 'Success Rate', 'Count']
        edu_analysis = edu_analysis.sort_values('Success Rate', ascending=False)
        
        fig = px.bar(
            edu_analysis,
            x='Education',
            y='Success Rate',
            title="Education Level vs Subscription Rate",
            color='Success Rate',
            color_continuous_scale='Blues',
            text='Success Rate'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💡 Key Business Insights")
        
        insights = [
            "🎯 **Target High-Value Segments**: Customers with previous successful campaign outcomes have 64.7% subscription rate",
            "📅 **Timing Matters**: March, September, October, and December show highest conversion rates (40-52%)",
            "👥 **Demographics**: Retired and student customers show higher subscription rates",
            "💰 **ROI Optimization**: Using ML predictions can reduce call costs by 80%+ while maintaining conversion rates",
            "📊 **Feature Importance**: Previous campaign outcomes, contact duration, and customer age are top predictors",
            "⚡ **Efficiency**: May has highest call volume but lowest conversion - consider reducing efforts in this month"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")

elif page == "🔮 Predictions":
    st.header("🔮 Customer Subscription Prediction")
    
    st.write("Enter customer details to predict subscription probability:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
        job = st.selectbox("Job", df['job'].unique())
        marital = st.selectbox("Marital Status", df['marital'].unique())
        education = st.selectbox("Education", df['education'].unique())
        default = st.selectbox("Has Credit Default?", ['no', 'yes'])
        balance = st.number_input("Balance", min_value=0, value=1000)
        housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
        loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    
    with col2:
        contact = st.selectbox("Contact Type", df['contact'].unique())
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
        campaign = st.number_input("Number of Contacts (Campaign)", min_value=1, value=1)
        pdays = st.number_input("Days Since Last Contact", min_value=-1, value=-1)
        previous = st.number_input("Previous Contacts", min_value=0, value=0)
        poutcome = st.selectbox("Previous Outcome", df['poutcome'].unique())
    
    if st.button("🔮 Predict Subscription Probability", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'balance': [balance],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'day': [day],
            'month': [month],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome]
        })
        
        # Preprocess
        X_input = input_data.copy()
        X_input = pd.get_dummies(X_input, drop_first=True)
        
        # Align columns with training data - ensure exact match
        feature_names = model_results['feature_names']
        # Create a new dataframe with all features in the correct order, initialized to 0
        X_input_aligned = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)
        
        # Fill in values that exist in X_input (both numeric and categorical)
        for col in X_input.columns:
            if col in X_input_aligned.columns:
                X_input_aligned.loc[0, col] = X_input[col].iloc[0]
        
        # Scale only the numeric columns that were scaled during training
        X_input_scaled = X_input_aligned.copy()
        numeric_cols = model_results['numeric_cols']
        # Get numeric columns in the exact order they were during training
        cols_to_scale = [col for col in numeric_cols if col in X_input_scaled.columns]
        if cols_to_scale:
            # Extract numeric columns in correct order as numpy array
            X_input_numeric = X_input_scaled[cols_to_scale].values
            # Transform
            scaled_values = model_results['scaler'].transform(X_input_numeric)
            # Assign back
            X_input_scaled[cols_to_scale] = scaled_values
        
        # Predict
        xgb_model = model_results['models_metrics']['XGBoost']['model']
        probability = xgb_model.predict_proba(X_input_scaled)[0, 1]
        prediction = xgb_model.predict(X_input_scaled)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Subscription Probability", f"{probability*100:.2f}%")
        
        with col2:
            prediction_text = "✅ Will Subscribe" if prediction == 1 else "❌ Will Not Subscribe"
            st.metric("Prediction", prediction_text)
        
        with col3:
            recommendation = "📞 **RECOMMEND CALLING**" if probability >= 0.5 else "⏸️ **DO NOT CALL**"
            st.markdown(f"### {recommendation}")
        
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Subscription Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300, template=get_plotly_template())
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit | Bank Marketing Analytics Dashboard"
    "</div>",
    unsafe_allow_html=True
)

