# Bank Marketing Analytics Dashboard

This project builds a **Machine Learning model** to predict whether a customer will subscribe to a **term deposit** using the Bank Marketing dataset from UCI Machine Learning Repository.
The goal is to help banks improve marketing efficiency, reduce costs, and target the right customers.
An impressive Streamlit dashboard for analyzing bank marketing campaign data with ML-powered predictions and ROI analysis.

## Features

- 📊 **Comprehensive EDA**: Interactive visualizations of customer demographics, temporal patterns, and correlations
- 🤖 **Model Performance**: Comparison of Logistic Regression, Random Forest, and XGBoost models
- 💰 **ROI Analysis**: Cost-benefit analysis with optimal threshold calculation
- 🎯 **Customer Insights**: Best contact times, customer segmentation, and key business insights
- 🔮 **Predictions**: Real-time subscription probability predictions for individual customers

## XGBoost Model

- **Accuracy**: 89% (without class weights)
- **Approach**: Uses SMOTE for handling imbalanced data

## 🚀 **Project Objectives**

### **1️⃣ Predict Customer Conversion (Main Objective)**
Build a model to predict whether a customer will say **"yes"** to a term deposit.

### **2️⃣ Identify Key Factors Influencing Subscription**
Understand which customer attributes (age, job, campaign calls, previous outcome, etc.) impact the final decision.

### **3️⃣ Handle Class Imbalance**
Use **SMOTE** to handle an imbalanced dataset and improve model performance.

### **4️⃣ Compare ML Models**
Train and evaluate:
- Logistic Regression  
- Random Forest  
- XGBoost  

Compare accuracy, precision, recall, and F1-score.

### **5️⃣ Explain Model Predictions**
Use **SHAP** to visualize:
- Global feature importance (summary plot)  
- Individual predictions (force plot)  

### **6️⃣ Streamlit Dashboard**
- Enter customer features in a form  
- Get a real-time prediction: **"Will this customer subscribe?"**  
- SHAP-based explanation for every prediction 

## 📂 **Dataset**

Source: UCI Machine Learning Repository  
Dataset: **Bank Marketing Data**  
Link: https://archive.ics.uci.edu/dataset/222/bank+marketing
- **45,211 records**
- **17 features** (demographic, financial, campaign-related)
- **Target**: Term deposit subscription (yes/no)
- **Challenge**: Highly imbalanced (11.7% subscription rate)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

1. Make sure `bank-full.csv` is in the same directory
2. Run the Streamlit app:
```bash
streamlit run streamlit_dashboard.py
```

3. The dashboard will open in your default web browser

## Dashboard Pages

1. **📈 Overview**: Dataset statistics and target distribution
2. **🔍 Exploratory Data Analysis**: Demographics, temporal analysis, job/education insights, correlations
3. **🤖 Model Performance**: Model comparison, confusion matrices, feature importance
4. **💰 ROI Analysis**: Profit optimization with customizable cost/profit parameters
5. **🎯 Customer Insights**: Best contact times, customer segments, key business insights
6. **🔮 Predictions**: Interactive prediction interface for individual customers

- **Data Preprocessing**:
  - SMOTE for handling imbalanced classes
  - StandardScaler for feature scaling
  - One-hot encoding for categorical variables

### Technical Skills Demonstrated:
- ✅ **Machine Learning**: Multiple models (Logistic Regression, Random Forest, XGBoost)
- ✅ **Data Preprocessing**: SMOTE, scaling, encoding, handling imbalanced data
- ✅ **Model Evaluation**: Comprehensive metrics, confusion matrices, ROC curves
- ✅ **Clustering**: K-Means customer segmentation
- ✅ **Business Intelligence**: ROI analysis, cost-benefit optimization
- ✅ **Data Visualization**: Interactive Plotly charts, professional styling
- ✅ **Web Development**: Streamlit dashboard with multiple pages

### Business Impact:
- **Cost Reduction**: Up to 80% reduction in marketing costs
- **ROI Optimization**: 1100%+ ROI with optimal threshold
- **Targeting Strategy**: Data-driven customer segmentation
- **Predictive Analytics**: Real-time customer subscription prediction

## 💡 Key Insights from Analysis

1. **Timing Matters**: March, September, October, December show 40-50% conversion
2. **Customer Segments**: Students and retired customers convert best
3. **Previous Success**: 64.7% conversion for customers with previous success
4. **Optimal Threshold**: 0.020 probability threshold maximizes profit
5. **Cost Efficiency**: Can reduce calls by 80% while maintaining conversions

## 🔧 Customization

You can customize:
- Cost per call (default: $20)
- Profit per conversion (default: $2000)
- Number of clusters for segmentation
- Model selection for predictions

## 🧪 **How to Run This Project Locally**

### **1. Clone the repo**
```bash
git clone https://github.com/your-username/bank-marketing-ml.git
cd bank-marketing-ml

🧑‍💻 Author
Nikita
📧 Email: nikitabalwada309@gmail.com
🌐 LinkedIn: https://www.linkedin.com/in/nikita-balwada29/

