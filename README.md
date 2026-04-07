# 💳 Credit Risk Prediction Engine (UCI Dataset)

### 🔍 Project Overview
Predicting credit card default is a core challenge for financial giants like **American Express**. This project uses the **UCI Credit Card Default Dataset** to build a robust classification model. The goal is to identify high-risk customers based on their demographic data, payment history, and billing statements.

### 📊 Dataset Insight (UCI Machine Learning Repository)
The dataset contains **30,000 observations** with 24 variables, including:
* **Demographics:** Gender, Education, Marriage, Age.
* **Payment History:** Past 6 months of payment status (PAY_0 to PAY_6).
* **Bill & Payment Amounts:** Monthly statement amounts and previous payments.
* **Target:** `default.payment.next.month` (Binary: 1 for default, 0 for not).

### 🛠️ Tech Stack & Advanced Skills
* **Modeling:** **XGBoost Classifier** (Optimized for financial tabular data).
* **Engineering:** **StandardScaler** for handling large variances in billing amounts.
* **Evaluation:** **ROC-AUC** and **Precision-Recall Curves** (Crucial for imbalanced financial data).
* **Architecture:** Modular, class-based Python script (`src/model_pipeline.py`) for production readiness.

### 📂 Repository Structure
* **`data/`**: Contains the `UCI_Credit_Card.csv` (Sample/Full version).
* **`notebooks/`**: Deep-dive EDA, Correlation Heatmaps, and Feature Importance analysis.
* **`src/`**: `model_pipeline.py` - Clean, scalable Python code for data cleaning and training.
* **`visualization/`**: Model performance charts (Confusion Matrix, ROC Curve).

### 🚀 Key Business Findings
* **Late Payments:** Customers with a delayed payment status in the most recent month (PAY_0) are significantly more likely to default.
* **Credit Utilization:** High billing amounts relative to credit limits (LIMIT_BAL) correlate strongly with credit risk.
* **Model Accuracy:** Achieved an ROC-AUC of **0.XX** (Update this after running your code).

---
**Developed by:** Himani | Data Engineer @ Deloitte | M.Tech (IIT Delhi) | M.Sc (IIT Mandi)
