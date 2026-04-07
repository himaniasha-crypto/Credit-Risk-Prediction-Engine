
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE # <--- Class Imbalance handle karne ke liye

class CreditRiskModel:
def __init__(self, data_path):
self.data_path = data_path
self.model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
self.scaler = StandardScaler()

def feature_engineering(self, df):
# --- FEATURE ENGINEERING FLEX ---
# 1. Credit Utilization Ratio: Kitna limit use kiya vs kitna mila (High ratio = High Risk)
df['utilization_ratio'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)

# 2. Average Payment Status: Pichle 6 mahine ka payment behavior
df['avg_payment_status'] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)

return df

def load_and_preprocess(self):
df = pd.read_csv(self.data_path)
df = self.feature_engineering(df) # Engineering apply ki

X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- HANDLING CLASS IMBALANCE (SMOTE) ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scaling
X_train_scaled = self.scaler.fit_transform(X_train_res)
X_test_scaled = self.scaler.transform(X_test)

return X_train_scaled, X_test_scaled, y_train_res, y_test

def train_and_evaluate(self):
X_train, X_test, y_train, y_test = self.load_and_preprocess()
self.model.fit(X_train, y_train)

preds = self.model.predict(X_test)
print("--- Model Performance with SMOTE & Feature Engineering ---")
print(classification_report(y_test, preds))

if __name__ == "__main__":
pipeline = CreditRiskModel('data/UCI_Credit_Card.csv')

