import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

class CreditRiskModel:
def __init__(self, data_path):
self.data_path = data_path
self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
self.scaler = StandardScaler()

def load_and_clean(self):
# Data loading logic
df = pd.read_csv(self.data_path)
# Basic cleaning (Amex values clean data!)
df = df.dropna()
return df

def train(self, X, y):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = self.scaler.fit_transform(X_train)
self.model.fit(X_train_scaled, y_train)
return X_test, y_test

def evaluate(self, X_test, y_test):
X_test_scaled = self.scaler.transform(X_test)
predictions = self.model.predict(X_test_scaled)
probs = self.model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, predictions))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
print("Credit Risk Pipeline Initialized...")
# Example usage (Uncomment when data is ready)
# pipeline = CreditRiskModel('data/sample_credit_data.csv')
