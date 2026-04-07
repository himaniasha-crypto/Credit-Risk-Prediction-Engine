
Himani <himani.asha@gmail.com>
3:38 PM (0 minutes ago)
to me

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

class CreditRiskModel:
def __init__(self, data_path):
self.data_path = data_path
self.model = XGBClassifier(
n_estimators=100,
learning_rate=0.1,
max_depth=5,
use_label_encoder=False,
eval_metric='logloss'
)
self.scaler = StandardScaler()

def load_and_preprocess(self):
# Loading UCI Dataset (UCI_Credit_Card.csv)
df = pd.read_csv(self.data_path)

# UCI specific cleaning: 'default.payment.next.month' is the target
X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling features (Crucial for UCI numerical columns like BILL_AMT)
X_train_scaled = self.scaler.fit_transform(X_train)
X_test_scaled = self.scaler.transform(X_test)

return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(self):
X_train, X_test, y_train, y_test = self.load_and_preprocess()

print("Training XGBoost Model on UCI Dataset...")
self.model.fit(X_train, y_train)

# Predictions
preds = self.model.predict(X_test)
probs = self.model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
# Path should match your filename in 'data/' folder
pipeline = CreditRiskModel('data/UCI_Credit_Card.csv')
# pipeline.train_and_evaluate() # Uncomment to run locally
