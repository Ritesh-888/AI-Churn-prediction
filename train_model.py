import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load and preprocess
df = pd.read_csv('telco_train.csv')
# Example preprocessing: drop customerID, encode categoricals, fill NAs
X = df.drop(['customerID', 'churned'], axis=1)
X = pd.get_dummies(X)
y = df['churned'].map({'No': 0, 'Yes': 1})

# Align and split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print(f'AUC-ROC on validation: {auc:.4f}')

# Save model and columns
joblib.dump(clf, 'model.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')