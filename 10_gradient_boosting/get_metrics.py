
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Mock matplotlib to avoid display issues if any
import matplotlib
matplotlib.use('Agg')

from main import generate_dataset, train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

def run():
    try:
        df = generate_dataset(1000)
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = train_model(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        if hasattr(model, 'feature_importances_'):
            import numpy as np
            importances = model.feature_importances_
            indices = np.argsort(importances)[-5:]
            print("\nTop 5 Features:")
            for idx in reversed(indices):
                print(f"feature_{idx}: {importances[idx]:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
