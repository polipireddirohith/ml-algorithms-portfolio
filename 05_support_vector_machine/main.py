"""
Support Vector Machine - Breast Cancer Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
import os

sns.set_style("whitegrid")

def load_dataset():
    """Load breast cancer dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='diagnosis')
    return X, y, data.target_names

def train_model(X_train, y_train):
    """Train SVM with RBF kernel"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def plot_results(y_test, y_pred, y_pred_proba, cm, metrics):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2)
    axes[0, 1].set_title('ROC Curve', fontweight='bold')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics
    axes[1, 0].axis('off')
    metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    # Prediction distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, label='Malignant', color='red')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, label='Benign', color='blue')
    axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Probability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/svm_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("SUPPORT VECTOR MACHINE - BREAST CANCER CLASSIFICATION")
    print("="*60)
    
    # Load data
    print("\n📊 Loading dataset...")
    X, y, class_names = load_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {class_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    print("\n🚀 Training SVM with RBF kernel...")
    model, scaler = train_model(X_train, y_train)
    print("✅ Model trained!")
    
    # Evaluate
    print("\n📈 Evaluating...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    X_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Support vectors
    print(f"\n🔍 Number of support vectors: {model.n_support_}")
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(y_test, y_pred, y_pred_proba, cm, metrics)
    
    print("\n" + "="*60)
    print("✅ SVM PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
