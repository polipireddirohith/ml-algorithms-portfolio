"""
XGBoost - Gradient Boosting for Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using sklearn GradientBoostingClassifier instead")
import os

sns.set_style("whitegrid")

def generate_dataset(n_samples=1000):
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def train_model(X_train, y_train):
    """Train XGBoost model"""
    if XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model

def plot_results(model, X_test, y_test, y_pred, y_pred_proba, cm, feature_names, metrics):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Feature Importance
    if XGBOOST_AVAILABLE:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_
    
    indices = np.argsort(importances)[-10:]  # Top 10 features
    
    axes[0, 0].barh(range(len(indices)), importances[indices],
                     color=plt.cm.plasma(np.linspace(0, 1, len(indices))), alpha=0.8)
    axes[0, 0].set_yticks(range(len(indices)))
    axes[0, 0].set_yticklabels([feature_names[i] for i in indices])
    axes[0, 0].set_xlabel('Importance', fontweight='bold')
    axes[0, 0].set_title('Top 10 Feature Importance', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=2)
    axes[0, 1].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0, 1].set_ylabel('True Positive Rate', fontweight='bold')
    axes[0, 1].set_title('ROC Curve', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted', fontweight='bold')
    axes[1, 0].set_ylabel('Actual', fontweight='bold')
    axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
    
    # Metrics
    axes[1, 1].axis('off')
    metrics_text = "Performance Metrics\n" + "="*35 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    
    if XGBOOST_AVAILABLE:
        metrics_text += f"\nModel: XGBoost\n"
        metrics_text += f"Trees: {model.n_estimators}\n"
        metrics_text += f"Max Depth: {model.max_depth}\n"
    else:
        metrics_text += f"\nModel: GradientBoosting\n"
        metrics_text += f"Trees: {model.n_estimators}\n"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('visualizations/xgboost_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("XGBOOST - GRADIENT BOOSTING CLASSIFICATION")
    print("="*60)
    
    # Generate data
    print("\n📊 Generating dataset...")
    df = generate_dataset(1000)
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\n🚀 Training XGBoost model...")
    model = train_model(X_train, y_train)
    print("✅ Model trained!")
    
    # Evaluate
    print("\n📈 Evaluating...")
    y_pred = model.predict(X_test)
    
    if XGBOOST_AVAILABLE:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    print("\n🔍 Top 5 Most Important Features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-5:]
    for idx in reversed(indices):
        print(f"{X.columns[idx]}: {importances[idx]:.4f}")
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(model, X_test, y_test, y_pred, y_pred_proba, cm, 
                 X.columns, metrics)
    
    print("\n" + "="*60)
    print("✅ XGBOOST PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
