"""
Logistic Regression - Customer Churn Prediction
Author: ML Portfolio
Description: Binary classification to predict customer churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def generate_dataset(n_samples=1000):
    """Generate synthetic customer churn dataset"""
    np.random.seed(42)
    
    # Features
    tenure = np.random.randint(1, 72, n_samples)  # months
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])  # 0: Month-to-month, 1: One year, 2: Two year
    internet_service = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # 0: No, 1: Yes
    tech_support = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 0: No, 1: Yes
    
    # Target: Churn (influenced by features)
    churn_probability = (
        0.5 - 0.01 * tenure +  # Longer tenure = less churn
        0.005 * monthly_charges +  # Higher charges = more churn
        -0.2 * contract_type +  # Longer contracts = less churn
        0.1 * internet_service -  # Internet service = slight increase
        0.15 * tech_support  # Tech support = less churn
    )
    
    # Apply sigmoid and add randomness
    churn_probability = 1 / (1 + np.exp(-churn_probability))
    churn = (churn_probability + np.random.normal(0, 0.1, n_samples)) > 0.5
    churn = churn.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'churn': churn
    })
    
    return df

def train_model(X_train, y_train):
    """Train Logistic Regression model"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model performance"""
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
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, y_pred, y_pred_proba, cm

def plot_results(y_test, y_pred, y_pred_proba, cm, feature_names, coefficients, metrics):
    """Create comprehensive visualizations"""
    
    os.makedirs('visualizations', exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted', fontweight='bold')
    ax1.set_ylabel('Actual', fontweight='bold')
    ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
    ax1.set_xticklabels(['No Churn', 'Churn'])
    ax1.set_yticklabels(['No Churn', 'Churn'])
    
    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')
    ax2.set_title('ROC Curve', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = fig.add_subplot(gs[0, 2])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax3.plot(recall, precision, linewidth=2, color='green')
    ax3.set_xlabel('Recall', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature Importance
    ax4 = fig.add_subplot(gs[1, :])
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
    ax4.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    ax4.set_xlabel('Coefficient Value', fontweight='bold')
    ax4.set_title('Feature Importance (Coefficients)', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Metrics Summary
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=12, 
             family='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 6. Prediction Distribution
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.6, label='No Churn', color='blue')
    ax6.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.6, label='Churn', color='red')
    ax6.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax6.set_xlabel('Predicted Probability', fontweight='bold')
    ax6.set_ylabel('Frequency', fontweight='bold')
    ax6.set_title('Prediction Probability Distribution', fontweight='bold', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig('visualizations/logistic_regression_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to 'visualizations/logistic_regression_results.png'")
    plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("LOGISTIC REGRESSION - CUSTOMER CHURN PREDICTION")
    print("="*60)
    
    # 1. Generate Dataset
    print("\n📊 Generating dataset...")
    df = generate_dataset(1000)
    print(f"Dataset shape: {df.shape}")
    print(f"\nChurn distribution:\n{df['churn'].value_counts()}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # 2. Prepare data
    print("\n🔧 Preparing data...")
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train model
    print("\n🚀 Training Logistic Regression model...")
    model, scaler = train_model(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # 4. Evaluate model
    print("\n📈 Evaluating model...")
    metrics, y_pred, y_pred_proba, cm = evaluate_model(model, scaler, X_test, y_test)
    
    print("\nPerformance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 5. Classification Report
    print("\n📋 Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # 6. Cross-validation
    print("\n🔄 Performing 5-fold cross-validation...")
    X_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC Scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 7. Feature coefficients
    print("\n🔍 Feature Coefficients:")
    print("-" * 40)
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")
    
    # 8. Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(y_test, y_pred, y_pred_proba, cm, X.columns, model.coef_[0], metrics)
    
    # 9. Sample predictions
    print("\n🎯 Sample Predictions:")
    print("-" * 60)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        actual = "Churn" if y_test.iloc[idx] == 1 else "No Churn"
        predicted = "Churn" if y_pred[idx] == 1 else "No Churn"
        probability = y_pred_proba[idx]
        print(f"Actual: {actual:10s} | Predicted: {predicted:10s} | Probability: {probability:.4f}")
    
    print("\n" + "="*60)
    print("✅ LOGISTIC REGRESSION PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
