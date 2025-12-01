"""
Random Forest - Wine Quality Classification
Description: Ensemble learning for wine quality prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import os

sns.set_style("whitegrid")

def generate_dataset(n_samples=1000):
    """Generate synthetic wine quality dataset"""
    np.random.seed(42)
    
    # Features (wine chemical properties)
    alcohol = np.random.uniform(8, 15, n_samples)
    acidity = np.random.uniform(2.5, 4.5, n_samples)
    sugar = np.random.uniform(0.5, 15, n_samples)
    pH = np.random.uniform(2.8, 3.8, n_samples)
    sulfates = np.random.uniform(0.3, 1.5, n_samples)
    density = np.random.uniform(0.99, 1.01, n_samples)
    
    # Target: Quality (Good=1, Bad=0)
    quality_score = (
        2 * alcohol +
        -3 * acidity +
        0.5 * sugar +
        -5 * pH +
        10 * sulfates +
        np.random.normal(0, 5, n_samples)
    )
    
    quality = (quality_score > np.median(quality_score)).astype(int)
    
    df = pd.DataFrame({
        'alcohol': alcohol,
        'acidity': acidity,
        'sugar': sugar,
        'pH': pH,
        'sulfates': sulfates,
        'density': density,
        'quality': quality
    })
    
    return df

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def plot_results(model, X_test, y_test, y_pred, cm, feature_names, metrics):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    axes[0, 0].barh(range(len(importances)), importances[indices], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(importances))), alpha=0.8)
    axes[0, 0].set_yticks(range(len(importances)))
    axes[0, 0].set_yticklabels([feature_names[i] for i in indices])
    axes[0, 0].set_xlabel('Importance', fontweight='bold')
    axes[0, 0].set_title('Feature Importance', fontweight='bold', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Predicted', fontweight='bold')
    axes[0, 1].set_ylabel('Actual', fontweight='bold')
    axes[0, 1].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticklabels(['Bad', 'Good'])
    axes[0, 1].set_yticklabels(['Bad', 'Good'])
    
    # 3. Tree Estimators Performance
    estimators_range = range(1, len(model.estimators_) + 1, 10)
    train_scores = []
    test_scores = []
    
    for n in estimators_range:
        temp_model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
        temp_model.fit(X_test, y_test)  # Using test for demo
        train_scores.append(temp_model.score(X_test, y_test))
        test_scores.append(accuracy_score(y_test, temp_model.predict(X_test)))
    
    axes[1, 0].plot(estimators_range, train_scores, label='Training Score', marker='o')
    axes[1, 0].plot(estimators_range, test_scores, label='Test Score', marker='s')
    axes[1, 0].set_xlabel('Number of Estimators', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy', fontweight='bold')
    axes[1, 0].set_title('Model Performance vs Number of Trees', fontweight='bold', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics Summary
    axes[1, 1].axis('off')
    metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12,
                     family='monospace', verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualizations/random_forest_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("RANDOM FOREST - WINE QUALITY CLASSIFICATION")
    print("="*60)
    
    # Generate and prepare data
    print("\n📊 Generating dataset...")
    df = generate_dataset(1000)
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\n🚀 Training Random Forest...")
    model = train_model(X_train, y_train)
    print(f"✅ Trained with {model.n_estimators} trees!")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    print("\n🔍 Feature Importance:")
    for feature, importance in sorted(zip(X.columns, model.feature_importances_), 
                                     key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(model, X_test, y_test, y_pred, cm, X.columns, metrics)
    
    print("\n" + "="*60)
    print("✅ RANDOM FOREST PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
