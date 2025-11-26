"""
Decision Trees - Iris Classification
Author: ML Portfolio
Description: Multi-class classification using Decision Trees
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_dataset():
    """Load Iris dataset"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    
    return X, y, iris.target_names

def train_model(X_train, y_train):
    """Train Decision Tree model"""
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Macro)': precision_score(y_test, y_pred, average='macro'),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro'),
        'F1-Score (Macro)': f1_score(y_test, y_pred, average='macro')
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, y_pred, cm

def plot_results(model, X, y, X_test, y_test, y_pred, cm, feature_names, class_names, metrics):
    """Create comprehensive visualizations"""
    
    os.makedirs('visualizations', exist_ok=True)
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Decision Tree Structure
    ax1 = fig.add_subplot(gs[0:2, :])
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax1)
    ax1.set_title('Decision Tree Structure', fontweight='bold', fontsize=14, pad=20)
    
    # 2. Feature Importance
    ax2 = fig.add_subplot(gs[2, 0])
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    ax2.barh(range(len(importances)), importances[indices], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(importances)))
    ax2.set_yticklabels([feature_names[i] for i in indices])
    ax2.set_xlabel('Importance', fontweight='bold')
    ax2.set_title('Feature Importance', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[2, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=True, ax=ax3,
                xticklabels=class_names, yticklabels=class_names)
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
    
    plt.savefig('visualizations/decision_tree_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to 'visualizations/decision_tree_results.png'")
    plt.close()
    
    # Create a separate plot for decision boundaries (2D)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot decision boundaries for different feature pairs
    feature_pairs = [(0, 1), (0, 2), (2, 3)]
    pair_names = [
        ('sepal length', 'sepal width'),
        ('sepal length', 'petal length'),
        ('petal length', 'petal width')
    ]
    
    for idx, (pair, names) in enumerate(zip(feature_pairs, pair_names)):
        ax = axes[idx]
        
        # Train a simple tree on just these two features
        X_pair = X.iloc[:, list(pair)].values
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_pair, y)
        
        # Create mesh
        x_min, x_max = X_pair[:, 0].min() - 0.5, X_pair[:, 0].max() + 0.5
        y_min, y_max = X_pair[:, 1].min() - 0.5, X_pair[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        # Predict
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        scatter = ax.scatter(X_pair[:, 0], X_pair[:, 1], c=y, 
                           cmap='viridis', edgecolor='k', s=50, alpha=0.8)
        ax.set_xlabel(names[0], fontweight='bold')
        ax.set_ylabel(names[1], fontweight='bold')
        ax.set_title(f'Decision Boundary: {names[0]} vs {names[1]}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/decision_boundaries.png', dpi=300, bbox_inches='tight')
    print("✅ Decision boundaries saved to 'visualizations/decision_boundaries.png'")
    plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("DECISION TREES - IRIS CLASSIFICATION")
    print("="*60)
    
    # 1. Load Dataset
    print("\n📊 Loading Iris dataset...")
    X, y, class_names = load_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"\nFeatures: {list(X.columns)}")
    print(f"Classes: {list(class_names)}")
    print(f"\nClass distribution:\n{y.value_counts()}")
    
    # 2. Prepare data
    print("\n🔧 Preparing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train model
    print("\n🚀 Training Decision Tree model...")
    model = train_model(X_train, y_train)
    print("✅ Model trained successfully!")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    
    # 4. Evaluate model
    print("\n📈 Evaluating model...")
    metrics, y_pred, cm = evaluate_model(model, X_test, y_test)
    
    print("\nPerformance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 5. Classification Report
    print("\n📋 Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # 6. Cross-validation
    print("\n🔄 Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 7. Feature importance
    print("\n🔍 Feature Importance:")
    print("-" * 40)
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    # 8. Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(model, X, y, X_test, y_test, y_pred, cm, 
                X.columns, class_names, metrics)
    
    # 9. Sample predictions
    print("\n🎯 Sample Predictions:")
    print("-" * 60)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        actual = class_names[y_test.iloc[idx]]
        predicted = class_names[y_pred[idx]]
        features = X_test.iloc[idx].values
        print(f"Actual: {actual:12s} | Predicted: {predicted:12s} | Features: {features}")
    
    print("\n" + "="*60)
    print("✅ DECISION TREES PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
