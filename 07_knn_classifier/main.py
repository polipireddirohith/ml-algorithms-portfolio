"""
K-Nearest Neighbors - Digit Recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

sns.set_style("whitegrid")

def load_dataset():
    """Load digits dataset"""
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y, digits.images

def find_optimal_k(X_train, y_train, X_test, y_test, max_k=20):
    """Find optimal K value"""
    k_range = range(1, max_k + 1, 2)
    train_scores = []
    test_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))
    
    return k_range, train_scores, test_scores

def plot_results(images, y_test, y_pred, cm, k_range, train_scores, test_scores, metrics):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Sample predictions
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'P:{y_pred[i]}\nA:{y_test[i]}', fontsize=8)
        plt.axis('off')
    
    # 2. K value selection
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(k_range, train_scores, marker='o', label='Train', linewidth=2)
    ax2.plot(k_range, test_scores, marker='s', label='Test', linewidth=2)
    ax2.set_xlabel('K Value', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('K Value Selection', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 1:])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax3)
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_title('Confusion Matrix', fontweight='bold')
    
    # 4. Metrics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    metrics_text = "Performance Metrics\n" + "="*40 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    ax4.text(0.3, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    plt.savefig('visualizations/knn_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("K-NEAREST NEIGHBORS - DIGIT RECOGNITION")
    print("="*60)
    
    # Load data
    print("\n📊 Loading digits dataset...")
    X, y, images = load_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal K
    print("\n🔍 Finding optimal K...")
    k_range, train_scores, test_scores = find_optimal_k(
        X_train_scaled, y_train, X_test_scaled, y_test, max_k=20
    )
    optimal_k = k_range[np.argmax(test_scores)]
    print(f"Optimal K: {optimal_k}")
    
    # Train model
    print(f"\n🚀 Training KNN with K={optimal_k}...")
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train_scaled, y_train)
    print("✅ Model trained!")
    
    # Evaluate
    print("\n📈 Evaluating...")
    y_pred = knn.predict(X_test_scaled)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Train Accuracy': knn.score(X_train_scaled, y_train),
        'Test Accuracy': knn.score(X_test_scaled, y_test)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    test_images = images[len(X_train):len(X_train)+len(X_test)]
    plot_results(test_images, y_test, y_pred, cm, k_range, 
                 train_scores, test_scores, metrics)
    
    print("\n" + "="*60)
    print("✅ KNN PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
