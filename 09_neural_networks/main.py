"""
Neural Networks - MNIST Digit Classification
Author: ML Portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    from sklearn.neural_network import MLPClassifier
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available, using sklearn MLPClassifier instead")
import os

sns.set_style("whitegrid")

def load_dataset():
    """Load digits dataset"""
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y, digits.images

def build_model_sklearn(input_dim, num_classes):
    """Build neural network using sklearn"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42,
        verbose=True
    )
    return model

def build_model_keras(input_dim, num_classes):
    """Build neural network using Keras"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_results(images, y_test, y_pred, cm, history, metrics):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Sample predictions
    for i in range(9):
        ax = fig.add_subplot(gs[0, i % 3])
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'Pred: {y_pred[i]}\nTrue: {y_test[i]}', fontsize=9)
            ax.axis('off')
    
    # Training history (if available)
    if history is not None and TENSORFLOW_AVAILABLE:
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Training History', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('Loss History', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True, ax=ax3)
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    ax3.set_title('Confusion Matrix', fontweight='bold')
    
    # Metrics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    metrics_text = "Performance Metrics\n" + "="*40 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    ax4.text(0.3, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.savefig('visualizations/neural_network_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("NEURAL NETWORKS - DIGIT CLASSIFICATION")
    print("="*60)
    
    # Load data
    print("\n📊 Loading digits dataset...")
    X, y, images = load_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    print("\n🚀 Building and training neural network...")
    
    history = None
    if TENSORFLOW_AVAILABLE:
        model = build_model_keras(X_train_scaled.shape[1], len(np.unique(y)))
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        print("✅ Model trained with TensorFlow/Keras!")
    else:
        model = build_model_sklearn(X_train_scaled.shape[1], len(np.unique(y)))
        model.fit(X_train_scaled, y_train)
        print("✅ Model trained with sklearn!")
    
    # Evaluate
    print("\n📈 Evaluating...")
    y_pred = model.predict(X_test_scaled)
    
    if TENSORFLOW_AVAILABLE:
        y_pred = np.argmax(y_pred, axis=1)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Train Accuracy': accuracy_score(y_train, 
                                         np.argmax(model.predict(X_train_scaled), axis=1) 
                                         if TENSORFLOW_AVAILABLE 
                                         else model.predict(X_train_scaled))
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    test_images = images[len(X_train):len(X_train)+len(X_test)]
    plot_results(test_images, y_test, y_pred, cm, history, metrics)
    
    print("\n" + "="*60)
    print("✅ NEURAL NETWORKS PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
