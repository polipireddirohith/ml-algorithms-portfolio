"""
Naive Bayes - Spam Detection
Author: ML Portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import os

sns.set_style("whitegrid")

def generate_dataset(n_samples=1000):
    """Generate synthetic email dataset"""
    np.random.seed(42)
    
    spam_words = ['free', 'win', 'prize', 'click', 'offer', 'buy', 'discount', 'urgent']
    ham_words = ['meeting', 'project', 'report', 'schedule', 'team', 'update', 'review']
    
    emails = []
    labels = []
    
    # Generate spam emails
    for _ in range(n_samples // 2):
        num_words = np.random.randint(5, 15)
        words = np.random.choice(spam_words, num_words, replace=True)
        emails.append(' '.join(words))
        labels.append(1)  # Spam
    
    # Generate ham emails
    for _ in range(n_samples // 2):
        num_words = np.random.randint(5, 15)
        words = np.random.choice(ham_words, num_words, replace=True)
        emails.append(' '.join(words))
        labels.append(0)  # Ham
    
    df = pd.DataFrame({'email': emails, 'label': labels})
    return df

def train_model(X_train, y_train):
    """Train Naive Bayes model"""
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def plot_results(y_test, y_pred, cm, feature_names, class_probs, metrics):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_xticklabels(['Ham', 'Spam'])
    axes[0, 0].set_yticklabels(['Ham', 'Spam'])
    
    # Class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    axes[0, 1].bar(['Ham', 'Spam'], counts, color=['blue', 'red'], alpha=0.7)
    axes[0, 1].set_title('Test Set Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Top features for spam
    if len(feature_names) > 0:
        top_spam_idx = np.argsort(class_probs[1])[-10:]
        top_spam_words = [feature_names[i] for i in top_spam_idx]
        top_spam_probs = class_probs[1][top_spam_idx]
        
        axes[1, 0].barh(top_spam_words, top_spam_probs, color='red', alpha=0.7)
        axes[1, 0].set_title('Top Spam Indicators', fontweight='bold')
        axes[1, 0].set_xlabel('Log Probability')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Metrics
    axes[1, 1].axis('off')
    metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric}: {value:.4f}\n"
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualizations/naive_bayes_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("NAIVE BAYES - SPAM DETECTION")
    print("="*60)
    
    # Generate data
    print("\n📊 Generating email dataset...")
    df = generate_dataset(1000)
    print(f"Dataset shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['email'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Train model
    print("\n🚀 Training Naive Bayes...")
    model, vectorizer = train_model(X_train, y_train)
    print("✅ Model trained!")
    
    # Evaluate
    print("\n📈 Evaluating...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Cross-validation
    print("\n🔄 Cross-validation...")
    X_vec = vectorizer.transform(df['email'])
    cv_scores = cross_val_score(model, X_vec, df['label'], cv=5, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    feature_names = vectorizer.get_feature_names_out()
    plot_results(y_test, y_pred, cm, feature_names, 
                 model.feature_log_prob_, metrics)
    
    print("\n" + "="*60)
    print("✅ NAIVE BAYES PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
