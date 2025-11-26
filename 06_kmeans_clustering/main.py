"""
K-Means Clustering - Customer Segmentation
Author: ML Portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

sns.set_style("whitegrid")

def generate_dataset(n_samples=500):
    """Generate synthetic customer data"""
    np.random.seed(42)
    
    # Create 3 natural clusters
    cluster1 = np.random.normal([25, 50000], [5, 10000], (n_samples//3, 2))
    cluster2 = np.random.normal([45, 80000], [7, 15000], (n_samples//3, 2))
    cluster3 = np.random.normal([35, 30000], [6, 8000], (n_samples//3, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    df = pd.DataFrame(data, columns=['age', 'annual_income'])
    df['spending_score'] = (
        0.5 * df['age'] + 
        0.0005 * df['annual_income'] + 
        np.random.normal(0, 10, len(df))
    )
    
    return df

def find_optimal_k(X_scaled, max_k=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    return K_range, inertias, silhouette_scores

def plot_results(df, X_scaled, labels, centroids, K_range, inertias, silhouette_scores):
    """Create visualizations"""
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Clusters (Age vs Income)
    scatter = axes[0, 0].scatter(df['age'], df['annual_income'], 
                                  c=labels, cmap='viridis', alpha=0.6, s=50)
    axes[0, 0].scatter(centroids[:, 0] * df['age'].std() + df['age'].mean(),
                        centroids[:, 1] * df['annual_income'].std() + df['annual_income'].mean(),
                        c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                        label='Centroids')
    axes[0, 0].set_xlabel('Age', fontweight='bold')
    axes[0, 0].set_ylabel('Annual Income ($)', fontweight='bold')
    axes[0, 0].set_title('Customer Segments (Age vs Income)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
    
    # 2. Elbow Method
    axes[0, 1].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Clusters (K)', fontweight='bold')
    axes[0, 1].set_ylabel('Inertia', fontweight='bold')
    axes[0, 1].set_title('Elbow Method', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Silhouette Score
    axes[1, 0].plot(K_range, silhouette_scores, marker='s', linewidth=2, 
                     markersize=8, color='green')
    axes[1, 0].set_xlabel('Number of Clusters (K)', fontweight='bold')
    axes[1, 0].set_ylabel('Silhouette Score', fontweight='bold')
    axes[1, 0].set_title('Silhouette Analysis', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cluster Statistics
    axes[1, 1].axis('off')
    cluster_stats = df.copy()
    cluster_stats['cluster'] = labels
    stats_text = "Cluster Statistics\n" + "="*40 + "\n\n"
    
    for cluster in range(len(centroids)):
        cluster_data = cluster_stats[cluster_stats['cluster'] == cluster]
        stats_text += f"Cluster {cluster}:\n"
        stats_text += f"  Size: {len(cluster_data)}\n"
        stats_text += f"  Avg Age: {cluster_data['age'].mean():.1f}\n"
        stats_text += f"  Avg Income: ${cluster_data['annual_income'].mean():,.0f}\n\n"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualizations/kmeans_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved!")
    plt.close()

def main():
    """Main execution"""
    print("="*60)
    print("K-MEANS CLUSTERING - CUSTOMER SEGMENTATION")
    print("="*60)
    
    # Generate data
    print("\n📊 Generating customer data...")
    df = generate_dataset(500)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    X = df[['age', 'annual_income', 'spending_score']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal K
    print("\n🔍 Finding optimal number of clusters...")
    K_range, inertias, silhouette_scores = find_optimal_k(X_scaled, max_k=10)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal K (by silhouette score): {optimal_k}")
    
    # Train K-Means
    print(f"\n🚀 Training K-Means with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print("✅ Clustering completed!")
    
    # Evaluate
    print("\n📈 Cluster Metrics:")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, labels):.4f}")
    
    # Cluster sizes
    print("\n📊 Cluster Sizes:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} customers ({count/len(labels)*100:.1f}%)")
    
    # Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(df, X_scaled, labels, kmeans.cluster_centers_, 
                 K_range, inertias, silhouette_scores)
    
    print("\n" + "="*60)
    print("✅ K-MEANS PROJECT COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
