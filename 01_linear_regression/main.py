"""
Linear Regression - House Price Prediction
Description: Implementation of Linear Regression for predicting house prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def generate_dataset(n_samples=1000):
    """Generate synthetic house price dataset"""
    np.random.seed(42)
    
    # Features
    square_feet = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    # Target: House price (with realistic relationship)
    price = (
        150 * square_feet +
        20000 * bedrooms +
        15000 * bathrooms -
        1000 * age +
        25000 * location_score +
        np.random.normal(0, 50000, n_samples)  # Add noise
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'price': price
    })
    
    return df

def train_model(X_train, y_train):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'R² Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return metrics, y_pred

def plot_results(y_test, y_pred, feature_names, coefficients, metrics):
    """Create comprehensive visualizations"""
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    axes[0, 0].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Price ($)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual Plot
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance (Coefficients)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
    axes[1, 0].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Feature Importance (Coefficients)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Metrics Summary
    axes[1, 1].axis('off')
    metrics_text = "Model Performance Metrics\n" + "="*40 + "\n\n"
    for metric, value in metrics.items():
        if metric == 'R² Score':
            metrics_text += f"{metric}: {value:.4f}\n"
        else:
            metrics_text += f"{metric}: ${value:,.2f}\n"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=14, 
                     family='monospace', verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('visualizations/linear_regression_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to 'visualizations/linear_regression_results.png'")
    plt.close()

def main():
    """Main execution function"""
    print("="*60)
    print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # 1. Generate Dataset
    print("\n📊 Generating dataset...")
    df = generate_dataset(1000)
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # 2. Prepare data
    print("\n🔧 Preparing data...")
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Train model
    print("\n🚀 Training Linear Regression model...")
    model = train_model(X_train, y_train)
    print("✅ Model trained successfully!")
    
    # 4. Evaluate model
    print("\n📈 Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    print("\nPerformance Metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        if metric == 'R² Score':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: ${value:,.2f}")
    
    # 5. Cross-validation
    print("\n🔄 Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='r2')
    print(f"CV R² Scores: {cv_scores}")
    print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 6. Feature coefficients
    print("\n🔍 Feature Coefficients:")
    print("-" * 40)
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:,.2f}")
    print(f"Intercept: {model.intercept_:,.2f}")
    
    # 7. Visualizations
    print("\n🎨 Creating visualizations...")
    plot_results(y_test, y_pred, X.columns, model.coef_, metrics)
    
    # 8. Sample predictions
    print("\n🎯 Sample Predictions:")
    print("-" * 40)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_indices:
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        error = abs(actual - predicted)
        print(f"Actual: ${actual:,.2f} | Predicted: ${predicted:,.2f} | Error: ${error:,.2f}")
    
    print("\n" + "="*60)
    print("✅ LINEAR REGRESSION PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
