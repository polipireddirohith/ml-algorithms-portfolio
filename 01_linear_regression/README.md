# Linear Regression - House Price Prediction

## 📖 Overview

Linear Regression is a fundamental supervised learning algorithm used for predicting continuous values. This project demonstrates house price prediction based on various features.

## 🎯 Algorithm Explanation

**Linear Regression** models the relationship between dependent variable (y) and independent variables (X) using a linear equation:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- y = Target variable (house price)
- β₀ = Intercept
- β₁, β₂, ..., βₙ = Coefficients
- x₁, x₂, ..., xₙ = Features
- ε = Error term

## 🔑 Key Concepts

1. **Cost Function**: Mean Squared Error (MSE)
2. **Optimization**: Gradient Descent or Normal Equation
3. **Assumptions**: Linearity, Independence, Homoscedasticity, Normality

## 📊 Dataset

- **Features**: Square footage, bedrooms, bathrooms, age, location score
- **Target**: House price
- **Size**: 1000 samples (synthetic data)

## 🎨 Visualizations

- Actual vs Predicted prices
- Residual plot
- Feature importance
- Learning curve

## 📈 Performance Metrics

- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## 🚀 Usage

```bash
pip install -r requirements.txt
python main.py
```


