# Logistic Regression - Customer Churn Prediction

## 📖 Overview

Logistic Regression is a supervised learning algorithm used for binary classification problems. This project predicts customer churn for a telecommunications company.

## 🎯 Algorithm Explanation

**Logistic Regression** uses the logistic (sigmoid) function to model the probability of a binary outcome:

```
P(y=1|X) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))
```

Where:
- P(y=1|X) = Probability of positive class
- e = Euler's number
- β = Coefficients

## 🔑 Key Concepts

1. **Sigmoid Function**: Maps any value to [0, 1]
2. **Cost Function**: Log Loss (Binary Cross-Entropy)
3. **Decision Boundary**: Threshold (typically 0.5)
4. **Optimization**: Gradient Descent

## 📊 Dataset

- **Features**: Contract type, monthly charges, tenure, services used
- **Target**: Churn (Yes/No)
- **Size**: 1000 customers (synthetic data)

## 🎨 Visualizations

- Confusion Matrix
- ROC Curve & AUC
- Precision-Recall Curve
- Feature Importance

## 📈 Performance Metrics

- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## 🚀 Usage

```bash
pip install -r requirements.txt
python main.py
```

## 💡 Interview Questions Covered

1. What is Logistic Regression?
2. Difference between Linear and Logistic Regression?
3. What is the sigmoid function?
4. How do you handle imbalanced classes?
5. What is ROC-AUC?

## 🔍 Real-World Applications

- Customer churn prediction
- Email spam detection
- Disease diagnosis
- Credit risk assessment
