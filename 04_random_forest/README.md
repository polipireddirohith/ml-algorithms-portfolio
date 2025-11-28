# Random Forest - Wine Quality Classification

## 📖 Overview

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

## 🎯 Algorithm Explanation

**Random Forest** creates multiple decision trees and aggregates their predictions:
- **Bagging**: Bootstrap Aggregating
- **Random Feature Selection**: Each tree uses random subset of features
- **Voting**: Classification by majority vote

## 🔑 Key Concepts

1. **Ensemble Learning**: Wisdom of crowds
2. **Out-of-Bag (OOB) Error**: Built-in validation
3. **Feature Importance**: Averaged across trees
4. **Hyperparameters**: n_estimators, max_depth, min_samples_split

## 📊 Dataset

- **Features**: Wine chemical properties
- **Target**: Wine quality (Good/Bad)
- **Size**: 1000 samples (synthetic)

## 🚀 Usage

```bash
pip install -r requirements.txt
python main.py
```

