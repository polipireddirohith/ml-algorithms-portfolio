# 🎓 Deep Interview Preparation Guide - ML Algorithms Portfolio

This guide provides a deep technical explanation of every project in your portfolio, tailored for an experienced Machine Learning candidate. It covers the theoretical foundations, implementation details, and potential interview questions for each algorithm.

---

## 1. Linear Regression (House Price Prediction)
**Project Goal:** Predict continuous values (house prices) based on features like square footage and number of bedrooms.

### 🧠 Technical Deep Dive
*   **Core Concept:** Linear Regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data. The most common method is **Ordinary Least Squares (OLS)**, which minimizes the sum of the squared differences (residuals) between the observed dependent variable and the predicted dependent variable.
*   **The Math:** The hypothesis is $h_\theta(x) = \theta_0 + \theta_1x_1 + ... + \theta_nx_n$. The cost function (MSE) is $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$. Gradient Descent is used to minimize this cost function.
*   **Assumptions:**
    1.  **Linearity:** The relationship between X and y is linear.
    2.  **Independence:** Observations are independent of each other.
    3.  **Homoscedasticity:** The variance of residual is the same for any value of X.
    4.  **Normality:** For any fixed value of X, Y is normally distributed.

### 💻 Implementation Highlights
*   **Data:** Synthetic dataset generated using `numpy`.
*   **Key Library:** `sklearn.linear_model.LinearRegression`.
*   **Evaluation:** Used **R² Score** (variance explained), **MAE** (average error magnitude), and **RMSE** (penalizes large errors).
*   **Visualization:** Residual plots were created to check for homoscedasticity (random scatter is good; patterns indicate issues).

### ❓ Key Interview Questions
1.  **Q: What is the difference between R² and Adjusted R²?**
    *   *A: R² increases every time you add a feature, even if it's noise. Adjusted R² penalizes the model for adding useless features, making it a better metric for multiple regression.*
2.  **Q: How do you handle multicollinearity?**
    *   *A: Multicollinearity (features highly correlated with each other) inflates standard errors. Detect it using VIF (Variance Inflation Factor). Fix it by removing correlated features or using regularization (Lasso/Ridge).*

---

## 2. Logistic Regression (Customer Churn)
**Project Goal:** Classify binary outcomes (churn vs. no churn).

### 🧠 Technical Deep Dive
*   **Core Concept:** Despite the name, it's a classification algorithm. It uses the **Sigmoid function** ($\sigma(z) = \frac{1}{1 + e^{-z}}$) to squash outputs between 0 and 1, representing probability.
*   **Cost Function:** Uses **Log Loss (Binary Cross-Entropy)**, not MSE. MSE is non-convex for logistic regression (many local minima), whereas Log Loss is convex.
*   **Decision Boundary:** The threshold (usually 0.5) decides the class. Changing this threshold trades off Precision vs. Recall.

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.linear_model.LogisticRegression`.
*   **Metrics:** **Confusion Matrix**, **Precision**, **Recall**, **F1-Score**. Accuracy is misleading for imbalanced datasets (like churn), so F1 is preferred.

### ❓ Key Interview Questions
1.  **Q: Why can't we use Linear Regression for classification?**
    *   *A: Linear regression outputs continuous values outside [0,1]. It is also sensitive to outliers which can shift the decision boundary drastically.*
2.  **Q: Explain the bias-variance trade-off in Logistic Regression.**
    *   *A: High regularization (small C) leads to high bias (underfitting). Low regularization (large C) leads to high variance (overfitting).*

---

## 3. Decision Trees (Iris Classification)
**Project Goal:** Multi-class classification of Iris flowers.

### 🧠 Technical Deep Dive
*   **Core Concept:** recursive partitioning of the data space. The tree splits data based on the feature that results in the highest **Information Gain** (reduction in entropy) or lowest **Gini Impurity**.
*   **Splitting Criteria:**
    *   **Gini Impurity:** $1 - \sum p_i^2$ (Faster to compute).
    *   **Entropy:** $- \sum p_i \log_2(p_i)$ (More computationally expensive).
*   **Overfitting:** Trees tend to overfit easily by growing too deep and memorizing noise.

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.tree.DecisionTreeClassifier`.
*   **Key Parameters:** `max_depth` (limits tree growth), `min_samples_split` (prevents splitting small nodes).
*   **Visualization:** `plot_tree` shows the actual decision logic.

### ❓ Key Interview Questions
1.  **Q: How do you prune a decision tree?**
    *   *A: Pre-pruning (stopping early via max_depth, min_samples_leaf) or Post-pruning (growing full tree then removing insignificant branches using Cost Complexity Pruning).*
2.  **Q: Why don't decision trees require feature scaling?**
    *   *A: They are rule-based, not distance-based. The split depends on the ordering of values, not their magnitude.*

---

## 4. Random Forest (Wine Quality)
**Project Goal:** Robust classification using an ensemble of trees.

### 🧠 Technical Deep Dive
*   **Core Concept:** An **Ensemble** method using **Bagging (Bootstrap Aggregation)**. It builds multiple decision trees on different random subsets of data (with replacement) and features.
*   **Feature Randomness:** At each split, it considers only a random subset of features. This decorrelates the trees, reducing variance.
*   **Prediction:** Majority vote (classification) or average (regression).

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.ensemble.RandomForestClassifier`.
*   **Key Parameter:** `n_estimators` (number of trees). More trees = more stable, but slower.

### ❓ Key Interview Questions
1.  **Q: What is Out-of-Bag (OOB) Error?**
    *   *A: Since each tree uses ~2/3 of the data, the remaining 1/3 (OOB samples) can be used for validation without a separate test set.*
2.  **Q: Random Forest vs. Decision Tree?**
    *   *A: RF has lower variance (less overfitting) but is less interpretable and slower to predict.*

---

## 5. Support Vector Machine (Breast Cancer)
**Project Goal:** High-margin classification for medical diagnosis.

### 🧠 Technical Deep Dive
*   **Core Concept:** Finds the hyperplane that maximizes the **Margin** (distance between the hyperplane and the nearest data points, called Support Vectors).
*   **Kernel Trick:** Maps data to a higher-dimensional space to make it linearly separable. Common kernels: Linear, Polynomial, RBF (Radial Basis Function).
*   **Parameters:**
    *   **C:** Regularization. High C = strict margin (can overfit). Low C = soft margin (allows misclassification).
    *   **Gamma:** Defines influence of a single point. High Gamma = close points have high weight (complex boundary).

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.svm.SVC`.
*   **Scaling:** **CRITICAL**. SVM is distance-based, so features must be scaled (StandardScaler) or the feature with the largest range will dominate.

### ❓ Key Interview Questions
1.  **Q: When would you use RBF kernel vs Linear kernel?**
    *   *A: Use Linear if features > samples (text classification). Use RBF if samples > features and non-linear relationship exists.*

---

## 6. K-Means Clustering (Customer Segmentation)
**Project Goal:** Unsupervised grouping of customers.

### 🧠 Technical Deep Dive
*   **Core Concept:** Iterative algorithm.
    1.  Initialize K centroids.
    2.  Assign points to nearest centroid.
    3.  Update centroids to the mean of assigned points.
    4.  Repeat until convergence.
*   **Objective:** Minimize **Inertia** (Sum of Squared Distances within clusters).

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.cluster.KMeans`.
*   **Elbow Method:** Used to find optimal K by plotting Inertia vs. K and looking for the "elbow".
*   **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters.

### ❓ Key Interview Questions
1.  **Q: How does K-Means handle outliers?**
    *   *A: Poorly. Centroids are means, which are sensitive to outliers. K-Medoids is more robust.*
2.  **Q: What if clusters are not spherical?**
    *   *A: K-Means fails. Use DBSCAN or Spectral Clustering for arbitrary shapes.*

---

## 7. KNN Classifier (Digit Recognition)
**Project Goal:** Simple, instance-based classification.

### 🧠 Technical Deep Dive
*   **Core Concept:** **Lazy Learner**. It doesn't "learn" a model. To predict, it finds the K nearest neighbors in the training set and takes the majority vote.
*   **Distance Metrics:** Euclidean (L2), Manhattan (L1), Minkowski.
*   **The "K":** Small K = noise sensitive (overfitting). Large K = smooth boundary (underfitting).

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.neighbors.KNeighborsClassifier`.
*   **Scaling:** Essential, as it relies on distance calculations.

### ❓ Key Interview Questions
1.  **Q: Why is KNN computationally expensive?**
    *   *A: It must calculate distance to *every* training point for *every* prediction. KD-Trees or Ball Trees can optimize this.*

---

## 8. Naive Bayes (Spam Detection)
**Project Goal:** Probabilistic text classification.

### 🧠 Technical Deep Dive
*   **Core Concept:** Based on **Bayes' Theorem**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
*   **"Naive" Assumption:** Assumes all features are **independent** given the class. This is rarely true in real life but works surprisingly well for text (Bag of Words).
*   **Types:**
    *   **Gaussian:** For continuous data.
    *   **Multinomial:** For discrete counts (e.g., word counts).
    *   **Bernoulli:** For binary features (word presence).

### 💻 Implementation Highlights
*   **Key Library:** `sklearn.naive_bayes.GaussianNB` (or MultinomialNB for text).

### ❓ Key Interview Questions
1.  **Q: What is the "Zero Frequency" problem?**
    *   *A: If a categorical variable has a category in test data not seen in training, probability becomes 0. Laplace Smoothing (adding 1) fixes this.*

---

## 9. Neural Networks (MNIST Digits)
**Project Goal:** Deep Learning for image classification.

### 🧠 Technical Deep Dive
*   **Core Concept:** Layers of neurons (perceptrons).
    *   **Forward Propagation:** Input -> Weights -> Activation -> Output.
    *   **Backpropagation:** Calculate loss -> Compute gradients (Chain Rule) -> Update weights.
*   **Activation Functions:**
    *   **ReLU:** $max(0, z)$. Solves vanishing gradient problem.
    *   **Softmax:** Converts output vector into probabilities summing to 1.
*   **Optimizer:** **Adam** (Adaptive Moment Estimation) is standard. It combines Momentum and RMSProp.

### 💻 Implementation Highlights
*   **Libraries:** `tensorflow.keras` (if available) or `sklearn.neural_network.MLPClassifier`.
*   **Architecture:** Input Layer (784 nodes) -> Hidden Layers (ReLU) -> Output Layer (10 nodes, Softmax).

### ❓ Key Interview Questions
1.  **Q: What is the Vanishing Gradient problem?**
    *   *A: In deep networks with Sigmoid/Tanh, gradients become tiny during backprop, stopping early layers from learning. ReLU fixes this.*
2.  **Q: Why do we need non-linear activation functions?**
    *   *A: Without them, a neural network is just a single linear regression model, no matter how many layers it has.*

---

## 10. Gradient Boosting (XGBoost)
**Project Goal:** State-of-the-art tabular data classification.

### 🧠 Technical Deep Dive
*   **Core Concept:** **Boosting**. Builds trees sequentially. Each new tree attempts to correct the **errors (residuals)** of the previous trees.
*   **Gradient Boosting:** Uses Gradient Descent to minimize the loss function by adding new trees.
*   **XGBoost (Extreme Gradient Boosting):** Optimized implementation.
    *   **Regularization:** L1/L2 built-in (unlike standard GBM).
    *   **Parallel Processing:** Handles sparse data and trains faster.
    *   **Tree Pruning:** Uses "max_depth" and prunes backwards.

### 💻 Implementation Highlights
*   **Library:** `xgboost` or `sklearn.ensemble.GradientBoostingClassifier`.
*   **Metrics:** ROC-AUC is often the primary metric for boosting models.

### ❓ Key Interview Questions
1.  **Q: XGBoost vs Random Forest?**
    *   *A: RF trains in parallel (bagging), XGBoost trains sequentially (boosting). XGBoost usually performs better but is more prone to overfitting if not tuned.*
2.  **Q: What is the learning rate in boosting?**
    *   *A: It shrinks the contribution of each tree. Lower learning rate requires more trees but generalizes better.*
