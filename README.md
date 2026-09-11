## Machine Learning Projects

### Regression

#### Used Car Price Prediction — Linear Models

Used-car price prediction using **Linear Regression, Ridge Regression, and Lasso Regression**, with preprocessing performed inside scikit-learn pipelines to avoid data leakage during cross-validation.

The project explores how regularization affects predictive performance and model sparsity while keeping the analysis focused specifically on the linear-model family.

**Highlights:**

- Linear Regression: R² ≈ 0.8765
- Linear Regression: MAE ≈ 791
- Linear Regression: RMSE ≈ 991
- Ridge Regression produced nearly identical predictive performance.
- Lasso achieved similar performance while setting **18 of 25 coefficients to zero**, producing a substantially sparser model.
- Numerical coefficients are interpreted after standardization, while categorical coefficients are interpreted relative to their reference categories.

[View project →](linear-regression/used-car-price-prediction/)

---

### Classification

#### Credit Card Fraud Detection — Gaussian Naive Bayes

Fraud detection on an imbalanced dataset of **1,000,000 transactions** using Gaussian Naive Bayes, stratified train/validation/test splitting, and validation-based decision-threshold selection.

Because only approximately **8.74% of transactions are fraudulent**, the project goes beyond accuracy and evaluates precision, recall, F1-score, ROC-AUC, and Average Precision.

**Highlights:**

- Selected decision threshold: 0.225
- Accuracy: 0.9703
- Precision: 0.7939
- Recall: 0.8916
- F1-score: 0.8399
- ROC-AUC: 0.9642
- Average Precision: 0.7289
- Threshold selection increased fraud recall substantially compared with the default 0.50 decision threshold.

[View project →](classification/credit-card-fraud-naive-bayes/)
