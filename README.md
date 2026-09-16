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

#### Customer Churn Prediction — Tree Ensembles

Customer churn prediction using **Decision Tree, Random Forest, and Gradient Boosting** with leakage-safe preprocessing, stratified cross-validation, hyperparameter tuning, decision-threshold analysis, and feature interpretation.

The project emphasizes generalization and the business trade-off between detecting more at-risk customers and limiting false retention alerts.

**Highlights:**

- Unrestricted Decision Tree exposed severe overfitting: train ROC-AUC 1.0000 vs. test ROC-AUC 0.6504.
- Regularization improved test ROC-AUC to 0.8324 with recall 0.7594.
- Tuned Random Forest reached ROC-AUC 0.8428 and precision 0.6678.
- Tuned Gradient Boosting achieved the strongest test ROC-AUC: **0.8471**.
- Out-of-fold threshold optimization raised Gradient Boosting recall from 0.5107 at the default threshold to 0.7246 at the max-F1 threshold.
- Permutation importance identified `tenure`, `Contract`, and `InternetService` as the strongest predictive inputs.

[View project →](classification/customer-churn-tree-ensembles/)
