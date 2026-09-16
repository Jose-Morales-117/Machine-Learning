# Customer Churn Prediction with Tree-Based Models

Customer churn classification project comparing **Decision Tree, Random Forest, and Gradient Boosting** models with leakage-safe preprocessing, stratified cross-validation, hyperparameter tuning, decision-threshold analysis, and feature interpretation.

The objective is not only to maximize classification performance, but also to understand the trade-off between detecting customers likely to churn and limiting false retention alerts.

## Project Overview

The dataset contains **7,043 customers** and 19 predictive features after excluding the customer identifier.

Target distribution:

- No churn: 73.46%
- Churn: 26.54%

Because the target is moderately imbalanced, the project evaluates models using precision, recall, F1-score, and ROC-AUC rather than relying on accuracy alone.

## Data Preparation

Key preprocessing decisions include:

- `customerID` is retained in the original dataset for traceability but excluded from model features.
- `TotalCharges` is converted from text to numeric.
- Eleven empty `TotalCharges` values correspond to customers with `tenure = 0` and are interpreted as customers who have not accumulated charges yet, so they are set to 0.
- Categorical variables are encoded with `OneHotEncoder(handle_unknown="ignore")` inside a `ColumnTransformer`.
- Numeric variables are passed through without scaling because the primary models are tree-based.
- The dataset is split 80/20 using a stratified train/test split.

## Exploratory Analysis

Initial EDA showed that churn is strongly associated with:

- Shorter customer tenure.
- Month-to-month contracts.
- Fiber-optic internet service.
- Lack of Online Security and Tech Support subscriptions.
- Electronic-check payment method.
- Higher monthly charges.

These relationships are treated as **associations, not causal effects**.

An interaction analysis also showed that fiber-optic customers had higher churn rates than DSL customers within each contract category, suggesting that `InternetService` contains predictive information beyond `Contract` alone.

## Methodology

The workflow consists of:

1. Data auditing and cleaning.
2. Exploratory churn analysis.
3. Stratified train/test splitting.
4. Leakage-safe preprocessing with scikit-learn pipelines.
5. Majority-class Dummy baseline.
6. Unrestricted Decision Tree to illustrate overfitting.
7. Decision Tree regularization using 5-fold stratified `GridSearchCV`.
8. Random Forest baseline and hyperparameter tuning.
9. Gradient Boosting baseline and hyperparameter tuning.
10. Model comparison using classification metrics and ROC-AUC.
11. Decision-threshold selection using out-of-fold training probabilities.
12. Feature interpretation using tree-based importance and permutation importance.

## Baseline

The majority-class Dummy classifier achieved:

| Metric | Score |
|---|---:|
| Accuracy | 0.7346 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1-score | 0.0000 |
| ROC-AUC | 0.5000 |

This illustrates why accuracy alone is misleading for this problem: predicting every customer as non-churn already produces approximately 73.5% accuracy.

## Decision Tree

An unrestricted Decision Tree severely overfit the training data:

| Metric | Train | Test |
|---|---:|---:|
| Accuracy | 0.9980 | 0.7225 |
| Precision | 0.9993 | 0.4781 |
| Recall | 0.9933 | 0.4973 |
| F1-score | 0.9963 | 0.4875 |
| ROC-AUC | 1.0000 | 0.6504 |

Tree complexity:

- Depth: 22
- Leaves: 1,102

After 5-fold stratified GridSearchCV, the best regularized configuration was:

```text
class_weight = balanced
max_depth = 5
min_samples_leaf = 10
min_samples_split = 2
```

Best cross-validation ROC-AUC: **0.8300**

Final test performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.7559 |
| Precision | 0.5279 |
| Recall | 0.7594 |
| F1-score | 0.6228 |
| ROC-AUC | 0.8324 |

The regularized tree reduced depth from 22 to 5 and leaves from 1,102 to 31 while substantially improving generalization.

## Random Forest

The tuned Random Forest used:

```text
n_estimators = 200
max_depth = 8
min_samples_leaf = 10
max_features = sqrt
class_weight = None
```

Best cross-validation ROC-AUC: **0.8484**

Final test performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.8020 |
| Precision | 0.6678 |
| Recall | 0.5053 |
| F1-score | 0.5753 |
| ROC-AUC | 0.8428 |

Compared with the regularized Decision Tree, Random Forest provided stronger overall risk discrimination and precision, but lower recall at the default 0.50 threshold.

## Gradient Boosting

The tuned Gradient Boosting model used:

```text
learning_rate = 0.1
max_depth = 1
min_samples_leaf = 10
n_estimators = 200
```

Best cross-validation ROC-AUC: **0.8504**

Final test performance at the default 0.50 threshold:

| Metric | Score |
|---|---:|
| Accuracy | 0.8013 |
| Precision | 0.6632 |
| Recall | 0.5107 |
| F1-score | 0.5770 |
| ROC-AUC | 0.8471 |

The train ROC-AUC was 0.8563, while cross-validation and test ROC-AUC were 0.8504 and 0.8471 respectively, indicating stable generalization.

## Decision Threshold Analysis

Out-of-fold training probabilities were used to choose thresholds without optimizing directly on the test set.

Two decision policies were considered:

| Policy | Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Default | 0.500 | 0.6632 | 0.5107 | 0.5770 |
| Max F1 | 0.355 | 0.5717 | 0.7246 | 0.6392 |
| Target ~75% recall | 0.320 | 0.5428 | 0.7460 | 0.6284 |

The **0.355 threshold** provides the strongest overall balance between precision and recall, while **0.320** represents a retention-oriented policy when missing churners is more costly than generating additional false alerts.

ROC-AUC remains 0.8471 because it evaluates ranking performance across thresholds rather than at a single classification cutoff.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Unrestricted Tree | 0.7225 | 0.4781 | 0.4973 | 0.4875 | 0.6504 |
| Regularized Tree | 0.7559 | 0.5279 | **0.7594** | 0.6228 | 0.8324 |
| Random Forest Baseline | 0.7835 | 0.6211 | 0.4733 | 0.5372 | 0.8207 |
| Tuned Random Forest | 0.8020 | **0.6678** | 0.5053 | 0.5753 | 0.8428 |
| Gradient Boosting Baseline | **0.8027** | 0.6655 | 0.5160 | 0.5813 | 0.8434 |
| Tuned Gradient Boosting | 0.8013 | 0.6632 | 0.5107 | 0.5770 | **0.8471** |

No single model dominates every objective. The regularized Decision Tree provides the highest recall at the default threshold, while tuned Gradient Boosting provides the strongest overall ROC-AUC.

## Feature Interpretation

Tree-based feature importance highlighted:

1. Month-to-month contract
2. Tenure
3. Fiber-optic internet service
4. No Online Security subscription
5. Electronic-check payment method
6. No Tech Support subscription

Permutation importance on the original input variables identified the strongest contributors to ROC-AUC as:

1. `tenure`
2. `Contract`
3. `InternetService`
4. `OnlineSecurity`
5. `PaymentMethod`
6. `TechSupport`

This largely confirms the patterns observed during EDA.

Feature importance reflects predictive usefulness and **must not be interpreted as causality**.

## Key Takeaways

- Accuracy alone was insufficient because a majority-class baseline already achieved approximately 73.5% accuracy while detecting no churners.
- An unrestricted Decision Tree severely overfit the training data.
- Cross-validation and regularization reduced Decision Tree complexity and substantially improved out-of-sample performance.
- Random Forest improved risk discrimination and precision through ensembling.
- Gradient Boosting achieved the strongest ROC-AUC, approximately **0.847** on the test set.
- Decision-threshold selection materially changed the precision/recall trade-off without retraining the model.
- For a retention-oriented use case, threshold selection should depend on the relative business cost of false positives and false negatives.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook / Google Colab

## Dataset

The dataset is **not included in this repository**.

To reproduce the notebook, place the Telco Customer Churn CSV at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Repository Structure

```text
customer-churn-tree-ensembles/
├── README.md
├── customer_churn_tree_ensembles.ipynb
├── .gitignore
└── data/
    └── .gitkeep
```

## Limitations

- The dataset is relatively small and should not be assumed to represent production telecom populations.
- Observed feature relationships are associative rather than causal.
- Feature importance can distribute or bias importance across encoded features and correlated variables.
- The decision threshold depends on the assumed business objective and cost of false positives versus false negatives.
- The holdout test set was used as a development benchmark during model comparison; a production-grade workflow would benefit from a fresh untouched holdout set or nested cross-validation for final model-selection claims.

## Author

José Alfredo Morales García

Engineering Physics | Data Science | Machine Learning
