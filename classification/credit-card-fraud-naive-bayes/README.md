# Credit Card Fraud Detection with Gaussian Naive Bayes

Credit card fraud classification project using **Gaussian Naive Bayes** with a focus on imbalanced-class evaluation and decision-threshold selection.

Rather than relying only on accuracy, the project evaluates the model using precision, recall, F1-score, ROC-AUC, and Average Precision.

## Project Overview

The dataset contains **1,000,000 transactions** and seven predictive features.

The target variable is imbalanced:

- Non-fraud transactions: 91.26%
- Fraudulent transactions: 8.74%

Because of this imbalance, accuracy alone can provide a misleading view of model performance.

## Features

The model uses the following variables:

- `distance_from_home`
- `distance_from_last_transaction`
- `ratio_to_median_purchase_price`
- `repeat_retailer`
- `used_chip`
- `used_pin_number`
- `online_order`

Target:

- `fraud`

## Methodology

The workflow consists of:

1. Dataset exploration and class-distribution analysis.
2. Stratified train/validation/test splitting.
3. Feature scaling using `StandardScaler`.
4. Training a `GaussianNB` classifier.
5. Generating fraud probabilities on the validation set.
6. Comparing multiple decision thresholds.
7. Selecting the threshold with the highest validation F1-score.
8. Performing the final evaluation on the test set.

The data split is approximately:

- Training: 64%
- Validation: 16%
- Test: 20%

The test set is reserved for final evaluation and is not used to select the decision threshold.

## Decision Threshold Analysis

The default decision threshold of `0.50` produced relatively low recall for fraudulent transactions.

Several thresholds were therefore evaluated using the validation set:

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.150 | 0.6509 | 0.9408 | 0.7694 |
| 0.175 | 0.7111 | 0.9239 | 0.8037 |
| 0.200 | 0.7624 | 0.9098 | 0.8296 |
| **0.225** | **0.7964** | **0.8872** | **0.8394** |
| 0.250 | 0.8024 | 0.8446 | 0.8230 |
| 0.500 | 0.7944 | 0.5917 | 0.6782 |

A threshold of **0.225** achieved the highest validation F1-score and was selected before final test evaluation.

## Final Test Results

Using the selected threshold of `0.225`:

| Metric | Score |
|---|---:|
| Accuracy | 0.9703 |
| Precision | 0.7939 |
| Recall | 0.8916 |
| F1-score | 0.8399 |
| ROC-AUC | 0.9642 |
| Average Precision | 0.7289 |

The model detected approximately **89% of fraudulent transactions** while maintaining approximately **79% precision**.

### Test Confusion Matrix

```text
[[178473   4046]
 [  1895  15586]]
```

This corresponds to:

- 178,473 true negatives
- 4,046 false positives
- 1,895 false negatives
- 15,586 true positives

## Key Takeaway

Threshold selection had a substantial effect on fraud detection.

Using the default threshold of `0.50`, fraud recall was approximately `0.59`. After selecting the threshold using validation data, recall increased to approximately `0.89` while maintaining similar precision.

This illustrates why classification metrics and decision thresholds should be considered carefully when working with imbalanced datasets.

## Technologies

- Python
- Pandas
- scikit-learn
- Jupyter Notebook / Google Colab

## Dataset

The dataset is **not included in this repository**.

To reproduce the notebook, place the dataset at:

```text
data/card_transdata.csv
```

The expected dataset contains the seven predictive features listed above and the target column `fraud`.

## Repository Structure

```text
credit-card-fraud-naive-bayes/
├── README.md
├── credit_card_fraud_naive_bayes.ipynb
├── .gitignore
└── data/
    └── .gitkeep
```

## Limitations

Gaussian Naive Bayes assumes that features follow class-conditional Gaussian distributions and treats features as conditionally independent.

Some variables in this dataset are binary, so these assumptions are not perfectly aligned with every feature.

Additionally, the selected decision threshold reflects the balance between precision and recall measured on this validation split. In a production fraud-detection system, the optimal threshold would depend on the business costs associated with false positives and false negatives.

## Author

José Alfredo Morales García

Engineering Physics | Data Science | Machine Learning
