
# Telco Customer Churn Prediction
Predictive Modelling for Retention Strategy

Authors: Reynold Takura Choruma, Blanca Fernandez & Marisa Oliveira
Project Type: Individual – Data Analytics / Machine Learning  
Year: 2026

---

## Project Overview

A major telecom provider was experiencing a churn rate of around 26–27%, with the highest attrition among high‑value fibre‑optic subscribers. This threatened recurring revenue and increased customer acquisition costs.

This project develops an end‑to‑end predictive churn model to identify at‑risk customers early, enabling proactive retention strategies and more efficient allocation of business resources.

---

## Key Outcomes (Final Modelling Results)

- Identified **≈69% of churn‑risk customers** with the chosen model (Recall ≈ 0.69).  
- Reduced **false retention alerts from 315 to 168** per cycle (**–147 alerts, ~47% reduction**).  
- Improved churn detection **precision from 0.49 to 0.61** (~24% relative improvement).  
- Achieved **PR‑AUC ≈ 0.652** and **MCC ≈ 0.51** for the final model, vs. a weaker baseline logistic model.  
- Showed that **tree‑based models (Random Forest, XGBoost)** achieve slightly higher ROC/PR‑AUC, but with less interpretability than the selected logistic model.

---

## Business Impact

- Retention teams can focus on genuinely at‑risk customers, reducing wasted effort and improving campaign ROI.  
- Fewer false alerts with still‑high recall means **better protection of recurring revenue** without overloading operations.  
- Clear feature effects (contract type, tenure, monthly charges, services) help business stakeholders design targeted retention strategies.

---

## Technical Approach

### 1. Data Preparation

**Data cleaning**

- Worked from the Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).  
- Validated and resolved 11 missing `TotalCharges` values using tenure consistency checks.  
- Assigned zero total charges to customers with 0‑month tenure (new customers without billable history).  

**Feature engineering**

- Dropped identifier column `customerID`.  
- Applied one‑hot encoding to categorical service and contract features using `pd.get_dummies(drop_first=True)`.  
- Standardised numerical variables (`MonthlyCharges`, `TotalCharges`, `tenure`, etc.) with `StandardScaler` for models that require scaling.

### 2. Handling Class Imbalance

Class distribution before resampling (train set):

- ~73% non‑churn  
- ~27% churn  

To address this imbalance:

- Implemented **SMOTE (Synthetic Minority Oversampling Technique)** on the training dataset only, keeping the test set untouched and realistic.  
- After SMOTE, the training data is **balanced** (churn vs non‑churn) for the models that use it.

---

## Modelling Strategy

### 3. Evaluation Metrics

Because of class imbalance, **accuracy** was not used as a primary metric. Instead:

- **PR‑AUC (Precision–Recall AUC)** – how well the model ranks true churners ahead of non‑churners.  
- **MCC (Matthews Correlation Coefficient)** – balanced measure of overall prediction quality.  
- **Precision, Recall, F1 score** – to understand operational trade‑offs between catching more churners and limiting false alerts.  
- **ROC AUC** – overall ranking quality across thresholds.

All metrics are computed via a single helper:

```python
def get_metrics(model, X, y):
    # returns (precision, recall, f1, mcc, roc_auc, pr_auc)
4. Models and Results
4.1 Comparison Table
| Model                         | Precision | Recall | F1   | MCC   | ROC‑AUC | PR‑AUC |
| ----------------------------- | --------- | ------ | ---- | ----- | ------- | ------ |
| Logistic Regression (Vanilla) | 0.49      | 0.82   | 0.61 | ~0.40 | ~0.79   | ~0.57  |
| Logistic Regression (SMOTE)   | 0.61      | 0.69   | 0.65 | 0.51  | 0.80    | 0.652  |
| Random Forest (tuned)         | 0.68      | 0.51   | 0.58 | 0.47  | 0.845   | 0.660  |
| XGBoost (tuned)               | 0.51      | 0.79   | 0.62 | 0.46  | 0.846   | 0.665  |

(Exact values as printed in the notebook; table summarises the key pattern.)

LogReg (vanilla): very high recall but low precision → many false positives.

LogReg (SMOTE): better balance, higher MCC and PR‑AUC; sharp drop in false positives.

Random Forest / XGBoost: strongest ROC/PR‑AUC, but with more complexity and less interpretability.

Final Model Selection – Logistic Regression + SMOTE
Although XGBoost achieved the highest PR‑AUC (~0.665), the final recommended model is Logistic Regression with SMOTE because:

It delivers strong, balanced performance across Precision (0.61), Recall (0.69), F1 (0.65), PR‑AUC (~0.652), and MCC (~0.51).

Coefficients are interpretable, enabling a direct link between features and churn risk (e.g., contract type, tenure, charges, add‑on services).

Behaviour is stable, simple to retrain, and straightforward to deploy as an initial production model (standard scaler + logistic regression).

XGBoost is retained as a secondary, higher‑complexity model for future iterations if the business prioritises maximum PR‑AUC and can accept more complexity.

Key Business Insights
1️⃣ Contract Risk

Month‑to‑month customers show much higher churn risk than one‑ or two‑year contracts.
👉 Action: design incentives for contract upgrades (discounted bundles, loyalty rewards, auto‑renew offers).

2️⃣ Price Sensitivity

Churned customers typically pay higher median monthly charges (around £15 more per month in this dataset).
👉 Action: review pricing and value perception for high‑charge segments; consider targeted discounts or plan optimisation.

3️⃣ Critical Risk Window

The first 0–12 months of tenure show the highest attrition, especially for month‑to‑month and fibre‑optic customers.
👉 Action: strengthen onboarding, early‑life engagement, and proactive outreach in the first year.

Operational Impact (Baseline vs Final Model)
| Metric          | Baseline Model (Vanilla LogReg) | Final Model (LogReg + SMOTE) |
| --------------- | ------------------------------- | ---------------------------- |
| False Positives | 315                             | 168                          |
| Precision       | 0.49                            | 0.61                         |
| Recall          | 0.82                            | 0.69                         |
| MCC             | ~0.40                           | 0.51                         |
Retention teams receive 147 fewer false alerts per scoring cycle, while still capturing the majority of true churners. This improves efficiency, reduces “alert fatigue”, and focuses effort on the customers most likely to churn.
Project Structure
.
├── 01_eda_telco_churn.ipynb        # EDA, churn patterns, data cleaning decisions
├── 02_modeling_telco_churn.ipynb   # Preprocessing, SMOTE, models, evaluation, selection
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
Tools & Technologies
Python

Pandas, NumPy

Scikit‑Learn

imbalanced‑learn (SMOTE)

XGBoost

Matplotlib, Seaborn

Jupyter Notebook

