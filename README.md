# Customer Churn Prediction (Telco) — Python

A Python data project focused on **predicting customer churn** (whether a customer will leave) using the classic *Telco Customer Churn* dataset.  
The repository includes **data cleaning**, **EDA (exploratory data analysis)**, and **machine learning modeling** with comparisons across multiple classification algorithms.

---

##  Repository contents

### Notebooks
- **`cleaning.ipynb`**  
  Data cleaning and preprocessing, producing a cleaned dataset.

- **`eda_telco_churn.ipynb`**  
  Exploratory Data Analysis to understand key churn drivers and patterns.

- **`Which customers are more likely to churn.ipynb`**  
  Additional analysis focused on identifying profiles/segments more likely to churn.

- **`First_modeling_ML (1).ipynb`**  
  Machine learning pipeline with model training and evaluation (includes model comparison + curves/metrics).

### Data
- **`customer_churn_clean.csv`**  
  Cleaned dataset generated from the cleaning process (ready for modeling).

### Visual outputs
- `algorithm_tournament_comparison.png`
- `metrics_comparison_with_mcc.png`
- `model_comparison_visual.png`
- `pr_curve_comparison.png`

---

##  Dataset

This project is based on the widely used *Telco Customer Churn* dataset (original file name often used in notebooks is:  
`WA_Fn-UseC_-Telco-Customer-Churn.csv`).

The repo already contains a cleaned version: **`customer_churn_clean.csv`**, so you can start directly with modeling if you want.

---

##  Tech stack

- **Language:** Python
- **Typical libraries used:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Imbalanced learning / boosting (if used in your notebook):** imbalanced-learn (SMOTE), xgboost
