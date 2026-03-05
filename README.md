Yes. For GitHub and LinkedIn portfolios, a README is actually better when it reads like a clean document instead of a technical notebook dump. Removing the code blocks makes it more readable for recruiters, hiring managers, and non-technical stakeholders.

Below is your clean README version (no code formatting, structured like a case study).



# Telco Customer Churn Prediction

# Predictive Modelling for Retention Strategy

Authors: Reynold Takura Choruma, Blanca Fernandez & Marisa Oliveira
Project Type: Group – Data Analytics / Machine Learning
Year: 2026



# Project Overview

A major telecom provider was experiencing a churn rate of approximately 26–27%, with the highest attrition occurring among high-value fibre-optic subscribers. This posed a risk to recurring revenue and increased the cost of acquiring replacement customers.

This project develops an end-to-end predictive churn model designed to identify at-risk customers early. The objective is to enable proactive retention strategies and allow the business to allocate retention resources more efficiently.



# Key Outcomes (Final Modelling Results)
	•	Identified approximately 69% of churn-risk customers using the selected model (Recall ≈ 0.69).
	•	Reduced false retention alerts from 315 to 168 per cycle, a reduction of 147 alerts (~47%).
	•	Improved churn detection precision from 0.49 to 0.61, representing roughly a 24% relative improvement.
	•	Achieved PR-AUC ≈ 0.652 and MCC ≈ 0.51 for the final model.
	•	Demonstrated that tree-based models (Random Forest and XGBoost) can achieve slightly higher ROC-AUC and PR-AUC, but with reduced interpretability compared to logistic regression.



## Business Impact

The model allows the business to move from reactive churn management to proactive retention intervention.

 Key operational benefits include:
	•	Retention teams can focus on customers genuinely at risk, reducing wasted outreach effort.
	•	Fewer false alerts allow retention campaigns to operate more efficiently and with higher ROI.
	•	Clear feature relationships (such as contract type, tenure, monthly charges, and add-on services) provide actionable insight for business decision-makers.



## Technical Approach

#### Data Preparation

#### Data Cleaning

 The project uses the Telco Customer Churn dataset.
	•	11 missing values in the TotalCharges field were validated and corrected using tenure consistency checks.
	•	Customers with 0-month tenure were assigned zero total charges, reflecting new customers without billable history.

 Feature Engineering
	•	The identifier column was removed to avoid leakage.
	•	Categorical service and contract variables were converted using one-hot encoding.
	•	Numerical variables such as MonthlyCharges, TotalCharges, and tenure were standardised to stabilise model training.



### Customer churn datasets are naturally imbalanced.

Original distribution:
	•	~73% non-churn customers
	•	~27% churn customers

To address this imbalance:
	•	SMOTE (Synthetic Minority Oversampling Technique) was applied to the training dataset only, ensuring the test set remained realistic.

This produced a balanced training dataset, improving the model’s ability to detect churn events.



## Modelling Strategy

### Evaluation Metrics

Because of class imbalance, accuracy was not used as the primary metric.

Instead, models were evaluated using:

PR-AUC (Precision–Recall AUC)
Measures how well the model ranks true churners ahead of non-churners.

MCC (Matthews Correlation Coefficient)
Provides a balanced measure of prediction quality across all confusion matrix components.

Precision, Recall, and F1 Score
Used to evaluate the operational trade-off between identifying churners and avoiding excessive false alerts.

ROC-AUC
Measures overall ranking performance across decision thresholds.



# Model Comparison

### Model performance summary:

| Model                         | Precision | Recall | F1   | MCC   | ROC‑AUC | PR‑AUC |
| ----------------------------- | --------- | ------ | ---- | ----- | ------- | ------ |
| Logistic Regression (Vanilla) | 0.49      | 0.82   | 0.61 | ~0.40 | ~0.79   | ~0.57  |
| Logistic Regression (SMOTE)   | 0.61      | 0.69   | 0.65 | 0.51  | 0.80    | 0.652  |
| Random Forest (tuned)         | 0.68      | 0.51   | 0.58 | 0.47  | 0.845   | 0.660  |
| XGBoost (tuned)               | 0.51      | 0.79   | 0.62 | 0.46  | 0.846   | 0.665  |

 Key observations:
	•	Vanilla Logistic Regression captures most churners but generates too many false alerts.
	•	Logistic Regression with SMOTE produces the best balance between precision and recall.
	•	Random Forest and XGBoost show strong ranking performance but introduce additional model complexity.



# Final Model Selection

The final recommended model is Logistic Regression with SMOTE.

Although XGBoost achieved slightly higher PR-AUC, Logistic Regression was selected because:
	•	It provides balanced performance across all metrics.
	•	Model coefficients remain interpretable, allowing business stakeholders to understand key churn drivers.
	•	The model is stable, transparent, and easy to deploy.

Tree-based models remain potential candidates for future iterations if maximum predictive performance becomes the primary objective.



# Key Business Insights

## Contract Risk

Customers on month-to-month contracts exhibit significantly higher churn risk compared with customers on one- or two-year contracts.

### Recommendation:
Introduce targeted incentives encouraging migration toward longer-term contracts, such as loyalty rewards or bundled service discounts.



## Price Sensitivity

Customers who churn tend to have higher monthly charges, often around £15 more per month in this dataset.

### Recommendation:
Review pricing strategies and perceived value in high-charge segments, and test targeted plan optimisation rather than broad price reductions.



## Critical Risk Window

The first 12 months of tenure represent the highest churn risk period, particularly among fibre-optic and month-to-month customers.

### Recommendation:
Strengthen onboarding and early-lifecycle engagement strategies, including proactive outreach and service guidance during the first year.



## Operational Impact

Metric	Baseline Model	Final Model
| Metric          | Baseline Model (Vanilla LogReg) | Final Model (LogReg + SMOTE) |
| --------------- | ------------------------------- | ---------------------------- |
| False Positives | 315                             | 168                          |
| Precision       | 0.49                            | 0.61                         |
| Recall          | 0.82                            | 0.69                         |
| MCC             | ~0.40                           | 0.51                         |

The final model reduces 147 false alerts per scoring cycle while still identifying the majority of churn-risk customers.

This improves operational efficiency and helps retention teams focus on the customers most likely to churn.



# Project Structure

Project files include:
	•	Exploratory data analysis notebook
	•	Modelling and evaluation notebook
	•	Telco churn dataset
	•	Documentation (README)


# Tools & Technologies

Python
Pandas
NumPy
Scikit-Learn
imbalanced-learn (SMOTE)
XGBoost
Matplotlib
Seaborn
Jupyter Notebook



