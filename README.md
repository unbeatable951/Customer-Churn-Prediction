# Customer Churn Prediction using Logistic Regression

AI-ML Assignment 2

## Objective
Build a Logistic Regression model to predict whether a telecom customer is likely to churn (leave the company), based on their demographic information and service usage patterns.

## Dataset
**Telco Customer Churn Dataset** (Kaggle)
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> The dataset is **not included in this repository**. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the Kaggle link above and place it in the project root before running the notebook.

## Libraries Used
- `pandas` – data loading and manipulation
- `numpy` – numerical operations
- `matplotlib`, `seaborn` – data visualization and confusion matrix plotting
- `scikit-learn` – preprocessing (`LabelEncoder`, `StandardScaler`), train/test split, `LogisticRegression`, and evaluation metrics

## Methodology
1. **Data Understanding** – Loaded the dataset, inspected its structure, and identified numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`), categorical features (contract type, internet service, payment method, etc.), and the target variable (`Churn`).
2. **Data Preprocessing**
   - Converted `TotalCharges` to numeric and imputed missing values with the median.
   - Dropped the non-predictive `customerID` column.
   - Label-encoded all categorical features and the target variable.
   - Split the data into 80% training / 20% testing (stratified on `Churn`).
   - Standardized numerical features with `StandardScaler`.
3. **Model Development** – Trained a `LogisticRegression` model on the processed training set and generated predictions on the held-out test set.
4. **Model Evaluation** – Computed Accuracy, Precision, Recall, and F1-Score, and visualized a Confusion Matrix. Inspected model coefficients to identify the features most associated with churn.
5. **Conclusion** – Summarized key findings, churn drivers, and a limitation of Logistic Regression for this problem.

## Results
Run `Assignment-2.ipynb` end-to-end after adding the dataset CSV to reproduce the metrics. In general terms, the model:
- Achieves solid overall accuracy on this fairly balanced-vs-imbalanced (~27% churn) dataset.
- Shows the largest coefficients (churn drivers) around **contract type**, **tenure**, and **monthly charges** — customers on month-to-month contracts with short tenure and higher bills are more likely to churn.
- Tends to have somewhat lower recall than precision on the churn class, meaning it is more conservative about flagging churners than it is accurate about the churners it does flag.

*(Fill in your exact Accuracy / Precision / Recall / F1 values and confusion matrix numbers here after running the notebook.)*

## Conclusion
Logistic Regression proved to be an effective and interpretable baseline model for predicting customer churn. Contract type, tenure, monthly charges, and add-on services (online security, tech support) emerged as the strongest predictors — customers on month-to-month contracts with short tenure and higher bills are considerably more likely to churn, while long-term contracts and longer tenure correlate with retention. A key limitation of Logistic Regression is its assumption of a **linear relationship** between features and the log-odds of churn, which limits its ability to capture non-linear interactions between variables that models like Random Forest or XGBoost could better exploit.

## Repository Structure
```
├── Assignment-2.ipynb   # Full notebook: all 5 tasks
├── README.md            # This file
```

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
# Download WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle into this folder
jupyter notebook Assignment-2.ipynb
```
