# Telecom Customer Churn Prediction

## Project Overview
This project analyzes telecom customer data to identify patterns associated with customer churn and develops machine learning models to predict customers who may leave a telecom service.

## Objectives
- Inspect and audit the raw dataset.
- Clean inconsistent values and handle missing data.
- Perform exploratory data analysis (EDA).
- Engineer features for machine learning.
- Train and compare classification models.
- Tune models and evaluate their performance.
- Present insights and predictions through a Streamlit dashboard.

## Dataset
The project uses a telecom customer dataset containing customer information, service subscriptions, tenure, billing details, support activity, satisfaction, and churn status.

The raw dataset is stored in `telecom_churn_raw.csv`.

## Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## Project Workflow
1. Data inspection and quality audit
2. Data cleaning and missing-value analysis
3. Exploratory data analysis
4. Feature engineering
5. Baseline model training
6. Hyperparameter tuning
7. Model comparison and evaluation
8. Interactive dashboard

## Models
The project evaluates Logistic Regression, Decision Tree, Random Forest, and XGBoost. Tuned Random Forest currently has the highest reported F1 score among the compared models.

## Current Best Model Results
- Accuracy: 73.39%
- Precision: 57.89%
- Recall: 67.90%
- F1 Score: 62.50%
- ROC-AUC: 79.27%

These are the current recorded evaluation results and should be regenerated and verified before final submission.

## Running the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the data-cleaning script:

```bash
python clean_data.py
```

Run exploratory data analysis:

```bash
python eda.py
```

Run model training:

```bash
python train_model.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Project Outputs
- `telecom_churn_clean.csv`: cleaned customer dataset
- `data/processed/telecom_churn_modeling.csv`: modeling dataset
- `reports/`: analysis reports and model comparisons
- `reports/plots/`: generated visualizations
- `models/`: saved machine learning models

## Important Note
Model performance depends on the dataset, preprocessing, train-test split, and evaluation settings. Results should be verified before making business decisions.