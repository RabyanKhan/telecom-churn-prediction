# Telecom Customer Churn Prediction

A machine learning project that analyzes telecom customer data, identifies patterns associated with customer churn, and compares predictive models to help understand customer retention.

## Project Overview

Customer churn occurs when customers stop using a company's services. Predicting potential churn can help businesses identify customers who may need additional attention and improve retention strategies.

This project covers data inspection, data quality auditing, preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, evaluation, and an interactive Streamlit application.

## Objectives

- Inspect and audit the dataset for quality issues.
- Analyze missing values, duplicates, and inconsistent data.
- Clean and prepare the data for machine learning.
- Perform exploratory data analysis and visualize important patterns.
- Engineer features for predictive modeling.
- Train and compare classification algorithms.
- Tune selected models and evaluate their performance.
- Save trained models and provide an interactive prediction interface.

## Technologies Used

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn, XGBoost
- **Model persistence:** Joblib
- **Interactive application:** Streamlit
- **Development tools:** VS Code, Jupyter Notebook, Git, GitHub

## Dataset

The project uses a telecom customer dataset containing customer, service, billing, tenure, and churn-related information.

**The dataset CSV files are not included in this public repository.** This avoids publishing data that may be sensitive or may not be licensed for redistribution.

To run the data-processing and training scripts:

1. Obtain the dataset from an authorized source.
2. Place the raw CSV in the project root directory.
3. Name it `telecom_churn_raw.csv`.
4. Ensure that its columns match those expected by the project scripts.

The cleaning script is designed to generate the cleaned data used in subsequent analysis. Exact results may vary depending on the dataset version and preprocessing decisions.

## Project Workflow

1. Raw data inspection and quality auditing
2. Missing-value and duplicate analysis
3. Data cleaning and validation
4. Exploratory data analysis
5. Feature engineering
6. Baseline model training
7. Hyperparameter tuning
8. Model comparison and evaluation
9. Saving trained models
10. Interactive prediction through Streamlit

## Machine Learning Models

The project evaluates the following classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Random Forest and XGBoost are also evaluated after hyperparameter tuning.

## Recorded Best Model Results

The current project reports the following results for the tuned Random Forest model:

| Metric | Recorded result |
|---|---:|
| Accuracy | 73.39% |
| Precision | 57.89% |
| Recall | 67.90% |
| F1 Score | 62.50% |
| ROC-AUC | 79.27% |

These are the project's recorded evaluation results. They should be reproduced and verified using the final evaluation pipeline before being treated as definitive results. The most suitable model depends on the business objective and the relative costs of missed churners and false alarms.

## Repository Structure

```text
telecom-churn-prediction/
├── app.py
├── clean_data.py
├── compare_models.py
├── create_fifth_plot.py
├── create_presentation.py
├── data_quality.py
├── eda.py
├── inspect_data.py
├── missing_analysis.py
├── train_model.py
├── tune_models.py
├── Telecom_Churn_Prediction.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── decision_tree_baseline.joblib
│   ├── logistic_regression_baseline.joblib
│   ├── random_forest_baseline.joblib
│   ├── random_forest_tuned.joblib
│   ├── telecom_churn_best_model.joblib
│   ├── xgboost_baseline.joblib
│   └── xgboost_tuned.joblib
└── reports/
    ├── Telecom_Churn_Prediction_Presentation.pptx
    ├── plots/
    └── screenshots/
```

The `reports/` directory also contains model comparisons, cleaning summaries, data quality reports, missing-value analyses, and tuning results. Generated datasets and some intermediate files are excluded from the public repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/RabyanKhan/telecom-churn-prediction.git
cd telecom-churn-prediction
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure the authorized raw dataset is available at the expected location before running data-processing scripts.

## Running the Project

Run the scripts from the project root in the following order, as needed:

**1. Inspect the dataset**

```bash
python inspect_data.py
```

**2. Audit data quality**

```bash
python data_quality.py
```

**3. Clean the dataset**

```bash
python clean_data.py
```

**4. Perform exploratory data analysis**

```bash
python eda.py
```

**5. Train the baseline models**

```bash
python train_model.py
```

**6. Tune the selected models**

```bash
python tune_models.py
```

**7. Compare model performance**

```bash
python compare_models.py
```

**8. Launch the Streamlit application**

```bash
python -m streamlit run app.py
```

The required input files must be present, and each script's expected inputs and outputs should be checked before running the complete pipeline.

## Project Outputs

- **`models/`** — serialized trained machine learning models.
- **`reports/`** — data quality reports, model comparisons, tuning results, charts, and presentation.
- **`reports/plots/`** — exploratory analysis and model evaluation visualizations.
- **`reports/screenshots/`** — a screenshot of a prediction result.
- **`Telecom_Churn_Prediction.ipynb`** — notebook documenting the project workflow.

## Limitations

- Model performance depends on dataset quality, preprocessing, feature selection, and evaluation methodology.
- Recorded metrics should be reproduced before drawing conclusions.
- Predictions indicate potential churn; they do not guarantee that a customer will leave.
- The public repository does not include the raw dataset, so users must obtain an authorized copy to reproduce data-dependent steps.

## Future Improvements

- Validate performance on additional datasets.
- Improve recall while monitoring precision and false-positive rates.
- Add explainability techniques to understand the factors influencing predictions.
- Improve the dashboard with additional model insights.
- Establish a reproducible evaluation pipeline.

## Author

**Rabyan Khan**

GitHub: [@RabyanKhan](https://github.com/RabyanKhan)

## License

No license has currently been specified. All rights to the project's contents remain subject to their applicable ownership and licensing terms.
