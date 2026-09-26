# Health Insurance Charges: EDA & Regression
A learning project: analysis of a public health-insurance dataset,covering data cleaning, exploratory analysis with visualizations, and linear regression.

## Methods
- Cleaning: missing-value check, deduplication, categorical encoding
- EDA: distributions, group comparisons, correlations, BMI × smoking interaction
- Modeling: linear regression on dummy-encoded features, 80/20 train/test split, evaluated with R² and MAE

## Key findings
- Smoking is the dominant cost driver (smokers pay several times more on average)
- Strong BMI × smoking interaction: BMI–charges correlation ≈ 0.81 among smokers vs ≈ 0.08 among non-smokers
- Regional variation: southeast has the highest average charges

## How to run
pip install -r requirements.txt
python insurance_analysis.py

## Contents
- `insurance_analysis.py` — full pipeline (cleaning → EDA → visualization → regression)
- `01–06_*.png` — figures
- `insurance.xlsx` — dataset (public Kaggle healthcare insurance data)
