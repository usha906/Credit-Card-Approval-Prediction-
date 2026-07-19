# Credit Approval Prediction System

A complete, runnable ML pipeline that predicts whether a credit/loan
application should be approved, plus a small web UI to try it interactively.

## What's included

```
credit_approval_system/
├── data_generator.py     # Builds a synthetic-but-realistic applicant dataset
├── train_model.py        # Preprocessing + trains/compares 3 models, saves the best
├── predict.py            # CLI: predict for one applicant (JSON) or a batch (CSV)
├── app.py                # Flask web app with a form UI
├── templates/index.html  # UI template used by app.py
├── requirements.txt
├── data/                 # Generated dataset lives here
└── models/               # Trained pipeline + metrics.json land here
```

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Generate data (optional — training does this automatically if missing)

```bash
python data_generator.py
```

This creates `data/credit_applications.csv` with ~5,000 synthetic applicants:
demographics, income, credit score, debt, loan terms, and an `approved`
label. The label isn't random — it's generated from a realistic underwriting
rule (credit score, debt-to-income, late payments, bankruptcy history,
employment, etc.) plus noise, so the models have real signal to learn.

**Using real data instead:** replace `data/credit_applications.csv` with your
own dataset (same column names), or edit `NUMERIC_FEATURES` /
`CATEGORICAL_FEATURES` / `TARGET_COL` at the top of `train_model.py` to match.

## 3. Train

```bash
python train_model.py
```

This builds a `ColumnTransformer` (median/mode imputation, scaling, one-hot
encoding), trains Logistic Regression, Random Forest, and Gradient Boosting,
evaluates each on a held-out test set (accuracy, precision, recall, F1,
ROC-AUC, confusion matrix), and saves the best one (by ROC-AUC) to
`models/best_model.joblib`. Full metrics are written to `models/metrics.json`.

## 4. Predict

**Single applicant:**
```bash
python predict.py --json '{"age":35,"annual_income":72000,"employment_years":6,
  "employment_type":"Salaried","home_ownership":"MORTGAGE","credit_score":710,
  "existing_debt":8000,"loan_amount":15000,"loan_term_months":36,
  "loan_purpose":"debt_consolidation","num_credit_lines":5,
  "num_late_payments":0,"has_bankruptcy":0,"debt_to_income":0.32}'
```

**Batch of applicants from a CSV** (must contain all feature columns):
```bash
python predict.py --csv data/new_applicants.csv
# writes data/new_applicants_predictions.csv
```

**No arguments** runs a built-in demo applicant so you can sanity-check the
model immediately after training.

## 5. Interactive web UI

```bash
python app.py
```

Open http://127.0.0.1:5000 — fill in the form and submit to see the
decision (Approved/Denied) and the model's approval probability.

## Notes on extending this for production use

- **Fairness/compliance**: real credit decisions are regulated (e.g. ECOA/Reg
  B in the US). Don't use protected-class attributes (race, sex, religion,
  national origin, age in some contexts) as features, and validate for
  disparate impact before any real-world use.
- **Explainability**: for real deployments, add SHAP or permutation feature
  importance so denied applicants can be given adverse-action reasons.
- **Class imbalance / threshold tuning**: the default decision threshold is
  0.5; in practice you'd tune this against business cost of false
  approvals vs. false denials.
- **Data drift**: retrain periodically and monitor incoming applicant
  distributions vs. training data.
