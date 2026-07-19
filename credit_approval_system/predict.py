"""
predict.py
----------
Load the trained pipeline and predict approval for:
  1) A single applicant, via --json '{"age": 35, ...}'
  2) A batch of applicants, via --csv path/to/applicants.csv
  3) A built-in interactive demo if run with no arguments.

Examples:
    python predict.py
    python predict.py --json '{"age":35,"annual_income":72000,"employment_years":6,
        "employment_type":"Salaried","home_ownership":"MORTGAGE","credit_score":710,
        "existing_debt":8000,"loan_amount":15000,"loan_term_months":36,
        "loan_purpose":"debt_consolidation","num_credit_lines":5,
        "num_late_payments":0,"has_bankruptcy":0,"debt_to_income":0.32}'
    python predict.py --csv data/new_applicants.csv
"""

import argparse
import json
import sys

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"


def load_pipeline():
    try:
        bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("No trained model found. Run `python train_model.py` first.")
        sys.exit(1)
    return bundle["pipeline"], bundle["features"], bundle["model_name"]


def predict_df(df: pd.DataFrame, pipeline, features) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[features]
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    out = df.copy()
    out["approval_probability"] = proba.round(4)
    out["decision"] = ["Approved" if p == 1 else "Denied" for p in pred]
    return out


def demo_applicant() -> dict:
    return {
        "age": 35,
        "annual_income": 72000,
        "employment_years": 6,
        "employment_type": "Salaried",
        "home_ownership": "MORTGAGE",
        "credit_score": 710,
        "existing_debt": 8000,
        "loan_amount": 15000,
        "loan_term_months": 36,
        "loan_purpose": "debt_consolidation",
        "num_credit_lines": 5,
        "num_late_payments": 0,
        "has_bankruptcy": 0,
        "debt_to_income": 0.32,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict credit approval.")
    parser.add_argument("--json", type=str, help="Single applicant as a JSON string")
    parser.add_argument("--csv", type=str, help="Path to CSV of applicants")
    args = parser.parse_args()

    pipeline, features, model_name = load_pipeline()
    print(f"Using model: {model_name}\n")

    if args.csv:
        df = pd.read_csv(args.csv)
        result = predict_df(df, pipeline, features)
        print(result[["decision", "approval_probability"]])
        out_path = args.csv.replace(".csv", "_predictions.csv")
        result.to_csv(out_path, index=False)
        print(f"\nSaved predictions -> {out_path}")

    elif args.json:
        applicant = json.loads(args.json)
        df = pd.DataFrame([applicant])
        result = predict_df(df, pipeline, features)
        row = result.iloc[0]
        print(f"Decision: {row['decision']}")
        print(f"Approval probability: {row['approval_probability']:.2%}")

    else:
        print("No --json or --csv given, running a built-in demo applicant:\n")
        applicant = demo_applicant()
        print(json.dumps(applicant, indent=2))
        df = pd.DataFrame([applicant])
        result = predict_df(df, pipeline, features)
        row = result.iloc[0]
        print(f"\nDecision: {row['decision']}")
        print(f"Approval probability: {row['approval_probability']:.2%}")


if __name__ == "__main__":
    main()
