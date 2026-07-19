"""
app.py
------
A minimal Flask web app providing a form-based UI to submit an applicant
and see the approval decision + probability, backed by the trained pipeline.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd

from train_model import ALL_FEATURES

app = Flask(__name__)

MODEL_PATH = "models/best_model.joblib"

EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Unemployed", "Retired"]
HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE"]
LOAN_PURPOSES = ["debt_consolidation", "home_improvement", "car", "education", "medical", "business"]


def load_bundle():
    return joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            bundle = load_bundle()
            pipeline, features = bundle["pipeline"], bundle["features"]

            applicant = {
                "age": float(request.form["age"]),
                "annual_income": float(request.form["annual_income"]),
                "employment_years": float(request.form["employment_years"]),
                "employment_type": request.form["employment_type"],
                "home_ownership": request.form["home_ownership"],
                "credit_score": float(request.form["credit_score"]),
                "existing_debt": float(request.form["existing_debt"]),
                "loan_amount": float(request.form["loan_amount"]),
                "loan_term_months": float(request.form["loan_term_months"]),
                "loan_purpose": request.form["loan_purpose"],
                "num_credit_lines": float(request.form["num_credit_lines"]),
                "num_late_payments": float(request.form["num_late_payments"]),
                "has_bankruptcy": int(request.form["has_bankruptcy"]),
            }
            applicant["debt_to_income"] = round(
                (applicant["existing_debt"] + applicant["loan_amount"]) / max(applicant["annual_income"], 1), 3
            )

            df = pd.DataFrame([applicant])[features]
            proba = pipeline.predict_proba(df)[:, 1][0]
            pred = pipeline.predict(df)[0]
            result = {
                "decision": "Approved" if pred == 1 else "Denied",
                "probability": round(proba * 100, 2),
            }
        except FileNotFoundError:
            error = "No trained model found. Run `python train_model.py` first."
        except Exception as e:  # noqa: BLE001
            error = f"Error: {e}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_values=form_values,
        employment_types=EMPLOYMENT_TYPES,
        home_ownership=HOME_OWNERSHIP,
        loan_purposes=LOAN_PURPOSES,
    )


if __name__ == "__main__":
    app.run(debug=True)
