"""
data_generator.py
------------------
Generates a realistic SYNTHETIC credit-approval dataset.

Why synthetic? You didn't attach a real dataset, so this creates one with
realistic features and a plausible (not random) approval rule, so the rest
of the pipeline (training/eval/prediction) is fully runnable end-to-end.

Swap this out at any time for a real dataset (e.g. Kaggle's "Credit Approval"
or "Give Me Some Credit") as long as you keep a similar column layout, or
adjust TARGET_COL / FEATURE lists in train_model.py to match.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_dataset(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, n_samples)
    annual_income = rng.lognormal(mean=10.8, sigma=0.5, size=n_samples).round(2)  # ~ $30k-$150k skewed
    employment_years = np.clip(rng.normal(loc=age - 22, scale=4, size=n_samples), 0, None).round(1)
    credit_score = np.clip(rng.normal(loc=650, scale=90, size=n_samples), 300, 850).round(0)
    existing_debt = np.clip(rng.normal(loc=annual_income * 0.25, scale=annual_income * 0.15), 0, None).round(2)
    loan_amount = np.clip(rng.normal(loc=annual_income * 0.4, scale=annual_income * 0.2), 500, None).round(2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    num_credit_lines = rng.integers(0, 15, n_samples)
    num_late_payments = rng.poisson(lam=1.2, size=n_samples)
    has_bankruptcy = rng.choice([0, 1], size=n_samples, p=[0.92, 0.08])
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n_samples, p=[0.4, 0.25, 0.35])
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Unemployed", "Retired"], size=n_samples, p=[0.55, 0.25, 0.1, 0.1]
    )
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "car", "education", "medical", "business"],
        size=n_samples,
    )

    debt_to_income = np.divide(existing_debt + loan_amount, annual_income, out=np.zeros_like(annual_income), where=annual_income != 0)

    # ---- Latent "approval score" -> plausible ground-truth rule (not random labels) ----
    score = (
        0.010 * (credit_score - 600)
        - 3.0 * debt_to_income
        + 0.05 * employment_years
        - 0.6 * num_late_payments
        - 1.8 * has_bankruptcy
        + 0.00002 * annual_income
        - 0.35 * (employment_type == "Unemployed")
        + 0.15 * (employment_type == "Salaried")
        - 0.02 * (num_credit_lines > 10)
    )
    score += rng.normal(0, 0.6, n_samples)  # noise so it's not a trivial linear separator
    approval_prob = 1 / (1 + np.exp(-score))
    approved = (rng.random(n_samples) < approval_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "employment_type": employment_type,
            "home_ownership": home_ownership,
            "credit_score": credit_score,
            "existing_debt": existing_debt,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "loan_purpose": purpose,
            "num_credit_lines": num_credit_lines,
            "num_late_payments": num_late_payments,
            "has_bankruptcy": has_bankruptcy,
            "debt_to_income": debt_to_income.round(3),
            "approved": approved,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/credit_applications.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["approved"].value_counts(normalize=True).rename("class balance"))
    print(df.head())
