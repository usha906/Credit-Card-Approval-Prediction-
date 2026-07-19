"""
train_model.py
---------------
Loads the credit-application data, builds a preprocessing + model pipeline,
trains/compares several classifiers, evaluates them, and saves the best
pipeline (preprocessing + model bundled together) to models/best_model.joblib.

Run:
    python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_generator import generate_dataset

TARGET_COL = "approved"
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "credit_score",
    "existing_debt",
    "loan_amount",
    "loan_term_months",
    "num_credit_lines",
    "num_late_payments",
    "has_bankruptcy",
    "debt_to_income",
]
CATEGORICAL_FEATURES = ["employment_type", "home_ownership", "loan_purpose"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DATA_PATH = "data/credit_applications.csv"
MODEL_PATH = "models/best_model.joblib"
METRICS_PATH = "models/metrics.json"


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main():
    print("Loading data...")
    df = load_data()
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    candidates = get_candidate_models()

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        print(f"\nTraining: {name}")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"  accuracy={metrics['accuracy']}  f1={metrics['f1']}  roc_auc={metrics['roc_auc']}")

    # Pick best model by ROC-AUC (a good default metric for imbalanced approval decisions)
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name} (roc_auc={results[best_name]['roc_auc']})")

    print("\nClassification report for best model:")
    print(classification_report(y_test, best_pipe.predict(X_test)))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "features": ALL_FEATURES, "model_name": best_name}, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\nSaved best model -> {MODEL_PATH}")
    print(f"Saved metrics    -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
