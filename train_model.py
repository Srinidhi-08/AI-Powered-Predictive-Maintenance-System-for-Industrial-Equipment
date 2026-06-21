"""
train_model.py
---------------
End-to-end training pipeline for the AI-Powered Predictive Maintenance System.

Stages:
    1. Load + clean data
    2. Exploratory checks (missing values, outliers)
    3. Feature engineering
    4. Encoding + scaling
    5. Train/test split
    6. Train & tune multiple classifiers (Logistic Regression, Decision Tree,
       Random Forest, Gradient Boosting)
    7. Cross-validate and compare on accuracy / precision / recall / F1 / ROC-AUC
    8. Persist the best model + preprocessing artifacts with joblib

Run:
    python train_model.py
Outputs (into model/):
    failure_model.pkl       -> best classifier (failure yes/no)
    failure_type_model.pkl  -> multiclass classifier (which failure type)
    scaler.pkl               -> fitted StandardScaler
    label_encoders.pkl       -> fitted LabelEncoders for categorical columns
    feature_columns.pkl      -> ordered list of feature names the model expects
    metrics.json              -> stored evaluation metrics for the dashboard
    feature_importance.json   -> feature importance for the analytics page
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "dataset" / "predictive_maintenance.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "Machine_Type",
    "Air_Temperature",
    "Process_Temperature",
    "Rotational_Speed",
    "Torque",
    "Tool_Wear",
    "Operating_Hours",
    "Maintenance_History",
]


def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # --- Missing value handling ---
    n_missing = df.isnull().sum().sum()
    print(f"Missing values found: {n_missing}")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["Machine_Type"] = df["Machine_Type"].fillna(df["Machine_Type"].mode()[0])

    # --- Outlier handling (IQR capping on sensor columns) ---
    sensor_cols = [
        "Air_Temperature",
        "Process_Temperature",
        "Rotational_Speed",
        "Torque",
        "Tool_Wear",
    ]
    for col in sensor_cols:
        q1, q3 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q3)

    # --- Feature engineering ---
    df["Temp_Difference"] = df["Process_Temperature"] - df["Air_Temperature"]
    df["Power"] = df["Torque"] * df["Rotational_Speed"] * 2 * np.pi / 60
    df["Wear_Torque_Ratio"] = df["Tool_Wear"] * df["Torque"]
    df["Hours_Per_Maintenance"] = df["Operating_Hours"] / (
        df["Maintenance_History"] + 1
    )

    return df


def encode_and_scale(df, fit=True, encoders=None, scaler=None, feature_cols=None):
    df = df.copy()
    if encoders is None:
        encoders = {}
    if fit:
        le = LabelEncoder()
        df["Machine_Type_Enc"] = le.fit_transform(df["Machine_Type"])
        encoders["Machine_Type"] = le
    else:
        le = encoders["Machine_Type"]
        df["Machine_Type_Enc"] = le.transform(df["Machine_Type"])

    engineered = [
        "Machine_Type_Enc",
        "Air_Temperature",
        "Process_Temperature",
        "Rotational_Speed",
        "Torque",
        "Tool_Wear",
        "Operating_Hours",
        "Maintenance_History",
        "Temp_Difference",
        "Power",
        "Wear_Torque_Ratio",
        "Hours_Per_Maintenance",
    ]

    X = df[engineered]

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, encoders, scaler, engineered


def train_and_compare(X_train, X_test, y_train, y_test):
    """Train multiple classifiers, tune the top candidates, and return the best one."""
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"C": [0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"max_depth": [5, 8, 12], "min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"n_estimators": [150, 250], "max_depth": [8, 12, None]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
    }

    results = {}
    fitted_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (estimator, param_grid) in candidates.items():
        print(f"\nTraining & tuning: {name} ...")
        search = GridSearchCV(
            estimator, param_grid, cv=cv, scoring="f1", n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        fitted_models[name] = best_model

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "best_params": search.best_params_,
        }
        results[name] = metrics
        print(f"  -> {metrics}")

    # Pick best model by F1 score (good balance for imbalanced failure data)
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\nBest model selected: {best_name}")
    return best_name, fitted_models[best_name], results, fitted_models


def main():
    df = load_and_clean()

    # ---- Binary failure classifier ----
    X, encoders, scaler, engineered_cols = encode_and_scale(df, fit=True)
    y = df["Failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_name, best_model, all_results, fitted_models = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_points = list(zip(np.round(fpr, 3).tolist(), np.round(tpr, 3).tolist()))[::3]

    # Feature importance (use Random Forest / Gradient Boosting if available, else coefs)
    importance_source = fitted_models.get("Random Forest", best_model)
    if hasattr(importance_source, "feature_importances_"):
        importances = importance_source.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(engineered_cols)) / len(engineered_cols)

    feature_importance = sorted(
        zip(engineered_cols, np.round(importances / importances.sum(), 4).tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    # ---- Multiclass failure-type classifier (only trained on rows that failed) ----
    failed_df = df[df["Failure"] == 1].copy()
    Xf, _, _, _ = encode_and_scale(
        failed_df, fit=False, encoders=encoders, scaler=scaler
    )
    yf = failed_df["Failure_Type"].values
    type_encoder = LabelEncoder()
    yf_enc = type_encoder.fit_transform(yf)

    type_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    if len(set(yf_enc)) > 1:
        type_model.fit(Xf, yf_enc)
        type_train_acc = round(type_model.score(Xf, yf_enc), 4)
    else:
        type_model.fit(Xf, yf_enc)
        type_train_acc = 1.0

    # ---- Persist everything ----
    joblib.dump(best_model, MODEL_DIR / "failure_model.pkl")
    joblib.dump(type_model, MODEL_DIR / "failure_type_model.pkl")
    joblib.dump(type_encoder, MODEL_DIR / "failure_type_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "label_encoders.pkl")
    joblib.dump(engineered_cols, MODEL_DIR / "feature_columns.pkl")

    metrics_out = {
        "best_model": best_name,
        "all_models": all_results,
        "confusion_matrix": cm,
        "roc_curve": roc_points,
        "failure_type_train_accuracy": type_train_acc,
        "failure_type_classes": type_encoder.classes_.tolist(),
        "dataset_size": len(df),
        "failure_rate": round(df["Failure"].mean(), 4),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(MODEL_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    print("\nAll artifacts saved to /model")
    print(f"Best model: {best_name}  |  F1: {all_results[best_name]['f1_score']}  |  ROC-AUC: {all_results[best_name]['roc_auc']}")


if __name__ == "__main__":
    main()
