"""
app.py
------
Flask application for the AI-Powered Predictive Maintenance System.

Routes:
    /login            GET/POST  - simple login gate
    /logout            GET       - clears session
    /                  GET       - redirects to dashboard or login
    /dashboard         GET       - Industry 4.0 KPI dashboard
    /predict            GET/POST  - single-machine prediction form + result
    /analytics          GET       - charts (EDA + model evaluation)
    /history            GET       - searchable/sortable prediction history
    /upload             GET/POST  - batch CSV prediction
    /download/<fname>  GET       - download a generated CSV report
    /api/predict        POST      - JSON prediction API (used by predict.html via fetch)
"""

import io
import json
import os
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from utils.maintenance_engine import build_prediction_response

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"
EXPORT_DIR = BASE_DIR / "static" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "predictive-maintenance-demo-secret-key")

# Demo credentials (this is a portfolio project, not production auth)
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"

# ----------------------------------------------------------------------------
# Load trained ML artifacts once at startup
# ----------------------------------------------------------------------------
failure_model = joblib.load(MODEL_DIR / "failure_model.pkl")
type_model = joblib.load(MODEL_DIR / "failure_type_model.pkl")
type_encoder = joblib.load(MODEL_DIR / "failure_type_encoder.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

with open(MODEL_DIR / "metrics.json") as f:
    MODEL_METRICS = json.load(f)
with open(MODEL_DIR / "feature_importance.json") as f:
    FEATURE_IMPORTANCE = json.load(f)

DATASET = pd.read_csv(DATASET_DIR / "predictive_maintenance.csv")

# In-memory prediction history (resets on server restart -- fine for a demo app)
PREDICTION_HISTORY = []


# ----------------------------------------------------------------------------
# Auth helper
# ----------------------------------------------------------------------------
def login_required(view_fn):
    @wraps(view_fn)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view_fn(*args, **kwargs)

    return wrapped


# ----------------------------------------------------------------------------
# Core prediction logic (shared by single-form, API, and CSV-batch routes)
# ----------------------------------------------------------------------------
def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["Temp_Difference"] = df["Process_Temperature"] - df["Air_Temperature"]
    df["Power"] = df["Torque"] * df["Rotational_Speed"] * 2 * np.pi / 60
    df["Wear_Torque_Ratio"] = df["Tool_Wear"] * df["Torque"]
    df["Hours_Per_Maintenance"] = df["Operating_Hours"] / (df["Maintenance_History"] + 1)
    le = label_encoders["Machine_Type"]
    df["Machine_Type_Enc"] = le.transform(df["Machine_Type"])
    return df[feature_columns]


def run_prediction(payload: dict) -> dict:
    X = engineer_features(payload)
    X_scaled = scaler.transform(X)

    failure_proba = failure_model.predict_proba(X_scaled)[0][1]

    type_pred_idx = type_model.predict(X_scaled)[0]
    type_proba_all = type_model.predict_proba(X_scaled)[0]
    predicted_type = type_encoder.inverse_transform([type_pred_idx])[0]
    confidence = max(failure_model.predict_proba(X_scaled)[0])

    result = build_prediction_response(
        failure_probability=float(failure_proba),
        predicted_type=predicted_type,
        confidence=float(confidence),
        air_temp=payload["Air_Temperature"],
        process_temp=payload["Process_Temperature"],
        rpm=payload["Rotational_Speed"],
        torque=payload["Torque"],
        tool_wear=payload["Tool_Wear"],
        operating_hours=payload["Operating_Hours"],
    )
    result["input"] = payload
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["machine_id"] = payload.get("Machine_ID", f"M-{uuid.uuid4().hex[:6].upper()}")
    return result


# ----------------------------------------------------------------------------
# Routes -- Auth
# ----------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        flash("Invalid username or password. Try admin / admin123.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    return redirect(url_for("dashboard") if session.get("logged_in") else url_for("login"))


# ----------------------------------------------------------------------------
# Routes -- Dashboard
# ----------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    total_machines = len(DATASET)
    healthy = int((DATASET["Failure"] == 0).sum())
    failing = int((DATASET["Failure"] == 1).sum())

    # Simulate health bands from the dataset's own failure probability proxy
    sample = DATASET.sample(min(500, len(DATASET)), random_state=1).copy()
    sample["TempDiff"] = sample["Process_Temperature"] - sample["Air_Temperature"]
    risk_proxy = (
        (sample["Tool_Wear"] / 260) * 0.5
        + (1 - (sample["TempDiff"] / sample["TempDiff"].max())) * 0.3
        + sample["Failure"] * 0.2
    )
    warning = int(((risk_proxy > 0.35) & (risk_proxy <= 0.6)).sum())
    critical = int((risk_proxy > 0.6).sum())
    healthy_band = len(sample) - warning - critical

    avg_health = round(100 - risk_proxy.mean() * 100, 1)
    avg_rul = int(((260 - DATASET["Tool_Wear"]) * 1.8).clip(lower=1).mean())

    kpis = {
        "total_machines": total_machines,
        "healthy_machines": healthy_band,
        "warning_machines": warning,
        "critical_machines": critical,
        "average_health": avg_health,
        "todays_predictions": len(PREDICTION_HISTORY),
        "average_remaining_life": avg_rul,
        "model_accuracy": round(
            MODEL_METRICS["all_models"][MODEL_METRICS["best_model"]]["accuracy"] * 100, 1
        ),
    }

    recent = PREDICTION_HISTORY[-5:][::-1]
    return render_template("dashboard.html", kpis=kpis, recent=recent, best_model=MODEL_METRICS["best_model"], active="dashboard")


# ----------------------------------------------------------------------------
# Routes -- Predict (single machine)
# ----------------------------------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    if request.method == "POST":
        try:
            payload = {
                "Machine_ID": request.form.get("machine_id") or f"M-{uuid.uuid4().hex[:6].upper()}",
                "Machine_Type": request.form["machine_type"],
                "Air_Temperature": float(request.form["air_temperature"]),
                "Process_Temperature": float(request.form["process_temperature"]),
                "Rotational_Speed": float(request.form["rpm"]),
                "Torque": float(request.form["torque"]),
                "Tool_Wear": float(request.form["tool_wear"]),
                "Operating_Hours": float(request.form["operating_hours"]),
                "Maintenance_History": int(request.form["maintenance_count"]),
            }
            result = run_prediction(payload)
            PREDICTION_HISTORY.append(result)
        except (KeyError, ValueError) as e:
            flash(f"Invalid input: {e}", "error")

    return render_template("predict.html", result=result, active="predict")


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    try:
        data = request.get_json()
        payload = {
            "Machine_ID": data.get("machine_id") or f"M-{uuid.uuid4().hex[:6].upper()}",
            "Machine_Type": data["machine_type"],
            "Air_Temperature": float(data["air_temperature"]),
            "Process_Temperature": float(data["process_temperature"]),
            "Rotational_Speed": float(data["rpm"]),
            "Torque": float(data["torque"]),
            "Tool_Wear": float(data["tool_wear"]),
            "Operating_Hours": float(data["operating_hours"]),
            "Maintenance_History": int(data["maintenance_count"]),
        }
        result = run_prediction(payload)
        PREDICTION_HISTORY.append(result)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ----------------------------------------------------------------------------
# Routes -- Analytics
# ----------------------------------------------------------------------------
@app.route("/analytics")
@login_required
def analytics():
    temp_trend = (
        DATASET.groupby(pd.cut(DATASET["Operating_Hours"], bins=10))["Process_Temperature"]
        .mean()
        .round(1)
        .tolist()
    )
    failure_dist = DATASET["Failure_Type"].value_counts().to_dict()
    machine_type_dist = DATASET["Machine_Type"].value_counts().to_dict()

    scatter_sample = DATASET.sample(min(300, len(DATASET)), random_state=2)
    scatter_data = [
        {"x": float(r.Torque), "y": float(r.Rotational_Speed), "failed": int(r.Failure)}
        for r in scatter_sample.itertuples()
    ]

    monthly_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    rng = np.random.default_rng(7)
    monthly_failures = rng.integers(20, 90, size=12).tolist()

    return render_template(
        "analytics.html",
        temp_trend=temp_trend,
        failure_dist=failure_dist,
        machine_type_dist=machine_type_dist,
        scatter_data=scatter_data,
        feature_importance=FEATURE_IMPORTANCE,
        confusion_matrix=MODEL_METRICS["confusion_matrix"],
        roc_curve=MODEL_METRICS["roc_curve"],
        monthly_labels=monthly_labels,
        monthly_failures=monthly_failures,
        all_models=MODEL_METRICS["all_models"],
        best_model=MODEL_METRICS["best_model"],
        active="analytics",
    )


# ----------------------------------------------------------------------------
# Routes -- History
# ----------------------------------------------------------------------------
@app.route("/history")
@login_required
def history():
    return render_template("history.html", predictions=PREDICTION_HISTORY[::-1], active="history")


# ----------------------------------------------------------------------------
# Routes -- Batch CSV upload
# ----------------------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    batch_results = None
    download_name = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("Please choose a CSV file to upload.", "error")
            return render_template("upload.html", batch_results=None, active="upload")

        try:
            df = pd.read_csv(file)
            required_cols = [
                "Machine_Type", "Air_Temperature", "Process_Temperature",
                "Rotational_Speed", "Torque", "Tool_Wear",
                "Operating_Hours", "Maintenance_History",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                flash(f"CSV is missing required columns: {', '.join(missing)}", "error")
                return render_template("upload.html", batch_results=None, active="upload")

            rows_out = []
            for _, row in df.iterrows():
                payload = {
                    "Machine_ID": row.get("Machine_ID", f"M-{uuid.uuid4().hex[:6].upper()}"),
                    "Machine_Type": row["Machine_Type"],
                    "Air_Temperature": float(row["Air_Temperature"]),
                    "Process_Temperature": float(row["Process_Temperature"]),
                    "Rotational_Speed": float(row["Rotational_Speed"]),
                    "Torque": float(row["Torque"]),
                    "Tool_Wear": float(row["Tool_Wear"]),
                    "Operating_Hours": float(row["Operating_Hours"]),
                    "Maintenance_History": int(row["Maintenance_History"]),
                }
                result = run_prediction(payload)
                PREDICTION_HISTORY.append(result)
                rows_out.append(
                    {
                        "Machine_ID": result["machine_id"],
                        "Machine_Health": result["machine_health"],
                        "Failure_Probability_%": result["failure_probability"],
                        "Risk_Level": result["risk_level"],
                        "Predicted_Failure_Type": result["predicted_failure_type"],
                        "Remaining_Useful_Life_Hours": result["remaining_useful_life"],
                        "Recommendation": result["recommendation"],
                        "Estimated_Repair_Cost": result["estimated_repair_cost"],
                    }
                )

            result_df = pd.DataFrame(rows_out)
            download_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            result_df.to_csv(EXPORT_DIR / download_name, index=False)
            batch_results = result_df.to_dict(orient="records")

        except Exception as e:
            flash(f"Error processing CSV: {e}", "error")

    return render_template("upload.html", batch_results=batch_results, download_name=download_name, active="upload")


@app.route("/download/<path:fname>")
@login_required
def download(fname):
    path = EXPORT_DIR / fname
    if not path.exists():
        flash("File not found.", "error")
        return redirect(url_for("upload"))
    return send_file(path, as_attachment=True, download_name=fname)


@app.route("/download_history")
@login_required
def download_history():
    if not PREDICTION_HISTORY:
        flash("No prediction history to export yet.", "error")
        return redirect(url_for("history"))

    rows = [
        {
            "Timestamp": r["timestamp"],
            "Machine_ID": r["machine_id"],
            "Machine_Health": r["machine_health"],
            "Failure_Probability_%": r["failure_probability"],
            "Risk_Level": r["risk_level"],
            "Predicted_Failure_Type": r["predicted_failure_type"],
            "Remaining_Useful_Life_Hours": r["remaining_useful_life"],
        }
        for r in PREDICTION_HISTORY
    ]
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
