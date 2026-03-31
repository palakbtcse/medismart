"""
MediSmart - Smart Medicine Recommendation System
Flask Backend with Decision Tree & Random Forest ML Models

Project Structure:
  medismart/
  ├── app.py               ← This file (Flask backend)
  ├── data/
  │   ├── symtoms_df.csv
  │   ├── medications.csv
  │   ├── description.csv
  │   ├── precautions_df.csv
  │   └── Symptom-severity.csv
  ├── templates/
  │   └── index.html       ← Frontend HTML
  ├── static/
  │   ├── style.css
  │   └── script.js
  └── requirements.txt

Run:
  pip install -r requirements.txt
  python app.py
"""

import os
import ast
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ─── APP INIT ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config["JSON_SORT_KEYS"] = False

# ─── PATHS ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

DATA_FILES = {
    "symptoms":   os.path.join(DATA_DIR, "symtoms_df.csv"),
    "severity":   os.path.join(DATA_DIR, "Symptom-severity.csv"),
    "medications":os.path.join(DATA_DIR, "medications.csv"),
    "description":os.path.join(DATA_DIR, "description.csv"),
    "precautions":os.path.join(DATA_DIR, "precautions_df.csv"),
}

# ─── GLOBAL OBJECTS (loaded once at startup) ────────────────────────────────
ALL_SYMPTOMS   = []   # sorted list of all unique symptom strings
SEVERITY_MAP   = {}   # symptom → weight (int)
DISEASE_INFO   = {}   # disease → {medications, description, precautions}
DT_MODEL       = None # trained Decision Tree
RF_MODEL       = None # trained Random Forest
MODEL_METRICS  = {}   # accuracy scores
FEATURE_COLS   = []   # ordered list of symptom columns used in model

# ─── DATA LOADING ───────────────────────────────────────────────────────────

def load_datasets():
    """Load and clean all CSV files into memory."""
    sym_df  = pd.read_csv(DATA_FILES["symptoms"])
    sev_df  = pd.read_csv(DATA_FILES["severity"])
    med_df  = pd.read_csv(DATA_FILES["medications"])
    desc_df = pd.read_csv(DATA_FILES["description"])
    prec_df = pd.read_csv(DATA_FILES["precautions"])
    return sym_df, sev_df, med_df, desc_df, prec_df


def build_severity_map(sev_df):
    """Build symptom → severity weight dictionary."""
    return {
        str(row["Symptom"]).strip(): int(row["weight"])
        for _, row in sev_df.iterrows()
    }


def build_disease_info(med_df, desc_df, prec_df):
    """Merge medications, descriptions, precautions per disease."""
    info = {}

    # Medications
    for _, row in med_df.iterrows():
        disease = str(row["Disease"]).strip()
        try:
            meds = ast.literal_eval(row["Medication"])
        except Exception:
            meds = [str(row["Medication"])]
        info.setdefault(disease, {})["medications"] = meds

    # Descriptions
    for _, row in desc_df.iterrows():
        disease = str(row["Disease"]).strip()
        info.setdefault(disease, {})["description"] = str(row["Description"])

    # Precautions
    prec_cols = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    for _, row in prec_df.iterrows():
        disease = str(row["Disease"]).strip()
        precs = [
            str(row[c]).strip()
            for c in prec_cols
            if pd.notna(row[c]) and str(row[c]).strip()
        ]
        info.setdefault(disease, {})["precautions"] = precs

    return info


def build_feature_matrix(sym_df):
    """
    Build binary feature matrix from symtoms_df.csv.
    Each row = one training example; columns = unique symptoms (0 or 1).
    Returns (X DataFrame, y Series, all_symptoms list).
    """
    sym_cols = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]

    # Clean
    for col in sym_cols:
        sym_df[col] = sym_df[col].astype(str).str.strip().replace("nan", "")

    # Collect all unique symptoms (filter out empty / NaN)
    all_syms = sorted({
        val
        for col in sym_cols
        for val in sym_df[col].unique()
        if isinstance(val, str) and val and val != "nan"
    })

    # Build binary feature matrix
    X = pd.DataFrame(0, index=sym_df.index, columns=all_syms)
    for col in sym_cols:
        for i, val in sym_df[col].items():
            if val and val in all_syms:
                X.at[i, val] = 1

    y = sym_df["Disease"].str.strip()
    return X, y, all_syms


# ─── MODEL TRAINING ─────────────────────────────────────────────────────────

def train_models(X, y):
    """Train Decision Tree and Random Forest; return models + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision Tree
    dt = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=20,
        random_state=42,
    )
    dt.fit(X_train, y_train)
    dt_pred   = dt.predict(X_test)
    dt_acc    = accuracy_score(y_test, dt_pred)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        criterion="entropy",
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred   = rf.predict(X_test)
    rf_acc    = accuracy_score(y_test, rf_pred)

    metrics = {
        "decision_tree": {
            "accuracy": round(dt_acc * 100, 2),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
        },
        "random_forest": {
            "accuracy": round(rf_acc * 100, 2),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
        },
    }
    return dt, rf, metrics


# ─── STARTUP: LOAD + TRAIN ───────────────────────────────────────────────────

def initialise():
    global ALL_SYMPTOMS, SEVERITY_MAP, DISEASE_INFO
    global DT_MODEL, RF_MODEL, MODEL_METRICS, FEATURE_COLS

    print("[MediSmart] Loading datasets…")
    sym_df, sev_df, med_df, desc_df, prec_df = load_datasets()

    print("[MediSmart] Building lookup tables…")
    SEVERITY_MAP  = build_severity_map(sev_df)
    DISEASE_INFO  = build_disease_info(med_df, desc_df, prec_df)

    print("[MediSmart] Building feature matrix…")
    X, y, ALL_SYMPTOMS = build_feature_matrix(sym_df)
    FEATURE_COLS = list(X.columns)

    print("[MediSmart] Training ML models…")
    DT_MODEL, RF_MODEL, MODEL_METRICS = train_models(X, y)

    print(f"[MediSmart] Decision Tree  accuracy: {MODEL_METRICS['decision_tree']['accuracy']}%")
    print(f"[MediSmart] Random Forest  accuracy: {MODEL_METRICS['random_forest']['accuracy']}%")
    print(f"[MediSmart] Unique symptoms : {len(ALL_SYMPTOMS)}")
    print(f"[MediSmart] Unique diseases : {len(DISEASE_INFO)}")
    print("[MediSmart] ✓ Ready — http://127.0.0.1:5000")


# ─── HELPER ─────────────────────────────────────────────────────────────────

def symptoms_to_vector(selected_symptoms):
    """Convert a list of symptom strings to an ordered binary feature DataFrame."""
    vec = {col: [0] for col in FEATURE_COLS}
    for sym in selected_symptoms:
        if sym in vec:
            vec[sym] = [1]
    return pd.DataFrame(vec, columns=FEATURE_COLS)


def build_disease_response(disease_name, selected_symptoms):
    """Build a rich response dict for a predicted disease."""
    info = DISEASE_INFO.get(disease_name, {})

    # Matched symptom analysis
    # We can't know disease symptoms from model alone —
    # look them up from the training data context
    matched = [s for s in selected_symptoms if SEVERITY_MAP.get(s, 0) > 0]
    severity_score = sum(SEVERITY_MAP.get(s, 3) for s in selected_symptoms)

    if severity_score >= 30:
        sev_label, sev_cls = "High", "sev-high"
    elif severity_score >= 15:
        sev_label, sev_cls = "Moderate", "sev-mod"
    else:
        sev_label, sev_cls = "Low", "sev-low"

    return {
        "disease":        disease_name,
        "description":    info.get("description", ""),
        "medications":    info.get("medications", []),
        "precautions":    info.get("precautions", []),
        "severity_score": severity_score,
        "severity_label": sev_label,
        "severity_cls":   sev_cls,
        "matched_symptoms": selected_symptoms,
    }


# ─── ROUTES ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    """Return all available symptoms."""
    return jsonify({
        "success":  True,
        "symptoms": ALL_SYMPTOMS,
        "count":    len(ALL_SYMPTOMS),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Request JSON:
    {
        "symptoms": ["fever", "headache", ...],
        "age":      25,
        "gender":   "male",
        "model":    "random_forest"   ← optional, default random_forest
    }

    Response JSON:
    {
        "success":     true,
        "predictions": [
            {
                "rank":           1,
                "model":          "random_forest",
                "disease":        "Malaria",
                "confidence":     0.87,
                "description":    "...",
                "medications":    [...],
                "precautions":    [...],
                "severity_label": "High",
                "severity_score": 42,
                ...
            }
        ],
        "meta": {
            "symptoms_received": 4,
            "age":    25,
            "gender": "male",
            "model_used": "random_forest"
        }
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    # Validate symptoms
    raw_symptoms = data.get("symptoms", [])
    if not isinstance(raw_symptoms, list) or len(raw_symptoms) == 0:
        return jsonify({"success": False, "error": "Please provide at least one symptom."}), 422

    # Clean + filter to known symptoms
    selected = [s.strip() for s in raw_symptoms if s.strip() in FEATURE_COLS]
    if not selected:
        return jsonify({
            "success": False,
            "error": "None of the provided symptoms are recognised. Please use /api/symptoms to get valid options."
        }), 422

    # Age & gender (informational — used for display; could extend for age-safety rules)
    age    = data.get("age", None)
    gender = data.get("gender", "")
    model_choice = data.get("model", "random_forest").lower()

    # Feature vector
    X_input = symptoms_to_vector(selected)

    # Predict using both models, return probabilities
    predictions = []

    for model_name, model_obj in [("decision_tree", DT_MODEL), ("random_forest", RF_MODEL)]:
        proba      = model_obj.predict_proba(X_input)[0]
        classes    = model_obj.classes_
        top_idx    = np.argsort(proba)[::-1][:3]   # top 3

        for rank, idx in enumerate(top_idx, 1):
            disease    = classes[idx]
            confidence = round(float(proba[idx]), 4)
            if confidence < 0.01:
                continue
            result = build_disease_response(disease, selected)
            result["rank"]       = rank
            result["model"]      = model_name
            result["confidence"] = confidence
            result["confidence_pct"] = round(confidence * 100, 1)
            predictions.append(result)

    # Deduplicate by disease name (keep highest confidence)
    seen     = {}
    combined = []
    for p in predictions:
        key = p["disease"]
        if key not in seen or p["confidence"] > seen[key]["confidence"]:
            seen[key] = p

    # Return top 3 by confidence from preferred model
    preferred_model = "random_forest" if model_choice == "random_forest" else "decision_tree"
    preferred = [p for p in seen.values() if p["model"] == preferred_model]
    preferred.sort(key=lambda x: x["confidence"], reverse=True)
    top3 = preferred[:3]

    # Re-rank
    for i, p in enumerate(top3):
        p["rank"] = i + 1

    return jsonify({
        "success":     True,
        "predictions": top3,
        "meta": {
            "symptoms_received": len(selected),
            "symptoms_used":     selected,
            "age":               age,
            "gender":            gender,
            "model_used":        preferred_model,
        },
    })


@app.route("/api/disease/<disease_name>", methods=["GET"])
def get_disease(disease_name):
    """Return full info for a specific disease."""
    info = DISEASE_INFO.get(disease_name)
    if not info:
        # Try case-insensitive search
        match = next(
            (k for k in DISEASE_INFO if k.lower() == disease_name.lower()), None
        )
        if not match:
            return jsonify({"success": False, "error": f"Disease '{disease_name}' not found."}), 404
        info = DISEASE_INFO[match]
        disease_name = match

    return jsonify({
        "success":    True,
        "disease":    disease_name,
        "info":       info,
    })


@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    """Return list of all diseases with basic info."""
    diseases = []
    for name, info in DISEASE_INFO.items():
        diseases.append({
            "disease":     name,
            "description": info.get("description", ""),
            "medications": info.get("medications", []),
            "precautions": info.get("precautions", []),
        })
    return jsonify({
        "success":  True,
        "diseases": diseases,
        "count":    len(diseases),
    })


@app.route("/api/models/stats", methods=["GET"])
def model_stats():
    """Return ML model accuracy and metadata."""
    return jsonify({
        "success": True,
        "models":  MODEL_METRICS,
        "dataset": {
            "total_symptoms": len(ALL_SYMPTOMS),
            "total_diseases": len(DISEASE_INFO),
        },
    })


@app.route("/api/severity", methods=["GET"])
def get_severity():
    """Return severity weights for all symptoms."""
    return jsonify({
        "success":  True,
        "severity": SEVERITY_MAP,
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status":  "ok",
        "service": "MediSmart API",
        "models":  {
            "decision_tree":  "loaded",
            "random_forest":  "loaded",
        },
    })


# ─── ERROR HANDLERS ──────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initialise()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,   # prevent double initialise() call
    )
