

from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Globals populated by training
df = None
scaler = None
knn = None
nb = None
dt = None
X_test = None
y_test = None
FEATURE_COLS = ["distance", "risk", "urgency", "incident_code", "area_code"]
_trained = False


def train_models(csv_path: str | Path | None = None):
    """Load data, featurize, train scaler and models and populate module globals.

    This function is idempotent and sets a module-level `_trained` flag.
    """
    global df, scaler, knn, nb, dt, X_test, y_test, _trained

    if _trained:
        return

    if csv_path is None:
        csv_path = Path(__file__).resolve().with_name("DisasterDeclarationsSummaries.csv")

    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(
            "pandas is required for training models. Install it with `pip install pandas` or add it to requirements.txt"
        ) from e

    df_local = pd.read_csv(csv_path)
    df_local = df_local[["incidentType", "declarationType", "designatedArea"]].dropna()

    risk_map = {
        "Fire": 3,
        "Hurricane": 3,
        "Flood": 3,
        "Earthquake": 3,
        "Tornado": 2,
        "Severe Storm": 2,
        "Winter Storm": 2,
        "Coastal Storm": 2,
        "Snowstorm": 2,
        "Other": 1,
    }
    df_local["risk"] = df_local["incidentType"].map(risk_map).fillna(1)
    df_local["incident_code"] = df_local["incidentType"].astype("category").cat.codes
    df_local["area_code"] = df_local["designatedArea"].astype("category").cat.codes

    np.random.seed(42)
    area_base = (df_local["area_code"] % 15) + 1
    noise = np.random.normal(0, 2.5, size=len(df_local))
    df_local["distance"] = (area_base + noise).clip(lower=0.5).round(2)
    df_local["urgency"] = (df_local["risk"] * 1.5 + df_local["incident_code"] * 0.3).round(2)

    priority_map = {"DR": 2, "EM": 1, "FM": 0}
    df_local["priority"] = df_local["declarationType"].map(priority_map)
    df_local = df_local.dropna(subset=["priority"])
    df_local["priority"] = df_local["priority"].astype(int)

    X = df_local[FEATURE_COLS].values
    y = df_local["priority"].values

    X_train, X_test_local, y_train_local, y_test_local = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler_local = StandardScaler()
    X_train = scaler_local.fit_transform(X_train)
    X_test_local = scaler_local.transform(X_test_local)

    knn_local = KNeighborsClassifier(n_neighbors=7, weights="distance")
    nb_local = GaussianNB()
    dt_local = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)

    knn_local.fit(X_train, y_train_local)
    nb_local.fit(X_train, y_train_local)
    dt_local.fit(X_train, y_train_local)

    # assign to module globals
    df = df_local
    scaler = scaler_local
    knn = knn_local
    nb = nb_local
    dt = dt_local
    X_test = X_test_local
    y_test = y_test_local
    _trained = True


def ensure_trained():
    """Ensure models are trained; no-op if already trained."""
    if not _trained:
        try:
            train_models()
        except Exception as exc:
            # If training fails (missing CSV, etc.) re-raise a clearer error
            raise RuntimeError("Model training failed — ensure dataset is available or call train_models() manually.") from exc


# 7. EVALUATION

def evaluate_models():
    ensure_trained()

    print("\n--- ML MODEL EVALUATION USING FEMA DATASET ---")
    print(f"Dataset Size : {len(df):,} records")
    print(f"Features Used: {', '.join(FEATURE_COLS)}")
    print("Labels       : 0 = LOW (FM)  |  1 = MEDIUM (EM)  |  2 = HIGH (DR)")
 
    models = {"kNN": knn, "Naive Bayes": nb, "Decision Tree": dt}
 
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n{'─'*45}")
        print(f"{name}  —  Accuracy: {accuracy_score(y_test, preds):.3f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
        print("\nClassification Report:")
        print(classification_report(y_test, preds, zero_division=0))
 
 
def get_model_metrics() -> dict:
    """Return per-model weighted metrics for dashboard display."""
    ensure_trained()

    models = {"knn": knn, "naive_bayes": nb, "decision_tree": dt}
    metrics = {}
 
    for key, model in models.items():
        preds = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        metrics[key] = {
            "accuracy":  round(float(accuracy_score(y_test, preds)), 3),
            "precision": round(float(precision), 3),
            "recall":    round(float(recall), 3),
            "f1":        round(float(f1), 3),
        }
 
    return metrics
 
 
# 8. PREDICTION FUNCTION  (used by simulation)
# Severity → incident_code mapping aligned with training encoding
_SEVERITY_CODE = {"critical": 3, "moderate": 2, "minor": 1}
 
# Incident-type urgency look-up for simulation
_SEVERITY_URGENCY = {
    "critical": round(_SEVERITY_CODE["critical"] * 1.5 * 1.3, 2),   # high-risk boost
    "moderate": round(_SEVERITY_CODE["moderate"] * 1.5 * 1.0, 2),
    "minor":    round(_SEVERITY_CODE["minor"]    * 1.5 * 0.7, 2),
}
 
def predict_priority(
    distance: float,
    risk: int,
    severity: str,
    area_code: int | None = None,
) -> tuple[int, int, int]:
    """
    Predict rescue priority for a live simulation input.
 
    Parameters
    ----------
    distance  : float  — ambulance-to-victim distance in km (0.5–30)
    risk      : int    — hazard risk score (1 = low, 2 = medium, 3 = high)
    severity  : str    — victim severity: 'critical' | 'moderate' | 'minor'
    area_code : int    — optional area encoding; defaults to dataset mode
 
    Returns
    -------
    (knn_pred, nb_pred, dt_pred)  — each is 0 (LOW), 1 (MEDIUM), or 2 (HIGH)
    """
    if severity not in _SEVERITY_CODE:
        raise ValueError(f"severity must be one of {list(_SEVERITY_CODE)}")

    ensure_trained()
 
    incident_code = _SEVERITY_CODE[severity]
    urgency       = _SEVERITY_URGENCY[severity]
    ac            = area_code if area_code is not None else int(df["area_code"].mode().iloc[0])
 
    raw_features = np.array([[distance, risk, urgency, incident_code, ac]], dtype=float)
 
    # Apply the same scaler fitted on training data
    features = scaler.transform(raw_features)
 
    knn_pred = int(knn.predict(features)[0])
    nb_pred  = int(nb.predict(features)[0])
    dt_pred  = int(dt.predict(features)[0])
 
    return knn_pred, nb_pred, dt_pred
 