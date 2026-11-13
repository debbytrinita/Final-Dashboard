"""
SCRIPT FASILITATOR — dijalankan SEBELUM pelatihan, bukan bagian dari sesi.
Menghasilkan final trained / selected best-performing model beserta metadata.
Peserta hanya MEMUAT file churn_model.joblib dan model_metadata.json.
"""
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")

NUM = ["tenure_months", "monthly_charges", "total_charges", "num_complaints", "support_calls"]
CAT = ["region", "segment", "contract_type", "payment_method", "internet_service"]
X, y = df[NUM + CAT], df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
model = Pipeline([("preprocess", pre),
                  ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                     learning_rate=0.05, random_state=42))])
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
metrics = {"roc_auc": round(roc_auc_score(y_te, proba), 4),
           "accuracy": round(accuracy_score(y_te, proba >= 0.5), 4),
           "f1": round(f1_score(y_te, proba >= 0.5), 4)}

joblib.dump(model, ROOT / "models" / "churn_model.joblib")
meta = {
    "model_name": "Customer Churn Classifier",
    "model_version": "1.0.0",
    "algorithm": "GradientBoostingClassifier (scikit-learn Pipeline)",
    "sklearn_version": sklearn.__version__,
    "target": "churn (1 = pelanggan berhenti berlangganan)",
    "numeric_features": NUM,
    "categorical_features": CAT,
    "feature_order": NUM + CAT,
    "categories": {c: sorted(df[c].unique().tolist()) for c in CAT},
    "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUM},
    "decision_threshold": 0.5,
    "risk_bins": {"Rendah": [0.0, 0.3], "Sedang": [0.3, 0.6], "Tinggi": [0.6, 1.0]},
    "holdout_metrics": metrics,
}
(ROOT / "models" / "model_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print("Model final tersimpan. Holdout metrics:", metrics)
