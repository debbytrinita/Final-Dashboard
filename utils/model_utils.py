"""utils/model_utils.py — fungsi bantu inference untuk Streamlit & Dash.""" 
import json 
from pathlib import Path 
import joblib 
import pandas as pd 
  
ROOT = Path(__file__).resolve().parents[1] 
MODEL_PATH = ROOT / "models" / "churn_model.joblib" 
META_PATH = ROOT / "models" / "model_metadata.json" 
  
  
def load_metadata(path=META_PATH) -> dict: 
    """Membaca kontrak input model (fitur, kategori, threshold, risk bins).""" 
    return json.loads(Path(path).read_text(encoding="utf-8")) 
  
  
def load_model(path=MODEL_PATH): 
    """Memuat final trained model (Pipeline preprocessing + classifier). Tanpa training ulang.""" 
    return joblib.load(path) 
  
  
def validate_input(df: pd.DataFrame, meta: dict) -> list[str]: 
    """Mengembalikan daftar pesan error; list kosong berarti input valid.""" 
    errors = [] 
    missing = [c for c in meta["feature_order"] if c not in df.columns] 
    if missing: 
        errors.append(f"Kolom wajib tidak ada: {missing}") 
        return errors 
    for c in meta["categorical_features"]: 
        unknown = set(df[c].dropna().unique()) - set(meta["categories"][c]) 
        if unknown: 
            errors.append(f"Nilai tidak dikenal pada '{c}': {sorted(unknown)}") 
    for c in meta["numeric_features"]: 
        if not pd.api.types.is_numeric_dtype(df[c]): 
            errors.append(f"Kolom '{c}' harus numerik") 
    return errors 
  
  
def prepare_input(df: pd.DataFrame, meta: dict) -> pd.DataFrame: 
    """Memilih & mengurutkan kolom sesuai urutan fitur saat model difinalisasi.""" 
    X = df[meta["feature_order"]].copy() 
    for c in meta["numeric_features"]: 
        X[c] = pd.to_numeric(X[c], errors="coerce") 
    return X   
  
def risk_category(p: float, meta: dict) -> str: 
    for label, (lo, hi) in meta["risk_bins"].items(): 
        if lo <= p < hi or (hi == 1.0 and p == 1.0): 
            return label 
    return "Tidak diketahui" 
  
  
def predict(model, df: pd.DataFrame, meta: dict) -> pd.DataFrame: 
    """Menjalankan inference dan menambahkan kolom output prediksi.""" 
    X = prepare_input(df, meta) 
    proba = model.predict_proba(X)[:, 1] 
    out = df.copy() 
    out["churn_probability"] = proba.round(4) 
    out["predicted_churn"] = (proba >= meta["decision_threshold"]).astype(int) 
    out["risk_category"] = [risk_category(p, meta) for p in proba] 
    return out 