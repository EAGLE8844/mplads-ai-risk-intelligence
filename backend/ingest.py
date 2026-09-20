from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from .db import Base, engine, SessionLocal
from .models import MPAllocation, CalamityConsent

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def clean_number(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    s = s.replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan

def clean_text(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None

def parse_date(value):
    if pd.isna(value):
        return None
    dt = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(dt) else dt.date()

def import_all():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Clear previous import so the script is idempotent.
    db.query(MPAllocation).delete()
    db.query(CalamityConsent).delete()
    db.commit()

    # -------------------------
    # MP allocation dataset
    # -------------------------
    f = DATA / "allocated_limit_mps.csv"
    df = pd.read_csv(f, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    state_col = "State"
    mp_col = "Hon'ble Members of Parliaments"
    constituency_col = "Constituency"
    amount_col = "Allocated AMOUNT ( ₹ )"
    sr_col = "Sr. No."

    df["allocated_amount_clean"] = df[amount_col].apply(clean_number)
    df[mp_col] = df[mp_col].apply(clean_text)
    df[state_col] = df[state_col].apply(clean_text)
    df[constituency_col] = df[constituency_col].apply(clean_text)

    # Remove totals/subtotals and rows without a real MP.
    text_blob = (
        df[[state_col, mp_col, constituency_col]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    total_mask = text_blob.str.contains(r"\bgrand\s*total\b|\btotal\b", regex=True)
    valid = (
        ~total_mask
        & df[mp_col].notna()
        & df[constituency_col].notna()
        & df["allocated_amount_clean"].notna()
    )
    df = df[valid].copy()

    # Basic data quality score.
    def quality(row):
        score = 100
        if not row[state_col]: score -= 20
        if not row[mp_col]: score -= 20
        if not row[constituency_col]: score -= 20
        if pd.isna(row["allocated_amount_clean"]): score -= 40
        elif row["allocated_amount_clean"] <= 0: score -= 30
        return max(score, 0)

    df["data_quality_score"] = df.apply(quality, axis=1)

    # Basic anomaly screening on allocation amount.
    amounts = df["allocated_amount_clean"].astype(float)
    log_amounts = np.log1p(amounts).to_numpy().reshape(-1, 1)

    if len(df) >= 20:
        model = IsolationForest(
            n_estimators=300,
            contamination=0.05,
            random_state=42
        )
        pred = model.fit_predict(log_amounts)
        decision = model.decision_function(log_amounts)

        df["is_anomaly"] = (pred == -1).astype(int)
        # Convert model decision into a simple 0-100 anomaly score.
        lo, hi = float(decision.min()), float(decision.max())
        if hi > lo:
            df["anomaly_score"] = 100 * (hi - decision) / (hi - lo)
        else:
            df["anomaly_score"] = 0.0
    else:
        df["is_anomaly"] = 0
        df["anomaly_score"] = 0.0

    for _, r in df.iterrows():
        sr = clean_number(r[sr_col])
        db.add(MPAllocation(
            sr_no=None if pd.isna(sr) else int(sr),
            state=r[state_col],
            mp_name=r[mp_col],
            constituency=r[constituency_col],
            allocated_amount=float(r["allocated_amount_clean"]),
            is_anomaly=int(r["is_anomaly"]),
            anomaly_score=float(r["anomaly_score"]),
            data_quality_score=float(r["data_quality_score"])
        ))

    # -------------------------
    # Calamity consent dataset
    # -------------------------
    f2 = DATA / "amount_consented_calamity.csv"
    cdf = pd.read_csv(f2, dtype=str)
    cdf.columns = [c.strip() for c in cdf.columns]

    c_type = "Calamity Type"
    c_name = "Calamity Name"
    c_mp = "Hon'ble Members of Parliament"
    c_date = "Date of Consent"
    c_amount = "Consent Amount ( ₹ )"
    c_sr = "Sr. No."

    cdf[c_mp] = cdf[c_mp].apply(clean_text)
    cdf[c_type] = cdf[c_type].apply(clean_text)
    cdf[c_name] = cdf[c_name].apply(clean_text)
    cdf["amount_clean"] = cdf[c_amount].apply(clean_number)

    for _, r in cdf.iterrows():
        sr = clean_number(r[c_sr])
        db.add(CalamityConsent(
            sr_no=None if pd.isna(sr) else int(sr),
            calamity_type=r[c_type],
            calamity_name=r[c_name],
            mp_name=r[c_mp],
            consent_date=parse_date(r[c_date]),
            consent_amount=None if pd.isna(r["amount_clean"]) else float(r["amount_clean"]),
            data_quality_score=100.0 if (
                r[c_mp] and r[c_name] and not pd.isna(r["amount_clean"])
            ) else 70.0
        ))

    db.commit()
    db.close()

    print(f"Imported {len(df)} MP allocation records.")
    print(f"Imported {len(cdf)} calamity consent records.")
    print("Database:", ROOT / "mplads.db")

if __name__ == "__main__":
    import_all()
