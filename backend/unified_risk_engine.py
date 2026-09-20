import os
import numpy as np
import pandas as pd


# ============================================================
# MPLADS AI — PHASE 1.8.1
# UNIFIED RISK INTELLIGENCE ENGINE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ANOMALY_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "allocation_anomaly_results.csv"
)

STATISTICAL_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "statistical_anomaly_results.csv"
)

PEER_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "peer_intelligence_results.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "unified_risk_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)


def normalize_score(series):
    return (
        safe_numeric(series)
        .clip(0, 100)
    )


def make_join_key(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    return numeric.apply(
        lambda x: str(int(x))
        if pd.notna(x)
        else str(x).strip()
    )


def find_column(df, candidates):

    normalized = {
        str(col).strip().lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace(".", ""): col
        for col in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
            .replace(".", "")
        )

        if key in normalized:
            return normalized[key]

    return None


def text_value(row, column, default="Unknown"):

    if column is None:
        return default

    value = row.get(
        column,
        default
    )

    if pd.isna(value):
        return default

    return str(value)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("MPLADS AI UNIFIED RISK INTELLIGENCE ENGINE")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    ANOMALY_FILE,
    STATISTICAL_FILE,
    PEER_FILE
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing required file:\n{path}"
        )


# ============================================================
# LOAD
# ============================================================

print("\nLoading intelligence layers...")

anomaly_df = pd.read_csv(
    ANOMALY_FILE
)

stat_df = pd.read_csv(
    STATISTICAL_FILE
)

peer_df = pd.read_csv(
    PEER_FILE
)

print(
    f"Allocation / ML records : {len(anomaly_df)}"
)

print(
    f"Statistical records     : {len(stat_df)}"
)

print(
    f"Peer records            : {len(peer_df)}"
)


# ============================================================
# SERIAL NUMBER
# ============================================================

sr_candidates = [
    "Sr. No.",
    "Sr No",
    "Serial No",
    "Serial Number",
    "sr_no",
    "srno"
]

anomaly_sr = find_column(
    anomaly_df,
    sr_candidates
)

stat_sr = find_column(
    stat_df,
    sr_candidates
)

peer_sr = find_column(
    peer_df,
    sr_candidates
)

if anomaly_sr is None:
    raise ValueError(
        "Serial number missing in anomaly file."
    )

if stat_sr is None:
    raise ValueError(
        "Serial number missing in statistical file."
    )

if peer_sr is None:
    raise ValueError(
        "Serial number missing in peer file."
    )


# ============================================================
# JOIN KEYS
# ============================================================

anomaly_df["_join_key"] = make_join_key(
    anomaly_df[anomaly_sr]
)

stat_df["_join_key"] = make_join_key(
    stat_df[stat_sr]
)

peer_df["_join_key"] = make_join_key(
    peer_df[peer_sr]
)


print("\n" + "=" * 70)
print("JOIN KEY DIAGNOSTICS")
print("=" * 70)

anomaly_keys = set(
    anomaly_df["_join_key"]
)

stat_keys = set(
    stat_df["_join_key"]
)

peer_keys = set(
    peer_df["_join_key"]
)

print(
    f"\nUnique anomaly keys     : {len(anomaly_keys)}"
)

print(
    f"Unique statistical keys : {len(stat_keys)}"
)

print(
    f"Unique peer keys        : {len(peer_keys)}"
)

print(
    f"\nStatistical matches     : "
    f"{len(anomaly_keys & stat_keys)}"
)

print(
    f"Peer matches            : "
    f"{len(anomaly_keys & peer_keys)}"
)


# ============================================================
# FIND REAL SOURCE COLUMNS
# ============================================================

# -------------------------
# Statistical
# -------------------------

stat_score_col = find_column(
    stat_df,
    [
        "statistical_score",
        "Statistical Score"
    ]
)

stat_alert_col = find_column(
    stat_df,
    [
        "statistical_alert_level",
        "statistical_alert",
        "Statistical Alert Level"
    ]
)

stat_z_col = find_column(
    stat_df,
    [
        "robust_z_score",
        "Robust Z-Score"
    ]
)

stat_percentile_col = find_column(
    stat_df,
    [
        "global_percentile",
        "Global Percentile"
    ]
)


# -------------------------
# Peer
# -------------------------

peer_risk_col = find_column(
    peer_df,
    [
        "peer_risk_score",
        "peer_risk",
        "Peer Risk Score"
    ]
)

peer_alert_col = find_column(
    peer_df,
    [
        "peer_alert_level",
        "peer_alert",
        "Peer Alert Level"
    ]
)

peer_confidence_col = find_column(
    peer_df,
    [
        "peer_confidence",
        "Peer Confidence"
    ]
)

peer_ratio_col = find_column(
    peer_df,
    [
        "peer_median_ratio",
        "peer_ratio",
        "Peer Median Ratio"
    ]
)

peer_deviation_col = find_column(
    peer_df,
    [
        "peer_deviation_pct",
        "peer_deviation_percent",
        "Peer Deviation"
    ]
)

peer_median_col = find_column(
    peer_df,
    [
        "peer_median",
        "Peer Median"
    ]
)

peer_group_col = find_column(
    peer_df,
    [
        "peer_group",
        "Peer Group"
    ]
)

peer_count_col = find_column(
    peer_df,
    [
        "peer_count",
        "Peer Count"
    ]
)


print("\n" + "=" * 70)
print("SOURCE SIGNAL DETECTION")
print("=" * 70)

print(
    f"\nStatistical score    : {stat_score_col}"
)

print(
    f"Statistical alert    : {stat_alert_col}"
)

print(
    f"Robust Z-score       : {stat_z_col}"
)

print(
    f"\nPeer risk score      : {peer_risk_col}"
)

print(
    f"Peer alert level     : {peer_alert_col}"
)

print(
    f"Peer confidence      : {peer_confidence_col}"
)


# ============================================================
# CREATE CLEAN STATISTICAL DATASET
# ============================================================

stat_clean = pd.DataFrame()

stat_clean["_join_key"] = stat_df[
    "_join_key"
]

if stat_score_col is not None:

    stat_clean["statistical_risk"] = (
        safe_numeric(
            stat_df[stat_score_col]
        )
    )

else:

    stat_clean["statistical_risk"] = 0.0


if stat_alert_col is not None:

    stat_clean["statistical_alert"] = (
        stat_df[stat_alert_col]
        .fillna("")
        .astype(str)
    )

else:

    stat_clean["statistical_alert"] = ""


if stat_z_col is not None:

    stat_clean["robust_z_score"] = (
        safe_numeric(
            stat_df[stat_z_col]
        )
    )

else:

    stat_clean["robust_z_score"] = 0.0


if stat_percentile_col is not None:

    stat_clean["statistical_percentile"] = (
        safe_numeric(
            stat_df[stat_percentile_col]
        )
    )

else:

    stat_clean["statistical_percentile"] = 0.0


stat_clean = stat_clean.drop_duplicates(
    subset="_join_key",
    keep="first"
)


# ============================================================
# CREATE CLEAN PEER DATASET
# ============================================================

peer_clean = pd.DataFrame()

peer_clean["_join_key"] = peer_df[
    "_join_key"
]


if peer_risk_col is not None:

    peer_clean["peer_risk"] = (
        safe_numeric(
            peer_df[peer_risk_col]
        )
    )

else:

    peer_clean["peer_risk"] = 0.0


if peer_alert_col is not None:

    peer_clean["peer_alert"] = (
        peer_df[peer_alert_col]
        .fillna("")
        .astype(str)
    )

else:

    peer_clean["peer_alert"] = ""


if peer_confidence_col is not None:

    peer_clean["peer_confidence"] = (
        peer_df[peer_confidence_col]
        .fillna("")
        .astype(str)
    )

else:

    peer_clean["peer_confidence"] = "INSUFFICIENT"


if peer_ratio_col is not None:

    peer_clean["peer_ratio"] = (
        safe_numeric(
            peer_df[peer_ratio_col]
        )
    )

else:

    peer_clean["peer_ratio"] = 0.0


if peer_deviation_col is not None:

    peer_clean["peer_deviation_pct"] = (
        safe_numeric(
            peer_df[peer_deviation_col]
        )
    )

else:

    peer_clean["peer_deviation_pct"] = 0.0


if peer_median_col is not None:

    peer_clean["peer_median"] = (
        safe_numeric(
            peer_df[peer_median_col]
        )
    )

else:

    peer_clean["peer_median"] = 0.0


if peer_group_col is not None:

    peer_clean["peer_group"] = (
        peer_df[peer_group_col]
        .fillna("")
        .astype(str)
    )

else:

    peer_clean["peer_group"] = ""


if peer_count_col is not None:

    peer_clean["peer_count"] = (
        safe_numeric(
            peer_df[peer_count_col]
        )
    )

else:

    peer_clean["peer_count"] = 0


peer_clean = peer_clean.drop_duplicates(
    subset="_join_key",
    keep="first"
)


# ============================================================
# MERGE
# ============================================================

print("\n" + "=" * 70)
print("MERGING INTELLIGENCE LAYERS")
print("=" * 70)

merged = anomaly_df.copy()


merged = merged.merge(
    stat_clean,
    on="_join_key",
    how="left"
)


merged = merged.merge(
    peer_clean,
    on="_join_key",
    how="left"
)


print(
    f"\nMerged records: {len(merged)}"
)


# ============================================================
# VERIFY SIGNALS
# ============================================================

merged["statistical_risk"] = safe_numeric(
    merged["statistical_risk"]
)

merged["peer_risk"] = safe_numeric(
    merged["peer_risk"]
)


stat_received = (
    merged["statistical_risk"] > 0
).sum()


peer_received = (
    merged["peer_risk"] > 0
).sum()


print("\n" + "=" * 70)
print("MERGE SUCCESS CHECK")
print("=" * 70)

print(
    f"\nRecords with statistical score > 0 : "
    f"{stat_received}"
)

print(
    f"Records with peer risk > 0          : "
    f"{peer_received}"
)


# ============================================================
# BASE ML SIGNAL
# ============================================================

ml_col = find_column(
    merged,
    [
        "anomaly_score",
        "Anomaly Score"
    ]
)

if ml_col is not None:

    merged["ml_risk"] = normalize_score(
        merged[ml_col]
    )

else:

    merged["ml_risk"] = 0.0


# ============================================================
# FINANCIAL SIGNAL
# ============================================================

percentile_col = find_column(
    merged,
    [
        "amount_percentile",
        "Amount Percentile"
    ]
)

if percentile_col is not None:

    merged["amount_percentile"] = safe_numeric(
        merged[percentile_col]
    )

else:

    merged["amount_percentile"] = 0.0


def financial_risk(percentile):

    if percentile >= 99:
        return 100.0

    if percentile >= 95:
        return 75.0

    if percentile >= 90:
        return 50.0

    if percentile >= 75:
        return 25.0

    return 0.0


merged["financial_risk"] = (
    merged["amount_percentile"]
    .apply(financial_risk)
)


# ============================================================
# PEER CONFIDENCE
# ============================================================

def confidence_multiplier(value):

    value = str(
        value
    ).upper().strip()

    if value == "HIGH":
        return 1.00

    if value == "MEDIUM":
        return 0.85

    if value == "LOW":
        return 0.60

    if value == "INSUFFICIENT":
        return 0.00

    return 0.50


merged["peer_confidence_multiplier"] = (
    merged["peer_confidence"]
    .apply(confidence_multiplier)
)


merged["adjusted_peer_risk"] = (
    merged["peer_risk"]
    * merged["peer_confidence_multiplier"]
)


# ============================================================
# STATISTICAL CONFIDENCE
# ============================================================

def statistical_multiplier(value):

    value = str(
        value
    ).upper().strip()

    if value == "HIGH":
        return 0.70

    if value == "MEDIUM":
        return 0.80

    if value == "WATCH":
        return 0.90

    return 0.85


merged["statistical_multiplier"] = (
    merged["statistical_alert"]
    .apply(statistical_multiplier)
)


merged["adjusted_statistical_risk"] = (
    merged["statistical_risk"]
    * merged["statistical_multiplier"]
)


# ============================================================
# DATA QUALITY
# ============================================================

quality_col = find_column(
    merged,
    [
        "metadata_risk",
        "Metadata Risk"
    ]
)

if quality_col is not None:

    merged["quality_risk"] = normalize_score(
        merged[quality_col]
    )

else:

    merged["quality_risk"] = 0.0


# ============================================================
# UNIFIED RISK SCORE
# ============================================================

print("\nCalculating unified risk score...")


merged["unified_risk_score"] = (

    0.35 * merged["ml_risk"]

    + 0.25 * merged["adjusted_peer_risk"]

    + 0.20 * merged["financial_risk"]

    + 0.15 * merged["adjusted_statistical_risk"]

    + 0.05 * merged["quality_risk"]
)


merged["unified_risk_score"] = (
    merged["unified_risk_score"]
    .clip(0, 100)
    .round(2)
)


# ============================================================
# RECORD TYPE
# ============================================================

record_type_col = find_column(
    merged,
    [
        "record_type",
        "Record Type"
    ]
)


if record_type_col is not None:

    valid_record = (
        merged[record_type_col]
        .astype(str)
        .str.upper()
        .eq("MP_RECORD")
    )

else:

    valid_record = pd.Series(
        True,
        index=merged.index
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(row):

    if not valid_record.loc[row.name]:

        return "NOT_APPLICABLE"

    score = float(
        row["unified_risk_score"]
    )

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MEDIUM"

    return "LOW"


merged["unified_risk_level"] = (
    merged.apply(
        calculate_risk_level,
        axis=1
    )
)


# ============================================================
# INVESTIGATION PRIORITY
# ============================================================

def calculate_priority(row):

    level = row[
        "unified_risk_level"
    ]

    score = float(
        row["unified_risk_score"]
    )

    if level == "CRITICAL":
        return "URGENT"

    if level == "HIGH":
        return "HIGH"

    if level == "MEDIUM" and score >= 50:
        return "WATCHLIST"

    if level == "MEDIUM":
        return "MONITOR"

    if level == "LOW":
        return "ROUTINE"

    return "NOT_APPLICABLE"


merged["investigation_priority"] = (
    merged.apply(
        calculate_priority,
        axis=1
    )
)


# ============================================================
# RISK SIGNAL EXPLANATION
# ============================================================

def collect_signals(row):

    signals = []

    ml = float(
        row["ml_risk"]
    )

    peer = float(
        row["adjusted_peer_risk"]
    )

    financial = float(
        row["financial_risk"]
    )

    statistical = float(
        row["adjusted_statistical_risk"]
    )

    quality = float(
        row["quality_risk"]
    )


    if ml >= 60:

        signals.append(
            "strong ML anomaly signal"
        )

    elif ml >= 30:

        signals.append(
            "moderate ML anomaly signal"
        )


    if peer >= 60:

        signals.append(
            "strong peer-group deviation"
        )

    elif peer >= 30:

        signals.append(
            "peer-group deviation"
        )


    if financial >= 75:

        signals.append(
            "allocation is in the top 5% financially"
        )

    elif financial >= 50:

        signals.append(
            "allocation is in the top 10% financially"
        )


    if statistical >= 60:

        signals.append(
            "strong statistical anomaly signal"
        )

    elif statistical >= 30:

        signals.append(
            "statistical anomaly signal"
        )


    if quality >= 50:

        signals.append(
            "data-quality concerns"
        )


    return signals


merged["risk_signals"] = (
    merged.apply(
        collect_signals,
        axis=1
    )
)


# ============================================================
# MEMBER / STATE
# ============================================================

member_col = find_column(
    merged,
    [
        "Hon'ble Members of Parliaments",
        "Hon'ble Members of Parliament"
    ]
)

state_col = find_column(
    merged,
    [
        "State"
    ]
)


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanation(row):

    level = row[
        "unified_risk_level"
    ]

    amount = float(
        row["amount_numeric"]
    )

    amount_text = (
        f"₹{amount:,.2f}"
    )

    if level == "NOT_APPLICABLE":

        return (
            "Record excluded from unified risk assessment."
        )


    signals = row[
        "risk_signals"
    ]


    if not signals:

        return (
            f"Allocation of {amount_text} shows no "
            "strong combined anomaly signal. "
            "Routine monitoring is recommended."
        )


    signal_text = ", ".join(
        signals
    )


    return (
        f"{level} risk candidate: allocation of "
        f"{amount_text} shows {signal_text}. "
        "Multiple independent intelligence signals "
        "were combined to prioritize this record "
        "for verification. This is an anomaly candidate "
        "and does not by itself indicate fraud."
    )


merged["unified_explanation"] = (
    merged.apply(
        generate_explanation,
        axis=1
    )
)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(level):

    if level == "CRITICAL":

        return (
            "Immediate verification of sanction, "
            "allocation, project documentation, "
            "implementing agency records, and "
            "expenditure details."
        )

    if level == "HIGH":

        return (
            "Detailed verification of allocation, "
            "supporting project records, sanction "
            "documentation, and implementing-agency details."
        )

    if level == "MEDIUM":

        return (
            "Monitor record and verify supporting "
            "documentation during the next review cycle."
        )

    if level == "LOW":

        return (
            "Routine monitoring; no immediate investigation "
            "required based on current signals."
        )

    return "No action required."


merged["recommended_action"] = (
    merged["unified_risk_level"]
    .apply(recommended_action)
)


# ============================================================
# ALERT STATUS
# ============================================================

def alert_status(level):

    if level == "CRITICAL":
        return "URGENT_INVESTIGATION"

    if level == "HIGH":
        return "INVESTIGATION_ALERT"

    if level == "MEDIUM":
        return "MONITORING_ALERT"

    if level == "LOW":
        return "ROUTINE"

    return "NOT_APPLICABLE"


merged["alert_status"] = (
    merged["unified_risk_level"]
    .apply(alert_status)
)


# ============================================================
# RANK
# ============================================================

rankable = merged[
    merged["unified_risk_level"]
    != "NOT_APPLICABLE"
].copy()

rankable = rankable.sort_values(
    "unified_risk_score",
    ascending=False
)

rank_map = {}

for rank, index in enumerate(
    rankable.index,
    start=1
):

    rank_map[index] = rank


merged["risk_rank"] = (
    merged.index
    .map(rank_map)
    .fillna(0)
    .astype(int)
)


# ============================================================
# COMPONENT SCORES
# ============================================================

merged["ml_component_score"] = (
    merged["ml_risk"]
    .round(2)
)

merged["peer_component_score"] = (
    merged["adjusted_peer_risk"]
    .round(2)
)

merged["financial_component_score"] = (
    merged["financial_risk"]
    .round(2)
)

merged["statistical_component_score"] = (
    merged["adjusted_statistical_risk"]
    .round(2)
)

merged["quality_component_score"] = (
    merged["quality_risk"]
    .round(2)
)


# ============================================================
# SORT
# ============================================================

merged = merged.sort_values(
    [
        "unified_risk_score",
        "amount_numeric"
    ],
    ascending=[
        False,
        False
    ]
)


# ============================================================
# REMOVE INTERNAL KEY
# ============================================================

merged = merged.drop(
    columns=["_join_key"],
    errors="ignore"
)


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("UNIFIED RISK INTELLIGENCE SUMMARY")
print("=" * 70)

print(
    f"\nTotal records : {len(merged)}"
)

print(
    f"Statistical signals received : {stat_received}"
)

print(
    f"Peer signals received        : {peer_received}"
)


print("\nUNIFIED RISK DISTRIBUTION")

distribution = (
    merged["unified_risk_level"]
    .value_counts()
)


for level in [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
    "NOT_APPLICABLE"
]:

    print(
        f"{level:<15}: "
        f"{distribution.get(level, 0)}"
    )


print("\nINVESTIGATION PRIORITY")

priority_distribution = (
    merged["investigation_priority"]
    .value_counts()
)


for priority in [
    "URGENT",
    "HIGH",
    "WATCHLIST",
    "MONITOR",
    "ROUTINE",
    "NOT_APPLICABLE"
]:

    print(
        f"{priority:<15}: "
        f"{priority_distribution.get(priority, 0)}"
    )


# ============================================================
# TOP 10
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 UNIFIED RISK CANDIDATES")
print("=" * 70)


top10 = merged[
    merged["unified_risk_level"]
    != "NOT_APPLICABLE"
].head(10)


for _, row in top10.iterrows():

    member = text_value(
        row,
        member_col
    )

    state = text_value(
        row,
        state_col
    )

    print(
        f"\nRank {int(row['risk_rank'])}"
    )

    print(
        f"Member: {member}"
    )

    print(
        f"State: {state}"
    )

    print(
        f"Amount: ₹"
        f"{float(row['amount_numeric']):,.2f}"
    )

    print(
        f"ML Score: "
        f"{float(row['ml_component_score']):.2f}"
    )

    print(
        f"Peer Score: "
        f"{float(row['peer_component_score']):.2f}"
    )

    print(
        f"Financial Score: "
        f"{float(row['financial_component_score']):.2f}"
    )

    print(
        f"Statistical Score: "
        f"{float(row['statistical_component_score']):.2f}"
    )

    print(
        f"Unified Risk: "
        f"{float(row['unified_risk_score']):.2f}"
    )

    print(
        f"Risk Level: "
        f"{row['unified_risk_level']}"
    )

    print(
        f"Priority: "
        f"{row['investigation_priority']}"
    )

    print(
        f"Explanation: "
        f"{row['unified_explanation']}"
    )

    print("-" * 70)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1.8.1 COMPLETE")
print("=" * 70)

print(
    "\nOutput saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nUnified risk scores combine ML, peer, "
    "financial, statistical, and data-quality signals."
)

print(
    "Risk scores are prioritization signals "
    "for verification, not proof of fraud."
)