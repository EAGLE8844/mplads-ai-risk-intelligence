import os
import numpy as np
import pandas as pd


# ============================================================
# MPLADS AI - PEER GROUP INTELLIGENCE ENGINE
# PHASE 1.6
# ============================================================

print("\n" + "=" * 70)
print("MPLADS AI PEER-GROUP INTELLIGENCE ENGINE")
print("=" * 70)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "allocation_anomaly_results.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "peer_intelligence_results.csv"
)


# ============================================================
# 2. CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        "\nInput file not found:\n"
        + INPUT_FILE
        + "\n\n"
        + "Make sure allocation_anomaly_results.csv "
        + "exists inside the processed folder."
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nRecords loaded: {len(df)}")


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "record_type",
    "ml_eligible",
    "amount_numeric",
    "state_clean"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f" - {column}"
            for column in missing_columns
        )
    )

print("Input validation successful.")


# ============================================================
# 5. CLEAN AMOUNT
# ============================================================

df["amount_numeric"] = pd.to_numeric(
    df["amount_numeric"],
    errors="coerce"
)


# ============================================================
# 6. CLEAN ML ELIGIBILITY
# ============================================================

df["ml_eligible"] = (
    df["ml_eligible"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin([
        "true",
        "1",
        "yes",
        "y"
    ])
)


# ============================================================
# 7. CLEAN STATE
# ============================================================

df["state_clean"] = (
    df["state_clean"]
    .astype(str)
    .str.strip()
)

df["state_clean"] = df[
    "state_clean"
].replace(
    [
        "",
        "nan",
        "None",
        "NA",
        "N/A"
    ],
    np.nan
)


# ============================================================
# 8. DETERMINE CONSTITUENCY TYPE
# ============================================================

if "Constituency" in df.columns:

    constituency = (
        df["Constituency"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["peer_type"] = np.select(
        [
            constituency.str.contains(
                r"\(SC\)",
                regex=True,
                na=False
            ),
            constituency.str.contains(
                r"\(ST\)",
                regex=True,
                na=False
            )
        ],
        [
            "SC",
            "ST"
        ],
        default="GENERAL"
    )

else:

    df["peer_type"] = "GENERAL"


# ============================================================
# 9. CREATE PEER GROUP
# ============================================================

df["peer_group"] = (
    df["state_clean"].fillna("UNKNOWN")
    + "_"
    + df["peer_type"]
)

print("\nPeer groups created.")


# ============================================================
# 10. SELECT VALID MP RECORDS
# ============================================================

eligible_mask = (
    (df["record_type"] == "MP_RECORD")
    &
    (df["ml_eligible"])
    &
    (df["amount_numeric"].notna())
    &
    (df["amount_numeric"] >= 0)
    &
    (df["state_clean"].notna())
)

eligible = df.loc[
    eligible_mask
].copy()

print(
    f"Peer analysis records: {len(eligible)}"
)


# ============================================================
# 11. PEER GROUP STATISTICS
# ============================================================

peer_stats = (
    eligible
    .groupby(
        "peer_group",
        dropna=False
    )["amount_numeric"]
    .agg(
        peer_count="count",
        peer_mean="mean",
        peer_median="median",
        peer_min="min",
        peer_max="max",
        peer_std="std"
    )
    .reset_index()
)


# ============================================================
# 12. MERGE PEER STATISTICS
# ============================================================

df = df.merge(
    peer_stats,
    on="peer_group",
    how="left"
)


# ============================================================
# 13. MP VS PEER MEDIAN RATIO
# ============================================================

df["peer_median_ratio"] = np.where(
    (
        df["peer_median"].notna()
        &
        (df["peer_median"] > 0)
    ),
    df["amount_numeric"] / df["peer_median"],
    np.nan
)


# ============================================================
# 14. PEER DEVIATION PERCENTAGE
# ============================================================

df["peer_deviation_pct"] = np.where(
    (
        df["peer_median"].notna()
        &
        (df["peer_median"] > 0)
    ),
    (
        (
            df["amount_numeric"]
            -
            df["peer_median"]
        )
        /
        df["peer_median"]
    ) * 100,
    np.nan
)


# ============================================================
# 15. PEER PERCENTILE
# ============================================================

df["peer_percentile"] = np.nan

for group_name, group in eligible.groupby(
    "peer_group",
    sort=False
):

    indices = group.index

    ranks = (
        group["amount_numeric"]
        .rank(
            method="average",
            pct=True
        )
        * 100
    )

    df.loc[
        indices,
        "peer_percentile"
    ] = ranks


# ============================================================
# 16. PEER CONFIDENCE
# ============================================================

def calculate_peer_confidence(count):

    if pd.isna(count):
        return "NONE"

    count = int(count)

    if count >= 10:
        return "HIGH"

    if count >= 5:
        return "MEDIUM"

    if count >= 3:
        return "LOW"

    return "INSUFFICIENT"


df["peer_confidence"] = (
    df["peer_count"]
    .apply(calculate_peer_confidence)
)


# ============================================================
# 17. PEER RISK SCORE
# ============================================================

def calculate_peer_risk(row):

    ratio = row["peer_median_ratio"]
    count = row["peer_count"]

    if pd.isna(ratio):
        return 0.0

    if pd.isna(count):
        return 0.0

    count = int(count)

    # Not enough comparable records
    if count < 3:
        return 0.0

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if ratio <= 1.10:
        score = 0

    # --------------------------------------------------------
    # SLIGHTLY ABOVE PEER
    # --------------------------------------------------------

    elif ratio <= 1.25:
        score = 20

    # --------------------------------------------------------
    # MODERATELY ABOVE PEER
    # --------------------------------------------------------

    elif ratio <= 1.50:
        score = 40

    # --------------------------------------------------------
    # SIGNIFICANTLY ABOVE PEER
    # --------------------------------------------------------

    elif ratio <= 2.00:
        score = 65

    # --------------------------------------------------------
    # EXTREMELY ABOVE PEER
    # --------------------------------------------------------

    else:
        score = 90

    # Small peer groups receive lower maximum confidence
    if count < 5:
        score = min(score, 40)

    elif count < 10:
        score = min(score, 65)

    return float(score)


df["peer_risk_score"] = df.apply(
    calculate_peer_risk,
    axis=1
)


# ============================================================
# 18. PEER ALERT LEVEL
# ============================================================

def calculate_peer_alert(score):

    if pd.isna(score):
        return "NORMAL"

    if score >= 65:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    if score >= 20:
        return "WATCH"

    return "NORMAL"


df["peer_alert_level"] = (
    df["peer_risk_score"]
    .apply(calculate_peer_alert)
)


# ============================================================
# 19. DEVIATION CATEGORY
# ============================================================

def deviation_category(ratio):

    if pd.isna(ratio):
        return "UNAVAILABLE"

    if ratio <= 1.10:
        return "NORMAL"

    if ratio <= 1.25:
        return "SLIGHTLY_ABOVE_PEER"

    if ratio <= 1.50:
        return "MODERATELY_ABOVE_PEER"

    if ratio <= 2.00:
        return "SIGNIFICANTLY_ABOVE_PEER"

    return "EXTREMELY_ABOVE_PEER"


df["peer_deviation_category"] = (
    df["peer_median_ratio"]
    .apply(deviation_category)
)


# ============================================================
# 20. PEER EXPLANATION
# ============================================================

def generate_peer_explanation(row):

    if row["record_type"] != "MP_RECORD":
        return (
            "Peer comparison not applicable "
            "for aggregate records."
        )

    if pd.isna(row["peer_median"]):
        return (
            "Peer comparison unavailable because "
            "no comparable peer group was identified."
        )

    ratio = row["peer_median_ratio"]
    count = row["peer_count"]

    if pd.isna(ratio):
        return "Peer comparison unavailable."

    count = int(count)

    median = row["peer_median"]

    # NORMAL
    if ratio <= 1.10:

        return (
            f"Allocation is within 10% of the "
            f"peer-group median of "
            f"₹{median:,.2f}. "
            f"Peer group contains {count} records."
        )

    # SLIGHT
    if ratio <= 1.25:

        return (
            f"Allocation is approximately "
            f"{ratio:.2f}x the peer-group median "
            f"of ₹{median:,.2f}."
        )

    # MODERATE
    if ratio <= 1.50:

        return (
            f"Allocation is approximately "
            f"{ratio:.2f}x the peer-group median "
            f"of ₹{median:,.2f}. "
            f"This represents a moderate "
            f"peer-group deviation."
        )

    # SIGNIFICANT
    if ratio <= 2.00:

        return (
            f"Allocation is approximately "
            f"{ratio:.2f}x the peer-group median "
            f"of ₹{median:,.2f}. "
            f"This is a significant deviation "
            f"and may require verification."
        )

    # EXTREME
    return (
        f"Allocation is approximately "
        f"{ratio:.2f}x the peer-group median "
        f"of ₹{median:,.2f}. "
        f"This is an extreme deviation "
        f"from comparable records and "
        f"requires verification."
    )


df["peer_explanation"] = df.apply(
    generate_peer_explanation,
    axis=1
)


# ============================================================
# 21. PEER DATA QUALITY
# ============================================================

def calculate_peer_data_quality(count):

    if pd.isna(count):
        return "INSUFFICIENT"

    count = int(count)

    if count >= 10:
        return "SUFFICIENT"

    if count >= 5:
        return "MODERATE"

    if count >= 3:
        return "LIMITED"

    return "INSUFFICIENT"


df["peer_data_quality"] = (
    df["peer_count"]
    .apply(calculate_peer_data_quality)
)


# ============================================================
# 22. PEER REVIEW FLAG
# ============================================================

df["peer_review_required"] = (
    df["peer_alert_level"].isin([
        "HIGH",
        "MEDIUM"
    ])
    &
    df["peer_data_quality"].isin([
        "SUFFICIENT",
        "MODERATE"
    ])
)


# ============================================================
# 23. ROUND NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "peer_mean",
    "peer_median",
    "peer_min",
    "peer_max",
    "peer_std",
    "peer_median_ratio",
    "peer_deviation_pct",
    "peer_percentile",
    "peer_risk_score"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].round(4)


# ============================================================
# 24. SAVE OUTPUT
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 25. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PEER INTELLIGENCE SUMMARY")
print("=" * 70)

total_groups = df["peer_group"].nunique()

sufficient_count = (
    df["peer_data_quality"] == "SUFFICIENT"
).sum()

moderate_count = (
    df["peer_data_quality"] == "MODERATE"
).sum()

limited_count = (
    df["peer_data_quality"] == "LIMITED"
).sum()

print(
    f"\nPeer groups created: {total_groups}"
)

print(
    f"Records with sufficient peer data: "
    f"{sufficient_count}"
)

print(
    f"Records with moderate peer data: "
    f"{moderate_count}"
)

print(
    f"Records with limited peer data: "
    f"{limited_count}"
)


# ============================================================
# 26. ALERT DISTRIBUTION
# ============================================================

print("\nPeer Alert Distribution:")

alert_distribution = (
    df[
        df["record_type"] == "MP_RECORD"
    ]["peer_alert_level"]
    .value_counts()
)

print(alert_distribution)


# ============================================================
# 27. CONFIDENCE DISTRIBUTION
# ============================================================

print("\nPeer Confidence Distribution:")

confidence_distribution = (
    df[
        df["record_type"] == "MP_RECORD"
    ]["peer_confidence"]
    .value_counts()
)

print(confidence_distribution)


# ============================================================
# 28. TOP PEER ANOMALIES
# ============================================================

top = (
    df[
        (df["record_type"] == "MP_RECORD")
        &
        (df["peer_risk_score"] > 0)
    ]
    .sort_values(
        [
            "peer_risk_score",
            "peer_median_ratio"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(10)
)


print("\n" + "=" * 70)
print("TOP 10 PEER ANOMALIES")
print("=" * 70)


# ============================================================
# 29. DISPLAY TOP RESULTS
# ============================================================

for _, row in top.iterrows():

    if "Hon'ble Members of Parliaments" in df.columns:

        name = row[
            "Hon'ble Members of Parliaments"
        ]

    elif "Hon'ble Members of Parliament" in df.columns:

        name = row[
            "Hon'ble Members of Parliament"
        ]

    else:

        name = "Unknown"

    serial = row.get(
        "Sr. No.",
        "N/A"
    )

    print(
        f"\nSr {serial} {name}"
    )

    print(
        f"State: {row['state_clean']}"
    )

    print(
        f"Peer Group: {row['peer_group']}"
    )

    if pd.notna(row["peer_count"]):

        print(
            f"Peer Count: "
            f"{int(row['peer_count'])}"
        )

    else:

        print(
            "Peer Count: N/A"
        )

    print(
        f"Amount: "
        f"₹{row['amount_numeric']:,.2f}"
    )

    if pd.notna(row["peer_median"]):

        print(
            f"Peer Median: "
            f"₹{row['peer_median']:,.2f}"
        )

    else:

        print(
            "Peer Median: N/A"
        )

    if pd.notna(row["peer_median_ratio"]):

        print(
            f"Peer Ratio: "
            f"{row['peer_median_ratio']:.2f}x"
        )

    else:

        print(
            "Peer Ratio: N/A"
        )

    if pd.notna(row["peer_deviation_pct"]):

        print(
            f"Peer Deviation: "
            f"{row['peer_deviation_pct']:.2f}%"
        )

    else:

        print(
            "Peer Deviation: N/A"
        )

    if pd.notna(row["peer_percentile"]):

        print(
            f"Peer Percentile: "
            f"{row['peer_percentile']:.2f}"
        )

    else:

        print(
            "Peer Percentile: N/A"
        )

    print(
        f"Peer Risk: "
        f"{row['peer_risk_score']:.2f}"
    )

    print(
        f"Alert: "
        f"{row['peer_alert_level']}"
    )

    print(
        f"Confidence: "
        f"{row['peer_confidence']}"
    )

    print(
        f"Explanation: "
        f"{row['peer_explanation']}"
    )


# ============================================================
# 30. COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PEER INTELLIGENCE COMPLETE")
print("=" * 70)

print(
    "\nOutput saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nNext step:"
)

print(
    "Phase 1.7 - Robust Statistical Anomaly Detection"
)

print("=" * 70)