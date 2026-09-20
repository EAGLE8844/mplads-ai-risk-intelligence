from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ============================================================
# MPLADS AI ANOMALY DETECTION ENGINE
# ============================================================
#
# Purpose:
# Detect unusual MP allocation patterns using:
#
# 1. Isolation Forest
# 2. Robust statistical deviation
# 3. Percentile analysis
# 4. Metadata quality
#
# Important:
# Aggregate rows such as "Grand Total" are NOT treated as
# MP-level anomalies.
#
# Output:
# processed/allocation_anomaly_results.csv
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "processed"
    / "allocation_processed.csv"
)

OUTPUT_DIR = (
    ROOT
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "allocation_anomaly_results.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CONTAMINATION = 0.03

RANDOM_STATE = 42

N_ESTIMATORS = 300


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_score(series):
    """
    Convert numerical values to a 0-100 scale.

    Higher value = greater anomaly.
    """

    series = pd.Series(
        series,
        dtype="float64"
    )

    minimum = series.min()
    maximum = series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or minimum == maximum
    ):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
        * 100
    )


def robust_deviation(values):
    """
    Calculate robust statistical deviation.

    Uses Median Absolute Deviation (MAD)
    instead of ordinary mean/std.

    This makes the calculation less sensitive
    to extreme values.
    """

    values = pd.Series(
        values,
        dtype="float64"
    )

    median = values.median()

    mad = np.median(
        np.abs(values - median)
    )

    if (
        mad == 0
        or pd.isna(mad)
    ):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index
        )

    robust_z = (
        0.6745
        *
        (values - median)
        /
        mad
    )

    return robust_z.abs()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 60)
    print("MPLADS AI ANOMALY ENGINE")
    print("=" * 60)

    print()
    print("Loading processed allocation data...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n\n"
            "Please run data_quality.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Records loaded:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    return df


# ============================================================
# RECORD TYPE CLASSIFICATION
# ============================================================

def classify_record_types(df):

    print()
    print("=" * 60)
    print("CLASSIFYING RECORD TYPES")
    print("=" * 60)

    def classify_record(row):

        # ----------------------------------------------------
        # Collect identifying fields
        # ----------------------------------------------------

        sr_no = str(
            row.get(
                "Sr. No.",
                ""
            )
        ).strip().lower()

        state = str(
            row.get(
                "state_clean",
                ""
            )
        ).strip().lower()

        mp = str(
            row.get(
                "mp_clean",
                ""
            )
        ).strip().lower()

        constituency = str(
            row.get(
                "constituency_clean",
                ""
            )
        ).strip().lower()

        combined = (
            sr_no
            + " "
            + state
            + " "
            + mp
            + " "
            + constituency
        )

        # ----------------------------------------------------
        # Aggregate rows
        # ----------------------------------------------------

        aggregate_keywords = [
            "grand total",
            "total",
            "subtotal",
            "sub total"
        ]

        if any(
            keyword in combined
            for keyword in aggregate_keywords
        ):
            return "AGGREGATE"

        # ----------------------------------------------------
        # Normal MP record
        # ----------------------------------------------------

        if (
            pd.notna(
                row.get("mp_clean")
            )
            and
            pd.notna(
                row.get("constituency_clean")
            )
        ):

            return "MP_RECORD"

        # ----------------------------------------------------
        # Incomplete record
        # ----------------------------------------------------

        return "INCOMPLETE"

    df["record_type"] = df.apply(
        classify_record,
        axis=1
    )

    # --------------------------------------------------------
    # ML eligibility
    # --------------------------------------------------------

    df["ml_eligible"] = (
        df["record_type"]
        == "MP_RECORD"
    )

    print()

    print(
        "Record type distribution:"
    )

    print(
        df["record_type"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "ML eligible records:",
        int(
            df["ml_eligible"].sum()
        )
    )

    print(
        "Excluded records:",
        int(
            (~df["ml_eligible"]).sum()
        )
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    print()
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # Convert amount to numeric
    # --------------------------------------------------------

    df["amount_numeric"] = pd.to_numeric(
        df["allocated_amount"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Valid financial records
    # --------------------------------------------------------

    valid_mask = (
        df["amount_numeric"].notna()
        &
        (
            df["amount_numeric"]
            > 0
        )
        &
        df["ml_eligible"]
    )

    df["amount_valid"] = valid_mask

    # --------------------------------------------------------
    # Log transformation
    #
    # Helps reduce the influence of extremely large values.
    # --------------------------------------------------------

    df["log_amount"] = np.nan

    df.loc[
        valid_mask,
        "log_amount"
    ] = np.log1p(
        df.loc[
            valid_mask,
            "amount_numeric"
        ]
    )

    # --------------------------------------------------------
    # Amount percentile
    # --------------------------------------------------------

    df["amount_percentile"] = np.nan

    if valid_mask.sum() > 0:

        valid_amounts = df.loc[
            valid_mask,
            "amount_numeric"
        ]

        df.loc[
            valid_mask,
            "amount_percentile"
        ] = (
            valid_amounts.rank(
                pct=True
            )
            * 100
        )

    # --------------------------------------------------------
    # Robust deviation
    # --------------------------------------------------------

    df["robust_deviation"] = np.nan

    if valid_mask.sum() > 0:

        robust_values = robust_deviation(
            df.loc[
                valid_mask,
                "log_amount"
            ]
        )

        df.loc[
            valid_mask,
            "robust_deviation"
        ] = robust_values

    # --------------------------------------------------------
    # Metadata columns
    # --------------------------------------------------------

    metadata_columns = [
        "missing_state",
        "missing_mp",
        "missing_constituency"
    ]

    for column in metadata_columns:

        if column not in df.columns:

            df[column] = False

        df[column] = (
            df[column]
            .fillna(False)
            .astype(bool)
        )

    # --------------------------------------------------------
    # Missing metadata count
    # --------------------------------------------------------

    df["metadata_missing_count"] = (
        df[
            metadata_columns
        ]
        .astype(int)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Metadata risk
    # --------------------------------------------------------

    df["metadata_risk"] = (
        df["metadata_missing_count"]
        / 3
        * 100
    )

    print(
        "Features created successfully."
    )

    return df


# ============================================================
# ISOLATION FOREST
# ============================================================

def run_isolation_forest(df):

    print()
    print("=" * 60)
    print("RUNNING ISOLATION FOREST")
    print("=" * 60)

    # --------------------------------------------------------
    # IMPORTANT:
    # Only valid MP records are passed to ML.
    # --------------------------------------------------------

    valid_mask = (
        df["ml_eligible"]
        &
        df["log_amount"].notna()
        &
        (
            df["amount_numeric"]
            > 0
        )
    )

    if valid_mask.sum() < 10:

        print(
            "Not enough valid records for Isolation Forest."
        )

        df["isolation_score"] = 0.0

        df["isolation_anomaly"] = False

        return df

    X = df.loc[
        valid_mask,
        ["log_amount"]
    ]

    print(
        "Records used for ML:",
        len(X)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )

    model.fit(X)

    # --------------------------------------------------------
    # Decision function
    #
    # Higher = more normal
    # Lower  = more anomalous
    # --------------------------------------------------------

    decision_scores = (
        model
        .decision_function(X)
    )

    # Convert:
    #
    # Higher = more anomalous

    raw_anomaly_scores = (
        -decision_scores
    )

    anomaly_scores = normalize_score(
        pd.Series(
            raw_anomaly_scores,
            index=X.index
        )
    )

    # --------------------------------------------------------
    # Store scores
    # --------------------------------------------------------

    df["isolation_score"] = 0.0

    df.loc[
        valid_mask,
        "isolation_score"
    ] = anomaly_scores

    # --------------------------------------------------------
    # Model classification
    # --------------------------------------------------------

    predictions = (
        model
        .predict(X)
    )

    df["isolation_anomaly"] = False

    df.loc[
        valid_mask,
        "isolation_anomaly"
    ] = (
        predictions == -1
    )

    anomaly_count = int(
        df["isolation_anomaly"]
        .sum()
    )

    print(
        "Potential Isolation Forest anomalies:",
        anomaly_count
    )

    return df


# ============================================================
# STATISTICAL ANOMALY SIGNAL
# ============================================================

def calculate_statistical_signal(df):

    print()
    print(
        "Calculating statistical anomaly signal..."
    )

    # --------------------------------------------------------
    # Cap robust deviation.
    #
    # Prevent one extreme value from dominating the score.
    # --------------------------------------------------------

    df["statistical_score"] = (
        df["robust_deviation"]
        .fillna(0)
        .clip(
            lower=0,
            upper=10
        )
        / 10
        * 100
    )

    return df


# ============================================================
# PERCENTILE SIGNAL
# ============================================================

def calculate_percentile_signal(df):

    print(
        "Calculating percentile signal..."
    )

    percentile = (
        df["amount_percentile"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Percentile-based signal
    #
    # >99th percentile = strong signal
    # >95th percentile = moderate signal
    # >90th percentile = weak signal
    # --------------------------------------------------------

    df["percentile_signal"] = np.where(

        percentile >= 99,

        100,

        np.where(

            percentile >= 95,

            75,

            np.where(

                percentile >= 90,

                40,

                0
            )
        )
    )

    return df


# ============================================================
# UNIFIED ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(df):

    print()
    print("=" * 60)
    print("CALCULATING UNIFIED ANOMALY SCORE")
    print("=" * 60)

    # --------------------------------------------------------
    # Weighted model
    #
    # Isolation Forest       = 45%
    # Statistical deviation  = 25%
    # Percentile             = 15%
    # Metadata quality       = 15%
    # --------------------------------------------------------

    df["anomaly_score"] = (

        0.45
        * df["isolation_score"]

        +

        0.25
        * df["statistical_score"]

        +

        0.15
        * df["percentile_signal"]

        +

        0.15
        * df["metadata_risk"]
    )

    df["anomaly_score"] = (
        df["anomaly_score"]
        .clip(
            lower=0,
            upper=100
        )
        .round(2)
    )

    # --------------------------------------------------------
    # Non-MP records
    #
    # These are retained but do not receive a risk score.
    # --------------------------------------------------------

    df.loc[
        ~df["ml_eligible"],
        "anomaly_score"
    ] = np.nan

    return df


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(df):

    print(
        "Classifying anomaly risk..."
    )

    def classify(row):

        # ----------------------------------------------------
        # Aggregate/incomplete records
        # ----------------------------------------------------

        if not row["ml_eligible"]:

            return "NOT_APPLICABLE"

        score = row["anomaly_score"]

        if pd.isna(score):

            return "NOT_APPLICABLE"

        # ----------------------------------------------------
        # Risk levels
        # ----------------------------------------------------

        if score >= 80:

            return "CRITICAL"

        elif score >= 60:

            return "HIGH"

        elif score >= 30:

            return "MEDIUM"

        else:

            return "LOW"

    df["risk_level"] = df.apply(
        classify,
        axis=1
    )

    return df


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def generate_explanations(df):

    print(
        "Generating explainable AI reasons..."
    )

    explanations = []

    for _, row in df.iterrows():

        reasons = []

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        if row["record_type"] == "AGGREGATE":

            reasons.append(
                "Aggregate/total record excluded "
                "from MP-level anomaly detection"
            )

            explanations.append(
                " | ".join(reasons)
            )

            continue

        # ----------------------------------------------------
        # Incomplete
        # ----------------------------------------------------

        if row["record_type"] == "INCOMPLETE":

            reasons.append(
                "Incomplete record excluded from "
                "MP-level anomaly scoring"
            )

            explanations.append(
                " | ".join(reasons)
            )

            continue

        # ----------------------------------------------------
        # Isolation Forest
        # ----------------------------------------------------

        if row["isolation_score"] >= 70:

            reasons.append(
                "Machine-learning model detected "
                "an unusual allocation pattern"
            )

        # ----------------------------------------------------
        # Statistical signal
        # ----------------------------------------------------

        if row["statistical_score"] >= 70:

            reasons.append(
                "Allocation differs significantly "
                "from the normal distribution"
            )

        # ----------------------------------------------------
        # Percentile
        # ----------------------------------------------------

        if row["percentile_signal"] >= 75:

            reasons.append(
                "Allocation lies in an unusually "
                "high percentile"
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        missing_count = int(
            row["metadata_missing_count"]
        )

        if missing_count > 0:

            reasons.append(
                f"{missing_count} important "
                "identifying field(s) missing"
            )

        # ----------------------------------------------------
        # Existing data-quality flag
        # ----------------------------------------------------

        if (
            "data_quality_alert" in row.index
            and bool(
                row["data_quality_alert"]
            )
        ):

            reasons.append(
                "Data-quality validation requires review"
            )

        # ----------------------------------------------------
        # No significant signals
        # ----------------------------------------------------

        if not reasons:

            reasons.append(
                "No strong anomaly signal detected"
            )

        explanations.append(
            " | ".join(reasons)
        )

    df["ai_explanation"] = explanations

    return df


# ============================================================
# ALERT GENERATION
# ============================================================

def create_alerts(df):

    print(
        "Generating investigation alerts..."
    )

    # --------------------------------------------------------
    # Only genuine MP records can require review.
    # --------------------------------------------------------

    df["requires_review"] = (

        df["ml_eligible"]

        &

        (
            df["anomaly_score"]
            >= 60
        )
    )

    # --------------------------------------------------------
    # Alert priority
    # --------------------------------------------------------

    df["alert_priority"] = np.select(

        [
            (
                df["ml_eligible"]
                &
                (
                    df["anomaly_score"]
                    >= 80
                )
            ),

            (
                df["ml_eligible"]
                &
                (
                    df["anomaly_score"]
                    >= 60
                )
            ),

            (
                df["ml_eligible"]
                &
                (
                    df["anomaly_score"]
                    >= 30
                )
            )
        ],

        [
            "CRITICAL",
            "HIGH",
            "MEDIUM"
        ],

        default="LOW"
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    print()
    print("=" * 60)
    print("SAVING ANOMALY RESULTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Highest-risk records first.
    #
    # NaN values naturally appear at the bottom.
    # --------------------------------------------------------

    df = df.sort_values(
        "anomaly_score",
        ascending=False,
        na_position="last"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RISK DISTRIBUTION")
    print("=" * 60)

    risk_counts = (
        df["risk_level"]
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
            f"{level:15s}: "
            f"{int(risk_counts.get(level, 0))}"
        )

    # --------------------------------------------------------
    # Review count
    # --------------------------------------------------------

    print()

    print(
        "Records requiring review:",
        int(
            df["requires_review"]
            .sum()
        )
    )

    return df


# ============================================================
# DISPLAY TOP RECORDS
# ============================================================

def display_top_records(df):

    print()
    print("=" * 60)
    print("TOP 10 MP RISK RECORDS")
    print("=" * 60)

    # --------------------------------------------------------
    # Only genuine MP records
    # --------------------------------------------------------

    top_records = (
        df[
            df["ml_eligible"]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(10)
    )

    display_columns = [
        "Sr. No.",
        "state_clean",
        "mp_clean",
        "constituency_clean",
        "amount_numeric",
        "amount_percentile",
        "isolation_score",
        "statistical_score",
        "anomaly_score",
        "risk_level",
        "requires_review"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in top_records.columns
    ]

    if len(top_records) == 0:

        print(
            "No valid MP records found."
        )

        return

    print()

    print(
        top_records[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TOP ALERT EXPLANATIONS")
    print("=" * 60)

    for _, row in top_records.head(5).iterrows():

        print()

        print(
            f"Record: {row.get('Sr. No.')}"
        )

        print(
            f"MP: {row.get('mp_clean')}"
        )

        print(
            f"Constituency: "
            f"{row.get('constituency_clean')}"
        )

        print(
            f"Allocation: "
            f"₹{row.get('amount_numeric'):,.2f}"
        )

        print(
            f"Risk Score: "
            f"{row.get('anomaly_score')}/100"
        )

        print(
            f"Risk Level: "
            f"{row.get('risk_level')}"
        )

        print(
            "Explanation:"
        )

        print(
            row.get(
                "ai_explanation"
            )
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Classify record types
    # --------------------------------------------------------

    df = classify_record_types(
        df
    )

    # --------------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------------

    df = create_features(
        df
    )

    # --------------------------------------------------------
    # 4. Machine learning
    # --------------------------------------------------------

    df = run_isolation_forest(
        df
    )

    # --------------------------------------------------------
    # 5. Statistical signal
    # --------------------------------------------------------

    df = calculate_statistical_signal(
        df
    )

    # --------------------------------------------------------
    # 6. Percentile signal
    # --------------------------------------------------------

    df = calculate_percentile_signal(
        df
    )

    # --------------------------------------------------------
    # 7. Unified anomaly score
    # --------------------------------------------------------

    df = calculate_anomaly_score(
        df
    )

    # --------------------------------------------------------
    # 8. Risk classification
    # --------------------------------------------------------

    df = classify_risk(
        df
    )

    # --------------------------------------------------------
    # 9. Explainability
    # --------------------------------------------------------

    df = generate_explanations(
        df
    )

    # --------------------------------------------------------
    # 10. Alerts
    # --------------------------------------------------------

    df = create_alerts(
        df
    )

    # --------------------------------------------------------
    # 11. Save
    # --------------------------------------------------------

    df = save_results(
        df
    )

    # --------------------------------------------------------
    # 12. Display results
    # --------------------------------------------------------

    display_top_records(
        df
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ANOMALY ENGINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        "Output file:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()