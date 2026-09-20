from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MPLADS AI RISK INTELLIGENCE ENGINE
# ============================================================
#
# Input:
#     processed/allocation_anomaly_results.csv
#
# Output:
#     processed/risk_results.csv
#
# Purpose:
#     Convert anomaly signals into an explainable
#     investigation-priority score.
#
# IMPORTANT:
#     This is NOT a "fraud probability".
#     It is an anomaly/risk prioritization score.
#
#     Final verification must be performed by authorized
#     officials/auditors.
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "processed"
    / "allocation_anomaly_results.csv"
)

OUTPUT_FILE = (
    ROOT
    / "processed"
    / "risk_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Score thresholds

MEDIUM_THRESHOLD = 30

HIGH_THRESHOLD = 60

CRITICAL_THRESHOLD = 80


# Review threshold

# Medium cases above this value will also be placed
# in the officer watchlist.

MEDIUM_REVIEW_THRESHOLD = 50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 65)
    print("MPLADS AI RISK INTELLIGENCE ENGINE")
    print("=" * 65)

    print()
    print("Loading anomaly results...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"""
Input file not found:

{INPUT_FILE}

Please run:

python backend/anomaly_engine.py

before running the risk engine.
"""
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
# COLUMN VALIDATION
# ============================================================

def validate_columns(df):

    print()
    print("=" * 65)
    print("VALIDATING INPUT DATA")
    print("=" * 65)

    required_columns = [
        "record_type",
        "ml_eligible",
        "amount_numeric",
        "amount_percentile",
        "isolation_score",
        "statistical_score",
        "metadata_risk",
        "anomaly_score",
        "risk_level",
        "ai_explanation"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("Missing required columns:")

        for column in missing_columns:
            print(
                " -",
                column
            )

        raise ValueError(
            "\nRisk Engine cannot continue because "
            "required columns are missing."
        )

    print(
        "Input validation successful."
    )

    return df


# ============================================================
# NUMERIC CLEANING
# ============================================================

def prepare_numeric_columns(df):

    print()
    print("=" * 65)
    print("PREPARING RISK SIGNALS")
    print("=" * 65)

    numeric_columns = [
        "amount_numeric",
        "amount_percentile",
        "isolation_score",
        "statistical_score",
        "metadata_risk",
        "anomaly_score"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Ensure boolean ML eligibility
    # --------------------------------------------------------

    if df["ml_eligible"].dtype != bool:

        df["ml_eligible"] = (
            df["ml_eligible"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                }
            )
            .fillna(False)
        )

    return df


# ============================================================
# FINANCIAL RISK
# ============================================================

def calculate_financial_risk(df):

    print(
        "Calculating financial risk..."
    )

    percentile = (
        df["amount_percentile"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Financial risk based on peer percentile.
    #
    # This does NOT mean a high amount is fraudulent.
    #
    # It means:
    #
    # "This amount is unusual relative to the dataset."
    # --------------------------------------------------------

    df["financial_risk"] = np.select(

        [
            percentile >= 99,
            percentile >= 95,
            percentile >= 90,
            percentile >= 75
        ],

        [
            100,
            75,
            50,
            25
        ],

        default=0
    )

    return df


# ============================================================
# MACHINE LEARNING RISK
# ============================================================

def calculate_ml_risk(df):

    print(
        "Calculating machine-learning risk..."
    )

    # Isolation Forest score is already normalized
    # to 0-100 by anomaly_engine.py.

    df["ml_risk"] = (
        df["isolation_score"]
        .fillna(0)
        .clip(
            0,
            100
        )
    )

    return df


# ============================================================
# STATISTICAL RISK
# ============================================================

def calculate_statistical_risk(df):

    print(
        "Calculating statistical risk..."
    )

    df["statistical_risk"] = (
        df["statistical_score"]
        .fillna(0)
        .clip(
            0,
            100
        )
    )

    return df


# ============================================================
# DATA QUALITY RISK
# ============================================================

def calculate_quality_risk(df):

    print(
        "Calculating data-quality risk..."
    )

    quality_risk = (
        df["metadata_risk"]
        .fillna(0)
        .clip(
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Existing data quality alert
    # --------------------------------------------------------

    if "data_quality_alert" in df.columns:

        quality_alert = (
            df["data_quality_alert"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        )

        quality_risk = np.maximum(
            quality_risk,
            np.where(
                quality_alert,
                50,
                0
            )
        )

    df["quality_risk"] = (
        quality_risk
        .clip(
            0,
            100
        )
    )

    return df


# ============================================================
# RISK FACTOR DETECTION
# ============================================================

def generate_risk_factors(df):

    print(
        "Generating risk factors..."
    )

    factors = []

    for _, row in df.iterrows():

        current_factors = []

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        if row["record_type"] == "AGGREGATE":

            current_factors.append(
                "Aggregate record"
            )

            factors.append(
                current_factors
            )

            continue

        # ----------------------------------------------------
        # Incomplete
        # ----------------------------------------------------

        if row["record_type"] == "INCOMPLETE":

            current_factors.append(
                "Incomplete record"
            )

            factors.append(
                current_factors
            )

            continue

        # ----------------------------------------------------
        # High financial percentile
        # ----------------------------------------------------

        percentile = row[
            "amount_percentile"
        ]

        if (
            pd.notna(percentile)
            and percentile >= 99
        ):

            current_factors.append(
                "Allocation is in the top 1% "
                "of observed records"
            )

        elif (
            pd.notna(percentile)
            and percentile >= 95
        ):

            current_factors.append(
                "Allocation is in the top 5% "
                "of observed records"
            )

        elif (
            pd.notna(percentile)
            and percentile >= 90
        ):

            current_factors.append(
                "Allocation is in the top 10% "
                "of observed records"
            )

        # ----------------------------------------------------
        # Machine learning
        # ----------------------------------------------------

        ml_score = row[
            "ml_risk"
        ]

        if ml_score >= 80:

            current_factors.append(
                "Isolation Forest detected "
                "a strong unusual pattern"
            )

        elif ml_score >= 60:

            current_factors.append(
                "Machine-learning model detected "
                "an unusual pattern"
            )

        # ----------------------------------------------------
        # Statistical signal
        # ----------------------------------------------------

        statistical = row[
            "statistical_risk"
        ]

        if statistical >= 80:

            current_factors.append(
                "Strong statistical deviation detected"
            )

        elif statistical >= 60:

            current_factors.append(
                "Significant statistical deviation detected"
            )

        # ----------------------------------------------------
        # Data quality
        # ----------------------------------------------------

        quality = row[
            "quality_risk"
        ]

        if quality >= 50:

            current_factors.append(
                "Data-quality validation requires review"
            )

        elif quality > 0:

            current_factors.append(
                "Minor metadata-quality concern detected"
            )

        # ----------------------------------------------------
        # Existing AI explanation
        # ----------------------------------------------------

        if (
            not current_factors
            and
            pd.notna(
                row.get(
                    "ai_explanation"
                )
            )
        ):

            explanation = str(
                row[
                    "ai_explanation"
                ]
            )

            if (
                explanation
                != "No strong anomaly signal detected"
            ):

                current_factors.append(
                    explanation
                )

        # ----------------------------------------------------
        # Default
        # ----------------------------------------------------

        if not current_factors:

            current_factors.append(
                "No significant risk factor detected"
            )

        factors.append(
            current_factors
        )

    df["risk_factors"] = factors

    # Convert list to readable string

    df["risk_factors_text"] = df[
        "risk_factors"
    ].apply(
        lambda x:
        " | ".join(x)
    )

    return df


# ============================================================
# FINAL RISK SCORE
# ============================================================

def calculate_final_risk_score(df):

    print()
    print("=" * 65)
    print("CALCULATING FINAL RISK SCORE")
    print("=" * 65)

    # --------------------------------------------------------
    # Risk engine weighting
    #
    # Existing AI anomaly score      50%
    # Financial unusualness          20%
    # ML signal                      15%
    # Statistical signal             10%
    # Data quality                    5%
    #
    # This is a prioritization score,
    # NOT probability of fraud.
    # --------------------------------------------------------

    df["risk_score"] = (

        0.50
        * df["anomaly_score"].fillna(0)

        +

        0.20
        * df["financial_risk"].fillna(0)

        +

        0.15
        * df["ml_risk"].fillna(0)

        +

        0.10
        * df["statistical_risk"].fillna(0)

        +

        0.05
        * df["quality_risk"].fillna(0)
    )

    df["risk_score"] = (
        df["risk_score"]
        .clip(
            0,
            100
        )
        .round(2)
    )

    # --------------------------------------------------------
    # Non-MP records don't receive a risk score.
    # --------------------------------------------------------

    df.loc[
        ~df["ml_eligible"],
        "risk_score"
    ] = np.nan

    return df


# ============================================================
# RISK LEVEL
# ============================================================

def classify_final_risk(df):

    print(
        "Classifying final risk..."
    )

    def classify(row):

        if not row["ml_eligible"]:

            return "NOT_APPLICABLE"

        score = row["risk_score"]

        if pd.isna(score):

            return "NOT_APPLICABLE"

        if score >= CRITICAL_THRESHOLD:

            return "CRITICAL"

        if score >= HIGH_THRESHOLD:

            return "HIGH"

        if score >= MEDIUM_THRESHOLD:

            return "MEDIUM"

        return "LOW"

    df["final_risk_level"] = df.apply(
        classify,
        axis=1
    )

    return df


# ============================================================
# INVESTIGATION PRIORITY
# ============================================================

def calculate_investigation_priority(df):

    print(
        "Calculating investigation priority..."
    )

    def priority(row):

        if not row["ml_eligible"]:

            return "NOT_APPLICABLE"

        score = row["risk_score"]

        level = row[
            "final_risk_level"
        ]

        if level == "CRITICAL":

            return "URGENT"

        if level == "HIGH":

            return "HIGH"

        if (
            level == "MEDIUM"
            and
            score >= MEDIUM_REVIEW_THRESHOLD
        ):

            return "WATCHLIST"

        if level == "MEDIUM":

            return "MONITOR"

        return "ROUTINE"

    df["investigation_priority"] = df.apply(
        priority,
        axis=1
    )

    return df


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def generate_recommended_actions(df):

    print(
        "Generating recommended actions..."
    )

    def action(row):

        level = row[
            "final_risk_level"
        ]

        priority = row[
            "investigation_priority"
        ]

        if level == "CRITICAL":

            return (
                "Immediate verification of "
                "allocation, sanction and supporting "
                "financial records"
            )

        if level == "HIGH":

            return (
                "Detailed verification of allocation "
                "and related project documentation"
            )

        if (
            level == "MEDIUM"
            and
            priority == "WATCHLIST"
        ):

            return (
                "Review allocation and compare with "
                "similar MPs/constituencies"
            )

        if level == "MEDIUM":

            return (
                "Monitor and perform periodic "
                "data/document verification"
            )

        if level == "LOW":

            return (
                "No immediate action; continue "
                "routine monitoring"
            )

        return (
            "Record requires data-quality review "
            "before risk assessment"
        )

    df["recommended_action"] = df.apply(
        action,
        axis=1
    )

    return df


# ============================================================
# ALERT STATUS
# ============================================================

def generate_alert_status(df):

    print(
        "Generating alert status..."
    )

    def alert(row):

        level = row[
            "final_risk_level"
        ]

        priority = row[
            "investigation_priority"
        ]

        if level == "CRITICAL":

            return "URGENT_ALERT"

        if level == "HIGH":

            return "INVESTIGATION_ALERT"

        if priority == "WATCHLIST":

            return "WATCHLIST_ALERT"

        if priority == "MONITOR":

            return "MONITOR_ALERT"

        if level == "LOW":

            return "NO_ALERT"

        return "DATA_REVIEW"

    df["alert_status"] = df.apply(
        alert,
        axis=1
    )

    return df


# ============================================================
# EXPLAINABLE ALERT
# ============================================================

def generate_alert_message(df):

    print(
        "Generating explainable alerts..."
    )

    messages = []

    for _, row in df.iterrows():

        if not row["ml_eligible"]:

            messages.append(
                "Not eligible for MP-level risk assessment"
            )

            continue

        mp = str(
            row.get(
                "mp_clean",
                "Unknown MP"
            )
        )

        constituency = str(
            row.get(
                "constituency_clean",
                "Unknown constituency"
            )
        )

        score = row[
            "risk_score"
        ]

        level = row[
            "final_risk_level"
        ]

        factors = row[
            "risk_factors_text"
        ]

        action = row[
            "recommended_action"
        ]

        message = (
            f"{level} RISK | "
            f"Score {score}/100 | "
            f"MP: {mp} | "
            f"Constituency: {constituency} | "
            f"Factors: {factors} | "
            f"Recommended action: {action}"
        )

        messages.append(
            message
        )

    df["alert_message"] = messages

    return df


# ============================================================
# RANK RECORDS
# ============================================================

def rank_risk_records(df):

    print(
        "Ranking risk records..."
    )

    df["risk_rank"] = np.nan

    eligible_mask = (
        df["ml_eligible"]
        &
        df["risk_score"].notna()
    )

    eligible = df.loc[
        eligible_mask
    ].sort_values(
        "risk_score",
        ascending=False
    )

    df.loc[
        eligible.index,
        "risk_rank"
    ] = range(
        1,
        len(eligible) + 1
    )

    return df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    print()
    print("=" * 65)
    print("SAVING RISK RESULTS")
    print("=" * 65)

    # --------------------------------------------------------
    # Sort highest-risk records first.
    # --------------------------------------------------------

    df = df.sort_values(
        "risk_score",
        ascending=False,
        na_position="last"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Risk results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    return df


# ============================================================
# RISK DISTRIBUTION
# ============================================================

def display_risk_distribution(df):

    print()
    print("=" * 65)
    print("FINAL RISK DISTRIBUTION")
    print("=" * 65)

    counts = (
        df[
            "final_risk_level"
        ]
        .value_counts()
    )

    levels = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
        "NOT_APPLICABLE"
    ]

    for level in levels:

        print(
            f"{level:15s}: "
            f"{int(counts.get(level, 0))}"
        )

    print()

    priority_counts = (
        df[
            "investigation_priority"
        ]
        .value_counts()
    )

    print(
        "INVESTIGATION PRIORITY"
    )

    print(
        "-" * 35
    )

    priorities = [
        "URGENT",
        "HIGH",
        "WATCHLIST",
        "MONITOR",
        "ROUTINE",
        "NOT_APPLICABLE"
    ]

    for priority in priorities:

        print(
            f"{priority:15s}: "
            f"{int(priority_counts.get(priority, 0))}"
        )


# ============================================================
# TOP RISK RECORDS
# ============================================================

def display_top_risk_records(df):

    print()
    print("=" * 65)
    print("TOP 10 RISK RECORDS")
    print("=" * 65)

    top = (
        df[
            df["ml_eligible"]
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    if len(top) == 0:

        print(
            "No eligible MP records found."
        )

        return

    columns = [
        "risk_rank",
        "Sr. No.",
        "state_clean",
        "mp_clean",
        "constituency_clean",
        "amount_numeric",
        "amount_percentile",
        "anomaly_score",
        "risk_score",
        "final_risk_level",
        "investigation_priority",
        "alert_status"
    ]

    columns = [
        column
        for column in columns
        if column in top.columns
    ]

    print()

    print(
        top[
            columns
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# DISPLAY ALERTS
# ============================================================

def display_top_alerts(df):

    print()
    print("=" * 65)
    print("TOP INVESTIGATION ALERTS")
    print("=" * 65)

    top = (
        df[
            df["ml_eligible"]
            &
            (
                df["investigation_priority"]
                .isin(
                    [
                        "URGENT",
                        "HIGH",
                        "WATCHLIST"
                    ]
                )
            )
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(5)
    )

    if len(top) == 0:

        print()
        print(
            "No high-priority investigation alerts."
        )

        print(
            "Medium-risk records will remain available "
            "for monitoring."
        )

        return

    for _, row in top.iterrows():

        print()
        print(
            "-" * 65
        )

        print(
            f"Risk Rank       : "
            f"{int(row['risk_rank'])}"
        )

        print(
            f"MP              : "
            f"{row['mp_clean']}"
        )

        print(
            f"Constituency    : "
            f"{row['constituency_clean']}"
        )

        print(
            f"State           : "
            f"{row['state_clean']}"
        )

        print(
            f"Allocation      : "
            f"₹{row['amount_numeric']:,.2f}"
        )

        print(
            f"Risk Score      : "
            f"{row['risk_score']}/100"
        )

        print(
            f"Risk Level      : "
            f"{row['final_risk_level']}"
        )

        print(
            f"Priority        : "
            f"{row['investigation_priority']}"
        )

        print(
            "Risk Factors    :"
        )

        print(
            row["risk_factors_text"]
        )

        print(
            "Recommended     :"
        )

        print(
            row["recommended_action"]
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    df = validate_columns(
        df
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    df = prepare_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    df = calculate_financial_risk(
        df
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    df = calculate_ml_risk(
        df
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    df = calculate_statistical_risk(
        df
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    df = calculate_quality_risk(
        df
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    df = generate_risk_factors(
        df
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    df = calculate_final_risk_score(
        df
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    df = classify_final_risk(
        df
    )

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    df = calculate_investigation_priority(
        df
    )

    # --------------------------------------------------------
    # STEP 12
    # --------------------------------------------------------

    df = generate_recommended_actions(
        df
    )

    # --------------------------------------------------------
    # STEP 13
    # --------------------------------------------------------

    df = generate_alert_status(
        df
    )

    # --------------------------------------------------------
    # STEP 14
    # --------------------------------------------------------

    df = generate_alert_message(
        df
    )

    # --------------------------------------------------------
    # STEP 15
    # --------------------------------------------------------

    df = rank_risk_records(
        df
    )

    # --------------------------------------------------------
    # STEP 16
    # --------------------------------------------------------

    df = save_results(
        df
    )

    # --------------------------------------------------------
    # STEP 17
    # --------------------------------------------------------

    display_risk_distribution(
        df
    )

    # --------------------------------------------------------
    # STEP 18
    # --------------------------------------------------------

    display_top_risk_records(
        df
    )

    # --------------------------------------------------------
    # STEP 19
    # --------------------------------------------------------

    display_top_alerts(
        df
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RISK ENGINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()