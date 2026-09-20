"""
MPLADS AI - ROBUST STATISTICAL ANOMALY ENGINE
Phase 1.7

Input:
    processed/allocation_anomaly_results.csv

Output:
    processed/statistical_anomaly_results.csv

Purpose:
    Detect unusual allocation values using:
        - Log transformation
        - Winsorization
        - MAD-based robust statistics
        - IQR outlier detection
        - Percentile analysis

Note:
    A statistical anomaly is an investigation signal.
    It is NOT proof of fraud.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
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
    "statistical_anomaly_results.csv"
)

LOWER_WINSOR_PERCENTILE = 1.0
UPPER_WINSOR_PERCENTILE = 99.0

MIN_SCALE = 1e-6

MAX_ROBUST_Z = 6.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_mp_name_column(dataframe):

    possible_columns = [
        "Hon'ble Members of Parliaments",
        "Hon'ble Members of Parliament",
        "MP Name",
        "mp_clean",
        "MP",
        "Member Name"
    ]

    for column in possible_columns:

        if column in dataframe.columns:
            return column

    return None


def find_state_column(dataframe):

    possible_columns = [
        "state_clean",
        "State",
        "state",
        "STATE"
    ]

    for column in possible_columns:

        if column in dataframe.columns:
            return column

    return None


def calculate_robust_z(values, center, scale):

    if scale is None:
        return np.zeros(len(values))

    if not np.isfinite(scale):
        return np.zeros(len(values))

    if scale < MIN_SCALE:
        return np.zeros(len(values))

    z = (
        values - center
    ) / scale

    z = np.clip(
        z,
        -MAX_ROBUST_Z,
        MAX_ROBUST_Z
    )

    return z


def calculate_statistical_score(
    robust_z,
    percentile,
    iqr_outlier
):

    # --------------------------------------------------------
    # PRIMARY ROBUST-Z SIGNAL
    # --------------------------------------------------------

    if robust_z >= 4.0:

        score = 100.0

    elif robust_z >= 3.0:

        score = 90.0

    elif robust_z >= 2.5:

        score = 75.0

    elif robust_z >= 2.0:

        score = 60.0

    elif robust_z >= 1.5:

        score = 40.0

    elif robust_z >= 1.0:

        score = 20.0

    else:

        score = 0.0


    # --------------------------------------------------------
    # PERCENTILE SUPPORT
    # --------------------------------------------------------

    if percentile >= 99.0:

        score = max(
            score,
            60.0
        )

    elif percentile >= 95.0:

        score = max(
            score,
            40.0
        )

    elif percentile >= 90.0:

        score = max(
            score,
            20.0
        )


    # --------------------------------------------------------
    # IQR SUPPORT
    # --------------------------------------------------------

    if iqr_outlier:

        score = max(
            score,
            60.0
        )


    return min(
        float(score),
        100.0
    )


def get_alert_level(score):

    if score >= 80:

        return "HIGH"

    elif score >= 60:

        return "MEDIUM"

    elif score >= 30:

        return "WATCH"

    return "NORMAL"


def get_deviation_category(
    robust_z,
    percentile,
    iqr_outlier
):

    if robust_z >= 4.0:

        return "EXTREME_DEVIATION"

    elif robust_z >= 3.0:

        return "VERY_HIGH_DEVIATION"

    elif robust_z >= 2.0:

        return "HIGH_DEVIATION"

    elif robust_z >= 1.5:

        return "MODERATE_DEVIATION"

    elif iqr_outlier:

        return "IQR_OUTLIER"

    elif percentile >= 99:

        return "TOP_1_PERCENT"

    elif percentile >= 95:

        return "TOP_5_PERCENT"

    elif percentile >= 90:

        return "TOP_10_PERCENT"

    return "NORMAL"


def generate_explanation(
    amount,
    robust_z,
    percentile,
    iqr_outlier
):

    reasons = []

    # --------------------------------------------------------
    # ROBUST DEVIATION
    # --------------------------------------------------------

    if robust_z >= 4.0:

        reasons.append(
            "extreme statistical deviation "
            f"(robust z-score {robust_z:.2f})"
        )

    elif robust_z >= 3.0:

        reasons.append(
            "very high statistical deviation "
            f"(robust z-score {robust_z:.2f})"
        )

    elif robust_z >= 2.0:

        reasons.append(
            "high statistical deviation "
            f"(robust z-score {robust_z:.2f})"
        )

    elif robust_z >= 1.5:

        reasons.append(
            "moderate statistical deviation "
            f"(robust z-score {robust_z:.2f})"
        )

    # --------------------------------------------------------
    # PERCENTILE
    # --------------------------------------------------------

    if percentile >= 99:

        reasons.append(
            "allocation is within the top 1% "
            "of observed values"
        )

    elif percentile >= 95:

        reasons.append(
            "allocation is within the top 5% "
            "of observed values"
        )

    elif percentile >= 90:

        reasons.append(
            "allocation is within the top 10% "
            "of observed values"
        )

    # --------------------------------------------------------
    # IQR
    # --------------------------------------------------------

    if iqr_outlier:

        reasons.append(
            "allocation exceeds the IQR upper fence"
        )

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if len(reasons) == 0:

        return (
            "Allocation is within the normal "
            "statistical range of the dataset."
        )

    explanation = (
        f"Allocation of ₹{amount:,.2f} was flagged because "
        + "; ".join(reasons)
        + ". This is an anomaly candidate requiring "
          "verification and does not by itself indicate fraud."
    )

    return explanation


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MPLADS AI ROBUST STATISTICAL ANOMALY ENGINE")
    print("=" * 70)
    print()

    # ========================================================
    # LOAD INPUT
    # ========================================================

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            "\nInput file not found:\n"
            + INPUT_FILE
        )

    print(
        "Loading allocation anomaly results..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Records loaded: {len(df)}"
    )

    print(
        f"Columns available: {len(df.columns)}"
    )

    print()

    # ========================================================
    # VALIDATION
    # ========================================================

    required_columns = [
        "record_type",
        "ml_eligible",
        "amount_numeric"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    print(
        "Input validation successful."
    )

    print()

    # ========================================================
    # CLEAN AMOUNT
    # ========================================================

    df["amount_numeric"] = pd.to_numeric(
        df["amount_numeric"],
        errors="coerce"
    )

    # ========================================================
    # CLEAN ML FLAG
    # ========================================================

    df["ml_eligible_clean"] = (
        df["ml_eligible"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(
            {
                "TRUE": True,
                "FALSE": False,
                "YES": True,
                "NO": False,
                "1": True,
                "0": False
            }
        )
    )

    # ========================================================
    # INITIALIZE OUTPUT COLUMNS
    # ========================================================

    df["log_amount"] = np.nan

    df["robust_z_score"] = np.nan

    df["standard_z_score"] = np.nan

    df["global_percentile"] = np.nan

    df["iqr_upper_fence"] = np.nan

    df["iqr_outlier"] = pd.Series(
        False,
        index=df.index,
        dtype="bool"
    )

    df["statistical_score"] = np.nan

    df["statistical_alert_level"] = pd.Series(
        "NOT_APPLICABLE",
        index=df.index,
        dtype="object"
    )

    df["deviation_category"] = pd.Series(
        "NOT_APPLICABLE",
        index=df.index,
        dtype="object"
    )

    df["statistical_explanation"] = pd.Series(
        "",
        index=df.index,
        dtype="object"
    )

    # ========================================================
    # ELIGIBLE RECORDS
    # ========================================================

    eligible_mask = (
        (
            df["record_type"]
            .astype(str)
            .str.upper()
            == "MP_RECORD"
        )
        &
        (
            df["ml_eligible_clean"]
            == True
        )
        &
        (
            df["amount_numeric"].notna()
        )
        &
        (
            df["amount_numeric"] > 0
        )
    )

    analysis_df = df.loc[
        eligible_mask
    ].copy()

    print(
        "Records eligible for statistical analysis: "
        f"{len(analysis_df)}"
    )

    print()

    if len(analysis_df) < 10:

        raise ValueError(
            "Insufficient valid records for "
            "statistical analysis."
        )

    # ========================================================
    # LOG TRANSFORMATION
    # ========================================================

    print(
        "Applying log transformation..."
    )

    analysis_df["log_amount"] = np.log1p(
        analysis_df["amount_numeric"]
    )

    raw_log = (
        analysis_df["log_amount"]
        .to_numpy()
    )

    # ========================================================
    # WINSORIZATION
    # ========================================================

    print(
        "Applying winsorization..."
    )

    lower_bound = np.percentile(
        raw_log,
        LOWER_WINSOR_PERCENTILE
    )

    upper_bound = np.percentile(
        raw_log,
        UPPER_WINSOR_PERCENTILE
    )

    winsorized_log = np.clip(
        raw_log,
        lower_bound,
        upper_bound
    )

    # ========================================================
    # ROBUST CENTER
    # ========================================================

    median_log = np.median(
        winsorized_log
    )

    # ========================================================
    # MAD
    # ========================================================

    absolute_deviation = np.abs(
        winsorized_log
        -
        median_log
    )

    mad = np.median(
        absolute_deviation
    )

    mad_scale = (
        mad * 1.4826
    )

    # ========================================================
    # IQR SCALE
    # ========================================================

    q1_log = np.percentile(
        winsorized_log,
        25
    )

    q3_log = np.percentile(
        winsorized_log,
        75
    )

    iqr_log = (
        q3_log
        -
        q1_log
    )

    iqr_scale = (
        iqr_log / 1.349
    )

    # ========================================================
    # STANDARD DEVIATION
    # ========================================================

    std_log = np.std(
        winsorized_log,
        ddof=1
    )

    # ========================================================
    # SELECT SCALE
    # ========================================================

    if (
        np.isfinite(mad_scale)
        and mad_scale >= MIN_SCALE
    ):

        robust_scale = mad_scale

        scale_method = "MAD"

    elif (
        np.isfinite(iqr_scale)
        and iqr_scale >= MIN_SCALE
    ):

        robust_scale = iqr_scale

        scale_method = "IQR"

    elif (
        np.isfinite(std_log)
        and std_log >= MIN_SCALE
    ):

        robust_scale = std_log

        scale_method = "STANDARD_DEVIATION"

    else:

        robust_scale = None

        scale_method = "PERCENTILE_ONLY"

    print(
        f"Robust scale method: {scale_method}"
    )

    # ========================================================
    # ROBUST Z SCORE
    # ========================================================

    if robust_scale is not None:

        robust_z = calculate_robust_z(
            winsorized_log,
            median_log,
            robust_scale
        )

    else:

        robust_z = np.zeros(
            len(analysis_df)
        )

    # ========================================================
    # STANDARD Z SCORE
    # ========================================================

    if (
        np.isfinite(std_log)
        and std_log >= MIN_SCALE
    ):

        standard_z = (
            winsorized_log
            -
            np.mean(winsorized_log)
        ) / std_log

        standard_z = np.clip(
            standard_z,
            -MAX_ROBUST_Z,
            MAX_ROBUST_Z
        )

    else:

        standard_z = np.zeros(
            len(analysis_df)
        )

    # ========================================================
    # GLOBAL PERCENTILE
    # ========================================================

    print(
        "Calculating global percentile..."
    )

    percentile_series = (
        analysis_df["amount_numeric"]
        .rank(
            method="average",
            pct=True
        )
        * 100
    )

    global_percentile = (
        percentile_series
        .to_numpy()
    )

    # ========================================================
    # IQR OUTLIER
    # ========================================================

    print(
        "Calculating IQR outlier signal..."
    )

    q1_amount = (
        analysis_df["amount_numeric"]
        .quantile(0.25)
    )

    q3_amount = (
        analysis_df["amount_numeric"]
        .quantile(0.75)
    )

    iqr_amount = (
        q3_amount
        -
        q1_amount
    )

    if (
        np.isfinite(iqr_amount)
        and iqr_amount > 0
    ):

        iqr_upper_fence = (
            q3_amount
            +
            1.5 * iqr_amount
        )

        iqr_outlier = (
            analysis_df["amount_numeric"]
            >
            iqr_upper_fence
        )

    else:

        iqr_upper_fence = np.nan

        iqr_outlier = (
            percentile_series >= 99
        )

    # ========================================================
    # SCORE EACH RECORD
    # ========================================================

    print(
        "Calculating statistical anomaly scores..."
    )

    statistical_scores = []

    statistical_alerts = []

    deviation_categories = []

    explanations = []

    amounts = (
        analysis_df["amount_numeric"]
        .to_numpy()
    )

    iqr_flags = np.asarray(
        iqr_outlier,
        dtype=bool
    )

    for i in range(
        len(analysis_df)
    ):

        amount = float(
            amounts[i]
        )

        z = float(
            robust_z[i]
        )

        percentile = float(
            global_percentile[i]
        )

        is_iqr_outlier = bool(
            iqr_flags[i]
        )

        score = calculate_statistical_score(
            robust_z=z,
            percentile=percentile,
            iqr_outlier=is_iqr_outlier
        )

        alert = get_alert_level(
            score
        )

        category = get_deviation_category(
            robust_z=z,
            percentile=percentile,
            iqr_outlier=is_iqr_outlier
        )

        explanation = generate_explanation(
            amount=amount,
            robust_z=z,
            percentile=percentile,
            iqr_outlier=is_iqr_outlier
        )

        statistical_scores.append(
            score
        )

        statistical_alerts.append(
            alert
        )

        deviation_categories.append(
            category
        )

        explanations.append(
            explanation
        )

    # ========================================================
    # WRITE RESULTS
    # ========================================================

    analysis_indices = (
        analysis_df.index
    )

    df.loc[
        analysis_indices,
        "log_amount"
    ] = analysis_df[
        "log_amount"
    ].to_numpy()

    df.loc[
        analysis_indices,
        "robust_z_score"
    ] = robust_z

    df.loc[
        analysis_indices,
        "standard_z_score"
    ] = standard_z

    df.loc[
        analysis_indices,
        "global_percentile"
    ] = global_percentile

    df.loc[
        analysis_indices,
        "iqr_upper_fence"
    ] = iqr_upper_fence

    df.loc[
        analysis_indices,
        "iqr_outlier"
    ] = iqr_flags

    df.loc[
        analysis_indices,
        "statistical_score"
    ] = statistical_scores

    df.loc[
        analysis_indices,
        "statistical_alert_level"
    ] = statistical_alerts

    df.loc[
        analysis_indices,
        "deviation_category"
    ] = deviation_categories

    df.loc[
        analysis_indices,
        "statistical_explanation"
    ] = explanations

    # ========================================================
    # REMOVE TEMP COLUMN
    # ========================================================

    df.drop(
        columns=[
            "ml_eligible_clean"
        ],
        inplace=True
    )

    # ========================================================
    # ROUND NUMBERS
    # ========================================================

    numeric_columns = [
        "log_amount",
        "robust_z_score",
        "standard_z_score",
        "global_percentile",
        "iqr_upper_fence",
        "statistical_score"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).round(4)

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Records analysed       : "
        f"{len(analysis_df)}"
    )

    print(
        f"Median log amount      : "
        f"{median_log:.6f}"
    )

    print(
        f"Q1 log amount          : "
        f"{q1_log:.6f}"
    )

    print(
        f"Q3 log amount          : "
        f"{q3_log:.6f}"
    )

    print(
        f"Log IQR                : "
        f"{iqr_log:.6f}"
    )

    print(
        f"MAD                    : "
        f"{mad:.6f}"
    )

    if robust_scale is not None:

        print(
            f"Robust scale           : "
            f"{robust_scale:.6f}"
        )

    else:

        print(
            "Robust scale           : unavailable"
        )

    print(
        f"Scale method           : "
        f"{scale_method}"
    )

    print(
        f"Amount Q1              : "
        f"₹{q1_amount:,.2f}"
    )

    print(
        f"Amount Q3              : "
        f"₹{q3_amount:,.2f}"
    )

    if np.isfinite(
        iqr_upper_fence
    ):

        print(
            f"IQR upper fence        : "
            f"₹{iqr_upper_fence:,.2f}"
        )

    else:

        print(
            "IQR upper fence        : unavailable"
        )

    # ========================================================
    # ALERT DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print("STATISTICAL ALERT DISTRIBUTION")
    print("=" * 70)

    alert_distribution = (
        df.loc[
            analysis_indices,
            "statistical_alert_level"
        ]
        .value_counts()
    )

    for level in [
        "NORMAL",
        "WATCH",
        "MEDIUM",
        "HIGH"
    ]:

        count = int(
            alert_distribution.get(
                level,
                0
            )
        )

        print(
            f"{level:<10}: {count}"
        )

    # ========================================================
    # FIND MP NAME COLUMN
    # ========================================================

    mp_name_column = find_mp_name_column(
        df
    )

    state_column = find_state_column(
        df
    )

    # ========================================================
    # TOP 10
    # ========================================================

    print()
    print("=" * 70)
    print("TOP 10 STATISTICAL ANOMALY CANDIDATES")
    print("=" * 70)

    top_records = (
        df.loc[
            analysis_indices
        ]
        .sort_values(
            by="statistical_score",
            ascending=False
        )
        .head(10)
    )

    for _, row in top_records.iterrows():

        # ----------------------------------------------------
        # MP NAME
        # ----------------------------------------------------

        if mp_name_column is not None:

            mp_name = row[
                mp_name_column
            ]

        else:

            mp_name = "Unknown"

        # ----------------------------------------------------
        # SERIAL NUMBER
        # ----------------------------------------------------

        serial_number = row.get(
            "Sr. No.",
            "N/A"
        )

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        if state_column is not None:

            state_name = row[
                state_column
            ]

        else:

            state_name = "N/A"

        print()

        print(
            f"Sr {serial_number} "
            f"{mp_name}"
        )

        print(
            f"State: {state_name}"
        )

        print(
            f"Amount: "
            f"₹{row['amount_numeric']:,.2f}"
        )

        print(
            f"Robust Z-Score: "
            f"{row['robust_z_score']:.2f}"
        )

        print(
            f"Percentile: "
            f"{row['global_percentile']:.2f}"
        )

        print(
            f"IQR Outlier: "
            f"{row['iqr_outlier']}"
        )

        print(
            f"Statistical Score: "
            f"{row['statistical_score']:.2f}"
        )

        print(
            f"Alert: "
            f"{row['statistical_alert_level']}"
        )

        print(
            f"Category: "
            f"{row['deviation_category']}"
        )

        print(
            "Explanation: "
            f"{row['statistical_explanation']}"
        )

        print(
            "-" * 70
        )

    # ========================================================
    # SAVE CONFIRMATION
    # ========================================================

    print()
    print("=" * 70)
    print("STATISTICAL ANOMALY ENGINE COMPLETE")
    print("=" * 70)

    print()

    print(
        "Output saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "Note: Statistical anomaly detection "
        "identifies records for verification. "
        "It does not establish fraud."
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()