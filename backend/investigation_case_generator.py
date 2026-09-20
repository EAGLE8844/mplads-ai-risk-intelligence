import os
import pandas as pd
import numpy as np


# ============================================================
# MPLADS AI INVESTIGATION CASE GENERATOR
# Phase 1.9
# ============================================================
#
# Purpose:
# Convert unified risk results into explainable investigation
# cases for auditors / monitoring authorities.
#
# IMPORTANT:
# This system identifies anomaly candidates and verification
# priorities. It does NOT declare fraud.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "unified_risk_results.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "processed",
    "investigation_cases.csv"
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("MPLADS AI INVESTIGATION CASE GENERATOR")
print("PHASE 1.9")
print("=" * 70)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):

    print("\nERROR: Input file not found.")

    print(
        f"\nExpected file:\n{INPUT_FILE}"
    )

    print(
        "\nPlease run the unified risk engine first."
    )

    raise SystemExit(1)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(
    f"\nRecords loaded: {len(df)}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    # Identification
    "Sr. No.",
    "State",
    "Hon'ble Members of Parliaments",
    "Constituency",
    "Allocated AMOUNT ( ₹ )",

    # ML
    "isolation_score",

    # Peer intelligence
    "peer_risk",
    "peer_alert",
    "peer_confidence",
    "peer_ratio",
    "peer_deviation_pct",
    "peer_median",
    "peer_group",
    "peer_count",

    # Statistical intelligence
    "statistical_risk",
    "statistical_alert",
    "robust_z_score",
    "statistical_percentile",

    # Financial
    "financial_risk",

    # Unified
    "unified_risk_score",
    "unified_risk_level",
    "investigation_priority"
]


# ============================================================
# COLUMN VALIDATION
# ============================================================

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print("\nERROR: Missing required columns:")

    for column in missing_columns:
        print(f" - {column}")

    print(
        "\nPlease verify that the unified risk engine "
        "has generated the expected output."
    )

    raise SystemExit(1)


print(
    "All required columns detected successfully."
)


# ============================================================
# STATISTICAL COMPONENT
# ============================================================
#
# IMPORTANT:
#
# The column `statistical_score` from the older anomaly engine
# may contain 0 even when the statistical detector has found
# a strong anomaly.
#
# The actual statistical intelligence in the unified output is
# represented by:
#
#   statistical_risk
#   statistical_alert
#   robust_z_score
#   statistical_percentile
#   adjusted_statistical_risk
#   statistical_component_score
#
# We therefore create one clean field:
#
#   effective_statistical_score
#
# which represents the statistical contribution used by the
# investigation layer.
# ============================================================


if "statistical_component_score" in df.columns:

    df["effective_statistical_score"] = pd.to_numeric(
        df["statistical_component_score"],
        errors="coerce"
    ).fillna(0)


elif "adjusted_statistical_risk" in df.columns:

    df["effective_statistical_score"] = pd.to_numeric(
        df["adjusted_statistical_risk"],
        errors="coerce"
    ).fillna(0)


else:

    df["effective_statistical_score"] = pd.to_numeric(
        df["statistical_risk"],
        errors="coerce"
    ).fillna(0)


# ============================================================
# NUMERIC CLEANING
# ============================================================

numeric_columns = [

    "Allocated AMOUNT ( ₹ )",

    "isolation_score",

    "peer_risk",
    "peer_ratio",
    "peer_deviation_pct",
    "peer_median",
    "peer_count",

    "statistical_risk",
    "robust_z_score",
    "statistical_percentile",
    "effective_statistical_score",

    "financial_risk",

    "unified_risk_score"
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# IDENTIFY INVESTIGATION CANDIDATES
# ============================================================
#
# HIGH / CRITICAL:
#   Always investigate.
#
# MEDIUM + WATCHLIST:
#   Investigation candidate.
#
# LOW:
#   Routine monitoring.
#
# This prevents every unusual record from becoming an alert.
# ============================================================

candidate_mask = (

    df["unified_risk_level"].isin(
        ["HIGH", "CRITICAL"]
    )

    |

    (
        (df["unified_risk_level"] == "MEDIUM")
        &
        (df["investigation_priority"] == "WATCHLIST")
    )

)


cases = df[
    candidate_mask
].copy()


# ============================================================
# SORT BY UNIFIED RISK
# ============================================================

cases = cases.sort_values(
    by="unified_risk_score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_signals(row):

    signals = []

    # --------------------------------------------------------
    # ML SIGNAL
    # --------------------------------------------------------

    if row["isolation_score"] >= 50:

        signals.append(
            "ML anomaly"
        )


    # --------------------------------------------------------
    # PEER SIGNAL
    # --------------------------------------------------------

    if row["peer_risk"] >= 40:

        signals.append(
            "Peer deviation"
        )


    # --------------------------------------------------------
    # FINANCIAL SIGNAL
    # --------------------------------------------------------

    if row["financial_risk"] >= 75:

        signals.append(
            "Financial exposure"
        )


    # --------------------------------------------------------
    # STATISTICAL SIGNAL
    # --------------------------------------------------------

    statistical_signal = (

        row["effective_statistical_score"] > 0

        or

        row["statistical_risk"] >= 75

        or

        str(
            row["statistical_alert"]
        ).upper()
        in ["HIGH", "MEDIUM"]

    )


    if statistical_signal:

        signals.append(
            "Statistical anomaly"
        )


    if not signals:

        return "No strong signal"


    return " + ".join(
        signals
    )


cases["risk_signals"] = cases.apply(
    detect_signals,
    axis=1
)


# ============================================================
# EVIDENCE STRENGTH
# ============================================================

def calculate_evidence_strength(row):

    signal_count = len(
        row["risk_signals"].split(" + ")
    )


    if signal_count >= 4:

        return "STRONG"


    if signal_count >= 3:

        return "STRONG"


    if signal_count == 2:

        return "MODERATE"


    return "LIMITED"


cases["evidence_strength"] = cases.apply(
    calculate_evidence_strength,
    axis=1
)


# ============================================================
# INVESTIGATION EXPLANATION
# ============================================================

def generate_explanation(row):

    explanations = []


    # --------------------------------------------------------
    # ML EXPLANATION
    # --------------------------------------------------------

    if row["isolation_score"] >= 50:

        explanations.append(

            f"ML anomaly score is "
            f"{row['isolation_score']:.2f}, "
            "indicating an unusual allocation pattern "
            "relative to the learned data distribution."

        )


    # --------------------------------------------------------
    # PEER EXPLANATION
    # --------------------------------------------------------

    if row["peer_risk"] >= 40:

        ratio = row["peer_ratio"]

        deviation = row["peer_deviation_pct"]

        if ratio > 0:

            explanations.append(

                f"Allocation is approximately "
                f"{ratio:.2f}x the peer-group median, "
                f"with a {deviation:.2f}% deviation."

            )


    # --------------------------------------------------------
    # FINANCIAL EXPLANATION
    # --------------------------------------------------------

    if row["financial_risk"] >= 75:

        amount = row[
            "Allocated AMOUNT ( ₹ )"
        ]

        explanations.append(

            f"Allocated amount is "
            f"₹{amount:,.2f}, "
            "placing the record in a high "
            "financial-exposure range."

        )


    # --------------------------------------------------------
    # STATISTICAL EXPLANATION
    # --------------------------------------------------------

    statistical_signal = (

        row["effective_statistical_score"] > 0

        or

        row["statistical_risk"] >= 75

        or

        str(
            row["statistical_alert"]
        ).upper()
        in ["HIGH", "MEDIUM"]

    )


    if statistical_signal:

        explanations.append(

            f"Statistical analysis flags the record "
            f"as {str(row['statistical_alert']).upper()} "
            f"with robust Z-score "
            f"{row['robust_z_score']:.2f} "
            f"and global percentile "
            f"{row['statistical_percentile']:.2f}."

        )


    if not explanations:

        return (
            "No single strong anomaly signal identified; "
            "record remains under automated monitoring."
        )


    return " ".join(
        explanations
    )


cases["investigation_explanation"] = cases.apply(
    generate_explanation,
    axis=1
)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def generate_recommended_action(row):

    priority = str(
        row["investigation_priority"]
    ).upper()


    if priority == "URGENT":

        return (

            "Prioritize immediate verification of "
            "allocation records, approvals, sanction "
            "details, supporting documents, and relevant "
            "implementation records."

        )


    if priority == "HIGH":

        return (

            "Prioritize verification of allocation records, "
            "supporting approvals, sanction details, and "
            "relevant implementation documentation."

        )


    if priority == "WATCHLIST":

        return (

            "Place on monitoring watchlist and verify "
            "supporting records if additional evidence "
            "emerges."

        )


    return (
        "Continue routine monitoring."
    )


cases["recommended_action"] = cases.apply(
    generate_recommended_action,
    axis=1
)


# ============================================================
# CASE STATUS
# ============================================================

cases["case_status"] = (
    "REQUIRES VERIFICATION"
)


# ============================================================
# CASE ID
# ============================================================

cases.insert(
    0,
    "case_id",
    [
        f"MPLADS-CASE-{number:04d}"
        for number
        in range(
            1,
            len(cases) + 1
        )
    ]
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

output_columns = [

    # Case identification
    "case_id",

    "Sr. No.",

    "State",

    "Hon'ble Members of Parliaments",

    "Constituency",

    "Allocated AMOUNT ( ₹ )",


    # Unified intelligence
    "unified_risk_score",

    "unified_risk_level",

    "investigation_priority",


    # Evidence
    "evidence_strength",

    "risk_signals",


    # ML intelligence
    "isolation_score",


    # Peer intelligence
    "peer_risk",

    "peer_alert",

    "peer_confidence",

    "peer_ratio",

    "peer_deviation_pct",

    "peer_median",

    "peer_group",

    "peer_count",


    # Statistical intelligence
    "statistical_risk",

    "statistical_alert",

    "robust_z_score",

    "statistical_percentile",

    "effective_statistical_score",


    # Financial intelligence
    "financial_risk",


    # Explainability
    "investigation_explanation",

    "recommended_action",

    "case_status"
]


cases = cases[
    output_columns
]


# ============================================================
# SAVE OUTPUT
# ============================================================

cases.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"\nInvestigation cases generated: "
    f"{len(cases)}"
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

print(
    "\nRisk distribution:"
)


if len(cases) > 0:

    print(
        cases[
            "unified_risk_level"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "No investigation cases."
    )


# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

print(
    "\nPriority distribution:"
)


if len(cases) > 0:

    print(
        cases[
            "investigation_priority"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "No priorities."
    )


# ============================================================
# SIGNAL COVERAGE
# ============================================================

print(
    "\nSignal coverage:"
)


ml_signal_count = (
    cases["isolation_score"] >= 50
).sum()


peer_signal_count = (
    cases["peer_risk"] >= 40
).sum()


financial_signal_count = (
    cases["financial_risk"] >= 75
).sum()


statistical_signal_count = (

    (
        cases[
            "effective_statistical_score"
        ] > 0
    )

    |

    (
        cases[
            "statistical_risk"
        ] >= 75
    )

    |

    (
        cases[
            "statistical_alert"
        ]
        .astype(str)
        .str.upper()
        .isin(
            ["HIGH", "MEDIUM"]
        )
    )

).sum()


print(
    f"ML signals          : "
    f"{ml_signal_count}"
)

print(
    f"Peer signals        : "
    f"{peer_signal_count}"
)

print(
    f"Financial signals   : "
    f"{financial_signal_count}"
)

print(
    f"Statistical signals : "
    f"{statistical_signal_count}"
)


# ============================================================
# TOP INVESTIGATION CASES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP INVESTIGATION CASES"
)

print(
    "=" * 70
)


for _, row in cases.head(10).iterrows():

    # Extract MP name separately.
    # This avoids the Windows/Python f-string apostrophe issue.
    mp_name = row[
        "Hon'ble Members of Parliaments"
    ]


    print(
        "\n" + "-" * 70
    )


    print(
        f"Case ID       : "
        f"{row['case_id']}"
    )


    print(
        f"MP            : "
        f"{mp_name}"
    )


    print(
        f"State         : "
        f"{row['State']}"
    )


    print(
        f"Constituency  : "
        f"{row['Constituency']}"
    )


    print(
        f"Amount        : ₹"
        f"{row['Allocated AMOUNT ( ₹ )']:,.2f}"
    )


    print(
        f"Unified Risk  : "
        f"{row['unified_risk_score']:.2f}"
    )


    print(
        f"Risk Level    : "
        f"{row['unified_risk_level']}"
    )


    print(
        f"Priority      : "
        f"{row['investigation_priority']}"
    )


    print(
        f"Signals       : "
        f"{row['risk_signals']}"
    )


    print(
        f"Evidence      : "
        f"{row['evidence_strength']}"
    )


    print(
        f"ML Score      : "
        f"{row['isolation_score']:.2f}"
    )


    print(
        f"Peer Risk     : "
        f"{row['peer_risk']:.2f}"
    )


    print(
        f"Peer Ratio    : "
        f"{row['peer_ratio']:.2f}x"
    )


    print(
        f"Peer Deviation: "
        f"{row['peer_deviation_pct']:.2f}%"
    )


    print(
        f"Peer Confidence: "
        f"{row['peer_confidence']}"
    )


    print(
        f"Statistical   : "
        f"{row['effective_statistical_score']:.2f}"
    )


    print(
        f"Stat Risk     : "
        f"{row['statistical_risk']:.2f}"
    )


    print(
        f"Stat Alert    : "
        f"{row['statistical_alert']}"
    )


    print(
        f"Robust Z      : "
        f"{row['robust_z_score']:.2f}"
    )


    print(
        f"Global Percentile: "
        f"{row['statistical_percentile']:.2f}"
    )


    print(
        f"Financial Risk: "
        f"{row['financial_risk']:.2f}"
    )


    print(
        f"Status        : "
        f"{row['case_status']}"
    )


# ============================================================
# COMPLETION
# ============================================================

print(
    "\n" + "=" * 70
)


print(
    "Investigation case generation "
    "completed successfully."
)


print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_FILE}"
)


print(
    "=" * 70
)