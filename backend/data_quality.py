from pathlib import Path
import re
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def clean_text(value):
    """Standardize text values."""

    if pd.isna(value):
        return None

    value = str(value)

    # Remove leading/trailing spaces
    value = value.strip()

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value)

    if value == "":
        return None

    return value


def clean_amount(value):
    """Convert Indian currency strings to numeric values."""

    if pd.isna(value):
        return np.nan

    value = str(value)

    # Remove currency symbols
    value = value.replace("₹", "")

    # Remove commas and spaces
    value = value.replace(",", "")
    value = value.replace(" ", "")

    # Keep only numeric characters
    value = re.sub(r"[^0-9.\-]", "", value)

    try:
        return float(value)
    except ValueError:
        return np.nan


# ---------------------------------------------------------
# Allocation Dataset
# ---------------------------------------------------------

def process_allocation_data():

    file = DATA_DIR / "allocated_limit_mps.csv"

    print("\nLoading allocation dataset...")

    df = pd.read_csv(file)

    # Standardize column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # ---------------------------------------------
    # Identify columns
    # ---------------------------------------------

    state_col = next(
        (c for c in df.columns if c.lower() == "state"),
        None
    )

    mp_col = next(
        (
            c for c in df.columns
            if "members of parliament" in c.lower()
            or "members of parliaments" in c.lower()
        ),
        None
    )

    constituency_col = next(
        (
            c for c in df.columns
            if "constituency" in c.lower()
        ),
        None
    )

    amount_col = next(
        (
            c for c in df.columns
            if "allocated amount" in c.lower()
        ),
        None
    )

    if not all([
        state_col,
        mp_col,
        constituency_col,
        amount_col
    ]):
        raise ValueError(
            "Could not identify required columns."
        )

    # ---------------------------------------------
    # Clean text
    # ---------------------------------------------

    df["state_clean"] = (
        df[state_col]
        .apply(clean_text)
    )

    df["mp_clean"] = (
        df[mp_col]
        .apply(clean_text)
    )

    df["constituency_clean"] = (
        df[constituency_col]
        .apply(clean_text)
    )

    # ---------------------------------------------
    # Clean amounts
    # ---------------------------------------------

    df["allocated_amount"] = (
        df[amount_col]
        .apply(clean_amount)
    )

    # ---------------------------------------------
    # Missing value flags
    # ---------------------------------------------

    df["missing_state"] = (
        df["state_clean"].isna()
    )

    df["missing_mp"] = (
        df["mp_clean"].isna()
    )

    df["missing_constituency"] = (
        df["constituency_clean"].isna()
    )

    df["missing_amount"] = (
        df["allocated_amount"].isna()
    )

    # ---------------------------------------------
    # Duplicate detection
    # ---------------------------------------------

    df["duplicate_record"] = (
        df.duplicated(
            subset=[
                "state_clean",
                "mp_clean",
                "constituency_clean",
                "allocated_amount"
            ],
            keep=False
        )
    )

    # ---------------------------------------------
    # Statistical extreme detection
    # ---------------------------------------------

    valid_amounts = df[
        df["allocated_amount"].notna()
        &
        (df["allocated_amount"] > 0)
    ]["allocated_amount"]

    if len(valid_amounts) > 0:

        q1 = valid_amounts.quantile(0.25)
        q3 = valid_amounts.quantile(0.75)

        iqr = q3 - q1

        upper_limit = q3 + 1.5 * iqr

        df["extreme_amount"] = (
            df["allocated_amount"] > upper_limit
        )

    else:

        df["extreme_amount"] = False

    # ---------------------------------------------
    # Data Quality Score
    # ---------------------------------------------

    df["data_quality_score"] = 100.0

    df.loc[
        df["missing_state"],
        "data_quality_score"
    ] -= 20

    df.loc[
        df["missing_mp"],
        "data_quality_score"
    ] -= 20

    df.loc[
        df["missing_constituency"],
        "data_quality_score"
    ] -= 20

    df.loc[
        df["missing_amount"],
        "data_quality_score"
    ] -= 25

    df.loc[
        df["duplicate_record"],
        "data_quality_score"
    ] -= 15

    df.loc[
        df["extreme_amount"],
        "data_quality_score"
    ] -= 15

    df["data_quality_score"] = (
        df["data_quality_score"]
        .clip(0, 100)
    )

    # ---------------------------------------------
    # Overall anomaly flag
    # ---------------------------------------------

    df["data_quality_alert"] = (
        (df["data_quality_score"] < 80)
        |
        df["extreme_amount"]
        |
        df["duplicate_record"]
    )

    # ---------------------------------------------
    # Save processed dataset
    # ---------------------------------------------

    output = OUTPUT_DIR / "allocation_processed.csv"

    df.to_csv(
        output,
        index=False
    )

    print("\nAllocation processing completed.")

    print(
        "Records:",
        len(df)
    )

    print(
        "Missing amounts:",
        df["missing_amount"].sum()
    )

    print(
        "Potential duplicates:",
        df["duplicate_record"].sum()
    )

    print(
        "Extreme amounts:",
        df["extreme_amount"].sum()
    )

    print(
        "Quality alerts:",
        df["data_quality_alert"].sum()
    )

    print(
        "\nSaved:",
        output
    )

    return df


# ---------------------------------------------------------
# Calamity Dataset
# ---------------------------------------------------------

def process_calamity_data():

    file = DATA_DIR / "amount_consented_calamity.csv"

    print("\nLoading calamity dataset...")

    df = pd.read_csv(file)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Identify columns

    type_col = next(
        (
            c for c in df.columns
            if "calamity type" in c.lower()
        ),
        None
    )

    name_col = next(
        (
            c for c in df.columns
            if "calamity name" in c.lower()
        ),
        None
    )

    mp_col = next(
        (
            c for c in df.columns
            if "members of parliament" in c.lower()
        ),
        None
    )

    date_col = next(
        (
            c for c in df.columns
            if "date of consent" in c.lower()
        ),
        None
    )

    amount_col = next(
        (
            c for c in df.columns
            if "consent amount" in c.lower()
        ),
        None
    )

    if amount_col is None:
        raise ValueError(
            "Could not find consent amount column."
        )

    # Clean text

    if type_col:
        df["calamity_type_clean"] = (
            df[type_col]
            .apply(clean_text)
        )

    if name_col:
        df["calamity_name_clean"] = (
            df[name_col]
            .apply(clean_text)
        )

    if mp_col:
        df["mp_clean"] = (
            df[mp_col]
            .apply(clean_text)
        )

    # Clean date

    if date_col:

        df["consent_date"] = pd.to_datetime(
            df[date_col],
            errors="coerce",
            dayfirst=True
        )

    # Clean amount

    df["consent_amount"] = (
        df[amount_col]
        .apply(clean_amount)
    )

    # Missing data

    df["missing_amount"] = (
        df["consent_amount"].isna()
    )

    # Duplicate detection

    duplicate_columns = []

    for col in [
        "calamity_type_clean",
        "calamity_name_clean",
        "mp_clean",
        "consent_amount"
    ]:
        if col in df.columns:
            duplicate_columns.append(col)

    df["duplicate_record"] = (
        df.duplicated(
            subset=duplicate_columns,
            keep=False
        )
    )

    # Quality score

    df["data_quality_score"] = 100.0

    for col in [
        "calamity_type_clean",
        "calamity_name_clean",
        "mp_clean"
    ]:

        if col in df.columns:

            df.loc[
                df[col].isna(),
                "data_quality_score"
            ] -= 20

    df.loc[
        df["missing_amount"],
        "data_quality_score"
    ] -= 25

    df.loc[
        df["duplicate_record"],
        "data_quality_score"
    ] -= 15

    df["data_quality_score"] = (
        df["data_quality_score"]
        .clip(0, 100)
    )

    df["data_quality_alert"] = (
        df["data_quality_score"] < 80
    )

    # Save

    output = (
        OUTPUT_DIR /
        "calamity_processed.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print("\nCalamity processing completed.")

    print(
        "Records:",
        len(df)
    )

    print(
        "Missing amounts:",
        df["missing_amount"].sum()
    )

    print(
        "Potential duplicates:",
        df["duplicate_record"].sum()
    )

    print(
        "Quality alerts:",
        df["data_quality_alert"].sum()
    )

    print(
        "\nSaved:",
        output
    )

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    allocation = process_allocation_data()

    calamity = process_calamity_data()

    print("\n===================================")
    print("DATA QUALITY ENGINE COMPLETED")
    print("===================================")

    print("\nProcessed files:")
    print("1.", OUTPUT_DIR / "allocation_processed.csv")
    print("2.", OUTPUT_DIR / "calamity_processed.csv")