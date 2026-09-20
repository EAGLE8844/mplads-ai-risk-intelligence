# MPLADS Final Dashboard

A standalone Streamlit dashboard for the MPLADS AI Risk Intelligence & Early Warning Platform.

## What it uses

The dashboard automatically discovers these files if they exist in `processed/`, `data/`, the project root, or the current working directory:

- `unified_risk_results.csv`
- `investigation_cases.csv`
- `allocation_anomaly_results.csv`
- `peer_intelligence_results.csv`
- `statistical_anomaly_results.csv`
- `Amount consented for Calamity (1)(2).csv` / `Amount consented for Calamity (1).csv`
- `Allocated Limit for Honble MPs (1)(1).csv` / `Allocated Limit for Honble MPs (1).csv`

It does not fabricate missing project-level data.

## Run on Windows

From the project root:

```bat
pip install -r requirements.txt
streamlit run dashboard\app.py
```

Or run the supplied `run_dashboard.bat`.

## Main screens

1. Executive Overview
2. AI Risk Explorer
3. Investigation Cases
4. Calamity Intelligence
5. Methodology & Data Scope

## Important interpretation

The dashboard presents anomaly and risk-prioritization signals. A high score is **not proof of fraud or wrongdoing**. Underlying official records must be verified by authorized personnel.
