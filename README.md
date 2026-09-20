# MPLADS AI Monitoring — Phase 1

A working Phase-1 MVP using the two real MPLADS CSV files supplied by the user.

## Features

- Imports and cleans the two MPLADS datasets
- Removes Grand Total/subtotal rows from MP allocation data
- Handles missing/invalid allocation amounts
- Stores cleaned records in SQLite
- FastAPI REST API
- Streamlit dashboard
- State-wise and constituency-wise allocation analytics
- Calamity consent analytics
- Basic allocation anomaly detection using Isolation Forest
- Data-quality metrics
- Project is intentionally designed so Phase 2 can add works, sanctions, expenditure, payments, progress, GPS and photos.

## Run on Windows

Open a terminal in this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m backend.ingest

uvicorn backend.main:app --reload
```

Open another terminal:

```powershell
.venv\Scripts\activate
streamlit run frontend/dashboard.py
```

- API: http://127.0.0.1:8000
- Swagger docs: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

## Run on Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m backend.ingest
uvicorn backend.main:app --reload
```

In another terminal:

```bash
source .venv/bin/activate
streamlit run frontend/dashboard.py
```

## Important

The anomaly score is a screening signal, NOT a finding of fraud. A high-risk record should be manually verified.

## Phase 2

Add official work/project, sanction, expenditure, payment, progress, implementing-agency, GPS and image datasets. The existing database/API structure can be extended without replacing the Phase-1 dashboard.
