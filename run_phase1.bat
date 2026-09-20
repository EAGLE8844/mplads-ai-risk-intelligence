@echo off
echo Installing dependencies...
python -m pip install -r requirements.txt
echo Importing MPLADS data...
python -m backend.ingest
start "MPLADS API" cmd /k "uvicorn backend.main:app --reload"
timeout /t 3 >nul
start "MPLADS Dashboard" cmd /k "streamlit run frontend/dashboard.py"
echo.
echo API: http://127.0.0.1:8000/docs
echo Dashboard: http://localhost:8501
pause
