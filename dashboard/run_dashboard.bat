@echo off
cd /d "%~dp0.."
echo Starting MPLADS AI Risk Intelligence Dashboard...
streamlit run dashboard\app.py
pause
