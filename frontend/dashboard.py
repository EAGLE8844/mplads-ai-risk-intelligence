import sys
from pathlib import Path

# Allows `streamlit run frontend/dashboard.py` to import the backend package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MPLADS AI Monitoring",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ MPLADS AI Monitoring — Phase 1")
st.caption("Real MPLADS allocation and calamity-consent data | AI screening is not a fraud finding.")

def get_json(path, params=None):
    try:
        r = requests.get(API + path, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(
            "Backend API is not running. Start it with:\n\n"
            "`uvicorn backend.main:app --reload`\n\n"
            f"Details: {e}"
        )
        st.stop()

summary = get_json("/api/summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MP Records", f"{summary['mp_records']:,}")
c2.metric("States / UTs", f"{summary['states']:,}")
c3.metric("Allocated", f"₹{summary['total_allocated']/1e7:,.2f} Cr")
c4.metric("Calamity Consent", f"₹{summary['calamity_total']/1e7:,.2f} Cr")
c5.metric("Allocation Alerts", f"{summary['allocation_anomalies']:,}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🧑‍💼 MP Explorer",
    "🚨 AI Alerts",
    "🌊 Calamity Analysis"
])

with tab1:
    states = pd.DataFrame(get_json("/api/states"))
    if not states.empty:
        states["Allocated (Cr)"] = states["allocated_amount"] / 1e7

        left, right = st.columns(2)

        with left:
            st.subheader("State-wise Allocation")
            fig = px.bar(
                states.sort_values("Allocated (Cr)", ascending=True),
                x="Allocated (Cr)",
                y="state",
                orientation="h",
                hover_data=["mp_count", "data_quality"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Data Quality by State")
            fig2 = px.bar(
                states.sort_values("data_quality"),
                x="state",
                y="data_quality",
                labels={"data_quality": "Quality Score", "state": "State"}
            )
            fig2.update_yaxes(range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("State Summary")
        st.dataframe(
            states.rename(columns={
                "state": "State",
                "mp_count": "MP Records",
                "allocated_amount": "Allocated Amount (₹)",
                "data_quality": "Data Quality"
            }),
            use_container_width=True,
            hide_index=True
        )

with tab2:
    states_list = ["All"] + [x["state"] for x in get_json("/api/states")]
    selected_state = st.selectbox("State", states_list)
    search = st.text_input("Search MP or constituency")

    params = {"search": search}
    if selected_state != "All":
        params["state"] = selected_state

    rows = pd.DataFrame(get_json("/api/mps", params=params))

    if not rows.empty:
        rows["allocated_amount"] = rows["allocated_amount"].round(2)
        rows["anomaly_score"] = rows["anomaly_score"].round(1)
        rows["data_quality_score"] = rows["data_quality_score"].round(1)
        rows["status"] = rows["is_anomaly"].map({True: "🔴 Review", False: "🟢 Normal"})

        st.dataframe(
            rows[[
                "state", "mp_name", "constituency",
                "allocated_amount", "status",
                "anomaly_score", "data_quality_score"
            ]].rename(columns={
                "state": "State",
                "mp_name": "MP",
                "constituency": "Constituency",
                "allocated_amount": "Allocated Amount (₹)",
                "status": "AI Screening",
                "anomaly_score": "Anomaly Score",
                "data_quality_score": "Data Quality"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No matching records.")

with tab3:
    alerts = pd.DataFrame(get_json("/api/mps", params={"only_anomalies": True}))
    st.subheader("Allocation Anomalies")
    st.info(
        "These alerts identify statistically unusual allocation amounts. "
        "They do not prove fraud or non-compliance."
    )

    if not alerts.empty:
        alerts["allocated_amount"] = alerts["allocated_amount"].round(2)
        alerts = alerts.sort_values("anomaly_score", ascending=False)

        st.dataframe(
            alerts[[
                "state", "mp_name", "constituency",
                "allocated_amount", "anomaly_score"
            ]].rename(columns={
                "state": "State",
                "mp_name": "MP",
                "constituency": "Constituency",
                "allocated_amount": "Allocated Amount (₹)",
                "anomaly_score": "Anomaly Score"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No allocation anomalies detected.")

with tab4:
    calamities = pd.DataFrame(get_json("/api/calamities"))

    if not calamities.empty:
        calamities["Consent (Cr)"] = calamities["consent_amount"] / 1e7

        fig = px.bar(
            calamities.sort_values("Consent (Cr)", ascending=True),
            x="Consent (Cr)",
            y="calamity_name",
            orientation="h",
            hover_data=["mp_count"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            calamities.rename(columns={
                "calamity_name": "Calamity",
                "mp_count": "MP Contributions",
                "consent_amount": "Consent Amount (₹)"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No calamity data found.")

st.divider()
st.caption(
    f"Average allocation data-quality score: {summary['avg_data_quality']:.1f}/100"
)
