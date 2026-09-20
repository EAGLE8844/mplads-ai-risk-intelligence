from pathlib import Path
import sys
import math
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# MPLADS AI Risk Intelligence & Early Warning Platform
# Standalone Streamlit dashboard. No API/database required.
# It consumes the real pipeline outputs when they exist and falls back to the
# two original MPLADS CSV datasets for the data-explorer/calamity views.
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="MPLADS AI Risk Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

SEARCH_ROOTS = [
    PROJECT_ROOT / "processed",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "backend" / ".." / "processed",
    Path.cwd() / "processed",
    Path.cwd() / "data",
]


def unique_paths(paths):
    out = []
    seen = set()
    for p in paths:
        p = p.resolve()
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


PROCESSED_ROOTS = unique_paths(SEARCH_ROOTS)


def locate(*names):
    for root in PROCESSED_ROOTS:
        for name in names:
            p = root / name
            if p.exists():
                return p
    # Also recursively search the project for exact basenames.
    for root in [PROJECT_ROOT, Path.cwd()]:
        if not root.exists():
            continue
        for name in names:
            try:
                hits = list(root.rglob(name))
            except Exception:
                hits = []
            if hits:
                return hits[0]
    return None


def read_csv(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def clean_amount(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def first_col(df, candidates, default=None):
    if df is None:
        return default
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in normalized:
            return normalized[c.lower()]
    for c in df.columns:
        low = str(c).strip().lower()
        if any(x.lower() in low for x in candidates):
            return c
    return default


def safe_num(df, col, default=0.0):
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def money_inr(x):
    if pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x/1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:,.2f} L"
    return f"₹{x:,.0f}"


def money_full(x):
    if pd.isna(x):
        return "—"
    return f"₹{float(x):,.2f}"


def risk_class(value):
    v = str(value).upper()
    if "CRITICAL" in v:
        return "critical"
    if "HIGH" in v:
        return "high"
    if "MEDIUM" in v or "WATCH" in v:
        return "medium"
    return "low"


def load_all():
    unified_path = locate("unified_risk_results.csv")
    cases_path = locate("investigation_cases.csv")
    anomaly_path = locate("allocation_anomaly_results.csv")
    peer_path = locate("peer_intelligence_results.csv")
    stat_path = locate("statistical_anomaly_results.csv")
    calamity_path = locate(
        "Amount consented for Calamity (1)(2).csv",
        "Amount consented for Calamity (1).csv",
        "amount_consented_calamity.csv",
    )
    allocation_path = locate(
        "Allocated Limit for Honble MPs (1)(1).csv",
        "Allocated Limit for Honble MPs (1).csv",
        "allocated_limit_mps.csv",
    )

    return {
        "unified": read_csv(unified_path),
        "cases": read_csv(cases_path),
        "anomaly": read_csv(anomaly_path),
        "peer": read_csv(peer_path),
        "stat": read_csv(stat_path),
        "calamity": read_csv(calamity_path),
        "allocation": read_csv(allocation_path),
        "paths": {
            "unified": unified_path,
            "cases": cases_path,
            "anomaly": anomaly_path,
            "peer": peer_path,
            "stat": stat_path,
            "calamity": calamity_path,
            "allocation": allocation_path,
        },
    }


DATA = load_all()

# Prefer the unified result because it contains the integrated intelligence
# layers. If unavailable, use anomaly output and still make the dashboard useful.
df = DATA["unified"] if DATA["unified"] is not None else DATA["anomaly"]

# Normalize the main table without changing the source files.
if df is not None:
    df = df.copy()
    df["__sr"] = safe_num(df, first_col(df, ["Sr. No.", "sr_no", "serial"]))
    df["__mp"] = df[first_col(df, ["Hon'ble Members of Parliaments", "Hon'ble Members of Parliament", "mp_name", "member"])] if first_col(df, ["Hon'ble Members of Parliaments", "Hon'ble Members of Parliament", "mp_name", "member"]) else "Unknown"
    df["__state"] = df[first_col(df, ["State", "state", "state_clean"])] if first_col(df, ["State", "state", "state_clean"]) else "Unknown"
    df["__const"] = df[first_col(df, ["Constituency", "constituency", "constituency_clean"])] if first_col(df, ["Constituency", "constituency", "constituency_clean"]) else "Unknown"
    amount_col = first_col(df, ["allocated_amount", "Allocated AMOUNT ( ₹ )", "allocated amount", "amount_numeric"])
    df["__amount"] = clean_amount(df[amount_col]) if amount_col else 0.0
    df["__amount"] = df["__amount"].fillna(safe_num(df, "amount_numeric"))
    df["__risk"] = safe_num(df, first_col(df, ["unified_risk_score", "risk_score", "anomaly_score"]), 0)
    df["__ml"] = safe_num(df, first_col(df, ["isolation_score", "ml_risk", "ml_score"]), 0)
    df["__peer"] = safe_num(df, first_col(df, ["peer_risk_score", "peer_risk", "peer_score"]), 0)
    df["__financial"] = safe_num(df, first_col(df, ["financial_risk", "financial_score"]), 0)
    df["__stat"] = safe_num(df, first_col(df, ["statistical_component_score", "adjusted_statistical_risk", "statistical_risk", "statistical_score"]), 0)
    df["__percentile"] = safe_num(df, first_col(df, ["amount_percentile", "global_percentile", "statistical_percentile"]), 0)
    df["__risk_level"] = df[first_col(df, ["risk_level", "unified_risk_level"])] if first_col(df, ["risk_level", "unified_risk_level"]) else df["__risk"].map(lambda x: "HIGH" if x >= 60 else "MEDIUM" if x >= 30 else "LOW")
    # Exclude aggregate/Grand Total rows from MP-level analytics.
    # The source files contain a final Grand Total row that is not an MP record.
    record_type_col = first_col(df, ["record_type", "Record Type"])
    if record_type_col:
        df = df[df[record_type_col].astype(str).str.upper().eq("MP_RECORD")].copy()
    else:
        df = df[df["__sr"].notna() & (df["__sr"] > 0)].copy()
    df["__priority"] = df[first_col(df, ["priority", "investigation_priority", "alert_priority"])] if first_col(df, ["priority", "investigation_priority", "alert_priority"]) else "ROUTINE"
    df["__peer_ratio"] = safe_num(df, first_col(df, ["peer_median_ratio", "peer_ratio"]), 0)
    df["__peer_dev"] = safe_num(df, first_col(df, ["peer_deviation_pct", "peer_deviation"]), 0)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    """
<style>
:root { --ink:#172033; --muted:#667085; --line:#e7eaf0; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
.hero { background: linear-gradient(135deg,#101828 0%,#1d2939 55%,#344054 100%); padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 18px; }
.hero h1 { font-size: 31px; margin: 0 0 6px 0; letter-spacing:-.5px; }
.hero p { color:#d0d5dd; margin:0; font-size:15px; }
.kpi { border:1px solid var(--line); border-radius:16px; padding:18px 18px 16px 18px; background:#fff; box-shadow:0 2px 9px rgba(16,24,40,.04); min-height:118px; }
.kpi-label { color:var(--muted); font-size:13px; font-weight:600; }
.kpi-value { color:var(--ink); font-size:27px; font-weight:750; margin-top:8px; }
.kpi-sub { color:#98a2b3; font-size:12px; margin-top:4px; }
.badge { display:inline-block; padding:4px 9px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-high { background:#fef3f2; color:#b42318; }
.badge-med { background:#fffaeb; color:#b54708; }
.badge-low { background:#ecfdf3; color:#027a48; }
.notice { background:#f8fafc; border:1px solid #e4e7ec; border-left:4px solid #475467; padding:13px 15px; border-radius:10px; color:#475467; font-size:13px; }
.section-title { font-size:19px; font-weight:750; color:#172033; margin:10px 0 8px; }
.small-muted { color:#667085; font-size:12px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🏛️ MPLADS AI Risk Intelligence & Early Warning Platform</h1>
  <p>Explainable AI screening of available MPLADS allocation and calamity-consent data — designed to prioritize records for human verification, not to declare fraud.</p>
</div>
""",
    unsafe_allow_html=True,
)

if df is None:
    st.error(
        "No MPLADS data was found. Put the existing project folders/files beside this dashboard, "
        "or place the processed CSV outputs in a `processed` folder."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔎 Control Center")
    state_options = ["All"] + sorted(df["__state"].dropna().astype(str).unique().tolist())
    selected_state = st.selectbox("State / UT", state_options)
    risk_options = ["All"] + sorted(df["__risk_level"].dropna().astype(str).unique().tolist())
    selected_risk = st.selectbox("Risk level", risk_options)
    search = st.text_input("Search MP / constituency", placeholder="e.g. Telangana or Rajender")
    min_score = st.slider("Minimum unified risk score", 0, 100, 0)
    st.divider()
    st.markdown("**Data source status**")
    for key, label in [
        ("unified", "Unified risk"), ("cases", "Investigation cases"),
        ("anomaly", "ML anomaly"), ("peer", "Peer intelligence"),
        ("stat", "Statistical"), ("calamity", "Calamity consent"),
    ]:
        icon = "🟢" if DATA["paths"].get(key) else "⚪"
        st.write(f"{icon} {label}")
    st.caption("Green = file found on disk. The dashboard never fabricates missing project-level data.")

view = st.radio(
    "Navigation",
    ["Executive Overview", "AI Risk Explorer", "Investigation Cases", "Calamity Intelligence", "Methodology & Data Scope"],
    horizontal=True,
)

filtered = df.copy()
if selected_state != "All":
    filtered = filtered[filtered["__state"].astype(str) == selected_state]
if selected_risk != "All":
    filtered = filtered[filtered["__risk_level"].astype(str).str.upper() == selected_risk.upper()]
if search.strip():
    q = search.strip().lower()
    mask = (
        filtered["__mp"].astype(str).str.lower().str.contains(q, na=False)
        | filtered["__const"].astype(str).str.lower().str.contains(q, na=False)
        | filtered["__state"].astype(str).str.lower().str.contains(q, na=False)
    )
    filtered = filtered[mask]
filtered = filtered[filtered["__risk"] >= min_score]

# -----------------------------------------------------------------------------
# Executive Overview
# -----------------------------------------------------------------------------
if view == "Executive Overview":
    mp_count = int((df["__sr"] > 0).sum()) if df["__sr"].notna().any() else len(df)
    states = int(df["__state"].nunique())
    total_alloc = df["__amount"].sum()
    high = int(df["__risk_level"].astype(str).str.upper().eq("HIGH").sum())
    critical = int(df["__risk_level"].astype(str).str.upper().eq("CRITICAL").sum())
    medium = int(df["__risk_level"].astype(str).str.upper().eq("MEDIUM").sum())
    cases = DATA["cases"]
    case_count = len(cases) if cases is not None else 0
    calamity = DATA["calamity"]
    calamity_total = 0
    calamity_count = 0
    if calamity is not None:
        cc = first_col(calamity, ["Consent Amount ( ₹ )", "consent_amount", "amount"])
        if cc:
            calamity_total = clean_amount(calamity[cc]).sum()
        calamity_count = len(calamity)

    cols = st.columns(5)
    metrics = [
        ("MP records", f"{mp_count:,}", "Available allocation records"),
        ("States / UTs", f"{states:,}", "Distinct state values"),
        ("Allocated amount", money_inr(total_alloc), "Sum of allocation records"),
        ("High / Critical", f"{high + critical:,}", f"{high} high · {critical} critical"),
        ("Investigation cases", f"{case_count:,}", f"{calamity_count} calamity-consent records"),
    ]
    for c, (label, value, sub) in zip(cols, metrics):
        c.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="notice">⚠️ <b>Interpretation rule:</b> an AI alert is an anomaly / prioritization signal. It is not proof of fraud, misconduct, or non-compliance.</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="section-title">Risk distribution</div>', unsafe_allow_html=True)
        risk_counts = df["__risk_level"].astype(str).str.upper().value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0).reset_index()
        risk_counts.columns = ["Risk", "Records"]
        fig = px.bar(risk_counts, x="Risk", y="Records", text="Records", category_orders={"Risk":["LOW","MEDIUM","HIGH","CRITICAL"]})
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=20,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Signal coverage</div>', unsafe_allow_html=True)
        signals = pd.DataFrame({
            "Signal": ["ML anomaly", "Peer intelligence", "Financial risk", "Statistical risk"],
            "Records with signal": [
                int((df["__ml"] > 0).sum()),
                int((df["__peer"] > 0).sum()),
                int((df["__financial"] > 0).sum()),
                int((df["__stat"] > 0).sum()),
            ],
        })
        fig = px.bar(signals, x="Records with signal", y="Signal", orientation="h", text="Records with signal")
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=20,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Top AI investigation candidates</div>', unsafe_allow_html=True)
    top = df.sort_values(["__risk", "__amount"], ascending=False).head(10).copy()
    display = pd.DataFrame({
        "Rank": range(1, len(top)+1),
        "MP": top["__mp"].values,
        "State": top["__state"].values,
        "Constituency": top["__const"].values,
        "Allocated": [money_full(x) for x in top["__amount"]],
        "Unified Risk": top["__risk"].round(2).values,
        "Risk": top["__risk_level"].values,
        "ML": top["__ml"].round(2).values,
        "Peer": top["__peer"].round(2).values,
        "Financial": top["__financial"].round(2).values,
        "Statistical": top["__stat"].round(2).values,
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# AI Risk Explorer
# -----------------------------------------------------------------------------
elif view == "AI Risk Explorer":
    st.markdown('<div class="section-title">AI Risk Explorer</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records after the sidebar filters.")

    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        # Scatter: financial exposure vs ML signal, size = risk.
        fig = px.scatter(
            filtered,
            x="__financial",
            y="__ml",
            size="__risk",
            hover_name="__mp",
            hover_data={"__state":True,"__const":True,"__amount":":,.0f","__risk":":.2f"},
            labels={"__financial":"Financial risk","__ml":"ML anomaly score","__risk":"Unified risk"},
        )
        fig.update_layout(height=430, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Record-level intelligence")
        shown = filtered.sort_values("__risk", ascending=False).copy()
        shown["Amount"] = shown["__amount"].map(money_full)
        shown["Risk"] = shown["__risk"].round(2)
        shown["ML"] = shown["__ml"].round(2)
        shown["Peer"] = shown["__peer"].round(2)
        shown["Financial"] = shown["__financial"].round(2)
        shown["Statistical"] = shown["__stat"].round(2)
        shown["Peer ratio"] = shown["__peer_ratio"].round(2).map(lambda x: f"{x:.2f}x" if x else "—")
        shown["Peer deviation"] = shown["__peer_dev"].round(2).map(lambda x: f"{x:+.2f}%" if x else "—")
        table = pd.DataFrame({
            "MP": shown["__mp"], "State":shown["__state"], "Constituency":shown["__const"],
            "Amount":shown["Amount"], "Risk":shown["Risk"], "Risk level":shown["__risk_level"],
            "ML":shown["ML"], "Peer":shown["Peer"], "Financial":shown["Financial"],
            "Statistical":shown["Statistical"], "Peer ratio":shown["Peer ratio"], "Peer deviation":shown["Peer deviation"]
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown("#### Inspect a record")
        options = list(filtered.index)
        chosen = st.selectbox("Select record", options, format_func=lambda i: f"{filtered.loc[i,'__mp']} — {filtered.loc[i,'__state']} — {money_full(filtered.loc[i,'__amount'])}")
        r = filtered.loc[chosen]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Unified risk", f"{r['__risk']:.2f}")
        c2.metric("ML score", f"{r['__ml']:.2f}")
        c3.metric("Peer risk", f"{r['__peer']:.2f}")
        c4.metric("Financial risk", f"{r['__financial']:.2f}")
        st.write(f"**{r['__mp']}** · {r['__const']}, {r['__state']} · {money_full(r['__amount'])}")

        # Evidence cards
        ev = []
        if r["__ml"] >= 50: ev.append(f"ML anomaly score is {r['__ml']:.2f}.")
        if r["__peer"] > 0:
            ratio = r["__peer_ratio"]
            dev = r["__peer_dev"]
            if ratio > 0: ev.append(f"Allocation is {ratio:.2f}× the peer-group median.")
            if dev != 0: ev.append(f"Peer deviation is {dev:+.2f}%.")
        if r["__financial"] >= 75: ev.append(f"Financial exposure signal is {r['__financial']:.0f}/100.")
        if r["__stat"] > 0: ev.append(f"Statistical component contributes {r['__stat']:.2f}/100.")
        if not ev: ev.append("No strong independent signal is visible at the selected thresholds.")
        st.markdown("**Why did the system flag it?**")
        for item in ev:
            st.write("• " + item)
        st.markdown('<div class="notice">Recommended interpretation: <b>requires verification</b>. Review the underlying official records before drawing any conclusion.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Investigation Cases
# -----------------------------------------------------------------------------
elif view == "Investigation Cases":
    cases = DATA["cases"]
    st.markdown('<div class="section-title">Investigation Case Queue</div>', unsafe_allow_html=True)
    st.caption("Cases are prioritization outputs generated from the unified risk layer.")
    if cases is None or cases.empty:
        st.warning("investigation_cases.csv was not found. Run investigation_case_generator.py first.")
    else:
        cases = cases.copy()
        member = first_col(cases, ["member", "mp", "Hon'ble Members of Parliaments", "Member"])
        state = first_col(cases, ["state", "State"])
        const = first_col(cases, ["constituency", "Constituency"])
        amount = first_col(cases, ["amount", "allocated_amount", "Allocated AMOUNT ( ₹ )"])
        score = first_col(cases, ["unified_risk_score", "unified risk", "risk_score"])
        level = first_col(cases, ["risk_level", "Risk Level"])
        priority = first_col(cases, ["priority", "Priority"])
        case_id = first_col(cases, ["case_id", "Case ID"])
        evidence = first_col(cases, ["evidence_strength", "strength", "Evidence"])
        actions = first_col(cases, ["recommended_action", "actions", "Actions"])
        signal_text = first_col(cases, ["signals", "signal_summary", "Signal"])

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Cases", len(cases))
        c2.metric("High", int(cases[level].astype(str).str.upper().eq("HIGH").sum()) if level else 0)
        c3.metric("Watchlist", int(cases[priority].astype(str).str.upper().eq("WATCHLIST").sum()) if priority else 0)
        c4.metric("Strong evidence", int(cases[evidence].astype(str).str.upper().eq("STRONG").sum()) if evidence else 0)

        show = cases.copy()
        if score:
            show["_score"] = pd.to_numeric(show[score], errors="coerce").fillna(0)
        else:
            show["_score"] = 0
        show = show.sort_values("_score", ascending=False)
        cols = {}
        if case_id: cols["Case ID"] = show[case_id]
        if member: cols["MP"] = show[member]
        if state: cols["State"] = show[state]
        if const: cols["Constituency"] = show[const]
        if amount: cols["Amount"] = clean_amount(show[amount]).map(money_full)
        cols["Unified Risk"] = show["_score"].round(2)
        if level: cols["Risk"] = show[level]
        if priority: cols["Priority"] = show[priority]
        if evidence: cols["Evidence"] = show[evidence]
        st.dataframe(pd.DataFrame(cols), use_container_width=True, hide_index=True)

        pick = st.selectbox("Open case", show.index, format_func=lambda i: f"{show.loc[i, member] if member else 'Case'} — risk {show.loc[i,'_score']:.2f}")
        r = show.loc[pick]
        st.markdown("---")
        a,b = st.columns([1,2])
        with a:
            st.markdown(f"### {r[case_id] if case_id else 'Investigation Case'}")
            st.metric("Unified risk", f"{r['_score']:.2f}")
            if level: st.write(f"**Risk:** {r[level]}")
            if priority: st.write(f"**Priority:** {r[priority]}")
            if evidence: st.write(f"**Evidence:** {r[evidence]}")
        with b:
            if member: st.write(f"**MP:** {r[member]}")
            if state: st.write(f"**State:** {r[state]}")
            if const: st.write(f"**Constituency:** {r[const]}")
            if amount: st.write(f"**Amount:** {money_full(clean_amount(pd.Series([r[amount]])).iloc[0])}")
            if signal_text: st.write(f"**Signals:** {r[signal_text]}")
            if actions: st.write(f"**Recommended action:** {r[actions]}")
            st.markdown('<div class="notice">This case should be treated as an <b>AI prioritization candidate for human verification</b>, not as a fraud finding.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Calamity Intelligence
# -----------------------------------------------------------------------------
elif view == "Calamity Intelligence":
    cal = DATA["calamity"]
    st.markdown('<div class="section-title">Calamity Consent Intelligence</div>', unsafe_allow_html=True)
    st.caption("This view uses the separate MPLADS calamity-consent dataset currently available.")
    if cal is None or cal.empty:
        st.warning("Calamity CSV not found.")
    else:
        cal = cal.copy()
        name = first_col(cal, ["Calamity Name", "calamity_name", "Calamity"])
        ctype = first_col(cal, ["Calamity Type", "calamity_type", "Type"])
        mp = first_col(cal, ["Hon'ble Members of Parliament", "Hon'ble Members of Parliaments", "mp_name", "Member"])
        date = first_col(cal, ["Date of Consent", "consent_date", "Date"])
        amount = first_col(cal, ["Consent Amount ( ₹ )", "consent_amount", "amount"])

        # The MPLADS calamity CSV also contains a final Grand Total row.
        # Exclude it so the dashboard does not double-count the consent amount
        # and does not display a blank calamity category.
        if name:
            name_text = cal[name].astype(str).str.strip()
            aggregate_mask = (
                name_text.str.lower().isin({"grand total", "total", "nan", "none", ""})
                | name_text.str.contains(r"grand\s*total", case=False, regex=True, na=False)
            )
        else:
            aggregate_mask = pd.Series(False, index=cal.index)
        if amount:
            amount_numeric = clean_amount(cal[amount])
            # Also catch a total row if its name is blank but its amount is the
            # sum of the individual rows.
            if name:
                detail_sum = amount_numeric[~aggregate_mask].sum()
                aggregate_mask = aggregate_mask | (
                    cal[name].astype(str).str.strip().eq("")
                    & amount_numeric.notna()
                    & (amount_numeric.sub(detail_sum).abs() < 0.01)
                )
        cal = cal.loc[~aggregate_mask].copy()
        cal["_amount"] = clean_amount(cal[amount]) if amount else 0

        c1,c2,c3 = st.columns(3)
        c1.metric("Consent records", len(cal))
        c2.metric("Total consent", money_inr(cal["_amount"].sum()))
        c3.metric("Calamity types", int(cal[name].nunique()) if name else 0)

        left,right = st.columns([1.25, 1])
        with left:
            if name:
                # Aggregate by calamity name and show every category. The
                # previous chart clipped the first/top label and also displayed
                # the Grand Total as a blank category.
                agg = (
                    cal.groupby(name, dropna=False)["_amount"]
                    .sum()
                    .sort_values(ascending=True)
                    .reset_index()
                )
                agg["Amount"] = agg["_amount"] / 1e7
                agg["Label"] = agg["Amount"].map(lambda x: f"₹{x:.2f} Cr")
                chart_height = max(430, 85 * len(agg) + 80)
                fig = px.bar(
                    agg, x="Amount", y=name, orientation="h", text="Label",
                    hover_data={"Amount":":.2f", "Label":False, name:True},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_yaxes(
                    automargin=True,
                    categoryorder="array",
                    categoryarray=agg[name].tolist(),
                    tickfont=dict(size=12),
                )
                fig.update_xaxes(rangemode="tozero", title="Consent Amount (₹ Crore)")
                fig.update_layout(
                    height=chart_height,
                    margin=dict(l=170, r=90, t=35, b=55),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with right:
            if ctype:
                typ = cal.groupby(ctype)["_amount"].sum().sort_values(ascending=False).reset_index()
                typ["Amount"] = typ["_amount"] / 1e7
                typ["Label"] = typ["Amount"].map(lambda x: f"₹{x:.2f} Cr")
                fig = px.pie(typ, names=ctype, values="Amount", hole=.55, hover_data={"Label":True})
                fig.update_layout(
                    height=max(430, 85 * len(typ) + 80),
                    margin=dict(l=20,r=20,t=35,b=20),
                    legend=dict(orientation="v", yanchor="middle", y=0.5),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        out = pd.DataFrame()
        if mp: out["MP"] = cal[mp]
        if name: out["Calamity"] = cal[name]
        if ctype: out["Type"] = cal[ctype]
        if date: out["Consent date"] = cal[date]
        out["Consent amount"] = cal["_amount"].map(money_full)
        st.dataframe(out, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Methodology & Data Scope
# -----------------------------------------------------------------------------
else:
    st.markdown('<div class="section-title">Methodology, Governance & Data Scope</div>', unsafe_allow_html=True)
    st.markdown(
        """
### What the current system actually analyzes

**Available MPLADS inputs**
- MP allocation records: state, MP, constituency and allocated amount.
- Calamity-consent records: calamity information, MP, consent date and consent amount.

**AI / analytics layers**
1. **Isolation Forest / ML anomaly detection** — identifies unusual allocation patterns.
2. **Peer intelligence** — compares a record with relevant peer groups and measures deviation from the peer median.
3. **Robust statistical analysis** — evaluates extreme values and upper-tail behavior.
4. **Financial exposure signal** — prioritizes records in the upper allocation percentiles.
5. **Unified risk engine** — combines independent signals into a 0–100 prioritization score.
6. **Investigation case generator** — converts high-priority candidates into explainable review cases.

### What is intentionally *not* claimed

The public data used here does not provide the project-level execution fields required to prove duplicate works, payment fraud, GPS anomalies, image recycling, delays, cost overruns, or incomplete assets. The dashboard therefore does **not** fabricate or simulate those fields.

### Auditor-first interpretation

> **The system does not decide who is guilty. It helps auditors decide what deserves attention first.**

Every alert should be validated against the underlying official records before any administrative action is taken.
"""
    )
    st.markdown("#### Files detected")
    rows = []
    for key, path in DATA["paths"].items():
        if key == "allocation" or key == "calamity" or key in {"unified","cases","anomaly","peer","stat"}:
            rows.append({"Layer":key, "File":str(path) if path else "Not found", "Status":"Available" if path else "Missing"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("MPLADS AI Risk Intelligence & Early Warning Platform · Prototype / decision-support system · Alerts are for verification prioritization only.")
