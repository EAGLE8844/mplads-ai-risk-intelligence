from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MPLADS Review Assistant", page_icon="🏛️", layout="wide")

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# ----------------------------- Data helpers -----------------------------
def locate(*names):
    """Find a data file regardless of whether the user copied the whole ZIP or only dashboard/."""
    # Direct locations first. Include dashboard/data because users commonly copy only dashboard/.
    roots = [
        APP_DIR / "data",
        ROOT / "data",
        ROOT / "processed",
        Path.cwd() / "data",
        Path.cwd() / "processed",
        ROOT,
        APP_DIR,
    ]
    wanted = [str(n).strip().lower() for n in names]

    # Exact filename match.
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            p = root / name
            if p.is_file():
                return p

    # Case-insensitive filename match.
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*.csv"):
                if p.name.strip().lower() in wanted:
                    return p
        except Exception:
            pass

    # Flexible filename matching for renamed/copy-suffixed Windows files.
    # This is intentionally based on distinctive phrases, not a hard-coded absolute path.
    phrases = []
    for n in names:
        q = str(n).lower().replace("_", " ")
        if "calamity" in q:
            phrases.append("calamity")
        if "allocated" in q and "honble" in q:
            phrases.append("allocated")
    phrases = set(phrases)
    for root in [ROOT, Path.cwd(), APP_DIR]:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*.csv"):
                low = p.name.lower()
                if "calamity" in phrases and "calamity" in low:
                    return p
                if "allocated" in phrases and "allocated" in low and "honble" in low:
                    return p
        except Exception:
            pass
    return None


def read_csv(path):
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def col(df, names):
    if df is None:
        return None
    exact = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in exact:
            return exact[n.lower()]
    for c in df.columns:
        s = str(c).strip().lower()
        if any(n.lower() in s for n in names):
            return c
    return None


def num(s):
    return pd.to_numeric(s.astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.replace("Rs.", "", regex=False).str.replace("Rs", "", regex=False), errors="coerce")


def money(x):
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


def risk_bucket(x):
    x = float(x or 0)
    if x >= 80:
        return "Critical"
    if x >= 60:
        return "Needs attention"
    if x >= 30:
        return "Worth checking"
    return "No major signal"


def risk_class(label):
    s = str(label).lower()
    if "critical" in s or "attention" in s:
        return "red"
    if "worth" in s or "watch" in s:
        return "amber"
    return "green"

# ----------------------------- Load actual outputs -----------------------------
paths = {
    "unified": locate("unified_risk_results.csv"),
    "cases": locate("investigation_cases.csv"),
    "anomaly": locate("allocation_anomaly_results.csv"),
    "calamity": locate("Amount consented for Calamity (1)(2).csv", "Amount consented for Calamity (1).csv", "Amount consented for Calamity.csv", "Amount consented for Calamity (1)(2).CSV"),
    "allocation": locate("Allocated Limit for Honble MPs (1)(1).csv", "Allocated Limit for Honble MPs (1).csv", "Allocated Limit for Honble MPs.csv"),
}
raw = {k: read_csv(v) for k, v in paths.items()}
main = raw["unified"] if raw["unified"] is not None else raw["anomaly"]

if main is not None:
    d = main.copy()
    sr = col(d, ["Sr. No.", "sr_no", "serial"])
    mp = col(d, ["Hon'ble Members of Parliaments", "Hon'ble Members of Parliament", "mp_name", "member"])
    state = col(d, ["State", "state"])
    const = col(d, ["Constituency", "constituency"])
    amount = col(d, ["allocated_amount", "Allocated AMOUNT", "allocated amount", "amount_numeric"])
    risk = col(d, ["unified_risk_score", "risk_score", "anomaly_score"])
    level = col(d, ["risk_level", "unified_risk_level"])
    ml = col(d, ["isolation_score", "ml_risk", "ml_score"])
    peer = col(d, ["peer_risk_score", "peer_risk", "peer_score"])
    fin = col(d, ["financial_risk", "financial_score"])
    stat = col(d, ["statistical_component_score", "adjusted_statistical_risk", "statistical_risk", "statistical_score"])
    ratio = col(d, ["peer_median_ratio", "peer_ratio"])
    deviation = col(d, ["peer_deviation_pct", "peer_deviation"])
    percentile = col(d, ["amount_percentile", "global_percentile", "statistical_percentile"])
    record_type = col(d, ["record_type", "Record Type"])

    d["_sr"] = pd.to_numeric(d[sr], errors="coerce") if sr else range(1, len(d)+1)
    d["_mp"] = d[mp].astype(str) if mp else "Unknown"
    d["_state"] = d[state].astype(str) if state else "Unknown"
    d["_const"] = d[const].astype(str) if const else "Unknown"
    d["_amount"] = num(d[amount]) if amount else 0.0
    d["_risk"] = pd.to_numeric(d[risk], errors="coerce").fillna(0) if risk else 0.0
    d["_ml"] = pd.to_numeric(d[ml], errors="coerce").fillna(0) if ml else 0.0
    d["_peer"] = pd.to_numeric(d[peer], errors="coerce").fillna(0) if peer else 0.0
    d["_fin"] = pd.to_numeric(d[fin], errors="coerce").fillna(0) if fin else 0.0
    d["_stat"] = pd.to_numeric(d[stat], errors="coerce").fillna(0) if stat else 0.0
    d["_ratio"] = pd.to_numeric(d[ratio], errors="coerce").fillna(0) if ratio else 0.0
    d["_dev"] = pd.to_numeric(d[deviation], errors="coerce").fillna(0) if deviation else 0.0
    d["_pct"] = pd.to_numeric(d[percentile], errors="coerce").fillna(0) if percentile else 0.0
    if level:
        d["_level_raw"] = d[level].astype(str).str.upper()
    else:
        d["_level_raw"] = d["_risk"].map(lambda x: "HIGH" if x >= 60 else "MEDIUM" if x >= 30 else "LOW")
    if record_type:
        d = d[d[record_type].astype(str).str.upper().eq("MP_RECORD")].copy()
    else:
        d = d[d["_sr"].notna() & (d["_sr"] > 0)].copy()
    d["_bucket"] = d["_risk"].map(risk_bucket)
    d["_class"] = d["_bucket"].map(risk_class)
else:
    d = None

# ----------------------------- Friendly styling -----------------------------
st.markdown("""
<style>
.stApp { background:#f6f8fb; }
.block-container { max-width:1450px; padding:1.4rem 2rem 3rem; }
.hero { background:linear-gradient(135deg,#0b2a4a,#145da0); color:white; border-radius:18px; padding:28px 32px; margin-bottom:18px; }
.hero h1 { margin:0; font-size:30px; letter-spacing:-.4px; }
.hero p { margin:8px 0 0; color:#dcecff; font-size:14px; max-width:900px; }
.hero .tag { display:inline-block; margin-top:15px; padding:6px 10px; border-radius:20px; background:rgba(255,255,255,.12); font-size:11px; font-weight:700; }
.nav-note { color:#667085; font-size:13px; margin:4px 0 18px; }
.card { background:white; border:1px solid #e5eaf0; border-radius:15px; padding:18px; box-shadow:0 4px 14px rgba(15,23,42,.04); }
.card h3 { margin:0 0 4px; color:#17324d; font-size:16px; }
.card p { color:#667085; font-size:12px; margin:0; line-height:1.5; }
.big { font-size:28px; font-weight:800; color:#0b2745; margin:6px 0; }
.answer { background:#edf6ff; border:1px solid #cfe6ff; border-radius:14px; padding:17px 19px; color:#29445f; margin:12px 0; }
.answer strong { color:#0b4f91; }
.simple-step { background:white; border:1px solid #e5eaf0; border-radius:14px; padding:16px; min-height:115px; }
.step-num { width:30px; height:30px; border-radius:50%; background:#e7f1ff; color:#1264ad; display:flex; align-items:center; justify-content:center; font-weight:800; margin-bottom:10px; }
.step-title { font-weight:800; color:#17324d; font-size:14px; }
.step-text { color:#667085; font-size:12px; line-height:1.45; margin-top:4px; }
.alert-card { border-radius:15px; padding:18px; border:1px solid #f0d4d0; background:#fff7f6; }
.alert-card.amber { border-color:#f2dfb4; background:#fffbf1; }
.alert-card.green { border-color:#cfe8d8; background:#f4fbf6; }
.label { color:#667085; font-size:11px; text-transform:uppercase; letter-spacing:.4px; font-weight:700; }
.value { color:#0b2745; font-size:22px; font-weight:800; margin-top:4px; }
.footer { color:#7b8794; font-size:11px; border-top:1px solid #e5eaf0; padding-top:15px; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏛️ MPLADS Review Assistant</h1>
  <p>A simple dashboard that helps reviewers find unusual allocation records first, understand why they were highlighted, and decide what should be verified.</p>
  <span class="tag">AI-ASSISTED • HUMAN VERIFICATION REQUIRED</span>
</div>
""", unsafe_allow_html=True)

if d is None or d.empty:
    st.error("MPLADS allocation results could not be found. Please place your processed CSV files in the project's processed folder.")
    st.stop()

# Friendly navigation
view = st.radio("", ["🏠 Overview", "🔎 Find records to check", "📋 Review cases", "🌪️ Calamity funds", "ℹ️ How it works"], horizontal=True, label_visibility="collapsed")

# ----------------------------- Overview -----------------------------
if view == "🏠 Overview":
    total = len(d)
    states = d["_state"].nunique()
    amount_total = d["_amount"].sum()
    high = int((d["_risk"] >= 60).sum())
    cases = raw["cases"]
    case_count = len(cases) if cases is not None else 0

    st.markdown("### What should a reviewer know first?")
    st.markdown("<div class='nav-note'>Start here. This page gives the overall picture without requiring any technical knowledge.</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    cards=[("Records reviewed",f"{total:,}","MP allocation records"),("States / UTs",f"{states:,}","Across the available data"),("Allocation amount",money(amount_total),"Total of MP-level records"),("Needs attention",f"{high:,}","Higher-priority review candidates")]
    for c,(lab,val,sub) in zip([c1,c2,c3,c4],cards):
        c.markdown(f"<div class='card'><div class='label'>{lab}</div><div class='big'>{val}</div><p>{sub}</p></div>",unsafe_allow_html=True)

    st.markdown("### How to read this dashboard")
    a,b,c = st.columns(3)
    for box,title,text,n in [(a,"1. Find unusual records","The system screens all available allocation records and brings the unusual ones to the top.","1"),(b,"2. Understand the reason","Open a record to see the signals behind the alert in plain language.","2"),(c,"3. Verify with official records","An alert is a prompt for human review — not a finding of fraud.","3")]:
        box.markdown(f"<div class='simple-step'><div class='step-num'>{n}</div><div class='step-title'>{title}</div><div class='step-text'>{text}</div></div>",unsafe_allow_html=True)

    st.markdown("### Records needing the most attention")
    top=d.sort_values("_risk",ascending=False).head(5)
    for _,r in top.iterrows():
        label=r["_bucket"]
        cls=r["_class"]
        left,right=st.columns([3.5,1])
        with left:
            st.markdown(f"<div class='alert-card {'' if cls=='red' else cls}'><div style='font-weight:800;color:#17324d;font-size:16px'>{r['_mp']}</div><div style='color:#667085;font-size:12px'>{r['_state']} • {r['_const']} • {money(r['_amount'])}</div><div style='margin-top:8px;color:#425466;font-size:12px'>The system found several unusual signals. Open <b>Find records to check</b> to see the reasons.</div></div>",unsafe_allow_html=True)
        with right:
            st.metric("Review level",label)
            st.metric("Risk score",f"{r['_risk']:.2f}")

# ----------------------------- Find records -----------------------------
elif view == "🔎 Find records to check":
    st.markdown("### Find records to check")
    st.markdown("<div class='nav-note'>You do not need to understand the algorithms. Start with a name, state, or review level.</div>",unsafe_allow_html=True)
    f1,f2,f3=st.columns([1.2,1,2])
    with f1: state_choice=st.selectbox("State / UT",["All"]+sorted(d["_state"].unique().tolist()))
    with f2: level_choice=st.selectbox("Review level",["All","Needs attention","Worth checking","No major signal"])
    with f3: query=st.text_input("Search",placeholder="MP, constituency, or state")
    x=d.copy()
    if state_choice!="All": x=x[x["_state"]==state_choice]
    if level_choice!="All": x=x[x["_bucket"]==level_choice]
    if query.strip():
        q=query.strip().lower(); x=x[x.apply(lambda r:q in str(r["_mp"]).lower() or q in str(r["_state"]).lower() or q in str(r["_const"]).lower(),axis=1)]
    x=x.sort_values("_risk",ascending=False)
    st.write(f"**{len(x):,} records** match your selection.")
    display=pd.DataFrame({"MP":x["_mp"],"State":x["_state"],"Constituency":x["_const"],"Allocation":x["_amount"].map(money),"Review level":x["_bucket"],"Risk score":x["_risk"].round(2)})
    st.dataframe(display,use_container_width=True,hide_index=True)
    if len(x):
        options=x.index.tolist()
        pick=st.selectbox("Open a record",options,format_func=lambda i:f"{x.loc[i,'_mp']} — {x.loc[i,'_bucket']} — {x.loc[i,'_risk']:.2f}")
        r=x.loc[pick]
        st.markdown("---")
        st.markdown(f"## {r['_mp']}")
        st.caption(f"{r['_state']} • {r['_const']}")
        a,b,c=st.columns(3)
        a.markdown(f"<div class='card'><div class='label'>Allocation</div><div class='value'>{money(r['_amount'])}</div><p>Amount in the available allocation dataset.</p></div>",unsafe_allow_html=True)
        b.markdown(f"<div class='card'><div class='label'>Review level</div><div class='value'>{r['_bucket']}</div><p>This is a screening priority, not a fraud finding.</p></div>",unsafe_allow_html=True)
        c.markdown(f"<div class='card'><div class='label'>Risk score</div><div class='value'>{r['_risk']:.2f} / 100</div><p>Combined score from the available signals.</p></div>",unsafe_allow_html=True)

        st.markdown("### Why was this record highlighted?")
        reasons=[]
        if r['_ml']>=50: reasons.append(f"The allocation shows an unusual pattern in the AI screening (signal {r['_ml']:.0f}/100).")
        if r['_peer']>=40:
            ratio_txt=f"about {r['_ratio']:.2f}×" if r['_ratio']>0 else "noticeably"
            dev_txt=f" ({r['_dev']:.1f}% above the peer median)" if r['_dev']>0 else ""
            reasons.append(f"Compared with similar records, the allocation is {ratio_txt} the peer median{dev_txt}.")
        if r['_fin']>=75: reasons.append("The allocation is in a financially significant upper range of the available records.")
        if r['_stat']>0 or r['_pct']>=95: reasons.append("A statistical check also places the record toward the extreme end of the distribution.")
        if not reasons: reasons.append("The combined screening score is elevated, but no single explanation signal is dominant in the available fields.")
        for reason in reasons: st.markdown(f"• {reason}")
        st.markdown("<div class='answer'><strong>What should happen next?</strong><br>Verify the underlying official allocation/supporting records before drawing any conclusion.</div>",unsafe_allow_html=True)

# ----------------------------- Cases -----------------------------
elif view == "📋 Review cases":
    st.markdown("### Review cases")
    st.markdown("<div class='nav-note'>These are the records your AI pipeline has already converted into a review queue.</div>",unsafe_allow_html=True)
    cases=raw["cases"]
    if cases is None or cases.empty:
        st.info("No investigation_cases.csv was found. Run the case generator first.")
    else:
        cases=cases.copy(); member=col(cases,["member","mp","Hon'ble Members of Parliaments","Member"]); state=col(cases,["state","State"]); const=col(cases,["constituency","Constituency"]); amount=col(cases,["amount","allocated_amount","Allocated AMOUNT"]); score=col(cases,["unified_risk_score","risk_score"]); priority=col(cases,["priority","Priority"]); evidence=col(cases,["evidence_strength","strength","Evidence"]); cid=col(cases,["case_id","Case ID"])
        s=pd.to_numeric(cases[score],errors="coerce").fillna(0) if score else pd.Series(0,index=cases.index)
        c1,c2,c3=st.columns(3); c1.metric("Cases to review",len(cases)); c2.metric("High priority",int(cases[priority].astype(str).str.upper().eq("HIGH").sum()) if priority else 0); c3.metric("Strong evidence coverage",int(cases[evidence].astype(str).str.upper().eq("STRONG").sum()) if evidence else 0)
        cases["_score"]=s; cases=cases.sort_values("_score",ascending=False)
        show=pd.DataFrame({"Case":cases[cid] if cid else [f"CASE-{i+1:04d}" for i in range(len(cases))],"MP":cases[member] if member else "Unknown","State":cases[state] if state else "Unknown","Amount":num(cases[amount]).map(money) if amount else "—","Review level":cases[priority] if priority else cases["_score"].map(risk_bucket),"Risk score":cases["_score"].round(2)})
        st.dataframe(show,use_container_width=True,hide_index=True)
        pick=st.selectbox("Open a case",cases.index,format_func=lambda i:f"{cases.loc[i,member] if member else 'Case'} — {cases.loc[i,'_score']:.2f}")
        r=cases.loc[pick]
        st.markdown("---")
        st.markdown(f"## {r[cid] if cid else 'Review case'}")
        st.markdown(f"### {r[member] if member else 'Unknown MP'}")
        st.caption(f"{r[state] if state else 'Unknown'} • {r[const] if const else 'Unknown'}")
        a,b,c=st.columns(3); a.metric("Risk score",f"{r['_score']:.2f}"); b.metric("Review level",str(r[priority]) if priority else risk_bucket(r['_score'])); c.metric("Evidence",str(r[evidence]) if evidence else "—")
        st.markdown("### Reviewer message")
        st.markdown("<div class='answer'><strong>This is an AI review candidate.</strong><br>The system found enough unusual signals to place this record in the review queue. Check the underlying official records before making any conclusion.</div>",unsafe_allow_html=True)
        for label,names in [("Signals detected",["signals","signal_summary","Signal"]),("Recommended next step",["recommended_action","actions","Actions"]),("Status",["status","Status"] )]:
            cc=col(cases,names)
            if cc: st.write(f"**{label}:** {r[cc]}")

# ----------------------------- Calamity -----------------------------
elif view == "🌪️ Calamity funds":
    st.markdown("### Calamity funds")
    st.markdown("<div class='nav-note'>A separate view of the MPLADS calamity-consent dataset that is currently available.</div>",unsafe_allow_html=True)
    cal=raw["calamity"]
    if cal is None or cal.empty:
        st.warning("Calamity data could not be located. The app searched the dashboard/data, project/data, project/processed, and project folders for the MPLADS calamity CSV.")
    else:
        cal=cal.copy(); name=col(cal,["Calamity Name","calamity_name","Calamity"]); typ=col(cal,["Calamity Type","calamity_type","Type"]); mp=col(cal,["Hon'ble Members of Parliament","Hon'ble Members of Parliaments","mp_name","Member"]); date=col(cal,["Date of Consent","consent_date","Date"]); amount=col(cal,["Consent Amount ( ₹ )","consent_amount","amount"])
        cal["_amount"]=num(cal[amount]) if amount else 0
        if name:
            text=cal[name].astype(str).str.strip().str.lower()
            mask=text.eq("")|text.isin(["grand total","total","nan","none"])|text.str.contains("grand\\s*total",regex=True,na=False)
            # Catch a nameless aggregate row whose amount equals the detail sum.
            detail_sum=cal.loc[~mask,"_amount"].sum()
            mask=mask | (cal[name].astype(str).str.strip().eq("") & (cal["_amount"]-detail_sum).abs().lt(.01))
            cal=cal.loc[~mask].copy()
        a,b,c=st.columns(3); a.metric("Consent records",len(cal)); b.metric("Total consent",money(cal['_amount'].sum())); c.metric("Calamity types",int(cal[name].nunique()) if name else 0)
        left,right=st.columns([1.35,1])
        with left:
            if name:
                agg=cal.groupby(name)['_amount'].sum().sort_values().reset_index(); agg['Crore']=agg['_amount']/1e7; agg['Label']=agg['Crore'].map(lambda x:f"₹{x:.2f} Cr")
                fig=px.bar(agg,x='Crore',y=name,orientation='h',text='Label'); fig.update_traces(textposition='outside',cliponaxis=False); fig.update_layout(height=max(400,70*len(agg)+90),margin=dict(l=180,r=90,t=25,b=50),showlegend=False); fig.update_xaxes(title='Consent amount (₹ Crore)'); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        with right:
            if typ:
                typdf=cal.groupby(typ)['_amount'].sum().reset_index(); typdf['Crore']=typdf['_amount']/1e7
                fig=px.pie(typdf,names=typ,values='Crore',hole=.55); fig.update_layout(height=440,margin=dict(l=10,r=10,t=25,b=10)); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        out=pd.DataFrame();
        if mp: out['MP']=cal[mp]
        if name: out['Calamity']=cal[name]
        if typ: out['Type']=cal[typ]
        if date: out['Consent date']=cal[date]
        out['Consent amount']=cal['_amount'].map(money_full)
        st.dataframe(out,use_container_width=True,hide_index=True)

# ----------------------------- How it works -----------------------------
else:
    st.markdown("### How it works")
    st.markdown("<div class='nav-note'>You can use the dashboard without knowing machine learning. This page explains the logic in everyday language.</div>",unsafe_allow_html=True)
    steps=[("1","Look for unusual patterns","The AI screens the available MP allocation records and looks for values that stand out."),("2","Compare similar records","The system compares records with relevant peer groups instead of looking only at the whole dataset."),("3","Check the financial size","Large allocations receive more attention because the financial exposure is greater."),("4","Use statistical checks","A separate statistical check looks for observations at the extreme end of the available distribution."),("5","Combine the signals","The independent signals are combined into one screening score from 0 to 100."),("6","Ask a human to verify","The result is a prioritization tool. It does not determine fraud or wrongdoing.")]
    for i in range(0,len(steps),2):
        a,b=st.columns(2)
        for box,item in zip([a,b],steps[i:i+2]):
            n,title,text=item; box.markdown(f"<div class='simple-step' style='margin-bottom:12px'><div class='step-num'>{n}</div><div class='step-title'>{title}</div><div class='step-text'>{text}</div></div>",unsafe_allow_html=True)
    st.markdown("### What data is currently available?")
    st.markdown("<div class='answer'><strong>Current prototype:</strong> MP allocation records plus a separate calamity-consent dataset. Project execution, payment, GPS, image and progress fields are not claimed when they are not present in the available source data.</div>",unsafe_allow_html=True)

st.markdown("<div class='footer'>MPLADS Review Assistant • Prototype decision-support system • AI alerts are for human verification only.</div>",unsafe_allow_html=True)
