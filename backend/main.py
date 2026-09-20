from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from .db import Base, engine, get_db
from .models import MPAllocation, CalamityConsent

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MPLADS AI Monitoring API",
    version="0.1.0",
    description="Phase-1 analytics API for MPLADS datasets."
)

@app.get("/")
def root():
    return {
        "name": "MPLADS AI Monitoring API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "mp_allocation_records": db.query(MPAllocation).count(),
        "calamity_records": db.query(CalamityConsent).count()
    }

@app.get("/api/summary")
def summary(db: Session = Depends(get_db)):
    mp_count = db.query(MPAllocation).count()
    state_count = db.query(MPAllocation.state).distinct().count()
    total_allocated = db.query(func.sum(MPAllocation.allocated_amount)).scalar() or 0
    calamity_count = db.query(CalamityConsent).count()
    calamity_total = db.query(func.sum(CalamityConsent.consent_amount)).scalar() or 0
    anomalies = db.query(MPAllocation).filter(MPAllocation.is_anomaly == 1).count()
    avg_quality = db.query(func.avg(MPAllocation.data_quality_score)).scalar() or 0

    return {
        "mp_records": mp_count,
        "states": state_count,
        "total_allocated": total_allocated,
        "calamity_records": calamity_count,
        "calamity_total": calamity_total,
        "allocation_anomalies": anomalies,
        "avg_data_quality": avg_quality
    }

@app.get("/api/states")
def states(db: Session = Depends(get_db)):
    rows = (
        db.query(
            MPAllocation.state,
            func.count(MPAllocation.id).label("mp_count"),
            func.sum(MPAllocation.allocated_amount).label("allocated_amount"),
            func.avg(MPAllocation.data_quality_score).label("data_quality")
        )
        .group_by(MPAllocation.state)
        .order_by(func.sum(MPAllocation.allocated_amount).desc())
        .all()
    )
    return [
        {
            "state": r.state,
            "mp_count": r.mp_count,
            "allocated_amount": r.allocated_amount or 0,
            "data_quality": round(r.data_quality or 0, 2)
        }
        for r in rows
    ]

@app.get("/api/mps")
def mps(
    search: str = Query("", description="Search MP or constituency"),
    state: str = Query("", description="Exact state filter"),
    only_anomalies: bool = False,
    db: Session = Depends(get_db)
):
    q = db.query(MPAllocation)

    if search:
        term = f"%{search.strip()}%"
        q = q.filter(
            (MPAllocation.mp_name.ilike(term)) |
            (MPAllocation.constituency.ilike(term))
        )

    if state:
        q = q.filter(MPAllocation.state == state)

    if only_anomalies:
        q = q.filter(MPAllocation.is_anomaly == 1)

    rows = q.order_by(MPAllocation.allocated_amount.desc()).limit(1000).all()

    return [
        {
            "id": r.id,
            "state": r.state,
            "mp_name": r.mp_name,
            "constituency": r.constituency,
            "allocated_amount": r.allocated_amount,
            "is_anomaly": bool(r.is_anomaly),
            "anomaly_score": round(r.anomaly_score or 0, 2),
            "data_quality_score": r.data_quality_score
        }
        for r in rows
    ]

@app.get("/api/calamities")
def calamities(db: Session = Depends(get_db)):
    rows = (
        db.query(
            CalamityConsent.calamity_name,
            func.count(CalamityConsent.id).label("mp_count"),
            func.sum(CalamityConsent.consent_amount).label("consent_amount")
        )
        .group_by(CalamityConsent.calamity_name)
        .order_by(func.sum(CalamityConsent.consent_amount).desc())
        .all()
    )
    return [
        {
            "calamity_name": r.calamity_name,
            "mp_count": r.mp_count,
            "consent_amount": r.consent_amount or 0
        }
        for r in rows
    ]
