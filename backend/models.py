from sqlalchemy import Column, Integer, String, Float, Date
from .db import Base

class MPAllocation(Base):
    __tablename__ = "mp_allocations"

    id = Column(Integer, primary_key=True)
    sr_no = Column(Integer, nullable=True)
    state = Column(String, index=True)
    mp_name = Column(String, index=True)
    constituency = Column(String, index=True)
    allocated_amount = Column(Float, nullable=True)
    is_anomaly = Column(Integer, default=0)
    anomaly_score = Column(Float, nullable=True)
    data_quality_score = Column(Float, default=100.0)

class CalamityConsent(Base):
    __tablename__ = "calamity_consents"

    id = Column(Integer, primary_key=True)
    sr_no = Column(Integer, nullable=True)
    calamity_type = Column(String, index=True)
    calamity_name = Column(String, index=True)
    mp_name = Column(String, index=True)
    consent_date = Column(Date, nullable=True)
    consent_amount = Column(Float, nullable=True)
    data_quality_score = Column(Float, default=100.0)
