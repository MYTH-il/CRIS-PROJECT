from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from datetime import datetime
from .db import Base


class CaseRecord(Base):
    __tablename__ = "case_records"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(32), index=True)
    fir_number = Column(String(64), index=True)
    report_date = Column(String(16), index=True)
    crime_type = Column(String(64), index=True)
    incident_description = Column(Text)
    incident_location = Column(String(256))
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="analyst")
    created_at = Column(DateTime, default=datetime.utcnow)


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(128))
    method = Column(String(8))
    user = Column(String(64))
    status = Column(Integer)
    duration_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
