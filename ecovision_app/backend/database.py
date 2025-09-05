# backend/database.py
from __future__ import annotations

import os
from datetime import datetime, date
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# ORM models (do NOT create their own engine/session)
from backend.models import EconomicIndicator, Country

# =============================================================================
# Single source of truth: Engine + SessionLocal
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ecovision.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False → SQLite + FastAPI threadpool compatibility
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# This is the ONLY SessionLocal the whole project should use
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# =============================================================================
# Dependency to get DB session (FastAPI)
# =============================================================================
def get_db():
    """
    FastAPI dependency that yields a scoped Session and closes it afterwards.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# Country CRUD
# =============================================================================
def get_countries(db: Session) -> List[Country]:
    return db.query(Country).all()

def get_country_by_code(db: Session, country_code: str) -> Optional[Country]:
    return db.query(Country).filter(Country.country_code == country_code).first()

def add_countries(db: Session, countries: List[dict]):
    """
    Upsert list of countries. Only provided fields are updated on conflicts.
    """
    inserted = 0
    updated = 0

    for payload in countries:
        existing = db.query(Country).filter(Country.country_code == payload["country_code"]).first()
        if existing:
            for k, v in payload.items():
                setattr(existing, k, v)
            updated += 1
        else:
            db.add(Country(**payload))
            inserted += 1

    db.commit()
    return {"message": f"{inserted} countries inserted, {updated} updated successfully"}

# =============================================================================
# Economic Indicator CRUD
# =============================================================================
def get_indicators(db: Session, limit: Optional[int] = None) -> List[EconomicIndicator]:
    q = db.query(EconomicIndicator)
    if limit:
        q = q.limit(limit)
    return q.all()

def filter_indicators(
    db: Session,
    indicator_code: Optional[str] = None,
    country_code: Optional[str] = None,
    year: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[EconomicIndicator]:
    """
    Flexible filter for indicators with optional sorting and limiting.
    sort supports field names on EconomicIndicator, prefix with '-' for desc.
    """
    q = db.query(EconomicIndicator)

    if indicator_code:
        q = q.filter(EconomicIndicator.indicator_code == indicator_code)
    if country_code:
        q = q.filter(EconomicIndicator.country_code == country_code)
    if year is not None:
        q = q.filter(EconomicIndicator.year == year)
    if year_from is not None:
        q = q.filter(EconomicIndicator.year >= year_from)
    if year_to is not None:
        q = q.filter(EconomicIndicator.year <= year_to)

    if sort:
        reverse = sort.startswith("-")
        field = sort.lstrip("-")
        if hasattr(EconomicIndicator, field):
            col = getattr(EconomicIndicator, field)
            q = q.order_by(col.desc() if reverse else col)

    if limit:
        q = q.limit(limit)

    return q.all()

def get_indicator_series(
    db: Session,
    codes: List[str],
    country_code: str,
    year_from: int,
    year_to: int,
) -> List[EconomicIndicator]:
    """
    Return ordered rows for a set of indicator codes for one country and range.
    """
    return (
        db.query(EconomicIndicator)
        .filter(EconomicIndicator.country_code == country_code)
        .filter(EconomicIndicator.indicator_code.in_(codes))
        .filter(EconomicIndicator.year >= year_from)
        .filter(EconomicIndicator.year <= year_to)
        .order_by(EconomicIndicator.indicator_code, EconomicIndicator.year)
        .all()
    )

def bulk_create_indicators(db: Session, new_records: List[dict]) -> List[EconomicIndicator]:
    """
    Insert a list of indicator dicts.
    - Auto-fills country_name/region when country_code is present.
    - Accepts last_updated as 'YYYY-MM-DD' or date object.
    """
    enriched: List[EconomicIndicator] = []

    for record in new_records:
        # Auto-fill from country_code (or region as fallback)
        country = None
        if record.get("country_code"):
            country = db.query(Country).filter(Country.country_code == record["country_code"]).first()
        elif record.get("region"):
            country = db.query(Country).filter(Country.region == record["region"]).first()

        if country:
            record.setdefault("country_name", country.country_name)
            record.setdefault("region", country.region)

        # Normalize last_updated
        if isinstance(record.get("last_updated"), str):
            try:
                record["last_updated"] = datetime.strptime(record["last_updated"], "%Y-%m-%d").date()
            except Exception:
                record["last_updated"] = date.today()

        enriched.append(EconomicIndicator(**record))

    db.add_all(enriched)
    db.commit()
    return enriched

# --- Region-average yearly series for a single indicator ---
def get_indicator_series_region_avg(
    db: Session,
    indicator_code: str,
    region: str,
    year_from: int,
    year_to: int,
):
    """
    Returns: [{"year": 2000, "value": 1.23}, ...]
    'value' is the average across all countries in the region for that year
    (ignoring nulls). If all values are null in a year, that year is omitted.
    """
    from sqlalchemy import func

    rows = (
        db.query(
            EconomicIndicator.year.label("year"),
            func.avg(EconomicIndicator.value).label("avg_value"),
        )
        .filter(EconomicIndicator.indicator_code == indicator_code)
        .filter(EconomicIndicator.region == region)
        .filter(EconomicIndicator.year >= year_from)
        .filter(EconomicIndicator.year <= year_to)
        .group_by(EconomicIndicator.year)
        .order_by(EconomicIndicator.year.asc())
        .all()
    )

    out = []
    for y, v in rows:
        if v is not None:
            out.append({"year": int(y), "value": float(v)})
    return out
