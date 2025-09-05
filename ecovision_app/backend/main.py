# backend/main.py
from __future__ import annotations

import asyncio
from datetime import date
from typing import List, Optional, Dict, AsyncGenerator
from collections import Counter

from fastapi import FastAPI, Depends, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, text as sqltext
from apscheduler.schedulers.background import BackgroundScheduler

#  Single source of truth: import SessionLocal/get_db from database.py only
from backend import database
from backend.database import get_db, SessionLocal
from backend.real_api import fetch_and_update_all
from backend.models import init_db, EconomicIndicator, Country

# ---------- AI helpers ----------
from backend.ai_chat import ask as ai_ask, stream_ask as ai_stream_ask
from backend.ai_chat import OLLAMA_URL as AI_OLLAMA_URL, MODEL as AI_MODEL
from backend.ai_rag import (
    rebuild_index as rag_rebuild_index,
    answer_with_data as rag_answer,
    search_docs as rag_search,
)

# ---------- Forecasting ----------
from backend.forecasting import forecast_country_indicator

# =============================================================================
# FastAPI app + CORS
# =============================================================================
app = FastAPI(title="EcoVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pydantic models
# =============================================================================
class IndicatorOut(BaseModel):
    indicator_code: str
    indicator_name: Optional[str] = None
    country_code: str
    country_name: Optional[str] = None
    region: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    year: int
    quarter: Optional[str] = None
    data_source: Optional[str] = None
    last_updated: Optional[date] = None
    category: Optional[str] = None
    model_config = {"from_attributes": True}

class LatestItem(BaseModel):
    year: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None

class LatestSnapshot(BaseModel):
    GDP_VALUE: LatestItem
    GDP_GROWTH: LatestItem
    CPI: LatestItem
    UNEMPLOYMENT: LatestItem
    TRADE_BALANCE: LatestItem

class SeriesPoint(BaseModel):
    year: int
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None

class IndicatorSeries(BaseModel):
    indicator_code: str
    indicator_name: Optional[str] = None
    points: List[SeriesPoint]

# =============================================================================
# Debug endpoints
# =============================================================================
@app.get("/api/debug/db-path")
def debug_db_path(db: Session = Depends(get_db)):
    """Return absolute SQLite file path used by this running process."""
    row = db.execute(sqltext("PRAGMA database_list;")).fetchone()
    return {"sqlite_path": row[2] if row else None}

@app.get("/api/debug/ai-docs")
def debug_ai_docs(db: Session = Depends(get_db)):
    """List tables and show ai_docs row count if present."""
    tables = db.execute(sqltext(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )).fetchall()
    table_names = [t[0] for t in tables]
    ai_docs_count = None
    if "ai_docs" in table_names:
        ai_docs_count = db.execute(sqltext("SELECT COUNT(*) FROM ai_docs")).fetchone()[0]
    return {"tables": table_names, "ai_docs_count": ai_docs_count}

@app.get("/api/health")
def api_health(db: Session = Depends(get_db)):
    """Quick green/red status for data + RAG index."""
    total = db.query(EconomicIndicator).count()
    countries = db.query(Country).count()
    tables = [t[0] for t in db.execute(sqltext(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )).fetchall()]
    ai_count = None
    if "ai_docs" in tables:
        ai_count = db.execute(sqltext("SELECT COUNT(*) FROM ai_docs")).fetchone()[0]
    return {
        "ok": (total > 0 and countries > 0 and (ai_count or 0) > 0),
        "countries": countries,
        "indicators": total,
        "ai_docs": ai_count,
    }

# =============================================================================
# Countries & Regions
# =============================================================================
@app.get("/api/countries")
def list_countries(db: Session = Depends(get_db)):
    return database.get_countries(db)

@app.get("/api/countries/{country_code}")
def get_country(country_code: str, db: Session = Depends(get_db)):
    country = database.get_country_by_code(db, country_code.upper())
    if not country:
        raise HTTPException(status_code=404, detail="Country not found")
    return country

@app.post("/api/countries/bulk")
def bulk_insert_countries(new_countries: List[dict], db: Session = Depends(get_db)):
    if not new_countries:
        raise HTTPException(status_code=400, detail="No country data provided")
    return database.add_countries(db, new_countries)

@app.get("/api/regions")
def list_regions(db: Session = Depends(get_db)):
    rows = (
        db.query(Country.region, func.count(Country.id))
        .filter(Country.region.isnot(None))
        .group_by(Country.region)
        .all()
    )
    return [{"region": r, "countries": c} for r, c in rows]

# =============================================================================
# Indicators
# =============================================================================
@app.get("/api/indicators", response_model=List[IndicatorOut])
def list_indicators(
    indicator_code: Optional[str] = None,
    country_code: Optional[str] = None,
    year: Optional[int] = None,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
    db: Session = Depends(get_db),
):
    return database.filter_indicators(
        db,
        indicator_code=indicator_code,
        country_code=country_code,
        year=year,
        sort=sort,
        limit=limit,
    )

@app.post("/api/indicators/bulk")
def bulk_insert_indicators(new_records: List[dict], db: Session = Depends(get_db)):
    if not new_records:
        raise HTTPException(status_code=400, detail="No indicator data provided")
    return database.bulk_create_indicators(db, new_records)

@app.get("/api/debug/counts")
def debug_counts(db: Session = Depends(get_db)):
    total = db.query(EconomicIndicator).count()
    by_country = (
        db.query(EconomicIndicator.country_code, func.count(EconomicIndicator.id))
        .group_by(EconomicIndicator.country_code)
        .all()
    )
    return {"total_records": total, "by_country": {cc: cnt for cc, cnt in by_country}}

# =============================================================================
# Latest snapshot (filterable by country or region)
# =============================================================================
@app.get("/api/indicators/latest", response_model=LatestSnapshot)
def latest_snapshot(
    country_code: Optional[str] = None,
    region: Optional[str] = None,
    db: Session = Depends(get_db),
):
    desired_units = {
        "GDP_VALUE": "USD",
        "GDP_GROWTH": "%",
        "CPI": "%",
        "UNEMPLOYMENT": "%",
        "TRADE_BALANCE": "% of GDP",
    }
    codes = ["GDP_VALUE", "GDP_GROWTH", "CPI", "UNEMPLOYMENT", "TRADE_BALANCE"]

    def latest_for_country(code: str, cc: str) -> LatestItem:
        row = (
            db.query(EconomicIndicator)
            .filter(
                EconomicIndicator.country_code == cc,
                EconomicIndicator.indicator_code == code,
                EconomicIndicator.value.isnot(None),
            )
            .order_by(desc(EconomicIndicator.year))
            .first()
        )
        if not row:
            return LatestItem(year=None, value=None, unit=desired_units[code], source=None)
        return LatestItem(
            year=row.year,
            value=row.value,
            unit=row.unit or desired_units[code],
            source=row.data_source,
        )

    def latest_for_region_avg(code: str, reg: str) -> LatestItem:
        base = db.query(EconomicIndicator).filter(
            EconomicIndicator.region == reg,
            EconomicIndicator.indicator_code == code,
        )
        latest_year = base.with_entities(func.max(EconomicIndicator.year)).scalar()
        if latest_year is None:
            return LatestItem(year=None, value=None, unit=desired_units[code], source=None)

        rows = (
            base.filter(EconomicIndicator.year == latest_year)
            .with_entities(EconomicIndicator.value, EconomicIndicator.unit, EconomicIndicator.data_source)
            .all()
        )
        values = [r[0] for r in rows if r[0] is not None]
        if not values:
            return LatestItem(year=latest_year, value=None, unit=desired_units[code], source=None)

        avg_val = sum(values) / len(values)
        units = [r[1] for r in rows if r[1]]
        unit = Counter(units).most_common(1)[0][0] if units else desired_units[code]
        sources = [r[2] for r in rows if r[2]]
        source = ", ".join(sorted(set(", ".join(sources).split(", ")))) if sources else None
        return LatestItem(year=latest_year, value=avg_val, unit=unit, source=source)

    cc = (country_code or "").upper().strip()
    reg = (region or "").strip()

    def get(code: str) -> LatestItem:
        if cc:
            return latest_for_country(code, cc)
        if reg:
            return latest_for_region_avg(code, reg)
        return latest_for_country(code, "DE")

    return LatestSnapshot(
        GDP_VALUE=get("GDP_VALUE"),
        GDP_GROWTH=get("GDP_GROWTH"),
        CPI=get("CPI"),
        UNEMPLOYMENT=get("UNEMPLOYMENT"),
        TRADE_BALANCE=get("TRADE_BALANCE"),
    )

# =============================================================================
# Root
# =============================================================================
@app.get("/")
def root():
    return {"message": "EcoVision API is running with SQLite backend"}

# =============================================================================
# Scheduler
# =============================================================================
scheduler = BackgroundScheduler()

def fetch_job():
    fetch_and_update_all()

def rag_rebuild_job():
    db = SessionLocal()
    try:
        asyncio.run(rag_rebuild_index(db, year_from=1990, year_to=2025))
    finally:
        db.close()

@app.on_event("startup")
def on_startup():
    init_db()
    # Ensure DB is populated at least once on boot
    fetch_and_update_all()

    if not scheduler.get_jobs():
        scheduler.add_job(fetch_job, "interval", hours=5, id="refresh_job", replace_existing=True)
        scheduler.add_job(
            rag_rebuild_job,
            trigger="cron",
            hour="2",
            minute="15",
            id="rag_rebuild_job",
            replace_existing=True,
        )
    scheduler.start()

@app.on_event("shutdown")
def on_shutdown():
    if scheduler.running:
        scheduler.shutdown()

# =============================================================================
# Series
# =============================================================================
@app.get("/api/indicators/series", response_model=Dict[str, IndicatorSeries])
def all_indicator_series(
    year_from: int = 2002,
    year_to: int = 2025,
    country_code: Optional[str] = None,
    region: Optional[str] = None,
    db: Session = Depends(get_db),
):
    codes = ["GDP_VALUE", "GDP_GROWTH", "CPI", "UNEMPLOYMENT", "TRADE_BALANCE"]

    base_q = (
        db.query(EconomicIndicator)
        .filter(EconomicIndicator.indicator_code.in_(codes))
        .filter(EconomicIndicator.year >= year_from)
        .filter(EconomicIndicator.year <= year_to)
    )

    q = base_q.filter(EconomicIndicator.region == region) if region else \
        base_q.filter(EconomicIndicator.country_code == (country_code or "DE"))

    rows = q.order_by(
        EconomicIndicator.indicator_code,
        EconomicIndicator.country_code,
        EconomicIndicator.year,
    ).all()

    if region:
        agg_vals: dict[str, dict[int, List[float]]] = {}
        name_map: dict[str, Optional[str]] = {}
        unit_map: dict[str, Optional[str]] = {}

        for r in rows:
            name_map.setdefault(r.indicator_code, r.indicator_name)
            unit_map.setdefault(r.indicator_code, r.unit)
            if r.value is None:
                continue
            agg_vals.setdefault(r.indicator_code, {}).setdefault(r.year, []).append(float(r.value))

        out: Dict[str, IndicatorSeries] = {}
        for code in codes:
            pts: List[SeriesPoint] = []
            for y in range(year_from, year_to + 1):
                vals = agg_vals.get(code, {}).get(y) or []
                v = (sum(vals) / len(vals)) if vals else None
                pts.append(SeriesPoint(year=y, value=v, unit=unit_map.get(code)))
            out[code] = IndicatorSeries(
                indicator_code=code,
                indicator_name=name_map.get(code),
                points=pts,
            )
        return out

    out: Dict[str, IndicatorSeries] = {}
    for r in rows:
        out.setdefault(
            r.indicator_code,
            IndicatorSeries(indicator_code=r.indicator_code, indicator_name=r.indicator_name, points=[]),
        ).points.append(
            SeriesPoint(year=r.year, value=r.value, unit=r.unit, source=r.data_source)
        )
    for code in codes:
        out.setdefault(code, IndicatorSeries(indicator_code=code, indicator_name=None, points=[]))
    return out

# =============================================================================
# AI Endpoints
# =============================================================================
class AskReq(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_tokens: Optional[int] = 768
    temperature: Optional[float] = 0.2

class SearchReq(BaseModel):
    query: str
    k: int = 16

@app.get("/api/ai/info")
def ai_info():
    return {"ollama_url": AI_OLLAMA_URL, "model": AI_MODEL}

@app.post("/api/ask")
async def ai_ask_once(req: AskReq):
    try:
        answer = await ai_ask(
            req.prompt,
            system=req.system,
            max_tokens=req.max_tokens or 768,
            temperature=req.temperature or 0.2,
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI backend error: {e}")

@app.post("/api/ask/stream")
async def ai_ask_stream(req: AskReq):
    async def sse_gen() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in ai_stream_ask(
                req.prompt,
                system=req.system,
                max_tokens=req.max_tokens or 768,
                temperature=req.temperature or 0.2,
            ):
                yield f"data: {chunk}\n\n".encode("utf-8")
        except Exception as e:
            yield f"data: [stream error] {e}\n\n".encode("utf-8")
    return StreamingResponse(sse_gen(), media_type="text/event-stream")

@app.post("/api/ai/index/rebuild")
async def ai_index_rebuild(
    year_from: int = 1990,
    year_to: int = 2025,
    db: Session = Depends(get_db),
):
    try:
        count = await rag_rebuild_index(db, year_from=year_from, year_to=year_to)
        return {"ok": True, "inserted_docs": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/search")
async def ai_search(req: SearchReq, db: Session = Depends(get_db)):
    try:
        results = await rag_search(db, req.query, k=req.k)
        return {"results": [{"text": t, "meta": m} for (t, m) in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/ask-data")
async def ai_ask_data(question: str = Body(..., embed=True), db: Session = Depends(get_db)):
    try:
        result = await rag_answer(db, question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Forecasting
# =============================================================================
@app.get("/api/forecast")
def api_forecast(
    indicator_code: str,
    country_code: str,
    horizon: int = 5,
    method: str = "auto",     # "auto" | "linear" | "ses"
    transform: str = "auto",  # "auto" | "none" | "log"
    db: Session = Depends(get_db),
):
    """
    Forecast a single country/indicator series (annual) for the next N years.
    Returns history and forecast with simple confidence bands.
    """
    horizon = max(1, min(horizon, 25))  # safety bounds
    method = (method or "auto").lower().strip()
    transform = (transform or "auto").lower().strip()
    if method not in {"auto", "linear", "ses"}:
        method = "auto"
    if transform not in {"auto", "none", "log"}:
        transform = "auto"

    res = forecast_country_indicator(
        db,
        indicator_code=indicator_code,
        country_code=country_code,
        horizon=horizon,
        method=method,
        transform=transform,
    )

    return {
        "indicator_code": res.indicator_code,
        "country_code": res.country_code,
        "model": res.model,
        "params": res.params,
        "rmse": res.rmse,
        "history": [{"year": p.year, "value": p.value} for p in res.history],
        "forecast": [
            {"year": p.year, "value": p.value, "lower": p.lower, "upper": p.upper}
            for p in res.forecast
        ],
    }

@app.post("/api/forecast/compare")
def forecast_compare(
    indicator_code: str = Body(...),
    countries: List[str] = Body(...),
    horizon: int = Body(3),
    method: str = Body("auto"),
    transform: str = Body("auto"),
    db: Session = Depends(get_db),
):
    """
    Compare next-year forecasts across a list of countries.
    Returns sorted results by forecast value (desc).
    """
    out = []
    for cc in countries:
        try:
            res = forecast_country_indicator(
                db,
                indicator_code=indicator_code,
                country_code=cc,
                horizon=max(1, min(int(horizon), 25)),
                method=(method or "auto"),
                transform=(transform or "auto"),
            )
            last_actual = res.history[-1].value if res.history else None
            next_year = res.forecast[0].value if res.forecast else None
            out.append({
                "country_code": cc,
                "model": res.model,
                "rmse": res.rmse,
                "last_actual": last_actual,
                "next_year_forecast": next_year,
            })
        except Exception as e:
            out.append({
                "country_code": cc,
                "error": str(e),
            })
    # Sort by next_year_forecast desc if it exists
    out.sort(key=lambda r: (r.get("next_year_forecast") is not None, r.get("next_year_forecast")), reverse=True)
    return {"indicator_code": indicator_code, "horizon": horizon, "results": out}

# =============================================================================
# Manual refresh endpoints
# =============================================================================
@app.post("/api/refresh")
def refresh_data(db: Session = Depends(get_db)):
    try:
        fetch_and_update_all()
        return {"ok": True, "message": "Refresh job started (check logs)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh-and-reindex")
def refresh_and_reindex(db: Session = Depends(get_db)):
    """
    One-shot: fetch data (WB/IMF) and rebuild RAG index in the SAME DB.
    """
    try:
        fetch_and_update_all()
        asyncio.run(rag_rebuild_index(db, year_from=1990, year_to=2025))
        return {"ok": True, "message": "Data refreshed and ai_docs rebuilt."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/debug/fetch-now")
def debug_fetch_now():
    try:
        fetch_and_update_all()
        return {"ok": True, "msg": "Fetch and update triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
