# backend/ai_rag.py
from __future__ import annotations

import re
import json
import math
import os
import time
import sqlite3
from typing import List, Dict, Tuple, Optional

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import text as sqltext

from backend.models import EconomicIndicator, Country
from backend.ai_chat import OLLAMA_URL, ask as llm_ask


# ----------------------------------------------------------------------
# Lexical helpers
# ----------------------------------------------------------------------

REGION_WORDS = {
    "europe": "Europe",
    "mena": "MENA",
    "middle east": "MENA",
    "africa": "Africa",
    "asia": "Asia",
    "north america": "North America",
    "south america": "South America",
}


def _extract_filters(question: str, countries: list[Country]) -> dict:
    q = question.lower()
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    year = years[0] if years else None

    region = None
    for k, v in REGION_WORDS.items():
        if k in q:
            region = v
            break

    country_codes = set()
    for c in countries:
        if c.country_name and c.country_name.lower() in q:
            country_codes.add(c.country_code.upper())
        elif c.country_code and c.country_code.lower() in q:
            country_codes.add(c.country_code.upper())

    return {"year": year, "region": region, "country_codes": list(country_codes)}


def _load_rows(db: Session):
    return db.execute(sqltext(SELECT_ALL_SQL)).fetchall()


def _filter_rows(rows, year=None, region=None, country_codes=None):
    """Simple lexical filter on stored text/meta JSON."""
    if not rows:
        return rows
    out = []
    for _id, text, meta_json, emb_json in rows:
        ok = True
        if year is not None:
            ok = ok and (f'"year": {year}' in meta_json or f"Year: {year}" in text)
        if ok and region:
            ok = ok and (f"Region: {region}" in text)
        if ok and country_codes:
            ok = ok and any(
                (f'"country_code": "{cc}"' in meta_json or f"({cc})" in text)
                for cc in country_codes
            )
        if ok:
            out.append((_id, text, meta_json, emb_json))
    return out


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Embedding model served by Ollama (pull one of these locally)
#   ollama pull mxbai-embed-large
#   ollama pull nomic-embed-text
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")


# ----------------------------------------------------------------------
# SQLite schema / statements
# ----------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ai_docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    embedding TEXT NOT NULL
);
"""

DELETE_ALL_SQL = "DELETE FROM ai_docs"

INSERT_DOC_SQL = """
INSERT INTO ai_docs (text, meta_json, embedding)
VALUES (:text, :meta_json, :embedding)
"""

SELECT_ALL_SQL = "SELECT id, text, meta_json, embedding FROM ai_docs"


# ----------------------------------------------------------------------
# Small SQLite helpers (WAL, busy timeout, and retry wrapper)
# ----------------------------------------------------------------------

def _apply_sqlite_pragmas(db: Session) -> None:
    """Put SQLite in WAL mode and set a busy timeout to reduce 'database is locked' errors."""
    try:
        db.execute(sqltext("PRAGMA journal_mode=WAL;"))
        db.execute(sqltext("PRAGMA busy_timeout=5000;"))  # 5s
        db.commit()
    except Exception:
        db.rollback()


def _is_locked_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return "locked" in s or "database is locked" in s or isinstance(exc, sqlite3.OperationalError)


def _exec_with_retry(
    db: Session,
    sql: str,
    params: Optional[dict] = None,
    retries: int = 5,
    delay: float = 0.15,
) -> None:
    """Execute a statement, retrying if we hit transient SQLite locking."""
    for attempt in range(retries):
        try:
            db.execute(sqltext(sql), params or {})
            return
        except Exception as e:
            if attempt < retries - 1 and _is_locked_error(e):
                time.sleep(delay * (attempt + 1))
                continue
            raise


# ----------------------------------------------------------------------
# Embeddings via Ollama (handles both "embedding" and "embeddings")
# ----------------------------------------------------------------------

async def _embed(texts: List[str]) -> List[List[float]]:
    """
    Call Ollama /api/embeddings with robust normalization.
    Some models (e.g. mxbai-embed-large) may return:
      - {"embeddings": [[...], [...], ...]}  for list input   (ideal)
      - {"embedding":  [...]}                even for list    (quirk)
    We detect mismatches and fall back to per-text requests.
    """
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=120.0) as client:
        def _normalize(payload_json: dict) -> List[List[float]]:
            if "embeddings" in payload_json and isinstance(payload_json["embeddings"], list):
                return payload_json["embeddings"]
            if "embedding" in payload_json and isinstance(payload_json["embedding"], list):
                return [payload_json["embedding"]]
            return []

        # Try batch first
        payload = {"model": EMBED_MODEL, "input": texts}
        r = await client.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
        r.raise_for_status()
        batch_vecs = _normalize(r.json())

        if len(batch_vecs) == len(texts):
            return batch_vecs

        # Fallback: per-text calls
        vecs: List[List[float]] = []
        for t in texts:
            pr = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "input": t},
            )
            pr.raise_for_status()
            pj = pr.json()
            one = _normalize(pj)
            if not one:
                raise RuntimeError(f"Unexpected embeddings response: keys={list(pj.keys())}")
            vecs.append(one[0])
        return vecs


# ----------------------------------------------------------------------
# Math
# ----------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ----------------------------------------------------------------------
# Document formatting (UPDATED: consistent numeric formatting)
# ----------------------------------------------------------------------

def _fmt(v, places=2, sep=False):
    """Format floats safely with fixed decimals; optional thousands separator."""
    if v is None:
        return None
    try:
        if sep:
            return f"{float(v):,.{places}f}"
        return f"{float(v):.{places}f}"
    except Exception:
        return str(v)


def _format_doc(country_name: str, country_code: str, year: int, vals: Dict[str, float | None]) -> str:
    """
    Create a compact, analyzable snippet containing the key indicators (and region if available),
    with consistent numeric formatting for better model grounding.
    """
    region     = vals.get("REGION")
    gdp_val    = vals.get("GDP_VALUE")
    gdp_growth = vals.get("GDP_GROWTH")
    cpi        = vals.get("CPI")
    unemp      = vals.get("UNEMPLOYMENT")
    tb         = vals.get("TRADE_BALANCE")

    parts: List[str] = []
    if region:
        parts.append(f"Region: {region}")
    parts.append(f"Country: {country_name} ({country_code})")
    parts.append(f"Year: {year}")

    if gdp_val is not None:
        parts.append(f"GDP (current USD): {_fmt(gdp_val, 2, sep=True)}")
    if gdp_growth is not None:
        parts.append(f"Real GDP Growth (%): {_fmt(gdp_growth, 2)}")
    if cpi is not None:
        parts.append(f"Inflation CPI YoY (%): {_fmt(cpi, 2)}")
    if unemp is not None:
        parts.append(f"Unemployment Rate (%): {_fmt(unemp, 2)}")
    if tb is not None:
        parts.append(f"Trade Balance (% of GDP): {_fmt(tb, 2)}")

    return " | ".join(parts)


# ----------------------------------------------------------------------
# Index building (safe for FastAPI-scoped Session)
# ----------------------------------------------------------------------

async def rebuild_index(db: Session, year_from: int = 1990, year_to: int = 2030) -> int:
    """
    Rebuild the vector index (ai_docs) from EconomicIndicator rows, with WAL and
    retry-on-lock logic. No nested transactions (works with FastAPI's Session).
    """
    _apply_sqlite_pragmas(db)

    # Ensure table exists, clear previous contents
    _exec_with_retry(db, CREATE_TABLE_SQL)
    _exec_with_retry(db, DELETE_ALL_SQL)
    db.commit()

    total_inserted = 0

    countries = db.query(Country).all()
    for c in countries:
        # Gather all rows for this country & year range
        rows = (
            db.query(EconomicIndicator)
            .filter(
                EconomicIndicator.country_code == c.country_code,
                EconomicIndicator.year >= year_from,
                EconomicIndicator.year <= year_to,
            )
            .all()
        )

        # Group by year and collect indicators
        by_year: Dict[int, Dict[str, float | None]] = {}
        for r in rows:
            d = by_year.setdefault(r.year, {})
            d[r.indicator_code] = r.value
            if "REGION" not in d:
                d["REGION"] = r.region

        to_embed_texts: List[str] = []
        meta_payloads: List[Dict] = []

        for year, vals in sorted(by_year.items()):
            # If all numeric indicators are None, skip
            if all(v is None for k, v in vals.items() if k != "REGION"):
                continue

            doc_text = _format_doc(c.country_name, c.country_code, year, vals)
            to_embed_texts.append(doc_text)
            meta_payloads.append({
                "country_code": c.country_code,
                "country_name": c.country_name,
                "region": vals.get("REGION"),
                "year": year,
                "present_indicators": [k for k in vals.keys() if k not in ("REGION",)],
            })

        if not to_embed_texts:
            continue

        # Embed this batch (no DB writes while waiting)
        embeddings = await _embed(to_embed_texts)
        if not embeddings:
            continue

        # Write this country's docs with retries; keep transactions short
        for doc_text, meta, emb in zip(to_embed_texts, meta_payloads, embeddings):
            _exec_with_retry(
                db,
                INSERT_DOC_SQL,
                {"text": doc_text, "meta_json": json.dumps(meta), "embedding": json.dumps(emb)},
            )
            total_inserted += 1

        db.commit()  # commit per country

    return total_inserted


# ----------------------------------------------------------------------
# Search API (cosine over local vectors)
# ----------------------------------------------------------------------

async def search_docs(db: Session, query: str, k: int = 8) -> List[Tuple[str, Dict]]:
    """
    Embed the query, compute cosine similarity vs all docs, return top-k (text, meta) tuples.
    """
    q_vecs = await _embed([query])
    if not q_vecs:
        return []
    q_emb = q_vecs[0]

    rows = db.execute(sqltext(SELECT_ALL_SQL)).fetchall()
    if not rows:
        return []

    scored: List[Tuple[float, str, Dict]] = []
    for _id, text, meta_json, emb_json in rows:
        emb = json.loads(emb_json)
        score = _cosine(q_emb, emb)
        scored.append((score, text, json.loads(meta_json)))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:max(1, k)]
    return [(t[1], t[2]) for t in top]


# ----------------------------------------------------------------------
# RAG: answer using ONLY indexed data
# ----------------------------------------------------------------------

async def answer_with_data(db: Session, question: str) -> Dict:
    # 1) Extract lexical filters from the question
    countries = db.query(Country).all()
    filt = _extract_filters(question, countries)

    # 2) Load all rows once (small-scale SQLite)
    rows = _load_rows(db)

    # 3) If we found filters, prefilter; else use all rows
    cand_rows = _filter_rows(
        rows,
        year=filt["year"],
        region=filt["region"],
        country_codes=filt["country_codes"],
    )

    # 4) Embed query once
    q_vecs = await _embed([question])
    if not q_vecs:
        return {"answer": "Could not embed the query.", "sources": []}
    q_emb = q_vecs[0]

    # 5) Re-rank candidates by cosine; if none, fall back to all rows
    pool = cand_rows if cand_rows else rows

    scored = []
    for _id, text, meta_json, emb_json in pool:
        emb = json.loads(emb_json)
        score = _cosine(q_emb, emb)
        scored.append((score, text, json.loads(meta_json)))

    scored.sort(key=lambda t: t[0], reverse=True)

    # widen k
    k = 50
    top_docs = [(t[1], t[2]) for t in scored[:k]]

    if not top_docs:
        return {
            "answer": (
                "I couldn't find any indexed documents to answer that. "
                "Try rebuilding the index with a wider year range."
            ),
            "sources": []
        }

    # 6) Build prompt
    context = "\n".join([f"- {text}" for text, _ in top_docs])
    system = (
        "You are an experienced market and macroeconomic analyst. "
        "Use ONLY the provided CONTEXT (country-year indicators) to answer. "
        "If not supported by the context, say you don't have data. "
        "Round all numeric values to 2 decimals. "
        "Be precise, cite countries/years when you use numbers."
    )
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Provide a concise, well-structured answer grounded in the context above."
    )

    answer = await llm_ask(prompt, system=system, max_tokens=600, temperature=0.2)
    return {"answer": answer, "sources": top_docs}
