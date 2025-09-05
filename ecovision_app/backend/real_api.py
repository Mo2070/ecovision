# backend/real_api.py
from __future__ import annotations

import requests
from datetime import date
from typing import Dict, Optional, List, Tuple

#  use the single SessionLocal from database (one engine, one DB file)
from backend import database
from backend.models import EconomicIndicator, Country

# ------------------------------------------------------------------------------
# External APIs
# ------------------------------------------------------------------------------
WB_API = (
    "https://api.worldbank.org/v2/country/{country3}/indicator/{indicator}"
    "?format=json&per_page=20000"
)
IMF_DM_API = (
    "https://www.imf.org/-/media/Websites/IMF/Data/Data-Mapper/API/v1/"
    "indicators/{indicator}/countries/{country2}"
)

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "EcoVision/1.0 (academic, contact: you@example.com)",
}

TIMEOUT = (8, 30)  # (connect, read) seconds

# ------------------------------------------------------------------------------
# Countries (ISO2 + ISO3)
# ------------------------------------------------------------------------------
COUNTRIES: List[Dict[str, str]] = [
    # North America
    {"code2": "US", "code3": "USA", "name": "United States", "region": "North America", "capital": "Washington D.C."},
    {"code2": "CA", "code3": "CAN", "name": "Canada",         "region": "North America", "capital": "Ottawa"},

    # South America
    {"code2": "BR", "code3": "BRA", "name": "Brazil",         "region": "South America", "capital": "Brasília"},
    {"code2": "AR", "code3": "ARG", "name": "Argentina",      "region": "South America", "capital": "Buenos Aires"},

    # Africa
    {"code2": "EG", "code3": "EGY", "name": "Egypt",          "region": "Africa",        "capital": "Cairo"},
    {"code2": "MA", "code3": "MAR", "name": "Morocco",        "region": "Africa",        "capital": "Rabat"},

    # Asia
    {"code2": "SA", "code3": "SAU", "name": "Saudi Arabia",   "region": "Asia",          "capital": "Riyadh"},
    {"code2": "QA", "code3": "QAT", "name": "Qatar",          "region": "Asia",          "capital": "Doha"},
    {"code2": "AE", "code3": "ARE", "name": "United Arab Emirates", "region": "Asia",    "capital": "Abu Dhabi"},
    {"code2": "CN", "code3": "CHN", "name": "China",          "region": "Asia",          "capital": "Beijing"},
    {"code2": "JP", "code3": "JPN", "name": "Japan",          "region": "Asia",          "capital": "Tokyo"},

    # Europe
    {"code2": "DE", "code3": "DEU", "name": "Germany",        "region": "Europe",        "capital": "Berlin"},
    {"code2": "FR", "code3": "FRA", "name": "France",         "region": "Europe",        "capital": "Paris"},
    {"code2": "IT", "code3": "ITA", "name": "Italy",          "region": "Europe",        "capital": "Rome"},
    {"code2": "NL", "code3": "NLD", "name": "Netherlands",    "region": "Europe",        "capital": "Amsterdam"},
    {"code2": "GB", "code3": "GBR", "name": "United Kingdom", "region": "Europe",        "capital": "London"},
    {"code2": "RU", "code3": "RUS", "name": "Russia",         "region": "Europe",        "capital": "Moscow"},
]

# ------------------------------------------------------------------------------
# Indicators map (canonical codes → API codes)
# ------------------------------------------------------------------------------
INDICATORS: Dict[str, Dict] = {
    "GDP_VALUE": {
        "name": "GDP (current US$)",
        "category": "GDP",
        "unit": "USD",
        "sources": {
            "World Bank": "NY.GDP.MKTP.CD",
        },
    },
    "GDP_GROWTH": {
        "name": "Real GDP Growth Rate",
        "category": "GDP",
        "unit": "%",
        "sources": {"World Bank": "NY.GDP.MKTP.KD.ZG", "IMF": "NGDP_RPCH"},
    },
    "CPI": {
        "name": "Consumer Price Index (YoY)",
        "category": "Inflation",
        "unit": "%",
        "sources": {"World Bank": "FP.CPI.TOTL.ZG", "IMF": "PCPIPCH"},
    },
    "UNEMPLOYMENT": {
        "name": "Unemployment Rate",
        "category": "Employment",
        "unit": "%",
        "sources": {"World Bank": "SL.UEM.TOTL.ZS", "IMF": "LUR"},
    },
    "TRADE_BALANCE": {
        "name": "External balance on goods & services",
        "category": "Trade",
        "unit": "% of GDP",
        "sources": {
            # Prefer WB % of GDP; we’ll fallback to EXP% - IMP% if needed
            "World Bank": "NE.RSB.GNFS.ZS",
        },
    },
}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def wb_series_all_years(country3: str, indicator_code: str) -> Dict[int, Optional[float]]:
    """Fetch full time series for a WB indicator (returns year → value)."""
    url = WB_API.format(country3=country3, indicator=indicator_code)
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()

    out: Dict[int, Optional[float]] = {}
    if not data or len(data) < 2 or data[1] is None:
        return out

    for row in data[1]:
        try:
            y = int(row.get("date"))
        except Exception:
            continue
        v = row.get("value")
        out[y] = float(v) if v is not None else None
    return out


def wb_trade_balance_pct_series(country3: str) -> Dict[int, Optional[float]]:
    """Prefer NE.RSB.GNFS.ZS; if missing, approximate with EXP% - IMP%."""
    s = wb_series_all_years(country3, "NE.RSB.GNFS.ZS")
    if s:
        return s
    exp = wb_series_all_years(country3, "NE.EXP.GNFS.ZS")
    imp = wb_series_all_years(country3, "NE.IMP.GNFS.ZS")
    years = set(exp) | set(imp)
    out: Dict[int, Optional[float]] = {}
    for y in years:
        e, m = exp.get(y), imp.get(y)
        out[y] = (e - m) if (e is not None and m is not None) else None
    return out


def imf_series_all_years(country2: str, indicator_code: Optional[str]) -> Dict[int, Optional[float]]:
    """Fetch IMF Data-Mapper series when an IMF code is provided (returns year → value)."""
    if not indicator_code:
        return {}

    url = IMF_DM_API.format(indicator=indicator_code, country2=country2)
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=TIMEOUT)
    except Exception:
        return {}

    if "application/json" not in (r.headers.get("Content-Type", "")).lower():
        return {}

    try:
        payload = r.json()
    except Exception:
        return {}

    out: Dict[int, Optional[float]] = {}

    # Some endpoints use `series`, others `values`
    series = (payload.get("series") or {}).get(country2.upper())
    if isinstance(series, dict):
        for y, v in series.items():
            if _is_int(y):
                out[int(y)] = float(v) if v is not None else None

    values = (payload.get("values") or {}).get(country2.upper())
    if isinstance(values, dict):
        for y, v in values.items():
            if _is_int(y):
                out[int(y)] = float(v) if v is not None else None

    return out


def upsert_indicator_session(
    db,
    *,
    country_code: str,
    indicator_name: str,
    indicator_code: str,
    value: float,
    unit: str,
    year: int,
    category: str,
    data_source: str,
) -> None:
    """
    Insert/update a single EconomicIndicator row (per country, per year, per indicator).
    """
    existing = (
        db.query(EconomicIndicator)
        .filter_by(country_code=country_code, indicator_code=indicator_code, year=year, quarter=None)
        .first()
    )
    if existing:
        existing.value = value
        existing.unit = unit
        existing.last_updated = date.today()
        existing.data_source = data_source
    else:
        country = db.query(Country).filter_by(country_code=country_code).first()
        db.add(
            EconomicIndicator(
                country_code=country_code,
                country_name=(country.country_name if country else None),
                region=(country.region if country else None),
                indicator_name=indicator_name,
                indicator_code=indicator_code,
                value=value,
                unit=unit,
                year=year,
                quarter=None,
                data_source=data_source,
                last_updated=date.today(),
                category=category,
            )
        )


# ------------------------------------------------------------------------------
# Main fetch & update
# ------------------------------------------------------------------------------
def fetch_and_update_all() -> Dict[str, int]:
    """
    Fetch full series for all configured countries & indicators from
    World Bank (and IMF where available). Upsert into the **single** SQLite DB.

    Returns a small stats dict.
    """
    print(" Fetching full series from World Bank & IMF (multi-country)...")

    db = database.SessionLocal()  #  the single SessionLocal
    upsert_count = 0
    country_upserts = 0

    try:
        # 1) Ensure Country rows exist / are updated
        for c in COUNTRIES:
            row = db.query(Country).filter_by(country_code=c["code2"]).first()
            if row:
                # keep this id but update fields
                row.country_name = c["name"]
                row.region = c["region"]
                row.capital_city = c["capital"]
            else:
                db.add(
                    Country(
                        country_code=c["code2"],
                        country_name=c["name"],
                        region=c["region"],
                        capital_city=c["capital"],
                    )
                )
                country_upserts += 1
        db.commit()

        # 2) For each country/indicator: fetch full series; average WB/IMF per year
        for c in COUNTRIES:
            code2, code3 = c["code2"], c["code3"]

            for ind_key, cfg in INDICATORS.items():
                # WB series
                if ind_key == "TRADE_BALANCE":
                    wb_series = wb_trade_balance_pct_series(code3)
                else:
                    wb_code = cfg["sources"]["World Bank"]
                    wb_series = wb_series_all_years(code3, wb_code)

                # IMF series (optional)
                imf_code = cfg["sources"].get("IMF")
                imf_series = imf_series_all_years(code2, imf_code) if imf_code else {}

                # Combine – use average when both exist
                years = sorted(set(wb_series) | set(imf_series))
                for y in years:
                    wb_val = wb_series.get(y)
                    imf_val = imf_series.get(y) if imf_series else None

                    if wb_val is not None and imf_val is not None:
                        final_val = (wb_val + imf_val) / 2.0
                        sources = "World Bank, IMF"
                    elif wb_val is not None:
                        final_val = wb_val
                        sources = "World Bank"
                    elif imf_val is not None:
                        final_val = imf_val
                        sources = "IMF"
                    else:
                        continue  # nothing to store

                    upsert_indicator_session(
                        db,
                        country_code=code2,
                        indicator_name=cfg["name"],
                        indicator_code=ind_key,
                        value=final_val,
                        unit=cfg["unit"],
                        year=y,
                        category=cfg["category"],
                        data_source=sources,
                    )
                    upsert_count += 1

            # Commit per country to avoid very large transactions
            db.commit()

        print(" Full series updated for selected countries")
        return {
            "countries_added_or_updated": country_upserts,
            "indicator_rows_upserted": upsert_count,
        }

    except Exception as e:
        db.rollback()
        # Log to console so you can see it while developing
        print(f" fetch_and_update_all failed: {e}")
        # Re-raise so FastAPI endpoint returns 500
        raise
    finally:
        db.close()


# For local debugging:
if __name__ == "__main__":
    stats = fetch_and_update_all()
    print(stats)
