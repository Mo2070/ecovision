# backend/forecasting.py
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import pstdev
from typing import List, Optional, Tuple, Dict

from sqlalchemy.orm import Session

from backend.models import EconomicIndicator

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class HistPoint:
    year: int
    value: float

@dataclass
class FcstPoint:
    year: int
    value: float
    lower: Optional[float] = None
    upper: Optional[float] = None

@dataclass
class ForecastResult:
    indicator_code: str
    country_code: str
    model: str
    params: Dict
    rmse: float
    history: List[HistPoint]
    forecast: List[FcstPoint]

# ----------------------------
# Utilities
# ----------------------------
def _select_transform(indicator_code: str, transform: str) -> str:
    """Choose transform. 'auto' => log for GDP_VALUE, else none."""
    if transform == "none":
        return "none"
    if transform == "log":
        return "log"
    # auto
    if indicator_code.upper() == "GDP_VALUE":
        return "log"
    return "none"

def _prep_series(db: Session, indicator_code: str, country_code: str) -> List[HistPoint]:
    rows = (
        db.query(EconomicIndicator)
        .filter(EconomicIndicator.country_code == country_code.upper())
        .filter(EconomicIndicator.indicator_code == indicator_code.upper())
        .filter(EconomicIndicator.value.isnot(None))
        .order_by(EconomicIndicator.year.asc())
        .all()
    )
    out: List[HistPoint] = [HistPoint(year=r.year, value=float(r.value)) for r in rows]
    # Drop duplicates by year if any (take last)
    dedup: Dict[int, float] = {}
    for p in out:
        dedup[p.year] = p.value
    out = [HistPoint(year=y, value=dedup[y]) for y in sorted(dedup)]
    return out

def _apply_transform(values: List[float], t: str) -> List[float]:
    if t == "log":
        return [log(v) for v in values if v is not None and v > 0]
    return values[:]  # none

def _invert_transform(values: List[float], t: str) -> List[float]:
    if t == "log":
        return [exp(v) for v in values]
    return values

# ----------------------------
# Models (Linear & SES)
# ----------------------------
def _fit_linear(years: List[int], ys: List[float]) -> Tuple[float, float]:
    """
    Fit y = a + b * year via OLS closed-form.
    Returns (a, b).
    """
    n = len(years)
    sx = sum(years)
    sy = sum(ys)
    sxx = sum(x * x for x in years)
    sxy = sum(x * y for x, y in zip(years, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        # fallback: horizontal mean
        b = 0.0
        a = sy / n
        return a, b
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b

def _predict_linear(a: float, b: float, x: int) -> float:
    return a + b * x

def _fit_ses(ys: List[float], alpha: Optional[float] = None) -> Tuple[float, float]:
    """
    SES with fixed alpha. If alpha None, grid-search in [0.1..0.9].
    Returns (alpha, last_level)
    """
    def ses_rmse(alpha_val: float) -> Tuple[float, float, float]:
        level = ys[0]
        preds = []
        for y in ys[1:]:
            preds.append(level)
            level = alpha_val * y + (1 - alpha_val) * level
        # Compare preds to ys[1:]
        if not preds:
            return alpha_val, level, 0.0
        errors = [y - p for y, p in zip(ys[1:], preds)]
        rmse = sqrt(sum(e * e for e in errors) / len(errors))
        return alpha_val, level, rmse

    if alpha is None:
        best = None
        for a in [i / 10.0 for i in range(1, 10)]:
            cand = ses_rmse(a)
            if best is None or cand[2] < best[2]:
                best = cand
        return best[0], best[1]
    else:
        a, lvl, _ = ses_rmse(alpha)
        return a, lvl

def _forecast_ses(last_level: float, alpha: float, steps: int) -> List[float]:
    # SES forecast is flat at last level
    return [last_level] * steps

# ----------------------------
# Backtest & intervals
# ----------------------------
def _holdout_backtest_linear(years: List[int], ys: List[float], test_len: int = 5) -> Tuple[float, Dict]:
    if len(ys) <= test_len + 3:
        # too short, just fit all
        a, b = _fit_linear(years, ys)
        fitted = [_predict_linear(a, b, x) for x in years]
        errors = [y - f for y, f in zip(ys, fitted)]
        rmse = sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
        return rmse, {"a": a, "b": b}

    years_tr, years_te = years[:-test_len], years[-test_len:]
    ys_tr, ys_te = ys[:-test_len], ys[-test_len:]
    a, b = _fit_linear(years_tr, ys_tr)
    preds = [_predict_linear(a, b, x) for x in years_te]
    errors = [y - p for y, p in zip(ys_te, preds)]
    rmse = sqrt(sum(e * e for e in errors) / len(errors))
    return rmse, {"a": a, "b": b}

def _holdout_backtest_ses(ys: List[float], test_len: int = 5) -> Tuple[float, Dict]:
    if len(ys) <= test_len + 2:
        # too short; fit all
        alpha, last_level = _fit_ses(ys)
        fitted = [ys[0]]  # first has no prediction
        level = ys[0]
        for y in ys[1:]:
            fitted.append(level)
            level = alpha * y + (1 - alpha) * level
        errors = [y - f for y, f in zip(ys[1:], fitted[1:])]
        rmse = sqrt(sum(e * e for e in errors) / len(errors)) if errors else 0.0
        return rmse, {"alpha": alpha, "level": last_level}

    ys_tr, ys_te = ys[:-test_len], ys[-test_len:]
    alpha, last_level = _fit_ses(ys_tr)
    # SES multi-step: flat at last_level
    preds = [last_level] * len(ys_te)
    errors = [y - p for y, p in zip(ys_te, preds)]
    rmse = sqrt(sum(e * e for e in errors) / len(errors))
    return rmse, {"alpha": alpha, "level": last_level}

def _conf_band_sigma(residuals: List[float]) -> float:
    if not residuals:
        return 0.0
    # population std as simple sigma
    return pstdev(residuals)

# ----------------------------
# Public API
# ----------------------------
def forecast_country_indicator(
    db: Session,
    *,
    indicator_code: str,
    country_code: str,
    horizon: int = 5,
    method: str = "auto",     # "auto" | "linear" | "ses"
    transform: str = "auto",  # "auto" | "none" | "log"
) -> ForecastResult:
    hist = _prep_series(db, indicator_code, country_code)
    if not hist:
        raise ValueError("No history found for this indicator/country.")

    years = [p.year for p in hist]
    values = [p.value for p in hist]

    # Choose transform
    t = _select_transform(indicator_code, transform)
    y_t = _apply_transform(values, t)
    if len(y_t) < 2:
        # too short for models; return flat continuation
        last_y = values[-1]
        fc_years = [years[-1] + i for i in range(1, horizon + 1)]
        fc_vals = [last_y] * horizon
        sigma = 0.0
        forecast = [FcstPoint(year=y, value=v, lower=v, upper=v) for y, v in zip(fc_years, fc_vals)]
        return ForecastResult(
            indicator_code=indicator_code,
            country_code=country_code,
            model="naive",
            params={},
            rmse=0.0,
            history=hist,
            forecast=forecast,
        )

    # Backtests
    if method == "linear":
        rmse_lin, params_lin = _holdout_backtest_linear(years, y_t, test_len=min(5, max(1, len(y_t)//5)))
        best = ("linear", rmse_lin, params_lin)
    elif method == "ses":
        rmse_ses, params_ses = _holdout_backtest_ses(y_t, test_len=min(5, max(1, len(y_t)//5)))
        best = ("ses", rmse_ses, params_ses)
    else:
        # auto: evaluate both
        tl = min(5, max(1, len(y_t)//5))
        rmse_lin, params_lin = _holdout_backtest_linear(years, y_t, test_len=tl)
        rmse_ses, params_ses = _holdout_backtest_ses(y_t, test_len=tl)
        if rmse_lin <= rmse_ses:
            best = ("linear", rmse_lin, params_lin)
        else:
            best = ("ses", rmse_ses, params_ses)

    model_name, rmse, params = best

    # Fit on full data for final forecast
    if model_name == "linear":
        a, b = _fit_linear(years, y_t)
        params = {"a": a, "b": b}
        fc_x = [years[-1] + i for i in range(1, horizon + 1)]
        fc_y_t = [a + b * x for x in fc_x]
        # In-sample residuals for sigma
        fitted_t = [a + b * x for x in years]
        residuals_t = [yt - ft for yt, ft in zip(y_t, fitted_t)]
    else:
        alpha, level = _fit_ses(y_t)
        params = {"alpha": alpha, "level": level}
        fc_y_t = _forecast_ses(level, alpha, horizon)
        # In-sample residuals for sigma
        fitted_t = [y_t[0]]
        lvl = y_t[0]
        for y in y_t[1:]:
            fitted_t.append(lvl)
            lvl = alpha * y + (1 - alpha) * lvl
        residuals_t = [yt - ft for yt, ft in zip(y_t[1:], fitted_t[1:])]

    sigma_t = _conf_band_sigma(residuals_t)
    # 95% bands in transformed space
    lower_t = [y - 1.96 * sigma_t for y in fc_y_t]
    upper_t = [y + 1.96 * sigma_t for y in fc_y_t]

    # Back-transform
    fc_vals = _invert_transform(fc_y_t, t)
    lower_vals = _invert_transform(lower_t, t)
    upper_vals = _invert_transform(upper_t, t)

    # Build results
    fc_years = [years[-1] + i for i in range(1, horizon + 1)]
    forecast = [
        FcstPoint(year=yr, value=val, lower=lo, upper=hi)
        for yr, val, lo, hi in zip(fc_years, fc_vals, lower_vals, upper_vals)
    ]

    return ForecastResult(
        indicator_code=indicator_code,
        country_code=country_code,
        model=model_name,
        params=params,
        rmse=rmse,
        history=hist,
        forecast=forecast,
    )
