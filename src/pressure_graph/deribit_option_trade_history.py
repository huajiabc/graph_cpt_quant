"""Deribit historical option-trade surfaces with conservative freshness gates.

The public TradingView endpoint exposes trade OHLCV bars for archived options.
Hours without trades can be forward-filled by the exchange, so this module only
uses bars with strictly positive volume and cost.  The resulting surface is a
research signal source, not a reconstruction of historical executable quotes.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


OPTION_NAME = re.compile(
    r"^(?P<currency>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-"
    r"(?P<strike>\d+(?:\.\d+)?)-(?P<option_type>[CP])$"
)
YEAR_SECONDS = 365.25 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class DeribitOptionSurfaceConfig:
    min_dte: float = 7.0
    max_dte: float = 45.0
    min_iv: float = 0.10
    max_iv: float = 3.00
    max_abs_log_moneyness: float = 0.55
    wing_delta_low: float = 0.12
    wing_delta_high: float = 0.40
    min_call_contracts: int = 2
    min_put_contracts: int = 2
    min_strikes: int = 3


def quarterly_expiries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return Deribit-style quarterly expiries (last Friday, 08:00 UTC)."""
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    expiries: list[pd.Timestamp] = []
    for year in range(start.year, end.year + 1):
        for month in (3, 6, 9, 12):
            month_end = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            month_end += pd.offsets.MonthEnd(0)
            days_back = (month_end.weekday() - 4) % 7
            expiry = pd.Timestamp(month_end - pd.Timedelta(days=days_back))
            expiry += pd.Timedelta(hours=8)
            if start <= expiry <= end:
                expiries.append(expiry)
    return expiries


def monthly_expiries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return the last-Friday 08:00 UTC expiry for every calendar month."""
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    months = pd.date_range(
        start.floor("D").replace(day=1),
        end.floor("D").replace(day=1),
        freq="MS",
    )
    expiries: list[pd.Timestamp] = []
    for month in months:
        month_end = month + pd.offsets.MonthEnd(0)
        days_back = (month_end.weekday() - 4) % 7
        expiry = pd.Timestamp(month_end - pd.Timedelta(days=days_back))
        expiry += pd.Timedelta(hours=8)
        if start <= expiry <= end:
            expiries.append(expiry)
    return expiries


def option_instrument(expiry: pd.Timestamp, strike: float, option_type: str) -> str:
    expiry = pd.Timestamp(expiry).tz_convert("UTC")
    strike_text = f"{float(strike):.8f}".rstrip("0").rstrip(".")
    expiry_text = f"{expiry.day}{expiry:%b%y}".upper()
    return f"BTC-{expiry_text}-{strike_text}-{option_type.upper()}"


def parse_option_instrument(instrument_name: str) -> dict[str, object]:
    match = OPTION_NAME.fullmatch(instrument_name.upper().strip())
    if match is None:
        raise ValueError(f"Invalid Deribit option instrument: {instrument_name}")
    expiry = pd.to_datetime(
        match.group("expiry"), format="%d%b%y", utc=True
    ) + pd.Timedelta(hours=8)
    return {
        "currency": match.group("currency"),
        "expiration_time": expiry,
        "strike": float(match.group("strike")),
        "option_type": "call" if match.group("option_type") == "C" else "put",
    }


def candidate_strikes(reference_price: float) -> list[float]:
    """Generate a deliberately coarse lattice known to exist in old archives."""
    if not np.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("reference_price must be positive and finite")
    step = 5_000.0 if reference_price < 40_000.0 else 10_000.0
    strikes = {
        max(step, round(reference_price * multiplier / step) * step)
        for multiplier in np.linspace(0.6, 1.4, 9)
    }
    return sorted(float(value) for value in strikes)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def inverse_option_price(
    spot: float,
    strike: float,
    years_to_expiry: float,
    volatility: float,
    option_type: str,
) -> float:
    """Black-Scholes value in BTC for Deribit's inverse BTC options."""
    if min(spot, strike, years_to_expiry, volatility) <= 0:
        return math.nan
    root_t = math.sqrt(years_to_expiry)
    d1 = (
        math.log(spot / strike) + 0.5 * volatility * volatility * years_to_expiry
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    strike_ratio = strike / spot
    if option_type.lower() == "call":
        return _normal_cdf(d1) - strike_ratio * _normal_cdf(d2)
    if option_type.lower() == "put":
        return strike_ratio * _normal_cdf(-d2) - _normal_cdf(-d1)
    raise ValueError(f"Unsupported option_type: {option_type}")


def implied_volatility(
    price_btc: float,
    spot: float,
    strike: float,
    years_to_expiry: float,
    option_type: str,
    lower: float = 0.01,
    upper: float = 5.0,
    iterations: int = 80,
) -> float:
    """Invert an inverse-option trade price with a deterministic bisection."""
    values = (price_btc, spot, strike, years_to_expiry)
    if not all(np.isfinite(value) and value > 0 for value in values):
        return math.nan
    low_price = inverse_option_price(spot, strike, years_to_expiry, lower, option_type)
    high_price = inverse_option_price(spot, strike, years_to_expiry, upper, option_type)
    tolerance = 1e-10
    if price_btc < low_price - tolerance or price_btc > high_price + tolerance:
        return math.nan
    left, right = lower, upper
    for _ in range(iterations):
        middle = 0.5 * (left + right)
        value = inverse_option_price(spot, strike, years_to_expiry, middle, option_type)
        if value < price_btc:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


def option_delta(
    spot: float,
    strike: float,
    years_to_expiry: float,
    volatility: float,
    option_type: str,
) -> float:
    if min(spot, strike, years_to_expiry, volatility) <= 0:
        return math.nan
    d1 = (
        math.log(spot / strike) + 0.5 * volatility * volatility * years_to_expiry
    ) / (volatility * math.sqrt(years_to_expiry))
    call_delta = _normal_cdf(d1)
    return call_delta if option_type.lower() == "call" else call_delta - 1.0


def normalize_active_trade_bars(
    option_bars: pd.DataFrame,
    underlying_bars: pd.DataFrame,
    instrument_name: str,
) -> pd.DataFrame:
    """Retain only actually traded option hours and attach contemporaneous BTC."""
    if option_bars.empty:
        return pd.DataFrame()
    metadata = parse_option_instrument(instrument_name)
    option = option_bars.copy()
    option["bar_open_time"] = pd.to_datetime(
        option["bar_open_time"], utc=True, errors="coerce"
    )
    for column in ("volume", "cost"):
        option[column] = pd.to_numeric(option[column], errors="coerce")
    option = option[
        option["volume"].gt(0)
        & option["cost"].gt(0)
        & option["bar_open_time"].lt(metadata["expiration_time"])
    ].copy()
    if option.empty:
        return pd.DataFrame()
    option["trade_vwap_btc"] = option["cost"] / option["volume"]

    underlying = underlying_bars.copy()
    underlying["bar_open_time"] = pd.to_datetime(
        underlying["bar_open_time"], utc=True, errors="coerce"
    )
    for column in ("open", "close"):
        underlying[column] = pd.to_numeric(underlying[column], errors="coerce")
    underlying["underlying_price"] = 0.5 * (
        underlying["open"] + underlying["close"]
    )
    option = option.merge(
        underlying[["bar_open_time", "underlying_price"]],
        on="bar_open_time",
        how="left",
        validate="many_to_one",
    )
    option["instrument_name"] = instrument_name
    option["expiration_time"] = metadata["expiration_time"]
    option["strike"] = metadata["strike"]
    option["option_type"] = metadata["option_type"]
    option["years_to_expiry"] = (
        option["expiration_time"]
        - (option["bar_open_time"] + pd.Timedelta(minutes=30))
    ).dt.total_seconds() / YEAR_SECONDS
    option["dte"] = option["years_to_expiry"] * 365.25
    option["log_moneyness"] = np.log(option["strike"] / option["underlying_price"])
    option["implied_volatility"] = [
        implied_volatility(price, spot, strike, years, kind)
        for price, spot, strike, years, kind in option[
            [
                "trade_vwap_btc",
                "underlying_price",
                "strike",
                "years_to_expiry",
                "option_type",
            ]
        ].itertuples(index=False, name=None)
    ]
    option["delta"] = [
        option_delta(spot, strike, years, volatility, kind)
        for spot, strike, years, volatility, kind in option[
            [
                "underlying_price",
                "strike",
                "years_to_expiry",
                "implied_volatility",
                "option_type",
            ]
        ].itertuples(index=False, name=None)
    ]
    option["abs_delta"] = option["delta"].abs()
    option["surface_date"] = option["bar_open_time"].dt.floor("D")
    keep = [
        "instrument_name",
        "bar_open_time",
        "surface_date",
        "expiration_time",
        "strike",
        "option_type",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "cost",
        "trade_vwap_btc",
        "underlying_price",
        "years_to_expiry",
        "dte",
        "log_moneyness",
        "implied_volatility",
        "delta",
        "abs_delta",
    ]
    return (
        option[keep]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(
            subset=[
                "bar_open_time",
                "trade_vwap_btc",
                "underlying_price",
                "implied_volatility",
            ]
        )
        .sort_values(["bar_open_time", "instrument_name"])
        .reset_index(drop=True)
    )


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric) & np.isfinite(weight) & (weight > 0)
    if not valid.any():
        return math.nan
    return float(np.average(numeric[valid], weights=weight[valid]))


def _daily_contract_rows(active_bars: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["surface_date", "expiration_time", "instrument_name"]
    for (surface_date, expiration_time, instrument), group in active_bars.groupby(
        keys, sort=True
    ):
        weights = group["volume"]
        rows.append(
            {
                "surface_date": surface_date,
                "expiration_time": expiration_time,
                "instrument_name": instrument,
                "strike": float(group["strike"].iloc[0]),
                "option_type": str(group["option_type"].iloc[0]),
                "dte": _weighted_average(group["dte"], weights),
                "log_moneyness": _weighted_average(group["log_moneyness"], weights),
                "implied_volatility": _weighted_average(
                    group["implied_volatility"], weights
                ),
                "abs_delta": _weighted_average(group["abs_delta"], weights),
                "volume": float(pd.to_numeric(weights, errors="coerce").sum()),
                "active_hours": int(group["bar_open_time"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_daily_option_surface(
    active_bars: pd.DataFrame,
    cfg: DeribitOptionSurfaceConfig = DeribitOptionSurfaceConfig(),
) -> pd.DataFrame:
    """Build one causal end-of-day surface row from actually traded contracts."""
    if active_bars.empty:
        return pd.DataFrame()
    contracts = _daily_contract_rows(active_bars)
    contracts = contracts[
        contracts["dte"].between(cfg.min_dte, cfg.max_dte)
        & contracts["implied_volatility"].between(cfg.min_iv, cfg.max_iv)
        & contracts["log_moneyness"].abs().le(cfg.max_abs_log_moneyness)
    ].copy()
    rows: list[dict[str, object]] = []
    for (surface_date, expiration_time), group in contracts.groupby(
        ["surface_date", "expiration_time"], sort=True
    ):
        calls = group[group["option_type"].eq("call")]
        puts = group[group["option_type"].eq("put")]
        wing_calls = calls[
            calls["abs_delta"].between(cfg.wing_delta_low, cfg.wing_delta_high)
        ]
        wing_puts = puts[
            puts["abs_delta"].between(cfg.wing_delta_low, cfg.wing_delta_high)
        ]
        if wing_calls.empty or wing_puts.empty:
            continue
        call25 = wing_calls.loc[(wing_calls["abs_delta"] - 0.25).abs().idxmin()]
        put25 = wing_puts.loc[(wing_puts["abs_delta"] - 0.25).abs().idxmin()]
        atm = group.loc[group["log_moneyness"].abs().nsmallest(2).index]
        atm_iv = _weighted_average(atm["implied_volatility"], atm["volume"])
        x = group["log_moneyness"].to_numpy(dtype=float)
        y = group["implied_volatility"].to_numpy(dtype=float)
        w = np.sqrt(group["volume"].to_numpy(dtype=float))
        slope = math.nan
        if len(group) >= 3 and np.ptp(x) > 0:
            slope = float(np.polyfit(x, y, 1, w=w)[0])
        rows.append(
            {
                "surface_date": surface_date,
                "feature_time": pd.Timestamp(surface_date) + pd.Timedelta(days=1),
                "expiration_time": expiration_time,
                "dte": float(group["dte"].median()),
                "contract_count": int(len(group)),
                "strike_count": int(group["strike"].nunique()),
                "call_contracts": int(len(calls)),
                "put_contracts": int(len(puts)),
                "active_hours": int(group["active_hours"].sum()),
                "total_volume": float(group["volume"].sum()),
                "call_volume": float(calls["volume"].sum()),
                "put_volume": float(puts["volume"].sum()),
                "put_call_log_volume_ratio": float(
                    np.log((puts["volume"].sum() + 1.0) / (calls["volume"].sum() + 1.0))
                ),
                "atm_iv": atm_iv,
                "put25_iv": float(put25["implied_volatility"]),
                "call25_iv": float(call25["implied_volatility"]),
                "downside_risk_reversal": float(
                    put25["implied_volatility"] - call25["implied_volatility"]
                ),
                "wing_convexity": float(
                    0.5
                    * (put25["implied_volatility"] + call25["implied_volatility"])
                    - atm_iv
                ),
                "iv_moneyness_slope": slope,
                "put25_abs_delta": float(put25["abs_delta"]),
                "call25_abs_delta": float(call25["abs_delta"]),
            }
        )
    surface = pd.DataFrame(rows)
    if surface.empty:
        return surface
    surface["quality_pass"] = (
        surface["call_contracts"].ge(cfg.min_call_contracts)
        & surface["put_contracts"].ge(cfg.min_put_contracts)
        & surface["strike_count"].ge(cfg.min_strikes)
        & surface[
            [
                "atm_iv",
                "put25_iv",
                "call25_iv",
                "downside_risk_reversal",
                "wing_convexity",
            ]
        ].notna().all(axis=1)
    )
    return surface.sort_values(["feature_time", "dte"]).reset_index(drop=True)


__all__ = [
    "DeribitOptionSurfaceConfig",
    "build_daily_option_surface",
    "candidate_strikes",
    "implied_volatility",
    "inverse_option_price",
    "monthly_expiries",
    "normalize_active_trade_bars",
    "option_delta",
    "option_instrument",
    "parse_option_instrument",
    "quarterly_expiries",
]
