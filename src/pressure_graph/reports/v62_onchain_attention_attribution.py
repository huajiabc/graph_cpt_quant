"""v6.2 On-chain Attention Attribution.

This report does not create a trading rule.  It audits whether the daily
market-level DeFiLlama attention events from v6.1 add information beyond
month, BTC state, and market impulse context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _max_contribution, _month_cap_expectancy
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v61_onchain_dex_attention_backfill import (
    V61Config,
    _build_events,
    _fetch_attention_charts,
    _read_features,
    _read_p2_trades,
)


REPORT_ROOT = Path("reports/v6_2_onchain_attention_attribution")
HORIZONS = {
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "24h": pd.Timedelta(hours=24),
    "48h": pd.Timedelta(hours=48),
    "72h": pd.Timedelta(hours=72),
}


@dataclass(frozen=True)
class V62Config:
    report_root: Path = REPORT_ROOT
    v61: V61Config = V61Config()
    conservative_available_lag_hours: int = 4
    matched_random_trials: int = 100
    random_seed: int = 620
    main_horizon: str = "24h"


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _asof_events(events: pd.DataFrame, cfg: V62Config, policy: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True, errors="coerce")
    out["event_date"] = out["event_time"].dt.floor("D")
    if policy == "same_day_naive":
        out["event_available_time"] = out["event_date"]
    elif policy == "next_day_conservative":
        out["event_available_time"] = out["event_date"] + pd.Timedelta(days=1) + pd.Timedelta(hours=cfg.conservative_available_lag_hours)
    else:
        raise KeyError(policy)
    out["asof_policy"] = policy
    out["event_available_lag_hours"] = (out["event_available_time"] - out["event_date"]).dt.total_seconds() / 3600.0
    return out.dropna(subset=["event_time", "event_available_time"]).reset_index(drop=True)


def _dedup_attention_days(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for event_date, group in events.groupby("event_date", sort=True):
        local = group.copy()
        primary = local.sort_values(["zscore", "percentile"], ascending=False).iloc[0]
        rows.append(
            {
                "attention_day_id": f"global|{pd.Timestamp(event_date).date()}",
                "event_date": event_date,
                "event_time": local["event_time"].min(),
                "event_available_time": local["event_available_time"].max(),
                "event_type": "dedup_attention_day",
                "primary_event_type": primary.get("event_type", ""),
                "event_count_on_day": int(len(local)),
                "event_types_on_day": "|".join(sorted(local["event_type"].astype(str).unique())),
                "attention_intensity": float(_num(local, "zscore").clip(lower=0).sum()),
                "max_zscore": float(_num(local, "zscore").max()),
                "max_percentile": float(_num(local, "percentile").max()),
                "source": "dedup_defillama_market",
                "confidence": "market_level",
                "asof_policy": local["asof_policy"].iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values("event_available_time").reset_index(drop=True)


def _event_units(events: pd.DataFrame, mode: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    if mode == "raw_events":
        out = events.copy()
        out["unit_id"] = out["event_id"].astype(str)
        out["primary_event_type"] = out["event_type"].astype(str)
        out["event_count_on_day"] = 1
        out["attention_intensity"] = _num(out, "zscore").clip(lower=0)
        return out
    if mode == "dedup_attention_days":
        out = _dedup_attention_days(events)
        out["unit_id"] = out["attention_day_id"].astype(str)
        return out
    if mode == "intensity_weighted_attention_days":
        out = _dedup_attention_days(events)
        out["unit_id"] = out["attention_day_id"].astype(str)
        return out
    raise KeyError(mode)


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["net20"] = _num(out, "net20") if "net20" in out.columns else _num(out, "net_return")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    out["trade_id"] = (
        out.get("signal_id", out["symbol"].astype(str) + "|" + out["entry_time"].astype(str) + "|" + out["candidate"].astype(str))
        .astype(str)
    )
    out = out.dropna(subset=["entry_time", "net20"]).sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)
    try:
        return _add_asof_burst_phase(out, "1h")
    except Exception:  # noqa: BLE001 - burst phase is diagnostic only
        out["burst_count_so_far"] = np.nan
        out["burst_phase_bucket"] = "unavailable"
        return out


def _market_time_series(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    local = features.copy()
    local["feature_time"] = pd.to_datetime(local["feature_time"], utc=True, errors="coerce")
    return (
        local.groupby("feature_time", as_index=False, sort=True)
        .agg(
            cex_volume_shock_rate=("cex_volume_shock", "mean"),
            cex_price_impulse_rate=("cex_price_impulse", "mean"),
            market_net20_12h=("net20_12h", "mean"),
            market_volume_z_1h=("volume_z_1h", "mean"),
        )
        .dropna(subset=["feature_time"])
    )


def _mode_text(values: pd.Series) -> str:
    values = values.dropna().astype(str)
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def _daily_context(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    local = features.copy()
    local["feature_time"] = pd.to_datetime(local["feature_time"], utc=True, errors="coerce")
    local["event_date"] = local["feature_time"].dt.floor("D")
    daily = (
        local.groupby("event_date", as_index=False, sort=True)
        .agg(
            btc_state=("btc_market_state", _mode_text),
            market_impulse_density_proxy=("cex_volume_shock", "mean"),
            market_price_impulse_density_proxy=("cex_price_impulse", "mean"),
            market_volume_proxy=("volume_z_1h", "mean"),
        )
        .dropna(subset=["event_date"])
    )
    daily["month"] = daily["event_date"].dt.strftime("%Y-%m")
    daily["is_weekend"] = daily["event_date"].dt.dayofweek.ge(5)
    for col, bucket in [
        ("market_impulse_density_proxy", "market_impulse_density_bucket"),
        ("market_volume_proxy", "market_volume_bucket"),
    ]:
        values = pd.to_numeric(daily[col], errors="coerce")
        try:
            daily[bucket] = pd.qcut(values.rank(method="first"), q=5, labels=False, duplicates="drop").astype("Int64").astype(str)
        except ValueError:
            daily[bucket] = "all"
        daily.loc[values.isna(), bucket] = "unknown"
    return daily


def _attach_context(units: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return units.copy()
    out = units.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], utc=True, errors="coerce").dt.floor("D")
    if daily.empty:
        out["context_covered"] = False
        return out
    merged = out.merge(daily, on="event_date", how="left", suffixes=("", "_context"))
    merged["context_covered"] = merged["btc_state"].notna()
    return merged


def _window_trades(units: pd.DataFrame, trades: pd.DataFrame, horizon: pd.Timedelta) -> pd.DataFrame:
    if units.empty or trades.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    trades_sorted = trades.sort_values("entry_time")
    for unit in units.itertuples(index=False):
        start = pd.Timestamp(getattr(unit, "event_available_time"))
        sample = trades_sorted[trades_sorted["entry_time"].gt(start) & trades_sorted["entry_time"].le(start + horizon)]
        if sample.empty:
            continue
        sample = sample.copy()
        sample["source_unit_id"] = str(getattr(unit, "unit_id", getattr(unit, "attention_day_id", "")))
        rows.append(sample)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates("trade_id").reset_index(drop=True)


def _sample_metrics(sample: pd.DataFrame, *, prefix: str = "") -> dict[str, Any]:
    if sample.empty:
        return {
            f"{prefix}cic_trades": 0,
            f"{prefix}net20": np.nan,
            f"{prefix}hit_rate": np.nan,
            f"{prefix}month_cap35_net20": np.nan,
            f"{prefix}max_symbol_contribution": np.nan,
        }
    local = sample.copy()
    local["net_return"] = _num(local, "net20")
    if "month" not in local.columns:
        local["month"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    return {
        f"{prefix}cic_trades": int(len(local)),
        f"{prefix}net20": float(_num(local, "net20").mean()),
        f"{prefix}hit_rate": float(_num(local, "net20").gt(0).mean()),
        f"{prefix}month_cap35_net20": _month_cap_expectancy(local),
        f"{prefix}max_symbol_contribution": _max_contribution(local, "symbol"),
    }


def _units_metrics(units: pd.DataFrame, trades: pd.DataFrame, horizon: pd.Timedelta) -> dict[str, Any]:
    sample = _window_trades(units, trades, horizon)
    return _sample_metrics(sample)


def _asof_policy_summary(events: pd.DataFrame, trades: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    rows = []
    horizon = HORIZONS[cfg.main_horizon]
    for policy in ("same_day_naive", "next_day_conservative"):
        units = _event_units(_asof_events(events, cfg, policy), "dedup_attention_days")
        metrics = _units_metrics(units, trades, horizon)
        rows.append(
            {
                "asof_policy": policy,
                "event_units": int(len(units)),
                "event_available_lag_hours": 0 if policy == "same_day_naive" else 24 + cfg.conservative_available_lag_hours,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _matched_random_table(units: pd.DataFrame, trades: pd.DataFrame, daily: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    if units.empty or daily.empty:
        return pd.DataFrame([{"scope": "no_units_or_context", "events": 0}])
    rng = np.random.default_rng(cfg.random_seed)
    horizon = HORIZONS[cfg.main_horizon]
    context_cols = ["month", "btc_state", "market_impulse_density_bucket", "market_volume_bucket"]
    units_ctx = _attach_context(units, daily)
    scopes: list[tuple[str, pd.DataFrame]] = [("dedup_all", units_ctx)]
    if "primary_event_type" in units_ctx.columns:
        for event_type, group in units_ctx.groupby("primary_event_type", sort=False):
            scopes.append((str(event_type), group))
    rows = []
    days = daily.copy()
    days["event_available_time"] = days["event_date"] + pd.Timedelta(days=1) + pd.Timedelta(hours=cfg.conservative_available_lag_hours)
    for scope, group in scopes:
        group = group[group["context_covered"].fillna(False).astype(bool)].copy()
        if group.empty:
            rows.append({"scope": scope, "events": 0})
            continue
        event_metrics = _units_metrics(group, trades, horizon)
        random_nets = []
        random_caps = []
        for trial in range(cfg.matched_random_trials):
            random_rows: list[dict[str, Any]] = []
            for idx, row in enumerate(group.itertuples(index=False)):
                mask = pd.Series(True, index=days.index)
                for col in context_cols:
                    if col in days.columns and hasattr(row, col):
                        mask &= days[col].astype(str).eq(str(getattr(row, col)))
                pool = days[mask & ~days["event_date"].isin(group["event_date"])]
                if pool.empty:
                    pool = days[days["month"].astype(str).eq(str(getattr(row, "month", "")))]
                if pool.empty:
                    pool = days
                selected = pool.iloc[int(rng.integers(0, len(pool)))]
                random_rows.append(
                    {
                        "unit_id": f"random|{trial}|{idx}",
                        "event_date": selected["event_date"],
                        "event_available_time": selected["event_available_time"],
                    }
                )
            random_units = pd.DataFrame(random_rows)
            random_sample = _window_trades(random_units, trades, horizon)
            random_nets.append(_sample_metrics(random_sample)["net20"])
            random_caps.append(_sample_metrics(random_sample)["month_cap35_net20"])
        random_arr = np.asarray(random_nets, dtype=float)
        random_arr = random_arr[np.isfinite(random_arr)]
        cap_arr = np.asarray(random_caps, dtype=float)
        cap_arr = cap_arr[np.isfinite(cap_arr)]
        event_net = float(event_metrics["net20"]) if pd.notna(event_metrics["net20"]) else np.nan
        rows.append(
            {
                "scope": scope,
                "events": int(len(group)),
                "matched_random_trials": cfg.matched_random_trials,
                "event_cic_trades": event_metrics["cic_trades"],
                "event_net20": event_net,
                "event_month_cap35_net20": event_metrics["month_cap35_net20"],
                "random_median_net20": float(np.nanmedian(random_arr)) if len(random_arr) else np.nan,
                "random_p75_net20": float(np.nanpercentile(random_arr, 75)) if len(random_arr) else np.nan,
                "random_p90_net20": float(np.nanpercentile(random_arr, 90)) if len(random_arr) else np.nan,
                "random_median_month_cap35_net20": float(np.nanmedian(cap_arr)) if len(cap_arr) else np.nan,
                "event_percentile": float((random_arr <= event_net).mean()) if len(random_arr) and pd.notna(event_net) else np.nan,
                "event_minus_random_median": event_net - float(np.nanmedian(random_arr)) if len(random_arr) and pd.notna(event_net) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_month(units: pd.DataFrame, trades: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    horizon = HORIZONS[cfg.main_horizon]
    if units.empty:
        return pd.DataFrame()
    all_metrics = _units_metrics(units, trades, horizon)
    rows = []
    units = units.copy()
    units["month"] = pd.to_datetime(units["event_date"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    for month in sorted(units["month"].dropna().unique()):
        sample_units = units[~units["month"].eq(month)]
        metrics = _units_metrics(sample_units, trades, horizon)
        rows.append(
            {
                "excluded_month": month,
                "remaining_events": int(len(sample_units)),
                "remaining_cic_trades": metrics["cic_trades"],
                "remaining_net20": metrics["net20"],
                "delta_vs_all": metrics["net20"] - all_metrics["net20"] if pd.notna(metrics["net20"]) and pd.notna(all_metrics["net20"]) else np.nan,
                "remaining_month_cap35_net20": metrics["month_cap35_net20"],
            }
        )
    return pd.DataFrame(rows)


def _leave_one_event_type(events: pd.DataFrame, trades: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    horizon = HORIZONS[cfg.main_horizon]
    all_units = _event_units(events, "dedup_attention_days")
    all_metrics = _units_metrics(all_units, trades, horizon)
    rows = []
    for event_type in sorted(events["event_type"].dropna().astype(str).unique()):
        reduced = events[~events["event_type"].astype(str).eq(event_type)]
        units = _event_units(reduced, "dedup_attention_days")
        metrics = _units_metrics(units, trades, horizon)
        rows.append(
            {
                "excluded_event_type": event_type,
                "remaining_events": int(len(units)),
                "remaining_cic_trades": metrics["cic_trades"],
                "remaining_net20": metrics["net20"],
                "delta_vs_all": metrics["net20"] - all_metrics["net20"] if pd.notna(metrics["net20"]) and pd.notna(all_metrics["net20"]) else np.nan,
                "remaining_month_cap35_net20": metrics["month_cap35_net20"],
            }
        )
    return pd.DataFrame(rows)


def _coevent_dedup_summary(events: pd.DataFrame, trades: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    rows = []
    horizon = HORIZONS[cfg.main_horizon]
    for mode in ("raw_events", "dedup_attention_days", "intensity_weighted_attention_days"):
        units = _event_units(events, mode)
        sample = _window_trades(units, trades, horizon)
        metrics = _sample_metrics(sample)
        row = {"event_unit_mode": mode, "event_units": int(len(units)), **metrics}
        if mode == "intensity_weighted_attention_days" and not units.empty:
            day_returns = []
            weights = []
            for unit in units.itertuples(index=False):
                unit_sample = _window_trades(pd.DataFrame([unit._asdict()]), trades, horizon)
                if unit_sample.empty:
                    continue
                day_returns.append(float(_num(unit_sample, "net20").mean()))
                weights.append(float(getattr(unit, "attention_intensity", 1.0)))
            row["intensity_weighted_net20"] = float(np.average(day_returns, weights=weights)) if day_returns and np.sum(weights) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _random_response_units(units: pd.DataFrame, daily: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    if units.empty or daily.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed + 3)
    days = daily.copy()
    days["event_available_time"] = days["event_date"] + pd.Timedelta(days=1) + pd.Timedelta(hours=cfg.conservative_available_lag_hours)
    selected = days.iloc[rng.integers(0, len(days), size=len(units))]
    out = pd.DataFrame(
        {
            "unit_id": [f"random_response|{idx}" for idx in range(len(units))],
            "event_available_time": selected["event_available_time"].to_numpy(),
            "event_date": selected["event_date"].to_numpy(),
        }
    )
    return out


def _response_curve(events: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame, daily: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    market = _market_time_series(features)
    if events.empty or market.empty:
        return pd.DataFrame([{"event_type": "no_events_or_features", "horizon": "", "events": 0}])
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("dedup_attention_day", _event_units(events, "dedup_attention_days"))]
    for event_type, group in events.groupby("event_type", sort=False):
        units = group.copy()
        units["unit_id"] = units["event_id"].astype(str)
        scopes.append((str(event_type), units))
    for event_type, units in scopes:
        random_units = _random_response_units(units, daily, cfg)
        for horizon_name, horizon in HORIZONS.items():
            shock_rates = []
            impulse_rates = []
            market_net = []
            for unit in units.itertuples(index=False):
                start = pd.Timestamp(getattr(unit, "event_available_time"))
                window = market[market["feature_time"].gt(start) & market["feature_time"].le(start + horizon)]
                if window.empty:
                    continue
                shock_rates.append(float(window["cex_volume_shock_rate"].max()))
                impulse_rates.append(float(window["cex_price_impulse_rate"].max()))
                market_net.append(float(window["market_net20_12h"].mean()))
            rand_shock = []
            rand_impulse = []
            if not random_units.empty:
                for unit in random_units.itertuples(index=False):
                    start = pd.Timestamp(getattr(unit, "event_available_time"))
                    window = market[market["feature_time"].gt(start) & market["feature_time"].le(start + horizon)]
                    if window.empty:
                        continue
                    rand_shock.append(float(window["cex_volume_shock_rate"].max()))
                    rand_impulse.append(float(window["cex_price_impulse_rate"].max()))
            cic_sample = _window_trades(units, trades, horizon)
            rows.append(
                {
                    "event_type": event_type,
                    "horizon": horizon_name,
                    "events": int(len(units)),
                    "covered_events": int(len(shock_rates)),
                    "cex_volume_shock_rate": float(np.nanmean(shock_rates)) if shock_rates else np.nan,
                    "cex_price_impulse_rate": float(np.nanmean(impulse_rates)) if impulse_rates else np.nan,
                    "cic_signal_rate": float(len(cic_sample) / len(units)) if len(units) else np.nan,
                    "p2_trade_rate": float(len(cic_sample) / len(units)) if len(units) else np.nan,
                    "future_net20": float(_num(cic_sample, "net20").mean()) if len(cic_sample) else np.nan,
                    "market_net20_12h_after_event": float(np.nanmean(market_net)) if market_net else np.nan,
                    "random_volume_shock_rate": float(np.nanmean(rand_shock)) if rand_shock else np.nan,
                    "random_price_impulse_rate": float(np.nanmean(rand_impulse)) if rand_impulse else np.nan,
                    "volume_shock_lift": float(np.nanmean(shock_rates) - np.nanmean(rand_shock)) if shock_rates and rand_shock else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _has_prior(events: pd.DataFrame, times: pd.Series, window: pd.Timedelta) -> list[bool]:
    if events.empty:
        return [False] * len(times)
    starts = events["event_available_time"].sort_values().reset_index(drop=True)
    flags = []
    for ts in pd.to_datetime(times, utc=True, errors="coerce"):
        prior = starts[starts.le(ts) & starts.gt(ts - window)]
        flags.append(bool(len(prior)))
    return flags


def _interaction_summary(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"module": "p2", "bucket": "no_trades", "trades": 0}])
    rows = []
    modules: list[tuple[str, pd.DataFrame]] = [
        ("P2_all", trades),
        ("CIC1", trades[trades["candidate"].astype(str).eq("CIC1_beta_extreme")]),
        ("CIC2", trades[trades["candidate"].astype(str).eq("CIC2_beta_broad")]),
    ]
    if "burst_count_so_far" in trades.columns:
        modules.append(("O6_late9_candidates", trades[_num(trades, "burst_count_so_far").ge(9)]))
    modules.append(("CP60", pd.DataFrame()))
    for module, sample in modules:
        if sample.empty:
            rows.append({"module": module, "lookback_window": "", "bucket": "insufficient_or_unavailable", "trades": 0})
            continue
        for lookback in [pd.Timedelta(hours=6), pd.Timedelta(hours=12), pd.Timedelta(hours=24), pd.Timedelta(hours=48)]:
            local = sample.copy()
            local["with_prior_onchain_event"] = _has_prior(events, local["entry_time"], lookback)
            for flag, group in local.groupby("with_prior_onchain_event", sort=False):
                metrics = _sample_metrics(group)
                rows.append(
                    {
                        "module": module,
                        "lookback_window": str(lookback),
                        "bucket": "with_prior_onchain_event" if flag else "without_prior_onchain_event",
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _event_type_attribution(events: pd.DataFrame, trades: pd.DataFrame, cfg: V62Config) -> pd.DataFrame:
    rows = []
    horizon = HORIZONS[cfg.main_horizon]
    for event_type, group in events.groupby("event_type", sort=False):
        units = group.copy()
        units["unit_id"] = units["event_id"].astype(str)
        metrics = _units_metrics(units, trades, horizon)
        rows.append({"event_type": event_type, "events": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def _write_notes(path: Path, asof: pd.DataFrame, matched: pd.DataFrame, interaction: pd.DataFrame) -> None:
    lines = [
        "# v6.2 On-chain Attention Attribution",
        "",
        "Status: attribution audit only. No strategy, selector, shadow, or real-live permission is changed.",
        "",
        "Main scope: market-level DeFiLlama daily attention events.",
        "Main as-of policy: next_day_conservative, event_available_time = event_date + 1 day + 04:00 UTC.",
        "Market impulse density is represented by a CEX cross-sectional volume-shock-rate proxy because the feature table has no native market_impulse_density column.",
        "",
    ]
    if not asof.empty and "asof_policy" in asof.columns:
        for row in asof.itertuples(index=False):
            lines.append(
                f"- {row.asof_policy}: events={int(row.event_units)}, trades={int(row.cic_trades)}, net20={row.net20:.4%}."
                if pd.notna(row.net20)
                else f"- {row.asof_policy}: events={int(row.event_units)}, trades={int(row.cic_trades)}, net20=nan."
            )
    if not matched.empty and "scope" in matched.columns:
        valid = matched[matched["scope"].eq("dedup_all")]
        if not valid.empty:
            row = valid.iloc[0]
            lines.extend(
                [
                    "",
                    f"Matched-random dedup_all: event_net20={row.get('event_net20', np.nan):.4%}, "
                    f"random_p75={row.get('random_p75_net20', np.nan):.4%}, percentile={row.get('event_percentile', np.nan):.2f}.",
                ]
            )
    if not interaction.empty and "module" in interaction.columns:
        p2 = interaction[(interaction["module"].eq("P2_all")) & (interaction["lookback_window"].eq(str(pd.Timedelta(hours=24))))]
        if not p2.empty:
            lines.append("")
            for row in p2.itertuples(index=False):
                lines.append(f"P2 24h {row.bucket}: trades={int(row.cic_trades)}, net20={row.net20:.4%}.")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Daily DeFiLlama events are not token/pool-level alpha.",
            "- Conservative as-of and matched random controls must pass before any shadow context is considered.",
            "- Token-level DEX attention requires symbol-to-contract mapping and is deferred to v6.3.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v62_onchain_attention_attribution(cfg: V62Config | None = None) -> dict[str, Path]:
    cfg = cfg or V62Config()
    report_root = ensure_dir(cfg.report_root)
    charts, source_coverage = _fetch_attention_charts(cfg.v61)
    raw_events = _build_events(charts, cfg.v61)
    same_day_events = _asof_events(raw_events, cfg, "same_day_naive")
    events = _asof_events(raw_events, cfg, "next_day_conservative")
    features = _read_features(cfg.v61)
    trades = _prepare_trades(_read_p2_trades(cfg.v61.trade_cache_path))
    daily = _daily_context(features)
    units = _event_units(events, "dedup_attention_days")

    asof_summary = _asof_policy_summary(raw_events, trades, cfg)
    matched = _matched_random_table(units, trades, daily, cfg)
    lomo = _leave_one_month(units, trades, cfg)
    loeo = _leave_one_event_type(events, trades, cfg)
    coevent = _coevent_dedup_summary(events, trades, cfg)
    response = _response_curve(events, features, trades, daily, cfg)
    interaction = _interaction_summary(events, trades)
    event_type_attr = _event_type_attribution(events, trades, cfg)
    attention_days = _attach_context(units, daily)
    coverage = pd.concat(
        [
            source_coverage,
            pd.DataFrame(
                [
                    {"source": "built_raw_events", "event_type": "all", "rows": int(len(raw_events)), "status": "ok" if len(raw_events) else "empty"},
                    {"source": "same_day_naive_events", "event_type": "all", "rows": int(len(same_day_events)), "status": "diagnostic_only"},
                    {"source": "next_day_conservative_events", "event_type": "all", "rows": int(len(events)), "status": "main"},
                    {"source": "dedup_attention_days", "event_type": "all", "rows": int(len(units)), "status": "main"},
                    {"source": "cex_feature_table", "event_type": "cex", "rows": int(len(features)), "status": "ok" if len(features) else "missing_or_empty"},
                    {"source": "p2_trade_cache", "event_type": "cic", "rows": int(len(trades)), "status": "ok" if len(trades) else "missing_or_empty"},
                ]
            ),
        ],
        ignore_index=True,
    )

    outputs = {
        "onchain_event_coverage": report_root / "onchain_event_coverage.csv",
        "asof_policy_summary": report_root / "asof_policy_summary.csv",
        "event_day_matched_random": report_root / "event_day_matched_random.csv",
        "leave_one_month": report_root / "leave_one_month.csv",
        "leave_one_event_type": report_root / "leave_one_event_type.csv",
        "coevent_dedup_summary": report_root / "coevent_dedup_summary.csv",
        "onchain_to_cex_response_curve": report_root / "onchain_to_cex_response_curve.csv",
        "onchain_cic_interaction": report_root / "onchain_cic_interaction.csv",
        "event_type_attribution": report_root / "event_type_attribution.csv",
        "onchain_attention_days": report_root / "onchain_attention_days.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["onchain_event_coverage"], index=False)
    asof_summary.to_csv(outputs["asof_policy_summary"], index=False)
    matched.to_csv(outputs["event_day_matched_random"], index=False)
    lomo.to_csv(outputs["leave_one_month"], index=False)
    loeo.to_csv(outputs["leave_one_event_type"], index=False)
    coevent.to_csv(outputs["coevent_dedup_summary"], index=False)
    response.to_csv(outputs["onchain_to_cex_response_curve"], index=False)
    interaction.to_csv(outputs["onchain_cic_interaction"], index=False)
    event_type_attr.to_csv(outputs["event_type_attribution"], index=False)
    attention_days.to_csv(outputs["onchain_attention_days"], index=False)
    _write_notes(outputs["candidate_notes"], asof_summary, matched, interaction)
    return outputs

