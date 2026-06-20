"""v6.1 On-chain / DEX Attention Event Backfill.

First on-chain pass: build market-level DEX/fee/stablecoin attention events
from public DeFiLlama datasets, then attribute whether those events lead CEX
volume shock, price impulse, and CIC/P2 outcomes.  This is a propagation audit,
not a trading rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v42_source_attribution_target_fusion import DEFAULT_TRADE_CACHE
from pressure_graph.reports.v50_perp_crowding_atlas import DEFAULT_FEATURE_PATH, UNIVERSE_COL, _available_columns, _bool, _num


REPORT_ROOT = Path("reports/v6_1_onchain_dex_attention_backfill")
DEFAULT_CACHE_ROOT = Path("data/external/defillama")

DEX_URL = "https://api.llama.fi/overview/dexs?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true"
FEES_URL = "https://api.llama.fi/overview/fees?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true"
STABLECOIN_URL = "https://stablecoins.llama.fi/stablecoincharts/all"


@dataclass(frozen=True)
class V61Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = DEFAULT_FEATURE_PATH
    trade_cache_path: Path = DEFAULT_TRADE_CACHE
    cache_root: Path = DEFAULT_CACHE_ROOT
    universe_col: str = UNIVERSE_COL
    cost_bps: float = 20.0
    z_threshold: float = 2.0
    pct_threshold: float = 0.95
    lookback_days: int = 90
    min_lookback_days: int = 30
    allow_network: bool = True
    random_seed: int = 610


FEATURE_COLUMNS = (
    "symbol",
    "feature_time",
    "warmup_complete",
    UNIVERSE_COL,
    "universe_static_current_top30",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "volume_z_1h",
    "volume_z_4h",
    "future_ret_4h",
    "future_ret_12h",
    "btc_market_state",
)

HORIZONS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "12h": pd.Timedelta(hours=12),
    "24h": pd.Timedelta(hours=24),
}


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def _get_json(url: str, path: Path, allow_network: bool) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not allow_network:
        return {}
    ensure_dir(path.parent)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text
    path.write_text(text, encoding="utf-8")
    return response.json()


def _chart_from_pairs(data: Any, event_type: str, source: str) -> pd.DataFrame:
    pairs = []
    if isinstance(data, dict):
        pairs = data.get("totalDataChart", [])
    elif isinstance(data, list):
        pairs = data
    rows = []
    for item in pairs or []:
        if isinstance(item, dict):
            ts = item.get("date") or item.get("timestamp")
            value = item.get("totalCirculatingUSD", {}).get("peggedUSD") if isinstance(item.get("totalCirculatingUSD"), dict) else item.get("value")
        else:
            ts, value = item[0], item[1]
        rows.append({"event_time": pd.to_datetime(int(ts), unit="s", utc=True, errors="coerce"), "raw_value": pd.to_numeric(value, errors="coerce")})
    out = pd.DataFrame(rows).dropna(subset=["event_time", "raw_value"])
    if out.empty:
        return out
    out["event_type"] = event_type
    out["source"] = source
    return out.sort_values("event_time").reset_index(drop=True)


def _fetch_attention_charts(cfg: V61Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = ensure_dir(cfg.cache_root)
    datasets = [
        ("chain_dex_volume_spike", "defillama_dex_volume", DEX_URL, cache / "defillama_dexs_overview.json"),
        ("protocol_fee_spike", "defillama_fees", FEES_URL, cache / "defillama_fees_overview.json"),
        ("stablecoin_supply_or_flow_spike", "defillama_stablecoins", STABLECOIN_URL, cache / "defillama_stablecoincharts_all.json"),
    ]
    frames = []
    coverage = []
    for event_type, source, url, path in datasets:
        try:
            data = _get_json(url, path, cfg.allow_network)
            frame = _chart_from_pairs(data, event_type, source)
            status = "ok" if not frame.empty else "missing_or_empty"
        except Exception as exc:  # noqa: BLE001 - one source should not kill the atlas
            frame = pd.DataFrame()
            status = f"error:{type(exc).__name__}"
        coverage.append(
            {
                "source": source,
                "event_type": event_type,
                "url": url,
                "cache_path": str(path),
                "rows": int(len(frame)),
                "first_time": frame["event_time"].min() if not frame.empty else pd.NaT,
                "last_time": frame["event_time"].max() if not frame.empty else pd.NaT,
                "status": status,
            }
        )
        if not frame.empty:
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), pd.DataFrame(coverage))


def _rolling_percentile(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    def pct(x: np.ndarray) -> float:
        if len(x) <= 1 or not np.isfinite(x[-1]):
            return np.nan
        prior = x[:-1]
        prior = prior[np.isfinite(prior)]
        if len(prior) == 0:
            return np.nan
        return float((prior <= x[-1]).mean())

    return values.rolling(window + 1, min_periods=min_periods + 1).apply(pct, raw=True)


def _build_events(charts: pd.DataFrame, cfg: V61Config) -> pd.DataFrame:
    if charts.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "symbol",
                "event_time",
                "event_type",
                "chain",
                "token_address",
                "source",
                "raw_value",
                "zscore",
                "percentile",
                "lookback_window",
                "mapped_cex_symbol",
                "confidence",
            ]
        )
    rows = []
    for (event_type, source), group in charts.groupby(["event_type", "source"], sort=False):
        local = group.sort_values("event_time").copy()
        value = _num(local, "raw_value")
        if event_type == "stablecoin_supply_or_flow_spike":
            metric = value.diff()
        else:
            metric = value
        mean = metric.shift(1).rolling(cfg.lookback_days, min_periods=cfg.min_lookback_days).mean()
        std = metric.shift(1).rolling(cfg.lookback_days, min_periods=cfg.min_lookback_days).std(ddof=0).replace(0, np.nan)
        local["zscore"] = (metric - mean) / std
        local["percentile"] = _rolling_percentile(metric, cfg.lookback_days, cfg.min_lookback_days)
        local["raw_metric"] = metric
        mask = local["zscore"].ge(cfg.z_threshold) | local["percentile"].ge(cfg.pct_threshold)
        for row in local[mask.fillna(False)].itertuples(index=False):
            ts = pd.Timestamp(getattr(row, "event_time"))
            rows.append(
                {
                    "event_id": f"{source}|{event_type}|{ts.date()}",
                    "symbol": "__MARKET__",
                    "event_time": ts,
                    "event_type": event_type,
                    "chain": "__GLOBAL__",
                    "token_address": "",
                    "source": source,
                    "raw_value": float(getattr(row, "raw_value")),
                    "zscore": float(getattr(row, "zscore")) if pd.notna(getattr(row, "zscore")) else np.nan,
                    "percentile": float(getattr(row, "percentile")) if pd.notna(getattr(row, "percentile")) else np.nan,
                    "lookback_window": cfg.lookback_days,
                    "mapped_cex_symbol": "__MARKET__",
                    "confidence": "market_level",
                }
            )
    return pd.DataFrame(rows).sort_values(["event_time", "event_type"]).reset_index(drop=True)


def _read_features(cfg: V61Config) -> pd.DataFrame:
    if not cfg.feature_path.exists():
        return pd.DataFrame()
    cols = _available_columns(cfg.feature_path, tuple(dict.fromkeys([*FEATURE_COLUMNS, cfg.universe_col])), cfg.universe_col)
    pf = pq.ParquetFile(cfg.feature_path)
    frames: list[pd.DataFrame] = []
    for idx in range(pf.num_row_groups):
        chunk = pf.read_row_group(idx, columns=cols).to_pandas()
        if cfg.universe_col in chunk.columns:
            chunk = chunk[_bool(chunk, cfg.universe_col)].copy()
        elif "universe_static_current_top30" in chunk.columns:
            chunk = chunk[_bool(chunk, "universe_static_current_top30")].copy()
        if "warmup_complete" in chunk.columns:
            chunk = chunk[_bool(chunk, "warmup_complete", True)].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["feature_time"] = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    out["cex_volume_shock"] = _num(out, "volume_z_1h").ge(2.0)
    out["cex_price_impulse"] = _num(out, "ret_1h").ge(0.004) & _num(out, "volume_z_1h").ge(1.5)
    out["net20_12h"] = _num(out, "future_ret_12h") - 2.0 * cfg.cost_bps / 10_000.0
    return out.dropna(subset=["symbol", "feature_time"]).sort_values(["symbol", "feature_time"]).reset_index(drop=True)


def _read_p2_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path, low_memory=False)
    if trades.empty or "candidate" not in trades.columns:
        return pd.DataFrame()
    out = trades[trades["candidate"].astype(str).isin(["CIC1_beta_extreme", "CIC2_beta_broad"])].copy()
    out["entry_time"] = pd.to_datetime(out.get("entry_time"), utc=True, errors="coerce")
    out["net20"] = _num(out, "net_return")
    return out.dropna(subset=["entry_time"]).copy()


def _event_response(events: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame, cfg: V61Config) -> pd.DataFrame:
    if events.empty or features.empty:
        return pd.DataFrame([{"event_type": "no_events_or_features", "horizon": "", "events": 0}])
    rows = []
    for event_type, group in events.groupby("event_type", sort=False):
        for horizon, delta in HORIZONS.items():
            vol_rates = []
            impulse_rates = []
            market_net = []
            cic_count = []
            cic_net = []
            for event in group.itertuples(index=False):
                ts = pd.Timestamp(getattr(event, "event_time"))
                window = features[features["feature_time"].gt(ts) & features["feature_time"].le(ts + delta)]
                if window.empty:
                    continue
                by_time = window.groupby("feature_time", sort=False).agg(
                    volume_shock_rate=("cex_volume_shock", "mean"),
                    price_impulse_rate=("cex_price_impulse", "mean"),
                    market_net20=("net20_12h", "mean"),
                )
                vol_rates.append(float(by_time["volume_shock_rate"].max()))
                impulse_rates.append(float(by_time["price_impulse_rate"].max()))
                market_net.append(float(by_time["market_net20"].mean()))
                trade_window = trades[trades["entry_time"].gt(ts) & trades["entry_time"].le(ts + delta)] if not trades.empty else pd.DataFrame()
                cic_count.append(int(len(trade_window)))
                cic_net.append(float(_num(trade_window, "net20").mean()) if len(trade_window) else np.nan)
            rows.append(
                {
                    "event_type": event_type,
                    "horizon": horizon,
                    "events": int(len(group)),
                    "covered_events": int(len(vol_rates)),
                    "max_cex_volume_shock_rate": float(np.nanmean(vol_rates)) if vol_rates else np.nan,
                    "max_cex_price_impulse_rate": _nanmean(impulse_rates),
                    "market_net20_12h_after_event": _nanmean(market_net),
                    "cic_trades_after_event": int(np.nansum(cic_count)) if cic_count else 0,
                    "cic_net20_after_event": _nanmean(cic_net),
                }
            )
    return pd.DataFrame(rows)


def _onchain_cic_fusion(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if events.empty or trades.empty:
        return pd.DataFrame([{"bucket": "no_events_or_trades", "trades": 0}])
    events = events.sort_values("event_time")
    rows = []
    for window in [pd.Timedelta(hours=4), pd.Timedelta(hours=12), pd.Timedelta(hours=24)]:
        flags = []
        for entry in trades["entry_time"]:
            prior = events[events["event_time"].le(entry) & events["event_time"].gt(entry - window)]
            flags.append(bool(len(prior)))
        local = trades.copy()
        local["with_prior_onchain_event"] = flags
        for flag, group in local.groupby("with_prior_onchain_event", sort=False):
            rows.append(
                {
                    "lookback_window": str(window),
                    "bucket": "with_prior_onchain_event" if flag else "without_prior_onchain_event",
                    "trades": int(len(group)),
                    "net20": float(_num(group, "net20").mean()) if len(group) else np.nan,
                    "hit_rate": float(_num(group, "net20").gt(0).mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _event_type_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame([{"event_type": "no_onchain_events_built", "events": 0}])
    return events.groupby(["event_type", "source", "confidence"], as_index=False, sort=False).agg(
        events=("event_id", "count"),
        first_event=("event_time", "min"),
        last_event=("event_time", "max"),
        avg_zscore=("zscore", "mean"),
        avg_percentile=("percentile", "mean"),
    )


def _random_time_control(events: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame, cfg: V61Config) -> pd.DataFrame:
    if events.empty or features.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed)
    days = pd.Series(features["feature_time"].dt.floor("D").dropna().unique()).sort_values()
    if days.empty:
        return pd.DataFrame()
    random_events = events.copy()
    random_events["event_time"] = rng.choice(days.to_numpy(), size=len(events), replace=True)
    random_events["event_type"] = "random_time_control"
    return _event_response(random_events, features, trades, cfg)


def _random_symbol_control(events: pd.DataFrame, features: pd.DataFrame, cfg: V61Config) -> pd.DataFrame:
    if events.empty or features.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed + 1)
    symbols = features["symbol"].dropna().astype(str).unique()
    rows = []
    for event_type, group in events.groupby("event_type", sort=False):
        symbol_nets = []
        shock_rates = []
        for event in group.itertuples(index=False):
            ts = pd.Timestamp(getattr(event, "event_time"))
            symbol = str(rng.choice(symbols))
            window = features[
                features["symbol"].astype(str).eq(symbol)
                & features["feature_time"].gt(ts)
                & features["feature_time"].le(ts + pd.Timedelta(hours=12))
            ]
            if window.empty:
                continue
            symbol_nets.append(float(_num(window, "net20_12h").mean()))
            shock_rates.append(float(window["cex_volume_shock"].mean()))
        rows.append(
            {
                "event_type": event_type,
                "control": "random_symbol_12h",
                "events": int(len(group)),
                "covered_events": int(len(symbol_nets)),
                "random_symbol_net20_12h": float(np.nanmean(symbol_nets)) if symbol_nets else np.nan,
                "random_symbol_volume_shock_rate": float(np.nanmean(shock_rates)) if shock_rates else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _coverage(source_coverage: pd.DataFrame, events: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = source_coverage.to_dict("records") if not source_coverage.empty else []
    rows.extend(
        [
            {"source": "built_onchain_attention_events", "event_type": "all", "rows": int(len(events)), "status": "ok" if len(events) else "empty"},
            {"source": "cex_feature_table", "event_type": "cex", "rows": int(len(features)), "status": "ok" if len(features) else "missing_or_empty"},
            {"source": "p2_trade_cache", "event_type": "cic", "rows": int(len(trades)), "status": "ok" if len(trades) else "missing_or_empty"},
        ]
    )
    return pd.DataFrame(rows)


def _write_notes(path: Path, coverage: pd.DataFrame, response: pd.DataFrame, fusion: pd.DataFrame) -> None:
    total = coverage[coverage["source"].eq("built_onchain_attention_events")]
    total_events = int(total["rows"].iloc[0]) if not total.empty else 0
    lines = [
        "# v6.1 On-chain / DEX Attention Event Backfill",
        "",
        "Status: propagation audit only. No strategy, gate, selector, shadow, or real-live permission is changed.",
        "",
        f"Built attention events: {total_events}.",
        "Current event level: market/global DeFiLlama DEX volume, fees, and stablecoin flow spikes.",
        "Token/pool-level GeckoTerminal/CoinGecko events still require token address mapping before use.",
        "",
    ]
    if not response.empty and "max_cex_volume_shock_rate" in response.columns:
        best = response.sort_values("max_cex_volume_shock_rate", ascending=False).head(1).iloc[0]
        lines.append(
            f"Best CEX volume response: {best['event_type']} {best['horizon']} shock_rate={best['max_cex_volume_shock_rate']:.4%}."
        )
    if not fusion.empty and "net20" in fusion.columns:
        valid = fusion.dropna(subset=["net20"])
        if not valid.empty:
            best = valid.sort_values("net20", ascending=False).head(1).iloc[0]
            lines.append(f"Best CIC fusion bucket: {best['lookback_window']} {best['bucket']} net20={best['net20']:.4%}, trades={int(best['trades'])}.")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Daily market-level events can only test broad propagation, not token-specific DEX lead-lag.",
            "- Passing propagation is required before any strategy candidate is considered.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v61_onchain_dex_attention_backfill(cfg: V61Config | None = None) -> dict[str, Path]:
    cfg = cfg or V61Config()
    report_root = ensure_dir(cfg.report_root)
    charts, source_coverage = _fetch_attention_charts(cfg)
    events = _build_events(charts, cfg)
    features = _read_features(cfg)
    trades = _read_p2_trades(cfg.trade_cache_path)
    coverage = _coverage(source_coverage, events, features, trades)
    mapping = pd.DataFrame(
        [
            {
                "mapped_cex_symbol": "__MARKET__",
                "mapping_level": "market",
                "confidence": "market_level",
                "events": int(len(events)),
                "notes": "Token/pool-level symbol mapping not available in v6.1.",
            }
        ]
    )
    response = _event_response(events, features, trades, cfg)
    fusion = _onchain_cic_fusion(events, trades)
    event_type = _event_type_summary(events)
    lead_lag = response.copy()
    random_time = _random_time_control(events, features, trades, cfg)
    random_symbol = _random_symbol_control(events, features, cfg)

    outputs = {
        "onchain_event_coverage": report_root / "onchain_event_coverage.csv",
        "symbol_mapping_quality": report_root / "symbol_mapping_quality.csv",
        "onchain_to_cex_response_curve": report_root / "onchain_to_cex_response_curve.csv",
        "onchain_cic_fusion_summary": report_root / "onchain_cic_fusion_summary.csv",
        "event_type_summary": report_root / "event_type_summary.csv",
        "lead_lag_horizon_summary": report_root / "lead_lag_horizon_summary.csv",
        "random_time_control": report_root / "random_time_control.csv",
        "random_symbol_control": report_root / "random_symbol_control.csv",
        "onchain_attention_events": report_root / "onchain_attention_events.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["onchain_event_coverage"], index=False)
    mapping.to_csv(outputs["symbol_mapping_quality"], index=False)
    response.to_csv(outputs["onchain_to_cex_response_curve"], index=False)
    fusion.to_csv(outputs["onchain_cic_fusion_summary"], index=False)
    event_type.to_csv(outputs["event_type_summary"], index=False)
    lead_lag.to_csv(outputs["lead_lag_horizon_summary"], index=False)
    random_time.to_csv(outputs["random_time_control"], index=False)
    random_symbol.to_csv(outputs["random_symbol_control"], index=False)
    events.to_csv(outputs["onchain_attention_events"], index=False)
    _write_notes(outputs["candidate_notes"], coverage, response, fusion)
    return outputs
