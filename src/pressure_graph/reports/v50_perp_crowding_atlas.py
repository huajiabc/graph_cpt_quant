"""v5.0 Perp Crowding Atlas.

This is an attribution-first atlas for funding/OI crowding.  It does not
promote a directional short, a selector, or a live overlay.  The first pass
asks whether perp crowding is better treated as relative-value exhaustion,
momentum confirmation, or a diagnostic context for the existing CIC/P2 long
stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v5_0_perp_crowding_atlas")
DEFAULT_FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
DEFAULT_TRADE_CACHE = Path("reports/v1_3a_checkpoint_robustness/_v09b_trades_tmp.csv")
UNIVERSE_COL = "universe_dynamic_monthly_top30"


@dataclass(frozen=True)
class V50Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = DEFAULT_FEATURE_PATH
    trade_cache_path: Path = DEFAULT_TRADE_CACHE
    universe_col: str = UNIVERSE_COL
    cost_bps: float = 20.0
    funding_high: float = 0.80
    funding_extreme: float = 0.90
    funding_not_hot: float = 0.70
    oi_high: float = 0.80
    oi_extreme: float = 0.90
    ret_high: float = 0.75
    ret_mid_low: float = 0.35
    ret_mid_high: float = 0.65
    volume_z_impulse: float = 1.0


FEATURE_COLUMNS = (
    "symbol",
    "feature_time",
    "bar_close_time",
    "close",
    "high",
    "low",
    "volume",
    "turnover",
    "warmup_complete",
    UNIVERSE_COL,
    "universe_static_current_top30",
    "funding_rate_settled",
    "funding_z",
    "funding_percentile",
    "oi_base",
    "oi_value_usdt",
    "oi_delta_1h",
    "oi_delta_4h",
    "oi_value_delta_1h",
    "oi_value_delta_4h",
    "oi_value_delta_z_1h",
    "oi_value_delta_z_4h",
    "oi_value_delta_1h_percentile",
    "oi_value_delta_4h_percentile",
    "oi_delta_1h_percentile",
    "oi_delta_4h_percentile",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "ret_4h_percentile",
    "volume_z_1h",
    "volume_z_4h",
    "volume_1h_percentile",
    "volume_4h_percentile",
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_market_state",
    "btc_vol_regime",
    "future_max_up_4h",
    "future_max_down_4h",
    "future_ret_4h",
    "future_max_up_12h",
    "future_max_down_12h",
    "future_ret_12h",
)


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _bool(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="bool")
    return frame[col].fillna(default).astype(bool)


def _net_at_cost(gross: pd.Series | np.ndarray | float, cost_bps: float) -> pd.Series:
    return pd.to_numeric(pd.Series(gross), errors="coerce") - 2.0 * float(cost_bps) / 10_000.0


def _pct01(series: pd.Series) -> pd.Series:
    """Normalize percentile-like columns that may be stored as 0-1 or 0-100."""
    values = pd.to_numeric(series, errors="coerce")
    max_value = values.dropna().quantile(0.99) if values.notna().any() else np.nan
    if np.isfinite(max_value) and max_value > 1.5:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)


def _month_cap35(frame: pd.DataFrame, value_col: str) -> float:
    if frame.empty or value_col not in frame.columns:
        return np.nan
    local = frame.copy()
    if "month" not in local.columns:
        local["month"] = pd.to_datetime(local.get("feature_time"), utc=True, errors="coerce").dt.strftime("%Y-%m")
    values = _num(local, value_col)
    total = float(values.sum())
    if not np.isfinite(total) or abs(total) < 1e-12:
        return float(values.mean()) if len(values) else np.nan
    month_sum = values.groupby(local["month"]).sum()
    capped_sum = month_sum.clip(lower=-0.35 * abs(total), upper=0.35 * abs(total)).sum()
    return float(capped_sum / len(values)) if len(values) else np.nan


def _max_contribution(frame: pd.DataFrame, key_col: str, value_col: str) -> float:
    if frame.empty or key_col not in frame.columns or value_col not in frame.columns:
        return np.nan
    total = float(_num(frame, value_col).sum())
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.nan
    grouped = _num(frame, value_col).groupby(frame[key_col].astype(str)).sum()
    return float((grouped / total).abs().max()) if len(grouped) else np.nan


def _available_columns(path: Path, desired: tuple[str, ...], universe_col: str) -> list[str]:
    pf = pq.ParquetFile(path)
    schema_cols = set(pf.schema.names)
    cols = []
    for col in desired:
        requested = universe_col if col == UNIVERSE_COL else col
        if requested in schema_cols and requested not in cols:
            cols.append(requested)
    return cols


def _read_feature_subset(cfg: V50Config) -> pd.DataFrame:
    if not cfg.feature_path.exists():
        return pd.DataFrame()
    desired = tuple(dict.fromkeys([*(FEATURE_COLUMNS), cfg.universe_col]))
    cols = _available_columns(cfg.feature_path, desired, cfg.universe_col)
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
    if "bar_close_time" in out.columns:
        out["bar_close_time"] = pd.to_datetime(out["bar_close_time"], utc=True, errors="coerce")
    else:
        out["bar_close_time"] = out["feature_time"]
    out = out.dropna(subset=["symbol", "feature_time"]).sort_values(["symbol", "feature_time"]).reset_index(drop=True)
    return out


def _add_future_24h(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty or "close" not in out.columns:
        out["future_ret_24h"] = np.nan
        return out
    local = out.sort_values(["symbol", "feature_time"]).copy()
    future_close = local.groupby("symbol", sort=False)["close"].shift(-96)
    future_time = local.groupby("symbol", sort=False)["feature_time"].shift(-96)
    valid = (pd.to_datetime(future_time, utc=True, errors="coerce") - local["feature_time"]).eq(pd.Timedelta(hours=24))
    local["future_ret_24h"] = np.where(valid, pd.to_numeric(future_close, errors="coerce") / _num(local, "close") - 1.0, np.nan)
    return local


def _add_context_columns(frame: pd.DataFrame, cfg: V50Config) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = _add_future_24h(frame)
    out["month"] = out["feature_time"].dt.strftime("%Y-%m")
    out["funding_pct01"] = _pct01(_num(out, "funding_percentile"))
    out["ret_4h_pct01"] = _pct01(_num(out, "ret_4h_percentile"))
    out["oi_crowding_percentile"] = _num(out, "oi_value_delta_4h_percentile").combine_first(_num(out, "oi_delta_4h_percentile"))
    out["oi_crowding_pct01"] = _pct01(out["oi_crowding_percentile"])
    out["funding_bucket"] = np.select(
        [
            out["funding_pct01"].ge(cfg.funding_extreme),
            out["funding_pct01"].ge(cfg.funding_high),
            out["funding_pct01"].le(0.20),
        ],
        ["funding_extreme", "funding_high", "funding_low"],
        default="funding_mid",
    )
    out["oi_bucket"] = np.select(
        [
            out["oi_crowding_pct01"].ge(cfg.oi_extreme),
            out["oi_crowding_pct01"].ge(cfg.oi_high),
            out["oi_crowding_pct01"].le(0.20),
        ],
        ["oi_extreme", "oi_high", "oi_low"],
        default="oi_mid",
    )
    ret_pct = out["ret_4h_pct01"]
    out["price_state"] = np.select(
        [ret_pct.ge(cfg.ret_high), ret_pct.le(0.25), ret_pct.between(cfg.ret_mid_low, cfg.ret_mid_high)],
        ["price_up", "price_down", "price_stall"],
        default="price_mid",
    )
    out["volume_impulse_state"] = np.where(
        (_num(out, "volume_z_1h").ge(cfg.volume_z_impulse) | _num(out, "volume_z_4h").ge(cfg.volume_z_impulse)),
        "volume_impulse",
        "volume_normal",
    )
    out["high_funding"] = out["funding_pct01"].ge(cfg.funding_high)
    out["funding_not_hot"] = out["funding_pct01"].lt(cfg.funding_not_hot)
    out["high_oi"] = out["oi_crowding_pct01"].ge(cfg.oi_high)
    out["oi_moderate"] = out["oi_crowding_pct01"].between(0.40, cfg.oi_high, inclusive="left")
    out["price_up"] = ret_pct.ge(cfg.ret_high)
    out["price_stall"] = ret_pct.between(cfg.ret_mid_low, cfg.ret_mid_high) | _num(out, "ret_4h").abs().le(0.003)
    out["price_impulse"] = out["price_up"] & out["volume_impulse_state"].eq("volume_impulse")

    btc = out[out["symbol"].astype(str).eq("BTCUSDT")][["feature_time", "future_ret_4h", "future_ret_12h", "future_ret_24h"]].copy()
    btc = btc.rename(columns={col: f"btc_{col}" for col in ["future_ret_4h", "future_ret_12h", "future_ret_24h"]})
    out = out.merge(btc, on="feature_time", how="left")
    market = (
        out.groupby("feature_time", sort=False)[["future_ret_4h", "future_ret_12h", "future_ret_24h"]]
        .mean(numeric_only=True)
        .rename(columns={col: f"market_{col}" for col in ["future_ret_4h", "future_ret_12h", "future_ret_24h"]})
        .reset_index()
    )
    out = out.merge(market, on="feature_time", how="left")
    for horizon in ("4h", "12h", "24h"):
        out[f"long_net20_{horizon}"] = _net_at_cost(_num(out, f"future_ret_{horizon}"), cfg.cost_bps).to_numpy()
        out[f"short_net20_{horizon}"] = _net_at_cost(-_num(out, f"future_ret_{horizon}"), cfg.cost_bps).to_numpy()
        out[f"short_long_btc_net20_{horizon}"] = _net_at_cost(_num(out, f"btc_future_ret_{horizon}") - _num(out, f"future_ret_{horizon}"), cfg.cost_bps).to_numpy()
        out[f"short_long_market_net20_{horizon}"] = _net_at_cost(_num(out, f"market_future_ret_{horizon}") - _num(out, f"future_ret_{horizon}"), cfg.cost_bps).to_numpy()
        out[f"long_short_btc_net20_{horizon}"] = _net_at_cost(_num(out, f"future_ret_{horizon}") - _num(out, f"btc_future_ret_{horizon}"), cfg.cost_bps).to_numpy()
    return out


def _state_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    idx = frame.index
    return {
        "all_top_universe": pd.Series(True, index=idx),
        "high_funding_high_oi": frame["high_funding"] & frame["high_oi"],
        "high_funding_high_oi_price_stall": frame["high_funding"] & frame["high_oi"] & frame["price_stall"],
        "high_funding_high_oi_price_up": frame["high_funding"] & frame["high_oi"] & frame["price_up"],
        "funding_not_hot_oi_moderate_price_impulse": frame["funding_not_hot"] & frame["oi_moderate"] & frame["price_impulse"],
        "oi_expansion_funding_safe": frame["high_oi"] & frame["funding_not_hot"],
        "funding_extreme_only": frame["funding_bucket"].eq("funding_extreme"),
        "oi_extreme_only": frame["oi_bucket"].eq("oi_extreme"),
    }


def _summary_metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    if frame.empty:
        return {
            "state": label,
            "events": 0,
            "long_net20_4h": np.nan,
            "long_net20_12h": np.nan,
            "long_net20_24h": np.nan,
            "short_net20_12h": np.nan,
            "short_long_btc_net20_12h": np.nan,
            "short_long_market_net20_12h": np.nan,
            "long_short_btc_net20_12h": np.nan,
            "hit_long_12h": np.nan,
            "underperform_btc_12h": np.nan,
            "month_cap35_long_net20_12h": np.nan,
            "month_cap35_short_long_btc_net20_12h": np.nan,
            "worst_month_long_net20_12h": np.nan,
            "max_symbol_contribution_long_net20_12h": np.nan,
        }
    local = frame.copy()
    monthly = local.groupby("month", sort=False)["long_net20_12h"].mean()
    return {
        "state": label,
        "events": int(len(local)),
        "long_net20_4h": float(_num(local, "long_net20_4h").mean()),
        "long_net20_12h": float(_num(local, "long_net20_12h").mean()),
        "long_net20_24h": float(_num(local, "long_net20_24h").mean()),
        "short_net20_12h": float(_num(local, "short_net20_12h").mean()),
        "short_long_btc_net20_12h": float(_num(local, "short_long_btc_net20_12h").mean()),
        "short_long_market_net20_12h": float(_num(local, "short_long_market_net20_12h").mean()),
        "long_short_btc_net20_12h": float(_num(local, "long_short_btc_net20_12h").mean()),
        "hit_long_12h": float(_num(local, "long_net20_12h").gt(0).mean()),
        "underperform_btc_12h": float(_num(local, "future_ret_12h").lt(_num(local, "btc_future_ret_12h")).mean()),
        "month_cap35_long_net20_12h": _month_cap35(local, "long_net20_12h"),
        "month_cap35_short_long_btc_net20_12h": _month_cap35(local, "short_long_btc_net20_12h"),
        "worst_month_long_net20_12h": float(monthly.min()) if len(monthly) else np.nan,
        "max_symbol_contribution_long_net20_12h": _max_contribution(local, "symbol", "long_net20_12h"),
    }


def _crowding_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_summary_metrics(frame[mask.fillna(False)], name) for name, mask in _state_masks(frame).items()])


def _funding_oi_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (funding, oi), group in frame.groupby(["funding_bucket", "oi_bucket"], sort=False, dropna=False):
        row = _summary_metrics(group, f"{funding}|{oi}")
        row["funding_bucket"] = funding
        row["oi_bucket"] = oi
        rows.append(row)
    return pd.DataFrame(rows)


def _relative_value_pair_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = _state_masks(frame)
    candidates = {
        "short_high_crowding_long_btc": masks["high_funding_high_oi"],
        "short_high_crowding_stall_long_btc": masks["high_funding_high_oi_price_stall"],
        "short_high_crowding_long_market": masks["high_funding_high_oi"],
        "long_funding_safe_momentum_short_btc": masks["funding_not_hot_oi_moderate_price_impulse"],
        "random_matched_short_long_btc": masks["all_top_universe"],
    }
    for name, mask in candidates.items():
        sample = frame[mask.fillna(False)].copy()
        if name == "random_matched_short_long_btc" and not sample.empty:
            sample = sample.sample(n=min(len(sample), int(masks["high_funding_high_oi"].sum())), random_state=42)
        if name == "short_high_crowding_long_market":
            value_col = "short_long_market_net20_12h"
        elif name == "long_funding_safe_momentum_short_btc":
            value_col = "long_short_btc_net20_12h"
        else:
            value_col = "short_long_btc_net20_12h"
        rows.append(
            {
                "candidate": name,
                "events": int(len(sample)),
                "pair_net20_4h": float(_num(sample, value_col.replace("12h", "4h")).mean()) if len(sample) else np.nan,
                "pair_net20_12h": float(_num(sample, value_col).mean()) if len(sample) else np.nan,
                "pair_net20_24h": float(_num(sample, value_col.replace("12h", "24h")).mean()) if len(sample) else np.nan,
                "pair_hit_rate_12h": float(_num(sample, value_col).gt(0).mean()) if len(sample) else np.nan,
                "month_cap35_pair_net20_12h": _month_cap35(sample, value_col),
                "worst_month_pair_net20_12h": float(_num(sample, value_col).groupby(sample["month"]).mean().min()) if len(sample) else np.nan,
                "max_symbol_contribution_pair_net20_12h": _max_contribution(sample, "symbol", value_col),
            }
        )
    return pd.DataFrame(rows)


def _load_p2_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path, low_memory=False)
    if trades.empty or "candidate" not in trades.columns:
        return pd.DataFrame()
    local = trades[trades["candidate"].astype(str).isin(["CIC1_beta_extreme", "CIC2_beta_broad"])].copy()
    if local.empty:
        return local
    local["entry_time"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce")
    local["signal_time"] = pd.to_datetime(local.get("signal_time", local["entry_time"]), utc=True, errors="coerce")
    local["month"] = local["entry_time"].dt.strftime("%Y-%m")
    if "base_signal_id" not in local.columns:
        local["base_signal_id"] = local["symbol"].astype(str) + "|" + local["signal_time"].astype(str)
    local["candidate_priority"] = np.select(
        [local["candidate"].astype(str).eq("CIC1_beta_extreme"), local["candidate"].astype(str).eq("CIC2_beta_broad")],
        [2, 1],
        default=0,
    )
    local = local.sort_values(["base_signal_id", "candidate_priority"], ascending=[True, False])
    local = local.drop_duplicates("base_signal_id", keep="first").copy()
    if "net_return" not in local.columns and "gross_return" in local.columns:
        local["net_return"] = _num(local, "gross_return") - 0.004
    return local


def _merge_trade_context(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or features.empty:
        return pd.DataFrame()
    feat_cols = [
        "symbol",
        "feature_time",
        "funding_bucket",
        "oi_bucket",
        "price_state",
        "volume_impulse_state",
        "funding_percentile",
        "oi_crowding_percentile",
        "ret_4h_percentile",
        "volume_z_1h",
        "btc_market_state",
    ]
    feat = features[[col for col in feat_cols if col in features.columns]].copy()
    rows = []
    for symbol, local_trades in trades.groupby("symbol", sort=False):
        local_feat = feat[feat["symbol"].astype(str).eq(str(symbol))].sort_values("feature_time")
        local_trades = local_trades.sort_values("entry_time").copy()
        if local_feat.empty:
            local_trades["feature_time"] = pd.NaT
            rows.append(local_trades)
            continue
        merged = pd.merge_asof(
            local_trades,
            local_feat,
            left_on="entry_time",
            right_on="feature_time",
            by="symbol",
            direction="backward",
            tolerance=pd.Timedelta(minutes=30),
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _crowding_vs_cic_interaction(features: pd.DataFrame, cfg: V50Config) -> pd.DataFrame:
    trades = _load_p2_trades(cfg.trade_cache_path)
    merged = _merge_trade_context(trades, features)
    if merged.empty:
        return pd.DataFrame(
            [
                {
                    "bucket": "no_p2_trade_cache",
                    "trades": 0,
                    "net20": np.nan,
                    "net30": np.nan,
                    "month_cap35_net20": np.nan,
                    "coverage": 0.0,
                }
            ]
        )
    merged["feature_covered"] = pd.to_datetime(merged.get("feature_time"), utc=True, errors="coerce").notna()
    merged["net20"] = _num(merged, "net_return")
    merged["net30"] = _num(merged, "gross_return", np.nan) - 0.006 if "gross_return" in merged.columns else _num(merged, "net20") - 0.002
    merged["crowding_state"] = np.select(
        [
            merged["funding_bucket"].astype(str).isin(["funding_high", "funding_extreme"]) & merged["oi_bucket"].astype(str).isin(["oi_high", "oi_extreme"]),
            merged["funding_bucket"].astype(str).isin(["funding_low", "funding_mid"]) & merged["price_state"].astype(str).eq("price_up"),
        ],
        ["high_funding_high_oi", "funding_not_hot_or_mid_price_up"],
        default="other",
    )
    rows = []
    for bucket_col in ["candidate", "crowding_state"]:
        for bucket, group in merged.groupby(bucket_col, sort=False, dropna=False):
            rows.append(
                {
                    "bucket_type": bucket_col,
                    "bucket": bucket,
                    "trades": int(len(group)),
                    "feature_covered_trades": int(group["feature_covered"].sum()),
                    "coverage": float(group["feature_covered"].mean()) if len(group) else np.nan,
                    "net20": float(_num(group, "net20").mean()) if len(group) else np.nan,
                    "net30": float(_num(group, "net30").mean()) if len(group) else np.nan,
                    "hit_rate": float(_num(group, "net20").gt(0).mean()) if len(group) else np.nan,
                    "month_cap35_net20": _month_cap35(group, "net20"),
                    "max_symbol_contribution": _max_contribution(group, "symbol", "net20"),
                }
            )
    for candidate in ["CIC1_beta_extreme", "CIC2_beta_broad"]:
        local = merged[merged["candidate"].astype(str).eq(candidate)].copy()
        for crowding, group in local.groupby("crowding_state", sort=False, dropna=False):
            rows.append(
                {
                    "bucket_type": "candidate_x_crowding_state",
                    "bucket": f"{candidate}|{crowding}",
                    "trades": int(len(group)),
                    "feature_covered_trades": int(group["feature_covered"].sum()),
                    "coverage": float(group["feature_covered"].mean()) if len(group) else np.nan,
                    "net20": float(_num(group, "net20").mean()) if len(group) else np.nan,
                    "net30": float(_num(group, "net30").mean()) if len(group) else np.nan,
                    "hit_rate": float(_num(group, "net20").gt(0).mean()) if len(group) else np.nan,
                    "month_cap35_net20": _month_cap35(group, "net20"),
                    "max_symbol_contribution": _max_contribution(group, "symbol", "net20"),
                }
            )
    return pd.DataFrame(rows)


def _crowding_action_atlas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = _state_masks(frame)
    actions = [
        ("no_long_if_high_funding_high_oi_stall", masks["high_funding_high_oi_price_stall"], "risk_overlay_candidate", "future long expectancy under crowded stall."),
        ("relative_value_short_crowded_long_btc", masks["high_funding_high_oi"], "rv_candidate", "short crowded symbol / long BTC proxy."),
        ("long_funding_safe_momentum", masks["funding_not_hot_oi_moderate_price_impulse"], "long_sibling_candidate", "price impulse with funding not hot and OI moderate."),
        ("no_overflow_if_high_crowding", masks["high_funding_high_oi"], "capacity_overlay_diagnostic", "possible no-overflow context; not tested as portfolio rule here."),
    ]
    for action, mask, action_type, note in actions:
        sample = frame[mask.fillna(False)].copy()
        rows.append(
            {
                "action_hint": action,
                "action_type": action_type,
                "events": int(len(sample)),
                "long_net20_12h": float(_num(sample, "long_net20_12h").mean()) if len(sample) else np.nan,
                "short_long_btc_net20_12h": float(_num(sample, "short_long_btc_net20_12h").mean()) if len(sample) else np.nan,
                "long_short_btc_net20_12h": float(_num(sample, "long_short_btc_net20_12h").mean()) if len(sample) else np.nan,
                "month_cap35_long_net20_12h": _month_cap35(sample, "long_net20_12h"),
                "month_cap35_short_long_btc_net20_12h": _month_cap35(sample, "short_long_btc_net20_12h"),
                "evidence_note": note,
                "promotion_status": "atlas_only",
            }
        )
    return pd.DataFrame(rows)


def _write_notes(
    path: Path,
    state: pd.DataFrame,
    rv: pd.DataFrame,
    cic: pd.DataFrame,
    action: pd.DataFrame,
    cfg: V50Config,
) -> None:
    high = state[state["state"].eq("high_funding_high_oi")]
    safe = state[state["state"].eq("funding_not_hot_oi_moderate_price_impulse")]
    rv_row = rv[rv["candidate"].eq("short_high_crowding_long_btc")]
    lines = [
        "# v5.0 Perp Crowding Atlas",
        "",
        "Status: atlas only. No live, shadow, selector, gate, or real-live permission is changed.",
        "",
        f"Universe: `{cfg.universe_col}`; cost convention: round-trip net subtracts `2 * {cfg.cost_bps}bps`.",
        "",
        "Key first-pass reads:",
    ]
    if not high.empty:
        row = high.iloc[0]
        lines.append(
            f"- High funding + high OI: events={int(row['events'])}, long12h={row['long_net20_12h']:.4%}, "
            f"short/BTC12h={row['short_long_btc_net20_12h']:.4%}, month_cap_long12h={row['month_cap35_long_net20_12h']:.4%}."
        )
    if not safe.empty:
        row = safe.iloc[0]
        lines.append(
            f"- Funding-not-hot + moderate OI + price impulse: events={int(row['events'])}, "
            f"long12h={row['long_net20_12h']:.4%}, long/BTC12h={row['long_short_btc_net20_12h']:.4%}."
        )
    if not rv_row.empty:
        row = rv_row.iloc[0]
        lines.append(
            f"- Relative value short crowded / long BTC: events={int(row['events'])}, "
            f"pair12h={row['pair_net20_12h']:.4%}, month_cap={row['month_cap35_pair_net20_12h']:.4%}."
        )
    if not cic.empty:
        best = cic.sort_values("net20", ascending=False).head(3)
        lines.append("")
        lines.append("CIC interaction top rows:")
        for row in best.itertuples(index=False):
            lines.append(f"- {getattr(row, 'bucket_type')}: {getattr(row, 'bucket')} trades={getattr(row, 'trades')} net20={getattr(row, 'net20'):.4%}")
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "- High funding/OI is not assumed to mean short. The atlas reports long, naked short, and hedged pair views side by side.",
            "- BTC/market hedge rows are relative-value proxies, not a portfolio engine.",
            "- Any promising state must still pass controls, month caps, execution realism, and forward logging before becoming a rule.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v50_perp_crowding_atlas(cfg: V50Config | None = None) -> dict[str, Path]:
    cfg = cfg or V50Config()
    report_root = ensure_dir(cfg.report_root)
    features = _add_context_columns(_read_feature_subset(cfg), cfg)

    state_summary = _crowding_state_summary(features)
    bucket_summary = _funding_oi_bucket_summary(features)
    rv_summary = _relative_value_pair_summary(features)
    cic_interaction = _crowding_vs_cic_interaction(features, cfg)
    action_atlas = _crowding_action_atlas(features)

    outputs = {
        "crowding_state_summary": report_root / "crowding_state_summary.csv",
        "funding_oi_bucket_summary": report_root / "funding_oi_bucket_summary.csv",
        "relative_value_pair_summary": report_root / "relative_value_pair_summary.csv",
        "crowding_vs_cic_interaction": report_root / "crowding_vs_cic_interaction.csv",
        "crowding_action_atlas": report_root / "crowding_action_atlas.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    state_summary.to_csv(outputs["crowding_state_summary"], index=False)
    bucket_summary.to_csv(outputs["funding_oi_bucket_summary"], index=False)
    rv_summary.to_csv(outputs["relative_value_pair_summary"], index=False)
    cic_interaction.to_csv(outputs["crowding_vs_cic_interaction"], index=False)
    action_atlas.to_csv(outputs["crowding_action_atlas"], index=False)
    _write_notes(outputs["candidate_notes"], state_summary, rv_summary, cic_interaction, action_atlas, cfg)
    return outputs
