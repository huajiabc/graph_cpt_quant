"""v4.0 Cross-Exchange Lead-Lag / Propagation Graph.

The first v4.0 pass is intentionally attribution-first. It builds a
Binance -> Bybit kline lead-lag atlas, checks data coverage, and evaluates
small frozen motifs with random/shuffled controls. It is not a live selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v4_0_cross_exchange_lead_lag")
DATA_ROOT = Path("data")
SOURCE_EXCHANGE = "binance"
TARGET_EXCHANGE = "bybit"
KLINE_DATASET = "klines"


@dataclass(frozen=True)
class V40Config:
    report_root: Path = REPORT_ROOT
    data_root: Path = DATA_ROOT
    source_exchange: str = SOURCE_EXCHANGE
    target_exchange: str = TARGET_EXCHANGE
    dataset: str = KLINE_DATASET
    top_n: int = 50
    max_pair_symbols: int = 12
    impulse_volume_z: float = 1.5
    impulse_ret_threshold: float = 0.001
    lag_ret_threshold: float = 0.001
    not_extended_ret_1h: float = 0.01
    market_density_threshold: float = 0.20
    cost_bps: float = 20.0
    random_permutations: int = 25


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _bool(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[col]
    if values.dtype == object:
        return values.astype(str).str.lower().isin(["true", "1", "yes"])
    return values.fillna(False).astype(bool)


def _symbol_files(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {path.stem.upper(): path for path in sorted(root.glob("*.parquet"))}


def _read_symbol_kline(path: Path, exchange: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    required = ["bar_open_time", "bar_close_time", "open", "high", "low", "close", "volume", "turnover"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        return pd.DataFrame()
    optional = [col for col in ["trades", "taker_buy_base", "taker_buy_quote"] if col in frame.columns]
    out = frame[required + optional].copy()
    out["exchange"] = str(frame["exchange"].iloc[0]) if "exchange" in frame.columns and len(frame) else exchange
    out["symbol"] = str(frame["symbol"].iloc[0]).upper() if "symbol" in frame.columns and len(frame) else path.stem.upper()
    out["bar_open_time"] = pd.to_datetime(out["bar_open_time"], utc=True, errors="coerce")
    out["bar_close_time"] = pd.to_datetime(out["bar_close_time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "turnover", *optional]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["bar_close_time", "close"]).sort_values(["symbol", "bar_close_time"])


def _select_common_symbols(source_dir: Path, target_dir: Path, top_n: int) -> tuple[list[str], pd.DataFrame]:
    source_files = _symbol_files(source_dir)
    target_files = _symbol_files(target_dir)
    common = sorted(set(source_files).intersection(target_files))
    rows = []
    for symbol in common:
        target = _read_symbol_kline(target_files[symbol], "target")
        turnover = float(_num(target, "turnover").sum()) if len(target) else 0.0
        rows.append({"symbol": symbol, "target_turnover_sum": turnover})
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return [], ranking
    ranking = ranking.sort_values(["target_turnover_sum", "symbol"], ascending=[False, True])
    selected = ranking["symbol"].head(int(top_n)).astype(str).tolist()
    ranking["selected"] = ranking["symbol"].isin(selected)
    return selected, ranking


def _load_klines(directory: Path, exchange: str, symbols: list[str]) -> pd.DataFrame:
    files = _symbol_files(directory)
    frames = [_read_symbol_kline(files[symbol], exchange) for symbol in symbols if symbol in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "bar_close_time"])


def _future_window_max(values: pd.Series, bars: int) -> pd.Series:
    shifted = [values.shift(-idx) for idx in range(1, bars + 1)]
    return pd.concat(shifted, axis=1).max(axis=1)


def _future_window_min(values: pd.Series, bars: int) -> pd.Series:
    shifted = [values.shift(-idx) for idx in range(1, bars + 1)]
    return pd.concat(shifted, axis=1).min(axis=1)


def _add_bar_features(frame: pd.DataFrame, cfg: V40Config, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    def _feature_group(group: pd.DataFrame) -> pd.DataFrame:
        out = group.sort_values("bar_close_time").copy()
        close = _num(out, "close")
        high = _num(out, "high")
        low = _num(out, "low")
        volume = _num(out, "volume")
        prev_close = close.shift(1)
        out[f"{prefix}_close"] = close
        out[f"{prefix}_ret_15m"] = close / prev_close - 1.0
        out[f"{prefix}_ret_1h"] = close / close.shift(4) - 1.0
        out[f"{prefix}_future_ret_15m"] = close.shift(-1) / close - 1.0
        out[f"{prefix}_future_ret_30m"] = close.shift(-2) / close - 1.0
        out[f"{prefix}_future_ret_1h"] = close.shift(-4) / close - 1.0
        out[f"{prefix}_future_ret_4h"] = close.shift(-16) / close - 1.0
        out[f"{prefix}_future_mfe_30m"] = _future_window_max(high, 2) / close - 1.0
        out[f"{prefix}_future_mae_30m"] = _future_window_min(low, 2) / close - 1.0
        out[f"{prefix}_future_mfe_1h"] = _future_window_max(high, 4) / close - 1.0
        out[f"{prefix}_future_mae_1h"] = _future_window_min(low, 4) / close - 1.0
        vol_mean = volume.shift(1).rolling(96, min_periods=3).mean()
        vol_std = volume.shift(1).rolling(96, min_periods=3).std(ddof=0).replace(0, np.nan)
        out[f"{prefix}_volume_z"] = (volume - vol_mean) / vol_std
        if "taker_buy_base" in out.columns:
            taker_buy = _num(out, "taker_buy_base")
            out[f"{prefix}_taker_buy_ratio"] = (taker_buy / volume.replace(0, np.nan)).clip(0.0, 1.0)
            buy_mean = taker_buy.shift(1).rolling(96, min_periods=3).mean()
            buy_std = taker_buy.shift(1).rolling(96, min_periods=3).std(ddof=0).replace(0, np.nan)
            out[f"{prefix}_taker_buy_z"] = (taker_buy - buy_mean) / buy_std
        else:
            out[f"{prefix}_taker_buy_ratio"] = np.nan
            out[f"{prefix}_taker_buy_z"] = np.nan
        out[f"{prefix}_bullish_volume_impulse"] = (
            (_num(out, f"{prefix}_ret_15m") >= cfg.impulse_ret_threshold)
            & (_num(out, f"{prefix}_volume_z") >= cfg.impulse_volume_z)
            & (_num(out, "close") >= _num(out, "open"))
        )
        out[f"{prefix}_taker_buy_impulse"] = (
            (_num(out, f"{prefix}_ret_15m") >= cfg.impulse_ret_threshold)
            & (
                (_num(out, f"{prefix}_taker_buy_ratio") >= 0.55)
                | (_num(out, f"{prefix}_taker_buy_z") >= cfg.impulse_volume_z)
            )
            & (_num(out, "close") >= _num(out, "open"))
        )
        prior_ret = _num(out, f"{prefix}_ret_15m").shift(1)
        rolling_high = high.shift(1).rolling(8, min_periods=3).max()
        pullback_then_reclaim = (
            (low <= prev_close * 0.995)
            & (close >= prev_close * 0.999)
            & (_num(out, f"{prefix}_ret_15m") > 0.0)
        )
        red_to_green = (prior_ret < 0.0) & (_num(out, f"{prefix}_ret_15m") > 0.0) & (close >= _num(out, "open"))
        out[f"{prefix}_reclaim_proxy"] = pullback_then_reclaim | red_to_green
        out[f"{prefix}_breakout_proxy"] = (close > rolling_high) & (_num(out, f"{prefix}_ret_15m") > 0.0)
        out[f"{prefix}_impulse_next_1h"] = pd.concat(
            [out[f"{prefix}_bullish_volume_impulse"].astype(float).shift(-idx) for idx in range(1, 5)],
            axis=1,
        ).max(axis=1).fillna(0.0).gt(0.0)
        return out

    frames = [_feature_group(group) for _, group in frame.groupby("symbol", sort=False)]
    return pd.concat(frames, ignore_index=True) if frames else frame.iloc[0:0].copy()


def _prefix_columns(frame: pd.DataFrame, prefix: str, symbol_col: str) -> pd.DataFrame:
    keep = ["symbol", "bar_close_time"]
    rename = {"symbol": symbol_col}
    for col in frame.columns:
        if col in {"symbol", "bar_close_time", "bar_open_time", "exchange"}:
            continue
        if col.startswith(prefix):
            keep.append(col)
    out = frame[keep].copy()
    out = out.rename(columns=rename)
    return out


def _forward_bool(series: pd.Series, bars: int) -> pd.Series:
    shifted = series.astype(float).shift(1)
    return shifted.rolling(bars, min_periods=1).max().fillna(0.0).gt(0.0)


def _build_same_symbol_frame(source: pd.DataFrame, target: pd.DataFrame, cfg: V40Config) -> pd.DataFrame:
    src = _prefix_columns(source, "source", "symbol")
    tgt = _prefix_columns(target, "target", "symbol")
    merged = tgt.merge(src, on=["symbol", "bar_close_time"], how="inner")
    if merged.empty:
        return merged
    merged = merged.sort_values(["symbol", "bar_close_time"]).reset_index(drop=True)
    merged["source_impulse_same_bar"] = _bool(merged, "source_bullish_volume_impulse")
    merged["source_taker_buy_impulse_same_bar"] = _bool(merged, "source_taker_buy_impulse")
    merged["target_impulse_same_bar"] = _bool(merged, "target_bullish_volume_impulse")
    for bars, name in [(1, "prior_15m"), (4, "prior_1h")]:
        merged[f"source_impulse_{name}"] = (
            merged.groupby("symbol", group_keys=False)["source_impulse_same_bar"].apply(lambda s: _forward_bool(s, bars))
        )
        merged[f"source_taker_buy_impulse_{name}"] = (
            merged.groupby("symbol", group_keys=False)["source_taker_buy_impulse_same_bar"].apply(
                lambda s: _forward_bool(s, bars)
            )
        )
        merged[f"target_impulse_{name}"] = (
            merged.groupby("symbol", group_keys=False)["target_impulse_same_bar"].apply(lambda s: _forward_bool(s, bars))
        )
    density = (
        merged.groupby("bar_close_time")["source_impulse_same_bar"]
        .mean()
        .rename("source_market_impulse_density")
        .reset_index()
    )
    merged = merged.merge(density, on="bar_close_time", how="left")
    merged["source_market_impulse_density_high"] = (
        _num(merged, "source_market_impulse_density").fillna(0.0) >= cfg.market_density_threshold
    )
    merged["target_lagging_same_symbol"] = (
        _num(merged, "target_ret_1h").fillna(0.0)
        < _num(merged, "source_ret_1h").fillna(0.0) - cfg.lag_ret_threshold
    )
    merged["target_not_extended"] = (
        (_num(merged, "target_ret_1h").fillna(0.0) < cfg.not_extended_ret_1h)
        & (_num(merged, "target_ret_1h").fillna(0.0) <= _num(merged, "source_ret_1h").fillna(0.0))
    )
    merged["target_already_extended"] = (
        (_num(merged, "target_ret_1h").fillna(0.0) >= cfg.not_extended_ret_1h)
        | (_num(merged, "target_ret_1h").fillna(0.0) > _num(merged, "source_ret_1h").fillna(0.0))
    )
    return merged


def _build_pair_frame(source: pd.DataFrame, target: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    pair_symbols = set(symbols)
    src = _prefix_columns(source[source["symbol"].isin(pair_symbols)], "source", "source_symbol")
    tgt = _prefix_columns(target[target["symbol"].isin(pair_symbols)], "target", "target_symbol")
    if src.empty or tgt.empty:
        return pd.DataFrame()
    pairs = tgt.merge(src, on="bar_close_time", how="inner")
    pairs = pairs.sort_values(["source_symbol", "target_symbol", "bar_close_time"]).reset_index(drop=True)
    pairs["edge_type"] = np.where(pairs["source_symbol"].eq(pairs["target_symbol"]), "same_symbol", "cross_symbol")
    pairs["source_impulse_same_bar"] = _bool(pairs, "source_bullish_volume_impulse")
    pairs["source_taker_buy_impulse_same_bar"] = _bool(pairs, "source_taker_buy_impulse")
    pairs["source_impulse_prior_1h"] = (
        pairs.groupby(["source_symbol", "target_symbol"], group_keys=False)["source_impulse_same_bar"]
        .apply(lambda s: _forward_bool(s, 4))
    )
    pairs["source_taker_buy_impulse_prior_1h"] = (
        pairs.groupby(["source_symbol", "target_symbol"], group_keys=False)["source_taker_buy_impulse_same_bar"]
        .apply(lambda s: _forward_bool(s, 4))
    )
    return pairs


def _net_at_cost(frame: pd.DataFrame, horizon_col: str, cost_bps: float) -> pd.Series:
    return _num(frame, horizon_col) - 2.0 * float(cost_bps) / 10_000.0


def _month_cap35(frame: pd.DataFrame, value_col: str) -> float:
    if frame.empty or "bar_close_time" not in frame.columns:
        return np.nan
    local = frame.copy()
    local["month"] = (
        pd.to_datetime(local["bar_close_time"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
    )
    vals = _num(local, value_col).dropna()
    if vals.empty:
        return np.nan
    total = float(vals.sum())
    if total <= 0:
        return float(vals.mean())
    month_sum = local.groupby("month")[value_col].sum(min_count=1)
    capped = month_sum.clip(upper=0.35 * total)
    return float(capped.sum() / len(vals))


def _summary_row(frame: pd.DataFrame, mask: pd.Series, label: str, cfg: V40Config) -> dict:
    subset = frame[mask.fillna(False)].copy()
    if not subset.empty:
        subset["net10"] = _net_at_cost(subset, "target_future_ret_1h", 10.0)
        subset["net20"] = _net_at_cost(subset, "target_future_ret_1h", 20.0)
        subset["net30"] = _net_at_cost(subset, "target_future_ret_1h", 30.0)
    return {
        "label": label,
        "events": int(len(subset)),
        "symbols": int(subset["symbol"].nunique()) if "symbol" in subset.columns and len(subset) else 0,
        "gross_ret_15m": float(_num(subset, "target_future_ret_15m").mean()) if len(subset) else np.nan,
        "gross_ret_1h": float(_num(subset, "target_future_ret_1h").mean()) if len(subset) else np.nan,
        "gross_ret_4h": float(_num(subset, "target_future_ret_4h").mean()) if len(subset) else np.nan,
        "net10": float(_num(subset, "net10").mean()) if len(subset) else np.nan,
        "net20": float(_num(subset, "net20").mean()) if len(subset) else np.nan,
        "net30": float(_num(subset, "net30").mean()) if len(subset) else np.nan,
        "hit_rate_1h_positive": float(_num(subset, "target_future_ret_1h").gt(0).mean()) if len(subset) else np.nan,
        "hit_mfe_1h_1pct": float(_num(subset, "target_future_mfe_1h").gt(0.01).mean()) if len(subset) else np.nan,
        "month_cap35_net20": _month_cap35(subset, "net20") if len(subset) else np.nan,
        "cost_single_side_bps": float(cfg.cost_bps),
    }


def _edge_atlas(pair_frame: pd.DataFrame, cfg: V40Config) -> pd.DataFrame:
    rows: list[dict] = []
    if pair_frame.empty:
        return pd.DataFrame(rows)
    for (source_symbol, target_symbol), group in pair_frame.groupby(["source_symbol", "target_symbol"], sort=False):
        impulse = group[_bool(group, "source_impulse_same_bar")]
        prior = group[_bool(group, "source_impulse_prior_1h")]
        n = int(len(prior))
        mean_net20 = float((_net_at_cost(prior, "target_future_ret_1h", cfg.cost_bps)).mean()) if n else np.nan
        hit_rate = float(_num(prior, "target_future_ret_1h").gt(0).mean()) if n else np.nan
        response_rate = float(_bool(impulse, "target_impulse_next_1h").mean()) if len(impulse) else np.nan
        shrinkage = n / (n + 30.0) if n else 0.0
        rows.append(
            {
                "source_exchange": cfg.source_exchange,
                "target_exchange": cfg.target_exchange,
                "source_symbol": source_symbol,
                "target_symbol": target_symbol,
                "edge_type": "same_symbol" if source_symbol == target_symbol else "cross_symbol",
                "overlapping_bars": int(len(group)),
                "source_impulses": int(_bool(group, "source_impulse_same_bar").sum()),
                "source_prior_1h_events": n,
                "target_response_impulse_1h_rate": response_rate,
                "target_future_ret_1h_after_source_prior": float(_num(prior, "target_future_ret_1h").mean()) if n else np.nan,
                "target_future_mfe_1h_after_source_prior": float(_num(prior, "target_future_mfe_1h").mean()) if n else np.nan,
                "net20_after_source_prior": mean_net20,
                "hit_rate_1h_positive": hit_rate,
                "edge_weight": float((0.0 if pd.isna(mean_net20) else mean_net20) * (0.0 if pd.isna(hit_rate) else hit_rate) * shrinkage),
            }
        )
    return pd.DataFrame(rows).sort_values(["edge_weight", "source_prior_1h_events"], ascending=[False, False])


def _source_response(same: pd.DataFrame, cfg: V40Config) -> pd.DataFrame:
    rows: list[dict] = []
    if same.empty:
        return pd.DataFrame(rows)
    windows = {
        "same_bar": _bool(same, "source_impulse_same_bar"),
        "prior_15m": _bool(same, "source_impulse_prior_15m"),
        "prior_1h": _bool(same, "source_impulse_prior_1h"),
        "taker_buy_prior_1h": _bool(same, "source_taker_buy_impulse_prior_1h"),
    }
    for symbol, group in same.groupby("symbol", sort=False):
        for window, mask in windows.items():
            row = _summary_row(group, mask.loc[group.index], f"{symbol}_{window}", cfg)
            row["symbol"] = symbol
            row["lead_window"] = window
            rows.append(row)
    total = []
    for window, mask in windows.items():
        row = _summary_row(same, mask, f"ALL_{window}", cfg)
        row["symbol"] = "ALL"
        row["lead_window"] = window
        total.append(row)
    return pd.DataFrame(total + rows)


def _lead_lag_motifs(same: pd.DataFrame, pairs: pd.DataFrame, cfg: V40Config) -> pd.DataFrame:
    rows: list[dict] = []
    if same.empty:
        return pd.DataFrame(rows)
    motifs = {
        "LX1_source_impulse_target_lag_reclaim": (
            _bool(same, "source_impulse_prior_1h")
            & _bool(same, "target_not_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
        "LX4_market_density_target_lag_reclaim": (
            _bool(same, "source_market_impulse_density_high")
            & _bool(same, "target_lagging_same_symbol")
            & _bool(same, "target_reclaim_proxy")
        ),
        "target_only_reclaim": _bool(same, "target_reclaim_proxy"),
        "target_reclaim_without_source_prior": _bool(same, "target_reclaim_proxy") & ~_bool(same, "source_impulse_prior_1h"),
        "same_exchange_target_impulse_reclaim": (
            _bool(same, "target_impulse_prior_1h")
            & _bool(same, "target_not_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
        "source_impulse_target_already_extended": (
            _bool(same, "source_impulse_prior_1h")
            & _bool(same, "target_already_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
        "LX2_binance_taker_buy_impulse_bybit_reclaim": (
            _bool(same, "source_taker_buy_impulse_prior_1h")
            & _bool(same, "target_not_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
    }
    for label, mask in motifs.items():
        row = _summary_row(same, mask, label, cfg)
        row["candidate"] = label.split("_", 1)[0]
        row["status"] = "evaluated"
        rows.append(row)
    if not pairs.empty:
        cross = pairs[~pairs["source_symbol"].eq(pairs["target_symbol"])].copy()
        if not cross.empty:
            cross = cross.rename(columns={"target_symbol": "symbol"})
            mask = (
                _bool(cross, "source_impulse_prior_1h")
                & (_num(cross, "target_volume_z") >= cfg.impulse_volume_z)
                & _bool(cross, "target_reclaim_proxy")
            )
            row = _summary_row(cross, mask, "LX3_cross_symbol_leader_impulse_beta_reclaim", cfg)
            row["candidate"] = "LX3"
            row["status"] = "evaluated_pair_limited"
            rows.append(row)
    if "source_taker_buy_ratio" not in same.columns or _num(same, "source_taker_buy_ratio").notna().sum() == 0:
        rows.append(
            {
                "label": "LX2_binance_taker_buy_impulse_bybit_reclaim",
                "candidate": "LX2",
                "events": 0,
                "status": "not_evaluated_no_taker_buy_coverage",
                "cost_single_side_bps": float(cfg.cost_bps),
            }
        )
    return pd.DataFrame(rows)


def _execution_comparison(same: pd.DataFrame, cfg: V40Config) -> pd.DataFrame:
    rows: list[dict] = []
    if same.empty:
        return pd.DataFrame(rows)
    conditions = {
        "target_reclaim_proxy": _bool(same, "target_reclaim_proxy"),
        "target_breakout_proxy": _bool(same, "target_breakout_proxy"),
        "source_prior_1h_plus_reclaim": _bool(same, "source_impulse_prior_1h") & _bool(same, "target_reclaim_proxy"),
        "source_prior_1h_plus_breakout": _bool(same, "source_impulse_prior_1h") & _bool(same, "target_breakout_proxy"),
        "source_taker_buy_prior_1h_plus_reclaim": (
            _bool(same, "source_taker_buy_impulse_prior_1h") & _bool(same, "target_reclaim_proxy")
        ),
    }
    horizons = {
        "15m": "target_future_ret_15m",
        "1h": "target_future_ret_1h",
        "4h": "target_future_ret_4h",
    }
    for execution, mask in conditions.items():
        subset = same[mask.fillna(False)].copy()
        for horizon, col in horizons.items():
            net20 = _net_at_cost(subset, col, cfg.cost_bps)
            rows.append(
                {
                    "execution": execution,
                    "hold_horizon": horizon,
                    "events": int(len(subset)),
                    "gross_return": float(_num(subset, col).mean()) if len(subset) else np.nan,
                    "net20_proxy": float(net20.mean()) if len(subset) else np.nan,
                    "hit_rate_positive": float(_num(subset, col).gt(0).mean()) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _mapped_source_mask(pair_frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    if pair_frame.empty:
        return pd.Series(False, index=pair_frame.index)
    target = pair_frame["target_symbol"].astype(str)
    mapped = target.map(mapping)
    return pair_frame["source_symbol"].astype(str).eq(mapped)


def _controls(same: pd.DataFrame, pairs: pd.DataFrame, symbols: list[str], cfg: V40Config) -> pd.DataFrame:
    rows: list[dict] = []
    if same.empty:
        return pd.DataFrame(rows)
    base_masks = {
        "real_same_symbol_source_prior": (
            _bool(same, "source_impulse_prior_1h")
            & _bool(same, "target_not_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
        "target_only_impulse_reclaim": (
            _bool(same, "target_impulse_prior_1h")
            & _bool(same, "target_not_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
        "target_reclaim_without_source_prior": (
            _bool(same, "target_reclaim_proxy") & ~_bool(same, "source_impulse_prior_1h")
        ),
        "source_prior_target_already_extended": (
            _bool(same, "source_impulse_prior_1h")
            & _bool(same, "target_already_extended")
            & _bool(same, "target_reclaim_proxy")
        ),
        "source_taker_buy_prior_target_reclaim": (
            _bool(same, "source_taker_buy_impulse_prior_1h") & _bool(same, "target_reclaim_proxy")
        ),
    }
    for label, mask in base_masks.items():
        row = _summary_row(same, mask, label, cfg)
        row["control_type"] = label
        rows.append(row)

    if not pairs.empty and len(symbols) > 1:
        shifted = {symbol: symbols[(idx + 1) % len(symbols)] for idx, symbol in enumerate(symbols)}
        reversed_map = {symbol: symbols[-idx - 1] for idx, symbol in enumerate(symbols)}
        for label, mapping in [
            ("random_cyclic_source_symbol", shifted),
            ("shuffled_reversed_source_target_mapping", reversed_map),
        ]:
            local = pairs[_mapped_source_mask(pairs, mapping)].copy()
            local = local.rename(columns={"target_symbol": "symbol"})
            mask = (
                _bool(local, "source_impulse_prior_1h")
                & (_num(local, "target_ret_1h").fillna(0.0) < cfg.not_extended_ret_1h)
                & _bool(local, "target_reclaim_proxy")
            )
            row = _summary_row(local, mask, label, cfg)
            row["control_type"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def _coverage_summary(
    cfg: V40Config,
    selected: list[str],
    ranking: pd.DataFrame,
    source: pd.DataFrame,
    target: pd.DataFrame,
    same: pd.DataFrame,
) -> pd.DataFrame:
    source_dir = cfg.data_root / "raw" / cfg.source_exchange / cfg.dataset
    target_dir = cfg.data_root / "raw" / cfg.target_exchange / cfg.dataset
    source_files = _symbol_files(source_dir)
    target_files = _symbol_files(target_dir)
    common = set(source_files).intersection(target_files)
    overlap_start = pd.to_datetime(same["bar_close_time"], utc=True, errors="coerce").min() if len(same) else pd.NaT
    overlap_end = pd.to_datetime(same["bar_close_time"], utc=True, errors="coerce").max() if len(same) else pd.NaT
    status = "ok"
    if len(selected) < 10:
        status = "insufficient_common_universe"
    if same.empty:
        status = "no_overlap"
    return pd.DataFrame(
        [
            {
                "source_exchange": cfg.source_exchange,
                "target_exchange": cfg.target_exchange,
                "source_symbols_available": len(source_files),
                "target_symbols_available": len(target_files),
                "common_symbols_available": len(common),
                "selected_symbols": len(selected),
                "selected_symbol_list": ",".join(selected),
                "source_rows": int(len(source)),
                "target_rows": int(len(target)),
                "same_symbol_overlap_rows": int(len(same)),
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "coverage_status": status,
                "top_n_requested": int(cfg.top_n),
                "ranking_rows": int(len(ranking)),
            }
        ]
    )


def _write_notes(
    root: Path,
    coverage: pd.DataFrame,
    motifs: pd.DataFrame,
    controls: pd.DataFrame,
    edge_atlas: pd.DataFrame,
    cfg: V40Config,
) -> None:
    cov = coverage.iloc[0].to_dict() if len(coverage) else {}
    lines = [
        "# v4.0 Cross-Exchange Lead-Lag / Propagation Graph",
        "",
        "## Scope",
        f"- Source: `{cfg.source_exchange}` raw `{cfg.dataset}`.",
        f"- Target: `{cfg.target_exchange}` raw `{cfg.dataset}`.",
        f"- Selected common symbols: {cov.get('selected_symbols', 0)}.",
        f"- Coverage status: `{cov.get('coverage_status', 'unknown')}`.",
        "",
        "## Interpretation",
    ]
    if cov.get("coverage_status") != "ok":
        lines.append(
            "- This run is an atlas / plumbing pass only. The local source universe is too small for alpha promotion."
        )
    else:
        lines.append("- Coverage is broad enough for first-pass motif attribution, but still not a live selector.")

    if len(motifs):
        focus = motifs[motifs.get("status", "evaluated").astype(str).eq("evaluated")].copy()
        if len(focus):
            best = focus.sort_values("net20", ascending=False).head(5)
            lines.append("")
            lines.append("## Best evaluated motif rows")
            for row in best.itertuples(index=False):
                lines.append(
                    f"- {row.label}: events={row.events}, net20={getattr(row, 'net20', np.nan):.4%}, "
                    f"hit={getattr(row, 'hit_rate_1h_positive', np.nan):.2%}."
                )
    if len(edge_atlas):
        best_edge = edge_atlas.head(1).iloc[0]
        lines.extend(
            [
                "",
                "## Strongest observed edge",
                (
                    f"- {best_edge.source_symbol}->{best_edge.target_symbol} "
                    f"({best_edge.edge_type}), events={int(best_edge.source_prior_1h_events)}, "
                    f"net20={best_edge.net20_after_source_prior:.4%}."
                ),
            ]
        )
    if len(controls):
        real = controls[controls["control_type"].eq("real_same_symbol_source_prior")]
        randomish = controls[controls["control_type"].str.contains("random|shuffled", case=False, na=False)]
        if len(real) and len(randomish):
            lines.extend(
                [
                    "",
                    "## Control warning",
                    (
                        f"- Real same-symbol net20={float(real['net20'].iloc[0]):.4%}; "
                        f"random/shuffled best={float(randomish['net20'].max()):.4%}. "
                        "Treat edge attribution as unproven unless real beats these controls on a broad universe."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Candidate status",
            "- LX1/LX4: evaluated as 15m kline lead-lag proxies.",
            "- LX2: evaluated when Binance kline taker-buy fields are present; otherwise kept as coverage-pending.",
            "- LX3: evaluated only on the limited cross-symbol pair universe available locally.",
            "- No v4.0 candidate is promoted to paper-live or real-live by this report.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v40_cross_exchange_lead_lag(cfg: V40Config = V40Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    source_dir = cfg.data_root / "raw" / cfg.source_exchange / cfg.dataset
    target_dir = cfg.data_root / "raw" / cfg.target_exchange / cfg.dataset
    selected, ranking = _select_common_symbols(source_dir, target_dir, cfg.top_n)
    source = _add_bar_features(_load_klines(source_dir, cfg.source_exchange, selected), cfg, "source")
    target = _add_bar_features(_load_klines(target_dir, cfg.target_exchange, selected), cfg, "target")
    same = _build_same_symbol_frame(source, target, cfg)
    pair_symbols = selected[: max(0, int(cfg.max_pair_symbols))]
    pairs = _build_pair_frame(source, target, pair_symbols)

    coverage = _coverage_summary(cfg, selected, ranking, source, target, same)
    edge_atlas = _edge_atlas(pairs, cfg)
    response = _source_response(same, cfg)
    motifs = _lead_lag_motifs(same, pairs, cfg)
    execution = _execution_comparison(same, cfg)
    controls = _controls(same, pairs, pair_symbols, cfg)

    outputs = {
        "coverage_summary": root / "coverage_summary.csv",
        "symbol_universe": root / "symbol_universe.csv",
        "cross_exchange_edge_atlas": root / "cross_exchange_edge_atlas.csv",
        "source_impulse_target_response": root / "source_impulse_target_response.csv",
        "lead_lag_motif_summary": root / "lead_lag_motif_summary.csv",
        "target_execution_comparison": root / "target_execution_comparison.csv",
        "random_shuffled_exchange_control": root / "random_shuffled_exchange_control.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["coverage_summary"], index=False)
    ranking.to_csv(outputs["symbol_universe"], index=False)
    edge_atlas.to_csv(outputs["cross_exchange_edge_atlas"], index=False)
    response.to_csv(outputs["source_impulse_target_response"], index=False)
    motifs.to_csv(outputs["lead_lag_motif_summary"], index=False)
    execution.to_csv(outputs["target_execution_comparison"], index=False)
    controls.to_csv(outputs["random_shuffled_exchange_control"], index=False)
    _write_notes(root, coverage, motifs, controls, edge_atlas, cfg)
    return outputs


__all__ = ["V40Config", "write_v40_cross_exchange_lead_lag"]
