"""v4.2 Source Attribution & Target Fusion.

This stage keeps cross-exchange research attribution-first.  v4.2A checks
whether 1m Binance source conditions add independent information over Bybit
target-only conditions.  v4.2B treats Binance source context as a diagnostic
feature for the existing CIC/P2 long stack; it does not create a live selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _pool_trades
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v40_cross_exchange_lead_lag import (
    DATA_ROOT,
    SOURCE_EXCHANGE,
    TARGET_EXCHANGE,
    _bool,
    _month_cap35,
    _net_at_cost,
    _num,
    _select_common_symbols,
)
from pressure_graph.reports.v41_cross_exchange_lead_lag_diagnostics import (
    _add_1m_source_features,
    _future_max,
    _future_min,
    _read_1m_symbol,
)


REPORT_ROOT = Path("reports/v4_2_source_attribution_target_fusion")
DEFAULT_TRADE_CACHE = Path("reports/v1_3a_checkpoint_robustness/_v09b_trades_tmp.csv")
SOURCE_1M_DATASET = "klines_1m_v4"
TARGET_1M_DATASET = "klines_1m_v4"
P2_POOL = "P2_CIC1_CIC2_COMBINED"


@dataclass(frozen=True)
class V42Config:
    report_root: Path = REPORT_ROOT
    data_root: Path = DATA_ROOT
    trade_cache_path: Path = DEFAULT_TRADE_CACHE
    source_exchange: str = SOURCE_EXCHANGE
    target_exchange: str = TARGET_EXCHANGE
    source_dataset_1m: str = SOURCE_1M_DATASET
    target_dataset_1m: str = TARGET_1M_DATASET
    top_n: int = 50
    horizon_minutes: int = 60
    cost_bps: float = 20.0
    random_seed: int = 42


def _source_dir(cfg: V42Config) -> Path:
    return cfg.data_root / "raw" / cfg.source_exchange / cfg.source_dataset_1m


def _target_dir(cfg: V42Config) -> Path:
    return cfg.data_root / "raw" / cfg.target_exchange / cfg.target_dataset_1m


def _add_1m_target_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.sort_values("bar_close_time").copy()
    close = _num(out, "close")
    high = _num(out, "high")
    low = _num(out, "low")
    volume = _num(out, "volume")
    out["target_ret_1m"] = close / close.shift(1) - 1.0
    out["target_ret_5m"] = close / close.shift(5) - 1.0
    vol_mean = volume.shift(1).rolling(240, min_periods=30).mean()
    vol_std = volume.shift(1).rolling(240, min_periods=30).std(ddof=0).replace(0, np.nan)
    out["target_volume_z"] = (volume - vol_mean) / vol_std
    prev_close = close.shift(1)
    prior_ret = _num(out, "target_ret_1m").shift(1)
    pullback_reclaim = (low <= prev_close * 0.997) & (close >= prev_close * 0.999) & _num(out, "target_ret_1m").gt(0)
    red_to_green = prior_ret.lt(0.0) & _num(out, "target_ret_1m").gt(0.0) & close.ge(_num(out, "open"))
    out["target_reclaim_proxy"] = pullback_reclaim | red_to_green
    out["target_bullish_volume_impulse"] = (
        _num(out, "target_ret_1m").ge(0.001)
        & _num(out, "target_volume_z").ge(2.0)
        & close.ge(_num(out, "open"))
    )
    out[f"target_future_ret_{horizon}m"] = close.shift(-horizon) / close - 1.0
    out[f"target_future_mfe_{horizon}m"] = _future_max(high, horizon) / close - 1.0
    out[f"target_future_mae_{horizon}m"] = _future_min(low, horizon) / close - 1.0
    keep = [
        "bar_close_time",
        "target_ret_1m",
        "target_ret_5m",
        "target_volume_z",
        "target_reclaim_proxy",
        "target_bullish_volume_impulse",
        f"target_future_ret_{horizon}m",
        f"target_future_mfe_{horizon}m",
        f"target_future_mae_{horizon}m",
    ]
    return out[keep]


def _decorate_source(frame: pd.DataFrame) -> pd.DataFrame:
    out = _add_1m_source_features(frame)
    if out.empty:
        return out
    out["source_impulse"] = _bool(out, "source_bullish_volume_impulse")
    out["source_taker_buy_impulse"] = (
        _num(out, "source_ret_1m").ge(0.0008)
        & (_num(out, "source_taker_buy_ratio").ge(0.58) | _num(out, "source_taker_buy_z").ge(2.0))
    )
    return out


def _coverage(symbols: list[str], cfg: V42Config) -> pd.DataFrame:
    rows = []
    for label, root in [("source_1m", _source_dir(cfg)), ("target_1m", _target_dir(cfg))]:
        files = {path.stem.upper() for path in root.glob("*.parquet")} if root.exists() else set()
        rows.append(
            {
                "dataset": label,
                "path": str(root),
                "selected_symbols_available": len([s for s in symbols if s in files]),
                "selected_symbols_required": len(symbols),
                "status": "ok" if symbols and all(s in files for s in symbols) else "missing_or_incomplete",
            }
        )
    rows.append(
        {
            "dataset": "cic_trade_cache",
            "path": str(cfg.trade_cache_path),
            "selected_symbols_available": int(cfg.trade_cache_path.exists()),
            "selected_symbols_required": 1,
            "status": "ok" if cfg.trade_cache_path.exists() else "missing_or_incomplete",
        }
    )
    return pd.DataFrame(rows)


def _bool_frame(columns: dict[str, pd.Series]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    frame = pd.concat(columns, axis=1)
    return frame.astype("boolean").fillna(False).astype(bool)


def _source_matrices(symbols: list[str], cfg: V42Config) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.Series]:
    source_by_symbol: dict[str, pd.DataFrame] = {}
    impulse_cols: dict[str, pd.Series] = {}
    taker_cols: dict[str, pd.Series] = {}
    ret5_cols: dict[str, pd.Series] = {}
    for symbol in symbols:
        path = _source_dir(cfg) / f"{symbol}.parquet"
        if not path.exists():
            continue
        source = _decorate_source(_read_1m_symbol(path, cfg.source_exchange, symbol))
        if source.empty:
            continue
        source = source.drop_duplicates("bar_close_time").set_index("bar_close_time").sort_index()
        source_by_symbol[symbol] = source
        impulse_cols[symbol] = _bool(source, "source_impulse")
        taker_cols[symbol] = _bool(source, "source_taker_buy_impulse")
        ret5_cols[symbol] = _num(source, "source_ret_5m")
    impulse_matrix = _bool_frame(impulse_cols)
    taker_matrix = _bool_frame(taker_cols)
    ret5_matrix = pd.concat(ret5_cols, axis=1) if ret5_cols else pd.DataFrame()
    density = impulse_matrix.mean(axis=1) if not impulse_matrix.empty else pd.Series(dtype=float)
    return source_by_symbol, {"impulse": impulse_matrix, "taker": taker_matrix, "ret5": ret5_matrix}, density


def _mapped_symbol(symbols: list[str], symbol: str, mode: str) -> str | None:
    if len(symbols) <= 1 or symbol not in symbols:
        return None
    idx = symbols.index(symbol)
    if mode == "cyclic":
        return symbols[(idx + 1) % len(symbols)]
    if mode == "reverse":
        return symbols[-idx - 1]
    raise ValueError(mode)


def _aligned_signal(matrix: pd.DataFrame, symbol: str | None, times: pd.Series) -> pd.Series:
    if symbol is None or matrix.empty or symbol not in matrix.columns:
        return pd.Series(False, index=times.index, dtype=bool)
    reindexed = matrix[symbol].reindex(pd.to_datetime(times, utc=True, errors="coerce"))
    return pd.Series(reindexed.astype("boolean").fillna(False).to_numpy(dtype=bool), index=times.index)


def _event_metric(frame: pd.DataFrame, label: str, cfg: V42Config, target_only_net20: float | None = None) -> dict:
    ret_col = f"target_future_ret_{cfg.horizon_minutes}m"
    mfe_col = f"target_future_mfe_{cfg.horizon_minutes}m"
    mae_col = f"target_future_mae_{cfg.horizon_minutes}m"
    if frame.empty or ret_col not in frame.columns:
        return {
            "candidate": label,
            "events": 0,
            "gross_return": np.nan,
            "net10": np.nan,
            "net20": np.nan,
            "net30": np.nan,
            "target_only_net20": target_only_net20,
            "incremental_lift": np.nan,
            "random_lift": np.nan,
            "shuffled_lift": np.nan,
            "density_matched_random_net20": np.nan,
            "density_matched_random_lift": np.nan,
            "month_cap35_net20": np.nan,
            "max_symbol_contribution": np.nan,
            "hit_1pct": np.nan,
            "adverse_0_5pct": np.nan,
        }
    local = frame.dropna(subset=[ret_col]).copy()
    if local.empty:
        return {
            "candidate": label,
            "events": 0,
            "gross_return": np.nan,
            "net10": np.nan,
            "net20": np.nan,
            "net30": np.nan,
            "target_only_net20": target_only_net20,
            "incremental_lift": np.nan,
            "random_lift": np.nan,
            "shuffled_lift": np.nan,
            "month_cap35_net20": np.nan,
            "max_symbol_contribution": np.nan,
            "hit_1pct": np.nan,
            "adverse_0_5pct": np.nan,
        }
    local["net10"] = _net_at_cost(local, ret_col, 10.0)
    local["net20"] = _net_at_cost(local, ret_col, 20.0)
    local["net30"] = _net_at_cost(local, ret_col, 30.0)
    symbol_net = local.groupby("symbol", dropna=False)["net20"].sum()
    total = float(symbol_net.sum())
    if total == 0:
        max_symbol = np.nan
    else:
        max_symbol = float((symbol_net / total).abs().max())
    return {
        "candidate": label,
        "events": int(len(local)),
        "gross_return": float(_num(local, ret_col).mean()),
        "net10": float(_num(local, "net10").mean()),
        "net20": float(_num(local, "net20").mean()),
        "net30": float(_num(local, "net30").mean()),
        "target_only_net20": target_only_net20,
        "incremental_lift": float(_num(local, "net20").mean() - target_only_net20)
        if target_only_net20 is not None and not pd.isna(target_only_net20)
        else np.nan,
        "random_lift": np.nan,
        "shuffled_lift": np.nan,
        "density_matched_random_net20": np.nan,
        "density_matched_random_lift": np.nan,
        "month_cap35_net20": _month_cap35(local, "net20"),
        "max_symbol_contribution": max_symbol,
        "hit_1pct": float(_num(local, mfe_col).ge(0.01).mean()),
        "adverse_0_5pct": float(_num(local, mae_col).le(-0.005).mean()),
    }


def _density_matched_random(target_only: pd.DataFrame, source_plus: pd.DataFrame, cfg: V42Config) -> pd.DataFrame:
    if target_only.empty or source_plus.empty or "market_density_bucket" not in target_only.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed)
    sampled = []
    counts = source_plus["market_density_bucket"].value_counts(dropna=False)
    for bucket, count in counts.items():
        local = target_only[target_only["market_density_bucket"].eq(bucket)]
        if local.empty:
            continue
        take = min(int(count), len(local))
        idx = rng.choice(local.index.to_numpy(), size=take, replace=False)
        sampled.append(local.loc[idx])
    return pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()


def _source_target_attribution_1m(symbols: list[str], cfg: V42Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, matrices, density = _source_matrices(symbols, cfg)
    impulse_matrix = matrices["impulse"]
    taker_matrix = matrices["taker"]
    ret5_matrix = matrices["ret5"]
    if impulse_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    density_bucket = pd.Series("missing", index=density.index, dtype=object)
    valid_density = density.dropna()
    if valid_density.nunique() >= 2:
        density_bucket.loc[valid_density.index] = pd.qcut(valid_density, q=5, duplicates="drop").astype(str)

    event_frames: dict[str, list[pd.DataFrame]] = {
        "A0_bybit_target_only_reclaim": [],
        "A1_binance_source_impulse_only": [],
        "A2_binance_source_plus_bybit_reclaim": [],
        "A3_random_source_plus_bybit_reclaim": [],
        "A4_shuffled_source_plus_bybit_reclaim": [],
        "A6_source_impulse_target_already_moved": [],
        "A7_source_impulse_target_lagging": [],
        "A8_source_taker_buy_plus_bybit_reclaim": [],
        "A9_random_source_target_lagging": [],
        "A10_shuffled_source_target_lagging": [],
    }
    all_target_rows: list[pd.DataFrame] = []
    event_detail_rows = []

    for symbol in symbols:
        target_path = _target_dir(cfg) / f"{symbol}.parquet"
        if not target_path.exists() or symbol not in impulse_matrix.columns:
            continue
        target = _add_1m_target_features(_read_1m_symbol(target_path, cfg.target_exchange, symbol), cfg.horizon_minutes)
        if target.empty:
            continue
        target = target.drop_duplicates("bar_close_time").sort_values("bar_close_time")
        source_impulse = _aligned_signal(impulse_matrix, symbol, target["bar_close_time"])
        source_taker = _aligned_signal(taker_matrix, symbol, target["bar_close_time"])
        random_symbol = _mapped_symbol(symbols, symbol, "cyclic")
        shuffled_symbol = _mapped_symbol(symbols, symbol, "reverse")
        random_impulse = _aligned_signal(impulse_matrix, random_symbol, target["bar_close_time"])
        shuffled_impulse = _aligned_signal(impulse_matrix, shuffled_symbol, target["bar_close_time"])
        source_ret_5m = (
            pd.Series(
                ret5_matrix[symbol].reindex(pd.to_datetime(target["bar_close_time"], utc=True, errors="coerce")).to_numpy(),
                index=target.index,
            )
            if symbol in ret5_matrix.columns
            else pd.Series(np.nan, index=target.index)
        )
        random_ret_5m = (
            pd.Series(
                ret5_matrix[random_symbol].reindex(pd.to_datetime(target["bar_close_time"], utc=True, errors="coerce")).to_numpy(),
                index=target.index,
            )
            if random_symbol in ret5_matrix.columns
            else pd.Series(np.nan, index=target.index)
        )
        shuffled_ret_5m = (
            pd.Series(
                ret5_matrix[shuffled_symbol].reindex(pd.to_datetime(target["bar_close_time"], utc=True, errors="coerce")).to_numpy(),
                index=target.index,
            )
            if shuffled_symbol in ret5_matrix.columns
            else pd.Series(np.nan, index=target.index)
        )
        local = target.copy()
        local["symbol"] = symbol
        local["source_impulse"] = source_impulse
        local["source_taker_buy_impulse"] = source_taker
        local["random_source_impulse"] = random_impulse
        local["shuffled_source_impulse"] = shuffled_impulse
        local["source_ret_5m"] = source_ret_5m
        local["random_source_ret_5m"] = random_ret_5m
        local["shuffled_source_ret_5m"] = shuffled_ret_5m
        local["source_target_ret_5m_gap"] = _num(local, "source_ret_5m") - _num(local, "target_ret_5m")
        local["random_source_target_ret_5m_gap"] = _num(local, "random_source_ret_5m") - _num(local, "target_ret_5m")
        local["shuffled_source_target_ret_5m_gap"] = _num(local, "shuffled_source_ret_5m") - _num(local, "target_ret_5m")
        local["target_lagging"] = _num(local, "source_target_ret_5m_gap").ge(0.003) & _num(local, "target_ret_5m").lt(
            _num(local, "source_ret_5m")
        )
        local["random_target_lagging"] = _num(local, "random_source_target_ret_5m_gap").ge(0.003) & _num(
            local, "target_ret_5m"
        ).lt(_num(local, "random_source_ret_5m"))
        local["shuffled_target_lagging"] = _num(local, "shuffled_source_target_ret_5m_gap").ge(0.003) & _num(
            local, "target_ret_5m"
        ).lt(_num(local, "shuffled_source_ret_5m"))
        local["target_already_moved"] = _num(local, "target_ret_5m").ge(_num(local, "source_ret_5m")) | _num(
            local, "target_ret_5m"
        ).ge(0.004)
        local["market_impulse_density"] = density.reindex(pd.to_datetime(local["bar_close_time"], utc=True, errors="coerce")).to_numpy()
        local["market_density_bucket"] = density_bucket.reindex(
            pd.to_datetime(local["bar_close_time"], utc=True, errors="coerce")
        ).fillna("missing").to_numpy()
        target_condition = _bool(local, "target_reclaim_proxy")
        masks = {
            "A0_bybit_target_only_reclaim": target_condition,
            "A1_binance_source_impulse_only": _bool(local, "source_impulse"),
            "A2_binance_source_plus_bybit_reclaim": _bool(local, "source_impulse") & target_condition,
            "A3_random_source_plus_bybit_reclaim": _bool(local, "random_source_impulse") & target_condition,
            "A4_shuffled_source_plus_bybit_reclaim": _bool(local, "shuffled_source_impulse") & target_condition,
            "A6_source_impulse_target_already_moved": _bool(local, "source_impulse") & _bool(local, "target_already_moved"),
            "A7_source_impulse_target_lagging": _bool(local, "source_impulse") & _bool(local, "target_lagging") & target_condition,
            "A8_source_taker_buy_plus_bybit_reclaim": _bool(local, "source_taker_buy_impulse") & target_condition,
            "A9_random_source_target_lagging": _bool(local, "random_source_impulse") & _bool(local, "random_target_lagging") & target_condition,
            "A10_shuffled_source_target_lagging": _bool(local, "shuffled_source_impulse")
            & _bool(local, "shuffled_target_lagging")
            & target_condition,
        }
        all_target_rows.append(local[target_condition].copy())
        for label, mask in masks.items():
            subset = local[mask.fillna(False)].copy()
            if len(subset):
                event_frames[label].append(subset)
                event_detail_rows.append({"candidate": label, "symbol": symbol, "events": int(len(subset))})

    combined = {label: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for label, parts in event_frames.items()}
    target_only = combined.get("A0_bybit_target_only_reclaim", pd.DataFrame())
    source_plus = combined.get("A2_binance_source_plus_bybit_reclaim", pd.DataFrame())
    density_random = _density_matched_random(
        pd.concat(all_target_rows, ignore_index=True) if all_target_rows else pd.DataFrame(),
        source_plus,
        cfg,
    )
    combined["A5_density_matched_random_target_reclaim"] = density_random
    target_net = _event_metric(target_only, "A0_bybit_target_only_reclaim", cfg)["net20"] if len(target_only) else np.nan
    rows = []
    for label, frame in combined.items():
        row = _event_metric(frame, label, cfg, target_net)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        net_lookup = dict(zip(summary["candidate"], summary["net20"]))
        source_net = net_lookup.get("A2_binance_source_plus_bybit_reclaim", np.nan)
        random_net = net_lookup.get("A3_random_source_plus_bybit_reclaim", np.nan)
        shuffled_net = net_lookup.get("A4_shuffled_source_plus_bybit_reclaim", np.nan)
        density_net = net_lookup.get("A5_density_matched_random_target_reclaim", np.nan)
        summary.loc[summary["candidate"].eq("A2_binance_source_plus_bybit_reclaim"), "random_lift"] = source_net - random_net
        summary.loc[summary["candidate"].eq("A2_binance_source_plus_bybit_reclaim"), "shuffled_lift"] = source_net - shuffled_net
        summary.loc[
            summary["candidate"].eq("A2_binance_source_plus_bybit_reclaim"),
            "density_matched_random_net20",
        ] = density_net
        summary.loc[
            summary["candidate"].eq("A2_binance_source_plus_bybit_reclaim"),
            "density_matched_random_lift",
        ] = source_net - density_net
        lag_net = net_lookup.get("A7_source_impulse_target_lagging", np.nan)
        lag_random_net = net_lookup.get("A9_random_source_target_lagging", np.nan)
        lag_shuffled_net = net_lookup.get("A10_shuffled_source_target_lagging", np.nan)
        summary.loc[summary["candidate"].eq("A7_source_impulse_target_lagging"), "random_lift"] = (
            lag_net - lag_random_net
        )
        summary.loc[summary["candidate"].eq("A7_source_impulse_target_lagging"), "shuffled_lift"] = (
            lag_net - lag_shuffled_net
        )
    detail = pd.DataFrame(event_detail_rows)
    return summary.sort_values("candidate"), detail


def _load_p2_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path)
    pool = _pool_trades(trades, P2_POOL)
    if pool.empty:
        return pool
    pool = pool[pd.to_numeric(pool.get("cost_single_side_bps", pd.Series(np.nan, index=pool.index)), errors="coerce").eq(10.0)].copy()
    for col in ["signal_time", "entry_time", "exit_time"]:
        pool[col] = pd.to_datetime(pool[col], utc=True, errors="coerce")
    for col in ["net_return", "gross_return", "volume_impulse_density", "cluster_impulse_density"]:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce")
    pool["month"] = pool["entry_time"].dt.strftime("%Y-%m")
    pool = pool.dropna(subset=["entry_time", "exit_time", "net_return"]).sort_values(["entry_time", "symbol"])
    if "burst_count_so_far" not in pool.columns:
        pool = _add_asof_burst_phase(pool)
    return pool


def _asof_values(series: pd.Series, times: pd.Series, *, max_age_seconds: int = 90) -> pd.Series:
    if series.empty:
        return pd.Series(np.nan, index=times.index, dtype="float64")
    source = pd.DataFrame(
        {
            "source_time": pd.to_datetime(series.index, utc=True, errors="coerce"),
            "value": pd.to_numeric(series.to_numpy(), errors="coerce"),
        }
    ).dropna(subset=["source_time"])
    decisions = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(times, utc=True, errors="coerce"),
            "_order": np.arange(len(times)),
        }
    )
    if source.empty or decisions.empty:
        return pd.Series(np.nan, index=times.index, dtype="float64")
    merged = pd.merge_asof(
        decisions.sort_values("decision_time"),
        source.sort_values("source_time"),
        left_on="decision_time",
        right_on="source_time",
        direction="backward",
    )
    age = (merged["decision_time"] - merged["source_time"]).dt.total_seconds()
    merged.loc[age.gt(max_age_seconds), "value"] = np.nan
    merged = merged.sort_values("_order")
    return pd.Series(merged["value"].to_numpy(), index=times.index)


def _has_source_context(matrix: pd.DataFrame, symbol: str, times: pd.Series) -> pd.Series:
    if matrix.empty or symbol not in matrix.columns:
        return pd.Series(False, index=times.index, dtype=bool)
    values = _asof_values(matrix[symbol].astype(float), times)
    return values.notna()


def _has_prior_event(matrix: pd.DataFrame, symbol: str, times: pd.Series, minutes: int) -> pd.Series:
    if matrix.empty or symbol not in matrix.columns:
        return pd.Series(False, index=times.index, dtype=bool)
    series = matrix[symbol].astype(float).sort_index()
    # Causal rolling count: include only source bars closed at or before decision time.
    rolling = series.rolling(f"{int(minutes)}min", min_periods=1).max()
    indexed = _asof_values(rolling, times)
    return pd.Series(indexed.fillna(0.0).gt(0.0).to_numpy(), index=times.index)


def _annotate_trade_source_context(pool: pd.DataFrame, symbols: list[str], cfg: V42Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pd.DataFrame()
    _, matrices, density = _source_matrices(symbols, cfg)
    impulse_matrix = matrices["impulse"]
    taker_matrix = matrices["taker"]
    rows = []
    coverage = []
    for symbol, group in pool.groupby("symbol", sort=False):
        local = group.copy()
        covered = symbol in impulse_matrix.columns and symbol in taker_matrix.columns
        local["binance_source_covered"] = _has_source_context(impulse_matrix, symbol, local["entry_time"]) if covered else False
        coverage.append(
            {
                "symbol": symbol,
                "trades": int(len(local)),
                "source_1m_covered": bool(covered),
                "entry_context_covered_trades": int(local["binance_source_covered"].sum()),
            }
        )
        local["binance_impulse_prior_5m"] = _has_prior_event(impulse_matrix, symbol, local["entry_time"], 5) if covered else False
        local["binance_impulse_prior_15m"] = _has_prior_event(impulse_matrix, symbol, local["entry_time"], 15) if covered else False
        local["binance_taker_buy_prior_5m"] = _has_prior_event(taker_matrix, symbol, local["entry_time"], 5) if covered else False
        local["binance_taker_buy_prior_15m"] = _has_prior_event(taker_matrix, symbol, local["entry_time"], 15) if covered else False
        local["binance_market_density_at_entry"] = (
            _asof_values(density, local["entry_time"]).to_numpy()
            if not density.empty
            else np.nan
        )
        rows.append(local)
    annotated = pd.concat(rows, ignore_index=True) if rows else pool.copy()
    return annotated, pd.DataFrame(coverage)


def _trade_bucket_row(sample: pd.DataFrame, bucket: str) -> dict:
    if sample.empty:
        return {
            "bucket": bucket,
            "trades": 0,
            "net20": np.nan,
            "net30": np.nan,
            "hit_rate": np.nan,
            "month_cap35_net20": np.nan,
            "CP60_delta": np.nan,
            "O6_delta": np.nan,
            "ProtectA_delta": np.nan,
            "max_symbol_contribution": np.nan,
        }
    local = sample.copy()
    local["net20"] = _num(local, "net_return")
    local["net30"] = _num(local, "gross_return") - 0.006 if "gross_return" in local.columns else _num(local, "net_return") - 0.002
    symbol_net = local.groupby("symbol", dropna=False)["net20"].sum()
    total = float(symbol_net.sum())
    return {
        "bucket": bucket,
        "trades": int(len(local)),
        "net20": float(_num(local, "net20").mean()),
        "net30": float(_num(local, "net30").mean()),
        "hit_rate": float(_num(local, "net20").gt(0).mean()),
        "month_cap35_net20": _month_cap_trade(local, "net20"),
        "CP60_delta": np.nan,
        "O6_delta": np.nan,
        "ProtectA_delta": np.nan,
        "max_symbol_contribution": float((symbol_net / total).abs().max()) if total else np.nan,
    }


def _month_cap_trade(frame: pd.DataFrame, value_col: str) -> float:
    if frame.empty:
        return np.nan
    local = frame.dropna(subset=[value_col]).copy()
    if local.empty:
        return np.nan
    total = float(_num(local, value_col).sum())
    if total <= 0:
        return float(_num(local, value_col).mean())
    monthly = _num(local, value_col).groupby(local["month"].astype(str), sort=False).sum()
    return float(monthly.clip(upper=0.35 * total).sum() / len(local))


def _fusion_with_cic(symbols: list[str], cfg: V42Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _load_p2_trades(cfg.trade_cache_path)
    if pool.empty:
        return pd.DataFrame(
            [
                {
                    "bucket": "blocked_missing_p2_trade_cache",
                    "trades": 0,
                    "net20": np.nan,
                    "net30": np.nan,
                    "hit_rate": np.nan,
                    "month_cap35_net20": np.nan,
                    "CP60_delta": np.nan,
                    "O6_delta": np.nan,
                    "ProtectA_delta": np.nan,
                    "max_symbol_contribution": np.nan,
                }
            ]
        ), pd.DataFrame()
    annotated, coverage = _annotate_trade_source_context(pool, symbols, cfg)
    covered = annotated[annotated["binance_source_covered"].astype(bool)].copy()
    rows = [
        _trade_bucket_row(annotated, "P2_all_trades"),
        _trade_bucket_row(covered, "P2_source_context_covered"),
    ]
    bucket_specs = [
        ("P2_with_binance_impulse_prior_15m", covered["binance_impulse_prior_15m"].astype(bool)),
        ("P2_without_binance_impulse_prior_15m", ~covered["binance_impulse_prior_15m"].astype(bool)),
        ("P2_with_binance_taker_buy_prior_15m", covered["binance_taker_buy_prior_15m"].astype(bool)),
        ("P2_without_binance_taker_buy_prior_15m", ~covered["binance_taker_buy_prior_15m"].astype(bool)),
        (
            "CIC1_with_binance_impulse_prior_15m",
            covered["candidate"].astype(str).eq("CIC1_beta_extreme") & covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "CIC1_without_binance_impulse_prior_15m",
            covered["candidate"].astype(str).eq("CIC1_beta_extreme") & ~covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "CIC2_with_binance_impulse_prior_15m",
            covered["candidate"].astype(str).eq("CIC2_beta_broad") & covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "CIC2_without_binance_impulse_prior_15m",
            covered["candidate"].astype(str).eq("CIC2_beta_broad") & ~covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "O6_eligible_with_binance_impulse_prior_15m",
            _num(covered, "burst_count_so_far").ge(9) & covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "O6_eligible_without_binance_impulse_prior_15m",
            _num(covered, "burst_count_so_far").ge(9) & ~covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "low_coimpulse_with_binance_impulse_prior_15m",
            _num(covered, "volume_impulse_density").le(_num(covered, "volume_impulse_density").median())
            & covered["binance_impulse_prior_15m"].astype(bool),
        ),
        (
            "low_coimpulse_without_binance_impulse_prior_15m",
            _num(covered, "volume_impulse_density").le(_num(covered, "volume_impulse_density").median())
            & ~covered["binance_impulse_prior_15m"].astype(bool),
        ),
    ]
    for label, mask in bucket_specs:
        rows.append(_trade_bucket_row(covered[mask.fillna(False)], label))
    summary = pd.DataFrame(rows)
    return summary, coverage


def _write_notes(
    root: Path,
    coverage: pd.DataFrame,
    attribution: pd.DataFrame,
    fusion: pd.DataFrame,
) -> None:
    lines = [
        "# v4.2 Source Attribution & Target Fusion",
        "",
        "## Scope",
        "- v4.2A evaluates 1m Binance -> Bybit attribution with target-only, random, shuffled, and density-matched controls.",
        "- v4.2B tests Binance source context as a diagnostic feature on existing CIC/P2 trades. It does not change live actions.",
        "",
        "## Coverage",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"- {row.dataset}: {row.selected_symbols_available}/{row.selected_symbols_required} `{row.status}`.")
    lines.append("")
    lines.append("## Attribution")
    if attribution.empty:
        lines.append("- No 1m attribution rows were produced.")
    else:
        a2 = attribution[attribution["candidate"].eq("A2_binance_source_plus_bybit_reclaim")]
        if not a2.empty:
            row = a2.iloc[0]
            lines.append(
                f"- A2 source+target net20={row.net20:.4%}, target-only lift={row.incremental_lift:.4%}, "
                f"random lift={row.random_lift:.4%}, shuffled lift={row.shuffled_lift:.4%}."
            )
        a7 = attribution[attribution["candidate"].eq("A7_source_impulse_target_lagging")]
        if not a7.empty:
            row = a7.iloc[0]
            lines.append(
                f"- A7 source impulse + target lagging is a small diagnostic pocket: events={int(row.events)}, "
                f"net20={row.net20:.4%}, random lift={row.random_lift:.4%}, shuffled lift={row.shuffled_lift:.4%}."
            )
        best = attribution.sort_values("net20", ascending=False).head(1).iloc[0]
        lines.append(f"- Best 1m attribution bucket: {best.candidate}, net20={best.net20:.4%}, events={int(best.events)}.")
    lines.append("")
    lines.append("## Fusion")
    if fusion.empty:
        lines.append("- No CIC fusion rows were produced.")
    else:
        covered = fusion[fusion["bucket"].eq("P2_source_context_covered")]
        if not covered.empty:
            row = covered.iloc[0]
            lines.append(f"- P2 source-context covered trades={int(row.trades)}, net20={row.net20:.4%}.")
        pairs = [
            ("P2_with_binance_impulse_prior_15m", "P2_without_binance_impulse_prior_15m"),
            ("P2_with_binance_taker_buy_prior_15m", "P2_without_binance_taker_buy_prior_15m"),
            ("O6_eligible_with_binance_impulse_prior_15m", "O6_eligible_without_binance_impulse_prior_15m"),
            ("low_coimpulse_with_binance_impulse_prior_15m", "low_coimpulse_without_binance_impulse_prior_15m"),
        ]
        for yes, no in pairs:
            y = fusion[fusion["bucket"].eq(yes)]
            n = fusion[fusion["bucket"].eq(no)]
            if y.empty or n.empty:
                continue
            lines.append(f"- {yes} vs {no}: {float(y.iloc[0].net20):.4%} vs {float(n.iloc[0].net20):.4%}.")
    lines.extend(
        [
            "",
            "## Decision",
            "- No standalone cross-exchange strategy is promoted by this report.",
            "- Binance source context can only be considered useful if source+target beats target-only and random/shuffled controls, and fusion buckets beat non-fusion buckets with enough covered CIC trades.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v42_source_attribution_target_fusion(cfg: V42Config = V42Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    selected, _ = _select_common_symbols(_source_dir(cfg), _target_dir(cfg), cfg.top_n)
    coverage = _coverage(selected, cfg)
    attribution, detail = _source_target_attribution_1m(selected, cfg)
    fusion, fusion_coverage = _fusion_with_cic(selected, cfg)

    outputs = {
        "coverage_audit": root / "coverage_audit.csv",
        "source_target_attribution_1m": root / "source_target_attribution_1m.csv",
        "source_target_event_detail": root / "source_target_event_detail.csv",
        "cross_exchange_fusion_with_cic": root / "cross_exchange_fusion_with_cic.csv",
        "fusion_coverage": root / "fusion_coverage.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["coverage_audit"], index=False)
    attribution.to_csv(outputs["source_target_attribution_1m"], index=False)
    detail.to_csv(outputs["source_target_event_detail"], index=False)
    fusion.to_csv(outputs["cross_exchange_fusion_with_cic"], index=False)
    fusion_coverage.to_csv(outputs["fusion_coverage"], index=False)
    _write_notes(root, coverage, attribution, fusion)
    return outputs


__all__ = ["V42Config", "write_v42_source_attribution_target_fusion"]
