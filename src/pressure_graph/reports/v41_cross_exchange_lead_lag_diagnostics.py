"""v4.1 Cross-Exchange Lead-Lag Diagnostics.

This stage is not a strategy search. It asks whether Binance -> Bybit
propagation exists at measurable horizons, whether the source condition adds
incremental value over Bybit-only baselines, and where the first-pass v4.0
motifs broke down.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v40_cross_exchange_lead_lag import (
    DATA_ROOT,
    KLINE_DATASET,
    REPORT_ROOT as V40_REPORT_ROOT,
    SOURCE_EXCHANGE,
    TARGET_EXCHANGE,
    V40Config,
    _add_bar_features,
    _bool,
    _build_same_symbol_frame,
    _load_klines,
    _month_cap35,
    _net_at_cost,
    _num,
    _select_common_symbols,
)


REPORT_ROOT = Path("reports/v4_1_cross_exchange_lead_lag_diagnostics")


@dataclass(frozen=True)
class V41Config:
    report_root: Path = REPORT_ROOT
    data_root: Path = DATA_ROOT
    source_exchange: str = SOURCE_EXCHANGE
    target_exchange: str = TARGET_EXCHANGE
    dataset: str = KLINE_DATASET
    top_n: int = 50
    max_edge_symbols: int = 50
    cost_bps: float = 20.0
    impulse_volume_z: float = 1.5
    impulse_ret_threshold: float = 0.001
    market_density_threshold: float = 0.20


AVAILABLE_HORIZONS = {
    15: ("target_future_ret_15m", "target_future_mfe_30m", "target_future_mae_30m"),
    30: ("target_future_ret_30m", "target_future_mfe_30m", "target_future_mae_30m"),
    60: ("target_future_ret_1h", "target_future_mfe_1h", "target_future_mae_1h"),
}
REQUESTED_HORIZONS = [1, 3, 5, 10, 15, 30, 60]
LOWER_TIMEFRAME_HORIZONS = [1, 3, 5, 10, 15, 30, 60]


def _v40_cfg(cfg: V41Config) -> V40Config:
    return V40Config(
        report_root=V40_REPORT_ROOT,
        data_root=cfg.data_root,
        source_exchange=cfg.source_exchange,
        target_exchange=cfg.target_exchange,
        dataset=cfg.dataset,
        top_n=cfg.top_n,
        impulse_volume_z=cfg.impulse_volume_z,
        impulse_ret_threshold=cfg.impulse_ret_threshold,
        market_density_threshold=cfg.market_density_threshold,
        cost_bps=cfg.cost_bps,
    )


def _prepare(cfg: V41Config) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v40 = _v40_cfg(cfg)
    source_dir = cfg.data_root / "raw" / cfg.source_exchange / cfg.dataset
    target_dir = cfg.data_root / "raw" / cfg.target_exchange / cfg.dataset
    selected, _ = _select_common_symbols(source_dir, target_dir, cfg.top_n)
    source = _add_bar_features(_load_klines(source_dir, cfg.source_exchange, selected), v40, "source")
    target = _add_bar_features(_load_klines(target_dir, cfg.target_exchange, selected), v40, "target")
    same = _build_same_symbol_frame(source, target, v40)
    if not same.empty:
        same["source_target_ret_15m_gap"] = _num(same, "source_ret_15m").fillna(0.0) - _num(
            same, "target_ret_15m"
        ).fillna(0.0)
        same["source_target_ret_1h_gap"] = _num(same, "source_ret_1h").fillna(0.0) - _num(
            same, "target_ret_1h"
        ).fillna(0.0)
        same["premium_proxy"] = _num(same, "source_close") / _num(same, "target_close") - 1.0
    return selected, source, target, same


def _event_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "binance_price_impulse": _num(frame, "source_ret_15m").ge(0.003),
        "binance_volume_shock": _num(frame, "source_volume_z").ge(1.5),
        "binance_taker_buy_imbalance": _num(frame, "source_taker_buy_ratio").ge(0.58)
        | _num(frame, "source_taker_buy_z").ge(1.5),
        "binance_taker_buy_plus_price": (
            (_num(frame, "source_ret_15m").ge(0.001))
            & (
                _num(frame, "source_taker_buy_ratio").ge(0.58)
                | _num(frame, "source_taker_buy_z").ge(1.5)
            )
        ),
        "binance_breakout": _bool(frame, "source_breakout_proxy"),
        "binance_bullish_volume_impulse": _bool(frame, "source_bullish_volume_impulse"),
    }


def _metric_row(frame: pd.DataFrame, ret_col: str, mfe_col: str, mae_col: str, cost_bps: float) -> dict:
    net = _net_at_cost(frame, ret_col, cost_bps)
    mfe = _num(frame, mfe_col)
    mae = _num(frame, mae_col)
    return {
        "events": int(len(frame)),
        "future_return": float(_num(frame, ret_col).mean()) if len(frame) else np.nan,
        "future_max_up": float(mfe.mean()) if len(frame) else np.nan,
        "future_max_down": float(mae.mean()) if len(frame) else np.nan,
        "net20": float(net.mean()) if len(frame) else np.nan,
        "hit_0_5pct": float(mfe.ge(0.005).mean()) if len(frame) else np.nan,
        "hit_1pct": float(mfe.ge(0.01).mean()) if len(frame) else np.nan,
        "hit_2pct": float(mfe.ge(0.02).mean()) if len(frame) else np.nan,
        "positive_close_return_rate": float(_num(frame, ret_col).gt(0).mean()) if len(frame) else np.nan,
        "adverse_0_5pct": float(mae.le(-0.005).mean()) if len(frame) else np.nan,
        "adverse_before_hit_proxy": float((mae.le(-0.005) & mfe.ge(0.005)).mean()) if len(frame) else np.nan,
    }


def _response_curve(same: pd.DataFrame, cfg: V41Config) -> pd.DataFrame:
    rows: list[dict] = []
    masks = _event_masks(same) if not same.empty else {}
    for event_type, mask in masks.items():
        for horizon in REQUESTED_HORIZONS:
            base = {
                "source_event_type": event_type,
                "target_horizon_minutes": horizon,
                "timeframe": "15m_full",
            }
            if horizon not in AVAILABLE_HORIZONS:
                rows.append({**base, "status": "requires_1m_or_5m_data", "events": 0})
                continue
            ret_col, mfe_col, mae_col = AVAILABLE_HORIZONS[horizon]
            subset = same[mask.fillna(False)].copy()
            rows.append(
                {
                    **base,
                    "status": "evaluated_15m_proxy",
                    **_metric_row(subset, ret_col, mfe_col, mae_col, cfg.cost_bps),
                    "month_cap35_net20": _month_cap35(
                        subset.assign(net20=_net_at_cost(subset, ret_col, cfg.cost_bps)), "net20"
                    )
                    if len(subset)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _future_max(values: pd.Series, bars: int) -> pd.Series:
    return values.iloc[::-1].shift(1).rolling(bars, min_periods=1).max().iloc[::-1]


def _future_min(values: pd.Series, bars: int) -> pd.Series:
    return values.iloc[::-1].shift(1).rolling(bars, min_periods=1).min().iloc[::-1]


def _read_1m_symbol(path: Path, exchange: str, symbol: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    required = ["bar_close_time", "open", "high", "low", "close", "volume", "turnover"]
    if any(col not in frame.columns for col in required):
        return pd.DataFrame()
    optional = [col for col in ["taker_buy_base", "taker_buy_quote", "trades"] if col in frame.columns]
    out = frame[required + optional].copy()
    out["exchange"] = exchange
    out["symbol"] = symbol
    out["bar_close_time"] = pd.to_datetime(out["bar_close_time"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "turnover", *optional]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["bar_close_time", "close"]).sort_values("bar_close_time")


def _add_1m_source_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("bar_close_time").copy()
    close = _num(out, "close")
    volume = _num(out, "volume")
    out["source_ret_1m"] = close / close.shift(1) - 1.0
    out["source_ret_5m"] = close / close.shift(5) - 1.0
    vol_mean = volume.shift(1).rolling(240, min_periods=30).mean()
    vol_std = volume.shift(1).rolling(240, min_periods=30).std(ddof=0).replace(0, np.nan)
    out["source_volume_z"] = (volume - vol_mean) / vol_std
    if "taker_buy_base" in out.columns:
        buy = _num(out, "taker_buy_base")
        out["source_taker_buy_ratio"] = (buy / volume.replace(0, np.nan)).clip(0.0, 1.0)
        buy_mean = buy.shift(1).rolling(240, min_periods=30).mean()
        buy_std = buy.shift(1).rolling(240, min_periods=30).std(ddof=0).replace(0, np.nan)
        out["source_taker_buy_z"] = (buy - buy_mean) / buy_std
    else:
        out["source_taker_buy_ratio"] = np.nan
        out["source_taker_buy_z"] = np.nan
    out["source_breakout_proxy"] = close.gt(_num(out, "high").shift(1).rolling(30, min_periods=10).max())
    out["source_bullish_volume_impulse"] = (
        _num(out, "source_ret_1m").ge(0.001)
        & _num(out, "source_volume_z").ge(2.0)
        & close.ge(_num(out, "open"))
    )
    return out[
        [
            "bar_close_time",
            "source_ret_1m",
            "source_ret_5m",
            "source_volume_z",
            "source_taker_buy_ratio",
            "source_taker_buy_z",
            "source_breakout_proxy",
            "source_bullish_volume_impulse",
        ]
    ]


def _add_1m_target_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values("bar_close_time").copy()
    close = _num(out, "close")
    high = _num(out, "high")
    low = _num(out, "low")
    keep = ["bar_close_time"]
    for horizon in LOWER_TIMEFRAME_HORIZONS:
        out[f"target_future_ret_{horizon}m"] = close.shift(-horizon) / close - 1.0
        out[f"target_future_mfe_{horizon}m"] = _future_max(high, horizon) / close - 1.0
        out[f"target_future_mae_{horizon}m"] = _future_min(low, horizon) / close - 1.0
        keep.extend(
            [
                f"target_future_ret_{horizon}m",
                f"target_future_mfe_{horizon}m",
                f"target_future_mae_{horizon}m",
            ]
        )
    return out[keep]


def _one_min_event_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "binance_1m_price_impulse": _num(frame, "source_ret_1m").ge(0.0015),
        "binance_1m_5m_price_impulse": _num(frame, "source_ret_5m").ge(0.004),
        "binance_1m_volume_shock": _num(frame, "source_volume_z").ge(2.0),
        "binance_1m_taker_buy_imbalance": _num(frame, "source_taker_buy_ratio").ge(0.58)
        | _num(frame, "source_taker_buy_z").ge(2.0),
        "binance_1m_taker_buy_plus_price": (
            _num(frame, "source_ret_1m").ge(0.0008)
            & (_num(frame, "source_taker_buy_ratio").ge(0.58) | _num(frame, "source_taker_buy_z").ge(2.0))
        ),
        "binance_1m_breakout": _bool(frame, "source_breakout_proxy"),
        "binance_1m_bullish_volume_impulse": _bool(frame, "source_bullish_volume_impulse"),
    }


def _empty_accumulator() -> dict:
    return {
        "events": 0,
        "future_return_sum": 0.0,
        "future_max_up_sum": 0.0,
        "future_max_down_sum": 0.0,
        "net20_sum": 0.0,
        "hit_0_5pct": 0,
        "hit_1pct": 0,
        "hit_2pct": 0,
        "positive_close_return_rate": 0,
        "adverse_0_5pct": 0,
        "adverse_before_hit_proxy": 0,
        "month_net": {},
    }


def _accumulate(acc: dict, subset: pd.DataFrame, horizon: int, cfg: V41Config) -> None:
    ret = _num(subset, f"target_future_ret_{horizon}m").dropna()
    if ret.empty:
        return
    local = subset.loc[ret.index].copy()
    mfe = _num(local, f"target_future_mfe_{horizon}m")
    mae = _num(local, f"target_future_mae_{horizon}m")
    net = ret - 2.0 * float(cfg.cost_bps) / 10_000.0
    acc["events"] += int(len(local))
    acc["future_return_sum"] += float(ret.sum())
    acc["future_max_up_sum"] += float(mfe.sum())
    acc["future_max_down_sum"] += float(mae.sum())
    acc["net20_sum"] += float(net.sum())
    acc["hit_0_5pct"] += int(mfe.ge(0.005).sum())
    acc["hit_1pct"] += int(mfe.ge(0.01).sum())
    acc["hit_2pct"] += int(mfe.ge(0.02).sum())
    acc["positive_close_return_rate"] += int(ret.gt(0).sum())
    acc["adverse_0_5pct"] += int(mae.le(-0.005).sum())
    acc["adverse_before_hit_proxy"] += int((mae.le(-0.005) & mfe.ge(0.005)).sum())
    months = pd.to_datetime(local["bar_close_time"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M").astype(str)
    for month, value in net.groupby(months).sum().items():
        acc["month_net"][month] = acc["month_net"].get(month, 0.0) + float(value)


def _acc_to_row(event_type: str, horizon: int, acc: dict) -> dict:
    n = int(acc["events"])
    month_cap = np.nan
    if n:
        total = float(sum(acc["month_net"].values()))
        if total <= 0:
            month_cap = acc["net20_sum"] / n
        else:
            month_cap = sum(min(v, 0.35 * total) for v in acc["month_net"].values()) / n
    return {
        "source_event_type": event_type,
        "target_horizon_minutes": horizon,
        "timeframe": "1m_sample",
        "status": "evaluated_1m_sample",
        "events": n,
        "future_return": acc["future_return_sum"] / n if n else np.nan,
        "future_max_up": acc["future_max_up_sum"] / n if n else np.nan,
        "future_max_down": acc["future_max_down_sum"] / n if n else np.nan,
        "net20": acc["net20_sum"] / n if n else np.nan,
        "hit_0_5pct": acc["hit_0_5pct"] / n if n else np.nan,
        "hit_1pct": acc["hit_1pct"] / n if n else np.nan,
        "hit_2pct": acc["hit_2pct"] / n if n else np.nan,
        "positive_close_return_rate": acc["positive_close_return_rate"] / n if n else np.nan,
        "adverse_0_5pct": acc["adverse_0_5pct"] / n if n else np.nan,
        "adverse_before_hit_proxy": acc["adverse_before_hit_proxy"] / n if n else np.nan,
        "month_cap35_net20": month_cap,
    }


def _response_curve_1m_sample(symbols: list[str], cfg: V41Config) -> pd.DataFrame:
    source_dir = cfg.data_root / "raw" / cfg.source_exchange / "klines_1m_v4"
    target_dir = cfg.data_root / "raw" / cfg.target_exchange / "klines_1m_v4"
    if not source_dir.exists() or not target_dir.exists():
        return pd.DataFrame()
    accumulators: dict[tuple[str, int], dict] = {}
    for symbol in symbols:
        source_path = source_dir / f"{symbol}.parquet"
        target_path = target_dir / f"{symbol}.parquet"
        if not source_path.exists() or not target_path.exists():
            continue
        source = _add_1m_source_features(_read_1m_symbol(source_path, cfg.source_exchange, symbol))
        target = _add_1m_target_features(_read_1m_symbol(target_path, cfg.target_exchange, symbol))
        if source.empty or target.empty:
            continue
        merged = target.merge(source, on="bar_close_time", how="inner")
        if merged.empty:
            continue
        masks = _one_min_event_masks(merged)
        for event_type, mask in masks.items():
            for horizon in LOWER_TIMEFRAME_HORIZONS:
                key = (event_type, horizon)
                acc = accumulators.setdefault(key, _empty_accumulator())
                _accumulate(acc, merged[mask.fillna(False)], horizon, cfg)
    rows = [_acc_to_row(event, horizon, acc) for (event, horizon), acc in accumulators.items()]
    return pd.DataFrame(rows)


def _mapped_source_signal(same: pd.DataFrame, symbols: list[str], source_col: str, mode: str) -> pd.Series:
    if same.empty:
        return pd.Series(False, index=same.index)
    if len(symbols) <= 1:
        return pd.Series(False, index=same.index)
    if mode == "cyclic":
        mapping = {symbol: symbols[(idx + 1) % len(symbols)] for idx, symbol in enumerate(symbols)}
    elif mode == "reverse":
        mapping = {symbol: symbols[-idx - 1] for idx, symbol in enumerate(symbols)}
    else:
        raise ValueError(mode)
    pivot = (
        same.pivot_table(index="bar_close_time", columns="symbol", values=source_col, aggfunc="max")
        .fillna(0.0)
        .astype(bool)
    )
    times = pd.to_datetime(same["bar_close_time"], utc=True, errors="coerce")
    mapped_symbols = same["symbol"].astype(str).map(mapping)
    out = []
    for ts, symbol in zip(times, mapped_symbols):
        if pd.isna(ts) or symbol not in pivot.columns or ts not in pivot.index:
            out.append(False)
        else:
            out.append(bool(pivot.at[ts, symbol]))
    return pd.Series(out, index=same.index, dtype=bool)


def _candidate_summary(frame: pd.DataFrame, mask: pd.Series, label: str, cfg: V41Config) -> dict:
    subset = frame[mask.fillna(False)].copy()
    if len(subset):
        subset["net20"] = _net_at_cost(subset, "target_future_ret_1h", cfg.cost_bps)
    return {
        "label": label,
        "events": int(len(subset)),
        "net20": float(_num(subset, "net20").mean()) if len(subset) else np.nan,
        "hit_rate": float(_num(subset, "target_future_ret_1h").gt(0).mean()) if len(subset) else np.nan,
        "month_cap35_net20": _month_cap35(subset, "net20") if len(subset) else np.nan,
    }


def _incremental_edge_summary(same: pd.DataFrame, symbols: list[str], cfg: V41Config) -> pd.DataFrame:
    if same.empty:
        return pd.DataFrame()
    random_source = _mapped_source_signal(same, symbols, "source_impulse_prior_1h", "cyclic")
    shuffled_source = _mapped_source_signal(same, symbols, "source_impulse_prior_1h", "reverse")
    specs = [
        (
            "LX1_source_prior_target_lag_reclaim",
            _bool(same, "source_impulse_prior_1h") & _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
            random_source & _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
            shuffled_source & _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
        ),
        (
            "LX2_taker_buy_prior_target_reclaim",
            _bool(same, "source_taker_buy_impulse_prior_1h") & _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
            random_source & _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
            shuffled_source & _bool(same, "target_not_extended") & _bool(same, "target_reclaim_proxy"),
        ),
        (
            "LX4_market_density_lag_reclaim",
            _bool(same, "source_market_impulse_density_high") & _bool(same, "target_lagging_same_symbol") & _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_lagging_same_symbol") & _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_lagging_same_symbol") & _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_lagging_same_symbol") & _bool(same, "target_reclaim_proxy"),
        ),
        (
            "premium_gap_high_target_reclaim",
            _num(same, "premium_proxy").ge(_num(same, "premium_proxy").quantile(0.8)) & _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_reclaim_proxy"),
            _bool(same, "target_reclaim_proxy"),
        ),
    ]
    rows: list[dict] = []
    for candidate, mask, baseline, random_mask, shuffled_mask in specs:
        cand = _candidate_summary(same, mask, "source_plus_target", cfg)
        targ = _candidate_summary(same, baseline, "target_only", cfg)
        rand = _candidate_summary(same, random_mask, "matched_random_source", cfg)
        shuf = _candidate_summary(same, shuffled_mask, "shuffled_source", cfg)
        rows.append(
            {
                "candidate": candidate,
                "source_plus_target_events": cand["events"],
                "source_plus_target_net20": cand["net20"],
                "target_only_events": targ["events"],
                "target_only_net20": targ["net20"],
                "incremental_lift": cand["net20"] - targ["net20"],
                "matched_random_net20": rand["net20"],
                "matched_random_lift": cand["net20"] - rand["net20"],
                "shuffled_net20": shuf["net20"],
                "shuffled_lift": cand["net20"] - shuf["net20"],
                "month_cap35_net20": cand["month_cap35_net20"],
            }
        )
    return pd.DataFrame(rows)


def _symbol_index(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol, group in frame.groupby("symbol", sort=False):
        local = group.sort_values("bar_close_time").set_index("bar_close_time")
        out[str(symbol)] = local
    return out


def _lead_lag_edge_matrix(source: pd.DataFrame, target: pd.DataFrame, symbols: list[str], cfg: V41Config) -> pd.DataFrame:
    source_map = _symbol_index(source[source["symbol"].isin(symbols)])
    target_map = _symbol_index(target[target["symbol"].isin(symbols)])
    rows: list[dict] = []
    for source_symbol in symbols[: cfg.max_edge_symbols]:
        src = source_map.get(source_symbol)
        if src is None or src.empty:
            continue
        event_times = src.index[_bool(src, "source_bullish_volume_impulse")]
        if len(event_times) == 0:
            continue
        for target_symbol in symbols[: cfg.max_edge_symbols]:
            tgt = target_map.get(target_symbol)
            if tgt is None or tgt.empty:
                continue
            aligned = tgt.reindex(event_times).dropna(subset=["target_future_ret_1h"])
            if aligned.empty:
                continue
            for horizon, (ret_col, mfe_col, mae_col) in AVAILABLE_HORIZONS.items():
                net = _net_at_cost(aligned, ret_col, cfg.cost_bps)
                hit = _num(aligned, ret_col).gt(0)
                adverse = _num(aligned, mae_col).le(-0.005)
                shrink = len(aligned) / (len(aligned) + 50.0)
                response_net = float(net.mean()) if len(aligned) else np.nan
                hit_rate = float(hit.mean()) if len(aligned) else np.nan
                adverse_rate = float(adverse.mean()) if len(aligned) else np.nan
                edge_weight = (
                    (0.0 if pd.isna(response_net) else response_net)
                    * (0.0 if pd.isna(hit_rate) else hit_rate)
                    * shrink
                    - (0.0 if pd.isna(adverse_rate) else adverse_rate) * 0.0005
                )
                rows.append(
                    {
                        "source_symbol": source_symbol,
                        "target_symbol": target_symbol,
                        "horizon_minutes": horizon,
                        "events": int(len(aligned)),
                        "response_net20": response_net,
                        "hit_rate": hit_rate,
                        "adverse_rate": adverse_rate,
                        "edge_weight": float(edge_weight),
                        "same_symbol": bool(source_symbol == target_symbol),
                    }
                )
    return pd.DataFrame(rows).sort_values(["edge_weight", "events"], ascending=[False, False])


def _bucket_summary(frame: pd.DataFrame, value_col: str, mask: pd.Series, label: str, cfg: V41Config) -> pd.DataFrame:
    local = frame[mask.fillna(False)].copy()
    if local.empty or _num(local, value_col).nunique(dropna=True) < 2:
        return pd.DataFrame()
    local["bucket"] = pd.qcut(_num(local, value_col), q=5, duplicates="drop")
    local["net20"] = _net_at_cost(local, "target_future_ret_1h", cfg.cost_bps)
    rows = []
    for bucket, group in local.groupby("bucket", observed=True):
        rows.append(
            {
                "diagnostic": label,
                "feature": value_col,
                "bucket": str(bucket),
                "events": int(len(group)),
                "feature_mean": float(_num(group, value_col).mean()),
                "net20": float(_num(group, "net20").mean()),
                "hit_rate": float(_num(group, "target_future_ret_1h").gt(0).mean()),
                "month_cap35_net20": _month_cap35(group, "net20"),
            }
        )
    return pd.DataFrame(rows)


def _gap_bucket_summary(same: pd.DataFrame, cfg: V41Config) -> pd.DataFrame:
    if same.empty:
        return pd.DataFrame()
    mask = _bool(same, "source_impulse_prior_1h") & _bool(same, "target_reclaim_proxy")
    return pd.concat(
        [
            _bucket_summary(same, "source_target_ret_15m_gap", mask, "source_target_gap_15m_after_source_prior_reclaim", cfg),
            _bucket_summary(same, "source_target_ret_1h_gap", mask, "source_target_gap_1h_after_source_prior_reclaim", cfg),
        ],
        ignore_index=True,
    )


def _premium_divergence_summary(same: pd.DataFrame, cfg: V41Config) -> pd.DataFrame:
    if same.empty:
        return pd.DataFrame()
    mask = _bool(same, "target_reclaim_proxy") & _bool(same, "target_not_extended")
    return _bucket_summary(same, "premium_proxy", mask, "binance_bybit_premium_proxy_target_reclaim", cfg)


def _timeframe_coverage(symbols: list[str], cfg: V41Config) -> pd.DataFrame:
    rows = []
    checks = [
        ("source_15m", cfg.data_root / "raw" / cfg.source_exchange / "klines"),
        ("target_15m", cfg.data_root / "raw" / cfg.target_exchange / "klines"),
        ("source_1m", cfg.data_root / "raw" / cfg.source_exchange / "klines_1m_v4"),
        ("target_1m", cfg.data_root / "raw" / cfg.target_exchange / "klines_1m_v4"),
    ]
    for label, path in checks:
        files = {p.stem.upper(): p for p in path.glob("*.parquet")} if path.exists() else {}
        common = [symbol for symbol in symbols if symbol in files]
        rows.append(
            {
                "dataset": label,
                "path": str(path),
                "selected_symbols_available": len(common),
                "selected_symbols_required": len(symbols),
                "status": "ok" if len(common) >= len(symbols) and len(symbols) else "missing_or_incomplete",
            }
        )
    return pd.DataFrame(rows)


def _write_notes(root: Path, coverage: pd.DataFrame, response: pd.DataFrame, incremental: pd.DataFrame, edges: pd.DataFrame) -> None:
    one_min_ready = bool(
        coverage[coverage["dataset"].isin(["source_1m", "target_1m"])]["status"].eq("ok").all()
    )
    lines = [
        "# v4.1 Cross-Exchange Lead-Lag Diagnostics",
        "",
        "## Scope",
        "- First pass uses Top50 common Binance UM -> Bybit 15m klines.",
        (
            "- Focused 1m response rows are evaluated from the v4 sample-month cache."
            if one_min_ready
            else "- 1m/3m/5m response rows are explicitly marked as requiring lower-timeframe data."
        ),
        "",
        "## Coverage",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"- {row.dataset}: {row.selected_symbols_available}/{row.selected_symbols_required} `{row.status}`.")
    lines.append("")
    lines.append("## Findings")
    if len(response):
        available = response[response["status"].isin(["evaluated_15m_proxy", "evaluated_1m_sample"])]
        if len(available):
            best = available.sort_values("net20", ascending=False).head(1).iloc[0]
            lines.append(
                f"- Best evaluated response row: {best.source_event_type} horizon={int(best.target_horizon_minutes)}m "
                f"timeframe={best.timeframe}, net20={best.net20:.4%}, events={int(best.events)}."
            )
    if len(incremental):
        best_inc = incremental.sort_values("incremental_lift", ascending=False).head(1).iloc[0]
        lines.append(
            f"- Best source-vs-target incremental row: {best_inc.candidate}, "
            f"lift={best_inc.incremental_lift:.4%}, net20={best_inc.source_plus_target_net20:.4%}."
        )
    if len(edges):
        best_edge = edges.head(1).iloc[0]
        lines.append(
            f"- Best edge matrix row: {best_edge.source_symbol}->{best_edge.target_symbol} "
            f"h={int(best_edge.horizon_minutes)}m, net20={best_edge.response_net20:.4%}, events={int(best_edge.events)}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "- No v4.1 diagnostic is promoted to paper-live.",
            "- Same-symbol and broad cross-symbol lead-lag remain unproven unless source+target beats target-only and random/shuffled controls.",
            (
                "- Lower-timeframe sample coverage is available; next blocker is target-only / random / shuffled attribution on 1m motifs."
                if one_min_ready
                else "- Next data blocker: populate `data/raw/binance/klines_1m_v4` and `data/raw/bybit/klines_1m_v4` for selected months before judging sub-15m propagation."
            ),
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v41_cross_exchange_lead_lag_diagnostics(cfg: V41Config = V41Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    selected, source, target, same = _prepare(cfg)
    coverage = _timeframe_coverage(selected, cfg)
    response_15m = _response_curve(same, cfg)
    response_1m = _response_curve_1m_sample(selected, cfg)
    response = pd.concat([response_15m, response_1m], ignore_index=True) if len(response_1m) else response_15m
    incremental = _incremental_edge_summary(same, selected, cfg)
    edge_symbols = selected[: cfg.max_edge_symbols]
    edges = _lead_lag_edge_matrix(source, target, edge_symbols, cfg)
    gap = _gap_bucket_summary(same, cfg)
    premium = _premium_divergence_summary(same, cfg)

    outputs = {
        "timeframe_coverage_audit": root / "timeframe_coverage_audit.csv",
        "response_curve": root / "response_curve.csv",
        "incremental_edge_vs_target_baseline": root / "incremental_edge_vs_target_baseline.csv",
        "lead_lag_edge_matrix": root / "lead_lag_edge_matrix.csv",
        "source_target_gap_bucket_summary": root / "source_target_gap_bucket_summary.csv",
        "premium_divergence_summary": root / "premium_divergence_summary.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["timeframe_coverage_audit"], index=False)
    response.to_csv(outputs["response_curve"], index=False)
    incremental.to_csv(outputs["incremental_edge_vs_target_baseline"], index=False)
    edges.to_csv(outputs["lead_lag_edge_matrix"], index=False)
    gap.to_csv(outputs["source_target_gap_bucket_summary"], index=False)
    premium.to_csv(outputs["premium_divergence_summary"], index=False)
    _write_notes(root, coverage, response, incremental, edges)
    return outputs


__all__ = ["V41Config", "write_v41_cross_exchange_lead_lag_diagnostics"]
