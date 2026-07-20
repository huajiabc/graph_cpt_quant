"""Directed taker-flow graph audit at 15-minute decision frequency."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    BTC,
    estimate_v106_betas,
)
from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    load_v132_bybit_klines,
)


METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
MEMBERSHIP_PATH = Path(
    "reports/v13_2_tg1_forward_temporal_extension/"
    "monthly_balanced_membership_extended.csv"
)
REPORT_ROOT = Path("reports/v14_1_directed_taker_flow_graph")
FINDINGS_PATH = Path(
    "docs/v141_directed_taker_flow_graph_findings_2026_07_15.md"
)
CANDIDATES = (
    "TFG1_POSITIVE_FLOW_PROPAGATION",
    "TFG2_NEGATIVE_FLOW_PROPAGATION",
)


@dataclass(frozen=True)
class V141Config:
    metrics_root: Path = METRICS_ROOT
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    lookback_days: int = 30
    minimum_history_days: int = 28
    min_edge_samples: int = 500
    shrinkage_n: int = 500
    leaders_per_follower: int = 3
    flow_lookback_bars: int = 7 * 24 * 4
    flow_min_periods: int = 5 * 24 * 4
    flow_z_threshold: float = 2.0
    min_source_rows: int = 2
    min_bucket_size: int = 3
    max_bucket_size: int = 5
    min_active_leaders: int = 2
    cooldown_minutes: int = 60
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    delayed_signal_bars: int = 96
    seed: int = 20260715


def _period(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if month < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def strict_15m_decision_time(stamps: pd.Series) -> pd.Series:
    """Map a source stamp to the first quarter-hour strictly after it."""
    values = pd.to_datetime(stamps, utc=True, errors="coerce")
    return (values + pd.Timedelta(nanoseconds=1)).dt.ceil("15min")


def load_v141_membership(cfg: V141Config = V141Config()) -> pd.DataFrame:
    membership = pd.read_csv(cfg.membership_path)
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    membership["symbol"] = membership["symbol"].astype(str)
    available = {path.stem for path in cfg.metrics_root.glob("*.parquet")}
    return (
        membership[
            membership["symbol"].ne(BTC)
            & membership["symbol"].isin(available)
        ]
        .dropna(subset=["month_start", "symbol"])
        .drop_duplicates(["month_start", "symbol"], keep="last")
        .sort_values(["month_start", "symbol"])
        .reset_index(drop=True)
    )


def _flow_feature_for_symbol(
    path: Path,
    cfg: V141Config,
) -> pd.DataFrame:
    raw = pd.read_parquet(
        path,
        columns=[
            "create_time",
            "bybit_symbol",
            "sum_taker_long_short_vol_ratio",
        ],
    )
    raw["create_time"] = pd.to_datetime(
        raw["create_time"], utc=True, errors="coerce"
    )
    raw["ratio"] = pd.to_numeric(
        raw["sum_taker_long_short_vol_ratio"], errors="coerce"
    )
    raw = raw.dropna(subset=["create_time", "ratio"])
    raw = raw[raw["ratio"].gt(0)]
    if raw.empty:
        return pd.DataFrame()
    raw["feature_time"] = strict_15m_decision_time(raw["create_time"])
    raw["log_taker_ratio"] = np.log(raw["ratio"])
    grouped = raw.groupby("feature_time", sort=True).agg(
        log_taker_ratio=("log_taker_ratio", "mean"),
        source_rows=("log_taker_ratio", "size"),
    )
    regular_index = pd.date_range(
        grouped.index.min(), grouped.index.max(), freq="15min", tz="UTC"
    )
    grouped = grouped.reindex(regular_index)
    grouped.index.name = "feature_time"
    grouped.loc[
        grouped["source_rows"].fillna(0).lt(cfg.min_source_rows),
        "log_taker_ratio",
    ] = np.nan
    shifted = grouped["log_taker_ratio"].shift(1)
    prior = shifted.rolling(
        cfg.flow_lookback_bars, min_periods=cfg.flow_min_periods
    )
    prior_mean = prior.mean()
    prior_std = prior.std(ddof=1)
    grouped["flow_z_15m"] = (
        (grouped["log_taker_ratio"] - prior_mean) / prior_std
    ).clip(-5.0, 5.0)
    symbol_values = raw["bybit_symbol"].dropna().astype(str)
    symbol = symbol_values.iloc[-1] if not symbol_values.empty else path.stem
    grouped["symbol"] = symbol
    return (
        grouped.reset_index()
        .dropna(subset=["log_taker_ratio"])
        .sort_values("feature_time")
        .reset_index(drop=True)
    )


def load_v141_flow_features(
    membership: pd.DataFrame,
    cfg: V141Config = V141Config(),
) -> pd.DataFrame:
    frames = []
    for symbol in sorted(membership["symbol"].astype(str).unique()):
        path = cfg.metrics_root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = _flow_feature_for_symbol(path, cfg)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["feature_time", "symbol"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def load_v141_price_matrices() -> dict[str, pd.DataFrame]:
    klines = load_v132_bybit_klines()
    close = klines.pivot_table(
        index="bar_close_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    regular_index = pd.date_range(
        close.index.min(), close.index.max(), freq="15min", tz="UTC"
    )
    close = close.reindex(regular_index)
    close.index.name = "feature_time"
    ret_15m = close.pct_change(fill_method=None)
    future_1h = close.shift(-4).div(close).sub(1.0)
    future_4h = close.shift(-16).div(close).sub(1.0)
    hourly_close = close.loc[close.index.minute == 0]
    hourly_return = hourly_close.pct_change(fill_method=None)
    return {
        "close": close,
        "ret_15m": ret_15m,
        "future_1h": future_1h,
        "future_4h": future_4h,
        "hourly_return": hourly_return,
    }


def build_v141_month_edges(
    flow_source: pd.DataFrame,
    residual_target: pd.DataFrame,
    month_start: pd.Timestamp,
    cfg: V141Config,
) -> pd.DataFrame:
    common = sorted(set(flow_source.columns).intersection(residual_target.columns))
    eligible = [
        symbol
        for symbol in common
        if int(flow_source[symbol].notna().sum()) >= cfg.min_edge_samples
        and int(residual_target[symbol].notna().sum()) >= cfg.min_edge_samples
    ]
    if len(eligible) < 4:
        return pd.DataFrame()
    joined = pd.concat(
        {
            "source": flow_source[eligible].clip(-5.0, 5.0),
            "target": residual_target[eligible],
        },
        axis=1,
    ).dropna(how="any")
    if len(joined) < cfg.min_edge_samples:
        return pd.DataFrame()
    source_rank = joined["source"].rank(pct=True, method="average").to_numpy(
        dtype=float
    )
    target_rank = joined["target"].rank(pct=True, method="average").to_numpy(
        dtype=float
    )
    source_std = source_rank.std(axis=0, ddof=1)
    target_std = target_rank.std(axis=0, ddof=1)
    valid = (source_std > 0) & (target_std > 0)
    source_z = np.zeros_like(source_rank)
    target_z = np.zeros_like(target_rank)
    source_z[:, valid] = (
        source_rank[:, valid] - source_rank[:, valid].mean(axis=0)
    ) / source_std[valid]
    target_z[:, valid] = (
        target_rank[:, valid] - target_rank[:, valid].mean(axis=0)
    ) / target_std[valid]
    n = len(joined)
    correlation = source_z.T @ target_z / (n - 1)
    shrinkage = np.sqrt(n / (n + cfg.shrinkage_n))
    rows = []
    for leader_index, leader in enumerate(eligible):
        if not valid[leader_index]:
            continue
        for follower_index, follower in enumerate(eligible):
            if leader == follower or not valid[follower_index]:
                continue
            forward = float(correlation[leader_index, follower_index])
            reverse = float(correlation[follower_index, leader_index])
            advantage = forward - reverse
            if forward <= 0 or advantage <= 0:
                continue
            rows.append(
                {
                    "month_start": month_start,
                    "leader_symbol": leader,
                    "follower_symbol": follower,
                    "sample_n": n,
                    "source_target_spearman": forward,
                    "reverse_spearman": reverse,
                    "direction_advantage": advantage,
                    "edge_weight": advantage * shrinkage,
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges = edges.sort_values(
        ["follower_symbol", "edge_weight"], ascending=[True, False]
    )
    edges["edge_rank"] = (
        edges.groupby("follower_symbol", sort=False).cumcount() + 1
    )
    return edges[edges["edge_rank"].le(cfg.leaders_per_follower)].reset_index(
        drop=True
    )


def _residual_future(
    future: pd.DataFrame,
    betas: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=future.index)
    if BTC not in future.columns:
        return out
    for symbol, beta in betas.items():
        if symbol == BTC or symbol not in future.columns:
            continue
        out[str(symbol)] = future[symbol] - float(beta) * future[BTC]
    return out


def build_v141_graph_and_contexts(
    flow_features: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    cfg: V141Config,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    flow_z = flow_features.pivot_table(
        index="feature_time",
        columns="symbol",
        values="flow_z_15m",
        aggfunc="last",
        observed=True,
    ).sort_index()
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    for raw_month, month_members in membership.groupby("month_start", sort=True):
        month = pd.Timestamp(raw_month)
        symbols = sorted(
            set(month_members["symbol"].astype(str))
            & set(flow_z.columns)
            & set(prices["close"].columns)
        )
        history_start = month - pd.Timedelta(days=cfg.lookback_days)
        history_end = month - pd.Timedelta(hours=1)
        history_index = flow_z.index[
            (flow_z.index >= history_start)
            & (flow_z.index < history_end)
            & (flow_z.index.minute == 0)
        ]
        if len(history_index) < cfg.min_edge_samples:
            continue
        if history_index.max() - history_index.min() < pd.Timedelta(
            days=cfg.minimum_history_days
        ):
            continue
        beta_history = prices["hourly_return"].loc[
            (prices["hourly_return"].index >= history_start)
            & (prices["hourly_return"].index < month)
        ]
        betas = estimate_v106_betas(beta_history)
        symbols = [symbol for symbol in symbols if symbol in betas.index]
        if len(symbols) < 4 or BTC not in prices["future_1h"].columns:
            continue
        historical_future = prices["future_1h"].reindex(history_index)
        historical_residual = _residual_future(historical_future, betas)
        edges = build_v141_month_edges(
            flow_z.reindex(index=history_index, columns=symbols),
            historical_residual.reindex(index=history_index, columns=symbols),
            month,
            cfg,
        )
        if edges.empty:
            continue
        next_month = month + pd.offsets.MonthBegin(1)
        target_index = flow_z.index[
            (flow_z.index >= month) & (flow_z.index < next_month)
        ]
        if target_index.empty:
            continue
        raw_1h = prices["future_1h"].reindex(target_index)
        raw_4h = prices["future_4h"].reindex(target_index)
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "symbols": symbols,
            "flow_z_15m": flow_z.reindex(index=target_index, columns=symbols),
            "ret_15m": prices["ret_15m"].reindex(
                index=target_index, columns=symbols
            ),
            "raw_future_1h": raw_1h.reindex(columns=symbols),
            "residual_future_1h": _residual_future(raw_1h, betas).reindex(
                columns=symbols
            ),
            "raw_future_4h": raw_4h.reindex(columns=symbols),
            "residual_future_4h": _residual_future(raw_4h, betas).reindex(
                columns=symbols
            ),
        }
        edge_frames.append(edges)
    all_edges = (
        pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    )
    return all_edges, contexts


def pressure_matrices(
    context: dict[str, Any],
    edges: pd.DataFrame,
    cfg: V141Config,
    signal_shift_bars: int = 0,
) -> dict[str, dict[str, pd.DataFrame]]:
    symbols: list[str] = context["symbols"]
    flow: pd.DataFrame = context["flow_z_15m"].reindex(columns=symbols)
    trailing_return: pd.DataFrame = context["ret_15m"].reindex(columns=symbols)
    positive_source = flow.sub(cfg.flow_z_threshold).clip(lower=0.0).where(
        trailing_return.gt(0), 0.0
    )
    negative_source = flow.mul(-1.0).sub(cfg.flow_z_threshold).clip(
        lower=0.0
    ).where(trailing_return.lt(0), 0.0)
    if signal_shift_bars:
        positive_source = positive_source.shift(signal_shift_bars)
        negative_source = negative_source.shift(signal_shift_bars)
    source_by_candidate = {
        CANDIDATES[0]: positive_source,
        CANDIDATES[1]: negative_source,
    }
    outputs = {
        candidate: {
            "pressure": pd.DataFrame(0.0, index=flow.index, columns=symbols),
            "active_leaders": pd.DataFrame(0, index=flow.index, columns=symbols),
        }
        for candidate in CANDIDATES
    }
    for follower, group in edges.groupby("follower_symbol", sort=False):
        follower = str(follower)
        if follower not in symbols:
            continue
        local = (
            group[group["leader_symbol"].astype(str).isin(symbols)]
            .sort_values("edge_rank")
            .drop_duplicates("leader_symbol", keep="first")
        )
        if local.empty:
            continue
        leaders = local["leader_symbol"].astype(str).tolist()
        weights = pd.to_numeric(local["edge_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        for candidate, source in source_by_candidate.items():
            values = source[leaders].fillna(0.0).to_numpy(dtype=float)
            active = values > 0
            counts = active.sum(axis=1)
            denominator = np.where(active, weights[None, :], 0.0).sum(axis=1)
            weighted = np.divide(
                np.where(active, values * weights[None, :], 0.0).sum(axis=1),
                denominator,
                out=np.zeros(len(values)),
                where=denominator > 0,
            )
            weighted[counts < cfg.min_active_leaders] = 0.0
            outputs[candidate]["pressure"][follower] = weighted
            outputs[candidate]["active_leaders"][follower] = counts
    return outputs


def build_v141_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V141Config,
    signal_shift_bars: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(minutes=cfg.cooldown_minutes)
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        signals = pressure_matrices(
            context, month_edges, cfg, signal_shift_bars=signal_shift_bars
        )
        for candidate, payload in signals.items():
            direction = 1.0 if candidate == CANDIDATES[0] else -1.0
            pressure = payload["pressure"]
            eligible_times = pressure.index[
                pressure.gt(0).sum(axis=1).ge(cfg.min_bucket_size)
            ]
            for timestamp in eligible_times:
                timestamp = pd.Timestamp(timestamp)
                last = last_by_candidate.get(candidate)
                if last is not None and timestamp - last < cooldown:
                    continue
                selected = (
                    pressure.loc[timestamp]
                    .loc[lambda values: values.gt(0)]
                    .sort_values(ascending=False)
                    .head(cfg.max_bucket_size)
                    .index.tolist()
                )
                raw_1h = direction * pd.to_numeric(
                    context["raw_future_1h"].loc[timestamp, selected],
                    errors="coerce",
                )
                residual_1h = direction * pd.to_numeric(
                    context["residual_future_1h"].loc[timestamp, selected],
                    errors="coerce",
                )
                finite = raw_1h.notna() & residual_1h.notna()
                if int(finite.sum()) < cfg.min_bucket_size:
                    continue
                selected = list(pd.Index(selected)[finite.to_numpy()])
                raw_1h = raw_1h[finite]
                residual_1h = residual_1h[finite]
                raw_4h = direction * pd.to_numeric(
                    context["raw_future_4h"].loc[timestamp, selected],
                    errors="coerce",
                )
                residual_4h = direction * pd.to_numeric(
                    context["residual_future_4h"].loc[timestamp, selected],
                    errors="coerce",
                )
                raw_gross_1h = float(raw_1h.mean())
                residual_gross_1h = float(residual_1h.mean())
                raw_gross_4h = float(raw_4h.mean())
                residual_gross_4h = float(residual_4h.mean())
                rows.append(
                    {
                        "candidate": candidate,
                        "feature_time": timestamp,
                        "entry_day": timestamp.strftime("%Y-%m-%d"),
                        "entry_month": timestamp.strftime("%Y-%m"),
                        "period": context["period"],
                        "direction": direction,
                        "bucket_size": len(selected),
                        "follower_symbols": "|".join(selected),
                        "mean_flow_pressure": float(
                            pressure.loc[timestamp, selected].mean()
                        ),
                        "raw_gross_1h": raw_gross_1h,
                        "raw_net_1h_20bp": raw_gross_1h - 0.002,
                        "raw_net_1h_30bp": raw_gross_1h - 0.003,
                        "residual_gross_1h": residual_gross_1h,
                        "residual_net_1h_40bp": residual_gross_1h - 0.004,
                        "raw_gross_4h": raw_gross_4h,
                        "raw_net_4h_20bp": raw_gross_4h - 0.002,
                        "residual_gross_4h": residual_gross_4h,
                        "residual_net_4h_40bp": residual_gross_4h - 0.004,
                    }
                )
                last_by_candidate[candidate] = timestamp
    return pd.DataFrame(rows)


def summarize_v141(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = (
            portfolios if scope == "all" else portfolios[portfolios["period"].eq(scope)]
        )
        for candidate in CANDIDATES:
            sample = scoped[scoped["candidate"].eq(candidate)]
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "portfolio_observations": len(sample),
                    "active_days": sample["entry_day"].nunique(),
                    "active_months": sample["entry_month"].nunique(),
                    "mean_bucket_size": sample["bucket_size"].mean(),
                    "mean_raw_gross_1h": sample["raw_gross_1h"].mean(),
                    "mean_raw_net_1h_20bp": sample["raw_net_1h_20bp"].mean(),
                    "mean_raw_net_1h_30bp": sample["raw_net_1h_30bp"].mean(),
                    "mean_residual_gross_1h": sample["residual_gross_1h"].mean(),
                    "mean_residual_net_1h_40bp": sample[
                        "residual_net_1h_40bp"
                    ].mean(),
                    "mean_raw_net_4h_20bp": sample["raw_net_4h_20bp"].mean(),
                    "mean_residual_net_4h_40bp": sample[
                        "residual_net_4h_40bp"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def randomize_v141_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V141Config,
) -> pd.DataFrame:
    rows = []
    for raw_month, group in edges.groupby("month_start", sort=True):
        month = pd.Timestamp(raw_month)
        symbols = contexts[month]["symbols"]
        month_code = month.year * 12 + month.month
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month_code)
        for follower, local in group.groupby("follower_symbol", sort=False):
            follower = str(follower)
            choices = [symbol for symbol in symbols if symbol != follower]
            source_rows = local.sort_values("edge_rank")
            take = min(len(source_rows), len(choices))
            leaders = rng.choice(choices, size=take, replace=False)
            for leader, row in zip(
                leaders, source_rows.head(take).itertuples(index=False), strict=True
            ):
                payload = row._asdict()
                payload["leader_symbol"] = str(leader)
                rows.append(payload)
    return pd.DataFrame(rows)


def reverse_v141_edges(edges: pd.DataFrame, cfg: V141Config) -> pd.DataFrame:
    out = edges.copy()
    out[["leader_symbol", "follower_symbol"]] = out[
        ["follower_symbol", "leader_symbol"]
    ].to_numpy()
    out = out.sort_values(
        ["month_start", "follower_symbol", "edge_weight"],
        ascending=[True, True, False],
    ).drop_duplicates(
        ["month_start", "leader_symbol", "follower_symbol"], keep="first"
    )
    out["edge_rank"] = (
        out.groupby(["month_start", "follower_symbol"], sort=False).cumcount() + 1
    )
    return out[out["edge_rank"].le(cfg.leaders_per_follower)].reset_index(
        drop=True
    )


def random_v141_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V141Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v141_edges(edges, contexts, iteration, cfg)
        portfolios = build_v141_portfolios(contexts, randomized, cfg)
        means = {}
        for candidate in CANDIDATES:
            sample = portfolios[portfolios["candidate"].eq(candidate)]
            mean = float(sample["residual_net_1h_40bp"].mean())
            means[candidate] = mean
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "portfolio_observations": len(sample),
                    "mean_residual_net_1h_40bp": mean,
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "portfolio_observations": len(portfolios),
                "mean_residual_net_1h_40bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(
    sample: pd.DataFrame,
    cfg: V141Config,
) -> tuple[float, float]:
    daily = [
        group["residual_net_1h_40bp"].dropna().to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    daily = [values for values in daily if len(values)]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(
            float(np.mean(np.concatenate([daily[index] for index in chosen])))
        )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _positive_share(values: pd.Series) -> float:
    positive = values.clip(lower=0.0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf


def audit_v141(
    real: pd.DataFrame,
    delayed: pd.DataFrame,
    reversed_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V141Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["candidate"].eq("FAMILY_MAX"),
        "mean_residual_net_1h_40bp",
    ].dropna()
    rows = []
    for candidate in CANDIDATES:
        lookup = {
            row.scope: row
            for row in summary[summary["candidate"].eq(candidate)].itertuples(
                index=False
            )
        }
        sample = real[real["candidate"].eq(candidate)]
        delayed_mean = float(
            delayed.loc[
                delayed["candidate"].eq(candidate), "residual_net_1h_40bp"
            ].mean()
        )
        reversed_mean = float(
            reversed_portfolios.loc[
                reversed_portfolios["candidate"].eq(candidate),
                "residual_net_1h_40bp",
            ].mean()
        )
        ci_low, ci_high = _bootstrap_ci(sample, cfg)
        real_mean = float(lookup["all"].mean_residual_net_1h_40bp)
        percentile = float(family.lt(real_mean).mean()) if len(family) else np.nan
        month_share = _positive_share(
            sample.groupby("entry_month")["residual_net_1h_40bp"].sum()
        )
        worst_period = float(
            sample.groupby("period")["residual_net_1h_40bp"].mean().min()
        )
        gates = {
            "full_observations_200": lookup["all"].portfolio_observations >= 200,
            "validation_observations_50": lookup["validation"].portfolio_observations
            >= 50,
            "holdout_observations_50": lookup["holdout"].portfolio_observations
            >= 50,
            "development_residual_net40_positive": lookup[
                "development"
            ].mean_residual_net_1h_40bp
            > 0,
            "validation_residual_net40_positive": lookup[
                "validation"
            ].mean_residual_net_1h_40bp
            > 0,
            "holdout_residual_net40_positive": lookup[
                "holdout"
            ].mean_residual_net_1h_40bp
            > 0,
            "development_raw_net20_positive": lookup[
                "development"
            ].mean_raw_net_1h_20bp
            > 0,
            "validation_raw_net20_positive": lookup[
                "validation"
            ].mean_raw_net_1h_20bp
            > 0,
            "holdout_raw_net20_positive": lookup["holdout"].mean_raw_net_1h_20bp
            > 0,
            "full_raw_net30_positive": lookup["all"].mean_raw_net_1h_30bp > 0,
            "bootstrap_lower_positive": ci_low > 0,
            "random_family_p95": percentile >= 0.95,
            "beats_reversed": real_mean > reversed_mean,
            "beats_delayed": real_mean > delayed_mean,
            "month_share_below_35pct": month_share <= 0.35,
            "worst_period_above_minus40bp": worst_period >= -0.004,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "verdict": "directed_flow_graph_shadow_candidate"
                if eligible
                else "reject_directed_flow_graph_candidate",
                "full_residual_net40": real_mean,
                "development_residual_net40": lookup[
                    "development"
                ].mean_residual_net_1h_40bp,
                "validation_residual_net40": lookup[
                    "validation"
                ].mean_residual_net_1h_40bp,
                "holdout_residual_net40": lookup[
                    "holdout"
                ].mean_residual_net_1h_40bp,
                "full_raw_net20": lookup["all"].mean_raw_net_1h_20bp,
                "full_raw_net30": lookup["all"].mean_raw_net_1h_30bp,
                "delayed_residual_net40": delayed_mean,
                "reversed_residual_net40": reversed_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "max_positive_month_share": month_share,
                "worst_period_mean": worst_period,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        )
    family_verdict = (
        "directed_flow_graph_shadow_candidate"
        if any(row["eligible"] for row in rows)
        else "reject_directed_flow_graph_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def _write_findings(
    path: Path,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    edges: pd.DataFrame,
    portfolios: pd.DataFrame,
) -> None:
    status = audit["family_verdict"].iloc[0]
    lines = [
        "# v14.1 Directed Taker-Flow Graph Findings",
        "",
        f"Verdict: `{status}`.",
        "",
        "The primary endpoint is one-hour BTC-residual return after 40 bp total cost. "
        "Four-hour values are diagnostic only.",
        "",
        "## Candidate audit",
        "",
        audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Primary summary",
        "",
        summary[
            [
                "scope",
                "candidate",
                "portfolio_observations",
                "mean_raw_net_1h_20bp",
                "mean_raw_net_1h_30bp",
                "mean_residual_net_1h_40bp",
                "mean_residual_net_4h_40bp",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Frozen graph months: `{edges['month_start'].nunique()}`; edges: `{len(edges)}`; "
        f"real portfolio observations: `{len(portfolios)}`.",
        "",
        "This retrospective audit grants no PaperLive, leverage, or live-order permission.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v141_directed_taker_flow_graph(
    cfg: V141Config = V141Config(),
) -> dict[str, Path]:
    membership = load_v141_membership(cfg)
    flow = load_v141_flow_features(membership, cfg)
    prices = load_v141_price_matrices()
    edges, contexts = build_v141_graph_and_contexts(
        flow, prices, membership, cfg
    )
    if edges.empty or not contexts:
        raise RuntimeError("v14.1 produced no causal monthly graph contexts")
    real = build_v141_portfolios(contexts, edges, cfg)
    delayed = build_v141_portfolios(
        contexts, edges, cfg, signal_shift_bars=cfg.delayed_signal_bars
    )
    reversed_edges = reverse_v141_edges(edges, cfg)
    reversed_portfolios = build_v141_portfolios(
        contexts, reversed_edges, cfg
    )
    summary = summarize_v141(real)
    controls = random_v141_controls(contexts, edges, cfg)
    audit = audit_v141(
        real, delayed, reversed_portfolios, summary, controls, cfg
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "flow_features": root / "taker_flow_features_15m.parquet",
        "membership": root / "monthly_membership.csv",
        "edges": root / "directed_flow_edges.csv",
        "portfolios": root / "timestamp_bucket_portfolios.parquet",
        "delayed": root / "delayed_signal_portfolios.parquet",
        "reversed": root / "reversed_edge_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    flow.to_parquet(outputs["flow_features"], index=False)
    membership.to_csv(outputs["membership"], index=False)
    edges.to_csv(outputs["edges"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    delayed.to_parquet(outputs["delayed"], index=False)
    reversed_portfolios.to_parquet(outputs["reversed"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "candidate_family": list(CANDIDATES),
                "flow_rows": len(flow),
                "flow_symbols": flow["symbol"].nunique(),
                "graph_months": edges["month_start"].nunique(),
                "edges": len(edges),
                "portfolio_observations": len(real),
                "random_iterations": cfg.random_iterations,
                "bootstrap_iterations": cfg.bootstrap_iterations,
                "family_verdict": audit["family_verdict"].iloc[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(cfg.findings_path, audit, summary, edges, real)
    return outputs
