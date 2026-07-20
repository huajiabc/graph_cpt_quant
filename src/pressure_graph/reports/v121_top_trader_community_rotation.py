"""Top-trader divergence, multi-coin buckets, and frozen-community rotation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import estimate_v106_betas


REPORT_ROOT = Path("reports/v12_1_top_trader_community_rotation")
METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
BTC = "BTCUSDT"
CANDIDATES = (
    "TD1_POSITION_VS_CROWD",
    "TD2_POSITION_VS_TOP_ACCOUNTS",
    "TD3_DIVERGENCE_PLUS_TAKER_FLOW",
    "TD4_FROZEN_COMMUNITY_ROTATION",
)
DIRECT_CANDIDATES = CANDIDATES[:3]


@dataclass(frozen=True)
class V121Config:
    metrics_root: Path = METRICS_ROOT
    feature_path: Path = FEATURE_PATH
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    start_time: pd.Timestamp = pd.Timestamp("2025-07-01", tz="UTC")
    rolling_hours: int = 30 * 24
    minimum_history_hours: int = 20 * 24
    direct_bucket_size: int = 9
    minimum_cross_section: int = 48
    minimum_community_coverage: int = 6
    focal_cost: float = 0.004
    stress_cost: float = 0.006
    one_way_cost: float = 0.002
    bootstrap_iterations: int = 2000
    direct_null_iterations: int = 200
    community_null_iterations: int = 100
    seed: int = 20260715


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def causal_rolling_zscore(
    series: pd.Series,
    window: int,
    minimum_history: int,
) -> pd.Series:
    history = series.shift(1)
    mean = history.rolling(window, min_periods=minimum_history).mean()
    std = history.rolling(window, min_periods=minimum_history).std(ddof=1)
    return series.sub(mean).div(std.where(std.gt(0))).clip(-5.0, 5.0)


def load_v121_metrics(cfg: V121Config = V121Config()) -> pd.DataFrame:
    """Load five-minute archives and retain the causally admissible hourly row."""
    frames: list[pd.DataFrame] = []
    columns = [
        "create_time",
        "bybit_symbol",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    for path in sorted(cfg.metrics_root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame["create_time"] = pd.to_datetime(
            frame["create_time"], utc=True, errors="coerce"
        )
        frame = frame[
            frame["create_time"].ge(cfg.start_time)
            & frame["create_time"].dt.minute.eq(55)
        ].copy()
        if frame.empty:
            continue
        for column in columns[2:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        valid = frame[columns[2:]].gt(0).all(axis=1)
        frame = frame[valid].copy()
        frame["feature_time"] = frame["create_time"].dt.floor("h") + pd.Timedelta(
            hours=1
        )
        frame["position_vs_crowd"] = np.log(
            frame["sum_toptrader_long_short_ratio"]
            / frame["count_long_short_ratio"]
        )
        frame["position_vs_top_accounts"] = np.log(
            frame["sum_toptrader_long_short_ratio"]
            / frame["count_toptrader_long_short_ratio"]
        )
        frame["taker_flow"] = np.log(frame["sum_taker_long_short_vol_ratio"])
        frame = frame.sort_values("feature_time").drop_duplicates(
            "feature_time", keep="last"
        )
        for source, target in (
            ("position_vs_crowd", "divergence_z"),
            ("position_vs_top_accounts", "size_bias_z"),
            ("taker_flow", "taker_z"),
        ):
            frame[target] = causal_rolling_zscore(
                frame[source], cfg.rolling_hours, cfg.minimum_history_hours
            )
        frames.append(
            frame[
                [
                    "bybit_symbol",
                    "feature_time",
                    "divergence_z",
                    "size_bias_z",
                    "taker_z",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    return (
        metrics.drop_duplicates(["bybit_symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "bybit_symbol"])
        .reset_index(drop=True)
    )


def _membership(cfg: V121Config) -> pd.DataFrame:
    membership = pd.read_csv(cfg.membership_path)
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    membership["symbol"] = membership["symbol"].astype(str)
    membership["community_id"] = membership["community_id"].astype(str)
    return membership.dropna(subset=["month_start", "symbol"])


def load_v121_prices(cfg: V121Config = V121Config()) -> pd.DataFrame:
    symbols = set(_membership(cfg)["symbol"]) | {BTC}
    columns = [
        "symbol",
        "feature_time",
        "ret_1h",
        "future_ret_4h",
        "warmup_complete",
    ]
    parquet = pq.ParquetFile(cfg.feature_path)
    frames: list[pd.DataFrame] = []
    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(index, columns=columns).to_pandas()
        chunk["feature_time"] = pd.to_datetime(
            chunk["feature_time"], utc=True, errors="coerce"
        )
        chunk = chunk[
            chunk["symbol"].astype(str).isin(symbols)
            & chunk["feature_time"].ge(cfg.start_time)
            & chunk["feature_time"].dt.minute.eq(0)
            & chunk["warmup_complete"].fillna(False).astype(bool)
        ].copy()
        if not chunk.empty:
            frames.append(chunk.drop(columns="warmup_complete"))
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True)
    for column in ("ret_1h", "future_ret_4h"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    return (
        prices.drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def estimate_v121_monthly_betas(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_month in sorted(membership["month_start"].unique()):
        month = pd.Timestamp(raw_month)
        exact = set(membership.loc[membership["month_start"].eq(month), "symbol"])
        history = prices[
            prices["feature_time"].ge(month - pd.Timedelta(days=30))
            & prices["feature_time"].lt(month)
            & prices["symbol"].isin(exact | {BTC})
        ]
        pivot = history.pivot_table(
            index="feature_time",
            columns="symbol",
            values="ret_1h",
            aggfunc="last",
            observed=True,
        )
        if BTC not in pivot.columns:
            continue
        betas = estimate_v106_betas(pivot)
        for symbol in sorted(exact & set(betas.index.astype(str))):
            rows.append(
                {
                    "month_start": month,
                    "symbol": symbol,
                    "btc_beta": float(betas[symbol]),
                }
            )
    return pd.DataFrame(rows)


def build_v121_aligned_panel(
    metrics: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V121Config = V121Config(),
) -> pd.DataFrame:
    membership = _membership(cfg)
    decision_prices = prices[
        prices["feature_time"].dt.hour.mod(4).eq(0)
        & prices["symbol"].ne(BTC)
    ].copy()
    decision_prices["month_start"] = pd.to_datetime(
        decision_prices["feature_time"].dt.strftime("%Y-%m-01"), utc=True
    )
    btc_future = (
        prices[
            prices["symbol"].eq(BTC)
            & prices["feature_time"].dt.hour.mod(4).eq(0)
        ][["feature_time", "future_ret_4h"]]
        .rename(columns={"future_ret_4h": "btc_future_ret_4h"})
        .drop_duplicates("feature_time", keep="last")
    )
    betas = estimate_v121_monthly_betas(prices, membership)
    aligned = metrics.merge(
        decision_prices[
            ["symbol", "feature_time", "month_start", "future_ret_4h"]
        ],
        left_on=["bybit_symbol", "feature_time"],
        right_on=["symbol", "feature_time"],
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.merge(
        membership[["month_start", "symbol", "community_id"]],
        on=["month_start", "symbol"],
        how="inner",
        validate="many_to_one",
    )
    aligned = aligned.merge(
        betas, on=["month_start", "symbol"], how="left", validate="many_to_one"
    ).merge(btc_future, on="feature_time", how="left", validate="many_to_one")
    aligned["residual_future_ret_4h"] = aligned["future_ret_4h"] - (
        aligned["btc_beta"] * aligned["btc_future_ret_4h"]
    )
    return aligned.sort_values(["feature_time", "symbol"]).reset_index(drop=True)


def _weighted_return(
    frame: pd.DataFrame, weights: dict[str, float], column: str
) -> float:
    values = frame.set_index("symbol")[column].reindex(weights)
    if values.isna().any():
        return float("nan")
    return float(sum(weights[symbol] * values[symbol] for symbol in weights))


def _bucket_weights(
    frame: pd.DataFrame, score: pd.Series, bucket_size: int
) -> tuple[dict[str, float], list[str], list[str]]:
    ranked = frame.assign(_score=score).dropna(subset=["_score", "future_ret_4h"])
    ranked = ranked.sort_values(["_score", "symbol"])
    low = ranked.head(bucket_size)["symbol"].astype(str).tolist()
    high = ranked.tail(bucket_size)["symbol"].astype(str).tolist()
    if len(low) < bucket_size or len(high) < bucket_size or set(low) & set(high):
        return {}, [], []
    weights = {symbol: 0.5 / len(high) for symbol in high}
    weights.update({symbol: -0.5 / len(low) for symbol in low})
    return weights, high, low


def build_v121_decisions(
    aligned: pd.DataFrame,
    cfg: V121Config = V121Config(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (month, timestamp), local in aligned.groupby(
        ["month_start", "feature_time"], sort=True, observed=True
    ):
        local = local.drop_duplicates("symbol", keep="last").copy()
        if len(local) < cfg.minimum_cross_section:
            continue
        active_score = (
            local["divergence_z"].rank(pct=True, method="average")
            + local["taker_z"].rank(pct=True, method="average")
        ) / 2.0
        score_map = {
            CANDIDATES[0]: local["divergence_z"],
            CANDIDATES[1]: local["size_bias_z"],
            CANDIDATES[2]: active_score,
        }
        for candidate, score in score_map.items():
            weights, high, low = _bucket_weights(
                local, score, cfg.direct_bucket_size
            )
            if not weights:
                continue
            valid = local.assign(_score=score).dropna(
                subset=["_score", "future_ret_4h"]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "month_start": month,
                    "feature_time": timestamp,
                    "period": _period(pd.Timestamp(timestamp)),
                    "coverage": len(valid),
                    "continuation_raw_return": _weighted_return(
                        local, weights, "future_ret_4h"
                    ),
                    "continuation_residual_return": _weighted_return(
                        local, weights, "residual_future_ret_4h"
                    ),
                    "spearman_ic": valid["_score"].corr(
                        valid["future_ret_4h"], method="spearman"
                    ),
                    "high_symbols": "|".join(high),
                    "low_symbols": "|".join(low),
                    "high_community": None,
                    "low_community": None,
                    "_weights": weights,
                }
            )

        community_groups = {
            community: group.dropna(subset=["divergence_z", "future_ret_4h"])
            for community, group in local.groupby("community_id", observed=True)
        }
        eligible = {
            community: group
            for community, group in community_groups.items()
            if len(group) >= cfg.minimum_community_coverage
        }
        if len(eligible) < 2:
            continue
        scores = {
            community: float(group["divergence_z"].median())
            for community, group in eligible.items()
        }
        low_community = min(scores, key=lambda value: (scores[value], value))
        high_community = max(scores, key=lambda value: (scores[value], value))
        high = sorted(eligible[high_community]["symbol"].astype(str).tolist())
        low = sorted(eligible[low_community]["symbol"].astype(str).tolist())
        weights = {symbol: 0.5 / len(high) for symbol in high}
        weights.update({symbol: -0.5 / len(low) for symbol in low})
        rows.append(
            {
                "candidate": CANDIDATES[3],
                "month_start": month,
                "feature_time": timestamp,
                "period": _period(pd.Timestamp(timestamp)),
                "coverage": len(local),
                "continuation_raw_return": _weighted_return(
                    local, weights, "future_ret_4h"
                ),
                "continuation_residual_return": _weighted_return(
                    local, weights, "residual_future_ret_4h"
                ),
                "spearman_ic": pd.Series(scores).corr(
                    pd.Series(
                        {
                            community: float(group["future_ret_4h"].mean())
                            for community, group in eligible.items()
                        }
                    ),
                    method="spearman",
                ),
                "high_symbols": "|".join(high),
                "low_symbols": "|".join(low),
                "high_community": high_community,
                "low_community": low_community,
                "_weights": weights,
            }
        )
    return pd.DataFrame(rows)


def freeze_v121_directions_and_costs(
    decisions: pd.DataFrame,
    cfg: V121Config = V121Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty:
        return decisions.copy(), pd.DataFrame()
    output = decisions.copy()
    direction_rows = []
    for candidate, local in output.groupby("candidate", sort=True):
        development_mean = float(
            local.loc[
                local["period"].eq("development"), "continuation_raw_return"
            ].mean()
        )
        sign = 1 if development_mean >= 0 else -1
        direction_rows.append(
            {
                "candidate": candidate,
                "development_continuation_mean": development_mean,
                "chosen_sign": sign,
                "chosen_direction": "continuation" if sign == 1 else "reversal",
            }
        )
        mask = output["candidate"].eq(candidate)
        output.loc[mask, "chosen_sign"] = sign
        output.loc[mask, "raw_return"] = (
            sign * output.loc[mask, "continuation_raw_return"]
        )
        output.loc[mask, "residual_return"] = (
            sign * output.loc[mask, "continuation_residual_return"]
        )
        output.loc[mask, "signed_ic"] = sign * output.loc[mask, "spearman_ic"]

    output["net_40bp"] = output["raw_return"] - cfg.focal_cost
    output["net_60bp"] = output["raw_return"] - cfg.stress_cost
    output["residual_net_40bp"] = output["residual_return"] - cfg.focal_cost
    output["realized_turnover"] = np.nan
    output["turnover_net_20bp_oneway"] = np.nan
    for candidate, indices in output.groupby("candidate", sort=True).groups.items():
        ordered = output.loc[indices].sort_values("feature_time")
        previous: dict[str, float] | None = None
        previous_time: pd.Timestamp | None = None
        turnovers: list[tuple[int, float]] = []
        for index, row in ordered.iterrows():
            sign = int(row["chosen_sign"])
            current = {
                symbol: sign * float(weight)
                for symbol, weight in row["_weights"].items()
            }
            timestamp = pd.Timestamp(row["feature_time"])
            if previous is None:
                turnover = sum(abs(value) for value in current.values())
            elif timestamp - previous_time > pd.Timedelta(hours=4):
                turnover = sum(abs(value) for value in previous.values()) + sum(
                    abs(value) for value in current.values()
                )
            else:
                symbols = set(previous) | set(current)
                turnover = sum(
                    abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
                    for symbol in symbols
                )
            turnovers.append((index, turnover))
            previous = current
            previous_time = timestamp
        if turnovers:
            last_index, last_turnover = turnovers[-1]
            turnovers[-1] = (last_index, last_turnover + 1.0)
        for index, turnover in turnovers:
            output.loc[index, "realized_turnover"] = turnover
            output.loc[index, "turnover_net_20bp_oneway"] = (
                output.loc[index, "raw_return"] - cfg.one_way_cost * turnover
            )
    return output, pd.DataFrame(direction_rows)


def _bootstrap_daily(
    local: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    daily = local.assign(day=local["feature_time"].dt.floor("D")).groupby(
        "day", observed=True
    )["net_40bp"].mean()
    values = daily.to_numpy(dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def build_v121_direct_nulls(
    aligned: pd.DataFrame,
    cfg: V121Config = V121Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    groups: dict[str, list[tuple[str, np.ndarray]]] = {
        candidate: [] for candidate in DIRECT_CANDIDATES
    }
    for timestamp, local in aligned.groupby("feature_time", sort=True):
        if len(local) < cfg.minimum_cross_section:
            continue
        active = (
            local["divergence_z"].rank(pct=True, method="average")
            + local["taker_z"].rank(pct=True, method="average")
        ) / 2.0
        scores = {
            CANDIDATES[0]: local["divergence_z"],
            CANDIDATES[1]: local["size_bias_z"],
            CANDIDATES[2]: active,
        }
        for candidate, score in scores.items():
            usable = local.assign(_score=score).dropna(
                subset=["_score", "future_ret_4h"]
            )
            if len(usable) >= 2 * cfg.direct_bucket_size:
                groups[candidate].append(
                    (
                        _period(pd.Timestamp(timestamp)),
                        usable["future_ret_4h"].to_numpy(dtype=float),
                    )
                )
    rows = []
    for candidate, candidate_groups in groups.items():
        for iteration in range(cfg.direct_null_iterations):
            sampled = []
            for period, returns in candidate_groups:
                order = rng.permutation(len(returns))
                n = cfg.direct_bucket_size
                gross = 0.5 * (
                    float(returns[order[:n]].mean())
                    - float(returns[order[n : 2 * n]].mean())
                )
                sampled.append((period, gross))
            frame = pd.DataFrame(sampled, columns=["period", "gross"])
            development_mean = frame.loc[
                frame["period"].eq("development"), "gross"
            ].mean()
            sign = 1 if development_mean >= 0 else -1
            rows.append(
                {
                    "candidate": candidate,
                    "iteration": iteration,
                    "null_type": "within_timestamp_random_bucket",
                    "mean_net_40bp": float(sign * frame["gross"].mean() - cfg.focal_cost),
                }
            )
    return pd.DataFrame(rows)


def build_v121_community_nulls(
    aligned: pd.DataFrame,
    cfg: V121Config = V121Config(),
    *,
    freeze_development_direction: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)
    rows = []
    month_inputs = []
    membership = _membership(cfg)
    for raw_month, member_group in membership.groupby(
        "month_start", sort=True, observed=True
    ):
        month = pd.Timestamp(raw_month)
        symbols = sorted(member_group["symbol"].astype(str).unique())
        local = aligned[aligned["month_start"].eq(month)]
        if local.empty:
            continue
        divergence = local.pivot_table(
            index="feature_time",
            columns="symbol",
            values="divergence_z",
            aggfunc="last",
            observed=True,
        ).reindex(columns=symbols)
        returns = local.pivot_table(
            index="feature_time",
            columns="symbol",
            values="future_ret_4h",
            aggfunc="last",
            observed=True,
        ).reindex(index=divergence.index, columns=symbols)
        month_inputs.append(
            (
                symbols,
                divergence.to_numpy(dtype=float),
                returns.to_numpy(dtype=float),
                np.asarray(
                    [_period(pd.Timestamp(timestamp)) for timestamp in divergence.index],
                    dtype=object,
                ),
            )
        )
    for iteration in range(cfg.community_null_iterations):
        sampled_gross: list[np.ndarray] = []
        sampled_period: list[np.ndarray] = []
        for symbols, divergence, returns, periods in month_inputs:
            permutation = rng.permutation(len(symbols))
            assignment = np.empty(len(symbols), dtype=np.int8)
            assignment[permutation] = np.arange(len(symbols), dtype=np.int16) // 9
            community_scores = np.full((len(divergence), 8), np.nan, dtype=float)
            community_returns = np.full((len(divergence), 8), np.nan, dtype=float)
            community_eligible = np.zeros((len(divergence), 8), dtype=bool)
            for community in range(8):
                member_mask = assignment == community
                local_divergence = divergence[:, member_mask]
                local_returns = returns[:, member_mask]
                valid = np.isfinite(local_divergence) & np.isfinite(local_returns)
                counts = valid.sum(axis=1)
                eligible = counts >= cfg.minimum_community_coverage
                community_eligible[:, community] = eligible
                if not eligible.any():
                    continue
                masked_divergence = np.where(valid, local_divergence, np.nan)
                community_scores[eligible, community] = np.nanmedian(
                    masked_divergence[eligible], axis=1
                )
                community_returns[eligible, community] = (
                    np.where(valid[eligible], local_returns[eligible], 0.0).sum(axis=1)
                    / counts[eligible]
                )
            usable = community_eligible.sum(axis=1) >= 2
            if not usable.any():
                continue
            high = np.argmax(
                np.where(community_eligible[usable], community_scores[usable], -np.inf),
                axis=1,
            )
            low = np.argmin(
                np.where(community_eligible[usable], community_scores[usable], np.inf),
                axis=1,
            )
            row_indices = np.arange(int(usable.sum()))
            gross = 0.5 * (
                community_returns[usable][row_indices, high]
                - community_returns[usable][row_indices, low]
            )
            sampled_gross.append(gross)
            sampled_period.append(periods[usable])
        gross = np.concatenate(sampled_gross)
        periods = np.concatenate(sampled_period)
        development_mean = float(gross[periods == "development"].mean())
        sign = (
            1 if development_mean >= 0 else -1
        ) if freeze_development_direction else 1
        rows.append(
            {
                "candidate": CANDIDATES[3],
                "iteration": iteration,
                "null_type": "random_monthly_communities",
                "mean_net_40bp": float(sign * gross.mean() - cfg.focal_cost),
            }
        )
    return pd.DataFrame(rows)


def summarize_v121(
    decisions: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V121Config = V121Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 3)
    rows = []
    for candidate, local in decisions.groupby("candidate", sort=True):
        ci_low, ci_high = _bootstrap_daily(local, cfg.bootstrap_iterations, rng)
        period_means = local.groupby("period", observed=True)["net_40bp"].mean()
        monthly = local.groupby("month_start", observed=True)["net_40bp"].sum()
        positive_months = monthly[monthly.gt(0)]
        concentration = (
            float(positive_months.max() / positive_months.sum())
            if positive_months.sum() > 0
            else float("nan")
        )
        candidate_null = nulls.loc[
            nulls["candidate"].eq(candidate), "mean_net_40bp"
        ]
        observed = float(local["net_40bp"].mean())
        null_percentile = (
            float(100.0 * candidate_null.le(observed).mean())
            if len(candidate_null)
            else float("nan")
        )
        values = local["net_40bp"].fillna(0.0).to_numpy(dtype=float)
        equity = np.cumprod(1.0 + values)
        peak = np.maximum.accumulate(np.r_[1.0, equity])
        drawdown = np.r_[1.0, equity] / peak - 1.0
        row = {
            "candidate": candidate,
            "decisions": len(local),
            "months": int(local["month_start"].nunique()),
            "median_coverage": float(local["coverage"].median()),
            "chosen_direction": (
                "continuation" if int(local["chosen_sign"].iloc[0]) == 1 else "reversal"
            ),
            "mean_gross_bp": float(local["raw_return"].mean() * 10_000),
            "mean_net_40bp_bp": observed * 10_000,
            "mean_net_60bp_bp": float(local["net_60bp"].mean() * 10_000),
            "mean_turnover_net_bp": float(
                local["turnover_net_20bp_oneway"].mean() * 10_000
            ),
            "mean_residual_net_40bp_bp": float(
                local["residual_net_40bp"].mean() * 10_000
            ),
            "mean_signed_ic": float(local["signed_ic"].mean()),
            "development_net_40bp_bp": float(
                period_means.get("development", np.nan) * 10_000
            ),
            "validation_net_40bp_bp": float(
                period_means.get("validation", np.nan) * 10_000
            ),
            "holdout_net_40bp_bp": float(
                period_means.get("holdout", np.nan) * 10_000
            ),
            "bootstrap_95_low_bp": ci_low * 10_000,
            "bootstrap_95_high_bp": ci_high * 10_000,
            "null_percentile": null_percentile,
            "positive_month_concentration": concentration,
            "worst_period_bp": float(period_means.min() * 10_000),
            "max_drawdown": float(drawdown.min()),
        }
        row["promote"] = bool(
            row["decisions"] >= 500
            and row["months"] >= 10
            and row["median_coverage"] >= 60
            and all(
                row[key] > 0
                for key in (
                    "development_net_40bp_bp",
                    "validation_net_40bp_bp",
                    "holdout_net_40bp_bp",
                    "mean_net_60bp_bp",
                    "mean_residual_net_40bp_bp",
                    "bootstrap_95_low_bp",
                )
            )
            and row["null_percentile"] >= 90
            and row["positive_month_concentration"] <= 0.35
            and row["worst_period_bp"] >= -40
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _findings_markdown(
    summary: pd.DataFrame,
    directions: pd.DataFrame,
    metrics: pd.DataFrame,
    aligned: pd.DataFrame,
) -> str:
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    verdict = "promote_forward_shadow_review" if promoted else "reject_all_as_tradable_alpha"
    best_turnover = summary.loc[summary["mean_turnover_net_bp"].idxmax()]
    td1 = summary.loc[summary["candidate"].eq(CANDIDATES[0])].iloc[0]
    td4 = summary.loc[summary["candidate"].eq(CANDIDATES[3])].iloc[0]
    lines = [
        "# v12.1 Top-Trader Divergence and Community Rotation Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"The Binance metrics panel contains {len(metrics):,} admissible hourly rows across "
        f"{metrics['bybit_symbol'].nunique()} symbols. After joining frozen memberships, "
        f"prices, and BTC betas, {len(aligned):,} symbol-time observations remained.",
        "",
        "## Candidate results",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Development-only direction freeze",
        "",
        directions.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        f"- The best realized-turnover result was `{best_turnover['candidate']}` at "
        f"{best_turnover['mean_turnover_net_bp']:.2f} bp per decision. Even the low-turnover "
        "accounting remained negative; the conservative full-replacement result was much worse.",
        f"- `{CANDIDATES[0]}` moved from {td1['development_net_40bp_bp']:.2f} bp net in "
        f"development to {td1['validation_net_40bp_bp']:.2f} bp in validation. Development-only "
        "direction selection therefore did not transfer chronologically.",
        f"- The frozen-community candidate reached only the {td4['null_percentile']:.1f}th "
        "percentile of random nine-symbol partitions. Price-community topology added no "
        "positioning-flow attribution.",
        "- BTC residualization did not rescue any family. The small gross effects were not "
        "hidden market beta, but they were far below executable costs and unstable by period.",
        "",
        "The large-trader metrics remain useful state variables, but this preregistered level-based "
        "rotation family is not a tradable alpha. A distinct follow-up must test positioning "
        "impulses or absorption, not tune these bucket sizes or costs after the fact.",
        "",
        "No existing PaperLive strategy was changed.",
    ]
    return "\n".join(lines) + "\n"


def write_v121_top_trader_community_rotation(
    cfg: V121Config = V121Config(),
) -> dict[str, Path]:
    metrics = load_v121_metrics(cfg)
    if metrics.empty:
        raise RuntimeError(f"No Binance metrics found under {cfg.metrics_root}")
    prices = load_v121_prices(cfg)
    if prices.empty:
        raise RuntimeError(f"No price features found at {cfg.feature_path}")
    aligned = build_v121_aligned_panel(metrics, prices, cfg)
    decisions = build_v121_decisions(aligned, cfg)
    decisions, directions = freeze_v121_directions_and_costs(decisions, cfg)
    direct_nulls = build_v121_direct_nulls(aligned, cfg)
    community_nulls = build_v121_community_nulls(aligned, cfg)
    nulls = pd.concat([direct_nulls, community_nulls], ignore_index=True)
    summary = summarize_v121(decisions, nulls, cfg)

    root = ensure_dir(cfg.report_root)
    paths = {
        "decisions": root / "decisions.parquet",
        "directions": root / "directions.csv",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "findings": Path(
            "docs/v121_top_trader_community_rotation_findings_2026_07_15.md"
        ),
        "metadata": root / "metadata.json",
    }
    decisions.drop(columns="_weights").to_parquet(paths["decisions"], index=False)
    directions.to_csv(paths["directions"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["findings"].write_text(
        _findings_markdown(summary, directions, metrics, aligned), encoding="utf-8"
    )
    paths["metadata"].write_text(
        json.dumps(
            {
                "metrics_rows": len(metrics),
                "metrics_symbols": int(metrics["bybit_symbol"].nunique()),
                "aligned_rows": len(aligned),
                "aligned_symbols": int(aligned["symbol"].nunique()),
                "decisions": len(decisions),
                "promoted": summary.loc[summary["promote"], "candidate"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
