"""Large-trader inventory absorption, positioning impulse, and propagation."""
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
from pressure_graph.reports.v121_top_trader_community_rotation import (
    BTC,
    _bootstrap_daily,
    _membership,
    _period,
    _weighted_return,
    build_v121_community_nulls,
    causal_rolling_zscore,
)


REPORT_ROOT = Path("reports/v12_2_positioning_absorption_impulse")
METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
CANDIDATES = (
    "AB1_4H_INVENTORY_ABSORPTION",
    "AB2_12H_INVENTORY_ABSORPTION",
    "AB3_12H_OI_CONFIRMED_ABSORPTION",
    "PI1_12H_POSITIONING_IMPULSE",
    "CA1_12H_COMMUNITY_ABSORPTION",
)
DIRECT_CANDIDATES = CANDIDATES[:4]


@dataclass(frozen=True)
class V122Config:
    metrics_root: Path = METRICS_ROOT
    feature_path: Path = FEATURE_PATH
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    start_time: pd.Timestamp = pd.Timestamp("2025-07-01", tz="UTC")
    rolling_hours: int = 30 * 24
    minimum_history_hours: int = 20 * 24
    bucket_size: int = 9
    minimum_cross_section: int = 48
    minimum_filtered_cross_section: int = 24
    minimum_community_coverage: int = 6
    focal_cost: float = 0.004
    stress_cost: float = 0.006
    one_way_cost: float = 0.002
    bootstrap_iterations: int = 2000
    direct_null_iterations: int = 200
    community_null_iterations: int = 100
    seed: int = 20260715


def _exact_lag(series: pd.Series, times: pd.Series, hours: int) -> np.ndarray:
    indexed = pd.Series(series.to_numpy(dtype=float), index=pd.DatetimeIndex(times))
    return indexed.reindex(pd.DatetimeIndex(times) - pd.Timedelta(hours=hours)).to_numpy()


def load_v122_metrics(cfg: V122Config = V122Config()) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "create_time",
        "bybit_symbol",
        "sum_open_interest_value",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    positive_columns = columns[2:]
    for path in sorted(cfg.metrics_root.glob("*.parquet")):
        raw = pd.read_parquet(path, columns=columns)
        raw["create_time"] = pd.to_datetime(
            raw["create_time"], utc=True, errors="coerce"
        )
        raw = raw[raw["create_time"].ge(cfg.start_time)].copy()
        for column in positive_columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw = raw[raw[positive_columns].gt(0).all(axis=1)].copy()
        if raw.empty:
            continue
        raw = raw.sort_values("create_time")
        raw["feature_time"] = (raw["create_time"] + pd.Timedelta(minutes=5)).dt.ceil(
            "h"
        )
        raw["position_vs_crowd"] = np.log(
            raw["sum_toptrader_long_short_ratio"] / raw["count_long_short_ratio"]
        )
        raw["log_oi_value"] = np.log(raw["sum_open_interest_value"])
        raw["log_taker_ratio"] = np.log(raw["sum_taker_long_short_vol_ratio"])
        hourly = (
            raw.groupby("feature_time", sort=True, observed=True)
            .agg(
                bybit_symbol=("bybit_symbol", "last"),
                position_vs_crowd=("position_vs_crowd", "last"),
                log_oi_value=("log_oi_value", "last"),
                taker_log_mean_1h=("log_taker_ratio", "mean"),
                source_rows=("create_time", "size"),
                source_last_time=("create_time", "max"),
            )
            .reset_index()
        )
        hourly = hourly[hourly["source_rows"].ge(10)].copy()
        past_position = _exact_lag(
            hourly["position_vs_crowd"], hourly["feature_time"], 4
        )
        past_oi = _exact_lag(hourly["log_oi_value"], hourly["feature_time"], 4)
        hourly["position_delta_4h"] = hourly["position_vs_crowd"] - past_position
        hourly["oi_delta_4h"] = hourly["log_oi_value"] - past_oi
        for source, target in (
            ("position_delta_4h", "position_impulse_z"),
            ("oi_delta_4h", "oi_impulse_z"),
            ("taker_log_mean_1h", "taker_flow_z"),
        ):
            hourly[target] = causal_rolling_zscore(
                hourly[source], cfg.rolling_hours, cfg.minimum_history_hours
            )
        hourly["absorption_score"] = (
            hourly["position_impulse_z"] - hourly["taker_flow_z"]
        )
        frames.append(
            hourly[
                [
                    "bybit_symbol",
                    "feature_time",
                    "source_rows",
                    "source_last_time",
                    "position_impulse_z",
                    "oi_impulse_z",
                    "taker_flow_z",
                    "absorption_score",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["bybit_symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "bybit_symbol"])
        .reset_index(drop=True)
    )


def load_v122_prices(cfg: V122Config = V122Config()) -> pd.DataFrame:
    symbols = set(_membership(cfg)["symbol"]) | {BTC}
    columns = [
        "symbol",
        "feature_time",
        "ret_1h",
        "future_ret_4h",
        "future_ret_12h",
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
    for column in ("ret_1h", "future_ret_4h", "future_ret_12h"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    return (
        prices.drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _monthly_betas(
    prices: pd.DataFrame, membership: pd.DataFrame
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
        betas = estimate_v106_betas(pivot)
        for symbol in sorted(exact & set(betas.index.astype(str))):
            rows.append(
                {"month_start": month, "symbol": symbol, "btc_beta": betas[symbol]}
            )
    return pd.DataFrame(rows)


def build_v122_aligned_panel(
    metrics: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V122Config = V122Config(),
) -> pd.DataFrame:
    membership = _membership(cfg)
    targets = prices[prices["symbol"].ne(BTC)].copy()
    targets["month_start"] = pd.to_datetime(
        targets["feature_time"].dt.strftime("%Y-%m-01"), utc=True
    )
    btc = prices[prices["symbol"].eq(BTC)][
        ["feature_time", "future_ret_4h", "future_ret_12h"]
    ].rename(
        columns={
            "future_ret_4h": "btc_future_ret_4h",
            "future_ret_12h": "btc_future_ret_12h",
        }
    )
    aligned = metrics.merge(
        targets[
            [
                "symbol",
                "feature_time",
                "month_start",
                "future_ret_4h",
                "future_ret_12h",
            ]
        ],
        left_on=["bybit_symbol", "feature_time"],
        right_on=["symbol", "feature_time"],
        how="inner",
        validate="one_to_one",
    ).merge(
        membership[["month_start", "symbol", "community_id"]],
        on=["month_start", "symbol"],
        how="inner",
        validate="many_to_one",
    )
    aligned = aligned.merge(
        _monthly_betas(prices, membership),
        on=["month_start", "symbol"],
        how="left",
        validate="many_to_one",
    ).merge(btc, on="feature_time", how="left", validate="many_to_one")
    for horizon in (4, 12):
        aligned[f"residual_future_ret_{horizon}h"] = aligned[
            f"future_ret_{horizon}h"
        ] - aligned["btc_beta"] * aligned[f"btc_future_ret_{horizon}h"]
    return aligned.sort_values(["feature_time", "symbol"]).reset_index(drop=True)


def _decision_row(
    local: pd.DataFrame,
    candidate: str,
    score: pd.Series,
    horizon: int,
    cfg: V122Config,
) -> dict[str, Any] | None:
    label = f"future_ret_{horizon}h"
    ranked = local.assign(_score=score).dropna(subset=["_score", label])
    ranked = ranked.sort_values(["_score", "symbol"])
    low = ranked.head(cfg.bucket_size)["symbol"].astype(str).tolist()
    high = ranked.tail(cfg.bucket_size)["symbol"].astype(str).tolist()
    if (
        len(low) < cfg.bucket_size
        or len(high) < cfg.bucket_size
        or set(low) & set(high)
    ):
        return None
    weights = {symbol: 0.5 / len(high) for symbol in high}
    weights.update({symbol: -0.5 / len(low) for symbol in low})
    if not weights:
        return None
    valid = ranked
    return {
        "candidate": candidate,
        "month_start": local["month_start"].iloc[0],
        "feature_time": local["feature_time"].iloc[0],
        "period": _period(pd.Timestamp(local["feature_time"].iloc[0])),
        "horizon_hours": horizon,
        "coverage": len(valid),
        "raw_return": _weighted_return(local, weights, f"future_ret_{horizon}h"),
        "residual_return": _weighted_return(
            local, weights, f"residual_future_ret_{horizon}h"
        ),
        "signed_ic": valid["_score"].corr(
            valid[f"future_ret_{horizon}h"], method="spearman"
        ),
        "high_symbols": "|".join(high),
        "low_symbols": "|".join(low),
        "high_community": None,
        "low_community": None,
        "_weights": weights,
    }


def build_v122_decisions(
    aligned: pd.DataFrame,
    cfg: V122Config = V122Config(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, local in aligned.groupby("feature_time", sort=True, observed=True):
        local = local.drop_duplicates("symbol", keep="last").copy()
        hour = int(local["feature_time"].iloc[0].hour)
        if hour % 4 == 0 and len(local) >= cfg.minimum_cross_section:
            row = _decision_row(
                local,
                CANDIDATES[0],
                local["absorption_score"],
                4,
                cfg,
            )
            if row is not None:
                rows.append(row)
        if hour % 12 != 0 or len(local) < cfg.minimum_cross_section:
            continue
        for candidate, score in (
            (CANDIDATES[1], local["absorption_score"]),
            (CANDIDATES[3], local["position_impulse_z"]),
        ):
            row = _decision_row(local, candidate, score, 12, cfg)
            if row is not None:
                rows.append(row)
        oi_local = local[local["oi_impulse_z"].ge(0)].copy()
        if len(oi_local) >= cfg.minimum_filtered_cross_section:
            row = _decision_row(
                oi_local,
                CANDIDATES[2],
                oi_local["absorption_score"],
                12,
                cfg,
            )
            if row is not None:
                rows.append(row)

        communities = {
            community: group.dropna(
                subset=["absorption_score", "future_ret_12h"]
            )
            for community, group in local.groupby("community_id", observed=True)
        }
        communities = {
            community: group
            for community, group in communities.items()
            if len(group) >= cfg.minimum_community_coverage
        }
        if len(communities) < 2:
            continue
        scores = {
            community: float(group["absorption_score"].median())
            for community, group in communities.items()
        }
        high_community = max(scores, key=lambda value: (scores[value], value))
        low_community = min(scores, key=lambda value: (scores[value], value))
        high = sorted(communities[high_community]["symbol"].astype(str))
        low = sorted(communities[low_community]["symbol"].astype(str))
        weights = {symbol: 0.5 / len(high) for symbol in high}
        weights.update({symbol: -0.5 / len(low) for symbol in low})
        community_returns = pd.Series(
            {
                community: group["future_ret_12h"].mean()
                for community, group in communities.items()
            }
        )
        rows.append(
            {
                "candidate": CANDIDATES[4],
                "month_start": local["month_start"].iloc[0],
                "feature_time": local["feature_time"].iloc[0],
                "period": _period(pd.Timestamp(local["feature_time"].iloc[0])),
                "horizon_hours": 12,
                "coverage": len(local),
                "raw_return": _weighted_return(local, weights, "future_ret_12h"),
                "residual_return": _weighted_return(
                    local, weights, "residual_future_ret_12h"
                ),
                "signed_ic": pd.Series(scores).corr(
                    community_returns, method="spearman"
                ),
                "high_symbols": "|".join(high),
                "low_symbols": "|".join(low),
                "high_community": high_community,
                "low_community": low_community,
                "_weights": weights,
            }
        )
    return pd.DataFrame(rows)


def apply_v122_costs(
    decisions: pd.DataFrame,
    cfg: V122Config = V122Config(),
) -> pd.DataFrame:
    output = decisions.copy()
    output["net_40bp"] = output["raw_return"] - cfg.focal_cost
    output["net_60bp"] = output["raw_return"] - cfg.stress_cost
    output["residual_net_40bp"] = output["residual_return"] - cfg.focal_cost
    output["realized_turnover"] = np.nan
    output["turnover_net_20bp_oneway"] = np.nan
    for _, indices in output.groupby("candidate", sort=True).groups.items():
        ordered = output.loc[indices].sort_values("feature_time")
        previous: dict[str, float] | None = None
        previous_time: pd.Timestamp | None = None
        turnovers: list[tuple[int, float]] = []
        for index, row in ordered.iterrows():
            current = {str(symbol): float(weight) for symbol, weight in row["_weights"].items()}
            timestamp = pd.Timestamp(row["feature_time"])
            horizon = pd.Timedelta(hours=int(row["horizon_hours"]))
            if previous is None:
                turnover = 1.0
            elif timestamp - previous_time > horizon:
                turnover = 2.0
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
            index, turnover = turnovers[-1]
            turnovers[-1] = (index, turnover + 1.0)
        for index, turnover in turnovers:
            output.loc[index, "realized_turnover"] = turnover
            output.loc[index, "turnover_net_20bp_oneway"] = (
                output.loc[index, "raw_return"] - cfg.one_way_cost * turnover
            )
    return output


def _direct_spec(candidate: str) -> tuple[str, int, bool]:
    if candidate == CANDIDATES[0]:
        return "absorption_score", 4, False
    if candidate == CANDIDATES[3]:
        return "position_impulse_z", 12, False
    return "absorption_score", 12, candidate == CANDIDATES[2]


def build_v122_direct_nulls(
    aligned: pd.DataFrame,
    cfg: V122Config = V122Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for candidate in DIRECT_CANDIDATES:
        score_column, horizon, oi_filter = _direct_spec(candidate)
        groups: list[np.ndarray] = []
        scheduled = aligned[aligned["feature_time"].dt.hour.mod(horizon).eq(0)]
        for _, local in scheduled.groupby("feature_time", sort=True, observed=True):
            if oi_filter:
                local = local[local["oi_impulse_z"].ge(0)]
                minimum = cfg.minimum_filtered_cross_section
            else:
                minimum = cfg.minimum_cross_section
            usable = local.dropna(subset=[score_column, f"future_ret_{horizon}h"])
            if len(usable) >= max(minimum, 2 * cfg.bucket_size):
                groups.append(usable[f"future_ret_{horizon}h"].to_numpy(dtype=float))
        for iteration in range(cfg.direct_null_iterations):
            gross = []
            for returns in groups:
                order = rng.permutation(len(returns))
                n = cfg.bucket_size
                gross.append(
                    0.5
                    * (
                        float(returns[order[:n]].mean())
                        - float(returns[order[n : 2 * n]].mean())
                    )
                )
            rows.append(
                {
                    "candidate": candidate,
                    "iteration": iteration,
                    "null_type": "within_timestamp_random_bucket",
                    "mean_net_40bp": float(np.mean(gross) - cfg.focal_cost),
                }
            )
    return pd.DataFrame(rows)


def build_v122_community_nulls(
    aligned: pd.DataFrame,
    cfg: V122Config = V122Config(),
) -> pd.DataFrame:
    community = aligned.loc[
        aligned["feature_time"].dt.hour.mod(12).eq(0),
        [
            "month_start",
            "feature_time",
            "symbol",
            "absorption_score",
            "future_ret_12h",
        ],
    ].rename(
        columns={
            "absorption_score": "divergence_z",
            "future_ret_12h": "future_ret_4h",
        }
    )
    nulls = build_v121_community_nulls(
        community,
        cfg,
        freeze_development_direction=False,
    )
    nulls["candidate"] = CANDIDATES[4]
    return nulls


def summarize_v122(
    decisions: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V122Config = V122Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 3)
    rows = []
    for candidate, local in decisions.groupby("candidate", sort=True):
        ci_low, ci_high = _bootstrap_daily(local, cfg.bootstrap_iterations, rng)
        periods = local.groupby("period", observed=True)["net_40bp"].mean()
        monthly = local.groupby("month_start", observed=True)["net_40bp"].sum()
        positive = monthly[monthly.gt(0)]
        concentration = (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
        )
        candidate_null = nulls.loc[
            nulls["candidate"].eq(candidate), "mean_net_40bp"
        ]
        observed = float(local["net_40bp"].mean())
        null_percentile = float(100 * candidate_null.le(observed).mean())
        minimum_coverage = (
            cfg.minimum_filtered_cross_section
            if candidate == CANDIDATES[2]
            else 60
        )
        row = {
            "candidate": candidate,
            "decisions": len(local),
            "months": int(local["month_start"].nunique()),
            "median_coverage": float(local["coverage"].median()),
            "horizon_hours": int(local["horizon_hours"].iloc[0]),
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
                periods.get("development", np.nan) * 10_000
            ),
            "validation_net_40bp_bp": float(
                periods.get("validation", np.nan) * 10_000
            ),
            "holdout_net_40bp_bp": float(periods.get("holdout", np.nan) * 10_000),
            "bootstrap_95_low_bp": ci_low * 10_000,
            "bootstrap_95_high_bp": ci_high * 10_000,
            "null_percentile": null_percentile,
            "positive_month_concentration": concentration,
            "worst_period_bp": float(periods.min() * 10_000),
            "worst_holding_bp": float(local["net_40bp"].min() * 10_000),
        }
        row["promote"] = bool(
            row["decisions"] >= 500
            and row["months"] >= 10
            and row["median_coverage"] >= minimum_coverage
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


def write_v122_positioning_absorption_impulse(
    cfg: V122Config = V122Config(),
) -> dict[str, Path]:
    metrics = load_v122_metrics(cfg)
    if metrics.empty:
        raise RuntimeError("No v12.2 metrics")
    prices = load_v122_prices(cfg)
    aligned = build_v122_aligned_panel(metrics, prices, cfg)
    decisions = apply_v122_costs(build_v122_decisions(aligned, cfg), cfg)
    nulls = pd.concat(
        [build_v122_direct_nulls(aligned, cfg), build_v122_community_nulls(aligned, cfg)],
        ignore_index=True,
    )
    summary = summarize_v122(decisions, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "decisions": root / "decisions.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v122_positioning_absorption_impulse_findings_2026_07_15.md"
        ),
    }
    decisions.drop(columns="_weights").to_parquet(paths["decisions"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "metrics_rows": len(metrics),
                "metrics_symbols": int(metrics["bybit_symbol"].nunique()),
                "aligned_rows": len(aligned),
                "decisions": len(decisions),
                "promoted": summary.loc[summary["promote"], "candidate"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    verdict = "promote_forward_candidate" if promoted else "reject_all_as_tradable_alpha"
    best = summary.loc[summary["mean_turnover_net_bp"].idxmax()]
    text = "\n".join(
        [
            "# v12.2 Positioning Absorption and Impulse Findings",
            "",
            f"Verdict: `{verdict}`.",
            "",
            f"The causal hourly panel contains {len(metrics):,} rows across "
            f"{metrics['bybit_symbol'].nunique()} symbols; {len(aligned):,} rows joined "
            "the frozen communities and price labels.",
            "",
            summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            f"The best realized-turnover candidate was `{best['candidate']}` at "
            f"{best['mean_turnover_net_bp']:.2f} bp per decision.",
            "",
            "The direction of every candidate was fixed before outcome inspection. "
            "No existing PaperLive strategy was changed.",
            "",
        ]
    )
    paths["findings"].write_text(text, encoding="utf-8")
    return paths
