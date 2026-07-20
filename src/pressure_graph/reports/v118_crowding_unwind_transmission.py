"""Crowding-unwind propagation through frozen price communities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import estimate_v106_betas
from pressure_graph.reports.v108_oi_leader_bucket import load_v108_features


REPORT_ROOT = Path("reports/v11_8_crowding_unwind_transmission")
ACCOUNT_RATIO_ROOT = Path("data/external/orthogonal_volatility/bybit_account_ratio_1h")
MEMBERSHIP_PATH = Path("reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv")
CANDIDATES = ("CU1_CROWDED_LONG_UNWIND_SHORT", "CU2_CROWDED_SHORT_SQUEEZE_LONG")


@dataclass(frozen=True)
class V118Config:
    account_ratio_root: Path = ACCOUNT_RATIO_ROOT
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    rolling_hours: int = 30 * 24
    minimum_history_hours: int = 20 * 24
    crowding_z_threshold: float = 2.0
    return_z_threshold: float = 1.5
    oi_z_threshold: float = -1.0
    minimum_followers: int = 3
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def rolling_v118_zscore(
    series: pd.Series,
    window: int,
    minimum_history: int,
) -> pd.Series:
    history = series.shift(1)
    mean = history.rolling(window, min_periods=minimum_history).mean()
    std = history.rolling(window, min_periods=minimum_history).std(ddof=1)
    return series.sub(mean).div(std.where(std.gt(0))).clip(-5.0, 5.0)


def load_v118_account_ratios(root: Path = ACCOUNT_RATIO_ROOT) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(
            path,
            columns=[
                "symbol",
                "account_ratio_time",
                "long_account_ratio",
                "short_account_ratio",
            ],
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    ratios = pd.concat(frames, ignore_index=True)
    ratios["account_ratio_time"] = pd.to_datetime(
        ratios["account_ratio_time"], utc=True, errors="coerce"
    )
    ratios["long_account_ratio"] = pd.to_numeric(
        ratios["long_account_ratio"], errors="coerce"
    )
    ratios["short_account_ratio"] = pd.to_numeric(
        ratios["short_account_ratio"], errors="coerce"
    )
    ratios = ratios.dropna(
        subset=[
            "symbol",
            "account_ratio_time",
            "long_account_ratio",
            "short_account_ratio",
        ]
    )
    ratios = ratios[
        ratios["long_account_ratio"].gt(0) & ratios["short_account_ratio"].gt(0)
    ].copy()
    ratios["crowding"] = np.log(
        ratios["long_account_ratio"] / ratios["short_account_ratio"]
    )
    return (
        ratios.drop_duplicates(["symbol", "account_ratio_time"], keep="last")
        .sort_values(["symbol", "account_ratio_time"])
        .reset_index(drop=True)
    )


def build_v118_aligned_panel(
    features: pd.DataFrame,
    ratios: pd.DataFrame,
    cfg: V118Config,
) -> pd.DataFrame:
    hourly = features[features["feature_time"].dt.minute.eq(0)].copy()
    hourly = hourly.sort_values(["symbol", "feature_time"]).reset_index(drop=True)
    hourly["return_z"] = hourly.groupby("symbol", group_keys=False)["ret_1h"].apply(
        lambda series: rolling_v118_zscore(
            series, cfg.rolling_hours, cfg.minimum_history_hours
        )
    )
    ratios = ratios.copy()
    ratios["crowding_z"] = ratios.groupby("symbol", group_keys=False)["crowding"].apply(
        lambda series: rolling_v118_zscore(
            series, cfg.rolling_hours, cfg.minimum_history_hours
        )
    )
    ratio_columns = [
        "symbol",
        "account_ratio_time",
        "long_account_ratio",
        "short_account_ratio",
        "crowding",
        "crowding_z",
    ]
    aligned = hourly.merge(
        ratios[ratio_columns],
        left_on=["symbol", "feature_time"],
        right_on=["symbol", "account_ratio_time"],
        how="left",
        validate="one_to_one",
    )
    aligned["account_ratio_age_minutes"] = (
        aligned["feature_time"] - aligned["account_ratio_time"]
    ).dt.total_seconds() / 60.0
    aligned.loc[
        aligned["account_ratio_age_minutes"].lt(0)
        | aligned["account_ratio_age_minutes"].gt(90),
        ["crowding", "crowding_z"],
    ] = np.nan
    aligned["month_start"] = pd.to_datetime(
        aligned["feature_time"].dt.strftime("%Y-%m-01"), utc=True
    )
    return aligned.sort_values(["feature_time", "symbol"]).reset_index(drop=True)


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(
        index="feature_time",
        columns="symbol",
        values=column,
        aggfunc="last",
        observed=True,
    ).sort_index()


def build_v118_contexts(
    panel: pd.DataFrame,
    membership_path: Path = MEMBERSHIP_PATH,
) -> tuple[dict[pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    membership = pd.read_csv(membership_path)
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    coverage_rows = []
    for raw_month in sorted(membership["month_start"].dropna().unique()):
        month = pd.Timestamp(raw_month)
        local_membership = membership[membership["month_start"].eq(month)]
        communities = {
            str(community): sorted(group["symbol"].astype(str).tolist())
            for community, group in local_membership.groupby("community_id", sort=True)
        }
        exact_symbols = set().union(*communities.values()) if communities else set()
        exact_membership = len(communities) == 8 and all(
            len(members) == 9 for members in communities.values()
        )
        history = panel[
            panel["feature_time"].ge(month - pd.Timedelta(days=30))
            & panel["feature_time"].lt(month)
        ]
        target = panel[panel["month_start"].eq(month)]
        if history.empty or target.empty or not exact_membership:
            continue
        history_return = _pivot(history, "ret_1h")
        target_future = _pivot(target, "future_ret_4h")
        target_crowding = _pivot(target, "crowding_z")
        target_return_z = _pivot(target, "return_z")
        target_oi_z = _pivot(target, "oi_value_delta_z_1h")
        if "BTCUSDT" not in history_return.columns or "BTCUSDT" not in target_future.columns:
            continue
        betas = estimate_v106_betas(history_return)
        available = (
            exact_symbols
            & set(target_future.columns)
            & set(target_crowding.columns)
            & set(target_return_z.columns)
            & set(target_oi_z.columns)
            & set(betas.index.astype(str))
        )
        if available != exact_symbols:
            continue
        symbols = sorted(exact_symbols)
        times = (
            target_future.index.intersection(target_crowding.index)
            .intersection(target_return_z.index)
            .intersection(target_oi_z.index)
        )
        raw_future = target_future.reindex(index=times, columns=symbols)
        residual_future = pd.DataFrame(index=times, columns=symbols, dtype=float)
        btc_future = target_future["BTCUSDT"].reindex(times)
        for symbol in symbols:
            residual_future[symbol] = raw_future[symbol] - float(betas[symbol]) * btc_future
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "target_times": times,
            "communities": communities,
            "crowding_z": target_crowding.reindex(index=times, columns=symbols),
            "return_z": target_return_z.reindex(index=times, columns=symbols),
            "oi_z": target_oi_z.reindex(index=times, columns=symbols),
            "raw_future_4h": raw_future,
            "residual_future_4h": residual_future,
            "btc_future_4h": btc_future,
            "betas": {symbol: float(betas[symbol]) for symbol in symbols},
        }
        coverage_rows.append(
            {
                "month_start": month,
                "symbols": len(symbols),
                "communities": len(communities),
                "hours": len(times),
                "crowding_z_coverage": float(
                    target_crowding.reindex(index=times, columns=symbols).notna().mean().mean()
                ),
            }
        )
    return contexts, pd.DataFrame(coverage_rows)


def _candidate_condition(
    context: dict[str, Any], candidate: str, cfg: V118Config, shift_hours: int
) -> pd.DataFrame:
    crowding = context["crowding_z"].shift(shift_hours) if shift_hours else context["crowding_z"]
    return_z = context["return_z"]
    oi_z = context["oi_z"]
    if candidate == CANDIDATES[0]:
        return (
            crowding.ge(cfg.crowding_z_threshold)
            & return_z.le(-cfg.return_z_threshold)
            & oi_z.le(cfg.oi_z_threshold)
        )
    return (
        crowding.le(-cfg.crowding_z_threshold)
        & return_z.ge(cfg.return_z_threshold)
        & oi_z.le(cfg.oi_z_threshold)
    )


def build_v118_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V118Config,
    community_overrides: dict[pd.Timestamp, dict[str, list[str]]] | None = None,
    crowding_shift_hours: int = 0,
) -> pd.DataFrame:
    rows = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        communities = (
            community_overrides.get(month, context["communities"])
            if community_overrides is not None
            else context["communities"]
        )
        raw_future = context["raw_future_4h"]
        residual_future = context["residual_future_4h"]
        for candidate in CANDIDATES:
            condition = _candidate_condition(
                context, candidate, cfg, crowding_shift_hours
            )
            direction = -1.0 if candidate == CANDIDATES[0] else 1.0
            eligible_times = condition.index[condition.any(axis=1)]
            for timestamp in eligible_times:
                last = last_by_candidate.get(candidate)
                if last is not None and pd.Timestamp(timestamp) - last < cooldown:
                    continue
                followers: set[str] = set()
                leaders: set[str] = set()
                triggered_communities = []
                for community_id, members in communities.items():
                    local_leaders = [
                        symbol
                        for symbol in members
                        if symbol in condition.columns and bool(condition.at[timestamp, symbol])
                    ]
                    if not local_leaders:
                        continue
                    triggered_communities.append(community_id)
                    leaders.update(local_leaders)
                    followers.update(set(members) - set(local_leaders))
                followers -= leaders
                selected = sorted(
                    symbol
                    for symbol in followers
                    if symbol in raw_future.columns
                    and pd.notna(raw_future.at[timestamp, symbol])
                    and pd.notna(residual_future.at[timestamp, symbol])
                )
                if len(selected) < cfg.minimum_followers:
                    continue
                raw_gross = direction * float(raw_future.loc[timestamp, selected].mean())
                residual_gross = direction * float(
                    residual_future.loc[timestamp, selected].mean()
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "feature_time": pd.Timestamp(timestamp),
                        "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                        "month_start": month,
                        "period": context["period"],
                        "direction": direction,
                        "leader_count": len(leaders),
                        "follower_count": len(selected),
                        "leader_symbols": "|".join(sorted(leaders)),
                        "follower_symbols": "|".join(selected),
                        "community_ids": "|".join(sorted(triggered_communities)),
                        "community_count": len(triggered_communities),
                        "raw_gross": raw_gross,
                        "raw_net20": raw_gross - 0.0020,
                        "raw_net30": raw_gross - 0.0030,
                        "residual_gross": residual_gross,
                        "residual_net40": residual_gross - 0.0040,
                        "crowding_shift_hours": crowding_shift_hours,
                    }
                )
                last_by_candidate[candidate] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def _random_communities(
    contexts: dict[pd.Timestamp, dict[str, Any]], rng: np.random.Generator, iteration: int
) -> dict[pd.Timestamp, dict[str, list[str]]]:
    overrides = {}
    for month, context in contexts.items():
        symbols = sorted(set().union(*context["communities"].values()))
        shuffled = list(rng.permutation(symbols))
        overrides[month] = {
            f"{month:%Y-%m}:R{iteration:03d}C{index:02d}": shuffled[index * 9 : (index + 1) * 9]
            for index in range(8)
        }
    return overrides


def summarize_v118(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        candidate_events = events[events["candidate"].eq(candidate)]
        for period in ("full", "development", "validation", "holdout"):
            sample = (
                candidate_events
                if period == "full"
                else candidate_events[candidate_events["period"].eq(period)]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "period": period,
                    "observations": len(sample),
                    "raw_gross": float(sample["raw_gross"].mean()) if len(sample) else np.nan,
                    "raw_net20": float(sample["raw_net20"].mean()) if len(sample) else np.nan,
                    "raw_net30": float(sample["raw_net30"].mean()) if len(sample) else np.nan,
                    "residual_net40": float(sample["residual_net40"].mean())
                    if len(sample)
                    else np.nan,
                    "win_rate_net20": float(sample["raw_net20"].gt(0).mean())
                    if len(sample)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _day_bootstrap(
    sample: pd.DataFrame, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    if sample.empty:
        return np.nan, np.nan
    day_values = sample.groupby("entry_day", sort=True)["raw_net20"].mean().to_numpy()
    draws = rng.choice(day_values, size=(iterations, len(day_values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _positive_share(sample: pd.DataFrame, key: str) -> float:
    positive = sample[sample["raw_net20"].gt(0)].copy()
    if positive.empty:
        return np.nan
    contributions: dict[str, float] = {}
    for row in positive.itertuples(index=False):
        keys = [item for item in str(getattr(row, key)).split("|") if item]
        if not keys:
            continue
        amount = float(row.raw_net20) / len(keys)
        for item in keys:
            contributions[item] = contributions.get(item, 0.0) + amount
    total = sum(contributions.values())
    return max(contributions.values()) / total if total > 0 and contributions else np.nan


def audit_v118(
    events: pd.DataFrame,
    shifted: pd.DataFrame,
    random_results: pd.DataFrame,
    cfg: V118Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    controls = []
    decisions = []
    slices = []
    for candidate in CANDIDATES:
        sample = events[events["candidate"].eq(candidate)].copy()
        validation = sample[sample["period"].eq("validation")]
        holdout = sample[sample["period"].eq("holdout")]
        shifted_sample = shifted[shifted["candidate"].eq(candidate)]
        lower, upper = _day_bootstrap(sample, cfg.bootstrap_iterations, rng)
        random_family = random_results[random_results["candidate"].eq(candidate)][
            "raw_net20"
        ].dropna()
        real = float(sample["raw_net20"].mean()) if len(sample) else np.nan
        percentile = (
            float(random_family.le(real).mean() * 100.0) if len(random_family) else np.nan
        )
        month_share = _positive_share(sample, "month_start")
        community_share = _positive_share(sample, "community_ids")
        controls.extend(
            [
                {"candidate": candidate, "control": "bootstrap_low", "value": lower},
                {"candidate": candidate, "control": "bootstrap_high", "value": upper},
                {"candidate": candidate, "control": "random_partition_percentile", "value": percentile},
                {
                    "candidate": candidate,
                    "control": "random_partition_mean",
                    "value": float(random_family.mean()) if len(random_family) else np.nan,
                },
                {
                    "candidate": candidate,
                    "control": "shifted_1d_net20",
                    "value": float(shifted_sample["raw_net20"].mean())
                    if len(shifted_sample)
                    else np.nan,
                },
                {
                    "candidate": candidate,
                    "control": "reversed_direction_net20",
                    "value": float((-sample["raw_gross"] - 0.0020).mean())
                    if len(sample)
                    else np.nan,
                },
                {"candidate": candidate, "control": "max_positive_month_share", "value": month_share},
                {"candidate": candidate, "control": "max_positive_community_share", "value": community_share},
            ]
        )
        if len(sample):
            ordered = sample.sort_values("feature_time").reset_index(drop=True)
            boundaries = np.linspace(0, len(ordered), 6, dtype=int)
            for index in range(5):
                fifth = ordered.iloc[boundaries[index] : boundaries[index + 1]]
                slices.append(
                    {
                        "candidate": candidate,
                        "chronological_fifth": index + 1,
                        "observations": len(fifth),
                        "raw_net20": float(fifth["raw_net20"].mean()),
                    }
                )
        gates = {
            "full_n_at_least_100": len(sample) >= 100,
            "validation_n_at_least_20": len(validation) >= 20,
            "holdout_n_at_least_20": len(holdout) >= 20,
            "full_net20_positive": real > 0,
            "validation_net20_positive": float(validation["raw_net20"].mean()) > 0
            if len(validation)
            else False,
            "holdout_net20_positive": float(holdout["raw_net20"].mean()) > 0
            if len(holdout)
            else False,
            "validation_residual_net40_positive": float(validation["residual_net40"].mean()) > 0
            if len(validation)
            else False,
            "holdout_residual_net40_positive": float(holdout["residual_net40"].mean()) > 0
            if len(holdout)
            else False,
            "bootstrap_low_positive": lower > 0,
            "random_percentile_at_least_95": percentile >= 95,
            "month_share_below_35pct": month_share <= 0.35,
            "community_share_below_35pct": community_share <= 0.35,
        }
        decisions.append(
            {
                "candidate": candidate,
                "verdict": "eligible_paperlive_review" if all(gates.values()) else "reject",
                **gates,
            }
        )
    return pd.DataFrame(controls), pd.DataFrame(slices), pd.DataFrame(decisions)


def write_v118_crowding_unwind_transmission(
    cfg: V118Config = V118Config(),
) -> dict[str, Path]:
    features = load_v108_features()
    ratios = load_v118_account_ratios(cfg.account_ratio_root)
    panel = build_v118_aligned_panel(features, ratios, cfg)
    contexts, coverage = build_v118_contexts(panel, cfg.membership_path)
    events = build_v118_portfolios(contexts, cfg)
    shifted = build_v118_portfolios(contexts, cfg, crowding_shift_hours=24)
    rng = np.random.default_rng(cfg.seed)
    random_rows = []
    for iteration in range(cfg.random_iterations):
        overrides = _random_communities(contexts, rng, iteration)
        random_events = build_v118_portfolios(contexts, cfg, overrides)
        for candidate in CANDIDATES:
            sample = random_events[random_events["candidate"].eq(candidate)]
            random_rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "observations": len(sample),
                    "raw_net20": float(sample["raw_net20"].mean())
                    if len(sample)
                    else np.nan,
                }
            )
    random_results = pd.DataFrame(random_rows)
    summary = summarize_v118(events)
    controls, slices, decision = audit_v118(events, shifted, random_results, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "coverage": root / "coverage.csv",
        "events": root / "events.parquet",
        "shifted": root / "shifted_events.parquet",
        "random": root / "random_partitions.csv",
        "summary": root / "summary.csv",
        "controls": root / "controls.csv",
        "slices": root / "chronological_fifths.csv",
        "decision": root / "decision.csv",
    }
    coverage.to_csv(outputs["coverage"], index=False)
    events.to_parquet(outputs["events"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    random_results.to_csv(outputs["random"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    slices.to_csv(outputs["slices"], index=False)
    decision.to_csv(outputs["decision"], index=False)
    return outputs


__all__ = [
    "CANDIDATES",
    "V118Config",
    "audit_v118",
    "build_v118_aligned_panel",
    "build_v118_contexts",
    "build_v118_portfolios",
    "load_v118_account_ratios",
    "rolling_v118_zscore",
    "summarize_v118",
    "write_v118_crowding_unwind_transmission",
]
