"""Weekly graph-relative premium-index value portfolios."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    FUNDING_ROOTS,
    REPORT_ROOT as V195_REPORT_ROOT,
    V195FeatureConfig,
    _community_target,
    _global_target,
    _neutralize_weights,
    _turnover,
    load_v195_funding,
)


REPORT_ROOT = Path("reports/v19_6_graph_premium_relative_value_weekly")
FINDINGS_PATH = Path(
    "docs/v196_graph_premium_relative_value_weekly_findings_2026_07_17.md"
)
FEATURE_PANEL_PATH = V195_REPORT_ROOT / "daily_symbol_feature_panel.parquet"
FSS3_PATH = Path("reports/v14_9_funding_sign_turnover_cap/weekly_portfolio.parquet")
GLOBAL_CANDIDATE = "GPRV1_GLOBAL_GRAPH_PEER_PREMIUM_WEEKLY"
COMMUNITY_CANDIDATE = "GPRV2_COMMUNITY_GRAPH_PEER_PREMIUM_WEEKLY"
CANDIDATES = (GLOBAL_CANDIDATE, COMMUNITY_CANDIDATE)


@dataclass(frozen=True)
class V196Config(V195FeatureConfig):
    feature_panel_path: Path = FEATURE_PANEL_PATH
    fss3_path: Path = FSS3_PATH
    holding_days: int = 7
    one_way_cost: float = 0.0020
    stress_one_way_cost: float = 0.0040
    null_iterations: int = 500
    bootstrap_iterations: int = 2_000
    bootstrap_block_weeks: int = 4
    seed: int = 19_600


def _funding_interval_lookup(
    funding: pd.DataFrame,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[tuple[pd.Timestamp, pd.Timestamp, str], float]:
    lookup: dict[tuple[pd.Timestamp, pd.Timestamp, str], float] = {}
    for symbol, local in funding.groupby("symbol", sort=True):
        time = pd.DatetimeIndex(local["funding_time"])
        values = local["funding_rate_settled"].to_numpy(dtype=float)
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        raw = time.view("int64")
        for entry, exit_time in intervals:
            left = int(np.searchsorted(raw, entry.value, side="right"))
            right = int(np.searchsorted(raw, exit_time.value, side="right"))
            lookup[(entry, exit_time, str(symbol))] = float(
                cumulative[right] - cumulative[left]
            )
    return lookup


def _portfolio_components(
    local: pd.DataFrame,
    weights: dict[str, float],
    price_return: pd.Series,
    future_funding: pd.Series,
) -> dict[str, object]:
    indexed = local.drop_duplicates("symbol").set_index("symbol")
    price_contributions: dict[str, float] = {}
    funding_contributions: dict[str, float] = {}
    residual_contributions: dict[str, float] = {}
    btc_return = float(price_return[BTC])
    for symbol, weight in weights.items():
        price_contributions[symbol] = float(weight * price_return[symbol])
        funding_contributions[symbol] = float(-weight * future_funding[symbol])
        if symbol == BTC:
            residual_contributions[symbol] = float(weight * btc_return)
        else:
            beta = float(indexed.at[symbol, "btc_beta"])
            residual_contributions[symbol] = float(
                weight * (price_return[symbol] - beta * btc_return)
            )
    price_value = float(sum(price_contributions.values()))
    funding_value = float(sum(funding_contributions.values()))
    residual_price = float(
        sum(value for symbol, value in residual_contributions.items() if symbol != BTC)
    )
    total_contributions = {
        symbol: price_contributions[symbol] + funding_contributions[symbol]
        for symbol in weights
    }
    long_gross = float(
        sum(total_contributions[symbol] for symbol, weight in weights.items()
            if symbol != BTC and weight > 0)
    )
    short_gross = float(
        sum(total_contributions[symbol] for symbol, weight in weights.items()
            if symbol != BTC and weight < 0)
    )
    btc_gross = float(total_contributions.get(BTC, 0.0))
    return {
        "price_return": price_value,
        "funding_return": funding_value,
        "gross_return": price_value + funding_value,
        "residual_gross_return": residual_price + funding_value,
        "long_gross_return": long_gross,
        "short_gross_return": short_gross,
        "btc_gross_return": btc_gross,
        "symbol_contributions": total_contributions,
    }


def _target_for_family(
    local: pd.DataFrame,
    family: str,
    cfg: V196Config,
) -> tuple[dict[str, float], list[str], list[str]]:
    if family == GLOBAL_CANDIDATE:
        return _global_target(local, "peer_premium_z", cfg)
    if family == COMMUNITY_CANDIDATE:
        return _community_target(local, cfg)
    if family == "GLOBAL_OWN_PREMIUM_CONTROL":
        return _global_target(local, "premium_z", cfg)
    if family == "GLOBAL_FUNDING_ORTHOGONAL_DIAGNOSTIC":
        return _global_target(local, "funding_orthogonal_premium_z", cfg)
    raise ValueError(f"unknown family: {family}")


def _build_target_map(
    feature_panel: pd.DataFrame,
    families: tuple[str, ...],
    cfg: V196Config,
) -> dict[str, dict[pd.Timestamp, dict[str, float]]]:
    maps = {family: {} for family in families}
    for entry, local in feature_panel.groupby("entry_time", sort=True):
        entry = pd.Timestamp(entry)
        if entry.weekday() != 0 or entry.hour != 0 or entry.minute != 0:
            continue
        for family in families:
            weights, _, _ = _target_for_family(local, family, cfg)
            if weights:
                maps[family][entry] = weights
    return maps


def _build_path_from_targets(
    candidate: str,
    target_map: dict[pd.Timestamp, dict[str, float]],
    feature_by_entry: dict[pd.Timestamp, pd.DataFrame],
    close: pd.DataFrame,
    funding_lookup: dict[tuple[pd.Timestamp, pd.Timestamp, str], float],
    cfg: V196Config,
    entry_to_target_time: dict[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for entry in sorted(feature_by_entry):
        target_time = entry if entry_to_target_time is None else entry_to_target_time.get(entry)
        if target_time is None or target_time not in target_map:
            continue
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        weights = target_map[target_time]
        symbols = list(weights)
        if entry not in close.index or exit_time not in close.index:
            continue
        prices = close.loc[[entry, exit_time], symbols]
        if prices.isna().any().any():
            continue
        price_return = prices.loc[exit_time] / prices.loc[entry] - 1.0
        future_funding = pd.Series(
            {
                symbol: funding_lookup.get((entry, exit_time, symbol), np.nan)
                for symbol in symbols
            },
            dtype=float,
        )
        if future_funding.isna().any():
            continue
        local = feature_by_entry[entry]
        if any(symbol != BTC and symbol not in set(local["symbol"]) for symbol in symbols):
            continue
        components = _portfolio_components(
            local, weights, price_return, future_funding
        )
        turnover = _turnover(previous, weights)
        rows.append(
            {
                "candidate": candidate,
                "entry_time": entry,
                "exit_time": exit_time,
                "target_feature_time": target_time,
                "period": str(local["period"].iloc[0]),
                "entry_month": entry.strftime("%Y-%m"),
                "coverage": len(local),
                "realized_turnover": turnover,
                "residual_btc_beta": float(
                    sum(
                        weight
                        * (
                            1.0
                            if symbol == BTC
                            else float(
                                local.set_index("symbol")["btc_beta"].get(symbol, np.nan)
                            )
                        )
                        for symbol, weight in weights.items()
                    )
                ),
                "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                "weights": weights,
                **components,
            }
        )
        previous = weights
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output.loc[output.index[-1], "realized_turnover"] += sum(
        abs(weight) for weight in previous.values()
    )
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"]
        - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    output["reversed_primary_net_return"] = (
        -output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    return output


def build_v196_paths(
    feature_panel: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: V196Config = V196Config(),
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[pd.Timestamp, dict[str, float]]],
]:
    feature_panel = feature_panel.copy()
    feature_panel["entry_time"] = pd.to_datetime(feature_panel["entry_time"], utc=True)
    feature_by_entry = {
        pd.Timestamp(entry): local.copy()
        for entry, local in feature_panel.groupby("entry_time", sort=True)
        if pd.Timestamp(entry).weekday() == 0
    }
    all_families = (
        *CANDIDATES,
        "GLOBAL_OWN_PREMIUM_CONTROL",
        "GLOBAL_FUNDING_ORTHOGONAL_DIAGNOSTIC",
    )
    target_maps = _build_target_map(feature_panel, all_families, cfg)
    possible_intervals = [
        (entry, entry + pd.Timedelta(days=cfg.holding_days))
        for entry in sorted(feature_by_entry)
        if entry in close.index
        and entry + pd.Timedelta(days=cfg.holding_days) in close.index
    ]
    funding_lookup = _funding_interval_lookup(funding, possible_intervals)
    primary_frames = []
    delayed_frames = []
    controls = []
    for candidate in CANDIDATES:
        primary_frames.append(
            _build_path_from_targets(
                candidate,
                target_maps[candidate],
                feature_by_entry,
                close,
                funding_lookup,
                cfg,
            )
        )
        ordered = sorted(target_maps[candidate])
        delayed_mapping = {
            ordered[index]: ordered[index - 1] for index in range(1, len(ordered))
        }
        delayed_frames.append(
            _build_path_from_targets(
                candidate,
                target_maps[candidate],
                feature_by_entry,
                close,
                funding_lookup,
                cfg,
                delayed_mapping,
            )
        )
    controls.append(
        _build_path_from_targets(
            "GLOBAL_OWN_PREMIUM_CONTROL",
            target_maps["GLOBAL_OWN_PREMIUM_CONTROL"],
            feature_by_entry,
            close,
            funding_lookup,
            cfg,
        )
    )
    controls.append(
        _build_path_from_targets(
            "GLOBAL_FUNDING_ORTHOGONAL_DIAGNOSTIC",
            target_maps["GLOBAL_FUNDING_ORTHOGONAL_DIAGNOSTIC"],
            feature_by_entry,
            close,
            funding_lookup,
            cfg,
        )
    )
    return (
        pd.concat(primary_frames, ignore_index=True),
        pd.concat(delayed_frames, ignore_index=True),
        pd.concat(controls, ignore_index=True),
        target_maps,
    )


def _random_target(
    local: pd.DataFrame,
    candidate: str,
    rng: np.random.Generator,
    cfg: V196Config,
) -> dict[str, float]:
    eligible = local.dropna(subset=["btc_beta"]).copy()
    if candidate == GLOBAL_CANDIDATE:
        count = 2 * cfg.global_bucket_size
        if len(eligible) < max(cfg.minimum_global_cross_section, count):
            return {}
        chosen = rng.choice(len(eligible), size=count, replace=False)
        symbols = eligible.iloc[chosen]["symbol"].astype(str).tolist()
        raw = {
            symbol: 0.5 / cfg.global_bucket_size
            for symbol in symbols[: cfg.global_bucket_size]
        }
        raw.update(
            {
                symbol: -0.5 / cfg.global_bucket_size
                for symbol in symbols[cfg.global_bucket_size :]
            }
        )
    else:
        pairs = []
        for _, group in eligible.groupby("community_id", sort=True):
            if len(group) < cfg.minimum_community_size:
                continue
            chosen = rng.choice(len(group), size=2, replace=False)
            symbols = group.iloc[chosen]["symbol"].astype(str).tolist()
            pairs.append((symbols[0], symbols[1]))
        if not pairs:
            return {}
        pair_weight = 0.5 / len(pairs)
        raw = {}
        for long_symbol, short_symbol in pairs:
            raw[long_symbol] = raw.get(long_symbol, 0.0) + pair_weight
            raw[short_symbol] = raw.get(short_symbol, 0.0) - pair_weight
    return _neutralize_weights(
        raw, eligible.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
    )


def build_v196_random_controls(
    feature_panel: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    observed: pd.DataFrame,
    cfg: V196Config = V196Config(),
) -> pd.DataFrame:
    feature_panel = feature_panel.copy()
    feature_panel["entry_time"] = pd.to_datetime(feature_panel["entry_time"], utc=True)
    observed_entries = sorted(pd.DatetimeIndex(observed["entry_time"].unique()))
    feature_by_entry = {
        entry: feature_panel[feature_panel["entry_time"].eq(entry)].copy()
        for entry in observed_entries
    }
    intervals = [(entry, entry + pd.Timedelta(days=cfg.holding_days)) for entry in observed_entries]
    funding_lookup = _funding_interval_lookup(funding, intervals)
    contexts = {}
    for entry in observed_entries:
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        symbols = sorted(set(feature_by_entry[entry]["symbol"]) | {BTC})
        price = close.loc[exit_time, symbols] / close.loc[entry, symbols] - 1.0
        future_funding = pd.Series(
            {
                symbol: funding_lookup.get((entry, exit_time, symbol), np.nan)
                for symbol in symbols
            },
            dtype=float,
        )
        contexts[entry] = (price, future_funding)
    rows = []
    for iteration in range(cfg.null_iterations):
        means = {}
        for candidate_index, candidate in enumerate(CANDIDATES):
            rng = np.random.default_rng(
                cfg.seed + iteration * 1009 + candidate_index * 100_003
            )
            previous: dict[str, float] = {}
            gross_values = []
            turnovers = []
            for entry in observed_entries:
                local = feature_by_entry[entry]
                weights = _random_target(local, candidate, rng, cfg)
                if not weights:
                    continue
                price, future_funding = contexts[entry]
                gross = float(
                    sum(
                        weight * float(price[symbol])
                        - weight * float(future_funding[symbol])
                        for symbol, weight in weights.items()
                    )
                )
                gross_values.append(gross)
                turnovers.append(_turnover(previous, weights))
                previous = weights
            if turnovers:
                turnovers[-1] += sum(abs(weight) for weight in previous.values())
            net = np.asarray(gross_values) - cfg.one_way_cost * np.asarray(turnovers)
            means[candidate] = float(net.mean())
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "weeks": len(net),
                    "mean_primary_net_return": means[candidate],
                }
            )
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "weeks": max(
                    row["weeks"]
                    for row in rows[-len(CANDIDATES) :]
                ),
                "mean_primary_net_return": max(means.values()),
            }
        )
    return pd.DataFrame(rows)


def _summary_metrics(
    sample: pd.DataFrame,
    random_controls: pd.DataFrame,
    fss3: pd.DataFrame,
    delayed: pd.DataFrame,
    control_mean: float,
    orthogonal_mean: float,
    cfg: V196Config,
) -> dict[str, object]:
    candidate = str(sample["candidate"].iloc[0])
    primary = sample["primary_net_return"].to_numpy(dtype=float)
    bootstrap = _moving_block_means(
        primary,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + CANDIDATES.index(candidate)),
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    periods = sample.groupby("period")["primary_net_return"].mean()
    monthly = sample.groupby("entry_month")["primary_net_return"].sum()
    positive_months = monthly.clip(lower=0)
    concentration = (
        float(positive_months.max() / positive_months.sum())
        if positive_months.sum() > 0
        else math.inf
    )
    loo = [
        float(sample.loc[~sample["entry_month"].eq(month), "primary_net_return"].mean())
        for month in sorted(sample["entry_month"].unique())
    ]
    symbol_totals: dict[str, float] = {}
    for contributions in sample["symbol_contributions"]:
        for symbol, value in dict(contributions).items():
            symbol_totals[symbol] = symbol_totals.get(symbol, 0.0) + float(value)
    positive_symbols = np.asarray(
        [value for value in symbol_totals.values() if value > 0], dtype=float
    )
    symbol_concentration = (
        float(positive_symbols.max() / positive_symbols.sum())
        if positive_symbols.size and positive_symbols.sum() > 0
        else math.inf
    )
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ]
    observed_mean = float(sample["primary_net_return"].mean())
    fss3_local = fss3[["entry_time", "primary_net_return"]].rename(
        columns={"primary_net_return": "fss3_primary_net_return"}
    )
    comparison = sample[["entry_time", "primary_net_return"]].merge(
        fss3_local, on="entry_time", how="inner"
    )
    correlation = float(
        comparison[["primary_net_return", "fss3_primary_net_return"]]
        .corr()
        .iloc[0, 1]
    )
    counts = sample["period"].value_counts()
    return {
        "candidate": candidate,
        "weeks": len(sample),
        "months": sample["entry_month"].nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "mean_turnover": float(sample["realized_turnover"].mean()),
        "mean_price_bp": float(sample["price_return"].mean() * 10_000),
        "mean_funding_bp": float(sample["funding_return"].mean() * 10_000),
        "mean_gross_bp": float(sample["gross_return"].mean() * 10_000),
        "mean_residual_gross_bp": float(
            sample["residual_gross_return"].mean() * 10_000
        ),
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": float(sample["stress_net_return"].mean() * 10_000),
        "development_primary_net_bp": float(periods.get("development", np.nan) * 10_000),
        "validation_primary_net_bp": float(periods.get("validation", np.nan) * 10_000),
        "holdout_primary_net_bp": float(periods.get("holdout", np.nan) * 10_000),
        "bootstrap_95_low_bp": float(low * 10_000),
        "bootstrap_95_high_bp": float(high * 10_000),
        "random_family_percentile": float(family.le(observed_mean).mean()),
        "reversed_primary_net_bp": float(
            sample["reversed_primary_net_return"].mean() * 10_000
        ),
        "delayed_primary_net_bp": float(delayed["primary_net_return"].mean() * 10_000),
        "control_primary_net_bp": control_mean * 10_000,
        "orthogonal_diagnostic_primary_net_bp": orthogonal_mean * 10_000,
        "long_gross_bp": float(sample["long_gross_return"].mean() * 10_000),
        "short_gross_bp": float(sample["short_gross_return"].mean() * 10_000),
        "positive_month_concentration": concentration,
        "minimum_leave_one_month_out_bp": min(loo) * 10_000,
        "positive_symbol_concentration": symbol_concentration,
        "worst_period_bp": float(periods.min() * 10_000),
        "fss3_overlap_weeks": len(comparison),
        "fss3_primary_return_correlation": correlation,
        "max_abs_residual_btc_beta": float(sample["residual_btc_beta"].abs().max()),
        "max_gross_notional_drift": float(
            (sample["gross_notional"] - 1.0).abs().max()
        ),
    }


def summarize_and_gate_v196(
    portfolio: pd.DataFrame,
    delayed: pd.DataFrame,
    controls: pd.DataFrame,
    random_controls: pd.DataFrame,
    fss3: pd.DataFrame,
    cfg: V196Config = V196Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    own_control = controls[
        controls["candidate"].eq("GLOBAL_OWN_PREMIUM_CONTROL")
    ]
    orthogonal = controls[
        controls["candidate"].eq("GLOBAL_FUNDING_ORTHOGONAL_DIAGNOSTIC")
    ]
    random_means = random_controls[random_controls["candidate"].isin(CANDIDATES)].groupby(
        "candidate"
    )["mean_primary_net_return"].mean()
    metrics = []
    for candidate in CANDIDATES:
        sample = portfolio[portfolio["candidate"].eq(candidate)].sort_values("entry_time")
        delayed_sample = delayed[delayed["candidate"].eq(candidate)]
        control_mean = (
            float(own_control["primary_net_return"].mean())
            if candidate == GLOBAL_CANDIDATE
            else float(random_means[candidate])
        )
        metrics.append(
            _summary_metrics(
                sample,
                random_controls,
                fss3,
                delayed_sample,
                control_mean,
                float(orthogonal["primary_net_return"].mean()),
                cfg,
            )
        )
    summary = pd.DataFrame(metrics)
    gate_rows = []
    outcomes = []
    for row in summary.itertuples(index=False):
        checks = {
            "complete_weeks_40": (row.weeks >= 40, row.weeks),
            "active_months_10": (row.months >= 10, row.months),
            "validation_weeks_8": (row.validation_weeks >= 8, row.validation_weeks),
            "holdout_weeks_12": (row.holdout_weeks >= 12, row.holdout_weeks),
            "development_primary_positive": (
                row.development_primary_net_bp > 0,
                row.development_primary_net_bp,
            ),
            "validation_primary_positive": (
                row.validation_primary_net_bp > 0,
                row.validation_primary_net_bp,
            ),
            "holdout_primary_positive": (
                row.holdout_primary_net_bp > 0,
                row.holdout_primary_net_bp,
            ),
            "full_stress_positive": (row.mean_stress_net_bp > 0, row.mean_stress_net_bp),
            "price_contribution_positive": (row.mean_price_bp > 0, row.mean_price_bp),
            "residual_gross_positive": (
                row.mean_residual_gross_bp > 0,
                row.mean_residual_gross_bp,
            ),
            "bootstrap_lower_positive": (
                row.bootstrap_95_low_bp > 0,
                row.bootstrap_95_low_bp,
            ),
            "random_family_percentile_95": (
                row.random_family_percentile >= 0.95,
                row.random_family_percentile,
            ),
            "beats_reversed_direction": (
                row.mean_primary_net_bp > row.reversed_primary_net_bp,
                row.mean_primary_net_bp - row.reversed_primary_net_bp,
            ),
            "beats_one_week_delay": (
                row.mean_primary_net_bp > row.delayed_primary_net_bp,
                row.mean_primary_net_bp - row.delayed_primary_net_bp,
            ),
            "beats_candidate_control": (
                row.mean_primary_net_bp > row.control_primary_net_bp,
                row.mean_primary_net_bp - row.control_primary_net_bp,
            ),
            "funding_orthogonal_diagnostic_positive": (
                row.orthogonal_diagnostic_primary_net_bp > 0,
                row.orthogonal_diagnostic_primary_net_bp,
            ),
            "long_sleeve_gross_positive": (row.long_gross_bp > 0, row.long_gross_bp),
            "short_sleeve_gross_positive": (row.short_gross_bp > 0, row.short_gross_bp),
            "mean_turnover_175": (row.mean_turnover <= 1.75, row.mean_turnover),
            "fss3_correlation_60": (
                abs(row.fss3_primary_return_correlation) <= 0.60,
                row.fss3_primary_return_correlation,
            ),
            "leave_one_month_out_positive": (
                row.minimum_leave_one_month_out_bp > 0,
                row.minimum_leave_one_month_out_bp,
            ),
            "positive_symbol_concentration_25": (
                row.positive_symbol_concentration <= 0.25,
                row.positive_symbol_concentration,
            ),
            "positive_month_concentration_35": (
                row.positive_month_concentration <= 0.35,
                row.positive_month_concentration,
            ),
            "worst_period_min40": (row.worst_period_bp >= -40, row.worst_period_bp),
            "btc_beta_neutral": (
                row.max_abs_residual_btc_beta <= 1e-12,
                row.max_abs_residual_btc_beta,
            ),
            "gross_notional_exact": (
                row.max_gross_notional_drift <= 1e-12,
                row.max_gross_notional_drift,
            ),
        }
        eligible = all(passed for passed, _ in checks.values())
        gate_rows.extend(
            {
                "candidate": row.candidate,
                "check": name,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for name, (passed, value) in checks.items()
        )
        outcomes.append(
            {
                **row._asdict(),
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_graph_premium_relative_value_weekly"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _contribution_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, sample in portfolio.groupby("candidate", sort=True):
        totals: dict[str, float] = {}
        for contributions in sample["symbol_contributions"]:
            for symbol, value in dict(contributions).items():
                totals[symbol] = totals.get(symbol, 0.0) + float(value)
        rows.extend(
            {
                "candidate": candidate,
                "symbol": symbol,
                "total_gross_contribution": value,
            }
            for symbol, value in sorted(totals.items())
        )
    return pd.DataFrame(rows)


def _serialize_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    output = portfolio.copy()
    output["weights"] = output["weights"].map(
        lambda value: "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))
    )
    output["symbol_contributions"] = output["symbol_contributions"].map(
        lambda value: "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))
    )
    return output


def _write_findings(outcome: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_graph_premium_relative_value_weekly"
    )
    text = [
        "# v19.6 Graph-Premium Relative-Value Weekly Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        "All decisions use exact as-of premium, graph membership, beta, and settled",
        "funding. Turnover, initial opening, and terminal close are fully charged.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v196_graph_premium_relative_value_weekly(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V196Config = V196Config(),
) -> dict[str, Path]:
    feature_panel = pd.read_parquet(cfg.feature_panel_path)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    funding = load_v195_funding(set(close.columns) | {BTC}, FUNDING_ROOTS)
    portfolio, delayed, controls, _ = build_v196_paths(
        feature_panel, close, funding, cfg
    )
    random_controls = build_v196_random_controls(
        feature_panel, close, funding, portfolio, cfg
    )
    fss3 = pd.read_parquet(cfg.fss3_path)
    fss3["entry_time"] = pd.to_datetime(fss3["entry_time"], utc=True)
    gates, outcome = summarize_and_gate_v196(
        portfolio, delayed, controls, random_controls, fss3, cfg
    )
    contributions = _contribution_table(portfolio)
    root = ensure_dir(report_root)
    outputs = {
        "portfolio": root / "weekly_portfolio.parquet",
        "delayed": root / "delayed_weekly_portfolio.parquet",
        "controls": root / "control_weekly_portfolios.parquet",
        "random_controls": root / "random_controls.parquet",
        "contributions": root / "symbol_contributions.csv",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    _serialize_portfolio(portfolio).to_parquet(outputs["portfolio"], index=False)
    _serialize_portfolio(delayed).to_parquet(outputs["delayed"], index=False)
    _serialize_portfolio(controls).to_parquet(outputs["controls"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    contributions.to_csv(outputs["contributions"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(outcome, findings_path)
    return outputs


__all__ = [
    "CANDIDATES",
    "COMMUNITY_CANDIDATE",
    "GLOBAL_CANDIDATE",
    "V196Config",
    "build_v196_paths",
    "build_v196_random_controls",
    "summarize_and_gate_v196",
    "write_v196_graph_premium_relative_value_weekly",
]
