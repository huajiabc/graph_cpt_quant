"""Post-hoc weekly short portfolios for rich graph-relative premium."""
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
    _neutralize_weights,
    _turnover,
    load_v195_funding,
)
from pressure_graph.reports.v196_graph_premium_relative_value_weekly import (
    COMMUNITY_CANDIDATE as PARENT_COMMUNITY,
    GLOBAL_CANDIDATE as PARENT_GLOBAL,
    REPORT_ROOT as V196_REPORT_ROOT,
    V196Config,
    _build_path_from_targets,
    _funding_interval_lookup,
)
from pressure_graph.reports.v197_rich_premium_short_feature_audit import (
    COMMUNITY_CHEAP_LONG,
    COMMUNITY_RICH_SHORT,
    GLOBAL_CHEAP_LONG,
    GLOBAL_RICH_SHORT,
    REPORT_ROOT as V197_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v19_8_rich_graph_premium_short_weekly")
FINDINGS_PATH = Path(
    "docs/v198_rich_graph_premium_short_weekly_findings_2026_07_17.md"
)
FEATURE_PANEL_PATH = V195_REPORT_ROOT / "daily_symbol_feature_panel.parquet"
TARGET_PATH = V197_REPORT_ROOT / "weekly_target_weights.parquet"
PARENT_PATH = V196_REPORT_ROOT / "weekly_portfolio.parquet"
GLOBAL_CANDIDATE = "RPS1_GLOBAL_RICH_GRAPH_PREMIUM_SHORT_WEEKLY"
COMMUNITY_CANDIDATE = "RPS2_COMMUNITY_RICH_GRAPH_PREMIUM_SHORT_WEEKLY"
CANDIDATES = (GLOBAL_CANDIDATE, COMMUNITY_CANDIDATE)
VARIANT_TO_CANDIDATE = {
    GLOBAL_RICH_SHORT: GLOBAL_CANDIDATE,
    COMMUNITY_RICH_SHORT: COMMUNITY_CANDIDATE,
}
CHEAP_VARIANT = {
    GLOBAL_CANDIDATE: GLOBAL_CHEAP_LONG,
    COMMUNITY_CANDIDATE: COMMUNITY_CHEAP_LONG,
}
PARENT_CANDIDATE = {
    GLOBAL_CANDIDATE: PARENT_GLOBAL,
    COMMUNITY_CANDIDATE: PARENT_COMMUNITY,
}


@dataclass(frozen=True)
class V198Config(V196Config):
    feature_panel_path: Path = FEATURE_PANEL_PATH
    target_path: Path = TARGET_PATH
    parent_path: Path = PARENT_PATH
    seed: int = 19_800


def _parse_weights(value: str) -> dict[str, float]:
    return {
        item.rsplit(":", 1)[0]: float(item.rsplit(":", 1)[1])
        for item in str(value).split("|")
        if item
    }


def _target_maps_from_v197(
    target_frame: pd.DataFrame,
) -> dict[str, dict[pd.Timestamp, dict[str, float]]]:
    frame = target_frame.copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["weights_map"] = frame["weights"].map(_parse_weights)
    return {
        family: {
            pd.Timestamp(row.entry_time): dict(row.weights_map)
            for row in frame[frame["family"].eq(family)].itertuples(index=False)
        }
        for family in frame["family"].unique()
    }


def _orthogonal_short_target(
    local: pd.DataFrame,
    scope: str,
    cfg: V198Config,
) -> dict[str, float]:
    eligible = local.dropna(
        subset=["funding_orthogonal_premium_z", "btc_beta"]
    ).copy()
    if scope == "global":
        if len(eligible) < cfg.minimum_global_cross_section:
            return {}
        selected = (
            eligible.sort_values(["funding_orthogonal_premium_z", "symbol"])
            .tail(cfg.global_bucket_size)["symbol"]
            .astype(str)
            .tolist()
        )
    else:
        selected = []
        for _, group in eligible.groupby("community_id", sort=True):
            ranked = group.sort_values(
                ["funding_orthogonal_premium_z", "symbol"]
            )
            if len(ranked) >= cfg.minimum_community_size:
                selected.append(str(ranked.iloc[-1]["symbol"]))
    if not selected:
        return {}
    raw = {symbol: -1.0 / len(selected) for symbol in selected}
    beta = eligible.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
    return _neutralize_weights(raw, beta)


def build_v198_paths(
    feature_panel: pd.DataFrame,
    targets: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: V198Config = V198Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = feature_panel.copy()
    panel["entry_time"] = pd.to_datetime(panel["entry_time"], utc=True)
    feature_by_entry = {
        pd.Timestamp(entry): local.copy()
        for entry, local in panel.groupby("entry_time", sort=True)
        if pd.Timestamp(entry).weekday() == 0
    }
    target_maps = _target_maps_from_v197(targets)
    intervals = [
        (entry, entry + pd.Timedelta(days=cfg.holding_days))
        for entry in sorted(feature_by_entry)
        if entry in close.index
        and entry + pd.Timedelta(days=cfg.holding_days) in close.index
    ]
    funding_lookup = _funding_interval_lookup(funding, intervals)
    primary = []
    delayed = []
    cheap = []
    orthogonal = []
    for variant, candidate in VARIANT_TO_CANDIDATE.items():
        mapping = target_maps[variant]
        primary.append(
            _build_path_from_targets(
                candidate,
                mapping,
                feature_by_entry,
                close,
                funding_lookup,
                cfg,
            )
        )
        ordered = sorted(mapping)
        delay_map = {
            ordered[index]: ordered[index - 1] for index in range(1, len(ordered))
        }
        delayed.append(
            _build_path_from_targets(
                candidate,
                mapping,
                feature_by_entry,
                close,
                funding_lookup,
                cfg,
                delay_map,
            )
        )
        cheap_variant = CHEAP_VARIANT[candidate]
        cheap.append(
            _build_path_from_targets(
                cheap_variant,
                target_maps[cheap_variant],
                feature_by_entry,
                close,
                funding_lookup,
                cfg,
            )
        )
        scope = "global" if candidate == GLOBAL_CANDIDATE else "community"
        orth_map = {
            entry: weights
            for entry, local in feature_by_entry.items()
            if (weights := _orthogonal_short_target(local, scope, cfg))
        }
        orthogonal.append(
            _build_path_from_targets(
                f"{candidate}_FUNDING_ORTHOGONAL",
                orth_map,
                feature_by_entry,
                close,
                funding_lookup,
                cfg,
            )
        )
    return (
        pd.concat(primary, ignore_index=True),
        pd.concat(delayed, ignore_index=True),
        pd.concat(cheap, ignore_index=True),
        pd.concat(orthogonal, ignore_index=True),
    )


def _random_one_sided_target(
    local: pd.DataFrame,
    variant: str,
    rng: np.random.Generator,
    cfg: V198Config,
) -> dict[str, float]:
    eligible = local.dropna(subset=["btc_beta"])
    direction = -1.0 if "RICH_PREMIUM_SHORT" in variant else 1.0
    if variant.startswith("GLOBAL_"):
        if len(eligible) < cfg.minimum_global_cross_section:
            return {}
        chosen = rng.choice(len(eligible), size=cfg.global_bucket_size, replace=False)
        selected = eligible.iloc[chosen]["symbol"].astype(str).tolist()
    else:
        selected = []
        for _, group in eligible.groupby("community_id", sort=True):
            if len(group) < cfg.minimum_community_size:
                continue
            selected.append(str(group.iloc[int(rng.integers(0, len(group)))]["symbol"]))
    if not selected:
        return {}
    raw = {symbol: direction / len(selected) for symbol in selected}
    beta = eligible.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
    return _neutralize_weights(raw, beta)


def build_v198_random_controls(
    feature_panel: pd.DataFrame,
    close: pd.DataFrame,
    funding: pd.DataFrame,
    observed: pd.DataFrame,
    cfg: V198Config = V198Config(),
) -> pd.DataFrame:
    panel = feature_panel.copy()
    panel["entry_time"] = pd.to_datetime(panel["entry_time"], utc=True)
    entries = sorted(pd.DatetimeIndex(observed["entry_time"].unique()))
    feature_by_entry = {
        entry: panel[panel["entry_time"].eq(entry)].copy() for entry in entries
    }
    intervals = [(entry, entry + pd.Timedelta(days=cfg.holding_days)) for entry in entries]
    funding_lookup = _funding_interval_lookup(funding, intervals)
    contexts = {}
    for entry in entries:
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        symbols = sorted(set(feature_by_entry[entry]["symbol"]) | {BTC})
        price = close.loc[exit_time, symbols] / close.loc[entry, symbols] - 1.0
        future_funding = {
            symbol: funding_lookup.get((entry, exit_time, symbol), np.nan)
            for symbol in symbols
        }
        contexts[entry] = (price, future_funding)
    rows = []
    variants = (
        GLOBAL_RICH_SHORT,
        GLOBAL_CHEAP_LONG,
        COMMUNITY_RICH_SHORT,
        COMMUNITY_CHEAP_LONG,
    )
    for iteration in range(cfg.null_iterations):
        means = {}
        weeks = {}
        for variant_index, variant in enumerate(variants):
            rng = np.random.default_rng(
                cfg.seed + iteration * 1009 + variant_index * 100_003
            )
            previous: dict[str, float] = {}
            gross_values = []
            turnovers = []
            for entry in entries:
                weights = _random_one_sided_target(
                    feature_by_entry[entry], variant, rng, cfg
                )
                if not weights:
                    continue
                price, future_funding = contexts[entry]
                gross_values.append(
                    float(
                        sum(
                            weight * float(price[symbol])
                            - weight * float(future_funding[symbol])
                            for symbol, weight in weights.items()
                        )
                    )
                )
                turnovers.append(_turnover(previous, weights))
                previous = weights
            if turnovers:
                turnovers[-1] += sum(abs(value) for value in previous.values())
            net = np.asarray(gross_values) - cfg.one_way_cost * np.asarray(turnovers)
            means[variant] = float(net.mean())
            weeks[variant] = len(net)
            rows.append(
                {
                    "iteration": iteration,
                    "variant": variant,
                    "weeks": len(net),
                    "mean_primary_net_return": means[variant],
                }
            )
        rows.append(
            {
                "iteration": iteration,
                "variant": "FOUR_DIRECTION_FAMILY_MAX",
                "weeks": max(weeks.values()),
                "mean_primary_net_return": max(means.values()),
            }
        )
    return pd.DataFrame(rows)


def _concentration_metrics(sample: pd.DataFrame) -> tuple[float, float, float]:
    monthly = sample.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    month_concentration = (
        float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf
    )
    symbol_totals: dict[str, float] = {}
    for contributions in sample["symbol_contributions"]:
        for symbol, value in dict(contributions).items():
            symbol_totals[symbol] = symbol_totals.get(symbol, 0.0) + float(value)
    positive = np.asarray([value for value in symbol_totals.values() if value > 0])
    symbol_concentration = (
        float(positive.max() / positive.sum())
        if positive.size and positive.sum() > 0
        else math.inf
    )
    leave_out = [
        float(sample.loc[~sample["entry_month"].eq(month), "primary_net_return"].mean())
        for month in sorted(sample["entry_month"].unique())
    ]
    return month_concentration, symbol_concentration, min(leave_out)


def summarize_and_gate_v198(
    portfolio: pd.DataFrame,
    delayed: pd.DataFrame,
    cheap: pd.DataFrame,
    orthogonal: pd.DataFrame,
    random_controls: pd.DataFrame,
    parent: pd.DataFrame,
    fss3: pd.DataFrame,
    cfg: V198Config = V198Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["variant"].eq("FOUR_DIRECTION_FAMILY_MAX"),
        "mean_primary_net_return",
    ]
    random_means = random_controls[
        random_controls["variant"].ne("FOUR_DIRECTION_FAMILY_MAX")
    ].groupby("variant")["mean_primary_net_return"].mean()
    gate_rows = []
    outcomes = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        sample = portfolio[portfolio["candidate"].eq(candidate)].sort_values("entry_time")
        delayed_sample = delayed[delayed["candidate"].eq(candidate)]
        cheap_variant = CHEAP_VARIANT[candidate]
        cheap_sample = cheap[cheap["candidate"].eq(cheap_variant)]
        orth_sample = orthogonal[
            orthogonal["candidate"].eq(f"{candidate}_FUNDING_ORTHOGONAL")
        ]
        parent_sample = parent[
            parent["candidate"].eq(PARENT_CANDIDATE[candidate])
        ]
        primary = sample["primary_net_return"].to_numpy(dtype=float)
        draws = _moving_block_means(
            primary,
            cfg.bootstrap_iterations,
            cfg.bootstrap_block_weeks,
            np.random.default_rng(cfg.seed + candidate_index),
        )
        low, high = np.quantile(draws, [0.025, 0.975])
        periods = sample.groupby("period")["primary_net_return"].mean()
        counts = sample["period"].value_counts()
        mean = float(sample["primary_net_return"].mean())
        variant = (
            GLOBAL_RICH_SHORT
            if candidate == GLOBAL_CANDIDATE
            else COMMUNITY_RICH_SHORT
        )
        percentile = float(family.le(mean).mean())
        month_concentration, symbol_concentration, minimum_loo = (
            _concentration_metrics(sample)
        )
        short_gross = float(sample["short_gross_return"].mean())
        btc_gross = float(sample["btc_gross_return"].mean())
        positive_denominator = max(short_gross, 0.0) + max(btc_gross, 0.0)
        short_share = (
            max(short_gross, 0.0) / positive_denominator
            if positive_denominator > 0
            else 0.0
        )
        comparison = sample[["entry_time", "primary_net_return"]].merge(
            fss3[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "fss3"}
            ),
            on="entry_time",
        )
        fss3_correlation = float(
            comparison[["primary_net_return", "fss3"]].corr().iloc[0, 1]
        )
        residual_price = float(
            (sample["residual_gross_return"] - sample["funding_return"]).mean()
        )
        metrics = {
            "candidate": candidate,
            "weeks": len(sample),
            "months": sample["entry_month"].nunique(),
            "validation_weeks": int(counts.get("validation", 0)),
            "holdout_weeks": int(counts.get("holdout", 0)),
            "mean_turnover": float(sample["realized_turnover"].mean()),
            "mean_price_bp": float(sample["price_return"].mean() * 10_000),
            "mean_residual_price_bp": residual_price * 10_000,
            "mean_funding_bp": float(sample["funding_return"].mean() * 10_000),
            "mean_gross_bp": float(sample["gross_return"].mean() * 10_000),
            "mean_primary_net_bp": mean * 10_000,
            "mean_stress_net_bp": float(sample["stress_net_return"].mean() * 10_000),
            "development_primary_net_bp": float(periods.get("development", np.nan) * 10_000),
            "validation_primary_net_bp": float(periods.get("validation", np.nan) * 10_000),
            "holdout_primary_net_bp": float(periods.get("holdout", np.nan) * 10_000),
            "bootstrap_95_low_bp": float(low * 10_000),
            "bootstrap_95_high_bp": float(high * 10_000),
            "random_four_direction_family_percentile": percentile,
            "reversed_primary_net_bp": float(
                sample["reversed_primary_net_return"].mean() * 10_000
            ),
            "delayed_primary_net_bp": float(
                delayed_sample["primary_net_return"].mean() * 10_000
            ),
            "cheap_long_primary_net_bp": float(
                cheap_sample["primary_net_return"].mean() * 10_000
            ),
            "parent_primary_net_bp": float(
                parent_sample["primary_net_return"].mean() * 10_000
            ),
            "matching_random_mean_net_bp": float(random_means[variant] * 10_000),
            "orthogonal_primary_net_bp": float(
                orth_sample["primary_net_return"].mean() * 10_000
            ),
            "short_alt_gross_bp": short_gross * 10_000,
            "btc_hedge_gross_bp": btc_gross * 10_000,
            "short_positive_contribution_share": short_share,
            "fss3_overlap_weeks": len(comparison),
            "fss3_primary_return_correlation": fss3_correlation,
            "minimum_leave_one_month_out_bp": minimum_loo * 10_000,
            "positive_symbol_concentration": symbol_concentration,
            "positive_month_concentration": month_concentration,
            "worst_period_bp": float(periods.min() * 10_000),
            "max_abs_residual_btc_beta": float(sample["residual_btc_beta"].abs().max()),
            "max_gross_notional_drift": float(
                (sample["gross_notional"] - 1.0).abs().max()
            ),
        }
        checks = {
            "complete_weeks_40": (metrics["weeks"] >= 40, metrics["weeks"]),
            "active_months_10": (metrics["months"] >= 10, metrics["months"]),
            "validation_weeks_8": (
                metrics["validation_weeks"] >= 8,
                metrics["validation_weeks"],
            ),
            "holdout_weeks_12": (
                metrics["holdout_weeks"] >= 12,
                metrics["holdout_weeks"],
            ),
            "development_primary_positive": (
                metrics["development_primary_net_bp"] > 0,
                metrics["development_primary_net_bp"],
            ),
            "validation_primary_positive": (
                metrics["validation_primary_net_bp"] > 0,
                metrics["validation_primary_net_bp"],
            ),
            "holdout_primary_positive": (
                metrics["holdout_primary_net_bp"] > 0,
                metrics["holdout_primary_net_bp"],
            ),
            "full_stress_positive": (
                metrics["mean_stress_net_bp"] > 0,
                metrics["mean_stress_net_bp"],
            ),
            "residual_price_positive": (
                metrics["mean_residual_price_bp"] > 0,
                metrics["mean_residual_price_bp"],
            ),
            "funding_contribution_positive": (
                metrics["mean_funding_bp"] > 0,
                metrics["mean_funding_bp"],
            ),
            "bootstrap_lower_positive": (
                metrics["bootstrap_95_low_bp"] > 0,
                metrics["bootstrap_95_low_bp"],
            ),
            "random_four_direction_family_percentile_99": (
                percentile >= 0.99,
                percentile,
            ),
            "beats_reversed_weights": (
                metrics["mean_primary_net_bp"] > metrics["reversed_primary_net_bp"],
                metrics["mean_primary_net_bp"] - metrics["reversed_primary_net_bp"],
            ),
            "beats_one_week_delay": (
                metrics["mean_primary_net_bp"] > metrics["delayed_primary_net_bp"],
                metrics["mean_primary_net_bp"] - metrics["delayed_primary_net_bp"],
            ),
            "beats_cheap_long_direction": (
                metrics["mean_primary_net_bp"] > metrics["cheap_long_primary_net_bp"],
                metrics["mean_primary_net_bp"] - metrics["cheap_long_primary_net_bp"],
            ),
            "beats_parent_double_sleeve": (
                metrics["mean_primary_net_bp"] > metrics["parent_primary_net_bp"],
                metrics["mean_primary_net_bp"] - metrics["parent_primary_net_bp"],
            ),
            "beats_matching_random_mean": (
                metrics["mean_primary_net_bp"] > metrics["matching_random_mean_net_bp"],
                metrics["mean_primary_net_bp"] - metrics["matching_random_mean_net_bp"],
            ),
            "orthogonal_short_positive": (
                metrics["orthogonal_primary_net_bp"] > 0,
                metrics["orthogonal_primary_net_bp"],
            ),
            "short_alt_gross_positive": (
                metrics["short_alt_gross_bp"] > 0,
                metrics["short_alt_gross_bp"],
            ),
            "short_positive_share_50": (short_share >= 0.50, short_share),
            "mean_turnover_085": (
                metrics["mean_turnover"] <= 0.85,
                metrics["mean_turnover"],
            ),
            "fss3_correlation_60": (
                abs(fss3_correlation) <= 0.60,
                fss3_correlation,
            ),
            "leave_one_month_out_positive": (
                metrics["minimum_leave_one_month_out_bp"] > 0,
                metrics["minimum_leave_one_month_out_bp"],
            ),
            "positive_symbol_concentration_25": (
                symbol_concentration <= 0.25,
                symbol_concentration,
            ),
            "positive_month_concentration_35": (
                month_concentration <= 0.35,
                month_concentration,
            ),
            "worst_period_min40": (
                metrics["worst_period_bp"] >= -40,
                metrics["worst_period_bp"],
            ),
            "btc_beta_neutral": (
                metrics["max_abs_residual_btc_beta"] <= 1e-12,
                metrics["max_abs_residual_btc_beta"],
            ),
            "gross_notional_exact": (
                metrics["max_gross_notional_drift"] <= 1e-12,
                metrics["max_gross_notional_drift"],
            ),
        }
        eligible = all(passed for passed, _ in checks.values())
        gate_rows.extend(
            {
                "candidate": candidate,
                "check": name,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for name, (passed, value) in checks.items()
        )
        outcomes.append(
            {
                **metrics,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "posthoc_offline_discovery_requires_natural_forward"
                    if eligible
                    else "reject_rich_graph_premium_short_weekly"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _serialize(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["weights"] = output["weights"].map(
        lambda value: "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))
    )
    output["symbol_contributions"] = output["symbol_contributions"].map(
        lambda value: "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))
    )
    return output


def _write_findings(outcome: pd.DataFrame, path: Path) -> None:
    verdict = (
        "posthoc_offline_discovery_requires_natural_forward"
        if outcome["eligible"].any()
        else "reject_rich_graph_premium_short_weekly"
    )
    text = [
        "# v19.8 Rich Graph-Premium Short Weekly Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        "This is a post-hoc sleeve follow-up with a four-direction family maximum",
        "and a 99% null gate. No result grants PaperLive, live, application,",
        "leverage, remote, or real-order permission.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v198_rich_graph_premium_short_weekly(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V198Config = V198Config(),
) -> dict[str, Path]:
    feature_panel = pd.read_parquet(cfg.feature_panel_path)
    targets = pd.read_parquet(cfg.target_path)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    funding = load_v195_funding(set(close.columns) | {BTC}, FUNDING_ROOTS)
    portfolio, delayed, cheap, orthogonal = build_v198_paths(
        feature_panel, targets, close, funding, cfg
    )
    random_controls = build_v198_random_controls(
        feature_panel, close, funding, portfolio, cfg
    )
    parent = pd.read_parquet(cfg.parent_path)
    fss3 = pd.read_parquet(cfg.fss3_path)
    for frame in (parent, fss3):
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    gates, outcome = summarize_and_gate_v198(
        portfolio,
        delayed,
        cheap,
        orthogonal,
        random_controls,
        parent,
        fss3,
        cfg,
    )
    root = ensure_dir(report_root)
    outputs = {
        "portfolio": root / "weekly_short_portfolio.parquet",
        "delayed": root / "delayed_short_portfolio.parquet",
        "cheap_long": root / "cheap_long_diagnostic.parquet",
        "orthogonal": root / "funding_orthogonal_short_diagnostic.parquet",
        "random_controls": root / "four_direction_random_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    _serialize(portfolio).to_parquet(outputs["portfolio"], index=False)
    _serialize(delayed).to_parquet(outputs["delayed"], index=False)
    _serialize(cheap).to_parquet(outputs["cheap_long"], index=False)
    _serialize(orthogonal).to_parquet(outputs["orthogonal"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(outcome, findings_path)
    return outputs


__all__ = [
    "CANDIDATES",
    "COMMUNITY_CANDIDATE",
    "GLOBAL_CANDIDATE",
    "V198Config",
    "build_v198_paths",
    "build_v198_random_controls",
    "summarize_and_gate_v198",
    "write_v198_rich_graph_premium_short_weekly",
]
