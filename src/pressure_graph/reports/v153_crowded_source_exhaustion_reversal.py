"""Direct source-coin reversal after causal crowding-liquidation shocks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v108_oi_leader_bucket import load_v108_features
from pressure_graph.reports.v118_crowding_unwind_transmission import (
    CANDIDATES as V118_CANDIDATES,
    V118Config,
    _candidate_condition,
    build_v118_aligned_panel,
    build_v118_contexts,
    load_v118_account_ratios,
)


REPORT_ROOT = Path("reports/v15_3_crowded_source_exhaustion_reversal")
FINDINGS_PATH = Path("docs/v153_crowded_source_exhaustion_reversal_findings_2026_07_16.md")
CANDIDATE = "LR1_CROWDED_SOURCE_EXHAUSTION_REVERSAL"
REVERSED_CONTROL = "LR1_REVERSED_SOURCE_CONTINUATION"


@dataclass(frozen=True)
class V153Config:
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    cooldown_hours: int = 4
    random_iterations: int = 500
    bootstrap_iterations: int = 2000
    seed: int = 20260716


def load_v153_contexts(
    source_cfg: V118Config = V118Config(),
) -> tuple[dict[pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    features = load_v108_features()
    ratios = load_v118_account_ratios(source_cfg.account_ratio_root)
    panel = build_v118_aligned_panel(features, ratios, source_cfg)
    return build_v118_contexts(panel, source_cfg.membership_path)


def _source_sides(
    context: dict[str, Any],
    source_cfg: V118Config,
    *,
    crowding_shift_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_sources = _candidate_condition(
        context,
        V118_CANDIDATES[0],
        source_cfg,
        crowding_shift_hours,
    )
    short_sources = _candidate_condition(
        context,
        V118_CANDIDATES[1],
        source_cfg,
        crowding_shift_hours,
    )
    return long_sources, short_sources


def source_beta_neutral_weights(
    long_symbols: list[str],
    short_symbols: list[str],
    betas: dict[str, float],
    *,
    reverse: bool = False,
) -> dict[str, float]:
    if reverse:
        long_symbols, short_symbols = short_symbols, long_symbols
    raw: dict[str, float] = {}
    if long_symbols and short_symbols:
        raw.update({symbol: 0.5 / len(long_symbols) for symbol in long_symbols})
        raw.update({symbol: -0.5 / len(short_symbols) for symbol in short_symbols})
    elif long_symbols:
        raw.update({symbol: 1.0 / len(long_symbols) for symbol in long_symbols})
    elif short_symbols:
        raw.update({symbol: -1.0 / len(short_symbols) for symbol in short_symbols})
    else:
        return {}
    hedge = -float(sum(weight * betas[symbol] for symbol, weight in raw.items()))
    unscaled = dict(raw)
    unscaled[BTC] = hedge
    gross = float(sum(abs(weight) for weight in unscaled.values()))
    return {symbol: weight / gross for symbol, weight in unscaled.items()}


def _event_components(
    timestamp: pd.Timestamp,
    context: dict[str, Any],
    weights: dict[str, float],
) -> tuple[float, float, float]:
    raw_future = context["raw_future_4h"]
    btc_future = float(context["btc_future_4h"].at[timestamp])
    alt_symbols = [symbol for symbol in weights if symbol != BTC]
    gross_return = float(
        sum(weights[symbol] * float(raw_future.at[timestamp, symbol]) for symbol in alt_symbols)
        + weights[BTC] * btc_future
    )
    residual_beta = float(
        sum(weights[symbol] * context["betas"][symbol] for symbol in alt_symbols)
        + weights[BTC]
    )
    gross_notional = float(sum(abs(weight) for weight in weights.values()))
    return gross_return, residual_beta, gross_notional


def build_v153_events(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V153Config = V153Config(),
    source_cfg: V118Config = V118Config(),
    *,
    crowding_shift_hours: int = 0,
    reverse: bool = False,
) -> pd.DataFrame:
    rows = []
    last_event: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        long_condition, short_condition = _source_sides(
            context,
            source_cfg,
            crowding_shift_hours=crowding_shift_hours,
        )
        eligible_times = long_condition.index[
            long_condition.any(axis=1) | short_condition.any(axis=1)
        ]
        for raw_timestamp in eligible_times:
            timestamp = pd.Timestamp(raw_timestamp)
            if last_event is not None and timestamp - last_event < cooldown:
                continue
            long_symbols = sorted(
                symbol
                for symbol in long_condition.columns
                if bool(long_condition.at[timestamp, symbol])
                and pd.notna(context["raw_future_4h"].at[timestamp, symbol])
            )
            short_symbols = sorted(
                symbol
                for symbol in short_condition.columns
                if bool(short_condition.at[timestamp, symbol])
                and pd.notna(context["raw_future_4h"].at[timestamp, symbol])
            )
            overlap = set(long_symbols) & set(short_symbols)
            long_symbols = [symbol for symbol in long_symbols if symbol not in overlap]
            short_symbols = [symbol for symbol in short_symbols if symbol not in overlap]
            weights = source_beta_neutral_weights(
                long_symbols,
                short_symbols,
                context["betas"],
                reverse=reverse,
            )
            if not weights or pd.isna(context["btc_future_4h"].at[timestamp]):
                continue
            gross_return, residual_beta, gross_notional = _event_components(
                timestamp, context, weights
            )
            turnover = 2.0 * gross_notional
            rows.append(
                {
                    "candidate": REVERSED_CONTROL if reverse else CANDIDATE,
                    "feature_time": timestamp,
                    "entry_day": timestamp.floor("D"),
                    "month_start": month,
                    "period": context["period"],
                    "long_source_count": len(long_symbols),
                    "short_source_count": len(short_symbols),
                    "long_source_symbols": "|".join(long_symbols),
                    "short_source_symbols": "|".join(short_symbols),
                    "btc_hedge_weight": weights[BTC],
                    "realized_turnover": turnover,
                    "gross_notional": gross_notional,
                    "residual_btc_beta": residual_beta,
                    "gross_return": gross_return,
                    "primary_net_return": gross_return - cfg.one_way_cost * turnover,
                    "stress_net_return": gross_return
                    - cfg.stress_one_way_cost * turnover,
                    "crowding_shift_hours": crowding_shift_hours,
                }
            )
            last_event = timestamp
    return pd.DataFrame(rows)


def build_v153_random_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    events: pd.DataFrame,
    cfg: V153Config = V153Config(),
) -> pd.DataFrame:
    rows = []
    contexts_by_month = {pd.Timestamp(month): context for month, context in contexts.items()}
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        returns = []
        for event in events.itertuples(index=False):
            context = contexts_by_month[pd.Timestamp(event.month_start)]
            symbols = np.asarray(sorted(context["betas"]))
            total = int(event.long_source_count + event.short_source_count)
            chosen = symbols[rng.choice(len(symbols), size=total, replace=False)]
            long_count = int(event.long_source_count)
            long_symbols = [str(symbol) for symbol in chosen[:long_count]]
            short_symbols = [str(symbol) for symbol in chosen[long_count:]]
            weights = source_beta_neutral_weights(
                long_symbols, short_symbols, context["betas"]
            )
            gross_return, _, gross_notional = _event_components(
                pd.Timestamp(event.feature_time), context, weights
            )
            returns.append(gross_return - cfg.one_way_cost * 2.0 * gross_notional)
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_sources_same_time_side_counts_beta_hedge_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def _day_block_bootstrap(
    events: pd.DataFrame,
    cfg: V153Config,
) -> tuple[float, float]:
    day_values = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in events.groupby("entry_day", sort=True, observed=True)
    ]
    rng = np.random.default_rng(cfg.seed + 2)
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        choices = rng.integers(0, len(day_values), size=len(day_values))
        draws[iteration] = float(
            np.concatenate([day_values[index] for index in choices]).mean()
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_v153(
    events: pd.DataFrame,
    reversed_control: pd.DataFrame,
    shifted_control: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V153Config = V153Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_low, bootstrap_high = _day_block_bootstrap(events, cfg)
    periods = events.groupby("period", observed=True)["primary_net_return"].mean()
    months = events.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    counts = events["period"].value_counts()
    observed_mean = float(events["primary_net_return"].mean())
    reversed_mean = float(reversed_control["primary_net_return"].mean())
    shifted_mean = float(shifted_control["primary_net_return"].mean())
    row = {
        "candidate": CANDIDATE,
        "events": len(events),
        "event_days": events["entry_day"].nunique(),
        "months": events["month_start"].nunique(),
        "validation_events": int(counts.get("validation", 0)),
        "holdout_events": int(counts.get("holdout", 0)),
        "long_only_events": int(events["short_source_count"].eq(0).sum()),
        "short_only_events": int(events["long_source_count"].eq(0).sum()),
        "two_sided_events": int(
            (events["long_source_count"].gt(0) & events["short_source_count"].gt(0)).sum()
        ),
        "mean_gross_bp": events["gross_return"].mean() * 10_000,
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": events["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_bp": bootstrap_low * 10_000,
        "bootstrap_95_high_bp": bootstrap_high * 10_000,
        "random_source_percentile": 100
        * random_controls["mean_primary_net_return"].le(observed_mean).mean(),
        "positive_month_concentration": concentration,
        "worst_period_bp": periods.min() * 10_000,
        "worst_event_bp": events["primary_net_return"].min() * 10_000,
        "reversed_control_mean_bp": reversed_mean * 10_000,
        "shifted_control_mean_bp": shifted_mean * 10_000,
        "max_abs_residual_btc_beta": events["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (events["gross_notional"] - 1.0).abs().max(),
    }
    row["promote"] = bool(
        row["events"] >= 200
        and row["months"] >= 10
        and row["validation_events"] >= 50
        and row["holdout_events"] >= 50
        and all(
            row[key] > 0
            for key in (
                "mean_stress_net_bp",
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["random_source_percentile"] >= 95
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["shifted_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-12
        and row["max_gross_notional_drift"] <= 1e-12
    )
    controls = pd.DataFrame(
        [
            {
                "control": REVERSED_CONTROL,
                "events": len(reversed_control),
                "mean_primary_net_bp": reversed_mean * 10_000,
            },
            {
                "control": "LR1_CROWDING_SHIFTED_24H",
                "events": len(shifted_control),
                "mean_primary_net_bp": shifted_mean * 10_000,
            },
        ]
    )
    return pd.DataFrame([row]), controls


def write_v153_crowded_source_exhaustion_reversal(
    cfg: V153Config = V153Config(),
    source_cfg: V118Config = V118Config(),
) -> dict[str, Path]:
    contexts, coverage = load_v153_contexts(source_cfg)
    events = build_v153_events(contexts, cfg, source_cfg)
    reversed_control = build_v153_events(contexts, cfg, source_cfg, reverse=True)
    shifted_control = build_v153_events(
        contexts, cfg, source_cfg, crowding_shift_hours=24
    )
    random_controls = build_v153_random_controls(contexts, events, cfg)
    summary, controls = summarize_v153(
        events, reversed_control, shifted_control, random_controls, cfg
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "coverage": root / "coverage.csv",
        "events": root / "events.parquet",
        "reversed": root / "reversed_control.parquet",
        "shifted": root / "shifted_control.parquet",
        "random": root / "random_sources.csv",
        "summary": root / "summary.csv",
        "controls": root / "control_summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    coverage.to_csv(paths["coverage"], index=False)
    events.to_parquet(paths["events"], index=False)
    reversed_control.to_parquet(paths["reversed"], index=False)
    shifted_control.to_parquet(paths["shifted"], index=False)
    random_controls.to_csv(paths["random"], index=False)
    summary.to_csv(paths["summary"], index=False)
    controls.to_csv(paths["controls"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "events": len(events),
                "random_iterations": cfg.random_iterations,
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_candidate"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v15.3 Crowded-Source Exhaustion Reversal Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Controls",
                "",
                controls.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The v11.8 event thresholds and causal feature timing were reused",
                "unchanged. This study trades source coins, not community followers.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
