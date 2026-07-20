"""Direct BTC shock propagation into causal receiver and insulator buckets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    BTC,
    KLINE_ROOT,
    V178Config,
    _month,
    _period,
    build_btc_source_signals,
    build_monthly_btc_receiver_graph,
    load_v178_market_data,
)


REPORT_ROOT = Path("reports/v17_9_btc_receiver_propagation")
FINDINGS_PATH = Path("docs/v179_btc_receiver_propagation_findings_2026_07_16.md")
RAW_CANDIDATE = "BRP1_SOURCE_DIRECTION_RECEIVER_BASKET"
NEUTRAL_CANDIDATE = "BRP2_BTC_NEUTRAL_RECEIVER_PROPAGATION"
SPREAD_CANDIDATE = "BRP3_RECEIVER_INSULATOR_PROPAGATION_SPREAD"
CANDIDATES = (RAW_CANDIDATE, NEUTRAL_CANDIDATE, SPREAD_CANDIDATE)


@dataclass(frozen=True)
class V179Config(V178Config):
    receiver_bucket_size: int = 8
    insulator_bucket_size: int = 8
    min_bucket_size: int = 5
    holding_bars: int = 1
    raw_primary_cost: float = 0.0020
    raw_stress_cost: float = 0.0030
    neutral_primary_cost: float = 0.0030
    neutral_stress_cost: float = 0.0040
    spread_primary_cost: float = 0.0030
    spread_stress_cost: float = 0.0040
    seed: int = 17_900


def assign_v179_buckets(
    graph: pd.DataFrame,
    cfg: V179Config = V179Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, local in graph.groupby("graph_month", sort=True):
        receivers = (
            local[local["signed_forward_correlation"].gt(0)]
            .sort_values(
                ["receiver_score", "signed_forward_correlation"], ascending=False
            )
            .head(cfg.receiver_bucket_size)
        )
        receiver_names = set(receivers["receiver"].astype(str))
        insulators = (
            local[~local["receiver"].astype(str).isin(receiver_names)]
            .sort_values(
                ["receiver_score", "signed_forward_correlation"], ascending=True
            )
            .head(cfg.insulator_bucket_size)
        )
        for bucket, bucket_frame in (
            ("receiver", receivers),
            ("insulator", insulators),
        ):
            for rank, row in enumerate(bucket_frame.itertuples(index=False), start=1):
                values = row._asdict()
                values.update(
                    {
                        "graph_month": pd.Timestamp(month),
                        "bucket": bucket,
                        "bucket_rank": rank,
                    }
                )
                rows.append(values)
    assignments = pd.DataFrame(rows)
    if not assignments.empty:
        assignments = assignments.sort_values(
            ["graph_month", "bucket", "bucket_rank"]
        ).reset_index(drop=True)
    return assignments


def _candidate_costs(cfg: V179Config) -> dict[str, tuple[float, float]]:
    return {
        RAW_CANDIDATE: (cfg.raw_primary_cost, cfg.raw_stress_cost),
        NEUTRAL_CANDIDATE: (cfg.neutral_primary_cost, cfg.neutral_stress_cost),
        SPREAD_CANDIDATE: (cfg.spread_primary_cost, cfg.spread_stress_cost),
    }


def _bucket_rows(
    assignments: pd.DataFrame,
    timestamp: pd.Timestamp,
    reverse_buckets: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = assignments[assignments["graph_month"].eq(_month(timestamp))]
    receiver_name = "insulator" if reverse_buckets else "receiver"
    insulator_name = "receiver" if reverse_buckets else "insulator"
    receivers = local[local["bucket"].eq(receiver_name)].sort_values("bucket_rank")
    insulators = local[local["bucket"].eq(insulator_name)].sort_values("bucket_rank")
    return receivers, insulators


def build_v179_events(
    signals: pd.DataFrame,
    assignments: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V179Config = V179Config(),
    holding_bars: int | None = None,
    reverse_buckets: bool = False,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    costs = _candidate_costs(cfg)
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        entry_time = pd.Timestamp(signal.feature_time)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        receivers, insulators = _bucket_rows(
            assignments, entry_time, reverse_buckets
        )
        receiver_names = receivers["receiver"].astype(str).tolist()
        insulator_names = insulators["receiver"].astype(str).tolist()
        if len(receiver_names) < cfg.min_bucket_size:
            continue
        required = [BTC, *receiver_names, *insulator_names]
        required = list(dict.fromkeys(required))
        prices = close.loc[[entry_time, exit_time], required]
        if prices[[BTC, *receiver_names]].isna().any().any():
            continue
        future = prices.loc[exit_time] / prices.loc[entry_time] - 1.0
        receiver_beta = receivers.set_index("receiver")["btc_beta"].astype(float)
        receiver_return = float(future[receiver_names].mean())
        mean_receiver_beta = float(receiver_beta.reindex(receiver_names).mean())
        direction = float(signal.direction)
        gross: dict[str, float] = {
            RAW_CANDIDATE: direction * receiver_return,
            NEUTRAL_CANDIDATE: direction
            * (receiver_return - mean_receiver_beta * float(future[BTC]))
            / (1.0 + abs(mean_receiver_beta)),
        }
        mean_insulator_beta = math.nan
        insulator_return = math.nan
        spread_beta = math.nan
        if (
            len(insulator_names) >= cfg.min_bucket_size
            and not prices[insulator_names].isna().any().any()
        ):
            insulator_beta = insulators.set_index("receiver")["btc_beta"].astype(float)
            insulator_return = float(future[insulator_names].mean())
            mean_insulator_beta = float(
                insulator_beta.reindex(insulator_names).mean()
            )
            spread_beta = 0.5 * (mean_receiver_beta - mean_insulator_beta)
            gross[SPREAD_CANDIDATE] = direction * (
                0.5 * (receiver_return - insulator_return)
                - spread_beta * float(future[BTC])
            ) / (1.0 + abs(spread_beta))
        common = {
            **signal._asdict(),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "holding_bars": horizon,
            "period": _period(entry_time),
            "entry_day": entry_time.strftime("%Y-%m-%d"),
            "entry_month": entry_time.strftime("%Y-%m"),
            "receiver_count": len(receiver_names),
            "insulator_count": len(insulator_names),
            "receivers": "|".join(receiver_names),
            "insulators": "|".join(insulator_names),
            "mean_receiver_beta": mean_receiver_beta,
            "mean_insulator_beta": mean_insulator_beta,
            "spread_beta": spread_beta,
            "btc_future_return": float(future[BTC]),
            "mean_receiver_future_return": receiver_return,
            "mean_insulator_future_return": insulator_return,
            "reverse_buckets": reverse_buckets,
        }
        for candidate, gross_return in gross.items():
            primary_cost, stress_cost = costs[candidate]
            rows.append(
                {
                    **common,
                    "candidate": candidate,
                    "gross_return": gross_return,
                    "primary_net_return": gross_return - primary_cost,
                    "stress_net_return": gross_return - stress_cost,
                }
            )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "candidate"]).reset_index(drop=True)
    return events


def summarize_v179(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        candidate_rows = events[events["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = (
                candidate_rows
                if scope == "all"
                else candidate_rows[candidate_rows["period"].eq(scope)]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": len(sample),
                    "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                    "active_months": (
                        sample["entry_month"].nunique() if len(sample) else 0
                    ),
                    "mean_gross_bp": (
                        float(sample["gross_return"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_primary_net_bp": (
                        float(sample["primary_net_return"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_stress_net_bp": (
                        float(sample["stress_net_return"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "win_rate_primary": (
                        float(sample["primary_net_return"].gt(0).mean())
                        if len(sample)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _event_contexts(
    events: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    candidates_by_time = events.groupby("entry_time")["candidate"].agg(set)
    base = events.sort_values("candidate").drop_duplicates("entry_time")
    for row in base.itertuples(index=False):
        local = graph[graph["graph_month"].eq(_month(row.entry_time))].set_index(
            "receiver"
        )
        names = local.index.astype(str).tolist()
        prices = close.reindex(
            index=[row.entry_time, row.exit_time], columns=[BTC, *names]
        )
        if prices[BTC].isna().any():
            continue
        future = prices.loc[row.exit_time] / prices.loc[row.entry_time] - 1.0
        valid = [
            name
            for name in names
            if np.isfinite(future.get(name, math.nan))
            and np.isfinite(local.at[name, "btc_beta"])
        ]
        if len(valid) < row.receiver_count + row.insulator_count:
            continue
        contexts.append(
            {
                "entry_time": row.entry_time,
                "direction": float(row.direction),
                "receiver_count": int(row.receiver_count),
                "insulator_count": int(row.insulator_count),
                "candidates": candidates_by_time.loc[row.entry_time],
                "future": future.reindex(valid).to_numpy(dtype=float),
                "beta": local.reindex(valid)["btc_beta"].to_numpy(dtype=float),
                "btc_future": float(future[BTC]),
            }
        )
    return contexts


def random_v179_controls(
    events: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V179Config = V179Config(),
) -> pd.DataFrame:
    contexts = _event_contexts(events, graph, close)
    costs = _candidate_costs(cfg)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in CANDIDATES}
        for context in contexts:
            receiver_count = int(context["receiver_count"])
            insulator_count = int(context["insulator_count"])
            total = receiver_count + insulator_count
            future = context["future"]
            beta = context["beta"]
            chosen = rng.choice(len(future), size=total, replace=False)
            receiver_indices = chosen[:receiver_count]
            insulator_indices = chosen[receiver_count:]
            receiver_return = float(np.mean(future[receiver_indices]))
            receiver_beta = float(np.mean(beta[receiver_indices]))
            direction = float(context["direction"])
            btc_future = float(context["btc_future"])
            gross = {
                RAW_CANDIDATE: direction * receiver_return,
                NEUTRAL_CANDIDATE: direction
                * (receiver_return - receiver_beta * btc_future)
                / (1.0 + abs(receiver_beta)),
            }
            if SPREAD_CANDIDATE in context["candidates"]:
                insulator_return = float(np.mean(future[insulator_indices]))
                insulator_beta = float(np.mean(beta[insulator_indices]))
                spread_beta = 0.5 * (receiver_beta - insulator_beta)
                gross[SPREAD_CANDIDATE] = direction * (
                    0.5 * (receiver_return - insulator_return)
                    - spread_beta * btc_future
                ) / (1.0 + abs(spread_beta))
            for candidate in context["candidates"]:
                if candidate in gross:
                    values[candidate].append(gross[candidate] - costs[candidate][0])
        means: dict[str, float] = {}
        for candidate in CANDIDATES:
            means[candidate] = (
                float(np.mean(values[candidate])) if values[candidate] else math.nan
            )
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "events": len(values[candidate]),
                    "mean_primary_net_return": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "events": max(len(values[name]) for name in CANDIDATES),
                "mean_primary_net_return": max(finite) if finite else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    sample: pd.DataFrame,
    cfg: V179Config,
    offset: int,
) -> tuple[float, float]:
    daily = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    if not daily:
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed + offset)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_profit_concentration(sample: pd.DataFrame) -> float:
    monthly = sample.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v179(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    reversed_bucket_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V179Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ].dropna()
    costs = _candidate_costs(cfg)
    gate_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        candidate_summary = summary[summary["candidate"].eq(candidate)].set_index(
            "scope"
        )
        sample = events[events["candidate"].eq(candidate)]
        low, high = _bootstrap(sample, cfg, candidate_index)
        real_mean = float(sample["primary_net_return"].mean())
        reversed_mean = float((-sample["gross_return"] - costs[candidate][0]).mean())
        delayed_row = delayed_summary[
            delayed_summary["candidate"].eq(candidate)
            & delayed_summary["scope"].eq("all")
        ]
        rank_reversed_row = reversed_bucket_summary[
            reversed_bucket_summary["candidate"].eq(candidate)
            & reversed_bucket_summary["scope"].eq("all")
        ]
        delayed_mean = float(delayed_row["mean_primary_net_bp"].iloc[0]) / 10_000
        rank_reversed_mean = (
            float(rank_reversed_row["mean_primary_net_bp"].iloc[0]) / 10_000
        )
        local_sensitivity = sensitivity[
            sensitivity["candidate"].eq(candidate) & sensitivity["scope"].eq("all")
        ].set_index("return_quantile")
        local_horizon = horizon_summary[
            horizon_summary["candidate"].eq(candidate)
            & horizon_summary["scope"].eq("all")
        ].set_index("holding_bars")
        random_percentile = float(family.le(real_mean).mean())
        concentration = _positive_profit_concentration(sample)
        checks: dict[str, tuple[bool, float]] = {
            "full_events_100": (
                candidate_summary.loc["all", "events"] >= 100,
                float(candidate_summary.loc["all", "events"]),
            ),
            "validation_events_20": (
                candidate_summary.loc["validation", "events"] >= 20,
                float(candidate_summary.loc["validation", "events"]),
            ),
            "holdout_events_25": (
                candidate_summary.loc["holdout", "events"] >= 25,
                float(candidate_summary.loc["holdout", "events"]),
            ),
            "development_primary_positive": (
                candidate_summary.loc["development", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["development", "mean_primary_net_bp"]),
            ),
            "validation_primary_positive": (
                candidate_summary.loc["validation", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["validation", "mean_primary_net_bp"]),
            ),
            "holdout_primary_positive": (
                candidate_summary.loc["holdout", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["holdout", "mean_primary_net_bp"]),
            ),
            "full_stress_positive": (
                candidate_summary.loc["all", "mean_stress_net_bp"] > 0,
                float(candidate_summary.loc["all", "mean_stress_net_bp"]),
            ),
            "bootstrap_lower_positive": (low > 0, low * 10_000),
            "random_family_percentile_95": (
                random_percentile >= 0.95,
                random_percentile,
            ),
            "beats_one_bar_delay": (
                real_mean > delayed_mean,
                (real_mean - delayed_mean) * 10_000,
            ),
            "beats_reversed_direction": (
                real_mean > reversed_mean,
                (real_mean - reversed_mean) * 10_000,
            ),
            "beats_rank_reversal": (
                real_mean > rank_reversed_mean,
                (real_mean - rank_reversed_mean) * 10_000,
            ),
            "source_q95_positive": (
                local_sensitivity.loc[0.95, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[0.95, "mean_primary_net_bp"]),
            ),
            "source_q99_positive": (
                local_sensitivity.loc[0.99, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[0.99, "mean_primary_net_bp"]),
            ),
            "holding_30m_positive": (
                local_horizon.loc[2, "mean_primary_net_bp"] > 0,
                float(local_horizon.loc[2, "mean_primary_net_bp"]),
            ),
            "holding_60m_positive": (
                local_horizon.loc[4, "mean_primary_net_bp"] > 0,
                float(local_horizon.loc[4, "mean_primary_net_bp"]),
            ),
            "positive_profit_concentration_35": (
                concentration <= 0.35,
                concentration,
            ),
        }
        eligible = all(passed for passed, _ in checks.values())
        gate_rows.extend(
            {
                "candidate": candidate,
                "check": check,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for check, (passed, value) in checks.items()
        )
        outcomes.append(
            {
                "candidate": candidate,
                "events": len(sample),
                "mean_primary_net_bp": real_mean * 10_000,
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_family_percentile": random_percentile,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "rank_reversed_primary_net_bp": rank_reversed_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    check for check, (passed, _) in checks.items() if not passed
                ),
            }
        )
    outcome = pd.DataFrame(outcomes)
    outcome["verdict"] = np.where(
        outcome["eligible"],
        "offline_research_candidate_only",
        "reject_btc_receiver_propagation",
    )
    return pd.DataFrame(gate_rows), outcome


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_btc_receiver_propagation"
    )
    text = [
        "# v17.9 BTC Receiver Propagation Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The monthly graph and all source thresholds use only prior completed bars.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v179_btc_receiver_propagation(
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V179Config = V179Config(),
) -> dict[str, Path]:
    close, btc_source = load_v178_market_data(kline_root)
    returns = close.pct_change(fill_method=None)
    signals = build_btc_source_signals(btc_source, cfg)
    graph = build_monthly_btc_receiver_graph(
        returns, signals["feature_time"].min(), signals["feature_time"].max(), cfg
    )
    assignments = assign_v179_buckets(graph, cfg)
    events = build_v179_events(signals, assignments, close, cfg)
    summary = summarize_v179(events)

    delayed_signals = build_btc_source_signals(btc_source, cfg, shift_bars=1)
    delayed_events = build_v179_events(delayed_signals, assignments, close, cfg)
    delayed_summary = summarize_v179(delayed_events)
    reversed_bucket_events = build_v179_events(
        signals, assignments, close, cfg, reverse_buckets=True
    )
    reversed_bucket_events = reversed_bucket_events.merge(
        events[["candidate", "entry_time"]],
        on=["candidate", "entry_time"],
        how="inner",
    )
    reversed_bucket_summary = summarize_v179(reversed_bucket_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for quantile in (0.95, 0.99):
        local_signals = build_btc_source_signals(
            btc_source, cfg, return_quantile=quantile
        )
        local = summarize_v179(
            build_v179_events(local_signals, assignments, close, cfg)
        )
        local["return_quantile"] = quantile
        sensitivity_frames.append(local)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (2, 4):
        local = summarize_v179(
            build_v179_events(
                signals, assignments, close, cfg, holding_bars=holding_bars
            )
        )
        local["holding_bars"] = holding_bars
        horizon_frames.append(local)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v179_controls(events, graph, close, cfg)
    gates, outcome = audit_v179(
        events,
        summary,
        delayed_summary,
        reversed_bucket_summary,
        sensitivity,
        horizon_summary,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "btc_source_signals.parquet",
        "graph": root / "monthly_btc_receiver_graph.parquet",
        "assignments": root / "monthly_bucket_assignments.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "rank_reversed_events": root / "rank_reversed_candidate_events.parquet",
        "rank_reversed_summary": root / "rank_reversed_period_summary.csv",
        "sensitivity": root / "source_threshold_sensitivity.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_bucket_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    graph.to_parquet(paths["graph"], index=False)
    assignments.to_parquet(paths["assignments"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    reversed_bucket_events.to_parquet(paths["rank_reversed_events"], index=False)
    reversed_bucket_summary.to_csv(paths["rank_reversed_summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    horizon_summary.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "CANDIDATES",
    "NEUTRAL_CANDIDATE",
    "RAW_CANDIDATE",
    "SPREAD_CANDIDATE",
    "V179Config",
    "assign_v179_buckets",
    "audit_v179",
    "build_v179_events",
    "random_v179_controls",
    "summarize_v179",
    "write_v179_btc_receiver_propagation",
]
