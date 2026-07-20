"""Cross-sectional receiver-versus-insulator spreads after BTC skew shocks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v173_deribit_skew_receiver_bucket import (
    ALTS,
    KLINE_ROOT,
    SURFACE_PATH,
    V173Config,
    build_monthly_receiver_graph,
    build_surface_signals,
    hourly_log_returns,
    load_v173_prices,
)


REPORT_ROOT = Path("reports/v17_5_deribit_skew_receiver_insulator_spread")
FINDINGS_PATH = Path(
    "docs/v175_deribit_skew_receiver_insulator_spread_findings_2026_07_16.md"
)
STRESS_CANDIDATE = "DSS1_STRESS_SHORT_RECEIVER_LONG_INSULATOR"
RELIEF_CANDIDATE = "DSS2_RELIEF_LONG_RECEIVER_SHORT_INSULATOR"
CANDIDATES = (STRESS_CANDIDATE, RELIEF_CANDIDATE)


@dataclass(frozen=True)
class V175Config:
    holding_hours: int = 24
    primary_cost: float = 0.0040
    stress_cost: float = 0.0060
    min_bucket_size: int = 3
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 17_500


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2024-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2025-01-01", tz="UTC"):
        return "validation"
    return "holdout"


def receiver_insulator_buckets(
    graph: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    month = pd.Timestamp(timestamp).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    local = graph[graph["graph_month"].eq(month)].copy()
    receivers = (
        local[local["selected"].eq(True)]
        .sort_values("receiver_rank")["receiver"]
        .astype(str)
        .tolist()
    )
    eligible = local[
        local["sample_n"].ge(1_000)
        & local["forward_abs_correlation"].notna()
        & ~local["receiver"].astype(str).isin(receivers)
    ].sort_values(
        ["forward_abs_correlation", "direction_advantage", "receiver"],
        ascending=[True, True, True],
    )
    insulators = eligible.head(len(receivers))["receiver"].astype(str).tolist()
    return receivers, insulators


def build_v175_events(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V175Config = V175Config(),
    holding_hours: int | None = None,
) -> pd.DataFrame:
    horizon = cfg.holding_hours if holding_hours is None else holding_hours
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        entry_time = pd.Timestamp(signal.feature_time)
        exit_time = entry_time + pd.Timedelta(hours=horizon)
        receivers, insulators = receiver_insulator_buckets(graph, entry_time)
        if (
            len(receivers) < cfg.min_bucket_size
            or len(insulators) != len(receivers)
            or entry_time not in prices.index
            or exit_time not in prices.index
        ):
            continue
        required = [*receivers, *insulators]
        if prices.loc[[entry_time, exit_time], required].isna().any().any():
            continue
        returns = prices.loc[exit_time, required] / prices.loc[entry_time, required] - 1.0
        receiver_return = float(returns[receivers].mean())
        insulator_return = float(returns[insulators].mean())
        if signal.event_type == "stress":
            candidate = STRESS_CANDIDATE
            gross = 0.5 * (insulator_return - receiver_return)
        else:
            candidate = RELIEF_CANDIDATE
            gross = 0.5 * (receiver_return - insulator_return)
        rows.append(
            {
                **signal._asdict(),
                "candidate": candidate,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "holding_hours": horizon,
                "period": _period(entry_time),
                "year": entry_time.year,
                "receivers": "|".join(receivers),
                "insulators": "|".join(insulators),
                "bucket_size": len(receivers),
                "receiver_return": receiver_return,
                "insulator_return": insulator_return,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["entry_time", "candidate"]).reset_index(drop=True)


def summarize_v175(events: pd.DataFrame) -> pd.DataFrame:
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
                    "events": int(len(sample)),
                    "active_years": int(sample["year"].nunique()),
                    "mean_gross_bp": float(sample["gross_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan,
                    "mean_primary_net_bp": float(
                        sample["primary_net_return"].mean() * 10_000
                    )
                    if len(sample)
                    else math.nan,
                    "mean_stress_net_bp": float(
                        sample["stress_net_return"].mean() * 10_000
                    )
                    if len(sample)
                    else math.nan,
                    "win_rate_primary": float(sample["primary_net_return"].gt(0).mean())
                    if len(sample)
                    else math.nan,
                    "sum_primary_net": float(sample["primary_net_return"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _random_controls(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V175Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in CANDIDATES}
        for signal in signals.itertuples(index=False):
            entry_time = pd.Timestamp(signal.feature_time)
            exit_time = entry_time + pd.Timedelta(hours=cfg.holding_hours)
            receivers, _ = receiver_insulator_buckets(graph, entry_time)
            size = len(receivers)
            if size < cfg.min_bucket_size or 2 * size > len(ALTS):
                continue
            selected = rng.choice(ALTS, size=2 * size, replace=False).tolist()
            first, second = selected[:size], selected[size:]
            if prices.loc[[entry_time, exit_time], selected].isna().any().any():
                continue
            returns = prices.loc[exit_time, selected] / prices.loc[entry_time, selected] - 1.0
            first_return = float(returns[first].mean())
            second_return = float(returns[second].mean())
            candidate = (
                STRESS_CANDIDATE
                if signal.event_type == "stress"
                else RELIEF_CANDIDATE
            )
            gross = (
                0.5 * (second_return - first_return)
                if signal.event_type == "stress"
                else 0.5 * (first_return - second_return)
            )
            values[candidate].append(gross - cfg.primary_cost)
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
                "events": max(len(values[candidate]) for candidate in CANDIDATES),
                "mean_primary_net_return": max(finite) if finite else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    values: np.ndarray,
    cfg: V175Config,
    offset: int,
) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed + offset)
    means = rng.choice(
        values,
        size=(cfg.bootstrap_iterations, len(values)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_year_share(sample: pd.DataFrame) -> float:
    yearly = sample.groupby("year")["primary_net_return"].sum().clip(lower=0)
    return float(yearly.max() / yearly.sum()) if yearly.sum() > 0 else math.inf


def audit_v175(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V175Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ].dropna()
    gates: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        candidate_summary = summary[summary["candidate"].eq(candidate)].set_index("scope")
        sample = events[events["candidate"].eq(candidate)]
        low, high = _bootstrap(
            sample["primary_net_return"].to_numpy(dtype=float), cfg, candidate_index
        )
        real_mean = float(sample["primary_net_return"].mean()) if len(sample) else math.nan
        random_percentile = float(family.le(real_mean).mean()) if len(family) else math.nan
        delayed = delayed_summary[
            delayed_summary["candidate"].eq(candidate)
            & delayed_summary["scope"].eq("all")
        ]
        delayed_mean = (
            float(delayed["mean_primary_net_bp"].iloc[0]) / 10_000
            if not delayed.empty
            else math.nan
        )
        local_sensitivity = sensitivity[
            sensitivity["candidate"].eq(candidate) & sensitivity["scope"].eq("all")
        ].set_index("holding_hours")
        year_share = _positive_year_share(sample)
        checks: dict[str, tuple[bool, float]] = {
            "full_events_25": (
                int(candidate_summary.loc["all", "events"]) >= 25,
                float(candidate_summary.loc["all", "events"]),
            ),
            "validation_events_5": (
                int(candidate_summary.loc["validation", "events"]) >= 5,
                float(candidate_summary.loc["validation", "events"]),
            ),
            "holdout_events_8": (
                int(candidate_summary.loc["holdout", "events"]) >= 8,
                float(candidate_summary.loc["holdout", "events"]),
            ),
            "full_primary_positive": (
                candidate_summary.loc["all", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["all", "mean_primary_net_bp"]),
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
            "random_family_percentile_90": (
                random_percentile >= 0.90,
                random_percentile,
            ),
            "beats_one_day_delay": (
                real_mean > delayed_mean,
                (real_mean - delayed_mean) * 10_000,
            ),
            "holding_8h_positive": (
                8 in local_sensitivity.index
                and local_sensitivity.loc[8, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[8, "mean_primary_net_bp"])
                if 8 in local_sensitivity.index
                else math.nan,
            ),
            "holding_48h_positive": (
                48 in local_sensitivity.index
                and local_sensitivity.loc[48, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[48, "mean_primary_net_bp"])
                if 48 in local_sensitivity.index
                else math.nan,
            ),
            "positive_year_share_50": (year_share <= 0.50, year_share),
        }
        eligible = all(passed for passed, _ in checks.values())
        gates.extend(
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
                "positive_year_share": year_share,
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
        "reject_receiver_insulator_spread",
    )
    return pd.DataFrame(gates), outcome


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_research_candidate_only"
        if bool(outcome["eligible"].any())
        else "reject_receiver_insulator_spread"
    )
    text = [
        "# v17.5 Deribit Skew Receiver-vs-Insulator Spread Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Both legs use closed Binance USD-M hourly prices and unit gross exposure.",
        "Deribit option trades are signal-only. No live permission changes.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v175_deribit_skew_receiver_insulator_spread(
    surface_path: Path = SURFACE_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    signal_cfg: V173Config = V173Config(),
    cfg: V175Config = V175Config(),
) -> dict[str, Path]:
    surface = pd.read_parquet(surface_path)
    surface["feature_time"] = pd.to_datetime(surface["feature_time"], utc=True)
    prices = load_v173_prices(kline_root)
    returns = hourly_log_returns(prices)
    graph = build_monthly_receiver_graph(
        returns,
        surface["feature_time"].min(),
        surface["feature_time"].max(),
        signal_cfg,
    )
    signals = build_surface_signals(surface, signal_cfg)
    events = build_v175_events(signals, graph, prices, cfg)
    summary = summarize_v175(events)

    delayed_signals = build_surface_signals(surface, signal_cfg, shift_days=1)
    delayed_events = build_v175_events(delayed_signals, graph, prices, cfg)
    delayed_summary = summarize_v175(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for holding_hours in (8, 48):
        local_events = build_v175_events(
            signals, graph, prices, cfg, holding_hours=holding_hours
        )
        local_summary = summarize_v175(local_events)
        local_summary["holding_hours"] = holding_hours
        sensitivity_frames.append(local_summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    random_controls = _random_controls(signals, graph, prices, cfg)
    gates, outcome = audit_v175(
        events,
        summary,
        delayed_summary,
        sensitivity,
        random_controls,
        cfg,
    )
    root = ensure_dir(report_root)
    paths = {
        "signals": root / "surface_signals.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "holding_sensitivity.csv",
        "random_controls": root / "random_bucket_pair_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "V175Config",
    "build_v175_events",
    "receiver_insulator_buckets",
    "summarize_v175",
    "write_v175_deribit_skew_receiver_insulator_spread",
]
