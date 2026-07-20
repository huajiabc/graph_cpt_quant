"""Direct BTC response after frozen leverage-flow source events."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _period
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    BTC,
    BUILD,
    UNWIND,
    V185Config,
    build_v185_source_signals,
)


REPORT_ROOT = Path("reports/v18_6_btc_leverage_event_direct_response")
FINDINGS_PATH = Path(
    "docs/v186_btc_leverage_event_direct_response_findings_2026_07_16.md"
)
BUILD_CANDIDATE = "LDR1_BTC_BUILD_CONTINUATION"
UNWIND_CANDIDATE = "LDR2_BTC_UNWIND_REVERSAL"
CANDIDATES = (BUILD_CANDIDATE, UNWIND_CANDIDATE)
CANDIDATE_BY_KIND = {BUILD: BUILD_CANDIDATE, UNWIND: UNWIND_CANDIDATE}
DIRECTION_MULTIPLIER = {BUILD: 1.0, UNWIND: -1.0}


@dataclass(frozen=True)
class V186Config(V185Config):
    primary_cost: float = 0.0010
    stress_cost: float = 0.0015
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_600


def build_v186_events(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V186Config = V186Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    """Build direct BTC event returns without receiver or graph selection."""
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        kind = str(signal.kind)
        if kind not in CANDIDATE_BY_KIND:
            continue
        source_time = pd.Timestamp(signal.source_feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        entry = float(close.at[entry_time, BTC])
        exit_price = float(close.at[exit_time, BTC])
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
            continue
        trade_direction = float(signal.source_sign) * DIRECTION_MULTIPLIER[kind]
        underlying_return = exit_price / entry - 1.0
        gross = trade_direction * underlying_return
        values = signal._asdict()
        values["candidate"] = CANDIDATE_BY_KIND[kind]
        rows.append(
            {
                **values,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "period": _period(source_time),
                "entry_day": source_time.strftime("%Y-%m-%d"),
                "entry_month": source_time.strftime("%Y-%m"),
                "trade_direction": trade_direction,
                "btc_underlying_return": underlying_return,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
                "reversed_primary_net_return": -gross - cfg.primary_cost,
            }
        )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "candidate"]).reset_index(
            drop=True
        )
    return events


def summarize_v186(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
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


def _month_close_index(close: pd.DataFrame, month: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return close.index[(close.index >= start) & (close.index < end)]


def _valid_circular_offsets(
    positions: np.ndarray,
    month_index: pd.DatetimeIndex,
    close_index: pd.DatetimeIndex,
    holding_bars: int,
) -> np.ndarray:
    count = len(month_index)
    if count <= 1:
        return np.array([], dtype=int)
    valid_entry = np.array(
        [
            timestamp + pd.Timedelta(minutes=15 * holding_bars) in close_index
            for timestamp in month_index
        ],
        dtype=bool,
    )
    offsets = [
        offset
        for offset in range(1, count)
        if valid_entry[(positions + offset) % count].all()
    ]
    return np.asarray(offsets, dtype=int)


def random_v186_circular_controls(
    events: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V186Config = V186Config(),
) -> pd.DataFrame:
    """Shift each candidate-month event sequence by a common nonzero offset."""
    contexts: dict[str, list[dict[str, object]]] = {
        candidate: [] for candidate in CANDIDATES
    }
    close_index = close.index
    for (candidate, month), sample in events.groupby(
        ["candidate", "entry_month"], sort=True
    ):
        month_index = _month_close_index(close, str(month))
        lookup = pd.Series(np.arange(len(month_index)), index=month_index)
        positions = lookup.reindex(pd.DatetimeIndex(sample["entry_time"])).to_numpy()
        if np.isnan(positions).any():
            continue
        positions = positions.astype(int)
        offsets = _valid_circular_offsets(
            positions, month_index, close_index, cfg.holding_bars
        )
        if len(offsets) == 0:
            continue
        contexts[str(candidate)].append(
            {
                "month_index": month_index,
                "positions": positions,
                "offsets": offsets,
                "directions": sample["trade_direction"].to_numpy(dtype=float),
            }
        )

    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        means: dict[str, float] = {}
        counts: dict[str, int] = {}
        for candidate in CANDIDATES:
            gross_values: list[float] = []
            for context in contexts[candidate]:
                offsets = context["offsets"]
                offset = int(offsets[rng.integers(0, len(offsets))])
                shifted = context["month_index"][
                    (context["positions"] + offset) % len(context["month_index"])
                ]
                exits = shifted + pd.Timedelta(minutes=15 * cfg.holding_bars)
                entry_prices = close.loc[shifted, BTC].to_numpy(dtype=float)
                exit_prices = close.loc[exits, BTC].to_numpy(dtype=float)
                future = exit_prices / entry_prices - 1.0
                gross_values.extend(
                    (context["directions"] * future).astype(float).tolist()
                )
            counts[candidate] = len(gross_values)
            means[candidate] = (
                float(np.mean(gross_values) - cfg.primary_cost)
                if gross_values
                else math.nan
            )
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "events": counts[candidate],
                    "mean_primary_net_return": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "events": max(counts.values()),
                "mean_primary_net_return": max(finite) if finite else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    sample: pd.DataFrame,
    cfg: V186Config,
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


def audit_v186(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V186Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"),
        "mean_primary_net_return",
    ].dropna()
    gate_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for index, candidate in enumerate(CANDIDATES):
        scoped = summary[summary["candidate"].eq(candidate)].set_index("scope")
        sample = events[events["candidate"].eq(candidate)]
        low, high = _bootstrap(sample, cfg, index)
        real_mean = float(sample["primary_net_return"].mean())
        reversed_mean = float(sample["reversed_primary_net_return"].mean())
        delayed = delayed_summary[
            delayed_summary["candidate"].eq(candidate)
            & delayed_summary["scope"].eq("all")
        ]
        delayed_mean = float(delayed["mean_primary_net_bp"].iloc[0]) / 10_000
        local_sensitivity = sensitivity[
            sensitivity["candidate"].eq(candidate)
            & sensitivity["scope"].eq("all")
        ].set_index("return_quantile")
        local_horizon = horizons[
            horizons["candidate"].eq(candidate) & horizons["scope"].eq("all")
        ].set_index("holding_bars")
        percentile = float(family.le(real_mean).mean())
        concentration = _positive_profit_concentration(sample)
        checks: dict[str, tuple[bool, float]] = {
            "full_events_100": (
                scoped.loc["all", "events"] >= 100,
                float(scoped.loc["all", "events"]),
            ),
            "validation_events_20": (
                scoped.loc["validation", "events"] >= 20,
                float(scoped.loc["validation", "events"]),
            ),
            "holdout_events_25": (
                scoped.loc["holdout", "events"] >= 25,
                float(scoped.loc["holdout", "events"]),
            ),
            "development_primary_positive": (
                scoped.loc["development", "mean_primary_net_bp"] > 0,
                float(scoped.loc["development", "mean_primary_net_bp"]),
            ),
            "validation_primary_positive": (
                scoped.loc["validation", "mean_primary_net_bp"] > 0,
                float(scoped.loc["validation", "mean_primary_net_bp"]),
            ),
            "holdout_primary_positive": (
                scoped.loc["holdout", "mean_primary_net_bp"] > 0,
                float(scoped.loc["holdout", "mean_primary_net_bp"]),
            ),
            "full_stress_positive": (
                scoped.loc["all", "mean_stress_net_bp"] > 0,
                float(scoped.loc["all", "mean_stress_net_bp"]),
            ),
            "bootstrap_lower_positive": (low > 0, low * 10_000),
            "random_family_percentile_95": (percentile >= 0.95, percentile),
            "beats_reversed_direction": (
                real_mean > reversed_mean,
                (real_mean - reversed_mean) * 10_000,
            ),
            "beats_one_bar_delay": (
                real_mean > delayed_mean,
                (real_mean - delayed_mean) * 10_000,
            ),
            "return_q85_positive": (
                local_sensitivity.loc[0.85, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[0.85, "mean_primary_net_bp"]),
            ),
            "return_q95_positive": (
                local_sensitivity.loc[0.95, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[0.95, "mean_primary_net_bp"]),
            ),
            "holding_15m_positive": (
                local_horizon.loc[1, "mean_primary_net_bp"] > 0,
                float(local_horizon.loc[1, "mean_primary_net_bp"]),
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
                "check": name,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for name, (passed, value) in checks.items()
        )
        outcomes.append(
            {
                "candidate": candidate,
                "events": len(sample),
                "mean_gross_bp": float(sample["gross_return"].mean() * 10_000),
                "mean_primary_net_bp": real_mean * 10_000,
                "mean_stress_net_bp": float(
                    sample["stress_net_return"].mean() * 10_000
                ),
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_family_percentile": percentile,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_btc_leverage_event_direct_response"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_btc_leverage_event_direct_response"
    )
    text = [
        "# v18.6 BTC Leverage-Event Direct Response Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The v18.5 source construction was reused without graph membership.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v186_btc_leverage_event_direct_response(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V186Config = V186Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    signals = build_v185_source_signals(close, panels, cfg)
    events = build_v186_events(signals, close, cfg)
    summary = summarize_v186(events)

    delayed_events = build_v186_events(signals, close, cfg, entry_delay_bars=1)
    delayed_summary = summarize_v186(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for quantile in (0.85, 0.95):
        local_signals = build_v185_source_signals(
            close, panels, cfg, return_quantile=quantile
        )
        local = summarize_v186(build_v186_events(local_signals, close, cfg))
        local["return_quantile"] = quantile
        sensitivity_frames.append(local)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (1, 4):
        local = summarize_v186(
            build_v186_events(signals, close, cfg, holding_bars=holding_bars)
        )
        local["holding_bars"] = holding_bars
        horizon_frames.append(local)
    horizons = pd.concat(horizon_frames, ignore_index=True)

    random_controls = random_v186_circular_controls(events, close, cfg)
    gates, outcome = audit_v186(
        events,
        summary,
        delayed_summary,
        sensitivity,
        horizons,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "source_signals.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "source_return_sensitivity.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_circular_controls.parquet",
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
    horizons.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "BUILD_CANDIDATE",
    "CANDIDATES",
    "UNWIND_CANDIDATE",
    "V186Config",
    "audit_v186",
    "build_v186_events",
    "random_v186_circular_controls",
    "summarize_v186",
    "write_v186_btc_leverage_event_direct_response",
]
