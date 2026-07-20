"""Long-horizon and continuous execution follow-up for v18.0 dispersion."""
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
    _period,
    load_v178_market_data,
)
from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    REPORT_ROOT as V180_REPORT_ROOT,
    V180Config,
    build_v180_events,
)


REPORT_ROOT = Path("reports/v18_1_residual_dispersion_long_horizon_execution")
FINDINGS_PATH = Path(
    "docs/v181_residual_dispersion_long_horizon_execution_findings_2026_07_16.md"
)
CANDIDATE = "RDC1_FIXED_Q975_LONG_HORIZON_EXECUTION"


@dataclass(frozen=True)
class V181Config(V180Config):
    primary_holding_bars: int = 16
    diagnostic_holding_bars: tuple[int, ...] = (8, 32, 48)
    continuous_primary_one_way_cost: float = 0.0015
    continuous_stress_one_way_cost: float = 0.0020
    seed: int = 18_100


def summarize_event_sleeves(
    events: pd.DataFrame,
    holding_bars: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "holding_bars": holding_bars,
                "holding_hours": holding_bars / 4,
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


def build_continuous_v181_book(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V181Config = V181Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = close.columns.astype(str).tolist()
    symbol_position = {symbol: index for index, symbol in enumerate(symbols)}
    time_position = {timestamp: index for index, timestamp in enumerate(close.index)}
    weight_sum = np.zeros((len(close), len(symbols)), dtype=np.float64)
    active_count = np.zeros(len(close), dtype=np.int32)
    sleeve_rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        entry_time = pd.Timestamp(signal.source_feature_time)
        start = time_position.get(entry_time)
        if start is None:
            continue
        end = start + cfg.primary_holding_bars
        if end >= len(close):
            continue
        laggards = str(signal.laggards).split("|")
        leaders = str(signal.leaders).split("|")
        if any(name not in symbol_position for name in [*laggards, *leaders]):
            continue
        spread_beta = float(signal.spread_beta)
        normalizer = 1.0 + abs(spread_beta)
        weights = np.zeros(len(symbols), dtype=np.float64)
        for name in laggards:
            weights[symbol_position[name]] += 0.5 / len(laggards) / normalizer
        for name in leaders:
            weights[symbol_position[name]] -= 0.5 / len(leaders) / normalizer
        weights[symbol_position[BTC]] -= spread_beta / normalizer
        weight_sum[start:end] += weights
        active_count[start:end] += 1
        sleeve_rows.append(
            {
                "source_feature_time": entry_time,
                "exit_time": close.index[end],
                "holding_bars": cfg.primary_holding_bars,
                "gross_exposure": float(np.abs(weights).sum()),
                "net_exposure": float(weights.sum()),
                "laggards": signal.laggards,
                "leaders": signal.leaders,
                "spread_beta": spread_beta,
            }
        )
    weights = np.divide(
        weight_sum,
        active_count[:, None],
        out=np.zeros_like(weight_sum),
        where=active_count[:, None] > 0,
    )
    future_returns = close.shift(-1).div(close).sub(1.0).to_numpy(dtype=float)
    gross_return = np.nansum(weights * future_returns, axis=1)
    previous = np.vstack([np.zeros((1, len(symbols))), weights[:-1]])
    turnover = np.abs(weights - previous).sum(axis=1)
    frame = pd.DataFrame(
        {
            "bar_time": close.index,
            "period": [_period(pd.Timestamp(value)) for value in close.index],
            "entry_day": close.index.strftime("%Y-%m-%d"),
            "entry_month": close.index.strftime("%Y-%m"),
            "active_sleeves": active_count,
            "gross_exposure": np.abs(weights).sum(axis=1),
            "net_exposure": weights.sum(axis=1),
            "turnover": turnover,
            "gross_return": gross_return,
            "primary_cost_return": (
                turnover * cfg.continuous_primary_one_way_cost
            ),
            "stress_cost_return": turnover * cfg.continuous_stress_one_way_cost,
        }
    )
    frame["primary_net_return"] = (
        frame["gross_return"] - frame["primary_cost_return"]
    )
    frame["stress_net_return"] = (
        frame["gross_return"] - frame["stress_cost_return"]
    )
    return frame, pd.DataFrame(sleeve_rows)


def summarize_continuous_v181(book: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = book if scope == "all" else book[book["period"].eq(scope)]
        active = sample[sample["active_sleeves"].gt(0) | sample["turnover"].gt(0)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "calendar_bars": len(sample),
                "active_bars": int(sample["active_sleeves"].gt(0).sum()),
                "active_days": active["entry_day"].nunique(),
                "mean_active_sleeves": (
                    float(active["active_sleeves"].mean())
                    if len(active)
                    else math.nan
                ),
                "mean_gross_exposure": (
                    float(active["gross_exposure"].mean())
                    if len(active)
                    else math.nan
                ),
                "total_turnover": float(sample["turnover"].sum()),
                "gross_return_sum_pct": float(sample["gross_return"].sum() * 100),
                "primary_cost_sum_pct": float(
                    sample["primary_cost_return"].sum() * 100
                ),
                "primary_net_sum_pct": float(
                    sample["primary_net_return"].sum() * 100
                ),
                "stress_net_sum_pct": float(sample["stress_net_return"].sum() * 100),
            }
        )
    return pd.DataFrame(rows)


def _event_bootstrap(
    events: pd.DataFrame,
    cfg: V181Config,
) -> tuple[float, float]:
    daily = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in events.groupby("entry_day", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _continuous_bootstrap(
    book: pd.DataFrame,
    cfg: V181Config,
) -> tuple[float, float]:
    daily = book.groupby("entry_day")["primary_net_return"].sum().to_numpy(dtype=float)
    rng = np.random.default_rng(cfg.seed + 1)
    totals = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        totals.append(float(daily[chosen].sum()))
    return float(np.quantile(totals, 0.025)), float(np.quantile(totals, 0.975))


def _positive_month_concentration(book: pd.DataFrame) -> float:
    monthly = book.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v181(
    primary_events: pd.DataFrame,
    event_summary: pd.DataFrame,
    continuous_book: pd.DataFrame,
    continuous_summary: pd.DataFrame,
    cfg: V181Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_scope = event_summary[
        event_summary["holding_bars"].eq(cfg.primary_holding_bars)
    ].set_index("scope")
    continuous_scope = continuous_summary.set_index("scope")
    event_low, event_high = _event_bootstrap(primary_events, cfg)
    continuous_low, continuous_high = _continuous_bootstrap(continuous_book, cfg)
    reversed_mean = float(
        (-primary_events["gross_return"] - cfg.primary_cost).mean()
    )
    real_mean = float(primary_events["primary_net_return"].mean())
    concentration = _positive_month_concentration(continuous_book)
    checks: dict[str, tuple[bool, float]] = {
        "four_hour_events_100": (
            primary_scope.loc["all", "events"] >= 100,
            float(primary_scope.loc["all", "events"]),
        ),
        "validation_events_20": (
            primary_scope.loc["validation", "events"] >= 20,
            float(primary_scope.loc["validation", "events"]),
        ),
        "holdout_events_25": (
            primary_scope.loc["holdout", "events"] >= 25,
            float(primary_scope.loc["holdout", "events"]),
        ),
        "event_development_primary_positive": (
            primary_scope.loc["development", "mean_primary_net_bp"] > 0,
            float(primary_scope.loc["development", "mean_primary_net_bp"]),
        ),
        "event_validation_primary_positive": (
            primary_scope.loc["validation", "mean_primary_net_bp"] > 0,
            float(primary_scope.loc["validation", "mean_primary_net_bp"]),
        ),
        "event_holdout_primary_positive": (
            primary_scope.loc["holdout", "mean_primary_net_bp"] > 0,
            float(primary_scope.loc["holdout", "mean_primary_net_bp"]),
        ),
        "event_full_stress_positive": (
            primary_scope.loc["all", "mean_stress_net_bp"] > 0,
            float(primary_scope.loc["all", "mean_stress_net_bp"]),
        ),
        "event_bootstrap_lower_positive": (event_low > 0, event_low * 10_000),
        "event_beats_reversed": (
            real_mean > reversed_mean,
            (real_mean - reversed_mean) * 10_000,
        ),
        "continuous_development_primary_positive": (
            continuous_scope.loc["development", "primary_net_sum_pct"] > 0,
            float(continuous_scope.loc["development", "primary_net_sum_pct"]),
        ),
        "continuous_validation_primary_positive": (
            continuous_scope.loc["validation", "primary_net_sum_pct"] > 0,
            float(continuous_scope.loc["validation", "primary_net_sum_pct"]),
        ),
        "continuous_holdout_primary_positive": (
            continuous_scope.loc["holdout", "primary_net_sum_pct"] > 0,
            float(continuous_scope.loc["holdout", "primary_net_sum_pct"]),
        ),
        "continuous_full_stress_positive": (
            continuous_scope.loc["all", "stress_net_sum_pct"] > 0,
            float(continuous_scope.loc["all", "stress_net_sum_pct"]),
        ),
        "continuous_bootstrap_lower_positive": (
            continuous_low > 0,
            continuous_low * 100,
        ),
        "two_hour_event_primary_positive": (
            event_summary.loc[
                event_summary["holding_bars"].eq(8)
                & event_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            > 0,
            float(
                event_summary.loc[
                    event_summary["holding_bars"].eq(8)
                    & event_summary["scope"].eq("all"),
                    "mean_primary_net_bp",
                ].iloc[0]
            ),
        ),
        "eight_hour_event_primary_positive": (
            event_summary.loc[
                event_summary["holding_bars"].eq(32)
                & event_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            > 0,
            float(
                event_summary.loc[
                    event_summary["holding_bars"].eq(32)
                    & event_summary["scope"].eq("all"),
                    "mean_primary_net_bp",
                ].iloc[0]
            ),
        ),
        "twelve_hour_event_primary_positive": (
            event_summary.loc[
                event_summary["holding_bars"].eq(48)
                & event_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            > 0,
            float(
                event_summary.loc[
                    event_summary["holding_bars"].eq(48)
                    & event_summary["scope"].eq("all"),
                    "mean_primary_net_bp",
                ].iloc[0]
            ),
        ),
        "positive_month_concentration_35": (
            concentration <= 0.35,
            concentration,
        ),
    }
    eligible = all(passed for passed, _ in checks.values())
    gates = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "check": name,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for name, (passed, value) in checks.items()
        ]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "events": len(primary_events),
                "event_mean_gross_bp": float(
                    primary_events["gross_return"].mean() * 10_000
                ),
                "event_mean_primary_net_bp": real_mean * 10_000,
                "event_bootstrap_95_low_bp": event_low * 10_000,
                "event_bootstrap_95_high_bp": event_high * 10_000,
                "continuous_gross_sum_pct": float(
                    continuous_book["gross_return"].sum() * 100
                ),
                "continuous_turnover": float(continuous_book["turnover"].sum()),
                "continuous_primary_net_sum_pct": float(
                    continuous_book["primary_net_return"].sum() * 100
                ),
                "continuous_stress_net_sum_pct": float(
                    continuous_book["stress_net_return"].sum() * 100
                ),
                "continuous_bootstrap_95_low_pct": continuous_low * 100,
                "continuous_bootstrap_95_high_pct": continuous_high * 100,
                "positive_month_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "forward_followup_candidate_only"
                    if eligible
                    else "reject_long_horizon_execution_extension"
                ),
            }
        ]
    )
    return gates, outcome


def _write_findings(
    outcome: pd.DataFrame,
    event_summary: pd.DataFrame,
    continuous_summary: pd.DataFrame,
    path: Path,
) -> None:
    text = [
        "# v18.1 Residual Dispersion Long-Horizon Execution Findings",
        "",
        f"Verdict: `{outcome['verdict'].iloc[0]}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event sleeves",
        "",
        event_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Continuous four-hour book",
        "",
        continuous_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "No event threshold or bucket membership was reselected from v18.0.",
        "No live, PaperLive, leverage, remote, application, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v181_residual_dispersion_long_horizon_execution(
    kline_root: Path = KLINE_ROOT,
    v180_report_root: Path = V180_REPORT_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V181Config = V181Config(),
) -> dict[str, Path]:
    close, _ = load_v178_market_data(kline_root)
    signals = pd.read_parquet(v180_report_root / "dispersion_signals.parquet")
    all_horizons = (cfg.primary_holding_bars, *cfg.diagnostic_holding_bars)
    events_by_horizon: dict[int, pd.DataFrame] = {}
    summaries: list[pd.DataFrame] = []
    for holding_bars in all_horizons:
        events = build_v180_events(
            signals,
            close,
            cfg,
            holding_bars=holding_bars,
        )
        events_by_horizon[holding_bars] = events
        summaries.append(summarize_event_sleeves(events, holding_bars))
    event_summary = pd.concat(summaries, ignore_index=True)
    primary_events = events_by_horizon[cfg.primary_holding_bars]
    valid_sources = set(primary_events["source_feature_time"])
    continuous_signals = signals[signals["source_feature_time"].isin(valid_sources)]
    continuous_book, sleeves = build_continuous_v181_book(
        continuous_signals, close, cfg
    )
    continuous_summary = summarize_continuous_v181(continuous_book)
    gates, outcome = audit_v181(
        primary_events,
        event_summary,
        continuous_book,
        continuous_summary,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "fixed_v180_signals.parquet",
        "events": root / "long_horizon_events.parquet",
        "event_summary": root / "event_horizon_summary.csv",
        "continuous_book": root / "continuous_four_hour_book.parquet",
        "continuous_sleeves": root / "continuous_four_hour_sleeves.parquet",
        "continuous_summary": root / "continuous_summary.csv",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    continuous_signals.to_parquet(paths["signals"], index=False)
    pd.concat(events_by_horizon.values(), ignore_index=True).to_parquet(
        paths["events"], index=False
    )
    event_summary.to_csv(paths["event_summary"], index=False)
    continuous_book.to_parquet(paths["continuous_book"], index=False)
    sleeves.to_parquet(paths["continuous_sleeves"], index=False)
    continuous_summary.to_csv(paths["continuous_summary"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, event_summary, continuous_summary, findings_path)
    return paths


__all__ = [
    "CANDIDATE",
    "V181Config",
    "audit_v181",
    "build_continuous_v181_book",
    "summarize_continuous_v181",
    "summarize_event_sleeves",
    "write_v181_residual_dispersion_long_horizon_execution",
]
