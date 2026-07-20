"""BTC continuation after high-breadth unwind cascades."""
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
    UNWIND,
    build_v185_source_signals,
)
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    V187Config,
    build_v187_monthly_risk,
)
from pressure_graph.reports.v189_volatility_breadth_regime_feature_audit import (
    build_v189_breadth,
)


REPORT_ROOT = Path("reports/v18_9_high_breadth_unwind_cascade")
FINDINGS_PATH = Path(
    "docs/v189_high_breadth_unwind_cascade_findings_2026_07_16.md"
)
CANDIDATE = "VBR1_HIGH_BREADTH_UNWIND_CASCADE_CONTINUATION"


@dataclass(frozen=True)
class V189Config(V187Config):
    source_return_quantile: float = 0.85
    breadth_quantile: float = 0.70
    holding_bars: int = 1
    primary_cost: float = 0.0010
    stress_cost: float = 0.0015
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_900


def build_v189_signals(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg: V189Config = V189Config(),
    source_return_quantile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_q = (
        cfg.source_return_quantile
        if source_return_quantile is None
        else source_return_quantile
    )
    base = build_v185_source_signals(
        close, panels, cfg, return_quantile=source_q
    )
    base = base[base["kind"].eq(UNWIND)].copy()
    returns = close.pct_change(fill_method=None)
    risk = build_v187_monthly_risk(
        returns,
        base["feature_time"].min(),
        base["feature_time"].max(),
        cfg,
    )
    breadth = build_v189_breadth(returns, risk)
    breadth["breadth_threshold"] = (
        breadth["volatility_breadth"]
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.breadth_quantile)
    )
    base["volatility_breadth"] = base["feature_time"].map(
        breadth["volatility_breadth"]
    )
    base["breadth_threshold"] = base["feature_time"].map(
        breadth["breadth_threshold"]
    )
    base["breadth_quantile"] = cfg.breadth_quantile
    base["transmitted_receivers"] = base["feature_time"].map(
        breadth["breadth_transmitted_receivers"]
    )
    base["valid_receivers"] = base["feature_time"].map(
        breadth["breadth_valid_receivers"]
    )
    selected = base[
        base["volatility_breadth"].ge(base["breadth_threshold"])
    ].reset_index(drop=True)
    return selected, base.reset_index(drop=True), risk, breadth.reset_index()


def build_v189_events(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V189Config = V189Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        entry = float(close.at[entry_time, BTC])
        exit_price = float(close.at[exit_time, BTC])
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
            continue
        direction = float(signal.source_sign)
        underlying = exit_price / entry - 1.0
        gross = direction * underlying
        values = signal._asdict()
        values["candidate"] = CANDIDATE
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
                "trade_direction": direction,
                "btc_underlying_return": underlying,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
                "reversed_primary_net_return": -gross - cfg.primary_cost,
            }
        )
    return pd.DataFrame(rows)


def summarize_v189(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(sample),
                "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                "active_months": (
                    sample["entry_month"].nunique() if len(sample) else 0
                ),
                "mean_breadth": (
                    float(sample["volatility_breadth"].mean())
                    if len(sample)
                    else math.nan
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


def random_v189_controls(
    selected_events: pd.DataFrame,
    base_events: pd.DataFrame,
    cfg: V189Config = V189Config(),
) -> pd.DataFrame:
    selected_counts = selected_events.groupby("entry_month").size().to_dict()
    base_by_month = {
        month: sample["primary_net_return"].to_numpy(dtype=float)
        for month, sample in base_events.groupby("entry_month", sort=True)
    }
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = []
        for month, count in selected_counts.items():
            pool = base_by_month[month]
            chosen = rng.choice(len(pool), size=int(count), replace=False)
            values.extend(pool[chosen].tolist())
        rows.append(
            {
                "iteration": iteration,
                "candidate": CANDIDATE,
                "events": len(values),
                "mean_primary_net_return": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    events: pd.DataFrame,
    cfg: V189Config,
) -> tuple[float, float]:
    daily = [
        sample["primary_net_return"].to_numpy(dtype=float)
        for _, sample in events.groupby("entry_day", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_profit_concentration(events: pd.DataFrame) -> float:
    monthly = events.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v189(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    complement_summary: pd.DataFrame,
    q90_summary: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V189Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = summary.set_index("scope")
    low, high = _bootstrap(events, cfg)
    real_mean = float(events["primary_net_return"].mean())
    reversed_mean = float(events["reversed_primary_net_return"].mean())
    delayed_mean = float(
        delayed_summary.loc[
            delayed_summary["scope"].eq("all"), "mean_primary_net_bp"
        ].iloc[0]
        / 10_000
    )
    complement_mean = float(
        complement_summary.loc[
            complement_summary["scope"].eq("all"), "mean_primary_net_bp"
        ].iloc[0]
        / 10_000
    )
    q90_mean = float(
        q90_summary.loc[q90_summary["scope"].eq("all"), "mean_primary_net_bp"].iloc[
            0
        ]
        / 10_000
    )
    horizon = horizons[horizons["scope"].eq("all")].set_index("holding_bars")
    percentile = float(
        random_controls["mean_primary_net_return"].le(real_mean).mean()
    )
    concentration = _positive_profit_concentration(events)
    checks: dict[str, tuple[bool, float]] = {
        "full_events_150": (
            scoped.loc["all", "events"] >= 150,
            float(scoped.loc["all", "events"]),
        ),
        "validation_events_20": (
            scoped.loc["validation", "events"] >= 20,
            float(scoped.loc["validation", "events"]),
        ),
        "holdout_events_40": (
            scoped.loc["holdout", "events"] >= 40,
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
        "random_percentile_95": (percentile >= 0.95, percentile),
        "beats_reversed_direction": (
            real_mean > reversed_mean,
            (real_mean - reversed_mean) * 10_000,
        ),
        "beats_one_bar_delay": (
            real_mean > delayed_mean,
            (real_mean - delayed_mean) * 10_000,
        ),
        "beats_non_high_breadth_complement": (
            real_mean > complement_mean,
            (real_mean - complement_mean) * 10_000,
        ),
        "source_q90_positive": (q90_mean > 0, q90_mean * 10_000),
        "holding_30m_positive": (
            horizon.loc[2, "mean_primary_net_bp"] > 0,
            float(horizon.loc[2, "mean_primary_net_bp"]),
        ),
        "holding_60m_positive": (
            horizon.loc[4, "mean_primary_net_bp"] > 0,
            float(horizon.loc[4, "mean_primary_net_bp"]),
        ),
        "positive_profit_concentration_35": (
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
                "events": len(events),
                "mean_gross_bp": float(events["gross_return"].mean() * 10_000),
                "mean_primary_net_bp": real_mean * 10_000,
                "mean_stress_net_bp": float(
                    events["stress_net_return"].mean() * 10_000
                ),
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_percentile": percentile,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "complement_primary_net_bp": complement_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_high_breadth_unwind_cascade"
                ),
            }
        ]
    )
    return gates, outcome


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = str(outcome.loc[0, "verdict"])
    text = [
        "# v18.9 High-Breadth Unwind Cascade Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The low-breadth candidate remained cancelled for inadequate coverage.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v189_high_breadth_unwind_cascade(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V189Config = V189Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    signals, base, risk, breadth = build_v189_signals(close, panels, cfg)
    events = build_v189_events(signals, close, cfg)
    summary = summarize_v189(events)
    delayed_events = build_v189_events(signals, close, cfg, entry_delay_bars=1)
    delayed_summary = summarize_v189(delayed_events)

    selected_keys = set(signals["source_feature_time"])
    complement_signals = base[~base["source_feature_time"].isin(selected_keys)]
    complement_events = build_v189_events(complement_signals, close, cfg)
    complement_summary = summarize_v189(complement_events)

    q90_signals, _, _, _ = build_v189_signals(
        close, panels, cfg, source_return_quantile=0.90
    )
    q90_events = build_v189_events(q90_signals, close, cfg)
    q90_summary = summarize_v189(q90_events)

    horizon_frames = []
    for holding_bars in (2, 4):
        local = summarize_v189(
            build_v189_events(signals, close, cfg, holding_bars=holding_bars)
        )
        local["holding_bars"] = holding_bars
        horizon_frames.append(local)
    horizons = pd.concat(horizon_frames, ignore_index=True)

    base_events = build_v189_events(base, close, cfg)
    random_controls = random_v189_controls(events, base_events, cfg)
    gates, outcome = audit_v189(
        events,
        summary,
        delayed_summary,
        complement_summary,
        q90_summary,
        horizons,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "selected_source_signals.parquet",
        "base_signals": root / "base_q85_unwind_signals.parquet",
        "risk": root / "monthly_risk_estimates.parquet",
        "breadth": root / "volatility_breadth.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "complement_events": root / "complement_events.parquet",
        "complement_summary": root / "complement_summary.csv",
        "q90_summary": root / "q90_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    base.to_parquet(paths["base_signals"], index=False)
    risk.to_parquet(paths["risk"], index=False)
    breadth.to_parquet(paths["breadth"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    complement_events.to_parquet(paths["complement_events"], index=False)
    complement_summary.to_csv(paths["complement_summary"], index=False)
    q90_summary.to_csv(paths["q90_summary"], index=False)
    horizons.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "CANDIDATE",
    "V189Config",
    "audit_v189",
    "build_v189_events",
    "build_v189_signals",
    "random_v189_controls",
    "summarize_v189",
    "write_v189_high_breadth_unwind_cascade",
]
