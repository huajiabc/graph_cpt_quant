"""Extreme cross-sectional residual dispersion compression at 15 minutes."""
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
    build_monthly_btc_receiver_graph,
    load_v178_market_data,
)


REPORT_ROOT = Path("reports/v18_0_extreme_residual_dispersion_compression")
FINDINGS_PATH = Path(
    "docs/v180_extreme_residual_dispersion_compression_findings_2026_07_16.md"
)
CANDIDATE = "RDC1_EXTREME_RESIDUAL_DISPERSION_COMPRESSION"


@dataclass(frozen=True)
class V180Config(V178Config):
    dispersion_lookback_bars: int = 30 * 96
    dispersion_min_bars: int = 20 * 96
    dispersion_quantile: float = 0.975
    dispersion_lower_quantile: float = 0.10
    dispersion_upper_quantile: float = 0.90
    min_cross_section: int = 30
    bucket_size: int = 5
    cooldown_bars: int = 4
    holding_bars: int = 1
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_000


def _monthly_beta_map(graph: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    return {
        pd.Timestamp(month): local.set_index("receiver")["btc_beta"].astype(float)
        for month, local in graph.groupby("graph_month", sort=True)
    }


def build_v180_signals(
    returns: pd.DataFrame,
    graph: pd.DataFrame,
    cfg: V180Config = V180Config(),
    dispersion_quantile: float | None = None,
) -> pd.DataFrame:
    quantile = (
        cfg.dispersion_quantile
        if dispersion_quantile is None
        else dispersion_quantile
    )
    beta_map = _monthly_beta_map(graph)
    dispersion_frames: list[pd.DataFrame] = []
    for month, beta in beta_map.items():
        end = month + pd.offsets.MonthBegin(1)
        local_returns = returns[(returns.index >= month) & (returns.index < end)]
        names = [name for name in beta.index.astype(str) if name in local_returns]
        if not names:
            continue
        residual = local_returns[names].sub(
            local_returns[BTC].to_numpy()[:, None] * beta.reindex(names).to_numpy(),
            axis=0,
        )
        cross_section = residual.notna().sum(axis=1)
        dispersion = residual.quantile(
            cfg.dispersion_upper_quantile, axis=1
        ) - residual.quantile(cfg.dispersion_lower_quantile, axis=1)
        dispersion_frames.append(
            pd.DataFrame(
                {
                    "dispersion": dispersion,
                    "cross_section": cross_section,
                    "graph_month": month,
                },
                index=local_returns.index,
            )
        )
    if not dispersion_frames:
        return pd.DataFrame()
    state = pd.concat(dispersion_frames).sort_index()
    state["dispersion_threshold"] = (
        state["dispersion"]
        .shift(1)
        .rolling(
            cfg.dispersion_lookback_bars,
            min_periods=cfg.dispersion_min_bars,
        )
        .quantile(quantile)
    )
    state["eligible"] = (
        state["cross_section"].ge(cfg.min_cross_section)
        & state["dispersion"].ge(state["dispersion_threshold"])
    )
    accepted: list[pd.Timestamp] = []
    last_time: pd.Timestamp | None = None
    cooldown = pd.Timedelta(minutes=15 * cfg.cooldown_bars)
    for timestamp in state.index[state["eligible"]]:
        timestamp = pd.Timestamp(timestamp)
        if last_time is None or timestamp - last_time >= cooldown:
            accepted.append(timestamp)
            last_time = timestamp

    rows: list[dict[str, object]] = []
    for timestamp in accepted:
        beta = beta_map.get(_month(timestamp))
        if beta is None or timestamp not in returns.index:
            continue
        names = [name for name in beta.index.astype(str) if name in returns.columns]
        current = returns.reindex(index=[timestamp], columns=[BTC, *names]).iloc[0]
        residual = current[names] - beta.reindex(names) * float(current[BTC])
        residual = residual.dropna()
        if len(residual) < cfg.min_cross_section:
            continue
        laggards = residual.nsmallest(cfg.bucket_size)
        leaders = residual.nlargest(cfg.bucket_size)
        if len(laggards) < cfg.bucket_size or len(leaders) < cfg.bucket_size:
            continue
        laggard_names = laggards.index.astype(str).tolist()
        leader_names = leaders.index.astype(str).tolist()
        mean_laggard_beta = float(beta.reindex(laggard_names).mean())
        mean_leader_beta = float(beta.reindex(leader_names).mean())
        rows.append(
            {
                "feature_time": timestamp,
                "source_feature_time": timestamp,
                "graph_month": _month(timestamp),
                "dispersion": float(state.at[timestamp, "dispersion"]),
                "dispersion_threshold": float(
                    state.at[timestamp, "dispersion_threshold"]
                ),
                "dispersion_quantile": quantile,
                "cross_section": int(len(residual)),
                "laggards": "|".join(laggard_names),
                "leaders": "|".join(leader_names),
                "mean_laggard_beta": mean_laggard_beta,
                "mean_leader_beta": mean_leader_beta,
                "spread_beta": 0.5 * (mean_laggard_beta - mean_leader_beta),
                "mean_laggard_current_residual": float(laggards.mean()),
                "mean_leader_current_residual": float(leaders.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("feature_time").reset_index(drop=True)


def build_v180_events(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V180Config = V180Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        laggards = str(signal.laggards).split("|")
        leaders = str(signal.leaders).split("|")
        required = [BTC, *laggards, *leaders]
        if entry_time not in close.index or exit_time not in close.index:
            continue
        prices = close.loc[[entry_time, exit_time], required]
        if prices.isna().any().any():
            continue
        future = prices.loc[exit_time] / prices.loc[entry_time] - 1.0
        laggard_return = float(future[laggards].mean())
        leader_return = float(future[leaders].mean())
        spread_beta = float(signal.spread_beta)
        gross = (
            0.5 * (laggard_return - leader_return)
            - spread_beta * float(future[BTC])
        ) / (1.0 + abs(spread_beta))
        rows.append(
            {
                **signal._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "period": _period(source_time),
                "entry_day": source_time.strftime("%Y-%m-%d"),
                "entry_month": source_time.strftime("%Y-%m"),
                "candidate": CANDIDATE,
                "btc_future_return": float(future[BTC]),
                "mean_laggard_future_return": laggard_return,
                "mean_leader_future_return": leader_return,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
            }
        )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values("entry_time").reset_index(drop=True)
    return events


def summarize_v180(events: pd.DataFrame) -> pd.DataFrame:
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
                "mean_dispersion_bp": (
                    float(sample["dispersion"].mean() * 10_000)
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


def random_v180_controls(
    events: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V180Config = V180Config(),
) -> pd.DataFrame:
    beta_map = _monthly_beta_map(graph)
    contexts: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        beta = beta_map[pd.Timestamp(event.graph_month)]
        names = beta.index.astype(str).tolist()
        prices = close.reindex(
            index=[event.entry_time, event.exit_time], columns=[BTC, *names]
        )
        if prices[BTC].isna().any():
            continue
        future = prices.loc[event.exit_time] / prices.loc[event.entry_time] - 1.0
        valid = [
            name
            for name in names
            if np.isfinite(future.get(name, math.nan))
            and np.isfinite(beta.get(name, math.nan))
        ]
        if len(valid) < 2 * cfg.bucket_size:
            continue
        contexts.append(
            {
                "future": future.reindex(valid).to_numpy(dtype=float),
                "beta": beta.reindex(valid).to_numpy(dtype=float),
                "btc_future": float(future[BTC]),
            }
        )
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values: list[float] = []
        for context in contexts:
            chosen = rng.choice(
                len(context["future"]), size=2 * cfg.bucket_size, replace=False
            )
            laggard_indices = chosen[: cfg.bucket_size]
            leader_indices = chosen[cfg.bucket_size :]
            future = context["future"]
            beta = context["beta"]
            spread_beta = 0.5 * (
                float(np.mean(beta[laggard_indices]))
                - float(np.mean(beta[leader_indices]))
            )
            gross = (
                0.5
                * (
                    float(np.mean(future[laggard_indices]))
                    - float(np.mean(future[leader_indices]))
                )
                - spread_beta * float(context["btc_future"])
            ) / (1.0 + abs(spread_beta))
            values.append(gross - cfg.primary_cost)
        rows.append(
            {
                "iteration": iteration,
                "candidate": "RANDOM_RANK",
                "events": len(values),
                "mean_primary_net_return": (
                    float(np.mean(values)) if values else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    events: pd.DataFrame,
    cfg: V180Config,
) -> tuple[float, float]:
    daily = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in events.groupby("entry_day", sort=True)
    ]
    if not daily:
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_profit_concentration(events: pd.DataFrame) -> float:
    monthly = events.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v180(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V180Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = summary.set_index("scope")
    real_mean = float(events["primary_net_return"].mean())
    reversed_mean = float((-events["gross_return"] - cfg.primary_cost).mean())
    delayed_mean = float(
        delayed_summary.loc[
            delayed_summary["scope"].eq("all"), "mean_primary_net_bp"
        ].iloc[0]
        / 10_000
    )
    low, high = _bootstrap(events, cfg)
    random_values = random_controls["mean_primary_net_return"].dropna()
    random_percentile = float(random_values.le(real_mean).mean())
    local_sensitivity = sensitivity[sensitivity["scope"].eq("all")].set_index(
        "dispersion_quantile"
    )
    local_horizon = horizon_summary[horizon_summary["scope"].eq("all")].set_index(
        "holding_bars"
    )
    concentration = _positive_profit_concentration(events)
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
        "random_rank_percentile_95": (
            random_percentile >= 0.95,
            random_percentile,
        ),
        "beats_reversed_direction": (
            real_mean > reversed_mean,
            (real_mean - reversed_mean) * 10_000,
        ),
        "beats_one_bar_delay": (
            real_mean > delayed_mean,
            (real_mean - delayed_mean) * 10_000,
        ),
        "dispersion_q95_positive": (
            local_sensitivity.loc[0.95, "mean_primary_net_bp"] > 0,
            float(local_sensitivity.loc[0.95, "mean_primary_net_bp"]),
        ),
        "dispersion_q99_positive": (
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
                "mean_primary_net_bp": real_mean * 10_000,
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_rank_percentile": random_percentile,
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
                    else "reject_extreme_residual_dispersion_compression"
                ),
            }
        ]
    )
    return gates, outcome


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    text = [
        "# v18.0 Extreme Residual Dispersion Compression Findings",
        "",
        f"Verdict: `{outcome['verdict'].iloc[0]}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "All beta estimates and dispersion thresholds use prior completed bars.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v180_extreme_residual_dispersion_compression(
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V180Config = V180Config(),
) -> dict[str, Path]:
    close, _ = load_v178_market_data(kline_root)
    returns = close.pct_change(fill_method=None)
    graph = build_monthly_btc_receiver_graph(
        returns, returns.index.min(), returns.index.max(), cfg
    )
    signals = build_v180_signals(returns, graph, cfg)
    events = build_v180_events(signals, close, cfg)
    summary = summarize_v180(events)
    delayed_events = build_v180_events(signals, close, cfg, entry_delay_bars=1)
    delayed_summary = summarize_v180(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for quantile in (0.95, 0.99):
        local_signals = build_v180_signals(
            returns, graph, cfg, dispersion_quantile=quantile
        )
        local = summarize_v180(build_v180_events(local_signals, close, cfg))
        local["dispersion_quantile"] = quantile
        sensitivity_frames.append(local)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (2, 4):
        local = summarize_v180(
            build_v180_events(signals, close, cfg, holding_bars=holding_bars)
        )
        local["holding_bars"] = holding_bars
        horizon_frames.append(local)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v180_controls(events, graph, close, cfg)
    gates, outcome = audit_v180(
        events,
        summary,
        delayed_summary,
        sensitivity,
        horizon_summary,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "graph": root / "monthly_btc_beta_graph.parquet",
        "signals": root / "dispersion_signals.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "dispersion_threshold_sensitivity.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_rank_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    graph.to_parquet(paths["graph"], index=False)
    signals.to_parquet(paths["signals"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    horizon_summary.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "CANDIDATE",
    "V180Config",
    "audit_v180",
    "build_v180_events",
    "build_v180_signals",
    "random_v180_controls",
    "summarize_v180",
    "write_v180_extreme_residual_dispersion_compression",
]
