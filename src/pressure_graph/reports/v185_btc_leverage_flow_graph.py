"""BTC leverage-flow impulses mapped into directed alt residual buckets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month, _period
from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    V180Config,
)
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)


REPORT_ROOT = Path("reports/v18_5_btc_leverage_flow_graph")
FINDINGS_PATH = Path("docs/v185_btc_leverage_flow_graph_findings_2026_07_16.md")
BTC = "BTCUSDT"
BUILD = "build"
UNWIND = "unwind"
BUILD_CANDIDATE = "LFG1_BTC_BUILD_DIRECTED_RESIDUAL_BUCKET"
UNWIND_CANDIDATE = "LFG2_BTC_UNWIND_DIRECTED_RESIDUAL_BUCKET"
CANDIDATES = (BUILD_CANDIDATE, UNWIND_CANDIDATE)
CANDIDATE_BY_KIND = {BUILD: BUILD_CANDIDATE, UNWIND: UNWIND_CANDIDATE}


@dataclass(frozen=True)
class V185Config(V180Config):
    source_lookback_bars: int = 30 * 96
    source_min_bars: int = 20 * 96
    source_return_quantile: float = 0.90
    source_flow_quantile: float = 0.75
    source_oi_tail_quantile: float = 0.70
    min_metric_breadth: int = 40
    cooldown_bars: int = 4
    graph_lookback_days: int = 30
    graph_min_samples: int = 2_000
    receiver_bucket_size: int = 8
    min_receiver_bucket: int = 5
    holding_bars: int = 2
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_500


def build_v185_features(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    returns = close.pct_change(fill_method=None)
    taker_ratio = panels["sum_taker_long_short_vol_ratio"].where(
        panels["sum_taker_long_short_vol_ratio"].gt(0)
    )
    open_interest = panels["sum_open_interest"].where(
        panels["sum_open_interest"].gt(0)
    )
    flow = np.log(taker_ratio)
    oi_change = np.log(open_interest).diff()
    impulses = {
        BUILD: flow * oi_change.clip(lower=0),
        UNWIND: flow * (-oi_change.clip(upper=0)),
    }
    return returns, flow, oi_change, impulses


def build_v185_source_signals(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg: V185Config = V185Config(),
    return_quantile: float | None = None,
) -> pd.DataFrame:
    returns, flow, oi_change, impulses = build_v185_features(close, panels)
    btc_return = returns[BTC]
    btc_flow = flow[BTC]
    btc_oi = oi_change[BTC]
    breadth = (
        panels["sum_taker_long_short_vol_ratio"].notna()
        & panels["sum_open_interest"].gt(0)
    ).sum(axis=1)
    quantile = (
        cfg.source_return_quantile if return_quantile is None else return_quantile
    )
    absolute_return = btc_return.abs()
    absolute_flow = btc_flow.abs()
    return_threshold = (
        absolute_return.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(quantile)
    )
    flow_threshold = (
        absolute_flow.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.source_flow_quantile)
    )
    oi_high = (
        btc_oi.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.source_oi_tail_quantile)
    )
    oi_low = (
        btc_oi.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(1.0 - cfg.source_oi_tail_quantile)
    )
    direction = np.sign(btc_return)
    base = (
        absolute_return.ge(return_threshold)
        & (direction * btc_flow).ge(flow_threshold)
        & direction.ne(0)
        & breadth.ge(cfg.min_metric_breadth)
    )
    rows: list[dict[str, object]] = []
    for kind, eligible in (
        (BUILD, base & btc_oi.ge(oi_high)),
        (UNWIND, base & btc_oi.le(oi_low)),
    ):
        accepted: list[pd.Timestamp] = []
        last_time: pd.Timestamp | None = None
        cooldown = pd.Timedelta(minutes=15 * cfg.cooldown_bars)
        for timestamp in eligible.index[eligible.fillna(False)]:
            timestamp = pd.Timestamp(timestamp)
            if last_time is None or timestamp - last_time >= cooldown:
                accepted.append(timestamp)
                last_time = timestamp
        for timestamp in accepted:
            rows.append(
                {
                    "feature_time": timestamp,
                    "source_feature_time": timestamp,
                    "kind": kind,
                    "candidate": CANDIDATE_BY_KIND[kind],
                    "btc_return_15m": float(btc_return.at[timestamp]),
                    "btc_flow": float(btc_flow.at[timestamp]),
                    "btc_oi_change": float(btc_oi.at[timestamp]),
                    "btc_impulse": float(impulses[kind].at[timestamp, BTC]),
                    "source_sign": float(np.sign(btc_flow.at[timestamp])),
                    "metric_breadth": int(breadth.at[timestamp]),
                    "return_threshold": float(return_threshold.at[timestamp]),
                    "flow_threshold": float(flow_threshold.at[timestamp]),
                    "oi_threshold": float(
                        oi_high.at[timestamp] if kind == BUILD else oi_low.at[timestamp]
                    ),
                    "return_quantile": quantile,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_time", "kind"]).reset_index(
        drop=True
    )


def build_monthly_v185_graph(
    returns: pd.DataFrame,
    impulses: dict[str, pd.DataFrame],
    first_month: pd.Timestamp,
    last_month: pd.Timestamp,
    cfg: V185Config = V185Config(),
) -> pd.DataFrame:
    start = pd.Timestamp(first_month).tz_convert("UTC").replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    end = pd.Timestamp(last_month).tz_convert("UTC").replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    months = pd.date_range(start, end, freq="MS")
    alts = sorted(set(returns.columns) - {BTC})
    rows: list[dict[str, object]] = []
    for month in months:
        history_mask = (returns.index >= month - pd.Timedelta(days=cfg.graph_lookback_days)) & (
            returns.index < month
        )
        history_returns = returns.loc[history_mask]
        for kind in (BUILD, UNWIND):
            local_rows: list[dict[str, object]] = []
            history_impulse = impulses[kind].loc[history_mask]
            for alt in alts:
                beta_pair = history_returns[[BTC, alt]].dropna()
                if (
                    len(beta_pair) < cfg.graph_min_samples
                    or beta_pair[BTC].var(ddof=1) <= 0
                ):
                    continue
                beta = float(
                    beta_pair[alt].cov(beta_pair[BTC])
                    / beta_pair[BTC].var(ddof=1)
                )
                residual = history_returns[alt] - beta * history_returns[BTC]
                forward = pd.DataFrame(
                    {
                        "source": history_impulse[BTC],
                        "residual_next": residual.shift(-1),
                    }
                ).dropna()
                reverse = pd.DataFrame(
                    {
                        "alt_source": history_impulse[alt],
                        "btc_next": history_returns[BTC].shift(-1),
                    }
                ).dropna()
                if (
                    len(forward) < cfg.graph_min_samples
                    or len(reverse) < cfg.graph_min_samples
                ):
                    continue
                forward_corr = float(
                    forward["source"].corr(
                        forward["residual_next"], method="spearman"
                    )
                )
                reverse_corr = float(
                    reverse["alt_source"].corr(reverse["btc_next"], method="spearman")
                )
                advantage = abs(forward_corr) - abs(reverse_corr)
                local_rows.append(
                    {
                        "graph_month": month,
                        "kind": kind,
                        "receiver": alt,
                        "forward_samples": len(forward),
                        "reverse_samples": len(reverse),
                        "btc_beta": beta,
                        "forward_correlation": forward_corr,
                        "reverse_correlation": reverse_corr,
                        "direction_advantage": advantage,
                        "edge_score": abs(forward_corr) + 0.5 * advantage,
                        "edge_sign": float(np.sign(forward_corr)),
                    }
                )
            local = pd.DataFrame(local_rows)
            selected = (
                local[local["direction_advantage"].gt(0)]
                .sort_values(
                    ["edge_score", "direction_advantage"], ascending=False
                )
                .head(cfg.receiver_bucket_size)
                if not local.empty
                else pd.DataFrame()
            )
            ranks = {
                receiver: rank
                for rank, receiver in enumerate(
                    selected.get("receiver", pd.Series(dtype=str)).astype(str), start=1
                )
            }
            for row in local_rows:
                receiver = str(row["receiver"])
                row["selected"] = receiver in ranks
                row["receiver_rank"] = ranks.get(receiver, math.nan)
                rows.append(row)
    graph = pd.DataFrame(rows)
    if not graph.empty:
        graph = graph.sort_values(
            ["graph_month", "kind", "selected", "receiver_rank", "receiver"],
            ascending=[True, True, False, True, True],
        ).reset_index(drop=True)
    return graph


def _selected_edges(
    graph: pd.DataFrame,
    timestamp: pd.Timestamp,
    kind: str,
) -> pd.DataFrame:
    return graph[
        graph["graph_month"].eq(_month(timestamp))
        & graph["kind"].eq(kind)
        & graph["selected"].astype(bool)
    ].sort_values("receiver_rank")


def build_v185_events(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V185Config = V185Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        edges = _selected_edges(graph, source_time, str(signal.kind))
        if len(edges) < cfg.min_receiver_bucket:
            continue
        receivers = edges["receiver"].astype(str).tolist()
        required = [BTC, *receivers]
        if entry_time not in close.index or exit_time not in close.index:
            continue
        prices = close.loc[[entry_time, exit_time], required]
        if prices.isna().any().any():
            continue
        future = prices.loc[exit_time] / prices.loc[entry_time] - 1.0
        edge_sign = edges.set_index("receiver")["edge_sign"].astype(float)
        beta = edges.set_index("receiver")["btc_beta"].astype(float)
        alt_weights = (
            float(signal.source_sign) * edge_sign.reindex(receivers) / len(receivers)
        )
        btc_hedge_weight = -float(
            (alt_weights * beta.reindex(receivers)).sum()
        )
        normalizer = float(alt_weights.abs().sum() + abs(btc_hedge_weight))
        gross = float(
            (
                (alt_weights * future[receivers]).sum()
                + btc_hedge_weight * float(future[BTC])
            )
            / normalizer
        )
        rows.append(
            {
                **signal._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "graph_month": _month(source_time),
                "period": _period(source_time),
                "entry_day": source_time.strftime("%Y-%m-%d"),
                "entry_month": source_time.strftime("%Y-%m"),
                "receiver_count": len(receivers),
                "receivers": "|".join(receivers),
                "edge_signs": "|".join(
                    f"{name}:{int(edge_sign[name]):+d}" for name in receivers
                ),
                "btc_hedge_weight": btc_hedge_weight,
                "normalizer": normalizer,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
                "reversed_primary_net_return": -gross - cfg.primary_cost,
            }
        )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "candidate"]).reset_index(drop=True)
    return events


def summarize_v185(events: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_receivers": (
                        float(sample["receiver_count"].mean())
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


def _random_contexts(
    events: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        local = graph[
            graph["graph_month"].eq(pd.Timestamp(event.graph_month))
            & graph["kind"].eq(str(event.kind))
            & graph["direction_advantage"].gt(0)
        ].set_index("receiver")
        names = local.index.astype(str).tolist()
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
            and np.isfinite(local.at[name, "btc_beta"])
            and local.at[name, "edge_sign"] != 0
        ]
        if len(valid) < int(event.receiver_count):
            continue
        contexts.append(
            {
                "candidate": str(event.candidate),
                "receiver_count": int(event.receiver_count),
                "source_sign": float(event.source_sign),
                "future": future.reindex(valid).to_numpy(dtype=float),
                "beta": local.reindex(valid)["btc_beta"].to_numpy(dtype=float),
                "edge_sign": local.reindex(valid)["edge_sign"].to_numpy(dtype=float),
                "btc_future": float(future[BTC]),
            }
        )
    return contexts


def random_v185_controls(
    events: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V185Config = V185Config(),
) -> pd.DataFrame:
    contexts = _random_contexts(events, graph, close)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in CANDIDATES}
        for context in contexts:
            chosen = rng.choice(
                len(context["future"]),
                size=int(context["receiver_count"]),
                replace=False,
            )
            alt_weights = (
                float(context["source_sign"])
                * context["edge_sign"][chosen]
                / len(chosen)
            )
            hedge = -float(np.sum(alt_weights * context["beta"][chosen]))
            normalizer = float(np.sum(np.abs(alt_weights)) + abs(hedge))
            gross = float(
                (
                    np.sum(alt_weights * context["future"][chosen])
                    + hedge * float(context["btc_future"])
                )
                / normalizer
            )
            values[str(context["candidate"])].append(gross - cfg.primary_cost)
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
    cfg: V185Config,
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


def audit_v185(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V185Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
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
            sensitivity["candidate"].eq(candidate) & sensitivity["scope"].eq("all")
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
                    else "reject_btc_leverage_flow_graph"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_btc_leverage_flow_graph"
    )
    text = [
        "# v18.5 BTC Leverage-Flow Directed Graph Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Every metric timestamp is an exact completed 15-minute archive point.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v185_btc_leverage_flow_graph(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V185Config = V185Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    returns, _, _, impulses = build_v185_features(close, panels)
    signals = build_v185_source_signals(close, panels, cfg)
    graph = build_monthly_v185_graph(
        returns,
        impulses,
        signals["feature_time"].min(),
        signals["feature_time"].max(),
        cfg,
    )
    events = build_v185_events(signals, graph, close, cfg)
    summary = summarize_v185(events)
    delayed_events = build_v185_events(
        signals, graph, close, cfg, entry_delay_bars=1
    )
    delayed_summary = summarize_v185(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for quantile in (0.85, 0.95):
        local_signals = build_v185_source_signals(
            close, panels, cfg, return_quantile=quantile
        )
        local = summarize_v185(build_v185_events(local_signals, graph, close, cfg))
        local["return_quantile"] = quantile
        sensitivity_frames.append(local)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (1, 4):
        local = summarize_v185(
            build_v185_events(
                signals, graph, close, cfg, holding_bars=holding_bars
            )
        )
        local["holding_bars"] = holding_bars
        horizon_frames.append(local)
    horizons = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v185_controls(events, graph, close, cfg)
    gates, outcome = audit_v185(
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
        "graph": root / "monthly_leverage_flow_graph.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "source_return_sensitivity.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_receiver_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    graph.to_parquet(paths["graph"], index=False)
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
    "V185Config",
    "audit_v185",
    "build_monthly_v185_graph",
    "build_v185_events",
    "build_v185_features",
    "build_v185_source_signals",
    "random_v185_controls",
    "summarize_v185",
    "write_v185_btc_leverage_flow_graph",
]
