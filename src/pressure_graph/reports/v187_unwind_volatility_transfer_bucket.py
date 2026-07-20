"""Cross-asset volatility-transfer buckets after BTC unwind events."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month, _period
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    BTC,
    UNWIND,
    V185Config,
    build_v185_features,
    build_v185_source_signals,
)


REPORT_ROOT = Path("reports/v18_7_unwind_volatility_transfer_bucket")
FINDINGS_PATH = Path(
    "docs/v187_unwind_volatility_transfer_bucket_findings_2026_07_16.md"
)
OVERSHOOT_CANDIDATE = "VTR1_UNWIND_RESIDUAL_OVERSHOOT_REVERSAL"
STRESS_CANDIDATE = "VTR2_UNWIND_SYNCHRONIZED_STRESS_REVERSAL"
CANDIDATES = (OVERSHOOT_CANDIDATE, STRESS_CANDIDATE)


@dataclass(frozen=True)
class V187Config(V185Config):
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    receiver_bucket_size: int = 8
    min_receiver_bucket: int = 5
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_700


def build_v187_monthly_risk(
    returns: pd.DataFrame,
    first_month: pd.Timestamp,
    last_month: pd.Timestamp,
    cfg: V187Config = V187Config(),
) -> pd.DataFrame:
    start = _month(pd.Timestamp(first_month))
    end = _month(pd.Timestamp(last_month))
    months = pd.date_range(start, end, freq="MS")
    rows: list[dict[str, object]] = []
    for month in months:
        history = returns.loc[
            (returns.index >= month - pd.Timedelta(days=cfg.risk_lookback_days))
            & (returns.index < month)
        ]
        for receiver in sorted(set(returns.columns) - {BTC}):
            paired = history[[BTC, receiver]].dropna()
            if len(paired) < cfg.risk_min_samples or paired[BTC].var(ddof=1) <= 0:
                continue
            beta = float(
                paired[receiver].cov(paired[BTC]) / paired[BTC].var(ddof=1)
            )
            residual = paired[receiver] - beta * paired[BTC]
            residual_volatility = float(residual.std(ddof=1))
            return_volatility = float(paired[receiver].std(ddof=1))
            if residual_volatility <= 0 or return_volatility <= 0:
                continue
            rows.append(
                {
                    "risk_month": month,
                    "receiver": receiver,
                    "samples": len(paired),
                    "btc_beta": beta,
                    "residual_volatility": residual_volatility,
                    "return_volatility": return_volatility,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["risk_month", "receiver"]).reset_index(drop=True)
    return frame


def _receiver_scores(
    timestamp: pd.Timestamp,
    source_sign: float,
    candidate: str,
    risk: pd.DataFrame,
    returns: pd.DataFrame,
    flow: pd.DataFrame,
    oi_change: pd.DataFrame,
) -> pd.DataFrame:
    local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
    receivers = [
        receiver
        for receiver in local.index.astype(str)
        if receiver in returns.columns
    ]
    if not receivers or timestamp not in returns.index:
        return pd.DataFrame()
    beta = local.reindex(receivers)["btc_beta"].astype(float)
    residual_vol = local.reindex(receivers)["residual_volatility"].astype(float)
    return_vol = local.reindex(receivers)["return_volatility"].astype(float)
    alt_return = returns.loc[timestamp, receivers].astype(float)
    btc_return = float(returns.at[timestamp, BTC])
    residual = alt_return - beta * btc_return
    residual_z = residual / residual_vol
    aligned_residual = source_sign * residual_z
    scores = pd.DataFrame(
        {
            "receiver": receivers,
            "btc_beta": beta.to_numpy(dtype=float),
            "residual_z": residual_z.to_numpy(dtype=float),
            "aligned_residual_z": aligned_residual.to_numpy(dtype=float),
        }
    ).set_index("receiver")
    if candidate == OVERSHOOT_CANDIDATE:
        scores["score"] = scores["aligned_residual_z"]
        return scores.replace([np.inf, -np.inf], np.nan).dropna().query("score > 0")

    aligned_return = source_sign * alt_return / return_vol
    aligned_flow = source_sign * flow.loc[timestamp, receivers].astype(float)
    unwind_intensity = -oi_change.loc[timestamp, receivers].astype(float)
    scores["aligned_return_z"] = aligned_return.to_numpy(dtype=float)
    scores["aligned_flow"] = aligned_flow.to_numpy(dtype=float)
    scores["unwind_intensity"] = unwind_intensity.to_numpy(dtype=float)
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    scores = scores[
        scores["aligned_return_z"].gt(0)
        & scores["aligned_flow"].gt(0)
        & scores["unwind_intensity"].gt(0)
    ].copy()
    if scores.empty:
        return scores
    scores["return_rank"] = scores["aligned_return_z"].rank(pct=True)
    scores["flow_rank"] = scores["aligned_flow"].rank(pct=True)
    scores["unwind_rank"] = scores["unwind_intensity"].rank(pct=True)
    scores["score"] = scores[["return_rank", "flow_rank", "unwind_rank"]].mean(
        axis=1
    )
    return scores


def build_v187_events(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    flow: pd.DataFrame,
    oi_change: pd.DataFrame,
    cfg: V187Config = V187Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
    ranking: str = "top",
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    unwind_signals = signals[signals["kind"].eq(UNWIND)]
    for signal in unwind_signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        for candidate in CANDIDATES:
            scores = _receiver_scores(
                source_time,
                float(signal.source_sign),
                candidate,
                risk,
                returns,
                flow,
                oi_change,
            )
            if len(scores) < cfg.min_receiver_bucket:
                continue
            scores = scores.sort_values("score", ascending=ranking != "top")
            selected = scores.head(cfg.receiver_bucket_size)
            receivers = selected.index.astype(str).tolist()
            if entry_time not in close.index or exit_time not in close.index:
                continue
            prices = close.loc[[entry_time, exit_time], [BTC, *receivers]]
            if prices.isna().any().any():
                continue
            future = prices.loc[exit_time] / prices.loc[entry_time] - 1.0
            direction = -float(signal.source_sign)
            alt_weights = pd.Series(direction / len(receivers), index=receivers)
            beta = selected["btc_beta"].astype(float)
            hedge = -float((alt_weights * beta).sum())
            normalizer = float(alt_weights.abs().sum() + abs(hedge))
            gross = float(
                (
                    (alt_weights * future[receivers]).sum()
                    + hedge * float(future[BTC])
                )
                / normalizer
            )
            values = signal._asdict()
            values["candidate"] = candidate
            rows.append(
                {
                    **values,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_delay_bars": entry_delay_bars,
                    "holding_bars": horizon,
                    "ranking": ranking,
                    "risk_month": _month(source_time),
                    "period": _period(source_time),
                    "entry_day": source_time.strftime("%Y-%m-%d"),
                    "entry_month": source_time.strftime("%Y-%m"),
                    "eligible_receivers": len(scores),
                    "receiver_count": len(receivers),
                    "receivers": "|".join(receivers),
                    "receiver_scores": "|".join(
                        f"{name}:{selected.at[name, 'score']:.8f}"
                        for name in receivers
                    ),
                    "trade_direction": direction,
                    "btc_hedge_weight": hedge,
                    "normalizer": normalizer,
                    "gross_return": gross,
                    "primary_net_return": gross - cfg.primary_cost,
                    "stress_net_return": gross - cfg.stress_cost,
                }
            )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "candidate"]).reset_index(
            drop=True
        )
    return events


def summarize_v187(events: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_eligible_receivers": (
                        float(sample["eligible_receivers"].mean())
                        if len(sample)
                        else math.nan
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


def random_v187_receiver_controls(
    events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    flow: pd.DataFrame,
    oi_change: pd.DataFrame,
    cfg: V187Config = V187Config(),
) -> pd.DataFrame:
    contexts: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        scores = _receiver_scores(
            pd.Timestamp(event.source_feature_time),
            float(event.source_sign),
            str(event.candidate),
            risk,
            returns,
            flow,
            oi_change,
        )
        names = scores.index.astype(str).tolist()
        if len(names) < int(event.receiver_count):
            continue
        future = (
            close.loc[event.exit_time, [BTC, *names]]
            / close.loc[event.entry_time, [BTC, *names]]
            - 1.0
        )
        valid = [
            name
            for name in names
            if np.isfinite(future.get(name, math.nan))
            and np.isfinite(scores.at[name, "btc_beta"])
        ]
        if len(valid) < int(event.receiver_count):
            continue
        contexts.append(
            {
                "candidate": str(event.candidate),
                "receiver_count": int(event.receiver_count),
                "trade_direction": float(event.trade_direction),
                "future": future.reindex(valid).to_numpy(dtype=float),
                "beta": scores.reindex(valid)["btc_beta"].to_numpy(dtype=float),
                "btc_future": float(future[BTC]),
            }
        )
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in CANDIDATES}
        for context in contexts:
            count = int(context["receiver_count"])
            chosen = rng.choice(len(context["future"]), size=count, replace=False)
            weights = np.repeat(float(context["trade_direction"]) / count, count)
            hedge = -float(np.sum(weights * context["beta"][chosen]))
            normalizer = float(np.sum(np.abs(weights)) + abs(hedge))
            gross = float(
                (
                    np.sum(weights * context["future"][chosen])
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
    cfg: V187Config,
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


def audit_v187(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    bottom_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V187Config,
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
        delayed_mean = float(
            delayed_summary.loc[
                delayed_summary["candidate"].eq(candidate)
                & delayed_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            / 10_000
        )
        bottom_mean = float(
            bottom_summary.loc[
                bottom_summary["candidate"].eq(candidate)
                & bottom_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            / 10_000
        )
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
            "beats_one_bar_delay": (
                real_mean > delayed_mean,
                (real_mean - delayed_mean) * 10_000,
            ),
            "beats_bottom_ranked_bucket": (
                real_mean > bottom_mean,
                (real_mean - bottom_mean) * 10_000,
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
                "bottom_primary_net_bp": bottom_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_unwind_volatility_transfer_bucket"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_unwind_volatility_transfer_bucket"
    )
    text = [
        "# v18.7 Unwind Volatility-Transfer Bucket Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Risk estimates use only the prior 30 days for each calendar month.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v187_unwind_volatility_transfer_bucket(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V187Config = V187Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    returns, flow, oi_change, _ = build_v185_features(close, panels)
    signals = build_v185_source_signals(close, panels, cfg)
    unwind_signals = signals[signals["kind"].eq(UNWIND)].reset_index(drop=True)
    risk = build_v187_monthly_risk(
        returns,
        unwind_signals["feature_time"].min(),
        unwind_signals["feature_time"].max(),
        cfg,
    )
    events = build_v187_events(
        unwind_signals, risk, close, returns, flow, oi_change, cfg
    )
    summary = summarize_v187(events)
    delayed_events = build_v187_events(
        unwind_signals,
        risk,
        close,
        returns,
        flow,
        oi_change,
        cfg,
        entry_delay_bars=1,
    )
    delayed_summary = summarize_v187(delayed_events)
    bottom_events = build_v187_events(
        unwind_signals,
        risk,
        close,
        returns,
        flow,
        oi_change,
        cfg,
        ranking="bottom",
    )
    bottom_summary = summarize_v187(bottom_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for quantile in (0.85, 0.95):
        local_signals = build_v185_source_signals(
            close, panels, cfg, return_quantile=quantile
        )
        local = build_v187_events(
            local_signals,
            risk,
            close,
            returns,
            flow,
            oi_change,
            cfg,
        )
        local_summary = summarize_v187(local)
        local_summary["return_quantile"] = quantile
        sensitivity_frames.append(local_summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (1, 4):
        local = build_v187_events(
            unwind_signals,
            risk,
            close,
            returns,
            flow,
            oi_change,
            cfg,
            holding_bars=holding_bars,
        )
        local_summary = summarize_v187(local)
        local_summary["holding_bars"] = holding_bars
        horizon_frames.append(local_summary)
    horizons = pd.concat(horizon_frames, ignore_index=True)

    random_controls = random_v187_receiver_controls(
        events, risk, close, returns, flow, oi_change, cfg
    )
    gates, outcome = audit_v187(
        events,
        summary,
        delayed_summary,
        bottom_summary,
        sensitivity,
        horizons,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "unwind_source_signals.parquet",
        "risk": root / "monthly_risk_estimates.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "bottom_events": root / "bottom_candidate_events.parquet",
        "bottom_summary": root / "bottom_period_summary.csv",
        "sensitivity": root / "source_return_sensitivity.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_receiver_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    unwind_signals.to_parquet(paths["signals"], index=False)
    risk.to_parquet(paths["risk"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    bottom_events.to_parquet(paths["bottom_events"], index=False)
    bottom_summary.to_csv(paths["bottom_summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    horizons.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "CANDIDATES",
    "OVERSHOOT_CANDIDATE",
    "STRESS_CANDIDATE",
    "V187Config",
    "audit_v187",
    "build_v187_events",
    "build_v187_monthly_risk",
    "random_v187_receiver_controls",
    "summarize_v187",
    "write_v187_unwind_volatility_transfer_bucket",
]
