"""Direct and receiver-bucket reversal after premium-confirmed OI unwind."""
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
    build_v185_source_signals,
)
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    V187Config,
    build_v187_monthly_risk,
)
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_panel,
)
from pressure_graph.reports.v191_premium_innovation_feature_audit import (
    build_v191_premium_features,
)


REPORT_ROOT = Path("reports/v19_2_premium_innovation_unwind_reversal")
FINDINGS_PATH = Path(
    "docs/v192_premium_innovation_unwind_reversal_findings_2026_07_17.md"
)
DIRECT_CANDIDATE = "PIR1_BTC_PREMIUM_SHOCK_REVERSAL"
BUCKET_CANDIDATE = "PIR2_PREMIUM_RECEIVER_REVERSAL_BUCKET"
CANDIDATES = (DIRECT_CANDIDATE, BUCKET_CANDIDATE)


@dataclass(frozen=True)
class V192Config(V187Config):
    source_return_quantile: float = 0.85
    premium_z_threshold: float = 1.0
    direct_primary_cost: float = 0.0010
    direct_stress_cost: float = 0.0015
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 19_200


def build_v192_signals(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    premium: pd.DataFrame,
    cfg: V192Config = V192Config(),
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
    innovation_z, breadth = build_v191_premium_features(close, premium, cfg)
    base["btc_premium_innovation_z"] = base["feature_time"].map(innovation_z[BTC])
    base["aligned_btc_premium_z"] = (
        base["source_sign"] * base["btc_premium_innovation_z"]
    )
    base["premium_innovation_breadth"] = base["feature_time"].map(
        breadth["premium_innovation_breadth"]
    )
    base["premium_breadth_q70"] = base["feature_time"].map(
        breadth["breadth_q70"]
    )
    base["source_side"] = np.where(
        base["source_sign"].lt(0), "long_liquidation", "short_cover"
    )
    selected = base[
        base["aligned_btc_premium_z"].ge(cfg.premium_z_threshold)
    ].reset_index(drop=True)
    return selected, base.reset_index(drop=True), innovation_z, breadth


def build_v192_direct_events(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V192Config = V192Config(),
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
        direction = -float(signal.source_sign)
        underlying = exit_price / entry - 1.0
        gross = direction * underlying
        values = signal._asdict()
        values["candidate"] = DIRECT_CANDIDATE
        rows.append(
            {
                **values,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "ranking": "btc_premium_filter",
                "period": _period(source_time),
                "entry_day": source_time.strftime("%Y-%m-%d"),
                "entry_month": source_time.strftime("%Y-%m"),
                "trade_direction": direction,
                "btc_underlying_return": underlying,
                "gross_return": gross,
                "primary_net_return": gross - cfg.direct_primary_cost,
                "stress_net_return": gross - cfg.direct_stress_cost,
                "reversed_primary_net_return": -gross - cfg.direct_primary_cost,
            }
        )
    return pd.DataFrame(rows)


def _bucket_scores(
    timestamp: pd.Timestamp,
    source_sign: float,
    risk: pd.DataFrame,
    innovation_z: pd.DataFrame,
    cfg: V192Config,
) -> pd.DataFrame:
    local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
    names = [name for name in local.index.astype(str) if name in innovation_z.columns]
    if not names or timestamp not in innovation_z.index:
        return pd.DataFrame()
    frame = pd.DataFrame(index=names)
    frame["btc_beta"] = local.reindex(names)["btc_beta"].astype(float)
    frame["premium_innovation_z"] = innovation_z.loc[timestamp, names].astype(
        float
    )
    frame["aligned_premium_z"] = source_sign * frame["premium_innovation_z"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame[frame["aligned_premium_z"].ge(cfg.premium_z_threshold)]


def build_v192_bucket_events(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    innovation_z: pd.DataFrame,
    cfg: V192Config = V192Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
    ranking: str = "top",
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        scores = _bucket_scores(
            source_time, float(signal.source_sign), risk, innovation_z, cfg
        )
        if len(scores) < cfg.min_receiver_bucket:
            continue
        selected = scores.sort_values(
            "aligned_premium_z", ascending=ranking != "top"
        ).head(cfg.receiver_bucket_size)
        receivers = selected.index.astype(str).tolist()
        entry_time = source_time + pd.Timedelta(minutes=15 * entry_delay_bars)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        prices = close.loc[[entry_time, exit_time], [BTC, *receivers]]
        if prices.isna().any().any():
            continue
        future = prices.loc[exit_time] / prices.loc[entry_time] - 1.0
        direction = -float(signal.source_sign)
        weights = pd.Series(direction / len(receivers), index=receivers)
        beta = selected["btc_beta"].astype(float)
        hedge = -float((weights * beta).sum())
        normalizer = float(weights.abs().sum() + abs(hedge))
        gross = float(
            ((weights * future[receivers]).sum() + hedge * future[BTC]) / normalizer
        )
        values = signal._asdict()
        values["candidate"] = BUCKET_CANDIDATE
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
                    f"{name}:{selected.at[name, 'aligned_premium_z']:.8f}"
                    for name in receivers
                ),
                "trade_direction": direction,
                "btc_hedge_weight": hedge,
                "normalizer": normalizer,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
                "reversed_primary_net_return": -gross - cfg.primary_cost,
            }
        )
    return pd.DataFrame(rows)


def _combine(direct: pd.DataFrame, bucket: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([direct, bucket], ignore_index=True, sort=False).sort_values(
        ["entry_time", "candidate"]
    ).reset_index(drop=True)


def build_v192_events(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    innovation_z: pd.DataFrame,
    cfg: V192Config = V192Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    direct = build_v192_direct_events(
        signals, close, cfg, holding_bars, entry_delay_bars
    )
    bucket = build_v192_bucket_events(
        signals,
        risk,
        close,
        innovation_z,
        cfg,
        holding_bars,
        entry_delay_bars,
    )
    return _combine(direct, bucket)


def summarize_v192(events: pd.DataFrame) -> pd.DataFrame:
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


def summarize_v192_sides(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for side in ("long_liquidation", "short_cover"):
            sample = local[local["source_side"].eq(side)]
            rows.append(
                {
                    "candidate": candidate,
                    "source_side": side,
                    "events": len(sample),
                    "mean_gross_bp": float(sample["gross_return"].mean() * 10_000),
                    "mean_primary_net_bp": float(
                        sample["primary_net_return"].mean() * 10_000
                    ),
                }
            )
    return pd.DataFrame(rows)


def _random_bucket_contexts(
    bucket_events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    innovation_z: pd.DataFrame,
    cfg: V192Config,
) -> list[dict[str, object]]:
    contexts = []
    for event in bucket_events.itertuples(index=False):
        scores = _bucket_scores(
            pd.Timestamp(event.source_feature_time),
            float(event.source_sign),
            risk,
            innovation_z,
            cfg,
        )
        names = scores.index.astype(str).tolist()
        future = (
            close.loc[event.exit_time, [BTC, *names]]
            / close.loc[event.entry_time, [BTC, *names]]
            - 1.0
        )
        valid = [
            name
            for name in names
            if np.isfinite(future[name]) and np.isfinite(scores.at[name, "btc_beta"])
        ]
        if len(valid) < int(event.receiver_count):
            continue
        contexts.append(
            {
                "receiver_count": int(event.receiver_count),
                "trade_direction": float(event.trade_direction),
                "future": future.reindex(valid).to_numpy(dtype=float),
                "beta": scores.reindex(valid)["btc_beta"].to_numpy(dtype=float),
                "btc_future": float(future[BTC]),
            }
        )
    return contexts


def random_v192_controls(
    selected_direct: pd.DataFrame,
    base_direct: pd.DataFrame,
    bucket_events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    innovation_z: pd.DataFrame,
    cfg: V192Config = V192Config(),
) -> pd.DataFrame:
    selected_counts = selected_direct.groupby("entry_month").size().to_dict()
    base_by_month = {
        month: sample["primary_net_return"].to_numpy(dtype=float)
        for month, sample in base_direct.groupby("entry_month", sort=True)
    }
    bucket_contexts = _random_bucket_contexts(
        bucket_events, risk, close, innovation_z, cfg
    )
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for iteration in range(cfg.random_iterations):
        direct_values = []
        for month, count in selected_counts.items():
            pool = base_by_month[month]
            chosen = rng.choice(len(pool), size=int(count), replace=False)
            direct_values.extend(pool[chosen].tolist())
        bucket_values = []
        for context in bucket_contexts:
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
            bucket_values.append(gross - cfg.primary_cost)
        means = {
            DIRECT_CANDIDATE: float(np.mean(direct_values)),
            BUCKET_CANDIDATE: float(np.mean(bucket_values)),
        }
        counts = {
            DIRECT_CANDIDATE: len(direct_values),
            BUCKET_CANDIDATE: len(bucket_values),
        }
        for candidate in CANDIDATES:
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "events": counts[candidate],
                    "mean_primary_net_return": means[candidate],
                }
            )
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "events": max(counts.values()),
                "mean_primary_net_return": max(means.values()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    sample: pd.DataFrame, cfg: V192Config, offset: int
) -> tuple[float, float]:
    daily = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + offset)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_profit_concentration(sample: pd.DataFrame) -> float:
    monthly = sample.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v192(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V192Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"),
        "mean_primary_net_return",
    ].dropna()
    gate_rows = []
    outcomes = []
    minimums = {
        DIRECT_CANDIDATE: (100, 10, 25),
        BUCKET_CANDIDATE: (75, 8, 20),
    }
    for index, candidate in enumerate(CANDIDATES):
        scoped = summary[summary["candidate"].eq(candidate)].set_index("scope")
        sample = events[events["candidate"].eq(candidate)]
        low, high = _bootstrap(sample, cfg, index)
        real_mean = float(sample["primary_net_return"].mean())
        reversed_mean = float(sample["reversed_primary_net_return"].mean())
        delayed_mean = float(
            delayed_summary.loc[
                delayed_summary["candidate"].eq(candidate)
                & delayed_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            / 10_000
        )
        ranking_mean = float(
            ranking_summary.loc[
                ranking_summary["candidate"].eq(candidate)
                & ranking_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            / 10_000
        )
        diagnostic = diagnostics[
            diagnostics["candidate"].eq(candidate)
            & diagnostics["scope"].eq("all")
        ].set_index("diagnostic")
        horizon = horizons[
            horizons["candidate"].eq(candidate) & horizons["scope"].eq("all")
        ].set_index("holding_bars")
        sides = side_summary[side_summary["candidate"].eq(candidate)].set_index(
            "source_side"
        )
        percentile = float(family.le(real_mean).mean())
        concentration = _positive_profit_concentration(sample)
        full_min, validation_min, holdout_min = minimums[candidate]
        checks = {
            "full_events_minimum": (
                scoped.loc["all", "events"] >= full_min,
                float(scoped.loc["all", "events"]),
            ),
            "validation_events_minimum": (
                scoped.loc["validation", "events"] >= validation_min,
                float(scoped.loc["validation", "events"]),
            ),
            "holdout_events_minimum": (
                scoped.loc["holdout", "events"] >= holdout_min,
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
            "beats_ranking_control": (
                real_mean > ranking_mean,
                (real_mean - ranking_mean) * 10_000,
            ),
            "source_q90_positive": (
                diagnostic.loc["source_q90", "mean_primary_net_bp"] > 0,
                float(diagnostic.loc["source_q90", "mean_primary_net_bp"]),
            ),
            "broad_premium_positive": (
                diagnostic.loc["broad_premium", "mean_primary_net_bp"] > 0,
                float(diagnostic.loc["broad_premium", "mean_primary_net_bp"]),
            ),
            "holding_15m_positive": (
                horizon.loc[1, "mean_primary_net_bp"] > 0,
                float(horizon.loc[1, "mean_primary_net_bp"]),
            ),
            "holding_60m_positive": (
                horizon.loc[4, "mean_primary_net_bp"] > 0,
                float(horizon.loc[4, "mean_primary_net_bp"]),
            ),
            "long_liquidation_positive": (
                sides.loc["long_liquidation", "mean_primary_net_bp"] > 0,
                float(sides.loc["long_liquidation", "mean_primary_net_bp"]),
            ),
            "short_cover_positive": (
                sides.loc["short_cover", "mean_primary_net_bp"] > 0,
                float(sides.loc["short_cover", "mean_primary_net_bp"]),
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
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "ranking_control_primary_net_bp": ranking_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_premium_innovation_unwind_reversal"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_premium_innovation_unwind_reversal"
    )
    text = [
        "# v19.2 Premium-Innovation Unwind Reversal Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        side_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Premium values are exact completed closes with no forward filling.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v192_premium_innovation_unwind_reversal(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V192Config = V192Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    premium = load_v190_premium_panel(premium_root)
    signals, base, innovation_z, breadth = build_v192_signals(
        close, panels, premium, cfg
    )
    returns = close.pct_change(fill_method=None)
    risk = build_v187_monthly_risk(
        returns, base["feature_time"].min(), base["feature_time"].max(), cfg
    )
    events = build_v192_events(signals, risk, close, innovation_z, cfg)
    summary = summarize_v192(events)
    side_summary = summarize_v192_sides(events)
    delayed_events = build_v192_events(
        signals, risk, close, innovation_z, cfg, entry_delay_bars=1
    )
    delayed_summary = summarize_v192(delayed_events)

    selected_keys = set(signals["source_feature_time"])
    complement = base[~base["source_feature_time"].isin(selected_keys)]
    direct_control = build_v192_direct_events(complement, close, cfg)
    bucket_control = build_v192_bucket_events(
        signals, risk, close, innovation_z, cfg, ranking="bottom"
    )
    ranking_events = _combine(direct_control, bucket_control)
    ranking_summary = summarize_v192(ranking_events)

    diagnostic_frames = []
    q90_signals, _, _, _ = build_v192_signals(
        close, panels, premium, cfg, source_return_quantile=0.90
    )
    broad_signals = signals[
        signals["premium_innovation_breadth"].ge(signals["premium_breadth_q70"])
    ]
    for name, local_signals in (
        ("source_q90", q90_signals),
        ("broad_premium", broad_signals),
    ):
        local_summary = summarize_v192(
            build_v192_events(local_signals, risk, close, innovation_z, cfg)
        )
        local_summary["diagnostic"] = name
        diagnostic_frames.append(local_summary)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)

    horizon_frames = []
    for holding_bars in (1, 4):
        local_summary = summarize_v192(
            build_v192_events(
                signals,
                risk,
                close,
                innovation_z,
                cfg,
                holding_bars=holding_bars,
            )
        )
        local_summary["holding_bars"] = holding_bars
        horizon_frames.append(local_summary)
    horizons = pd.concat(horizon_frames, ignore_index=True)

    selected_direct = events[events["candidate"].eq(DIRECT_CANDIDATE)]
    base_direct = build_v192_direct_events(base, close, cfg)
    bucket_events = events[events["candidate"].eq(BUCKET_CANDIDATE)]
    random_controls = random_v192_controls(
        selected_direct,
        base_direct,
        bucket_events,
        risk,
        close,
        innovation_z,
        cfg,
    )
    gates, outcome = audit_v192(
        events,
        summary,
        side_summary,
        delayed_summary,
        ranking_summary,
        diagnostics,
        horizons,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "selected_source_signals.parquet",
        "base_signals": root / "base_q85_unwind_signals.parquet",
        "risk": root / "monthly_risk_estimates.parquet",
        "innovation_z": root / "premium_innovation_z.parquet",
        "breadth": root / "premium_innovation_breadth.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "side_summary": root / "source_side_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "ranking_events": root / "ranking_control_events.parquet",
        "ranking_summary": root / "ranking_control_summary.csv",
        "diagnostics": root / "diagnostic_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    base.to_parquet(paths["base_signals"], index=False)
    risk.to_parquet(paths["risk"], index=False)
    innovation_z.to_parquet(paths["innovation_z"])
    breadth.reset_index().to_parquet(paths["breadth"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    side_summary.to_csv(paths["side_summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    ranking_events.to_parquet(paths["ranking_events"], index=False)
    ranking_summary.to_csv(paths["ranking_summary"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    horizons.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, side_summary, findings_path)
    return paths


__all__ = [
    "BUCKET_CANDIDATE",
    "CANDIDATES",
    "DIRECT_CANDIDATE",
    "V192Config",
    "audit_v192",
    "build_v192_bucket_events",
    "build_v192_direct_events",
    "build_v192_events",
    "build_v192_signals",
    "random_v192_controls",
    "summarize_v192",
    "summarize_v192_sides",
    "write_v192_premium_innovation_unwind_reversal",
]
