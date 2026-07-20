"""BTC and altcoin reversal after top-trader absorption during unwind events."""
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


REPORT_ROOT = Path("reports/v18_8_top_trader_absorption")
FINDINGS_PATH = Path("docs/v188_top_trader_absorption_findings_2026_07_16.md")
DIRECT_CANDIDATE = "TDA1_BTC_TOPTRADER_ABSORPTION_REVERSAL"
BUCKET_CANDIDATE = "TDA2_ALT_TOPTRADER_ABSORPTION_BUCKET"
CANDIDATES = (DIRECT_CANDIDATE, BUCKET_CANDIDATE)


@dataclass(frozen=True)
class V188Config(V187Config):
    source_return_quantile: float = 0.85
    absorption_quantile: float = 0.50
    direct_primary_cost: float = 0.0010
    direct_stress_cost: float = 0.0015
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_800


def build_v188_features(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame | pd.Series]:
    returns = close.pct_change(fill_method=None)
    flow = np.log(
        panels["sum_taker_long_short_vol_ratio"].where(
            panels["sum_taker_long_short_vol_ratio"].gt(0)
        )
    )
    oi_change = np.log(
        panels["sum_open_interest"].where(panels["sum_open_interest"].gt(0))
    ).diff()
    top_position = np.log(
        panels["sum_toptrader_long_short_ratio"].where(
            panels["sum_toptrader_long_short_ratio"].gt(0)
        )
    )
    top_change = top_position.diff()
    btc_direction = np.sign(returns[BTC])
    btc_absorption = -btc_direction * top_change[BTC]
    return {
        "returns": returns,
        "flow": flow,
        "oi_change": oi_change,
        "top_change": top_change,
        "btc_absorption": btc_absorption,
    }


def build_v188_signals(
    close: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    cfg: V188Config = V188Config(),
    source_return_quantile: float | None = None,
    absorption_quantile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_q = (
        cfg.source_return_quantile
        if source_return_quantile is None
        else source_return_quantile
    )
    absorption_q = (
        cfg.absorption_quantile
        if absorption_quantile is None
        else absorption_quantile
    )
    base = build_v185_source_signals(
        close, panels, cfg, return_quantile=source_q
    )
    base = base[base["kind"].eq(UNWIND)].copy()
    features = build_v188_features(close, panels)
    btc_absorption = features["btc_absorption"]
    threshold = (
        btc_absorption.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(absorption_q)
    )
    base["btc_toptrader_absorption"] = base["feature_time"].map(btc_absorption)
    base["absorption_threshold"] = base["feature_time"].map(threshold)
    base["absorption_quantile"] = absorption_q
    selected = base[
        base["btc_toptrader_absorption"].ge(base["absorption_threshold"])
    ].reset_index(drop=True)
    return selected, base.reset_index(drop=True)


def build_v188_direct_events(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V188Config = V188Config(),
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
                "ranking": "absorption_filter",
                "period": _period(source_time),
                "entry_day": source_time.strftime("%Y-%m-%d"),
                "entry_month": source_time.strftime("%Y-%m"),
                "trade_direction": direction,
                "btc_underlying_return": underlying,
                "gross_return": gross,
                "primary_net_return": gross - cfg.direct_primary_cost,
                "stress_net_return": gross - cfg.direct_stress_cost,
            }
        )
    return pd.DataFrame(rows)


def _bucket_scores(
    timestamp: pd.Timestamp,
    source_sign: float,
    risk: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
) -> pd.DataFrame:
    local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
    names = local.index.astype(str).tolist()
    if not names:
        return pd.DataFrame()
    returns = features["returns"]
    flow = features["flow"]
    oi_change = features["oi_change"]
    top_change = features["top_change"]
    frame = pd.DataFrame(index=names)
    frame["btc_beta"] = local.reindex(names)["btc_beta"].astype(float)
    frame["aligned_return_z"] = (
        source_sign
        * returns.loc[timestamp, names].astype(float)
        / local.reindex(names)["return_volatility"].astype(float)
    )
    frame["aligned_flow"] = source_sign * flow.loc[timestamp, names].astype(float)
    frame["unwind_intensity"] = -oi_change.loc[timestamp, names].astype(float)
    frame["toptrader_absorption"] = (
        -source_sign * top_change.loc[timestamp, names].astype(float)
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    feature_columns = (
        "aligned_return_z",
        "aligned_flow",
        "unwind_intensity",
        "toptrader_absorption",
    )
    frame = frame[frame[list(feature_columns)].gt(0).all(axis=1)].copy()
    if frame.empty:
        return frame
    rank_columns = []
    for column in feature_columns:
        rank_name = f"{column}_rank"
        frame[rank_name] = frame[column].rank(pct=True)
        rank_columns.append(rank_name)
    frame["score"] = frame[rank_columns].mean(axis=1)
    return frame


def build_v188_bucket_events(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
    cfg: V188Config = V188Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
    ranking: str = "top",
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        scores = _bucket_scores(
            source_time, float(signal.source_sign), risk, features
        )
        if len(scores) < cfg.min_receiver_bucket:
            continue
        selected = scores.sort_values("score", ascending=ranking != "top").head(
            cfg.receiver_bucket_size
        )
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
        hedge = -float((weights * selected["btc_beta"]).sum())
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
                    f"{name}:{selected.at[name, 'score']:.8f}" for name in receivers
                ),
                "trade_direction": direction,
                "btc_hedge_weight": hedge,
                "normalizer": normalizer,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
            }
        )
    return pd.DataFrame(rows)


def summarize_v188(events: pd.DataFrame) -> pd.DataFrame:
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


def _bucket_random_contexts(
    events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        scores = _bucket_scores(
            pd.Timestamp(event.source_feature_time),
            float(event.source_sign),
            risk,
            features,
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


def random_v188_controls(
    selected_direct: pd.DataFrame,
    base_direct: pd.DataFrame,
    bucket_events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
    cfg: V188Config = V188Config(),
) -> pd.DataFrame:
    selected_counts = selected_direct.groupby("entry_month").size().to_dict()
    base_by_month = {
        month: sample["primary_net_return"].to_numpy(dtype=float)
        for month, sample in base_direct.groupby("entry_month", sort=True)
    }
    bucket_contexts = _bucket_random_contexts(
        bucket_events, risk, close, features
    )
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
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
    sample: pd.DataFrame,
    cfg: V188Config,
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


def audit_v188(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V188Config,
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
        ranking_mean = float(
            ranking_summary.loc[
                ranking_summary["candidate"].eq(candidate)
                & ranking_summary["scope"].eq("all"),
                "mean_primary_net_bp",
            ].iloc[0]
            / 10_000
        )
        local_diagnostics = diagnostics[
            diagnostics["candidate"].eq(candidate)
            & diagnostics["scope"].eq("all")
        ].set_index("diagnostic")
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
            "validation_events_10": (
                scoped.loc["validation", "events"] >= 10,
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
            "beats_ranking_control": (
                real_mean > ranking_mean,
                (real_mean - ranking_mean) * 10_000,
            ),
            "source_q90_positive": (
                local_diagnostics.loc["source_q90", "mean_primary_net_bp"] > 0,
                float(
                    local_diagnostics.loc["source_q90", "mean_primary_net_bp"]
                ),
            ),
            "absorption_q55_positive": (
                local_diagnostics.loc[
                    "absorption_q55", "mean_primary_net_bp"
                ]
                > 0,
                float(
                    local_diagnostics.loc[
                        "absorption_q55", "mean_primary_net_bp"
                    ]
                ),
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
                "ranking_control_primary_net_bp": ranking_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_top_trader_absorption"
                ),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _combine_events(direct: pd.DataFrame, bucket: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([direct, bucket], ignore_index=True, sort=False).sort_values(
        ["entry_time", "candidate"]
    ).reset_index(drop=True)


def _build_both(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame | pd.Series],
    cfg: V188Config,
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    direct = build_v188_direct_events(
        signals, close, cfg, holding_bars, entry_delay_bars
    )
    bucket = build_v188_bucket_events(
        signals,
        risk,
        close,
        features,
        cfg,
        holding_bars,
        entry_delay_bars,
    )
    return _combine_events(direct, bucket)


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if outcome["eligible"].any()
        else "reject_top_trader_absorption"
    )
    text = [
        "# v18.8 Top-Trader Absorption Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Primary q85/q50 thresholds were selected from feature coverage only.",
        "No live, PaperLive, application, leverage, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v188_top_trader_absorption(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V188Config = V188Config(),
) -> dict[str, Path]:
    close, panels = load_v184_exact_panels(metrics_root, kline_root)
    features = build_v188_features(close, panels)
    signals, base_signals = build_v188_signals(close, panels, cfg)
    risk = build_v187_monthly_risk(
        features["returns"],
        signals["feature_time"].min(),
        signals["feature_time"].max(),
        cfg,
    )
    events = _build_both(signals, risk, close, features, cfg)
    summary = summarize_v188(events)
    delayed_events = _build_both(
        signals, risk, close, features, cfg, entry_delay_bars=1
    )
    delayed_summary = summarize_v188(delayed_events)

    selected_keys = set(signals["source_feature_time"])
    complement = base_signals[
        ~base_signals["source_feature_time"].isin(selected_keys)
    ]
    direct_control = build_v188_direct_events(complement, close, cfg)
    bucket_control = build_v188_bucket_events(
        signals, risk, close, features, cfg, ranking="bottom"
    )
    ranking_events = _combine_events(direct_control, bucket_control)
    ranking_summary = summarize_v188(ranking_events)

    diagnostic_frames: list[pd.DataFrame] = []
    for name, source_q, absorption_q in (
        ("source_q90", 0.90, cfg.absorption_quantile),
        ("absorption_q55", cfg.source_return_quantile, 0.55),
    ):
        local_signals, _ = build_v188_signals(
            close,
            panels,
            cfg,
            source_return_quantile=source_q,
            absorption_quantile=absorption_q,
        )
        local_events = _build_both(
            local_signals, risk, close, features, cfg
        )
        local_summary = summarize_v188(local_events)
        local_summary["diagnostic"] = name
        diagnostic_frames.append(local_summary)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (1, 4):
        local_events = _build_both(
            signals, risk, close, features, cfg, holding_bars=holding_bars
        )
        local_summary = summarize_v188(local_events)
        local_summary["holding_bars"] = holding_bars
        horizon_frames.append(local_summary)
    horizons = pd.concat(horizon_frames, ignore_index=True)

    selected_direct = events[events["candidate"].eq(DIRECT_CANDIDATE)]
    base_direct = build_v188_direct_events(base_signals, close, cfg)
    bucket_events = events[events["candidate"].eq(BUCKET_CANDIDATE)]
    random_controls = random_v188_controls(
        selected_direct,
        base_direct,
        bucket_events,
        risk,
        close,
        features,
        cfg,
    )
    gates, outcome = audit_v188(
        events,
        summary,
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
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "ranking_events": root / "ranking_control_events.parquet",
        "ranking_summary": root / "ranking_control_summary.csv",
        "diagnostics": root / "threshold_diagnostics.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    base_signals.to_parquet(paths["base_signals"], index=False)
    risk.to_parquet(paths["risk"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    ranking_events.to_parquet(paths["ranking_events"], index=False)
    ranking_summary.to_csv(paths["ranking_summary"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    horizons.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "BUCKET_CANDIDATE",
    "CANDIDATES",
    "DIRECT_CANDIDATE",
    "V188Config",
    "audit_v188",
    "build_v188_bucket_events",
    "build_v188_direct_events",
    "build_v188_features",
    "build_v188_signals",
    "random_v188_controls",
    "summarize_v188",
    "write_v188_top_trader_absorption",
]
