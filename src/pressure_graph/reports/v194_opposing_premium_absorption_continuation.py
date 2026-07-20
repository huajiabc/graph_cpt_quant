"""Continuation after BTC price shocks absorb opposing premium-index pressure."""
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
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    V187Config,
    build_v187_monthly_risk,
)
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_ohlc_panels,
)
from pressure_graph.reports.v193_premium_pressure_shape_feature_audit import (
    build_v193_pressure_features,
)


REPORT_ROOT = Path("reports/v19_4_opposing_premium_absorption_continuation")
FINDINGS_PATH = Path(
    "docs/v194_opposing_premium_absorption_continuation_findings_2026_07_17.md"
)
DIRECT_CANDIDATE = "PPA1_BTC_OPPOSING_PREMIUM_ABSORPTION_CONTINUATION"
BUCKET_CANDIDATE = "PPA2_OPPOSING_PREMIUM_RECEIVER_CONTINUATION_BUCKET"
CANDIDATES = (DIRECT_CANDIDATE, BUCKET_CANDIDATE)


@dataclass(frozen=True)
class V194Config(V187Config):
    source_return_quantile: float = 0.85
    source_range_z_threshold: float = 1.0
    receiver_range_z_threshold: float = 1.0
    body_z_threshold: float = 0.50
    close_location_threshold: float = 0.50
    direct_primary_cost: float = 0.0010
    direct_stress_cost: float = 0.0015
    seed: int = 19_400


def build_v194_signals(
    close: pd.DataFrame,
    premium_ohlc: dict[str, pd.DataFrame],
    cfg: V194Config = V194Config(),
    source_return_quantile: float | None = None,
    source_range_z_threshold: float | None = None,
    shape: str = "absorption",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    quantile = (
        cfg.source_return_quantile
        if source_return_quantile is None
        else source_return_quantile
    )
    range_threshold = (
        cfg.source_range_z_threshold
        if source_range_z_threshold is None
        else source_range_z_threshold
    )
    if quantile not in (0.85, 0.90):
        raise ValueError("source_return_quantile must be 0.85 or 0.90")
    if shape not in ("absorption", "through"):
        raise ValueError("shape must be absorption or through")
    features = build_v193_pressure_features(close, premium_ohlc, cfg)
    label = f"q{int(round(quantile * 100))}"
    source_sign = np.sign(features["price_return"][BTC])
    base = pd.DataFrame(
        {
            "feature_time": close.index,
            "source_feature_time": close.index,
            "btc_return_15m": features["price_return"][BTC].to_numpy(),
            "source_sign": source_sign.to_numpy(),
            "return_threshold": features["price_thresholds"][label].to_numpy(),
            "return_quantile": quantile,
            "btc_premium_range_z": features["range_z"][BTC].to_numpy(),
            "btc_aligned_body_z": (
                source_sign * features["body_z"][BTC]
            ).to_numpy(),
            "btc_aligned_close_location": (
                source_sign * features["close_location"][BTC]
            ).to_numpy(),
        }
    )
    base["source_side"] = np.where(
        base["source_sign"].gt(0), "up_move", "down_move"
    )
    price_shock = base["btc_return_15m"].abs().ge(base["return_threshold"])
    range_shock = base["btc_premium_range_z"].ge(range_threshold)
    if shape == "absorption":
        shape_mask = base["btc_aligned_body_z"].le(-cfg.body_z_threshold) & base[
            "btc_aligned_close_location"
        ].le(-cfg.close_location_threshold)
    else:
        shape_mask = base["btc_aligned_body_z"].ge(cfg.body_z_threshold) & base[
            "btc_aligned_close_location"
        ].ge(cfg.close_location_threshold)
    selected = base[price_shock & range_shock & shape_mask].copy()
    selected["source_shape"] = shape
    return selected.reset_index(drop=True), base.reset_index(drop=True), features


def build_v194_direct_events(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V194Config = V194Config(),
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
        values["candidate"] = DIRECT_CANDIDATE
        rows.append(
            {
                **values,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_delay_bars": entry_delay_bars,
                "holding_bars": horizon,
                "ranking": str(values.get("source_shape", "price_shock_pool")),
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


def _receiver_scores(
    timestamp: pd.Timestamp,
    source_sign: float,
    risk: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    cfg: V194Config,
    shape: str = "absorption",
) -> pd.DataFrame:
    local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
    names = [
        name
        for name in local.index.astype(str)
        if name in features["range_z"].columns
    ]
    if not names or timestamp not in features["range_z"].index:
        return pd.DataFrame()
    frame = pd.DataFrame(index=names)
    frame["btc_beta"] = local.reindex(names)["btc_beta"].astype(float)
    frame["range_z"] = features["range_z"].loc[timestamp, names].astype(float)
    frame["aligned_body_z"] = (
        source_sign * features["body_z"].loc[timestamp, names].astype(float)
    )
    frame["aligned_close_location"] = (
        source_sign
        * features["close_location"].loc[timestamp, names].astype(float)
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    range_mask = frame["range_z"].ge(cfg.receiver_range_z_threshold)
    if shape == "absorption":
        shape_mask = frame["aligned_body_z"].le(-cfg.body_z_threshold) & frame[
            "aligned_close_location"
        ].le(-cfg.close_location_threshold)
    elif shape == "through":
        shape_mask = frame["aligned_body_z"].ge(cfg.body_z_threshold) & frame[
            "aligned_close_location"
        ].ge(cfg.close_location_threshold)
    else:
        raise ValueError("shape must be absorption or through")
    return frame[range_mask & shape_mask]


def build_v194_bucket_events(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    cfg: V194Config = V194Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
    ranking: str = "top",
    shape: str = "absorption",
) -> pd.DataFrame:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        source_time = pd.Timestamp(signal.source_feature_time)
        scores = _receiver_scores(
            source_time, float(signal.source_sign), risk, features, cfg, shape
        )
        if len(scores) < cfg.min_receiver_bucket:
            continue
        selected = scores.sort_values(
            "range_z", ascending=ranking != "top"
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
        direction = float(signal.source_sign)
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
                "receiver_shape": shape,
                "risk_month": _month(source_time),
                "period": _period(source_time),
                "entry_day": source_time.strftime("%Y-%m-%d"),
                "entry_month": source_time.strftime("%Y-%m"),
                "eligible_receivers": len(scores),
                "receiver_count": len(receivers),
                "receivers": "|".join(receivers),
                "receiver_scores": "|".join(
                    f"{name}:{selected.at[name, 'range_z']:.8f}"
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


def build_v194_events(
    signals: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    cfg: V194Config = V194Config(),
    holding_bars: int | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    return _combine(
        build_v194_direct_events(
            signals, close, cfg, holding_bars, entry_delay_bars
        ),
        build_v194_bucket_events(
            signals,
            risk,
            close,
            features,
            cfg,
            holding_bars,
            entry_delay_bars,
        ),
    )


def summarize_v194(events: pd.DataFrame) -> pd.DataFrame:
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


def summarize_v194_sides(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for side in ("up_move", "down_move"):
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
    features: dict[str, pd.DataFrame],
    cfg: V194Config,
) -> list[dict[str, object]]:
    contexts = []
    for event in bucket_events.itertuples(index=False):
        scores = _receiver_scores(
            pd.Timestamp(event.source_feature_time),
            float(event.source_sign),
            risk,
            features,
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


def random_v194_controls(
    selected_direct: pd.DataFrame,
    base_direct: pd.DataFrame,
    bucket_events: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    cfg: V194Config = V194Config(),
) -> pd.DataFrame:
    selected_counts = selected_direct.groupby("entry_month").size().to_dict()
    base_by_month = {
        month: sample["primary_net_return"].to_numpy(dtype=float)
        for month, sample in base_direct.groupby("entry_month", sort=True)
    }
    bucket_contexts = _random_bucket_contexts(
        bucket_events, risk, close, features, cfg
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
    sample: pd.DataFrame, cfg: V194Config, offset: int
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


def audit_v194(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    side_summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    horizons: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V194Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"),
        "mean_primary_net_return",
    ].dropna()
    gate_rows = []
    outcomes = []
    minimums = {
        DIRECT_CANDIDATE: (150, 30, 50),
        BUCKET_CANDIDATE: (60, 15, 20),
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
        control_mean = float(
            control_summary.loc[
                control_summary["candidate"].eq(candidate)
                & control_summary["scope"].eq("all"),
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
            "beats_shape_or_ranking_control": (
                real_mean > control_mean,
                (real_mean - control_mean) * 10_000,
            ),
            "source_q90_positive": (
                diagnostic.loc["source_q90", "mean_primary_net_bp"] > 0,
                float(diagnostic.loc["source_q90", "mean_primary_net_bp"]),
            ),
            "range_z15_positive": (
                diagnostic.loc["range_z15", "mean_primary_net_bp"] > 0,
                float(diagnostic.loc["range_z15", "mean_primary_net_bp"]),
            ),
            "holding_15m_positive": (
                horizon.loc[1, "mean_primary_net_bp"] > 0,
                float(horizon.loc[1, "mean_primary_net_bp"]),
            ),
            "holding_60m_positive": (
                horizon.loc[4, "mean_primary_net_bp"] > 0,
                float(horizon.loc[4, "mean_primary_net_bp"]),
            ),
            "up_move_positive": (
                sides.loc["up_move", "mean_primary_net_bp"] > 0,
                float(sides.loc["up_move", "mean_primary_net_bp"]),
            ),
            "down_move_positive": (
                sides.loc["down_move", "mean_primary_net_bp"] > 0,
                float(sides.loc["down_move", "mean_primary_net_bp"]),
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
                "control_primary_net_bp": control_mean * 10_000,
                "positive_profit_concentration": concentration,
                "eligible": eligible,
                "failed_gates": "|".join(
                    name for name, (passed, _) in checks.items() if not passed
                ),
                "verdict": (
                    "offline_research_candidate_only"
                    if eligible
                    else "reject_opposing_premium_absorption_continuation"
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
        else "reject_opposing_premium_absorption_continuation"
    )
    text = [
        "# v19.4 Opposing-Premium Absorption Continuation Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        side_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Premium OHLC values are exact completed bars with shifted historical",
        "scales and no forward filling. No live, PaperLive, application, leverage,",
        "remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v194_opposing_premium_absorption_continuation(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V194Config = V194Config(),
) -> dict[str, Path]:
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    premium_ohlc = load_v190_premium_ohlc_panels(premium_root)
    signals, base, features = build_v194_signals(close, premium_ohlc, cfg)
    returns = close.pct_change(fill_method=None)
    risk = build_v187_monthly_risk(
        returns, signals["feature_time"].min(), signals["feature_time"].max(), cfg
    )
    events = build_v194_events(signals, risk, close, features, cfg)
    summary = summarize_v194(events)
    side_summary = summarize_v194_sides(events)
    delayed_events = build_v194_events(
        signals, risk, close, features, cfg, entry_delay_bars=1
    )
    delayed_summary = summarize_v194(delayed_events)

    through_signals, _, _ = build_v194_signals(
        close, premium_ohlc, cfg, shape="through"
    )
    direct_control = build_v194_direct_events(through_signals, close, cfg)
    bucket_control = build_v194_bucket_events(
        signals, risk, close, features, cfg, ranking="bottom"
    )
    control_events = _combine(direct_control, bucket_control)
    control_summary = summarize_v194(control_events)

    diagnostic_frames = []
    for name, kwargs in (
        ("source_q90", {"source_return_quantile": 0.90}),
        ("range_z15", {"source_range_z_threshold": 1.50}),
    ):
        local_signals, _, _ = build_v194_signals(
            close, premium_ohlc, cfg, **kwargs
        )
        local_summary = summarize_v194(
            build_v194_events(local_signals, risk, close, features, cfg)
        )
        local_summary["diagnostic"] = name
        diagnostic_frames.append(local_summary)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)

    horizon_frames = []
    for holding_bars in (1, 4):
        local_summary = summarize_v194(
            build_v194_events(
                signals,
                risk,
                close,
                features,
                cfg,
                holding_bars=holding_bars,
            )
        )
        local_summary["holding_bars"] = holding_bars
        horizon_frames.append(local_summary)
    horizons = pd.concat(horizon_frames, ignore_index=True)

    price_shock_pool = base[
        base["btc_return_15m"].abs().ge(base["return_threshold"])
        & base["source_sign"].ne(0)
    ].copy()
    price_shock_pool["source_shape"] = "price_shock_pool"
    selected_direct = events[events["candidate"].eq(DIRECT_CANDIDATE)]
    base_direct = build_v194_direct_events(price_shock_pool, close, cfg)
    bucket_events = events[events["candidate"].eq(BUCKET_CANDIDATE)]
    random_controls = random_v194_controls(
        selected_direct,
        base_direct,
        bucket_events,
        risk,
        close,
        features,
        cfg,
    )
    gates, outcome = audit_v194(
        events,
        summary,
        side_summary,
        delayed_summary,
        control_summary,
        diagnostics,
        horizons,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "selected_source_signals.parquet",
        "base": root / "base_feature_rows.parquet",
        "risk": root / "monthly_risk_estimates.parquet",
        "range_z": root / "premium_range_z.parquet",
        "body_z": root / "premium_body_z.parquet",
        "close_location": root / "premium_close_location.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "side_summary": root / "source_side_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "control_events": root / "shape_ranking_control_events.parquet",
        "control_summary": root / "shape_ranking_control_summary.csv",
        "diagnostics": root / "diagnostic_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    base.to_parquet(paths["base"], index=False)
    risk.to_parquet(paths["risk"], index=False)
    features["range_z"].to_parquet(paths["range_z"])
    features["body_z"].to_parquet(paths["body_z"])
    features["close_location"].to_parquet(paths["close_location"])
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    side_summary.to_csv(paths["side_summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    control_events.to_parquet(paths["control_events"], index=False)
    control_summary.to_csv(paths["control_summary"], index=False)
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
    "V194Config",
    "audit_v194",
    "build_v194_bucket_events",
    "build_v194_direct_events",
    "build_v194_events",
    "build_v194_signals",
    "random_v194_controls",
    "summarize_v194",
    "summarize_v194_sides",
    "write_v194_opposing_premium_absorption_continuation",
]
