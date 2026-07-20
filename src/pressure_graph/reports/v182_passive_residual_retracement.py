"""Passive retracement entries for the audited v18.0 residual extremes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    BTC,
    EXCLUDED_RECEIVERS,
    KLINE_ROOT,
    _period,
    load_v178_market_data,
)
from pressure_graph.reports.v180_extreme_residual_dispersion_compression import (
    REPORT_ROOT as V180_REPORT_ROOT,
    V180Config,
)


REPORT_ROOT = Path("reports/v18_2_passive_residual_retracement")
FINDINGS_PATH = Path("docs/v182_passive_residual_retracement_findings_2026_07_16.md")
CANDIDATE = "PRR1_PASSIVE_RESIDUAL_RETRACEMENT"


@dataclass(frozen=True)
class V182Config(V180Config):
    entry_offset: float = 0.0010
    sensitivity_offsets: tuple[float, ...] = (0.0005, 0.0020)
    post_fill_holding_bars: int = 4
    alt_primary_cost: float = 0.0010
    alt_stress_cost: float = 0.0020
    btc_primary_cost: float = 0.0008
    btc_stress_cost: float = 0.0012
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 18_200


def load_v182_high_low(
    root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_frames: list[pd.Series] = []
    low_frames: list[pd.Series] = []
    for path in sorted(root.glob("*.parquet")):
        symbol = path.stem.upper()
        if symbol in EXCLUDED_RECEIVERS - {BTC}:
            continue
        frame = pd.read_parquet(path, columns=["bar_close_time", "high", "low"])
        frame["bar_close_time"] = pd.to_datetime(
            frame["bar_close_time"], utc=True, errors="coerce"
        )
        for column in ("high", "low"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.dropna(subset=["bar_close_time"])
            .drop_duplicates("bar_close_time", keep="last")
            .sort_values("bar_close_time")
            .set_index("bar_close_time")
        )
        high_frames.append(frame["high"].rename(symbol))
        low_frames.append(frame["low"].rename(symbol))
    high = pd.concat(high_frames, axis=1).sort_index()
    low = pd.concat(low_frames, axis=1).sort_index()
    regular = pd.date_range(high.index.min(), high.index.max(), freq="15min", tz="UTC")
    high = high.reindex(regular)
    low = low.reindex(regular)
    high.index.name = "bar_close_time"
    low.index.name = "bar_close_time"
    return high, low


def _beta_map(graph: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    return {
        pd.Timestamp(month): local.set_index("receiver")["btc_beta"].astype(float)
        for month, local in graph.groupby("graph_month", sort=True)
    }


def _simulate_v182_event(
    signal: object,
    buy_names: list[str],
    short_names: list[str],
    beta: pd.Series,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    cfg: V182Config,
    entry_offset: float,
    order_delay_bars: int,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    source_time = pd.Timestamp(signal.source_feature_time)
    reference_time = source_time + pd.Timedelta(minutes=15 * order_delay_bars)
    fill_time = reference_time + pd.Timedelta(minutes=15)
    exit_time = fill_time + pd.Timedelta(minutes=15 * cfg.post_fill_holding_bars)
    required_times = (reference_time, fill_time, exit_time)
    if any(timestamp not in close.index for timestamp in required_times):
        return None, []
    normalizer = 1.0 + abs(float(signal.spread_beta))
    intended_weight = 0.1 / normalizer
    leg_rows: list[dict[str, object]] = []
    alt_gross = 0.0
    filled_beta = 0.0
    filled_alt_allocation = 0.0
    laggard_fills = 0
    leader_fills = 0
    for side, names in (("laggard_buy", buy_names), ("leader_short", short_names)):
        for name in names:
            reference = float(close.at[reference_time, name])
            exit_price = float(close.at[exit_time, name])
            if side == "laggard_buy":
                limit_price = reference * (1.0 - entry_offset)
                touched = float(low.at[fill_time, name]) <= limit_price
                signed_weight = intended_weight
            else:
                limit_price = reference * (1.0 + entry_offset)
                touched = float(high.at[fill_time, name]) >= limit_price
                signed_weight = -intended_weight
            finite = all(
                np.isfinite(value)
                for value in (reference, exit_price, limit_price, beta.get(name, math.nan))
            )
            filled = bool(finite and touched)
            contribution = (
                signed_weight * (exit_price / limit_price - 1.0) if filled else 0.0
            )
            if filled:
                alt_gross += contribution
                filled_beta += signed_weight * float(beta[name])
                filled_alt_allocation += abs(signed_weight)
                laggard_fills += int(side == "laggard_buy")
                leader_fills += int(side == "leader_short")
            leg_rows.append(
                {
                    "source_feature_time": source_time,
                    "reference_time": reference_time,
                    "fill_time": fill_time,
                    "exit_time": exit_time,
                    "symbol": name,
                    "side": side,
                    "reference_close": reference,
                    "limit_price": limit_price,
                    "fill_bar_high": float(high.at[fill_time, name]),
                    "fill_bar_low": float(low.at[fill_time, name]),
                    "filled": filled,
                    "signed_weight": signed_weight if filled else 0.0,
                    "btc_beta": float(beta.get(name, math.nan)),
                    "gross_contribution": contribution,
                    "entry_offset": entry_offset,
                    "order_delay_bars": order_delay_bars,
                }
            )
    btc_hedge_weight = -filled_beta
    btc_return = float(close.at[exit_time, BTC] / close.at[fill_time, BTC] - 1.0)
    btc_gross = btc_hedge_weight * btc_return
    gross = alt_gross + btc_gross
    primary_cost = (
        filled_alt_allocation * cfg.alt_primary_cost
        + abs(btc_hedge_weight) * cfg.btc_primary_cost
    )
    stress_cost = (
        filled_alt_allocation * cfg.alt_stress_cost
        + abs(btc_hedge_weight) * cfg.btc_stress_cost
    )
    filled_count = laggard_fills + leader_fills
    event = {
        **signal._asdict(),
        "candidate": CANDIDATE,
        "reference_time": reference_time,
        "fill_time": fill_time,
        "exit_time": exit_time,
        "entry_offset": entry_offset,
        "order_delay_bars": order_delay_bars,
        "period": _period(source_time),
        "entry_day": source_time.strftime("%Y-%m-%d"),
        "entry_month": source_time.strftime("%Y-%m"),
        "laggard_fills": laggard_fills,
        "leader_fills": leader_fills,
        "filled_count": filled_count,
        "filled_fraction": filled_count / (len(buy_names) + len(short_names)),
        "laggard_fill_fraction": laggard_fills / len(buy_names),
        "leader_fill_fraction": leader_fills / len(short_names),
        "traded": filled_count > 0,
        "filled_alt_allocation": filled_alt_allocation,
        "btc_hedge_weight": btc_hedge_weight,
        "alt_gross_return": alt_gross,
        "btc_gross_return": btc_gross,
        "gross_return": gross,
        "primary_cost_return": primary_cost,
        "stress_cost_return": stress_cost,
        "primary_net_return": gross - primary_cost,
        "stress_net_return": gross - stress_cost,
        "reversed_primary_net_return": -gross - primary_cost,
    }
    return event, leg_rows


def build_v182_events(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    cfg: V182Config = V182Config(),
    entry_offset: float | None = None,
    order_delay_bars: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    offset = cfg.entry_offset if entry_offset is None else entry_offset
    betas = _beta_map(graph)
    event_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        beta = betas.get(pd.Timestamp(signal.graph_month))
        if beta is None:
            continue
        event, local_legs = _simulate_v182_event(
            signal,
            str(signal.laggards).split("|"),
            str(signal.leaders).split("|"),
            beta,
            close,
            high,
            low,
            cfg,
            offset,
            order_delay_bars,
        )
        if event is not None:
            event_rows.append(event)
            leg_rows.extend(local_legs)
    events = pd.DataFrame(event_rows)
    legs = pd.DataFrame(leg_rows)
    if not events.empty:
        events = events.sort_values("source_feature_time").reset_index(drop=True)
    return events, legs


def summarize_v182(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        traded = sample[sample["traded"].astype(bool)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "source_events": len(sample),
                "traded_events": len(traded),
                "active_days": traded["entry_day"].nunique(),
                "active_months": traded["entry_month"].nunique(),
                "laggard_fill_rate": (
                    float(sample["laggard_fill_fraction"].mean())
                    if len(sample)
                    else math.nan
                ),
                "leader_fill_rate": (
                    float(sample["leader_fill_fraction"].mean())
                    if len(sample)
                    else math.nan
                ),
                "mean_filled_fraction": (
                    float(sample["filled_fraction"].mean())
                    if len(sample)
                    else math.nan
                ),
                "mean_gross_bp": (
                    float(sample["gross_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan
                ),
                "mean_primary_cost_bp": (
                    float(sample["primary_cost_return"].mean() * 10_000)
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
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> list[dict[str, object]]:
    betas = _beta_map(graph)
    contexts: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        beta = betas[pd.Timestamp(event.graph_month)]
        names = [name for name in beta.index.astype(str) if name in close]
        reference_time = pd.Timestamp(event.reference_time)
        fill_time = pd.Timestamp(event.fill_time)
        exit_time = pd.Timestamp(event.exit_time)
        valid = [
            name
            for name in names
            if all(
                np.isfinite(value)
                for value in (
                    close.at[reference_time, name],
                    high.at[fill_time, name],
                    low.at[fill_time, name],
                    close.at[exit_time, name],
                    beta[name],
                )
            )
        ]
        if len(valid) < 10:
            continue
        contexts.append(
            {
                "reference": close.loc[reference_time, valid].to_numpy(dtype=float),
                "high": high.loc[fill_time, valid].to_numpy(dtype=float),
                "low": low.loc[fill_time, valid].to_numpy(dtype=float),
                "exit": close.loc[exit_time, valid].to_numpy(dtype=float),
                "beta": beta.reindex(valid).to_numpy(dtype=float),
                "btc_return": float(
                    close.at[exit_time, BTC] / close.at[fill_time, BTC] - 1.0
                ),
                "normalizer": 1.0 + abs(float(event.spread_beta)),
            }
        )
    return contexts


def random_v182_controls(
    events: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    cfg: V182Config = V182Config(),
) -> pd.DataFrame:
    contexts = _random_contexts(events, graph, close, high, low)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values: list[float] = []
        traded = 0
        for context in contexts:
            chosen = rng.choice(len(context["reference"]), size=10, replace=False)
            buy = chosen[:5]
            short = chosen[5:]
            reference = context["reference"]
            buy_limit = reference[buy] * (1.0 - cfg.entry_offset)
            short_limit = reference[short] * (1.0 + cfg.entry_offset)
            buy_fill = context["low"][buy] <= buy_limit
            short_fill = context["high"][short] >= short_limit
            weight = 0.1 / float(context["normalizer"])
            alt_gross = float(
                np.sum(
                    weight
                    * (context["exit"][buy][buy_fill] / buy_limit[buy_fill] - 1.0)
                )
                - np.sum(
                    weight
                    * (
                        context["exit"][short][short_fill]
                        / short_limit[short_fill]
                        - 1.0
                    )
                )
            )
            filled_beta = float(
                weight * np.sum(context["beta"][buy][buy_fill])
                - weight * np.sum(context["beta"][short][short_fill])
            )
            hedge_weight = -filled_beta
            gross = alt_gross + hedge_weight * float(context["btc_return"])
            allocation = weight * (int(buy_fill.sum()) + int(short_fill.sum()))
            cost = (
                allocation * cfg.alt_primary_cost
                + abs(hedge_weight) * cfg.btc_primary_cost
            )
            values.append(gross - cost)
            traded += int(bool(buy_fill.any() or short_fill.any()))
        rows.append(
            {
                "iteration": iteration,
                "candidate": "RANDOM_RANK",
                "source_events": len(values),
                "traded_events": traded,
                "mean_primary_net_return": (
                    float(np.mean(values)) if values else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(events: pd.DataFrame, cfg: V182Config) -> tuple[float, float]:
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


def _positive_profit_concentration(events: pd.DataFrame) -> float:
    monthly = events.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v182(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V182Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = summary.set_index("scope")
    real_mean = float(events["primary_net_return"].mean())
    reversed_mean = float(events["reversed_primary_net_return"].mean())
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
        "entry_offset_bp"
    )
    concentration = _positive_profit_concentration(events)
    checks: dict[str, tuple[bool, float]] = {
        "traded_events_100": (
            scoped.loc["all", "traded_events"] >= 100,
            float(scoped.loc["all", "traded_events"]),
        ),
        "validation_traded_20": (
            scoped.loc["validation", "traded_events"] >= 20,
            float(scoped.loc["validation", "traded_events"]),
        ),
        "holdout_traded_25": (
            scoped.loc["holdout", "traded_events"] >= 25,
            float(scoped.loc["holdout", "traded_events"]),
        ),
        "laggard_fill_rate_40": (
            scoped.loc["all", "laggard_fill_rate"] >= 0.40,
            float(scoped.loc["all", "laggard_fill_rate"]),
        ),
        "leader_fill_rate_40": (
            scoped.loc["all", "leader_fill_rate"] >= 0.40,
            float(scoped.loc["all", "leader_fill_rate"]),
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
        "beats_one_bar_delay": (
            real_mean > delayed_mean,
            (real_mean - delayed_mean) * 10_000,
        ),
        "beats_reversed_direction": (
            real_mean > reversed_mean,
            (real_mean - reversed_mean) * 10_000,
        ),
        "offset_5bp_positive": (
            local_sensitivity.loc[5.0, "mean_primary_net_bp"] > 0,
            float(local_sensitivity.loc[5.0, "mean_primary_net_bp"]),
        ),
        "offset_20bp_positive": (
            local_sensitivity.loc[20.0, "mean_primary_net_bp"] > 0,
            float(local_sensitivity.loc[20.0, "mean_primary_net_bp"]),
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
                "source_events": len(events),
                "traded_events": int(events["traded"].sum()),
                "mean_gross_bp": float(events["gross_return"].mean() * 10_000),
                "mean_primary_cost_bp": float(
                    events["primary_cost_return"].mean() * 10_000
                ),
                "mean_primary_net_bp": real_mean * 10_000,
                "mean_stress_net_bp": float(
                    events["stress_net_return"].mean() * 10_000
                ),
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
                    else "reject_passive_residual_retracement"
                ),
            }
        ]
    )
    return gates, outcome


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    text = [
        "# v18.2 Passive Residual Retracement Findings",
        "",
        f"Verdict: `{outcome['verdict'].iloc[0]}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "All partial fills are retained and unfilled allocations remain cash.",
        "No live, PaperLive, leverage, remote, application, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v182_passive_residual_retracement(
    kline_root: Path = KLINE_ROOT,
    v180_report_root: Path = V180_REPORT_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V182Config = V182Config(),
) -> dict[str, Path]:
    close, _ = load_v178_market_data(kline_root)
    high, low = load_v182_high_low(kline_root)
    signals = pd.read_parquet(v180_report_root / "dispersion_signals.parquet")
    graph = pd.read_parquet(v180_report_root / "monthly_btc_beta_graph.parquet")
    events, legs = build_v182_events(signals, graph, close, high, low, cfg)
    summary = summarize_v182(events)
    delayed_events, _ = build_v182_events(
        signals, graph, close, high, low, cfg, order_delay_bars=1
    )
    delayed_summary = summarize_v182(delayed_events)
    sensitivity_frames: list[pd.DataFrame] = []
    for offset in cfg.sensitivity_offsets:
        local_events, _ = build_v182_events(
            signals, graph, close, high, low, cfg, entry_offset=offset
        )
        local = summarize_v182(local_events)
        local["entry_offset_bp"] = offset * 10_000
        sensitivity_frames.append(local)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    random_controls = random_v182_controls(events, graph, close, high, low, cfg)
    gates, outcome = audit_v182(
        events,
        summary,
        delayed_summary,
        sensitivity,
        random_controls,
        cfg,
    )

    root = ensure_dir(report_root)
    paths = {
        "signals": root / "fixed_v180_signals.parquet",
        "events": root / "candidate_events.parquet",
        "legs": root / "passive_fill_legs.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "entry_offset_sensitivity.csv",
        "random_controls": root / "random_rank_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    events.to_parquet(paths["events"], index=False)
    legs.to_parquet(paths["legs"], index=False)
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
    "CANDIDATE",
    "V182Config",
    "audit_v182",
    "build_v182_events",
    "load_v182_high_low",
    "random_v182_controls",
    "summarize_v182",
    "write_v182_passive_residual_retracement",
]
