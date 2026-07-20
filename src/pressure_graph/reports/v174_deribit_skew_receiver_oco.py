"""Price-directed OCO monetization of BTC option-skew stress transmission."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v173_deribit_skew_receiver_bucket import (
    ALTS,
    BTC,
    KLINE_ROOT,
    SURFACE_PATH,
    V173Config,
    build_monthly_receiver_graph,
    build_surface_signals,
    hourly_log_returns,
    load_v173_prices,
)


REPORT_ROOT = Path("reports/v17_4_deribit_skew_receiver_oco")
FINDINGS_PATH = Path("docs/v174_deribit_skew_receiver_oco_findings_2026_07_16.md")
PROMOTION_CANDIDATE = "DOS1_STRESS_RECEIVER_OCO"
DIAGNOSTIC_CANDIDATE = "DOS2_RELIEF_RECEIVER_OCO"
ALL_CANDIDATES = (PROMOTION_CANDIDATE, DIAGNOSTIC_CANDIDATE)


@dataclass(frozen=True)
class V174Config:
    reference_hours: int = 6
    entry_window_hours: int = 6
    holding_hours: int = 24
    min_filled_receivers: int = 2
    primary_cost: float = 0.0030
    stress_cost: float = 0.0050
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 17_400


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2024-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2025-01-01", tz="UTC"):
        return "validation"
    return "holdout"


def load_v174_ohlc(root: Path = KLINE_ROOT) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol in (BTC, *ALTS):
        frame = pd.read_parquet(
            root / f"{symbol}.parquet",
            columns=["feature_time", "high", "low", "close"],
        )
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
        for column in ("high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        output[symbol] = (
            frame.dropna()
            .drop_duplicates("feature_time", keep="last")
            .sort_values("feature_time")
            .set_index("feature_time")
        )
    return output


def simulate_receiver_oco_leg(
    ohlc: pd.DataFrame,
    signal_time: pd.Timestamp,
    reference_hours: int = 6,
    entry_window_hours: int = 6,
    holding_hours: int = 24,
) -> dict[str, object]:
    signal_time = pd.Timestamp(signal_time)
    reference_times = pd.date_range(
        signal_time - pd.Timedelta(hours=reference_hours - 1),
        signal_time,
        freq="h",
    )
    reference = ohlc.reindex(reference_times)
    if len(reference) != reference_hours or reference[["high", "low"]].isna().any().any():
        return {"status": "missing_reference", "filled": False, "ambiguous": False}
    reference_high = float(reference["high"].max())
    reference_low = float(reference["low"].min())
    exit_time = signal_time + pd.Timedelta(hours=holding_hours)
    if exit_time not in ohlc.index or not np.isfinite(ohlc.at[exit_time, "close"]):
        return {"status": "missing_exit", "filled": False, "ambiguous": False}
    exit_price = float(ohlc.at[exit_time, "close"])
    entry_times = pd.date_range(
        signal_time + pd.Timedelta(hours=1),
        signal_time + pd.Timedelta(hours=entry_window_hours),
        freq="h",
    )
    for entry_time, bar in ohlc.reindex(entry_times).iterrows():
        if not np.isfinite(bar.get("high", np.nan)) or not np.isfinite(
            bar.get("low", np.nan)
        ):
            continue
        long_hit = float(bar["high"]) >= reference_high
        short_hit = float(bar["low"]) <= reference_low
        if long_hit and short_hit:
            return {
                "status": "same_bar_dual_trigger",
                "filled": False,
                "ambiguous": True,
                "entry_time": entry_time,
                "reference_high": reference_high,
                "reference_low": reference_low,
            }
        if long_hit:
            return {
                "status": "long_break",
                "filled": True,
                "ambiguous": False,
                "side": "long",
                "entry_time": entry_time,
                "entry_price": reference_high,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "gross_return": exit_price / reference_high - 1.0,
            }
        if short_hit:
            return {
                "status": "short_break",
                "filled": True,
                "ambiguous": False,
                "side": "short",
                "entry_time": entry_time,
                "entry_price": reference_low,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "gross_return": 1.0 - exit_price / reference_low,
            }
    return {
        "status": "unfilled",
        "filled": False,
        "ambiguous": False,
        "reference_high": reference_high,
        "reference_low": reference_low,
    }


def _selected_receivers(graph: pd.DataFrame, timestamp: pd.Timestamp) -> list[str]:
    month = pd.Timestamp(timestamp).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    return (
        graph[graph["graph_month"].eq(month) & graph["selected"].eq(True)]
        .sort_values("receiver_rank")["receiver"]
        .astype(str)
        .tolist()
    )


def _volatility_diagnostics(
    returns: pd.DataFrame,
    signal_time: pd.Timestamp,
    receivers: list[str],
) -> dict[str, float]:
    prior_times = pd.date_range(
        signal_time - pd.Timedelta(hours=23), signal_time, freq="h"
    )
    future_times = pd.date_range(
        signal_time + pd.Timedelta(hours=1),
        signal_time + pd.Timedelta(hours=24),
        freq="h",
    )

    def metrics(times: pd.DatetimeIndex) -> tuple[float, float]:
        values = returns.reindex(times)[receivers]
        if len(values) != 24 or values.isna().any().any():
            return math.nan, math.nan
        rv = values.pow(2).sum().pow(0.5).mean()
        downside = values.clip(upper=0).pow(2).sum().pow(0.5).mean()
        return float(rv), float(downside)

    prior_rv, prior_downside = metrics(prior_times)
    future_rv, future_downside = metrics(future_times)
    return {
        "prior_receiver_rv_24h": prior_rv,
        "future_receiver_rv_24h": future_rv,
        "receiver_rv_expansion": future_rv / prior_rv
        if np.isfinite(prior_rv) and prior_rv > 0 and np.isfinite(future_rv)
        else math.nan,
        "prior_receiver_downside_24h": prior_downside,
        "future_receiver_downside_24h": future_downside,
        "receiver_downside_expansion": future_downside / prior_downside
        if np.isfinite(prior_downside)
        and prior_downside > 0
        and np.isfinite(future_downside)
        else math.nan,
    }


def build_v174_events(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    cfg: V174Config = V174Config(),
    reference_hours: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window = cfg.reference_hours if reference_hours is None else reference_hours
    event_rows: list[dict[str, object]] = []
    leg_rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(signal.feature_time)
        receivers = _selected_receivers(graph, signal_time)
        if len(receivers) < 3:
            continue
        local_legs: list[dict[str, object]] = []
        for receiver in receivers:
            result = simulate_receiver_oco_leg(
                ohlc[receiver],
                signal_time,
                reference_hours=window,
                entry_window_hours=cfg.entry_window_hours,
                holding_hours=cfg.holding_hours,
            )
            leg = {
                "signal_time": signal_time,
                "event_type": signal.event_type,
                "receiver": receiver,
                "reference_hours": window,
                **result,
            }
            local_legs.append(leg)
            leg_rows.append(leg)
        filled = [row for row in local_legs if bool(row["filled"])]
        ambiguous = [row for row in local_legs if bool(row["ambiguous"])]
        if len(filled) < cfg.min_filled_receivers:
            continue
        gross = float(
            sum(float(row["gross_return"]) for row in filled) / len(receivers)
        )
        filled_fraction = len(filled) / len(receivers)
        candidate = (
            PROMOTION_CANDIDATE
            if signal.event_type == "stress"
            else DIAGNOSTIC_CANDIDATE
        )
        event_rows.append(
            {
                **signal._asdict(),
                "candidate": candidate,
                "entry_time": signal_time,
                "exit_time": signal_time + pd.Timedelta(hours=cfg.holding_hours),
                "period": _period(signal_time),
                "year": signal_time.year,
                "receivers": "|".join(receivers),
                "receiver_count": len(receivers),
                "reference_hours": window,
                "filled_receivers": len(filled),
                "ambiguous_receivers": len(ambiguous),
                "filled_fraction": filled_fraction,
                "ambiguous_fraction": len(ambiguous) / len(receivers),
                "long_receivers": sum(row.get("side") == "long" for row in filled),
                "short_receivers": sum(row.get("side") == "short" for row in filled),
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost * filled_fraction,
                "stress_net_return": gross - cfg.stress_cost * filled_fraction,
                **_volatility_diagnostics(returns, signal_time, receivers),
            }
        )
    events = pd.DataFrame(event_rows)
    legs = pd.DataFrame(leg_rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "candidate"]).reset_index(drop=True)
    return events, legs


def summarize_v174(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in ALL_CANDIDATES:
        candidate_rows = events[events["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = (
                candidate_rows
                if scope == "all"
                else candidate_rows[candidate_rows["period"].eq(scope)]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": int(len(sample)),
                    "active_years": int(sample["year"].nunique()),
                    "mean_filled_fraction": float(sample["filled_fraction"].mean())
                    if len(sample)
                    else math.nan,
                    "mean_ambiguous_fraction": float(sample["ambiguous_fraction"].mean())
                    if len(sample)
                    else math.nan,
                    "mean_gross_bp": float(sample["gross_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan,
                    "mean_primary_net_bp": float(
                        sample["primary_net_return"].mean() * 10_000
                    )
                    if len(sample)
                    else math.nan,
                    "mean_stress_net_bp": float(
                        sample["stress_net_return"].mean() * 10_000
                    )
                    if len(sample)
                    else math.nan,
                    "win_rate_primary": float(sample["primary_net_return"].gt(0).mean())
                    if len(sample)
                    else math.nan,
                    "mean_rv_expansion": float(sample["receiver_rv_expansion"].mean())
                    if len(sample)
                    else math.nan,
                    "mean_downside_expansion": float(
                        sample["receiver_downside_expansion"].mean()
                    )
                    if len(sample)
                    else math.nan,
                    "sum_primary_net": float(sample["primary_net_return"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _random_controls(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame],
    cfg: V174Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in ALL_CANDIDATES}
        for signal in signals.itertuples(index=False):
            signal_time = pd.Timestamp(signal.feature_time)
            count = len(_selected_receivers(graph, signal_time))
            if count < 3:
                continue
            receivers = sorted(rng.choice(ALTS, size=count, replace=False).tolist())
            results = [
                simulate_receiver_oco_leg(
                    ohlc[receiver],
                    signal_time,
                    reference_hours=cfg.reference_hours,
                    entry_window_hours=cfg.entry_window_hours,
                    holding_hours=cfg.holding_hours,
                )
                for receiver in receivers
            ]
            filled = [result for result in results if bool(result["filled"])]
            if len(filled) < cfg.min_filled_receivers:
                continue
            gross = sum(float(result["gross_return"]) for result in filled) / count
            net = gross - cfg.primary_cost * len(filled) / count
            candidate = (
                PROMOTION_CANDIDATE
                if signal.event_type == "stress"
                else DIAGNOSTIC_CANDIDATE
            )
            values[candidate].append(net)
        means: dict[str, float] = {}
        for candidate in ALL_CANDIDATES:
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
                "events": max(len(values[candidate]) for candidate in ALL_CANDIDATES),
                "mean_primary_net_return": max(finite) if finite else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, cfg: V174Config) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed + 1)
    means = rng.choice(
        values,
        size=(cfg.bootstrap_iterations, len(values)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_year_share(sample: pd.DataFrame) -> float:
    yearly = sample.groupby("year")["primary_net_return"].sum().clip(lower=0)
    return float(yearly.max() / yearly.sum()) if yearly.sum() > 0 else math.inf


def audit_v174(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V174Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_summary = summary[summary["candidate"].eq(PROMOTION_CANDIDATE)].set_index(
        "scope"
    )
    sample = events[events["candidate"].eq(PROMOTION_CANDIDATE)]
    low, high = _bootstrap(sample["primary_net_return"].to_numpy(dtype=float), cfg)
    real_mean = float(sample["primary_net_return"].mean()) if len(sample) else math.nan
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ].dropna()
    random_percentile = float(family.le(real_mean).mean()) if len(family) else math.nan
    delayed = delayed_summary[
        delayed_summary["candidate"].eq(PROMOTION_CANDIDATE)
        & delayed_summary["scope"].eq("all")
    ]
    delayed_mean = (
        float(delayed["mean_primary_net_bp"].iloc[0]) / 10_000
        if not delayed.empty
        else math.nan
    )
    sensitivity = sensitivity_summary[
        sensitivity_summary["candidate"].eq(PROMOTION_CANDIDATE)
        & sensitivity_summary["scope"].eq("all")
    ].set_index("reference_hours")
    year_share = _positive_year_share(sample)
    checks: dict[str, tuple[bool, float]] = {
        "full_events_25": (
            int(candidate_summary.loc["all", "events"]) >= 25,
            float(candidate_summary.loc["all", "events"]),
        ),
        "validation_events_5": (
            int(candidate_summary.loc["validation", "events"]) >= 5,
            float(candidate_summary.loc["validation", "events"]),
        ),
        "holdout_events_8": (
            int(candidate_summary.loc["holdout", "events"]) >= 8,
            float(candidate_summary.loc["holdout", "events"]),
        ),
        "filled_fraction_60": (
            candidate_summary.loc["all", "mean_filled_fraction"] >= 0.60,
            float(candidate_summary.loc["all", "mean_filled_fraction"]),
        ),
        "ambiguous_fraction_15": (
            candidate_summary.loc["all", "mean_ambiguous_fraction"] <= 0.15,
            float(candidate_summary.loc["all", "mean_ambiguous_fraction"]),
        ),
        "full_primary_positive": (
            candidate_summary.loc["all", "mean_primary_net_bp"] > 0,
            float(candidate_summary.loc["all", "mean_primary_net_bp"]),
        ),
        "validation_primary_positive": (
            candidate_summary.loc["validation", "mean_primary_net_bp"] > 0,
            float(candidate_summary.loc["validation", "mean_primary_net_bp"]),
        ),
        "holdout_primary_positive": (
            candidate_summary.loc["holdout", "mean_primary_net_bp"] > 0,
            float(candidate_summary.loc["holdout", "mean_primary_net_bp"]),
        ),
        "full_stress_positive": (
            candidate_summary.loc["all", "mean_stress_net_bp"] > 0,
            float(candidate_summary.loc["all", "mean_stress_net_bp"]),
        ),
        "bootstrap_lower_positive": (low > 0, low * 10_000),
        "random_family_percentile_90": (
            random_percentile >= 0.90,
            random_percentile,
        ),
        "beats_one_day_delay": (
            real_mean > delayed_mean,
            (real_mean - delayed_mean) * 10_000,
        ),
        "reference_4h_positive": (
            4 in sensitivity.index
            and sensitivity.loc[4, "mean_primary_net_bp"] > 0,
            float(sensitivity.loc[4, "mean_primary_net_bp"])
            if 4 in sensitivity.index
            else math.nan,
        ),
        "reference_8h_positive": (
            8 in sensitivity.index
            and sensitivity.loc[8, "mean_primary_net_bp"] > 0,
            float(sensitivity.loc[8, "mean_primary_net_bp"])
            if 8 in sensitivity.index
            else math.nan,
        ),
        "positive_year_share_50": (year_share <= 0.50, year_share),
    }
    eligible = all(passed for passed, _ in checks.values())
    gates = pd.DataFrame(
        [
            {
                "candidate": PROMOTION_CANDIDATE,
                "check": check,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for check, (passed, value) in checks.items()
        ]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": PROMOTION_CANDIDATE,
                "events": len(sample),
                "mean_primary_net_bp": real_mean * 10_000,
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_family_percentile": random_percentile,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "positive_year_share": year_share,
                "eligible": eligible,
                "failed_gates": "|".join(
                    check for check, (passed, _) in checks.items() if not passed
                ),
                "verdict": "offline_research_candidate_only"
                if eligible
                else "reject_deribit_skew_receiver_oco",
            }
        ]
    )
    return gates, outcome


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    verdict = str(outcome["verdict"].iloc[0])
    text = [
        "# v17.4 Deribit Skew-Stress Receiver OCO Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "OCO directions use only post-signal hourly high/low touches. Same-hour dual",
        "touches and unfilled allocations remain cash. Option bars are signal-only;",
        "no option execution is simulated. No live permission changes.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v174_deribit_skew_receiver_oco(
    surface_path: Path = SURFACE_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    signal_cfg: V173Config = V173Config(),
    cfg: V174Config = V174Config(),
) -> dict[str, Path]:
    surface = pd.read_parquet(surface_path)
    surface["feature_time"] = pd.to_datetime(surface["feature_time"], utc=True)
    prices = load_v173_prices(kline_root)
    returns = hourly_log_returns(prices)
    ohlc = load_v174_ohlc(kline_root)
    graph = build_monthly_receiver_graph(
        returns,
        surface["feature_time"].min(),
        surface["feature_time"].max(),
        signal_cfg,
    )
    signals = build_surface_signals(surface, signal_cfg)
    events, legs = build_v174_events(signals, graph, ohlc, returns, cfg)
    summary = summarize_v174(events)

    delayed_signals = build_surface_signals(surface, signal_cfg, shift_days=1)
    delayed_events, _ = build_v174_events(
        delayed_signals, graph, ohlc, returns, cfg
    )
    delayed_summary = summarize_v174(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for reference_hours in (4, 8):
        local_events, _ = build_v174_events(
            signals,
            graph,
            ohlc,
            returns,
            cfg,
            reference_hours=reference_hours,
        )
        local_summary = summarize_v174(local_events)
        local_summary["reference_hours"] = reference_hours
        sensitivity_frames.append(local_summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    random_controls = _random_controls(signals, graph, ohlc, cfg)
    gates, outcome = audit_v174(
        events,
        summary,
        delayed_summary,
        sensitivity,
        random_controls,
        cfg,
    )
    root = ensure_dir(report_root)
    paths = {
        "signals": root / "surface_signals.parquet",
        "events": root / "candidate_events.parquet",
        "legs": root / "oco_legs.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "reference_window_sensitivity.csv",
        "random_controls": root / "random_bucket_controls.parquet",
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
    "V174Config",
    "build_v174_events",
    "load_v174_ohlc",
    "simulate_receiver_oco_leg",
    "summarize_v174",
    "write_v174_deribit_skew_receiver_oco",
]
