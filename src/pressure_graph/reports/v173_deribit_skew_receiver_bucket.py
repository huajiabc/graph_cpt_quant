"""Causal BTC option-skew shocks transmitted into graph receiver buckets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


SURFACE_PATH = Path(
    "data/external/deribit_quarterly_option_trades/daily_trade_surface.parquet"
)
DVOL_PATH = Path("data/external/orthogonal_volatility/deribit_dvol_1h/BTC.parquet")
KLINE_ROOT = Path("data/external/binance_um_long_history/klines_1h")
REPORT_ROOT = Path("reports/v17_3_deribit_skew_receiver_bucket")
FINDINGS_PATH = Path(
    "docs/v173_deribit_skew_receiver_bucket_findings_2026_07_16.md"
)
BTC = "BTCUSDT"
ALTS = (
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ETCUSDT",
)
PROMOTION_CANDIDATES = (
    "DSR1_STRESS_RECEIVER_SHORT",
    "DSR2_STRESS_RECEIVER_BTC_NEUTRAL",
)
ALL_CANDIDATES = (*PROMOTION_CANDIDATES, "DSR3_RELIEF_RECEIVER_LONG")


@dataclass(frozen=True)
class V173Config:
    graph_lookback_days: int = 90
    graph_min_samples: int = 1_000
    max_receivers: int = 4
    min_receivers: int = 3
    robust_lookback: int = 120
    robust_min_periods: int = 40
    signal_threshold: float = 1.0
    maximum_surface_gap_days: int = 2
    cooldown_days: int = 3
    holding_hours: int = 24
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 17_300


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2024-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2025-01-01", tz="UTC"):
        return "validation"
    return "holdout"


def load_v173_prices(root: Path = KLINE_ROOT) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in (BTC, *ALTS):
        frame = pd.read_parquet(root / f"{symbol}.parquet", columns=["feature_time", "close"])
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["symbol"] = symbol
        frames.append(frame[["feature_time", "symbol", "close"]])
    stacked = pd.concat(frames, ignore_index=True).dropna()
    return stacked.pivot_table(
        index="feature_time", columns="symbol", values="close", aggfunc="last"
    ).sort_index()


def hourly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().replace([np.inf, -np.inf], np.nan)


def _bucket_beta(history: pd.DataFrame, receivers: list[str]) -> float:
    pair = pd.DataFrame(
        {
            "btc": history[BTC],
            "bucket": history[receivers].mean(axis=1, skipna=False),
        }
    ).dropna()
    if len(pair) < 2 or float(pair["btc"].var(ddof=1)) <= 0:
        return math.nan
    beta = float(pair["bucket"].cov(pair["btc"]) / pair["btc"].var(ddof=1))
    return float(np.clip(beta, 0.5, 2.0))


def build_monthly_receiver_graph(
    returns: pd.DataFrame,
    first_month: pd.Timestamp,
    last_month: pd.Timestamp,
    cfg: V173Config = V173Config(),
) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(first_month).tz_convert("UTC").replace(day=1, hour=0),
        pd.Timestamp(last_month).tz_convert("UTC").replace(day=1, hour=0),
        freq="MS",
    )
    rows: list[dict[str, object]] = []
    for month in months:
        history = returns[
            (returns.index >= month - pd.Timedelta(days=cfg.graph_lookback_days))
            & (returns.index < month)
        ]
        local: list[dict[str, object]] = []
        for alt in ALTS:
            pair = pd.DataFrame(
                {
                    "btc_lag_abs": history[BTC].abs().shift(1),
                    "alt_abs": history[alt].abs(),
                    "alt_lag_abs": history[alt].abs().shift(1),
                    "btc_abs": history[BTC].abs(),
                }
            ).dropna()
            sample_n = len(pair)
            forward = (
                float(pair["btc_lag_abs"].corr(pair["alt_abs"]))
                if sample_n >= cfg.graph_min_samples
                else math.nan
            )
            reverse = (
                float(pair["alt_lag_abs"].corr(pair["btc_abs"]))
                if sample_n >= cfg.graph_min_samples
                else math.nan
            )
            local.append(
                {
                    "graph_month": month,
                    "receiver": alt,
                    "sample_n": sample_n,
                    "forward_abs_correlation": forward,
                    "reverse_abs_correlation": reverse,
                    "direction_advantage": forward - reverse,
                }
            )
        local_frame = pd.DataFrame(local)
        eligible = local_frame[
            local_frame["sample_n"].ge(cfg.graph_min_samples)
            & local_frame["forward_abs_correlation"].gt(0)
        ].sort_values(
            ["direction_advantage", "forward_abs_correlation"],
            ascending=False,
        )
        selected = eligible.head(cfg.max_receivers)
        selected_names = selected["receiver"].astype(str).tolist()
        beta = (
            _bucket_beta(history, selected_names)
            if len(selected_names) >= cfg.min_receivers
            else math.nan
        )
        rank = {name: index + 1 for index, name in enumerate(selected_names)}
        for row in local:
            name = str(row["receiver"])
            row["selected"] = name in rank and len(selected_names) >= cfg.min_receivers
            row["receiver_rank"] = rank.get(name, math.nan)
            row["bucket_beta"] = beta
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["graph_month", "selected", "receiver_rank", "receiver"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def _rolling_mad(values: np.ndarray) -> float:
    median = float(np.nanmedian(values))
    return float(np.nanmedian(np.abs(values - median)))


def build_surface_signals(
    surface: pd.DataFrame,
    cfg: V173Config = V173Config(),
    threshold: float | None = None,
    shift_days: int = 0,
) -> pd.DataFrame:
    frame = surface[surface["quality_pass"].eq(True)].copy()
    frame["feature_time"] = pd.to_datetime(
        frame["feature_time"], utc=True, errors="coerce"
    )
    frame["expiration_time"] = pd.to_datetime(
        frame["expiration_time"], utc=True, errors="coerce"
    )
    frame = frame.sort_values("feature_time").reset_index(drop=True)
    gap = frame["feature_time"].diff()
    adjacent = (
        frame["expiration_time"].eq(frame["expiration_time"].shift(1))
        & gap.le(pd.Timedelta(days=cfg.maximum_surface_gap_days))
    )
    frame["risk_reversal_innovation"] = frame["downside_risk_reversal"].diff().where(
        adjacent
    )
    frame["atm_iv_innovation"] = frame["atm_iv"].diff().where(adjacent)
    prior = frame["risk_reversal_innovation"].shift(1).rolling(
        cfg.robust_lookback, min_periods=cfg.robust_min_periods
    )
    center = prior.median()
    scale = 1.4826 * prior.apply(_rolling_mad, raw=True)
    frame["risk_reversal_robust_z"] = (
        (frame["risk_reversal_innovation"] - center) / scale.replace(0.0, np.nan)
    )
    cutoff = cfg.signal_threshold if threshold is None else threshold
    frame["event_type"] = ""
    frame.loc[
        frame["risk_reversal_robust_z"].ge(cutoff)
        & frame["atm_iv_innovation"].gt(0),
        "event_type",
    ] = "stress"
    frame.loc[
        frame["risk_reversal_robust_z"].le(-cutoff)
        & frame["atm_iv_innovation"].lt(0),
        "event_type",
    ] = "relief"
    accepted: list[int] = []
    last_time: pd.Timestamp | None = None
    cooldown = pd.Timedelta(days=cfg.cooldown_days)
    for index, row in frame[frame["event_type"].ne("")].iterrows():
        timestamp = pd.Timestamp(row["feature_time"])
        if last_time is None or timestamp - last_time >= cooldown:
            accepted.append(index)
            last_time = timestamp
    result = frame.loc[accepted].copy()
    result["source_feature_time"] = result["feature_time"]
    if shift_days:
        result["feature_time"] += pd.Timedelta(days=shift_days)
    result["threshold"] = cutoff
    result["signal_shift_days"] = shift_days
    return result.reset_index(drop=True)


def _selected_receivers(graph: pd.DataFrame, month: pd.Timestamp) -> tuple[list[str], float]:
    local = graph[graph["graph_month"].eq(month) & graph["selected"].eq(True)].sort_values(
        "receiver_rank"
    )
    if local.empty:
        return [], math.nan
    return local["receiver"].astype(str).tolist(), float(local["bucket_beta"].iloc[0])


def build_candidate_events(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V173Config = V173Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        entry_time = pd.Timestamp(signal.feature_time)
        exit_time = entry_time + pd.Timedelta(hours=cfg.holding_hours)
        graph_month = entry_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        receivers, beta = _selected_receivers(graph, graph_month)
        if len(receivers) < cfg.min_receivers or not np.isfinite(beta):
            continue
        required = [BTC, *receivers]
        if (
            entry_time not in prices.index
            or exit_time not in prices.index
            or prices.loc[[entry_time, exit_time], required].isna().any().any()
        ):
            continue
        returns = prices.loc[exit_time, required] / prices.loc[entry_time, required] - 1.0
        bucket_return = float(returns[receivers].mean())
        btc_return = float(returns[BTC])
        common: dict[str, Any] = {
            **signal._asdict(),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "period": _period(entry_time),
            "year": entry_time.year,
            "graph_month": graph_month,
            "receivers": "|".join(receivers),
            "receiver_count": len(receivers),
            "bucket_beta": beta,
            "bucket_return": bucket_return,
            "btc_return": btc_return,
        }
        if signal.event_type == "stress":
            dsr1 = -bucket_return
            rows.append(
                {
                    **common,
                    "candidate": "DSR1_STRESS_RECEIVER_SHORT",
                    "gross_return": dsr1,
                    "primary_net_return": dsr1 - 0.0020,
                    "stress_net_return": dsr1 - 0.0040,
                }
            )
            dsr2 = (-bucket_return + beta * btc_return) / (1.0 + beta)
            rows.append(
                {
                    **common,
                    "candidate": "DSR2_STRESS_RECEIVER_BTC_NEUTRAL",
                    "gross_return": dsr2,
                    "primary_net_return": dsr2 - 0.0030,
                    "stress_net_return": dsr2 - 0.0050,
                }
            )
        elif signal.event_type == "relief":
            rows.append(
                {
                    **common,
                    "candidate": "DSR3_RELIEF_RECEIVER_LONG",
                    "gross_return": bucket_return,
                    "primary_net_return": bucket_return - 0.0020,
                    "stress_net_return": bucket_return - 0.0040,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["entry_time", "candidate"]).reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
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
                    "sum_primary_net": float(sample["primary_net_return"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _random_bucket_controls(
    real_events: pd.DataFrame,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V173Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    event_base = real_events.drop_duplicates("entry_time").sort_values("entry_time")
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in ALL_CANDIDATES}
        for event in event_base.itertuples(index=False):
            entry_time = pd.Timestamp(event.entry_time)
            exit_time = pd.Timestamp(event.exit_time)
            receiver_count = int(event.receiver_count)
            sample = sorted(rng.choice(ALTS, size=receiver_count, replace=False).tolist())
            if prices.loc[[entry_time, exit_time], sample].isna().any().any():
                continue
            bucket_return = float(
                (prices.loc[exit_time, sample] / prices.loc[entry_time, sample] - 1.0).mean()
            )
            history = returns[
                (returns.index >=
                    event.graph_month - pd.Timedelta(days=cfg.graph_lookback_days)
                )
                & (returns.index < event.graph_month)
            ]
            beta = _bucket_beta(history, sample)
            if not np.isfinite(beta):
                continue
            btc_return = float(prices.loc[exit_time, BTC] / prices.loc[entry_time, BTC] - 1.0)
            if event.event_type == "stress":
                values["DSR1_STRESS_RECEIVER_SHORT"].append(-bucket_return - 0.0020)
                values["DSR2_STRESS_RECEIVER_BTC_NEUTRAL"].append(
                    (-bucket_return + beta * btc_return) / (1.0 + beta) - 0.0030
                )
            else:
                values["DSR3_RELIEF_RECEIVER_LONG"].append(bucket_return - 0.0020)
        means: dict[str, float] = {}
        for candidate, candidate_values in values.items():
            means[candidate] = (
                float(np.mean(candidate_values)) if candidate_values else math.nan
            )
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "events": len(candidate_values),
                    "mean_primary_net_return": means[candidate],
                }
            )
        family_values = [
            means[candidate]
            for candidate in PROMOTION_CANDIDATES
            if np.isfinite(means[candidate])
        ]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "events": int(
                    max(
                        len(values[candidate]) for candidate in PROMOTION_CANDIDATES
                    )
                ),
                "mean_primary_net_return": max(family_values)
                if family_values
                else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_interval(values: np.ndarray, cfg: V173Config, offset: int) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed + offset)
    means = rng.choice(
        values,
        size=(cfg.bootstrap_iterations, len(values)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_year_share(sample: pd.DataFrame) -> float:
    yearly = sample.groupby("year")["primary_net_return"].sum().clip(lower=0)
    return float(yearly.max() / yearly.sum()) if yearly.sum() > 0 else math.inf


def _data_quality(surface: pd.DataFrame, dvol: pd.DataFrame) -> dict[str, float]:
    valid = surface[surface["quality_pass"].eq(True)].copy()
    valid["feature_time"] = pd.to_datetime(valid["feature_time"], utc=True)
    dvol = dvol.copy()
    dvol["feature_time"] = (
        pd.to_datetime(dvol["dvol_time"], utc=True).dt.floor("D")
        + pd.Timedelta(days=1)
    )
    daily = dvol.groupby("feature_time")["close"].mean().div(100.0).rename("dvol")
    overlap = valid.merge(daily, on="feature_time")
    return {
        "quality_surface_rows": float(len(valid)),
        "quarterly_expiries": float(valid["expiration_time"].nunique()),
        "dvol_overlap_rows": float(len(overlap)),
        "atm_dvol_correlation": float(overlap["atm_iv"].corr(overlap["dvol"])),
    }


def audit_v173(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    data_quality: dict[str, float],
    cfg: V173Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gates: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ].dropna()
    for candidate_index, candidate in enumerate(PROMOTION_CANDIDATES):
        candidate_summary = summary[summary["candidate"].eq(candidate)].set_index("scope")
        delayed = delayed_summary[
            (delayed_summary["candidate"].eq(candidate))
            & delayed_summary["scope"].eq("all")
        ]
        delayed_mean = (
            float(delayed["mean_primary_net_bp"].iloc[0]) / 10_000
            if not delayed.empty
            else math.nan
        )
        sensitivity = sensitivity_summary[
            sensitivity_summary["candidate"].eq(candidate)
            & sensitivity_summary["scope"].eq("all")
        ].set_index("threshold")
        sample = events[events["candidate"].eq(candidate)]
        low, high = _bootstrap_interval(
            sample["primary_net_return"].to_numpy(dtype=float), cfg, candidate_index
        )
        real_mean = float(sample["primary_net_return"].mean()) if len(sample) else math.nan
        random_percentile = float(family.le(real_mean).mean()) if len(family) else math.nan
        year_share = _positive_year_share(sample)
        checks: dict[str, tuple[bool, float]] = {
            "quality_surface_rows_400": (
                data_quality["quality_surface_rows"] >= 400,
                data_quality["quality_surface_rows"],
            ),
            "quarterly_expiries_16": (
                data_quality["quarterly_expiries"] >= 16,
                data_quality["quarterly_expiries"],
            ),
            "atm_dvol_correlation_80": (
                data_quality["atm_dvol_correlation"] >= 0.80,
                data_quality["atm_dvol_correlation"],
            ),
            "full_events_30": (
                int(candidate_summary.loc["all", "events"]) >= 30,
                float(candidate_summary.loc["all", "events"]),
            ),
            "validation_events_5": (
                int(candidate_summary.loc["validation", "events"]) >= 5,
                float(candidate_summary.loc["validation", "events"]),
            ),
            "holdout_events_10": (
                int(candidate_summary.loc["holdout", "events"]) >= 10,
                float(candidate_summary.loc["holdout", "events"]),
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
            "threshold_075_positive": (
                0.75 in sensitivity.index
                and sensitivity.loc[0.75, "mean_primary_net_bp"] > 0,
                float(sensitivity.loc[0.75, "mean_primary_net_bp"])
                if 0.75 in sensitivity.index
                else math.nan,
            ),
            "threshold_125_positive": (
                1.25 in sensitivity.index
                and sensitivity.loc[1.25, "mean_primary_net_bp"] > 0,
                float(sensitivity.loc[1.25, "mean_primary_net_bp"])
                if 1.25 in sensitivity.index
                else math.nan,
            ),
            "positive_year_share_50": (year_share <= 0.50, year_share),
        }
        eligible = all(passed for passed, _ in checks.values())
        for check, (passed, value) in checks.items():
            gates.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "passed": bool(passed),
                    "value": float(value),
                    "eligible": eligible,
                }
            )
        outcomes.append(
            {
                "candidate": candidate,
                "events": int(len(sample)),
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
            }
        )
    outcome = pd.DataFrame(outcomes)
    outcome["verdict"] = np.where(
        outcome["eligible"],
        "offline_research_candidate_only",
        "reject_deribit_skew_receiver_alpha",
    )
    return pd.DataFrame(gates), outcome


def _write_findings(
    outputs: dict[str, pd.DataFrame],
    path: Path,
) -> None:
    outcome = outputs["outcome"]
    verdict = (
        "offline_research_candidate_only"
        if bool(outcome["eligible"].any())
        else "reject_deribit_skew_receiver_alpha"
    )
    text = [
        "# v17.3 Deribit Skew-to-Receiver-Bucket Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        outputs["summary"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "BTC option trade surfaces are used only as completed-day signals. The",
        "traded legs are Binance USD-M closed-bar returns; option execution is not",
        "simulated. The BTC-neutral result is normalized to unit gross exposure.",
        "No PaperLive, application, remote, leverage, or real-order permission changes.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v173_deribit_skew_receiver_bucket(
    surface_path: Path = SURFACE_PATH,
    dvol_path: Path = DVOL_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V173Config = V173Config(),
) -> dict[str, Path]:
    surface = pd.read_parquet(surface_path)
    surface["feature_time"] = pd.to_datetime(surface["feature_time"], utc=True)
    dvol = pd.read_parquet(dvol_path)
    prices = load_v173_prices(kline_root)
    returns = hourly_log_returns(prices)
    graph = build_monthly_receiver_graph(
        returns,
        surface["feature_time"].min(),
        surface["feature_time"].max(),
        cfg,
    )
    signals = build_surface_signals(surface, cfg)
    events = build_candidate_events(signals, graph, prices, cfg)
    summary = summarize_events(events)

    delayed_signals = build_surface_signals(surface, cfg, shift_days=1)
    delayed_events = build_candidate_events(delayed_signals, graph, prices, cfg)
    delayed_summary = summarize_events(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for threshold in (0.75, 1.25):
        local_signals = build_surface_signals(surface, cfg, threshold=threshold)
        local_events = build_candidate_events(local_signals, graph, prices, cfg)
        local_summary = summarize_events(local_events)
        local_summary["threshold"] = threshold
        sensitivity_frames.append(local_summary)
    sensitivity_summary = pd.concat(sensitivity_frames, ignore_index=True)
    random_controls = _random_bucket_controls(events, returns, prices, cfg)
    quality = _data_quality(surface, dvol)
    gates, outcome = audit_v173(
        events,
        summary,
        delayed_summary,
        sensitivity_summary,
        random_controls,
        quality,
        cfg,
    )
    quality_frame = pd.DataFrame([quality])
    frames = {
        "summary": summary,
        "outcome": outcome,
    }
    root = ensure_dir(report_root)
    paths = {
        "graph": root / "monthly_receiver_graph.parquet",
        "signals": root / "surface_signals.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "threshold_sensitivity.csv",
        "random_controls": root / "random_bucket_controls.parquet",
        "data_quality": root / "data_quality.csv",
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
    sensitivity_summary.to_csv(paths["sensitivity"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    quality_frame.to_csv(paths["data_quality"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(frames, findings_path)
    return paths


__all__ = [
    "V173Config",
    "build_candidate_events",
    "build_monthly_receiver_graph",
    "build_surface_signals",
    "hourly_log_returns",
    "load_v173_prices",
    "summarize_events",
    "write_v173_deribit_skew_receiver_bucket",
]
