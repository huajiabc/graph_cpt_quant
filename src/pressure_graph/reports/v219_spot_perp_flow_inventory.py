"""Preregistered spot-versus-perpetual taker-flow inventory reveal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    build_v187_monthly_risk,
)
from pressure_graph.reports.v218_spot_perp_flow_inventory_feature_audit import (
    CANDIDATES,
    GLOBAL_SPREAD,
    PERP_ROOT,
    REPORT_ROOT as V218_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v21_9_spot_perp_flow_inventory")
FINDINGS_PATH = Path("docs/v219_spot_perp_flow_inventory_findings_2026_07_17.md")
CANDIDATE_FEATURES_PATH = V218_REPORT_ROOT / "candidate_feature_events.parquet"
DECISION_FEATURES_PATH = V218_REPORT_ROOT / "decision_symbol_features.parquet"
ELIGIBLE_PERIODS = ("development", "validation", "holdout")
BTC_FALLBACK_PATH = Path("data/external/binance_um_long_history/klines_1h/BTCUSDT.parquet")


@dataclass(frozen=True)
class V219Config:
    holding_hours: int = 12
    primary_round_trip_cost: float = 0.0020
    stress_round_trip_cost: float = 0.0040
    risk_lookback_days: int = 30
    risk_min_samples: int = 480
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 21_900
    minimum_events: int = 300
    minimum_period_events: int = 60
    minimum_active_months: int = 10
    maximum_contribution_share: float = 0.35
    maximum_selection_share: float = 0.35


def _symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def _pairs(value: object) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in str(value).split("|"):
        if ">" in item:
            high, low = item.split(">", 1)
            pairs.append((high, low))
    return pairs


def load_v219_perp_close(
    root: Path,
    symbols: set[str],
) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for symbol in sorted(symbols | {BTC}):
        path = root / f"{symbol}.parquet"
        if symbol == BTC and not path.exists():
            path = BTC_FALLBACK_PATH
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["feature_time", "close"])
        frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True, errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        series = (
            frame.dropna(subset=["feature_time", "close"])
            .drop_duplicates("feature_time", keep="last")
            .sort_values("feature_time")
            .set_index("feature_time")["close"]
            .rename(symbol)
        )
        frames.append(series)
    close = pd.concat(frames, axis=1).sort_index()
    close.index.name = "feature_time"
    return close


def beta_neutral_spread_weights(
    longs: list[str],
    shorts: list[str],
    beta: pd.Series,
) -> dict[str, float]:
    if not longs or not shorts or set(longs) & set(shorts):
        return {}
    raw = {symbol: 0.5 / len(longs) for symbol in longs}
    raw.update({symbol: -0.5 / len(shorts) for symbol in shorts})
    hedge = -float(sum(weight * float(beta[symbol]) for symbol, weight in raw.items()))
    gross = float(sum(abs(weight) for weight in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def build_v219_events(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V219Config = V219Config(),
    *,
    holding_hours: int | None = None,
    additional_entry_delay_hours: int = 0,
) -> pd.DataFrame:
    horizon = cfg.holding_hours if holding_hours is None else holding_hours
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    decisions_by_time = {
        pd.Timestamp(timestamp): local.copy()
        for timestamp, local in decisions.groupby("feature_time", sort=False)
    }
    rows: list[dict[str, object]] = []
    for item in candidates.itertuples(index=False):
        feature_time = pd.Timestamp(item.feature_time)
        entry_time = pd.Timestamp(item.entry_time) + pd.Timedelta(
            hours=additional_entry_delay_hours
        )
        exit_time = entry_time + pd.Timedelta(hours=horizon)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        longs = _symbols(item.long_symbols)
        shorts = _symbols(item.short_symbols)
        selected = longs + shorts
        month = _month(feature_time)
        endpoint = close.reindex(index=[entry_time, exit_time], columns=[BTC, *selected])
        beta = pd.Series(
            {symbol: risk_lookup.get((month, symbol), np.nan) for symbol in selected},
            dtype=float,
        )
        if endpoint.isna().any().any() or beta.isna().any():
            continue
        weights = beta_neutral_spread_weights(longs, shorts, beta)
        if not weights:
            continue
        future = endpoint.loc[exit_time].div(endpoint.loc[entry_time]).sub(1.0)
        contributions = {
            symbol: float(weight * future[symbol]) for symbol, weight in weights.items()
        }
        gross = float(sum(contributions.values()))
        local = decisions_by_time[feature_time]
        eligible = local[local["feature_eligible"]]
        pool = []
        for symbol in eligible["symbol"].astype(str):
            if symbol not in close.columns:
                continue
            pair = close.reindex(index=[entry_time, exit_time], columns=[symbol])
            if pair[symbol].notna().all() and np.isfinite(risk_lookup.get((month, symbol), np.nan)):
                pool.append(symbol)
        pool = sorted(set(pool))
        pool_beta = {symbol: float(risk_lookup.loc[(month, symbol)]) for symbol in pool}
        pool_future = close.loc[exit_time, pool].div(close.loc[entry_time, pool]).sub(1.0).to_dict()
        pool_future[BTC] = float(future[BTC])
        rows.append(
            {
                **item._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_day": entry_time.floor("D"),
                "holding_hours": horizon,
                "additional_entry_delay_hours": additional_entry_delay_hours,
                "realized_long_count": len(longs),
                "realized_short_count": len(shorts),
                "btc_hedge_weight": float(weights[BTC]),
                "alt_dollar_exposure": float(
                    sum(weight for symbol, weight in weights.items() if symbol != BTC)
                ),
                "residual_btc_beta": float(
                    weights[BTC] + sum(weights[symbol] * float(beta[symbol]) for symbol in selected)
                ),
                "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                "long_contribution": float(sum(contributions[symbol] for symbol in longs)),
                "short_contribution": float(sum(contributions[symbol] for symbol in shorts)),
                "btc_hedge_contribution": float(contributions[BTC]),
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_round_trip_cost,
                "stress_net_return": gross - cfg.stress_round_trip_cost,
                "reversed_primary_net_return": -gross - cfg.primary_round_trip_cost,
                "weights": weights,
                "symbol_contributions": contributions,
                "eligible_pool_symbols": pool,
                "eligible_pool_betas": pool_beta,
                "eligible_pool_future_returns": pool_future,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["entry_time", "candidate"]).reset_index(drop=True)


def summarize_v219(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        for scope in ("all", *ELIGIBLE_PERIODS):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": len(sample),
                    "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                    "active_months": sample["entry_month"].nunique() if len(sample) else 0,
                    "mean_long_count": (
                        float(sample["realized_long_count"].mean()) if len(sample) else math.nan
                    ),
                    "mean_short_count": (
                        float(sample["realized_short_count"].mean()) if len(sample) else math.nan
                    ),
                    "mean_long_bp": (
                        float(sample["long_contribution"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_short_bp": (
                        float(sample["short_contribution"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_btc_hedge_bp": (
                        float(sample["btc_hedge_contribution"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "mean_gross_bp": (
                        float(sample["gross_return"].mean() * 10_000) if len(sample) else math.nan
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
                    "mean_reversed_primary_net_bp": (
                        float(sample["reversed_primary_net_return"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                    "positive_primary_fraction": (
                        float(sample["primary_net_return"].gt(0).mean())
                        if len(sample)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def random_v219_controls(
    events: pd.DataFrame,
    cfg: V219Config = V219Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)]
        totals = np.zeros(cfg.random_iterations, dtype=float)
        usable = 0
        rng = np.random.default_rng(cfg.seed + offset)
        for item in local.itertuples(index=False):
            beta_series = pd.Series(item.eligible_pool_betas, dtype=float)
            future_series = pd.Series(item.eligible_pool_future_returns, dtype=float)
            if candidate == GLOBAL_SPREAD:
                pool = list(item.eligible_pool_symbols)
                long_count = int(item.realized_long_count)
                short_count = int(item.realized_short_count)
                if len(pool) < long_count + short_count:
                    continue
                beta = beta_series.reindex(pool).to_numpy()
                future = future_series.reindex(pool).to_numpy()
                order = np.argsort(rng.random((cfg.random_iterations, len(pool))), axis=1)[
                    :, : long_count + short_count
                ]
                long_index = order[:, :long_count]
                short_index = order[:, long_count:]
                raw_beta = 0.5 * (beta[long_index].mean(axis=1) - beta[short_index].mean(axis=1))
                raw_alt = 0.5 * (future[long_index].mean(axis=1) - future[short_index].mean(axis=1))
            else:
                pairs = _pairs(item.community_pairs)
                if not pairs:
                    continue
                signs = rng.choice(
                    np.array([-1.0, 1.0]),
                    size=(cfg.random_iterations, len(pairs)),
                )
                beta_diff = np.array([beta_series[high] - beta_series[low] for high, low in pairs])
                future_diff = np.array(
                    [future_series[high] - future_series[low] for high, low in pairs]
                )
                raw_beta = 0.5 * (signs * beta_diff).mean(axis=1)
                raw_alt = 0.5 * (signs * future_diff).mean(axis=1)
            hedge = -raw_beta
            totals += (raw_alt + hedge * float(future_series[BTC])) / (1.0 + np.abs(hedge))
            usable += 1
        means = totals / usable
        rows.extend(
            {
                "candidate": candidate,
                "iteration": iteration,
                "events": usable,
                "mean_gross_return": float(means[iteration]),
            }
            for iteration in range(cfg.random_iterations)
        )
    return pd.DataFrame(rows)


def day_block_bootstrap_v219(
    events: pd.DataFrame,
    cfg: V219Config = V219Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)]
        daily = local.groupby("entry_day")["primary_net_return"].agg(["sum", "count"])
        rng = np.random.default_rng(cfg.seed + 100 + offset)
        means: list[float] = []
        for _ in range(cfg.bootstrap_iterations):
            indices = rng.integers(0, len(daily), size=len(daily))
            sample = daily.iloc[indices]
            means.append(float(sample["sum"].sum() / sample["count"].sum()))
        rows.append(
            {
                "candidate": candidate,
                "events": len(local),
                "active_days": len(daily),
                "mean_primary_net_bp": float(local["primary_net_return"].mean() * 10_000),
                "lower_95_primary_net_bp": float(np.quantile(means, 0.025) * 10_000),
                "upper_95_primary_net_bp": float(np.quantile(means, 0.975) * 10_000),
            }
        )
    return pd.DataFrame(rows)


def _month_share(events: pd.DataFrame) -> float:
    grouped = events.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    total = float(grouped.sum())
    return float(grouped.max() / total) if total > 0 else 1.0


def _selection_share(events: pd.DataFrame) -> float:
    counts: dict[str, int] = {}
    for item in events.itertuples(index=False):
        for symbol in set(_symbols(item.long_symbols) + _symbols(item.short_symbols)):
            counts[symbol] = counts.get(symbol, 0) + 1
    return max(counts.values()) / len(events) if counts and len(events) else 1.0


def concentration_v219(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "maximum_month_positive_pnl_share": _month_share(
                    events[events["candidate"].eq(candidate)]
                ),
                "maximum_symbol_selection_share": _selection_share(
                    events[events["candidate"].eq(candidate)]
                ),
            }
            for candidate in CANDIDATES
        ]
    )


def cost_frontier_v219(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        gross = float(local["gross_return"].mean() * 10_000)
        for cost in (5, 10, 20, 40):
            rows.append(
                {
                    "candidate": candidate,
                    "round_trip_cost_bp": cost,
                    "events": len(local),
                    "mean_gross_bp": gross,
                    "mean_net_bp": gross - cost,
                }
            )
    return pd.DataFrame(rows)


def audit_v219(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    bootstrap: pd.DataFrame,
    concentration: pd.DataFrame,
    cfg: V219Config = V219Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        table = summary[summary["candidate"].eq(candidate)].set_index("scope")
        delayed = delayed_summary[
            delayed_summary["candidate"].eq(candidate) & delayed_summary["scope"].eq("all")
        ].iloc[0]
        placebo = placebo_summary[
            placebo_summary["candidate"].eq(candidate) & placebo_summary["scope"].eq("all")
        ].iloc[0]
        boot = bootstrap[bootstrap["candidate"].eq(candidate)].iloc[0]
        conc = concentration[concentration["candidate"].eq(candidate)].iloc[0]
        random = random_controls[random_controls["candidate"].eq(candidate)]["mean_gross_return"]
        gross = float(local["gross_return"].mean())
        percentile = float(random.le(gross).mean())
        period_events = [int(table.loc[period, "events"]) for period in ELIGIBLE_PERIODS]
        period_net = [
            float(table.loc[period, "mean_primary_net_bp"]) for period in ELIGIBLE_PERIODS
        ]
        checks = {
            "minimum_total_events": (
                len(local) >= cfg.minimum_events,
                len(local),
                cfg.minimum_events,
            ),
            "minimum_each_period_events": (
                min(period_events) >= cfg.minimum_period_events,
                min(period_events),
                cfg.minimum_period_events,
            ),
            "minimum_active_months": (
                int(table.loc["all", "active_months"]) >= cfg.minimum_active_months,
                int(table.loc["all", "active_months"]),
                cfg.minimum_active_months,
            ),
            "gross_exceeds_20bp_cost": (
                gross > cfg.primary_round_trip_cost,
                gross * 10_000,
                20.0,
            ),
            "all_period_primary_net_positive": (
                min(period_net) > 0,
                min(period_net),
                0.0,
            ),
            "delayed_primary_net_positive": (
                float(delayed["mean_primary_net_bp"]) > 0,
                float(delayed["mean_primary_net_bp"]),
                0.0,
            ),
            "reversed_primary_net_negative": (
                float(table.loc["all", "mean_reversed_primary_net_bp"]) < 0,
                float(table.loc["all", "mean_reversed_primary_net_bp"]),
                0.0,
            ),
            "random_control_percentile_at_least_0_95": (
                percentile >= 0.95,
                percentile,
                0.95,
            ),
            "day_bootstrap_lower_95_positive": (
                float(boot["lower_95_primary_net_bp"]) > 0,
                float(boot["lower_95_primary_net_bp"]),
                0.0,
            ),
            "shifted_24h_placebo_net_nonpositive": (
                float(placebo["mean_primary_net_bp"]) <= 0,
                float(placebo["mean_primary_net_bp"]),
                0.0,
            ),
            "month_contribution_at_most_35pct": (
                float(conc["maximum_month_positive_pnl_share"]) <= cfg.maximum_contribution_share,
                float(conc["maximum_month_positive_pnl_share"]),
                cfg.maximum_contribution_share,
            ),
            "symbol_selection_at_most_35pct": (
                float(conc["maximum_symbol_selection_share"]) <= cfg.maximum_selection_share,
                float(conc["maximum_symbol_selection_share"]),
                cfg.maximum_selection_share,
            ),
        }
        for check, (passed, value, threshold) in checks.items():
            gate_rows.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "passed": bool(passed),
                    "value": value,
                    "threshold": threshold,
                }
            )
        eligible = all(passed for passed, _, _ in checks.values())
        outcomes.append(
            {
                "candidate": candidate,
                "events": len(local),
                "mean_gross_bp": gross * 10_000,
                "mean_primary_net_bp": float(table.loc["all", "mean_primary_net_bp"]),
                "mean_stress_net_bp": float(table.loc["all", "mean_stress_net_bp"]),
                "random_control_percentile": percentile,
                "day_bootstrap_lower_95_primary_net_bp": float(boot["lower_95_primary_net_bp"]),
                "eligible": eligible,
                "status": ("offline_candidate_natural_forward_only" if eligible else "rejected"),
            }
        )
    return pd.DataFrame(gate_rows), pd.DataFrame(outcomes)


def _serialize_mapping(value: dict[str, float]) -> str:
    return "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))


def _serialize_symbols(value: list[str]) -> str:
    return "|".join(value)


def _write_event_frame(frame: pd.DataFrame, path: Path) -> None:
    serial = frame.copy()
    for column in (
        "weights",
        "symbol_contributions",
        "eligible_pool_betas",
        "eligible_pool_future_returns",
    ):
        serial[column] = serial[column].map(_serialize_mapping)
    serial["eligible_pool_symbols"] = serial["eligible_pool_symbols"].map(_serialize_symbols)
    serial.to_parquet(path, index=False)


def _write_findings(
    outcome: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "offline_candidate_natural_forward_only"
        if bool(outcome["eligible"].any())
        else "reject_spot_perp_flow_inventory_candidates"
    )
    controls = (
        delayed_summary[delayed_summary["scope"].eq("all")][
            ["candidate", "mean_gross_bp", "mean_primary_net_bp"]
        ]
        .rename(
            columns={
                "mean_gross_bp": "delayed_gross_bp",
                "mean_primary_net_bp": "delayed_net20_bp",
            }
        )
        .merge(
            placebo_summary[placebo_summary["scope"].eq("all")][
                ["candidate", "mean_gross_bp", "mean_primary_net_bp"]
            ].rename(
                columns={
                    "mean_gross_bp": "placebo_24h_gross_bp",
                    "mean_primary_net_bp": "placebo_24h_net20_bp",
                }
            ),
            on="candidate",
        )
    )
    text = [
        "# v21.9 Spot-Perpetual Flow-Inventory Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Chronological results:",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Timing controls:",
        "",
        controls.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Alternate horizons:",
        "",
        horizon_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Concentration:",
        "",
        concentration.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The reveal follows the frozen v21.9 preregistration. Spot is information "
        "only; all realized returns are Binance USD-M perpetual returns. Primary "
        "and stress costs are 20/40 bp round trip on unit gross.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v219_reveal(
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    decision_features_path: Path = DECISION_FEATURES_PATH,
    perp_root: Path = PERP_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V219Config = V219Config(),
) -> dict[str, Path]:
    candidates = pd.read_parquet(candidate_features_path)
    decisions = pd.read_parquet(decision_features_path)
    for frame in (candidates, decisions):
        for column in ("feature_time", "entry_time", "entry_day", "month_start"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
    symbols = set(decisions["symbol"].astype(str))
    close = load_v219_perp_close(perp_root, symbols)
    risk = build_v187_monthly_risk(
        close.pct_change(fill_method=None),
        candidates["feature_time"].min(),
        candidates["feature_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    events = build_v219_events(candidates, decisions, risk, close, cfg)
    summary = summarize_v219(events)
    delayed_events = build_v219_events(
        candidates,
        decisions,
        risk,
        close,
        cfg,
        additional_entry_delay_hours=1,
    )
    delayed_summary = summarize_v219(delayed_events)
    placebo_events = build_v219_events(
        candidates,
        decisions,
        risk,
        close,
        cfg,
        additional_entry_delay_hours=24,
    )
    placebo_summary = summarize_v219(placebo_events)
    horizon_frames: list[pd.DataFrame] = []
    for horizon in (4, 24):
        local_events = build_v219_events(
            candidates,
            decisions,
            risk,
            close,
            cfg,
            holding_hours=horizon,
        )
        local = summarize_v219(local_events)
        local["holding_hours"] = horizon
        horizon_frames.append(local)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = random_v219_controls(events, cfg)
    bootstrap = day_block_bootstrap_v219(events, cfg)
    concentration = concentration_v219(events)
    cost_frontier = cost_frontier_v219(events)
    gates, outcome = audit_v219(
        events,
        summary,
        delayed_summary,
        placebo_summary,
        random_controls,
        bootstrap,
        concentration,
        cfg,
    )
    root = ensure_dir(report_root)
    outputs = {
        "risk": root / "monthly_btc_risk.parquet",
        "events": root / "candidate_events.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "placebo_events": root / "shifted_24h_placebo_events.parquet",
        "placebo_summary": root / "shifted_24h_placebo_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "cost_frontier": root / "cost_frontier.csv",
        "random_controls": root / "random_controls.parquet",
        "bootstrap": root / "day_block_bootstrap_summary.csv",
        "concentration": root / "concentration_summary.csv",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    risk.to_parquet(outputs["risk"], index=False)
    for frame, key in (
        (events, "events"),
        (delayed_events, "delayed_events"),
        (placebo_events, "placebo_events"),
    ):
        _write_event_frame(frame, outputs[key])
    summary.to_csv(outputs["summary"], index=False)
    delayed_summary.to_csv(outputs["delayed_summary"], index=False)
    placebo_summary.to_csv(outputs["placebo_summary"], index=False)
    horizon_summary.to_csv(outputs["horizons"], index=False)
    cost_frontier.to_csv(outputs["cost_frontier"], index=False)
    random_controls.to_parquet(outputs["random_controls"], index=False)
    bootstrap.to_csv(outputs["bootstrap"], index=False)
    concentration.to_csv(outputs["concentration"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outcome.to_csv(outputs["outcome"], index=False)
    _write_findings(
        outcome,
        summary,
        delayed_summary,
        placebo_summary,
        horizon_summary,
        concentration,
        findings_path,
    )
    return outputs


__all__ = [
    "V219Config",
    "audit_v219",
    "beta_neutral_spread_weights",
    "build_v219_events",
    "load_v219_perp_close",
    "random_v219_controls",
    "summarize_v219",
    "write_v219_reveal",
]
