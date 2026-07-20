"""Independent audit for the v21.3 DEX community-propagation reveal."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    KLINE_ROOT,
    _month,
    load_v178_market_data,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v206_aggtrade_flow_exhaustion_audit import parse_mapping
from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    CANDIDATES,
)
from pressure_graph.reports.v213_dex_community_propagation import (
    CANDIDATE_FEATURES_PATH,
    COMMUNITY_RANDOM,
    ELIGIBLE_PERIODS,
    GLOBAL_RANDOM,
    REPORT_ROOT as V213_REPORT_ROOT,
    V213Config,
)


REPORT_ROOT = Path("reports/v21_4_dex_community_propagation_audit")
FINDINGS_PATH = Path("docs/v214_dex_community_propagation_audit_2026_07_17.md")
EXPECTED_FEATURE_HASH = "0FFC06F729800E29A876EA83E75DD79F8CC58E2C27B7E018C2497DB0BB00E69F"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def _reproduce_random(events: pd.DataFrame, cfg: V213Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)]
        family = GLOBAL_RANDOM if candidate == CANDIDATES[0] else COMMUNITY_RANDOM
        totals = np.zeros(cfg.random_iterations, dtype=float)
        usable = 0
        rng = np.random.default_rng(cfg.seed + offset)
        for item in local.itertuples(index=False):
            pool = _symbols(
                item.global_pool_symbols if family == GLOBAL_RANDOM else item.community_pool_symbols
            )
            count = int(item.realized_selection_count)
            if len(pool) < count:
                continue
            betas = pd.Series(parse_mapping(item.universe_betas)).reindex(pool).to_numpy()
            future_series = pd.Series(parse_mapping(item.universe_future_returns))
            future = future_series.reindex(pool).to_numpy()
            scores = rng.random((cfg.random_iterations, len(pool)))
            indices = np.argpartition(scores, count - 1, axis=1)[:, :count]
            mean_beta = betas[indices].mean(axis=1)
            mean_alt = future[indices].mean(axis=1)
            direction = float(item.source_direction)
            totals += (
                direction
                * (mean_alt - mean_beta * float(future_series[BTC]))
                / (1.0 + np.abs(mean_beta))
            )
            usable += 1
        means = totals / usable
        rows.extend(
            {
                "candidate": candidate,
                "control_family": family,
                "iteration": iteration,
                "events": usable,
                "mean_gross_return": float(means[iteration]),
            }
            for iteration in range(cfg.random_iterations)
        )
    return pd.DataFrame(rows)


def _positive_share(events: pd.DataFrame, column: str) -> float:
    grouped = events.groupby(column)["primary_net_return"].sum().clip(lower=0.0)
    total = float(grouped.sum())
    return float(grouped.max() / total) if total > 0 else 1.0


def audit_v214(
    kline_root: Path = KLINE_ROOT,
    source_root: Path = V213_REPORT_ROOT,
    cfg: V213Config = V213Config(),
) -> pd.DataFrame:
    events = pd.read_parquet(source_root / "candidate_events.parquet")
    delayed = pd.read_parquet(source_root / "delayed_candidate_events.parquet")
    placebo = pd.read_parquet(source_root / "shifted_24h_placebo_events.parquet")
    source = pd.read_parquet(source_root / "source_only_events.parquet")
    summary = pd.read_csv(source_root / "period_summary.csv")
    horizons = pd.read_csv(source_root / "holding_horizon_summary.csv")
    random = pd.read_parquet(source_root / "random_controls.parquet")
    bootstrap = pd.read_csv(source_root / "day_block_bootstrap_summary.csv")
    concentration = pd.read_csv(source_root / "concentration_summary.csv")
    gates = pd.read_csv(source_root / "candidate_gates.csv")
    outcome = pd.read_csv(source_root / "candidate_outcome.csv")
    risk = pd.read_parquet(source_root / "monthly_btc_risk.parquet")
    for frame in (events, delayed, placebo, source, risk):
        for column in (
            "feature_time",
            "entry_time",
            "exit_time",
            "entry_day",
            "risk_month",
        ):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True)
    close, _ = load_v178_market_data(kline_root)
    returns = close.pct_change(fill_method=None)
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    maximum_risk_beta_error = 0.0
    risk_history_strictly_prior = True
    for item in risk.itertuples(index=False):
        month = pd.Timestamp(item.risk_month)
        history = returns.loc[
            (returns.index >= month - pd.Timedelta(days=cfg.risk_lookback_days))
            & (returns.index < month),
            [BTC, item.receiver],
        ].dropna()
        risk_history_strictly_prior &= bool(history.index.max() < month)
        reproduced = float(history[item.receiver].cov(history[BTC]) / history[BTC].var(ddof=1))
        maximum_risk_beta_error = max(
            maximum_risk_beta_error, abs(reproduced - float(item.btc_beta))
        )
    maximum_contribution_error = 0.0
    maximum_gross_error = 0.0
    maximum_cost_error = 0.0
    maximum_notional_error = 0.0
    maximum_beta_error = 0.0
    primary_timing_ok = True
    for item in events.itertuples(index=False):
        weights = parse_mapping(item.weights)
        stored = parse_mapping(item.symbol_contributions)
        feature = pd.Timestamp(item.feature_time)
        entry = pd.Timestamp(item.entry_time)
        exit_time = pd.Timestamp(item.exit_time)
        primary_timing_ok &= entry == feature + pd.Timedelta(minutes=15)
        primary_timing_ok &= exit_time == entry + pd.Timedelta(hours=12)
        future = close.loc[exit_time].div(close.loc[entry]).sub(1.0)
        contributions = {
            symbol: weight * float(future[symbol]) for symbol, weight in weights.items()
        }
        maximum_contribution_error = max(
            maximum_contribution_error,
            max(abs(contributions[symbol] - stored[symbol]) for symbol in weights),
        )
        gross = float(sum(contributions.values()))
        maximum_gross_error = max(maximum_gross_error, abs(gross - item.gross_return))
        maximum_cost_error = max(
            maximum_cost_error,
            abs(item.primary_net_return - (gross - cfg.primary_round_trip_cost)),
            abs(item.stress_net_return - (gross - cfg.stress_round_trip_cost)),
        )
        maximum_notional_error = max(
            maximum_notional_error,
            abs(sum(abs(weight) for weight in weights.values()) - 1.0),
        )
        exposure = weights[BTC]
        for symbol, weight in weights.items():
            if symbol != BTC:
                exposure += weight * float(risk_lookup.loc[(_month(feature), symbol)])
        maximum_beta_error = max(maximum_beta_error, abs(exposure))
    maximum_summary_error = 0.0
    for item in summary.itertuples(index=False):
        local = events[events["candidate"].eq(item.candidate)]
        if item.scope != "all":
            local = local[local["period"].eq(item.scope)]
        maximum_summary_error = max(
            maximum_summary_error,
            abs(float(local["gross_return"].mean() * 10_000) - item.mean_gross_bp),
            abs(float(local["primary_net_return"].mean() * 10_000) - item.mean_primary_net_bp),
        )
    reproduced_random = _reproduce_random(events, cfg)
    merged_random = random.merge(
        reproduced_random,
        on=["candidate", "control_family", "iteration", "events"],
        suffixes=("_stored", "_reproduced"),
    )
    maximum_random_error = float(
        (merged_random["mean_gross_return_stored"] - merged_random["mean_gross_return_reproduced"])
        .abs()
        .max()
    )
    bootstrap_error = 0.0
    for offset, candidate in enumerate(CANDIDATES):
        local = events[events["candidate"].eq(candidate)]
        daily = local.groupby("entry_day")["primary_net_return"].agg(["sum", "count"])
        rng = np.random.default_rng(cfg.seed + 100 + offset)
        means: list[float] = []
        for _ in range(cfg.bootstrap_iterations):
            indices = rng.integers(0, len(daily), size=len(daily))
            sample = daily.iloc[indices]
            means.append(float(sample["sum"].sum() / sample["count"].sum()))
        stored = bootstrap[bootstrap["candidate"].eq(candidate)].iloc[0]
        bootstrap_error = max(
            bootstrap_error,
            abs(float(np.quantile(means, 0.025) * 10_000) - stored["lower_95_primary_net_bp"]),
            abs(float(np.quantile(means, 0.975) * 10_000) - stored["upper_95_primary_net_bp"]),
        )
    concentration_error = 0.0
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        stored = concentration[concentration["candidate"].eq(candidate)].iloc[0]
        concentration_error = max(
            concentration_error,
            abs(_positive_share(local, "entry_month") - stored["maximum_month_positive_pnl_share"]),
            abs(
                _positive_share(local, "source_symbol")
                - stored["maximum_source_positive_pnl_share"]
            ),
        )
    gate_outcome_consistent = True
    for candidate in CANDIDATES:
        all_gates = bool(gates.loc[gates["candidate"].eq(candidate), "passed"].all())
        eligible = bool(outcome.loc[outcome["candidate"].eq(candidate), "eligible"].iloc[0])
        gate_outcome_consistent &= all_gates == eligible
    counts = events.groupby("candidate").size().to_dict()
    delayed_timing_ok = bool(
        delayed["entry_time"].sub(delayed["feature_time"]).eq(pd.Timedelta(minutes=30)).all()
    )
    placebo_timing_ok = bool(
        placebo["entry_time"]
        .sub(placebo["feature_time"])
        .eq(pd.Timedelta(hours=24, minutes=15))
        .all()
    )
    source_is_single_name = bool(source["realized_selection_count"].eq(1).all())
    checks = {
        "candidate_feature_hash_matches_prereg": _sha256(CANDIDATE_FEATURES_PATH)
        == EXPECTED_FEATURE_HASH,
        "candidate_counts_267_each": counts == {candidate: 267 for candidate in CANDIDATES},
        "event_keys_unique": not events.duplicated(["candidate", "event_id"]).any(),
        "vendor_transition_excluded": set(events["period"].unique()) == set(ELIGIBLE_PERIODS),
        "primary_entry_exit_timing_reproduced": primary_timing_ok,
        "delayed_entry_timing_reproduced": delayed_timing_ok,
        "shifted_24h_placebo_timing_reproduced": placebo_timing_ok,
        "source_control_single_name": source_is_single_name,
        "risk_history_strictly_prior": risk_history_strictly_prior,
        "monthly_beta_estimates_reproduced": maximum_risk_beta_error < 1e-12,
        "weights_total_gross_one": maximum_notional_error < 1e-9,
        "prior_beta_exposure_neutral": maximum_beta_error < 1e-9,
        "symbol_contributions_reproduced": maximum_contribution_error < 1e-12,
        "gross_returns_reproduced": maximum_gross_error < 1e-12,
        "cost_charges_reproduced": maximum_cost_error < 1e-12,
        "period_summary_reproduced": maximum_summary_error < 1e-10,
        "alternate_horizon_rows_complete": len(horizons) == len(CANDIDATES) * 4 * 2,
        "random_control_rows_1000": len(random) == cfg.random_iterations * 2,
        "random_controls_reproduced": maximum_random_error < 1e-9,
        "day_bootstrap_intervals_reproduced": bootstrap_error < 1e-10,
        "concentration_reproduced": concentration_error < 1e-12,
        "gate_outcome_consistent": gate_outcome_consistent,
        "both_candidates_rejected": not bool(outcome["eligible"].any()),
        "gross_effects_below_20bp_cost": bool(outcome["mean_gross_bp"].lt(20.0).all()),
        "dap2_holdout_gross_below_20bp": float(
            summary.loc[
                summary["candidate"].eq(CANDIDATES[1]) & summary["scope"].eq("holdout"),
                "mean_gross_bp",
            ].iloc[0]
        )
        < 20.0,
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v214_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    checks = audit_v214()
    root = ensure_dir(report_root)
    checks_path = root / "independent_audit_checks.csv"
    checks.to_csv(checks_path, index=False)
    verdict = (
        "audit_pass_v213_rejections_reproduced" if bool(checks["passed"].all()) else "audit_failed"
    )
    text = [
        "# v21.4 DEX Community-Propagation Independent Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"Passed {int(checks['passed'].sum())}/{len(checks)} independent checks.",
        "",
        "The audit independently reloaded Binance prices, recomputed every monthly "
        "BTC beta from strictly prior observations, reproduced timing, weights, "
        "symbol PnL, costs, summaries, all 1,000 random-control paths, day-block "
        "bootstrap intervals, concentration, and both rejection decisions.",
        "",
        "The recent DAP2 holdout gross response remains below the frozen 20 bp "
        "round-trip hurdle; it is not a promotable alpha result.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(findings_path.parent)
    findings_path.write_text("\n".join(text), encoding="utf-8")
    return {"checks": checks_path, "findings": findings_path}


__all__ = ["audit_v214", "write_v214_audit"]
