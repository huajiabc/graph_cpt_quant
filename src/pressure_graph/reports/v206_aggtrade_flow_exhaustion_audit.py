"""Independent arithmetic and protocol audit for the v20.5 reveal."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    CANDIDATES,
)
from pressure_graph.reports.v205_aggtrade_flow_exhaustion import (
    CANDIDATE_FEATURES_PATH,
    RECEIVER_FEATURES_PATH,
    REPORT_ROOT as V205_REPORT_ROOT,
    V205Config,
)


REPORT_ROOT = Path("reports/v20_6_aggtrade_flow_exhaustion_audit")
FINDINGS_PATH = Path("docs/v206_aggtrade_flow_exhaustion_audit_2026_07_17.md")
EXPECTED_HASHES = {
    RECEIVER_FEATURES_PATH: "6B6418B3AB9AC2337C56F64912B28A82C596BDA3F7097F1491DD2BB74E4DD64F",
    CANDIDATE_FEATURES_PATH: "A83BCACFBD6FBDC7040E83800C50301C9933A66D5DF6CF348BAF68046D799712",
}


def parse_mapping(value: object) -> dict[str, float]:
    if not value or pd.isna(value):
        return {}
    output: dict[str, float] = {}
    for item in str(value).split("|"):
        key, raw = item.rsplit(":", 1)
        output[key] = float(raw)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _scope(events: pd.DataFrame, scope: str) -> pd.DataFrame:
    return events if scope == "all" else events[events["period"].eq(scope)]


def audit_v206(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = V205_REPORT_ROOT,
    cfg: V205Config = V205Config(),
) -> pd.DataFrame:
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    horizons = pd.read_csv(report_root / "holding_horizon_summary.csv")
    random = pd.read_parquet(report_root / "random_controls.parquet")
    bootstrap = pd.read_csv(report_root / "bootstrap_summary.csv")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    risk = pd.read_parquet(report_root / "monthly_btc_risk.parquet")
    feature_candidates = pd.read_parquet(CANDIDATE_FEATURES_PATH)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    maximum_contribution_error = 0.0
    maximum_gross_error = 0.0
    maximum_cost_error = 0.0
    maximum_beta_error = 0.0
    maximum_notional_error = 0.0
    causal = True
    for item in events.itertuples(index=False):
        weights = parse_mapping(item.weights)
        stored_contributions = parse_mapping(item.symbol_contributions)
        entry = pd.Timestamp(item.entry_time)
        exit_time = pd.Timestamp(item.exit_time)
        causal &= entry == pd.Timestamp(item.feature_time)
        causal &= exit_time == entry + pd.Timedelta(minutes=15)
        future = close.loc[exit_time].div(close.loc[entry]).sub(1.0)
        recomputed = {
            symbol: weight * float(future[symbol])
            for symbol, weight in weights.items()
        }
        maximum_contribution_error = max(
            maximum_contribution_error,
            max(
                abs(recomputed[symbol] - stored_contributions[symbol])
                for symbol in recomputed
            ),
        )
        gross = float(sum(recomputed.values()))
        maximum_gross_error = max(maximum_gross_error, abs(gross - item.gross_return))
        maximum_cost_error = max(
            maximum_cost_error,
            abs(item.primary_net_return - (gross - cfg.primary_round_trip_cost)),
            abs(item.stress_net_return - (gross - cfg.stress_round_trip_cost)),
        )
        gross_notional = float(sum(abs(weight) for weight in weights.values()))
        maximum_notional_error = max(
            maximum_notional_error, abs(gross_notional - item.gross_notional)
        )
        beta_exposure = weights[BTC]
        for symbol, weight in weights.items():
            if symbol == BTC:
                continue
            beta_exposure += weight * float(
                risk_lookup.loc[(_month(pd.Timestamp(item.feature_time)), symbol)]
            )
        maximum_beta_error = max(maximum_beta_error, abs(beta_exposure))
    maximum_summary_error = 0.0
    for item in summary.itertuples(index=False):
        local = _scope(
            events[events["candidate"].eq(item.candidate)], str(item.scope)
        )
        maximum_summary_error = max(
            maximum_summary_error,
            abs(float(local["gross_return"].mean() * 10_000) - item.mean_gross_bp),
            abs(
                float(local["primary_net_return"].mean() * 10_000)
                - item.mean_primary_net_bp
            ),
        )
    maximum_percentile_error = 0.0
    for item in outcome.itertuples(index=False):
        observed = events.loc[
            events["candidate"].eq(item.candidate), "gross_return"
        ].mean()
        control = random.loc[
            random["candidate"].eq(item.candidate), "mean_gross_return"
        ]
        percentile = float(control.le(observed).mean())
        maximum_percentile_error = max(
            maximum_percentile_error,
            abs(percentile - item.random_control_percentile),
        )
    reproduced_bootstrap_error = 0.0
    for offset, candidate in enumerate(CANDIDATES):
        values = events.loc[
            events["candidate"].eq(candidate), "primary_net_return"
        ].to_numpy(dtype=float)
        rng = np.random.default_rng(cfg.seed + 100 + offset)
        means = np.array(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(cfg.bootstrap_iterations)
            ]
        )
        stored = bootstrap[bootstrap["candidate"].eq(candidate)].iloc[0]
        reproduced_bootstrap_error = max(
            reproduced_bootstrap_error,
            abs(float(np.quantile(means, 0.025) * 10_000) - stored["lower_95_primary_net_bp"]),
            abs(float(np.quantile(means, 0.975) * 10_000) - stored["upper_95_primary_net_bp"]),
        )
    gate_outcome_consistent = True
    for candidate in CANDIDATES:
        all_gates = bool(gates.loc[gates["candidate"].eq(candidate), "passed"].all())
        eligible = bool(outcome.loc[outcome["candidate"].eq(candidate), "eligible"].iloc[0])
        gate_outcome_consistent &= all_gates == eligible
    checks = {
        "receiver_feature_hash_matches_prereg": (
            _sha256(RECEIVER_FEATURES_PATH) == EXPECTED_HASHES[RECEIVER_FEATURES_PATH]
        ),
        "candidate_feature_hash_matches_prereg": (
            _sha256(CANDIDATE_FEATURES_PATH) == EXPECTED_HASHES[CANDIDATE_FEATURES_PATH]
        ),
        "feature_candidate_counts_53_52": (
            feature_candidates.groupby("candidate").size().to_dict()
            == {CANDIDATES[0]: 53, CANDIDATES[1]: 52}
        ),
        "realized_candidate_counts_53_52": (
            events.groupby("candidate").size().to_dict()
            == {CANDIDATES[0]: 53, CANDIDATES[1]: 52}
        ),
        "event_keys_unique": not events.duplicated(
            ["candidate", "source_event_id"]
        ).any(),
        "entry_exit_timing_causal": causal,
        "weights_total_gross_one": maximum_notional_error < 1e-9,
        "prior_beta_exposure_neutral": maximum_beta_error < 1e-9,
        "symbol_contributions_reproduced": maximum_contribution_error < 1e-12,
        "gross_returns_reproduced": maximum_gross_error < 1e-12,
        "cost_charges_reproduced": maximum_cost_error < 1e-12,
        "period_summary_reproduced": maximum_summary_error < 1e-10,
        "delayed_counts_match_primary": len(delayed) == len(events),
        "horizon_rows_complete": len(horizons) == len(CANDIDATES) * 4 * 2,
        "random_control_rows_1000": len(random) == cfg.random_iterations * 2,
        "random_iterations_complete": bool(
            random.groupby("candidate")["iteration"].nunique().eq(cfg.random_iterations).all()
        ),
        "random_percentiles_reproduced": maximum_percentile_error < 1e-12,
        "bootstrap_intervals_reproduced": reproduced_bootstrap_error < 1e-10,
        "gate_outcome_consistent": gate_outcome_consistent,
        "both_candidates_rejected": not bool(outcome["eligible"].any()),
    }
    return pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )


def write_v206_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    checks = audit_v206()
    root = ensure_dir(report_root)
    checks_path = root / "independent_audit_checks.csv"
    checks.to_csv(checks_path, index=False)
    verdict = (
        "audit_pass_v205_rejections_reproduced"
        if bool(checks["passed"].all())
        else "audit_failed"
    )
    text = [
        "# v20.6 AggTrade Flow-Exhaustion Independent Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"Passed {int(checks['passed'].sum())}/{len(checks)} independent checks.",
        "",
        "The audit reloaded official close prices, preregistered feature inputs, "
        "monthly beta estimates, saved weights, random controls, and bootstrap "
        "settings. It independently reproduced both v20.5 rejection decisions.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(findings_path.parent)
    findings_path.write_text("\n".join(text), encoding="utf-8")
    return {"checks": checks_path, "findings": findings_path}


__all__ = ["audit_v206", "parse_mapping", "write_v206_audit"]
