"""Independent arithmetic and chronology audit for the v20.1 reveal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v201_reference_price_transmission import (
    CANDIDATES,
    CANDIDATE_RULES,
    FEATURE_EVENTS_PATH,
    REPORT_ROOT as V201_REPORT_ROOT,
    V201Config,
    select_v201_feature_events,
)


REPORT_ROOT = Path("reports/v20_1_reference_price_transmission_audit")
FINDINGS_PATH = Path(
    "docs/v201_reference_price_transmission_audit_2026_07_17.md"
)


def parse_mapping(value: object) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in str(value).split("|"):
        if not item:
            continue
        key, number = item.rsplit(":", maxsplit=1)
        output[key] = float(number)
    return output


def audit_v201_reveal(
    reveal_root: Path = V201_REPORT_ROOT,
    feature_events_path: Path = FEATURE_EVENTS_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    cfg: V201Config = V201Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(reveal_root / "candidate_events.parquet")
    delayed = pd.read_parquet(reveal_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(reveal_root / "period_summary.csv")
    delayed_summary = pd.read_csv(reveal_root / "delayed_period_summary.csv")
    horizons = pd.read_csv(reveal_root / "holding_horizon_summary.csv")
    gates = pd.read_csv(reveal_root / "candidate_gates.csv")
    outcome = pd.read_csv(reveal_root / "candidate_outcome.csv")
    controls = pd.read_parquet(reveal_root / "random_receiver_controls.parquet")
    risk = pd.read_parquet(reveal_root / "monthly_btc_risk.parquet")
    feature_events = pd.read_parquet(feature_events_path)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)

    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True)
    events["exit_time"] = pd.to_datetime(events["exit_time"], utc=True)
    events["feature_time"] = pd.to_datetime(events["feature_time"], utc=True)
    risk["risk_month"] = pd.to_datetime(risk["risk_month"], utc=True)
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    maximum_errors = {
        "gross_notional": 0.0,
        "beta_residual": 0.0,
        "contribution": 0.0,
        "gross_return": 0.0,
        "primary_cost": 0.0,
        "stress_cost": 0.0,
        "reversed_cost": 0.0,
    }
    receiver_mismatch = 0
    direction_mismatch = 0
    rule_mismatch = 0
    price_missing = 0
    for item in events.itertuples(index=False):
        weights = parse_mapping(item.weights)
        contributions = parse_mapping(item.symbol_contributions)
        receivers = [name for name in str(item.receivers).split("|") if name]
        if set(weights) - {BTC} != set(receivers):
            receiver_mismatch += 1
        expected_direction = (
            float(item.source_sign)
            if "REFERENCE" in str(item.candidate)
            else -float(item.source_sign)
        )
        if any(
            np.sign(weights[symbol]) != np.sign(expected_direction)
            for symbol in receivers
        ):
            direction_mismatch += 1
        source_scope, family, source_setting, receiver_threshold = CANDIDATE_RULES[
            str(item.candidate)
        ]
        if not (
            str(item.source_scope) == source_scope
            and str(item.family) == family
            and str(item.source_setting) == source_setting
            and float(item.receiver_z_threshold) == receiver_threshold
            and int(item.receiver_count) >= 3
        ):
            rule_mismatch += 1
        gross_notional = sum(abs(value) for value in weights.values())
        maximum_errors["gross_notional"] = max(
            maximum_errors["gross_notional"], abs(gross_notional - 1.0)
        )
        month = pd.Timestamp(item.feature_time).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        beta_residual = weights.get(BTC, 0.0) + sum(
            weights[symbol] * risk_lookup.get((month, symbol), np.nan)
            for symbol in receivers
        )
        maximum_errors["beta_residual"] = max(
            maximum_errors["beta_residual"], abs(float(beta_residual))
        )
        prices = close.reindex(
            index=[item.entry_time, item.exit_time], columns=list(weights)
        )
        if prices.isna().any().any():
            price_missing += 1
            continue
        future = prices.loc[item.exit_time].div(prices.loc[item.entry_time]).sub(1.0)
        recomputed = {
            symbol: weights[symbol] * float(future[symbol]) for symbol in weights
        }
        maximum_errors["contribution"] = max(
            maximum_errors["contribution"],
            max(
                abs(recomputed[symbol] - contributions[symbol])
                for symbol in recomputed
            ),
        )
        gross = sum(recomputed.values())
        maximum_errors["gross_return"] = max(
            maximum_errors["gross_return"], abs(gross - float(item.gross_return))
        )
        maximum_errors["primary_cost"] = max(
            maximum_errors["primary_cost"],
            abs(
                float(item.primary_net_return)
                - (gross - cfg.primary_round_trip_cost)
            ),
        )
        maximum_errors["stress_cost"] = max(
            maximum_errors["stress_cost"],
            abs(
                float(item.stress_net_return)
                - (gross - cfg.stress_round_trip_cost)
            ),
        )
        maximum_errors["reversed_cost"] = max(
            maximum_errors["reversed_cost"],
            abs(
                float(item.reversed_primary_net_return)
                - (-gross - cfg.primary_round_trip_cost)
            ),
        )

    expected_feature_counts = {
        candidate: len(select_v201_feature_events(feature_events, candidate))
        for candidate in CANDIDATES
    }
    observed_counts = events["candidate"].value_counts().to_dict()
    summary_all = summary[summary["scope"].eq("all")].set_index("candidate")
    delayed_all = delayed_summary[
        delayed_summary["scope"].eq("all")
    ].set_index("candidate")
    outcome_indexed = outcome.set_index("candidate")
    summary_errors = []
    delayed_errors = []
    outcome_errors = []
    for candidate in CANDIDATES:
        local = events[events["candidate"].eq(candidate)]
        local_delayed = delayed[delayed["candidate"].eq(candidate)]
        summary_errors.extend(
            [
                abs(float(summary_all.at[candidate, "events"]) - len(local)),
                abs(
                    float(summary_all.at[candidate, "mean_gross_bp"])
                    - float(local["gross_return"].mean() * 10_000)
                ),
                abs(
                    float(summary_all.at[candidate, "mean_primary_net_bp"])
                    - float(local["primary_net_return"].mean() * 10_000)
                ),
            ]
        )
        delayed_errors.extend(
            [
                abs(float(delayed_all.at[candidate, "events"]) - len(local_delayed)),
                abs(
                    float(delayed_all.at[candidate, "mean_primary_net_bp"])
                    - float(local_delayed["primary_net_return"].mean() * 10_000)
                ),
            ]
        )
        outcome_errors.extend(
            [
                abs(float(outcome_indexed.at[candidate, "events"]) - len(local)),
                abs(
                    float(outcome_indexed.at[candidate, "mean_gross_bp"])
                    - float(local["gross_return"].mean() * 10_000)
                ),
                abs(
                    float(outcome_indexed.at[candidate, "mean_primary_net_bp"])
                    - float(local["primary_net_return"].mean() * 10_000)
                ),
            ]
        )

    feature_count_values = "|".join(
        f"{candidate}:{expected_feature_counts[candidate]}" for candidate in CANDIDATES
    )
    observed_count_values = "|".join(
        f"{candidate}:{observed_counts.get(candidate, 0)}" for candidate in CANDIDATES
    )
    checks: dict[str, tuple[bool, object]] = {
        "candidate_set_exact": (
            set(events["candidate"]) == set(CANDIDATES),
            events["candidate"].nunique(),
        ),
        "feature_counts_match_frozen_rules": (
            list(expected_feature_counts.values()) == [986, 619, 227, 360],
            feature_count_values,
        ),
        "realized_counts_expected_after_risk": (
            list(observed_counts.get(candidate, 0) for candidate in CANDIDATES)
            == [951, 597, 227, 360],
            observed_count_values,
        ),
        "candidate_rules_exact": (rule_mismatch == 0, rule_mismatch),
        "receivers_match_weights": (receiver_mismatch == 0, receiver_mismatch),
        "directions_match_prereg": (direction_mismatch == 0, direction_mismatch),
        "feature_entry_exit_chronology": (
            events["entry_time"].eq(events["feature_time"]).all()
            and events["exit_time"].eq(
                events["entry_time"] + pd.Timedelta(minutes=15)
            ).all(),
            int(
                (~events["entry_time"].eq(events["feature_time"])).sum()
                + (
                    ~events["exit_time"].eq(
                        events["entry_time"] + pd.Timedelta(minutes=15)
                    )
                ).sum()
            ),
        ),
        "no_missing_repricing": (price_missing == 0, price_missing),
        "gross_notional_exact": (
            maximum_errors["gross_notional"] <= 2e-11,
            maximum_errors["gross_notional"],
        ),
        "beta_neutral_exact": (
            maximum_errors["beta_residual"] <= 2e-11,
            maximum_errors["beta_residual"],
        ),
        "contributions_reprice": (
            maximum_errors["contribution"] <= 2e-11,
            maximum_errors["contribution"],
        ),
        "gross_return_reprices": (
            maximum_errors["gross_return"] <= 2e-11,
            maximum_errors["gross_return"],
        ),
        "primary_cost_exact": (
            maximum_errors["primary_cost"] <= 2e-11,
            maximum_errors["primary_cost"],
        ),
        "stress_cost_exact": (
            maximum_errors["stress_cost"] <= 2e-11,
            maximum_errors["stress_cost"],
        ),
        "reversed_cost_exact": (
            maximum_errors["reversed_cost"] <= 2e-11,
            maximum_errors["reversed_cost"],
        ),
        "summary_recomputes": (max(summary_errors) <= 1e-10, max(summary_errors)),
        "delayed_summary_recomputes": (
            max(delayed_errors) <= 1e-10,
            max(delayed_errors),
        ),
        "outcome_recomputes": (max(outcome_errors) <= 1e-10, max(outcome_errors)),
        "horizon_grid_exact": (
            set(horizons["holding_bars"]) == {2, 4}
            and len(horizons) == len(CANDIDATES) * 4 * 2,
            len(horizons),
        ),
        "random_family_complete": (
            len(controls) == cfg.random_iterations * (len(CANDIDATES) + 1)
            and controls["iteration"].nunique() == cfg.random_iterations
            and controls["candidate"].eq("FAMILY_MAX").sum()
            == cfg.random_iterations,
            len(controls),
        ),
        "all_candidates_rejected": (
            ~outcome["eligible"].astype(bool).any()
            and outcome["verdict"].eq(
                "reject_reference_price_transmission"
            ).all(),
            int(outcome["eligible"].astype(bool).sum()),
        ),
        "gate_status_matches_outcome": (
            all(
                not bool(gates[gates["candidate"].eq(candidate)]["passed"].all())
                and not bool(outcome_indexed.at[candidate, "eligible"])
                for candidate in CANDIDATES
            ),
            int(gates["passed"].sum()),
        ),
    }
    audit = pd.DataFrame(
        [
            {"check": name, "passed": bool(passed), "value": value}
            for name, (passed, value) in checks.items()
        ]
    )
    audit["verdict"] = (
        "audit_pass_reference_transmission_rejection_reproduced"
        if audit["passed"].all()
        else "audit_failure_reference_transmission_requires_investigation"
    )
    details = pd.DataFrame(
        [
            {"metric": name, "maximum_absolute_error": value}
            for name, value in maximum_errors.items()
        ]
    )
    return audit, details


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v20.1 Reference-Price Transmission Independent Audit",
        "",
        f"Verdict: `{audit['verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The audit independently reparsed serialized weights, repriced every event "
        "from source futures closes, rechecked frozen selection rules, BTC-beta "
        "neutrality, costs, summaries, controls, and the rejection state.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v201_independent_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, details = audit_v201_reveal()
    root = ensure_dir(report_root)
    outputs = {
        "audit": root / "audit_checks.csv",
        "errors": root / "repricing_errors.csv",
        "findings": findings_path,
    }
    audit.to_csv(outputs["audit"], index=False)
    details.to_csv(outputs["errors"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = [
    "audit_v201_reveal",
    "parse_mapping",
    "write_v201_independent_audit",
]
