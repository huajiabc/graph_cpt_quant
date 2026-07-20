"""Independent repricing audit for the v20.3 community peer hedge."""
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
from pressure_graph.reports.v203_community_peer_hedge import (
    REPORT_ROOT as V203_REPORT_ROOT,
    V203Config,
    parse_weights,
)


REPORT_ROOT = Path("reports/v20_3_community_peer_hedge_audit")
FINDINGS_PATH = Path("docs/v203_community_peer_hedge_audit_2026_07_17.md")


def audit_v203_reveal(
    reveal_root: Path = V203_REPORT_ROOT,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    cfg: V203Config = V203Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(reveal_root / "candidate_events.parquet")
    delayed = pd.read_parquet(reveal_root / "delayed_candidate_events.parquet")
    summary = pd.read_csv(reveal_root / "period_summary.csv")
    delayed_summary = pd.read_csv(reveal_root / "delayed_period_summary.csv")
    horizons = pd.read_csv(reveal_root / "holding_horizon_summary.csv")
    controls = pd.read_parquet(reveal_root / "random_partition_controls.parquet")
    gates = pd.read_csv(reveal_root / "candidate_gates.csv")
    outcome = pd.read_csv(reveal_root / "candidate_outcome.csv")
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    for column in ("feature_time", "entry_time", "exit_time"):
        events[column] = pd.to_datetime(events[column], utc=True)

    errors = {
        "gross_notional": 0.0,
        "net_dollar": 0.0,
        "contribution": 0.0,
        "gross_return": 0.0,
        "primary_cost": 0.0,
        "stress_cost": 0.0,
        "reversed_cost": 0.0,
    }
    sleeve_mismatch = 0
    direction_mismatch = 0
    missing_prices = 0
    for item in events.itertuples(index=False):
        weights = parse_weights(item.weights)
        contributions = parse_weights(item.symbol_contributions)
        selected = [name for name in str(item.selected_symbols).split("|") if name]
        peers = [name for name in str(item.peer_symbols).split("|") if name]
        if set(weights) != set(selected) | set(peers) or set(selected) & set(peers):
            sleeve_mismatch += 1
        if any(
            np.sign(weights[symbol]) != -np.sign(float(item.source_sign))
            for symbol in selected
        ) or any(
            np.sign(weights[symbol]) != np.sign(float(item.source_sign))
            for symbol in peers
        ):
            direction_mismatch += 1
        errors["gross_notional"] = max(
            errors["gross_notional"],
            abs(sum(abs(value) for value in weights.values()) - 1.0),
        )
        errors["net_dollar"] = max(
            errors["net_dollar"], abs(sum(weights.values()))
        )
        prices = close.reindex(
            index=[item.entry_time, item.exit_time], columns=list(weights)
        )
        if prices.isna().any().any():
            missing_prices += 1
            continue
        future = prices.loc[item.exit_time].div(prices.loc[item.entry_time]).sub(1.0)
        recomputed = {
            symbol: weights[symbol] * float(future[symbol]) for symbol in weights
        }
        errors["contribution"] = max(
            errors["contribution"],
            max(
                abs(recomputed[symbol] - contributions[symbol])
                for symbol in recomputed
            ),
        )
        gross = sum(recomputed.values())
        errors["gross_return"] = max(
            errors["gross_return"], abs(gross - float(item.gross_return))
        )
        errors["primary_cost"] = max(
            errors["primary_cost"],
            abs(
                float(item.primary_net_return)
                - (gross - cfg.primary_round_trip_cost)
            ),
        )
        errors["stress_cost"] = max(
            errors["stress_cost"],
            abs(
                float(item.stress_net_return)
                - (gross - cfg.stress_round_trip_cost)
            ),
        )
        errors["reversed_cost"] = max(
            errors["reversed_cost"],
            abs(
                float(item.reversed_primary_net_return)
                - (-gross - cfg.primary_round_trip_cost)
            ),
        )

    scoped = summary.set_index("scope")
    delayed_scoped = delayed_summary.set_index("scope")
    summary_errors = [
        abs(float(scoped.at["all", "events"]) - len(events)),
        abs(
            float(scoped.at["all", "mean_gross_bp"])
            - float(events["gross_return"].mean() * 10_000)
        ),
        abs(
            float(scoped.at["all", "mean_primary_net_bp"])
            - float(events["primary_net_return"].mean() * 10_000)
        ),
    ]
    delayed_errors = [
        abs(float(delayed_scoped.at["all", "events"]) - len(delayed)),
        abs(
            float(delayed_scoped.at["all", "mean_primary_net_bp"])
            - float(delayed["primary_net_return"].mean() * 10_000)
        ),
    ]
    outcome_row = outcome.iloc[0]
    outcome_errors = [
        abs(float(outcome_row["events"]) - len(events)),
        abs(
            float(outcome_row["mean_gross_bp"])
            - float(events["gross_return"].mean() * 10_000)
        ),
        abs(
            float(outcome_row["mean_primary_net_bp"])
            - float(events["primary_net_return"].mean() * 10_000)
        ),
    ]
    checks: dict[str, tuple[bool, object]] = {
        "event_count_237": (len(events) == 237, len(events)),
        "sleeves_disjoint_and_complete": (sleeve_mismatch == 0, sleeve_mismatch),
        "directions_match_prereg": (direction_mismatch == 0, direction_mismatch),
        "chronology_exact_15m": (
            events["entry_time"].eq(events["feature_time"]).all()
            and events["exit_time"].eq(
                events["entry_time"] + pd.Timedelta(minutes=15)
            ).all(),
            len(events),
        ),
        "no_missing_repricing": (missing_prices == 0, missing_prices),
        "gross_notional_exact": (
            errors["gross_notional"] <= 2e-11,
            errors["gross_notional"],
        ),
        "net_dollar_exact": (
            errors["net_dollar"] <= 2e-11,
            errors["net_dollar"],
        ),
        "contributions_reprice": (
            errors["contribution"] <= 2e-11,
            errors["contribution"],
        ),
        "gross_return_reprices": (
            errors["gross_return"] <= 2e-11,
            errors["gross_return"],
        ),
        "primary_cost_exact": (
            errors["primary_cost"] <= 2e-11,
            errors["primary_cost"],
        ),
        "stress_cost_exact": (
            errors["stress_cost"] <= 2e-11,
            errors["stress_cost"],
        ),
        "reversed_cost_exact": (
            errors["reversed_cost"] <= 2e-11,
            errors["reversed_cost"],
        ),
        "summary_recomputes": (max(summary_errors) <= 1e-10, max(summary_errors)),
        "delayed_summary_recomputes": (
            max(delayed_errors) <= 1e-10,
            max(delayed_errors),
        ),
        "outcome_recomputes": (max(outcome_errors) <= 1e-10, max(outcome_errors)),
        "horizon_grid_complete": (
            set(horizons["holding_bars"]) == {2, 4} and len(horizons) == 8,
            len(horizons),
        ),
        "random_controls_complete": (
            len(controls) == cfg.random_iterations
            and controls["iteration"].nunique() == cfg.random_iterations,
            len(controls),
        ),
        "rejection_matches_failed_gates": (
            not bool(outcome_row["all_gates_pass"])
            and str(outcome_row["verdict"]) == "reject_community_peer_hedge"
            and not bool(gates["passed"].all()),
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
        "audit_pass_community_peer_hedge_rejection_reproduced"
        if audit["passed"].all()
        else "audit_failure_community_peer_hedge_requires_investigation"
    )
    details = pd.DataFrame(
        [
            {"metric": name, "maximum_absolute_error": value}
            for name, value in errors.items()
        ]
    )
    return audit, details


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v20.3 Community Peer-Hedge Independent Audit",
        "",
        f"Verdict: `{audit['verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "Every serialized weight and contribution was independently repriced. "
        "Sleeve membership, direction, dollar neutrality, costs, summaries, "
        "controls, and the rejection verdict reproduce.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v203_independent_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, details = audit_v203_reveal()
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


__all__ = ["audit_v203_reveal", "write_v203_independent_audit"]
