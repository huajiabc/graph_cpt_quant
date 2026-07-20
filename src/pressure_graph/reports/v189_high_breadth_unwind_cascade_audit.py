"""Independent audit for the v18.9 high-breadth cascade candidate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC, UNWIND
from pressure_graph.reports.v189_high_breadth_unwind_cascade import (
    CANDIDATE,
    REPORT_ROOT,
    V189Config,
)


AUDIT_ROOT = Path("reports/v18_9_high_breadth_unwind_cascade_audit")
FINDINGS_PATH = Path(
    "docs/v189_high_breadth_unwind_cascade_audit_2026_07_16.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def audit_v189(
    report_root: Path = REPORT_ROOT,
    cfg: V189Config = V189Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "selected_source_signals.parquet")
    base = pd.read_parquet(report_root / "base_q85_unwind_signals.parquet")
    risk = pd.read_parquet(report_root / "monthly_risk_estimates.parquet")
    breadth = pd.read_parquet(report_root / "volatility_breadth.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    complement = pd.read_parquet(report_root / "complement_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    random = pd.read_parquet(report_root / "random_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    close, _ = load_v184_exact_panels()

    for frame in (signals, base):
        for column in ("feature_time", "source_feature_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    risk["risk_month"] = pd.to_datetime(risk["risk_month"], utc=True)
    breadth["feature_time"] = pd.to_datetime(breadth["feature_time"], utc=True)
    breadth = breadth.set_index("feature_time")
    for frame in (events, delayed, complement):
        for column in (
            "feature_time",
            "source_feature_time",
            "entry_time",
            "exit_time",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)

    _check(
        rows,
        "base_q85_unwind_only",
        base["kind"].eq(UNWIND).all()
        and base["return_quantile"].eq(cfg.source_return_quantile).all(),
        len(base),
    )
    returns = close.pct_change(fill_method=None)
    breadth_errors: list[float] = []
    count_errors: list[int] = []
    for event in base.itertuples(index=False):
        timestamp = pd.Timestamp(event.feature_time)
        local = risk[risk["risk_month"].eq(_month(timestamp))].set_index(
            "receiver"
        )
        names = local.index.astype(str).tolist()
        standardized = (
            float(event.source_sign)
            * returns.loc[timestamp, names]
            / local.reindex(names)["return_volatility"]
        )
        valid = int(standardized.notna().sum())
        transmitted = int(standardized.gt(1.0).sum())
        if valid == 0:
            breadth_errors.append(
                0.0 if pd.isna(event.volatility_breadth) else np.inf
            )
            continue
        expected = transmitted / valid
        breadth_errors.append(abs(expected - float(event.volatility_breadth)))
        count_errors.extend(
            [
                abs(valid - int(event.valid_receivers)),
                abs(transmitted - int(event.transmitted_receivers)),
            ]
        )
    _check(
        rows,
        "event_breadth_formula_exact",
        max(breadth_errors) <= 1e-12 and max(count_errors) == 0,
        max(breadth_errors),
    )
    expected_threshold = (
        breadth["volatility_breadth"]
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.breadth_quantile)
    )
    threshold_error = (
        base["breadth_threshold"]
        - base["feature_time"].map(expected_threshold)
    ).abs().max()
    _check(
        rows,
        "breadth_threshold_exact_prior_window",
        threshold_error <= 1e-12,
        threshold_error,
    )
    expected_keys = set(
        base.loc[
            base["volatility_breadth"].ge(base["breadth_threshold"]),
            "source_feature_time",
        ]
    )
    _check(
        rows,
        "high_breadth_selection_exact",
        expected_keys == set(signals["source_feature_time"]),
        len(expected_keys.symmetric_difference(set(signals["source_feature_time"]))),
    )
    _check(
        rows,
        "all_selected_signals_have_events",
        set(events["source_feature_time"]) == expected_keys,
        len(events),
    )
    _check(
        rows,
        "event_holding_exact_15m",
        (events["exit_time"] - events["entry_time"])
        .eq(pd.Timedelta(minutes=15))
        .all(),
        len(events),
    )

    entry = pd.Series(
        [close.at[timestamp, BTC] for timestamp in events["entry_time"]],
        index=events.index,
        dtype=float,
    )
    exit_price = pd.Series(
        [close.at[timestamp, BTC] for timestamp in events["exit_time"]],
        index=events.index,
        dtype=float,
    )
    expected_underlying = exit_price / entry - 1.0
    expected_gross = events["source_sign"] * expected_underlying
    return_error = max(
        (events["btc_underlying_return"] - expected_underlying).abs().max(),
        (events["gross_return"] - expected_gross).abs().max(),
        (
            events["primary_net_return"]
            - (events["gross_return"] - cfg.primary_cost)
        ).abs().max(),
        (
            events["stress_net_return"]
            - (events["gross_return"] - cfg.stress_cost)
        ).abs().max(),
        (
            events["reversed_primary_net_return"]
            - (-events["gross_return"] - cfg.primary_cost)
        ).abs().max(),
    )
    _check(
        rows,
        "direction_return_and_cost_formulas_exact",
        return_error <= 1e-12,
        return_error,
    )
    reported = summary[summary["scope"].eq("all")].iloc[0]
    summary_error = abs(
        float(events["primary_net_return"].mean() * 10_000)
        - float(reported["mean_primary_net_bp"])
    )
    _check(rows, "summary_exact", summary_error <= 1e-10, summary_error)
    _check(
        rows,
        "delay_same_sources",
        set(events["source_feature_time"]) == set(delayed["source_feature_time"]),
        len(
            set(events["source_feature_time"]).symmetric_difference(
                set(delayed["source_feature_time"])
            )
        ),
    )
    complement_keys = set(base["source_feature_time"]) - expected_keys
    _check(
        rows,
        "complement_control_exact",
        set(complement["source_feature_time"]) == complement_keys,
        len(complement),
    )
    _check(
        rows,
        "random_iterations_complete",
        random["iteration"].nunique() == cfg.random_iterations,
        int(random["iteration"].nunique()),
    )
    _check(
        rows,
        "random_controls_preserve_event_count",
        random["events"].eq(len(events)).all(),
        int(random["events"].min()),
    )
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "candidate_ineligible", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_high_breadth_unwind_cascade").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    _check(
        rows,
        "candidate_name_exact",
        events["candidate"].eq(CANDIDATE).all(),
        int(events["candidate"].ne(CANDIDATE).sum()),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v189" in path.name.lower()
        or "high_breadth_unwind" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_high_breadth_cascade_rejected",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.9 High-Breadth Unwind Cascade Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The high-breadth continuation candidate remains rejected. No live or",
        "application scope changed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v189_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v189()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v189", "write_v189_audit"]
