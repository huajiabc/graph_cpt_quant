"""Independent audit for the v18.7 unwind volatility-transfer buckets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import (
    BTC,
    UNWIND,
    build_v185_features,
)
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    CANDIDATES,
    OVERSHOOT_CANDIDATE,
    REPORT_ROOT,
    V187Config,
)


AUDIT_ROOT = Path("reports/v18_7_unwind_volatility_transfer_bucket_audit")
FINDINGS_PATH = Path(
    "docs/v187_unwind_volatility_transfer_bucket_audit_2026_07_16.md"
)
V185_REPORT_ROOT = Path("reports/v18_5_btc_leverage_flow_graph")


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def _independent_scores(
    event: object,
    risk: pd.DataFrame,
    returns: pd.DataFrame,
    flow: pd.DataFrame,
    oi_change: pd.DataFrame,
) -> pd.DataFrame:
    timestamp = pd.Timestamp(event.source_feature_time)
    source_sign = float(event.source_sign)
    local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
    names = local.index.astype(str).tolist()
    beta = local["btc_beta"].astype(float)
    residual_vol = local["residual_volatility"].astype(float)
    return_vol = local["return_volatility"].astype(float)
    alt_return = returns.loc[timestamp, names].astype(float)
    residual_z = (alt_return - beta * float(returns.at[timestamp, BTC])) / residual_vol
    frame = pd.DataFrame(
        {
            "btc_beta": beta,
            "aligned_residual_z": source_sign * residual_z,
        }
    )
    if str(event.candidate) == OVERSHOOT_CANDIDATE:
        frame["score"] = frame["aligned_residual_z"]
        return frame.replace([np.inf, -np.inf], np.nan).dropna().query("score > 0")
    frame["aligned_return_z"] = source_sign * alt_return / return_vol
    frame["aligned_flow"] = source_sign * flow.loc[timestamp, names].astype(float)
    frame["unwind_intensity"] = -oi_change.loc[timestamp, names].astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[
        frame["aligned_return_z"].gt(0)
        & frame["aligned_flow"].gt(0)
        & frame["unwind_intensity"].gt(0)
    ].copy()
    frame["score"] = pd.concat(
        [
            frame["aligned_return_z"].rank(pct=True),
            frame["aligned_flow"].rank(pct=True),
            frame["unwind_intensity"].rank(pct=True),
        ],
        axis=1,
    ).mean(axis=1)
    return frame


def audit_v187(
    report_root: Path = REPORT_ROOT,
    cfg: V187Config = V187Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "unwind_source_signals.parquet")
    prior_signals = pd.read_parquet(V185_REPORT_ROOT / "source_signals.parquet")
    risk = pd.read_parquet(report_root / "monthly_risk_estimates.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    bottom = pd.read_parquet(report_root / "bottom_candidate_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    random = pd.read_parquet(report_root / "random_receiver_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    close, panels = load_v184_exact_panels()
    returns, flow, oi_change, _ = build_v185_features(close, panels)

    for frame in (signals, prior_signals):
        for column in ("feature_time", "source_feature_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    risk["risk_month"] = pd.to_datetime(risk["risk_month"], utc=True)
    for frame in (events, delayed, bottom):
        for column in (
            "feature_time",
            "source_feature_time",
            "entry_time",
            "exit_time",
            "risk_month",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)

    expected_signals = prior_signals[prior_signals["kind"].eq(UNWIND)].reset_index(
        drop=True
    )
    shared = sorted(set(signals.columns) & set(expected_signals.columns))
    _check(
        rows,
        "source_signals_exact_v185_unwind_freeze",
        signals[shared].reset_index(drop=True).equals(
            expected_signals[shared].reset_index(drop=True)
        ),
        len(signals),
    )
    _check(
        rows,
        "risk_month_normalized",
        risk["risk_month"].dt.day.eq(1).all()
        and risk["risk_month"].dt.hour.eq(0).all()
        and risk["risk_month"].dt.minute.eq(0).all(),
        len(risk),
    )
    _check(
        rows,
        "risk_minimum_samples",
        risk["samples"].ge(cfg.risk_min_samples).all(),
        int(risk["samples"].min()),
    )
    risk_errors: list[float] = []
    sample_errors: list[int] = []
    for item in risk.itertuples(index=False):
        history = returns.loc[
            (
                returns.index
                >= item.risk_month - pd.Timedelta(days=cfg.risk_lookback_days)
            )
            & (returns.index < item.risk_month),
            [BTC, item.receiver],
        ].dropna()
        beta = float(
            history[item.receiver].cov(history[BTC]) / history[BTC].var(ddof=1)
        )
        residual = history[item.receiver] - beta * history[BTC]
        risk_errors.extend(
            [
                abs(beta - float(item.btc_beta)),
                abs(residual.std(ddof=1) - float(item.residual_volatility)),
                abs(
                    history[item.receiver].std(ddof=1)
                    - float(item.return_volatility)
                ),
            ]
        )
        sample_errors.append(abs(len(history) - int(item.samples)))
    _check(
        rows,
        "risk_prior_window_formulas_exact",
        max(risk_errors) <= 1e-12 and max(sample_errors) == 0,
        max(risk_errors),
    )

    _check(
        rows,
        "events_unwind_only",
        events["kind"].eq(UNWIND).all(),
        int(events["kind"].ne(UNWIND).sum()),
    )
    _check(
        rows,
        "event_holding_exact_30m",
        (events["exit_time"] - events["entry_time"])
        .eq(pd.Timedelta(minutes=30))
        .all(),
        len(events),
    )
    _check(
        rows,
        "entry_is_completed_source_close",
        events["entry_time"].eq(events["source_feature_time"]).all(),
        int(events["entry_time"].ne(events["source_feature_time"]).sum()),
    )
    _check(
        rows,
        "receiver_count_bounds",
        events["receiver_count"].between(
            cfg.min_receiver_bucket, cfg.receiver_bucket_size
        ).all(),
        int(events["receiver_count"].min()),
    )

    selection_errors: list[int] = []
    gross_errors: list[float] = []
    for event in events.itertuples(index=False):
        scores = _independent_scores(event, risk, returns, flow, oi_change)
        expected = scores.sort_values("score", ascending=False).head(
            cfg.receiver_bucket_size
        )
        receivers = str(event.receivers).split("|")
        selection_errors.append(len(set(receivers).symmetric_difference(expected.index)))
        future = (
            close.loc[event.exit_time, [BTC, *receivers]]
            / close.loc[event.entry_time, [BTC, *receivers]]
            - 1.0
        )
        weights = pd.Series(
            -float(event.source_sign) / len(receivers), index=receivers
        )
        beta = expected.reindex(receivers)["btc_beta"]
        hedge = -float((weights * beta).sum())
        normalizer = float(weights.abs().sum() + abs(hedge))
        gross = float(
            ((weights * future[receivers]).sum() + hedge * future[BTC])
            / normalizer
        )
        gross_errors.append(abs(gross - float(event.gross_return)))
    _check(
        rows,
        "top_receiver_selection_exact",
        max(selection_errors) == 0,
        max(selection_errors),
    )
    _check(
        rows,
        "beta_hedged_gross_formula_exact",
        max(gross_errors) <= 1e-12,
        max(gross_errors),
    )
    cost_error = max(
        (
            events["primary_net_return"]
            - (events["gross_return"] - cfg.primary_cost)
        ).abs().max(),
        (
            events["stress_net_return"]
            - (events["gross_return"] - cfg.stress_cost)
        ).abs().max(),
    )
    _check(rows, "event_cost_formulas_exact", cost_error <= 1e-12, cost_error)

    for candidate, sample in events.groupby("candidate", sort=True):
        reported = summary[
            summary["candidate"].eq(candidate) & summary["scope"].eq("all")
        ].iloc[0]
        actual = float(sample["primary_net_return"].mean() * 10_000)
        error = abs(actual - float(reported["mean_primary_net_bp"]))
        _check(rows, f"{candidate}_summary_exact", error <= 1e-10, error)
        source_keys = set(sample["source_feature_time"])
        delayed_keys = set(
            delayed.loc[
                delayed["candidate"].eq(candidate), "source_feature_time"
            ]
        )
        bottom_keys = set(
            bottom.loc[bottom["candidate"].eq(candidate), "source_feature_time"]
        )
        _check(
            rows,
            f"{candidate}_controls_same_sources",
            source_keys == delayed_keys == bottom_keys,
            len(source_keys.symmetric_difference(delayed_keys | bottom_keys)),
        )

    _check(
        rows,
        "random_iterations_complete",
        random["iteration"].nunique() == cfg.random_iterations,
        int(random["iteration"].nunique()),
    )
    random_wide = random.pivot(
        index="iteration", columns="candidate", values="mean_primary_net_return"
    )
    family_error = (
        random_wide["FAMILY_MAX"] - random_wide[list(CANDIDATES)].max(axis=1)
    ).abs().max()
    _check(rows, "random_family_max_exact", family_error <= 1e-12, family_error)
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(
        rows,
        "rejection_verdict_exact",
        outcome["verdict"].eq("reject_unwind_volatility_transfer_bucket").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v187" in path.name.lower()
        or "volatility_transfer_bucket" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_volatility_transfer_buckets_rejected",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.7 Unwind Volatility-Transfer Bucket Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "Both event-time receiver rankings are rejected. No live or application",
        "scope changed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v187_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v187()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v187", "write_v187_audit"]
