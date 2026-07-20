"""Independent audit for the v18.8 top-trader absorption round."""
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
from pressure_graph.reports.v188_top_trader_absorption import (
    BUCKET_CANDIDATE,
    CANDIDATES,
    DIRECT_CANDIDATE,
    REPORT_ROOT,
    V188Config,
)


AUDIT_ROOT = Path("reports/v18_8_top_trader_absorption_audit")
FINDINGS_PATH = Path("docs/v188_top_trader_absorption_audit_2026_07_16.md")


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def _independent_bucket_scores(
    event: object,
    risk: pd.DataFrame,
    returns: pd.DataFrame,
    flow: pd.DataFrame,
    oi_change: pd.DataFrame,
    top_change: pd.DataFrame,
) -> pd.DataFrame:
    timestamp = pd.Timestamp(event.source_feature_time)
    source_sign = float(event.source_sign)
    local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
    names = local.index.astype(str).tolist()
    frame = pd.DataFrame(index=names)
    frame["btc_beta"] = local["btc_beta"].astype(float)
    frame["aligned_return_z"] = (
        source_sign
        * returns.loc[timestamp, names].astype(float)
        / local["return_volatility"].astype(float)
    )
    frame["aligned_flow"] = source_sign * flow.loc[timestamp, names].astype(float)
    frame["unwind_intensity"] = -oi_change.loc[timestamp, names].astype(float)
    frame["toptrader_absorption"] = (
        -source_sign * top_change.loc[timestamp, names].astype(float)
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    columns = (
        "aligned_return_z",
        "aligned_flow",
        "unwind_intensity",
        "toptrader_absorption",
    )
    frame = frame[frame[list(columns)].gt(0).all(axis=1)].copy()
    ranks = pd.concat([frame[column].rank(pct=True) for column in columns], axis=1)
    frame["score"] = ranks.mean(axis=1)
    return frame


def audit_v188(
    report_root: Path = REPORT_ROOT,
    cfg: V188Config = V188Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signals = pd.read_parquet(report_root / "selected_source_signals.parquet")
    base = pd.read_parquet(report_root / "base_q85_unwind_signals.parquet")
    risk = pd.read_parquet(report_root / "monthly_risk_estimates.parquet")
    events = pd.read_parquet(report_root / "candidate_events.parquet")
    delayed = pd.read_parquet(report_root / "delayed_candidate_events.parquet")
    ranking = pd.read_parquet(report_root / "ranking_control_events.parquet")
    summary = pd.read_csv(report_root / "period_summary.csv")
    random = pd.read_parquet(report_root / "random_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    close, panels = load_v184_exact_panels()

    for frame in (signals, base):
        for column in ("feature_time", "source_feature_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    risk["risk_month"] = pd.to_datetime(risk["risk_month"], utc=True)
    for frame in (events, delayed, ranking):
        for column in (
            "feature_time",
            "source_feature_time",
            "entry_time",
            "exit_time",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        if "risk_month" in frame.columns:
            frame["risk_month"] = pd.to_datetime(frame["risk_month"], utc=True)

    returns = close.pct_change(fill_method=None)
    flow = np.log(
        panels["sum_taker_long_short_vol_ratio"].where(
            panels["sum_taker_long_short_vol_ratio"].gt(0)
        )
    )
    oi_change = np.log(
        panels["sum_open_interest"].where(panels["sum_open_interest"].gt(0))
    ).diff()
    top_position = np.log(
        panels["sum_toptrader_long_short_ratio"].where(
            panels["sum_toptrader_long_short_ratio"].gt(0)
        )
    )
    top_change = top_position.diff()
    btc_absorption = -np.sign(returns[BTC]) * top_change[BTC]
    absorption_threshold = (
        btc_absorption.shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .quantile(cfg.absorption_quantile)
    )

    _check(
        rows,
        "base_q85_unwind_only",
        base["kind"].eq(UNWIND).all()
        and base["return_quantile"].eq(cfg.source_return_quantile).all(),
        len(base),
    )
    threshold_error = (
        base["absorption_threshold"]
        - base["feature_time"].map(absorption_threshold)
    ).abs().max()
    absorption_error = (
        base["btc_toptrader_absorption"]
        - base["feature_time"].map(btc_absorption)
    ).abs().max()
    _check(
        rows,
        "absorption_features_exact_prior_window",
        max(threshold_error, absorption_error) <= 1e-12,
        max(threshold_error, absorption_error),
    )
    expected_keys = set(
        base.loc[
            base["btc_toptrader_absorption"].ge(base["absorption_threshold"]),
            "source_feature_time",
        ]
    )
    _check(
        rows,
        "selected_signal_filter_exact",
        expected_keys == set(signals["source_feature_time"]),
        len(expected_keys.symmetric_difference(set(signals["source_feature_time"]))),
    )
    _check(
        rows,
        "risk_prior_sample_minimum",
        risk["samples"].ge(cfg.risk_min_samples).all(),
        int(risk["samples"].min()),
    )

    direct = events[events["candidate"].eq(DIRECT_CANDIDATE)]
    bucket = events[events["candidate"].eq(BUCKET_CANDIDATE)]
    _check(
        rows,
        "all_selected_signals_have_direct_events",
        set(direct["source_feature_time"]) == set(signals["source_feature_time"]),
        len(direct),
    )
    _check(
        rows,
        "event_holding_exact_30m",
        (events["exit_time"] - events["entry_time"])
        .eq(pd.Timedelta(minutes=30))
        .all(),
        len(events),
    )
    direct_entry = pd.Series(
        [close.at[timestamp, BTC] for timestamp in direct["entry_time"]],
        index=direct.index,
        dtype=float,
    )
    direct_exit = pd.Series(
        [close.at[timestamp, BTC] for timestamp in direct["exit_time"]],
        index=direct.index,
        dtype=float,
    )
    direct_gross = -direct["source_sign"] * (direct_exit / direct_entry - 1.0)
    direct_error = max(
        (direct["gross_return"] - direct_gross).abs().max(),
        (
            direct["primary_net_return"]
            - (direct["gross_return"] - cfg.direct_primary_cost)
        ).abs().max(),
        (
            direct["stress_net_return"]
            - (direct["gross_return"] - cfg.direct_stress_cost)
        ).abs().max(),
    )
    _check(rows, "direct_return_and_cost_exact", direct_error <= 1e-12, direct_error)

    selection_errors: list[int] = []
    bucket_gross_errors: list[float] = []
    for event in bucket.itertuples(index=False):
        scores = _independent_bucket_scores(
            event, risk, returns, flow, oi_change, top_change
        )
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
        hedge = -float((weights * expected.reindex(receivers)["btc_beta"]).sum())
        normalizer = float(weights.abs().sum() + abs(hedge))
        gross = float(
            ((weights * future[receivers]).sum() + hedge * future[BTC])
            / normalizer
        )
        bucket_gross_errors.append(abs(gross - float(event.gross_return)))
    _check(
        rows,
        "bucket_top_selection_exact",
        max(selection_errors) == 0,
        max(selection_errors),
    )
    _check(
        rows,
        "bucket_beta_hedged_gross_exact",
        max(bucket_gross_errors) <= 1e-12,
        max(bucket_gross_errors),
    )
    bucket_cost_error = max(
        (
            bucket["primary_net_return"]
            - (bucket["gross_return"] - cfg.primary_cost)
        ).abs().max(),
        (
            bucket["stress_net_return"]
            - (bucket["gross_return"] - cfg.stress_cost)
        ).abs().max(),
    )
    _check(
        rows,
        "bucket_cost_formula_exact",
        bucket_cost_error <= 1e-12,
        bucket_cost_error,
    )

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
        _check(
            rows,
            f"{candidate}_delay_same_sources",
            source_keys == delayed_keys,
            len(source_keys.symmetric_difference(delayed_keys)),
        )

    direct_control = ranking[ranking["candidate"].eq(DIRECT_CANDIDATE)]
    complement_keys = set(base["source_feature_time"]) - set(
        signals["source_feature_time"]
    )
    _check(
        rows,
        "direct_ranking_control_is_complement",
        set(direct_control["source_feature_time"]) == complement_keys,
        len(direct_control),
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
        outcome["verdict"].eq("reject_top_trader_absorption").all(),
        "|".join(outcome["verdict"].astype(str)),
    )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v188" in path.name.lower() or "top_trader_absorption" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_top_trader_absorption_rejected",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v18.8 Top-Trader Absorption Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "Both top-trader absorption candidates remain rejected. No live or",
        "application scope changed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v188_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v188()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v188", "write_v188_audit"]
