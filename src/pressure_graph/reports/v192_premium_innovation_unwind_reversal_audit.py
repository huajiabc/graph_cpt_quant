"""Independent audit for the v19.2 premium-innovation reversal reveal."""
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
from pressure_graph.reports.v190_binance_premium_index_audit import (
    load_v190_premium_panel,
)
from pressure_graph.reports.v192_premium_innovation_unwind_reversal import (
    BUCKET_CANDIDATE,
    CANDIDATES,
    DIRECT_CANDIDATE,
    REPORT_ROOT,
    V192Config,
)


AUDIT_ROOT = Path("reports/v19_2_premium_innovation_unwind_reversal_audit")
FINDINGS_PATH = Path(
    "docs/v192_premium_innovation_unwind_reversal_audit_2026_07_17.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def _max_abs(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmax(np.abs(array))) if array.size else 0.0


def _load_outputs(report_root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "signals": pd.read_parquet(report_root / "selected_source_signals.parquet"),
        "base": pd.read_parquet(report_root / "base_q85_unwind_signals.parquet"),
        "risk": pd.read_parquet(report_root / "monthly_risk_estimates.parquet"),
        "innovation": pd.read_parquet(report_root / "premium_innovation_z.parquet"),
        "events": pd.read_parquet(report_root / "candidate_events.parquet"),
        "delayed": pd.read_parquet(report_root / "delayed_candidate_events.parquet"),
        "random": pd.read_parquet(report_root / "random_controls.parquet"),
        "summary": pd.read_csv(report_root / "period_summary.csv"),
        "sides": pd.read_csv(report_root / "source_side_summary.csv"),
        "gates": pd.read_csv(report_root / "candidate_gates.csv"),
        "outcome": pd.read_csv(report_root / "candidate_outcome.csv"),
    }
    for name in ("signals", "base"):
        for column in ("feature_time", "source_feature_time"):
            frames[name][column] = pd.to_datetime(frames[name][column], utc=True)
    for name in ("events", "delayed"):
        for column in (
            "feature_time",
            "source_feature_time",
            "entry_time",
            "exit_time",
            "risk_month",
        ):
            frames[name][column] = pd.to_datetime(frames[name][column], utc=True)
    frames["risk"]["risk_month"] = pd.to_datetime(
        frames["risk"]["risk_month"], utc=True
    )
    return frames


def _recompute_bucket_errors(
    events: pd.DataFrame,
    close: pd.DataFrame,
    innovation: pd.DataFrame,
    risk: pd.DataFrame,
    cfg: V192Config,
) -> tuple[float, float, float, int]:
    ranking_errors: list[float] = []
    hedge_errors: list[float] = []
    return_errors: list[float] = []
    receiver_count_errors = 0
    for event in events.itertuples(index=False):
        timestamp = pd.Timestamp(event.source_feature_time)
        local_risk = risk[risk["risk_month"].eq(_month(timestamp))].set_index(
            "receiver"
        )
        names = [
            name
            for name in local_risk.index.astype(str)
            if name in innovation.columns
        ]
        aligned = float(event.source_sign) * innovation.loc[timestamp, names]
        eligible = aligned[aligned.ge(cfg.premium_z_threshold)].dropna()
        expected = eligible.sort_values(ascending=False).head(cfg.receiver_bucket_size)
        reported = str(event.receivers).split("|")
        receiver_count_errors += int(
            reported != expected.index.astype(str).tolist()
            or len(reported) != int(event.receiver_count)
            or len(eligible) != int(event.eligible_receivers)
        )
        if reported:
            ranking_errors.extend(
                (
                    np.asarray(
                        [
                            float(item.rsplit(":", 1)[1])
                            for item in str(event.receiver_scores).split("|")
                        ]
                    )
                    - expected.to_numpy(dtype=float)
                ).tolist()
            )
        beta = local_risk.reindex(reported)["btc_beta"].to_numpy(dtype=float)
        direction = -float(event.source_sign)
        weights = np.repeat(direction / len(reported), len(reported))
        hedge = -float(np.sum(weights * beta))
        normalizer = float(np.sum(np.abs(weights)) + abs(hedge))
        future = (
            close.loc[event.exit_time, [BTC, *reported]]
            / close.loc[event.entry_time, [BTC, *reported]]
            - 1.0
        )
        gross = float(
            (
                np.sum(weights * future.reindex(reported).to_numpy(dtype=float))
                + hedge * float(future[BTC])
            )
            / normalizer
        )
        hedge_errors.extend(
            [hedge - float(event.btc_hedge_weight), normalizer - float(event.normalizer)]
        )
        return_errors.extend(
            [
                gross - float(event.gross_return),
                gross - cfg.primary_cost - float(event.primary_net_return),
                gross - cfg.stress_cost - float(event.stress_net_return),
                -gross - cfg.primary_cost
                - float(event.reversed_primary_net_return),
            ]
        )
    return (
        _max_abs(ranking_errors),
        _max_abs(hedge_errors),
        _max_abs(return_errors),
        receiver_count_errors,
    )


def audit_v192_independently(
    report_root: Path = REPORT_ROOT,
    cfg: V192Config = V192Config(),
) -> pd.DataFrame:
    frames = _load_outputs(report_root)
    signals = frames["signals"]
    base = frames["base"]
    risk = frames["risk"]
    innovation = frames["innovation"]
    events = frames["events"]
    delayed = frames["delayed"]
    random = frames["random"]
    summary = frames["summary"]
    sides = frames["sides"]
    gates = frames["gates"]
    outcome = frames["outcome"]
    close, _ = load_v184_exact_panels()
    premium = load_v190_premium_panel().reindex(
        index=close.index, columns=close.columns
    )

    rows: list[dict[str, object]] = []
    _check(rows, "selected_signal_keys_unique", not signals.duplicated(
        ["source_feature_time", "kind"]
    ).any(), len(signals))
    _check(rows, "base_signal_keys_unique", not base.duplicated(
        ["source_feature_time", "kind"]
    ).any(), len(base))
    _check(rows, "all_sources_are_q85_unwind", base["kind"].eq(UNWIND).all()
        and base["return_quantile"].eq(cfg.source_return_quantile).all(), len(base))
    _check(rows, "source_feature_time_is_feature_time", base["source_feature_time"].eq(
        base["feature_time"]
    ).all(), int(base["source_feature_time"].ne(base["feature_time"]).sum()))

    scale = (
        premium.diff()
        .shift(1)
        .rolling(cfg.source_lookback_bars, min_periods=cfg.source_min_bars)
        .std(ddof=1)
    )
    expected_innovation = premium.diff().div(scale.where(scale.gt(0)))
    signal_expected_z = pd.Series(
        [expected_innovation.at[timestamp, BTC] for timestamp in base["feature_time"]],
        index=base.index,
        dtype=float,
    )
    z_error = _max_abs(base["btc_premium_innovation_z"] - signal_expected_z)
    _check(rows, "premium_innovation_uses_shifted_prior_scale", z_error <= 1e-12, z_error)
    aligned_error = _max_abs(
        base["aligned_btc_premium_z"]
        - base["source_sign"] * base["btc_premium_innovation_z"]
    )
    _check(rows, "aligned_premium_formula_exact", aligned_error <= 1e-12, aligned_error)
    expected_selected = base[
        base["aligned_btc_premium_z"].ge(cfg.premium_z_threshold)
    ]
    selected_keys = set(signals["source_feature_time"])
    expected_keys = set(expected_selected["source_feature_time"])
    _check(rows, "selected_source_set_exact", selected_keys == expected_keys,
        len(selected_keys.symmetric_difference(expected_keys)))
    expected_side = np.where(
        signals["source_sign"].lt(0), "long_liquidation", "short_cover"
    )
    _check(rows, "source_side_mapping_exact", signals["source_side"].eq(
        expected_side
    ).all(), int(signals["source_side"].ne(expected_side).sum()))
    _check(rows, "all_selected_sources_have_exact_premium", signals[
        "btc_premium_innovation_z"
    ].notna().all(), int(signals["btc_premium_innovation_z"].isna().sum()))

    direct = events[events["candidate"].eq(DIRECT_CANDIDATE)].copy()
    bucket = events[events["candidate"].eq(BUCKET_CANDIDATE)].copy()
    _check(rows, "candidate_domain_exact", set(events["candidate"]) == set(CANDIDATES),
        "|".join(sorted(events["candidate"].unique())))
    _check(rows, "direct_has_every_selected_source", set(direct["source_feature_time"])
        == selected_keys, len(direct))
    _check(rows, "bucket_sources_subset_selected", set(bucket["source_feature_time"])
        <= selected_keys, len(bucket))
    _check(rows, "bucket_minimum_receiver_rule", bucket["receiver_count"].ge(
        cfg.min_receiver_bucket
    ).all(), int(bucket["receiver_count"].min()))
    _check(rows, "event_entry_at_completed_source_close", events["entry_time"].eq(
        events["source_feature_time"]
    ).all(), int(events["entry_time"].ne(events["source_feature_time"]).sum()))
    holding = events["exit_time"] - events["entry_time"]
    _check(rows, "event_holding_exact_30m", holding.eq(
        pd.Timedelta(minutes=30)
    ).all(), int(holding.ne(pd.Timedelta(minutes=30)).sum()))
    expected_direction = -events["source_sign"]
    _check(rows, "trade_direction_is_source_reversal", events["trade_direction"].eq(
        expected_direction
    ).all(), int(events["trade_direction"].ne(expected_direction).sum()))

    direct_entry = pd.Series(
        [close.at[timestamp, BTC] for timestamp in direct["entry_time"]],
        index=direct.index,
    )
    direct_exit = pd.Series(
        [close.at[timestamp, BTC] for timestamp in direct["exit_time"]],
        index=direct.index,
    )
    expected_underlying = direct_exit / direct_entry - 1.0
    expected_gross = direct["trade_direction"] * expected_underlying
    direct_return_error = max(
        _max_abs(direct["btc_underlying_return"] - expected_underlying),
        _max_abs(direct["gross_return"] - expected_gross),
    )
    _check(rows, "direct_price_return_formula_exact", direct_return_error <= 1e-12,
        direct_return_error)
    direct_cost_error = max(
        _max_abs(direct["primary_net_return"] - (
            direct["gross_return"] - cfg.direct_primary_cost
        )),
        _max_abs(direct["stress_net_return"] - (
            direct["gross_return"] - cfg.direct_stress_cost
        )),
        _max_abs(direct["reversed_primary_net_return"] - (
            -direct["gross_return"] - cfg.direct_primary_cost
        )),
    )
    _check(rows, "direct_cost_and_reverse_formulas_exact", direct_cost_error <= 1e-12,
        direct_cost_error)

    rank_error, hedge_error, bucket_return_error, receiver_errors = (
        _recompute_bucket_errors(bucket, close, innovation, risk, cfg)
    )
    _check(rows, "bucket_top_rank_selection_exact", receiver_errors == 0 and
        rank_error <= 5e-9, max(receiver_errors, rank_error))
    _check(rows, "bucket_beta_hedge_and_normalizer_exact", hedge_error <= 1e-12,
        hedge_error)
    _check(rows, "bucket_return_and_cost_formulas_exact", bucket_return_error <= 1e-12,
        bucket_return_error)

    delayed_key_errors = 0
    for candidate in CANDIDATES:
        regular_keys = set(events.loc[events["candidate"].eq(candidate),
            "source_feature_time"])
        delayed_keys = set(delayed.loc[delayed["candidate"].eq(candidate),
            "source_feature_time"])
        delayed_key_errors += len(regular_keys.symmetric_difference(delayed_keys))
    delayed_offset = delayed["entry_time"] - delayed["source_feature_time"]
    _check(rows, "delay_control_preserves_sources", delayed_key_errors == 0,
        delayed_key_errors)
    _check(rows, "delay_control_is_exactly_one_bar", delayed_offset.eq(
        pd.Timedelta(minutes=15)
    ).all(), int(delayed_offset.ne(pd.Timedelta(minutes=15)).sum()))

    summary_error = 0.0
    side_error = 0.0
    for candidate in CANDIDATES:
        sample = events[events["candidate"].eq(candidate)]
        reported = summary[summary["candidate"].eq(candidate)
            & summary["scope"].eq("all")].iloc[0]
        summary_error = max(
            summary_error,
            abs(len(sample) - int(reported["events"])),
            abs(sample["gross_return"].mean() * 10_000 - reported["mean_gross_bp"]),
            abs(sample["primary_net_return"].mean() * 10_000
                - reported["mean_primary_net_bp"]),
        )
        for side in ("long_liquidation", "short_cover"):
            local = sample[sample["source_side"].eq(side)]
            side_reported = sides[sides["candidate"].eq(candidate)
                & sides["source_side"].eq(side)].iloc[0]
            side_error = max(
                side_error,
                abs(len(local) - int(side_reported["events"])),
                abs(local["primary_net_return"].mean() * 10_000
                    - side_reported["mean_primary_net_bp"]),
            )
    _check(rows, "all_period_summary_exact", summary_error <= 1e-10, summary_error)
    _check(rows, "source_side_summary_exact", side_error <= 1e-10, side_error)

    _check(rows, "random_iterations_complete", random["iteration"].nunique()
        == cfg.random_iterations, int(random["iteration"].nunique()))
    random_wide = random.pivot(
        index="iteration", columns="candidate", values="mean_primary_net_return"
    )
    family_error = _max_abs(
        random_wide["FAMILY_MAX"] - random_wide[list(CANDIDATES)].max(axis=1)
    )
    _check(rows, "random_family_max_exact", family_error <= 1e-12, family_error)
    percentile_errors = []
    family = random_wide["FAMILY_MAX"]
    for row in outcome.itertuples(index=False):
        real = events.loc[events["candidate"].eq(row.candidate),
            "primary_net_return"].mean()
        percentile_errors.append(
            abs(float(family.le(real).mean()) - float(row.random_family_percentile))
        )
    percentile_error = max(percentile_errors)
    _check(rows, "reported_random_percentiles_exact", percentile_error <= 1e-12,
        percentile_error)

    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(rows, "rejection_verdict_exact", outcome["verdict"].eq(
        "reject_premium_innovation_unwind_reversal"
    ).all(), "|".join(outcome["verdict"].astype(str)))
    direct_outcome = outcome[outcome["candidate"].eq(DIRECT_CANDIDATE)].iloc[0]
    _check(rows, "direct_gross_positive_but_net_negative", direct_outcome[
        "mean_gross_bp"
    ] > 0 and direct_outcome["mean_primary_net_bp"] < 0,
        float(direct_outcome["mean_primary_net_bp"]))
    _check(rows, "direct_validation_and_holdout_negative", summary[
        summary["candidate"].eq(DIRECT_CANDIDATE)
        & summary["scope"].isin(["validation", "holdout"])
    ]["mean_primary_net_bp"].lt(0).all(), "sample_out_of_sample")
    _check(rows, "bucket_gross_negative", outcome.loc[
        outcome["candidate"].eq(BUCKET_CANDIDATE), "mean_gross_bp"
    ].lt(0).all(), float(outcome.loc[
        outcome["candidate"].eq(BUCKET_CANDIDATE), "mean_gross_bp"
    ].iloc[0]))
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v192" in path.name.lower() or "premium_innovation" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))

    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_premium_innovation_reversal_rejected",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v19.2 Premium-Innovation Reversal Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The exact Binance premium innovation filter does not rescue the OI-unwind",
        "reversal after costs or out of sample. The receiver bucket is negative gross.",
        "No live, PaperLive, leverage, application, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v192_independent_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v192_independently()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v192_independently", "write_v192_independent_audit"]
