"""Independent audit for the v19.4 opposing-premium continuation reveal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v190_binance_premium_index_audit import (
    load_v190_premium_ohlc_panels,
)
from pressure_graph.reports.v194_opposing_premium_absorption_continuation import (
    BUCKET_CANDIDATE,
    CANDIDATES,
    DIRECT_CANDIDATE,
    REPORT_ROOT,
    V194Config,
)


AUDIT_ROOT = Path(
    "reports/v19_4_opposing_premium_absorption_continuation_audit"
)
FINDINGS_PATH = Path(
    "docs/v194_opposing_premium_absorption_continuation_audit_2026_07_17.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def _max_abs(values: pd.Series | np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmax(np.abs(array))) if array.size else 0.0


def _load_outputs(report_root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "signals": pd.read_parquet(report_root / "selected_source_signals.parquet"),
        "risk": pd.read_parquet(report_root / "monthly_risk_estimates.parquet"),
        "range_z": pd.read_parquet(report_root / "premium_range_z.parquet"),
        "body_z": pd.read_parquet(report_root / "premium_body_z.parquet"),
        "location": pd.read_parquet(report_root / "premium_close_location.parquet"),
        "events": pd.read_parquet(report_root / "candidate_events.parquet"),
        "delayed": pd.read_parquet(report_root / "delayed_candidate_events.parquet"),
        "random": pd.read_parquet(report_root / "random_controls.parquet"),
        "summary": pd.read_csv(report_root / "period_summary.csv"),
        "sides": pd.read_csv(report_root / "source_side_summary.csv"),
        "gates": pd.read_csv(report_root / "candidate_gates.csv"),
        "outcome": pd.read_csv(report_root / "candidate_outcome.csv"),
    }
    for column in ("feature_time", "source_feature_time"):
        frames["signals"][column] = pd.to_datetime(
            frames["signals"][column], utc=True
        )
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
    range_z: pd.DataFrame,
    body_z: pd.DataFrame,
    location: pd.DataFrame,
    risk: pd.DataFrame,
    cfg: V194Config,
) -> tuple[float, float, int]:
    ranking_errors: list[float] = []
    formula_errors: list[float] = []
    selection_errors = 0
    for event in events.itertuples(index=False):
        timestamp = pd.Timestamp(event.source_feature_time)
        local = risk[risk["risk_month"].eq(_month(timestamp))].set_index("receiver")
        names = [name for name in local.index.astype(str) if name in range_z.columns]
        frame = pd.DataFrame(index=names)
        frame["btc_beta"] = local.reindex(names)["btc_beta"].astype(float)
        frame["range_z"] = range_z.loc[timestamp, names].astype(float)
        frame["aligned_body_z"] = (
            float(event.source_sign) * body_z.loc[timestamp, names].astype(float)
        )
        frame["aligned_location"] = (
            float(event.source_sign) * location.loc[timestamp, names].astype(float)
        )
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        eligible = frame[
            frame["range_z"].ge(cfg.receiver_range_z_threshold)
            & frame["aligned_body_z"].le(-cfg.body_z_threshold)
            & frame["aligned_location"].le(-cfg.close_location_threshold)
        ]
        expected = eligible.sort_values("range_z", ascending=False).head(
            cfg.receiver_bucket_size
        )
        reported = str(event.receivers).split("|")
        selection_errors += int(
            reported != expected.index.astype(str).tolist()
            or int(event.eligible_receivers) != len(eligible)
            or int(event.receiver_count) != len(reported)
        )
        reported_scores = np.asarray(
            [float(item.rsplit(":", 1)[1]) for item in str(event.receiver_scores).split("|")]
        )
        ranking_errors.extend(
            (reported_scores - expected["range_z"].to_numpy(dtype=float)).tolist()
        )
        beta = expected["btc_beta"].to_numpy(dtype=float)
        direction = float(event.source_sign)
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
        formula_errors.extend(
            [
                hedge - float(event.btc_hedge_weight),
                normalizer - float(event.normalizer),
                gross - float(event.gross_return),
                gross - cfg.primary_cost - float(event.primary_net_return),
                gross - cfg.stress_cost - float(event.stress_net_return),
                -gross - cfg.primary_cost
                - float(event.reversed_primary_net_return),
            ]
        )
    return _max_abs(ranking_errors), _max_abs(formula_errors), selection_errors


def audit_v194_independently(
    report_root: Path = REPORT_ROOT,
    cfg: V194Config = V194Config(),
) -> pd.DataFrame:
    frames = _load_outputs(report_root)
    signals = frames["signals"]
    risk = frames["risk"]
    events = frames["events"]
    delayed = frames["delayed"]
    random = frames["random"]
    summary = frames["summary"]
    sides = frames["sides"]
    gates = frames["gates"]
    outcome = frames["outcome"]
    close, _ = load_v184_exact_panels()
    ohlc = {
        field: panel.reindex(index=close.index, columns=close.columns)
        for field, panel in load_v190_premium_ohlc_panels().items()
    }
    rows: list[dict[str, object]] = []

    _check(rows, "source_keys_unique", not signals.duplicated(
        ["source_feature_time"]
    ).any(), len(signals))
    _check(rows, "source_feature_time_is_feature_time", signals[
        "source_feature_time"
    ].eq(signals["feature_time"]).all(), int(signals["source_feature_time"].ne(
        signals["feature_time"]
    ).sum()))
    price_return = close[BTC].pct_change(fill_method=None)
    price_threshold = price_return.abs().shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    ).quantile(cfg.source_return_quantile)
    premium_range = ohlc["high"] - ohlc["low"]
    range_mean = premium_range.shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    ).mean()
    range_scale = premium_range.shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    ).std(ddof=1)
    expected_range_z = (premium_range - range_mean).div(
        range_scale.where(range_scale.gt(0))
    )
    body = ohlc["close"] - ohlc["open"]
    body_scale = body.shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    ).std(ddof=1)
    expected_body_z = body.div(body_scale.where(body_scale.gt(0)))
    expected_location = (
        2.0 * (ohlc["close"] - ohlc["low"]).div(premium_range.where(
            premium_range.gt(0)
        ))
        - 1.0
    ).clip(-1.0, 1.0)
    source_times = pd.DatetimeIndex(signals["feature_time"])
    feature_error = max(
        _max_abs(signals["btc_premium_range_z"].to_numpy()
            - expected_range_z.loc[source_times, BTC].to_numpy()),
        _max_abs(signals["btc_aligned_body_z"].to_numpy()
            - signals["source_sign"].to_numpy()
            * expected_body_z.loc[source_times, BTC].to_numpy()),
        _max_abs(signals["btc_aligned_close_location"].to_numpy()
            - signals["source_sign"].to_numpy()
            * expected_location.loc[source_times, BTC].to_numpy()),
    )
    _check(rows, "shifted_premium_shape_features_exact", feature_error <= 1e-12,
        feature_error)
    expected_mask = (
        price_return.abs().ge(price_threshold)
        & expected_range_z[BTC].ge(cfg.source_range_z_threshold)
        & (np.sign(price_return) * expected_body_z[BTC]).le(-cfg.body_z_threshold)
        & (np.sign(price_return) * expected_location[BTC]).le(
            -cfg.close_location_threshold
        )
    )
    expected_keys = set(expected_mask[expected_mask].index)
    actual_keys = set(signals["source_feature_time"])
    _check(rows, "frozen_source_set_exact", expected_keys == actual_keys,
        len(expected_keys.symmetric_difference(actual_keys)))
    threshold_error = _max_abs(
        signals["return_threshold"].to_numpy()
        - price_threshold.loc[source_times].to_numpy()
    )
    _check(rows, "shifted_price_threshold_exact", threshold_error <= 1e-12,
        threshold_error)
    side_expected = np.where(signals["source_sign"].gt(0), "up_move", "down_move")
    _check(rows, "source_side_mapping_exact", signals["source_side"].eq(
        side_expected
    ).all(), int(signals["source_side"].ne(side_expected).sum()))

    direct = events[events["candidate"].eq(DIRECT_CANDIDATE)].copy()
    bucket = events[events["candidate"].eq(BUCKET_CANDIDATE)].copy()
    _check(rows, "candidate_domain_exact", set(events["candidate"]) == set(CANDIDATES),
        "|".join(sorted(events["candidate"].unique())))
    _check(rows, "direct_has_every_source", set(direct["source_feature_time"])
        == actual_keys, len(direct))
    _check(rows, "bucket_sources_subset", set(bucket["source_feature_time"])
        <= actual_keys, len(bucket))
    _check(rows, "bucket_min_receiver_rule", bucket["receiver_count"].ge(
        cfg.min_receiver_bucket
    ).all(), int(bucket["receiver_count"].min()))
    _check(rows, "entry_at_completed_source_close", events["entry_time"].eq(
        events["source_feature_time"]
    ).all(), int(events["entry_time"].ne(events["source_feature_time"]).sum()))
    holding = events["exit_time"] - events["entry_time"]
    _check(rows, "holding_exact_30m", holding.eq(pd.Timedelta(minutes=30)).all(),
        int(holding.ne(pd.Timedelta(minutes=30)).sum()))
    _check(rows, "trade_direction_continues_source", events["trade_direction"].eq(
        events["source_sign"]
    ).all(), int(events["trade_direction"].ne(events["source_sign"]).sum()))

    entry = pd.Series(
        [close.at[timestamp, BTC] for timestamp in direct["entry_time"]],
        index=direct.index,
    )
    exit_price = pd.Series(
        [close.at[timestamp, BTC] for timestamp in direct["exit_time"]],
        index=direct.index,
    )
    expected_underlying = exit_price / entry - 1.0
    expected_gross = direct["trade_direction"] * expected_underlying
    direct_error = max(
        _max_abs(direct["btc_underlying_return"] - expected_underlying),
        _max_abs(direct["gross_return"] - expected_gross),
        _max_abs(direct["primary_net_return"] - (
            expected_gross - cfg.direct_primary_cost
        )),
        _max_abs(direct["stress_net_return"] - (
            expected_gross - cfg.direct_stress_cost
        )),
        _max_abs(direct["reversed_primary_net_return"] - (
            -expected_gross - cfg.direct_primary_cost
        )),
    )
    _check(rows, "direct_return_cost_reverse_exact", direct_error <= 1e-12,
        direct_error)

    rank_error, bucket_error, selection_errors = _recompute_bucket_errors(
        bucket,
        close,
        frames["range_z"],
        frames["body_z"],
        frames["location"],
        risk,
        cfg,
    )
    _check(rows, "bucket_top_range_selection_exact", selection_errors == 0
        and rank_error <= 5e-9, max(selection_errors, rank_error))
    _check(rows, "bucket_hedge_return_cost_exact", bucket_error <= 1e-12,
        bucket_error)

    delay_key_errors = 0
    for candidate in CANDIDATES:
        regular = set(events.loc[events["candidate"].eq(candidate),
            "source_feature_time"])
        shifted = set(delayed.loc[delayed["candidate"].eq(candidate),
            "source_feature_time"])
        delay_key_errors += len(regular.symmetric_difference(shifted))
    _check(rows, "delay_control_preserves_sources", delay_key_errors == 0,
        delay_key_errors)
    delay = delayed["entry_time"] - delayed["source_feature_time"]
    _check(rows, "delay_control_exact_one_bar", delay.eq(
        pd.Timedelta(minutes=15)
    ).all(), int(delay.ne(pd.Timedelta(minutes=15)).sum()))

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
        for side in ("up_move", "down_move"):
            local = sample[sample["source_side"].eq(side)]
            reported_side = sides[sides["candidate"].eq(candidate)
                & sides["source_side"].eq(side)].iloc[0]
            side_error = max(
                side_error,
                abs(len(local) - int(reported_side["events"])),
                abs(local["primary_net_return"].mean() * 10_000
                    - reported_side["mean_primary_net_bp"]),
            )
    _check(rows, "period_summary_exact", summary_error <= 1e-10, summary_error)
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
    family = random_wide["FAMILY_MAX"]
    percentile_errors = []
    for row in outcome.itertuples(index=False):
        mean = events.loc[events["candidate"].eq(row.candidate),
            "primary_net_return"].mean()
        percentile_errors.append(
            abs(float(family.le(mean).mean()) - float(row.random_family_percentile))
        )
    percentile_error = max(percentile_errors)
    _check(rows, "reported_random_percentiles_exact", percentile_error <= 1e-12,
        percentile_error)
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(rows, "rejection_verdict_exact", outcome["verdict"].eq(
        "reject_opposing_premium_absorption_continuation"
    ).all(), "|".join(outcome["verdict"].astype(str)))
    direct_row = outcome[outcome["candidate"].eq(DIRECT_CANDIDATE)].iloc[0]
    bucket_row = outcome[outcome["candidate"].eq(BUCKET_CANDIDATE)].iloc[0]
    _check(rows, "direct_continuation_gross_negative", direct_row["mean_gross_bp"] < 0,
        float(direct_row["mean_gross_bp"]))
    _check(rows, "bucket_continuation_gross_positive_but_net_negative",
        bucket_row["mean_gross_bp"] > 0 and bucket_row["mean_primary_net_bp"] < 0,
        float(bucket_row["mean_primary_net_bp"]))
    _check(rows, "bucket_gross_positive_all_periods", summary[
        summary["candidate"].eq(BUCKET_CANDIDATE)
    ]["mean_gross_bp"].gt(0).all(), "all_periods")
    bucket_times = pd.DatetimeIndex(bucket.sort_values("entry_time")["entry_time"])
    overlaps = int(bucket_times.to_series().diff().le(pd.Timedelta(minutes=30)).sum())
    _check(rows, "bucket_overlap_too_sparse_for_stateful_rescue", overlaps <= 5,
        overlaps)
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v194" in path.name.lower() or "opposing_premium" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))

    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_absorption_continuation_rejected_subcost_bucket_retained",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v19.4 Opposing-Premium Continuation Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The receiver bucket retains a directionally consistent but sub-cost gross",
        "effect. Events are too isolated for stateful netting to close the cost gap.",
        "No live, PaperLive, leverage, application, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v194_independent_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v194_independently()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v194_independently", "write_v194_independent_audit"]
