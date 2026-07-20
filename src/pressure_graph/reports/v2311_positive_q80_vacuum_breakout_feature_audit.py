"""Feature-only audit for a denser positive q80 book-vacuum breakout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)
from pressure_graph.reports.v161_hourly_liquidity_withdrawal_amplification import (
    load_v161_features,
)
from pressure_graph.reports.v224_alt_book_vacuum_pressure_feature_audit import (
    FEATURE_ROOT,
    V224Config,
    add_v224_symbol_states,
    build_v224_bucket_states,
    select_v224_events,
)
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    build_v233_hourly_context,
    load_v233_btc_15m,
)


REPORT_ROOT = Path("reports/v23_11_positive_q80_vacuum_breakout_feature_audit")
FINDINGS_PATH = Path(
    "docs/v2311_positive_q80_vacuum_breakout_feature_audit_2026_07_17.md"
)
CANDIDATE = "DVB6_POSITIVE_Q80_VACUUM_0625SIGMA_BREAKOUT"


@dataclass(frozen=True)
class V2311Config:
    feature_root: Path = FEATURE_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    candidate: str = CANDIDATE
    pressure_quantile: float = 0.80
    minimum_directional_symbols: int = 11
    minimum_withdrawing_symbols: int = 5
    cooldown_hours: int = 4
    sigma_multiple: float = 0.625
    path_hours: int = 4
    bar_minutes: int = 15
    minimum_events: int = 80
    minimum_period_events: int = 20
    minimum_active_months: int = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_v2311_features(
    cfg: V2311Config = V2311Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_v161_features(cfg.feature_root)
    v224_cfg = replace(
        V224Config(),
        pressure_quantile=cfg.pressure_quantile,
        minimum_directional_symbols=cfg.minimum_directional_symbols,
        minimum_withdrawing_symbols=cfg.minimum_withdrawing_symbols,
        cooldown_hours=cfg.cooldown_hours,
        minimum_events=0,
        minimum_period_events=0,
        minimum_direction_period_events=0,
        minimum_active_months=0,
    )
    symbol_states = add_v224_symbol_states(raw, v224_cfg)
    bucket_states = build_v224_bucket_states(symbol_states, v224_cfg)
    events = select_v224_events(bucket_states, v224_cfg)
    events = events[events["signal_direction"].eq(1)].copy()
    bars = load_v233_btc_15m(V233Config())
    hourly = build_v233_hourly_context(bars, V233Config())
    features = events.merge(
        hourly,
        on="entry_time",
        how="left",
        validate="one_to_one",
    )
    expected = cfg.path_hours * 60 // cfg.bar_minutes
    available = set(bars["bar_open_time"])
    features["path_timestamp_count"] = [
        sum(
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset) in available
            for offset in range(expected)
        )
        for entry in features["entry_time"]
    ]
    features = features[
        features["entry_spot"].gt(0)
        & features["causal_hourly_sigma"].gt(0)
        & features["path_timestamp_count"].eq(expected)
    ].copy()
    features["upper_stop_price"] = features["entry_spot"] * np.exp(
        cfg.sigma_multiple * features["causal_hourly_sigma"]
    )
    features["lower_stop_price"] = features["entry_spot"] * np.exp(
        -cfg.sigma_multiple * features["causal_hourly_sigma"]
    )
    features["candidate"] = cfg.candidate
    columns = [
        "candidate",
        "feature_time",
        "entry_time",
        "entry_month",
        "period",
        "signal_direction",
        "bucket_pressure",
        "prior_abs_pressure_threshold",
        "covered_symbols",
        "directional_symbol_count",
        "directional_breadth",
        "withdrawing_symbol_count",
        "withdrawal_breadth",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "causal_hourly_sigma",
        "upper_stop_price",
        "lower_stop_price",
        "path_timestamp_count",
    ]
    return (
        symbol_states,
        bucket_states,
        features[columns].sort_values("entry_time").reset_index(drop=True),
    )


def summarize_v2311(
    features: pd.DataFrame,
    cfg: V2311Config = V2311Config(),
) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features[features["period"].eq(scope)]
        rows.append(
            {
                "candidate": cfg.candidate,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "median_pressure_ratio": float(
                    (
                        local["bucket_pressure"]
                        / local["prior_abs_pressure_threshold"]
                    ).median()
                ),
                "median_withdrawal_breadth": float(
                    local["withdrawal_breadth"].median()
                ),
                "median_barrier_distance_bp": float(
                    np.log(local["upper_stop_price"] / local["entry_spot"]).median()
                    * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v2311_features(
    bucket_states: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V2311Config = V2311Config(),
) -> pd.DataFrame:
    all_row = summary[summary["scope"].eq("all")].iloc[0]
    periods = summary[summary["scope"].ne("all")]
    expected = cfg.path_hours * 60 // cfg.bar_minutes
    gaps = features["entry_time"].sort_values().diff().dropna()
    columns = " ".join(features.columns).lower()
    checks = {
        "bucket_hour_keys_unique": bucket_states["decision_time"].is_unique,
        "all_events_have_positive_pressure": features["bucket_pressure"].gt(0).all(),
        "all_events_exceed_causal_pressure_threshold": bool(
            features["bucket_pressure"]
            .ge(features["prior_abs_pressure_threshold"])
            .all()
        ),
        "directional_breadth_frozen_11_of_16": bool(
            features["directional_symbol_count"]
            .ge(cfg.minimum_directional_symbols)
            .all()
        ),
        "withdrawal_breadth_frozen_5_of_16": bool(
            features["withdrawing_symbol_count"]
            .ge(cfg.minimum_withdrawing_symbols)
            .all()
        ),
        "false_transition_and_four_hour_cooldown": bool(
            gaps.ge(pd.Timedelta(hours=cfg.cooldown_hours)).all()
        ),
        "entry_equals_completed_feature_hour": features["entry_time"].eq(
            features["feature_time"]
        ).all(),
        "causal_sigma_positive_and_finite": bool(
            np.isfinite(features["causal_hourly_sigma"]).all()
            and features["causal_hourly_sigma"].gt(0).all()
        ),
        "barrier_width_exactly_0625_sigma": bool(
            np.allclose(
                np.log(features["upper_stop_price"] / features["entry_spot"]),
                cfg.sigma_multiple * features["causal_hourly_sigma"],
                atol=1e-12,
            )
        ),
        "complete_16_bar_timestamp_paths": features["path_timestamp_count"].eq(
            expected
        ).all(),
        "minimum_total_events": int(all_row["events"]) >= cfg.minimum_events,
        "minimum_each_period_events": periods["events"].ge(
            cfg.minimum_period_events
        ).all(),
        "minimum_active_months": int(all_row["active_months"])
        >= cfg.minimum_active_months,
        "no_post_entry_outcome_columns": not any(
            token in columns
            for token in (
                "trigger",
                "fill",
                "exit",
                "future",
                "return",
                "pnl",
                "gross",
                "net",
            )
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v2311_positive_q80_vacuum_breakout_feature_audit(
    cfg: V2311Config = V2311Config(),
) -> dict[str, Path]:
    _, bucket_states, features = build_v2311_features(cfg)
    summary = summarize_v2311(features, cfg)
    checks = audit_v2311_features(bucket_states, features, summary, cfg)
    hashes = pd.DataFrame(
        [
            {
                "input": str(cfg.feature_root / f"{symbol}.parquet"),
                "sha256": _sha256(cfg.feature_root / f"{symbol}.parquet"),
            }
            for symbol in FROZEN_SYMBOLS
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "positive_q80_breakout_features.parquet",
        "bucket_states": root / "hourly_q80_bucket_states.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    features.to_parquet(paths["features"], index=False)
    bucket_states.to_parquet(paths["bucket_states"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    hashes.to_csv(paths["hashes"], index=False)
    verdict = (
        "feature_viable_freeze_positive_q80_0625sigma_breakout"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.11 Positive-q80 Vacuum Breakout Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "This denser mechanism-preserving candidate keeps positive bucket",
                "pressure, 11/16 directional agreement, 5/16 depth withdrawal,",
                "false transitions, and four-hour cooldown. Only the causal pressure",
                "threshold changes from q90 to the predeclared q80. BTC OCO barriers",
                "are frozen at plus/minus 0.625 trailing hourly sigma.",
                "",
                "No post-entry high, low, trigger, fill, direction, or return was used.",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2311Config",
    "audit_v2311_features",
    "build_v2311_features",
    "summarize_v2311",
    "write_v2311_positive_q80_vacuum_breakout_feature_audit",
]
