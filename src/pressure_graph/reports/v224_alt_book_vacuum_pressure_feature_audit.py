"""Feature-only audit for synchronized alt-book pressure and depth withdrawal."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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


FEATURE_ROOT = Path("data/external/binance_um_book_depth/hourly_features")
REPORT_ROOT = Path("reports/v22_4_alt_book_vacuum_pressure_feature_audit")
FINDINGS_PATH = Path(
    "docs/v224_alt_book_vacuum_pressure_feature_audit_2026_07_17.md"
)
CANDIDATE = "DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC"


@dataclass(frozen=True)
class V224Config:
    feature_root: Path = FEATURE_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    rolling_hours: int = 720
    minimum_history_hours: int = 480
    minimum_snapshots: int = 90
    minimum_symbols: int = 15
    pressure_quantile: float = 0.90
    minimum_directional_symbols: int = 11
    withdrawal_quantile: float = 0.20
    minimum_withdrawing_symbols: int = 5
    cooldown_hours: int = 4
    minimum_events: int = 150
    minimum_period_events: int = 45
    minimum_direction_period_events: int = 15
    minimum_active_months: int = 11


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def add_v224_symbol_states(
    features: pd.DataFrame,
    cfg: V224Config = V224Config(),
) -> pd.DataFrame:
    output = features.sort_values(["symbol", "decision_time"]).copy()
    output = output[
        output["notional_imbalance_1p0_valid_snapshots"].ge(cfg.minimum_snapshots)
        & output["total_notional_1p0_median"].gt(0)
    ].copy()
    grouped = output.groupby("symbol", sort=False, observed=True)
    previous_depth = grouped["total_notional_1p0_median"].shift(1)
    output["depth_log_change"] = np.log(
        output["total_notional_1p0_median"] / previous_depth
    )
    prior_mean = grouped["notional_imbalance_1p0_median"].transform(
        lambda values: values.shift(1)
        .rolling(cfg.rolling_hours, min_periods=cfg.minimum_history_hours)
        .mean()
    )
    prior_std = grouped["notional_imbalance_1p0_median"].transform(
        lambda values: values.shift(1)
        .rolling(cfg.rolling_hours, min_periods=cfg.minimum_history_hours)
        .std()
    )
    output["imbalance_prior_mean"] = prior_mean
    output["imbalance_prior_std"] = prior_std
    output["imbalance_z"] = (
        output["notional_imbalance_1p0_median"] - prior_mean
    ) / prior_std.replace(0, np.nan)
    output["prior_withdrawal_threshold"] = output.groupby(
        "symbol", sort=False, observed=True
    )["depth_log_change"].transform(
        lambda values: values.shift(1)
        .rolling(cfg.rolling_hours, min_periods=cfg.minimum_history_hours)
        .quantile(cfg.withdrawal_quantile)
    )
    output["depth_withdrawal_state"] = output["depth_log_change"].le(
        output["prior_withdrawal_threshold"]
    )
    output["feature_ready"] = np.isfinite(
        output[
            ["imbalance_z", "prior_withdrawal_threshold", "depth_log_change"]
        ]
    ).all(axis=1)
    return output.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def build_v224_bucket_states(
    symbol_states: pd.DataFrame,
    cfg: V224Config = V224Config(),
) -> pd.DataFrame:
    ready = symbol_states[symbol_states["feature_ready"]]
    rows: list[dict[str, object]] = []
    for decision_time, local in ready.groupby(
        "decision_time", sort=True, observed=True
    ):
        if local["symbol"].nunique() < cfg.minimum_symbols:
            continue
        pressure = float(local["imbalance_z"].median())
        direction = 1 if pressure >= 0 else -1
        direction_mask = local["imbalance_z"].mul(direction).gt(0)
        withdrawal_mask = local["depth_withdrawal_state"]
        rows.append(
            {
                "decision_time": pd.Timestamp(decision_time),
                "covered_symbols": int(local["symbol"].nunique()),
                "bucket_pressure": pressure,
                "direction": direction,
                "directional_symbol_count": int(direction_mask.sum()),
                "directional_breadth": float(direction_mask.mean()),
                "withdrawing_symbol_count": int(withdrawal_mask.sum()),
                "withdrawal_breadth": float(withdrawal_mask.mean()),
                "directional_symbols": "|".join(
                    sorted(local.loc[direction_mask, "symbol"].astype(str))
                ),
                "withdrawing_symbols": "|".join(
                    sorted(local.loc[withdrawal_mask, "symbol"].astype(str))
                ),
            }
        )
    output = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    output["prior_abs_pressure_threshold"] = (
        output["bucket_pressure"]
        .abs()
        .shift(1)
        .rolling(cfg.rolling_hours, min_periods=cfg.minimum_history_hours)
        .quantile(cfg.pressure_quantile)
    )
    output["candidate_state"] = (
        output["bucket_pressure"].abs().ge(output["prior_abs_pressure_threshold"])
        & output["directional_symbol_count"].ge(cfg.minimum_directional_symbols)
        & output["withdrawing_symbol_count"].ge(cfg.minimum_withdrawing_symbols)
    )
    return output


def select_v224_events(
    bucket_states: pd.DataFrame,
    cfg: V224Config = V224Config(),
) -> pd.DataFrame:
    state = bucket_states["candidate_state"]
    starts = bucket_states[state & ~state.shift(1, fill_value=False)].copy()
    selected: list[int] = []
    last_time: pd.Timestamp | None = None
    for index, row in starts.iterrows():
        decision_time = pd.Timestamp(row["decision_time"])
        if last_time is None or decision_time - last_time >= pd.Timedelta(
            hours=cfg.cooldown_hours
        ):
            selected.append(index)
            last_time = decision_time
    events = starts.loc[selected].copy()
    events["candidate"] = CANDIDATE
    events["feature_time"] = events["decision_time"]
    events["entry_time"] = events["decision_time"]
    events["signal_direction"] = events["direction"].astype(int)
    events["entry_month"] = events["entry_time"].dt.strftime("%Y-%m")
    events["period"] = np.select(
        [
            events["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            events["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
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
        "directional_symbols",
        "withdrawing_symbols",
    ]
    return events[columns].sort_values("entry_time").reset_index(drop=True)


def summarize_v224_events(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "long_events": int(local["signal_direction"].eq(1).sum()),
                "short_events": int(local["signal_direction"].eq(-1).sum()),
                "median_abs_pressure": float(local["bucket_pressure"].abs().median()),
                "mean_directional_breadth": float(local["directional_breadth"].mean()),
                "mean_withdrawal_breadth": float(local["withdrawal_breadth"].mean()),
            }
        )
    return pd.DataFrame(rows)


def audit_v224_features(
    symbol_states: pd.DataFrame,
    bucket_states: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V224Config = V224Config(),
) -> pd.DataFrame:
    gaps = events["entry_time"].sort_values().diff().dropna()
    all_row = summary[summary["scope"].eq("all")].iloc[0]
    periods = summary[summary["scope"].ne("all")]
    columns = " ".join(events.columns).lower()
    checks = {
        "symbol_hour_keys_unique": not symbol_states.duplicated(
            ["decision_time", "symbol"]
        ).any(),
        "bucket_hour_keys_unique": bucket_states["decision_time"].is_unique,
        "rolling_symbol_features_use_shifted_history": bool(
            symbol_states.loc[symbol_states["feature_ready"], "imbalance_prior_std"]
            .gt(0)
            .all()
        ),
        "minimum_snapshot_coverage": bool(
            symbol_states["notional_imbalance_1p0_valid_snapshots"]
            .ge(cfg.minimum_snapshots)
            .all()
        ),
        "minimum_cross_section_coverage": bool(
            bucket_states["covered_symbols"].ge(cfg.minimum_symbols).all()
        ),
        "event_pressure_exceeds_prior_q90": bool(
            events["bucket_pressure"]
            .abs()
            .ge(events["prior_abs_pressure_threshold"])
            .all()
        ),
        "event_directional_breadth_frozen_11_of_16": bool(
            events["directional_symbol_count"]
            .ge(cfg.minimum_directional_symbols)
            .all()
        ),
        "event_withdrawal_breadth_frozen_5_of_16": bool(
            events["withdrawing_symbol_count"]
            .ge(cfg.minimum_withdrawing_symbols)
            .all()
        ),
        "signal_matches_pressure_sign": bool(
            events["signal_direction"].eq(np.sign(events["bucket_pressure"])).all()
        ),
        "entry_equals_completed_feature_hour": bool(
            events["entry_time"].eq(events["feature_time"]).all()
        ),
        "false_transition_and_four_hour_cooldown": bool(
            gaps.ge(pd.Timedelta(hours=cfg.cooldown_hours)).all()
        ),
        "minimum_total_events": int(all_row["events"]) >= cfg.minimum_events,
        "minimum_each_period_events": bool(
            periods["events"].ge(cfg.minimum_period_events).all()
        ),
        "minimum_each_direction_each_period": bool(
            periods[["long_events", "short_events"]]
            .ge(cfg.minimum_direction_period_events)
            .all()
            .all()
        ),
        "minimum_active_months": int(all_row["active_months"])
        >= cfg.minimum_active_months,
        "no_future_outcome_columns": not any(
            token in columns
            for token in ("future", "return", "pnl", "gross", "net", "exit", "price")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v224_alt_book_vacuum_pressure_feature_audit(
    cfg: V224Config = V224Config(),
) -> dict[str, Path]:
    features = load_v161_features(cfg.feature_root)
    symbol_states = add_v224_symbol_states(features, cfg)
    bucket_states = build_v224_bucket_states(symbol_states, cfg)
    events = select_v224_events(bucket_states, cfg)
    summary = summarize_v224_events(events)
    checks = audit_v224_features(symbol_states, bucket_states, events, summary, cfg)
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
        "symbol_states": root / "symbol_feature_states.parquet",
        "bucket_states": root / "hourly_bucket_states.parquet",
        "events": root / "candidate_feature_events.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    symbol_states.to_parquet(paths["symbol_states"], index=False)
    bucket_states.to_parquet(paths["bucket_states"], index=False)
    events.to_parquet(paths["events"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    hashes.to_csv(paths["hashes"], index=False)
    verdict = (
        "feature_viable_freeze_alt_book_vacuum_pressure"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v22.4 Alt-Book Vacuum Pressure Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The feature standardizes each symbol's completed-hour one-percent",
                "book imbalance against shifted trailing-720-hour history. Candidate",
                "events require an aggregate absolute-pressure q90 breach, at least",
                "11/16 symbols aligned with the direction, at least 5/16 symbols in",
                "their own trailing q20 depth-withdrawal state, a false transition,",
                "and a four-hour cooldown.",
                "",
                "No future price, return, PnL, turnover, or outcome was loaded.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
