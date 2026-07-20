"""Outcome-free audit for q90 book pressure confirmed by broad taker buying."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)


Q90_FEATURE_PATH = Path(
    "reports/v22_4_alt_book_vacuum_pressure_feature_audit/"
    "candidate_feature_events.parquet"
)
METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
REPORT_ROOT = Path("reports/v23_23_q90_broad_taker_confirmation_feature_audit")
FINDINGS_PATH = Path(
    "docs/v2323_q90_broad_taker_confirmation_feature_audit_2026_07_17.md"
)
CANDIDATE = "BTF1_Q90_POSITIVE_PRESSURE_BROAD_TAKER_BUY"


@dataclass(frozen=True)
class V2323Config:
    q90_feature_path: Path = Q90_FEATURE_PATH
    metrics_root: Path = METRICS_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_metric_symbols: int = 16
    minimum_buy_symbols: int = 9
    minimum_events: int = 24
    minimum_period_events: int = 7
    minimum_active_months: int = 9


def load_v2323_inputs(
    cfg: V2323Config = V2323Config(),
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    features = pd.read_parquet(cfg.q90_feature_path)
    for column in ("feature_time", "entry_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="raise")
    positive = features.loc[features["signal_direction"].eq(1)].copy()
    metrics: dict[str, pd.Series] = {}
    for symbol in FROZEN_SYMBOLS:
        frame = pd.read_parquet(
            cfg.metrics_root / f"{symbol}.parquet",
            columns=["create_time", "sum_taker_long_short_vol_ratio"],
        )
        frame["create_time"] = pd.to_datetime(
            frame["create_time"], utc=True, errors="raise"
        )
        values = pd.to_numeric(
            frame["sum_taker_long_short_vol_ratio"], errors="coerce"
        )
        metrics[symbol] = pd.Series(
            values.to_numpy(float),
            index=pd.DatetimeIndex(frame["create_time"]),
            name=symbol,
        ).loc[lambda series: ~series.index.duplicated(keep="last")]
    return positive.sort_values("entry_time").reset_index(drop=True), metrics


def build_v2323_features(
    positive_q90: pd.DataFrame,
    metrics: dict[str, pd.Series],
    cfg: V2323Config = V2323Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for event in positive_q90.itertuples(index=False):
        time = pd.Timestamp(event.entry_time)
        ratios = {
            symbol: float(series.at[time])
            for symbol, series in metrics.items()
            if time in series.index and np.isfinite(series.at[time]) and series.at[time] > 0
        }
        logs = {symbol: float(np.log(value)) for symbol, value in ratios.items()}
        buying = sorted(symbol for symbol, value in logs.items() if value > 0)
        selling = sorted(symbol for symbol, value in logs.items() if value <= 0)
        row = event._asdict()
        row.update(
            {
                "candidate": CANDIDATE,
                "metric_feature_time": time,
                "metric_symbol_count": len(logs),
                "taker_buy_symbol_count": len(buying),
                "taker_buy_breadth": len(buying) / len(logs) if logs else np.nan,
                "median_log_taker_ratio": float(np.median(list(logs.values())))
                if logs
                else np.nan,
                "taker_buy_symbols": "|".join(buying),
                "taker_sell_symbols": "|".join(selling),
            }
        )
        rows.append(row)
    all_events = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    selected = all_events.loc[
        all_events["metric_symbol_count"].ge(cfg.minimum_metric_symbols)
        & all_events["taker_buy_symbol_count"].ge(cfg.minimum_buy_symbols)
    ].copy()
    return all_events, selected.reset_index(drop=True)


def summarize_v2323(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = selected if scope == "all" else selected.loc[selected["period"].eq(scope)]
        rows.append(
            {
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "median_taker_buy_symbols": float(
                    local["taker_buy_symbol_count"].median()
                ),
                "median_taker_buy_breadth": float(local["taker_buy_breadth"].median()),
                "median_log_taker_ratio": float(
                    local["median_log_taker_ratio"].median()
                ),
                "median_bucket_pressure": float(local["bucket_pressure"].median()),
            }
        )
    return pd.DataFrame(rows)


def audit_v2323(
    positive_q90: pd.DataFrame,
    all_events: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V2323Config = V2323Config(),
) -> pd.DataFrame:
    period_counts = selected["period"].value_counts()
    checks = {
        "exactly_53_positive_q90_ancestors": len(positive_q90) == 53,
        "every_q90_event_has_metric_row": len(all_events) == len(positive_q90),
        "complete_16_symbol_metric_coverage": all_events[
            "metric_symbol_count"
        ].eq(16).all(),
        "minimum_selected_events": len(selected) >= cfg.minimum_events,
        "minimum_period_events": all(
            int(period_counts.get(period, 0)) >= cfg.minimum_period_events
            for period in ("development", "validation", "holdout")
        ),
        "minimum_active_months": selected["entry_month"].nunique()
        >= cfg.minimum_active_months,
        "positive_pressure_only": selected["signal_direction"].eq(1).all(),
        "broad_taker_buy_rule_exact": selected["taker_buy_symbol_count"].ge(
            cfg.minimum_buy_symbols
        ).all(),
        "metric_time_equals_decision_time": selected["metric_feature_time"].eq(
            selected["entry_time"]
        ).all(),
        "feature_time_equals_entry_time": selected["feature_time"].eq(
            selected["entry_time"]
        ).all(),
        "breadth_reconciles": np.allclose(
            selected["taker_buy_breadth"],
            selected["taker_buy_symbol_count"] / selected["metric_symbol_count"],
        ),
        "symbol_lists_reconcile": all(
            len(str(row.taker_buy_symbols).split("|"))
            == int(row.taker_buy_symbol_count)
            for row in selected.itertuples(index=False)
        ),
        "summary_reconciles": int(summary.loc[summary["scope"].eq("all"), "events"].iloc[0])
        == len(selected),
        "outcome_columns_absent": {
            "exit_spot",
            "gross_return",
            "primary_net_return",
            "stress_net_return",
        }.isdisjoint(selected.columns),
    }
    return pd.DataFrame(
        [{"check": check, "passed": bool(passed)} for check, passed in checks.items()]
    )


def feature_hash_v2323(features: pd.DataFrame) -> str:
    payload = features.sort_values("entry_time").to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S%z",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def write_v2323_q90_broad_taker_confirmation_feature_audit(
    cfg: V2323Config = V2323Config(),
) -> dict[str, Path]:
    positive, metrics = load_v2323_inputs(cfg)
    all_events, selected = build_v2323_features(positive, metrics, cfg)
    summary = summarize_v2323(selected)
    checks = audit_v2323(positive, all_events, selected, summary, cfg)
    feature_hash = feature_hash_v2323(selected)
    root = ensure_dir(cfg.report_root)
    paths = {
        "all_events": root / "positive_q90_taker_context.parquet",
        "features": root / "broad_taker_confirmed_features.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    all_events.to_parquet(paths["all_events"], index=False)
    selected.to_parquet(paths["features"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    passed = bool(checks["passed"].all())
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "feature_hash": feature_hash,
                "all_checks_passed": passed,
                "outcomes_loaded": False,
                "config": {
                    **asdict(cfg),
                    "q90_feature_path": str(cfg.q90_feature_path),
                    "metrics_root": str(cfg.metrics_root),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "feature_viable_freeze_broad_taker_confirmation" if passed else "feature_audit_failed"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.23 q90 Broad-Taker Confirmation Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                f"Feature hash: `{feature_hash}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Every q90 event is joined to the exact completed Binance metrics",
                "timestamp. Selection requires at least 9 of 16 taker long/short",
                "volume ratios above one. No BTC exit price or return was loaded.",
                "",
                "This is a second-stage diagnostic of a post-selected ancestor.",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2323Config",
    "audit_v2323",
    "build_v2323_features",
    "feature_hash_v2323",
    "load_v2323_inputs",
    "summarize_v2323",
    "write_v2323_q90_broad_taker_confirmation_feature_audit",
]
