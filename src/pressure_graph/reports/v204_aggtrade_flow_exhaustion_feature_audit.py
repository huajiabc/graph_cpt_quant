"""Feature-only audit for aggTrade flow exhaustion in extreme graph events."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.binance_aggtrade_event_history import (
    FEATURE_PATH,
    build_extreme_overshoot_tasks,
)
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import KLINE_ROOT
from pressure_graph.reports.v201_reference_price_transmission import (
    FEATURE_EVENTS_PATH,
)


REPORT_ROOT = Path("reports/v20_4_aggtrade_flow_exhaustion_feature_audit")
FINDINGS_PATH = Path(
    "docs/v204_aggtrade_flow_exhaustion_feature_audit_2026_07_17.md"
)
EVENT_WIDE = "RFX1_EVENT_WIDE_LATE_FLOW_REVERSAL_FADE"
STATE_SPREAD = "RFX2_EXHAUSTED_VS_PERSISTENT_FLOW_SPREAD"
CANDIDATES = (EVENT_WIDE, STATE_SPREAD)


@dataclass(frozen=True)
class V204FeatureConfig:
    expected_tasks: int = 936
    expected_events: int = 216
    expected_symbols: int = 45
    window_minutes: int = 15
    minimum_trades: int = 100
    maximum_boundary_gap_seconds: float = 10.0
    minimum_candidate_events: int = 45
    minimum_period_events: int = 5


def add_v204_quality_fields(
    receiver_features: pd.DataFrame,
    cfg: V204FeatureConfig = V204FeatureConfig(),
) -> pd.DataFrame:
    output = receiver_features.copy()
    for column in (
        "feature_time",
        "window_start",
        "window_end",
        "first_trade_time",
        "last_trade_time",
    ):
        output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    output["first_gap_seconds"] = (
        output["first_trade_time"] - output["window_start"]
    ).dt.total_seconds()
    output["last_gap_seconds"] = (
        output["window_end"] - output["last_trade_time"]
    ).dt.total_seconds()
    finite_columns = [
        "price_return",
        "aligned_price_return",
        "buy_sell_imbalance",
        "aligned_buy_sell_imbalance",
        "early_aligned_imbalance",
        "middle_aligned_imbalance",
        "late_aligned_imbalance",
        "aligned_flow_exhaustion",
        "large_aligned_imbalance",
    ]
    finite = np.isfinite(output[finite_columns].to_numpy(dtype=float)).all(axis=1)
    output["quality_ok"] = (
        finite
        & output["trade_count"].ge(cfg.minimum_trades)
        & output["first_gap_seconds"].between(
            0.0, cfg.maximum_boundary_gap_seconds, inclusive="both"
        )
        & output["last_gap_seconds"].between(
            0.0, cfg.maximum_boundary_gap_seconds, inclusive="both"
        )
    )
    output["strict_exhausted"] = (
        output["quality_ok"]
        & output["early_aligned_imbalance"].gt(0.0)
        & output["late_aligned_imbalance"].lt(0.0)
    )
    output["persistent_flow"] = (
        output["quality_ok"]
        & output["early_aligned_imbalance"].gt(0.0)
        & output["late_aligned_imbalance"].ge(0.0)
    )
    return output


def add_kline_validation(
    receiver_features: pd.DataFrame,
    kline_root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for symbol, local in receiver_features.groupby("symbol", sort=True):
        kline = pd.read_parquet(
            kline_root / f"{symbol}.parquet",
            columns=["bar_open_time", "open", "close"],
        )
        kline["bar_open_time"] = pd.to_datetime(
            kline["bar_open_time"], utc=True, errors="coerce"
        )
        kline["kline_price_return"] = (
            pd.to_numeric(kline["close"], errors="coerce")
            / pd.to_numeric(kline["open"], errors="coerce")
            - 1.0
        )
        lookup = (
            kline.dropna(subset=["bar_open_time"])
            .drop_duplicates("bar_open_time", keep="last")
            .set_index("bar_open_time")["kline_price_return"]
        )
        enriched = local.copy()
        enriched["kline_price_return"] = enriched["window_start"].map(lookup)
        enriched["kline_abs_return_difference"] = (
            enriched["price_return"] - enriched["kline_price_return"]
        ).abs()
        frames.append(enriched)
    output = pd.concat(frames, ignore_index=True).sort_values(
        ["feature_time", "community_id", "symbol"]
    )
    rows: list[dict[str, object]] = []
    for symbol, local in output.groupby("symbol", sort=True):
        rows.append(
            {
                "symbol": symbol,
                "windows": len(local),
                "matched_windows": int(local["kline_price_return"].notna().sum()),
                "return_correlation": float(
                    local[["price_return", "kline_price_return"]].corr().iloc[0, 1]
                ),
                "median_abs_return_difference": float(
                    local["kline_abs_return_difference"].median()
                ),
                "p99_abs_return_difference": float(
                    local["kline_abs_return_difference"].quantile(0.99)
                ),
                "maximum_abs_return_difference": float(
                    local["kline_abs_return_difference"].max()
                ),
            }
        )
    return output.reset_index(drop=True), pd.DataFrame(rows)


def _symbols(local: pd.DataFrame, mask: pd.Series | None = None) -> str:
    selected = local if mask is None else local[mask]
    return "|".join(sorted(selected["symbol"].astype(str)))


def build_v204_candidate_features(receiver_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, local in receiver_features.groupby("source_event_id", sort=True):
        local = local.sort_values("symbol")
        first = local.iloc[0]
        quality = local["quality_ok"]
        exhausted = local["strict_exhausted"]
        persistent = local["persistent_flow"]
        late_opposes = quality & local["late_flow_opposes_source"]
        quality_count = int(quality.sum())
        base = {
            "source_event_id": first["source_event_id"],
            "feature_time": first["feature_time"],
            "period": first["period"],
            "entry_day": pd.Timestamp(first["feature_time"]).date(),
            "entry_month": pd.Timestamp(first["feature_time"]).strftime("%Y-%m"),
            "community_id": first["community_id"],
            "source_sign": float(first["source_sign"]),
            "original_receiver_count": len(local),
            "quality_receiver_count": quality_count,
            "late_opposes_count": int(late_opposes.sum()),
            "late_opposes_fraction": (
                float(late_opposes.sum() / quality_count) if quality_count else math.nan
            ),
            "strict_exhausted_count": int(exhausted.sum()),
            "persistent_count": int(persistent.sum()),
            "mean_late_aligned_imbalance": float(
                local.loc[quality, "late_aligned_imbalance"].mean()
            ),
            "mean_aligned_flow_exhaustion": float(
                local.loc[quality, "aligned_flow_exhaustion"].mean()
            ),
            "all_quality_receivers": _symbols(local, quality),
            "strict_exhausted_receivers": _symbols(local, exhausted),
            "persistent_receivers": _symbols(local, persistent),
        }
        if quality_count >= 3 and late_opposes.sum() / quality_count >= 0.5:
            rows.append(
                {
                    **base,
                    "candidate": EVENT_WIDE,
                    "candidate_receiver_count": quality_count,
                    "candidate_receivers": _symbols(local, quality),
                }
            )
        if exhausted.sum() >= 1 and persistent.sum() >= 2:
            rows.append(
                {
                    **base,
                    "candidate": STATE_SPREAD,
                    "candidate_receiver_count": int(exhausted.sum() + persistent.sum()),
                    "candidate_receivers": _symbols(local, exhausted | persistent),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["feature_time", "candidate", "community_id"]
    ).reset_index(drop=True)


def summarize_v204_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = candidates[candidates["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": len(sample),
                    "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                    "active_months": (
                        sample["entry_month"].nunique() if len(sample) else 0
                    ),
                    "mean_candidate_receivers": (
                        float(sample["candidate_receiver_count"].mean())
                        if len(sample)
                        else math.nan
                    ),
                    "mean_late_opposes_fraction": (
                        float(sample["late_opposes_fraction"].mean())
                        if len(sample)
                        else math.nan
                    ),
                    "mean_strict_exhausted_count": (
                        float(sample["strict_exhausted_count"].mean())
                        if len(sample)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_v204_features(
    expected_tasks: pd.DataFrame,
    receivers: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    cfg: V204FeatureConfig = V204FeatureConfig(),
) -> pd.DataFrame:
    expected_ids = set(expected_tasks["task_id"].astype(str))
    observed_ids = set(receivers["task_id"].astype(str))
    all_coverage = coverage[coverage["scope"].eq("all")].set_index("candidate")
    period_coverage = coverage[coverage["scope"].ne("all")]
    checks = {
        "expected_task_count_936": len(expected_tasks) == cfg.expected_tasks,
        "observed_task_count_936": len(receivers) == cfg.expected_tasks,
        "task_ids_exactly_match_frozen_set": observed_ids == expected_ids,
        "task_ids_unique": receivers["task_id"].nunique() == len(receivers),
        "source_events_216": (
            receivers["source_event_id"].nunique() == cfg.expected_events
        ),
        "symbols_45": receivers["symbol"].nunique() == cfg.expected_symbols,
        "window_is_exactly_15_minutes": bool(
            receivers["window_end"]
            .sub(receivers["window_start"])
            .eq(pd.Timedelta(minutes=cfg.window_minutes))
            .all()
        ),
        "feature_time_equals_window_end": bool(
            receivers["feature_time"].eq(receivers["window_end"]).all()
        ),
        "all_windows_pass_quality_gate": bool(receivers["quality_ok"].all()),
        "all_kline_windows_matched": bool(
            receivers["kline_price_return"].notna().all()
        ),
        "aggregate_kline_return_correlation_gt_0_9999": bool(
            receivers[["price_return", "kline_price_return"]]
            .corr()
            .iloc[0, 1]
            > 0.9999
        ),
        "kline_return_difference_p99_lt_10bp": bool(
            receivers["kline_abs_return_difference"].quantile(0.99) < 0.001
        ),
        "source_provenance_present": bool(
            (
                receivers["pages"].notna()
                | receivers["source_mode"].eq("checked_zip_crc_archive")
            ).all()
        ),
        "candidate_all_sample_coverage": bool(
            all_coverage["events"].ge(cfg.minimum_candidate_events).all()
        ),
        "candidate_each_period_coverage": bool(
            period_coverage["events"].ge(cfg.minimum_period_events).all()
        ),
        "no_future_return_columns": not any(
            any(token in column.lower() for token in ("future", "gross", "net_return"))
            for column in candidates.columns
        ),
    }
    return pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )


def _write_findings(
    checks: pd.DataFrame,
    coverage: pd.DataFrame,
    receivers: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "feature_viable_freeze_two_candidates"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    correlation = receivers[["price_return", "kline_price_return"]].corr().iloc[0, 1]
    p99_difference = receivers["kline_abs_return_difference"].quantile(0.99)
    text = [
        "# v20.4 AggTrade Flow-Exhaustion Feature Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"All {len(receivers):,} frozen receiver windows are present across "
        f"{receivers['source_event_id'].nunique()} events and "
        f"{receivers['symbol'].nunique()} symbols. Every window passed the frozen "
        "trade-count and boundary-coverage quality gate.",
        "",
        f"The aggTrade first/last return agrees with the official 15-minute kline: "
        f"correlation={correlation:.8f}, p99 absolute difference="
        f"{p99_difference * 10_000:.4f} bp.",
        "",
        coverage[coverage["scope"].eq("all")].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "Frozen feature rules:",
        "",
        "- RFX1: at least half of the original extreme overshoot receivers have "
        "late-third taker flow opposite to the graph source direction; fade the "
        "full quality-screened receiver bucket.",
        "- RFX2: at least one receiver flips from early-third source-aligned flow "
        "to late-third opposing flow and at least two retain source-aligned flow; "
        "test the exhausted-versus-persistent within-event spread.",
        "",
        "The thresholds are structural zero/sign rules and were frozen without "
        "reading post-event returns. This is post-hoc offline discovery and any "
        "positive reveal remains natural-forward-only, not promotion evidence.",
        "",
        "No future price, candidate PnL, live, PaperLive, application, leverage, "
        "remote, or order state was read or changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v204_feature_audit(
    receiver_path: Path = FEATURE_PATH,
    feature_events_path: Path = FEATURE_EVENTS_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V204FeatureConfig = V204FeatureConfig(),
) -> dict[str, Path]:
    raw_receivers = pd.read_parquet(receiver_path)
    receivers = add_v204_quality_fields(raw_receivers, cfg)
    receivers, kline_validation = add_kline_validation(receivers, kline_root)
    feature_events = pd.read_parquet(feature_events_path)
    expected_tasks = build_extreme_overshoot_tasks(feature_events)
    candidates = build_v204_candidate_features(receivers)
    coverage = summarize_v204_coverage(candidates)
    checks = audit_v204_features(expected_tasks, receivers, candidates, coverage, cfg)
    root = ensure_dir(report_root)
    outputs = {
        "receivers": root / "receiver_features.parquet",
        "candidates": root / "candidate_feature_events.parquet",
        "coverage": root / "feature_coverage_summary.csv",
        "kline_validation": root / "kline_price_validation_by_symbol.csv",
        "checks": root / "data_quality_checks.csv",
        "findings": findings_path,
    }
    receivers.to_parquet(outputs["receivers"], index=False)
    candidates.to_parquet(outputs["candidates"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    kline_validation.to_csv(outputs["kline_validation"], index=False)
    checks.to_csv(outputs["checks"], index=False)
    _write_findings(checks, coverage, receivers, findings_path)
    return outputs


__all__ = [
    "CANDIDATES",
    "EVENT_WIDE",
    "STATE_SPREAD",
    "V204FeatureConfig",
    "add_kline_validation",
    "add_v204_quality_fields",
    "audit_v204_features",
    "build_v204_candidate_features",
    "summarize_v204_coverage",
    "write_v204_feature_audit",
]
