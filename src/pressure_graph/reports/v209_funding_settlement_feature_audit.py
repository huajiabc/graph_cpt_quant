"""Feature-only audit for synchronized funding-settlement receiver buckets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _period
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    load_v195_funding,
)


FUNDING_ROOT = Path("data/external/binance_um_carry/funding")
REPORT_ROOT = Path("reports/v20_9_funding_settlement_feature_audit")
FINDINGS_PATH = Path("docs/v209_funding_settlement_feature_audit_2026_07_17.md")
ALL_NEGATIVE = "FSE1_ALL_NEGATIVE_SETTLEMENT_REBOUND"
NEW_NEGATIVE = "FSE2_NEW_NEGATIVE_ONSET_REBOUND"
CANDIDATES = (ALL_NEGATIVE, NEW_NEGATIVE)


@dataclass(frozen=True)
class V209FeatureConfig:
    history_start: pd.Timestamp = pd.Timestamp("2025-07-01", tz="UTC")
    first_candidate_time: pd.Timestamp = pd.Timestamp("2025-08-01", tz="UTC")
    last_candidate_time: pd.Timestamp = pd.Timestamp(
        "2026-06-04 16:00", tz="UTC"
    )
    expected_symbols: int = 45
    minimum_synchronized_coverage: int = 45
    minimum_bucket_size: int = 5
    entry_delay_bars: int = 1
    holding_bars: int = 4
    minimum_candidate_events: int = 250
    minimum_period_events: int = 100


def build_v209_funding_features(
    funding: pd.DataFrame,
    symbols: set[str],
    cfg: V209FeatureConfig = V209FeatureConfig(),
) -> pd.DataFrame:
    local = funding[
        funding["symbol"].isin(symbols)
        & funding["funding_time"].between(
            cfg.history_start, cfg.last_candidate_time, inclusive="both"
        )
    ].copy()
    local = local.sort_values(["symbol", "funding_time"]).reset_index(drop=True)
    local["prior_funding_time"] = local.groupby("symbol")["funding_time"].shift()
    local["prior_funding_rate"] = local.groupby("symbol")[
        "funding_rate_settled"
    ].shift()
    local["funding_interval_hours"] = (
        local["funding_time"] - local["prior_funding_time"]
    ).dt.total_seconds().div(3600)
    local["negative_funding"] = local["funding_rate_settled"].lt(0.0)
    local["new_negative_onset"] = (
        local["negative_funding"] & local["prior_funding_rate"].ge(0.0)
    )
    coverage = local.groupby("funding_time")["symbol"].transform("size")
    local["synchronized_coverage"] = coverage
    local["synchronized_settlement"] = (
        coverage.ge(cfg.minimum_synchronized_coverage)
        & local["funding_time"].dt.hour.isin([0, 8, 16])
        & local["funding_time"].dt.minute.eq(0)
    )
    return local


def _symbols(local: pd.DataFrame, mask: pd.Series) -> str:
    return "|".join(sorted(local.loc[mask, "symbol"].astype(str)))


def build_v209_candidate_features(
    funding_features: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V209FeatureConfig = V209FeatureConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    synchronized = funding_features[
        funding_features["synchronized_settlement"]
        & funding_features["funding_time"].ge(cfg.first_candidate_time)
    ]
    for settlement_time, local in synchronized.groupby("funding_time", sort=True):
        settlement_time = pd.Timestamp(settlement_time)
        entry_time = settlement_time + pd.Timedelta(
            minutes=15 * cfg.entry_delay_bars
        )
        exit_time = entry_time + pd.Timedelta(minutes=15 * cfg.holding_bars)
        if entry_time not in close.index or exit_time not in close.index:
            continue
        base = {
            "settlement_time": settlement_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "period": _period(settlement_time),
            "entry_day": entry_time.date(),
            "entry_month": entry_time.strftime("%Y-%m"),
            "synchronized_coverage": len(local),
            "cross_section_mean_funding": float(
                local["funding_rate_settled"].mean()
            ),
            "cross_section_median_funding": float(
                local["funding_rate_settled"].median()
            ),
            "negative_count": int(local["negative_funding"].sum()),
            "new_negative_count": int(local["new_negative_onset"].sum()),
        }
        rules = (
            (ALL_NEGATIVE, local["negative_funding"]),
            (NEW_NEGATIVE, local["new_negative_onset"]),
        )
        for candidate, mask in rules:
            selected = local[mask]
            if len(selected) < cfg.minimum_bucket_size:
                continue
            symbols = selected["symbol"].astype(str).tolist()
            endpoints_available = bool(
                close.reindex(index=[entry_time, exit_time], columns=[BTC, *symbols])
                .notna()
                .all()
                .all()
            )
            rows.append(
                {
                    **base,
                    "candidate": candidate,
                    "selection_count": len(selected),
                    "selection_symbols": _symbols(local, mask),
                    "mean_selected_funding": float(
                        selected["funding_rate_settled"].mean()
                    ),
                    "median_selected_funding": float(
                        selected["funding_rate_settled"].median()
                    ),
                    "minimum_selected_funding": float(
                        selected["funding_rate_settled"].min()
                    ),
                    "maximum_selected_funding": float(
                        selected["funding_rate_settled"].max()
                    ),
                    "mean_prior_funding": float(
                        selected["prior_funding_rate"].mean()
                    ),
                    "price_endpoints_available": endpoints_available,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["settlement_time", "candidate"]
    ).reset_index(drop=True)


def summarize_v209_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_selection_count": (
                        float(sample["selection_count"].mean())
                        if len(sample)
                        else math.nan
                    ),
                    "median_selection_count": (
                        float(sample["selection_count"].median())
                        if len(sample)
                        else math.nan
                    ),
                    "mean_selected_funding_bp": (
                        float(sample["mean_selected_funding"].mean() * 10_000)
                        if len(sample)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_v209_features(
    funding_features: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    symbols: set[str],
    cfg: V209FeatureConfig = V209FeatureConfig(),
) -> pd.DataFrame:
    synchronized = funding_features[
        funding_features["synchronized_settlement"]
    ]
    synchronized_counts = synchronized.groupby("funding_time")["symbol"].nunique()
    all_coverage = coverage[coverage["scope"].eq("all")]
    period_coverage = coverage[coverage["scope"].ne("all")]
    checks = {
        "research_universe_45_symbols": len(symbols) == cfg.expected_symbols,
        "funding_keys_unique": not funding_features.duplicated(
            ["symbol", "funding_time"]
        ).any(),
        "synchronized_events_1017": synchronized["funding_time"].nunique() == 1017,
        "synchronized_cross_section_exact_45": bool(
            synchronized_counts.eq(cfg.expected_symbols).all()
        ),
        "synchronized_hours_only_0_8_16": set(
            synchronized["funding_time"].dt.hour.unique()
        )
        == {0, 8, 16},
        "prior_funding_strictly_before_current": bool(
            funding_features.dropna(subset=["prior_funding_time"])
            .eval("prior_funding_time < funding_time")
            .all()
        ),
        "candidate_signal_precedes_entry_by_15m": bool(
            candidates["entry_time"]
            .sub(candidates["settlement_time"])
            .eq(pd.Timedelta(minutes=15))
            .all()
        ),
        "holding_window_exactly_60m": bool(
            candidates["exit_time"]
            .sub(candidates["entry_time"])
            .eq(pd.Timedelta(minutes=60))
            .all()
        ),
        "all_selected_rates_negative": bool(
            candidates["maximum_selected_funding"].lt(0.0).all()
        ),
        "minimum_bucket_size_5": bool(
            candidates["selection_count"].ge(cfg.minimum_bucket_size).all()
        ),
        "all_price_endpoints_available": bool(
            candidates["price_endpoints_available"].all()
        ),
        "candidate_all_sample_coverage": bool(
            all_coverage["events"].ge(cfg.minimum_candidate_events).all()
        ),
        "candidate_each_period_coverage": bool(
            period_coverage["events"].ge(cfg.minimum_period_events).all()
        ),
        "candidate_keys_unique": not candidates.duplicated(
            ["candidate", "settlement_time"]
        ).any(),
        "no_post_event_return_columns": not any(
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
    path: Path,
) -> None:
    verdict = (
        "feature_viable_freeze_two_settlement_candidates"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    text = [
        "# v20.9 Funding-Settlement Feature Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        coverage[coverage["scope"].eq("all")].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "Frozen feature rules:",
        "",
        "- FSE1: at a synchronized 00/08/16 UTC Binance USD-M funding settlement, "
        "select every alt with a just-settled negative funding rate; require at "
        "least five names.",
        "- FSE2: select only alts whose just-settled rate is negative while their "
        "immediately prior settled rate was non-negative; require at least five names.",
        "",
        "The signal is observed at settlement. The frozen entry is one full "
        "15-minute bar later and the primary holding window is 60 minutes. No "
        "post-event return was calculated during this feature audit.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state was "
        "read or changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v209_feature_audit(
    funding_root: Path = FUNDING_ROOT,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V209FeatureConfig = V209FeatureConfig(),
) -> dict[str, Path]:
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    symbols = set(close.columns) - {BTC}
    funding = load_v195_funding(symbols, (funding_root,))
    features = build_v209_funding_features(funding, symbols, cfg)
    candidates = build_v209_candidate_features(features, close, cfg)
    coverage = summarize_v209_coverage(candidates)
    checks = audit_v209_features(features, candidates, coverage, symbols, cfg)
    root = ensure_dir(report_root)
    outputs = {
        "funding_features": root / "funding_symbol_features.parquet",
        "candidates": root / "candidate_feature_events.parquet",
        "coverage": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "findings": findings_path,
    }
    features.to_parquet(outputs["funding_features"], index=False)
    candidates.to_parquet(outputs["candidates"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    checks.to_csv(outputs["checks"], index=False)
    _write_findings(checks, coverage, findings_path)
    return outputs


__all__ = [
    "ALL_NEGATIVE",
    "CANDIDATES",
    "NEW_NEGATIVE",
    "V209FeatureConfig",
    "audit_v209_features",
    "build_v209_candidate_features",
    "build_v209_funding_features",
    "summarize_v209_coverage",
    "write_v209_feature_audit",
]
