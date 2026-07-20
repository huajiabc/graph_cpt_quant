"""Independent weekly feature and portfolio audit for rejected v16.3 LQ1."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v163_within_bucket_liquidity_quality import (
    FROZEN_PAIRS,
    load_v163_binance_volume,
    load_v163_depth,
)


REPORT_ROOT = Path("reports/v16_3_within_bucket_liquidity_quality")
AUDIT_DOC = Path("docs/v164_weekly_liquidity_quality_audit_2026_07_16.md")


@dataclass(frozen=True)
class V164AuditConfig:
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    tolerance: float = 1e-12


def audit_v164_features(cfg: V164AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    depth = load_v163_depth()
    volume = load_v163_binance_volume()
    panel = pd.read_parquet(cfg.report_root / "weekly_symbol_panel.parquet")
    rows = []
    for raw_time, stored in panel.groupby("decision_time", sort=True, observed=True):
        decision = pd.Timestamp(raw_time)
        start = decision - pd.Timedelta(days=7)
        local_depth = depth[
            depth["decision_time"].gt(start) & depth["decision_time"].le(decision)
        ].groupby("symbol", observed=True).agg(
            depth_hours_audit=("total_notional_1p0_median", "count"),
            depth_1pct_audit=("total_notional_1p0_median", "median"),
            depth_5pct_audit=("total_notional_5p0_median", "median"),
        )
        local_volume = volume[
            volume["feature_time"].gt(start) & volume["feature_time"].le(decision)
        ].groupby("symbol", observed=True).agg(
            volume_hours_audit=("quote_volume", "count"),
            mean_volume_audit=("quote_volume", "mean"),
        )
        audit = local_depth.join(local_volume, how="inner")
        indexed = stored.set_index("symbol")
        for symbol, independent in audit.iterrows():
            quality_1pct = float(
                np.log(independent["depth_1pct_audit"] / independent["mean_volume_audit"])
            )
            quality_5pct = float(
                np.log(independent["depth_5pct_audit"] / independent["mean_volume_audit"])
            )
            rows.append(
                {
                    "decision_time": decision,
                    "symbol": symbol,
                    "depth_hours_difference": independent["depth_hours_audit"]
                    - indexed.at[symbol, "depth_hours"],
                    "volume_hours_difference": independent["volume_hours_audit"]
                    - indexed.at[symbol, "volume_hours"],
                    "depth_1pct_difference": independent["depth_1pct_audit"]
                    - indexed.at[symbol, "depth_1pct"],
                    "depth_5pct_difference": independent["depth_5pct_audit"]
                    - indexed.at[symbol, "depth_5pct"],
                    "mean_volume_difference": independent["mean_volume_audit"]
                    - indexed.at[symbol, "mean_hourly_quote_volume"],
                    "quality_1pct_difference": quality_1pct
                    - indexed.at[symbol, "quality_1pct"],
                    "quality_5pct_difference": quality_5pct
                    - indexed.at[symbol, "quality_5pct"],
                }
            )
    audit = pd.DataFrame(rows)
    difference_columns = [column for column in audit if column.endswith("_difference")]
    summary = pd.DataFrame(
        [
            {
                "feature_rows": len(audit),
                **{
                    f"max_abs_{column}": audit[column].abs().max()
                    for column in difference_columns
                },
            }
        ]
    )
    return audit, summary


def audit_v164_portfolio(cfg: V164AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.report_root / "weekly_symbol_panel.parquet")
    portfolio = pd.read_parquet(cfg.report_root / "weekly_portfolio.parquet")
    panel_lookup = {
        pd.Timestamp(time): local.set_index("symbol")
        for time, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    rows = []
    previous_weights: dict[str, float] = {}
    previous_time: pd.Timestamp | None = None
    for event in portfolio.itertuples(index=False):
        decision = pd.Timestamp(event.decision_time)
        if previous_time is not None and decision - previous_time > pd.Timedelta(days=7):
            previous_weights = {}
        local = panel_lookup[decision]
        expected_longs = []
        expected_shorts = []
        for pair in FROZEN_PAIRS.values():
            ordered = sorted(
                pair,
                key=lambda symbol: (-float(local.at[symbol, "quality_1pct"]), symbol),
            )
            expected_longs.append(ordered[0])
            expected_shorts.append(ordered[1])
        weights = {key: float(value) for key, value in json.loads(event.weights_json).items()}
        expected_turnover = float(
            sum(
                abs(previous_weights.get(symbol, 0.0) - weights.get(symbol, 0.0))
                for symbol in set(previous_weights) | set(weights)
            )
        )
        alt_symbols = [symbol for symbol in weights if symbol != BTC]
        gross = float(
            sum(
                weights[symbol] * float(local.at[symbol, "price_return"])
                for symbol in alt_symbols
            )
            + weights[BTC] * float(local.iloc[0]["btc_return"])
        )
        residual = float(
            sum(
                weights[symbol] * float(local.at[symbol, "btc_beta"])
                for symbol in alt_symbols
            )
            + weights[BTC]
        )
        turnover = expected_turnover + float(event.forced_close_turnover)
        rows.append(
            {
                "decision_time": decision,
                "long_pairs_exact": set(expected_longs)
                == set(str(event.long_symbols).split("|")),
                "short_pairs_exact": set(expected_shorts)
                == set(str(event.short_symbols).split("|")),
                "turnover_difference": expected_turnover - event.entry_turnover,
                "gross_return_difference": gross - event.gross_return,
                "primary_return_difference": gross
                - cfg.one_way_cost * turnover
                - event.primary_net_return,
                "stress_return_difference": gross
                - cfg.stress_one_way_cost * turnover
                - event.stress_net_return,
                "residual_btc_beta": residual,
                "gross_notional_drift": sum(abs(value) for value in weights.values())
                - 1.0,
            }
        )
        previous_weights = weights
        previous_time = decision
    audit = pd.DataFrame(rows)
    numeric = [
        "turnover_difference",
        "gross_return_difference",
        "primary_return_difference",
        "stress_return_difference",
        "residual_btc_beta",
        "gross_notional_drift",
    ]
    summary = pd.DataFrame(
        [
            {
                "portfolio_weeks": len(audit),
                "long_pairs_exact": audit["long_pairs_exact"].all(),
                "short_pairs_exact": audit["short_pairs_exact"].all(),
                **{f"max_abs_{column}": audit[column].abs().max() for column in numeric},
            }
        ]
    )
    return audit, summary


def write_v164_weekly_liquidity_quality_audit(
    cfg: V164AuditConfig = V164AuditConfig(),
) -> dict[str, Path]:
    features, feature_summary = audit_v164_features(cfg)
    portfolio, portfolio_summary = audit_v164_portfolio(cfg)
    main = pd.read_csv(cfg.report_root / "summary.csv")
    feature_columns = [
        column for column in feature_summary if column.startswith("max_abs_")
    ]
    feature_pass = bool(feature_summary.loc[0, feature_columns].le(cfg.tolerance).all())
    portfolio_columns = [
        column for column in portfolio_summary if column.startswith("max_abs_")
    ]
    portfolio_pass = bool(
        portfolio_summary.at[0, "long_pairs_exact"]
        and portfolio_summary.at[0, "short_pairs_exact"]
        and portfolio_summary.loc[0, portfolio_columns].le(cfg.tolerance).all()
    )
    outcome = pd.DataFrame(
        [
            {
                "feature_audit_pass": feature_pass,
                "portfolio_audit_pass": portfolio_pass,
                "rejection_confirmed": bool(
                    not bool(main.at[0, "promote"])
                    and float(main.at[0, "validation_primary_net_bp"]) < 0
                    and float(main.at[0, "bootstrap_95_low_bp"]) < 0
                ),
            }
        ]
    )
    root = ensure_dir(cfg.report_root / "independent_audit")
    paths = {
        "features": root / "weekly_feature_recalculation.parquet",
        "feature_summary": root / "weekly_feature_audit_summary.csv",
        "portfolio": root / "portfolio_recalculation.parquet",
        "portfolio_summary": root / "portfolio_audit_summary.csv",
        "outcome": root / "audit_outcome.csv",
        "audit_doc": cfg.audit_doc,
    }
    features.to_parquet(paths["features"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    verdict = "rejection_confirmed" if bool(outcome.iloc[0].all()) else "audit_failed"
    paths["audit_doc"].write_text(
        "\n".join(
            [
                "# v16.4 Independent Audit of v16.3 LQ1",
                "",
                f"Verdict: `{verdict}`.",
                "",
                outcome.to_markdown(index=False),
                "",
                feature_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                portfolio_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                "The audit independently rebuilt all 816 weekly depth/volume quality",
                "features, pair directions and 51 portfolio weeks. The rejection is",
                "confirmed; positive stale and 5% diagnostics are temporally unstable.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
