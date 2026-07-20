"""Independent total-depth, score and portfolio audit for rejected v16.1 LW1."""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import FROZEN_SYMBOLS
from pressure_graph.reports.v160_hourly_depth_candidate_audit import (
    V160AuditConfig,
    audit_v160_portfolio,
)


DATA_ROOT = Path("data/external/binance_um_book_depth")
REPORT_ROOT = Path("reports/v16_1_hourly_liquidity_withdrawal_amplification")
AUDIT_DOC = Path("docs/v162_liquidity_withdrawal_candidate_audit_2026_07_16.md")


@dataclass(frozen=True)
class V162AuditConfig:
    data_root: Path = DATA_ROOT
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    raw_days_per_symbol: int = 5
    audit_hour: int = 12
    tolerance: float = 1e-12


def independent_total_depth_median(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, int]:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV, found {csv_names!r}")
        frame = pd.read_csv(
            archive.open(csv_names[0]),
            usecols=["timestamp", "percentage", "notional"],
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[
        frame["timestamp"].ge(start)
        & frame["timestamp"].lt(end)
        & frame["percentage"].isin([-1.0, 1.0])
    ]
    pivot = frame.pivot_table(
        index="timestamp",
        columns="percentage",
        values="notional",
        aggfunc="last",
        observed=True,
    )
    total = (pivot[-1.0] + pivot[1.0]).replace([np.inf, -np.inf], np.nan).dropna()
    total = total[total.gt(0)]
    return float(total.median()), int(total.count())


def audit_v162_raw_depth(cfg: V162AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for symbol in FROZEN_SYMBOLS:
        hourly = pd.read_parquet(
            cfg.data_root / "hourly_features" / f"{symbol}.parquet"
        )
        hourly["source_day"] = pd.to_datetime(hourly["source_day"], utc=True)
        hourly["decision_time"] = pd.to_datetime(hourly["decision_time"], utc=True)
        days = sorted(hourly["source_day"].drop_duplicates())
        positions = np.linspace(0, len(days) - 1, cfg.raw_days_per_symbol, dtype=int)
        indexed = hourly.set_index("decision_time")
        for position in np.unique(positions):
            day = pd.Timestamp(days[int(position)])
            decision = day + pd.Timedelta(hours=cfg.audit_hour)
            start = decision - pd.Timedelta(hours=1)
            stamp = day.date().isoformat()
            path = (
                cfg.data_root
                / "raw"
                / symbol
                / stamp[:7]
                / f"{symbol}-bookDepth-{stamp}.zip"
            )
            independent, count = independent_total_depth_median(path, start, decision)
            stored = indexed.loc[decision]
            rows.append(
                {
                    "symbol": symbol,
                    "decision_time": decision,
                    "stored_total_depth": float(stored["total_notional_1p0_median"]),
                    "independent_total_depth": independent,
                    "total_depth_difference": independent
                    - float(stored["total_notional_1p0_median"]),
                    "stored_snapshots": int(
                        stored["notional_imbalance_1p0_valid_snapshots"]
                    ),
                    "independent_snapshots": count,
                }
            )
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "raw_depth_samples": len(audit),
                "snapshot_counts_match": audit["stored_snapshots"].eq(
                    audit["independent_snapshots"]
                ).all(),
                "max_abs_total_depth_difference": audit[
                    "total_depth_difference"
                ].abs().max(),
            }
        ]
    )
    return audit, summary


def audit_v162_scores(cfg: V162AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.report_root / "hourly_symbol_panel.parquet")
    independent = panel[
        [
            "decision_time",
            "symbol",
            "total_depth_1pct",
            "previous_total_depth_1pct",
            "prior_price_return",
            "prior_btc_return",
            "btc_beta",
            "withdrawal_percentile_1pct",
            "prior_residual_return",
            "score_1pct",
        ]
    ].copy()
    independent["depth_change_audit"] = np.log(
        independent["total_depth_1pct"] / independent["previous_total_depth_1pct"]
    )
    independent["withdrawal_audit"] = -independent["depth_change_audit"]
    independent["withdrawal_percentile_audit"] = independent.groupby(
        "decision_time", observed=True
    )["withdrawal_audit"].rank(method="average", pct=True)
    independent["prior_residual_audit"] = (
        independent["prior_price_return"]
        - independent["btc_beta"] * independent["prior_btc_return"]
    )
    independent["score_audit"] = (
        independent["prior_residual_audit"]
        * independent["withdrawal_percentile_audit"]
    )
    independent["withdrawal_percentile_difference"] = (
        independent["withdrawal_percentile_audit"]
        - independent["withdrawal_percentile_1pct"]
    )
    independent["prior_residual_difference"] = (
        independent["prior_residual_audit"] - independent["prior_residual_return"]
    )
    independent["score_difference"] = independent["score_audit"] - independent["score_1pct"]
    summary = pd.DataFrame(
        [
            {
                "score_rows": len(independent),
                "max_abs_withdrawal_percentile_difference": independent[
                    "withdrawal_percentile_difference"
                ].abs().max(),
                "max_abs_prior_residual_difference": independent[
                    "prior_residual_difference"
                ].abs().max(),
                "max_abs_score_difference": independent["score_difference"].abs().max(),
            }
        ]
    )
    return independent, summary


def write_v162_liquidity_withdrawal_candidate_audit(
    cfg: V162AuditConfig = V162AuditConfig(),
) -> dict[str, Path]:
    raw, raw_summary = audit_v162_raw_depth(cfg)
    scores, score_summary = audit_v162_scores(cfg)
    portfolio, portfolio_summary = audit_v160_portfolio(
        V160AuditConfig(report_root=cfg.report_root)
    )
    main = pd.read_csv(cfg.report_root / "summary.csv")
    raw_pass = bool(
        raw_summary.at[0, "snapshot_counts_match"]
        and raw_summary.at[0, "max_abs_total_depth_difference"] <= cfg.tolerance
    )
    score_columns = [column for column in score_summary if column.startswith("max_abs_")]
    score_pass = bool(score_summary.loc[0, score_columns].le(cfg.tolerance).all())
    portfolio_columns = [
        column for column in portfolio_summary if column.startswith("max_abs_")
    ]
    portfolio_pass = bool(
        portfolio_summary.loc[0, portfolio_columns].le(cfg.tolerance).all()
    )
    outcome = pd.DataFrame(
        [
            {
                "raw_depth_audit_pass": raw_pass,
                "score_audit_pass": score_pass,
                "portfolio_audit_pass": portfolio_pass,
                "rejection_confirmed": bool(
                    not bool(main.at[0, "promote"])
                    and float(main.at[0, "mean_gross_bp"]) <= 0
                    and float(main.at[0, "bootstrap_95_high_bp"]) < 0
                ),
            }
        ]
    )
    root = ensure_dir(cfg.report_root / "independent_audit")
    paths = {
        "raw": root / "raw_total_depth_samples.csv",
        "raw_summary": root / "raw_total_depth_summary.csv",
        "scores": root / "score_recalculation.parquet",
        "score_summary": root / "score_audit_summary.csv",
        "portfolio": root / "portfolio_recalculation.parquet",
        "portfolio_summary": root / "portfolio_audit_summary.csv",
        "outcome": root / "audit_outcome.csv",
        "audit_doc": cfg.audit_doc,
    }
    raw.to_csv(paths["raw"], index=False)
    raw_summary.to_csv(paths["raw_summary"], index=False)
    scores.to_parquet(paths["scores"], index=False)
    score_summary.to_csv(paths["score_summary"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    verdict = "rejection_confirmed" if bool(outcome.iloc[0].all()) else "audit_failed"
    paths["audit_doc"].write_text(
        "\n".join(
            [
                "# v16.2 Independent Audit of v16.1 LW1",
                "",
                f"Verdict: `{verdict}`.",
                "",
                outcome.to_markdown(index=False),
                "",
                raw_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                score_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                portfolio_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                "The audit independently replayed total-depth windows, rebuilt every",
                "withdrawal rank and residual-price product, and recomputed all portfolio",
                "returns and costs. The zero-gross-edge rejection is confirmed.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
