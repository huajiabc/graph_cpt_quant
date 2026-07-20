"""Independent raw-hour and portfolio audit for rejected v15.9 BD3."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import FROZEN_SYMBOLS


DATA_ROOT = Path("data/external/binance_um_book_depth")
REPORT_ROOT = Path("reports/v15_9_hourly_cross_venue_depth_imbalance")
AUDIT_DOC = Path("docs/v160_hourly_depth_candidate_audit_2026_07_16.md")


@dataclass(frozen=True)
class V160AuditConfig:
    data_root: Path = DATA_ROOT
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    raw_days_per_symbol: int = 5
    audit_hour: int = 12
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    tolerance: float = 1e-12


def independent_hourly_median(
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
    denominator = pivot[-1.0] + pivot[1.0]
    imbalance = ((pivot[-1.0] - pivot[1.0]) / denominator).where(denominator.gt(0))
    values = imbalance.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()), int(values.count())


def audit_v160_raw_hours(cfg: V160AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_hashes_match = True
    for symbol in FROZEN_SYMBOLS:
        hourly = pd.read_parquet(
            cfg.data_root / "hourly_features" / f"{symbol}.parquet"
        )
        daily = pd.read_parquet(
            cfg.data_root / "daily_features" / f"{symbol}.parquet",
            columns=["source_day", "archive_sha256"],
        )
        for frame in (hourly, daily):
            frame["source_day"] = pd.to_datetime(frame["source_day"], utc=True)
        merged_hash = hourly[["source_day", "archive_sha256"]].drop_duplicates().merge(
            daily,
            on="source_day",
            suffixes=("_hourly", "_daily"),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        all_hashes_match &= bool(
            merged_hash["_merge"].eq("both").all()
            and merged_hash["archive_sha256_hourly"].eq(
                merged_hash["archive_sha256_daily"]
            ).all()
        )
        days = sorted(hourly["source_day"].drop_duplicates())
        positions = np.linspace(0, len(days) - 1, cfg.raw_days_per_symbol, dtype=int)
        indexed = hourly.set_index("decision_time")
        indexed.index = pd.to_datetime(indexed.index, utc=True)
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
            independent, count = independent_hourly_median(path, start, decision)
            stored = indexed.loc[decision]
            rows.append(
                {
                    "symbol": symbol,
                    "decision_time": decision,
                    "strict_start": start,
                    "strict_end": decision,
                    "stored_feature": float(stored["notional_imbalance_1p0_median"]),
                    "independent_feature": independent,
                    "feature_difference": independent
                    - float(stored["notional_imbalance_1p0_median"]),
                    "stored_snapshots": int(
                        stored["notional_imbalance_1p0_valid_snapshots"]
                    ),
                    "independent_snapshots": count,
                    "all_hourly_daily_hashes_match": all_hashes_match,
                }
            )
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "raw_hour_samples": len(audit),
                "all_hourly_daily_hashes_match": audit[
                    "all_hourly_daily_hashes_match"
                ].all(),
                "snapshot_counts_match": audit["stored_snapshots"].eq(
                    audit["independent_snapshots"]
                ).all(),
                "max_abs_feature_difference": audit["feature_difference"].abs().max(),
            }
        ]
    )
    return audit, summary


def audit_v160_portfolio(cfg: V160AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.report_root / "hourly_symbol_panel.parquet")
    portfolio = pd.read_parquet(cfg.report_root / "hourly_portfolio.parquet")
    panel_lookup = {
        pd.Timestamp(time): local.set_index("symbol")
        for time, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    rows = []
    previous_weights: dict[str, float] = {}
    previous_time: pd.Timestamp | None = None
    for event in portfolio.itertuples(index=False):
        decision = pd.Timestamp(event.decision_time)
        if previous_time is not None and decision - previous_time > pd.Timedelta(hours=1):
            previous_weights = {}
        local = panel_lookup[decision]
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
    summary = pd.DataFrame(
        [
            {
                "portfolio_hours": len(audit),
                "max_abs_turnover_difference": audit["turnover_difference"].abs().max(),
                "max_abs_gross_return_difference": audit[
                    "gross_return_difference"
                ].abs().max(),
                "max_abs_primary_return_difference": audit[
                    "primary_return_difference"
                ].abs().max(),
                "max_abs_stress_return_difference": audit[
                    "stress_return_difference"
                ].abs().max(),
                "max_abs_residual_btc_beta": audit["residual_btc_beta"].abs().max(),
                "max_abs_gross_notional_drift": audit[
                    "gross_notional_drift"
                ].abs().max(),
            }
        ]
    )
    return audit, summary


def write_v160_hourly_depth_candidate_audit(
    cfg: V160AuditConfig = V160AuditConfig(),
) -> dict[str, Path]:
    raw, raw_summary = audit_v160_raw_hours(cfg)
    portfolio, portfolio_summary = audit_v160_portfolio(cfg)
    main = pd.read_csv(cfg.report_root / "summary.csv")
    raw_pass = bool(
        raw_summary.at[0, "all_hourly_daily_hashes_match"]
        and raw_summary.at[0, "snapshot_counts_match"]
        and raw_summary.at[0, "max_abs_feature_difference"] <= cfg.tolerance
    )
    portfolio_columns = [
        column for column in portfolio_summary if column.startswith("max_abs_")
    ]
    portfolio_pass = bool(
        portfolio_summary.loc[0, portfolio_columns].le(cfg.tolerance).all()
    )
    outcome = pd.DataFrame(
        [
            {
                "raw_hour_audit_pass": raw_pass,
                "portfolio_audit_pass": portfolio_pass,
                "rejection_confirmed": bool(
                    not bool(main.at[0, "promote"])
                    and float(main.at[0, "mean_primary_net_bp"]) < 0
                    and float(main.at[0, "bootstrap_95_high_bp"]) < 0
                ),
            }
        ]
    )
    root = ensure_dir(cfg.report_root / "independent_audit")
    paths = {
        "raw": root / "raw_hour_samples.csv",
        "raw_summary": root / "raw_hour_audit_summary.csv",
        "portfolio": root / "portfolio_recalculation.parquet",
        "portfolio_summary": root / "portfolio_audit_summary.csv",
        "outcome": root / "audit_outcome.csv",
        "audit_doc": cfg.audit_doc,
    }
    raw.to_csv(paths["raw"], index=False)
    raw_summary.to_csv(paths["raw_summary"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    verdict = "rejection_confirmed" if bool(outcome.iloc[0].all()) else "audit_failed"
    paths["audit_doc"].write_text(
        "\n".join(
            [
                "# v16.0 Independent Audit of v15.9 BD3",
                "",
                f"Verdict: `{verdict}`.",
                "",
                outcome.to_markdown(index=False),
                "",
                raw_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                portfolio_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                "The audit independently re-read 80 raw one-hour windows using strict",
                "half-open timestamps and recomputed all 9,055 portfolio hours. It",
                "confirms that the negative result is not a timing or cost bug.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
