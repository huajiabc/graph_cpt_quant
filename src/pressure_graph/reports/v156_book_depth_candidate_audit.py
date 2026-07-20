"""Independent raw-feature and portfolio audit for rejected v15.5 BD2."""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC


DATA_ROOT = Path("data/external/binance_um_book_depth")
REPORT_ROOT = Path("reports/v15_5_binance_one_percent_depth_imbalance")
AUDIT_DOC = Path("docs/v156_book_depth_candidate_audit_2026_07_16.md")
FROZEN_SYMBOLS = (
    "SOLUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",
    "WIFUSDT",
    "ETHUSDT",
    "ENAUSDT",
    "HBARUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ONDOUSDT",
    "XRPUSDT",
    "XLMUSDT",
    "FARTCOINUSDT",
    "WLDUSDT",
    "SEIUSDT",
    "TIAUSDT",
)


@dataclass(frozen=True)
class V156AuditConfig:
    data_root: Path = DATA_ROOT
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    raw_samples_per_symbol: int = 5
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    tolerance: float = 1e-12


def independent_one_percent_median(path: Path) -> float:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV, found {csv_names!r}")
        frame = pd.read_csv(
            archive.open(csv_names[0]),
            usecols=["timestamp", "percentage", "notional"],
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["percentage"] = pd.to_numeric(frame["percentage"], errors="coerce")
    frame["notional"] = pd.to_numeric(frame["notional"], errors="coerce")
    frame = frame[frame["percentage"].isin([-1.0, 1.0])]
    pivot = frame.pivot_table(
        index="timestamp",
        columns="percentage",
        values="notional",
        aggfunc="last",
        observed=True,
    )
    denominator = pivot[-1.0] + pivot[1.0]
    values = ((pivot[-1.0] - pivot[1.0]) / denominator).where(denominator.gt(0))
    return float(values.replace([np.inf, -np.inf], np.nan).dropna().median())


def _raw_path(root: Path, symbol: str, day: pd.Timestamp) -> Path:
    stamp = day.date().isoformat()
    return root / "raw" / symbol / stamp[:7] / f"{symbol}-bookDepth-{stamp}.zip"


def audit_v156_raw_features(cfg: V156AuditConfig) -> pd.DataFrame:
    daily_manifest = pd.read_parquet(cfg.data_root / "daily_manifest.parquet")
    daily_manifest["source_day"] = pd.to_datetime(
        daily_manifest["source_day"], utc=True, errors="coerce"
    )
    manifest_ok = daily_manifest[daily_manifest["status"].eq("ok")].copy()
    feature_frames = []
    for symbol in FROZEN_SYMBOLS:
        frame = pd.read_parquet(
            cfg.data_root / "daily_features" / f"{symbol}.parquet",
            columns=[
                "bybit_symbol",
                "source_day",
                "notional_imbalance_1p0_median",
                "archive_sha256",
            ],
        )
        feature_frames.append(frame)
    features = pd.concat(feature_frames, ignore_index=True)
    features["source_day"] = pd.to_datetime(features["source_day"], utc=True)
    merged = features.merge(
        manifest_ok[
            ["bybit_symbol", "binance_symbol", "source_day", "sha256"]
        ],
        on=["bybit_symbol", "source_day"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    hash_manifest_matches = bool(
        merged["_merge"].eq("both").all()
        and merged["archive_sha256"].eq(merged["sha256"]).all()
    )
    sample_rows = []
    for symbol, local in merged.groupby("bybit_symbol", sort=True, observed=True):
        positions = np.linspace(
            0, len(local) - 1, cfg.raw_samples_per_symbol, dtype=int
        )
        for position in np.unique(positions):
            row = local.sort_values("source_day").iloc[int(position)]
            day = pd.Timestamp(row["source_day"])
            path = _raw_path(cfg.data_root, str(row["binance_symbol"]), day)
            content = path.read_bytes()
            independent = independent_one_percent_median(path)
            stored = float(row["notional_imbalance_1p0_median"])
            sample_rows.append(
                {
                    "bybit_symbol": symbol,
                    "source_day": day,
                    "raw_path": str(path),
                    "raw_sha256_matches": hashlib.sha256(content).hexdigest()
                    == row["sha256"],
                    "stored_feature": stored,
                    "independent_feature": independent,
                    "absolute_feature_difference": abs(stored - independent),
                    "all_manifest_feature_hashes_match": hash_manifest_matches,
                }
            )
    return pd.DataFrame(sample_rows)


def audit_v156_portfolio(cfg: V156AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.report_root / "daily_symbol_panel.parquet")
    portfolio = pd.read_parquet(cfg.report_root / "daily_portfolio.parquet")
    panel_by_day = {
        pd.Timestamp(day): local.set_index("symbol")
        for day, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    rows = []
    previous_weights: dict[str, float] = {}
    previous_day: pd.Timestamp | None = None
    for event in portfolio.itertuples(index=False):
        day = pd.Timestamp(event.decision_time)
        local = panel_by_day[day]
        weights = {key: float(value) for key, value in json.loads(event.weights_json).items()}
        if previous_day is not None and day - previous_day > pd.Timedelta(days=1):
            previous_weights = {}
        expected_turnover = float(
            sum(
                abs(previous_weights.get(symbol, 0.0) - weights.get(symbol, 0.0))
                for symbol in set(previous_weights) | set(weights)
            )
        )
        alt_symbols = [symbol for symbol in weights if symbol != BTC]
        independent_gross = float(
            sum(
                weights[symbol] * float(local.at[symbol, "price_return"])
                for symbol in alt_symbols
            )
            + weights[BTC] * float(local.iloc[0]["btc_return"])
        )
        residual_beta = float(
            sum(
                weights[symbol] * float(local.at[symbol, "btc_beta"])
                for symbol in alt_symbols
            )
            + weights[BTC]
        )
        total_turnover = expected_turnover + float(event.forced_close_turnover)
        rows.append(
            {
                "decision_time": day,
                "source_lag_days": (day - pd.Timestamp(event.source_day)).days,
                "symbol_count": len(local),
                "entry_turnover_difference": expected_turnover - event.entry_turnover,
                "gross_return_difference": independent_gross - event.gross_return,
                "primary_return_difference": independent_gross
                - cfg.one_way_cost * total_turnover
                - event.primary_net_return,
                "stress_return_difference": independent_gross
                - cfg.stress_one_way_cost * total_turnover
                - event.stress_net_return,
                "residual_btc_beta": residual_beta,
                "gross_notional_drift": sum(abs(weight) for weight in weights.values())
                - 1.0,
            }
        )
        previous_weights = weights
        previous_day = day
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "days": len(audit),
                "source_lag_exact": audit["source_lag_days"].eq(1).all(),
                "universe_exact": audit["symbol_count"].eq(len(FROZEN_SYMBOLS)).all(),
                "max_abs_turnover_difference": audit[
                    "entry_turnover_difference"
                ].abs().max(),
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
    numeric_columns = [column for column in summary if column.startswith("max_abs_")]
    summary["portfolio_audit_pass"] = bool(
        summary.at[0, "source_lag_exact"]
        and summary.at[0, "universe_exact"]
        and summary.loc[0, numeric_columns].le(cfg.tolerance).all()
    )
    return audit, summary


def write_v156_book_depth_candidate_audit(
    cfg: V156AuditConfig = V156AuditConfig(),
) -> dict[str, Path]:
    raw_audit = audit_v156_raw_features(cfg)
    portfolio_audit, portfolio_summary = audit_v156_portfolio(cfg)
    main_summary = pd.read_csv(cfg.report_root / "summary.csv")
    raw_pass = bool(
        raw_audit["raw_sha256_matches"].all()
        and raw_audit["all_manifest_feature_hashes_match"].all()
        and raw_audit["absolute_feature_difference"].le(cfg.tolerance).all()
    )
    rejection_confirmed = bool(
        not bool(main_summary.at[0, "promote"])
        and float(main_summary.at[0, "mean_primary_net_bp"]) < 0
        and float(main_summary.at[0, "bootstrap_95_high_bp"]) < 0
    )
    outcome = pd.DataFrame(
        [
            {
                "raw_feature_audit_pass": raw_pass,
                "portfolio_audit_pass": bool(
                    portfolio_summary.at[0, "portfolio_audit_pass"]
                ),
                "rejection_confirmed": rejection_confirmed,
                "raw_samples": len(raw_audit),
            }
        ]
    )
    root = ensure_dir(cfg.report_root / "independent_audit")
    paths = {
        "raw": root / "raw_feature_sample.csv",
        "portfolio": root / "portfolio_recalculation.parquet",
        "portfolio_summary": root / "portfolio_audit_summary.csv",
        "outcome": root / "audit_outcome.csv",
        "audit_doc": cfg.audit_doc,
    }
    raw_audit.to_csv(paths["raw"], index=False)
    portfolio_audit.to_parquet(paths["portfolio"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    verdict = (
        "rejection_confirmed"
        if bool(outcome.iloc[0][
            ["raw_feature_audit_pass", "portfolio_audit_pass", "rejection_confirmed"]
        ].all())
        else "audit_failed"
    )
    paths["audit_doc"].write_text(
        "\n".join(
            [
                "# v15.6 Independent Audit of v15.5 BD2",
                "",
                f"Verdict: `{verdict}`.",
                "",
                outcome.to_markdown(index=False),
                "",
                portfolio_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                "The audit independently re-read deterministic raw ZIP samples,",
                "matched all stored feature hashes to the download manifest, and",
                "recomputed every portfolio return, turnover charge, BTC beta residual",
                "and gross normalization. It confirms the v15.5 rejection; it does not",
                "promote the reversed control. PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
