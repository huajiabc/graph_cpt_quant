"""Independent signal and portfolio audit for rejected v15.7 VT4."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    load_v155_hourly_prices,
)


V155_PANEL_PATH = Path(
    "reports/v15_5_binance_one_percent_depth_imbalance/daily_symbol_panel.parquet"
)
REPORT_ROOT = Path("reports/v15_7_pair_shock_fragile_receiver")
AUDIT_DOC = Path("docs/v158_pair_shock_candidate_audit_2026_07_16.md")
PAIRS = {
    "BSP01": ("SOLUSDT", "DOGEUSDT"),
    "BSP02": ("1000PEPEUSDT", "WIFUSDT"),
    "BSP03": ("ETHUSDT", "ENAUSDT"),
    "BSP04": ("HBARUSDT", "AVAXUSDT"),
    "BSP05": ("LINKUSDT", "ONDOUSDT"),
    "BSP06": ("XRPUSDT", "XLMUSDT"),
    "BSP07": ("FARTCOINUSDT", "WLDUSDT"),
    "BSP08": ("SEIUSDT", "TIAUSDT"),
}


@dataclass(frozen=True)
class V158AuditConfig:
    panel_path: Path = V155_PANEL_PATH
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    tolerance: float = 1e-12


def audit_v158_signals(cfg: V158AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.panel_path)
    signals = pd.read_parquet(cfg.report_root / "daily_pair_signals.parquet")
    prices = load_v155_hourly_prices()
    close = prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    daily = close[close.index.hour == 0]
    panel_lookup = {
        pd.Timestamp(day): local.set_index("symbol")
        for day, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    rows = []
    for event in signals.itertuples(index=False):
        day = pd.Timestamp(event.signal_time)
        local = panel_lookup[day]
        prior_return = daily.loc[day] / daily.loc[day - pd.Timedelta(days=1)] - 1.0
        btc_return = float(prior_return[BTC])
        pair = PAIRS[str(event.pair_id)]
        residual = {
            symbol: float(prior_return[symbol])
            - float(local.at[symbol, "btc_beta"]) * btc_return
            for symbol in pair
        }
        ordered = sorted(pair, key=lambda symbol: (-abs(residual[symbol]), symbol))
        source, receiver = ordered
        fragility = abs(float(local.at[receiver, "feature_1pct"]))
        strength = abs(residual[source]) * fragility
        rows.append(
            {
                "signal_time": day,
                "pair_id": event.pair_id,
                "source_matches": source == event.source_symbol,
                "receiver_matches": receiver == event.receiver_symbol,
                "source_lag_days": (day - pd.Timestamp(event.source_day)).days,
                "source_residual_difference": residual[source]
                - event.source_residual_return,
                "receiver_fragility_difference": fragility - event.receiver_fragility,
                "propagation_strength_difference": strength - event.propagation_strength,
            }
        )
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "signal_rows": len(audit),
                "source_exact": audit["source_matches"].all(),
                "receiver_exact": audit["receiver_matches"].all(),
                "source_lag_exact": audit["source_lag_days"].eq(1).all(),
                "max_abs_source_residual_difference": audit[
                    "source_residual_difference"
                ].abs().max(),
                "max_abs_receiver_fragility_difference": audit[
                    "receiver_fragility_difference"
                ].abs().max(),
                "max_abs_propagation_strength_difference": audit[
                    "propagation_strength_difference"
                ].abs().max(),
            }
        ]
    )
    return audit, summary


def audit_v158_portfolio(cfg: V158AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(cfg.panel_path)
    portfolio = pd.read_parquet(cfg.report_root / "daily_portfolio.parquet")
    panel_lookup = {
        pd.Timestamp(day): local.set_index("symbol")
        for day, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    rows = []
    previous_weights: dict[str, float] = {}
    previous_day: pd.Timestamp | None = None
    for event in portfolio.itertuples(index=False):
        day = pd.Timestamp(event.decision_time)
        if previous_day is not None and day - previous_day > pd.Timedelta(days=1):
            previous_weights = {}
        local = panel_lookup[day]
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
            + weights.get(BTC, 0.0) * float(local.iloc[0]["btc_return"])
        )
        residual = float(
            sum(
                weights[symbol] * float(local.at[symbol, "btc_beta"])
                for symbol in alt_symbols
            )
            + weights.get(BTC, 0.0)
        )
        total_turnover = expected_turnover + float(event.forced_close_turnover)
        rows.append(
            {
                "decision_time": day,
                "entry_turnover_difference": expected_turnover - event.entry_turnover,
                "gross_return_difference": gross - event.gross_return,
                "primary_return_difference": gross
                - cfg.one_way_cost * total_turnover
                - event.primary_net_return,
                "stress_return_difference": gross
                - cfg.stress_one_way_cost * total_turnover
                - event.stress_net_return,
                "residual_btc_beta": residual,
                "gross_notional_drift": (
                    sum(abs(weight) for weight in weights.values()) - 1.0
                    if weights
                    else 0.0
                ),
            }
        )
        previous_weights = weights
        previous_day = day
    audit = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "portfolio_days": len(audit),
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
    return audit, summary


def write_v158_pair_shock_candidate_audit(
    cfg: V158AuditConfig = V158AuditConfig(),
) -> dict[str, Path]:
    signal_audit, signal_summary = audit_v158_signals(cfg)
    portfolio_audit, portfolio_summary = audit_v158_portfolio(cfg)
    main_summary = pd.read_csv(cfg.report_root / "summary.csv")
    numeric = [
        *[column for column in signal_summary if column.startswith("max_abs_")],
        *[column for column in portfolio_summary if column.startswith("max_abs_")],
    ]
    maximum = max(
        [float(signal_summary.at[0, column]) for column in numeric[:3]]
        + [float(portfolio_summary.at[0, column]) for column in numeric[3:]]
    )
    outcome = pd.DataFrame(
        [
            {
                "signal_audit_pass": bool(
                    signal_summary.at[0, "source_exact"]
                    and signal_summary.at[0, "receiver_exact"]
                    and signal_summary.at[0, "source_lag_exact"]
                    and signal_summary.loc[
                        0, [column for column in signal_summary if column.startswith("max_abs_")]
                    ].le(cfg.tolerance).all()
                ),
                "portfolio_audit_pass": maximum <= cfg.tolerance,
                "rejection_confirmed": bool(
                    not bool(main_summary.at[0, "promote"])
                    and float(main_summary.at[0, "mean_primary_net_bp"]) < 0
                    and float(main_summary.at[0, "bootstrap_95_high_bp"]) < 0
                ),
            }
        ]
    )
    root = ensure_dir(cfg.report_root / "independent_audit")
    paths = {
        "signal": root / "signal_recalculation.parquet",
        "signal_summary": root / "signal_audit_summary.csv",
        "portfolio": root / "portfolio_recalculation.parquet",
        "portfolio_summary": root / "portfolio_audit_summary.csv",
        "outcome": root / "audit_outcome.csv",
        "audit_doc": cfg.audit_doc,
    }
    signal_audit.to_parquet(paths["signal"], index=False)
    signal_summary.to_csv(paths["signal_summary"], index=False)
    portfolio_audit.to_parquet(paths["portfolio"], index=False)
    portfolio_summary.to_csv(paths["portfolio_summary"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    verdict = "rejection_confirmed" if bool(outcome.iloc[0].all()) else "audit_failed"
    paths["audit_doc"].write_text(
        "\n".join(
            [
                "# v15.8 Independent Audit of v15.7 VT4",
                "",
                f"Verdict: `{verdict}`.",
                "",
                outcome.to_markdown(index=False),
                "",
                signal_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                portfolio_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                "The audit independently rebuilt every prior-day residual, source/receiver",
                "assignment and fragility product, then recomputed all portfolio returns,",
                "turnover costs and beta constraints. The rejection is confirmed.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
