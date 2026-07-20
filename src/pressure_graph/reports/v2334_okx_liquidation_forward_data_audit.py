"""Causal data audit for the forward OKX liquidation collector."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.okx_liquidation_forward import EVENT_KEY
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)


DATA_ROOT = Path("data/external/okx_liquidation_forward")
REPORT_ROOT = Path("reports/v23_34_okx_liquidation_forward_data_audit")
FINDINGS_PATH = Path(
    "docs/v2334_okx_liquidation_forward_data_audit_2026_07_17.md"
)
EXPECTED_SYMBOLS = ("BTCUSDT", *FROZEN_SYMBOLS)
FORBIDDEN_RESEARCH_COLUMNS = {
    "future_return",
    "forward_return",
    "pnl",
    "label",
    "target",
    "strategy_return",
}


def load_v2334_liquidations(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    frames = []
    for path in sorted(data_root.glob("*USDT.parquet")):
        frame = pd.read_parquet(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no liquidation parquet files under {data_root}")
    events = pd.concat(frames, ignore_index=True)
    for column in ("event_time", "first_seen_at", "last_seen_at"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="coerce")
    for column in (
        "timestamp_ms",
        "contracts",
        "bankruptcy_price",
        "bankruptcy_loss",
        "contract_value",
        "notional_usd",
    ):
        events[column] = pd.to_numeric(events[column], errors="coerce")
    return events.sort_values(["event_time", "bybit_symbol"]).reset_index(drop=True)


def build_v2334_five_minute_buckets(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = events.copy()
    local["bucket_time"] = local["event_time"].dt.floor("5min")
    forced_sell = local["position_side"].eq("long") & local[
        "liquidation_side"
    ].eq("sell")
    forced_buy = local["position_side"].eq("short") & local[
        "liquidation_side"
    ].eq("buy")
    local["forced_sell_events"] = forced_sell.astype(int)
    local["forced_buy_events"] = forced_buy.astype(int)
    local["forced_sell_usd"] = local["notional_usd"].where(forced_sell, 0.0)
    local["forced_buy_usd"] = local["notional_usd"].where(forced_buy, 0.0)

    by_symbol = (
        local.groupby(["bucket_time", "bybit_symbol"], as_index=False)
        .agg(
            liquidation_events=("timestamp_ms", "size"),
            forced_sell_events=("forced_sell_events", "sum"),
            forced_buy_events=("forced_buy_events", "sum"),
            forced_sell_usd=("forced_sell_usd", "sum"),
            forced_buy_usd=("forced_buy_usd", "sum"),
            first_knowledge_time=("first_seen_at", "min"),
            last_knowledge_time=("first_seen_at", "max"),
        )
        .sort_values(["bucket_time", "bybit_symbol"])
        .reset_index(drop=True)
    )
    by_symbol["total_liquidation_usd"] = (
        by_symbol["forced_sell_usd"] + by_symbol["forced_buy_usd"]
    )
    by_symbol["net_forced_buy_usd"] = (
        by_symbol["forced_buy_usd"] - by_symbol["forced_sell_usd"]
    )
    by_symbol["log_buy_sell_imbalance"] = np.log1p(
        by_symbol["forced_buy_usd"]
    ) - np.log1p(by_symbol["forced_sell_usd"])

    market = (
        by_symbol.groupby("bucket_time", as_index=False)
        .agg(
            active_symbols=("bybit_symbol", "nunique"),
            liquidation_events=("liquidation_events", "sum"),
            forced_sell_events=("forced_sell_events", "sum"),
            forced_buy_events=("forced_buy_events", "sum"),
            forced_sell_usd=("forced_sell_usd", "sum"),
            forced_buy_usd=("forced_buy_usd", "sum"),
            total_liquidation_usd=("total_liquidation_usd", "sum"),
            net_forced_buy_usd=("net_forced_buy_usd", "sum"),
            first_knowledge_time=("first_knowledge_time", "min"),
            last_knowledge_time=("last_knowledge_time", "max"),
        )
        .sort_values("bucket_time")
        .reset_index(drop=True)
    )
    market["log_buy_sell_imbalance"] = np.log1p(
        market["forced_buy_usd"]
    ) - np.log1p(market["forced_sell_usd"])
    return by_symbol, market


def _symbol_summary(events: pd.DataFrame) -> pd.DataFrame:
    local = events.copy()
    local["forced_sell_usd"] = local["notional_usd"].where(
        local["position_side"].eq("long"), 0.0
    )
    local["forced_buy_usd"] = local["notional_usd"].where(
        local["position_side"].eq("short"), 0.0
    )
    local["knowledge_lag_seconds"] = (
        local["first_seen_at"] - local["event_time"]
    ).dt.total_seconds()
    summary = (
        local.groupby("bybit_symbol", as_index=False)
        .agg(
            events=("timestamp_ms", "size"),
            first_event_time=("event_time", "min"),
            last_event_time=("event_time", "max"),
            first_known_at=("first_seen_at", "min"),
            last_known_at=("first_seen_at", "max"),
            forced_sell_events=("position_side", lambda x: int(x.eq("long").sum())),
            forced_buy_events=("position_side", lambda x: int(x.eq("short").sum())),
            forced_sell_usd=("forced_sell_usd", "sum"),
            forced_buy_usd=("forced_buy_usd", "sum"),
            total_liquidation_usd=("notional_usd", "sum"),
            median_event_usd=("notional_usd", "median"),
            p95_event_usd=("notional_usd", lambda x: float(x.quantile(0.95))),
            max_event_usd=("notional_usd", "max"),
            median_knowledge_lag_seconds=("knowledge_lag_seconds", "median"),
            p95_knowledge_lag_seconds=(
                "knowledge_lag_seconds",
                lambda x: float(x.quantile(0.95)),
            ),
            source_payloads=("source_payload_sha256", "nunique"),
        )
        .sort_values("bybit_symbol")
        .reset_index(drop=True)
    )
    summary["event_span_hours"] = (
        summary["last_event_time"] - summary["first_event_time"]
    ).dt.total_seconds() / 3600.0
    return summary


def audit_v2334(
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = load_v2334_liquidations(data_root)
    manifest = pd.read_csv(data_root / "manifest.csv")
    mapping = pd.read_csv(data_root / "instrument_map.csv")
    coverage = json.loads((data_root / "coverage.json").read_text(encoding="utf-8"))
    summary = _symbol_summary(events)
    by_symbol, market = build_v2334_five_minute_buckets(events)

    expected = set(EXPECTED_SYMBOLS)
    event_symbols = set(events["bybit_symbol"].astype(str))
    manifest_symbols = set(manifest["bybit_symbol"].astype(str))
    mapping_symbols = set(mapping["bybit_symbol"].astype(str))
    event_duplicates = int(events.duplicated(list(EVENT_KEY)).sum())
    numeric = events[
        ["contracts", "bankruptcy_price", "contract_value", "notional_usd"]
    ]
    expected_notional = events["contracts"] * events["contract_value"]
    non_stable = ~events["contract_value_currency"].isin(["USD", "USDT", "USDC"])
    expected_notional = expected_notional.where(
        ~non_stable, expected_notional * events["bankruptcy_price"]
    )
    direction_pairs = set(
        map(
            tuple,
            events[["position_side", "liquidation_side"]].drop_duplicates().to_numpy(),
        )
    )
    manifest_counts = manifest.set_index("bybit_symbol")["total_rows"].astype(int)
    actual_counts = events.groupby("bybit_symbol").size()
    raw_paths_exist = manifest["raw_path"].dropna().map(lambda x: Path(str(x)).exists())
    output_paths_exist = manifest["output_path"].dropna().map(
        lambda x: Path(str(x)).exists()
    )
    hash_valid = events["source_payload_sha256"].astype(str).map(
        lambda x: bool(re.fullmatch(r"[0-9A-F]{64}", x))
    )
    forbidden_present = sorted(FORBIDDEN_RESEARCH_COLUMNS & set(events.columns))
    market_span_hours = (
        events["event_time"].max() - events["event_time"].min()
    ).total_seconds() / 3600.0
    btc_rows = int(actual_counts.get("BTCUSDT", 0))
    eth_rows = int(actual_counts.get("ETHUSDT", 0))

    checks = {
        "expected_symbols_exact_17": event_symbols
        == manifest_symbols
        == mapping_symbols
        == expected,
        "coverage_declares_17_successes": coverage["requested_symbols"] == 17
        and coverage["mapped_symbols"] == 17
        and coverage["successful_symbols"] == 17
        and coverage["symbols_with_events"] == 17,
        "per_response_knowledge_time_basis": coverage.get("knowledge_time_basis")
        == "per_response_received_at",
        "manifest_has_no_errors": manifest["error"].isna().all(),
        "current_raw_and_output_paths_exist": raw_paths_exist.all()
        and output_paths_exist.all(),
        "event_keys_unique": event_duplicates == 0,
        "manifest_counts_reconcile": manifest_counts.index.equals(actual_counts.index)
        and np.array_equal(
            manifest_counts.sort_index().to_numpy(dtype=np.int64),
            actual_counts.sort_index().to_numpy(dtype=np.int64),
        ),
        "coverage_total_reconciles": int(coverage["total_rows"]) == len(events),
        "causal_timestamps_coherent": events["event_time"].le(
            events["first_seen_at"]
        ).all()
        and events["first_seen_at"].le(events["last_seen_at"]).all(),
        "liquidation_directions_exact": direction_pairs
        == {("long", "sell"), ("short", "buy")},
        "positive_finite_contract_fields": np.isfinite(numeric).all().all()
        and numeric.gt(0).all().all(),
        "notional_formula_reconciles": np.allclose(
            events["notional_usd"], expected_notional, rtol=1e-12, atol=1e-9
        ),
        "payload_hashes_valid": hash_valid.all(),
        "mapping_live_linear_usdt": mapping["mapping_error"].isna().all()
        and mapping["state"].eq("live").all()
        and mapping["settle_currency"].eq("USDT").all()
        and mapping["contract_type"].eq("linear").all()
        and mapping["okx_inst_id"].is_unique,
        "market_snapshot_span_at_least_23h": market_span_hours >= 23.0,
        "btc_eth_event_density_present": btc_rows >= 1_000 and eth_rows >= 1_000,
        "five_minute_buckets_reconcile": int(by_symbol["liquidation_events"].sum())
        == len(events)
        and int(market["liquidation_events"].sum()) == len(events)
        and np.isclose(market["total_liquidation_usd"].sum(), events["notional_usd"].sum()),
        "raw_source_contains_no_outcome_columns": not forbidden_present,
    }
    values = {
        "expected_symbols_exact_17": len(event_symbols),
        "coverage_declares_17_successes": coverage["successful_symbols"],
        "per_response_knowledge_time_basis": coverage.get("knowledge_time_basis"),
        "manifest_has_no_errors": int(manifest["error"].notna().sum()),
        "current_raw_and_output_paths_exist": int(raw_paths_exist.sum()),
        "event_keys_unique": event_duplicates,
        "manifest_counts_reconcile": int(actual_counts.sum()),
        "coverage_total_reconciles": int(coverage["total_rows"]),
        "causal_timestamps_coherent": int(
            events["event_time"].gt(events["first_seen_at"]).sum()
        ),
        "liquidation_directions_exact": sorted(direction_pairs),
        "positive_finite_contract_fields": int(numeric.isna().sum().sum()),
        "notional_formula_reconciles": float(
            np.max(np.abs(events["notional_usd"] - expected_notional))
        ),
        "payload_hashes_valid": int(hash_valid.sum()),
        "mapping_live_linear_usdt": int(mapping["okx_inst_id"].nunique()),
        "market_snapshot_span_at_least_23h": market_span_hours,
        "btc_eth_event_density_present": f"BTC={btc_rows};ETH={eth_rows}",
        "five_minute_buckets_reconcile": len(market),
        "raw_source_contains_no_outcome_columns": forbidden_present,
    }
    audit = pd.DataFrame(
        [
            {"check": name, "passed": bool(passed), "value": values[name]}
            for name, passed in checks.items()
        ]
    )
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_forward_liquidation_source_ready",
        "audit_failure_requires_investigation",
    )
    return audit, summary, by_symbol, market


def _write_findings(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    total_events = int(summary["events"].sum())
    total_usd = float(summary["total_liquidation_usd"].sum())
    sell_usd = float(summary["forced_sell_usd"].sum())
    buy_usd = float(summary["forced_buy_usd"].sum())
    failed = audit.loc[~audit["passed"]]
    text = [
        "# v23.34 OKX Liquidation Forward Data Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        f"The collector currently stores {total_events:,} unique liquidation events "
        f"across 17 mapped USDT swaps, totaling ${total_usd:,.2f} notional.",
        f"Long-position forced sells total ${sell_usd:,.2f}; short-position forced "
        f"buys total ${buy_usd:,.2f}.",
        "",
        "This is a current-snapshot-plus-forward source, not a historical liquidation "
        "backfill. The initial roughly 24-hour snapshot may only be used at decisions "
        "after it became known. Every future feature must require "
        "`first_seen_at <= decision_time` and use event windows ending strictly before "
        "the decision timestamp.",
        "",
        "The audit establishes data integrity and causal availability only. It does not "
        "claim predictive alpha and does not inspect future strategy returns.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2334_audit(
    data_root: Path = DATA_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, summary, by_symbol, market = audit_v2334(data_root)
    root = ensure_dir(report_root)
    outputs = {
        "audit": root / "audit_checks.csv",
        "symbol_summary": root / "symbol_summary.csv",
        "five_minute_by_symbol": root / "liquidation_5m_by_symbol.parquet",
        "five_minute_market": root / "liquidation_5m_market.parquet",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    audit.to_csv(outputs["audit"], index=False)
    summary.to_csv(outputs["symbol_summary"], index=False)
    by_symbol.to_parquet(outputs["five_minute_by_symbol"], index=False)
    market.to_parquet(outputs["five_minute_market"], index=False)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "source": "OKX public liquidation-orders endpoint",
                "collection_mode": "current_snapshot_plus_forward",
                "historical_backfill_available": False,
                "knowledge_time_basis": "per_response_received_at",
                "causal_eligibility_rule": "first_seen_at <= decision_time",
                "feature_window_rule": "decision_time - lookback <= event_time < decision_time",
                "direction_semantics": {
                    "long/sell": "long-position liquidation; forced sell",
                    "short/buy": "short-position liquidation; forced buy",
                },
                "expected_symbols": list(EXPECTED_SYMBOLS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(audit, summary, findings_path)
    return outputs


__all__ = [
    "DATA_ROOT",
    "EXPECTED_SYMBOLS",
    "audit_v2334",
    "build_v2334_five_minute_buckets",
    "load_v2334_liquidations",
    "write_v2334_audit",
]
