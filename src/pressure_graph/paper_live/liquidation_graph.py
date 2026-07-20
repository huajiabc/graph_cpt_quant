"""Remote-started LIVE_DIAGNOSTIC ledger for liquidation graph features."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.okx_liquidation_forward import (
    OkxLiquidationConfig,
    collect_okx_liquidation_snapshot,
)
from pressure_graph.paper_live.q90 import (
    FROZEN_SYMBOLS,
)


FACTOR_ID = "LIQUIDATION_GRAPH_BUCKET_VOLATILITY_STATE"


@dataclass(frozen=True)
class LiquidationGraphLiveConfig:
    live_root: Path
    source_root: Path
    contract_root: Path
    report_root: Path
    ledger_path: Path
    collection_interval_minutes: int = 15
    minimum_hourly_decisions: int = 336
    minimum_utc_days: int = 14
    scope: str = "live_shadow"
    enabled: bool = True
    push_policy: str = "record_only"
    real_orders_allowed: bool = False
    leverage_allowed: bool = False


def load_liquidation_graph_live_config(
    path: str | Path,
) -> LiquidationGraphLiveConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    application = payload["application"]
    data = payload["data"]
    contract = payload["contract"]
    cfg = LiquidationGraphLiveConfig(
        live_root=Path(data["live_root"]),
        source_root=Path(data["source_root"]),
        contract_root=Path(data["contract_root"]),
        report_root=Path(data["report_root"]),
        ledger_path=Path(data["ledger_path"]),
        collection_interval_minutes=int(
            data["collection_interval_minutes"]
        ),
        minimum_hourly_decisions=int(
            contract["minimum_hourly_decisions"]
        ),
        minimum_utc_days=int(contract["minimum_utc_days"]),
        scope=str(application["scope"]),
        enabled=bool(application["enabled"]),
        push_policy=str(application["push_policy"]),
        real_orders_allowed=bool(application["real_orders_allowed"]),
        leverage_allowed=bool(application["leverage_allowed"]),
    )
    if cfg.scope != "live_shadow" or cfg.push_policy != "record_only":
        raise ValueError(
            "liquidation graph diagnostic must be live_shadow/record_only"
        )
    if cfg.real_orders_allowed or cfg.leverage_allowed:
        raise ValueError(
            "liquidation graph diagnostic cannot enable orders or leverage"
        )
    return cfg


def _load_events(root: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(root / f"{symbol}.parquet")
        for symbol in ("BTCUSDT", *FROZEN_SYMBOLS)
    ]
    events = pd.concat(frames, ignore_index=True)
    for column in ("event_time", "first_seen_at", "last_seen_at"):
        events[column] = pd.to_datetime(
            events[column], utc=True, errors="coerce"
        )
    events["notional_usd"] = pd.to_numeric(
        events["notional_usd"], errors="coerce"
    )
    return (
        events.dropna(
            subset=["event_time", "first_seen_at", "notional_usd"]
        )
        .drop_duplicates(
            [
                "okx_inst_id",
                "timestamp_ms",
                "position_side",
                "liquidation_side",
                "contracts",
                "bankruptcy_price",
            ],
            keep="last",
        )
        .sort_values(["event_time", "bybit_symbol"])
        .reset_index(drop=True)
    )


def _write_source_audit(
    cfg: LiquidationGraphLiveConfig,
    manifest: pd.DataFrame,
    coverage: dict[str, object],
) -> dict[str, Path]:
    events = _load_events(cfg.source_root)
    expected = {"BTCUSDT", *FROZEN_SYMBOLS}
    checks = {
        "expected_symbols_exact_17": set(events["bybit_symbol"]) == expected,
        "manifest_has_no_errors": manifest["error"].isna().all(),
        "coverage_declares_17_successes": int(
            coverage["successful_symbols"]
        )
        == 17,
        "event_keys_unique": not events.duplicated(
            [
                "okx_inst_id",
                "timestamp_ms",
                "position_side",
                "liquidation_side",
                "contracts",
                "bankruptcy_price",
            ]
        ).any(),
        "causal_timestamps_coherent": events["event_time"]
        .le(events["first_seen_at"])
        .all()
        and events["first_seen_at"].le(events["last_seen_at"]).all(),
        "positive_finite_notional": bool(
            np.isfinite(events["notional_usd"]).all()
            and events["notional_usd"].gt(0).all()
        ),
        "knowledge_time_basis_frozen": coverage["knowledge_time_basis"]
        == "per_response_received_at",
        "no_outcome_columns": not any(
            token in column.lower()
            for column in events.columns
            for token in ("future", "return", "pnl", "target", "label")
        ),
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    root = ensure_dir(cfg.report_root / "source_audit")
    audit_path = root / "audit_checks.csv"
    audit.to_csv(audit_path, index=False)
    findings = cfg.report_root / "source_audit.md"
    findings.write_text(
        "\n".join(
            [
                "# Liquidation Graph Remote Source Audit",
                "",
                f"- checks_passed: {int(audit['passed'].sum())}/{len(audit)}",
                f"- unique_events: {len(events)}",
                f"- symbols: {events['bybit_symbol'].nunique()}",
                "- outcome_columns_loaded: `False`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not bool(audit["passed"].all()):
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise RuntimeError(f"liquidation source audit failed: {failed}")
    return {"audit": audit_path, "findings": findings}


def _remote_contract(
    cfg: LiquidationGraphLiveConfig,
    coverage: dict[str, object],
) -> dict[str, Path]:
    metadata = cfg.contract_root / "metadata.json"
    if metadata.exists():
        return {
            "metadata": metadata,
            "prereg": cfg.contract_root / "contract.md",
        }
    root = ensure_dir(cfg.contract_root)
    causal_start = pd.Timestamp(coverage["batch_completed_at"])
    payload = {
        "status": "frozen_remote_forward_contract_no_outcomes",
        "causal_start": causal_start.isoformat(),
        "feature_windows_minutes": [5, 15, 60],
        "hourly_minimum_decisions": cfg.minimum_hourly_decisions,
        "hourly_minimum_days": cfg.minimum_utc_days,
        "outcomes_loaded": False,
        "remote_causal_start_rule": "first_successful_remote_batch_completion",
    }
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    prereg = root / "contract.md"
    prereg.write_text(
        "\n".join(
            [
                "# Liquidation Graph Remote Forward Contract",
                "",
                f"- causal_start: {causal_start.isoformat()}",
                "- windows: 5m, 15m, 60m",
                "- event inclusion: `window_start <= event_time < decision_time`",
                "- knowledge inclusion: `first_seen_at <= decision_time`",
                f"- evaluation gate: {cfg.minimum_hourly_decisions} decisions across {cfg.minimum_utc_days} UTC days",
                "- outcomes_loaded: `False`",
                "- real_orders_allowed: `False`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"metadata": metadata, "prereg": prereg}


def _window_features(events: pd.DataFrame) -> dict[str, float]:
    sell = events["position_side"].eq("long") & events[
        "liquidation_side"
    ].eq("sell")
    buy = events["position_side"].eq("short") & events[
        "liquidation_side"
    ].eq("buy")
    sell_usd = float(events.loc[sell, "notional_usd"].sum())
    buy_usd = float(events.loc[buy, "notional_usd"].sum())
    total = sell_usd + buy_usd
    symbol_usd = events.groupby("bybit_symbol")["notional_usd"].sum()
    btc_usd = float(
        events.loc[
            events["bybit_symbol"].eq("BTCUSDT"), "notional_usd"
        ].sum()
    )
    return {
        "event_count": float(len(events)),
        "forced_sell_events": float(sell.sum()),
        "forced_buy_events": float(buy.sum()),
        "forced_sell_usd": sell_usd,
        "forced_buy_usd": buy_usd,
        "total_usd": total,
        "net_forced_buy_usd": buy_usd - sell_usd,
        "log_buy_sell_imbalance": float(
            np.log1p(buy_usd) - np.log1p(sell_usd)
        ),
        "active_symbols": float(events["bybit_symbol"].nunique()),
        "forced_sell_breadth": float(
            events.loc[sell, "bybit_symbol"].nunique()
        ),
        "forced_buy_breadth": float(
            events.loc[buy, "bybit_symbol"].nunique()
        ),
        "btc_notional_share": btc_usd / total if total > 0 else 0.0,
        "symbol_notional_hhi": float(
            np.square(symbol_usd / total).sum()
        )
        if total > 0
        else 0.0,
        "max_event_usd": float(events["notional_usd"].max())
        if len(events)
        else 0.0,
        "median_event_usd": float(events["notional_usd"].median())
        if len(events)
        else 0.0,
    }


def _build_features(
    events: pd.DataFrame, decisions: pd.DatetimeIndex
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decision in decisions:
        known = events[
            events["event_time"].lt(decision)
            & events["first_seen_at"].le(decision)
        ]
        row: dict[str, object] = {"decision_time": decision}
        for window in (5, 15, 60):
            local = known[
                known["event_time"].ge(
                    decision - pd.Timedelta(minutes=window)
                )
            ]
            for name, value in _window_features(local).items():
                row[f"liq_{window}m_{name}"] = value
        row["liq_5m_to_15m_notional_share"] = row[
            "liq_5m_total_usd"
        ] / max(float(row["liq_15m_total_usd"]), 1.0)
        row["liq_15m_to_60m_notional_share"] = row[
            "liq_15m_total_usd"
        ] / max(float(row["liq_60m_total_usd"]), 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_hourly_ledger(
    cfg: LiquidationGraphLiveConfig,
    coverage: dict[str, object],
    contract: dict[str, object],
) -> dict[str, Path]:
    causal_start = pd.Timestamp(contract["causal_start"])
    completed = pd.Timestamp(coverage["batch_completed_at"])
    first = causal_start.ceil("h")
    last = completed.floor("h")
    available = (
        pd.date_range(first, last, freq="h", tz="UTC")
        if first <= last
        else pd.DatetimeIndex([], tz="UTC")
    )
    existing = (
        pd.read_parquet(cfg.ledger_path)
        if cfg.ledger_path.exists()
        else pd.DataFrame()
    )
    if not existing.empty:
        existing["decision_time"] = pd.to_datetime(
            existing["decision_time"], utc=True, errors="coerce"
        )
    existing_times = set(
        existing.get(
            "decision_time", pd.Series(dtype="datetime64[ns, UTC]")
        )
    )
    missing = pd.DatetimeIndex(
        [decision for decision in available if decision not in existing_times]
    )
    if len(missing):
        appended = _build_features(_load_events(cfg.source_root), missing)
        appended["source_snapshot_completed_at"] = completed
        appended["source_total_rows"] = int(coverage["total_rows"])
        appended["knowledge_time_rule"] = "first_seen_at <= decision_time"
        existing = pd.concat([existing, appended], ignore_index=True)
    ledger = (
        existing.sort_values("decision_time")
        .drop_duplicates("decision_time", keep="first")
        .reset_index(drop=True)
        if not existing.empty
        else existing
    )
    if not ledger.empty:
        write_parquet(ledger, cfg.ledger_path)
    report = ensure_dir(cfg.report_root / "hourly_ledger")
    metadata_path = report / "metadata.json"
    metadata = {
        "status": "forward_hourly_feature_ledger_active",
        "causal_start": causal_start.isoformat(),
        "latest_snapshot_completed_at": completed.isoformat(),
        "available_hourly_decisions": len(available),
        "new_hourly_decisions": len(missing),
        "ledger_rows": len(ledger),
        "outcomes_loaded": False,
        "hourly_minimum_decisions": cfg.minimum_hourly_decisions,
        "hourly_minimum_days": cfg.minimum_utc_days,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    audit_path = report / "audit_checks.csv"
    checks = {
        "decision_keys_unique": ledger.empty
        or ledger["decision_time"].is_unique,
        "decisions_not_before_causal_start": ledger.empty
        or ledger["decision_time"].ge(causal_start).all(),
        "feature_values_finite": ledger.empty
        or np.isfinite(
            ledger[
                [column for column in ledger if column.startswith("liq_")]
            ]
        )
        .all()
        .all(),
        "outcomes_not_loaded": True,
    }
    pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    ).to_csv(audit_path, index=False)
    findings = cfg.report_root / "hourly_ledger.md"
    findings.write_text(
        "\n".join(
            [
                "# Liquidation Graph Hourly Forward Ledger",
                "",
                f"- causal_start: {causal_start.isoformat()}",
                f"- ledger_rows: {len(ledger)}",
                f"- appended_now: {len(missing)}",
                "- outcomes_loaded: `False`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ledger": cfg.ledger_path,
        "metadata": metadata_path,
        "audit": audit_path,
        "findings": findings,
    }


def write_liquidation_graph_live_diagnostic(
    cfg: LiquidationGraphLiveConfig,
    *,
    observed_at: object | None = None,
) -> dict[str, Path]:
    observed = (
        pd.Timestamp.now(tz="UTC")
        if observed_at is None
        else pd.Timestamp(observed_at)
    )
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    ensure_dir(cfg.live_root)
    manifest = collect_okx_liquidation_snapshot(
        ["BTCUSDT", *FROZEN_SYMBOLS],
        OkxLiquidationConfig(output_root=cfg.source_root),
    )
    coverage = json.loads(
        (cfg.source_root / "coverage.json").read_text(encoding="utf-8")
    )
    audit_paths = _write_source_audit(cfg, manifest, coverage)
    contract_paths = _remote_contract(cfg, coverage)
    contract = json.loads(
        (cfg.contract_root / "metadata.json").read_text(encoding="utf-8")
    )
    ledger_paths = _write_hourly_ledger(cfg, coverage, contract)
    ledger_metadata = json.loads(
        ledger_paths["metadata"].read_text(encoding="utf-8")
    )
    ledger_rows = int(ledger_metadata["ledger_rows"])
    causal_start = pd.Timestamp(contract["causal_start"])
    snapshot_completed = pd.Timestamp(coverage["batch_completed_at"])
    snapshot_completed = (
        snapshot_completed.tz_localize("UTC")
        if snapshot_completed.tzinfo is None
        else snapshot_completed.tz_convert("UTC")
    )
    status_observed = max(observed, snapshot_completed)
    elapsed_days = max(
        0.0, (status_observed - causal_start).total_seconds() / 86_400
    )
    gate_ready = bool(
        ledger_rows >= cfg.minimum_hourly_decisions
        and elapsed_days >= cfg.minimum_utc_days
    )
    status_payload = {
        "status": (
            "EVALUATION_GATE_READY_DIAGNOSTIC_ONLY"
            if gate_ready
            else "ACCUMULATING_LIVE_DIAGNOSTIC"
        ),
        "observed_at_utc": status_observed.isoformat(),
        "factor_id": FACTOR_ID,
        "factor_status": "LIVE_DIAGNOSTIC",
        "strategy_status": None,
        "scope": cfg.scope,
        "enabled": cfg.enabled,
        "push_policy": cfg.push_policy,
        "real_orders_allowed": False,
        "leverage_allowed": False,
        "remote_causal_start": causal_start.isoformat(),
        "latest_snapshot_completed_at": coverage["batch_completed_at"],
        "successful_symbols": int(manifest["error"].isna().sum()),
        "new_liquidation_events": int(manifest["new_rows"].sum()),
        "total_liquidation_events": int(manifest["total_rows"].sum()),
        "hourly_ledger_rows": ledger_rows,
        "minimum_hourly_decisions": cfg.minimum_hourly_decisions,
        "elapsed_utc_days": elapsed_days,
        "minimum_utc_days": cfg.minimum_utc_days,
        "evaluation_gate_ready": gate_ready,
        "outcomes_loaded": False,
        "paper_live_strategy": False,
    }
    status_json = cfg.report_root / "live_status.json"
    status_md = cfg.report_root / "live_status.md"
    status_json.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status_md.write_text(
        "\n".join(
            [
                "# v23.42 Liquidation Graph Live Diagnostic",
                "",
                f"- status: `{status_payload['status']}`",
                f"- factor_status: `{status_payload['factor_status']}`",
                f"- observed_at_utc: {status_payload['observed_at_utc']}",
                f"- remote_causal_start: {status_payload['remote_causal_start']}",
                f"- hourly_ledger_rows: {ledger_rows}",
                f"- total_liquidation_events: {status_payload['total_liquidation_events']}",
                f"- evaluation_gate_ready: `{gate_ready}`",
                "- outcomes_loaded: `False`",
                "- real_orders_allowed: `False`",
                "- leverage_allowed: `False`",
                "",
                "This is a frozen graph-level volatility-state factor ledger.",
                "It is not a directional strategy or a PaperLive order application.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status_json": status_json,
        "status_md": status_md,
        "ledger": ledger_paths["ledger"],
        "ledger_metadata": ledger_paths["metadata"],
        "source_audit": audit_paths["audit"],
        "contract_metadata": contract_paths["metadata"],
    }


__all__ = [
    "FACTOR_ID",
    "LiquidationGraphLiveConfig",
    "load_liquidation_graph_live_config",
    "write_liquidation_graph_live_diagnostic",
]
