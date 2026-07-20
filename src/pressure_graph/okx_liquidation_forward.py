"""Idempotent forward collection of public OKX liquidation snapshots."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


OKX_BASE_URL = "https://www.okx.com"
DEFAULT_OUTPUT_ROOT = Path("data/external/okx_liquidation_forward")
EVENT_KEY = (
    "bybit_symbol",
    "okx_inst_id",
    "timestamp_ms",
    "position_side",
    "liquidation_side",
    "contracts",
    "bankruptcy_price",
)


@dataclass(frozen=True)
class OkxLiquidationConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    base_url: str = OKX_BASE_URL
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_sleep_seconds: float = 1.0
    request_sleep_seconds: float = 0.15
    limit: int = 100
    merge_existing: bool = True


@dataclass(frozen=True)
class OkxLiquidationSymbolResult:
    bybit_symbol: str
    okx_inst_family: str | None
    okx_inst_id: str | None
    snapshot_rows: int
    new_rows: int
    total_rows: int
    first_event_time: str | None
    last_event_time: str | None
    output_path: str | None
    raw_path: str | None
    error: str | None = None


def okx_base_candidates(bybit_symbol: str) -> tuple[str, ...]:
    symbol = bybit_symbol.upper().strip()
    if not symbol.endswith("USDT"):
        raise ValueError(f"expected a USDT symbol, got {bybit_symbol!r}")
    base = symbol.removesuffix("USDT")
    candidates = [base]
    if base.startswith("1000") and len(base) > 4:
        candidates.append(base[4:])
    return tuple(dict.fromkeys(candidates))


def _get_json(
    client: httpx.Client,
    path: str,
    params: dict[str, str],
    cfg: OkxLiquidationConfig,
) -> dict[str, Any]:
    for attempt in range(cfg.retry_attempts):
        try:
            response = client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code")) != "0":
                raise RuntimeError(
                    f"OKX error {payload.get('code')}: {payload.get('msg')}"
                )
            return payload
        except (httpx.HTTPError, json.JSONDecodeError, RuntimeError):
            if attempt + 1 >= cfg.retry_attempts:
                raise
            time.sleep(cfg.retry_sleep_seconds * (attempt + 1))
    raise RuntimeError("unreachable retry state")


def parse_okx_swap_instruments(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("data", []):
        if str(item.get("instType")) != "SWAP":
            continue
        inst_family = str(item.get("instFamily") or item.get("uly") or "")
        contract_value_currency = str(item.get("ctValCcy", ""))
        base_currency = str(item.get("baseCcy", ""))
        if not base_currency and contract_value_currency not in {"USD", "USDT", "USDC"}:
            base_currency = contract_value_currency
        if not base_currency and inst_family.endswith("-USDT"):
            base_currency = inst_family.removesuffix("-USDT")
        rows.append(
            {
                "okx_inst_id": str(item.get("instId", "")),
                "okx_inst_family": inst_family,
                "base_currency": base_currency,
                "quote_currency": str(item.get("quoteCcy") or item.get("settleCcy") or ""),
                "settle_currency": str(item.get("settleCcy", "")),
                "contract_value": pd.to_numeric(item.get("ctVal"), errors="coerce"),
                "contract_value_currency": contract_value_currency,
                "contract_type": str(item.get("ctType", "")),
                "state": str(item.get("state", "")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "okx_inst_id",
                "okx_inst_family",
                "base_currency",
                "quote_currency",
                "settle_currency",
                "contract_value",
                "contract_value_currency",
                "contract_type",
                "state",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("okx_inst_id", keep="last")


def map_okx_swap_instruments(
    instruments: pd.DataFrame,
    bybit_symbols: list[str],
) -> pd.DataFrame:
    live = instruments.loc[
        instruments["state"].eq("live")
        & instruments["settle_currency"].eq("USDT")
        & instruments["okx_inst_id"].str.endswith("-USDT-SWAP")
    ].copy()
    rows = []
    for symbol in sorted({value.upper().strip() for value in bybit_symbols}):
        match = pd.DataFrame()
        for base in okx_base_candidates(symbol):
            family = f"{base}-USDT"
            match = live.loc[live["okx_inst_family"].eq(family)]
            if not match.empty:
                break
        if match.empty:
            rows.append({"bybit_symbol": symbol, "mapping_error": "no_live_okx_usdt_swap"})
            continue
        selected = match.sort_values("okx_inst_id").iloc[0].to_dict()
        rows.append({"bybit_symbol": symbol, **selected, "mapping_error": None})
    return pd.DataFrame(rows).sort_values("bybit_symbol").reset_index(drop=True)


def _canonical_payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def parse_okx_liquidations(
    payload: dict[str, Any],
    mapping: pd.Series,
    retrieved_at: pd.Timestamp,
) -> pd.DataFrame:
    payload_hash = _canonical_payload_hash(payload)
    rows = []
    for group in payload.get("data", []):
        inst_id = str(group.get("instId") or mapping["okx_inst_id"])
        inst_family = str(
            group.get("instFamily") or group.get("uly") or mapping["okx_inst_family"]
        )
        for detail in group.get("details", []):
            timestamp_ms = pd.to_numeric(
                detail.get("ts", detail.get("time")), errors="coerce"
            )
            contracts = pd.to_numeric(detail.get("sz"), errors="coerce")
            bankruptcy_price = pd.to_numeric(detail.get("bkPx"), errors="coerce")
            bankruptcy_loss = pd.to_numeric(detail.get("bkLoss"), errors="coerce")
            if not np.isfinite(timestamp_ms) or not np.isfinite(contracts):
                continue
            contract_value = float(mapping["contract_value"])
            contract_ccy = str(mapping["contract_value_currency"])
            base_ccy = str(mapping["base_currency"])
            if contract_ccy == base_ccy and np.isfinite(bankruptcy_price):
                notional_usd = float(contracts * contract_value * bankruptcy_price)
            elif contract_ccy in {"USD", "USDT", "USDC"}:
                notional_usd = float(contracts * contract_value)
            else:
                notional_usd = float("nan")
            rows.append(
                {
                    "bybit_symbol": str(mapping["bybit_symbol"]),
                    "okx_inst_family": inst_family,
                    "okx_inst_id": inst_id,
                    "event_time": pd.to_datetime(
                        int(timestamp_ms), unit="ms", utc=True, errors="coerce"
                    ),
                    "timestamp_ms": int(timestamp_ms),
                    "position_side": str(detail.get("posSide", "")),
                    "liquidation_side": str(detail.get("side", "")),
                    "contracts": float(contracts),
                    "bankruptcy_price": float(bankruptcy_price),
                    "bankruptcy_loss": float(bankruptcy_loss),
                    "currency": str(detail.get("ccy", "")),
                    "contract_value": contract_value,
                    "contract_value_currency": contract_ccy,
                    "notional_usd": notional_usd,
                    "first_seen_at": retrieved_at,
                    "last_seen_at": retrieved_at,
                    "source_payload_sha256": payload_hash,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                *EVENT_KEY,
                "okx_inst_family",
                "event_time",
                "bankruptcy_loss",
                "currency",
                "contract_value",
                "contract_value_currency",
                "notional_usd",
                "first_seen_at",
                "last_seen_at",
                "source_payload_sha256",
            ]
        )
    return (
        pd.DataFrame(rows)
        .dropna(subset=["event_time"])
        .drop_duplicates(list(EVENT_KEY), keep="last")
        .sort_values("event_time")
        .reset_index(drop=True)
    )


def _recompute_notional_usd(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild notionals from immutable contract fields after snapshot merges."""
    if frame.empty:
        return frame.copy()
    required = {
        "contracts",
        "bankruptcy_price",
        "contract_value",
        "contract_value_currency",
    }
    if not required.issubset(frame.columns):
        return frame.copy()

    repaired = frame.copy()
    contracts = pd.to_numeric(repaired["contracts"], errors="coerce")
    bankruptcy_price = pd.to_numeric(
        repaired["bankruptcy_price"], errors="coerce"
    )
    contract_value = pd.to_numeric(repaired["contract_value"], errors="coerce")
    contract_ccy = repaired["contract_value_currency"].astype(str).str.upper()
    stable_contract = contract_ccy.isin({"USD", "USDT", "USDC"})
    calculated = contracts * contract_value
    calculated = calculated.where(
        stable_contract, calculated * bankruptcy_price
    )
    valid = (
        np.isfinite(calculated)
        & calculated.gt(0)
        & contracts.gt(0)
        & contract_value.gt(0)
    )
    repaired.loc[valid, "notional_usd"] = calculated.loc[valid].astype(float)
    return repaired


def _repair_knowledge_times(frame: pd.DataFrame) -> pd.DataFrame:
    """Conservatively repair legacy batch-start timestamps that predate an event."""
    if frame.empty:
        return frame.copy()
    required = {"event_time", "first_seen_at", "last_seen_at"}
    if not required.issubset(frame.columns):
        return frame.copy()

    repaired = frame.copy()
    event_time = pd.to_datetime(repaired["event_time"], utc=True, errors="coerce")
    first_seen = pd.to_datetime(
        repaired["first_seen_at"], utc=True, errors="coerce"
    )
    last_seen = pd.to_datetime(repaired["last_seen_at"], utc=True, errors="coerce")
    premature = event_time.gt(first_seen)
    conservative_first_seen = last_seen.where(last_seen.ge(event_time), event_time)
    first_seen = first_seen.where(~premature, conservative_first_seen)
    last_seen = last_seen.where(last_seen.ge(first_seen), first_seen)
    repaired["first_seen_at"] = first_seen
    repaired["last_seen_at"] = last_seen
    return repaired


def merge_okx_liquidations(new: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, int]:
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    existing_keys = (
        set(map(tuple, existing[list(EVENT_KEY)].itertuples(index=False, name=None)))
        if not existing.empty
        else set()
    )
    new_keys = set(map(tuple, new[list(EVENT_KEY)].itertuples(index=False, name=None)))
    new_rows = len(new_keys - existing_keys)
    if existing.empty:
        repaired = _repair_knowledge_times(_recompute_notional_usd(new))
        return repaired.sort_values("event_time").reset_index(drop=True), new_rows
    combined = pd.concat([existing, new], ignore_index=True)
    combined["first_seen_at"] = pd.to_datetime(
        combined["first_seen_at"], utc=True, errors="coerce"
    )
    combined["last_seen_at"] = pd.to_datetime(
        combined["last_seen_at"], utc=True, errors="coerce"
    )
    combined = combined.sort_values([*EVENT_KEY, "last_seen_at"])
    first_seen = combined.groupby(list(EVENT_KEY), dropna=False)["first_seen_at"].min()
    last_seen = combined.groupby(list(EVENT_KEY), dropna=False)["last_seen_at"].max()
    deduplicated = combined.drop_duplicates(list(EVENT_KEY), keep="last").set_index(
        list(EVENT_KEY)
    )
    deduplicated["first_seen_at"] = first_seen
    deduplicated["last_seen_at"] = last_seen
    repaired = _repair_knowledge_times(
        _recompute_notional_usd(deduplicated.reset_index())
    )
    return (
        repaired.sort_values("event_time").reset_index(drop=True),
        new_rows,
    )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def collect_okx_liquidation_snapshot(
    bybit_symbols: list[str],
    cfg: OkxLiquidationConfig = OkxLiquidationConfig(),
) -> pd.DataFrame:
    requested = sorted({symbol.upper().strip() for symbol in bybit_symbols if symbol.strip()})
    batch_started_at = pd.Timestamp.now(tz="UTC")
    root = ensure_dir(cfg.output_root)
    snapshot_root = ensure_dir(
        root / "snapshots" / batch_started_at.strftime("%Y%m%dT%H%M%S%fZ")
    )
    results = []
    with httpx.Client(
        base_url=cfg.base_url,
        timeout=cfg.timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "graph-quant-research/1.0"},
    ) as client:
        instrument_payload = _get_json(
            client, "/api/v5/public/instruments", {"instType": "SWAP"}, cfg
        )
        instruments = parse_okx_swap_instruments(instrument_payload)
        mapping = map_okx_swap_instruments(instruments, requested)
        mapping.to_csv(root / "instrument_map.csv", index=False)
        for item in mapping.itertuples(index=False):
            row = pd.Series(item._asdict())
            symbol = str(row["bybit_symbol"])
            if pd.notna(row.get("mapping_error")):
                results.append(
                    asdict(
                        OkxLiquidationSymbolResult(
                            bybit_symbol=symbol,
                            okx_inst_family=None,
                            okx_inst_id=None,
                            snapshot_rows=0,
                            new_rows=0,
                            total_rows=0,
                            first_event_time=None,
                            last_event_time=None,
                            output_path=None,
                            raw_path=None,
                            error=str(row["mapping_error"]),
                        )
                    )
                )
                continue
            try:
                payload = _get_json(
                    client,
                    "/api/v5/public/liquidation-orders",
                    {
                        "instType": "SWAP",
                        "instFamily": str(row["okx_inst_family"]),
                        "state": "filled",
                        "limit": str(cfg.limit),
                    },
                    cfg,
                )
                response_received_at = pd.Timestamp.now(tz="UTC")
                raw_path = snapshot_root / f"{symbol}.json"
                raw_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
                )
                snapshot = parse_okx_liquidations(
                    payload, row, response_received_at
                )
                output_path = root / f"{symbol}.parquet"
                if cfg.merge_existing:
                    merged, new_rows = merge_okx_liquidations(snapshot, output_path)
                else:
                    merged, new_rows = snapshot, len(snapshot)
                _atomic_write_parquet(merged, output_path)
                results.append(
                    asdict(
                        OkxLiquidationSymbolResult(
                            bybit_symbol=symbol,
                            okx_inst_family=str(row["okx_inst_family"]),
                            okx_inst_id=str(row["okx_inst_id"]),
                            snapshot_rows=len(snapshot),
                            new_rows=new_rows,
                            total_rows=len(merged),
                            first_event_time=(
                                merged["event_time"].min().isoformat()
                                if not merged.empty
                                else None
                            ),
                            last_event_time=(
                                merged["event_time"].max().isoformat()
                                if not merged.empty
                                else None
                            ),
                            output_path=str(output_path),
                            raw_path=str(raw_path),
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001 - retain per-symbol failures
                results.append(
                    asdict(
                        OkxLiquidationSymbolResult(
                            bybit_symbol=symbol,
                            okx_inst_family=str(row["okx_inst_family"]),
                            okx_inst_id=str(row["okx_inst_id"]),
                            snapshot_rows=0,
                            new_rows=0,
                            total_rows=0,
                            first_event_time=None,
                            last_event_time=None,
                            output_path=None,
                            raw_path=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                )
            time.sleep(cfg.request_sleep_seconds)
    manifest = pd.DataFrame(results).sort_values("bybit_symbol").reset_index(drop=True)
    manifest.to_csv(root / "manifest.csv", index=False)
    successful = manifest.loc[manifest["error"].isna()]
    batch_completed_at = pd.Timestamp.now(tz="UTC")
    coverage = {
        "retrieved_at": batch_started_at.isoformat(),
        "batch_started_at": batch_started_at.isoformat(),
        "batch_completed_at": batch_completed_at.isoformat(),
        "knowledge_time_basis": "per_response_received_at",
        "requested_symbols": len(requested),
        "mapped_symbols": int(manifest["okx_inst_id"].notna().sum()),
        "successful_symbols": len(successful),
        "symbols_with_events": int(successful["total_rows"].gt(0).sum()),
        "snapshot_rows": int(successful["snapshot_rows"].sum()),
        "new_rows": int(successful["new_rows"].sum()),
        "total_rows": int(successful["total_rows"].sum()),
        "config": {**asdict(cfg), "output_root": str(cfg.output_root)},
    }
    (root / "coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return manifest


__all__ = [
    "EVENT_KEY",
    "OkxLiquidationConfig",
    "collect_okx_liquidation_snapshot",
    "map_okx_swap_instruments",
    "merge_okx_liquidations",
    "okx_base_candidates",
    "parse_okx_liquidations",
    "parse_okx_swap_instruments",
]
