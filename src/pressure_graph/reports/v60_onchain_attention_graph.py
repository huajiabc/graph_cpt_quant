"""v6.0 CEX-DEX / On-chain Attention Graph readiness atlas.

The project does not currently have a native historical on-chain event store.
This report therefore formalizes the ingestion contract, writes symbol mapping
seeds, and, when an optional event file is present, attributes CEX follow-through
without changing any live or shadow decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v50_perp_crowding_atlas import DEFAULT_FEATURE_PATH, UNIVERSE_COL, _available_columns, _bool, _num


REPORT_ROOT = Path("reports/v6_0_onchain_attention_graph")
DEFAULT_EVENT_PATH = Path("data/external/onchain_attention_events.csv")
DEFAULT_BYBIT_INSTRUMENTS = Path("data/raw/bybit/instruments.parquet")


@dataclass(frozen=True)
class V60Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = DEFAULT_FEATURE_PATH
    event_path: Path = DEFAULT_EVENT_PATH
    bybit_instruments_path: Path = DEFAULT_BYBIT_INSTRUMENTS
    universe_col: str = UNIVERSE_COL
    cost_bps: float = 20.0


EVENT_SCHEMA = [
    ("event_id", "stable id; optional, generated if missing"),
    ("event_time", "UTC timestamp when external attention event became observable"),
    ("symbol", "CEX target symbol such as AAVEUSDT"),
    ("base_asset", "base asset such as AAVE"),
    ("event_type", "dex_volume_spike | tvl_change | stablecoin_inflow | bridge_inflow | smart_wallet | protocol_fee | holder_growth"),
    ("attention_score", "as-of normalized score, higher means stronger external attention"),
    ("chain", "source chain if applicable"),
    ("protocol_slug", "source protocol slug if applicable"),
    ("source", "data provider"),
]


def _read_features(cfg: V60Config) -> pd.DataFrame:
    if not cfg.feature_path.exists():
        return pd.DataFrame()
    cols = _available_columns(
        cfg.feature_path,
        (
            "symbol",
            "feature_time",
            cfg.universe_col,
            "universe_static_current_top30",
            "warmup_complete",
            "future_ret_4h",
            "future_ret_12h",
            "ret_4h",
            "volume_z_1h",
            "btc_market_state",
        ),
        cfg.universe_col,
    )
    pf = pq.ParquetFile(cfg.feature_path)
    frames: list[pd.DataFrame] = []
    for idx in range(pf.num_row_groups):
        chunk = pf.read_row_group(idx, columns=cols).to_pandas()
        if cfg.universe_col in chunk.columns:
            chunk = chunk[_bool(chunk, cfg.universe_col)].copy()
        elif "universe_static_current_top30" in chunk.columns:
            chunk = chunk[_bool(chunk, "universe_static_current_top30")].copy()
        if "warmup_complete" in chunk.columns:
            chunk = chunk[_bool(chunk, "warmup_complete", True)].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["feature_time"] = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    return out.dropna(subset=["symbol", "feature_time"]).sort_values(["symbol", "feature_time"]).reset_index(drop=True)


def _base_from_symbol(symbol: str) -> str:
    for suffix in ("USDT", "USDC", "USD"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _symbol_mapping_seed(features: pd.DataFrame, cfg: V60Config) -> pd.DataFrame:
    symbols = sorted(features["symbol"].dropna().astype(str).unique().tolist()) if not features.empty else []
    rows = []
    instruments = pd.DataFrame()
    if cfg.bybit_instruments_path.exists():
        instruments = pd.read_parquet(cfg.bybit_instruments_path, columns=["symbol", "baseCoin", "launch_time"])
    inst = instruments.drop_duplicates("symbol").set_index("symbol") if not instruments.empty else pd.DataFrame()
    for symbol in symbols:
        base = str(inst.loc[symbol, "baseCoin"]) if not inst.empty and symbol in inst.index else _base_from_symbol(symbol)
        rows.append(
            {
                "symbol": symbol,
                "base_asset": base,
                "suggested_onchain_key": base.lower(),
                "protocol_slug": "",
                "chain": "",
                "mapping_status": "seed_unverified",
            }
        )
    return pd.DataFrame(rows)


def _coverage(cfg: V60Config, features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "cex_feature_table",
                "path": str(cfg.feature_path),
                "exists": cfg.feature_path.exists(),
                "rows": int(len(features)),
                "status": "ok" if len(features) else "missing_or_empty",
            },
            {
                "dataset": "onchain_attention_events",
                "path": str(cfg.event_path),
                "exists": cfg.event_path.exists(),
                "rows": int(len(events)),
                "status": "ok" if len(events) else "missing_optional_source",
            },
            {
                "dataset": "bybit_instruments",
                "path": str(cfg.bybit_instruments_path),
                "exists": cfg.bybit_instruments_path.exists(),
                "rows": int(len(pd.read_parquet(cfg.bybit_instruments_path, columns=["symbol"])) if cfg.bybit_instruments_path.exists() else 0),
                "status": "ok" if cfg.bybit_instruments_path.exists() else "missing",
            },
        ]
    )


def _read_events(cfg: V60Config) -> pd.DataFrame:
    if not cfg.event_path.exists():
        return pd.DataFrame(columns=[name for name, _ in EVENT_SCHEMA])
    events = pd.read_csv(cfg.event_path)
    if events.empty:
        return events
    events["event_time"] = pd.to_datetime(events.get("event_time"), utc=True, errors="coerce")
    if "symbol" not in events.columns and "base_asset" in events.columns:
        events["symbol"] = events["base_asset"].astype(str).str.upper() + "USDT"
    if "event_id" not in events.columns:
        events["event_id"] = (
            events.get("source", "unknown").astype(str)
            + "|"
            + events.get("symbol", "").astype(str)
            + "|"
            + events["event_time"].astype(str)
            + "|"
            + events.groupby(["symbol", "event_time"], dropna=False).cumcount().astype(str)
        )
    return events.dropna(subset=["event_time", "symbol"]).copy()


def _event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame([{"event_type": "no_onchain_events_loaded", "events": 0}])
    rows = []
    for event_type, group in events.groupby("event_type", sort=False, dropna=False):
        rows.append(
            {
                "event_type": event_type,
                "events": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "first_event_time": group["event_time"].min(),
                "last_event_time": group["event_time"].max(),
                "avg_attention_score": float(_num(group, "attention_score").mean()) if "attention_score" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _followthrough(events: pd.DataFrame, features: pd.DataFrame, cfg: V60Config) -> pd.DataFrame:
    if events.empty or features.empty:
        return pd.DataFrame([{"bucket": "no_events_or_features", "events": 0, "net20_4h": np.nan, "net20_12h": np.nan}])
    rows = []
    feat = features.sort_values(["symbol", "feature_time"]).copy()
    for symbol, local_events in events.groupby("symbol", sort=False):
        local_feat = feat[feat["symbol"].astype(str).eq(str(symbol))]
        if local_feat.empty:
            continue
        merged = pd.merge_asof(
            local_events.sort_values("event_time"),
            local_feat,
            left_on="event_time",
            right_on="feature_time",
            by="symbol",
            direction="forward",
            tolerance=pd.Timedelta(minutes=30),
        )
        rows.append(merged)
    aligned = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if aligned.empty:
        return pd.DataFrame([{"bucket": "no_aligned_cex_followthrough", "events": 0, "net20_4h": np.nan, "net20_12h": np.nan}])
    aligned["net20_4h"] = _num(aligned, "future_ret_4h") - 2.0 * cfg.cost_bps / 10_000.0
    aligned["net20_12h"] = _num(aligned, "future_ret_12h") - 2.0 * cfg.cost_bps / 10_000.0
    out = []
    for bucket, group in aligned.groupby("event_type", sort=False, dropna=False):
        out.append(
            {
                "bucket": bucket,
                "events": int(len(group)),
                "aligned_events": int(group["feature_time"].notna().sum()),
                "net20_4h": float(_num(group, "net20_4h").mean()),
                "net20_12h": float(_num(group, "net20_12h").mean()),
                "hit_rate_12h": float(_num(group, "net20_12h").gt(0).mean()),
            }
        )
    return pd.DataFrame(out)


def _write_notes(path: Path, coverage: pd.DataFrame) -> None:
    event_row = coverage[coverage["dataset"].eq("onchain_attention_events")]
    event_status = event_row["status"].iloc[0] if not event_row.empty else "unknown"
    lines = [
        "# v6.0 CEX-DEX / On-chain Attention Graph",
        "",
        "Status: data-layer/readiness atlas only. No strategy, selector, shadow, or live permission is changed.",
        "",
        f"On-chain event source status: `{event_status}`.",
        "",
        "Expected next data source:",
        "- `data/external/onchain_attention_events.csv` with event_time, symbol/base_asset, event_type, attention_score, chain/protocol_slug, source.",
        "",
        "Research use once data exists:",
        "- external attention spike -> CEX volume/price follow-through",
        "- external attention spike + CEX not extended -> reclaim/breakout",
        "- on-chain event as diagnostic feature for CIC/P2/O6 candidates",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v60_onchain_attention_graph(cfg: V60Config | None = None) -> dict[str, Path]:
    cfg = cfg or V60Config()
    report_root = ensure_dir(cfg.report_root)
    features = _read_features(cfg)
    events = _read_events(cfg)
    coverage = _coverage(cfg, features, events)
    schema = pd.DataFrame(EVENT_SCHEMA, columns=["field", "description"])
    mapping = _symbol_mapping_seed(features, cfg)
    event_summary = _event_summary(events)
    follow = _followthrough(events, features, cfg)

    outputs = {
        "onchain_data_coverage": report_root / "onchain_data_coverage.csv",
        "onchain_event_schema": report_root / "onchain_event_schema.csv",
        "symbol_mapping_seed": report_root / "symbol_mapping_seed.csv",
        "onchain_attention_event_summary": report_root / "onchain_attention_event_summary.csv",
        "cex_followthrough_atlas": report_root / "cex_followthrough_atlas.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["onchain_data_coverage"], index=False)
    schema.to_csv(outputs["onchain_event_schema"], index=False)
    mapping.to_csv(outputs["symbol_mapping_seed"], index=False)
    event_summary.to_csv(outputs["onchain_attention_event_summary"], index=False)
    follow.to_csv(outputs["cex_followthrough_atlas"], index=False)
    _write_notes(outputs["candidate_notes"], coverage)
    return outputs

