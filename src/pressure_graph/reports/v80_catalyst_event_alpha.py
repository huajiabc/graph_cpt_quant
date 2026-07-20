"""v8.0 Event / Listing / Catalyst Alpha Atlas.

This first catalyst pass uses exchange listing metadata that is already local
and writes an explicit schema for future news/announcement events.  It is a
coverage and post-listing response atlas, not a live catalyst strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v50_perp_crowding_atlas import _max_contribution, _month_cap35, _num


REPORT_ROOT = Path("reports/v8_0_catalyst_event_alpha")
DEFAULT_BYBIT_INSTRUMENTS = Path("data/raw/bybit/instruments.parquet")
DEFAULT_BYBIT_KLINES = Path("data/raw/bybit/klines")
DEFAULT_EXTERNAL_EVENTS = Path("data/external/catalyst_events.csv")


@dataclass(frozen=True)
class V80Config:
    report_root: Path = REPORT_ROOT
    bybit_instruments_path: Path = DEFAULT_BYBIT_INSTRUMENTS
    bybit_kline_root: Path = DEFAULT_BYBIT_KLINES
    external_event_path: Path = DEFAULT_EXTERNAL_EVENTS
    cost_bps: float = 20.0
    max_event_entry_delay_minutes: int = 30


EVENT_SCHEMA = [
    ("event_id", "stable catalyst id"),
    ("event_time", "UTC timestamp when catalyst became observable"),
    ("symbol", "target perp symbol, e.g. AAVEUSDT"),
    ("event_type", "listing | contract_listing | unlock | upgrade | airdrop | protocol_news | policy_macro"),
    ("source", "exchange/rss/manual/vendor"),
    ("headline", "short human-readable catalyst text"),
    ("confidence", "0-1 source confidence"),
    ("url", "optional source URL"),
]


def _read_instruments(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cols = ["symbol", "baseCoin", "status", "launch_time", "isPreListing", "symbolType"]
    available = pd.read_parquet(path)
    keep = [col for col in cols if col in available.columns]
    out = available[keep].copy()
    out["launch_time"] = pd.to_datetime(out.get("launch_time"), utc=True, errors="coerce")
    return out.dropna(subset=["symbol", "launch_time"]).copy()


def _read_external_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[name for name, _ in EVENT_SCHEMA])
    events = pd.read_csv(path)
    if events.empty:
        return events
    events["event_time"] = pd.to_datetime(events.get("event_time"), utc=True, errors="coerce")
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
    return events.dropna(subset=["symbol", "event_time"]).copy()


def _event_coverage(cfg: V80Config, instruments: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    raw_files = list(cfg.bybit_kline_root.glob("*.parquet")) if cfg.bybit_kline_root.exists() else []
    return pd.DataFrame(
        [
            {
                "dataset": "bybit_listing_metadata",
                "path": str(cfg.bybit_instruments_path),
                "exists": cfg.bybit_instruments_path.exists(),
                "rows": int(len(instruments)),
                "status": "ok" if len(instruments) else "missing_or_empty",
            },
            {
                "dataset": "bybit_raw_klines",
                "path": str(cfg.bybit_kline_root),
                "exists": cfg.bybit_kline_root.exists(),
                "rows": int(len(raw_files)),
                "status": "ok" if raw_files else "missing_or_empty",
            },
            {
                "dataset": "external_catalyst_events",
                "path": str(cfg.external_event_path),
                "exists": cfg.external_event_path.exists(),
                "rows": int(len(external)),
                "status": "ok" if len(external) else "missing_optional_source",
            },
        ]
    )


def _listing_events(instruments: pd.DataFrame, cfg: V80Config) -> pd.DataFrame:
    if instruments.empty:
        return pd.DataFrame()
    rows = []
    for row in instruments.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        rows.append(
            {
                "event_id": f"bybit_listing|{symbol}|{pd.Timestamp(getattr(row, 'launch_time')).isoformat()}",
                "event_time": pd.Timestamp(getattr(row, "launch_time")),
                "symbol": symbol,
                "event_type": "bybit_perp_listing",
                "source": "bybit_instruments",
                "base_asset": getattr(row, "baseCoin", ""),
                "raw_kline_available": (cfg.bybit_kline_root / f"{symbol}.parquet").exists(),
            }
        )
    return pd.DataFrame(rows)


def _future_from_raw(
    path: Path,
    event_time: pd.Timestamp,
    cost_bps: float,
    max_event_entry_delay_minutes: int,
) -> dict[str, float | int | str]:
    if not path.exists():
        return {"status": "missing_raw_kline"}
    try:
        frame = pd.read_parquet(path, columns=["bar_close_time", "open", "high", "low", "close", "turnover"])
    except Exception:  # noqa: BLE001 - malformed one-symbol file should not kill the atlas
        return {"status": "read_error"}
    if frame.empty:
        return {"status": "empty_raw_kline"}
    frame["bar_close_time"] = pd.to_datetime(frame["bar_close_time"], utc=True, errors="coerce")
    frame = frame.sort_values("bar_close_time")
    post = frame[frame["bar_close_time"].ge(event_time)].copy()
    if post.empty:
        return {"status": "no_bar_after_event"}
    entry = post.iloc[0]
    entry_time = pd.Timestamp(entry["bar_close_time"])
    entry_delay_minutes = float((entry_time - event_time).total_seconds() / 60.0)
    if entry_delay_minutes > max_event_entry_delay_minutes:
        return {
            "status": "event_time_not_covered",
            "entry_time": entry_time.isoformat(),
            "entry_delay_minutes": entry_delay_minutes,
        }
    entry_close = float(entry["close"])
    out: dict[str, float | int | str] = {
        "status": "ok",
        "entry_time": entry_time.isoformat(),
        "entry_delay_minutes": entry_delay_minutes,
        "entry_close": entry_close,
    }
    for label, bars in [("4h", 16), ("24h", 96), ("72h", 288), ("7d", 672)]:
        window = frame[(frame["bar_close_time"].ge(entry_time)) & (frame["bar_close_time"].le(entry_time + pd.Timedelta(minutes=15 * bars)))]
        if len(window) <= 1 or not np.isfinite(entry_close) or entry_close == 0:
            out[f"ret_{label}"] = np.nan
            out[f"net20_{label}"] = np.nan
            out[f"mfe_{label}"] = np.nan
            out[f"mae_{label}"] = np.nan
            continue
        close = float(window.iloc[-1]["close"])
        out[f"ret_{label}"] = close / entry_close - 1.0
        out[f"net20_{label}"] = close / entry_close - 1.0 - 2.0 * cost_bps / 10_000.0
        out[f"mfe_{label}"] = float(_num(window, "high").max()) / entry_close - 1.0
        out[f"mae_{label}"] = float(_num(window, "low").min()) / entry_close - 1.0
    out["turnover_first_24h"] = float(_num(window if "window" in locals() else frame.head(0), "turnover").head(96).sum()) if "turnover" in frame.columns else np.nan
    return out


def _post_listing_response(events: pd.DataFrame, cfg: V80Config) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame([{"event_type": "no_listing_events", "events": 0}])
    rows = []
    for row in events.itertuples(index=False):
        event = row._asdict()
        symbol = str(event["symbol"])
        response = _future_from_raw(
            cfg.bybit_kline_root / f"{symbol}.parquet",
            pd.Timestamp(event["event_time"]),
            cfg.cost_bps,
            cfg.max_event_entry_delay_minutes,
        )
        rows.append({**event, **response, "month": pd.Timestamp(event["event_time"]).strftime("%Y-%m")})
    return pd.DataFrame(rows)


def _listing_event_summary(response: pd.DataFrame) -> pd.DataFrame:
    if response.empty or "status" not in response.columns:
        return pd.DataFrame()
    rows = []
    for status, group in response.groupby("status", sort=False):
        rows.append(
            {
                "status": status,
                "events": int(len(group)),
                "symbols": int(group["symbol"].nunique()) if "symbol" in group else 0,
                "net20_24h": float(_num(group, "net20_24h").mean()) if "net20_24h" in group else np.nan,
                "net20_7d": float(_num(group, "net20_7d").mean()) if "net20_7d" in group else np.nan,
                "hit_rate_24h": float(_num(group, "net20_24h").gt(0).mean()) if "net20_24h" in group else np.nan,
                "month_cap35_net20_24h": _month_cap35(group, "net20_24h") if "net20_24h" in group else np.nan,
                "max_symbol_contribution_net20_24h": _max_contribution(group, "symbol", "net20_24h") if "net20_24h" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _external_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame([{"event_type": "no_external_events_loaded", "events": 0}])
    return events.groupby(["event_type", "source"], as_index=False, sort=False).agg(
        events=("event_id", "count"),
        symbols=("symbol", "nunique"),
        first_event=("event_time", "min"),
        last_event=("event_time", "max"),
    )


def _write_notes(path: Path, coverage: pd.DataFrame, listing_summary: pd.DataFrame) -> None:
    external = coverage[coverage["dataset"].eq("external_catalyst_events")]
    ext_status = external["status"].iloc[0] if not external.empty else "unknown"
    ok = listing_summary[listing_summary["status"].eq("ok")] if not listing_summary.empty else pd.DataFrame()
    lines = [
        "# v8.0 Event / Listing / Catalyst Alpha Atlas",
        "",
        "Status: catalyst coverage/listing atlas only. No catalyst strategy, shadow, or live permission is changed.",
        f"External catalyst source status: `{ext_status}`.",
    ]
    if not ok.empty:
        row = ok.iloc[0]
        lines.append(f"- Bybit listing events with raw kline replay: events={int(row['events'])}, net20_24h={row['net20_24h']:.4%}, net20_7d={row['net20_7d']:.4%}.")
    else:
        uncovered = int(listing_summary.loc[listing_summary["status"].eq("event_time_not_covered"), "events"].sum())
        lines.append(
            "- No listing event has raw kline coverage close enough to launch time; "
            f"event_time_not_covered={uncovered}. No listing-return claim is valid."
        )
    lines.extend(
        [
            "",
            "Next data source contract:",
            "- `data/external/catalyst_events.csv` with event_time, symbol, event_type, source, headline, confidence, url.",
            "",
            "Guardrails:",
            "- Listing launch-time response is not equivalent to news alpha.",
            "- Any catalyst motif must be tested against no-event matched symbols and execution availability before promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v80_catalyst_event_alpha(cfg: V80Config | None = None) -> dict[str, Path]:
    cfg = cfg or V80Config()
    report_root = ensure_dir(cfg.report_root)
    instruments = _read_instruments(cfg.bybit_instruments_path)
    external = _read_external_events(cfg.external_event_path)
    coverage = _event_coverage(cfg, instruments, external)
    schema = pd.DataFrame(EVENT_SCHEMA, columns=["field", "description"])
    listing = _listing_events(instruments, cfg)
    response = _post_listing_response(listing, cfg)
    listing_summary = _listing_event_summary(response)
    external_summary = _external_event_summary(external)

    outputs = {
        "catalyst_data_coverage": report_root / "catalyst_data_coverage.csv",
        "catalyst_event_schema": report_root / "catalyst_event_schema.csv",
        "listing_events": report_root / "listing_events.csv",
        "post_listing_response": report_root / "post_listing_response.csv",
        "listing_event_summary": report_root / "listing_event_summary.csv",
        "external_event_summary": report_root / "external_event_summary.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["catalyst_data_coverage"], index=False)
    schema.to_csv(outputs["catalyst_event_schema"], index=False)
    listing.to_csv(outputs["listing_events"], index=False)
    response.to_csv(outputs["post_listing_response"], index=False)
    listing_summary.to_csv(outputs["listing_event_summary"], index=False)
    external_summary.to_csv(outputs["external_event_summary"], index=False)
    _write_notes(outputs["candidate_notes"], coverage, listing_summary)
    return outputs

