from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pressure_graph.clients import BybitClient
from pressure_graph.config import load_config
from pressure_graph.io import ensure_dir


TIME_COLUMNS = {
    "klines": "bar_open_time",
    "funding": "funding_time",
    "open_interest": "oi_time",
}


def _read_parquet_time(path: Path, column: str) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        data = pd.read_parquet(path, columns=[column])
    except Exception:
        return None
    if data.empty or column not in data.columns:
        return None
    ts = pd.to_datetime(data[column], utc=True, errors="coerce").max()
    return None if pd.isna(ts) else pd.Timestamp(ts)


def _processed_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        data = pd.read_parquet(path, columns=["symbol", "feature_time"])
    except Exception as exc:  # noqa: BLE001 - health check should report, not crash
        return {"exists": True, "error": str(exc)}
    if data.empty:
        return {"exists": True, "rows": 0, "symbols": 0, "latest_feature_time": None}
    latest = pd.to_datetime(data["feature_time"], utc=True, errors="coerce").max()
    return {
        "exists": True,
        "rows": int(len(data)),
        "symbols": int(data["symbol"].nunique()),
        "latest_feature_time": None if pd.isna(latest) else pd.Timestamp(latest),
    }


def _raw_summary(live_root: Path) -> list[dict[str, Any]]:
    raw_root = live_root / "raw" / "bybit"
    rows: list[dict[str, Any]] = []
    for subdir, time_col in TIME_COLUMNS.items():
        path = raw_root / subdir
        files = sorted(path.glob("*.parquet")) if path.exists() else []
        latest_by_symbol: list[tuple[str, pd.Timestamp]] = []
        for file in files:
            ts = _read_parquet_time(file, time_col)
            if ts is not None:
                latest_by_symbol.append((file.stem, ts))
        latest = max((ts for _, ts in latest_by_symbol), default=None)
        oldest_latest = min((ts for _, ts in latest_by_symbol), default=None)
        rows.append(
            {
                "dataset": subdir,
                "files": len(files),
                "symbols_with_time": len(latest_by_symbol),
                "latest_time": latest,
                "oldest_symbol_latest_time": oldest_latest,
            }
        )
    return rows


def _status_summary(report_root: Path) -> dict[str, str]:
    status = report_root / "current_status.md"
    if not status.exists():
        return {"current_status_exists": "false"}
    text = status.read_text(encoding="utf-8", errors="replace")
    out = {"current_status_exists": "true"}
    for key in [
        "latest_feature_time",
        "data_stale",
        "primary_portfolio_trades",
        "sample_status",
        "evaluation_status",
    ]:
        match = re.search(rf"^- {re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
        if match:
            out[key] = match.group(1).strip()
    return out


def _api_probe(base_config) -> dict[str, Any]:
    client = BybitClient(str(base_config.exchanges.bybit.base_url), base_config.exchanges.bybit.category)
    try:
        end = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=15)
        start = end - pd.Timedelta(minutes=45)
        klines = client.klines("BTCUSDT", start, end, base_config.experiment.base_interval)
        latest = None
        if not klines.empty:
            latest = pd.to_datetime(klines["bar_close_time"], utc=True, errors="coerce").max()
        return {
            "ok": True,
            "rows": int(len(klines)),
            "latest_btc_bar_close": None if latest is None or pd.isna(latest) else pd.Timestamp(latest),
        }
    except Exception as exc:  # noqa: BLE001 - probe is diagnostic
        return {"ok": False, "error": str(exc)}
    finally:
        client.close()


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact live freshness health report.")
    parser.add_argument("--base-config", default="configs/v0_3.yaml")
    parser.add_argument("--live-root", default="data/live_v07d2")
    parser.add_argument("--report-root", default="reports/v0_7d2_cic_mir1_paper_live")
    parser.add_argument(
        "--processed-path",
        default="data/live_v07d2/processed/v0_7d2_live_features.parquet",
    )
    parser.add_argument(
        "--health-report",
        default="reports/v0_7d2_cic_mir1_paper_live/live_health_status.md",
    )
    parser.add_argument("--skip-api-probe", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    live_root = Path(args.live_root)
    report_root = Path(args.report_root)
    processed = _processed_summary(Path(args.processed_path))
    raw = _raw_summary(live_root)
    status = _status_summary(report_root)
    api = {"skipped": True} if args.skip_api_probe else _api_probe(base_config)
    now = pd.Timestamp.now(tz="UTC")

    lines = [
        "# Live Health Status",
        "",
        f"- checked_at_utc: {now.isoformat()}",
        f"- api_probe_ok: {_fmt(api.get('ok', api.get('skipped')))}",
        f"- api_probe_rows: {_fmt(api.get('rows'))}",
        f"- api_probe_latest_btc_bar_close: {_fmt(api.get('latest_btc_bar_close'))}",
        f"- api_probe_error: {_fmt(api.get('error'))}",
        "",
        "## Current Status",
    ]
    for key, value in status.items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines.extend(["", "## Processed Features"])
    for key, value in processed.items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines.extend(["", "## Raw Freshness"])
    for row in raw:
        lines.append(
            "- {dataset}: files={files}, symbols_with_time={symbols_with_time}, "
            "latest={latest_time}, oldest_symbol_latest={oldest_symbol_latest_time}".format(
                **{key: _fmt(value) for key, value in row.items()}
            )
        )

    out = Path(args.health_report)
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
