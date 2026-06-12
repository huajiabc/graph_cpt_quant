from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

try:  # pragma: no cover - optional speed path
    import orjson
except Exception:  # pragma: no cover - environment without orjson
    orjson = None

from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.orderbook import FEATURE_COLUMNS, compute_orderbook_features
from pressure_graph.reports.v09e import SOURCE_ROOT, write_v09e_orderbook_capacity_ranking


REPORT_ROOT = Path("reports/v0_9e1_historical_orderbook_replay")
CONFLICT_ROOT = Path("reports/v0_9c_orderflow_capacity_ranking")
V09D_CONFLICT_ROOT = Path("reports/v0_9d_cic_capacity_architecture")
HISTORY_ROOT = Path("data/orderbook_history/bybit_official")
REPLAY_ORDERBOOK_ROOT = HISTORY_ROOT / "replay_orderbook"
RAW_ORDERBOOK_ROOT = HISTORY_ROOT / "raw" / "linear"

API_BASE = "https://api2.bybit.com"
LIST_FILES_PATH = "/quote/public/support/download/list-files"
PRODUCT_ID = "orderbook"
DEFAULT_POOL = "P2_CIC1_CIC2_COMBINED"
DEFAULT_MAX_POSITIONS = 8
TARGET_CANDIDATES = {"CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"}
CANDIDATE_ALIASES = {
    "CIC1_beta_extreme": "CIC1_FILTERED_MIR1",
    "CIC2_beta_broad": "CIC2_FILTERED_MIR1",
}


@dataclass(frozen=True)
class V09E1Config:
    source_root: Path = SOURCE_ROOT
    conflict_root: Path = CONFLICT_ROOT
    historical_conflict_root: Path = V09D_CONFLICT_ROOT
    report_root: Path = REPORT_ROOT
    history_root: Path = HISTORY_ROOT
    replay_orderbook_root: Path = REPLAY_ORDERBOOK_ROOT
    raw_orderbook_root: Path = RAW_ORDERBOOK_ROOT
    pool: str = DEFAULT_POOL
    max_positions: int = DEFAULT_MAX_POSITIONS
    max_files: int | None = None
    download: bool = True
    run_ranking: bool = True
    max_staleness_minutes: int = 30


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        return pd.DataFrame()


def _coerce_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _event_id(frame: pd.DataFrame) -> pd.Series:
    if "trade_id" in frame.columns:
        trade_id = frame["trade_id"].fillna("").astype(str)
    else:
        trade_id = pd.Series("", index=frame.index)
    fallback = (
        frame.get("candidate", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + frame.get("symbol", pd.Series("", index=frame.index)).astype(str)
        + "|"
        + _coerce_utc(frame.get("entry_time", pd.Series(pd.NaT, index=frame.index))).astype(str)
    )
    return np.where(trade_id.str.len() > 0, trade_id, fallback)


def load_replay_targets(
    source_root: Path = SOURCE_ROOT,
    conflict_root: Path = CONFLICT_ROOT,
    *,
    pool: str = DEFAULT_POOL,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name in ["selected_trades.csv", "skipped_trades.csv"]:
        frame = _read_csv(conflict_root / name)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_file"] = name
        if "pool" in frame.columns:
            frame = frame[frame["pool"].astype(str).eq(pool)]
        if "max_positions" in frame.columns:
            frame = frame[pd.to_numeric(frame["max_positions"], errors="coerce").eq(max_positions)]
        frames.append(frame)
    if frames:
        data = pd.concat(frames, ignore_index=True)
    else:
        data = _read_csv(source_root / "orderflow_shadow_trades.csv")
        if data.empty:
            return pd.DataFrame()
        data = data[data["candidate"].astype(str).isin(TARGET_CANDIDATES)].copy()
        data["source_file"] = "orderflow_shadow_trades.csv"
        data["pool"] = pool
        data["max_positions"] = max_positions
        if "selection_status" not in data.columns:
            data["selection_status"] = np.where(data.get("portfolio_accepted", False).astype(bool), "selected", "candidate")

    if data.empty:
        return pd.DataFrame()
    out = data.copy()
    out["entry_time"] = _coerce_utc(out["entry_time"])
    out["exit_time"] = _coerce_utc(out.get("exit_time", pd.Series(pd.NaT, index=out.index)))
    out["decision_time"] = out["entry_time"]
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["candidate"] = out["candidate"].astype(str)
    out["event_id"] = _event_id(out)
    out["date"] = out["decision_time"].dt.strftime("%Y-%m-%d")
    source_text = out.get("source_file", pd.Series("", index=out.index)).fillna("").astype(str)
    status_text = out.get("selection_status", pd.Series("", index=out.index)).fillna("").astype(str)
    skip_text = out.get("skip_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    out["is_skipped_target"] = source_text.eq("skipped_trades.csv") | status_text.str.contains(
        "skipped", case=False, na=False
    ) | skip_text.str.len().gt(0)
    skipped_dates = set(out.loc[out["is_skipped_target"], "date"].dropna().astype(str))
    if skipped_dates:
        out = out[out["date"].astype(str).isin(skipped_dates)].copy()
    out["decision_bucket"] = out["decision_time"].dt.floor("15min")
    out["conflict_set_id"] = (
        out["pool"].astype(str)
        + "|max"
        + out["max_positions"].astype(str)
        + "|"
        + out["decision_bucket"].astype(str)
    )
    dedupe_cols = ["event_id", "symbol", "candidate", "decision_time"]
    out = out.dropna(subset=["symbol", "decision_time", "date"]).drop_duplicates(dedupe_cols)
    return out.sort_values(["date", "symbol", "decision_time", "candidate"]).reset_index(drop=True)


def _target_source_has_rows(source_root: Path, *, pool: str, max_positions: int) -> bool:
    for name in ["selected_trades.csv", "skipped_trades.csv"]:
        frame = _read_csv(source_root / name)
        if frame.empty:
            continue
        if "pool" in frame.columns:
            frame = frame[frame["pool"].astype(str).eq(pool)]
        if "max_positions" in frame.columns:
            frame = frame[pd.to_numeric(frame["max_positions"], errors="coerce").eq(max_positions)]
        if not frame.empty:
            return True
    return False


def _resolve_conflict_root(cfg: V09E1Config) -> Path:
    if _target_source_has_rows(cfg.historical_conflict_root, pool=cfg.pool, max_positions=cfg.max_positions):
        return cfg.historical_conflict_root
    return cfg.conflict_root


def _write_ranking_source(report_root: Path, targets: pd.DataFrame) -> Path:
    source_root = ensure_dir(report_root / "_ranking_source")
    out = targets.copy()
    if "candidate" in out.columns:
        out["historical_candidate"] = out["candidate"].astype(str)
        out["candidate"] = out["candidate"].astype(str).replace(CANDIDATE_ALIASES)
    if "net_return_20bp" not in out.columns and "net_return" in out.columns:
        out["net_return_20bp"] = out["net_return"]
    if "net_return_10bp" not in out.columns and "gross_return" in out.columns:
        out["net_return_10bp"] = pd.to_numeric(out["gross_return"], errors="coerce") - 0.002
    if "net_return_10bp" not in out.columns and "net_return" in out.columns:
        out["net_return_10bp"] = out["net_return"]
    out.to_csv(source_root / "orderflow_shadow_trades.csv", index=False)
    return source_root


def build_download_manifest(targets: pd.DataFrame, *, max_files: int | None = None) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    local = targets.copy()
    skipped_dates = set(local.loc[local["is_skipped_target"], "date"].dropna().astype(str))
    local["date_has_skipped_target"] = local["date"].astype(str).isin(skipped_dates)
    manifest = (
        local.groupby(["symbol", "date"], as_index=False, sort=True)
        .agg(
            target_events=("event_id", "nunique"),
            skipped_target_events=("is_skipped_target", "sum"),
            date_has_skipped_target=("date_has_skipped_target", "max"),
            first_decision_time=("decision_time", "min"),
            last_decision_time=("decision_time", "max"),
        )
        .sort_values(["date_has_skipped_target", "date", "skipped_target_events", "symbol"], ascending=[False, True, False, True])
        .reset_index(drop=True)
    )
    if max_files is not None:
        manifest = manifest.head(max_files).copy()
    return manifest


def query_bybit_orderbook_file(symbol: str, date: str) -> dict[str, object]:
    params = {
        "bizType": "contract",
        "productId": PRODUCT_ID,
        "symbols": symbol,
        "interval": "daily",
        "startDay": date,
        "endDay": date,
    }
    url = f"{API_BASE}{LIST_FILES_PATH}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.bybit.com/derivatives/en/history-data",
            "Origin": "https://www.bybit.com",
            "platform": "pc",
        },
    )
    with urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("result", {}).get("list", []) or []
    if not rows:
        return _fallback_orderbook_file(symbol, date, status="fallback_url_no_api_rows")
    row = rows[0]
    return {
        "symbol": symbol,
        "date": date,
        "download_status": "remote_file_found",
        "url": row.get("url", ""),
        "filename": row.get("filename", ""),
        "size": int(row.get("size", 0) or 0),
    }


def _fallback_orderbook_file(symbol: str, date: str, *, status: str) -> dict[str, object]:
    filename = f"{date}_{symbol}_ob200.data.zip"
    return {
        "symbol": symbol,
        "date": date,
        "download_status": status,
        "url": f"https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{filename}",
        "filename": filename,
        "size": 0,
    }


def enrich_download_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    rows = []
    for row in manifest.itertuples(index=False):
        payload = row._asdict()
        try:
            payload.update(query_bybit_orderbook_file(str(row.symbol), str(row.date)))
        except Exception as exc:
            payload.update(_fallback_orderbook_file(str(row.symbol), str(row.date), status="fallback_url_after_api_error"))
            payload["error"] = str(exc)
        rows.append(payload)
    return pd.DataFrame(rows)


def _local_zip_path(raw_root: Path, symbol: str, filename: str) -> Path:
    return raw_root / symbol / filename


def download_manifest_files(manifest: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    rows = []
    ensure_dir(raw_root)
    for row in manifest.itertuples(index=False):
        payload = row._asdict()
        filename = str(payload.get("filename") or "")
        url = str(payload.get("url") or "")
        if not filename or not url:
            payload["local_status"] = "no_url"
            payload["local_path"] = ""
            rows.append(payload)
            continue
        path = _local_zip_path(raw_root, str(payload["symbol"]), filename)
        ensure_dir(path.parent)
        expected_size = int(payload.get("size") or 0)
        if path.exists() and (not expected_size or path.stat().st_size == expected_size) and zipfile.is_zipfile(path):
            payload["local_status"] = "cached"
            payload["local_path"] = str(path)
            rows.append(payload)
            continue
        if path.exists() and not zipfile.is_zipfile(path):
            path.unlink()
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            if shutil.which("curl"):
                result = subprocess.run(
                    [
                        "curl",
                        "-L",
                        "--fail",
                        "--connect-timeout",
                        "10",
                        "--speed-time",
                        "30",
                        "--speed-limit",
                        "1024",
                        "--retry",
                        "0",
                        "--max-time",
                        "300",
                        "-A",
                        "Mozilla/5.0",
                        "-o",
                        str(tmp),
                        url,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError((result.stderr or result.stdout).strip())
            else:
                request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=900) as response, tmp.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected_size and tmp.stat().st_size != expected_size:
                raise RuntimeError(f"downloaded size {tmp.stat().st_size} != expected {expected_size}")
            if not zipfile.is_zipfile(tmp):
                head = tmp.read_bytes()[:120]
                raise RuntimeError(f"downloaded file is not a zip: {head!r}")
            tmp.replace(path)
            payload["local_status"] = "downloaded"
            payload["local_path"] = str(path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            payload["local_status"] = "download_error"
            payload["local_path"] = str(path)
            payload["error"] = str(exc)
        rows.append(payload)
    return pd.DataFrame(rows)


def _apply_updates(book: dict[float, float], updates: list[list[str]]) -> None:
    for price_raw, size_raw in updates:
        price = float(price_raw)
        size = float(size_raw)
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size


def _loads_json_line(line: bytes) -> dict[str, object]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line.decode("utf-8"))


def _levels_from_book(
    *,
    symbol: str,
    snapshot_time: pd.Timestamp,
    snapshot_cts: pd.Timestamp | pd.NaT,
    update_id: object,
    seq: object,
    bids: dict[float, float],
    asks: dict[float, float],
    depth: int = 200,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side, source in [("bid", sorted(bids.items(), reverse=True)[:depth]), ("ask", sorted(asks.items())[:depth])]:
        for level, (price, size) in enumerate(source, start=1):
            rows.append(
                {
                    "exchange": "bybit",
                    "symbol": symbol,
                    "snapshot_time": snapshot_time,
                    "exchange_ts": snapshot_cts,
                    "update_id": update_id,
                    "seq": seq,
                    "side": side,
                    "level": level,
                    "price": float(price),
                    "size": float(size),
                    "notional": float(price) * float(size),
                }
            )
    return pd.DataFrame(rows)


def parse_orderbook_zip_for_targets(zip_path: Path, targets: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if targets.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame()
    local_targets = targets.copy().sort_values("decision_time").reset_index(drop=True)
    local_targets["target_ms"] = (local_targets["decision_time"].astype("int64") // 1_000_000).astype("int64")
    target_records = local_targets.to_dict("records")
    target_idx = 0
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    initialized = False
    last_snapshot_time = pd.NaT
    last_snapshot_cts = pd.NaT
    last_update_id: object = np.nan
    last_seq: object = np.nan
    feature_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []

    def capture(target: dict[str, object], status: str) -> None:
        nonlocal feature_rows
        if not initialized:
            status_rows.append(
                {
                    "event_id": target.get("event_id"),
                    "symbol": symbol,
                    "decision_time": target.get("decision_time"),
                    "replay_status": "no_book_before_decision",
                }
            )
            return
        levels = _levels_from_book(
            symbol=symbol,
            snapshot_time=last_snapshot_time,
            snapshot_cts=last_snapshot_cts,
            update_id=last_update_id,
            seq=last_seq,
            bids=bids,
            asks=asks,
        )
        features = compute_orderbook_features(levels)
        if features.empty:
            status_rows.append(
                {
                    "event_id": target.get("event_id"),
                    "symbol": symbol,
                    "decision_time": target.get("decision_time"),
                    "snapshot_time": last_snapshot_time,
                    "snapshot_cts": last_snapshot_cts,
                    "replay_status": "empty_feature",
                }
            )
            return
        for key in [
            "event_id",
            "trade_id",
            "signal_id",
            "candidate",
            "selection_status",
            "skip_reason",
            "conflict_set_id",
            "source_file",
        ]:
            features[key] = target.get(key, "")
        features["decision_time"] = target.get("decision_time")
        features["target_snapshot_age_ms"] = (
            (pd.Timestamp(target["decision_time"]) - pd.Timestamp(last_snapshot_cts if pd.notna(last_snapshot_cts) else last_snapshot_time)).total_seconds()
            * 1000
        )
        features["source"] = "bybit_official_historical_ob200"
        feature_rows.append(features)
        status_rows.append(
            {
                "event_id": target.get("event_id"),
                "symbol": symbol,
                "decision_time": target.get("decision_time"),
                "snapshot_time": last_snapshot_time,
                "snapshot_cts": last_snapshot_cts,
                "target_snapshot_age_ms": features["target_snapshot_age_ms"].iloc[0],
                "replay_status": status,
            }
        )

    with zipfile.ZipFile(zip_path) as archive:
        info = archive.infolist()[0]
        with archive.open(info) as handle:
            for line in handle:
                if target_idx >= len(target_records):
                    break
                obj = _loads_json_line(line)
                data = obj.get("data", {})
                ts_ms = int(obj.get("ts") or 0)
                cts_ms = int(obj.get("cts") or ts_ms)
                asof_ms = cts_ms or ts_ms
                while target_idx < len(target_records) and asof_ms > int(target_records[target_idx]["target_ms"]):
                    capture(target_records[target_idx], "covered")
                    target_idx += 1
                typ = obj.get("type")
                if typ == "snapshot":
                    bids = {float(price): float(size) for price, size in data.get("b", []) if float(size) > 0}
                    asks = {float(price): float(size) for price, size in data.get("a", []) if float(size) > 0}
                    initialized = True
                else:
                    _apply_updates(bids, data.get("b", []))
                    _apply_updates(asks, data.get("a", []))
                last_snapshot_time = pd.to_datetime(ts_ms, unit="ms", utc=True)
                last_snapshot_cts = pd.to_datetime(cts_ms, unit="ms", utc=True)
                last_update_id = data.get("u")
                last_seq = data.get("seq")
    while target_idx < len(target_records):
        capture(target_records[target_idx], "covered_eof")
        target_idx += 1
    features = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame(columns=FEATURE_COLUMNS)
    status = pd.DataFrame(status_rows)
    return features, status


def _write_replay_features(features: pd.DataFrame, replay_orderbook_root: Path) -> None:
    feature_dir = ensure_dir(replay_orderbook_root / "features")
    if features.empty:
        return
    for symbol, group in features.groupby("symbol", sort=False):
        path = feature_dir / f"{symbol}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            if not existing.empty:
                group = pd.concat([existing, group], ignore_index=True)
        group = group.sort_values("snapshot_time").drop_duplicates(["symbol", "snapshot_time", "event_id"], keep="last")
        write_parquet(group, path)


def _read_existing_replay_features(replay_orderbook_root: Path, targets: pd.DataFrame) -> pd.DataFrame:
    feature_dir = replay_orderbook_root / "features"
    if targets.empty or not feature_dir.exists():
        return pd.DataFrame()
    frames = []
    target_events = set(targets["event_id"].dropna().astype(str))
    for symbol in sorted(targets["symbol"].dropna().astype(str).unique()):
        path = feature_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty or "event_id" not in frame.columns:
            continue
        frame = frame[frame["event_id"].astype(str).isin(target_events)].copy()
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cached_parse_status(existing: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame()
    target_lookup = targets.drop_duplicates("event_id").set_index("event_id")
    rows = []
    for row in existing.itertuples(index=False):
        event_id = str(getattr(row, "event_id", ""))
        target = target_lookup.loc[event_id] if event_id in target_lookup.index else None
        decision_time = pd.Timestamp(target["decision_time"]) if target is not None else pd.NaT
        snapshot_time = pd.Timestamp(getattr(row, "snapshot_time", pd.NaT))
        snapshot_cts = pd.Timestamp(getattr(row, "snapshot_cts", getattr(row, "exchange_ts", pd.NaT)))
        asof = snapshot_cts if pd.notna(snapshot_cts) else snapshot_time
        age_ms = (decision_time - asof).total_seconds() * 1000 if pd.notna(decision_time) and pd.notna(asof) else np.nan
        rows.append(
            {
                "event_id": event_id,
                "symbol": getattr(row, "symbol", ""),
                "decision_time": decision_time,
                "snapshot_time": snapshot_time,
                "snapshot_cts": snapshot_cts,
                "target_snapshot_age_ms": age_ms,
                "replay_status": "cached_feature",
            }
        )
    return pd.DataFrame(rows)


def run_historical_orderbook_replay(cfg: V09E1Config = V09E1Config()) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    conflict_root = _resolve_conflict_root(cfg)
    targets = load_replay_targets(cfg.source_root, conflict_root, pool=cfg.pool, max_positions=cfg.max_positions)
    ranking_source_root = _write_ranking_source(report_root, targets)
    existing = _read_existing_replay_features(cfg.replay_orderbook_root, targets)
    existing_events = set(existing.get("event_id", pd.Series(dtype=str)).dropna().astype(str))
    parse_targets = targets[~targets["event_id"].astype(str).isin(existing_events)].copy()
    manifest = build_download_manifest(parse_targets, max_files=cfg.max_files)
    manifest = enrich_download_manifest(manifest) if not manifest.empty else manifest
    manifest = download_manifest_files(manifest, cfg.raw_orderbook_root) if cfg.download and not manifest.empty else manifest

    feature_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        local_path_raw = str(getattr(row, "local_path", "") or "")
        if not local_path_raw:
            continue
        local_path = Path(local_path_raw)
        if not local_path.is_file():
            continue
        sample = targets[targets["symbol"].eq(str(row.symbol)) & targets["date"].eq(str(row.date))]
        features, status = parse_orderbook_zip_for_targets(local_path, sample, str(row.symbol))
        if not features.empty:
            feature_frames.append(features)
        if not status.empty:
            status_frames.append(status)
    new_features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    features = pd.concat([existing, new_features], ignore_index=True) if not existing.empty or not new_features.empty else pd.DataFrame()
    cached_status = _cached_parse_status(existing, targets)
    parse_status = pd.concat([cached_status, *status_frames], ignore_index=True) if not cached_status.empty or status_frames else pd.DataFrame()
    _write_replay_features(features, cfg.replay_orderbook_root)

    outputs = {
        "replay_targets": report_root / "replay_targets.csv",
        "download_manifest": report_root / "download_manifest.csv",
        "historical_orderbook_features": report_root / "historical_orderbook_features.csv",
        "replay_parse_status": report_root / "replay_parse_status.csv",
        "ranking_source": ranking_source_root / "orderflow_shadow_trades.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    targets.to_csv(outputs["replay_targets"], index=False)
    manifest.to_csv(outputs["download_manifest"], index=False)
    features.to_csv(outputs["historical_orderbook_features"], index=False)
    parse_status.to_csv(outputs["replay_parse_status"], index=False)

    ranking_outputs: dict[str, Path] = {}
    if cfg.run_ranking:
        ranking_outputs = write_v09e_orderbook_capacity_ranking(
            ranking_source_root,
            cfg.replay_orderbook_root,
            cfg.report_root,
            max_staleness_minutes=cfg.max_staleness_minutes,
        )
    _write_notes(outputs["candidate_notes"], targets, manifest, features, parse_status, conflict_root)
    outputs.update(ranking_outputs)
    return outputs


def _write_notes(
    path: Path,
    targets: pd.DataFrame,
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    parse_status: pd.DataFrame,
    conflict_root: Path,
) -> None:
    covered = int(parse_status["replay_status"].astype(str).str.startswith("covered").sum()) if not parse_status.empty else 0
    downloaded = int(manifest.get("local_status", pd.Series(dtype=str)).astype(str).isin(["cached", "downloaded"]).sum()) if not manifest.empty else 0
    total_size = pd.to_numeric(manifest.get("size", pd.Series(dtype=float)), errors="coerce").sum() if not manifest.empty else 0
    lines = [
        "# v0.9E.1 Historical Orderbook Replay",
        "",
        f"targets: {len(targets)}",
        f"symbol_days: {len(manifest)}",
        f"downloaded_or_cached_files: {downloaded}",
        f"remote_size_bytes: {int(total_size) if pd.notna(total_size) else 0}",
        f"replayed_features: {len(features)}",
        f"covered_targets: {covered}",
        f"conflict_source: {conflict_root}",
        "",
        "Scope: only CIC/P2 max8 selected + skipped conflict windows are targeted.",
        "This report is shadow-only and does not enable real-live trading.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "REPORT_ROOT",
    "V09E1Config",
    "build_download_manifest",
    "load_replay_targets",
    "parse_orderbook_zip_for_targets",
    "run_historical_orderbook_replay",
]
