from __future__ import annotations

import zipfile
from pathlib import Path
import json

import pandas as pd
import pytest

from pressure_graph.reports.v09e1 import build_download_manifest, load_replay_targets, parse_orderbook_zip_for_targets


def _write_orderbook_zip(path: Path, rows: list[dict[str, object]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        payload = "\n".join(json.dumps(row, separators=(",", ":")) for row in rows).encode() + b"\n"
        archive.writestr(path.name.replace(".zip", ""), payload)


def test_parse_orderbook_zip_replays_last_book_before_decision(tmp_path) -> None:
    zip_path = tmp_path / "2026-01-01_AAAUSDT_ob200.data.zip"
    rows = [
        {
            "topic": "orderbook.200.AAAUSDT",
            "type": "snapshot",
            "ts": 1767225600000,
            "cts": 1767225600000,
            "data": {
                "s": "AAAUSDT",
                "b": [["99", "10"]],
                "a": [["101", "10"]],
                "u": 1,
                "seq": 1,
            },
        },
        {
            "topic": "orderbook.200.AAAUSDT",
            "type": "delta",
            "ts": 1767225604000,
            "cts": 1767225604000,
            "data": {
                "s": "AAAUSDT",
                "b": [["100", "5"]],
                "a": [["100.5", "4"]],
                "u": 2,
                "seq": 2,
            },
        },
        {
            "topic": "orderbook.200.AAAUSDT",
            "type": "delta",
            "ts": 1767225606000,
            "cts": 1767225606000,
            "data": {
                "s": "AAAUSDT",
                "b": [["100.1", "5"]],
                "a": [["100.4", "4"]],
                "u": 3,
                "seq": 3,
            },
        },
    ]
    _write_orderbook_zip(zip_path, rows)
    targets = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "trade_id": "trade-1",
                "signal_id": "signal-1",
                "candidate": "CIC1_FILTERED_MIR1",
                "symbol": "AAAUSDT",
                "decision_time": pd.Timestamp("2026-01-01 00:00:05Z"),
                "selection_status": "selected",
                "skip_reason": "",
                "conflict_set_id": "c1",
                "source_file": "selected_trades.csv",
            }
        ]
    )

    features, status = parse_orderbook_zip_for_targets(zip_path, targets, "AAAUSDT")

    assert status.iloc[0]["replay_status"] == "covered"
    row = features.iloc[0]
    assert row["best_bid"] == pytest.approx(100.0)
    assert row["best_ask"] == pytest.approx(100.5)
    assert row["event_id"] == "event-1"
    assert row["target_snapshot_age_ms"] == pytest.approx(1000.0)


def test_load_replay_targets_uses_p2_max8_conflict_rows(tmp_path) -> None:
    source_root = tmp_path / "source"
    conflict_root = tmp_path / "conflict"
    source_root.mkdir()
    conflict_root.mkdir()
    selected = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "signal_id": "s1",
                "candidate": "CIC1_FILTERED_MIR1",
                "symbol": "AAAUSDT",
                "entry_time": "2026-01-01 00:05:00Z",
                "exit_time": "2026-01-01 01:00:00Z",
                "pool": "P2_CIC1_CIC2_COMBINED",
                "max_positions": 8,
                "selection_status": "selected",
            },
            {
                "trade_id": "t2",
                "signal_id": "s2",
                "candidate": "CIC2_FILTERED_MIR1",
                "symbol": "BBBUSDT",
                "entry_time": "2026-01-02 00:05:00Z",
                "exit_time": "2026-01-02 01:00:00Z",
                "pool": "P2_CIC1_CIC2_COMBINED",
                "max_positions": 5,
                "selection_status": "selected",
            },
        ]
    )
    selected.to_csv(conflict_root / "selected_trades.csv", index=False)
    pd.DataFrame().to_csv(conflict_root / "skipped_trades.csv", index=False)

    targets = load_replay_targets(source_root, conflict_root)
    manifest = build_download_manifest(targets)

    assert len(targets) == 1
    assert targets.iloc[0]["event_id"] == "t1"
    assert manifest.iloc[0]["symbol"] == "AAAUSDT"
    assert manifest.iloc[0]["date"] == "2026-01-01"
