from __future__ import annotations

import pandas as pd

from pressure_graph.okx_liquidation_forward import (
    map_okx_swap_instruments,
    merge_okx_liquidations,
    okx_base_candidates,
    parse_okx_liquidations,
    parse_okx_swap_instruments,
)


def _instrument_payload() -> dict:
    return {
        "code": "0",
        "data": [
            {
                "instType": "SWAP",
                "instId": "BTC-USDT-SWAP",
                "instFamily": "BTC-USDT",
                "baseCcy": "",
                "quoteCcy": "",
                "settleCcy": "USDT",
                "ctVal": "0.01",
                "ctValCcy": "BTC",
                "ctType": "linear",
                "state": "live",
            },
            {
                "instType": "SWAP",
                "instId": "PEPE-USDT-SWAP",
                "instFamily": "PEPE-USDT",
                "baseCcy": "PEPE",
                "quoteCcy": "USDT",
                "settleCcy": "USDT",
                "ctVal": "1000",
                "ctValCcy": "PEPE",
                "ctType": "linear",
                "state": "live",
            },
        ],
    }


def test_multiplier_alias_maps_to_okx_base() -> None:
    assert okx_base_candidates("1000PEPEUSDT") == ("1000PEPE", "PEPE")
    instruments = parse_okx_swap_instruments(_instrument_payload())
    mapping = map_okx_swap_instruments(
        instruments, ["BTCUSDT", "1000PEPEUSDT", "MISSINGUSDT"]
    )
    assert mapping.loc[
        mapping["bybit_symbol"].eq("1000PEPEUSDT"), "okx_inst_id"
    ].iloc[0] == "PEPE-USDT-SWAP"
    assert mapping.loc[
        mapping["bybit_symbol"].eq("MISSINGUSDT"), "mapping_error"
    ].iloc[0] == "no_live_okx_usdt_swap"
    assert (
        mapping.loc[mapping["bybit_symbol"].eq("BTCUSDT"), "base_currency"].iloc[0]
        == "BTC"
    )


def test_parse_liquidation_computes_notional() -> None:
    instruments = parse_okx_swap_instruments(_instrument_payload())
    mapping = map_okx_swap_instruments(instruments, ["BTCUSDT"]).iloc[0]
    payload = {
        "code": "0",
        "data": [
            {
                "instType": "SWAP",
                "instId": "BTC-USDT-SWAP",
                "instFamily": "BTC-USDT",
                "details": [
                    {
                        "bkLoss": "0",
                        "bkPx": "60000",
                        "posSide": "long",
                        "side": "sell",
                        "sz": "2",
                        "ts": "1784160000000",
                    }
                ],
            }
        ],
    }
    frame = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-16T00:00:00Z")
    )
    assert len(frame) == 1
    assert frame.loc[0, "notional_usd"] == 1_200.0
    assert frame.loc[0, "position_side"] == "long"
    assert str(frame.loc[0, "event_time"].tz) == "UTC"


def test_merge_is_idempotent_and_updates_last_seen(tmp_path) -> None:
    instruments = parse_okx_swap_instruments(_instrument_payload())
    mapping = map_okx_swap_instruments(instruments, ["BTCUSDT"]).iloc[0]
    payload = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "details": [
                    {
                        "bkLoss": "0",
                        "bkPx": "60000",
                        "posSide": "long",
                        "side": "sell",
                        "sz": "2",
                        "ts": "1784160000000",
                    }
                ],
            }
        ],
    }
    first = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-16T00:00:00Z")
    )
    path = tmp_path / "BTCUSDT.parquet"
    first.to_parquet(path, index=False)
    second = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-16T01:00:00Z")
    )
    merged, new_rows = merge_okx_liquidations(second, path)
    assert new_rows == 0
    assert len(merged) == 1
    assert merged.loc[0, "first_seen_at"] == pd.Timestamp("2026-07-16T00:00:00Z")
    assert merged.loc[0, "last_seen_at"] == pd.Timestamp("2026-07-16T01:00:00Z")


def test_merge_repairs_legacy_missing_notional(tmp_path) -> None:
    instruments = parse_okx_swap_instruments(_instrument_payload())
    mapping = map_okx_swap_instruments(instruments, ["BTCUSDT"]).iloc[0]
    payload = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "details": [
                    {
                        "bkLoss": "0",
                        "bkPx": "60000",
                        "posSide": "short",
                        "side": "buy",
                        "sz": "2",
                        "ts": "1784160000000",
                    }
                ],
            }
        ],
    }
    legacy = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-16T00:00:00Z")
    )
    legacy["notional_usd"] = float("nan")
    path = tmp_path / "BTCUSDT.parquet"
    legacy.to_parquet(path, index=False)

    current = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-16T01:00:00Z")
    )
    merged, new_rows = merge_okx_liquidations(current, path)

    assert new_rows == 0
    assert merged.loc[0, "notional_usd"] == 1_200.0


def test_merge_conservatively_repairs_premature_legacy_first_seen(tmp_path) -> None:
    instruments = parse_okx_swap_instruments(_instrument_payload())
    mapping = map_okx_swap_instruments(instruments, ["BTCUSDT"]).iloc[0]
    payload = {
        "code": "0",
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "details": [
                    {
                        "bkLoss": "0",
                        "bkPx": "60000",
                        "posSide": "long",
                        "side": "sell",
                        "sz": "2",
                        "ts": "1784160000000",
                    }
                ],
            }
        ],
    }
    legacy = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-15T23:59:59Z")
    )
    path = tmp_path / "BTCUSDT.parquet"
    legacy.to_parquet(path, index=False)
    current = parse_okx_liquidations(
        payload, mapping, pd.Timestamp("2026-07-16T01:00:00Z")
    )

    merged, _ = merge_okx_liquidations(current, path)

    assert merged.loc[0, "first_seen_at"] == pd.Timestamp("2026-07-16T01:00:00Z")
    assert merged.loc[0, "event_time"] <= merged.loc[0, "first_seen_at"]
