"""Frozen causal liquidation features and forward evaluation contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    DATA_ROOT,
    load_v2334_liquidations,
)


REPORT_ROOT = Path("reports/v23_35_liquidation_pre_event_feature_contract")
PREREG_PATH = Path(
    "docs/v2335_liquidation_pre_event_feature_contract_2026_07_17.md"
)
FEATURE_WINDOWS_MINUTES = (5, 15, 60)


@dataclass(frozen=True)
class V2335Contract:
    hourly_minimum_decisions: int = 336
    hourly_minimum_days: int = 14
    q90_minimum_events: int = 30
    q90_minimum_events_per_third: int = 10
    interaction_model_minimum_decisions: int = 1_000
    boosted_model_minimum_decisions: int = 2_000


def _utc_index(values: pd.Series | pd.Index | list[pd.Timestamp]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))
    if index.isna().any():
        raise ValueError("decision times must all be valid UTC timestamps")
    return index.sort_values().unique()


def _window_features(events: pd.DataFrame, decision: pd.Timestamp) -> dict[str, float]:
    sell = events["position_side"].eq("long") & events["liquidation_side"].eq(
        "sell"
    )
    buy = events["position_side"].eq("short") & events["liquidation_side"].eq(
        "buy"
    )
    sell_usd = float(events.loc[sell, "notional_usd"].sum())
    buy_usd = float(events.loc[buy, "notional_usd"].sum())
    total_usd = sell_usd + buy_usd
    symbol_usd = events.groupby("bybit_symbol")["notional_usd"].sum()
    concentration = (
        float(np.square(symbol_usd / total_usd).sum()) if total_usd > 0 else 0.0
    )
    btc_usd = float(
        events.loc[events["bybit_symbol"].eq("BTCUSDT"), "notional_usd"].sum()
    )
    return {
        "event_count": float(len(events)),
        "forced_sell_events": float(sell.sum()),
        "forced_buy_events": float(buy.sum()),
        "forced_sell_usd": sell_usd,
        "forced_buy_usd": buy_usd,
        "total_usd": total_usd,
        "net_forced_buy_usd": buy_usd - sell_usd,
        "log_buy_sell_imbalance": float(np.log1p(buy_usd) - np.log1p(sell_usd)),
        "active_symbols": float(events["bybit_symbol"].nunique()),
        "forced_sell_breadth": float(events.loc[sell, "bybit_symbol"].nunique()),
        "forced_buy_breadth": float(events.loc[buy, "bybit_symbol"].nunique()),
        "btc_notional_share": btc_usd / total_usd if total_usd > 0 else 0.0,
        "symbol_notional_hhi": concentration,
        "max_event_usd": float(events["notional_usd"].max()) if len(events) else 0.0,
        "median_event_usd": (
            float(events["notional_usd"].median()) if len(events) else 0.0
        ),
        "decision_time_epoch": float(decision.timestamp()),
    }


def build_v2335_causal_features(
    events: pd.DataFrame,
    decision_times: pd.Series | pd.Index | list[pd.Timestamp],
    windows_minutes: tuple[int, ...] = FEATURE_WINDOWS_MINUTES,
) -> pd.DataFrame:
    decisions = _utc_index(decision_times)
    local = events.copy()
    for column in ("event_time", "first_seen_at"):
        local[column] = pd.to_datetime(local[column], utc=True, errors="coerce")
    local["notional_usd"] = pd.to_numeric(local["notional_usd"], errors="coerce")
    local = local.dropna(subset=["event_time", "first_seen_at", "notional_usd"])

    rows: list[dict[str, object]] = []
    for decision in decisions:
        known_pre_event = local[
            local["event_time"].lt(decision)
            & local["first_seen_at"].le(decision)
        ]
        row: dict[str, object] = {"decision_time": decision}
        for window in windows_minutes:
            start = decision - pd.Timedelta(minutes=window)
            window_events = known_pre_event[known_pre_event["event_time"].ge(start)]
            for name, value in _window_features(window_events, decision).items():
                if name == "decision_time_epoch":
                    continue
                row[f"liq_{window}m_{name}"] = value
        row["liq_5m_to_15m_notional_share"] = row["liq_5m_total_usd"] / max(
            float(row["liq_15m_total_usd"]), 1.0
        )
        row["liq_15m_to_60m_notional_share"] = row["liq_15m_total_usd"] / max(
            float(row["liq_60m_total_usd"]), 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)


def _write_prereg(path: Path, causal_start: pd.Timestamp) -> None:
    text = [
        "# v23.35 Liquidation Pre-Event Feature Contract",
        "",
        "Status: `frozen_forward_contract_no_outcome_search`.",
        "",
        f"Conservative causal start: `{causal_start.isoformat()}`.",
        "",
        "## Causal inclusion rule",
        "",
        "At decision time `t`, an event is eligible only when "
        "`t-window <= event_time < t` and `first_seen_at <= t`. The initial OKX "
        "snapshot is not treated as historical data before it became known.",
        "",
        "## Frozen feature families",
        "",
        "For 5, 15, and 60 minute windows: forced-sell and forced-buy counts and "
        "notional, net forced-buy pressure, log imbalance, active-symbol breadth, "
        "BTC share, cross-symbol notional HHI, and event-size summaries. Two burst "
        "shares compare 5/15 and 15/60 minute notional. No other transformation may "
        "be introduced after outcomes are inspected in the first evaluation.",
        "",
        "## Forward hypotheses",
        "",
        "1. Broad, high-notional liquidation bursts predict continued BTC realized "
        "volatility over the next 1 and 4 hours.",
        "2. Forced-buy versus forced-sell imbalance predicts the first BTC excursion "
        "side, conditional on a volatility event.",
        "3. Broad low-concentration cascades transmit more strongly to BTC than "
        "single-symbol concentrated liquidations.",
        "4. On frozen q90 book-vacuum events, pre-event liquidation intensity is "
        "tested only as an OCO trigger/avoidance overlay, not as a reselected base rule.",
        "",
        "## Evaluation gates",
        "",
        "The hourly volatility panel requires at least 336 decisions across 14 UTC "
        "days. The q90 overlay requires at least 30 events and 10 events in each "
        "chronological third. Regularized interactions are forbidden below 1,000 "
        "hourly decisions; boosted/tree models are forbidden below 2,000 and then "
        "require nested walk-forward validation against the frozen linear baseline.",
        "",
        "Primary outcomes are next-1h and next-4h BTC log-range/realized absolute "
        "movement. Directional secondary outcomes are first upper-versus-lower "
        "excursion under symmetric barriers. Costs enter only when translating a "
        "validated volatility relation into an executable OCO strategy.",
        "",
        "No PaperLive, live, leverage, remote, application, or order state changes "
        "are authorized by this contract.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2335_contract(
    data_root: Path = DATA_ROOT,
    report_root: Path = REPORT_ROOT,
    prereg_path: Path = PREREG_PATH,
    contract: V2335Contract = V2335Contract(),
) -> dict[str, Path]:
    coverage = json.loads((data_root / "coverage.json").read_text(encoding="utf-8"))
    existing_metadata = report_root / "metadata.json"
    if existing_metadata.exists():
        frozen = json.loads(existing_metadata.read_text(encoding="utf-8"))
        causal_start = pd.Timestamp(frozen["causal_start"])
    else:
        causal_start = pd.Timestamp(coverage["batch_completed_at"])
    events = load_v2334_liquidations(data_root)
    snapshot = build_v2335_causal_features(events, [causal_start])
    root = ensure_dir(report_root)
    paths = {
        "feature_snapshot": root / "causal_feature_snapshot.parquet",
        "checks": root / "contract_checks.csv",
        "metadata": root / "metadata.json",
        "prereg": prereg_path,
    }
    feature_columns = [column for column in snapshot if column != "decision_time"]
    checks = pd.DataFrame(
        [
            {
                "check": "snapshot_decision_at_conservative_causal_start",
                "passed": snapshot.loc[0, "decision_time"] == causal_start,
            },
            {
                "check": "snapshot_features_all_finite",
                "passed": bool(np.isfinite(snapshot[feature_columns]).all().all()),
            },
            {
                "check": "nested_event_counts",
                "passed": bool(
                    snapshot.loc[0, "liq_5m_event_count"]
                    <= snapshot.loc[0, "liq_15m_event_count"]
                    <= snapshot.loc[0, "liq_60m_event_count"]
                ),
            },
            {
                "check": "nested_liquidation_notionals",
                "passed": bool(
                    snapshot.loc[0, "liq_5m_total_usd"]
                    <= snapshot.loc[0, "liq_15m_total_usd"] + 1e-9
                    <= snapshot.loc[0, "liq_60m_total_usd"] + 1e-9
                ),
            },
            {
                "check": "no_outcome_columns",
                "passed": not any(
                    token in column
                    for column in snapshot.columns
                    for token in ("return", "pnl", "label", "target", "future")
                ),
            },
        ]
    )
    snapshot.to_parquet(paths["feature_snapshot"], index=False)
    checks.to_csv(paths["checks"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "status": "frozen_forward_contract_no_outcome_search",
                "causal_start": causal_start.isoformat(),
                "feature_windows_minutes": list(FEATURE_WINDOWS_MINUTES),
                "hourly_minimum_decisions": contract.hourly_minimum_decisions,
                "hourly_minimum_days": contract.hourly_minimum_days,
                "q90_minimum_events": contract.q90_minimum_events,
                "q90_minimum_events_per_third": contract.q90_minimum_events_per_third,
                "interaction_model_minimum_decisions": (
                    contract.interaction_model_minimum_decisions
                ),
                "boosted_model_minimum_decisions": (
                    contract.boosted_model_minimum_decisions
                ),
                "outcomes_loaded": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_prereg(prereg_path, causal_start)
    return paths


__all__ = [
    "FEATURE_WINDOWS_MINUTES",
    "V2335Contract",
    "build_v2335_causal_features",
    "write_v2335_contract",
]
