"""Exact Bybit taker-flow states around independently defined long events."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v10_0_exact_taker_flow_alpha")
EVENT_PATH = Path("reports/v0_1/entry_policy_1m_trades.csv")
RAW_ROOT = Path("data/raw/bybit/public_trading_parquet")
CACHE_ROOT = Path("data/processed/v100_exact_taker_flow_1m")
SEED = 20260714
CANDIDATES = (
    "OF1_CONFIRM_LONG",
    "OF2_SELL_ABSORPTION_LONG",
    "OF3_AVOID_BUY_EXHAUSTION",
)
ALL_STATES = (*CANDIDATES, "ALL_COVERED_EVENTS")


@dataclass(frozen=True)
class V100Config:
    event_path: Path = EVENT_PATH
    raw_root: Path = RAW_ROOT
    cache_root: Path = CACHE_ROOT
    report_root: Path = REPORT_ROOT
    cooldown_minutes: int = 60
    random_iterations: int = 500
    bootstrap_iterations: int = 2000
    seed: int = SEED
    cache_workers: int = 4


def load_v100_events(cfg: V100Config = V100Config()) -> pd.DataFrame:
    usecols = ["exchange", "symbol", "path_name", "signal_time"]
    events = pd.read_csv(cfg.event_path, usecols=usecols)
    events["signal_time"] = pd.to_datetime(events["signal_time"], utc=True, errors="coerce")
    events = (
        events.dropna(subset=["symbol", "path_name", "signal_time"])
        .drop_duplicates(usecols)
        .sort_values(["signal_time", "symbol", "path_name"])
        .reset_index(drop=True)
    )
    last_by_symbol: dict[str, pd.Timestamp] = {}
    keep = []
    cooldown = pd.Timedelta(minutes=cfg.cooldown_minutes)
    for row in events.itertuples(index=False):
        last = last_by_symbol.get(str(row.symbol))
        accepted = last is None or pd.Timestamp(row.signal_time) - last >= cooldown
        keep.append(accepted)
        if accepted:
            last_by_symbol[str(row.symbol)] = pd.Timestamp(row.signal_time)
    events = events.loc[keep].copy()
    events["event_id"] = [
        f"{row.path_name}|{row.symbol}|{pd.Timestamp(row.signal_time).isoformat()}"
        for row in events.itertuples(index=False)
    ]
    events["period"] = np.select(
        [
            events["signal_time"].lt(pd.Timestamp("2026-05-01", tz="UTC")),
            events["signal_time"].lt(pd.Timestamp("2026-05-21", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    return events


def _aggregate_trade_file(path: Path) -> pd.DataFrame:
    trades = pd.read_parquet(
        path, columns=["exchange", "symbol", "timestamp", "price", "turnover", "side"]
    )
    if trades.empty:
        return pd.DataFrame()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["turnover"] = pd.to_numeric(trades["turnover"], errors="coerce")
    trades = trades.dropna(subset=["timestamp", "price", "turnover", "side"])
    if trades.empty:
        return pd.DataFrame()
    trades["bar_open_time"] = trades["timestamp"].dt.floor("1min")
    side = trades["side"].astype(str).str.lower()
    trades["buy_turnover"] = trades["turnover"].where(side.eq("buy"), 0.0)
    trades["sell_turnover"] = trades["turnover"].where(side.eq("sell"), 0.0)
    grouped = trades.groupby("bar_open_time", sort=True)
    price = grouped["price"].ohlc()
    out = price.reset_index()
    out["buy_turnover"] = grouped["buy_turnover"].sum().to_numpy()
    out["sell_turnover"] = grouped["sell_turnover"].sum().to_numpy()
    out["turnover"] = grouped["turnover"].sum().to_numpy()
    out["trade_count"] = grouped.size().to_numpy()
    out["exchange"] = str(trades["exchange"].iloc[0])
    out["symbol"] = str(trades["symbol"].iloc[0])
    return out[
        [
            "exchange",
            "symbol",
            "bar_open_time",
            "open",
            "high",
            "low",
            "close",
            "buy_turnover",
            "sell_turnover",
            "turnover",
            "trade_count",
        ]
    ]


def _symbol_source_files(symbol: str, events: pd.DataFrame, cfg: V100Config) -> list[Path]:
    days: set[str] = set()
    for timestamp in events.loc[events["symbol"].eq(symbol), "signal_time"]:
        day = pd.Timestamp(timestamp).floor("D")
        for offset in (-1, 0, 1):
            days.add((day + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"))
    root = cfg.raw_root / symbol
    return [root / f"{symbol}{day}.parquet" for day in sorted(days) if (root / f"{symbol}{day}.parquet").exists()]


def _build_symbol_cache(symbol: str, events: pd.DataFrame, cfg: V100Config) -> Path:
    output = cfg.cache_root / f"{symbol}.parquet"
    sources = _symbol_source_files(symbol, events, cfg)
    if output.exists() and sources:
        newest_source = max(path.stat().st_mtime_ns for path in sources)
        if output.stat().st_mtime_ns >= newest_source:
            return output
    frames = []
    for path in sources:
        frame = _aggregate_trade_file(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no exact trade files for {symbol}")
    data = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["symbol", "bar_open_time"], keep="last")
        .sort_values("bar_open_time")
    )
    ensure_dir(output.parent)
    temporary = output.with_suffix(".tmp.parquet")
    data.to_parquet(temporary, index=False)
    temporary.replace(output)
    return output


def build_v100_minute_cache(
    events: pd.DataFrame,
    cfg: V100Config = V100Config(),
) -> dict[str, Path]:
    ensure_dir(cfg.cache_root)
    outputs: dict[str, Path] = {}
    symbols = sorted(events["symbol"].astype(str).unique())
    with ThreadPoolExecutor(max_workers=max(1, cfg.cache_workers)) as executor:
        futures = {
            executor.submit(_build_symbol_cache, symbol, events, cfg): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            outputs[symbol] = future.result()
    return outputs


def load_v100_minute_cache(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol, path in paths.items():
        frame = pd.read_parquet(path).sort_values("bar_open_time").reset_index(drop=True)
        frame["bar_open_time"] = pd.to_datetime(
            frame["bar_open_time"], utc=True, errors="coerce"
        )
        frames[symbol] = frame.dropna(subset=["bar_open_time"]).reset_index(drop=True)
    return frames


def _window_metrics(frame: pd.DataFrame, end: pd.Timestamp, minutes: int) -> dict[str, float]:
    start = end - pd.Timedelta(minutes=minutes)
    times = frame["bar_open_time"].to_numpy(dtype="datetime64[ns]")
    start_value = np.datetime64(pd.Timestamp(start).tz_convert("UTC").tz_localize(None))
    end_value = np.datetime64(pd.Timestamp(end).tz_convert("UTC").tz_localize(None))
    left = int(np.searchsorted(times, start_value, side="left"))
    right = int(np.searchsorted(times, end_value, side="left"))
    window = frame.iloc[left:right]
    buy = float(pd.to_numeric(window["buy_turnover"], errors="coerce").sum())
    sell = float(pd.to_numeric(window["sell_turnover"], errors="coerce").sum())
    total = buy + sell
    return {
        "populated_minutes": int(len(window)),
        "buy_turnover": buy,
        "sell_turnover": sell,
        "turnover": total,
        "imbalance": (buy - sell) / total if total > 0 else np.nan,
        "return": (
            float(window["close"].iloc[-1] / window["open"].iloc[0] - 1.0)
            if len(window)
            else np.nan
        ),
    }


def _price_at_or_after(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    tolerance_minutes: int = 2,
) -> float:
    times = frame["bar_open_time"].to_numpy(dtype="datetime64[ns]")
    target = np.datetime64(pd.Timestamp(timestamp).tz_convert("UTC").tz_localize(None))
    index = int(np.searchsorted(times, target, side="left"))
    if index >= len(frame):
        return np.nan
    actual = pd.Timestamp(frame.iloc[index]["bar_open_time"])
    if actual - pd.Timestamp(timestamp) > pd.Timedelta(minutes=tolerance_minutes):
        return np.nan
    return float(frame.iloc[index]["open"])


def _extract_event(
    event: pd.Series,
    minute_data: dict[str, pd.DataFrame],
    shift_minutes: int = 0,
) -> dict[str, Any] | None:
    symbol = str(event["symbol"])
    frame = minute_data.get(symbol)
    if frame is None or frame.empty:
        return None
    signal_time = pd.Timestamp(event["signal_time"]) + pd.Timedelta(minutes=shift_minutes)
    five = _window_metrics(frame, signal_time, 5)
    fifteen = _window_metrics(frame, signal_time, 15)
    if five["populated_minutes"] < 4 or fifteen["populated_minutes"] < 12:
        return None
    entry = _price_at_or_after(frame, signal_time)
    if not np.isfinite(entry) or entry <= 0:
        return None
    row: dict[str, Any] = {
        "event_id": str(event["event_id"]),
        "exchange": str(event["exchange"]),
        "symbol": symbol,
        "path_name": str(event["path_name"]),
        "original_signal_time": pd.Timestamp(event["signal_time"]),
        "signal_time": signal_time,
        "period": str(event["period"]),
        "shift_minutes": shift_minutes,
        "entry_price": entry,
        "imbalance_5m": five["imbalance"],
        "imbalance_15m": fifteen["imbalance"],
        "return_5m": five["return"],
        "return_15m": fifteen["return"],
        "turnover_5m": five["turnover"],
        "turnover_15m": fifteen["turnover"],
        "turnover_acceleration": (
            five["turnover"] / (fifteen["turnover"] / 3.0)
            if fifteen["turnover"] > 0
            else np.nan
        ),
        "populated_minutes_5m": five["populated_minutes"],
        "populated_minutes_15m": fifteen["populated_minutes"],
    }
    for horizon in (15, 60, 240):
        exit_price = _price_at_or_after(
            frame, signal_time + pd.Timedelta(minutes=horizon)
        )
        gross = exit_price / entry - 1.0 if np.isfinite(exit_price) else np.nan
        row[f"exit_price_{horizon}m"] = exit_price
        row[f"gross_return_{horizon}m"] = gross
        for cost in (10, 20, 30):
            row[f"net_return_{horizon}m_{cost}bp"] = gross - cost / 10_000.0
    return row


def add_v100_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    confirm = (
        out["imbalance_5m"].ge(0.10)
        & out["imbalance_15m"].ge(0.0)
        & out["return_5m"].gt(0.0)
        & out["turnover_acceleration"].ge(1.0)
    )
    absorption = out["imbalance_5m"].le(-0.10) & out["return_5m"].ge(0.0)
    exhaustion = out["imbalance_5m"].ge(0.10) & out["return_5m"].le(0.0)
    out["OF1_CONFIRM_LONG"] = confirm
    out["OF2_SELL_ABSORPTION_LONG"] = absorption
    out["OF3_AVOID_BUY_EXHAUSTION"] = ~exhaustion
    out["ALL_COVERED_EVENTS"] = True
    return out


def build_v100_event_panel(
    events: pd.DataFrame,
    minute_data: dict[str, pd.DataFrame],
    shift_minutes: int = 0,
) -> pd.DataFrame:
    rows = []
    for _, event in events.iterrows():
        row = _extract_event(event, minute_data, shift_minutes=shift_minutes)
        if row is not None:
            rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    required = [f"gross_return_{horizon}m" for horizon in (15, 60, 240)]
    panel = panel.dropna(subset=required).copy()
    panel["entry_day"] = panel["signal_time"].dt.strftime("%Y-%m-%d")
    return add_v100_states(panel)


def summarize_v100_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", panel)]
    scopes.extend(
        (period, panel[panel["period"].eq(period)])
        for period in ("development", "validation", "holdout")
    )
    for scope, scoped in scopes:
        for path in ("all", "short_squeeze", "momentum_ignition"):
            path_data = scoped if path == "all" else scoped[scoped["path_name"].eq(path)]
            for candidate in ALL_STATES:
                sample = path_data[path_data[candidate].fillna(False).astype(bool)]
                row: dict[str, Any] = {
                    "scope": scope,
                    "path_name": path,
                    "candidate": candidate,
                    "trades": int(len(sample)),
                    "symbols": int(sample["symbol"].nunique()),
                    "active_days": int(sample["entry_day"].nunique()),
                }
                for horizon in (15, 60, 240):
                    gross = pd.to_numeric(sample[f"gross_return_{horizon}m"], errors="coerce")
                    row[f"mean_gross_{horizon}m"] = float(gross.mean())
                    for cost in (10, 20, 30):
                        net = pd.to_numeric(
                            sample[f"net_return_{horizon}m_{cost}bp"], errors="coerce"
                        )
                        row[f"mean_net_{horizon}m_{cost}bp"] = float(net.mean())
                        row[f"sum_net_{horizon}m_{cost}bp"] = float(net.sum())
                net20 = pd.to_numeric(sample["net_return_60m_20bp"], errors="coerce")
                row["median_net_60m_20bp"] = float(net20.median())
                row["win_rate_net_60m_20bp"] = float(net20.gt(0).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_and_concentration(
    panel: pd.DataFrame,
    candidate: str,
    cfg: V100Config,
) -> dict[str, float]:
    sample = panel[panel[candidate].fillna(False).astype(bool)].copy()
    daily = [
        pd.to_numeric(group["net_return_60m_20bp"], errors="coerce").to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + ALL_STATES.index(candidate))
    boot = []
    if daily:
        for _ in range(cfg.bootstrap_iterations):
            chosen = rng.integers(0, len(daily), len(daily))
            values = np.concatenate([daily[index] for index in chosen])
            boot.append(float(np.mean(values)))
    day_sum = sample.groupby("entry_day")["net_return_60m_20bp"].sum()
    positive = pd.to_numeric(day_sum, errors="coerce").clip(lower=0)
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    return {
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
        "max_positive_day_share": concentration,
        "worst_day_sum": float(pd.to_numeric(day_sum, errors="coerce").min()) if len(day_sum) else np.nan,
    }


def _valid_random_anchors(
    event: pd.Series,
    minute_data: dict[str, pd.DataFrame],
) -> list[pd.Timestamp]:
    day = pd.Timestamp(event["signal_time"]).floor("D")
    anchors = pd.date_range(
        day + pd.Timedelta(minutes=15),
        day + pd.Timedelta(hours=23, minutes=45),
        freq="15min",
    )
    valid = []
    for anchor in anchors:
        temporary = event.copy()
        temporary["signal_time"] = anchor
        if _extract_event(temporary, minute_data, shift_minutes=0) is not None:
            valid.append(anchor)
    return valid


def random_v100_controls(
    events: pd.DataFrame,
    minute_data: dict[str, pd.DataFrame],
    cfg: V100Config,
) -> pd.DataFrame:
    anchor_map = {
        str(event["event_id"]): _valid_random_anchors(event, minute_data)
        for _, event in events.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + iteration)
        random_events = events.copy()
        sampled_times = []
        for _, event in random_events.iterrows():
            anchors = anchor_map[str(event["event_id"])]
            sampled_times.append(anchors[int(rng.integers(0, len(anchors)))] if anchors else pd.NaT)
        random_events["signal_time"] = sampled_times
        random_events = random_events.dropna(subset=["signal_time"])
        panel = build_v100_event_panel(random_events, minute_data)
        means = {}
        for candidate in CANDIDATES:
            sample = panel[panel[candidate].fillna(False).astype(bool)]
            means[candidate] = float(
                pd.to_numeric(sample["net_return_60m_20bp"], errors="coerce").mean()
            )
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "trades": len(sample),
                    "mean_net_60m_20bp": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "trades": len(panel),
                "mean_net_60m_20bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def audit_v100(
    panel: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V100Config,
) -> pd.DataFrame:
    shifted_summary = summarize_v100_panel(shifted)
    family = random_controls[random_controls["candidate"].eq("FAMILY_MAX")][
        "mean_net_60m_20bp"
    ]
    rows: list[dict[str, Any]] = []
    eligible_map: dict[str, bool] = {}
    for candidate in CANDIDATES:
        full = summary[
            summary["scope"].eq("all")
            & summary["path_name"].eq("all")
            & summary["candidate"].eq(candidate)
        ].iloc[0]
        validation = summary[
            summary["scope"].eq("validation")
            & summary["path_name"].eq("all")
            & summary["candidate"].eq(candidate)
        ].iloc[0]
        holdout = summary[
            summary["scope"].eq("holdout")
            & summary["path_name"].eq("all")
            & summary["candidate"].eq(candidate)
        ].iloc[0]
        shifted_row = shifted_summary[
            shifted_summary["scope"].eq("all")
            & shifted_summary["path_name"].eq("all")
            & shifted_summary["candidate"].eq(candidate)
        ].iloc[0]
        stability = _bootstrap_and_concentration(panel, candidate, cfg)
        real_mean = float(full.mean_net_60m_20bp)
        family_percentile = float(pd.to_numeric(family, errors="coerce").lt(real_mean).mean())
        path_rows = summary[
            summary["scope"].eq("all")
            & summary["path_name"].isin(["short_squeeze", "momentum_ignition"])
            & summary["candidate"].eq(candidate)
        ]
        min_path = float(path_rows["mean_net_60m_20bp"].min())
        gates = {
            "full_trades_at_least_80": (full.trades >= 80, float(full.trades)),
            "validation_trades_at_least_25": (
                validation.trades >= 25,
                float(validation.trades),
            ),
            "holdout_trades_at_least_25": (holdout.trades >= 25, float(holdout.trades)),
            "six_symbols": (full.symbols >= 6, float(full.symbols)),
            "ten_active_days": (full.active_days >= 10, float(full.active_days)),
            "validation_net20_positive": (
                validation.mean_net_60m_20bp > 0,
                float(validation.mean_net_60m_20bp),
            ),
            "holdout_net20_positive": (
                holdout.mean_net_60m_20bp > 0,
                float(holdout.mean_net_60m_20bp),
            ),
            "full_net30_positive": (full.mean_net_60m_30bp > 0, float(full.mean_net_60m_30bp)),
            "bootstrap_lower_positive": (
                stability["bootstrap_ci_low"] > 0,
                stability["bootstrap_ci_low"],
            ),
            "family_random_p90": (family_percentile >= 0.90, family_percentile),
            "beats_shifted_placebo": (
                real_mean > shifted_row.mean_net_60m_20bp,
                real_mean - float(shifted_row.mean_net_60m_20bp),
            ),
            "both_paths_nonnegative": (min_path >= 0, min_path),
            "positive_day_share_below_35pct": (
                stability["max_positive_day_share"] <= 0.35,
                stability["max_positive_day_share"],
            ),
        }
        eligible = all(passed for passed, _ in gates.values())
        eligible_map[candidate] = eligible
        for check, (passed, value) in gates.items():
            rows.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "passed": bool(passed),
                    "value": value,
                    "eligible": eligible,
                    **stability,
                }
            )
    audit = pd.DataFrame(rows)
    audit["verdict"] = (
        "exact_flow_research_candidate_only"
        if any(eligible_map.values())
        else "reject_exact_taker_flow_overlay"
    )
    return audit


def _write_notes(
    root: Path,
    events: pd.DataFrame,
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    verdict = str(audit["verdict"].iloc[0]) if not audit.empty else "not_run"
    lines = [
        "# v10.0 Exact Taker-Flow State Alpha",
        "",
        f"Status: `{verdict}`. Offline conditional research only.",
        "",
        f"Events after cooldown: {len(events)}; fully covered: {len(panel)}.",
        "",
        "## Primary 60-minute net20",
    ]
    focal = summary[
        summary["path_name"].eq("all")
        & summary["candidate"].isin(ALL_STATES)
        & summary["scope"].isin(["all", "validation", "holdout"])
    ]
    for row in focal.sort_values(["candidate", "scope"]).itertuples(index=False):
        lines.append(
            f"- {row.candidate}/{row.scope}: trades={row.trades}, "
            f"mean_net20={row.mean_net_60m_20bp:.4%}, "
            f"mean_net30={row.mean_net_60m_30bp:.4%}."
        )
    eligible = audit[audit["eligible"].eq(True)]["candidate"].drop_duplicates().tolist()
    lines.extend(
        [
            "",
            "## Decision",
            f"- Eligible: {', '.join(eligible) if eligible else 'none'}.",
            "- Archived days are event-conditioned; no unconditional intraday claim is allowed.",
            "- P2 and all live permissions remain unchanged.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v100_exact_taker_flow_alpha(
    cfg: V100Config = V100Config(),
) -> dict[str, Path]:
    events = load_v100_events(cfg)
    cache_paths = build_v100_minute_cache(events, cfg)
    minute_data = load_v100_minute_cache(cache_paths)
    panel = build_v100_event_panel(events, minute_data)
    shifted = build_v100_event_panel(events, minute_data, shift_minutes=60)
    summary = summarize_v100_panel(panel)
    random_controls = random_v100_controls(events, minute_data, cfg)
    audit = audit_v100(panel, shifted, summary, random_controls, cfg)
    cache_summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "path": path,
                "minute_rows": len(minute_data[symbol]),
                "start": minute_data[symbol]["bar_open_time"].min(),
                "end": minute_data[symbol]["bar_open_time"].max(),
            }
            for symbol, path in sorted(cache_paths.items())
        ]
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "event_panel": root / "event_panel.parquet",
        "shifted_panel": root / "shifted_60m_panel.parquet",
        "summary": root / "candidate_summary.csv",
        "random_controls": root / "random_time_controls.csv",
        "audit": root / "candidate_audit.csv",
        "cache_summary": root / "cache_summary.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    panel.to_parquet(outputs["event_panel"], index=False)
    shifted.to_parquet(outputs["shifted_panel"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    random_controls.to_csv(outputs["random_controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    cache_summary.to_csv(outputs["cache_summary"], index=False)
    _write_notes(root, events, panel, summary, audit)
    return outputs
