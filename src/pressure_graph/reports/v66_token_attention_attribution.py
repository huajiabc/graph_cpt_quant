"""v6.6 Token / Pool Attention Attribution.

Attribution pass for token-level DEX attention after v6.5 coverage passes.
This report tests whether same-symbol token/pool attention adds information
inside the mapped P2/CIC/O6 universe.  It is not a trading rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _max_contribution, _month_cap_expectancy
from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config, _read_p2_trades
from pressure_graph.reports.v64_onchain_attention_score import _prepare_trades


REPORT_ROOT = Path("reports/v6_6_token_attention_attribution")
DEFAULT_TOKEN_EVENTS = Path("reports/v6_3_token_pool_dex_attention/token_pool_attention_events.csv")
DEFAULT_TOKEN_MAPPING = Path("reports/v6_3_token_pool_dex_attention/token_pool_mapping.csv")
DEFAULT_MARKET_ATTENTION_DAYS = Path("reports/v6_2_onchain_attention_attribution/onchain_attention_days.csv")


@dataclass(frozen=True)
class V66Config:
    report_root: Path = REPORT_ROOT
    token_events_path: Path = DEFAULT_TOKEN_EVENTS
    token_mapping_path: Path = DEFAULT_TOKEN_MAPPING
    market_attention_days_path: Path = DEFAULT_MARKET_ATTENTION_DAYS
    v61: V61Config = V61Config()
    prior_windows: tuple[str, ...] = ("6h", "12h", "24h", "48h")
    main_prior_window: str = "24h"
    random_trials: int = 100
    random_seed: int = 660


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _parse_window(value: str) -> pd.Timedelta:
    return pd.Timedelta(value)


def _read_mapping(path: Path) -> pd.DataFrame:
    mapping = _read_csv(path)
    if mapping.empty or "cex_symbol" not in mapping.columns:
        return pd.DataFrame()
    out = mapping.copy()
    out["cex_symbol"] = out["cex_symbol"].fillna("").astype(str).str.upper()
    out["mapping_confidence"] = out.get("mapping_confidence", "").fillna("").astype(str).str.upper()
    out = out[out["mapping_confidence"].isin(["A", "B"])].copy()
    return out.drop_duplicates("cex_symbol").reset_index(drop=True)


def _read_token_events(path: Path, mapped_symbols: set[str]) -> pd.DataFrame:
    events = _read_csv(path)
    if events.empty or "cex_symbol" not in events.columns or "event_available_time" not in events.columns:
        return pd.DataFrame()
    out = events.copy()
    out["cex_symbol"] = out["cex_symbol"].fillna("").astype(str).str.upper()
    out = out[out["cex_symbol"].isin(mapped_symbols)].copy()
    out["event_available_time"] = pd.to_datetime(out["event_available_time"], utc=True, errors="coerce")
    out["event_time"] = pd.to_datetime(out.get("event_time"), utc=True, errors="coerce")
    out["event_date"] = out["event_available_time"].dt.floor("D")
    out["event_type"] = out.get("event_type", "unknown").fillna("unknown").astype(str)
    out["source"] = out.get("source", "unknown").fillna("unknown").astype(str)
    return out.dropna(subset=["cex_symbol", "event_available_time"]).sort_values(["cex_symbol", "event_available_time"]).reset_index(drop=True)


def _read_market_days(path: Path) -> set[pd.Timestamp]:
    days = _read_csv(path)
    if days.empty:
        return set()
    col = "event_date" if "event_date" in days.columns else days.columns[0]
    series = pd.to_datetime(days[col], utc=True, errors="coerce").dt.floor("D").dropna()
    return set(series.tolist())


def _prepare_p2_trades(cfg: V66Config, mapped_symbols: set[str]) -> pd.DataFrame:
    trades = _prepare_trades(_read_p2_trades(cfg.v61.trade_cache_path))
    if trades.empty:
        return trades
    out = trades.copy()
    out["symbol"] = out["symbol"].fillna("").astype(str).str.upper()
    out = out[out["symbol"].isin(mapped_symbols)].copy()
    out["candidate"] = out["candidate"].fillna("").astype(str)
    out["is_cic1"] = out["candidate"].eq("CIC1_beta_extreme")
    out["is_cic2"] = out["candidate"].eq("CIC2_beta_broad")
    out["is_o6"] = _num(out, "burst_count_so_far").ge(9)
    out["month"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    out = out.dropna(subset=["entry_time", "net20"]).reset_index(drop=True)
    out["trade_id"] = (
        out["symbol"].astype(str)
        + "|"
        + out["candidate"].astype(str)
        + "|"
        + out["entry_time"].astype(str)
        + "|"
        + out.index.astype(str)
    )
    return out


def _event_times_by_symbol(events: pd.DataFrame) -> dict[str, np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    if events.empty:
        return grouped
    for symbol, group in events.groupby("cex_symbol", sort=False):
        times = pd.to_datetime(group["event_available_time"], utc=True, errors="coerce").dropna().sort_values()
        grouped[str(symbol)] = times.astype("int64").to_numpy()
    return grouped


def _attach_prior_flags(trades: pd.DataFrame, events: pd.DataFrame, windows: tuple[str, ...], prefix: str = "token") -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    times_by_symbol = _event_times_by_symbol(events)
    entries = pd.to_datetime(out["entry_time"], utc=True, errors="coerce").astype("int64").to_numpy()
    symbols = out["symbol"].astype(str).to_numpy()
    for window_text in windows:
        delta_ns = int(_parse_window(window_text).value)
        flags = np.zeros(len(out), dtype=bool)
        counts = np.zeros(len(out), dtype=int)
        for idx, (symbol, entry_ns) in enumerate(zip(symbols, entries, strict=False)):
            times = times_by_symbol.get(symbol)
            if times is None or len(times) == 0 or pd.isna(entry_ns):
                continue
            left = np.searchsorted(times, entry_ns - delta_ns, side="right")
            right = np.searchsorted(times, entry_ns, side="right")
            count = max(int(right - left), 0)
            counts[idx] = count
            flags[idx] = count > 0
        out[f"{prefix}_prior_{window_text}"] = flags
        out[f"{prefix}_prior_{window_text}_count"] = counts
    return out


def _sample_metrics(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "trades": 0,
            "net20": np.nan,
            "hit_rate": np.nan,
            "month_cap35_net20": np.nan,
            "max_symbol_contribution": np.nan,
        }
    local = sample.copy()
    local["net_return"] = _num(local, "net20")
    if "month" not in local.columns:
        local["month"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    return {
        "trades": int(len(local)),
        "net20": float(_num(local, "net20").mean()),
        "hit_rate": float(_num(local, "net20").gt(0).mean()),
        "month_cap35_net20": _month_cap_expectancy(local),
        "max_symbol_contribution": _max_contribution(local, "symbol"),
    }


def _modules(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("P2_all", frame),
        ("CIC1", frame[frame["is_cic1"]]),
        ("CIC2", frame[frame["is_cic2"]]),
        ("O6_late9", frame[frame["is_o6"]]),
    ]


def _fusion_summary(scored: pd.DataFrame, cfg: V66Config) -> pd.DataFrame:
    rows = []
    if scored.empty:
        return pd.DataFrame()
    for window in cfg.prior_windows:
        prior_col = f"token_prior_{window}"
        for module, sample in _modules(scored):
            prior = sample[sample[prior_col]]
            no_prior = sample[~sample[prior_col]]
            prior_metrics = _sample_metrics(prior)
            no_metrics = _sample_metrics(no_prior)
            rows.append(
                {
                    "module": module,
                    "lookback_window": window,
                    "prior_trades": prior_metrics["trades"],
                    "prior_net20": prior_metrics["net20"],
                    "prior_hit_rate": prior_metrics["hit_rate"],
                    "prior_month_cap35_net20": prior_metrics["month_cap35_net20"],
                    "prior_max_symbol_contribution": prior_metrics["max_symbol_contribution"],
                    "no_prior_trades": no_metrics["trades"],
                    "no_prior_net20": no_metrics["net20"],
                    "no_prior_hit_rate": no_metrics["hit_rate"],
                    "no_prior_month_cap35_net20": no_metrics["month_cap35_net20"],
                    "prior_minus_no_prior_net20": prior_metrics["net20"] - no_metrics["net20"]
                    if pd.notna(prior_metrics["net20"]) and pd.notna(no_metrics["net20"])
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _event_type_summary(scored: pd.DataFrame, events: pd.DataFrame, cfg: V66Config) -> pd.DataFrame:
    if scored.empty or events.empty:
        return pd.DataFrame()
    window = _parse_window(cfg.main_prior_window)
    rows = []
    events_by_symbol = {symbol: group.sort_values("event_available_time") for symbol, group in events.groupby("cex_symbol", sort=False)}
    for trade in scored.itertuples(index=False):
        symbol = str(trade.symbol)
        entry = pd.Timestamp(trade.entry_time)
        local = events_by_symbol.get(symbol, pd.DataFrame())
        if local.empty:
            continue
        prior = local[local["event_available_time"].le(entry) & local["event_available_time"].gt(entry - window)]
        for event_type, group in prior.groupby(["event_type", "source"], sort=False):
            rows.append(
                {
                    "trade_id": getattr(trade, "trade_id", ""),
                    "symbol": symbol,
                    "candidate": getattr(trade, "candidate", ""),
                    "is_o6": bool(getattr(trade, "is_o6", False)),
                    "event_type": event_type[0],
                    "source": event_type[1],
                    "event_count": int(len(group)),
                    "net20": float(getattr(trade, "net20")),
                    "month": getattr(trade, "month", ""),
                }
            )
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return pd.DataFrame()
    summary = []
    for (event_type, source), group in ledger.groupby(["event_type", "source"], sort=False):
        dedup = group.drop_duplicates("trade_id")
        summary.append({"event_type": event_type, "source": source, "prior_trade_events": int(len(group)), **_sample_metrics(dedup)})
    return pd.DataFrame(summary).sort_values("trades", ascending=False).reset_index(drop=True)


def _random_same_token_events(events: pd.DataFrame, trades: pd.DataFrame, rng: np.random.Generator, window: pd.Timedelta) -> pd.DataFrame:
    rows = []
    if events.empty or trades.empty:
        return pd.DataFrame()
    for symbol, group in events.groupby("cex_symbol", sort=False):
        local_trades = trades[trades["symbol"].eq(symbol)]
        if local_trades.empty:
            continue
        start = pd.Timestamp(local_trades["entry_time"].min()) - window
        end = pd.Timestamp(local_trades["entry_time"].max())
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        start_ns = int(start.value)
        end_ns = int(end.value)
        random_ns = rng.integers(start_ns, end_ns + 1, size=len(group))
        for ts_ns in random_ns:
            rows.append({"cex_symbol": symbol, "event_available_time": pd.to_datetime(ts_ns, utc=True), "event_type": "same_token_random_time", "source": "random_control"})
    return pd.DataFrame(rows)


def _random_symbol_events(events: pd.DataFrame, mapping: pd.DataFrame, rng: np.random.Generator, same_chain: bool) -> pd.DataFrame:
    if events.empty or mapping.empty:
        return pd.DataFrame()
    rows = []
    symbol_chain = mapping.set_index("cex_symbol")["chain"].fillna("").astype(str).to_dict() if "chain" in mapping.columns else {}
    symbols = sorted(mapping["cex_symbol"].astype(str).unique())
    for event in events.itertuples(index=False):
        source_symbol = str(event.cex_symbol)
        pool = [sym for sym in symbols if sym != source_symbol]
        if same_chain:
            chain = symbol_chain.get(source_symbol, "")
            pool = [sym for sym in pool if symbol_chain.get(sym, "") == chain]
        if not pool:
            continue
        target = pool[int(rng.integers(0, len(pool)))]
        rows.append(
            {
                "cex_symbol": target,
                "event_available_time": getattr(event, "event_available_time"),
                "event_type": "same_chain_random_token" if same_chain else "same_day_random_token",
                "source": "random_control",
            }
        )
    return pd.DataFrame(rows)


def _control_distribution(scored: pd.DataFrame, events: pd.DataFrame, mapping: pd.DataFrame, cfg: V66Config) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    window = cfg.main_prior_window
    window_delta = _parse_window(window)
    prior_col = f"token_prior_{window}"
    actual = scored[scored[prior_col]]
    for module, actual_sample in _modules(actual):
        actual_metrics = _sample_metrics(actual_sample)
        for control_name in ("same_token_random_time", "same_day_random_token", "same_chain_random_token"):
            nets = []
            counts = []
            caps = []
            for _ in range(cfg.random_trials):
                if control_name == "same_token_random_time":
                    random_events = _random_same_token_events(events, scored, rng, window_delta)
                elif control_name == "same_day_random_token":
                    random_events = _random_symbol_events(events, mapping, rng, same_chain=False)
                else:
                    random_events = _random_symbol_events(events, mapping, rng, same_chain=True)
                random_scored = _attach_prior_flags(scored, random_events, (window,), prefix="random")
                sample = dict(_modules(random_scored))[module]
                selected = sample[sample[f"random_prior_{window}"]]
                metrics = _sample_metrics(selected)
                nets.append(metrics["net20"])
                counts.append(metrics["trades"])
                caps.append(metrics["month_cap35_net20"])
            net_arr = np.asarray(nets, dtype=float)
            net_arr = net_arr[np.isfinite(net_arr)]
            count_arr = np.asarray(counts, dtype=float)
            cap_arr = np.asarray(caps, dtype=float)
            cap_arr = cap_arr[np.isfinite(cap_arr)]
            rows.append(
                {
                    "module": module,
                    "lookback_window": window,
                    "control": control_name,
                    "actual_prior_trades": actual_metrics["trades"],
                    "actual_prior_net20": actual_metrics["net20"],
                    "actual_prior_month_cap35_net20": actual_metrics["month_cap35_net20"],
                    "random_trials": cfg.random_trials,
                    "random_median_trades": float(np.nanmedian(count_arr)) if len(count_arr) else np.nan,
                    "random_median_net20": float(np.nanmedian(net_arr)) if len(net_arr) else np.nan,
                    "random_p75_net20": float(np.nanpercentile(net_arr, 75)) if len(net_arr) else np.nan,
                    "random_p90_net20": float(np.nanpercentile(net_arr, 90)) if len(net_arr) else np.nan,
                    "random_median_month_cap35_net20": float(np.nanmedian(cap_arr)) if len(cap_arr) else np.nan,
                    "actual_percentile": float((net_arr <= actual_metrics["net20"]).mean())
                    if len(net_arr) and pd.notna(actual_metrics["net20"])
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _market_control(scored: pd.DataFrame, events: pd.DataFrame, market_days: set[pd.Timestamp], cfg: V66Config) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    out = scored.copy()
    window = cfg.main_prior_window
    prior_col = f"token_prior_{window}"
    out["entry_day"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce").dt.floor("D")
    out["market_attention_day"] = out["entry_day"].isin(market_days)
    rows = []
    for module, sample in _modules(out):
        buckets = {
            "token_prior_market_day": sample[sample[prior_col] & sample["market_attention_day"]],
            "token_prior_no_market_day": sample[sample[prior_col] & ~sample["market_attention_day"]],
            "no_token_market_day": sample[~sample[prior_col] & sample["market_attention_day"]],
            "no_token_no_market_day": sample[~sample[prior_col] & ~sample["market_attention_day"]],
        }
        for bucket, group in buckets.items():
            rows.append({"module": module, "bucket": bucket, **_sample_metrics(group)})

    if events.empty or not market_days:
        return pd.DataFrame(rows)
    local_events = events.copy()
    local_events["event_day"] = pd.to_datetime(local_events["event_available_time"], utc=True, errors="coerce").dt.floor("D")
    non_market_events = local_events[~local_events["event_day"].isin(market_days)].copy()
    non_market_scored = _attach_prior_flags(scored.drop(columns=[col for col in scored.columns if col.startswith("non_market_token_prior_")], errors="ignore"), non_market_events, (window,), prefix="non_market_token")
    flag = f"non_market_token_prior_{window}"
    for module, sample in _modules(non_market_scored):
        rows.append({"module": module, "bucket": "token_prior_excluding_market_attention_days", **_sample_metrics(sample[sample[flag]])})
    return pd.DataFrame(rows)


def _leave_one_month(scored: pd.DataFrame, cfg: V66Config) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    window = cfg.main_prior_window
    prior_col = f"token_prior_{window}"
    rows = []
    months = sorted(scored["month"].dropna().astype(str).unique())
    for excluded in months:
        local = scored[~scored["month"].astype(str).eq(excluded)].copy()
        for module, sample in _modules(local):
            prior = sample[sample[prior_col]]
            no_prior = sample[~sample[prior_col]]
            pm = _sample_metrics(prior)
            nm = _sample_metrics(no_prior)
            rows.append(
                {
                    "excluded_month": excluded,
                    "module": module,
                    "prior_trades": pm["trades"],
                    "prior_net20": pm["net20"],
                    "no_prior_trades": nm["trades"],
                    "no_prior_net20": nm["net20"],
                    "prior_minus_no_prior_net20": pm["net20"] - nm["net20"] if pd.notna(pm["net20"]) and pd.notna(nm["net20"]) else np.nan,
                    "prior_month_cap35_net20": pm["month_cap35_net20"],
                }
            )
    return pd.DataFrame(rows)


def _token_response_curve(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if events.empty or trades.empty:
        return pd.DataFrame()
    horizons = {"6h": pd.Timedelta(hours=6), "12h": pd.Timedelta(hours=12), "24h": pd.Timedelta(hours=24), "48h": pd.Timedelta(hours=48)}
    rows = []
    trades_by_symbol = {symbol: group.sort_values("entry_time") for symbol, group in trades.groupby("symbol", sort=False)}
    for (event_type, source), group in events.groupby(["event_type", "source"], sort=False):
        for horizon_name, horizon in horizons.items():
            matched_trades = []
            event_with_trade = 0
            for event in group.itertuples(index=False):
                symbol = str(event.cex_symbol)
                start = pd.Timestamp(event.event_available_time)
                local = trades_by_symbol.get(symbol, pd.DataFrame())
                if local.empty:
                    continue
                sample = local[local["entry_time"].gt(start) & local["entry_time"].le(start + horizon)]
                if len(sample):
                    event_with_trade += 1
                    matched_trades.append(sample)
            matched = pd.concat(matched_trades, ignore_index=True).drop_duplicates("trade_id") if matched_trades else pd.DataFrame()
            rows.append(
                {
                    "event_type": event_type,
                    "source": source,
                    "horizon": horizon_name,
                    "events": int(len(group)),
                    "events_with_p2_trade": int(event_with_trade),
                    "event_to_p2_trade_rate": float(event_with_trade / len(group)) if len(group) else np.nan,
                    **_sample_metrics(matched),
                }
            )
    return pd.DataFrame(rows)


def _coverage_summary(mapping: pd.DataFrame, events: pd.DataFrame, trades: pd.DataFrame, scored: pd.DataFrame, cfg: V66Config) -> pd.DataFrame:
    prior_col = f"token_prior_{cfg.main_prior_window}"
    return pd.DataFrame(
        [
            {"dataset": "mapped_A_B_symbols", "rows": int(mapping["cex_symbol"].nunique()) if not mapping.empty else 0, "status": "reference"},
            {"dataset": "token_events", "rows": int(len(events)), "status": "ok" if len(events) else "missing"},
            {"dataset": "mapped_p2_trades", "rows": int(len(trades)), "status": "ok" if len(trades) else "missing"},
            {"dataset": f"token_prior_{cfg.main_prior_window}_trades", "rows": int(scored[prior_col].sum()) if not scored.empty and prior_col in scored.columns else 0, "status": "coverage"},
        ]
    )


def _write_notes(
    path: Path,
    coverage: pd.DataFrame,
    fusion: pd.DataFrame,
    controls: pd.DataFrame,
    market_control: pd.DataFrame,
    cfg: V66Config,
) -> None:
    lines = [
        "# v6.6 Token / Pool Attention Attribution",
        "",
        "Status: attribution audit only. No strategy, gate, selector, shadow, or real-live permission is changed.",
        "",
    ]
    cov = coverage.set_index("dataset") if not coverage.empty else pd.DataFrame()
    for dataset in ("mapped_A_B_symbols", "token_events", "mapped_p2_trades", f"token_prior_{cfg.main_prior_window}_trades"):
        if dataset in cov.index:
            lines.append(f"- {dataset}: {int(cov.loc[dataset, 'rows'])}.")
    main = fusion[(fusion["lookback_window"].astype(str).eq(cfg.main_prior_window))] if not fusion.empty else pd.DataFrame()
    if not main.empty:
        lines.extend(["", "Main 24h fusion:"])
        for row in main.sort_values("module").to_dict("records"):
            lines.append(
                f"- {row['module']}: prior={int(row['prior_trades'])} net20={row['prior_net20']:.4%}, "
                f"without={int(row['no_prior_trades'])} net20={row['no_prior_net20']:.4%}, "
                f"lift={row['prior_minus_no_prior_net20']:.4%}."
            )
    if not controls.empty:
        p2 = controls[(controls["module"].eq("P2_all"))]
        if not p2.empty:
            best = p2.sort_values("actual_percentile", ascending=False).iloc[0]
            lines.extend(
                [
                    "",
                    f"Best P2 random-control percentile: {best['control']} percentile={best['actual_percentile']:.2f}, "
                    f"actual_net20={best['actual_prior_net20']:.4%}, random_p75={best['random_p75_net20']:.4%}.",
                ]
            )
    if not market_control.empty:
        sample = market_control[(market_control["module"].eq("P2_all")) & (market_control["bucket"].eq("token_prior_excluding_market_attention_days"))]
        if not sample.empty:
            row = sample.iloc[0]
            lines.append(
                f"Token-prior excluding market-level attention days: trades={int(row['trades'])}, net20={row['net20']:.4%}."
            )
    lines.extend(
        [
            "",
            "Promotion guardrails:",
            "- Token-prior must beat same-token random time and random-token controls, not only no-prior.",
            "- Any usable context must survive leave-one-month and month-cap checks.",
            "- This report can only promote a future shadow context after forward validation; it cannot create live permissions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v66_token_attention_attribution(cfg: V66Config | None = None) -> dict[str, Path]:
    cfg = cfg or V66Config()
    report_root = ensure_dir(cfg.report_root)
    mapping = _read_mapping(cfg.token_mapping_path)
    mapped_symbols = set(mapping["cex_symbol"].astype(str)) if not mapping.empty else set()
    events = _read_token_events(cfg.token_events_path, mapped_symbols)
    trades = _prepare_p2_trades(cfg, mapped_symbols)
    scored = _attach_prior_flags(trades, events, cfg.prior_windows, prefix="token")
    market_days = _read_market_days(cfg.market_attention_days_path)

    coverage = _coverage_summary(mapping, events, trades, scored, cfg)
    fusion = _fusion_summary(scored, cfg)
    event_types = _event_type_summary(scored, events, cfg)
    controls = _control_distribution(scored, events, mapping, cfg)
    market = _market_control(scored, events, market_days, cfg)
    leave_month = _leave_one_month(scored, cfg)
    response = _token_response_curve(events, scored)

    outputs = {
        "coverage": report_root / "token_attention_coverage.csv",
        "scored_trade_ledger": report_root / "token_attention_scored_trade_ledger.csv",
        "token_cic_o6_fusion_summary": report_root / "token_cic_o6_fusion_summary.csv",
        "token_event_type_summary": report_root / "token_event_type_summary.csv",
        "random_control_summary": report_root / "random_control_summary.csv",
        "market_level_control_summary": report_root / "market_level_control_summary.csv",
        "leave_one_month": report_root / "leave_one_month.csv",
        "token_to_p2_response_curve": report_root / "token_to_p2_response_curve.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["coverage"], index=False)
    scored.to_csv(outputs["scored_trade_ledger"], index=False)
    fusion.to_csv(outputs["token_cic_o6_fusion_summary"], index=False)
    event_types.to_csv(outputs["token_event_type_summary"], index=False)
    controls.to_csv(outputs["random_control_summary"], index=False)
    market.to_csv(outputs["market_level_control_summary"], index=False)
    leave_month.to_csv(outputs["leave_one_month"], index=False)
    response.to_csv(outputs["token_to_p2_response_curve"], index=False)
    _write_notes(outputs["candidate_notes"], coverage, fusion, controls, market, cfg)
    return outputs
