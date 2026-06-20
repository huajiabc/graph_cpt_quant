"""v6.5 Token / Pool Coverage Expansion.

Coverage sprint for token-level DEX attention.  This report does not test or
promote alpha.  It answers whether the P2/CIC/O6 symbols that actually drive
the long stack have reliable token/pool mappings and whether existing token
pool events overlap pre-trade windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config, _read_p2_trades
from pressure_graph.reports.v64_onchain_attention_score import _prepare_trades


REPORT_ROOT = Path("reports/v6_5_token_pool_coverage_expansion")
DEFAULT_TOKEN_MAPPING = Path("reports/v6_3_token_pool_dex_attention/token_pool_mapping.csv")
DEFAULT_TOKEN_EVENTS = Path("reports/v6_3_token_pool_dex_attention/token_pool_attention_events.csv")
DEFAULT_OHLCV_COVERAGE = Path("reports/v6_3_token_pool_dex_attention/token_pool_ohlcv_coverage.csv")


@dataclass(frozen=True)
class V65Config:
    report_root: Path = REPORT_ROOT
    token_mapping_path: Path = DEFAULT_TOKEN_MAPPING
    token_events_path: Path = DEFAULT_TOKEN_EVENTS
    ohlcv_coverage_path: Path = DEFAULT_OHLCV_COVERAGE
    v61: V61Config = V61Config()
    prior_windows: tuple[str, ...] = ("6h", "12h", "24h", "48h")
    main_prior_window: str = "24h"
    target_mapped_trade_coverage_low: float = 0.40
    target_mapped_trade_coverage_high: float = 0.60
    target_token_prior_p2_trades: int = 100
    target_token_prior_o6_trades: int = 30


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[col].fillna("").astype(str)


def _parse_window(value: str) -> pd.Timedelta:
    return pd.Timedelta(value)


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _confidence_rank(value: str) -> int:
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    return order.get(str(value).upper(), 4)


def _read_mapping(path: Path) -> pd.DataFrame:
    mapping = _read_csv_optional(path)
    if mapping.empty:
        return pd.DataFrame()
    if "cex_symbol" not in mapping.columns and "symbol" in mapping.columns:
        mapping = mapping.rename(columns={"symbol": "cex_symbol"})
    if "cex_symbol" not in mapping.columns:
        return pd.DataFrame()
    out = mapping.copy()
    out["cex_symbol"] = _text(out, "cex_symbol").str.upper()
    out["mapping_confidence"] = _text(out, "mapping_confidence").str.upper().replace("", "D")
    out["confidence_rank"] = out["mapping_confidence"].map(_confidence_rank).fillna(4).astype(int)
    out["pool_24h_volume_usd_num"] = _num(out, "pool_24h_volume_usd").fillna(-1)
    out["pool_liquidity_usd_num"] = _num(out, "pool_liquidity_usd").fillna(-1)
    out = out.sort_values(
        ["cex_symbol", "confidence_rank", "pool_24h_volume_usd_num", "pool_liquidity_usd_num"],
        ascending=[True, True, False, False],
    )
    out = out.drop_duplicates("cex_symbol", keep="first").reset_index(drop=True)
    return out


def _read_token_events(path: Path) -> pd.DataFrame:
    events = _read_csv_optional(path)
    if events.empty or "cex_symbol" not in events.columns or "event_available_time" not in events.columns:
        return pd.DataFrame()
    out = events.copy()
    out["cex_symbol"] = _text(out, "cex_symbol").str.upper()
    out["event_available_time"] = pd.to_datetime(out["event_available_time"], utc=True, errors="coerce")
    out["event_time"] = pd.to_datetime(out.get("event_time"), utc=True, errors="coerce")
    return out.dropna(subset=["cex_symbol", "event_available_time"]).sort_values(["cex_symbol", "event_available_time"]).reset_index(drop=True)


def _prepare_p2_trades(cfg: V65Config) -> pd.DataFrame:
    trades = _prepare_trades(_read_p2_trades(cfg.v61.trade_cache_path))
    if trades.empty:
        return trades
    out = trades.copy()
    out["symbol"] = _text(out, "symbol").str.upper()
    out["candidate"] = _text(out, "candidate")
    out["is_cic1"] = out["candidate"].eq("CIC1_beta_extreme")
    out["is_cic2"] = out["candidate"].eq("CIC2_beta_broad")
    out["is_o6"] = _num(out, "burst_count_so_far").ge(9)
    return out


def _mapping_status(row: pd.Series) -> str:
    confidence = str(row.get("mapping_confidence", "") or "").upper()
    token = str(row.get("token_address", "") or "").strip()
    pool = str(row.get("pool_address", "") or "").strip()
    if confidence in {"", "NAN", "NONE", "<NA>"}:
        return "missing_mapping"
    if confidence in {"A", "B"} and token and pool:
        return "mapped_A_B"
    if confidence in {"A", "B"}:
        return "mapped_A_B_incomplete"
    if confidence == "C":
        return "mapped_C_review"
    if confidence == "D":
        return "unmapped_D"
    return "mapping_unknown"


def _add_dex_relevance(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return mapping
    out = mapping.copy()
    usable = out["mapping_confidence"].astype(str).str.upper().isin(["A", "B"])
    volume = _num(out, "pool_24h_volume_usd")
    liquidity = _num(out, "pool_liquidity_usd")
    vol_rank = pd.Series(0.0, index=out.index, dtype="float64")
    liq_rank = pd.Series(0.0, index=out.index, dtype="float64")
    if usable.any():
        vol_rank.loc[usable] = volume.loc[usable].rank(pct=True).fillna(0.0)
        liq_rank.loc[usable] = liquidity.loc[usable].rank(pct=True).fillna(0.0)
    out["dex_relevance_score"] = (0.6 * vol_rank + 0.4 * liq_rank).round(6)
    out["dex_relevance_bucket"] = np.select(
        [
            out["dex_relevance_score"].ge(0.75),
            out["dex_relevance_score"].ge(0.40),
            out["dex_relevance_score"].gt(0),
        ],
        ["high", "medium", "low"],
        default="none",
    )
    return out


def _trade_weighted_mapping_coverage(trades: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "p2_trades",
                "cic1_trades",
                "cic2_trades",
                "o6_trades",
                "net20_contribution",
                "mapping_status",
                "chain",
                "token_address",
                "top_pool",
                "mapping_confidence",
            ]
        )
    grouped = trades.groupby("symbol", as_index=False).agg(
        p2_trades=("symbol", "size"),
        cic1_trades=("is_cic1", "sum"),
        cic2_trades=("is_cic2", "sum"),
        o6_trades=("is_o6", "sum"),
        net20_contribution=("net20", "sum"),
        net20_mean=("net20", "mean"),
    )
    merged = grouped.merge(mapping, left_on="symbol", right_on="cex_symbol", how="left")
    for col in ("mapping_confidence", "chain", "token_address", "pool_address", "pool_dex", "pool_quote_token", "mapping_source", "coingecko_id"):
        if col not in merged.columns:
            merged[col] = ""
        else:
            merged[col] = merged[col].fillna("").astype(str)
    for col in ("pool_liquidity_usd", "pool_24h_volume_usd", "dex_relevance_score"):
        if col not in merged.columns:
            merged[col] = np.nan
    merged["mapping_status"] = merged.apply(_mapping_status, axis=1)
    merged["top_pool"] = merged["pool_address"]
    cols = [
        "symbol",
        "p2_trades",
        "cic1_trades",
        "cic2_trades",
        "o6_trades",
        "net20_contribution",
        "net20_mean",
        "mapping_status",
        "chain",
        "token_address",
        "top_pool",
        "pool_dex",
        "pool_quote_token",
        "pool_liquidity_usd",
        "pool_24h_volume_usd",
        "dex_relevance_score",
        "dex_relevance_bucket",
        "mapping_confidence",
        "mapping_source",
        "coingecko_id",
    ]
    return merged[cols].sort_values(["p2_trades", "o6_trades", "net20_contribution"], ascending=[False, False, False]).reset_index(drop=True)


def _missing_mapping_priority(coverage: pd.DataFrame, token_events: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    event_symbols = set(token_events.get("cex_symbol", pd.Series(dtype=str)).astype(str).str.upper()) if not token_events.empty else set()
    out = coverage.copy()
    out["has_token_events"] = out["symbol"].isin(event_symbols)
    out["has_coingecko_id"] = _text(out, "coingecko_id").ne("")
    out["has_known_contract"] = _text(out, "token_address").ne("")
    out["possible_chains"] = _text(out, "chain")
    out["manual_review_needed"] = ~out["mapping_status"].eq("mapped_A_B")
    out["data_backfill_needed"] = out["mapping_status"].eq("mapped_A_B") & ~out["has_token_events"]
    out["coverage_gap_reason"] = np.select(
        [
            out["mapping_status"].ne("mapped_A_B"),
            out["data_backfill_needed"],
        ],
        [
            "mapping_missing_or_not_A_B",
            "mapped_but_no_token_events",
        ],
        default="covered_by_mapping_and_events",
    )
    out["trade_count"] = _num(out, "p2_trades").fillna(0).astype(int)
    out["o6_trade_count"] = _num(out, "o6_trades").fillna(0).astype(int)
    out["historical_net20"] = _num(out, "net20_contribution")
    out["priority_score"] = (
        _num(out, "p2_trades").fillna(0)
        + 1.5 * _num(out, "o6_trades").fillna(0)
        + 50.0 * _num(out, "net20_contribution").abs().fillna(0)
    )
    queue = out[out["coverage_gap_reason"].ne("covered_by_mapping_and_events")].copy()
    cols = [
        "symbol",
        "trade_count",
        "o6_trade_count",
        "historical_net20",
        "has_coingecko_id",
        "has_known_contract",
        "possible_chains",
        "manual_review_needed",
        "data_backfill_needed",
        "mapping_status",
        "mapping_confidence",
        "token_address",
        "top_pool",
        "coverage_gap_reason",
        "priority_score",
    ]
    return queue[cols].sort_values("priority_score", ascending=False).reset_index(drop=True)


def _prior_flags(trades: pd.DataFrame, token_events: pd.DataFrame, windows: tuple[str, ...]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    events_by_symbol = {symbol: group.sort_values("event_available_time") for symbol, group in token_events.groupby("cex_symbol", sort=False)} if not token_events.empty else {}
    rows: list[dict[str, Any]] = []
    for trade in trades.itertuples(index=False):
        entry = pd.Timestamp(trade.entry_time)
        symbol = str(trade.symbol)
        row = {
            "trade_id": getattr(trade, "trade_id", ""),
            "symbol": symbol,
            "candidate": getattr(trade, "candidate", ""),
            "entry_time": entry,
            "is_cic1": bool(getattr(trade, "is_cic1", False)),
            "is_cic2": bool(getattr(trade, "is_cic2", False)),
            "is_o6": bool(getattr(trade, "is_o6", False)),
        }
        events = events_by_symbol.get(symbol, pd.DataFrame())
        for window_text in windows:
            window = _parse_window(window_text)
            count = 0
            if not events.empty:
                count = int(events["event_available_time"].le(entry).mul(events["event_available_time"].gt(entry - window)).sum())
            row[f"token_prior_{window_text}"] = count > 0
            row[f"token_prior_{window_text}_count"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def _token_event_overlap_audit(trades: pd.DataFrame, mapping: pd.DataFrame, token_events: pd.DataFrame, cfg: V65Config) -> pd.DataFrame:
    symbols = sorted(trades["symbol"].dropna().astype(str).unique()) if not trades.empty else []
    flags = _prior_flags(trades, token_events, cfg.prior_windows)
    mapping_index = mapping.set_index("cex_symbol") if not mapping.empty and "cex_symbol" in mapping.columns else pd.DataFrame()
    event_counts = token_events.groupby("cex_symbol").size().to_dict() if not token_events.empty else {}
    rows = []
    main_col = f"token_prior_{cfg.main_prior_window}"
    for symbol in symbols:
        sample = trades[trades["symbol"].eq(symbol)]
        flag_sample = flags[flags["symbol"].eq(symbol)] if not flags.empty else pd.DataFrame()
        map_row = mapping_index.loc[symbol] if not mapping_index.empty and symbol in mapping_index.index else pd.Series(dtype=object)
        status = _mapping_status(map_row)
        mapped = status == "mapped_A_B"
        row: dict[str, Any] = {
            "symbol": symbol,
            "mapped": bool(mapped),
            "mapping_confidence": str(map_row.get("mapping_confidence", "")) if len(map_row) else "",
            "chain": str(map_row.get("chain", "")) if len(map_row) else "",
            "token_address": str(map_row.get("token_address", "")) if len(map_row) else "",
            "top_pool": str(map_row.get("pool_address", "")) if len(map_row) else "",
            "dex_relevance_score": float(map_row.get("dex_relevance_score", 0.0)) if len(map_row) and pd.notna(map_row.get("dex_relevance_score", np.nan)) else 0.0,
            "token_events": int(event_counts.get(symbol, 0)),
            "p2_trades": int(len(sample)),
        }
        for window_text in cfg.prior_windows:
            col = f"token_prior_{window_text}"
            row[col] = int(flag_sample[col].sum()) if not flag_sample.empty and col in flag_sample.columns else 0
        if not flag_sample.empty and main_col in flag_sample.columns:
            main = flag_sample[flag_sample[main_col]]
            row["cic1_prior_count"] = int(main["is_cic1"].sum())
            row["cic2_prior_count"] = int(main["is_cic2"].sum())
            row["o6_prior_count"] = int(main["is_o6"].sum())
        else:
            row["cic1_prior_count"] = 0
            row["cic2_prior_count"] = 0
            row["o6_prior_count"] = 0
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["p2_trades", f"token_prior_{cfg.main_prior_window}", "token_events"], ascending=[False, False, False]).reset_index(drop=True)


def _coverage_targets(trades: pd.DataFrame, trade_coverage: pd.DataFrame, overlap: pd.DataFrame, cfg: V65Config) -> pd.DataFrame:
    total_trades = int(len(trades))
    total_o6 = int(trades["is_o6"].sum()) if not trades.empty and "is_o6" in trades.columns else 0
    mapped_symbols = set(trade_coverage.loc[trade_coverage["mapping_status"].eq("mapped_A_B"), "symbol"]) if not trade_coverage.empty else set()
    mapped_trades = int(trades["symbol"].isin(mapped_symbols).sum()) if total_trades else 0
    main_col = f"token_prior_{cfg.main_prior_window}"
    token_prior_p2 = int(overlap[main_col].sum()) if not overlap.empty and main_col in overlap.columns else 0
    token_prior_o6 = int(overlap["o6_prior_count"].sum()) if not overlap.empty and "o6_prior_count" in overlap.columns else 0
    mapped_coverage = mapped_trades / total_trades if total_trades else np.nan
    return pd.DataFrame(
        [
            {
                "metric": "p2_trades",
                "value": total_trades,
                "target": "",
                "status": "reference",
            },
            {
                "metric": "o6_trades",
                "value": total_o6,
                "target": "",
                "status": "reference",
            },
            {
                "metric": "mapped_A_B_trade_coverage",
                "value": mapped_coverage,
                "target": f">={cfg.target_mapped_trade_coverage_low:.0%}",
                "status": "passed_minimum" if pd.notna(mapped_coverage) and mapped_coverage >= cfg.target_mapped_trade_coverage_low else "below_target",
            },
            {
                "metric": f"token_prior_{cfg.main_prior_window}_p2_trades",
                "value": token_prior_p2,
                "target": f">={cfg.target_token_prior_p2_trades}",
                "status": "passed" if token_prior_p2 >= cfg.target_token_prior_p2_trades else "below_target",
            },
            {
                "metric": f"token_prior_{cfg.main_prior_window}_o6_trades",
                "value": token_prior_o6,
                "target": f">={cfg.target_token_prior_o6_trades}",
                "status": "passed" if token_prior_o6 >= cfg.target_token_prior_o6_trades else "below_target",
            },
        ]
    )


def _dex_relevance_summary(trade_coverage: pd.DataFrame) -> pd.DataFrame:
    if trade_coverage.empty:
        return pd.DataFrame()
    cols = [
        "symbol",
        "p2_trades",
        "o6_trades",
        "mapping_status",
        "pool_24h_volume_usd",
        "pool_liquidity_usd",
        "dex_relevance_score",
        "dex_relevance_bucket",
        "chain",
        "top_pool",
    ]
    return trade_coverage[cols].sort_values(["dex_relevance_score", "p2_trades"], ascending=[False, False]).reset_index(drop=True)


def _write_notes(
    path: Path,
    trade_coverage: pd.DataFrame,
    missing: pd.DataFrame,
    overlap: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: V65Config,
) -> None:
    target_index = targets.set_index("metric") if not targets.empty else pd.DataFrame()
    mapped_cov = target_index.loc["mapped_A_B_trade_coverage", "value"] if "mapped_A_B_trade_coverage" in target_index.index else np.nan
    token_prior_p2 = target_index.loc[f"token_prior_{cfg.main_prior_window}_p2_trades", "value"] if f"token_prior_{cfg.main_prior_window}_p2_trades" in target_index.index else 0
    token_prior_o6 = target_index.loc[f"token_prior_{cfg.main_prior_window}_o6_trades", "value"] if f"token_prior_{cfg.main_prior_window}_o6_trades" in target_index.index else 0
    top_missing = missing.head(8)["symbol"].tolist() if not missing.empty else []
    top_event_backfill = (
        missing[missing["coverage_gap_reason"].astype(str).eq("mapped_but_no_token_events")].head(8)["symbol"].tolist()
        if not missing.empty and "coverage_gap_reason" in missing.columns
        else []
    )
    top_mapping_review = (
        missing[missing["coverage_gap_reason"].astype(str).eq("mapping_missing_or_not_A_B")].head(8)["symbol"].tolist()
        if not missing.empty and "coverage_gap_reason" in missing.columns
        else []
    )
    mapped_symbols = int(trade_coverage["mapping_status"].eq("mapped_A_B").sum()) if not trade_coverage.empty else 0
    event_symbols = int((overlap["token_events"] > 0).sum()) if not overlap.empty and "token_events" in overlap.columns else 0
    mapped_status = (
        "passed"
        if pd.notna(mapped_cov) and mapped_cov >= cfg.target_mapped_trade_coverage_low
        else "below target"
    )
    lines = [
        "# v6.5 Token / Pool Coverage Expansion",
        "",
        "Status: coverage sprint only. No alpha, gate, selector, shadow, or real-live permission is changed.",
        "",
        "Coverage snapshot:",
        f"- Symbols with A/B usable mapping in the P2 trade-weighted universe: {mapped_symbols}.",
        f"- Symbols with token pool events in current v6.3 artifacts: {event_symbols}.",
        f"- Trade-weighted A/B mapping coverage: {mapped_cov:.2%} ({mapped_status})."
        if pd.notna(mapped_cov)
        else "- Trade-weighted A/B mapping coverage: unavailable.",
        f"- Token-prior {cfg.main_prior_window} P2/CIC trades: {int(token_prior_p2)}; target >= {cfg.target_token_prior_p2_trades}.",
        f"- Token-prior {cfg.main_prior_window} O6 trades: {int(token_prior_o6)}; target >= {cfg.target_token_prior_o6_trades}.",
        "",
    ]
    if top_missing:
        lines.append("Highest-priority coverage queue:")
        for symbol in top_missing:
            lines.append(f"- {symbol}")
        lines.append("")
    if top_event_backfill:
        lines.append("Mapped symbols that need deeper historical token-event backfill:")
        for symbol in top_event_backfill:
            lines.append(f"- {symbol}")
        lines.append("")
    if top_mapping_review:
        lines.append("Remaining symbols that still need A/B mapping or alternate-source review:")
        for symbol in top_mapping_review:
            lines.append(f"- {symbol}")
        lines.append("")
    lines.extend(
        [
            "Interpretation:",
            "- v6.6 token attribution should wait until both A/B mapping and token-event overlap reach the coverage targets.",
            "- A/B mappings are the only eligible universe for future token-level attribution; C/D rows remain diagnostic or manual-review only.",
            "- DEX relevance is a coverage diagnostic, not a trading score.",
            "",
            "Next operational step:",
            "- If mapping coverage is below target, fill missing_mapping_priority.csv from the top down with canonical token and top-pool mappings.",
            "- If mapping coverage has passed but token-prior coverage is below target, prioritize mapped_but_no_token_events rows for deeper historical pool-event backfill or an alternate historical DEX source.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v65_token_pool_coverage_expansion(cfg: V65Config | None = None) -> dict[str, Path]:
    cfg = cfg or V65Config()
    report_root = ensure_dir(cfg.report_root)
    trades = _prepare_p2_trades(cfg)
    mapping = _add_dex_relevance(_read_mapping(cfg.token_mapping_path))
    token_events = _read_token_events(cfg.token_events_path)

    trade_coverage = _trade_weighted_mapping_coverage(trades, mapping)
    missing = _missing_mapping_priority(trade_coverage, token_events)
    overlap = _token_event_overlap_audit(trades, mapping, token_events, cfg)
    targets = _coverage_targets(trades, trade_coverage, overlap, cfg)
    dex_relevance = _dex_relevance_summary(trade_coverage)

    outputs = {
        "trade_weighted_mapping_coverage": report_root / "trade_weighted_mapping_coverage.csv",
        "missing_mapping_priority": report_root / "missing_mapping_priority.csv",
        "token_event_overlap_audit": report_root / "token_event_overlap_audit.csv",
        "coverage_targets": report_root / "coverage_targets.csv",
        "dex_relevance_summary": report_root / "dex_relevance_summary.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    trade_coverage.to_csv(outputs["trade_weighted_mapping_coverage"], index=False)
    missing.to_csv(outputs["missing_mapping_priority"], index=False)
    overlap.to_csv(outputs["token_event_overlap_audit"], index=False)
    targets.to_csv(outputs["coverage_targets"], index=False)
    dex_relevance.to_csv(outputs["dex_relevance_summary"], index=False)
    _write_notes(outputs["candidate_notes"], trade_coverage, missing, overlap, targets, cfg)
    return outputs
