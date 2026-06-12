"""v1.2s2 Long Risk-Off Overlay — the actual product of the short-side research.

v1.2s proved the failure motifs are not tradeable as shorts, but that their value
(doc §8) is *cutting long exposure / forbidding new longs*. This module measures
exactly that: take the existing CIC long basket (v0.9D capacity trade cache) and,
using the short failure-motif confirmations as a risk-off gate, ask whether the
long book's drawdown and risk-adjusted return improve.

Two gate modes (strict as-of — a long is only gated by confirmations whose
feature_time <= the long's own signal_time):

- symbol gate:  suppress new longs on a symbol within a cooldown window after that
                symbol's own failure-motif confirmation (a failed CIC reclaim on
                this name is a reason to stand aside on this name).
- market gate:  suppress new longs market-wide when failure-motif breadth in a
                trailing window crosses a threshold (the doc's S6 breadth-collapse /
                市场转弱 — many names failing at once).

Removed longs free basket capacity that first-come selection backfills, so the
comparison answers: did standing aside on these names beat carrying them?

Tier: research only. No shadow / paper-live / real-live wiring. The default motif
set excludes S2 (mistimed exhaustion proxy, worst standalone behaviour in v1.2s).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v09b import select_portfolio
from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool, _portfolio_metrics
from pressure_graph.reports.v12s_short_motif_atlas import (
    DETECTORS,
    MOTIF_PARAMS,
    ShortAtlasConfig,
)

REPORT_ROOT = Path("reports/v1_2s2_long_risk_off_overlay")
TRADE_CACHE_PATH = Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet")
POOL_NAME = "P2_CIC1_CIC2_COMBINED"
BAR = pd.Timedelta(minutes=15)
BAR_NS = int(BAR.value)  # 15 minutes in nanoseconds — gate math runs in epoch ns (tz-safe).


def _epoch_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("int64").to_numpy()


@dataclass(frozen=True)
class RiskOffConfig:
    report_root: Path = REPORT_ROOT
    trade_cache_path: Path = TRADE_CACHE_PATH
    top_n: int = 30
    motifs: tuple[str, ...] = ("S1", "S3", "S5")
    symbol_cooldown_bars: int = 32  # 8h: a recent failure suppresses new longs here.
    breadth_window_bars: int = 16  # 4h trailing window for market breadth.
    breadth_threshold: int = 3  # distinct symbols failing -> market risk-off.
    max_positions_grid: tuple[int, ...] = (5, 8, 10)


def stream_risk_off_events(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbols: list[str],
    config: ExperimentConfig,
    cfg: RiskOffConfig,
) -> pd.DataFrame:
    """Per-symbol failure-motif confirmation times (feature_time of the confirm bar)."""
    atlas_cfg = ShortAtlasConfig(top_n=cfg.top_n, motifs=cfg.motifs)
    rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= cfg.top_n].copy()
        if data.empty:
            continue
        data = data.sort_values("bar_open_time").reset_index(drop=True)
        feature_time = pd.to_datetime(data["feature_time"], utc=True, errors="coerce")
        for motif_code in cfg.motifs:
            signals = DETECTORS[motif_code](data, MOTIF_PARAMS[motif_code])
            for confirmation_idx, _ in signals:
                ts = feature_time.iloc[confirmation_idx]
                if pd.notna(ts):
                    rows.append({"symbol": str(symbol), "motif": motif_code, "feature_time": ts})
        if idx % 25 == 0:
            print(f"v1.2s2 risk-off events: {idx}/{len(symbols)} symbols, {len(rows)} events", flush=True)
    events = pd.DataFrame(rows, columns=["symbol", "motif", "feature_time"])
    if not events.empty:
        events = events.sort_values("feature_time").reset_index(drop=True)
    return events


def _prepare_pool(cfg: RiskOffConfig) -> pd.DataFrame:
    trades = read_parquet(cfg.trade_cache_path)
    pool = _focus_pool(trades, POOL_NAME)
    if pool.empty:
        return pool
    pool["signal_time"] = pd.to_datetime(pool["signal_time"], utc=True, errors="coerce")
    pool = pool.dropna(subset=["signal_time"]).reset_index(drop=True)
    if "rank_first_come_first_served" not in pool.columns:
        pool["rank_first_come_first_served"] = 0.0
    return pool


def _apply_symbol_gate(pool: pd.DataFrame, events: pd.DataFrame, cfg: RiskOffConfig) -> pd.Series:
    """Gate a long if a same-symbol failure confirmed within the cooldown before its signal."""
    window_ns = cfg.symbol_cooldown_bars * BAR_NS
    gated = pd.Series(False, index=pool.index)
    if events.empty:
        return gated
    events_ns = _epoch_ns(events["feature_time"])
    by_symbol = {
        str(sym): np.sort(events_ns[idx.to_numpy()])
        for sym, idx in events.reset_index(drop=True).groupby("symbol").groups.items()
    }
    pool_ns = _epoch_ns(pool["signal_time"])
    symbols = pool["symbol"].astype(str).to_numpy()
    for i in range(len(pool)):
        times = by_symbol.get(symbols[i])
        if times is None:
            continue
        signal = pool_ns[i]
        left = np.searchsorted(times, signal - window_ns, side="right")
        right = np.searchsorted(times, signal, side="right")
        if right > left:
            gated.iloc[i] = True
    return gated


def _breadth_at(events: pd.DataFrame, cfg: RiskOffConfig) -> tuple[np.ndarray, np.ndarray]:
    """Distinct-symbol breadth timeline sampled at each event time (trailing window)."""
    if events.empty:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=int)
    times = events["feature_time"].to_numpy()
    times_ns = _epoch_ns(events["feature_time"])
    window_ns = cfg.breadth_window_bars * BAR_NS
    symbols = events["symbol"].to_numpy()
    breadth = np.zeros(len(times_ns), dtype=int)
    for i in range(len(times_ns)):
        mask = (times_ns <= times_ns[i]) & (times_ns > times_ns[i] - window_ns)
        breadth[i] = len(set(symbols[mask]))
    return times, breadth


def _apply_market_gate(pool: pd.DataFrame, events: pd.DataFrame, cfg: RiskOffConfig) -> pd.Series:
    gated = pd.Series(False, index=pool.index)
    if events.empty:
        return gated
    event_ns = _epoch_ns(events["feature_time"])
    symbols = events["symbol"].to_numpy()
    window_ns = cfg.breadth_window_bars * BAR_NS
    pool_ns = _epoch_ns(pool["signal_time"])
    for i in range(len(pool)):
        signal = pool_ns[i]
        mask = (event_ns <= signal) & (event_ns > signal - window_ns)
        if len(set(symbols[mask])) >= cfg.breadth_threshold:
            gated.iloc[i] = True
    return gated


def _mode_metrics(pool: pd.DataFrame, gated: pd.Series, label: str, max_positions: int) -> dict[str, object]:
    kept = pool[~gated.to_numpy()].copy()
    removed = pool[gated.to_numpy()].copy()
    selected, skipped = select_portfolio(kept, score_col="rank_first_come_first_served", max_positions=max_positions)
    metrics = _portfolio_metrics(
        selected,
        skipped,
        architecture="long_risk_off_overlay",
        pool=POOL_NAME,
        rule=label,
        max_positions=max_positions,
        notes="",
    )
    net = pd.to_numeric(selected.get("net_return", pd.Series(dtype=float)), errors="coerce")
    removed_net = pd.to_numeric(removed.get("net_return", pd.Series(dtype=float)), errors="coerce")
    dd = float(metrics.get("max_drawdown_proxy", np.nan))
    portfolio_net = float(metrics.get("portfolio_net20", np.nan))
    metrics["mode"] = label
    metrics["longs_gated"] = int(len(removed))
    metrics["gated_realized_net_mean"] = float(removed_net.mean()) if len(removed_net) else np.nan
    metrics["gated_loss_share"] = float((removed_net < 0).mean()) if len(removed_net) else np.nan
    metrics["return_per_drawdown"] = float(portfolio_net / abs(dd)) if dd and np.isfinite(dd) and dd != 0 else np.nan
    return metrics


def _overlay_table(pool: pd.DataFrame, events: pd.DataFrame, cfg: RiskOffConfig) -> pd.DataFrame:
    symbol_gate = _apply_symbol_gate(pool, events, cfg)
    market_gate = _apply_market_gate(pool, events, cfg)
    combined = symbol_gate | market_gate
    no_gate = pd.Series(False, index=pool.index)
    rows: list[dict[str, object]] = []
    for max_positions in cfg.max_positions_grid:
        rows.append(_mode_metrics(pool, no_gate, "baseline", max_positions))
        rows.append(_mode_metrics(pool, symbol_gate, "symbol_risk_off", max_positions))
        rows.append(_mode_metrics(pool, market_gate, "market_risk_off", max_positions))
        rows.append(_mode_metrics(pool, combined, "combined_risk_off", max_positions))
    return pd.DataFrame(rows)


def _write_notes(report_root: Path, table: pd.DataFrame, events: pd.DataFrame, cfg: RiskOffConfig) -> None:
    lines = [
        "# v1.2s2 Long Risk-Off Overlay",
        "",
        "Product of the v1.2s short research: the failure motifs are not tradeable as",
        "shorts, but used as a long risk-off gate they should cut drawdown. This report",
        "applies S1/S3/S5 confirmations as a symbol-level and market-breadth gate on the",
        "CIC P2 long basket and compares against the un-gated baseline. Research only.",
        "",
        f"- gate motifs: {', '.join(cfg.motifs)} (S2 excluded — mistimed in v1.2s)",
        f"- symbol cooldown: {cfg.symbol_cooldown_bars} bars; breadth: >= {cfg.breadth_threshold} symbols in {cfg.breadth_window_bars} bars",
        f"- risk-off events: {len(events)}",
        "",
    ]
    focus = table[table["max_positions"].astype(str).eq("8")] if not table.empty else pd.DataFrame()
    if not focus.empty:
        base = focus[focus["mode"].eq("baseline")].iloc[0]
        lines.append("## P2 max8 (20bp)")
        for row in focus.itertuples(index=False):
            d_net = row.portfolio_net20 - base["portfolio_net20"]
            d_dd = row.max_drawdown_proxy - base["max_drawdown_proxy"]
            lines.append(
                f"- **{row.mode}**: net20={row.portfolio_net20:.4%} (Δ{d_net:+.4%}), "
                f"max_dd={row.max_drawdown_proxy:.4%} (Δ{d_dd:+.4%}), "
                f"ret/dd={row.return_per_drawdown:.2f}, gated={row.longs_gated}, "
                f"gated_realized_net={row.gated_realized_net_mean:.4%}."
            )
        lines.append("")
        best = focus[focus["mode"].ne("baseline")].sort_values("return_per_drawdown", ascending=False)
        if not best.empty:
            b = best.iloc[0]
            improved = b["max_drawdown_proxy"] >= base["max_drawdown_proxy"] and b["portfolio_net20"] >= base["portfolio_net20"]
            verdict = (
                "Risk-off improves both net and drawdown — wire as a long-book gate."
                if improved
                else "Risk-off trades some net for drawdown; judge against the long book's risk budget."
            )
            lines.append(f"## Verdict\n- Best mode by ret/dd: {b['mode']}. {verdict}")
            lines.append("")
    lines.extend(
        [
            "## Discipline",
            "- Strict as-of: a long is only gated by confirmations with feature_time <= its signal_time.",
            "- Gated longs free capacity that first-come selection backfills (next-best long).",
            "- gated_realized_net < 0 means the gate removed losers on average (the intended effect).",
            "- No paper-live / real-live permission changes.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v12s2_long_risk_off_overlay(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: RiskOffConfig = RiskOffConfig(),
) -> dict[str, Path]:
    report_root = ensure_dir(cfg.report_root)
    if not cfg.trade_cache_path.exists():
        raise FileNotFoundError(f"long trade cache not found: {cfg.trade_cache_path} (run run-v09d first)")
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    events = stream_risk_off_events(feature_path, rank30, rank90, symbols, config, cfg)
    pool = _prepare_pool(cfg)
    table = _overlay_table(pool, events, cfg) if not pool.empty else pd.DataFrame()
    times, breadth = _breadth_at(events, cfg)
    breadth_timeline = pd.DataFrame({"feature_time": times, "breadth": breadth})

    outputs = {
        "overlay_summary": report_root / "overlay_summary.csv",
        "risk_off_events": report_root / "risk_off_events.csv",
        "breadth_timeline": report_root / "breadth_timeline.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    table.to_csv(outputs["overlay_summary"], index=False)
    events.to_csv(outputs["risk_off_events"], index=False)
    breadth_timeline.to_csv(outputs["breadth_timeline"], index=False)
    _write_notes(report_root, table, events, cfg)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "RiskOffConfig",
    "stream_risk_off_events",
    "write_v12s2_long_risk_off_overlay",
]
