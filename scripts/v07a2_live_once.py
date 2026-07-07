from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.config import load_config
from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.features import build_feature_table
from pressure_graph.io import ensure_dir, raw_path, read_parquet, write_parquet
from pressure_graph.paper_live.v07a2 import (
    PAPER_DATA_ROOT,
    REPORT_ROOT,
    _daily_summary,
    _market_gate_audit,
    _shadow_baselines_live,
    _summary,
    _write_status,
    add_v07a2_live_columns,
    build_v07a2_paper_ledger,
)
from pressure_graph.reports.v03 import _add_v03_report_columns
from pressure_graph.reports.v04 import RANK30_COL, RANK90_COL
from pressure_graph.reports.v06c import _market_regime_features
from pressure_graph.reports.v07a import _add_motif_columns
from v06a3_live_once import (
    _concat_symbol_dir,
    _carry_forward_rank_context,
    _floor_15m,
    _live_symbols,
    _rank_context,
    _read_optional,
    _refresh_live_raw,
    _month_start,
)


def _build_live_prepared(live_root: Path, rank30: pd.DataFrame, rank90: pd.DataFrame, base_config, paper_config):
    raw_root = live_root / "raw" / "bybit"
    instruments = _read_optional(raw_root / "instruments.parquet")
    klines = _concat_symbol_dir(raw_root / "klines")
    funding = _concat_symbol_dir(raw_root / "funding")
    oi = _concat_symbol_dir(raw_root / "open_interest")
    if klines.empty:
        raise FileNotFoundError("No live klines found after refresh.")
    features = build_feature_table(klines, funding, oi, instruments, base_config)
    features["month_start"] = _month_start(features["bar_open_time"])
    rank30 = _carry_forward_rank_context(rank30, features["month_start"])
    rank90 = _carry_forward_rank_context(rank90, features["month_start"])
    context30 = rank30.rename(
        columns={RANK30_COL: "turnover_rank_30d", "trailing_30d_turnover": "trailing_30d_turnover"}
    )[["month_start", "symbol", "turnover_rank_30d", "trailing_30d_turnover"]]
    context90 = rank90.rename(
        columns={RANK90_COL: "turnover_rank_90d", "trailing_90d_turnover": "trailing_90d_turnover"}
    )[["month_start", "symbol", "turnover_rank_90d", "trailing_90d_turnover"]]
    features = features.merge(context30, on=["month_start", "symbol"], how="left")
    features = features.merge(context90, on=["month_start", "symbol"], how="left")
    features[RANK30_COL] = pd.to_numeric(features["turnover_rank_30d"], errors="coerce")
    features[RANK90_COL] = pd.to_numeric(features["turnover_rank_90d"], errors="coerce")
    features["dynamic_all_rank"] = features[RANK30_COL]
    features["dynamic_all_trailing_turnover"] = pd.to_numeric(
        features["trailing_30d_turnover"], errors="coerce"
    )
    max_top_n = max(candidate.universe_top_n for candidate in paper_config.candidates)
    features = features[pd.to_numeric(features[RANK30_COL], errors="coerce") <= max_top_n].copy()
    prepared = _add_v03_report_columns(features, base_config)
    prepared[RANK30_COL] = pd.to_numeric(prepared[RANK30_COL], errors="coerce")
    prepared[RANK90_COL] = pd.to_numeric(prepared[RANK90_COL], errors="coerce")
    prepared["dynamic_all_rank"] = prepared[RANK30_COL]
    prepared["dynamic_all_trailing_turnover"] = pd.to_numeric(
        prepared["dynamic_all_trailing_turnover"], errors="coerce"
    )
    prepared = add_v07a2_live_columns(prepared, paper_config)
    regime = _market_regime_features(prepared)
    if not regime.empty:
        regime["volume_impulse_density_high"] = (
            pd.to_numeric(regime["volume_impulse_density"], errors="coerce")
            >= paper_config.market_graph.density_threshold
        )
        regime["low_volume_impulse_density"] = (
            pd.to_numeric(regime["volume_impulse_density"], errors="coerce")
            <= paper_config.market_graph.low_density_threshold
        )
        prepared = _add_motif_columns(prepared, regime, base_config)
    return add_v07a2_live_columns(prepared, paper_config)


def _write_outputs(prepared: pd.DataFrame, paper_config, signal_days: int, report_root: Path, paper_root: Path) -> None:
    report_root = ensure_dir(report_root)
    paper_root = ensure_dir(paper_root)
    latest = pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()
    signal_start = latest - pd.Timedelta(days=signal_days)
    signals, trades, baseline_signals, baseline_trades = build_v07a2_paper_ledger(
        prepared,
        paper_config,
        signal_start,
    )
    daily = _daily_summary(signals, trades, paper_config)
    candidate_summary = _summary(signals, trades, ["candidate", "candidate_role"])
    primary_summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=True)
    baseline_summary = _summary(baseline_signals, baseline_trades, ["candidate", "baseline_kind"])
    skipped = signals[signals["status"].eq("skipped")].copy() if not signals.empty else signals.copy()
    for root in [report_root, paper_root]:
        write_parquet(signals, root / "paper_signals.parquet")
        write_parquet(trades, root / "paper_trades.parquet")
        write_parquet(baseline_signals, root / "baseline_signals.parquet")
        write_parquet(baseline_trades, root / "baseline_trades.parquet")
    daily.to_csv(report_root / "daily_summary.csv", index=False)
    candidate_summary.to_csv(report_root / "candidate_summary.csv", index=False)
    primary_summary.to_csv(report_root / "primary_summary.csv", index=False)
    baseline_summary.to_csv(report_root / "baseline_summary.csv", index=False)
    skipped.to_csv(report_root / "skipped_signals.csv", index=False)
    _market_gate_audit(signals, trades, prepared).to_csv(report_root / "market_gate_audit.csv", index=False)
    _shadow_baselines_live(signals, trades, baseline_signals, baseline_trades, paper_config).to_csv(
        report_root / "shadow_baselines_live.csv",
        index=False,
    )
    _write_status(report_root, prepared, signals, trades, baseline_trades, paper_config)
    print(f"latest_feature_time={latest}")
    print(f"signals={len(signals)} trades={len(trades)} baseline_trades={len(baseline_trades)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated v0.7A.2 MIR1 paper-live refresh.")
    parser.add_argument("--base-config", default="configs/v0_3.yaml")
    parser.add_argument("--paper-config", default="configs/v0_7a2_mir1_paper_live.yaml")
    parser.add_argument("--live-root", default="data/live_v07a2")
    parser.add_argument("--history-days", type=int, default=45)
    parser.add_argument("--signal-days", type=int, default=7)
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()
    base_config = load_config(args.base_config)
    paper_config = load_v07a2_config(args.paper_config)
    feature_path = base_config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments_path = raw_path(base_config.paths.data_root, "bybit", "instruments")
    instruments = read_parquet(instruments_path)
    rank30, rank90 = _rank_context(feature_path, instruments, base_config)
    top_n = max(candidate.universe_top_n for candidate in paper_config.candidates)
    symbols = _live_symbols(rank30, _floor_15m(), top_n, args.max_symbols or None)
    live_root = Path(args.live_root)
    print(f"live_symbols={len(symbols)} {','.join(symbols)}")
    _refresh_live_raw(symbols, base_config, live_root, args.history_days)
    prepared = _build_live_prepared(live_root, rank30, rank90, base_config, paper_config)
    write_parquet(prepared, live_root / "processed" / "v0_7a2_live_features.parquet")
    _write_outputs(prepared, paper_config, args.signal_days, REPORT_ROOT, PAPER_DATA_ROOT)


if __name__ == "__main__":
    main()
