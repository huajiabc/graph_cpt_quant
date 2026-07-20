from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.config import load_config
from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.features import build_feature_table
from pressure_graph.io import raw_path, read_parquet, write_parquet
from pressure_graph.paper_live.v07d2 import (
    PAPER_DATA_ROOT,
    REPORT_ROOT,
    S2_PAPER_LIVE_ROOT,
    add_v07d2_live_columns,
    write_v07d2_outputs,
    write_v07d2_s2_paper_live,
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
    _month_start,
    _rank_context,
    _read_optional,
    _refresh_live_raw,
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
    features["dynamic_all_trailing_turnover"] = pd.to_numeric(features["trailing_30d_turnover"], errors="coerce")
    max_top_n = max(candidate.universe_top_n for candidate in paper_config.candidates)
    features = features[pd.to_numeric(features[RANK30_COL], errors="coerce") <= max_top_n].copy()
    prepared = _add_v03_report_columns(features, base_config)
    prepared[RANK30_COL] = pd.to_numeric(prepared[RANK30_COL], errors="coerce")
    prepared[RANK90_COL] = pd.to_numeric(prepared[RANK90_COL], errors="coerce")
    prepared["dynamic_all_rank"] = prepared[RANK30_COL]
    prepared["dynamic_all_trailing_turnover"] = pd.to_numeric(
        prepared["dynamic_all_trailing_turnover"], errors="coerce"
    )
    prepared = add_v07d2_live_columns(prepared, paper_config)
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
    return add_v07d2_live_columns(prepared, paper_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated v0.7D.2 CIC-filtered MIR1 paper-live refresh.")
    parser.add_argument("--base-config", default="configs/v0_3.yaml")
    parser.add_argument("--paper-config", default="configs/v0_7d2_cic_mir1_paper_live.yaml")
    parser.add_argument("--live-root", default="data/live_v07d2")
    parser.add_argument("--history-days", type=int, default=45)
    parser.add_argument("--signal-days", type=int, default=7)
    parser.add_argument("--max-symbols", type=int, default=0)
    args = parser.parse_args()
    base_config = load_config(args.base_config)
    paper_config = load_v07a2_config(args.paper_config)
    feature_path = base_config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = read_parquet(raw_path(base_config.paths.data_root, "bybit", "instruments"))
    rank30, rank90 = _rank_context(feature_path, instruments, base_config)
    top_n = max(candidate.universe_top_n for candidate in paper_config.candidates)
    symbols = _live_symbols(rank30, _floor_15m(), top_n, args.max_symbols or None)
    live_root = Path(args.live_root)
    print(f"live_symbols={len(symbols)} {','.join(symbols)}")
    _refresh_live_raw(symbols, base_config, live_root, args.history_days)
    prepared = _build_live_prepared(live_root, rank30, rank90, base_config, paper_config)
    write_parquet(prepared, live_root / "processed" / "v0_7d2_live_features.parquet")
    observed_at = pd.Timestamp.now(tz="UTC")
    outputs = write_v07d2_outputs(
        prepared,
        paper_config,
        args.signal_days,
        REPORT_ROOT,
        PAPER_DATA_ROOT,
        forward_mode=True,
        observed_at=observed_at,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")

    from pressure_graph.live.gates import evaluate_live_gates, write_live_gate_artifacts

    cumulative_signals = read_parquet(outputs["forward_signals"])
    cumulative_checkpoints = read_parquet(outputs["forward_checkpoint_trades"])
    cumulative_primary = cumulative_checkpoints[
        cumulative_checkpoints.get("portfolio_id", pd.Series(dtype=str))
        .astype(str)
        .eq(paper_config.forward_primary.portfolio_id)
    ].copy()
    cumulative_baselines = read_parquet(outputs["forward_baseline_trades"])
    gate_decision = evaluate_live_gates(
        prepared,
        cumulative_primary,
        cumulative_baselines,
        paper_config,
        now=observed_at,
    )
    gate_outputs = write_live_gate_artifacts(
        report_root=REPORT_ROOT,
        decision=gate_decision,
        cumulative_signals=cumulative_signals,
        observed_at=observed_at,
        cumulative_primary=cumulative_primary,
    )
    for name, path in gate_outputs.items():
        print(f"gate.{name}: {path}")

    try:
        from pressure_graph.reports.v94_forward_monitoring import write_forward_monitoring

        monitoring_outputs = write_forward_monitoring(
            REPORT_ROOT,
            primary_portfolio_id=paper_config.forward_primary.portfolio_id,
            observed_at=observed_at,
        )
        for name, path in monitoring_outputs.items():
            print(f"monitoring.{name}: {path}")
    except Exception as exc:  # noqa: BLE001 - monitoring must never break the primary refresh
        print(f"forward monitoring skipped: {exc}")

    if gate_decision.data_stale:
        raise SystemExit(2)

    s2_outputs = write_v07d2_s2_paper_live(prepared, args.signal_days, S2_PAPER_LIVE_ROOT)
    for name, path in s2_outputs.items():
        print(f"s2.{name}: {path}")

    # v1.2s2 long risk-off gate, SHADOW ONLY: records would-be suppressions next to
    # the live report and changes no paper trade. Guarded so it can never break the
    # primary refresh.
    try:
        from pressure_graph.live.risk_off_gate import write_risk_off_shadow

        risk_off_outputs = write_risk_off_shadow(prepared, Path(REPORT_ROOT) / "risk_off_shadow")
        for name, path in risk_off_outputs.items():
            print(f"risk_off.{name}: {path}")
    except Exception as exc:  # noqa: BLE001 - shadow must never affect the live refresh
        print(f"risk_off shadow skipped: {exc}")


if __name__ == "__main__":
    main()
