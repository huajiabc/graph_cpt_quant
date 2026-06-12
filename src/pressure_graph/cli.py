from __future__ import annotations

from pathlib import Path

import typer

from pressure_graph.config import load_config
from pressure_graph.config.v02 import load_v02_config
from pressure_graph.config.v05 import load_v05_config
from pressure_graph.config.v06a3 import load_v06a3_config
from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.pipeline import (
    build_features_from_raw,
    build_v03_features_from_raw,
    collect_exchange,
    run_reports_from_features,
    run_v01_reports_from_features,
    collect_v01_1m_execution_data,
    run_v01_1m_execution_validation,
    run_v02_execution_reality_check,
    run_v03a_attribution_from_features,
    run_v03_reports_from_features,
    run_v04_regime_liquidity_gate_from_features,
    run_v05_paper_live_from_features,
    run_v06a3_paper_live_from_features,
    collect_v06a2_execution_data_from_features,
    run_v06a1_impulse_reclaim_validation_from_features,
    run_v06a2_execution_concentration_from_features,
    run_v06a31_entry_gate_revalidation_from_features,
    run_v06b_flush_reversal_from_features,
    run_v06c_attack_regime_detector_from_features,
    run_v07a_motif_atlas_from_features,
    run_v07a1_1m_reconciliation_from_features,
    run_v07a1_mir1_validation_from_features,
    run_v07a2_mir1_paper_live_from_features,
    run_v07d2_cic_mir1_paper_live_from_features,
    run_v07b_neighbor_graph_from_features,
    run_v07b1_nir1_validation_from_features,
    run_v07b2_neighbor_edge_attribution_from_features,
    run_v07c_leader_beta_rotation_from_features,
    run_v07c1_gate_width_audit_from_features,
    run_v07c2_leader_beta_continuation_from_features,
    run_v07d_co_impulse_continuation_from_features,
    run_v07d1_cic_execution_integration_from_features,
    run_v09a_cluster_impulse_graph_from_features,
    run_v09b_portfolio_ranking_from_features,
    run_v09b2_ranking_failure_attribution,
    run_v09c_orderflow_capacity_ranking,
    run_v09e_orderbook_capacity_ranking,
    run_v09e1_historical_orderbook_replay,
    run_v09e2_upside_vacuum_validation_report,
    run_v09d_cic_capacity_architecture_from_features,
    run_v09d1_burst_capacity_execution_from_features,
    run_v10a_cic_basket_portfolio_from_features,
    run_v10b_slot_turnover_attribution_from_features,
    run_v10c_burst_phase_allocation_from_features,
    run_v10d_late_burst_overflow_from_features,
    run_v10_short_mirror_failure_from_features,
    run_v11_orderflow_burst_ranking_from_features,
    run_v06a_reclaim_alpha_from_features,
)
from pressure_graph.reports.v06a1 import load_v06a1_config
from pressure_graph.reports.v06a2 import load_v06a2_config
from pressure_graph.orderflow import OrderflowRunConfig, write_v08_orderflow_shadow
from pressure_graph.orderflow_history import (
    DEFAULT_CANDIDATES,
    OrderflowHistoryConfig,
    write_cic_event_orderflow,
)
from pressure_graph.orderbook import OrderbookRunConfig, write_v085_orderbook_snapshot


app = typer.Typer(help="Crypto Pressure Graph v0 experiment CLI")


def _symbol_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _exchange_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


@app.command()
def collect(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    exchange: str = typer.Option("bybit", help="Exchange: bybit or binance."),
    days: int | None = typer.Option(None, help="History days. Defaults by exchange role."),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT."),
    all_eligible: bool = typer.Option(False, help="Download all currently eligible USDT perps."),
    skip_existing: bool = typer.Option(False, help="Skip symbols with cached kline/funding/OI files."),
    workers: int = typer.Option(1, help="Parallel symbol download workers."),
) -> None:
    cfg = load_config(config)
    selected = collect_exchange(exchange, cfg, days, _symbol_list(symbols), all_eligible, skip_existing, workers)
    typer.echo(f"Collected {exchange}: {len(selected)} symbols")


@app.command("build-features")
def build_features(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    exchanges: str = typer.Option("bybit,binance", help="Comma-separated exchanges to build."),
) -> None:
    cfg = load_config(config)
    df = build_features_from_raw(cfg, _exchange_list(exchanges))
    typer.echo(f"Wrote features: {len(df):,} rows")


@app.command("run-paths")
def run_paths(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    universe_col: str = typer.Option(
        "universe_dynamic_monthly_top30", help="Universe flag column for primary report."
    ),
) -> None:
    cfg = load_config(config)
    outputs = run_reports_from_features(cfg, universe_col)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command()
def report(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    universe_col: str = typer.Option(
        "universe_dynamic_monthly_top30", help="Universe flag column for primary report."
    ),
) -> None:
    run_paths(config, universe_col)


@app.command("run-v01")
def run_v01(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    universe_col: str = typer.Option(
        "universe_dynamic_monthly_top30", help="Universe flag column for v0.1 report."
    ),
) -> None:
    cfg = load_config(config)
    outputs = run_v01_reports_from_features(cfg, universe_col)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("collect-v01-1m")
def collect_v01_1m(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    universe_col: str = typer.Option(
        "universe_dynamic_monthly_top30", help="Universe flag column for v0.1 1m validation."
    ),
    source: str = typer.Option("api", help="1m source: api or public-trades."),
    max_symbol_days: int | None = typer.Option(
        None, help="Optional cap for public-trades symbol-day downloads."
    ),
    symbol_day_offset: int = typer.Option(
        0, help="Offset for batching public-trades symbol-day downloads."
    ),
    public_trade_workers: int = typer.Option(
        4, help="Parallel workers for public-trades downloads."
    ),
) -> None:
    cfg = load_config(config)
    out_dir = collect_v01_1m_execution_data(
        cfg,
        universe_col,
        source=source,
        max_symbol_days=max_symbol_days,
        symbol_day_offset=symbol_day_offset,
        public_trade_workers=public_trade_workers,
    )
    typer.echo(f"1m execution data: {out_dir}")


@app.command("run-v01-1m")
def run_v01_1m(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    universe_col: str = typer.Option(
        "universe_dynamic_monthly_top30", help="Universe flag column for v0.1 1m validation."
    ),
    source: str = typer.Option("api", help="1m source: api or public-trades."),
) -> None:
    cfg = load_config(config)
    outputs = run_v01_1m_execution_validation(cfg, universe_col, source=source)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v02")
def run_v02(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to v0 base config."),
    frozen_config: Path = typer.Option(
        Path("configs/v0_2_frozen_candidates.yaml"), help="Path to frozen v0.2 candidates."
    ),
    source: str = typer.Option("public-trades", help="Execution source for tick validation."),
    include_baselines: bool = typer.Option(
        True, "--include-baselines/--skip-baselines", help="Run matched-random and entry-only baselines."
    ),
) -> None:
    cfg = load_config(config)
    v02 = load_v02_config(frozen_config)
    outputs = run_v02_execution_reality_check(cfg, v02, source, include_baselines)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("build-v03-features")
def build_v03_features(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to v0 base config."),
    streaming: bool = typer.Option(
        False, help="Memory-bounded batch build for cgroup-limited hosts."
    ),
    workers: int = typer.Option(5, help="Streaming worker processes."),
    batch_size: int = typer.Option(12, help="Symbols per streaming batch."),
) -> None:
    cfg = load_config(config)
    if streaming:
        from pressure_graph.features.streaming_build import (
            StreamingBuildConfig,
            build_v03_features_streaming,
        )

        output = build_v03_features_streaming(
            cfg, StreamingBuildConfig(batch_size=batch_size, workers=workers)
        )
        typer.echo(f"Wrote v0.3 all-eligible features (streaming): {output}")
        return
    df = build_v03_features_from_raw(cfg)
    typer.echo(f"Wrote v0.3 all-eligible features: {len(df):,} rows")


@app.command("run-v03")
def run_v03(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to v0 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v03_reports_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v03a")
def run_v03a(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v03a_attribution_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v04")
def run_v04(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v04_regime_liquidity_gate_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v05-paper")
def run_v05_paper(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    paper_config: Path = typer.Option(
        Path("configs/v0_5_paper_live.yaml"), help="Path to v0.5 paper-live config."
    ),
    days: int | None = typer.Option(
        30,
        help="Latest days to replay with the fixed paper-live engine. Use 0 for all available rows.",
    ),
) -> None:
    cfg = load_config(config)
    v05 = load_v05_config(paper_config)
    outputs = run_v05_paper_live_from_features(cfg, v05, days=None if days == 0 else days)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v06a3-paper")
def run_v06a3_paper(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    paper_config: Path = typer.Option(
        Path("configs/v0_6a3_paper_live.yaml"), help="Path to v0.6A.3 paper-live config."
    ),
    days: int | None = typer.Option(
        30,
        help="Latest days to replay with the fixed v0.6A.3 paper-live engine. Use 0 for all available rows.",
    ),
) -> None:
    cfg = load_config(config)
    v06a3 = load_v06a3_config(paper_config)
    outputs = run_v06a3_paper_live_from_features(cfg, v06a3, days=None if days == 0 else days)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v06a")
def run_v06a(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v06a_reclaim_alpha_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v06a1")
def run_v06a1(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    frozen_config: Path = typer.Option(
        Path("configs/v0_6a1_frozen_candidates.yaml"), help="Path to frozen v0.6A.1 candidates."
    ),
) -> None:
    cfg = load_config(config)
    v06a1 = load_v06a1_config(frozen_config)
    outputs = run_v06a1_impulse_reclaim_validation_from_features(cfg, v06a1)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("collect-v06a2-1m")
def collect_v06a2_1m(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    frozen_config: Path = typer.Option(
        Path("configs/v0_6a2_execution_concentration.yaml"),
        help="Path to frozen v0.6A.2 execution config.",
    ),
    source: str = typer.Option("api", help="Execution data source: api or public-trades."),
    max_symbol_days: int | None = typer.Option(
        None, help="Optional cap for public-trades symbol-day downloads."
    ),
    symbol_day_offset: int = typer.Option(
        0, help="Offset for batching public-trades symbol-day downloads."
    ),
    public_trade_workers: int = typer.Option(
        4, help="Parallel workers for public-trades downloads."
    ),
) -> None:
    cfg = load_config(config)
    v06a2 = load_v06a2_config(frozen_config)
    out_dir = collect_v06a2_execution_data_from_features(
        cfg,
        v06a2,
        source=source,
        max_symbol_days=max_symbol_days,
        symbol_day_offset=symbol_day_offset,
        public_trade_workers=public_trade_workers,
    )
    typer.echo(f"v0.6A.2 execution data: {out_dir}")


@app.command("run-v06a2")
def run_v06a2(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    frozen_config: Path = typer.Option(
        Path("configs/v0_6a2_execution_concentration.yaml"),
        help="Path to frozen v0.6A.2 execution config.",
    ),
) -> None:
    cfg = load_config(config)
    v06a2 = load_v06a2_config(frozen_config)
    outputs = run_v06a2_execution_concentration_from_features(cfg, v06a2)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v06a31")
def run_v06a31(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    frozen_config: Path = typer.Option(
        Path("configs/v0_6a2_execution_concentration.yaml"),
        help="Path to frozen v0.6A.2 execution config.",
    ),
) -> None:
    cfg = load_config(config)
    v06a2 = load_v06a2_config(frozen_config)
    outputs = run_v06a31_entry_gate_revalidation_from_features(cfg, v06a2)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v06b")
def run_v06b(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v06b_flush_reversal_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v06c")
def run_v06c(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v06c_attack_regime_detector_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07a")
def run_v07a(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    symbol_offset: int = typer.Option(0, help="Start offset for batched symbol runs."),
    symbol_limit: int | None = typer.Option(None, help="Optional number of symbols for batched symbol runs."),
    batch_only: bool = typer.Option(False, help="Write one v0.7A batch without final assembly."),
    assemble_only: bool = typer.Option(False, help="Assemble existing v0.7A batches into final reports."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07a_motif_atlas_from_features(
        cfg,
        symbol_offset=symbol_offset,
        symbol_limit=symbol_limit,
        batch_only=batch_only,
        assemble_only=assemble_only,
    )
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07a1")
def run_v07a1(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07a1_mir1_validation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07a1-reconcile")
def run_v07a1_reconcile(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07a1_1m_reconciliation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07a2-paper")
def run_v07a2_paper(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    paper_config: Path = typer.Option(
        Path("configs/v0_7a2_mir1_paper_live.yaml"), help="Path to v0.7A.2 MIR1 paper-live config."
    ),
    days: int | None = typer.Option(
        30,
        help="Latest days to replay with the fixed v0.7A.2 paper-live engine. Use 0 for all available rows.",
    ),
) -> None:
    cfg = load_config(config)
    v07a2 = load_v07a2_config(paper_config)
    outputs = run_v07a2_mir1_paper_live_from_features(cfg, v07a2, days=None if days == 0 else days)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07d2-paper")
def run_v07d2_paper(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    paper_config: Path = typer.Option(
        Path("configs/v0_7d2_cic_mir1_paper_live.yaml"),
        help="Path to v0.7D.2 CIC-filtered MIR1 paper-live config.",
    ),
    days: int | None = typer.Option(
        30,
        help="Latest days to replay with the fixed v0.7D.2 paper-live engine. Use 0 for all available rows.",
    ),
) -> None:
    cfg = load_config(config)
    v07d2 = load_v07a2_config(paper_config)
    outputs = run_v07d2_cic_mir1_paper_live_from_features(cfg, v07d2, days=None if days == 0 else days)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07b")
def run_v07b(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07b_neighbor_graph_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07b1")
def run_v07b1(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07b1_nir1_validation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07b2")
def run_v07b2(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07b2_neighbor_edge_attribution_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07c")
def run_v07c(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07c_leader_beta_rotation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07c1")
def run_v07c1(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07c1_gate_width_audit_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07c2")
def run_v07c2(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07c2_leader_beta_continuation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07d")
def run_v07d(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07d_co_impulse_continuation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v07d1")
def run_v07d1(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v07d1_cic_execution_integration_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v08-orderflow-shadow")
def run_v08_orderflow_shadow(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    report_root: Path = typer.Option(
        Path("reports/v0_8_orderflow_shadow"), help="Output report directory."
    ),
    orderflow_root: Path = typer.Option(
        Path("data/orderflow/v0_8/bybit"), help="Raw and 1m orderflow cache directory."
    ),
    demand_queue_path: Path = typer.Option(
        Path("data/orderflow/demand_queue.parquet"), help="Event-driven orderflow demand queue."
    ),
    source_report_root: Path = typer.Option(
        Path("reports/v0_7d2_cic_mir1_paper_live"),
        help="v0.7D.2 paper-live report directory to annotate.",
    ),
    live_feature_path: Path = typer.Option(
        Path("data/live_v07d2/processed/v0_7d2_live_features.parquet"),
        help="Latest v0.7D.2 live feature table for selecting TopN symbols.",
    ),
    top_n: int = typer.Option(50, help="Dynamic universe TopN to cache for orderflow."),
    max_symbols: int = typer.Option(0, help="Optional symbol cap. Use 0 for all selected symbols."),
    recent_trade_limit: int = typer.Option(1000, help="Bybit recent trades limit per symbol."),
    retain_days: int = typer.Option(7, help="Days to retain in local recent-trade cache."),
    report_lookback_days: int = typer.Option(
        7, help="Recent paper-report lookback days for adding active symbols."
    ),
) -> None:
    cfg = load_config(config)
    run_cfg = OrderflowRunConfig(
        report_root=report_root,
        orderflow_root=orderflow_root,
        demand_queue_path=demand_queue_path,
        source_report_root=source_report_root,
        live_feature_path=live_feature_path,
        top_n=top_n,
        max_symbols=None if max_symbols == 0 else max_symbols,
        recent_trade_limit=recent_trade_limit,
        retain_days=retain_days,
        report_lookback_days=report_lookback_days,
    )
    outputs = write_v08_orderflow_shadow(cfg, run_cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v085-orderbook-snapshot")
def run_v085_orderbook_snapshot(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
    report_root: Path = typer.Option(
        Path("reports/v0_8_5_orderbook_snapshot"), help="Output report directory."
    ),
    orderbook_root: Path = typer.Option(
        Path("data/orderbook/v0_8_5/bybit"), help="Raw and feature orderbook cache directory."
    ),
    demand_queue_path: Path = typer.Option(
        Path("data/orderflow/demand_queue.parquet"), help="Event-driven orderflow demand queue."
    ),
    live_feature_path: Path = typer.Option(
        Path("data/live_v07d2/processed/v0_7d2_live_features.parquet"),
        help="Latest v0.7D.2 live feature table for selecting TopN fallback symbols.",
    ),
    top_n: int = typer.Option(50, help="Dynamic universe TopN fallback."),
    max_symbols: int = typer.Option(10, help="Optional symbol cap. Use 0 for all selected symbols."),
    depth_limit: int = typer.Option(200, help="Bybit orderbook depth limit."),
    retain_days: int = typer.Option(7, help="Days to retain in local snapshot cache."),
) -> None:
    cfg = load_config(config)
    run_cfg = OrderbookRunConfig(
        report_root=report_root,
        orderbook_root=orderbook_root,
        demand_queue_path=demand_queue_path,
        live_feature_path=live_feature_path,
        top_n=top_n,
        max_symbols=None if max_symbols == 0 else max_symbols,
        depth_limit=depth_limit,
        retain_days=retain_days,
    )
    outputs = write_v085_orderbook_snapshot(cfg, run_cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09a")
def run_v09a(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v09a_cluster_impulse_graph_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09b")
def run_v09b(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v09b_portfolio_ranking_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09b2")
def run_v09b2() -> None:
    outputs = run_v09b2_ranking_failure_attribution()
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09c")
def run_v09c() -> None:
    outputs = run_v09c_orderflow_capacity_ranking()
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09e")
def run_v09e() -> None:
    outputs = run_v09e_orderbook_capacity_ranking()
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09e1")
def run_v09e1(
    max_files: int | None = typer.Option(
        None,
        help="Optional cap on symbol-day orderbook files to download/replay.",
    ),
    download: bool = typer.Option(
        True,
        "--download/--no-download",
        help="Download missing Bybit historical orderbook files before replay.",
    ),
    run_ranking: bool = typer.Option(
        True,
        "--run-ranking/--no-ranking",
        help="Run v0.9E ranking after historical replay features are written.",
    ),
) -> None:
    outputs = run_v09e1_historical_orderbook_replay(
        max_files=max_files,
        download=download,
        run_ranking=run_ranking,
    )
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09e2")
def run_v09e2(
    max_files: int | None = typer.Option(
        None,
        help="Optional cap on missing official orderbook symbol-day files to download/replay.",
    ),
    download: bool = typer.Option(
        True,
        "--download/--no-download",
        help="Download missing Bybit official historical orderbook zip files.",
    ),
) -> None:
    outputs = run_v09e2_upside_vacuum_validation_report(max_files=max_files, download=download)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09d")
def run_v09d(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v09d_cic_capacity_architecture_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v09d1")
def run_v09d1(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v09d1_burst_capacity_execution_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v10-short")
def run_v10_short(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v10_short_mirror_failure_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v10a-cic-basket")
def run_v10a_cic_basket(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v10a_cic_basket_portfolio_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v10b-slot-turnover")
def run_v10b_slot_turnover(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v10b_slot_turnover_attribution_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v10c-burst-phase")
def run_v10c_burst_phase(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v10c_burst_phase_allocation_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v10d-late-overflow")
def run_v10d_late_overflow(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    cfg = load_config(config)
    outputs = run_v10d_late_burst_overflow_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("collect-orderflow-history")
def collect_orderflow_history(
    trade_cache: Path = typer.Option(
        Path("reports/v0_9d_cic_capacity_architecture/capacity_trade_cache.parquet"),
        help="v0.9D capacity trade cache parquet.",
    ),
    candidates: str = typer.Option(
        ",".join(DEFAULT_CANDIDATES),
        help="Comma-separated candidate names to backfill.",
    ),
    workers: int = typer.Option(8, help="Parallel archive download workers."),
    max_events: int | None = typer.Option(None, help="Cap events for smoke runs."),
) -> None:
    """Backfill historical event-window orderflow from Binance UM aggTrades archives."""
    candidate_tuple = tuple(item.strip() for item in candidates.split(",") if item.strip())
    cfg = OrderflowHistoryConfig(
        trade_cache_path=trade_cache,
        candidates=candidate_tuple or DEFAULT_CANDIDATES,
        download_workers=workers,
        max_events=max_events,
    )
    outputs = write_cic_event_orderflow(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-v11-orderflow-ranking")
def run_v11_orderflow_ranking(
    config: Path = typer.Option(Path("configs/v0_3.yaml"), help="Path to v0.3 base config."),
) -> None:
    """Selected-vs-skipped orderflow ranking experiment over the CIC burst pool."""
    cfg = load_config(config)
    outputs = run_v11_orderflow_burst_ranking_from_features(cfg)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


@app.command("run-all")
def run_all(
    config: Path = typer.Option(Path("configs/v0.yaml"), help="Path to experiment config."),
    exchange: str = typer.Option("bybit", help="Exchange: bybit or binance."),
    days: int | None = typer.Option(None, help="History days. Defaults by exchange role."),
    symbols: str | None = typer.Option(None, help="Comma-separated symbols for smoke runs."),
    all_eligible: bool = typer.Option(False, help="Download all currently eligible USDT perps."),
    skip_existing: bool = typer.Option(False, help="Skip symbols with cached kline/funding/OI files."),
    workers: int = typer.Option(1, help="Parallel symbol download workers."),
    universe_col: str = typer.Option(
        "universe_dynamic_monthly_top30", help="Universe flag column for primary report."
    ),
) -> None:
    cfg = load_config(config)
    selected = collect_exchange(exchange, cfg, days, _symbol_list(symbols), all_eligible, skip_existing, workers)
    typer.echo(f"Collected {exchange}: {len(selected)} symbols")
    df = build_features_from_raw(cfg, [exchange])
    typer.echo(f"Wrote features: {len(df):,} rows")
    outputs = run_reports_from_features(cfg, universe_col)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()
