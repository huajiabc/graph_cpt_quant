from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pressure_graph.clients import BinanceClient, BybitClient
from pressure_graph.clients.bybit_public import download_public_trading_day, public_trades_to_1m_ohlcv
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.v05 import V05Config
from pressure_graph.config.v06a3 import V06A3Config
from pressure_graph.config.v07a2 import V07A2Config
from pressure_graph.config.v02 import V02Config, load_v02_config
from pressure_graph.backtest import ENTRY_POLICIES
from pressure_graph.backtest.minute_execution import simulate_1m_execution, write_1m_execution_outputs
from pressure_graph.config.models import ExecutionRule
from pressure_graph.features import build_feature_table
from pressure_graph.io import ensure_dir, processed_path, raw_path, read_parquet, write_parquet
from pressure_graph.labels import add_future_labels
from pressure_graph.reports import write_reports, write_v01_reports
from pressure_graph.reports.stats import development_holdout_split
from pressure_graph.reports.v01 import prepare_v01_dataset
from pressure_graph.reports.v02 import write_v02_reports
from pressure_graph.reports.v03 import write_v03_reports_from_feature_path
from pressure_graph.reports.v03a import write_v03a_attribution
from pressure_graph.reports.v04 import write_v04_regime_liquidity_gate
from pressure_graph.reports.v06a import write_v06a_reclaim_alpha
from pressure_graph.reports.v06a1 import V06A1Config, write_v06a1_impulse_reclaim_validation
from pressure_graph.reports.v06a2 import (
    V06A2Config,
    collect_v06a2_1m_execution_data,
    write_v06a2_execution_concentration,
)
from pressure_graph.reports.v06a31 import write_v06a31_entry_gate_revalidation
from pressure_graph.reports.v06b import write_v06b_flush_reversal
from pressure_graph.reports.v06c import write_v06c_attack_regime_detector
from pressure_graph.reports.v07a import write_v07a_motif_atlas
from pressure_graph.reports.v07a1 import write_v07a1_1m_reconciliation, write_v07a1_mir1_validation
from pressure_graph.reports.v07b import write_v07b_neighbor_graph
from pressure_graph.reports.v07b1 import write_v07b1_nir1_validation
from pressure_graph.reports.v07b2 import write_v07b2_neighbor_edge_attribution
from pressure_graph.reports.v07c import write_v07c_leader_beta_rotation
from pressure_graph.reports.v07c1 import write_v07c1_gate_width_audit
from pressure_graph.reports.v07c2 import write_v07c2_leader_beta_continuation
from pressure_graph.reports.v07d import write_v07d_co_impulse_continuation
from pressure_graph.reports.v07d1 import write_v07d1_cic_execution_integration
from pressure_graph.reports.v09a import write_v09a_cluster_impulse_graph
from pressure_graph.reports.v09b import write_v09b_portfolio_ranking
from pressure_graph.reports.v09b2 import write_v09b2_ranking_failure_attribution
from pressure_graph.reports.v09c import write_v09c_orderflow_capacity_ranking
from pressure_graph.reports.v09e import write_v09e_orderbook_capacity_ranking
from pressure_graph.reports.v09e1 import V09E1Config, run_historical_orderbook_replay
from pressure_graph.reports.v09e2 import V09E2Config, run_v09e2_upside_vacuum_validation
from pressure_graph.reports.v09d import write_v09d_cic_capacity_architecture
from pressure_graph.reports.v09d1 import write_v09d1_burst_capacity_execution
from pressure_graph.reports.v10a_cic_basket_portfolio import (
    V10AConfig,
    _load_or_build_trades,
    write_v10a_cic_basket_portfolio,
)
from pressure_graph.reports.v10b_slot_turnover_attribution import write_v10b_slot_turnover_attribution
from pressure_graph.reports.v10c_burst_phase_allocation import write_v10c_burst_phase_allocation
from pressure_graph.reports.v10d_late_burst_overflow import write_v10d_late_burst_overflow
from pressure_graph.reports.v10_short_mirror import write_v10_short_mirror_failure
from pressure_graph.reports.v11_orderflow_burst_ranking import (
    V11Config,
    write_v11_orderflow_burst_ranking,
)
from pressure_graph.reports.v12s_short_motif_atlas import (
    ShortAtlasConfig,
    write_v12s_short_motif_atlas,
)
from pressure_graph.reports.v12s2_long_risk_off_overlay import (
    RiskOffConfig,
    write_v12s2_long_risk_off_overlay,
)
from pressure_graph.reports.v12s3_current_stack_risk_off_overlay import (
    CurrentStackConfig,
    write_v12s3_current_stack_risk_off,
)
from pressure_graph.reports.v3_3_failure_path_search import (
    V33Config,
    write_v3_3_failure_path_search,
)
from pressure_graph.reports.v3_4_true_short_sleeve import (
    V34Config,
    write_v3_4_true_short_sleeve,
)
from pressure_graph.reports.v3_5_failure_risk_layer_bridge import (
    V35Config,
    write_v3_5_failure_risk_layer_bridge,
)
from pressure_graph.reports.v4s_failure_state_graph import (
    V4SConfig,
    write_v4s_failure_state_graph,
)
from pressure_graph.reports.v6s_path_c_short_validation import (
    V6SConfig,
    write_v6s_path_c_short_validation,
)
from pressure_graph.reports.v7s_short_alpha import (
    V7SConfig,
    write_v7s_short_alpha,
)
from pressure_graph.reports.failure_overlay_shadow import (
    ShadowConfig,
    write_failure_overlay_shadow,
)
from pressure_graph.paper_live import (
    write_v05_paper_live,
    write_v06a3_paper_live,
    write_v07a2_mir1_paper_live,
    write_v07d2_cic_mir1_paper_live,
)
from pressure_graph.universe.selection import apply_universe_flags, filter_instruments, static_current_top_symbols


def utc_floor_15m(ts: pd.Timestamp | None = None) -> pd.Timestamp:
    ts = ts or pd.Timestamp.now(tz="UTC")
    return ts.floor("15min")


def create_client(exchange: str, config: ExperimentConfig):
    if exchange == "bybit":
        bybit = config.exchanges.bybit
        return BybitClient(str(bybit.base_url), bybit.category)
    if exchange == "binance":
        return BinanceClient(str(config.exchanges.binance.base_url))
    raise ValueError(f"unsupported exchange: {exchange}")


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def _concat_symbol_files(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in sorted(path.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _eligible_symbols_from_current_snapshot(
    exchange: str,
    instruments: pd.DataFrame,
    tickers: pd.DataFrame,
    config: ExperimentConfig,
    end: pd.Timestamp,
    all_eligible: bool,
) -> list[str]:
    if all_eligible:
        eligible = filter_instruments(instruments, end, config)
        if "symbol" in eligible.columns:
            if exchange == "bybit":
                eligible = eligible[eligible["symbol"].str.endswith("USDT")]
            if exchange == "binance" and "quoteAsset" in eligible.columns:
                eligible = eligible[eligible["quoteAsset"].eq("USDT")]
            symbols = sorted(eligible["symbol"].dropna().unique().tolist())
            if "BTCUSDT" not in symbols:
                symbols = ["BTCUSDT", *symbols]
            return list(dict.fromkeys(symbols))
    symbols = static_current_top_symbols(tickers, instruments, end, config)
    if "BTCUSDT" not in symbols:
        symbols = ["BTCUSDT", *symbols]
    return list(dict.fromkeys(symbols))


def collect_exchange(
    exchange: str,
    config: ExperimentConfig,
    days: int | None = None,
    symbols: list[str] | None = None,
    all_eligible: bool = False,
    skip_existing: bool = False,
    workers: int = 1,
) -> list[str]:
    end = utc_floor_15m()
    if days is None:
        days = (
            config.exchanges.binance.oi_max_history_days
            if exchange == "binance"
            else config.experiment.start_days
        )
    start = end - pd.Timedelta(days=days)
    client = create_client(exchange, config)
    try:
        if exchange == "bybit":
            instruments = client.instruments(config.exchanges.bybit.settle_coin)
            tickers = client.tickers(config.exchanges.bybit.settle_coin)
        else:
            instruments = client.instruments()
            tickers = client.tickers()

        write_parquet(instruments, raw_path(config.paths.data_root, exchange, "instruments"))
        write_parquet(tickers, raw_path(config.paths.data_root, exchange, "tickers"))

        if symbols:
            selected = list(dict.fromkeys(["BTCUSDT", *symbols]))
        else:
            selected = _eligible_symbols_from_current_snapshot(
                exchange, instruments, tickers, config, end, all_eligible
            )

        def symbol_is_cached(symbol: str) -> bool:
            paths = [
                raw_path(config.paths.data_root, exchange, "klines", symbol),
                raw_path(config.paths.data_root, exchange, "funding", symbol),
                raw_path(config.paths.data_root, exchange, "open_interest", symbol),
            ]
            return all(path.exists() and path.stat().st_size > 0 for path in paths)

        symbols_to_fetch = [
            symbol for symbol in selected if not (skip_existing and symbol_is_cached(symbol))
        ]

        def collect_symbol(symbol: str) -> str:
            local_client = create_client(exchange, config)
            try:
                klines = local_client.klines(symbol, start, end, config.experiment.base_interval)
                funding = local_client.funding_history(symbol, start, end)
                oi = local_client.open_interest(symbol, start, end, config.experiment.base_interval)
                write_parquet(klines, raw_path(config.paths.data_root, exchange, "klines", symbol))
                write_parquet(funding, raw_path(config.paths.data_root, exchange, "funding", symbol))
                write_parquet(oi, raw_path(config.paths.data_root, exchange, "open_interest", symbol))
                return symbol
            finally:
                local_client.close()

        failed: list[str] = []
        if workers > 1 and symbols_to_fetch:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(collect_symbol, symbol): symbol for symbol in symbols_to_fetch}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - one flaky symbol must not kill the run
                        failed.append(symbol)
                        print(f"[collect] {symbol} failed: {exc}", flush=True)
            if failed:
                print(
                    f"[collect] {len(failed)} symbols failed and can be resumed with "
                    f"--skip-existing: {','.join(sorted(failed))}",
                    flush=True,
                )
            return selected

        for symbol in symbols_to_fetch:
            try:
                klines = client.klines(symbol, start, end, config.experiment.base_interval)
                funding = client.funding_history(symbol, start, end)
                oi = client.open_interest(symbol, start, end, config.experiment.base_interval)
            except Exception as exc:  # noqa: BLE001 - one flaky symbol must not kill the run
                failed.append(symbol)
                print(f"[collect] {symbol} failed: {exc}", flush=True)
                continue
            write_parquet(klines, raw_path(config.paths.data_root, exchange, "klines", symbol))
            write_parquet(funding, raw_path(config.paths.data_root, exchange, "funding", symbol))
            write_parquet(oi, raw_path(config.paths.data_root, exchange, "open_interest", symbol))
        if failed:
            print(
                f"[collect] {len(failed)} symbols failed and can be resumed with "
                f"--skip-existing: {','.join(sorted(failed))}",
                flush=True,
            )
        return selected
    finally:
        client.close()


def load_raw_exchange(exchange: str, config: ExperimentConfig) -> tuple[pd.DataFrame, ...]:
    root = config.paths.data_root
    instruments = _read_optional_parquet(raw_path(root, exchange, "instruments"))
    tickers = _read_optional_parquet(raw_path(root, exchange, "tickers"))
    klines = _concat_symbol_files(root / "raw" / exchange / "klines")
    funding = _concat_symbol_files(root / "raw" / exchange / "funding")
    oi = _concat_symbol_files(root / "raw" / exchange / "open_interest")
    return instruments, tickers, klines, funding, oi


def build_features_from_raw(
    config: ExperimentConfig,
    exchanges: list[str] | None = None,
) -> pd.DataFrame:
    exchanges = exchanges or ["bybit", "binance"]
    frames: list[pd.DataFrame] = []
    for exchange in exchanges:
        instruments, tickers, klines, funding, oi = load_raw_exchange(exchange, config)
        if klines.empty:
            continue
        klines = apply_universe_flags(klines, instruments, tickers, config)
        features = build_feature_table(klines, funding, oi, instruments, config)
        labeled = add_future_labels(features)
        frames.append(labeled)
    if not frames:
        raise FileNotFoundError("No raw klines found. Run collect first.")
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["exchange", "symbol", "bar_open_time"]
    )
    output = processed_path(
        config.paths.data_root,
        config.experiment.name,
        "perp_pressure_features.parquet",
    )
    write_parquet(out, output)
    return out


def load_processed_features(config: ExperimentConfig) -> pd.DataFrame:
    path = processed_path(
        config.paths.data_root,
        config.experiment.name,
        "perp_pressure_features.parquet",
    )
    return read_parquet(path)


def build_v03_features_from_raw(config: ExperimentConfig) -> pd.DataFrame:
    exchange = "bybit"
    instruments, tickers, klines, funding, oi = load_raw_exchange(exchange, config)
    if klines.empty:
        raise FileNotFoundError("No Bybit raw klines found. Run collect --exchange bybit first.")
    klines = apply_universe_flags(klines, instruments, tickers, config)
    features = build_feature_table(klines, funding, oi, instruments, config)
    labeled = add_future_labels(features)
    out = labeled
    output = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    write_parquet(out, output)
    return out


def load_v03_features(config: ExperimentConfig) -> pd.DataFrame:
    path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    return read_parquet(path)


def run_v03_reports_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v03_reports_from_feature_path(features_path, instruments, config, config.paths.report_root)


def run_v03a_attribution_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v03a_attribution(features_path, instruments, config)


def run_v04_regime_liquidity_gate_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v04_regime_liquidity_gate(features_path, instruments, config)


def run_v05_paper_live_from_features(
    config: ExperimentConfig,
    v05_config: V05Config,
    days: int | None = 30,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v05_paper_live(features_path, instruments, config, v05_config, days=days)


def run_v06a3_paper_live_from_features(
    config: ExperimentConfig,
    v06a3_config: V06A3Config,
    days: int | None = 30,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06a3_paper_live(features_path, instruments, config, v06a3_config, days=days)


def run_v06a_reclaim_alpha_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06a_reclaim_alpha(features_path, instruments, config)


def run_v06a1_impulse_reclaim_validation_from_features(
    config: ExperimentConfig,
    v06a1_config: V06A1Config,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06a1_impulse_reclaim_validation(features_path, instruments, config, v06a1_config)


def collect_v06a2_execution_data_from_features(
    config: ExperimentConfig,
    v06a2_config: V06A2Config,
    source: str = "api",
    max_symbol_days: int | None = None,
    symbol_day_offset: int = 0,
    public_trade_workers: int = 4,
) -> Path:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return collect_v06a2_1m_execution_data(
        features_path,
        instruments,
        config,
        v06a2_config,
        source=source,
        max_symbol_days=max_symbol_days,
        symbol_day_offset=symbol_day_offset,
        public_trade_workers=public_trade_workers,
    )


def run_v06a2_execution_concentration_from_features(
    config: ExperimentConfig,
    v06a2_config: V06A2Config,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06a2_execution_concentration(features_path, instruments, config, v06a2_config)


def run_v06a31_entry_gate_revalidation_from_features(
    config: ExperimentConfig,
    v06a2_config: V06A2Config,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06a31_entry_gate_revalidation(features_path, instruments, config, v06a2_config)


def run_v06b_flush_reversal_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06b_flush_reversal(features_path, instruments, config)


def run_v06c_attack_regime_detector_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06c_attack_regime_detector(features_path, instruments, config)


def run_v07a_motif_atlas_from_features(
    config: ExperimentConfig,
    symbol_offset: int = 0,
    symbol_limit: int | None = None,
    batch_only: bool = False,
    assemble_only: bool = False,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07a_motif_atlas(
        features_path,
        instruments,
        config,
        symbol_offset=symbol_offset,
        symbol_limit=symbol_limit,
        batch_only=batch_only,
        assemble_only=assemble_only,
    )


def run_v07a1_mir1_validation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07a1_mir1_validation(features_path, instruments, config)


def run_v07a1_1m_reconciliation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07a1_1m_reconciliation(features_path, instruments, config)


def run_v07a2_mir1_paper_live_from_features(
    config: ExperimentConfig,
    v07a2_config: V07A2Config,
    days: int | None = 30,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07a2_mir1_paper_live(features_path, instruments, config, v07a2_config, days=days)


def run_v07d2_cic_mir1_paper_live_from_features(
    config: ExperimentConfig,
    v07d2_config: V07A2Config,
    days: int | None = 30,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07d2_cic_mir1_paper_live(features_path, instruments, config, v07d2_config, days=days)


def run_v07b_neighbor_graph_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07b_neighbor_graph(features_path, instruments, config)


def run_v07b1_nir1_validation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07b1_nir1_validation(features_path, instruments, config)


def run_v07b2_neighbor_edge_attribution_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07b2_neighbor_edge_attribution(features_path, instruments, config)


def run_v07c_leader_beta_rotation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07c_leader_beta_rotation(features_path, instruments, config)


def run_v07c1_gate_width_audit_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07c1_gate_width_audit(features_path, instruments, config)


def run_v07c2_leader_beta_continuation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07c2_leader_beta_continuation(features_path, instruments, config)


def run_v07d_co_impulse_continuation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07d_co_impulse_continuation(features_path, instruments, config)


def run_v07d1_cic_execution_integration_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v07d1_cic_execution_integration(features_path, instruments, config)


def run_v09a_cluster_impulse_graph_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v09a_cluster_impulse_graph(features_path, instruments, config)


def run_v09b_portfolio_ranking_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v09b_portfolio_ranking(features_path, instruments, config)


def run_v09b2_ranking_failure_attribution() -> dict[str, Path]:
    return write_v09b2_ranking_failure_attribution()


def run_v09c_orderflow_capacity_ranking() -> dict[str, Path]:
    return write_v09c_orderflow_capacity_ranking()


def run_v09e_orderbook_capacity_ranking() -> dict[str, Path]:
    return write_v09e_orderbook_capacity_ranking()


def run_v09e1_historical_orderbook_replay(
    *,
    max_files: int | None = None,
    download: bool = True,
    run_ranking: bool = True,
) -> dict[str, Path]:
    return run_historical_orderbook_replay(
        V09E1Config(max_files=max_files, download=download, run_ranking=run_ranking)
    )


def run_v09e2_upside_vacuum_validation_report(
    *,
    max_files: int | None = None,
    download: bool = True,
) -> dict[str, Path]:
    return run_v09e2_upside_vacuum_validation(V09E2Config(max_files=max_files, download=download))


def run_v09d_cic_capacity_architecture_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v09d_cic_capacity_architecture(features_path, instruments, config)


def run_v09d1_burst_capacity_execution_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v09d1_burst_capacity_execution(features_path, instruments, config)


def run_v10a_cic_basket_portfolio_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v10a_cic_basket_portfolio(features_path, instruments, config)


def run_v10b_slot_turnover_attribution_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v10b_slot_turnover_attribution(features_path, instruments, config)


def run_v10c_burst_phase_allocation_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v10c_burst_phase_allocation(features_path, instruments, config)


def run_v10d_late_burst_overflow_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v10d_late_burst_overflow(features_path, instruments, config)


def run_v11_orderflow_burst_ranking_from_features(
    config: ExperimentConfig,
    v11_config: V11Config | None = None,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    v10a_cfg = V10AConfig()
    trades = _load_or_build_trades(
        features_path, instruments, config, ensure_dir(v10a_cfg.v09d_root), v10a_cfg
    )
    return write_v11_orderflow_burst_ranking(trades, v11_config or V11Config())


def run_v12s_short_motif_atlas_from_features(
    config: ExperimentConfig,
    short_config: ShortAtlasConfig | None = None,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v12s_short_motif_atlas(features_path, instruments, config, short_config or ShortAtlasConfig())


def run_v12s2_long_risk_off_overlay_from_features(
    config: ExperimentConfig,
    risk_off_config: RiskOffConfig | None = None,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v12s2_long_risk_off_overlay(features_path, instruments, config, risk_off_config or RiskOffConfig())


def run_v12s3_current_stack_risk_off_from_features(
    config: ExperimentConfig,
    stack_config: CurrentStackConfig | None = None,
) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v12s3_current_stack_risk_off(
        features_path, instruments, config, stack_config or CurrentStackConfig()
    )


def run_v3_4_true_short_sleeve_from_features(
    config: ExperimentConfig,
    sleeve_config: V34Config | None = None,
) -> dict[str, Path]:
    """v3.4 true-short-sleeve research line (SS1A..SS3B + Fast/Swing + 3-action)."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v3_4_true_short_sleeve(
        features_path, instruments, config, sleeve_config or V34Config()
    )


def run_v3_3_failure_path_search_from_features(
    config: ExperimentConfig,
    v33_config: V33Config | None = None,
    *,
    use_synthetic_fitness: bool = False,
) -> dict[str, Path]:
    """v3.3 ACO + GA + SA meta-search over the v1.2s3 long risk-off stack."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v3_3_failure_path_search(
        features_path,
        instruments,
        config,
        v33_config or V33Config(),
        use_synthetic_fitness=use_synthetic_fitness,
    )


def run_v3_5_failure_risk_layer_bridge_from_features(
    config: ExperimentConfig,
    v35_config: V35Config | None = None,
) -> dict[str, Path]:
    """v3.5 Failure Risk Layer Bridge: F0..F5 × B0..B3 over the current long stack."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v3_5_failure_risk_layer_bridge(
        features_path, instruments, config, v35_config or V35Config()
    )


def run_v4s_failure_state_graph_from_features(
    config: ExperimentConfig,
    v4s_config: V4SConfig | None = None,
) -> dict[str, Path]:
    """v4S Failure State Graph: 3 paths × 7 actions atop the current long stack."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v4s_failure_state_graph(
        features_path, instruments, config, v4s_config or V4SConfig()
    )


def run_v6s_path_c_short_validation_from_features(
    config: ExperimentConfig,
    v6s_config: V6SConfig | None = None,
) -> dict[str, Path]:
    """v6S Path C Short Validation — discipline-grade test of the v4S survivor."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v6s_path_c_short_validation(
        features_path, instruments, config, v6s_config or V6SConfig()
    )


def run_v7s_short_alpha_from_features(
    config: ExperimentConfig,
    v7s_config: V7SConfig | None = None,
) -> dict[str, Path]:
    """v7S Short Alpha Exploration — orthogonal short lane (Direction E only this commit)."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v7s_short_alpha(
        features_path, instruments, config, v7s_config or V7SConfig()
    )


def run_failure_overlay_shadow_from_features(
    config: ExperimentConfig,
    shadow_config: ShadowConfig | None = None,
) -> dict[str, Path]:
    """F3 / F5 live shadow recorder — idempotent re-run against the v0.9D cache."""
    features_path = (
        config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = _read_optional_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_failure_overlay_shadow(
        features_path, instruments, config, shadow_config or ShadowConfig()
    )


def run_v10_short_mirror_failure_from_features(config: ExperimentConfig) -> dict[str, Path]:
    feature_path = (
        Path(config.paths.data_root)
        / "processed"
        / "v0_3"
        / "perp_pressure_features_all_eligible.parquet"
    )
    instruments = pd.read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v10_short_mirror_failure(feature_path, instruments, config)


def run_reports_from_features(
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
) -> dict[str, Path]:
    df = load_processed_features(config)
    return write_reports(df, config, universe_col)


def run_v01_reports_from_features(
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
) -> dict[str, Path]:
    df = load_processed_features(config)
    return write_v01_reports(df, config, universe_col)


def _merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + pd.Timedelta(minutes=1):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _v01_holdout_prepared(config: ExperimentConfig) -> pd.DataFrame:
    df = load_processed_features(config)
    prepared = prepare_v01_dataset(df, config)
    _, holdout = development_holdout_split(prepared, config.validation.final_holdout_months)
    return holdout.copy()


def _v01_holdout_events(config: ExperimentConfig, universe_col: str) -> pd.DataFrame:
    holdout = _v01_holdout_prepared(config)
    universe = holdout[universe_col].fillna(False) if universe_col in holdout.columns else True
    signal_mask = (
        holdout["short_squeeze_signal_event"].fillna(False)
        | holdout["momentum_ignition_signal_event"].fillna(False)
    )
    return holdout[universe & signal_mask].copy()


def collect_v01_1m_execution_data(
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
    exchange: str = "bybit",
    source: str = "api",
    max_symbol_days: int | None = None,
    symbol_day_offset: int = 0,
    public_trade_workers: int = 4,
) -> Path:
    if exchange != "bybit":
        raise ValueError("v0.1 1m execution validation currently supports Bybit only")
    events = _v01_holdout_events(config, universe_col)
    if source == "public-trades":
        return collect_v01_1m_public_trades(
            config, events, exchange, max_symbol_days, symbol_day_offset, public_trade_workers
        )
    out_dir = ensure_dir(config.paths.data_root / "raw" / exchange / "klines_1m_execution")
    if events.empty:
        return out_dir
    client = BybitClient(str(config.exchanges.bybit.base_url), config.exchanges.bybit.category)
    try:
        for symbol, group in events.groupby("symbol", sort=False):
            intervals = []
            for row in group.itertuples(index=False):
                start = pd.Timestamp(row.feature_time)
                end = start + pd.Timedelta(hours=16)
                intervals.append((start, end))
            frames = []
            for start, end in _merge_intervals(intervals):
                frames.append(client.klines(symbol, start, end, "1m"))
            if frames:
                data = (
                    pd.concat(frames, ignore_index=True)
                    .drop_duplicates(["exchange", "symbol", "bar_open_time"])
                    .sort_values("bar_open_time")
                )
                write_parquet(data, out_dir / f"{symbol}.parquet")
        return out_dir
    finally:
        client.close()


def collect_v01_1m_public_trades(
    config: ExperimentConfig,
    events: pd.DataFrame,
    exchange: str = "bybit",
    max_symbol_days: int | None = None,
    symbol_day_offset: int = 0,
    public_trade_workers: int = 4,
) -> Path:
    out_dir = ensure_dir(config.paths.data_root / "raw" / exchange / "klines_1m_execution_public")
    cache_root = ensure_dir(config.paths.data_root / "raw" / exchange / "public_trading")
    if events.empty:
        return out_dir
    pairs: list[tuple[str, pd.Timestamp]] = []
    for row in events.itertuples(index=False):
        start = pd.Timestamp(row.feature_time).floor("D")
        end = (pd.Timestamp(row.feature_time) + pd.Timedelta(hours=16)).floor("D")
        day = start
        while day <= end:
            pairs.append((str(row.symbol), day))
            day += pd.Timedelta(days=1)
    pairs = sorted(set(pairs), key=lambda item: (item[0], item[1]))
    if symbol_day_offset:
        pairs = pairs[symbol_day_offset:]
    if max_symbol_days is not None:
        pairs = pairs[:max_symbol_days]

    def fetch(pair: tuple[str, pd.Timestamp]) -> tuple[str, Path | None]:
        symbol, day = pair
        path = download_public_trading_day(symbol, day, cache_root)
        return symbol, path

    workers = max(1, public_trade_workers)
    if workers == 1:
        results = [fetch(pair) for pair in pairs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch, pair) for pair in pairs]
            for future in as_completed(futures):
                results.append(future.result())

    by_symbol: dict[str, list[pd.DataFrame]] = {}
    for symbol, path in results:
        if path is None:
            continue
        minute = public_trades_to_1m_ohlcv(path)
        if minute.empty:
            continue
        by_symbol.setdefault(symbol, []).append(minute)
    for symbol, frames in by_symbol.items():
        out_path = out_dir / f"{symbol}.parquet"
        if out_path.exists():
            frames.insert(0, read_parquet(out_path))
        data = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["exchange", "symbol", "bar_open_time"])
            .sort_values("bar_open_time")
        )
        write_parquet(data, out_path)
    return out_dir


def _load_v01_1m_bars(
    config: ExperimentConfig,
    exchange: str = "bybit",
    source: str = "api",
) -> pd.DataFrame:
    dataset = "klines_1m_execution_public" if source == "public-trades" else "klines_1m_execution"
    path = config.paths.data_root / "raw" / exchange / dataset
    return _concat_symbol_files(path)


def run_v01_1m_execution_validation(
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
    exchange: str = "bybit",
    source: str = "api",
) -> dict[str, Path]:
    holdout = _v01_holdout_prepared(config)
    universe = holdout[universe_col].fillna(False) if universe_col in holdout.columns else True
    signals_15m = holdout[universe].copy()
    minute_bars = _load_v01_1m_bars(config, exchange, source)
    if minute_bars.empty:
        raise FileNotFoundError("No 1m execution bars found. Run collect-v01-1m first.")

    def vol_regime_rule(row: pd.Series) -> ExecutionRule:
        vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
        if pd.isna(vol_pct) or vol_pct < 40:
            return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
        if vol_pct < 80:
            return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)

    configs: list[tuple[str, ExecutionRule, object | None]] = [
        ("fast", config.execution.rules["fast"], None),
        ("swing", config.execution.rules["swing"], None),
        ("vol_regime_fast", config.execution.rules["fast"], vol_regime_rule),
    ]
    path_signals = {
        "short_squeeze": "short_squeeze_signal_event",
        "momentum_ignition": "momentum_ignition_signal_event",
    }
    frames = []
    for path_name, signal_col in path_signals.items():
        for policy in ENTRY_POLICIES:
            for rule_name, rule, resolver in configs:
                for cost in [5, 10]:
                    frames.append(
                        simulate_1m_execution(
                            signals_15m,
                            minute_bars,
                            signal_col,
                            path_name,
                            policy,
                            rule_name,
                            rule,
                            cost,
                            resolver,
                        )
                    )
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return write_1m_execution_outputs(trades, Path("reports/v0_1"))


def run_v02_execution_reality_check(
    base_config: ExperimentConfig,
    v02_config: V02Config | None = None,
    source: str = "public-trades",
    include_baselines: bool = True,
) -> dict[str, Path]:
    v02_config = v02_config or load_v02_config()
    holdout = _v01_holdout_prepared(base_config)
    universe = (
        holdout[v02_config.universe_col].fillna(False)
        if v02_config.universe_col in holdout.columns
        else True
    )
    signal_rows = holdout[universe].copy()

    one_min_summary_path = Path("reports/v0_1/entry_policy_1m_comparison.csv")
    one_min_summary = pd.read_csv(one_min_summary_path) if one_min_summary_path.exists() else pd.DataFrame()
    if not one_min_summary.empty:
        one_min_summary = one_min_summary.copy()
        # Map v0.1 path/policy rows to frozen candidate ids.
        candidate_map = {
            (item.path_name, item.entry_policy, item.execution_rule): item.candidate
            for item in v02_config.candidates
        }
        one_min_summary["candidate"] = [
            candidate_map.get((row.path_name, row.entry_policy, row.execution_rule), "")
            for row in one_min_summary.itertuples(index=False)
        ]
        one_min_summary = one_min_summary[one_min_summary["candidate"].ne("")]

    cache_root = (
        base_config.paths.data_root / "raw" / "bybit" / "public_trading"
        if source == "public-trades"
        else None
    )
    return write_v02_reports(
        signal_rows,
        one_min_summary,
        cache_root,
        base_config,
        v02_config,
        include_baselines=include_baselines,
    )
