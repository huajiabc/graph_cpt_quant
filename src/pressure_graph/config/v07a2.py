from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class V07A2Experiment:
    name: str
    mode: str
    strategy_id: str
    primary_candidate: str
    timezone: str
    base_interval: str
    frozen: bool
    paper_live_allowed: bool
    real_live_allowed: bool
    tick_validation_status: str
    historical_validation_status: str
    candidate_rating: str


@dataclass(frozen=True)
class V07A2Exchange:
    name: str
    market: str


@dataclass(frozen=True)
class V07A2MarketGraph:
    density_threshold: float
    low_density_threshold: float
    strict_signal_and_entry_gate: bool


@dataclass(frozen=True)
class V07A2Signal:
    anchor: str
    volume_z_4h_min: float
    ret_4h_min: float
    cooldown_bars: int
    frozen: bool


@dataclass(frozen=True)
class V07A2Candidate:
    candidate: str
    role: str
    universe_top_n: int
    market_gate: str
    event_col: str
    entry_policy: str
    pullback_pct: float
    valid_bars: int
    execution_rule: str
    rationale: str


@dataclass(frozen=True)
class V07A2Baseline:
    baseline: str
    candidate: str
    market_gate: str
    event_col: str
    entry_policy: str
    pullback_pct: float
    valid_bars: int
    execution_rule: str


@dataclass(frozen=True)
class V07A2Baselines:
    enabled: bool
    items: list[V07A2Baseline]


@dataclass(frozen=True)
class V07A2Costs:
    single_side_bps: list[float]


@dataclass(frozen=True)
class V07A2Portfolio:
    primary_max_positions: int
    shadow_max_positions: list[int]
    one_position_per_symbol: bool
    pyramiding: bool
    sizing: str
    ranking: list[str]


@dataclass(frozen=True)
class V07A2Logging:
    save_signals: bool
    save_trades: bool
    save_baselines: bool
    save_daily_summary: bool


@dataclass(frozen=True)
class V07A2Stops:
    stale_data_bars: int
    rolling_trade_window: int
    pause_if_rolling_10bp_net_lte_zero: bool
    pause_if_baseline_lift_lte_zero: bool


@dataclass(frozen=True)
class V07A2Config:
    experiment: V07A2Experiment
    exchange: V07A2Exchange
    market_graph: V07A2MarketGraph
    signal: V07A2Signal
    candidates: list[V07A2Candidate]
    baselines: V07A2Baselines
    costs: V07A2Costs
    portfolio: V07A2Portfolio
    logging: V07A2Logging
    stops: V07A2Stops


def _section(payload: dict, name: str) -> dict:
    value = payload.get(name, {})
    return value if isinstance(value, dict) else {}


def load_v07a2_config(path: str | Path = "configs/v0_7a2_mir1_paper_live.yaml") -> V07A2Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    market_graph = _section(payload, "market_graph")
    baselines = _section(payload, "baselines")
    portfolio = _section(payload, "portfolio")
    stops = _section(payload, "stops")
    return V07A2Config(
        experiment=V07A2Experiment(**_section(payload, "experiment")),
        exchange=V07A2Exchange(**_section(payload, "exchange")),
        market_graph=V07A2MarketGraph(
            density_threshold=float(market_graph.get("density_threshold", 0.10)),
            low_density_threshold=float(market_graph.get("low_density_threshold", 0.02)),
            strict_signal_and_entry_gate=bool(market_graph.get("strict_signal_and_entry_gate", True)),
        ),
        signal=V07A2Signal(**_section(payload, "signal")),
        candidates=[V07A2Candidate(**item) for item in payload.get("candidates", [])],
        baselines=V07A2Baselines(
            enabled=bool(baselines.get("enabled", True)),
            items=[V07A2Baseline(**item) for item in baselines.get("items", [])],
        ),
        costs=V07A2Costs(
            single_side_bps=[
                float(item) for item in _section(payload, "costs").get("single_side_bps", [5, 10, 20, 30, 50])
            ]
        ),
        portfolio=V07A2Portfolio(
            primary_max_positions=int(portfolio.get("primary_max_positions", 3)),
            shadow_max_positions=[int(item) for item in portfolio.get("shadow_max_positions", [1, 5, 10])],
            one_position_per_symbol=bool(portfolio.get("one_position_per_symbol", True)),
            pyramiding=bool(portfolio.get("pyramiding", False)),
            sizing=str(portfolio.get("sizing", "equal_notional")),
            ranking=list(portfolio.get("ranking", [])),
        ),
        logging=V07A2Logging(**_section(payload, "logging")),
        stops=V07A2Stops(
            stale_data_bars=int(stops.get("stale_data_bars", 2)),
            rolling_trade_window=int(stops.get("rolling_trade_window", 100)),
            pause_if_rolling_10bp_net_lte_zero=bool(
                stops.get("pause_if_rolling_10bp_net_lte_zero", True)
            ),
            pause_if_baseline_lift_lte_zero=bool(stops.get("pause_if_baseline_lift_lte_zero", True)),
        ),
    )
