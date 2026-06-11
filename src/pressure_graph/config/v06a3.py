from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class V06A3Experiment:
    name: str
    mode: str
    strategy_id: str
    primary_candidate: str
    timezone: str
    base_interval: str
    frozen: bool
    real_live_allowed: bool
    tick_validation_status: str


@dataclass(frozen=True)
class V06A3Exchange:
    name: str
    market: str


@dataclass(frozen=True)
class V06A3BtcGate:
    enabled: bool
    required_state: str


@dataclass(frozen=True)
class V06A3Regime:
    btc_gate: V06A3BtcGate
    shadow_states: list[str]


@dataclass(frozen=True)
class V06A3Signal:
    anchor: str
    volume_z_4h_min: float
    ret_4h_min: float
    cooldown_bars: int
    frozen: bool


@dataclass(frozen=True)
class V06A3TransientHot:
    turnover_rank_30d_max: int
    turnover_rank_90d_min: int


@dataclass(frozen=True)
class V06A3Liquidity:
    transient_hot_veto: bool
    transient_hot: V06A3TransientHot


@dataclass(frozen=True)
class V06A3Candidate:
    candidate: str
    role: str
    universe_top_n: int
    entry_policy: str
    pullback_pct: float
    valid_bars: int
    execution_rule: str
    rationale: str


@dataclass(frozen=True)
class V06A3Baselines:
    enabled: bool
    kinds: list[str]


@dataclass(frozen=True)
class V06A3Costs:
    single_side_bps: list[float]


@dataclass(frozen=True)
class V06A3Portfolio:
    primary_max_positions: int
    shadow_max_positions: list[int]
    one_position_per_symbol: bool
    pyramiding: bool
    sizing: str
    ranking: list[str]


@dataclass(frozen=True)
class V06A3Logging:
    save_signals: bool
    save_trades: bool
    save_baselines: bool
    save_daily_summary: bool


@dataclass(frozen=True)
class V06A3Stops:
    stale_data_bars: int
    rolling_trade_window: int
    pause_if_rolling_10bp_net_lte_zero: bool
    pause_if_baseline_lift_lte_zero: bool


@dataclass(frozen=True)
class V06A3Config:
    experiment: V06A3Experiment
    exchange: V06A3Exchange
    regime: V06A3Regime
    signal: V06A3Signal
    liquidity: V06A3Liquidity
    candidates: list[V06A3Candidate]
    baselines: V06A3Baselines
    costs: V06A3Costs
    portfolio: V06A3Portfolio
    logging: V06A3Logging
    stops: V06A3Stops


def _section(payload: dict, name: str) -> dict:
    value = payload.get(name, {})
    return value if isinstance(value, dict) else {}


def load_v06a3_config(path: str | Path = "configs/v0_6a3_paper_live.yaml") -> V06A3Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    regime = _section(payload, "regime")
    btc_gate = regime.get("btc_gate", {})
    liquidity = _section(payload, "liquidity")
    transient_hot = liquidity.get("transient_hot", {})
    baselines = _section(payload, "baselines")
    portfolio = _section(payload, "portfolio")
    stops = _section(payload, "stops")
    return V06A3Config(
        experiment=V06A3Experiment(**_section(payload, "experiment")),
        exchange=V06A3Exchange(**_section(payload, "exchange")),
        regime=V06A3Regime(
            btc_gate=V06A3BtcGate(
                enabled=bool(btc_gate.get("enabled", True)),
                required_state=str(btc_gate.get("required_state", "BTC_up")),
            ),
            shadow_states=list(regime.get("shadow_states", ["BTC_chop", "BTC_down"])),
        ),
        signal=V06A3Signal(**_section(payload, "signal")),
        liquidity=V06A3Liquidity(
            transient_hot_veto=bool(liquidity.get("transient_hot_veto", True)),
            transient_hot=V06A3TransientHot(
                turnover_rank_30d_max=int(transient_hot.get("turnover_rank_30d_max", 30)),
                turnover_rank_90d_min=int(transient_hot.get("turnover_rank_90d_min", 50)),
            ),
        ),
        candidates=[V06A3Candidate(**item) for item in payload.get("candidates", [])],
        baselines=V06A3Baselines(
            enabled=bool(baselines.get("enabled", True)),
            kinds=list(
                baselines.get(
                    "kinds",
                    [
                        "entry_only_reclaim",
                        "volume_shock_next_open",
                        "volume_shock_pullback_only",
                        "matched_random_reclaim",
                    ],
                )
            ),
        ),
        costs=V06A3Costs(
            single_side_bps=[
                float(item) for item in _section(payload, "costs").get("single_side_bps", [5, 10, 20, 30, 50])
            ]
        ),
        portfolio=V06A3Portfolio(
            primary_max_positions=int(portfolio.get("primary_max_positions", 3)),
            shadow_max_positions=[int(item) for item in portfolio.get("shadow_max_positions", [1, 5, 10])],
            one_position_per_symbol=bool(portfolio.get("one_position_per_symbol", True)),
            pyramiding=bool(portfolio.get("pyramiding", False)),
            sizing=str(portfolio.get("sizing", "equal_notional")),
            ranking=list(portfolio.get("ranking", [])),
        ),
        logging=V06A3Logging(**_section(payload, "logging")),
        stops=V06A3Stops(
            stale_data_bars=int(stops.get("stale_data_bars", 2)),
            rolling_trade_window=int(stops.get("rolling_trade_window", 100)),
            pause_if_rolling_10bp_net_lte_zero=bool(
                stops.get("pause_if_rolling_10bp_net_lte_zero", True)
            ),
            pause_if_baseline_lift_lte_zero=bool(stops.get("pause_if_baseline_lift_lte_zero", True)),
        ),
    )
