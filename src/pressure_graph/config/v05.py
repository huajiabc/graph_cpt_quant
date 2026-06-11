from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class V05Experiment:
    name: str
    mode: str
    candidate: str
    timezone: str
    base_interval: str


@dataclass(frozen=True)
class V05Exchange:
    name: str
    market: str


@dataclass(frozen=True)
class V05TransientHot:
    enabled: bool
    turnover_rank_30d_max: int
    turnover_rank_90d_min: int


@dataclass(frozen=True)
class V05Universe:
    mode: str
    min_listing_age_days: int
    exclude_stablecoins: bool
    turnover_rank_30d_max: int
    transient_hot: V05TransientHot


@dataclass(frozen=True)
class V05BtcGate:
    enabled: bool
    required_state: str


@dataclass(frozen=True)
class V05Regime:
    btc_gate: V05BtcGate
    shadow_states: list[str]


@dataclass(frozen=True)
class V05Signal:
    path: str
    frozen: bool


@dataclass(frozen=True)
class V05Entry:
    type: str
    pullback_pct: float
    valid_bars: int
    fill_models: list[str]


@dataclass(frozen=True)
class V05Exit:
    type: str
    tp: float
    sl: float
    max_hold_bars: int


@dataclass(frozen=True)
class V05Costs:
    single_side_bps: list[float]


@dataclass(frozen=True)
class V05Portfolio:
    primary_max_positions: int
    shadow_max_positions: list[int]
    one_position_per_symbol: bool
    pyramiding: bool
    sizing: str
    ranking: list[str]


@dataclass(frozen=True)
class V05Logging:
    save_signals: bool
    save_trades: bool
    save_skipped: bool
    save_daily_summary: bool


@dataclass(frozen=True)
class V05Stops:
    disable_if_btc_not_up: bool
    stale_data_bars: int
    rolling_trade_window: int
    pause_if_rolling_10bp_net_lte_zero: bool
    pause_if_matched_lift_lte_zero: bool


@dataclass(frozen=True)
class V05Config:
    experiment: V05Experiment
    exchange: V05Exchange
    universe: V05Universe
    regime: V05Regime
    signal: V05Signal
    entry: V05Entry
    exit: V05Exit
    costs: V05Costs
    portfolio: V05Portfolio
    logging: V05Logging
    stops: V05Stops


def _section(payload: dict, name: str) -> dict:
    value = payload.get(name, {})
    return value if isinstance(value, dict) else {}


def load_v05_config(path: str | Path = "configs/v0_5_paper_live.yaml") -> V05Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    universe = _section(payload, "universe")
    regime = _section(payload, "regime")
    portfolio = _section(payload, "portfolio")
    transient_hot = universe.get("transient_hot", {})
    btc_gate = regime.get("btc_gate", {})
    return V05Config(
        experiment=V05Experiment(**_section(payload, "experiment")),
        exchange=V05Exchange(**_section(payload, "exchange")),
        universe=V05Universe(
            mode=str(universe.get("mode", "dynamic_top30_monthly_lock")),
            min_listing_age_days=int(universe.get("min_listing_age_days", 365)),
            exclude_stablecoins=bool(universe.get("exclude_stablecoins", True)),
            turnover_rank_30d_max=int(universe.get("turnover_rank_30d_max", 30)),
            transient_hot=V05TransientHot(
                enabled=bool(transient_hot.get("enabled", True)),
                turnover_rank_30d_max=int(transient_hot.get("turnover_rank_30d_max", 30)),
                turnover_rank_90d_min=int(transient_hot.get("turnover_rank_90d_min", 50)),
            ),
        ),
        regime=V05Regime(
            btc_gate=V05BtcGate(
                enabled=bool(btc_gate.get("enabled", True)),
                required_state=str(btc_gate.get("required_state", "BTC_up")),
            ),
            shadow_states=list(regime.get("shadow_states", ["BTC_chop", "BTC_down"])),
        ),
        signal=V05Signal(**_section(payload, "signal")),
        entry=V05Entry(**_section(payload, "entry")),
        exit=V05Exit(**_section(payload, "exit")),
        costs=V05Costs(
            single_side_bps=[float(item) for item in _section(payload, "costs").get("single_side_bps", [5, 10, 20])]
        ),
        portfolio=V05Portfolio(
            primary_max_positions=int(portfolio.get("primary_max_positions", 3)),
            shadow_max_positions=[int(item) for item in portfolio.get("shadow_max_positions", [1, 5, 10])],
            one_position_per_symbol=bool(portfolio.get("one_position_per_symbol", True)),
            pyramiding=bool(portfolio.get("pyramiding", False)),
            sizing=str(portfolio.get("sizing", "equal_notional")),
            ranking=list(portfolio.get("ranking", [])),
        ),
        logging=V05Logging(**_section(payload, "logging")),
        stops=V05Stops(**_section(payload, "stops")),
    )
