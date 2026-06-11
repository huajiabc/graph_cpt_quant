from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ExperimentSection(BaseModel):
    name: str = "v0"
    timezone: str = "UTC"
    base_interval: Literal["15m"] = "15m"
    feature_windows: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    label_windows: list[str] = Field(default_factory=lambda: ["4h", "12h"])
    direction: Literal["long_only"] = "long_only"
    start_days: int = 365


class PathSection(BaseModel):
    data_root: Path = Path("data")
    report_root: Path = Path("reports/v0")


class BinanceConfig(BaseModel):
    enabled: bool = True
    role: str = "smoke_and_recent_oi_sanity"
    market: str = "usdt_m_perp"
    base_url: HttpUrl = "https://fapi.binance.com"
    oi_max_history_days: int = 30


class BybitConfig(BaseModel):
    enabled: bool = True
    role: str = "main_12m_experiment"
    category: str = "linear"
    settle_coin: str = "USDT"
    base_url: HttpUrl = "https://api.bybit.com"


class ExchangesConfig(BaseModel):
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    bybit: BybitConfig = Field(default_factory=BybitConfig)


class UniverseConfig(BaseModel):
    modes: list[str] = Field(
        default_factory=lambda: ["static_current_top30", "dynamic_monthly_top30"]
    )
    top_n: int = 30
    dynamic_lookback_days: int = 30
    min_listing_age_days: int = 365
    exclude_stablecoins: bool = True
    max_missing_ratio: float = 0.05


class TimestampConfig(BaseModel):
    primary_key_time: Literal["bar_open_time"] = "bar_open_time"
    feature_time: Literal["bar_close_time"] = "bar_close_time"
    entry_time: Literal["next_bar_open_time"] = "next_bar_open_time"


class FeatureConfig(BaseModel):
    rolling_window_days: int = 30
    min_history_days: int = 14
    percentile_method: Literal["current_vs_prior_window"] = "current_vs_prior_window"
    funding_asof_max_age_intervals: int = 2
    oi_value_usdt: bool = True


class EventConfig(BaseModel):
    deduplicate: bool = True
    cooldown_bars_4h: int = 16
    cooldown_bars_12h: int = 48
    one_position_per_symbol: bool = True


class ShortSqueezeRule(BaseModel):
    btc_ret_4h_min: float = -0.015
    oi_percentile_grid: list[float] = Field(default_factory=lambda: [70, 75, 80])
    funding_percentile_grid: list[float] = Field(default_factory=lambda: [50, 60, 70, 80, 90])
    funding_z_max: float = 0.5
    volume_z_grid: list[float] = Field(default_factory=lambda: [0.8, 1.0, 1.2])


class MomentumIgnitionRule(BaseModel):
    btc_ret_4h_min: float = -0.015
    ret_4h_percentile_min: float = 70
    volume_z_grid: list[float] = Field(default_factory=lambda: [0.8, 1.0, 1.2])
    oi_percentile_min_grid: list[float] = Field(default_factory=lambda: [60, 70])
    oi_percentile_max: float = 95
    funding_percentile_max_grid: list[float] = Field(default_factory=lambda: [60, 70, 80])


class CrowdedLongRiskRule(BaseModel):
    funding_percentile_min: float = 90
    oi_percentile_min: float = 75
    ret_4h_percentile_max: float = 50
    mode: str = "veto_and_report"


class PathRulesConfig(BaseModel):
    short_squeeze: ShortSqueezeRule = Field(default_factory=ShortSqueezeRule)
    momentum_ignition: MomentumIgnitionRule = Field(default_factory=MomentumIgnitionRule)
    crowded_long_risk: CrowdedLongRiskRule = Field(default_factory=CrowdedLongRiskRule)


class ExecutionRule(BaseModel):
    tp: float
    sl: float
    max_hold_bars: int


class ExecutionConfig(BaseModel):
    entry: Literal["next_15m_open"] = "next_15m_open"
    ambiguity_policy: Literal["sl_first", "tp_first", "skip"] = "sl_first"
    ambiguity_policies: list[Literal["sl_first", "tp_first", "skip"]] = Field(
        default_factory=lambda: ["sl_first", "tp_first", "skip"]
    )
    cost_single_side_bps: list[float] = Field(default_factory=lambda: [5, 10, 20])
    rules: dict[str, ExecutionRule] = Field(
        default_factory=lambda: {
            "fast": ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16),
            "swing": ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48),
        }
    )


class ValidationConfig(BaseModel):
    walk_forward: bool = True
    final_holdout_months: int = 3
    report_bar_level: bool = True
    report_event_level: bool = True
    primary_level: Literal["event", "bar"] = "event"
    max_symbol_contribution: float = 0.35
    max_month_contribution: float = 0.35
    sample_shrinkage_k: int = 100


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    paths: PathSection = Field(default_factory=PathSection)
    exchanges: ExchangesConfig = Field(default_factory=ExchangesConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    timestamps: TimestampConfig = Field(default_factory=TimestampConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    path_rules: PathRulesConfig = Field(default_factory=PathRulesConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @property
    def bars_per_day(self) -> int:
        return 96

    @property
    def rolling_window_bars(self) -> int:
        return self.features.rolling_window_days * self.bars_per_day

    @property
    def min_history_bars(self) -> int:
        return self.features.min_history_days * self.bars_per_day


def load_config(path: str | Path = "configs/v0.yaml") -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return ExperimentConfig.model_validate(payload)
