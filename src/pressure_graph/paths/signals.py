from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import pandas as pd

from pressure_graph.config import ExperimentConfig


KEYS = ["exchange", "symbol"]


@dataclass(frozen=True)
class PathVariant:
    path_name: str
    signal_col: str
    params: dict[str, float]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def apply_crowded_veto(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    crowded = out.get("crowded_long_risk", False)
    out["short_squeeze_signal"] = out["short_squeeze_signal_raw"] & ~crowded
    out["momentum_ignition_signal"] = out["momentum_ignition_signal_raw"] & ~crowded
    return out


def add_path_signals(
    df: pd.DataFrame,
    config: ExperimentConfig,
    short_oi_pct: float | None = None,
    short_funding_pct: float | None = None,
    short_volume_z: float | None = None,
    momentum_oi_min_pct: float | None = None,
    momentum_funding_pct: float | None = None,
    momentum_volume_z: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    short_rule = config.path_rules.short_squeeze
    momentum_rule = config.path_rules.momentum_ignition
    crowded_rule = config.path_rules.crowded_long_risk

    short_oi_pct = short_oi_pct if short_oi_pct is not None else 75
    short_funding_pct = short_funding_pct if short_funding_pct is not None else 60
    short_volume_z = short_volume_z if short_volume_z is not None else 1.0
    momentum_oi_min_pct = momentum_oi_min_pct if momentum_oi_min_pct is not None else 60
    momentum_funding_pct = momentum_funding_pct if momentum_funding_pct is not None else 80
    momentum_volume_z = momentum_volume_z if momentum_volume_z is not None else 1.0

    market_ok = _num(out, "btc_ret_4h") > short_rule.btc_ret_4h_min
    short_oi_up = (
        (_num(out, "oi_value_delta_1h_percentile") > short_oi_pct)
        | (_num(out, "oi_value_delta_4h_percentile") > short_oi_pct)
    )
    funding_not_hot = (
        (_num(out, "funding_percentile") < short_funding_pct)
        | (_num(out, "funding_z") < short_rule.funding_z_max)
    )
    price_resilient = (_num(out, "ret_1h") > 0) | (_num(out, "ret_4h") > 0)
    volume_confirm = _num(out, "volume_z_1h") > short_volume_z
    out["short_squeeze_signal_raw"] = (
        out.get("warmup_complete", True)
        & market_ok
        & short_oi_up
        & funding_not_hot
        & price_resilient
        & volume_confirm
    )

    momentum_market_ok = _num(out, "btc_ret_4h") > momentum_rule.btc_ret_4h_min
    price_breakout = _num(out, "ret_4h_percentile") > momentum_rule.ret_4h_percentile_min
    volume_expansion = _num(out, "volume_z_4h") > momentum_volume_z
    oi_moderate = (
        (_num(out, "oi_value_delta_4h_percentile") > momentum_oi_min_pct)
        & (_num(out, "oi_value_delta_4h_percentile") < momentum_rule.oi_percentile_max)
    )
    funding_safe = _num(out, "funding_percentile") < momentum_funding_pct
    out["momentum_ignition_signal_raw"] = (
        out.get("warmup_complete", True)
        & momentum_market_ok
        & price_breakout
        & volume_expansion
        & oi_moderate
        & funding_safe
    )

    out["crowded_long_risk"] = (
        out.get("warmup_complete", True)
        & (_num(out, "funding_percentile") > crowded_rule.funding_percentile_min)
        & (_num(out, "oi_value_delta_4h_percentile") > crowded_rule.oi_percentile_min)
        & (_num(out, "ret_4h_percentile") < crowded_rule.ret_4h_percentile_max)
    )
    return apply_crowded_veto(out)


def _event_flags(signals: pd.Series, cooldown_bars: int) -> pd.Series:
    signal_arr = signals.fillna(False).astype(bool).to_numpy()
    events = [False] * len(signal_arr)
    last_event = -10**9
    for idx, is_signal in enumerate(signal_arr):
        if is_signal and idx - last_event >= cooldown_bars:
            events[idx] = True
            last_event = idx
    return pd.Series(events, index=signals.index)


def add_event_columns(
    df: pd.DataFrame,
    signal_cols: Iterable[str],
    cooldown_bars: int,
) -> pd.DataFrame:
    out = df.sort_values(KEYS + ["bar_open_time"]).copy()
    for signal_col in signal_cols:
        event_col = f"{signal_col}_event"
        out[event_col] = (
            out.groupby(KEYS, group_keys=False, sort=False)[signal_col]
            .apply(lambda s: _event_flags(s, cooldown_bars))
            .reindex(out.index)
        )
    return out


def iter_parameter_variants(config: ExperimentConfig) -> list[PathVariant]:
    variants: list[PathVariant] = []
    short = config.path_rules.short_squeeze
    for oi_pct, funding_pct, volume_z in product(
        short.oi_percentile_grid, short.funding_percentile_grid, short.volume_z_grid
    ):
        variants.append(
            PathVariant(
                "short_squeeze",
                f"short_squeeze_oi{oi_pct:g}_fund{funding_pct:g}_vol{volume_z:g}",
                {
                    "short_oi_pct": oi_pct,
                    "short_funding_pct": funding_pct,
                    "short_volume_z": volume_z,
                },
            )
        )

    momentum = config.path_rules.momentum_ignition
    for oi_min, funding_pct, volume_z in product(
        momentum.oi_percentile_min_grid,
        momentum.funding_percentile_max_grid,
        momentum.volume_z_grid,
    ):
        variants.append(
            PathVariant(
                "momentum_ignition",
                f"momentum_oi{oi_min:g}_fund{funding_pct:g}_vol{volume_z:g}",
                {
                    "momentum_oi_min_pct": oi_min,
                    "momentum_funding_pct": funding_pct,
                    "momentum_volume_z": volume_z,
                },
            )
        )
    return variants
