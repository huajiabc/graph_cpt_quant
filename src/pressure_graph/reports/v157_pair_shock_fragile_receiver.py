"""Causal within-bucket shock transmission into order-book-fragile receivers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
    load_v155_hourly_prices,
)


V155_PANEL_PATH = Path(
    "reports/v15_5_binance_one_percent_depth_imbalance/daily_symbol_panel.parquet"
)
REPORT_ROOT = Path("reports/v15_7_pair_shock_fragile_receiver")
FINDINGS_PATH = Path("docs/v157_pair_shock_fragile_receiver_findings_2026_07_16.md")
CANDIDATE = "VT4_PAIR_SHOCK_TO_FRAGILE_RECEIVER"
REVERSED_CONTROL = "VT4_REVERSED_PROPAGATION"
STALE_CONTROL = "VT4_ONE_DAY_STALE_SIGNAL"
SOURCE_ONLY_CONTROL = "VT4_SOURCE_SHOCK_ONLY_NO_DEPTH"
FROZEN_PAIRS = {
    "BSP01": ("SOLUSDT", "DOGEUSDT"),
    "BSP02": ("1000PEPEUSDT", "WIFUSDT"),
    "BSP03": ("ETHUSDT", "ENAUSDT"),
    "BSP04": ("HBARUSDT", "AVAXUSDT"),
    "BSP05": ("LINKUSDT", "ONDOUSDT"),
    "BSP06": ("XRPUSDT", "XLMUSDT"),
    "BSP07": ("FARTCOINUSDT", "WLDUSDT"),
    "BSP08": ("SEIUSDT", "TIAUSDT"),
}


@dataclass(frozen=True)
class V157Config:
    panel_path: Path = V155_PANEL_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    side_count: int = 2
    retention_count: int = 4
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    bootstrap_block_days: int = 7
    seed: int = 20260720


def load_v157_panel(path: Path = V155_PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    for column in ("decision_time", "source_day", "stale_source_day"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def build_v157_signals(
    panel: pd.DataFrame,
    hourly_prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = hourly_prices.copy()
    prices["feature_time"] = pd.to_datetime(prices["feature_time"], utc=True, errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    hourly_close = prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    daily_close = hourly_close[hourly_close.index.hour == 0]
    rows = []
    for raw_day, local in panel.groupby("decision_time", sort=True, observed=True):
        day = pd.Timestamp(raw_day)
        previous_day = day - pd.Timedelta(days=1)
        if day not in daily_close.index or previous_day not in daily_close.index:
            continue
        indexed = local.set_index("symbol")
        past_return = daily_close.loc[day] / daily_close.loc[previous_day] - 1.0
        btc_past_return = float(past_return[BTC])
        residuals = {
            symbol: float(past_return[symbol])
            - float(indexed.at[symbol, "btc_beta"]) * btc_past_return
            for symbol in FROZEN_SYMBOLS
        }
        for pair_id, pair in FROZEN_PAIRS.items():
            ordered = sorted(pair, key=lambda symbol: (-abs(residuals[symbol]), symbol))
            source, receiver = ordered
            source_residual = residuals[source]
            receiver_fragility = abs(float(indexed.at[receiver, "feature_1pct"]))
            rows.append(
                {
                    "signal_time": day,
                    "source_day": indexed.iloc[0]["source_day"],
                    "pair_id": pair_id,
                    "source_symbol": source,
                    "receiver_symbol": receiver,
                    "source_residual_return": source_residual,
                    "source_direction": float(np.sign(source_residual)),
                    "source_shock": abs(source_residual),
                    "receiver_fragility": receiver_fragility,
                    "propagation_strength": abs(source_residual) * receiver_fragility,
                    "source_btc_beta": float(indexed.at[source, "btc_beta"]),
                    "receiver_btc_beta": float(indexed.at[receiver, "btc_beta"]),
                    "receiver_future_return": float(indexed.at[receiver, "price_return"]),
                    "btc_future_return": float(indexed.iloc[0]["btc_return"]),
                    "period": indexed.iloc[0]["period"],
                }
            )
    return pd.DataFrame(rows).sort_values(["signal_time", "pair_id"]).reset_index(drop=True)


def select_v157_side(
    candidates: list[tuple[str, float]],
    previous: set[str],
    cfg: V157Config = V157Config(),
) -> list[str]:
    ordered = sorted(candidates, key=lambda item: (-item[1], item[0]))
    eligible = {symbol for symbol, _ in ordered[: cfg.retention_count]}
    retained = [symbol for symbol, _ in ordered if symbol in previous and symbol in eligible]
    for symbol, _ in ordered:
        if len(retained) >= cfg.side_count:
            break
        if symbol not in retained:
            retained.append(symbol)
    return retained[: cfg.side_count]


def beta_neutral_v157_weights(
    longs: list[str],
    shorts: list[str],
    betas: dict[str, float],
) -> dict[str, float]:
    if len(longs) == 0 or len(shorts) == 0:
        return {}
    alt = {symbol: 0.5 / len(longs) for symbol in longs}
    alt.update({symbol: -0.5 / len(shorts) for symbol in shorts})
    hedge = -float(sum(weight * betas[symbol] for symbol, weight in alt.items()))
    gross = float(sum(abs(weight) for weight in alt.values()) + abs(hedge))
    weights = {symbol: weight / gross for symbol, weight in alt.items()}
    weights[BTC] = hedge / gross
    return weights


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return float(
        sum(
            abs(previous.get(symbol, 0.0) - current.get(symbol, 0.0))
            for symbol in set(previous) | set(current)
        )
    )


def _ranked_candidates(
    signal: pd.DataFrame,
    score_column: str,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    positive = [
        (str(row.receiver_symbol), float(getattr(row, score_column)))
        for row in signal.itertuples(index=False)
        if float(row.source_direction) > 0
    ]
    negative = [
        (str(row.receiver_symbol), float(getattr(row, score_column)))
        for row in signal.itertuples(index=False)
        if float(row.source_direction) < 0
    ]
    return positive, negative


def build_v157_portfolio(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    cfg: V157Config = V157Config(),
    *,
    candidate: str = CANDIDATE,
    score_column: str = "propagation_strength",
    reverse: bool = False,
    signal_lag_days: int = 0,
    random_seed: int | None = None,
) -> pd.DataFrame:
    signal_lookup = {
        pd.Timestamp(day): local.sort_values("pair_id").reset_index(drop=True)
        for day, local in signals.groupby("signal_time", sort=True, observed=True)
    }
    panel_lookup = {
        pd.Timestamp(day): local.set_index("symbol")
        for day, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    previous_positive: set[str] = set()
    previous_negative: set[str] = set()
    previous_day: pd.Timestamp | None = None
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    for day in sorted(panel_lookup):
        if previous_day is not None and day - previous_day > pd.Timedelta(days=1):
            forced_close = float(sum(abs(weight) for weight in previous_weights.values()))
            rows[-1]["forced_close_turnover"] = forced_close
            rows[-1]["realized_turnover"] = float(rows[-1]["entry_turnover"]) + forced_close
            rows[-1]["primary_net_return"] = float(rows[-1]["gross_return"]) - (
                cfg.one_way_cost * float(rows[-1]["realized_turnover"])
            )
            rows[-1]["stress_net_return"] = float(rows[-1]["gross_return"]) - (
                cfg.stress_one_way_cost * float(rows[-1]["realized_turnover"])
            )
            previous_weights = {}
            previous_positive = set()
            previous_negative = set()
        signal_day = day - pd.Timedelta(days=signal_lag_days)
        signal = signal_lookup.get(signal_day, pd.DataFrame()).copy()
        if rng is not None and not signal.empty:
            receiver_columns = ["receiver_symbol", "receiver_fragility"]
            receivers = signal[receiver_columns].to_numpy(copy=True)
            signal.loc[:, receiver_columns] = receivers[rng.permutation(len(receivers))]
            signal["propagation_strength"] = (
                signal["source_shock"] * signal["receiver_fragility"]
            )
        positive, negative = _ranked_candidates(signal, score_column) if not signal.empty else ([], [])
        active = len(positive) >= cfg.side_count and len(negative) >= cfg.side_count
        if active:
            positive_selected = select_v157_side(positive, previous_positive, cfg)
            negative_selected = select_v157_side(negative, previous_negative, cfg)
        else:
            positive_selected = []
            negative_selected = []
        longs = negative_selected if reverse else positive_selected
        shorts = positive_selected if reverse else negative_selected
        local = panel_lookup[day]
        betas = local["btc_beta"].astype(float).to_dict()
        weights = beta_neutral_v157_weights(longs, shorts, betas)
        gross_return = (
            float(
                sum(
                    weights[symbol] * float(local.at[symbol, "price_return"])
                    for symbol in longs + shorts
                )
                + weights[BTC] * float(local.iloc[0]["btc_return"])
            )
            if weights
            else 0.0
        )
        residual_beta = (
            float(
                sum(weights[symbol] * betas[symbol] for symbol in longs + shorts)
                + weights[BTC]
            )
            if weights
            else 0.0
        )
        gross_notional = float(sum(abs(weight) for weight in weights.values()))
        entry_turnover = _turnover(previous_weights, weights)
        rows.append(
            {
                "candidate": candidate,
                "decision_time": day,
                "signal_time": signal_day,
                "period": local.iloc[0]["period"],
                "active": bool(weights),
                "positive_receivers": "|".join(positive_selected),
                "negative_receivers": "|".join(negative_selected),
                "long_symbols": "|".join(longs),
                "short_symbols": "|".join(shorts),
                "weights_json": json.dumps(weights, sort_keys=True),
                "btc_hedge_weight": weights.get(BTC, 0.0),
                "entry_turnover": entry_turnover,
                "forced_close_turnover": 0.0,
                "realized_turnover": entry_turnover,
                "gross_notional": gross_notional,
                "residual_btc_beta": residual_beta,
                "gross_return": gross_return,
                "primary_net_return": gross_return - cfg.one_way_cost * entry_turnover,
                "stress_net_return": gross_return
                - cfg.stress_one_way_cost * entry_turnover,
            }
        )
        previous_weights = weights
        previous_positive = set(positive_selected)
        previous_negative = set(negative_selected)
        previous_day = day
    return pd.DataFrame(rows)


def build_v157_random_controls(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    cfg: V157Config = V157Config(),
) -> pd.DataFrame:
    signal_lookup = {
        pd.Timestamp(day): local.sort_values("pair_id").reset_index(drop=True)
        for day, local in signals.groupby("signal_time", sort=True, observed=True)
    }
    panel_lookup = {
        pd.Timestamp(day): local.set_index("symbol")
        for day, local in panel.groupby("decision_time", sort=True, observed=True)
    }
    contexts = [(day, signal_lookup[day], panel_lookup[day]) for day in sorted(panel_lookup)]
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        previous_weights: dict[str, float] = {}
        previous_positive: set[str] = set()
        previous_negative: set[str] = set()
        previous_day: pd.Timestamp | None = None
        total_net = 0.0
        total_turnover = 0.0
        active_days = 0
        for day, signal, local in contexts:
            if previous_day is not None and day - previous_day > pd.Timedelta(days=1):
                forced_close = float(sum(abs(weight) for weight in previous_weights.values()))
                total_net -= cfg.one_way_cost * forced_close
                total_turnover += forced_close
                previous_weights = {}
                previous_positive = set()
                previous_negative = set()
            receiver_order = rng.permutation(len(signal))
            receiver_symbols = signal["receiver_symbol"].to_numpy()[receiver_order]
            receiver_fragility = signal["receiver_fragility"].to_numpy(dtype=float)[
                receiver_order
            ]
            direction = signal["source_direction"].to_numpy(dtype=float)
            score = signal["source_shock"].to_numpy(dtype=float) * receiver_fragility
            positive = [
                (str(receiver_symbols[index]), float(score[index]))
                for index in range(len(signal))
                if direction[index] > 0
            ]
            negative = [
                (str(receiver_symbols[index]), float(score[index]))
                for index in range(len(signal))
                if direction[index] < 0
            ]
            active = len(positive) >= cfg.side_count and len(negative) >= cfg.side_count
            if active:
                longs = select_v157_side(positive, previous_positive, cfg)
                shorts = select_v157_side(negative, previous_negative, cfg)
                active_days += 1
            else:
                longs = []
                shorts = []
            betas = local["btc_beta"].astype(float).to_dict()
            weights = beta_neutral_v157_weights(longs, shorts, betas)
            turnover = _turnover(previous_weights, weights)
            gross_return = (
                float(
                    sum(
                        weights[symbol] * float(local.at[symbol, "price_return"])
                        for symbol in longs + shorts
                    )
                    + weights[BTC] * float(local.iloc[0]["btc_return"])
                )
                if weights
                else 0.0
            )
            total_net += gross_return - cfg.one_way_cost * turnover
            total_turnover += turnover
            previous_weights = weights
            previous_positive = set(longs)
            previous_negative = set(shorts)
            previous_day = day
        rows.append(
            {
                "iteration": iteration,
                "mean_primary_net_return": total_net / len(contexts),
                "mean_turnover": total_turnover / len(contexts),
                "active_days": active_days,
            }
        )
    return pd.DataFrame(rows)


def _moving_block_bootstrap(values: np.ndarray, cfg: V157Config) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed + 2)
    offsets = np.arange(cfg.bootstrap_block_days)
    block_count = int(np.ceil(len(values) / cfg.bootstrap_block_days))
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draws[iteration] = float(values[indices.ravel()[: len(values)]].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_v157(
    portfolio: pd.DataFrame,
    reversed_control: pd.DataFrame,
    stale_control: pd.DataFrame,
    source_only_control: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V157Config = V157Config(),
) -> pd.DataFrame:
    bootstrap_low, bootstrap_high = _moving_block_bootstrap(
        portfolio["primary_net_return"].to_numpy(dtype=float), cfg
    )
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    active_counts = portfolio[portfolio["active"]]["period"].value_counts()
    monthly = portfolio.assign(month=portfolio["decision_time"].dt.strftime("%Y-%m")).groupby(
        "month", observed=True
    )["primary_net_return"].sum()
    positive = monthly[monthly.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    observed_mean = float(portfolio["primary_net_return"].mean())
    active = portfolio[portfolio["active"]]
    row = {
        "candidate": CANDIDATE,
        "calendar_days": len(portfolio),
        "active_days": len(active),
        "active_validation_days": int(active_counts.get("validation", 0)),
        "active_holdout_days": int(active_counts.get("holdout", 0)),
        "mean_gross_bp": portfolio["gross_return"].mean() * 10_000,
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": portfolio["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_bp": bootstrap_low * 10_000,
        "bootstrap_95_high_bp": bootstrap_high * 10_000,
        "random_pairing_percentile": 100
        * random_controls["mean_primary_net_return"].le(observed_mean).mean(),
        "positive_month_concentration": concentration,
        "mean_calendar_turnover": portfolio["realized_turnover"].mean(),
        "reversed_control_mean_bp": reversed_control["primary_net_return"].mean()
        * 10_000,
        "stale_control_mean_bp": stale_control["primary_net_return"].mean() * 10_000,
        "source_only_control_mean_bp": source_only_control[
            "primary_net_return"
        ].mean()
        * 10_000,
        "max_abs_residual_btc_beta": active["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (active["gross_notional"] - 1.0).abs().max(),
    }
    row["promote"] = bool(
        row["calendar_days"] == 375
        and row["active_days"] >= 250
        and row["active_validation_days"] >= 65
        and row["active_holdout_days"] >= 65
        and all(
            row[key] > 0
            for key in (
                "mean_primary_net_bp",
                "mean_stress_net_bp",
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["random_pairing_percentile"] >= 99
        and row["positive_month_concentration"] <= 0.35
        and row["mean_calendar_turnover"] <= 0.60
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["stale_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["source_only_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-10
        and row["max_gross_notional_drift"] <= 1e-10
    )
    return pd.DataFrame([row])


def write_v157_pair_shock_fragile_receiver(
    cfg: V157Config = V157Config(),
) -> dict[str, Path]:
    panel = load_v157_panel(cfg.panel_path)
    signals = build_v157_signals(panel, load_v155_hourly_prices())
    portfolio = build_v157_portfolio(signals, panel, cfg)
    reversed_control = build_v157_portfolio(
        signals, panel, cfg, candidate=REVERSED_CONTROL, reverse=True
    )
    stale_control = build_v157_portfolio(
        signals, panel, cfg, candidate=STALE_CONTROL, signal_lag_days=1
    )
    source_only_control = build_v157_portfolio(
        signals,
        panel,
        cfg,
        candidate=SOURCE_ONLY_CONTROL,
        score_column="source_shock",
    )
    random_controls = build_v157_random_controls(signals, panel, cfg)
    summary = summarize_v157(
        portfolio,
        reversed_control,
        stale_control,
        source_only_control,
        random_controls,
        cfg,
    )
    controls = pd.DataFrame(
        [
            {
                "control": name,
                "mean_primary_net_bp": frame["primary_net_return"].mean() * 10_000,
                "active_days": int(frame["active"].sum()),
            }
            for name, frame in (
                (REVERSED_CONTROL, reversed_control),
                (STALE_CONTROL, stale_control),
                (SOURCE_ONLY_CONTROL, source_only_control),
            )
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "signals": root / "daily_pair_signals.parquet",
        "portfolio": root / "daily_portfolio.parquet",
        "reversed": root / "reversed_control.parquet",
        "stale": root / "stale_control.parquet",
        "source_only": root / "source_only_control.parquet",
        "random": root / "random_pairings.csv",
        "summary": root / "summary.csv",
        "controls": root / "control_summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    reversed_control.to_parquet(paths["reversed"], index=False)
    stale_control.to_parquet(paths["stale"], index=False)
    source_only_control.to_parquet(paths["source_only"], index=False)
    random_controls.to_csv(paths["random"], index=False)
    summary.to_csv(paths["summary"], index=False)
    controls.to_csv(paths["controls"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    serialized_config = {
        **asdict(cfg),
        "panel_path": str(cfg.panel_path),
        "report_root": str(cfg.report_root),
        "findings_path": str(cfg.findings_path),
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "promoted": promoted,
                "config": serialized_config,
                "frozen_pairs": FROZEN_PAIRS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_candidate"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v15.7 Pair-Shock to Fragile-Receiver Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Frozen controls",
                "",
                controls.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The graph pairs, source/receiver definition, one-percent fragility,",
                "two-per-side holding band, cost model and gates were frozen before",
                "inspecting propagation returns. PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
