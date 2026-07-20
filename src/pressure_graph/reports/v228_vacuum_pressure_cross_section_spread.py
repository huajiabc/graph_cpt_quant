"""Beta-neutral cross-sectional spread inside broad book-vacuum events."""

from __future__ import annotations

import hashlib
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


FEATURE_PATH = Path(
    "reports/v22_7_vacuum_pressure_cross_section_feature_audit/"
    "ranked_symbol_features.parquet"
)
REPORT_ROOT = Path("reports/v22_8_vacuum_pressure_cross_section_spread")
FINDINGS_PATH = Path(
    "docs/v228_vacuum_pressure_cross_section_spread_findings_2026_07_17.md"
)
CANDIDATE = "DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4"
FEATURE_SHA256 = "5B7D351886C63DA3178B81C101BF24A47CA90085651A6654B414226674DD546E"


@dataclass(frozen=True)
class V228Config:
    feature_path: Path = FEATURE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    beta_lookback_days: int = 30
    minimum_beta_samples: int = 500
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    raw_cost: float = 0.0020
    random_iterations: int = 1000
    bootstrap_iterations: int = 2000
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v228_inputs(
    cfg: V228Config = V228Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(cfg.feature_path)
    prices = load_v155_hourly_prices()
    for frame, columns in (
        (features, ("feature_time", "entry_time")),
        (prices, ("feature_time",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return features, prices


def _price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()


def estimate_v228_monthly_betas(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V228Config = V228Config(),
) -> pd.DataFrame:
    matrix = _price_matrix(prices)
    returns = matrix.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    rows = []
    for month_text in sorted(features["entry_month"].unique()):
        month_start = pd.Timestamp(f"{month_text}-01", tz="UTC")
        history = returns[
            (returns.index >= month_start - pd.Timedelta(days=cfg.beta_lookback_days))
            & (returns.index < month_start)
        ]
        for symbol in FROZEN_SYMBOLS:
            paired = history[[symbol, BTC]].dropna()
            if len(paired) < cfg.minimum_beta_samples:
                continue
            btc = paired[BTC].to_numpy(dtype=float)
            alt = paired[symbol].to_numpy(dtype=float)
            variance = float(np.var(btc, ddof=0))
            if variance <= 0 or not np.isfinite(variance):
                continue
            covariance = float(
                np.mean((alt - alt.mean()) * (btc - btc.mean()))
            )
            rows.append(
                {
                    "entry_month": month_text,
                    "month_start": month_start,
                    "history_start": month_start
                    - pd.Timedelta(days=cfg.beta_lookback_days),
                    "history_end_exclusive": month_start,
                    "symbol": symbol,
                    "beta_samples": len(paired),
                    "btc_beta": covariance / variance,
                }
            )
    return pd.DataFrame(rows).sort_values(["month_start", "symbol"]).reset_index(
        drop=True
    )


def _neutralize(
    raw: dict[str, float],
    beta_map: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    alt_beta = float(sum(raw[symbol] * beta_map[symbol] for symbol in raw))
    unscaled = dict(raw)
    unscaled[BTC] = -alt_beta
    gross = float(sum(abs(weight) for weight in unscaled.values()))
    weights = {symbol: weight / gross for symbol, weight in unscaled.items()}
    residual = float(
        sum(weights[symbol] * beta_map[symbol] for symbol in raw) + weights[BTC]
    )
    return weights, residual, float(sum(abs(value) for value in weights.values()))


def build_v228_outcomes(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    monthly_betas: pd.DataFrame,
    cfg: V228Config = V228Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = _price_matrix(prices)
    beta_maps = {
        month: dict(zip(local["symbol"], local["btc_beta"], strict=True))
        for month, local in monthly_betas.groupby("entry_month", observed=True)
    }
    rows = []
    weight_rows = []
    for entry, local in features.groupby("entry_time", sort=True, observed=True):
        entry = pd.Timestamp(entry)
        month = str(local["entry_month"].iloc[0])
        beta_map = beta_maps.get(month, {})
        selected = local["symbol"].astype(str).tolist()
        if any(symbol not in beta_map for symbol in selected):
            continue
        required_times = [
            entry,
            entry + pd.Timedelta(hours=1),
            entry + pd.Timedelta(hours=4),
            entry + pd.Timedelta(hours=5),
            entry + pd.Timedelta(hours=8),
        ]
        if any(time not in matrix.index for time in required_times):
            continue
        required_symbols = [BTC, *selected]
        local_prices = matrix.loc[required_times, required_symbols]
        if not bool(np.isfinite(local_prices.to_numpy(dtype=float)).all()):
            continue
        raw = dict(zip(local["symbol"], local["raw_weight"], strict=True))
        weights, residual_beta, gross_notional = _neutralize(raw, beta_map)
        horizon_returns = {}
        for horizon in (1, 4, 8):
            future = entry + pd.Timedelta(hours=horizon)
            horizon_returns[horizon] = {
                symbol: float(local_prices.at[future, symbol] / local_prices.at[entry, symbol] - 1)
                for symbol in weights
            }
        gross_returns = {
            horizon: float(
                sum(weights[symbol] * returns[symbol] for symbol in weights)
            )
            for horizon, returns in horizon_returns.items()
        }
        delayed_returns = {
            symbol: float(
                local_prices.at[entry + pd.Timedelta(hours=5), symbol]
                / local_prices.at[entry + pd.Timedelta(hours=1), symbol]
                - 1
            )
            for symbol in weights
        }
        delayed_gross = float(
            sum(weights[symbol] * delayed_returns[symbol] for symbol in weights)
        )
        raw_gross = float(
            sum(raw[symbol] * horizon_returns[4][symbol] for symbol in raw)
        )
        long_component = float(
            sum(
                weights[symbol] * horizon_returns[4][symbol]
                for symbol in raw
                if raw[symbol] > 0
            )
        )
        short_component = float(
            sum(
                weights[symbol] * horizon_returns[4][symbol]
                for symbol in raw
                if raw[symbol] < 0
            )
        )
        btc_component = float(weights[BTC] * horizon_returns[4][BTC])
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "entry_day": entry.floor("D"),
                "entry_month": month,
                "period": local["period"].iloc[0],
                "long_symbols": "|".join(
                    sorted(symbol for symbol in raw if raw[symbol] > 0)
                ),
                "short_symbols": "|".join(
                    sorted(symbol for symbol in raw if raw[symbol] < 0)
                ),
                "btc_hedge_weight": weights[BTC],
                "residual_btc_beta": residual_beta,
                "gross_notional": gross_notional,
                "long_alt_return_4h": long_component,
                "short_alt_return_4h": short_component,
                "btc_hedge_return_4h": btc_component,
                "gross_return_1h": gross_returns[1],
                "gross_return_4h": gross_returns[4],
                "gross_return_8h": gross_returns[8],
                "primary_net_return_4h": gross_returns[4] - cfg.primary_cost,
                "stress_net_return_4h": gross_returns[4] - cfg.stress_cost,
                "reversed_primary_net_return_4h": -gross_returns[4]
                - cfg.primary_cost,
                "delayed_gross_return_4h": delayed_gross,
                "delayed_primary_net_return_4h": delayed_gross - cfg.primary_cost,
                "raw_dollar_neutral_gross_return_4h": raw_gross,
                "raw_dollar_neutral_net_return_4h": raw_gross - cfg.raw_cost,
            }
        )
        weight_rows.extend(
            {
                "entry_time": entry,
                "symbol": symbol,
                "weight": weight,
                "is_btc_hedge": symbol == BTC,
                "btc_beta": 1.0 if symbol == BTC else beta_map[symbol],
            }
            for symbol, weight in weights.items()
        )
    return (
        pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True),
        pd.DataFrame(weight_rows).sort_values(["entry_time", "symbol"]).reset_index(
            drop=True
        ),
    )


def build_v228_random_controls(
    outcomes: pd.DataFrame,
    prices: pd.DataFrame,
    monthly_betas: pd.DataFrame,
    cfg: V228Config = V228Config(),
) -> pd.DataFrame:
    matrix = _price_matrix(prices)
    beta_maps = {
        month: dict(zip(local["symbol"], local["btc_beta"], strict=True))
        for month, local in monthly_betas.groupby("entry_month", observed=True)
    }
    contexts = []
    symbols = np.asarray(sorted(FROZEN_SYMBOLS))
    for event in outcomes.itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        month = str(event.entry_month)
        returns = {
            symbol: float(
                matrix.at[entry + pd.Timedelta(hours=4), symbol]
                / matrix.at[entry, symbol]
                - 1
            )
            for symbol in [BTC, *symbols]
        }
        contexts.append((beta_maps[month], returns))
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        values = []
        for beta_map, returns in contexts:
            shuffled = rng.permutation(symbols)
            raw = {symbol: 0.125 for symbol in shuffled[:4]}
            raw.update({symbol: -0.125 for symbol in shuffled[-4:]})
            weights, _, _ = _neutralize(raw, beta_map)
            gross = float(
                sum(weights[symbol] * returns[symbol] for symbol in weights)
            )
            values.append(gross - cfg.primary_cost)
        rows.append(
            {
                "iteration": iteration,
                "control": "within_event_random_top4_bottom4_ranks",
                "events": len(values),
                "mean_primary_net_return_4h": float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def _day_cluster_bootstrap(
    outcomes: pd.DataFrame,
    cfg: V228Config,
) -> tuple[float, float]:
    groups = [
        local["primary_net_return_4h"].to_numpy(dtype=float)
        for _, local in outcomes.groupby("entry_day", sort=False, observed=True)
    ]
    rng = np.random.default_rng(cfg.seed + 2)
    draws = []
    for _ in range(cfg.bootstrap_iterations):
        indices = rng.integers(0, len(groups), size=len(groups))
        draws.append(float(np.concatenate([groups[index] for index in indices]).mean()))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_v228(
    outcomes: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V228Config = V228Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    low, high = _day_cluster_bootstrap(outcomes, cfg)
    periods = outcomes.groupby("period", observed=True).agg(
        events=("entry_time", "size"),
        mean_gross_bp=("gross_return_4h", lambda x: float(x.mean() * 10_000)),
        mean_primary_net_bp=("primary_net_return_4h", lambda x: float(x.mean() * 10_000)),
        mean_stress_net_bp=("stress_net_return_4h", lambda x: float(x.mean() * 10_000)),
    ).reset_index()
    horizons = pd.DataFrame(
        [
            {
                "horizon_hours": horizon,
                "mean_gross_bp": float(outcomes[f"gross_return_{horizon}h"].mean() * 10_000),
            }
            for horizon in (1, 4, 8)
        ]
    )
    period_means = outcomes.groupby("period", observed=True)[
        "primary_net_return_4h"
    ].mean()
    month_pnl = outcomes.groupby("entry_month", observed=True)[
        "primary_net_return_4h"
    ].sum()
    day_pnl = outcomes.groupby("entry_day", observed=True)[
        "primary_net_return_4h"
    ].sum()
    positive_months = month_pnl[month_pnl.gt(0)]
    positive_days = day_pnl[day_pnl.gt(0)]
    observed = float(outcomes["primary_net_return_4h"].mean())
    counts = outcomes["period"].value_counts()
    row: dict[str, object] = {
        "candidate": CANDIDATE,
        "events": len(outcomes),
        "active_days": outcomes["entry_day"].nunique(),
        "active_months": outcomes["entry_month"].nunique(),
        "development_events": int(counts.get("development", 0)),
        "validation_events": int(counts.get("validation", 0)),
        "holdout_events": int(counts.get("holdout", 0)),
        "mean_gross_1h_bp": float(outcomes["gross_return_1h"].mean() * 10_000),
        "mean_gross_4h_bp": float(outcomes["gross_return_4h"].mean() * 10_000),
        "mean_gross_8h_bp": float(outcomes["gross_return_8h"].mean() * 10_000),
        "mean_primary_net_4h_bp": observed * 10_000,
        "mean_stress_net_4h_bp": float(outcomes["stress_net_return_4h"].mean() * 10_000),
        "development_primary_net_4h_bp": float(period_means.get("development", np.nan) * 10_000),
        "validation_primary_net_4h_bp": float(period_means.get("validation", np.nan) * 10_000),
        "holdout_primary_net_4h_bp": float(period_means.get("holdout", np.nan) * 10_000),
        "raw_dollar_neutral_gross_4h_bp": float(outcomes["raw_dollar_neutral_gross_return_4h"].mean() * 10_000),
        "raw_dollar_neutral_net_4h_bp": float(outcomes["raw_dollar_neutral_net_return_4h"].mean() * 10_000),
        "long_alt_component_4h_bp": float(outcomes["long_alt_return_4h"].mean() * 10_000),
        "short_alt_component_4h_bp": float(outcomes["short_alt_return_4h"].mean() * 10_000),
        "btc_hedge_component_4h_bp": float(outcomes["btc_hedge_return_4h"].mean() * 10_000),
        "bootstrap_95_low_primary_bp": low * 10_000,
        "bootstrap_95_high_primary_bp": high * 10_000,
        "random_rank_percentile": float(
            100
            * random_controls["mean_primary_net_return_4h"].le(observed).mean()
        ),
        "reversed_primary_net_4h_bp": float(outcomes["reversed_primary_net_return_4h"].mean() * 10_000),
        "delayed_primary_net_4h_bp": float(outcomes["delayed_primary_net_return_4h"].mean() * 10_000),
        "positive_month_concentration": float(
            positive_months.max() / positive_months.sum()
        )
        if positive_months.sum() > 0
        else np.inf,
        "positive_day_concentration": float(positive_days.max() / positive_days.sum())
        if positive_days.sum() > 0
        else np.inf,
        "max_abs_residual_btc_beta": float(outcomes["residual_btc_beta"].abs().max()),
        "max_gross_notional_drift": float((outcomes["gross_notional"] - 1).abs().max()),
    }
    row["promote"] = bool(
        row["events"] >= 150
        and row["active_months"] >= 11
        and row["development_events"] >= 45
        and row["validation_events"] >= 45
        and row["holdout_events"] >= 45
        and all(
            float(row[key]) > 0
            for key in (
                "mean_gross_4h_bp",
                "mean_primary_net_4h_bp",
                "mean_stress_net_4h_bp",
                "development_primary_net_4h_bp",
                "validation_primary_net_4h_bp",
                "holdout_primary_net_4h_bp",
                "raw_dollar_neutral_net_4h_bp",
                "bootstrap_95_low_primary_bp",
            )
        )
        and row["random_rank_percentile"] >= 95
        and row["mean_primary_net_4h_bp"] > row["reversed_primary_net_4h_bp"]
        and row["mean_primary_net_4h_bp"] > row["delayed_primary_net_4h_bp"]
        and row["positive_month_concentration"] <= 0.35
        and row["positive_day_concentration"] <= 0.20
        and row["max_abs_residual_btc_beta"] <= 1e-12
        and row["max_gross_notional_drift"] <= 1e-12
    )
    return pd.DataFrame([row]), periods, horizons


def build_v228_cost_frontier(outcomes: pd.DataFrame) -> pd.DataFrame:
    gross = float(outcomes["gross_return_4h"].mean())
    return pd.DataFrame(
        [
            {
                "round_trip_cost_bp": cost,
                "mean_net_return_bp": gross * 10_000 - cost,
            }
            for cost in (0, 10, 20, 30, 40, 50)
        ]
    )


def write_v228_vacuum_pressure_cross_section_spread(
    cfg: V228Config = V228Config(),
) -> dict[str, Path]:
    features, prices = load_v228_inputs(cfg)
    monthly_betas = estimate_v228_monthly_betas(features, prices, cfg)
    outcomes, weights = build_v228_outcomes(features, prices, monthly_betas, cfg)
    random_controls = build_v228_random_controls(outcomes, prices, monthly_betas, cfg)
    summary, periods, horizons = summarize_v228(outcomes, random_controls, cfg)
    cost_frontier = build_v228_cost_frontier(outcomes)
    root = ensure_dir(cfg.report_root)
    paths = {
        "events": root / "candidate_events.parquet",
        "weights": root / "event_weights.parquet",
        "betas": root / "monthly_betas.parquet",
        "random": root / "random_rank_controls.csv",
        "summary": root / "candidate_outcome.csv",
        "periods": root / "period_summary.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "cost_frontier": root / "cost_frontier.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    outcomes.to_parquet(paths["events"], index=False)
    weights.to_parquet(paths["weights"], index=False)
    monthly_betas.to_parquet(paths["betas"], index=False)
    random_controls.to_csv(paths["random"], index=False)
    summary.to_csv(paths["summary"], index=False)
    periods.to_csv(paths["periods"], index=False)
    horizons.to_csv(paths["horizons"], index=False)
    cost_frontier.to_csv(paths["cost_frontier"], index=False)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "feature_sha256": _sha256(cfg.feature_path),
                "preregistered_feature_sha256": FEATURE_SHA256,
                "promoted": summary.loc[summary["promote"], "candidate"].tolist(),
                "config": serialized,
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = (
        "retain_cross_section_research_candidate"
        if bool(summary.iloc[0]["promote"])
        else "reject_vacuum_pressure_cross_section_spread"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v22.8 Vacuum-Pressure Cross-Section Spread Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Chronological periods",
                "",
                periods.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Holding horizons",
                "",
                horizons.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Only the preregistered Top4-minus-Bottom4 rank spread was evaluated.",
                "The one/eight-hour and raw-dollar-neutral views cannot rescue the primary.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
