"""Hourly cross-venue continuation from causal Binance depth imbalance."""
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
    beta_neutral_v155_weights,
    load_v155_hourly_prices,
    select_v155_sides,
)


FEATURE_ROOT = Path("data/external/binance_um_book_depth/hourly_features")
REPORT_ROOT = Path("reports/v15_9_hourly_cross_venue_depth_imbalance")
FINDINGS_PATH = Path(
    "docs/v159_hourly_cross_venue_depth_imbalance_findings_2026_07_16.md"
)
CANDIDATE = "BD3_HOURLY_CROSS_VENUE_DEPTH_CONTINUATION"
REVERSED_CONTROL = "BD3_REVERSED_HOURLY_DEPTH_REVERSAL"
STALE_CONTROL = "BD3_ONE_HOUR_STALE_DEPTH"


@dataclass(frozen=True)
class V159Config:
    feature_root: Path = FEATURE_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    beta_window_hours: int = 720
    minimum_beta_samples: int = 500
    minimum_snapshots: int = 90
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    bootstrap_block_hours: int = 24
    seed: int = 20260722


def load_v159_features(
    root: Path = FEATURE_ROOT,
) -> pd.DataFrame:
    columns = [
        "decision_time",
        "source_day",
        "symbol",
        "notional_imbalance_1p0_median",
        "notional_imbalance_1p0_valid_snapshots",
        "notional_imbalance_5p0_median",
        "archive_sha256",
    ]
    frames = [
        pd.read_parquet(root / f"{symbol}.parquet", columns=columns)
        for symbol in FROZEN_SYMBOLS
    ]
    features = pd.concat(frames, ignore_index=True)
    for column in ("decision_time", "source_day"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="coerce")
    return (
        features.dropna(subset=["decision_time", "symbol"])
        .drop_duplicates(["decision_time", "symbol"], keep="last")
        .sort_values(["decision_time", "symbol"])
        .reset_index(drop=True)
    )


def rolling_v159_betas(
    hourly_returns: pd.DataFrame,
    cfg: V159Config = V159Config(),
) -> pd.DataFrame:
    btc = pd.to_numeric(hourly_returns[BTC], errors="coerce")
    outputs = {}
    for symbol in FROZEN_SYMBOLS:
        alt = pd.to_numeric(hourly_returns[symbol], errors="coerce")
        valid = alt.notna() & btc.notna()
        local_alt = alt.where(valid)
        local_btc = btc.where(valid)
        rolling = {"window": cfg.beta_window_hours, "min_periods": 1}
        count = valid.rolling(**rolling).sum()
        alt_mean = local_alt.rolling(**rolling).sum() / count
        btc_mean = local_btc.rolling(**rolling).sum() / count
        cross_mean = (local_alt * local_btc).rolling(**rolling).sum() / count
        square_mean = (local_btc * local_btc).rolling(**rolling).sum() / count
        covariance = cross_mean - alt_mean * btc_mean
        variance = square_mean - btc_mean * btc_mean
        outputs[symbol] = (covariance / variance).where(
            count.ge(cfg.minimum_beta_samples) & variance.gt(0)
        )
    return pd.DataFrame(outputs, index=hourly_returns.index)


def build_v159_hourly_panel(
    features: pd.DataFrame,
    hourly_prices: pd.DataFrame,
    cfg: V159Config = V159Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = features.rename(
        columns={
            "notional_imbalance_1p0_median": "feature_1pct",
            "notional_imbalance_1p0_valid_snapshots": "valid_snapshots",
            "notional_imbalance_5p0_median": "feature_5pct",
        }
    ).copy()
    stale = base[["decision_time", "symbol", "feature_1pct", "valid_snapshots"]].copy()
    stale["decision_time"] += pd.Timedelta(hours=1)
    stale = stale.rename(
        columns={
            "feature_1pct": "stale_feature_1pct",
            "valid_snapshots": "stale_valid_snapshots",
        }
    )
    aligned = base.merge(
        stale,
        on=["decision_time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    required_feature = (
        aligned["feature_1pct"].notna()
        & aligned["stale_feature_1pct"].notna()
        & aligned["valid_snapshots"].ge(cfg.minimum_snapshots)
        & aligned["stale_valid_snapshots"].ge(cfg.minimum_snapshots)
    )
    aligned["feature_ready"] = required_feature
    feature_coverage = aligned.groupby("decision_time", observed=True).agg(
        symbols=("symbol", "nunique"),
        feature_ready_symbols=("feature_ready", "sum"),
    )
    feature_ready_hours = feature_coverage.index[
        feature_coverage["symbols"].eq(len(FROZEN_SYMBOLS))
        & feature_coverage["feature_ready_symbols"].eq(len(FROZEN_SYMBOLS))
    ]

    prices = hourly_prices.copy()
    prices["feature_time"] = pd.to_datetime(prices["feature_time"], utc=True, errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    close = prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    future = close.shift(-1) / close - 1.0
    betas = rolling_v159_betas(returns, cfg)
    required_symbols = list(FROZEN_SYMBOLS) + [BTC]
    price_ready = close[required_symbols].notna().all(axis=1) & close[
        required_symbols
    ].shift(-1).notna().all(axis=1)
    beta_ready = betas[list(FROZEN_SYMBOLS)].notna().all(axis=1)
    usable_times = feature_ready_hours.intersection(close.index[price_ready & beta_ready])

    local = aligned[aligned["decision_time"].isin(usable_times)].copy()
    future_long = (
        future[list(FROZEN_SYMBOLS)]
        .stack(future_stack=True)
        .rename("price_return")
        .rename_axis(index=["decision_time", "symbol"])
        .reset_index()
    )
    beta_long = (
        betas[list(FROZEN_SYMBOLS)]
        .stack(future_stack=True)
        .rename("btc_beta")
        .rename_axis(index=["decision_time", "symbol"])
        .reset_index()
    )
    local = local.merge(
        future_long,
        on=["decision_time", "symbol"],
        how="inner",
        validate="one_to_one",
    ).merge(
        beta_long,
        on=["decision_time", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    local["btc_return"] = local["decision_time"].map(future[BTC])
    local["period"] = np.select(
        [
            local["decision_time"].le(pd.Timestamp("2025-12-31 23:00", tz="UTC")),
            local["decision_time"].le(pd.Timestamp("2026-03-31 23:00", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    coverage = feature_coverage.reset_index()
    coverage["price_ready"] = coverage["decision_time"].map(price_ready).fillna(False)
    coverage["beta_ready"] = coverage["decision_time"].map(beta_ready).fillna(False)
    coverage["usable"] = coverage["decision_time"].isin(usable_times)
    return (
        local.sort_values(["decision_time", "symbol"]).reset_index(drop=True),
        coverage.sort_values("decision_time").reset_index(drop=True),
    )


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return float(
        sum(
            abs(previous.get(symbol, 0.0) - current.get(symbol, 0.0))
            for symbol in set(previous) | set(current)
        )
    )


def build_v159_portfolio(
    panel: pd.DataFrame,
    cfg: V159Config = V159Config(),
    *,
    feature_column: str = "feature_1pct",
    candidate: str = CANDIDATE,
    reverse: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    previous_longs: set[str] = set()
    previous_shorts: set[str] = set()
    previous_time: pd.Timestamp | None = None
    for raw_time, local in panel.groupby("decision_time", sort=True, observed=True):
        decision = pd.Timestamp(raw_time)
        if previous_time is not None and decision - previous_time > pd.Timedelta(hours=1):
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
            previous_longs = set()
            previous_shorts = set()
        local = local.sort_values("symbol").reset_index(drop=True)
        order = (
            local.sort_values([feature_column, "symbol"], ascending=[False, True])["symbol"]
            .astype(str)
            .tolist()
        )
        if reverse:
            order = list(reversed(order))
        longs, shorts = select_v155_sides(order, previous_longs, previous_shorts)
        betas = local.set_index("symbol")["btc_beta"].astype(float).to_dict()
        weights = beta_neutral_v155_weights(longs, shorts, betas)
        indexed = local.set_index("symbol")
        gross_return = float(
            sum(
                weights[symbol] * float(indexed.at[symbol, "price_return"])
                for symbol in longs + shorts
            )
            + weights[BTC] * float(local.iloc[0]["btc_return"])
        )
        residual_beta = float(
            sum(weights[symbol] * betas[symbol] for symbol in longs + shorts)
            + weights[BTC]
        )
        gross_notional = float(sum(abs(weight) for weight in weights.values()))
        entry_turnover = _turnover(previous_weights, weights)
        rows.append(
            {
                "candidate": candidate,
                "decision_time": decision,
                "period": local.iloc[0]["period"],
                "long_symbols": "|".join(longs),
                "short_symbols": "|".join(shorts),
                "weights_json": json.dumps(weights, sort_keys=True),
                "btc_hedge_weight": weights[BTC],
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
        previous_longs = set(longs)
        previous_shorts = set(shorts)
        previous_time = decision
    return pd.DataFrame(rows)


def build_v159_random_controls(
    panel: pd.DataFrame,
    cfg: V159Config = V159Config(),
) -> pd.DataFrame:
    symbols = np.asarray(sorted(FROZEN_SYMBOLS))
    beta = panel.pivot(index="decision_time", columns="symbol", values="btc_beta").reindex(
        columns=symbols
    )
    returns = panel.pivot(
        index="decision_time", columns="symbol", values="price_return"
    ).reindex(index=beta.index, columns=symbols)
    btc_return = panel.groupby("decision_time", observed=True)["btc_return"].first().reindex(
        beta.index
    )
    times = beta.index
    beta_values = beta.to_numpy(dtype=float)
    return_values = returns.to_numpy(dtype=float)
    btc_values = btc_return.to_numpy(dtype=float)
    gaps = np.r_[False, np.diff(times.asi8) > pd.Timedelta(hours=1).value]
    path_count = cfg.random_iterations
    symbol_count = len(symbols)
    rng = np.random.default_rng(cfg.seed + 1)
    previous_alt = np.zeros((path_count, symbol_count), dtype=float)
    previous_btc = np.zeros(path_count, dtype=float)
    previous_longs = np.zeros((path_count, symbol_count), dtype=bool)
    previous_shorts = np.zeros((path_count, symbol_count), dtype=bool)
    total_net = np.zeros(path_count, dtype=float)
    total_turnover = np.zeros(path_count, dtype=float)
    path_indices = np.arange(path_count)
    rank_values = np.arange(symbol_count)

    def select_side(
        rank: np.ndarray,
        previous: np.ndarray,
        eligible: np.ndarray,
        priority: np.ndarray,
    ) -> np.ndarray:
        retained = previous & eligible
        needed = 4 - retained.sum(axis=1)
        candidate_priority = np.where(eligible & ~retained, priority, symbol_count + 1)
        fill_order = np.argsort(candidate_priority, axis=1)[:, :4]
        selected = retained.copy()
        for slot in range(4):
            mask = needed > slot
            selected[path_indices[mask], fill_order[mask, slot]] = True
        return selected

    for row_index in range(len(times)):
        if gaps[row_index]:
            forced_close = np.abs(previous_alt).sum(axis=1) + np.abs(previous_btc)
            total_net -= cfg.one_way_cost * forced_close
            total_turnover += forced_close
            previous_alt.fill(0.0)
            previous_btc.fill(0.0)
            previous_longs.fill(False)
            previous_shorts.fill(False)
        random_keys = rng.random((path_count, symbol_count))
        order = np.argsort(random_keys, axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, rank_values[None, :], axis=1)
        top_half = rank < 8
        bottom_half = ~top_half
        longs = select_side(rank, previous_longs, top_half, rank)
        shorts = select_side(rank, previous_shorts, bottom_half, symbol_count - 1 - rank)
        raw_alt = 0.125 * longs.astype(float) - 0.125 * shorts.astype(float)
        raw_btc = -(raw_alt @ beta_values[row_index])
        gross = np.abs(raw_alt).sum(axis=1) + np.abs(raw_btc)
        alt = raw_alt / gross[:, None]
        btc = raw_btc / gross
        turnover = np.abs(alt - previous_alt).sum(axis=1) + np.abs(btc - previous_btc)
        gross_return = alt @ return_values[row_index] + btc * btc_values[row_index]
        total_net += gross_return - cfg.one_way_cost * turnover
        total_turnover += turnover
        previous_alt = alt
        previous_btc = btc
        previous_longs = longs
        previous_shorts = shorts
    return pd.DataFrame(
        {
            "iteration": np.arange(path_count),
            "mean_primary_net_return": total_net / len(times),
            "mean_turnover": total_turnover / len(times),
        }
    )


def _moving_block_bootstrap(values: np.ndarray, cfg: V159Config) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed + 2)
    offsets = np.arange(cfg.bootstrap_block_hours)
    block_count = int(np.ceil(len(values) / cfg.bootstrap_block_hours))
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draws[iteration] = float(values[indices.ravel()[: len(values)]].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_v159(
    portfolio: pd.DataFrame,
    reversed_control: pd.DataFrame,
    stale_control: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V159Config = V159Config(),
) -> pd.DataFrame:
    bootstrap_low, bootstrap_high = _moving_block_bootstrap(
        portfolio["primary_net_return"].to_numpy(dtype=float), cfg
    )
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    counts = portfolio["period"].value_counts()
    monthly = portfolio.assign(month=portfolio["decision_time"].dt.strftime("%Y-%m")).groupby(
        "month", observed=True
    )["primary_net_return"].sum()
    positive = monthly[monthly.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    observed_mean = float(portfolio["primary_net_return"].mean())
    row = {
        "candidate": CANDIDATE,
        "hours": len(portfolio),
        "months": portfolio["decision_time"].dt.strftime("%Y-%m").nunique(),
        "validation_hours": int(counts.get("validation", 0)),
        "holdout_hours": int(counts.get("holdout", 0)),
        "mean_gross_bp": portfolio["gross_return"].mean() * 10_000,
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": portfolio["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_bp": bootstrap_low * 10_000,
        "bootstrap_95_high_bp": bootstrap_high * 10_000,
        "random_ranking_percentile": 100
        * random_controls["mean_primary_net_return"].le(observed_mean).mean(),
        "positive_month_concentration": concentration,
        "mean_one_way_turnover": portfolio["realized_turnover"].mean(),
        "reversed_control_mean_bp": reversed_control["primary_net_return"].mean()
        * 10_000,
        "stale_control_mean_bp": stale_control["primary_net_return"].mean() * 10_000,
        "max_abs_residual_btc_beta": portfolio["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (portfolio["gross_notional"] - 1.0).abs().max(),
    }
    row["promote"] = bool(
        row["hours"] >= 7500
        and row["months"] >= 12
        and row["validation_hours"] >= 1800
        and row["holdout_hours"] >= 2200
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
        and row["random_ranking_percentile"] >= 99
        and row["positive_month_concentration"] <= 0.25
        and row["mean_one_way_turnover"] <= 0.25
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["stale_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-10
        and row["max_gross_notional_drift"] <= 1e-10
    )
    return pd.DataFrame([row])


def write_v159_hourly_cross_venue_depth_imbalance(
    cfg: V159Config = V159Config(),
) -> dict[str, Path]:
    features = load_v159_features(cfg.feature_root)
    panel, coverage = build_v159_hourly_panel(features, load_v155_hourly_prices(), cfg)
    portfolio = build_v159_portfolio(panel, cfg)
    reversed_control = build_v159_portfolio(
        panel, cfg, candidate=REVERSED_CONTROL, reverse=True
    )
    stale_control = build_v159_portfolio(
        panel, cfg, feature_column="stale_feature_1pct", candidate=STALE_CONTROL
    )
    diagnostic_5pct = build_v159_portfolio(
        panel, cfg, feature_column="feature_5pct", candidate="BD3_5PCT_DIAGNOSTIC_ONLY"
    )
    random_controls = build_v159_random_controls(panel, cfg)
    summary = summarize_v159(
        portfolio, reversed_control, stale_control, random_controls, cfg
    )
    controls = pd.DataFrame(
        [
            {
                "control": name,
                "hours": len(frame),
                "mean_primary_net_bp": frame["primary_net_return"].mean() * 10_000,
            }
            for name, frame in (
                (REVERSED_CONTROL, reversed_control),
                (STALE_CONTROL, stale_control),
                ("BD3_5PCT_DIAGNOSTIC_ONLY_NON_PROMOTABLE", diagnostic_5pct),
            )
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "hourly_symbol_panel.parquet",
        "coverage": root / "coverage.csv",
        "portfolio": root / "hourly_portfolio.parquet",
        "reversed": root / "reversed_control.parquet",
        "stale": root / "stale_control.parquet",
        "diagnostic_5pct": root / "five_percent_diagnostic.parquet",
        "random": root / "random_rankings.csv",
        "summary": root / "summary.csv",
        "controls": root / "control_summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    panel.to_parquet(paths["panel"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    reversed_control.to_parquet(paths["reversed"], index=False)
    stale_control.to_parquet(paths["stale"], index=False)
    diagnostic_5pct.to_parquet(paths["diagnostic_5pct"], index=False)
    random_controls.to_csv(paths["random"], index=False)
    summary.to_csv(paths["summary"], index=False)
    controls.to_csv(paths["controls"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    serialized_config = {
        **asdict(cfg),
        "feature_root": str(cfg.feature_root),
        "report_root": str(cfg.report_root),
        "findings_path": str(cfg.findings_path),
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "promoted": promoted,
                "config": serialized_config,
                "frozen_symbols": list(FROZEN_SYMBOLS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_candidate"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v15.9 Hourly Cross-Venue Depth Imbalance Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Frozen controls",
                "",
                controls.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Every feature uses only snapshots strictly before the Bybit entry",
                "hour. The 5% row is diagnostic-only. PaperLive and remote state are",
                "unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
