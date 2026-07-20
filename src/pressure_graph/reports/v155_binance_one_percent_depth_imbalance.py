"""Frozen cross-venue daily strategy from Binance one-percent book depth."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    hourly_bybit_prices,
    load_v132_bybit_klines,
)


FEATURE_ROOT = Path("data/external/binance_um_book_depth/daily_features")
REPORT_ROOT = Path("reports/v15_5_binance_one_percent_depth_imbalance")
FINDINGS_PATH = Path(
    "docs/v155_binance_one_percent_depth_imbalance_findings_2026_07_16.md"
)
CANDIDATE = "BD2_PRIOR_DAY_ONE_PERCENT_DEPTH_CONTINUATION"
REVERSED_CONTROL = "BD2_REVERSED_ONE_PERCENT_DEPTH_REVERSAL"
STALE_CONTROL = "BD2_STALE_TWO_DAY_ONE_PERCENT_DEPTH"
FROZEN_SYMBOLS = (
    "SOLUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",
    "WIFUSDT",
    "ETHUSDT",
    "ENAUSDT",
    "HBARUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ONDOUSDT",
    "XRPUSDT",
    "XLMUSDT",
    "FARTCOINUSDT",
    "WLDUSDT",
    "SEIUSDT",
    "TIAUSDT",
)


@dataclass(frozen=True)
class V155Config:
    feature_root: Path = FEATURE_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    lookback_days: int = 30
    minimum_beta_samples: int = 500
    long_count: int = 4
    short_count: int = 4
    retention_count: int = 8
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    bootstrap_block_days: int = 7
    seed: int = 20260716


def load_v155_features(
    root: Path = FEATURE_ROOT,
    symbols: tuple[str, ...] = FROZEN_SYMBOLS,
) -> pd.DataFrame:
    columns = [
        "bybit_symbol",
        "source_day",
        "notional_imbalance_1p0_median",
        "notional_imbalance_5p0_median",
        "snapshot_count",
        "archive_sha256",
    ]
    frames = []
    for symbol in symbols:
        path = root / f"{symbol}.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["symbol"] = symbol
        frames.append(frame.drop(columns="bybit_symbol"))
    features = pd.concat(frames, ignore_index=True)
    features["source_day"] = pd.to_datetime(features["source_day"], utc=True, errors="coerce")
    for column in (
        "notional_imbalance_1p0_median",
        "notional_imbalance_5p0_median",
        "snapshot_count",
    ):
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return (
        features.dropna(subset=["symbol", "source_day"])
        .drop_duplicates(["symbol", "source_day"], keep="last")
        .sort_values(["source_day", "symbol"])
        .reset_index(drop=True)
    )


def load_v155_hourly_prices() -> pd.DataFrame:
    prices = hourly_bybit_prices(load_v132_bybit_klines())
    return prices[prices["symbol"].isin(set(FROZEN_SYMBOLS) | {BTC})].reset_index(
        drop=True
    )


def estimate_v155_betas(
    hourly_returns: pd.DataFrame,
    symbols: tuple[str, ...] = FROZEN_SYMBOLS,
    minimum_samples: int = 500,
) -> dict[str, float]:
    if BTC not in hourly_returns.columns:
        return {}
    btc = pd.to_numeric(hourly_returns[BTC], errors="coerce")
    betas: dict[str, float] = {}
    for symbol in symbols:
        if symbol not in hourly_returns.columns:
            continue
        alt = pd.to_numeric(hourly_returns[symbol], errors="coerce")
        valid = alt.notna() & btc.notna()
        if int(valid.sum()) < minimum_samples:
            continue
        local_alt = alt[valid].to_numpy(dtype=float)
        local_btc = btc[valid].to_numpy(dtype=float)
        variance = float(np.var(local_btc, ddof=0))
        if not np.isfinite(variance) or variance <= 0:
            continue
        covariance = float(
            np.mean((local_alt - local_alt.mean()) * (local_btc - local_btc.mean()))
        )
        betas[symbol] = covariance / variance
    return betas


def build_v155_daily_panel(
    features: pd.DataFrame,
    hourly_prices: pd.DataFrame,
    cfg: V155Config = V155Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        "symbol",
        "source_day",
        "notional_imbalance_1p0_median",
        "notional_imbalance_5p0_median",
        "snapshot_count",
        "archive_sha256",
    ]
    base = features[feature_columns].copy()
    base["decision_time"] = base["source_day"] + pd.Timedelta(days=1)
    base = base.rename(
        columns={
            "notional_imbalance_1p0_median": "feature_1pct",
            "notional_imbalance_5p0_median": "feature_5pct",
        }
    )
    stale = features[
        ["symbol", "source_day", "notional_imbalance_1p0_median"]
    ].copy()
    stale["decision_time"] = stale["source_day"] + pd.Timedelta(days=2)
    stale = stale.rename(
        columns={
            "source_day": "stale_source_day",
            "notional_imbalance_1p0_median": "stale_feature_1pct",
        }
    )
    aligned = base.merge(
        stale,
        on=["symbol", "decision_time"],
        how="left",
        validate="one_to_one",
    )

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
    hourly_return = hourly_close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    daily_close = hourly_close[hourly_close.index.hour == 0]
    required = list(FROZEN_SYMBOLS) + [BTC]
    coverage_rows = []
    panel_rows = []
    for decision_time, local in aligned.groupby("decision_time", sort=True, observed=True):
        decision = pd.Timestamp(decision_time)
        exact_symbols = set(local["symbol"].astype(str)) == set(FROZEN_SYMBOLS)
        finite_primary = bool(
            len(local) == len(FROZEN_SYMBOLS)
            and np.isfinite(local["feature_1pct"].to_numpy(dtype=float)).all()
        )
        finite_stale = bool(
            len(local) == len(FROZEN_SYMBOLS)
            and np.isfinite(local["stale_feature_1pct"].to_numpy(dtype=float)).all()
        )
        price_ready = bool(
            decision in daily_close.index
            and decision + pd.Timedelta(days=1) in daily_close.index
            and daily_close.loc[decision, required].notna().all()
            and daily_close.loc[decision + pd.Timedelta(days=1), required].notna().all()
        )
        history = hourly_return[
            (hourly_return.index > decision - pd.Timedelta(days=cfg.lookback_days))
            & (hourly_return.index <= decision)
        ]
        betas = estimate_v155_betas(
            history,
            minimum_samples=cfg.minimum_beta_samples,
        )
        beta_ready = set(betas) == set(FROZEN_SYMBOLS)
        usable = exact_symbols and finite_primary and finite_stale and price_ready and beta_ready
        coverage_rows.append(
            {
                "decision_time": decision,
                "symbols": len(local),
                "exact_universe": exact_symbols,
                "finite_primary": finite_primary,
                "finite_stale": finite_stale,
                "price_ready": price_ready,
                "beta_ready": beta_ready,
                "minimum_beta_samples_observed": int(
                    history[list(FROZEN_SYMBOLS) + [BTC]].notna().sum().min()
                )
                if set(required).issubset(history.columns)
                else 0,
                "usable": usable,
            }
        )
        if not usable:
            continue
        entry = daily_close.loc[decision, required]
        exit_price = daily_close.loc[decision + pd.Timedelta(days=1), required]
        returns = exit_price / entry - 1.0
        period = (
            "development"
            if decision <= pd.Timestamp("2025-12-31", tz="UTC")
            else "validation"
            if decision <= pd.Timestamp("2026-03-31", tz="UTC")
            else "holdout"
        )
        for row in local.itertuples(index=False):
            panel_rows.append(
                {
                    "decision_time": decision,
                    "source_day": row.source_day,
                    "stale_source_day": row.stale_source_day,
                    "period": period,
                    "symbol": row.symbol,
                    "feature_1pct": float(row.feature_1pct),
                    "stale_feature_1pct": float(row.stale_feature_1pct),
                    "feature_5pct": float(row.feature_5pct),
                    "snapshot_count": int(row.snapshot_count),
                    "archive_sha256": row.archive_sha256,
                    "btc_beta": float(betas[row.symbol]),
                    "price_return": float(returns[row.symbol]),
                    "btc_return": float(returns[BTC]),
                }
            )
    return pd.DataFrame(panel_rows), pd.DataFrame(coverage_rows)


def select_v155_sides(
    order: list[str],
    previous_longs: set[str],
    previous_shorts: set[str],
    cfg: V155Config = V155Config(),
) -> tuple[list[str], list[str]]:
    rank = {symbol: index + 1 for index, symbol in enumerate(order)}
    longs = sorted(
        (symbol for symbol in previous_longs if rank[symbol] <= cfg.retention_count),
        key=rank.get,
    )
    for symbol in order:
        if len(longs) >= cfg.long_count:
            break
        if symbol not in longs and rank[symbol] <= cfg.retention_count:
            longs.append(symbol)
    shorts = sorted(
        (
            symbol
            for symbol in previous_shorts
            if rank[symbol] > len(order) - cfg.retention_count
        ),
        key=rank.get,
        reverse=True,
    )
    for symbol in reversed(order):
        if len(shorts) >= cfg.short_count:
            break
        if symbol not in shorts and rank[symbol] > len(order) - cfg.retention_count:
            shorts.append(symbol)
    return longs, shorts


def beta_neutral_v155_weights(
    longs: list[str],
    shorts: list[str],
    betas: dict[str, float],
) -> dict[str, float]:
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


def build_v155_portfolio(
    panel: pd.DataFrame,
    cfg: V155Config = V155Config(),
    *,
    feature_column: str = "feature_1pct",
    candidate: str = CANDIDATE,
    reverse: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    previous_longs: set[str] = set()
    previous_shorts: set[str] = set()
    previous_time: pd.Timestamp | None = None
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    for raw_time, local in panel.groupby("decision_time", sort=True, observed=True):
        decision = pd.Timestamp(raw_time)
        if previous_time is not None and decision - previous_time > pd.Timedelta(days=1):
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
        if rng is None:
            order = (
                local.sort_values([feature_column, "symbol"], ascending=[False, True])[
                    "symbol"
                ]
                .astype(str)
                .tolist()
            )
        else:
            symbols = np.asarray(sorted(local["symbol"].astype(str)))
            order = [str(symbol) for symbol in symbols[rng.permutation(len(symbols))]]
        if reverse:
            order = list(reversed(order))
        longs, shorts = select_v155_sides(
            order, previous_longs, previous_shorts, cfg
        )
        betas = local.set_index("symbol")["btc_beta"].astype(float).to_dict()
        weights = beta_neutral_v155_weights(longs, shorts, betas)
        indexed = local.set_index("symbol")
        gross_return = float(
            sum(weights[symbol] * float(indexed.at[symbol, "price_return"]) for symbol in longs + shorts)
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
                "source_day": local.iloc[0]["source_day"],
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


def build_v155_random_controls(
    panel: pd.DataFrame,
    cfg: V155Config = V155Config(),
) -> pd.DataFrame:
    symbols = np.asarray(sorted(FROZEN_SYMBOLS))
    contexts = []
    for raw_time, local in panel.groupby("decision_time", sort=True, observed=True):
        indexed = local.set_index("symbol").reindex(symbols)
        contexts.append(
            (
                pd.Timestamp(raw_time),
                indexed["btc_beta"].to_numpy(dtype=float),
                indexed["price_return"].to_numpy(dtype=float),
                float(local.iloc[0]["btc_return"]),
            )
        )
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        previous_alt = np.zeros(len(symbols), dtype=float)
        previous_btc = 0.0
        previous_longs: set[int] = set()
        previous_shorts: set[int] = set()
        previous_time: pd.Timestamp | None = None
        total_net = 0.0
        total_turnover = 0.0
        for decision, betas, returns, btc_return in contexts:
            if previous_time is not None and decision - previous_time > pd.Timedelta(days=1):
                forced_close = float(np.abs(previous_alt).sum() + abs(previous_btc))
                total_net -= cfg.one_way_cost * forced_close
                total_turnover += forced_close
                previous_alt = np.zeros(len(symbols), dtype=float)
                previous_btc = 0.0
                previous_longs = set()
                previous_shorts = set()
            order = rng.permutation(len(symbols))
            rank = np.empty(len(symbols), dtype=int)
            rank[order] = np.arange(len(symbols))
            longs = sorted(
                (index for index in previous_longs if rank[index] < cfg.retention_count),
                key=rank.__getitem__,
            )
            for index in order:
                selected = int(index)
                if len(longs) >= cfg.long_count:
                    break
                if selected not in longs and rank[selected] < cfg.retention_count:
                    longs.append(selected)
            shorts = sorted(
                (
                    index
                    for index in previous_shorts
                    if rank[index] >= len(symbols) - cfg.retention_count
                ),
                key=rank.__getitem__,
                reverse=True,
            )
            for index in order[::-1]:
                selected = int(index)
                if len(shorts) >= cfg.short_count:
                    break
                if selected not in shorts and rank[selected] >= len(symbols) - cfg.retention_count:
                    shorts.append(selected)
            raw_alt = np.zeros(len(symbols), dtype=float)
            raw_alt[longs] = 0.5 / len(longs)
            raw_alt[shorts] = -0.5 / len(shorts)
            raw_btc = -float(np.dot(raw_alt, betas))
            gross = float(np.abs(raw_alt).sum() + abs(raw_btc))
            alt = raw_alt / gross
            btc = raw_btc / gross
            turnover = float(
                np.abs(alt - previous_alt).sum() + abs(btc - previous_btc)
            )
            gross_return = float(np.dot(alt, returns) + btc * btc_return)
            total_net += gross_return - cfg.one_way_cost * turnover
            total_turnover += turnover
            previous_alt = alt
            previous_btc = btc
            previous_longs = set(longs)
            previous_shorts = set(shorts)
            previous_time = decision
        rows.append(
            {
                "iteration": iteration,
                "mean_primary_net_return": total_net / len(contexts),
                "mean_turnover": total_turnover / len(contexts),
            }
        )
    return pd.DataFrame(rows)


def _moving_block_bootstrap(
    values: np.ndarray,
    cfg: V155Config,
) -> tuple[float, float]:
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


def summarize_v155(
    portfolio: pd.DataFrame,
    reversed_control: pd.DataFrame,
    stale_control: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V155Config = V155Config(),
) -> pd.DataFrame:
    bootstrap_low, bootstrap_high = _moving_block_bootstrap(
        portfolio["primary_net_return"].to_numpy(dtype=float), cfg
    )
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    counts = portfolio["period"].value_counts()
    monthly = portfolio.assign(
        month=portfolio["decision_time"].dt.to_period("M").astype(str)
    ).groupby("month", observed=True)["primary_net_return"].sum()
    positive = monthly[monthly.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    observed_mean = float(portfolio["primary_net_return"].mean())
    row = {
        "candidate": CANDIDATE,
        "days": len(portfolio),
        "months": portfolio["decision_time"].dt.to_period("M").nunique(),
        "validation_days": int(counts.get("validation", 0)),
        "holdout_days": int(counts.get("holdout", 0)),
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
        row["days"] >= 300
        and row["months"] >= 10
        and row["validation_days"] >= 80
        and row["holdout_days"] >= 80
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
        and row["random_ranking_percentile"] >= 95
        and row["positive_month_concentration"] <= 0.35
        and row["mean_one_way_turnover"] <= 0.50
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["stale_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-10
        and row["max_gross_notional_drift"] <= 1e-10
    )
    return pd.DataFrame([row])


def write_v155_binance_one_percent_depth_imbalance(
    cfg: V155Config = V155Config(),
) -> dict[str, Path]:
    features = load_v155_features(cfg.feature_root)
    prices = load_v155_hourly_prices()
    panel, coverage = build_v155_daily_panel(features, prices, cfg)
    portfolio = build_v155_portfolio(panel, cfg)
    reversed_control = build_v155_portfolio(
        panel, cfg, candidate=REVERSED_CONTROL, reverse=True
    )
    stale_control = build_v155_portfolio(
        panel, cfg, feature_column="stale_feature_1pct", candidate=STALE_CONTROL
    )
    diagnostic_5pct = build_v155_portfolio(
        panel, cfg, feature_column="feature_5pct", candidate="BD2_5PCT_DIAGNOSTIC_ONLY"
    )
    random_controls = build_v155_random_controls(panel, cfg)
    summary = summarize_v155(
        portfolio, reversed_control, stale_control, random_controls, cfg
    )
    controls = pd.DataFrame(
        [
            {
                "control": REVERSED_CONTROL,
                "days": len(reversed_control),
                "mean_primary_net_bp": reversed_control["primary_net_return"].mean()
                * 10_000,
            },
            {
                "control": STALE_CONTROL,
                "days": len(stale_control),
                "mean_primary_net_bp": stale_control["primary_net_return"].mean()
                * 10_000,
            },
            {
                "control": "BD2_5PCT_DIAGNOSTIC_ONLY_NON_PROMOTABLE",
                "days": len(diagnostic_5pct),
                "mean_primary_net_bp": diagnostic_5pct["primary_net_return"].mean()
                * 10_000,
            },
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "daily_symbol_panel.parquet",
        "coverage": root / "coverage.csv",
        "portfolio": root / "daily_portfolio.parquet",
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
                "# v15.5 Binance One-Percent Depth Imbalance Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Frozen controls",
                "",
                controls.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The 1% continuation direction, 4/4 bucket, top/bottom-eight holding",
                "band, 24-hour horizon, beta hedge, cost model, sample splits and gates",
                "were frozen before inspecting any portfolio return. The 5% row is",
                "diagnostic-only and cannot be promoted from this study. PaperLive and",
                "remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
