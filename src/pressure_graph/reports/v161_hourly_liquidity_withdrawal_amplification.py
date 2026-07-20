"""Hourly residual-price continuation scaled by displayed-liquidity withdrawal."""
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
from pressure_graph.reports.v159_hourly_cross_venue_depth_imbalance import (
    V159Config,
    build_v159_hourly_panel,
    build_v159_portfolio,
)


FEATURE_ROOT = Path("data/external/binance_um_book_depth/hourly_features")
REPORT_ROOT = Path("reports/v16_1_hourly_liquidity_withdrawal_amplification")
FINDINGS_PATH = Path(
    "docs/v161_hourly_liquidity_withdrawal_amplification_findings_2026_07_16.md"
)
CANDIDATE = "LW1_HOURLY_LIQUIDITY_WITHDRAWAL_AMPLIFICATION"
PRICE_ONLY_CONTROL = "LW1_PRICE_ONLY_RESIDUAL_MOMENTUM"
REVERSED_CONTROL = "LW1_REVERSED_WITHDRAWAL_AMPLIFICATION"
STALE_CONTROL = "LW1_ONE_HOUR_STALE_WITHDRAWAL_SIGNAL"


@dataclass(frozen=True)
class V161Config:
    feature_root: Path = FEATURE_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_snapshots: int = 90
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    bootstrap_block_hours: int = 24
    seed: int = 20260724


def load_v161_features(root: Path = FEATURE_ROOT) -> pd.DataFrame:
    columns = [
        "decision_time",
        "source_day",
        "symbol",
        "notional_imbalance_1p0_median",
        "notional_imbalance_1p0_valid_snapshots",
        "notional_imbalance_5p0_median",
        "total_notional_1p0_median",
        "total_notional_5p0_median",
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


def add_v161_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for band in ("1pct", "5pct"):
        current = pd.to_numeric(out[f"total_depth_{band}"], errors="coerce")
        previous = pd.to_numeric(out[f"previous_total_depth_{band}"], errors="coerce")
        out[f"depth_change_{band}"] = np.log(current / previous).where(
            current.gt(0) & previous.gt(0)
        )
        out[f"withdrawal_{band}"] = -out[f"depth_change_{band}"]
        out[f"withdrawal_percentile_{band}"] = out.groupby(
            "decision_time", observed=True
        )[f"withdrawal_{band}"].rank(method="average", pct=True)
        out[f"score_{band}"] = (
            out["prior_residual_return"] * out[f"withdrawal_percentile_{band}"]
        )
    return out


def build_v161_panel(
    features: pd.DataFrame,
    hourly_prices: pd.DataFrame,
    cfg: V161Config = V161Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v159_cfg = V159Config(
        feature_root=cfg.feature_root,
        minimum_snapshots=cfg.minimum_snapshots,
        one_way_cost=cfg.one_way_cost,
        stress_one_way_cost=cfg.stress_one_way_cost,
    )
    base, coverage = build_v159_hourly_panel(features, hourly_prices, v159_cfg)
    depth = features[
        [
            "decision_time",
            "symbol",
            "total_notional_1p0_median",
            "total_notional_5p0_median",
        ]
    ].rename(
        columns={
            "total_notional_1p0_median": "total_depth_1pct",
            "total_notional_5p0_median": "total_depth_5pct",
        }
    )
    previous_depth = depth.copy()
    previous_depth["decision_time"] += pd.Timedelta(hours=1)
    previous_depth = previous_depth.rename(
        columns={
            "total_depth_1pct": "previous_total_depth_1pct",
            "total_depth_5pct": "previous_total_depth_5pct",
        }
    )
    panel = base.merge(
        depth,
        on=["decision_time", "symbol"],
        how="left",
        validate="one_to_one",
    ).merge(
        previous_depth,
        on=["decision_time", "symbol"],
        how="left",
        validate="one_to_one",
    )

    prices = hourly_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    prior_return = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    prior_long = (
        prior_return[list(FROZEN_SYMBOLS)]
        .stack(future_stack=True)
        .rename("prior_price_return")
        .rename_axis(index=["decision_time", "symbol"])
        .reset_index()
    )
    panel = panel.merge(
        prior_long,
        on=["decision_time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    panel["prior_btc_return"] = panel["decision_time"].map(prior_return[BTC])
    panel["prior_residual_return"] = (
        panel["prior_price_return"] - panel["btc_beta"] * panel["prior_btc_return"]
    )
    panel = add_v161_scores(panel)
    stale = panel[["decision_time", "symbol", "score_1pct"]].copy()
    stale["decision_time"] += pd.Timedelta(hours=1)
    stale = stale.rename(columns={"score_1pct": "stale_score_1pct"})
    panel = panel.merge(
        stale,
        on=["decision_time", "symbol"],
        how="left",
        validate="one_to_one",
    )
    required = [
        "score_1pct",
        "score_5pct",
        "prior_residual_return",
        "stale_score_1pct",
    ]
    panel["v161_feature_ready"] = np.isfinite(panel[required]).all(axis=1)
    ready = panel.groupby("decision_time", observed=True)["v161_feature_ready"].agg(
        ["count", "sum"]
    )
    usable_times = ready.index[
        ready["count"].eq(len(FROZEN_SYMBOLS))
        & ready["sum"].eq(len(FROZEN_SYMBOLS))
    ]
    panel = panel[panel["decision_time"].isin(usable_times)].copy()
    coverage = coverage.merge(
        ready.rename(columns={"count": "v161_symbols", "sum": "v161_ready_symbols"}),
        left_on="decision_time",
        right_index=True,
        how="left",
    )
    coverage["v161_usable"] = coverage["decision_time"].isin(usable_times)
    return (
        panel.sort_values(["decision_time", "symbol"]).reset_index(drop=True),
        coverage.sort_values("decision_time").reset_index(drop=True),
    )


def build_v161_random_controls(
    panel: pd.DataFrame,
    cfg: V161Config = V161Config(),
) -> pd.DataFrame:
    symbols = np.asarray(sorted(FROZEN_SYMBOLS))
    matrices = {
        column: panel.pivot(index="decision_time", columns="symbol", values=column).reindex(
            columns=symbols
        )
        for column in (
            "prior_residual_return",
            "withdrawal_percentile_1pct",
            "btc_beta",
            "price_return",
        )
    }
    times = matrices["btc_beta"].index
    residual = matrices["prior_residual_return"].reindex(times).to_numpy(dtype=float)
    withdrawal = matrices["withdrawal_percentile_1pct"].reindex(times).to_numpy(
        dtype=float
    )
    betas = matrices["btc_beta"].reindex(times).to_numpy(dtype=float)
    returns = matrices["price_return"].reindex(times).to_numpy(dtype=float)
    btc_return = (
        panel.groupby("decision_time", observed=True)["btc_return"]
        .first()
        .reindex(times)
        .to_numpy(dtype=float)
    )
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

    for time_index in range(len(times)):
        if gaps[time_index]:
            forced_close = np.abs(previous_alt).sum(axis=1) + np.abs(previous_btc)
            total_net -= cfg.one_way_cost * forced_close
            total_turnover += forced_close
            previous_alt.fill(0.0)
            previous_btc.fill(0.0)
            previous_longs.fill(False)
            previous_shorts.fill(False)
        random_keys = rng.random((path_count, symbol_count))
        permutation = np.argsort(random_keys, axis=1)
        shuffled_withdrawal = withdrawal[time_index][permutation]
        score = residual[time_index][None, :] * shuffled_withdrawal
        order = np.argsort(-score, axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, rank_values[None, :], axis=1)
        top_half = rank < 8
        longs = select_side(rank, previous_longs, top_half, rank)
        shorts = select_side(
            rank,
            previous_shorts,
            ~top_half,
            symbol_count - 1 - rank,
        )
        raw_alt = 0.125 * longs.astype(float) - 0.125 * shorts.astype(float)
        raw_btc = -(raw_alt @ betas[time_index])
        gross = np.abs(raw_alt).sum(axis=1) + np.abs(raw_btc)
        alt = raw_alt / gross[:, None]
        btc = raw_btc / gross
        turnover = np.abs(alt - previous_alt).sum(axis=1) + np.abs(btc - previous_btc)
        gross_return = alt @ returns[time_index] + btc * btc_return[time_index]
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


def _moving_block_bootstrap(values: np.ndarray, cfg: V161Config) -> tuple[float, float]:
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


def summarize_v161(
    portfolio: pd.DataFrame,
    price_only: pd.DataFrame,
    reversed_control: pd.DataFrame,
    stale_control: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V161Config = V161Config(),
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
        "random_depth_pairing_percentile": 100
        * random_controls["mean_primary_net_return"].le(observed_mean).mean(),
        "positive_month_concentration": concentration,
        "mean_one_way_turnover": portfolio["realized_turnover"].mean(),
        "price_only_mean_bp": price_only["primary_net_return"].mean() * 10_000,
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
        and row["random_depth_pairing_percentile"] >= 99
        and row["positive_month_concentration"] <= 0.25
        and row["mean_one_way_turnover"] <= 0.35
        and row["mean_primary_net_bp"] > row["price_only_mean_bp"]
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["stale_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-10
        and row["max_gross_notional_drift"] <= 1e-10
    )
    return pd.DataFrame([row])


def write_v161_hourly_liquidity_withdrawal_amplification(
    cfg: V161Config = V161Config(),
) -> dict[str, Path]:
    features = load_v161_features(cfg.feature_root)
    panel, coverage = build_v161_panel(features, load_v155_hourly_prices(), cfg)
    portfolio = build_v159_portfolio(
        panel,
        V159Config(one_way_cost=cfg.one_way_cost, stress_one_way_cost=cfg.stress_one_way_cost),
        feature_column="score_1pct",
        candidate=CANDIDATE,
    )
    price_only = build_v159_portfolio(
        panel,
        V159Config(one_way_cost=cfg.one_way_cost, stress_one_way_cost=cfg.stress_one_way_cost),
        feature_column="prior_residual_return",
        candidate=PRICE_ONLY_CONTROL,
    )
    reversed_control = build_v159_portfolio(
        panel,
        V159Config(one_way_cost=cfg.one_way_cost, stress_one_way_cost=cfg.stress_one_way_cost),
        feature_column="score_1pct",
        candidate=REVERSED_CONTROL,
        reverse=True,
    )
    stale_control = build_v159_portfolio(
        panel,
        V159Config(one_way_cost=cfg.one_way_cost, stress_one_way_cost=cfg.stress_one_way_cost),
        feature_column="stale_score_1pct",
        candidate=STALE_CONTROL,
    )
    diagnostic_5pct = build_v159_portfolio(
        panel,
        V159Config(one_way_cost=cfg.one_way_cost, stress_one_way_cost=cfg.stress_one_way_cost),
        feature_column="score_5pct",
        candidate="LW1_FIVE_PERCENT_DIAGNOSTIC_ONLY",
    )
    random_controls = build_v161_random_controls(panel, cfg)
    summary = summarize_v161(
        portfolio,
        price_only,
        reversed_control,
        stale_control,
        random_controls,
        cfg,
    )
    controls = pd.DataFrame(
        [
            {
                "control": name,
                "hours": len(frame),
                "mean_primary_net_bp": frame["primary_net_return"].mean() * 10_000,
            }
            for name, frame in (
                (PRICE_ONLY_CONTROL, price_only),
                (REVERSED_CONTROL, reversed_control),
                (STALE_CONTROL, stale_control),
                ("LW1_FIVE_PERCENT_DIAGNOSTIC_ONLY_NON_PROMOTABLE", diagnostic_5pct),
            )
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "hourly_symbol_panel.parquet",
        "coverage": root / "coverage.csv",
        "portfolio": root / "hourly_portfolio.parquet",
        "price_only": root / "price_only_control.parquet",
        "reversed": root / "reversed_control.parquet",
        "stale": root / "stale_control.parquet",
        "diagnostic_5pct": root / "five_percent_diagnostic.parquet",
        "random": root / "random_depth_pairings.csv",
        "summary": root / "summary.csv",
        "controls": root / "control_summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    panel.to_parquet(paths["panel"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    price_only.to_parquet(paths["price_only"], index=False)
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
                "# v16.1 Hourly Liquidity-Withdrawal Amplification Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Frozen controls",
                "",
                controls.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Total-depth change, residual-price direction, score multiplication,",
                "costs and gates were frozen before inspecting returns. The 5% row is",
                "diagnostic-only. PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
