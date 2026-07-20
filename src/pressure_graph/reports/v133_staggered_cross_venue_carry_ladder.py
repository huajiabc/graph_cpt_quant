"""Seven-weekday staggered implementation of the cross-venue carry signal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v121_top_trader_community_rotation import _period
from pressure_graph.reports.v126_turnover_governed_cross_venue_carry import (
    _select_hold_band,
)
from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    BINANCE_ROOT,
    RAW_BYBIT_ROOT,
    RECENT_ROOT,
    _combined_funding,
    hourly_bybit_prices,
    load_v132_binance_prices,
    load_v132_bybit_klines,
)


MEMBERSHIP_PATH = Path(
    "reports/v13_2_tg1_forward_temporal_extension/monthly_balanced_membership_extended.csv"
)
MONDAY_TG1_PATH = Path("reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet")
REPORT_ROOT = Path("reports/v13_3_staggered_cross_venue_carry_ladder")
CANDIDATE = "SL1_7COHORT_30D_TOP9_HOLD18"


@dataclass(frozen=True)
class V133Config:
    membership_path: Path = MEMBERSHIP_PATH
    monday_tg1_path: Path = MONDAY_TG1_PATH
    report_root: Path = REPORT_ROOT
    first_entry: pd.Timestamp = pd.Timestamp("2025-08-04", tz="UTC")
    bucket_size: int = 9
    hold_rank: int = 18
    cohort_count: int = 7
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 500
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def load_v133_membership(cfg: V133Config = V133Config()) -> pd.DataFrame:
    membership = pd.read_csv(cfg.membership_path)
    membership["month_start"] = pd.to_datetime(membership["month_start"], utc=True, errors="coerce")
    return (
        membership.dropna(subset=["month_start", "symbol"])
        .drop_duplicates(["month_start", "symbol"], keep="last")
        .sort_values(["month_start", "symbol"])
        .reset_index(drop=True)
    )


def build_v133_daily_panel(
    bybit_funding: pd.DataFrame,
    binance_funding: pd.DataFrame,
    bybit_prices: pd.DataFrame,
    binance_prices: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: V133Config = V133Config(),
) -> pd.DataFrame:
    """Build strictly causal daily scores and next-day pair returns."""
    bybit_close = bybit_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    binance_close = binance_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="binance_close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    if bybit_close.empty or binance_close.empty:
        return pd.DataFrame()

    last_price_time = min(bybit_close.index.max(), binance_close.index.max())
    last_membership_month = membership["month_start"].max()
    membership_end = last_membership_month + pd.offsets.MonthBegin(1)
    last_day = min(last_price_time.floor("D"), membership_end)
    entries = pd.date_range(
        cfg.first_entry,
        last_day - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    )
    bybit_groups = {
        str(symbol): group.sort_values("funding_time")
        for symbol, group in bybit_funding.groupby("symbol", observed=True)
    }
    binance_groups = {
        str(symbol): group.sort_values("funding_time")
        for symbol, group in binance_funding.groupby("symbol", observed=True)
    }
    month_symbols = {
        pd.Timestamp(month): sorted(group["symbol"].astype(str).unique())
        for month, group in membership.groupby("month_start", observed=True)
    }

    rows: list[dict[str, object]] = []
    for entry in entries:
        exit_time = entry + pd.Timedelta(days=1)
        if any(
            time not in frame.index
            for frame in (bybit_close, binance_close)
            for time in (entry, exit_time)
        ):
            continue
        month = entry.floor("D").replace(day=1)
        for symbol in month_symbols.get(month, []):
            if symbol not in bybit_close.columns or symbol not in binance_close.columns:
                continue
            prices = (
                bybit_close.at[entry, symbol],
                bybit_close.at[exit_time, symbol],
                binance_close.at[entry, symbol],
                binance_close.at[exit_time, symbol],
            )
            if not all(np.isfinite(value) and value > 0 for value in prices):
                continue
            bybit = bybit_groups.get(symbol)
            binance = binance_groups.get(symbol)
            if bybit is None or binance is None:
                continue
            bt = bybit["funding_time"]
            br = bybit["funding_rate_settled"]
            nt = binance["funding_time"]
            nr = binance["funding_rate_settled"]
            bybit_30d = br[bt.ge(entry - pd.Timedelta(days=30)) & bt.lt(entry)]
            binance_30d = nr[nt.ge(entry - pd.Timedelta(days=30)) & nt.lt(entry)]
            if bybit_30d.empty or binance_30d.empty:
                continue
            future_bybit = float(br[bt.gt(entry) & bt.le(exit_time)].sum())
            future_binance = float(nr[nt.gt(entry) & nt.le(exit_time)].sum())
            bybit_return = float(prices[1] / prices[0] - 1.0)
            binance_return = float(prices[3] / prices[2] - 1.0)
            price_basis = bybit_return - binance_return
            funding_spread = future_binance - future_bybit
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "month_start": month,
                    "symbol": symbol,
                    "score_30d": float(binance_30d.sum() - bybit_30d.sum()),
                    "bybit_return": bybit_return,
                    "binance_return": binance_return,
                    "price_basis_return": price_basis,
                    "funding_spread_return": funding_spread,
                    "pair_gross_return": price_basis + funding_spread,
                }
            )
    return pd.DataFrame(rows)


def _random_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    cfg: V133Config,
    rng: np.random.Generator,
) -> list[str]:
    usable = local.dropna(subset=["score_30d", "pair_gross_return"])
    symbols = sorted(usable.loc[usable["score_30d"].gt(0), "symbol"].astype(str).unique())
    if len(symbols) < cfg.bucket_size:
        return []
    ranked = list(np.asarray(symbols)[rng.permutation(len(symbols))])
    ranks = {str(symbol): rank for rank, symbol in enumerate(ranked, start=1)}
    retained = [
        symbol for symbol in previous if symbol in ranks and ranks[symbol] <= cfg.hold_rank
    ][: cfg.bucket_size]
    selected = retained.copy()
    for symbol in ranked:
        symbol = str(symbol)
        if len(selected) >= cfg.bucket_size:
            break
        if symbol not in selected:
            selected.append(symbol)
    return selected


def _daily_ladder_path(
    panel: pd.DataFrame,
    cfg: V133Config,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    states: dict[int, list[str]] = {weekday: [] for weekday in range(cfg.cohort_count)}
    rows: list[dict[str, object]] = []
    for entry, frame in panel.groupby("entry_time", sort=True, observed=True):
        local = frame.set_index("symbol", drop=False)
        weekday = int(pd.Timestamp(entry).weekday())
        previous = states[weekday]
        if rng is None:
            selected = _select_hold_band(frame, previous, cfg.bucket_size, cfg.hold_rank)
        else:
            selected = _random_hold_band(frame, previous, cfg, rng)
        if selected:
            states[weekday] = selected
        cohort_turnover = 0.0
        if selected:
            previous_weights = {symbol: 1.0 / cfg.bucket_size for symbol in previous}
            current_weights = {symbol: 1.0 / cfg.bucket_size for symbol in selected}
            cohort_turnover = sum(
                abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in set(previous_weights) | set(current_weights)
            )

        active = [holdings for holdings in states.values() if holdings]
        components: dict[str, float] = {}
        for output_name, column in (
            ("price_basis_return", "price_basis_return"),
            ("funding_spread_return", "funding_spread_return"),
            ("gross_return", "pair_gross_return"),
        ):
            value = 0.0
            for holdings in active:
                available = [symbol for symbol in holdings if symbol in local.index]
                if len(available) != cfg.bucket_size:
                    continue
                value += float(local.loc[available, column].mean()) / cfg.cohort_count
            components[output_name] = value
        rows.append(
            {
                "entry_time": entry,
                "exit_time": frame["exit_time"].iloc[0],
                "active_cohorts": len(active),
                "coverage": len(frame),
                "rebalance_weekday": weekday,
                "selected_symbols": "|".join(selected),
                "portfolio_turnover": cohort_turnover / cfg.cohort_count,
                **components,
            }
        )
    return pd.DataFrame(rows)


def _week_start(times: pd.Series) -> pd.Series:
    return times.dt.floor("D") - pd.to_timedelta(times.dt.weekday, unit="D")


def aggregate_v133_weeks(
    daily: pd.DataFrame,
    cfg: V133Config = V133Config(),
    observed_costs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate only fully invested, complete Monday-to-Monday calendar weeks."""
    if daily.empty:
        return pd.DataFrame()
    local = daily.copy()
    local["entry_time"] = pd.to_datetime(local["entry_time"], utc=True)
    local["week_start"] = _week_start(local["entry_time"])
    first_week = local["week_start"].min()
    startup_turnover = float(
        local.loc[local["week_start"].eq(first_week), "portfolio_turnover"].sum()
    )
    groups = []
    for week, frame in local.groupby("week_start", sort=True, observed=True):
        if week == first_week or len(frame) != 7 or frame["active_cohorts"].min() != 7:
            continue
        groups.append(
            {
                "candidate": CANDIDATE,
                "entry_time": week,
                "exit_time": week + pd.Timedelta(days=7),
                "month_start": week.replace(day=1),
                "period": _period(week),
                "coverage": float(frame["coverage"].median()),
                "price_basis_return": float(frame["price_basis_return"].sum()),
                "funding_spread_return": float(frame["funding_spread_return"].sum()),
                "gross_return": float(frame["gross_return"].sum()),
                "realized_turnover": float(frame["portfolio_turnover"].sum()),
            }
        )
    weekly = pd.DataFrame(groups)
    if weekly.empty:
        return weekly
    if observed_costs is None:
        weekly.loc[weekly.index[0], "realized_turnover"] += startup_turnover
        weekly.loc[weekly.index[-1], "realized_turnover"] += 1.0
    else:
        cost_map = observed_costs.set_index("entry_time")["realized_turnover"]
        weekly["realized_turnover"] = weekly["entry_time"].map(cost_map)
        weekly = weekly.dropna(subset=["realized_turnover"]).reset_index(drop=True)
    weekly["primary_net_return"] = (
        weekly["gross_return"] - cfg.one_way_cost * weekly["realized_turnover"]
    )
    weekly["stress_net_return"] = (
        weekly["gross_return"] - cfg.stress_one_way_cost * weekly["realized_turnover"]
    )
    return weekly


def build_v133_nulls(
    panel: pd.DataFrame,
    observed: pd.DataFrame,
    cfg: V133Config = V133Config(),
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.null_iterations):
        rng = np.random.default_rng(cfg.seed + 10_000 + iteration)
        daily = _daily_ladder_path(panel, cfg, rng=rng)
        weekly = aggregate_v133_weeks(daily, cfg, observed_costs=observed)
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_positive_spread_ranking_observed_cost",
                "mean_primary_net_return": float(weekly["primary_net_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _moving_block_means(
    values: np.ndarray,
    iterations: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=float)
    block_count = int(np.ceil(len(values) / block_length))
    draws = np.empty(iterations, dtype=float)
    offsets = np.arange(block_length)
    for iteration in range(iterations):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draws[iteration] = values[indices.ravel()[: len(values)]].mean()
    return draws


def summarize_v133(
    weekly: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V133Config = V133Config(),
) -> pd.DataFrame:
    values = weekly["primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        values,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 2),
    )
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = weekly.groupby("period", observed=True)["primary_net_return"].mean()
    months = weekly.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    monday = pd.read_parquet(cfg.monday_tg1_path)[["entry_time", "primary_net_return"]].rename(
        columns={"primary_net_return": "monday_primary_net_return"}
    )
    monday["entry_time"] = pd.to_datetime(monday["entry_time"], utc=True)
    overlap = weekly.merge(monday, on="entry_time", how="inner")
    correlation = float(overlap["primary_net_return"].corr(overlap["monday_primary_net_return"]))
    observed = float(values.mean())
    counts = weekly["period"].value_counts()
    row = {
        "candidate": CANDIDATE,
        "weeks": len(weekly),
        "months": int(weekly["month_start"].nunique()),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "median_coverage": float(weekly["coverage"].median()),
        "mean_turnover": float(weekly["realized_turnover"].mean()),
        "mean_price_basis_bp": float(weekly["price_basis_return"].mean() * 10_000),
        "mean_funding_spread_bp": float(weekly["funding_spread_return"].mean() * 10_000),
        "mean_gross_bp": float(weekly["gross_return"].mean() * 10_000),
        "mean_primary_net_bp": observed * 10_000,
        "mean_stress_net_bp": float(weekly["stress_net_return"].mean() * 10_000),
        "development_primary_net_bp": float(periods.get("development", np.nan) * 10_000),
        "validation_primary_net_bp": float(periods.get("validation", np.nan) * 10_000),
        "holdout_primary_net_bp": float(periods.get("holdout", np.nan) * 10_000),
        "bootstrap_95_low_bp": float(ci_low * 10_000),
        "bootstrap_95_high_bp": float(ci_high * 10_000),
        "null_percentile": float(100 * nulls["mean_primary_net_return"].le(observed).mean()),
        "positive_month_concentration": concentration,
        "worst_period_bp": float(periods.min() * 10_000),
        "monday_tg1_correlation": correlation,
        "monday_overlap_weeks": len(overlap),
    }
    row["promote"] = bool(
        row["weeks"] >= 45
        and row["months"] >= 11
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 10
        and row["mean_turnover"] <= 0.50
        and row["mean_funding_spread_bp"] > 0
        and all(
            row[key] > 0
            for key in (
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "mean_stress_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["null_percentile"] >= 90
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and abs(row["monday_tg1_correlation"]) < 0.95
    )
    return pd.DataFrame([row])


def write_v133_staggered_cross_venue_carry_ladder(
    cfg: V133Config = V133Config(),
) -> dict[str, Path]:
    membership = load_v133_membership(cfg)
    bybit_prices = hourly_bybit_prices(load_v132_bybit_klines())
    binance_prices = load_v132_binance_prices()
    bybit_funding = _combined_funding(
        RAW_BYBIT_ROOT / "funding", RECENT_ROOT / "bybit_funding", "symbol"
    )
    binance_funding = _combined_funding(
        BINANCE_ROOT / "funding",
        RECENT_ROOT / "binance_funding",
        "bybit_symbol",
    )
    panel = build_v133_daily_panel(
        bybit_funding,
        binance_funding,
        bybit_prices,
        binance_prices,
        membership,
        cfg,
    )
    daily = _daily_ladder_path(panel, cfg)
    weekly = aggregate_v133_weeks(daily, cfg)
    nulls = build_v133_nulls(panel, weekly, cfg)
    summary = summarize_v133(weekly, nulls, cfg)

    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "daily_symbol_panel.parquet",
        "daily": root / "daily_ladder.parquet",
        "weekly": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v133_staggered_cross_venue_carry_ladder_findings_2026_07_15.md"),
    }
    panel.to_parquet(paths["panel"], index=False)
    daily.to_parquet(paths["daily"], index=False)
    weekly.to_parquet(paths["weekly"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "panel_rows": len(panel),
                "days": int(panel["entry_time"].nunique()),
                "weeks": len(weekly),
                "last_evaluated_week": weekly["entry_time"].max().isoformat(),
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_as_tradable_alpha"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v13.3 Staggered Cross-Venue Carry Ladder Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The seven weekday cohorts use only prior settled funding. Returns are",
                "non-overlapping calendar-week sums of daily marked pair PnL. The first",
                "week is a burn-in whose entry cost is carried into evaluation; terminal",
                "close cost is charged to the last complete week. PaperLive was unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
