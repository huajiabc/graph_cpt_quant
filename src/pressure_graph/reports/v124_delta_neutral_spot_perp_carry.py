"""Weekly delta-neutral Binance spot / Bybit perpetual funding carry."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v121_top_trader_community_rotation import (
    _membership,
    _period,
)
from pressure_graph.reports.v123_cross_sectional_funding_carry import (
    load_v123_funding,
    load_v123_prices,
)


REPORT_ROOT = Path("reports/v12_4_delta_neutral_spot_perp_carry")
SPOT_ROOT = Path("data/external/binance_spot_1h")
FUNDING_ROOT = Path("data/raw/bybit/funding")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
CANDIDATES = (
    "DN1_7D_TOP_FUNDING",
    "DN2_30D_TOP_FUNDING",
    "DN3_COMMUNITY_TOP_FUNDING",
)


@dataclass(frozen=True)
class V124Config:
    spot_root: Path = SPOT_ROOT
    funding_root: Path = FUNDING_ROOT
    feature_path: Path = FEATURE_PATH
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    first_entry: pd.Timestamp = pd.Timestamp("2025-08-04", tz="UTC")
    holding_days: int = 7
    bucket_size: int = 9
    community_count: int = 8
    focal_cost: float = 0.004
    stress_cost: float = 0.008
    one_way_cost: float = 0.002
    direct_null_iterations: int = 500
    community_null_iterations: int = 200
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def load_v124_spot(cfg: V124Config = V124Config()) -> pd.DataFrame:
    frames = []
    for symbol in sorted(_membership(cfg)["symbol"].astype(str).unique()):
        path = cfg.spot_root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["bybit_symbol", "feature_time", "close"]
        ).rename(columns={"bybit_symbol": "symbol", "close": "spot_close"})
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    spot = pd.concat(frames, ignore_index=True)
    spot["feature_time"] = pd.to_datetime(
        spot["feature_time"], utc=True, errors="coerce"
    )
    spot["spot_close"] = pd.to_numeric(spot["spot_close"], errors="coerce")
    return (
        spot.dropna(subset=["symbol", "feature_time", "spot_close"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def build_v124_weekly_panel(
    funding: pd.DataFrame,
    perpetual_prices: pd.DataFrame,
    spot_prices: pd.DataFrame,
    cfg: V124Config = V124Config(),
) -> pd.DataFrame:
    membership = _membership(cfg)
    perp_close = perpetual_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    spot_close = spot_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="spot_close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    if perp_close.empty or spot_close.empty:
        return pd.DataFrame()
    last_price_time = min(perp_close.index.max(), spot_close.index.max())
    last_entry = last_price_time - pd.Timedelta(days=cfg.holding_days)
    entries = pd.date_range(cfg.first_entry, last_entry, freq="7D", tz="UTC")
    funding_by_symbol = {
        symbol: group.sort_values("funding_time")
        for symbol, group in funding.groupby("symbol", observed=True)
    }
    rows = []
    for entry in entries:
        exit_time = entry + pd.Timedelta(days=cfg.holding_days)
        if (
            entry not in perp_close.index
            or exit_time not in perp_close.index
            or entry not in spot_close.index
            or exit_time not in spot_close.index
        ):
            continue
        month = entry.floor("D").replace(day=1)
        local_membership = membership[membership["month_start"].eq(month)]
        if local_membership.empty:
            continue
        for item in local_membership.itertuples(index=False):
            symbol = str(item.symbol)
            if symbol not in perp_close.columns or symbol not in spot_close.columns:
                continue
            prices = (
                perp_close.loc[entry, symbol],
                perp_close.loc[exit_time, symbol],
                spot_close.loc[entry, symbol],
                spot_close.loc[exit_time, symbol],
            )
            if not all(np.isfinite(value) and value > 0 for value in prices):
                continue
            symbol_funding = funding_by_symbol.get(symbol)
            if symbol_funding is None:
                continue
            time = symbol_funding["funding_time"]
            rate = symbol_funding["funding_rate_settled"]
            history_7d = rate[
                time.ge(entry - pd.Timedelta(days=7)) & time.lt(entry)
            ]
            history_30d = rate[
                time.ge(entry - pd.Timedelta(days=30)) & time.lt(entry)
            ]
            if history_7d.empty or history_30d.empty:
                continue
            future_funding = float(rate[time.gt(entry) & time.le(exit_time)].sum())
            perp_return = float(prices[1] / prices[0] - 1.0)
            spot_return = float(prices[3] / prices[2] - 1.0)
            basis_return = spot_return - perp_return
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "month_start": month,
                    "period": _period(entry),
                    "community_id": str(item.community_id),
                    "symbol": symbol,
                    "score_7d": float(history_7d.sum()),
                    "score_30d": float(history_30d.mean()),
                    "future_funding": future_funding,
                    "spot_return": spot_return,
                    "perp_return": perp_return,
                    "basis_return": basis_return,
                    "pair_gross_return": basis_return + future_funding,
                }
            )
    return pd.DataFrame(rows)


def _top_positive_weights(
    local: pd.DataFrame, score_column: str, bucket_size: int
) -> tuple[dict[str, float], list[str]]:
    usable = local.dropna(subset=[score_column, "pair_gross_return"])
    usable = usable[usable[score_column].gt(0)].sort_values(
        [score_column, "symbol"]
    )
    selected = usable.tail(bucket_size)["symbol"].astype(str).tolist()
    if len(selected) < bucket_size:
        return {}, []
    return {symbol: 1.0 / bucket_size for symbol in selected}, selected


def _components(local: pd.DataFrame, weights: dict[str, float]) -> dict[str, float]:
    indexed = local.set_index("symbol")
    return {
        name: float(
            sum(weights[symbol] * indexed.loc[symbol, column] for symbol in weights)
        )
        for name, column in (
            ("spot_return", "spot_return"),
            ("perp_return", "perp_return"),
            ("basis_return", "basis_return"),
            ("funding_return", "future_funding"),
            ("gross_return", "pair_gross_return"),
        )
    }


def build_v124_portfolios(
    panel: pd.DataFrame, cfg: V124Config = V124Config()
) -> pd.DataFrame:
    rows = []
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        for candidate, score_column in zip(CANDIDATES[:2], ("score_7d", "score_30d")):
            weights, selected = _top_positive_weights(
                local, score_column, cfg.bucket_size
            )
            if not weights:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "entry_time": entry,
                    "exit_time": local["exit_time"].iloc[0],
                    "month_start": local["month_start"].iloc[0],
                    "period": local["period"].iloc[0],
                    "coverage": len(local),
                    "selected_symbols": "|".join(selected),
                    "_weights": weights,
                    **_components(local, weights),
                }
            )

        community_selected = []
        for community, group in local.groupby("community_id", observed=True):
            usable = group.dropna(subset=["score_7d", "pair_gross_return"])
            usable = usable[usable["score_7d"].gt(0)].sort_values(
                ["score_7d", "symbol"]
            )
            if usable.empty:
                continue
            community_selected.append((str(community), str(usable.iloc[-1]["symbol"])))
        if len(community_selected) != cfg.community_count:
            continue
        weights = {
            symbol: 1.0 / cfg.community_count
            for _, symbol in community_selected
        }
        if len(weights) != cfg.community_count:
            continue
        rows.append(
            {
                "candidate": CANDIDATES[2],
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "selected_symbols": "|".join(
                    symbol for _, symbol in community_selected
                ),
                "_weights": weights,
                **_components(local, weights),
            }
        )
    return pd.DataFrame(rows)


def apply_v124_costs(
    portfolios: pd.DataFrame, cfg: V124Config = V124Config()
) -> pd.DataFrame:
    output = portfolios.copy()
    output["net_40bp"] = output["gross_return"] - cfg.focal_cost
    output["net_80bp"] = output["gross_return"] - cfg.stress_cost
    output["realized_turnover"] = np.nan
    output["turnover_net_20bp_oneway"] = np.nan
    for _, indices in output.groupby("candidate", sort=True).groups.items():
        ordered = output.loc[indices].sort_values("entry_time")
        previous: dict[str, float] | None = None
        turnovers = []
        for index, row in ordered.iterrows():
            current = row["_weights"]
            if previous is None:
                turnover = 1.0
            else:
                symbols = set(previous) | set(current)
                turnover = sum(
                    abs(current.get(symbol, 0.0) - previous.get(symbol, 0.0))
                    for symbol in symbols
                )
            turnovers.append((index, turnover))
            previous = current
        if turnovers:
            index, turnover = turnovers[-1]
            turnovers[-1] = (index, turnover + 1.0)
        for index, turnover in turnovers:
            output.loc[index, "realized_turnover"] = turnover
            output.loc[index, "turnover_net_20bp_oneway"] = (
                output.loc[index, "gross_return"] - cfg.one_way_cost * turnover
            )
    return output


def build_v124_nulls(
    panel: pd.DataFrame, cfg: V124Config = V124Config()
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    weeks = [group.copy() for _, group in panel.groupby("entry_time", sort=True)]
    rows = []
    for candidate, score_column in zip(CANDIDATES[:2], ("score_7d", "score_30d")):
        for iteration in range(cfg.direct_null_iterations):
            returns = []
            for local in weeks:
                usable = local.dropna(subset=[score_column, "pair_gross_return"])
                usable = usable[usable[score_column].gt(0)]
                if len(usable) < cfg.bucket_size:
                    continue
                selected = rng.choice(
                    usable["pair_gross_return"].to_numpy(dtype=float),
                    size=cfg.bucket_size,
                    replace=False,
                )
                returns.append(float(selected.mean()))
            rows.append(
                {
                    "candidate": candidate,
                    "iteration": iteration,
                    "null_type": "within_week_random_positive_funding",
                    "mean_net_40bp": float(np.mean(returns) - cfg.focal_cost),
                }
            )

    months = {
        pd.Timestamp(month): sorted(group["symbol"].astype(str).unique())
        for month, group in _membership(cfg).groupby("month_start", observed=True)
    }
    for iteration in range(cfg.community_null_iterations):
        assignments = {}
        for month, symbols in months.items():
            shuffled = np.asarray(symbols)[rng.permutation(len(symbols))]
            assignments[month] = {
                str(symbol): int(index // 9) for index, symbol in enumerate(shuffled)
            }
        returns = []
        for local in weeks:
            month = pd.Timestamp(local["month_start"].iloc[0])
            randomized = local.assign(
                random_community=local["symbol"].map(assignments[month])
            )
            selected = []
            for _, group in randomized.groupby("random_community", observed=True):
                usable = group.dropna(subset=["score_7d", "pair_gross_return"])
                usable = usable[usable["score_7d"].gt(0)].sort_values(
                    ["score_7d", "symbol"]
                )
                if usable.empty:
                    continue
                selected.append(float(usable.iloc[-1]["pair_gross_return"]))
            if len(selected) == cfg.community_count:
                returns.append(float(np.mean(selected)))
        rows.append(
            {
                "candidate": CANDIDATES[2],
                "iteration": iteration,
                "null_type": "random_monthly_communities_top_positive",
                "mean_net_40bp": float(np.mean(returns) - cfg.focal_cost),
            }
        )
    return pd.DataFrame(rows)


def summarize_v124(
    portfolios: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V124Config = V124Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)
    rows = []
    for candidate, local in portfolios.groupby("candidate", sort=True):
        values = local["net_40bp"].to_numpy(dtype=float)
        draws = rng.choice(
            values, size=(cfg.bootstrap_iterations, len(values)), replace=True
        ).mean(axis=1)
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        periods = local.groupby("period", observed=True)["net_40bp"].mean()
        months = local.groupby("month_start", observed=True)["net_40bp"].sum()
        positive = months[months.gt(0)]
        concentration = (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
        )
        candidate_null = nulls.loc[
            nulls["candidate"].eq(candidate), "mean_net_40bp"
        ]
        observed = float(local["net_40bp"].mean())
        counts = local["period"].value_counts()
        row = {
            "candidate": candidate,
            "weeks": len(local),
            "months": int(local["month_start"].nunique()),
            "validation_weeks": int(counts.get("validation", 0)),
            "holdout_weeks": int(counts.get("holdout", 0)),
            "median_coverage": float(local["coverage"].median()),
            "mean_spot_bp": float(local["spot_return"].mean() * 10_000),
            "mean_perp_bp": float(local["perp_return"].mean() * 10_000),
            "mean_basis_bp": float(local["basis_return"].mean() * 10_000),
            "mean_funding_bp": float(local["funding_return"].mean() * 10_000),
            "mean_gross_bp": float(local["gross_return"].mean() * 10_000),
            "mean_net_40bp_bp": observed * 10_000,
            "mean_net_80bp_bp": float(local["net_80bp"].mean() * 10_000),
            "mean_turnover_net_bp": float(
                local["turnover_net_20bp_oneway"].mean() * 10_000
            ),
            "development_net_40bp_bp": float(
                periods.get("development", np.nan) * 10_000
            ),
            "validation_net_40bp_bp": float(
                periods.get("validation", np.nan) * 10_000
            ),
            "holdout_net_40bp_bp": float(periods.get("holdout", np.nan) * 10_000),
            "bootstrap_95_low_bp": float(ci_low * 10_000),
            "bootstrap_95_high_bp": float(ci_high * 10_000),
            "null_percentile": float(100 * candidate_null.le(observed).mean()),
            "positive_month_concentration": concentration,
            "worst_period_bp": float(periods.min() * 10_000),
        }
        row["promote"] = bool(
            row["weeks"] >= 40
            and row["months"] >= 10
            and row["validation_weeks"] >= 10
            and row["holdout_weeks"] >= 8
            and row["mean_funding_bp"] > 0
            and all(
                row[key] > 0
                for key in (
                    "development_net_40bp_bp",
                    "validation_net_40bp_bp",
                    "holdout_net_40bp_bp",
                    "mean_net_80bp_bp",
                    "bootstrap_95_low_bp",
                )
            )
            and row["null_percentile"] >= 90
            and row["positive_month_concentration"] <= 0.35
            and row["worst_period_bp"] >= -40
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_v124_delta_neutral_spot_perp_carry(
    cfg: V124Config = V124Config(),
) -> dict[str, Path]:
    funding = load_v123_funding(cfg)
    perpetual = load_v123_prices(cfg)
    spot = load_v124_spot(cfg)
    panel = build_v124_weekly_panel(funding, perpetual, spot, cfg)
    portfolios = apply_v124_costs(build_v124_portfolios(panel, cfg), cfg)
    nulls = build_v124_nulls(panel, cfg)
    summary = summarize_v124(portfolios, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolios": root / "weekly_portfolios.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v124_delta_neutral_spot_perp_carry_findings_2026_07_15.md"
        ),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolios.drop(columns="_weights").to_parquet(paths["portfolios"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "spot_rows": len(spot),
                "spot_symbols": int(spot["symbol"].nunique()),
                "funding_rows": len(funding),
                "panel_rows": len(panel),
                "weeks": int(panel["entry_time"].nunique()),
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_candidate" if promoted else "reject_all_as_tradable_alpha"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v12.4 Delta-Neutral Spot/Perpetual Carry Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The basis leg uses bars fully closed by each weekly decision time; funding "
                "settlements are strictly as-of for scores and `(entry, exit]` for PnL. "
                "No existing PaperLive strategy was changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
