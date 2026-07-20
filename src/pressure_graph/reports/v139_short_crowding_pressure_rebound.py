"""Continuous funding/OI/taker short-crowding pressure rebound."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v134_negative_funding_beta_neutral_rebound import (
    _weights_and_components,
)
from pressure_graph.reports.v135_adaptive_negative_funding_breadth import (
    summarize_v135,
)
from pressure_graph.reports.v138_negative_funding_oi_state import METRICS_ROOT


PANEL_PATH = Path("reports/v13_8_negative_funding_oi_state/weekly_symbol_panel.parquet")
REPORT_ROOT = Path("reports/v13_9_short_crowding_pressure_rebound")
CANDIDATE = "NF7_SHORT_CROWDING_PRESSURE_ADAPTIVE4TO9_BTC_BETA_NEUTRAL"


@dataclass(frozen=True)
class V139Config:
    panel_path: Path = PANEL_PATH
    metrics_root: Path = METRICS_ROOT
    report_root: Path = REPORT_ROOT
    minimum_breadth: int = 4
    maximum_breadth: int = 9
    hold_rank: int = 18
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 20260715


def _weekly_taker_for_symbol(
    path: Path,
    entries: pd.DatetimeIndex,
) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=["create_time", "sum_taker_long_short_vol_ratio"])
    raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True, errors="coerce")
    raw["sum_taker_long_short_vol_ratio"] = pd.to_numeric(
        raw["sum_taker_long_short_vol_ratio"], errors="coerce"
    )
    raw = (
        raw.dropna(subset=["create_time", "sum_taker_long_short_vol_ratio"])
        .loc[lambda frame: frame["sum_taker_long_short_vol_ratio"].gt(0)]
        .drop_duplicates("create_time", keep="last")
        .sort_values("create_time")
    )
    times = pd.DatetimeIndex(raw["create_time"])
    values = np.log(raw["sum_taker_long_short_vol_ratio"].to_numpy(dtype=float))
    rows = []
    for entry in entries:
        start = int(times.searchsorted(entry - pd.Timedelta(days=7), side="left"))
        stop = int(times.searchsorted(entry, side="left"))
        if stop <= start:
            continue
        rows.append(
            {
                "symbol": path.stem,
                "entry_time": entry,
                "taker_log_mean_7d": float(values[start:stop].mean()),
                "taker_rows_7d": stop - start,
            }
        )
    return pd.DataFrame(rows)


def load_v139_panel(cfg: V139Config = V139Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    entries = pd.DatetimeIndex(sorted(panel["entry_time"].unique()))
    frames = []
    for symbol in sorted(panel["symbol"].astype(str).unique()):
        path = cfg.metrics_root / f"{symbol}.parquet"
        if path.exists():
            frames.append(_weekly_taker_for_symbol(path, entries))
    taker = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return (
        panel.merge(taker, on=["symbol", "entry_time"], how="left", validate="one_to_one")
        .sort_values(["entry_time", "symbol"])
        .reset_index(drop=True)
    )


def _ranked_pressure(local: pd.DataFrame) -> pd.DataFrame:
    ranked = local.dropna(
        subset=[
            "score_7d",
            "oi_change_7d",
            "taker_log_mean_7d",
            "price_return",
            "btc_beta",
        ]
    )
    ranked = ranked[ranked["score_7d"].lt(0)].copy()
    ranked["funding_rank"] = (-ranked["score_7d"]).rank(pct=True, method="average")
    ranked["oi_rank"] = ranked["oi_change_7d"].rank(pct=True, method="average")
    ranked["sell_rank"] = (-ranked["taker_log_mean_7d"]).rank(pct=True, method="average")
    ranked["pressure_score"] = ranked[["funding_rank", "oi_rank", "sell_rank"]].mean(axis=1)
    return ranked.sort_values(["pressure_score", "symbol"], ascending=[False, True])


def _select_pressure_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    cfg: V139Config,
) -> list[str]:
    ranked = _ranked_pressure(local)
    if len(ranked) < cfg.minimum_breadth:
        return []
    target = min(cfg.maximum_breadth, len(ranked))
    ranks = {str(symbol): rank for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)}
    selected = [
        symbol for symbol in previous if symbol in ranks and ranks[symbol] <= cfg.hold_rank
    ][:target]
    for symbol in ranked["symbol"].astype(str):
        if len(selected) >= target:
            break
        if symbol not in selected:
            selected.append(symbol)
    return selected


def build_v139_portfolio(
    panel: pd.DataFrame,
    cfg: V139Config = V139Config(),
) -> pd.DataFrame:
    rows = []
    previous: list[str] = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        ranked = _ranked_pressure(local)
        selected = _select_pressure_hold_band(local, previous, cfg)
        if not selected:
            if previous_weights is not None and rows:
                rows[-1]["realized_turnover"] += sum(
                    abs(weight) for weight in previous_weights.values()
                )
            previous = []
            previous_weights = None
            continue
        weights, components = _weights_and_components(local, selected, cfg)
        if not weights:
            continue
        if previous_weights is None:
            turnover = sum(abs(weight) for weight in weights.values())
        else:
            turnover = sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in set(weights) | set(previous_weights)
            )
        selected_scores = ranked.set_index("symbol").loc[selected, "pressure_score"]
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": int(local["taker_log_mean_7d"].notna().sum()),
                "eligible_negative_names": len(ranked),
                "selected_breadth": len(selected),
                "breadth_state": "full9" if len(selected) == 9 else "contracted4to8",
                "mean_selected_pressure_score": float(selected_scores.mean()),
                "selected_symbols": "|".join(selected),
                "retained_names": len(set(selected) & set(previous)),
                "realized_turnover": turnover,
                "_weights": weights,
                **components,
            }
        )
        previous = selected
        previous_weights = weights
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output.loc[output.index[-1], "realized_turnover"] += sum(
        abs(weight) for weight in output.iloc[-1]["_weights"].values()
    )
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"] - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v139_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V139Config = V139Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    groups = {
        pd.Timestamp(entry): frame
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }
    rows = []
    for iteration in range(cfg.null_iterations):
        returns = []
        for row in portfolio.itertuples(index=False):
            local = groups[pd.Timestamp(row.entry_time)]
            eligible = _ranked_pressure(local)["symbol"].astype(str).tolist()
            selected = list(
                np.asarray(eligible)[
                    rng.choice(len(eligible), size=int(row.selected_breadth), replace=False)
                ]
            )
            _, components = _weights_and_components(local, selected, cfg)
            returns.append(
                components["gross_return"] - cfg.one_way_cost * float(row.realized_turnover)
            )
        rows.append(
            {
                "iteration": iteration,
                "null_type": "random_negative_same_breadth_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def write_v139_short_crowding_pressure_rebound(
    cfg: V139Config = V139Config(),
) -> dict[str, Path]:
    panel = load_v139_panel(cfg)
    portfolio = build_v139_portfolio(panel, cfg)
    nulls = build_v139_nulls(panel, portfolio, cfg)
    summary = summarize_v135(portfolio, nulls, cfg)
    summary["candidate"] = CANDIDATE
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v139_short_crowding_pressure_rebound_findings_2026_07_15.md"),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolio.drop(columns="_weights").to_parquet(paths["portfolio"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "panel_rows": len(panel),
                "taker_coverage": int(panel["taker_log_mean_7d"].notna().sum()),
                "weeks": len(portfolio),
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
                "# v13.9 Continuous Short-Crowding Pressure Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Funding, OI, and taker windows are strictly pre-entry. Binance metrics",
                "contribute no return. PaperLive and leverage/status permissions remain",
                "unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
