"""Dual-venue negative-funding confirmation for the beta-neutral rebound basket."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    BINANCE_ROOT,
    RECENT_ROOT,
    _combined_funding,
)
from pressure_graph.reports.v134_negative_funding_beta_neutral_rebound import (
    _weights_and_components,
)
from pressure_graph.reports.v135_adaptive_negative_funding_breadth import (
    PANEL_PATH,
    summarize_v135,
)


REPORT_ROOT = Path("reports/v13_7_cross_venue_consensus_negative_funding")
CANDIDATE = "NF4_DUAL_VENUE_NEGATIVE_ADAPTIVE4TO9_BTC_BETA_NEUTRAL"


@dataclass(frozen=True)
class V137Config:
    panel_path: Path = PANEL_PATH
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


def load_v137_panel(cfg: V137Config = V137Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    binance = _combined_funding(
        BINANCE_ROOT / "funding",
        RECENT_ROOT / "binance_funding",
        "bybit_symbol",
    )
    groups = {
        str(symbol): frame.sort_values("funding_time")
        for symbol, frame in binance.groupby("symbol", observed=True)
    }
    scores = []
    for row in panel.itertuples(index=False):
        frame = groups.get(str(row.symbol))
        if frame is None:
            scores.append(np.nan)
            continue
        time = frame["funding_time"]
        rate = frame["funding_rate_settled"]
        window = rate[time.ge(row.entry_time - pd.Timedelta(days=7)) & time.lt(row.entry_time)]
        scores.append(float(window.sum()) if not window.empty else np.nan)
    panel["binance_score_7d"] = scores
    panel["consensus_score_7d"] = panel["score_7d"] + panel["binance_score_7d"]
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _dual_negative(local: pd.DataFrame) -> pd.DataFrame:
    usable = local.dropna(
        subset=[
            "score_7d",
            "binance_score_7d",
            "consensus_score_7d",
            "price_return",
            "btc_beta",
        ]
    )
    return usable[usable["score_7d"].lt(0) & usable["binance_score_7d"].lt(0)].sort_values(
        ["consensus_score_7d", "symbol"], ascending=[True, True]
    )


def _select_dual_negative_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    cfg: V137Config,
) -> list[str]:
    ranked = _dual_negative(local)
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


def build_v137_portfolio(
    panel: pd.DataFrame,
    cfg: V137Config = V137Config(),
) -> pd.DataFrame:
    rows = []
    previous: list[str] = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        eligible_count = len(_dual_negative(local))
        selected = _select_dual_negative_hold_band(local, previous, cfg)
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
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": int(local["binance_score_7d"].notna().sum()),
                "eligible_negative_names": eligible_count,
                "selected_breadth": len(selected),
                "breadth_state": "full9" if len(selected) == 9 else "contracted4to8",
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


def build_v137_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V137Config = V137Config(),
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
            eligible = _dual_negative(local)["symbol"].astype(str).tolist()
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
                "null_type": "random_dual_negative_same_breadth_observed_cost",
                "mean_primary_net_return": float(np.mean(returns)),
            }
        )
    return pd.DataFrame(rows)


def write_v137_cross_venue_consensus_negative_funding(
    cfg: V137Config = V137Config(),
) -> dict[str, Path]:
    panel = load_v137_panel(cfg)
    portfolio = build_v137_portfolio(panel, cfg)
    nulls = build_v137_nulls(panel, portfolio, cfg)
    summary = summarize_v135(portfolio, nulls, cfg)
    summary["candidate"] = CANDIDATE
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v137_cross_venue_consensus_negative_funding_findings_2026_07_15.md"),
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
                "weeks": len(portfolio),
                "last_entry": portfolio["entry_time"].max().isoformat(),
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
                "# v13.7 Cross-Venue Consensus Negative-Funding Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Both venue funding windows are strictly pre-entry. Binance confirms",
                "the signal but contributes no return. PaperLive and leverage/status",
                "permissions remain unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
