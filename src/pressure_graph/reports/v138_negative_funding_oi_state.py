"""Negative-funding rebound split by causal seven-day open-interest state."""

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
    PANEL_PATH,
    summarize_v135,
)


METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
REPORT_ROOT = Path("reports/v13_8_negative_funding_oi_state")
CANDIDATES = (
    "NF5_OI_BUILD_ADAPTIVE4TO9_BTC_BETA_NEUTRAL",
    "NF6_OI_UNWIND_ADAPTIVE4TO9_BTC_BETA_NEUTRAL",
)


@dataclass(frozen=True)
class V138Config:
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


def _oi_changes_for_symbol(path: Path, entries: pd.DatetimeIndex) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=["create_time", "sum_open_interest"])
    raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True, errors="coerce")
    raw["sum_open_interest"] = pd.to_numeric(raw["sum_open_interest"], errors="coerce")
    raw = (
        raw.dropna(subset=["create_time", "sum_open_interest"])
        .loc[lambda frame: frame["sum_open_interest"].gt(0)]
        .drop_duplicates("create_time", keep="last")
        .sort_values("create_time")
    )
    times = pd.DatetimeIndex(raw["create_time"])
    values = raw["sum_open_interest"].to_numpy(dtype=float)
    rows = []
    for entry in entries:
        current_index = int(times.searchsorted(entry, side="left") - 1)
        past_time = entry - pd.Timedelta(days=7)
        past_index = int(times.searchsorted(past_time, side="left") - 1)
        if current_index < 0 or past_index < 0:
            continue
        rows.append(
            {
                "symbol": path.stem,
                "entry_time": entry,
                "oi_current_time": times[current_index],
                "oi_past_time": times[past_index],
                "oi_change_7d": float(np.log(values[current_index]) - np.log(values[past_index])),
            }
        )
    return pd.DataFrame(rows)


def load_v138_panel(cfg: V138Config = V138Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    entries = pd.DatetimeIndex(sorted(panel["entry_time"].unique()))
    frames = []
    for symbol in sorted(panel["symbol"].astype(str).unique()):
        path = cfg.metrics_root / f"{symbol}.parquet"
        if path.exists():
            frames.append(_oi_changes_for_symbol(path, entries))
    oi = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return (
        panel.merge(oi, on=["symbol", "entry_time"], how="left", validate="one_to_one")
        .sort_values(["entry_time", "symbol"])
        .reset_index(drop=True)
    )


def _eligible_oi_state(local: pd.DataFrame, candidate: str) -> pd.DataFrame:
    usable = local.dropna(subset=["score_7d", "oi_change_7d", "price_return", "btc_beta"])
    usable = usable[usable["score_7d"].lt(0)]
    if candidate == CANDIDATES[0]:
        usable = usable[usable["oi_change_7d"].gt(0)]
    elif candidate == CANDIDATES[1]:
        usable = usable[usable["oi_change_7d"].le(0)]
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    return usable.sort_values(["score_7d", "symbol"], ascending=[True, True])


def _select_oi_hold_band(
    local: pd.DataFrame,
    previous: list[str],
    candidate: str,
    cfg: V138Config,
) -> list[str]:
    ranked = _eligible_oi_state(local, candidate)
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


def _candidate_portfolio(
    panel: pd.DataFrame,
    candidate: str,
    cfg: V138Config,
) -> pd.DataFrame:
    rows = []
    previous: list[str] = []
    previous_weights: dict[str, float] | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        eligible_count = len(_eligible_oi_state(local, candidate))
        selected = _select_oi_hold_band(local, previous, candidate, cfg)
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
                "candidate": candidate,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": int(local["oi_change_7d"].notna().sum()),
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


def build_v138_portfolios(
    panel: pd.DataFrame,
    cfg: V138Config = V138Config(),
) -> pd.DataFrame:
    return pd.concat(
        [_candidate_portfolio(panel, candidate, cfg) for candidate in CANDIDATES],
        ignore_index=True,
    )


def build_v138_nulls(
    panel: pd.DataFrame,
    portfolios: pd.DataFrame,
    cfg: V138Config = V138Config(),
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    groups = {
        pd.Timestamp(entry): frame
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }
    rows = []
    for candidate in CANDIDATES:
        observed = portfolios[portfolios["candidate"].eq(candidate)]
        for iteration in range(cfg.null_iterations):
            returns = []
            for row in observed.itertuples(index=False):
                local = groups[pd.Timestamp(row.entry_time)]
                eligible = _eligible_oi_state(local, candidate)["symbol"].astype(str).tolist()
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
                    "candidate": candidate,
                    "iteration": iteration,
                    "null_type": "random_same_oi_state_same_breadth_observed_cost",
                    "mean_primary_net_return": float(np.mean(returns)),
                }
            )
    return pd.DataFrame(rows)


def summarize_v138(
    portfolios: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V138Config = V138Config(),
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        local = portfolios[portfolios["candidate"].eq(candidate)].copy()
        candidate_nulls = nulls[nulls["candidate"].eq(candidate)]
        summary = summarize_v135(local, candidate_nulls, cfg)
        summary.loc[0, "candidate"] = candidate
        summary.loc[0, "full_weeks"] = int(local["breadth_state"].eq("full9").sum())
        row = summary.iloc[0].to_dict()
        row["promote"] = bool(
            row["weeks"] >= 45
            and row["months"] >= 11
            and row["validation_weeks"] >= 10
            and row["holdout_weeks"] >= 10
            and row["contracted_weeks"] >= 5
            and row["full_weeks"] >= 5
            and row["mean_turnover"] <= 0.50
            and row["mean_funding_bp"] > 0
            and all(
                row[key] > 0
                for key in (
                    "development_primary_net_bp",
                    "validation_primary_net_bp",
                    "holdout_primary_net_bp",
                    "mean_stress_net_bp",
                    "full9_primary_net_bp",
                    "contracted_primary_net_bp",
                    "bootstrap_95_low_bp",
                )
            )
            and row["null_percentile"] >= 95
            and row["positive_month_concentration"] <= 0.35
            and row["worst_period_bp"] >= -40
            and row["max_abs_residual_btc_beta"] <= 1e-12
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_v138_negative_funding_oi_state(
    cfg: V138Config = V138Config(),
) -> dict[str, Path]:
    panel = load_v138_panel(cfg)
    portfolios = build_v138_portfolios(panel, cfg)
    nulls = build_v138_nulls(panel, portfolios, cfg)
    summary = summarize_v138(portfolios, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolios": root / "weekly_portfolios.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v138_negative_funding_oi_state_findings_2026_07_15.md"),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolios.drop(columns="_weights").to_parquet(paths["portfolios"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "panel_rows": len(panel),
                "oi_coverage": int(panel["oi_change_7d"].notna().sum()),
                "portfolio_weeks": {
                    candidate: int(
                        portfolios.loc[
                            portfolios["candidate"].eq(candidate), "entry_time"
                        ].nunique()
                    )
                    for candidate in CANDIDATES
                },
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_oi_state_family"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v13.8 Negative-Funding OI-State Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "OI snapshots are strictly pre-entry and pre-lookback-boundary. Binance",
                "OI contributes no return. The two-direction family uses a 95th-percentile",
                "random-basket gate. PaperLive and leverage/status permissions are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
