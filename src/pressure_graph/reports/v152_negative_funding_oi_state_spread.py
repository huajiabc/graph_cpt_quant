"""OI-build minus OI-unwind spread inside the negative-funding universe."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)
from pressure_graph.reports.v138_negative_funding_oi_state import (
    REPORT_ROOT as V138_REPORT_ROOT,
)
from pressure_graph.reports.v147_funding_sign_spread import beta_neutral_components
from pressure_graph.reports.v149_funding_sign_turnover_cap import (
    _execute_capped_array_transition,
    _neutralize_alt_array,
    execute_capped_transition,
    weight_turnover,
)


PANEL_PATH = V138_REPORT_ROOT / "weekly_symbol_panel.parquet"
FSS3_PATH = Path("reports/v14_9_funding_sign_turnover_cap/weekly_portfolio.parquet")
TG1_PATH = Path("reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet")
REPORT_ROOT = Path("reports/v15_2_negative_funding_oi_state_spread")
FINDINGS_PATH = Path("docs/v152_negative_funding_oi_state_spread_findings_2026_07_16.md")
CANDIDATE = "OI1_NEG_FUNDING_BUILD_MINUS_UNWIND"
REVERSED_CONTROL = "OI1_REVERSED_UNWIND_MINUS_BUILD"


@dataclass(frozen=True)
class V152Config:
    panel_path: Path = PANEL_PATH
    fss3_path: Path = FSS3_PATH
    tg1_path: Path = TG1_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_side_breadth: int = 2
    transition_turnover_cap: float = 0.70
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    bisection_iterations: int = 48
    seed: int = 20260716


def load_v152_panel(cfg: V152Config = V152Config()) -> pd.DataFrame:
    panel = pd.read_parquet(cfg.panel_path)
    for column in ("entry_time", "exit_time", "month_start"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="coerce")
    return panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def oi_state_target(
    local: pd.DataFrame,
    cfg: V152Config = V152Config(),
    *,
    direction: int = 1,
) -> tuple[dict[str, float], dict[str, float], list[str], list[str], int]:
    eligible = local.dropna(
        subset=[
            "score_7d",
            "oi_change_7d",
            "price_return",
            "future_funding",
            "btc_beta",
        ]
    )
    eligible = eligible[eligible["score_7d"].lt(0)].sort_values(
        ["oi_change_7d", "symbol"], ascending=[False, True]
    )
    side_count = len(eligible) // 2
    if side_count < cfg.minimum_side_breadth:
        return {}, {}, [], [], len(eligible)
    build = eligible.iloc[:side_count]["symbol"].astype(str).tolist()
    unwind = eligible.iloc[-side_count:]["symbol"].astype(str).tolist()
    long_symbols, short_symbols = (build, unwind) if direction > 0 else (unwind, build)
    raw = {symbol: 0.5 / side_count for symbol in long_symbols}
    raw.update({symbol: -0.5 / side_count for symbol in short_symbols})
    weights, components = beta_neutral_components(local, raw)
    return weights, components, long_symbols, short_symbols, len(eligible)


def build_v152_portfolio(
    panel: pd.DataFrame,
    cfg: V152Config = V152Config(),
    *,
    direction: int = 1,
) -> pd.DataFrame:
    candidate = CANDIDATE if direction > 0 else REVERSED_CONTROL
    rows: list[dict] = []
    previous_weights: dict[str, float] | None = None
    previous_entry: pd.Timestamp | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        entry = pd.Timestamp(entry)
        target, _, target_long, target_short, eligible_breadth = oi_state_target(
            local, cfg, direction=direction
        )
        has_gap = previous_entry is not None and entry - previous_entry > pd.Timedelta(
            days=7, minutes=1
        )
        if has_gap and previous_weights is not None:
            rows[-1]["realized_turnover"] += sum(abs(w) for w in previous_weights.values())
            previous_weights = None
        if not target:
            if previous_weights is not None:
                rows[-1]["realized_turnover"] += sum(abs(w) for w in previous_weights.values())
                previous_weights = None
            previous_entry = entry
            continue
        weights, components, fraction, rebalance_turnover, cap_breach = (
            execute_capped_transition(
                local,
                previous_weights,
                target,
                cfg.transition_turnover_cap,
                cfg.bisection_iterations,
            )
        )
        indexed = local.set_index("symbol")
        executed_alt = [symbol for symbol in weights if symbol != BTC]
        long_oi = [float(indexed.at[symbol, "oi_change_7d"]) for symbol in executed_alt if weights[symbol] > 0]
        short_oi = [float(indexed.at[symbol, "oi_change_7d"]) for symbol in executed_alt if weights[symbol] < 0]
        rows.append(
            {
                "candidate": candidate,
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "coverage": len(local),
                "eligible_breadth": eligible_breadth,
                "side_breadth": len(target_long),
                "breadth_state": "thin4to7" if eligible_breadth <= 7 else "broad8plus",
                "target_long_symbols": "|".join(target_long),
                "target_short_symbols": "|".join(target_short),
                "selected_long_symbols": "|".join(
                    sorted(symbol for symbol in executed_alt if weights[symbol] > 0)
                ),
                "selected_short_symbols": "|".join(
                    sorted(symbol for symbol in executed_alt if weights[symbol] < 0)
                ),
                "mean_executed_long_oi_change": float(np.mean(long_oi)),
                "mean_executed_short_oi_change": float(np.mean(short_oi)),
                "executed_target_fraction": fraction,
                "target_tracking_l1": weight_turnover(weights, target),
                "cap_applicable": previous_weights is not None,
                "cap_binding": previous_weights is not None and fraction < 1.0 - 1e-10,
                "rebalance_turnover": rebalance_turnover,
                "cap_breach": cap_breach,
                "realized_turnover": rebalance_turnover,
                "_weights": weights,
                **components,
            }
        )
        previous_weights = weights
        previous_entry = entry
    if rows and previous_weights is not None:
        rows[-1]["realized_turnover"] += sum(abs(w) for w in previous_weights.values())
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"] - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v152_nulls(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V152Config = V152Config(),
) -> pd.DataFrame:
    observed = portfolio.sort_values("entry_time").reset_index(drop=True)
    all_symbols = sorted(panel["symbol"].astype(str).unique())
    symbol_to_index = {symbol: index for index, symbol in enumerate(all_symbols)}
    groups = {
        pd.Timestamp(entry): local
        for entry, local in panel.groupby("entry_time", sort=True, observed=True)
    }
    weeks = []
    for row in observed.itertuples(index=False):
        local = groups[pd.Timestamp(row.entry_time)]
        eligible = local.dropna(
            subset=["score_7d", "oi_change_7d", "price_return", "future_funding", "btc_beta"]
        )
        eligible = eligible[eligible["score_7d"].lt(0)]
        indexed = eligible.set_index("symbol")
        symbols = sorted(eligible["symbol"].astype(str).unique())
        indices = np.asarray([symbol_to_index[symbol] for symbol in symbols], dtype=int)
        beta = np.zeros(len(all_symbols), dtype=float)
        price = np.zeros(len(all_symbols), dtype=float)
        funding = np.zeros(len(all_symbols), dtype=float)
        beta[indices] = indexed.loc[symbols, "btc_beta"].to_numpy(dtype=float)
        price[indices] = indexed.loc[symbols, "price_return"].to_numpy(dtype=float)
        funding[indices] = indexed.loc[symbols, "future_funding"].to_numpy(dtype=float)
        mask = np.zeros(len(all_symbols), dtype=bool)
        mask[indices] = True
        weeks.append(
            {
                "indices": indices,
                "beta": beta,
                "price": price,
                "funding": funding,
                "mask": mask,
                "btc_return": float(eligible.iloc[0]["btc_return"]),
                "btc_funding": float(eligible.iloc[0]["btc_future_funding"]),
                "side_count": int(row.side_breadth),
            }
        )
    rows = []
    for iteration in range(cfg.null_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        previous_alt: np.ndarray | None = None
        previous_btc = 0.0
        gross_returns = []
        turnovers = []
        for week in weeks:
            chosen = week["indices"][rng.permutation(len(week["indices"]))]
            side_count = week["side_count"]
            raw = np.zeros(len(all_symbols), dtype=float)
            raw[chosen[:side_count]] = 0.5 / side_count
            raw[chosen[-side_count:]] = -0.5 / side_count
            target_alt, target_btc = _neutralize_alt_array(raw, week["beta"])
            alt, btc, turnover = _execute_capped_array_transition(
                previous_alt,
                previous_btc,
                target_alt,
                target_btc,
                week["beta"],
                week["mask"],
                cfg.transition_turnover_cap,
                cfg.bisection_iterations,
            )
            gross_returns.append(
                float(
                    np.dot(alt, week["price"])
                    + btc * week["btc_return"]
                    - np.dot(alt, week["funding"])
                    - btc * week["btc_funding"]
                )
            )
            turnovers.append(turnover)
            previous_alt = alt
            previous_btc = btc
        if turnovers and previous_alt is not None:
            turnovers[-1] += float(np.abs(previous_alt).sum() + abs(previous_btc))
        primary = np.asarray(gross_returns) - cfg.one_way_cost * np.asarray(turnovers)
        rows.append(
            {
                "iteration": iteration,
                "candidate": CANDIDATE,
                "null_type": "permuted_oi_ranks_identical_capped_execution",
                "mean_primary_net_return": float(primary.mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_reference_returns(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)[["entry_time", "primary_net_return"]].copy()
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    return frame.rename(columns={"primary_net_return": name})


def summarize_v152(
    portfolio: pd.DataFrame,
    reversed_control: pd.DataFrame,
    nulls: pd.DataFrame,
    cfg: V152Config = V152Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = portfolio.sort_values("entry_time")
    values = sample["primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        values,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 2),
    )
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = sample.groupby("period", observed=True)["primary_net_return"].mean()
    states = sample.groupby("breadth_state", observed=True)["primary_net_return"].mean()
    months = sample.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive = months[months.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    counts = sample["period"].value_counts()
    applicable = sample[sample["cap_applicable"]]
    reversed_mean = float(reversed_control["primary_net_return"].mean())
    observed_mean = float(values.mean())
    references = sample[["entry_time", "primary_net_return"]]
    references = references.merge(
        _load_reference_returns(cfg.fss3_path, "fss3_return"), on="entry_time", how="inner"
    ).merge(_load_reference_returns(cfg.tg1_path, "tg1_return"), on="entry_time", how="inner")
    fss3_correlation = float(references[["primary_net_return", "fss3_return"]].corr().iloc[0, 1])
    tg1_correlation = float(references[["primary_net_return", "tg1_return"]].corr().iloc[0, 1])
    row = {
        "candidate": CANDIDATE,
        "weeks": len(sample),
        "months": sample["month_start"].nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "thin_weeks": int(sample["breadth_state"].eq("thin4to7").sum()),
        "median_eligible_breadth": sample["eligible_breadth"].median(),
        "mean_turnover": sample["realized_turnover"].mean(),
        "cap_binding_weeks": int(sample["cap_binding"].sum()),
        "mean_executed_target_fraction": sample["executed_target_fraction"].mean(),
        "max_capped_transition_turnover": applicable["rebalance_turnover"].max(),
        "max_cap_breach": sample["cap_breach"].max(),
        "mean_oi_state_gap": (
            sample["mean_executed_long_oi_change"] - sample["mean_executed_short_oi_change"]
        ).mean(),
        "mean_price_bp": sample["price_return"].mean() * 10_000,
        "mean_funding_bp": sample["funding_return"].mean() * 10_000,
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": sample["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "thin_primary_net_bp": states.get("thin4to7", np.nan) * 10_000,
        "broad_primary_net_bp": states.get("broad8plus", np.nan) * 10_000,
        "bootstrap_95_low_bp": ci_low * 10_000,
        "bootstrap_95_high_bp": ci_high * 10_000,
        "null_percentile": 100 * nulls["mean_primary_net_return"].le(observed_mean).mean(),
        "positive_month_concentration": concentration,
        "worst_period_bp": periods.min() * 10_000,
        "reversed_control_mean_bp": reversed_mean * 10_000,
        "correlation_with_fss3": fss3_correlation,
        "correlation_with_tg1": tg1_correlation,
        "max_abs_residual_btc_beta": sample["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (sample["gross_notional"] - 1.0).abs().max(),
    }
    row["promote"] = bool(
        row["weeks"] >= 45
        and row["months"] >= 11
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 10
        and row["thin_weeks"] >= 5
        and row["mean_turnover"] <= 0.75
        and row["max_capped_transition_turnover"] <= cfg.transition_turnover_cap + 1e-10
        and row["max_cap_breach"] <= 1e-10
        and all(
            row[key] > 0
            for key in (
                "mean_price_bp",
                "mean_stress_net_bp",
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "thin_primary_net_bp",
                "broad_primary_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["null_percentile"] >= 95
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
        and abs(row["correlation_with_fss3"]) <= 0.50
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-12
        and row["max_gross_notional_drift"] <= 1e-12
    )
    control = pd.DataFrame(
        [
            {
                "candidate": REVERSED_CONTROL,
                "weeks": len(reversed_control),
                "mean_primary_net_bp": reversed_mean * 10_000,
                "mean_stress_net_bp": reversed_control["stress_net_return"].mean() * 10_000,
            }
        ]
    )
    return pd.DataFrame([row]), control


def write_v152_negative_funding_oi_state_spread(
    cfg: V152Config = V152Config(),
) -> dict[str, Path]:
    panel = load_v152_panel(cfg)
    portfolio = build_v152_portfolio(panel, cfg, direction=1)
    reversed_control = build_v152_portfolio(panel, cfg, direction=-1)
    nulls = build_v152_nulls(panel, portfolio, cfg)
    summary, control = summarize_v152(portfolio, reversed_control, nulls, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_portfolio.parquet",
        "reversed": root / "reversed_control.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "control": root / "control_summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    portfolio.drop(columns="_weights").to_parquet(paths["portfolio"], index=False)
    reversed_control.drop(columns="_weights").to_parquet(paths["reversed"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    control.to_csv(paths["control"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "weeks": len(portfolio),
                "null_iterations": cfg.null_iterations,
                "promoted": promoted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_candidate"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v15.2 Negative-Funding OI-State Spread Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Reversed control",
                "",
                control.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The OI direction, within-negative-funding split, execution cap and all",
                "gates were frozen before return inspection. PaperLive is unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
