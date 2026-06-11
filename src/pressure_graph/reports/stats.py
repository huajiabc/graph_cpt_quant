from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pressure_graph.backtest import simulate_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.paths import add_event_columns, add_path_signals, iter_parameter_variants


PATH_SIGNAL_COLS = [
    "short_squeeze_signal",
    "momentum_ignition_signal",
    "crowded_long_risk",
]


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean())


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").quantile(q))


def baseline_masks(df: pd.DataFrame, universe_col: str) -> dict[str, pd.Series]:
    base = df[universe_col].fillna(False).astype(bool) if universe_col in df.columns else True
    return {
        "all_universe_baseline": base,
        "BTC_market_safe_only": base & (pd.to_numeric(df["btc_ret_4h"], errors="coerce") > -0.015),
        "ret_4h_percentile_only": base
        & (pd.to_numeric(df["ret_4h_percentile"], errors="coerce") > 70),
        "OI_up_only": base & (pd.to_numeric(df["oi_value_delta_4h_percentile"], errors="coerce") > 75),
        "funding_not_hot_only": base & (pd.to_numeric(df["funding_percentile"], errors="coerce") < 60),
    }


def _metric_row(
    df: pd.DataFrame,
    mask: pd.Series,
    name: str,
    level: str,
    config: ExperimentConfig,
    baseline_name: str | None = None,
    baseline_metrics: dict[str, float] | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    sample = df[mask.fillna(False)]
    n = len(sample)
    if trades is None or trades.empty:
        expectancy = float("nan")
        expectancy_funding = float("nan")
    else:
        expectancy = _safe_mean(trades["net_return_ex_fee_slippage"])
        expectancy_funding = _safe_mean(trades["net_return_ex_fee_slippage_funding"])

    row: dict[str, float | int | str] = {
        "path_name": name,
        "level": level,
        "sample_n": n,
        "hit_3pct_4h": _safe_mean(sample.get("hit_3pct_4h", pd.Series(dtype=float))),
        "hit_5pct_4h": _safe_mean(sample.get("hit_5pct_4h", pd.Series(dtype=float))),
        "hit_3pct_12h": _safe_mean(sample.get("hit_3pct_12h", pd.Series(dtype=float))),
        "hit_5pct_12h": _safe_mean(sample.get("hit_5pct_12h", pd.Series(dtype=float))),
        "dd_2pct_before_hit_3pct": _safe_mean(
            sample.get("dd_2pct_before_hit_3pct_4h", pd.Series(dtype=float))
        ),
        "dd_3pct_before_hit_5pct": _safe_mean(
            sample.get("dd_3pct_before_hit_5pct_4h", pd.Series(dtype=float))
        ),
        "mean_max_up_4h": _safe_mean(sample.get("future_max_up_4h", pd.Series(dtype=float))),
        "median_max_up_4h": _safe_quantile(
            sample.get("future_max_up_4h", pd.Series(dtype=float)), 0.5
        ),
        "p25_max_up_4h": _safe_quantile(sample.get("future_max_up_4h", pd.Series(dtype=float)), 0.25),
        "mean_max_down_4h": _safe_mean(sample.get("future_max_down_4h", pd.Series(dtype=float))),
        "median_max_down_4h": _safe_quantile(
            sample.get("future_max_down_4h", pd.Series(dtype=float)), 0.5
        ),
        "p25_max_down_4h": _safe_quantile(
            sample.get("future_max_down_4h", pd.Series(dtype=float)), 0.25
        ),
        "cost_adjusted_expectancy": expectancy,
        "cost_adjusted_expectancy_funding": expectancy_funding,
        "baseline_name": baseline_name or "",
    }
    raw_score = (
        0.35 * float(row["hit_3pct_4h"] or 0)
        + 0.25 * float(row["hit_5pct_4h"] or 0)
        + 0.20 * float(row["median_max_up_4h"] or 0)
        - 0.30 * float(row["dd_2pct_before_hit_3pct"] or 0)
        - 0.20 * abs(float(row["p25_max_down_4h"] or 0))
    )
    row["raw_score"] = raw_score
    row["adjusted_score"] = raw_score * n / (n + config.validation.sample_shrinkage_k)
    if baseline_metrics:
        row["hit_3pct_lift"] = float(row["hit_3pct_4h"]) - baseline_metrics["hit_3pct_4h"]
        row["hit_5pct_lift"] = float(row["hit_5pct_4h"]) - baseline_metrics["hit_5pct_4h"]
        row["median_max_up_lift"] = (
            float(row["median_max_up_4h"]) - baseline_metrics["median_max_up_4h"]
        )
        row["dd_before_hit_delta"] = (
            float(row["dd_2pct_before_hit_3pct"])
            - baseline_metrics["dd_2pct_before_hit_3pct"]
        )
    else:
        row["hit_3pct_lift"] = np.nan
        row["hit_5pct_lift"] = np.nan
        row["median_max_up_lift"] = np.nan
        row["dd_before_hit_delta"] = np.nan
    row["max_symbol_contribution"] = _max_contribution(sample, "symbol")
    row["max_month_contribution"] = _max_month_contribution(sample)
    row["grade"] = candidate_grade(row, config)
    return row


def _max_contribution(sample: pd.DataFrame, col: str) -> float:
    if sample.empty or col not in sample.columns:
        return float("nan")
    counts = sample[col].value_counts(normalize=True)
    return float(counts.iloc[0]) if not counts.empty else float("nan")


def _max_month_contribution(sample: pd.DataFrame) -> float:
    if sample.empty:
        return float("nan")
    months = pd.to_datetime(sample["bar_open_time"], utc=True).dt.strftime("%Y-%m")
    counts = months.value_counts(normalize=True)
    return float(counts.iloc[0]) if not counts.empty else float("nan")


def candidate_grade(row: dict[str, float | int | str], config: ExperimentConfig) -> str:
    n = int(row.get("sample_n", 0) or 0)
    lift = float(row.get("hit_3pct_lift", 0) or 0)
    dd_delta = float(row.get("dd_before_hit_delta", 0) or 0)
    expectancy = float(row.get("cost_adjusted_expectancy", 0) or 0)
    score = float(row.get("adjusted_score", 0) or 0)
    concentration_bad = (
        float(row.get("max_symbol_contribution", 0) or 0) > config.validation.max_symbol_contribution
        or float(row.get("max_month_contribution", 0) or 0) > config.validation.max_month_contribution
    )
    if n < 30 or lift <= 0 or dd_delta > 0.02:
        return "D"
    if concentration_bad:
        return "C"
    if n >= 200 and lift >= 0.03 and expectancy > 0 and score > 0:
        return "A"
    if n >= 80 and lift > 0 and score > -0.02:
        return "B"
    return "C"


def _baseline_metric_dict(df: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    sample = df[mask.fillna(False)]
    return {
        "hit_3pct_4h": _safe_mean(sample.get("hit_3pct_4h", pd.Series(dtype=float))),
        "hit_5pct_4h": _safe_mean(sample.get("hit_5pct_4h", pd.Series(dtype=float))),
        "median_max_up_4h": _safe_quantile(
            sample.get("future_max_up_4h", pd.Series(dtype=float)), 0.5
        ),
        "dd_2pct_before_hit_3pct": _safe_mean(
            sample.get("dd_2pct_before_hit_3pct_4h", pd.Series(dtype=float))
        ),
    }


def path_stats(
    df: pd.DataFrame,
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
    level: str = "event",
) -> pd.DataFrame:
    data = df.copy()
    signal_cols = PATH_SIGNAL_COLS
    if level == "event":
        data = add_event_columns(data, signal_cols, config.events.cooldown_bars_4h)
        signal_cols = [f"{col}_event" for col in signal_cols]

    base_masks = baseline_masks(data, universe_col)
    all_base_metrics = _baseline_metric_dict(data, base_masks["all_universe_baseline"])
    rows: list[dict[str, float | int | str]] = []
    for baseline_name, mask in base_masks.items():
        rows.append(_metric_row(data, mask, baseline_name, level, config, baseline_name=""))

    primary_cost = config.execution.cost_single_side_bps[0]
    primary_rule = config.execution.rules["fast"]
    for signal_col in signal_cols:
        path_name = signal_col.removesuffix("_event").removesuffix("_signal")
        mask = data[signal_col].fillna(False).astype(bool)
        mask = mask & base_masks["all_universe_baseline"]
        trades = simulate_trades(
            data,
            signal_col,
            path_name,
            primary_rule,
            primary_cost,
            config.execution.ambiguity_policy,
            config.events.one_position_per_symbol,
        )
        rows.append(
            _metric_row(
                data,
                mask,
                path_name,
                level,
                config,
                baseline_name="all_universe_baseline",
                baseline_metrics=all_base_metrics,
                trades=trades,
            )
        )
    return pd.DataFrame(rows)


def development_holdout_split(
    df: pd.DataFrame,
    final_holdout_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    times = pd.to_datetime(df["bar_open_time"], utc=True)
    max_time = times.max()
    last_month = pd.Timestamp(max_time.year, max_time.month, 1, tz="UTC")
    holdout_start = last_month - pd.DateOffset(months=final_holdout_months - 1)
    dev = df[times < holdout_start].copy()
    holdout = df[times >= holdout_start].copy()
    return dev, holdout


def walk_forward_report(
    df: pd.DataFrame,
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if df.empty:
        return pd.DataFrame()
    data = add_event_columns(df, PATH_SIGNAL_COLS, config.events.cooldown_bars_4h)
    data["window"] = pd.to_datetime(data["bar_open_time"], utc=True).dt.strftime("%Y-%m")
    for window, group in data.groupby("window", sort=True):
        stats = path_stats(group, config, universe_col, level=config.validation.primary_level)
        stats["window"] = window
        rows.append(stats)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parameter_grid_stats(
    df: pd.DataFrame,
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    base = baseline_masks(df, universe_col)["all_universe_baseline"]
    base_metrics = _baseline_metric_dict(df, base)
    for variant in iter_parameter_variants(config):
        data = add_path_signals(df, config, **variant.params)
        signal_col = (
            "short_squeeze_signal"
            if variant.path_name == "short_squeeze"
            else "momentum_ignition_signal"
        )
        data = add_event_columns(data, [signal_col], config.events.cooldown_bars_4h)
        event_col = f"{signal_col}_event"
        row = _metric_row(
            data,
            data[event_col].fillna(False).astype(bool) & base,
            variant.signal_col,
            "event",
            config,
            baseline_name="all_universe_baseline",
            baseline_metrics=base_metrics,
        )
        row.update({"family": variant.path_name, **variant.params})
        rows.append(row)
    return pd.DataFrame(rows)


def write_heatmaps(grid_stats: pd.DataFrame, report_root: Path) -> None:
    if grid_stats.empty:
        return
    out_dir = ensure_dir(report_root / "param_heatmaps")
    for family, group in grid_stats.groupby("family"):
        if family == "short_squeeze":
            x_col, y_col = "short_oi_pct", "short_funding_pct"
        else:
            x_col, y_col = "momentum_oi_min_pct", "momentum_funding_pct"
        if x_col not in group.columns or y_col not in group.columns:
            continue
        pivot = group.pivot_table(
            values="hit_3pct_lift",
            index=y_col,
            columns=x_col,
            aggfunc="mean",
        )
        if pivot.empty:
            continue
        plt.figure(figsize=(7, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0)
        plt.title(f"{family} hit_3pct_lift")
        plt.tight_layout()
        plt.savefig(out_dir / f"{family}_hit_3pct_lift.png", dpi=160)
        plt.close()


def candidate_grades(path_stats_df: pd.DataFrame) -> pd.DataFrame:
    paths = path_stats_df[
        path_stats_df["path_name"].isin(["short_squeeze", "momentum_ignition", "crowded_long_risk"])
    ].copy()
    return paths.sort_values(["grade", "adjusted_score"], ascending=[True, False])


def write_candidate_list(path_stats_df: pd.DataFrame, report_root: Path) -> None:
    candidates = candidate_grades(path_stats_df)
    lines = ["# Crypto Pressure Graph v0 Candidate List", ""]
    if candidates.empty:
        lines.append("No candidate paths passed the reporting filters.")
    else:
        for grade, group in candidates.groupby("grade", sort=True):
            lines.append(f"## {grade} 类")
            for row in group.itertuples(index=False):
                lines.append(
                    f"- {row.path_name}: n={row.sample_n}, "
                    f"hit_3pct_lift={row.hit_3pct_lift:.3f}, "
                    f"expectancy={row.cost_adjusted_expectancy:.4f}, "
                    f"score={row.adjusted_score:.4f}"
                )
            lines.append("")
    ensure_dir(report_root)
    (report_root / "candidate_list.md").write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    df: pd.DataFrame,
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
) -> dict[str, Path]:
    report_root = ensure_dir(config.paths.report_root)
    data = add_path_signals(df, config)
    bar_stats = path_stats(data, config, universe_col, level="bar")
    event_stats = path_stats(data, config, universe_col, level="event")
    stats = pd.concat([bar_stats, event_stats], ignore_index=True)
    stats_path = report_root / "path_stats.csv"
    stats.to_csv(stats_path, index=False)

    dev, holdout = development_holdout_split(data, config.validation.final_holdout_months)
    dev_wf = walk_forward_report(dev, config, universe_col)
    holdout_stats = path_stats(holdout, config, universe_col, level=config.validation.primary_level)
    dev_path = report_root / "development_walk_forward.csv"
    holdout_path = report_root / "final_holdout.csv"
    dev_wf.to_csv(dev_path, index=False)
    holdout_stats.to_csv(holdout_path, index=False)

    grid = parameter_grid_stats(dev, config, universe_col)
    grid_path = report_root / "parameter_grid_stats.csv"
    grid.to_csv(grid_path, index=False)
    write_heatmaps(grid, report_root)
    write_candidate_list(holdout_stats, report_root)
    return {
        "path_stats": stats_path,
        "development_walk_forward": dev_path,
        "final_holdout": holdout_path,
        "parameter_grid_stats": grid_path,
        "candidate_list": report_root / "candidate_list.md",
    }
