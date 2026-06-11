from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest import ENTRY_POLICIES, simulate_entry_policy_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.features import add_v01_features
from pressure_graph.io import ensure_dir
from pressure_graph.labels import add_touch_diagnostics
from pressure_graph.paths import add_event_columns, add_path_signals
from pressure_graph.reports.stats import development_holdout_split


PATHS = {
    "short_squeeze": "short_squeeze_signal",
    "momentum_ignition": "momentum_ignition_signal",
}
RAW_PATHS = {
    "short_squeeze_raw": "short_squeeze_signal_raw",
    "momentum_raw": "momentum_ignition_signal_raw",
}
EVENT_SUFFIX = "_event"


def _mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean()) if len(series) else float("nan")


def _median(series: pd.Series) -> float:
    if not len(series):
        return float("nan")
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def prepare_v01_dataset(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = add_v01_features(df, config)
    out = add_touch_diagnostics(out, {"4h": 16})
    out = add_path_signals(out, config)
    out = add_signal_age(out, ["short_squeeze_signal", "momentum_ignition_signal"])
    out["short_squeeze_rejected_by_veto"] = (
        out["short_squeeze_signal_raw"].fillna(False)
        & out["crowded_long_risk"].fillna(False)
        & ~out["short_squeeze_signal"].fillna(False)
    )
    out["momentum_rejected_by_veto"] = (
        out["momentum_ignition_signal_raw"].fillna(False)
        & out["crowded_long_risk"].fillna(False)
        & ~out["momentum_ignition_signal"].fillna(False)
    )
    signal_cols = [
        "short_squeeze_signal",
        "momentum_ignition_signal",
        "crowded_long_risk",
        "short_squeeze_signal_raw",
        "momentum_ignition_signal_raw",
        "short_squeeze_rejected_by_veto",
        "momentum_rejected_by_veto",
    ]
    return add_event_columns(out, signal_cols, config.events.cooldown_bars_4h)


def _signal_age_for_group(group: pd.DataFrame, signal_col: str) -> pd.Series:
    ages: list[int] = []
    current = 0
    for is_signal in group[signal_col].fillna(False).astype(bool):
        current = current + 1 if is_signal else 0
        ages.append(current)
    return pd.Series(ages, index=group.index)


def add_signal_age(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    for signal_col in signal_cols:
        if signal_col not in out.columns:
            continue
        out[f"{signal_col}_age"] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False)
            .apply(lambda group: _signal_age_for_group(group, signal_col))
            .reindex(out.index)
        )
    return out


def _metrics(sample: pd.DataFrame) -> dict[str, float | int]:
    return {
        "sample_n": len(sample),
        "hit_3pct_4h": _mean(sample.get("hit_3pct_4h", pd.Series(dtype=float))),
        "hit_5pct_4h": _mean(sample.get("hit_5pct_4h", pd.Series(dtype=float))),
        "clean_hit_3pct_4h": _mean(sample.get("clean_hit_3pct_4h", pd.Series(dtype=float))),
        "dirty_hit_3pct_4h": _mean(sample.get("dirty_hit_3pct_4h", pd.Series(dtype=float))),
        "bad_drop_no_hit_4h": _mean(sample.get("bad_drop_no_hit_4h", pd.Series(dtype=float))),
        "no_event_4h": _mean(sample.get("no_event_4h", pd.Series(dtype=float))),
        "dd_2pct_before_hit_3pct": _mean(
            sample.get("dd_2pct_before_hit_3pct_4h", pd.Series(dtype=float))
        ),
        "median_max_up_4h": _median(sample.get("future_max_up_4h", pd.Series(dtype=float))),
        "median_max_down_4h": _median(sample.get("future_max_down_4h", pd.Series(dtype=float))),
        "median_bars_to_hit": _median(sample.get("bars_to_hit_3pct_4h", pd.Series(dtype=float))),
        "median_bars_to_dd": _median(sample.get("bars_to_dd_2pct_4h", pd.Series(dtype=float))),
        "median_bars_to_first_touch": _median(
            sample.get("bars_to_first_touch_4h", pd.Series(dtype=float))
        ),
        "max_symbol_contribution": _max_contribution(sample, "symbol"),
        "max_month_contribution": _max_month_contribution(sample),
    }


def _max_contribution(sample: pd.DataFrame, col: str) -> float:
    if sample.empty or col not in sample.columns:
        return float("nan")
    counts = sample[col].value_counts(normalize=True)
    return float(counts.iloc[0]) if not counts.empty else float("nan")


def _max_month_contribution(sample: pd.DataFrame) -> float:
    if sample.empty or "bar_open_time" not in sample.columns:
        return float("nan")
    months = pd.to_datetime(sample["bar_open_time"], utc=True).dt.strftime("%Y-%m")
    counts = months.value_counts(normalize=True)
    return float(counts.iloc[0]) if not counts.empty else float("nan")


def _rows_for_signals(
    df: pd.DataFrame,
    signal_map: dict[str, str],
    universe_col: str,
    use_event: bool = True,
) -> list[dict[str, float | int | str]]:
    rows = []
    universe = df[universe_col].fillna(False) if universe_col in df.columns else True
    for path_name, signal_col in signal_map.items():
        col = f"{signal_col}{EVENT_SUFFIX}" if use_event else signal_col
        if col not in df.columns:
            continue
        sample = df[df[col].fillna(False) & universe]
        rows.append({"path_name": path_name, "level": "event" if use_event else "bar", **_metrics(sample)})
    return rows


def first_touch_stats(df: pd.DataFrame, universe_col: str) -> pd.DataFrame:
    rows = []
    universe = df[universe_col].fillna(False) if universe_col in df.columns else True
    for path_name, signal_col in PATHS.items():
        event_col = f"{signal_col}{EVENT_SUFFIX}"
        sample = df[df[event_col].fillna(False) & universe]
        counts = sample["first_touch_3up_2down_4h"].value_counts(dropna=False)
        total = counts.sum()
        for touch, count in counts.items():
            rows.append(
                {
                    "path_name": path_name,
                    "first_touch": touch,
                    "sample_n": int(count),
                    "share": float(count / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def clean_hit_path_stats(df: pd.DataFrame, universe_col: str) -> pd.DataFrame:
    rows = _rows_for_signals(df, PATHS | {"crowded_long_risk": "crowded_long_risk"}, universe_col)
    return pd.DataFrame(rows)


def _bucket_series(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if name in {"BTC state"}:
        return series.astype(str)
    if "percentile" in name:
        return pd.cut(numeric, bins=[-np.inf, 20, 40, 60, 80, 90, 95, np.inf])
    if "volume_z" in name:
        return pd.cut(numeric, bins=[-np.inf, 0, 1, 2, 3, np.inf])
    if name in {"close_location_value", "upper_wick_ratio"}:
        return pd.cut(numeric, bins=[-np.inf, 0.25, 0.5, 0.65, 0.8, np.inf])
    return pd.qcut(numeric.rank(method="first"), q=5, duplicates="drop")


def drawdown_attribution(df: pd.DataFrame, universe_col: str) -> pd.DataFrame:
    dimensions = {
        "BTC state": "btc_market_state",
        "symbol_volatility_percentile": "symbol_volatility_percentile",
        "ret_15m_percentile": "ret_15m_percentile",
        "ret_1h_percentile": "ret_1h_percentile",
        "ret_4h_percentile": "ret_4h_percentile",
        "volume_z_1h": "volume_z_1h",
        "volume_z_4h": "volume_z_4h",
        "oi_delta_1h_percentile": "oi_value_delta_1h_percentile",
        "oi_delta_4h_percentile": "oi_value_delta_4h_percentile",
        "funding_percentile": "funding_percentile",
        "candle_range_percentile": "range_pct_percentile",
        "upper_wick_ratio": "upper_wick_ratio",
        "close_location_value": "close_location_value",
        "short_squeeze_signal_age": "short_squeeze_signal_age",
        "momentum_ignition_signal_age": "momentum_ignition_signal_age",
    }
    rows = []
    universe = df[universe_col].fillna(False) if universe_col in df.columns else True
    for path_name, signal_col in PATHS.items():
        sample = df[df[f"{signal_col}{EVENT_SUFFIX}"].fillna(False) & universe].copy()
        for dim_name, col in dimensions.items():
            if col not in sample.columns or sample.empty:
                continue
            sample["_bucket"] = _bucket_series(sample[col], dim_name).astype(str)
            for bucket, group in sample.groupby("_bucket", dropna=False):
                rows.append(
                    {
                        "path_name": path_name,
                        "dimension": dim_name,
                        "bucket": bucket,
                        **_metrics(group),
                    }
                )
    return pd.DataFrame(rows)


def _filter_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    filters = {
        "F1_btc_1h_4h_positive": (df["btc_ret_1h"] > 0) & (df["btc_ret_4h"] > 0),
        "F2_ret_15m_pct_lt_90": df["ret_15m_percentile"] < 90,
        "F3_ret_1h_pct_lt_90": df["ret_1h_percentile"] < 90,
        "F4_close_location_gt_065": df["close_location_value"] > 0.65,
        "F5_upper_wick_lt_040": df["upper_wick_ratio"] < 0.40,
        "F6_range_pct_pct_lt_90": df["range_pct_percentile"] < 90,
        "F7_oi_delta_pct_lt_95": df["oi_value_delta_4h_percentile"] < 95,
        "F8_volume_z_lt_3": (df["volume_z_1h"] < 3.0) & (df["volume_z_4h"] < 3.0),
        "F9_symbol_vol_pct_lt_90": df["symbol_volatility_percentile"] < 90,
        "F10_not_crowded_long_risk": ~df["crowded_long_risk"].fillna(False),
    }
    combos = [
        ("F1_btc_1h_4h_positive", "F4_close_location_gt_065"),
        ("F1_btc_1h_4h_positive", "F5_upper_wick_lt_040"),
        ("F2_ret_15m_pct_lt_90", "F4_close_location_gt_065"),
        ("F2_ret_15m_pct_lt_90", "F5_upper_wick_lt_040"),
        ("F4_close_location_gt_065", "F7_oi_delta_pct_lt_95"),
        ("F5_upper_wick_lt_040", "F7_oi_delta_pct_lt_95"),
        ("F7_oi_delta_pct_lt_95", "F8_volume_z_lt_3"),
        ("F1_btc_1h_4h_positive", "F10_not_crowded_long_risk"),
        ("F4_close_location_gt_065", "F10_not_crowded_long_risk"),
        ("F7_oi_delta_pct_lt_95", "F10_not_crowded_long_risk"),
    ]
    for left, right in combos:
        filters[f"{left}+{right}"] = filters[left] & filters[right]
    return filters


def filter_ablation(df: pd.DataFrame, universe_col: str) -> pd.DataFrame:
    rows = []
    universe = df[universe_col].fillna(False) if universe_col in df.columns else True
    filters = _filter_masks(df)
    for path_name, signal_col in PATHS.items():
        base_mask = df[f"{signal_col}{EVENT_SUFFIX}"].fillna(False) & universe
        base_metrics = _metrics(df[base_mask])
        rows.append({"path_name": path_name, "filter_name": "control", **base_metrics})
        for filter_name, filter_mask in filters.items():
            sample = df[base_mask & filter_mask.fillna(False)]
            metrics = _metrics(sample)
            rows.append(
                {
                    "path_name": path_name,
                    "filter_name": filter_name,
                    **metrics,
                    "sample_retention": (
                        metrics["sample_n"] / base_metrics["sample_n"]
                        if base_metrics["sample_n"]
                        else np.nan
                    ),
                    "clean_hit_delta": metrics["clean_hit_3pct_4h"]
                    - base_metrics["clean_hit_3pct_4h"],
                    "dd_delta": metrics["dd_2pct_before_hit_3pct"]
                    - base_metrics["dd_2pct_before_hit_3pct"],
                }
            )
    return pd.DataFrame(rows)


def volatility_matched_baseline(df: pd.DataFrame, universe_col: str) -> pd.DataFrame:
    data = df.copy()
    universe = data[universe_col].fillna(False) if universe_col in data.columns else True
    data = data[universe].copy()
    data["vol_bucket"] = pd.cut(
        pd.to_numeric(data["symbol_volatility_percentile"], errors="coerce"),
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["vol_0_20", "vol_20_40", "vol_40_60", "vol_60_80", "vol_80_100"],
    ).astype(str)
    data["month"] = pd.to_datetime(data["bar_open_time"], utc=True).dt.strftime("%Y-%m")
    rows = []
    group_cols = ["symbol", "month", "vol_bucket", "btc_market_state"]
    for path_name, signal_col in PATHS.items():
        event_col = f"{signal_col}{EVENT_SUFFIX}"
        signal = data[event_col].fillna(False)
        signal_sample = data[signal]
        matched_non_signal = []
        for _, group in data.groupby(group_cols, sort=False):
            sig_count = int(group[event_col].fillna(False).sum())
            if sig_count <= 0:
                continue
            non_sig = group[~group[event_col].fillna(False)]
            if non_sig.empty:
                continue
            matched_non_signal.append(non_sig)
        baseline = pd.concat(matched_non_signal, ignore_index=True) if matched_non_signal else data.iloc[0:0]
        sig_metrics = _metrics(signal_sample)
        base_metrics = _metrics(baseline)
        rows.append(
            {
                "path_name": path_name,
                "signal_n": sig_metrics["sample_n"],
                "matched_baseline_n": base_metrics["sample_n"],
                "signal_hit_3pct_4h": sig_metrics["hit_3pct_4h"],
                "matched_hit_3pct_4h": base_metrics["hit_3pct_4h"],
                "hit_3pct_lift_vs_vol_matched_baseline": sig_metrics["hit_3pct_4h"]
                - base_metrics["hit_3pct_4h"],
                "signal_dd_before_hit": sig_metrics["dd_2pct_before_hit_3pct"],
                "matched_dd_before_hit": base_metrics["dd_2pct_before_hit_3pct"],
                "dd_delta_vs_vol_matched_baseline": sig_metrics["dd_2pct_before_hit_3pct"]
                - base_metrics["dd_2pct_before_hit_3pct"],
                "signal_clean_hit": sig_metrics["clean_hit_3pct_4h"],
                "matched_clean_hit": base_metrics["clean_hit_3pct_4h"],
                "clean_hit_lift_vs_vol_matched_baseline": sig_metrics["clean_hit_3pct_4h"]
                - base_metrics["clean_hit_3pct_4h"],
            }
        )
    return pd.DataFrame(rows)


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trade_n": 0,
            "gross_expectancy": np.nan,
            "net_expectancy": np.nan,
            "net_expectancy_funding": np.nan,
            "win_rate": np.nan,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "sl_first_rate": np.nan,
            "median_bars_to_entry": np.nan,
        }
    return {
        "trade_n": len(trades),
        "gross_expectancy": _mean(trades["gross_return"]),
        "net_expectancy": _mean(trades["net_return_ex_fee_slippage"]),
        "net_expectancy_funding": _mean(trades["net_return_ex_fee_slippage_funding"]),
        "win_rate": _mean(trades["net_return_ex_fee_slippage"] > 0),
        "tp_rate": _mean(trades["exit_reason"].astype(str).str.startswith("tp")),
        "sl_rate": _mean(trades["exit_reason"].astype(str).str.startswith("sl")),
        "sl_first_rate": _mean(trades["exit_reason"].astype(str).str.startswith("sl")),
        "median_bars_to_entry": _median(trades["bars_from_signal_to_entry"]),
    }


def entry_policy_comparison(df: pd.DataFrame, config: ExperimentConfig, universe_col: str) -> pd.DataFrame:
    rows = []
    universe = df[universe_col].fillna(False) if universe_col in df.columns else True
    data = df[universe].copy()

    def vol_regime_rule(row: pd.Series) -> ExecutionRule:
        vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
        if pd.isna(vol_pct) or vol_pct < 40:
            return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
        if vol_pct < 80:
            return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)

    for path_name, signal_col in PATHS.items():
        event_col = f"{signal_col}{EVENT_SUFFIX}"
        for policy in ENTRY_POLICIES:
            fixed_rules: list[tuple[str, ExecutionRule, Callable[[pd.Series], ExecutionRule] | None]] = [
                (name, rule, None) for name, rule in config.execution.rules.items()
            ]
            fixed_rules.append(
                ("vol_regime_fast", ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16), vol_regime_rule)
            )
            for rule_name, rule, resolver in fixed_rules:
                for cost in [5, 10]:
                    trades = simulate_entry_policy_trades(
                        data,
                        event_col,
                        path_name,
                        policy,
                        rule,
                        cost,
                        config.execution.ambiguity_policy,
                        config.events.one_position_per_symbol,
                        signal_col,
                        resolver,
                    )
                    rows.append(
                        {
                            "path_name": path_name,
                            "entry_policy": policy.name,
                            "execution_rule": rule_name,
                            "cost_single_side_bps": cost,
                            **_trade_metrics(trades),
                        }
                    )
    return pd.DataFrame(rows)


def veto_overlay(df: pd.DataFrame, universe_col: str) -> pd.DataFrame:
    signal_map = {
        "short_squeeze_raw": "short_squeeze_signal_raw",
        "short_squeeze_keep_after_veto": "short_squeeze_signal",
        "short_squeeze_rejected_by_veto": "short_squeeze_rejected_by_veto",
        "momentum_raw": "momentum_ignition_signal_raw",
        "momentum_keep_after_veto": "momentum_ignition_signal",
        "momentum_rejected_by_veto": "momentum_rejected_by_veto",
    }
    return pd.DataFrame(_rows_for_signals(df, signal_map, universe_col))


def write_v01_candidate_list(report_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    entry = tables["entry_policy_comparison"]
    filt = tables["filter_ablation"]
    matched = tables["vol_matched_baseline"]
    clean = tables.get("clean_hit_path_stats", pd.DataFrame())

    lines = ["# Crypto Pressure Graph v0.1 Candidate List", ""]
    lines.append("## Entry Policies")
    if entry.empty:
        lines.append("- No entry policy rows.")
    else:
        scored = entry if "clean_path_score" in entry.columns else add_clean_path_scores(entry, clean)
        best = scored[scored["cost_single_side_bps"] == 5].sort_values(
            ["clean_path_score", "net_expectancy"], ascending=False
        )
        for row in best.itertuples(index=False):
            lines.append(
                f"- {row.path_name} {row.entry_policy} / {row.execution_rule}: "
                f"score={row.clean_path_score:.4f}, trades={row.trade_n}, "
                f"net={row.net_expectancy:.4%}, win={row.win_rate:.2%}, "
                f"tp={row.tp_rate:.2%}, sl={row.sl_rate:.2%}"
            )
            if len(lines) >= 17:
                break
    lines.append("")
    lines.append("## Filters")
    if not filt.empty:
        best_filters = filt[filt["filter_name"] != "control"].sort_values(
            ["dd_delta", "clean_hit_delta"], ascending=[True, False]
        )
        for row in best_filters.groupby("path_name").head(5).itertuples(index=False):
            lines.append(
                f"- {row.path_name} {row.filter_name}: n={row.sample_n}, "
                f"clean_delta={row.clean_hit_delta:.2%}, dd_delta={row.dd_delta:.2%}"
            )
    lines.append("")
    lines.append("## Volatility Matched Baseline")
    if not matched.empty:
        for row in matched.itertuples(index=False):
            lines.append(
                f"- {row.path_name}: hit_lift={row.hit_3pct_lift_vs_vol_matched_baseline:.2%}, "
                f"clean_lift={row.clean_hit_lift_vs_vol_matched_baseline:.2%}, "
                f"dd_delta={row.dd_delta_vs_vol_matched_baseline:.2%}"
            )
    (report_root / "candidate_list.md").write_text("\n".join(lines), encoding="utf-8")


def add_clean_path_scores(entry: pd.DataFrame, clean_stats: pd.DataFrame) -> pd.DataFrame:
    if entry.empty or clean_stats.empty:
        out = entry.copy()
        out["clean_path_score"] = np.nan
        return out
    metric_cols = [
        "clean_hit_3pct_4h",
        "hit_5pct_4h",
        "median_max_up_4h",
        "dd_2pct_before_hit_3pct",
        "median_max_down_4h",
        "max_symbol_contribution",
        "max_month_contribution",
        "clean_path_score",
    ]
    entry = entry.drop(columns=[col for col in metric_cols if col in entry.columns])
    path_metrics = clean_stats[
        [
            "path_name",
            "clean_hit_3pct_4h",
            "hit_5pct_4h",
            "median_max_up_4h",
            "dd_2pct_before_hit_3pct",
            "median_max_down_4h",
            "max_symbol_contribution",
            "max_month_contribution",
        ]
    ].copy()
    out = entry.merge(path_metrics, on="path_name", how="left")
    concentration = out[["max_symbol_contribution", "max_month_contribution"]].max(axis=1)
    concentration_penalty = (concentration - 0.35).clip(lower=0).fillna(0)
    out["clean_path_score"] = (
        0.35 * out["clean_hit_3pct_4h"].fillna(0)
        + 0.20 * out["hit_5pct_4h"].fillna(0)
        + 0.20 * out["median_max_up_4h"].fillna(0)
        - 0.35 * out["dd_2pct_before_hit_3pct"].fillna(0)
        - 0.25 * out["sl_first_rate"].fillna(out["sl_rate"]).fillna(0)
        - 0.20 * out["median_max_down_4h"].abs().fillna(0)
        + 0.30 * out["net_expectancy"].fillna(0)
        - 0.15 * concentration_penalty
    )
    return out


def write_v01_reports(
    df: pd.DataFrame,
    config: ExperimentConfig,
    universe_col: str = "universe_dynamic_monthly_top30",
) -> dict[str, Path]:
    report_root = ensure_dir(Path("reports/v0_1"))
    prepared = prepare_v01_dataset(df, config)
    _, holdout = development_holdout_split(prepared, config.validation.final_holdout_months)

    tables: dict[str, pd.DataFrame] = {
        "drawdown_attribution": drawdown_attribution(holdout, universe_col),
        "first_touch_stats": first_touch_stats(holdout, universe_col),
        "entry_policy_comparison": entry_policy_comparison(holdout, config, universe_col),
        "filter_ablation": filter_ablation(holdout, universe_col),
        "vol_matched_baseline": volatility_matched_baseline(holdout, universe_col),
        "veto_overlay": veto_overlay(holdout, universe_col),
        "clean_hit_path_stats": clean_hit_path_stats(holdout, universe_col),
    }
    tables["entry_policy_comparison"] = add_clean_path_scores(
        tables["entry_policy_comparison"], tables["clean_hit_path_stats"]
    )
    outputs = {}
    for name, table in tables.items():
        path = report_root / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path
    write_v01_candidate_list(report_root, tables)
    outputs["candidate_list"] = report_root / "candidate_list.md"
    return outputs
