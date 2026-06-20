"""v4.3 A7 Lagging Pocket Autopsy.

The v4.2 A7 pocket was very strong but tiny.  This report audits whether it is
a real cross-exchange lag pocket or an artefact from a few extreme events,
symbols, months, or lag-definition choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v40_cross_exchange_lead_lag import (
    DATA_ROOT,
    SOURCE_EXCHANGE,
    TARGET_EXCHANGE,
    _bool,
    _month_cap35,
    _net_at_cost,
    _num,
    _select_common_symbols,
)
from pressure_graph.reports.v41_cross_exchange_lead_lag_diagnostics import _read_1m_symbol
from pressure_graph.reports.v42_source_attribution_target_fusion import (
    SOURCE_1M_DATASET,
    TARGET_1M_DATASET,
    _add_1m_target_features,
    _aligned_signal,
    _decorate_source,
    _mapped_symbol,
    _source_dir,
    _source_matrices,
    _target_dir,
)


REPORT_ROOT = Path("reports/v4_3_a7_lagging_pocket_autopsy")


@dataclass(frozen=True)
class V43Config:
    report_root: Path = REPORT_ROOT
    data_root: Path = DATA_ROOT
    source_exchange: str = SOURCE_EXCHANGE
    target_exchange: str = TARGET_EXCHANGE
    source_dataset_1m: str = SOURCE_1M_DATASET
    target_dataset_1m: str = TARGET_1M_DATASET
    top_n: int = 50
    horizon_minutes: int = 60
    lag_threshold: float = 0.003
    sensitivity_thresholds: tuple[float, ...] = (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075)


def _with_raw_target_features(raw: pd.DataFrame, cfg: V43Config) -> pd.DataFrame:
    features = _add_1m_target_features(raw, cfg.horizon_minutes)
    if raw.empty or features.empty:
        return pd.DataFrame()
    keep = raw[["bar_close_time", "open", "high", "low", "close", "volume", "turnover"]].copy()
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")
    return features.merge(keep, on="bar_close_time", how="left")


def _source_ret_for_symbol(ret5_matrix: pd.DataFrame, symbol: str | None, times: pd.Series, index: pd.Index) -> pd.Series:
    if symbol is None or ret5_matrix.empty or symbol not in ret5_matrix.columns:
        return pd.Series(np.nan, index=index)
    return pd.Series(
        ret5_matrix[symbol].reindex(pd.to_datetime(times, utc=True, errors="coerce")).to_numpy(),
        index=index,
    )


def _first_touch_one(raw: pd.DataFrame, decision_time: pd.Timestamp, entry_price: float, threshold: float, horizon: int) -> dict[str, Any]:
    future = raw[pd.to_datetime(raw["bar_close_time"], utc=True, errors="coerce").gt(decision_time)].head(int(horizon))
    if future.empty or not np.isfinite(entry_price) or entry_price <= 0:
        return {
            "hit_any": False,
            "adverse_any": False,
            "first_touch": "no_path",
            "hit_minutes": np.nan,
            "adverse_minutes": np.nan,
        }
    up = entry_price * (1.0 + threshold)
    down = entry_price * (1.0 - threshold)
    hit_rows = future[pd.to_numeric(future["high"], errors="coerce").ge(up)]
    adverse_rows = future[pd.to_numeric(future["low"], errors="coerce").le(down)]
    hit_time = pd.NaT if hit_rows.empty else pd.Timestamp(hit_rows.iloc[0]["bar_close_time"])
    adverse_time = pd.NaT if adverse_rows.empty else pd.Timestamp(adverse_rows.iloc[0]["bar_close_time"])
    if pd.isna(hit_time) and pd.isna(adverse_time):
        first = "none"
    elif pd.isna(adverse_time) or hit_time < adverse_time:
        first = "hit_first"
    elif pd.isna(hit_time) or adverse_time < hit_time:
        first = "adverse_first"
    else:
        first = "same_bar"
    return {
        "hit_any": not pd.isna(hit_time),
        "adverse_any": not pd.isna(adverse_time),
        "first_touch": first,
        "hit_minutes": (hit_time - decision_time).total_seconds() / 60.0 if not pd.isna(hit_time) else np.nan,
        "adverse_minutes": (adverse_time - decision_time).total_seconds() / 60.0 if not pd.isna(adverse_time) else np.nan,
    }


def _build_lag_ledger(
    symbols: list[str],
    cfg: V43Config,
    *,
    lag_threshold: float,
    matrices: dict[str, pd.DataFrame] | None = None,
    density: pd.Series | None = None,
) -> pd.DataFrame:
    if matrices is None or density is None:
        _, matrices, density = _source_matrices(symbols, cfg)  # type: ignore[arg-type]
    impulse_matrix = matrices["impulse"]
    ret5_matrix = matrices["ret5"]
    if impulse_matrix.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for symbol in symbols:
        target_path = _target_dir(cfg) / f"{symbol}.parquet"  # type: ignore[arg-type]
        if not target_path.exists() or symbol not in impulse_matrix.columns:
            continue
        raw = _read_1m_symbol(target_path, cfg.target_exchange, symbol)
        target = _with_raw_target_features(raw, cfg)
        if target.empty:
            continue
        target = target.drop_duplicates("bar_close_time").sort_values("bar_close_time")
        controls = {
            "real": symbol,
            "random_cyclic": _mapped_symbol(symbols, symbol, "cyclic"),
            "shuffled_reverse": _mapped_symbol(symbols, symbol, "reverse"),
        }
        for control_type, source_symbol in controls.items():
            source_impulse = _aligned_signal(impulse_matrix, source_symbol, target["bar_close_time"])
            source_ret_5m = _source_ret_for_symbol(ret5_matrix, source_symbol, target["bar_close_time"], target.index)
            local = target.copy()
            local["symbol"] = symbol
            local["source_symbol"] = source_symbol
            local["control_type"] = control_type
            local["source_impulse"] = source_impulse
            local["source_ret_5m"] = source_ret_5m
            local["source_target_ret_5m_gap"] = _num(local, "source_ret_5m") - _num(local, "target_ret_5m")
            local["target_lagging"] = _num(local, "source_target_ret_5m_gap").ge(lag_threshold) & _num(
                local, "target_ret_5m"
            ).lt(_num(local, "source_ret_5m"))
            mask = _bool(local, "source_impulse") & _bool(local, "target_lagging") & _bool(local, "target_reclaim_proxy")
            subset = local[mask.fillna(False)].copy()
            if subset.empty:
                continue
            subset["decision_time"] = pd.to_datetime(subset["bar_close_time"], utc=True, errors="coerce")
            subset["month"] = subset["decision_time"].dt.strftime("%Y-%m")
            subset["market_impulse_density"] = density.reindex(subset["decision_time"]).to_numpy()
            ret_col = f"target_future_ret_{cfg.horizon_minutes}m"
            mfe_col = f"target_future_mfe_{cfg.horizon_minutes}m"
            mae_col = f"target_future_mae_{cfg.horizon_minutes}m"
            subset["gross_return"] = _num(subset, ret_col)
            subset["future_mfe"] = _num(subset, mfe_col)
            subset["future_mae"] = _num(subset, mae_col)
            for cost in [10.0, 20.0, 30.0, 50.0]:
                subset[f"net{int(cost)}"] = _net_at_cost(subset, ret_col, cost)
            subset["lag_threshold"] = float(lag_threshold)
            rows.append(subset)
    if not rows:
        return pd.DataFrame()
    ledger = pd.concat(rows, ignore_index=True)
    cols = [
        "control_type",
        "symbol",
        "source_symbol",
        "bar_close_time",
        "decision_time",
        "month",
        "lag_threshold",
        "source_ret_5m",
        "target_ret_5m",
        "source_target_ret_5m_gap",
        "target_reclaim_proxy",
        "target_volume_z",
        "market_impulse_density",
        "close",
        "gross_return",
        "net10",
        "net20",
        "net30",
        "net50",
        "future_mfe",
        "future_mae",
    ]
    return ledger[[col for col in cols if col in ledger.columns]].dropna(subset=["gross_return"])


def _add_first_touch(ledger: pd.DataFrame, cfg: V43Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ledger.empty:
        return ledger.copy(), pd.DataFrame()
    out = ledger.copy()
    real_index = out.index[out["control_type"].astype(str).eq("real")]
    real_source = out.loc[real_index].copy()
    agg_rows = []
    for threshold in [0.005, 0.01, 0.02]:
        touch_rows = []
        for symbol, group in real_source.groupby("symbol", sort=False):
            raw = _read_1m_symbol(_target_dir(cfg) / f"{symbol}.parquet", cfg.target_exchange, symbol)  # type: ignore[arg-type]
            if raw.empty:
                continue
            for idx, row in group.iterrows():
                touch = _first_touch_one(
                    raw,
                    pd.Timestamp(row["decision_time"]),
                    float(row["close"]),
                    threshold,
                    cfg.horizon_minutes,
                )
                touch["index"] = idx
                touch_rows.append(touch)
        touch_frame = pd.DataFrame(touch_rows).set_index("index") if touch_rows else pd.DataFrame()
        if not touch_frame.empty:
            label = str(threshold).replace(".", "p")
            out[f"first_touch_{label}"] = touch_frame["first_touch"]
            out[f"hit_any_{label}"] = touch_frame["hit_any"]
            out[f"adverse_any_{label}"] = touch_frame["adverse_any"]
            out[f"hit_minutes_{label}"] = touch_frame["hit_minutes"]
            out[f"adverse_minutes_{label}"] = touch_frame["adverse_minutes"]
            real = out.loc[real_index]
            values = real[f"first_touch_{label}"].astype(str)
            agg_rows.append(
                {
                    "threshold": threshold,
                    "events": int(len(real)),
                    "hit_first_rate": float(values.eq("hit_first").mean()) if len(real) else np.nan,
                    "adverse_first_rate": float(values.eq("adverse_first").mean()) if len(real) else np.nan,
                    "same_bar_rate": float(values.eq("same_bar").mean()) if len(real) else np.nan,
                    "hit_any_rate": float(real[f"hit_any_{label}"].astype(bool).mean()) if len(real) else np.nan,
                    "adverse_any_rate": float(real[f"adverse_any_{label}"].astype(bool).mean()) if len(real) else np.nan,
                    "median_hit_minutes": float(pd.to_numeric(real[f"hit_minutes_{label}"], errors="coerce").median()),
                    "median_adverse_minutes": float(pd.to_numeric(real[f"adverse_minutes_{label}"], errors="coerce").median()),
                }
            )
    return out, pd.DataFrame(agg_rows)


def _group_contribution(ledger: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if ledger.empty or "control_type" not in ledger.columns:
        return pd.DataFrame()
    real = ledger[ledger["control_type"].eq("real")].copy()
    if real.empty:
        return pd.DataFrame()
    grouped = real.groupby(group_col, dropna=False).agg(
        events=("net20", "size"),
        net20_sum=("net20", "sum"),
        net20_mean=("net20", "mean"),
        net30_mean=("net30", "mean"),
        gross_mean=("gross_return", "mean"),
        mfe_mean=("future_mfe", "mean"),
        mae_mean=("future_mae", "mean"),
    )
    total = float(grouped["net20_sum"].sum())
    grouped["contribution_pct"] = grouped["net20_sum"] / total if total else np.nan
    return grouped.reset_index().sort_values("net20_sum", ascending=False)


def _top_trade_contribution(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "control_type" not in ledger.columns:
        return pd.DataFrame()
    real = ledger[ledger["control_type"].eq("real")].copy()
    if real.empty:
        return pd.DataFrame()
    total = float(real["net20"].sum())
    real["contribution_pct"] = real["net20"] / total if total else np.nan
    return real.sort_values("net20", ascending=False).head(25)


def _mfe_mae_distribution(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "control_type" not in ledger.columns:
        return pd.DataFrame()
    real = ledger[ledger["control_type"].eq("real")].copy()
    rows = []
    for col in ["gross_return", "net20", "net30", "future_mfe", "future_mae", "source_target_ret_5m_gap"]:
        values = pd.to_numeric(real.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if values.empty:
            continue
        row = {"metric": col, "events": int(len(values))}
        for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
            row[f"p{int(q * 100)}"] = float(values.quantile(q))
        row["mean"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _cost_stress(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "control_type" not in ledger.columns:
        return pd.DataFrame()
    real = ledger[ledger["control_type"].eq("real")].copy()
    rows = []
    for cost in [10, 20, 30, 50]:
        col = f"net{cost}"
        if col not in real.columns:
            continue
        symbol_sum = real.groupby("symbol", dropna=False)[col].sum()
        month_sum = real.groupby("month", dropna=False)[col].sum()
        total = float(symbol_sum.sum())
        rows.append(
            {
                "cost_bps": cost,
                "events": int(len(real)),
                "net": float(pd.to_numeric(real[col], errors="coerce").mean()) if len(real) else np.nan,
                "month_cap35": _month_cap35(real.rename(columns={col: "value"}), "value") if len(real) else np.nan,
                "max_symbol_contribution": float((symbol_sum / total).abs().max()) if total else np.nan,
                "max_month_contribution": float((month_sum / total).abs().max()) if total else np.nan,
                "hit_rate": float(pd.to_numeric(real[col], errors="coerce").gt(0).mean()) if len(real) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _random_shuffled_by_month(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "control_type" not in ledger.columns:
        return pd.DataFrame()
    rows = []
    for (month, control), group in ledger.groupby(["month", "control_type"], sort=True, dropna=False):
        rows.append(
            {
                "month": month,
                "control_type": control,
                "events": int(len(group)),
                "net20": float(pd.to_numeric(group["net20"], errors="coerce").mean()),
                "net30": float(pd.to_numeric(group["net30"], errors="coerce").mean()),
                "gross_return": float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    pivot = table.pivot(index="month", columns="control_type", values="net20")
    for control in ["random_cyclic", "shuffled_reverse"]:
        if "real" in pivot.columns and control in pivot.columns:
            pivot[f"real_vs_{control}_lift"] = pivot["real"] - pivot[control]
    return table.merge(pivot.reset_index(), on="month", how="left")


def _sensitivity(
    symbols: list[str],
    cfg: V43Config,
    *,
    matrices: dict[str, pd.DataFrame],
    density: pd.Series,
) -> pd.DataFrame:
    rows = []
    for threshold in cfg.sensitivity_thresholds:
        ledger = _build_lag_ledger(symbols, cfg, lag_threshold=threshold, matrices=matrices, density=density)
        if ledger.empty:
            continue
        for control, group in ledger.groupby("control_type", sort=False):
            rows.append(
                {
                    "lag_threshold": threshold,
                    "control_type": control,
                    "events": int(len(group)),
                    "net20": float(pd.to_numeric(group["net20"], errors="coerce").mean()),
                    "net30": float(pd.to_numeric(group["net30"], errors="coerce").mean()),
                    "month_cap35_net20": _month_cap35(group, "net20"),
                    "max_symbol_contribution": _max_group_contribution(group, "symbol", "net20"),
                    "max_month_contribution": _max_group_contribution(group, "month", "net20"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    real = out[out["control_type"].eq("real")][["lag_threshold", "net20"]].rename(columns={"net20": "real_net20"})
    random = out[out["control_type"].eq("random_cyclic")][["lag_threshold", "net20"]].rename(columns={"net20": "random_net20"})
    shuffled = out[out["control_type"].eq("shuffled_reverse")][["lag_threshold", "net20"]].rename(columns={"net20": "shuffled_net20"})
    lifts = real.merge(random, on="lag_threshold", how="left").merge(shuffled, on="lag_threshold", how="left")
    lifts["real_vs_random_lift"] = lifts["real_net20"] - lifts["random_net20"]
    lifts["real_vs_shuffled_lift"] = lifts["real_net20"] - lifts["shuffled_net20"]
    return out.merge(lifts, on="lag_threshold", how="left")


def _max_group_contribution(frame: pd.DataFrame, group_col: str, value_col: str) -> float:
    if frame.empty:
        return np.nan
    grouped = frame.groupby(group_col, dropna=False)[value_col].sum()
    total = float(grouped.sum())
    return float((grouped / total).abs().max()) if total else np.nan


def _write_notes(root: Path, ledger: pd.DataFrame, cost: pd.DataFrame, touch: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    real = ledger[ledger.get("control_type", pd.Series(dtype=str)).astype(str).eq("real")] if not ledger.empty else ledger
    lines = [
        "# v4.3 A7 Lagging Pocket Autopsy",
        "",
        "## Scope",
        "- Audit v4.2 A7: Binance source impulse + Bybit target lagging + Bybit reclaim on 1m data.",
        "- This is diagnostic only; it does not create a live selector.",
        "",
        "## Main Pocket",
    ]
    if real.empty:
        lines.append("- No A7 real events found.")
    else:
        max_trade = float((real["net20"] / real["net20"].sum()).abs().max()) if real["net20"].sum() else np.nan
        lines.append(f"- events={len(real)}, net20={real['net20'].mean():.4%}, net30={real['net30'].mean():.4%}, net50={real['net50'].mean():.4%}.")
        lines.append(f"- max_trade_contribution={max_trade:.2%}; max_symbol_contribution={_max_group_contribution(real, 'symbol', 'net20'):.2%}; max_month_contribution={_max_group_contribution(real, 'month', 'net20'):.2%}.")
    if not touch.empty:
        row = touch[touch["threshold"].eq(0.01)]
        if not row.empty:
            r = row.iloc[0]
            lines.append(f"- first-touch 1pct: hit_first={r.hit_first_rate:.2%}, adverse_first={r.adverse_first_rate:.2%}, adverse_any={r.adverse_any_rate:.2%}.")
    if not cost.empty:
        row50 = cost[cost["cost_bps"].eq(50)]
        if not row50.empty:
            lines.append(f"- 50bp stress net={row50.iloc[0].net:.4%}, month_cap35={row50.iloc[0].month_cap35:.4%}.")
    lines.append("")
    lines.append("## Decision")
    lines.append("- Require stability across symbol, month, first-touch, cost, and lag-threshold sensitivity before any shadow discussion.")
    if not sensitivity.empty:
        pivot = sensitivity[sensitivity["control_type"].eq("real")]
        positive = int(pd.to_numeric(pivot["net30"], errors="coerce").gt(0).sum())
        lines.append(f"- Lag sensitivity positive net30 thresholds: {positive}/{len(pivot)}.")
    lines.append("- If concentration or adverse-first remains high, keep A7 as right-tail diagnostic only.")
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v43_a7_lagging_pocket_autopsy(cfg: V43Config = V43Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    symbols, _ = _select_common_symbols(_source_dir(cfg), _target_dir(cfg), cfg.top_n)  # type: ignore[arg-type]
    _, matrices, density = _source_matrices(symbols, cfg)  # type: ignore[arg-type]
    ledger = _build_lag_ledger(symbols, cfg, lag_threshold=cfg.lag_threshold, matrices=matrices, density=density)
    ledger, first_touch = _add_first_touch(ledger, cfg)
    symbol_contrib = _group_contribution(ledger, "symbol")
    month_contrib = _group_contribution(ledger, "month")
    top_trade = _top_trade_contribution(ledger)
    mfe_mae = _mfe_mae_distribution(ledger)
    cost = _cost_stress(ledger)
    by_month = _random_shuffled_by_month(ledger)
    sensitivity = _sensitivity(symbols, cfg, matrices=matrices, density=density)

    outputs = {
        "a7_trade_ledger": root / "a7_trade_ledger.csv",
        "a7_symbol_contribution": root / "a7_symbol_contribution.csv",
        "a7_month_contribution": root / "a7_month_contribution.csv",
        "a7_top_trade_contribution": root / "a7_top_trade_contribution.csv",
        "a7_first_touch": root / "a7_first_touch.csv",
        "a7_mfe_mae_distribution": root / "a7_mfe_mae_distribution.csv",
        "a7_liquidity_cost_stress": root / "a7_liquidity_cost_stress.csv",
        "a7_random_shuffled_by_month": root / "a7_random_shuffled_by_month.csv",
        "a7_lag_definition_sensitivity": root / "a7_lag_definition_sensitivity.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    ledger.to_csv(outputs["a7_trade_ledger"], index=False)
    symbol_contrib.to_csv(outputs["a7_symbol_contribution"], index=False)
    month_contrib.to_csv(outputs["a7_month_contribution"], index=False)
    top_trade.to_csv(outputs["a7_top_trade_contribution"], index=False)
    first_touch.to_csv(outputs["a7_first_touch"], index=False)
    mfe_mae.to_csv(outputs["a7_mfe_mae_distribution"], index=False)
    cost.to_csv(outputs["a7_liquidity_cost_stress"], index=False)
    by_month.to_csv(outputs["a7_random_shuffled_by_month"], index=False)
    sensitivity.to_csv(outputs["a7_lag_definition_sensitivity"], index=False)
    _write_notes(root, ledger, cost, first_touch, sensitivity)
    return outputs


__all__ = ["V43Config", "write_v43_a7_lagging_pocket_autopsy"]
