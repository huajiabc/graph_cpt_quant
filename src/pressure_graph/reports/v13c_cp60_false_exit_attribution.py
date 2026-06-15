"""v1.3C CP60 false-exit attribution.

This report explains the CP60 early exits found in v1.3A. It does not change
paper-live behavior and does not propose a new checkpoint rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig, _load_or_build_trades
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v13a_checkpoint_robustness import (
    _checkpoint_o6_integration,
    _load_price_frame,
    _pool_base20,
)


REPORT_ROOT = Path("reports/v1_3c_cp60_false_exit_attribution")
NEUTRAL_DELTA = 0.001
NUMERIC_FEATURES = (
    "checkpoint_net",
    "checkpoint_gross",
    "mfe_so_far",
    "mae_so_far",
    "price_position_vs_entry",
    "post_entry_1h_high",
    "post_entry_1h_low",
    "market_impulse_density",
    "beta_extreme_strength",
    "local_volume_shock_strength",
    "cluster_density",
)
CATEGORICAL_FEATURES = ("cic_type", "btc_state")


@dataclass(frozen=True)
class V13CConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    neutral_delta: float = NEUTRAL_DELTA


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def _first_numeric(frame: pd.DataFrame, cols: tuple[str, ...], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="float64")
    for col in cols:
        if col not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), candidate)
    return out


def _first_text(frame: pd.DataFrame, cols: tuple[str, ...], default: str = "unknown") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="object")
    for col in cols:
        if col not in frame.columns:
            continue
        candidate = frame[col].astype("object")
        valid = candidate.notna() & candidate.astype(str).ne("")
        out = out.where(~out.eq(default) | ~valid, candidate.astype(str))
    return out.fillna(default).astype(str)


def _cic_type(candidate: object) -> str:
    text = str(candidate)
    if text.startswith("CIC1"):
        return "CIC1"
    if text.startswith("CIC2"):
        return "CIC2"
    return text or "unknown"


def _exit_class(delta: float, neutral_delta: float) -> str:
    if pd.isna(delta):
        return "unknown"
    if delta > neutral_delta:
        return "true_good_exit"
    if delta < -neutral_delta:
        return "false_exit"
    return "neutral_exit"


def _build_cp60_selected_ledger(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    root: Path,
    v10a: V10AConfig,
) -> pd.DataFrame:
    trades = _load_or_build_trades(feature_path, instruments, config, root, v10a)
    base = _add_asof_burst_phase(_pool_base20(trades), "1h")
    if base.empty:
        raise ValueError("No P2 CIC trades available for v1.3C false-exit attribution.")
    prices = _load_price_frame(feature_path, base)
    integration, ledger, _ = _checkpoint_o6_integration(base, prices)
    if integration.empty or ledger.empty:
        raise ValueError("No CP60 integration ledger available for v1.3C false-exit attribution.")
    return ledger[ledger["portfolio_id"].eq("S1_P2_MAX8_CHECKPOINT_60M")].copy()


def _classify_cp60_exits(ledger: pd.DataFrame, neutral_delta: float = NEUTRAL_DELTA) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    exits = ledger[ledger.get("checkpoint_early_exit", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)].copy()
    if exits.empty:
        return exits
    exits["cic_type"] = exits["candidate"].map(_cic_type)
    exits["checkpoint_net"] = _num(exits, "checkpoint_net_at_cost")
    exits["checkpoint_gross"] = _num(exits, "checkpoint_gross_return")
    exits["mfe_so_far"] = _num(exits, "checkpoint_mfe")
    exits["mae_so_far"] = _num(exits, "checkpoint_mae")
    exits["price_position_vs_entry"] = exits["checkpoint_gross"]
    exits["post_entry_1h_high"] = exits["mfe_so_far"]
    exits["post_entry_1h_low"] = exits["mae_so_far"]
    exits["market_impulse_density"] = _first_numeric(exits, ("volume_impulse_density", "rank_market_impulse_density"))
    exits["beta_extreme_strength"] = _first_numeric(
        exits,
        ("c2_beta_extension_score", "rank_beta_extreme_strength", "ret_4h_percentile"),
    )
    exits["local_volume_shock_strength"] = _first_numeric(
        exits,
        ("volume_z_1h", "volume_z_4h", "rank_local_volume_shock_strength"),
    )
    exits["cluster_density"] = _first_numeric(exits, ("cluster_impulse_density", "rank_cluster_impulse_density"))
    exits["btc_state"] = _first_text(exits, ("btc_state_at_entry", "btc_market_state", "btc_state"))
    exits["net_if_kept"] = _num(exits, "net_return_at_cost")
    exits["net_if_exited"] = _num(exits, "effective_net_return")
    exits["delta_exit_vs_keep"] = exits["net_if_exited"] - exits["net_if_kept"]
    exits["exit_class"] = exits["delta_exit_vs_keep"].map(lambda value: _exit_class(float(value), neutral_delta))
    exits["neutral_delta"] = float(neutral_delta)
    preferred = [
        "signal_id",
        "symbol",
        "candidate",
        "cic_type",
        "entry_time",
        "checkpoint_time",
        "exit_time",
        "checkpoint_net",
        "checkpoint_gross",
        "mfe_so_far",
        "mae_so_far",
        "price_position_vs_entry",
        "post_entry_1h_high",
        "post_entry_1h_low",
        "market_impulse_density",
        "beta_extreme_strength",
        "local_volume_shock_strength",
        "cluster_density",
        "btc_state",
        "net_if_kept",
        "net_if_exited",
        "delta_exit_vs_keep",
        "exit_class",
    ]
    cols = [col for col in preferred if col in exits.columns]
    rest = [col for col in exits.columns if col not in cols]
    return exits[cols + rest].sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def _summary_by_class(exits: pd.DataFrame) -> pd.DataFrame:
    if exits.empty:
        return pd.DataFrame()
    rows = []
    total = len(exits)
    for exit_class, group in exits.groupby("exit_class", sort=False, dropna=False):
        rows.append(
            {
                "exit_class": exit_class,
                "trades": int(len(group)),
                "trade_share": float(len(group) / total) if total else np.nan,
                "avg_delta_exit_vs_keep": float(_num(group, "delta_exit_vs_keep").mean()),
                "sum_delta_exit_vs_keep": float(_num(group, "delta_exit_vs_keep").sum()),
                "avg_checkpoint_net": float(_num(group, "checkpoint_net").mean()),
                "avg_mfe_so_far": float(_num(group, "mfe_so_far").mean()),
                "avg_mae_so_far": float(_num(group, "mae_so_far").mean()),
                "avg_market_impulse_density": float(_num(group, "market_impulse_density").mean()),
                "avg_beta_extreme_strength": float(_num(group, "beta_extreme_strength").mean()),
                "avg_local_volume_shock_strength": float(_num(group, "local_volume_shock_strength").mean()),
                "avg_cluster_density": float(_num(group, "cluster_density").mean()),
            }
        )
    return pd.DataFrame(rows)


def _feature_bucket_summary(exits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if exits.empty:
        return pd.DataFrame()
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(exits.get(feature), errors="coerce")
        valid = exits[values.notna()].copy()
        if valid.empty:
            continue
        unique = values.dropna().nunique()
        if unique < 2:
            continue
        buckets = min(5, int(unique))
        valid["bucket"] = pd.qcut(pd.to_numeric(valid[feature], errors="coerce"), q=buckets, duplicates="drop")
        for bucket, group in valid.groupby("bucket", sort=True, observed=False):
            classes = group["exit_class"].astype(str)
            rows.append(
                {
                    "feature": feature,
                    "bucket": str(bucket),
                    "trades": int(len(group)),
                    "feature_min": float(pd.to_numeric(group[feature], errors="coerce").min()),
                    "feature_max": float(pd.to_numeric(group[feature], errors="coerce").max()),
                    "true_good_exit_rate": float(classes.eq("true_good_exit").mean()),
                    "false_exit_rate": float(classes.eq("false_exit").mean()),
                    "neutral_exit_rate": float(classes.eq("neutral_exit").mean()),
                    "avg_delta_exit_vs_keep": float(_num(group, "delta_exit_vs_keep").mean()),
                    "sum_delta_exit_vs_keep": float(_num(group, "delta_exit_vs_keep").sum()),
                }
            )
    return pd.DataFrame(rows)


def _categorical_summary(exits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if exits.empty:
        return pd.DataFrame()
    for feature in CATEGORICAL_FEATURES:
        if feature not in exits.columns:
            continue
        for value, group in exits.groupby(feature, sort=False, dropna=False):
            classes = group["exit_class"].astype(str)
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "trades": int(len(group)),
                    "true_good_exit_rate": float(classes.eq("true_good_exit").mean()),
                    "false_exit_rate": float(classes.eq("false_exit").mean()),
                    "neutral_exit_rate": float(classes.eq("neutral_exit").mean()),
                    "avg_delta_exit_vs_keep": float(_num(group, "delta_exit_vs_keep").mean()),
                    "sum_delta_exit_vs_keep": float(_num(group, "delta_exit_vs_keep").sum()),
                    "avg_checkpoint_net": float(_num(group, "checkpoint_net").mean()),
                    "avg_mfe_so_far": float(_num(group, "mfe_so_far").mean()),
                    "avg_mae_so_far": float(_num(group, "mae_so_far").mean()),
                }
            )
    return pd.DataFrame(rows)


def _feature_class_contrast(exits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if exits.empty:
        return pd.DataFrame()
    for feature in NUMERIC_FEATURES:
        row: dict[str, object] = {"feature": feature}
        for exit_class, group in exits.groupby("exit_class", sort=False, dropna=False):
            row[f"{exit_class}_mean"] = float(pd.to_numeric(group.get(feature), errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_notes(root: Path, summary: pd.DataFrame, categorical: pd.DataFrame) -> None:
    lines = [
        "# v1.3C CP60 False-Exit Attribution",
        "",
        "Purpose: explain CP60 early exits. This report is diagnostic only and does not change live rules.",
        f"Neutral band: abs(exit_vs_keep_delta) <= {NEUTRAL_DELTA:.2%}.",
        "",
        "## Exit Class Summary",
    ]
    if summary.empty:
        lines.append("- No CP60 exits.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.exit_class}: trades={row.trades}, share={row.trade_share:.2%}, "
                f"avg_delta={row.avg_delta_exit_vs_keep:.4%}, sum_delta={row.sum_delta_exit_vs_keep:.4%}."
            )
    cic = categorical[categorical["feature"].astype(str).eq("cic_type")] if not categorical.empty else pd.DataFrame()
    if not cic.empty:
        lines.extend(["", "## CIC Type"])
        for row in cic.itertuples(index=False):
            lines.append(
                f"- {row.value}: trades={row.trades}, true={row.true_good_exit_rate:.2%}, "
                f"false={row.false_exit_rate:.2%}, avg_delta={row.avg_delta_exit_vs_keep:.4%}."
            )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Do not promote CP60_v2 from this report alone.",
            "- Use this to identify whether a small number of as-of features can separate false exits from true weak exits.",
            "- Live CP60 remains shadow-only.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v13c_cp60_false_exit_attribution(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V13CConfig = V13CConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    ledger = _build_cp60_selected_ledger(feature_path, instruments, config, root, cfg.v10a)
    exits = _classify_cp60_exits(ledger, cfg.neutral_delta)
    summary = _summary_by_class(exits)
    buckets = _feature_bucket_summary(exits)
    categorical = _categorical_summary(exits)
    contrast = _feature_class_contrast(exits)
    outputs = {
        "cp60_exit_classification": root / "cp60_exit_classification.csv",
        "exit_class_summary": root / "exit_class_summary.csv",
        "feature_bucket_summary": root / "feature_bucket_summary.csv",
        "categorical_summary": root / "categorical_summary.csv",
        "feature_class_contrast": root / "feature_class_contrast.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    exits.to_csv(outputs["cp60_exit_classification"], index=False)
    summary.to_csv(outputs["exit_class_summary"], index=False)
    buckets.to_csv(outputs["feature_bucket_summary"], index=False)
    categorical.to_csv(outputs["categorical_summary"], index=False)
    contrast.to_csv(outputs["feature_class_contrast"], index=False)
    _write_notes(root, summary, categorical)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V13CConfig",
    "write_v13c_cp60_false_exit_attribution",
]
