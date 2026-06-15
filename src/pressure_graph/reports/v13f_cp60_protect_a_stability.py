"""v1.3F CP60 Protect_A burst/month stability audit.

This report keeps CP60_all unchanged and stress-tests whether the beta-high
Protect_A research candidate is overly dependent on one protected burst or
one protected month. It is offline-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig
from pressure_graph.reports.v13d_cp60_context_protection import (
    PROTECTION_COST_BPS,
    _apply_protected_checkpoint,
    _cp60_exit_mask,
    _exit_class_for_protection_sample,
    _num,
    _portfolio_summary,
    _protection_masks,
    _simulate_core_max8,
)
from pressure_graph.reports.v13e_cp60_beta_protection_stability import (
    _cp60_sample,
    _portfolio_net,
    _prepare_sample_at_cost,
    _protect_a_sample,
    _selected_protected_exits,
    _simulate_max8_o6,
)


REPORT_ROOT = Path("reports/v1_3f_cp60_protect_a_stability")


@dataclass(frozen=True)
class V13FConfig:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()


def _prepare_sample(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    root: Path,
    cfg: V13FConfig,
) -> pd.DataFrame:
    return _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, PROTECTION_COST_BPS)


def _baseline_nets(sample: pd.DataFrame) -> dict[str, float]:
    cp_sample = _cp60_sample(sample)
    protect_sample = _protect_a_sample(sample)
    cp_ledger, cp_skipped = _simulate_core_max8(cp_sample, "CP60_all")
    protect_ledger, protect_skipped = _simulate_core_max8(protect_sample, "Protect_A_beta_high")
    cp_o6_ledger, _ = _simulate_max8_o6(cp_sample, "S3_CP60_O6")
    protect_o6_ledger, _ = _simulate_max8_o6(protect_sample, "S3_Protect_A_O6")
    return {
        "cp60_all_net20": float(_portfolio_summary("CP60_all", cp_ledger, cp_skipped)["portfolio_net20"]),
        "protect_a_net20": float(
            _portfolio_summary("Protect_A_beta_high", protect_ledger, protect_skipped)["portfolio_net20"]
        ),
        "s3_cp60_o6_net20": _portfolio_net(cp_o6_ledger),
        "s3_protect_a_o6_net20": _portfolio_net(protect_o6_ledger),
    }


def _full_protect_ledger(sample: pd.DataFrame) -> pd.DataFrame:
    protect_sample = _protect_a_sample(sample)
    protect_ledger, _ = _simulate_core_max8(protect_sample, "Protect_A_beta_high")
    protected = _selected_protected_exits(protect_ledger)
    if protected.empty:
        return protected
    if "month" not in protected.columns:
        protected["month"] = pd.to_datetime(protected["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    return protected


def _simulate_mask(sample: pd.DataFrame, protect_mask: pd.Series, rule_name: str) -> dict[str, object]:
    local = _apply_protected_checkpoint(sample, protect_mask, rule_name)
    ledger, skipped = _simulate_core_max8(local, rule_name)
    o6_ledger, o6_skipped = _simulate_max8_o6(local, f"S3_{rule_name}_O6")
    protected = _selected_protected_exits(ledger)
    protected_o6 = _selected_protected_exits(o6_ledger)
    summary = _portfolio_summary(rule_name, ledger, skipped)
    return {
        "net20": float(summary["portfolio_net20"]),
        "selected_trades": int(summary["selected_trades"]),
        "skipped_trades": int(summary["skipped_trades"]),
        "protected_exits": int(len(protected)),
        "o6_net20": _portfolio_net(o6_ledger),
        "o6_selected_trades": int(len(o6_ledger)),
        "o6_skipped_trades": int(len(o6_skipped)),
        "o6_protected_exits": int(len(protected_o6)),
    }


def _protect_a_mask(sample: pd.DataFrame) -> pd.Series:
    return _protection_masks(sample)["Protect_A_beta_high"].reindex(sample.index).fillna(False).astype(bool)


def _remove_protected_keys(sample: pd.DataFrame, protected_keys: set[str]) -> pd.Series:
    mask = _protect_a_mask(sample)
    if not protected_keys:
        return mask
    return mask & ~sample["trade_key"].astype(str).isin(protected_keys)


def _leave_one_burst_out(sample: pd.DataFrame) -> pd.DataFrame:
    baselines = _baseline_nets(sample)
    protected = _full_protect_ledger(sample)
    if protected.empty or "burst_id" not in protected.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for burst_id, group in protected.groupby("burst_id", sort=False, dropna=False):
        keys = set(group["trade_key"].astype(str))
        result = _simulate_mask(sample, _remove_protected_keys(sample, keys), "Protect_A_minus_burst")
        rows.append(
            {
                "removed_burst_id": burst_id,
                "removed_protected_exits": int(len(group)),
                "removed_symbols": ",".join(sorted(set(group["symbol"].astype(str)))),
                "removed_months": ",".join(sorted(set(group["month"].astype(str)))) if "month" in group.columns else "",
                "removed_delta_vs_cp60_sum": float(_num(group, "delta_vs_cp60").sum()),
                "remaining_net20": result["net20"],
                "delta_vs_CP60_all": float(result["net20"] - baselines["cp60_all_net20"]),
                "still_above_CP60_all": bool(result["net20"] > baselines["cp60_all_net20"]),
                "remaining_s3_o6_net20": result["o6_net20"],
                "delta_vs_S3_CP60_O6": float(result["o6_net20"] - baselines["s3_cp60_o6_net20"]),
                "still_above_S3_CP60_O6": bool(result["o6_net20"] > baselines["s3_cp60_o6_net20"]),
                "remaining_protected_exits": result["protected_exits"],
                "remaining_o6_protected_exits": result["o6_protected_exits"],
            }
        )
    return pd.DataFrame(rows)


def _leave_one_month_out(sample: pd.DataFrame) -> pd.DataFrame:
    baselines = _baseline_nets(sample)
    protected = _full_protect_ledger(sample)
    if protected.empty:
        return pd.DataFrame()
    if "month" not in protected.columns:
        protected["month"] = pd.to_datetime(protected["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    rows: list[dict[str, object]] = []
    for month, group in protected.groupby("month", sort=True, dropna=False):
        keys = set(group["trade_key"].astype(str))
        result = _simulate_mask(sample, _remove_protected_keys(sample, keys), "Protect_A_minus_month")
        rows.append(
            {
                "removed_month": month,
                "removed_protected_exits": int(len(group)),
                "removed_bursts": int(group["burst_id"].nunique()) if "burst_id" in group.columns else np.nan,
                "removed_symbols": ",".join(sorted(set(group["symbol"].astype(str)))),
                "removed_delta_vs_cp60_sum": float(_num(group, "delta_vs_cp60").sum()),
                "remaining_net20": result["net20"],
                "delta_vs_CP60_all": float(result["net20"] - baselines["cp60_all_net20"]),
                "still_above_CP60_all": bool(result["net20"] > baselines["cp60_all_net20"]),
                "remaining_s3_o6_net20": result["o6_net20"],
                "delta_vs_S3_CP60_O6": float(result["o6_net20"] - baselines["s3_cp60_o6_net20"]),
                "still_above_S3_CP60_O6": bool(result["o6_net20"] > baselines["s3_cp60_o6_net20"]),
                "remaining_protected_exits": result["protected_exits"],
                "remaining_o6_protected_exits": result["o6_protected_exits"],
            }
        )
    return pd.DataFrame(rows)


def _cap_mask_by_burst(sample: pd.DataFrame, cap: int | None) -> pd.Series:
    mask = _protect_a_mask(sample) & _cp60_exit_mask(sample)
    if cap is None:
        return mask
    allowed = pd.Series(False, index=sample.index)
    candidates = sample[mask].copy()
    if candidates.empty:
        return allowed
    candidates["_checkpoint_sort"] = pd.to_datetime(candidates["checkpoint_time"], utc=True, errors="coerce")
    candidates["_entry_sort"] = pd.to_datetime(candidates["entry_time"], utc=True, errors="coerce")
    sort_cols = ["burst_id", "_checkpoint_sort", "_entry_sort", "symbol"] if "burst_id" in candidates.columns else ["_checkpoint_sort", "_entry_sort", "symbol"]
    candidates = candidates.sort_values(sort_cols)
    group_key = "burst_id" if "burst_id" in candidates.columns else pd.Series("unknown", index=candidates.index)
    keep_idx = candidates.groupby(group_key, sort=False, dropna=False).head(cap).index
    allowed.loc[keep_idx] = True
    return allowed


def _selected_exit_class_counts(sample: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, int]:
    if ledger.empty:
        return {"false_exits_saved": 0, "true_good_exits_lost": 0, "neutral_exits_saved": 0}
    protected = _selected_protected_exits(ledger)
    if protected.empty:
        return {"false_exits_saved": 0, "true_good_exits_lost": 0, "neutral_exits_saved": 0}
    cp_sample = _apply_protected_checkpoint(sample, pd.Series(False, index=sample.index), "CP60_all")
    classes = _exit_class_for_protection_sample(cp_sample, 0.001).set_index("trade_key")
    keys = protected["trade_key"].astype(str)
    local = classes.loc[classes.index.intersection(keys)] if not classes.empty else pd.DataFrame()
    if local.empty or "exit_class" not in local.columns:
        return {"false_exits_saved": 0, "true_good_exits_lost": 0, "neutral_exits_saved": 0}
    cls = local["exit_class"].astype(str)
    return {
        "false_exits_saved": int(cls.eq("false_exit").sum()),
        "true_good_exits_lost": int(cls.eq("true_good_exit").sum()),
        "neutral_exits_saved": int(cls.eq("neutral_exit").sum()),
    }


def _protected_exit_cap_summary(sample: pd.DataFrame) -> pd.DataFrame:
    baselines = _baseline_nets(sample)
    rows: list[dict[str, object]] = []
    specs = [("CP60_all", 0), ("Protect_A_cap1_per_burst", 1), ("Protect_A_cap2_per_burst", 2), ("Protect_A_uncapped", None)]
    for rule_name, cap in specs:
        mask = pd.Series(False, index=sample.index) if cap == 0 else _cap_mask_by_burst(sample, cap)
        local = _apply_protected_checkpoint(sample, mask, rule_name)
        ledger, skipped = _simulate_core_max8(local, rule_name)
        o6_ledger, o6_skipped = _simulate_max8_o6(local, f"S3_{rule_name}_O6")
        summary = _portfolio_summary(rule_name, ledger, skipped)
        protected = _selected_protected_exits(ledger)
        protected_burst_counts = protected.groupby("burst_id", sort=False).size() if not protected.empty and "burst_id" in protected.columns else pd.Series(dtype=int)
        class_counts = _selected_exit_class_counts(sample, ledger)
        rows.append(
            {
                "rule": rule_name,
                "protect_cap_per_burst": "uncapped" if cap is None else int(cap),
                "net20": float(summary["portfolio_net20"]),
                "delta_vs_CP60_all": float(summary["portfolio_net20"] - baselines["cp60_all_net20"]),
                "selected_trades": int(summary["selected_trades"]),
                "skipped_trades": int(summary["skipped_trades"]),
                "protected_exits": int(len(protected)),
                "protected_bursts": int(protected_burst_counts.size),
                "max_protected_exits_per_burst": int(protected_burst_counts.max()) if len(protected_burst_counts) else 0,
                "false_exits_saved": class_counts["false_exits_saved"],
                "true_good_exits_lost": class_counts["true_good_exits_lost"],
                "neutral_exits_saved": class_counts["neutral_exits_saved"],
                "s3_o6_net20": _portfolio_net(o6_ledger),
                "delta_vs_S3_CP60_O6": float(_portfolio_net(o6_ledger) - baselines["s3_cp60_o6_net20"]),
                "s3_o6_selected_trades": int(len(o6_ledger)),
                "s3_o6_skipped_trades": int(len(o6_skipped)),
            }
        )
    return pd.DataFrame(rows)


def _write_notes(root: Path, burst: pd.DataFrame, month: pd.DataFrame, cap: pd.DataFrame) -> None:
    lines = [
        "# v1.3F CP60 Protect_A Burst/Month Stability",
        "",
        "Purpose: test whether Protect_A beta-high protection survives burst/month concentration checks.",
        "Status: offline diagnostic only. CP60_all live shadow remains unchanged.",
        "",
        "## Leave-One-Burst-Out",
    ]
    if burst.empty:
        lines.append("- No protected bursts.")
    else:
        lines.append(f"- protected_bursts={len(burst)}")
        lines.append(f"- still_above_CP60_all={int(burst['still_above_CP60_all'].sum())}/{len(burst)}")
        lines.append(f"- min_delta_vs_CP60_all={burst['delta_vs_CP60_all'].min():.4%}")
        lines.append(f"- min_delta_vs_S3_CP60_O6={burst['delta_vs_S3_CP60_O6'].min():.4%}")
    lines.extend(["", "## Leave-One-Month-Out"])
    if month.empty:
        lines.append("- No protected months.")
    else:
        lines.append(f"- protected_months={len(month)}")
        lines.append(f"- still_above_CP60_all={int(month['still_above_CP60_all'].sum())}/{len(month)}")
        lines.append(f"- min_delta_vs_CP60_all={month['delta_vs_CP60_all'].min():.4%}")
        lines.append(f"- min_delta_vs_S3_CP60_O6={month['delta_vs_S3_CP60_O6'].min():.4%}")
        ex_oct = month[month["removed_month"].astype(str).eq("2025-10")]
        if not ex_oct.empty:
            row = ex_oct.iloc[0]
            lines.append(
                f"- ex_2025_10_delta_vs_CP60_all={row['delta_vs_CP60_all']:.4%}; "
                f"ex_2025_10_delta_vs_S3_CP60_O6={row['delta_vs_S3_CP60_O6']:.4%}."
            )
    lines.extend(["", "## Burst Protection Caps"])
    if cap.empty:
        lines.append("- No cap rows.")
    else:
        for row in cap.itertuples(index=False):
            lines.append(
                f"- {row.rule}: net20={row.net20:.4%}, delta_vs_CP60={row.delta_vs_CP60_all:.4%}, "
                f"protected_exits={row.protected_exits}, max_per_burst={row.max_protected_exits_per_burst}."
            )
    lines.extend(
        [
            "",
            "Discipline: Protect_A can remain an offline research candidate, but it should not replace CP60_all "
            "unless burst/month stability and future live counterfactual evidence both hold.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v13f_cp60_protect_a_stability(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V13FConfig = V13FConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg)
    burst = _leave_one_burst_out(sample)
    month = _leave_one_month_out(sample)
    cap = _protected_exit_cap_summary(sample)
    outputs = {
        "leave_one_burst_out": root / "leave_one_burst_out.csv",
        "leave_one_month_out": root / "leave_one_month_out.csv",
        "protected_exit_cap_summary": root / "protected_exit_cap_summary.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    burst.to_csv(outputs["leave_one_burst_out"], index=False)
    month.to_csv(outputs["leave_one_month_out"], index=False)
    cap.to_csv(outputs["protected_exit_cap_summary"], index=False)
    _write_notes(root, burst, month, cap)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V13FConfig",
    "write_v13f_cp60_protect_a_stability",
]
