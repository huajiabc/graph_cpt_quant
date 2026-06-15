"""v2.1A Holdout Autopsy.

This report explains why the current CIC basket architectures degrade in the
holdout window. It does not search, tune, or promote new rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import (
    VALIDATION_END,
    MotifSpec,
    V20Config,
    _metrics_for_sample,
    _period_sample,
    _prepare_sample,
    _simulate_portfolio,
    _spec_nodes,
)


REPORT_ROOT = Path("reports/v2_1a_holdout_autopsy")


@dataclass(frozen=True)
class V21AConfig:
    report_root: Path = REPORT_ROOT
    v20: V20Config = V20Config()


AUTOPSY_SPECS: dict[str, MotifSpec] = {
    "B0_P2_max8": MotifSpec(max_positions=8),
    "B3_P2_max8_CP60_O6": MotifSpec(
        max_positions=8,
        overflow_rule="O6_late9",
        checkpoint_rule="CP60",
    ),
    "B4_P2_max8_ProtectA_cap2_O6": MotifSpec(
        max_positions=8,
        overflow_rule="O6_late9",
        checkpoint_rule="Protect_A_cap2",
        protect_cap=2,
    ),
    "ACO_validation_best_max5_ProtectA_cap2_O6": MotifSpec(
        market_context="market_impulse_density_high",
        local_state="cic1_cic2",
        max_positions=5,
        overflow_rule="O6_late9",
        checkpoint_rule="Protect_A_cap2",
        protect_cap=2,
    ),
    "SA_search_best_max3_ProtectA_cap1_O6": MotifSpec(
        market_context="market_impulse_density_high",
        local_state="cic1_cic2",
        max_positions=3,
        overflow_rule="O6_late9",
        checkpoint_rule="Protect_A_cap1",
        protect_cap=1,
    ),
}


def _first_numeric(frame: pd.DataFrame, cols: tuple[str, ...], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="float64")
    for col in cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out


def _first_text(frame: pd.DataFrame, cols: tuple[str, ...], default: str = "unknown") -> pd.Series:
    out = pd.Series(default, index=frame.index, dtype="object")
    for col in cols:
        if col not in frame.columns:
            continue
        values = frame[col].astype("object")
        valid = values.notna() & values.astype(str).ne("")
        out = out.where(~out.eq(default) | ~valid, values.astype(str))
    return out.fillna(default).astype(str)


def _cic_type(value: object) -> str:
    text = str(value)
    if text.startswith("CIC1"):
        return "CIC1"
    if text.startswith("CIC2"):
        return "CIC2"
    return text or "unknown"


def _burst_phase(count: object) -> str:
    value = pd.to_numeric(pd.Series([count]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "unknown"
    if value <= 3:
        return "order_1_3"
    if value <= 8:
        return "order_4_8"
    if value <= 14:
        return "order_9_14"
    return "order_15_plus"


def _qbucket(values: pd.Series, prefix: str, bins: int = 3) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = [f"{prefix}_q{i + 1}" for i in range(bins)]
    out = pd.Series(f"{prefix}_unknown", index=values.index, dtype="object")
    valid = numeric[numeric.notna()]
    if valid.nunique() < 2:
        return out
    try:
        bucketed = pd.qcut(valid, q=min(bins, valid.nunique()), labels=labels[: min(bins, valid.nunique())], duplicates="drop")
    except ValueError:
        return out
    out.loc[bucketed.index] = bucketed.astype(str)
    return out


def _annotate_ledger(ledger: pd.DataFrame, denominator: int) -> pd.DataFrame:
    out = ledger.copy()
    if out.empty:
        return out
    out["denominator"] = int(denominator)
    out["portfolio_contribution_net20"] = pd.to_numeric(out["weighted_return"], errors="coerce") / max(1, denominator)
    out["trade_net20"] = pd.to_numeric(out.get("effective_net_return"), errors="coerce")
    out["cic_type"] = out.get("candidate", pd.Series("", index=out.index)).map(_cic_type)
    out["btc_state"] = _first_text(out, ("btc_state_at_entry", "btc_market_state", "btc_state"))
    out["market_impulse_density"] = _first_numeric(
        out,
        ("market_impulse_density", "volume_impulse_density", "rank_market_impulse_density"),
    )
    out["cluster_density"] = _first_numeric(out, ("cluster_impulse_density", "cluster_density", "rank_cluster_impulse_density"))
    out["beta_strength"] = _first_numeric(
        out,
        ("beta_extreme_strength", "c2_beta_extension_score", "rank_beta_extreme_strength", "ret_4h_percentile"),
    )
    out["local_shock_strength"] = _first_numeric(
        out,
        ("local_volume_shock_strength", "volume_z_1h", "volume_z_4h", "rank_local_volume_shock_strength"),
    )
    out["burst_phase"] = out.get("burst_count_so_far", pd.Series(np.nan, index=out.index)).map(_burst_phase)
    out["checkpoint_component"] = np.select(
        [
            out.get("checkpoint_early_exit", pd.Series(False, index=out.index)).fillna(False).astype(bool),
            out.get("kept_due_to_protection", pd.Series(False, index=out.index)).fillna(False).astype(bool),
            out.get("sleeve", pd.Series("", index=out.index)).astype(str).eq("overflow"),
        ],
        ["CP60_exit", "ProtectA_kept", "O6_overflow"],
        default="core_normal",
    )
    out["market_density_bucket"] = _qbucket(out["market_impulse_density"], "market")
    out["beta_bucket"] = _qbucket(out["beta_strength"], "beta")
    out["cluster_bucket"] = _qbucket(out["cluster_density"], "cluster")
    out["state_cluster"] = (
        out["btc_state"].astype(str)
        + "|"
        + out["market_density_bucket"].astype(str)
        + "|"
        + out["beta_bucket"].astype(str)
        + "|"
        + out["burst_phase"].astype(str)
        + "|"
        + out["cic_type"].astype(str)
    )
    return out


def _ledger_for_spec(sample: pd.DataFrame, candidate_id: str, spec: MotifSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger, skipped = _simulate_portfolio(sample, spec)
    if not ledger.empty:
        ledger = _annotate_ledger(ledger, spec.max_positions)
        ledger["candidate_id"] = candidate_id
        ledger["nodes"] = _spec_nodes(spec)
    if not skipped.empty:
        skipped = skipped.copy()
        skipped["candidate_id"] = candidate_id
        skipped["nodes"] = _spec_nodes(spec)
        skipped["candidate_status"] = "skipped"
        skipped["future_trade_net20"] = pd.to_numeric(skipped.get("effective_net_return"), errors="coerce")
        skipped["cic_type"] = skipped.get("candidate", pd.Series("", index=skipped.index)).map(_cic_type)
        skipped["burst_phase"] = skipped.get("burst_count_so_far", pd.Series(np.nan, index=skipped.index)).map(_burst_phase)
    return ledger, skipped


def _group_summary(ledger: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in ledger.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        returns = pd.to_numeric(group["portfolio_contribution_net20"], errors="coerce")
        trade_returns = pd.to_numeric(group["trade_net20"], errors="coerce")
        row.update(
            {
                "selected_trades": int(len(group)),
                "core_trades": int(group.get("sleeve", pd.Series("", index=group.index)).astype(str).eq("core").sum()),
                "overflow_trades": int(group.get("sleeve", pd.Series("", index=group.index)).astype(str).eq("overflow").sum()),
                "cp60_exits": int(group.get("checkpoint_early_exit", pd.Series(False, index=group.index)).fillna(False).astype(bool).sum()),
                "protected_exits": int(group.get("kept_due_to_protection", pd.Series(False, index=group.index)).fillna(False).astype(bool).sum()),
                "portfolio_net20": float(returns.sum()),
                "avg_trade_net20": float(trade_returns.mean()) if len(trade_returns) else np.nan,
                "hit_rate": float(trade_returns.gt(0).mean()) if len(trade_returns) else np.nan,
                "avg_market_impulse_density": float(pd.to_numeric(group.get("market_impulse_density"), errors="coerce").mean()),
                "avg_beta_strength": float(pd.to_numeric(group.get("beta_strength"), errors="coerce").mean()),
                "avg_cluster_density": float(pd.to_numeric(group.get("cluster_density"), errors="coerce").mean()),
                "avg_burst_count_so_far": float(pd.to_numeric(group.get("burst_count_so_far"), errors="coerce").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate_id", "portfolio_net20"], ascending=[True, True]).reset_index(drop=True)


def _selected_vs_skipped_summary(ledger: pd.DataFrame, skipped: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(set(ledger.get("candidate_id", pd.Series(dtype=str))).union(set(skipped.get("candidate_id", pd.Series(dtype=str))))):
        selected = ledger[ledger["candidate_id"].eq(candidate_id)] if not ledger.empty else pd.DataFrame()
        skipped_group = skipped[skipped["candidate_id"].eq(candidate_id)] if not skipped.empty else pd.DataFrame()
        rows.append(
            {
                "candidate_id": candidate_id,
                "selected_trades": int(len(selected)),
                "skipped_trades": int(len(skipped_group)),
                "selected_avg_net20": float(pd.to_numeric(selected.get("trade_net20"), errors="coerce").mean()) if len(selected) else np.nan,
                "skipped_avg_future_net20": float(pd.to_numeric(skipped_group.get("future_trade_net20"), errors="coerce").mean()) if len(skipped_group) else np.nan,
                "selected_minus_skipped_avg_net20": (
                    float(pd.to_numeric(selected.get("trade_net20"), errors="coerce").mean())
                    - float(pd.to_numeric(skipped_group.get("future_trade_net20"), errors="coerce").mean())
                    if len(selected) and len(skipped_group)
                    else np.nan
                ),
                "portfolio_full_skips": int(skipped_group.get("skip_reason", pd.Series("", index=skipped_group.index)).astype(str).str.contains("portfolio_full|overflow_full", regex=True).sum())
                if len(skipped_group)
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _period_summary(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, spec in AUTOPSY_SPECS.items():
        row: dict[str, Any] = {"candidate_id": candidate_id, "nodes": _spec_nodes(spec)}
        for period in ("search", "validation", "holdout", "full"):
            metrics = _metrics_for_sample(_period_sample(sample, period), spec)
            for key, value in metrics.items():
                row[f"{period}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _notes(root: Path, period_summary: pd.DataFrame, cp_o6: pd.DataFrame, selected_vs_skipped: pd.DataFrame) -> None:
    def fmt_pct(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "n/a" if pd.isna(numeric) else f"{numeric:.4%}"

    lines = [
        "# v2.1A Holdout Autopsy",
        "",
        "Status: offline diagnostic only. No paper-live or real-live rule changes.",
        f"Holdout period: entry_time >= {VALIDATION_END.isoformat()}.",
        "",
        "## Candidate Holdout",
    ]
    cols = ["candidate_id", "holdout_trades", "holdout_portfolio_net20", "holdout_worst_month_net20", "holdout_worst_burst_net20"]
    for row in period_summary[cols].itertuples(index=False):
        lines.append(
            f"- {row.candidate_id}: trades={row.holdout_trades}, net20={row.holdout_portfolio_net20:.4%}, "
            f"worst_month={row.holdout_worst_month_net20:.4%}, worst_burst={row.holdout_worst_burst_net20:.4%}."
        )
    b4 = period_summary.loc[period_summary["candidate_id"].eq("B4_P2_max8_ProtectA_cap2_O6")]
    if not b4.empty:
        b4_id = "B4_P2_max8_ProtectA_cap2_O6"
        component = cp_o6[cp_o6["candidate_id"].eq(b4_id)].copy()
        if not component.empty:
            lines.extend(["", "## B4 Component Damage"])
            for row in component.sort_values("portfolio_net20").itertuples(index=False):
                lines.append(
                    f"- {row.checkpoint_component}: trades={row.selected_trades}, net20={row.portfolio_net20:.4%}, "
                    f"avg_trade={row.avg_trade_net20:.4%}."
                )
        svs = selected_vs_skipped[selected_vs_skipped["candidate_id"].eq(b4_id)]
        if not svs.empty:
            row = svs.iloc[0]
            lines.extend(
                [
                    "",
                    "## B4 Selected vs Skipped",
                    (
                        f"- selected={int(row.selected_trades)}, skipped={int(row.skipped_trades)}, "
                        f"selected_avg={fmt_pct(row.selected_avg_net20)}, skipped_avg={fmt_pct(row.skipped_avg_future_net20)}."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "Interpretation: v2.1A only identifies where holdout damage concentrates. "
            "State discovery / router rules should be built only after these failure buckets are stable.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21a_holdout_autopsy(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21AConfig = V21AConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v20)
    holdout = _period_sample(sample, "holdout")
    ledgers: list[pd.DataFrame] = []
    skipped_rows: list[pd.DataFrame] = []
    for candidate_id, spec in AUTOPSY_SPECS.items():
        ledger, skipped = _ledger_for_spec(holdout, candidate_id, spec)
        ledgers.append(ledger)
        skipped_rows.append(skipped)
    non_empty_ledgers = [frame for frame in ledgers if not frame.empty]
    non_empty_skipped = [frame for frame in skipped_rows if not frame.empty]
    holdout_ledger = pd.concat(non_empty_ledgers, ignore_index=True) if non_empty_ledgers else pd.DataFrame()
    holdout_skipped = pd.concat(non_empty_skipped, ignore_index=True) if non_empty_skipped else pd.DataFrame()
    period_summary = _period_summary(sample)

    outputs = {
        "strategy_period_summary": root / "strategy_period_summary.csv",
        "holdout_trade_ledger": root / "holdout_trade_ledger.csv",
        "holdout_skipped_candidates": root / "holdout_skipped_candidates.csv",
        "holdout_selected_vs_skipped": root / "holdout_selected_vs_skipped.csv",
        "holdout_by_market_regime": root / "holdout_by_market_regime.csv",
        "holdout_by_burst_phase": root / "holdout_by_burst_phase.csv",
        "holdout_by_cic_type": root / "holdout_by_cic_type.csv",
        "holdout_by_symbol": root / "holdout_by_symbol.csv",
        "holdout_by_month": root / "holdout_by_month.csv",
        "holdout_by_cp60_o6": root / "holdout_by_cp60_o6.csv",
        "holdout_by_state_cluster": root / "holdout_by_state_cluster.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    selected_vs_skipped = _selected_vs_skipped_summary(holdout_ledger, holdout_skipped)
    by_market = _group_summary(holdout_ledger, ["candidate_id", "btc_state", "market_density_bucket"])
    by_burst = _group_summary(holdout_ledger, ["candidate_id", "burst_phase"])
    by_cic = _group_summary(holdout_ledger, ["candidate_id", "cic_type"])
    by_symbol = _group_summary(holdout_ledger, ["candidate_id", "symbol"])
    by_month = _group_summary(holdout_ledger, ["candidate_id", "month"])
    by_cp_o6 = _group_summary(holdout_ledger, ["candidate_id", "checkpoint_component"])
    by_state = _group_summary(holdout_ledger, ["candidate_id", "state_cluster"])

    period_summary.to_csv(outputs["strategy_period_summary"], index=False)
    holdout_ledger.to_csv(outputs["holdout_trade_ledger"], index=False)
    holdout_skipped.to_csv(outputs["holdout_skipped_candidates"], index=False)
    selected_vs_skipped.to_csv(outputs["holdout_selected_vs_skipped"], index=False)
    by_market.to_csv(outputs["holdout_by_market_regime"], index=False)
    by_burst.to_csv(outputs["holdout_by_burst_phase"], index=False)
    by_cic.to_csv(outputs["holdout_by_cic_type"], index=False)
    by_symbol.to_csv(outputs["holdout_by_symbol"], index=False)
    by_month.to_csv(outputs["holdout_by_month"], index=False)
    by_cp_o6.to_csv(outputs["holdout_by_cp60_o6"], index=False)
    by_state.to_csv(outputs["holdout_by_state_cluster"], index=False)
    _notes(root, period_summary, by_cp_o6, selected_vs_skipped)
    return outputs


__all__ = [
    "AUTOPSY_SPECS",
    "REPORT_ROOT",
    "V21AConfig",
    "write_v21a_holdout_autopsy",
]
