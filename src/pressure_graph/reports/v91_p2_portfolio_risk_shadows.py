from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


P2_EW = "P2_EW"
P2_VOL = "P2_VOL"
P2_BETA = "P2_BETA"
P2_CORR = "P2_CORR"
ARMS = (P2_EW, P2_VOL, P2_BETA, P2_CORR)


@dataclass(frozen=True)
class P2RiskShadowConfig:
    max_positions: int = 8
    total_exposure_cap: float = 8.0
    target_stop_risk: float = 0.025
    vol_size_min: float = 0.50
    vol_size_max: float = 1.25
    beta_exposure_cap: float = 6.0
    beta_min_residual_size: float = 0.25
    corr_cluster_max_positions: int = 2


def _num(row: object, name: str, default: float = np.nan) -> float:
    value = pd.to_numeric(getattr(row, name, default), errors="coerce")
    return float(value) if pd.notna(value) else default


def _vol_size(row: object, cfg: P2RiskShadowConfig) -> tuple[float, bool]:
    entry = _num(row, "entry_price")
    stop = _num(row, "sl_price")
    if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0:
        return 1.0, False
    distance = abs(stop / entry - 1.0)
    if not np.isfinite(distance) or distance <= 0:
        return 1.0, False
    return float(np.clip(cfg.target_stop_risk / distance, cfg.vol_size_min, cfg.vol_size_max)), True


def _cluster(row: object) -> tuple[str, bool]:
    value = str(
        getattr(row, "correlation_cluster_id_at_entry", "")
        or getattr(row, "cluster_id_at_entry", "")
        or ""
    )
    if value and value.lower() != "nan":
        return value, True
    return f"missing:{getattr(row, 'symbol', '')}", False


def _candidate_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if pool.empty:
        return pool.copy()
    out = pool.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="coerce")
    if "candidate_priority" not in out.columns:
        out["candidate_priority"] = np.where(
            out.get("candidate", pd.Series("", index=out.index)).astype(str).eq("CIC1_FILTERED_MIR1"), 2, 1
        )
    key = out.get("shadow_base_signal_id", out.get("signal_id", out.index.astype(str))).astype(str)
    out["_p2_signal_key"] = key
    return (
        out.sort_values(
            ["entry_time", "symbol", "candidate_priority"], ascending=[True, True, False]
        )
        .drop_duplicates("_p2_signal_key", keep="first")
        .reset_index(drop=True)
    )


def _select_arm(
    pool: pd.DataFrame,
    arm: str,
    cfg: P2RiskShadowConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _candidate_pool(pool)
    if data.empty:
        return data.copy(), data.copy()
    active: list[dict[str, object]] = []
    selected: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        active = [item for item in active if pd.Timestamp(item["exit_time"]) > entry_time]
        active_symbols = {str(item["symbol"]) for item in active}
        exposure_before = float(sum(float(item["size"]) for item in active))
        beta_before = float(sum(float(item["size"]) * float(item["abs_beta"]) for item in active))
        cluster, cluster_covered = _cluster(row)
        cluster_before = int(sum(str(item["cluster"]) == cluster for item in active))
        beta = abs(_num(row, "btc_beta_7d_at_entry"))
        beta_covered = bool(np.isfinite(beta))
        size = 1.0
        size_input_covered = True
        reason = ""

        if str(row.symbol) in active_symbols:
            reason = "symbol_already_active"
        elif len(active) >= cfg.max_positions:
            reason = "max_positions"
        elif arm == P2_VOL:
            size, size_input_covered = _vol_size(row, cfg)
        elif arm == P2_BETA:
            if not beta_covered:
                reason = "missing_btc_beta"
            else:
                remaining = cfg.beta_exposure_cap - beta_before
                size = min(1.0, remaining / max(beta, 1e-12))
                if size < cfg.beta_min_residual_size:
                    reason = "beta_exposure_cap"
        elif arm == P2_CORR and cluster_before >= cfg.corr_cluster_max_positions:
            reason = "correlation_cluster_cap"

        if not reason and exposure_before + size > cfg.total_exposure_cap:
            size = cfg.total_exposure_cap - exposure_before
            if size < cfg.beta_min_residual_size:
                reason = "total_exposure_cap"

        payload = row._asdict()
        payload.update(
            {
                "risk_shadow_arm": arm,
                "position_size": max(0.0, float(size)),
                "selected": not bool(reason),
                "skip_reason": reason,
                "concurrent_positions_before": len(active),
                "total_exposure_before": exposure_before,
                "total_exposure_after": exposure_before + (size if not reason else 0.0),
                "btc_beta_7d_abs": beta,
                "btc_beta_input_covered": beta_covered,
                "btc_beta_exposure_before": beta_before,
                "btc_beta_exposure_after": beta_before + (size * beta if not reason and beta_covered else 0.0),
                "correlation_cluster": cluster,
                "correlation_cluster_input_covered": cluster_covered,
                "cluster_positions_before": cluster_before,
                "size_input_covered": size_input_covered,
            }
        )
        for cost in (10, 20, 30):
            value = _num(row, f"net_return_{cost}bp")
            payload[f"weighted_net{cost}"] = float(size * value) if not reason and np.isfinite(value) else np.nan
        if reason:
            skipped.append(payload)
            continue
        selected.append(payload)
        active.append(
            {
                "exit_time": pd.Timestamp(row.exit_time),
                "symbol": str(row.symbol),
                "size": float(size),
                "abs_beta": float(beta) if beta_covered else 0.0,
                "cluster": cluster,
            }
        )
    return pd.DataFrame(selected), pd.DataFrame(skipped)


def _summary_row(arm: str, selected: pd.DataFrame, skipped: pd.DataFrame, cfg: P2RiskShadowConfig) -> dict[str, object]:
    burst = (
        selected.groupby("burst_id", dropna=False)["weighted_net20"].sum() / cfg.total_exposure_cap
        if not selected.empty and "burst_id" in selected.columns
        else pd.Series(dtype=float)
    )
    return {
        "risk_shadow_arm": arm,
        "selected_trades": int(len(selected)),
        "skipped_candidates": int(len(skipped)),
        "total_position_units": float(pd.to_numeric(selected.get("position_size"), errors="coerce").sum()) if not selected.empty else 0.0,
        "portfolio_net10": float(pd.to_numeric(selected.get("weighted_net10"), errors="coerce").sum()) / cfg.total_exposure_cap if not selected.empty else 0.0,
        "portfolio_net20": float(pd.to_numeric(selected.get("weighted_net20"), errors="coerce").sum()) / cfg.total_exposure_cap if not selected.empty else 0.0,
        "portfolio_net30": float(pd.to_numeric(selected.get("weighted_net30"), errors="coerce").sum()) / cfg.total_exposure_cap if not selected.empty else 0.0,
        "max_total_exposure": float(pd.to_numeric(selected.get("total_exposure_after"), errors="coerce").max()) if not selected.empty else 0.0,
        "max_btc_beta_exposure": float(pd.to_numeric(selected.get("btc_beta_exposure_after"), errors="coerce").max()) if not selected.empty else 0.0,
        "worst_burst_net20": float(burst.min()) if not burst.empty else np.nan,
        "beta_coverage": float(selected.get("btc_beta_input_covered", pd.Series(dtype=bool)).mean()) if not selected.empty else np.nan,
        "cluster_coverage": float(selected.get("correlation_cluster_input_covered", pd.Series(dtype=bool)).mean()) if not selected.empty else np.nan,
    }


def build_p2_portfolio_risk_shadows(
    pool: pd.DataFrame,
    cfg: P2RiskShadowConfig = P2RiskShadowConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for arm in ARMS:
        selected, skipped = _select_arm(pool, arm, cfg)
        selected_frames.append(selected)
        skipped_frames.append(skipped)
        summary_rows.append(_summary_row(arm, selected, skipped, cfg))
    selected_all = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    skipped_all = pd.concat(skipped_frames, ignore_index=True, sort=False) if skipped_frames else pd.DataFrame()
    return selected_all, skipped_all, pd.DataFrame(summary_rows)


__all__ = [
    "ARMS",
    "P2_BETA",
    "P2_CORR",
    "P2_EW",
    "P2_VOL",
    "P2RiskShadowConfig",
    "build_p2_portfolio_risk_shadows",
]
