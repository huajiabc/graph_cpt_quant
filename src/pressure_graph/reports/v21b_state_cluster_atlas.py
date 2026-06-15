"""v2.1B State Cluster Atlas.

The atlas discovers event-state clusters for the CIC/P2 candidate pool using
as-of state features. It is a diagnostic map, not a router or live rule.
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
    SEARCH_END,
    VALIDATION_END,
    V20Config,
    _beta_high_mask,
    _cp60_would_exit,
    _prepare_sample,
)


REPORT_ROOT = Path("reports/v2_1b_state_cluster_atlas")
SEED = 20260614
DEFAULT_CLUSTERS = 8


@dataclass(frozen=True)
class V21BConfig:
    report_root: Path = REPORT_ROOT
    v20: V20Config = V20Config()
    seed: int = SEED
    n_clusters: int = DEFAULT_CLUSTERS


STATE_FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    "market_impulse_density": ("market_impulse_density", "volume_impulse_density", "rank_market_impulse_density"),
    "cluster_density": ("cluster_density", "cluster_impulse_density", "rank_cluster_impulse_density"),
    "beta_strength": ("beta_extreme_strength", "c2_beta_extension_score", "rank_beta_extreme_strength", "ret_4h_percentile"),
    "local_shock_strength": ("local_volume_shock_strength", "volume_z_1h", "volume_z_4h", "rank_local_volume_shock_strength"),
    "ret_4h": ("ret_4h",),
    "ret_4h_percentile": ("ret_4h_percentile",),
    "leader_prior_1h": ("c2_leader_prior_1h_ratio",),
    "directed_edge_weight_1h": ("c2_directed_edge_weight_active_1h",),
    "symbol_volatility_percentile": ("symbol_volatility_percentile",),
    "liquidity_rank": ("rank_liquidity", "dynamic_all_rank"),
    "burst_count_so_far": ("burst_count_so_far",),
    "minutes_since_burst_start": ("time_since_burst_start_so_far", "minutes_since_burst_start"),
    "same_timestamp_peer_count": ("same_timestamp_peer_count",),
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


def _period(entry_time: pd.Series) -> pd.Series:
    entry = pd.to_datetime(entry_time, utc=True, errors="coerce")
    out = pd.Series("holdout", index=entry.index, dtype="object")
    out.loc[entry.lt(SEARCH_END)] = "search"
    out.loc[entry.ge(SEARCH_END) & entry.lt(VALIDATION_END)] = "validation"
    return out


def _month_cap(contrib: pd.Series, cap: float = 0.35) -> float:
    values = pd.to_numeric(contrib, errors="coerce").dropna()
    if values.empty:
        return np.nan
    total = float(values.sum())
    cap_value = total * cap if total > 0 else 0.0
    capped = [min(value, cap_value) if value > 0 and cap_value > 0 else value for value in values]
    return float(np.sum(capped))


def _safe_mean(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return np.nan
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def _safe_quantile(frame: pd.DataFrame, col: str, q: float) -> float:
    if col not in frame.columns or frame.empty:
        return np.nan
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else np.nan


def _build_state_features(sample: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=sample.index)
    for feature, cols in STATE_FEATURE_SOURCES.items():
        features[feature] = _first_numeric(sample, cols)
    cic = sample.get("candidate", pd.Series("", index=sample.index)).map(_cic_type)
    features["is_cic1"] = cic.eq("CIC1").astype(float)
    features["is_cic2"] = cic.eq("CIC2").astype(float)
    btc = _first_text(sample, ("btc_state_at_entry", "btc_market_state", "btc_state"))
    features["btc_up"] = btc.eq("BTC_up").astype(float)
    features["btc_chop"] = btc.eq("BTC_chop").astype(float)
    features["btc_down"] = btc.eq("BTC_down").astype(float)
    return features


def _standardize(features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median(axis=0, skipna=True).fillna(0.0)
    filled = numeric.fillna(medians)
    q75 = filled.quantile(0.75)
    q25 = filled.quantile(0.25)
    scale = (q75 - q25).replace(0.0, np.nan).fillna(filled.std(axis=0).replace(0.0, np.nan)).fillna(1.0)
    z = ((filled - medians) / scale).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return z.to_numpy(dtype=float), z


def _kmeans(features: pd.DataFrame, n_clusters: int, seed: int) -> np.ndarray:
    x, _ = _standardize(features)
    n = len(x)
    if n == 0:
        return np.array([], dtype=int)
    k = int(max(1, min(n_clusters, n)))
    weights = np.linspace(1.0, 2.0, x.shape[1]) if x.shape[1] else np.array([1.0])
    projection = x @ weights[: x.shape[1]]
    order = np.argsort(projection, kind="mergesort")
    initial = order[np.rint(np.linspace(0, n - 1, k)).astype(int)]
    centers = x[initial].copy()
    labels = np.full(n, -1, dtype=int)
    for _ in range(80):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for idx in range(k):
            members = x[labels == idx]
            if len(members):
                centers[idx] = members.mean(axis=0)
            else:
                centers[idx] = x[distances.min(axis=1).argmax()]
    return labels


def _assign_cluster_ids(sample: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = sample.copy()
    out["_raw_cluster"] = labels
    ordering = (
        out.groupby("_raw_cluster", sort=False)["net_return_at_cost"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    mapping = {int(row["_raw_cluster"]): f"SDG{idx:02d}" for idx, row in ordering.iterrows()}
    out["state_cluster_id"] = out["_raw_cluster"].map(mapping).astype(str)
    map_rows = [{"raw_cluster": raw, "state_cluster_id": cluster_id} for raw, cluster_id in mapping.items()]
    return out.drop(columns=["_raw_cluster"]), pd.DataFrame(map_rows)


def _prepare_membership(sample: pd.DataFrame, cfg: V21BConfig) -> pd.DataFrame:
    out = sample.copy()
    out["period"] = _period(out["entry_time"])
    out["cic_type"] = out.get("candidate", pd.Series("", index=out.index)).map(_cic_type)
    out["btc_state"] = _first_text(out, ("btc_state_at_entry", "btc_market_state", "btc_state"))
    out["market_impulse_density"] = _first_numeric(out, STATE_FEATURE_SOURCES["market_impulse_density"])
    out["cluster_density"] = _first_numeric(out, STATE_FEATURE_SOURCES["cluster_density"])
    out["beta_strength"] = _first_numeric(out, STATE_FEATURE_SOURCES["beta_strength"])
    out["local_shock_strength"] = _first_numeric(out, STATE_FEATURE_SOURCES["local_shock_strength"])
    out["cp60_would_exit"] = _cp60_would_exit(out)
    out["beta_high_protect_candidate"] = out["cp60_would_exit"] & _beta_high_mask(out)
    out["late_burst_o6_candidate"] = pd.to_numeric(out.get("burst_count_so_far"), errors="coerce").ge(9)
    out["net20"] = pd.to_numeric(out["net_return_at_cost"], errors="coerce")
    out["mfe_12h"] = pd.to_numeric(out.get("mfe_12h"), errors="coerce")
    out["mae_12h"] = pd.to_numeric(out.get("mae_12h"), errors="coerce")
    out["mfe_24h"] = pd.to_numeric(out.get("mfe_24h"), errors="coerce")
    out["mae_24h"] = pd.to_numeric(out.get("mae_24h"), errors="coerce")
    out["hit_10pct_12h"] = out.get("hit_10pct_12h", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["hit_20pct_24h"] = out.get("hit_20pct_24h", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    features = _build_state_features(out)
    labels = _kmeans(features, cfg.n_clusters, cfg.seed)
    clustered, _ = _assign_cluster_ids(out, labels)
    clustered["state_feature_vector_complete"] = features.notna().mean(axis=1)
    return clustered.sort_values(["state_cluster_id", "entry_time", "symbol"]).reset_index(drop=True)


def _dominant(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "unknown"
    return str(values.value_counts().idxmax())


def _cluster_summary_rows(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()
    for keys, group in data.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        month_returns = pd.to_numeric(group["net20"], errors="coerce").groupby(group["month"].astype(str), sort=False).sum()
        cp = group[group["cp60_would_exit"].fillna(False).astype(bool)]
        cp_delta = pd.to_numeric(cp.get("checkpoint_net_at_cost"), errors="coerce") - pd.to_numeric(cp.get("net_return_at_cost"), errors="coerce")
        late = group[group["late_burst_o6_candidate"].fillna(False).astype(bool)]
        row.update(
            {
                "trades": int(len(group)),
                "cic1_trades": int(group["cic_type"].eq("CIC1").sum()),
                "cic2_trades": int(group["cic_type"].eq("CIC2").sum()),
                "dominant_cic_type": _dominant(group["cic_type"]),
                "dominant_btc_state": _dominant(group["btc_state"]),
                "net20_sum": float(pd.to_numeric(group["net20"], errors="coerce").sum()),
                "net20_avg": _safe_mean(group, "net20"),
                "net20_median": float(pd.to_numeric(group["net20"], errors="coerce").median()) if len(group) else np.nan,
                "hit_rate": float(pd.to_numeric(group["net20"], errors="coerce").gt(0).mean()) if len(group) else np.nan,
                "hit10_12h": float(group["hit_10pct_12h"].mean()) if len(group) else np.nan,
                "hit20_24h": float(group["hit_20pct_24h"].mean()) if len(group) else np.nan,
                "median_mfe_12h": _safe_quantile(group, "mfe_12h", 0.50),
                "p90_mfe_12h": _safe_quantile(group, "mfe_12h", 0.90),
                "median_mae_12h": _safe_quantile(group, "mae_12h", 0.50),
                "p10_mae_12h": _safe_quantile(group, "mae_12h", 0.10),
                "avg_market_impulse_density": _safe_mean(group, "market_impulse_density"),
                "avg_cluster_density": _safe_mean(group, "cluster_density"),
                "avg_beta_strength": _safe_mean(group, "beta_strength"),
                "avg_local_shock_strength": _safe_mean(group, "local_shock_strength"),
                "avg_burst_count_so_far": _safe_mean(group, "burst_count_so_far"),
                "cp60_would_exit_rate": float(group["cp60_would_exit"].mean()) if len(group) else np.nan,
                "cp60_delta_vs_keep_avg": float(cp_delta.mean()) if len(cp_delta) else np.nan,
                "cp60_delta_vs_keep_sum": float(cp_delta.sum()) if len(cp_delta) else 0.0,
                "protect_a_candidate_rate": float(group["beta_high_protect_candidate"].mean()) if len(group) else np.nan,
                "late_burst_o6_candidate_rate": float(group["late_burst_o6_candidate"].mean()) if len(group) else np.nan,
                "late_burst_net20_avg": _safe_mean(late, "net20") if len(late) else np.nan,
                "month_cap35_net20": _month_cap(month_returns),
                "max_month_contribution": float(month_returns[month_returns > 0].max() / month_returns[month_returns > 0].sum())
                if float(month_returns[month_returns > 0].sum()) > 0
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _state_cluster_summary(membership: pd.DataFrame) -> pd.DataFrame:
    summary = _cluster_summary_rows(membership, ["state_cluster_id"])
    if summary.empty:
        return summary
    holdout = _cluster_summary_rows(membership[membership["period"].eq("holdout")], ["state_cluster_id"])
    validation = _cluster_summary_rows(membership[membership["period"].eq("validation")], ["state_cluster_id"])
    out = summary.copy()
    if not validation.empty:
        validation_cols = {
            col: f"validation_{col}"
            for col in validation.columns
            if col != "state_cluster_id"
        }
        out = out.merge(validation.rename(columns=validation_cols), on="state_cluster_id", how="left")
    if not holdout.empty:
        holdout_cols = {
            col: f"holdout_{col}"
            for col in holdout.columns
            if col != "state_cluster_id"
        }
        out = out.merge(holdout.rename(columns=holdout_cols), on="state_cluster_id", how="left")
    sort_cols = [col for col in ("holdout_net20_sum", "net20_sum") if col in out.columns]
    ascending = [True if col == "holdout_net20_sum" else False for col in sort_cols]
    return out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def _period_summary(membership: pd.DataFrame) -> pd.DataFrame:
    out = _cluster_summary_rows(membership, ["state_cluster_id", "period"])
    return out.sort_values(["period", "net20_sum"], ascending=[True, True]).reset_index(drop=True) if not out.empty else out


def _action_summary(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in membership.groupby("state_cluster_id", sort=False):
        cp = group[group["cp60_would_exit"].fillna(False).astype(bool)]
        cp_delta = pd.to_numeric(cp.get("checkpoint_net_at_cost"), errors="coerce") - pd.to_numeric(cp.get("net_return_at_cost"), errors="coerce")
        late = group[group["late_burst_o6_candidate"].fillna(False).astype(bool)]
        protect = group[group["beta_high_protect_candidate"].fillna(False).astype(bool)]
        rows.extend(
            [
                {
                    "state_cluster_id": cluster_id,
                    "action": "CP60_if_no_followthrough",
                    "eligible_trades": int(len(cp)),
                    "eligible_rate": float(len(cp) / len(group)) if len(group) else np.nan,
                    "avg_delta_if_action_vs_keep": float(cp_delta.mean()) if len(cp_delta) else np.nan,
                    "sum_delta_if_action_vs_keep": float(cp_delta.sum()) if len(cp_delta) else 0.0,
                    "eligible_net20_avg": _safe_mean(cp, "net20") if len(cp) else np.nan,
                },
                {
                    "state_cluster_id": cluster_id,
                    "action": "Protect_A_beta_high",
                    "eligible_trades": int(len(protect)),
                    "eligible_rate": float(len(protect) / len(group)) if len(group) else np.nan,
                    "avg_delta_if_action_vs_keep": np.nan,
                    "sum_delta_if_action_vs_keep": 0.0,
                    "eligible_net20_avg": _safe_mean(protect, "net20") if len(protect) else np.nan,
                },
                {
                    "state_cluster_id": cluster_id,
                    "action": "O6_late_burst_overflow",
                    "eligible_trades": int(len(late)),
                    "eligible_rate": float(len(late) / len(group)) if len(group) else np.nan,
                    "avg_delta_if_action_vs_keep": np.nan,
                    "sum_delta_if_action_vs_keep": 0.0,
                    "eligible_net20_avg": _safe_mean(late, "net20") if len(late) else np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def _feature_profile(membership: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "market_impulse_density",
        "cluster_density",
        "beta_strength",
        "local_shock_strength",
        "ret_4h",
        "ret_4h_percentile",
        "mfe_12h",
        "mae_12h",
        "burst_count_so_far",
    ]
    rows: list[dict[str, Any]] = []
    for cluster_id, group in membership.groupby("state_cluster_id", sort=False):
        for col in feature_cols:
            if col not in group.columns:
                continue
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "state_cluster_id": cluster_id,
                    "feature": col,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p25": float(values.quantile(0.25)),
                    "p75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def _notes(root: Path, summary: pd.DataFrame, period: pd.DataFrame, action: pd.DataFrame) -> None:
    def fmt_pct(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return "n/a" if pd.isna(numeric) else f"{numeric:.4%}"

    lines = [
        "# v2.1B State Cluster Atlas",
        "",
        "Status: offline diagnostic only. Clusters are not gates, selectors, or live rules.",
        "",
        "## Weakest Holdout Clusters",
    ]
    holdout_cols = ["state_cluster_id", "holdout_trades", "holdout_net20_sum", "holdout_net20_avg", "dominant_cic_type", "dominant_btc_state"]
    available = [col for col in holdout_cols if col in summary.columns]
    weak = summary[summary.get("holdout_trades", pd.Series(0, index=summary.index)).fillna(0).gt(0)].copy()
    for row in weak.sort_values("holdout_net20_sum", ascending=True).head(5)[available].itertuples(index=False):
        data = dict(zip(available, row, strict=False))
        lines.append(
            f"- {data.get('state_cluster_id')}: holdout_trades={int(data.get('holdout_trades', 0))}, "
            f"holdout_sum={data.get('holdout_net20_sum', np.nan):.4%}, "
            f"holdout_avg={data.get('holdout_net20_avg', np.nan):.4%}, "
            f"dominant={data.get('dominant_cic_type', 'unknown')}/{data.get('dominant_btc_state', 'unknown')}."
        )
    lines.extend(["", "## Strongest Full-Sample Clusters"])
    for row in summary.sort_values("net20_sum", ascending=False).head(5).itertuples(index=False):
        lines.append(
            f"- {row.state_cluster_id}: trades={row.trades}, net20_sum={row.net20_sum:.4%}, "
            f"avg={row.net20_avg:.4%}, hit10_12h={row.hit10_12h:.2%}, dominant={row.dominant_cic_type}/{row.dominant_btc_state}."
        )
    if not period.empty:
        pivot = period.pivot_table(
            index="state_cluster_id",
            columns="period",
            values="net20_sum",
            aggfunc="sum",
        ).reset_index()
        if {"search", "holdout"}.issubset(pivot.columns):
            flip = pivot[pivot["search"].gt(0) & pivot["holdout"].lt(0)].copy()
            if not flip.empty:
                flip["flip_damage"] = flip["holdout"]
                lines.extend(["", "## Holdout Flip Clusters"])
                for row in flip.sort_values("flip_damage").head(5).itertuples(index=False):
                    lines.append(
                        f"- {row.state_cluster_id}: search_net={fmt_pct(row.search)}, "
                        f"validation_net={fmt_pct(getattr(row, 'validation', np.nan))}, holdout_net={fmt_pct(row.holdout)}."
                    )
    cp = action[action["action"].eq("CP60_if_no_followthrough")].sort_values("sum_delta_if_action_vs_keep", ascending=False)
    if not cp.empty:
        lines.extend(["", "## CP60 Applicability"])
        for row in cp.head(5).itertuples(index=False):
            lines.append(
                f"- {row.state_cluster_id}: eligible={row.eligible_trades}, "
                f"sum_delta={row.sum_delta_if_action_vs_keep:.4%}, avg_delta={row.avg_delta_if_action_vs_keep:.4%}."
            )
    lines.extend(
        [
            "",
            "Next: use these clusters as the input vocabulary for v2.1C state-transition graph / v2.2 router research.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21b_state_cluster_atlas(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21BConfig = V21BConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v20)
    membership = _prepare_membership(sample, cfg)
    summary = _state_cluster_summary(membership)
    period = _period_summary(membership)
    action = _action_summary(membership)
    profile = _feature_profile(membership)
    holdout = membership[membership["period"].eq("holdout")].copy()

    outputs = {
        "state_cluster_membership": root / "state_cluster_membership.csv",
        "state_cluster_summary": root / "state_cluster_summary.csv",
        "state_cluster_period_summary": root / "state_cluster_period_summary.csv",
        "state_cluster_action_summary": root / "state_cluster_action_summary.csv",
        "state_cluster_feature_profile": root / "state_cluster_feature_profile.csv",
        "state_cluster_holdout_autopsy": root / "state_cluster_holdout_autopsy.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["state_cluster_membership"], index=False)
    summary.to_csv(outputs["state_cluster_summary"], index=False)
    period.to_csv(outputs["state_cluster_period_summary"], index=False)
    action.to_csv(outputs["state_cluster_action_summary"], index=False)
    profile.to_csv(outputs["state_cluster_feature_profile"], index=False)
    holdout.to_csv(outputs["state_cluster_holdout_autopsy"], index=False)
    _notes(root, summary, period, action)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "STATE_FEATURE_SOURCES",
    "V21BConfig",
    "write_v21b_state_cluster_atlas",
]
