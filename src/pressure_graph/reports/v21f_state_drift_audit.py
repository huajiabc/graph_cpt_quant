"""v2.1F State Drift / Novelty Audit.

Audits whether holdout damage is associated with feature drift inside the
discovered state clusters. The report is diagnostic only: it does not create a
router, gate, selector, or live rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v20_graph_motif_search import V20Config, _prepare_sample
from pressure_graph.reports.v21b_state_cluster_atlas import V21BConfig, _prepare_membership


REPORT_ROOT = Path("reports/v2_1f_state_drift_audit")
DRIFT_FEATURES = [
    "market_impulse_density",
    "cluster_density",
    "beta_strength",
    "local_shock_strength",
    "ret_4h",
    "ret_4h_percentile",
    "leader_prior_1h",
    "directed_edge_weight_1h",
    "symbol_volatility_percentile",
    "liquidity_rank",
    "burst_count_so_far",
    "minutes_since_burst_start",
    "same_timestamp_peer_count",
]


@dataclass(frozen=True)
class V21FConfig:
    report_root: Path = REPORT_ROOT
    v21b: V21BConfig = V21BConfig(v20=V20Config())
    min_cluster_reference_events: int = 5
    min_history_months: int = 2
    novelty_buckets: int = 5


def _available_features(frame: pd.DataFrame) -> list[str]:
    return [col for col in DRIFT_FEATURES if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any()]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _entry_month(frame: pd.DataFrame) -> pd.Series:
    entry = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    return entry.dt.strftime("%Y-%m")


def _safe_std(values: pd.Series) -> float:
    std = float(pd.to_numeric(values, errors="coerce").std(ddof=0))
    return std if np.isfinite(std) and std > 1e-12 else np.nan


def _safe_iqr(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    iqr = float(numeric.quantile(0.75) - numeric.quantile(0.25))
    if np.isfinite(iqr) and iqr > 1e-12:
        return iqr
    return _safe_std(numeric)


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 5) -> float:
    exp = pd.to_numeric(expected, errors="coerce").dropna()
    act = pd.to_numeric(actual, errors="coerce").dropna()
    if len(exp) < 5 or len(act) < 2:
        return np.nan
    edges = np.unique(exp.quantile(np.linspace(0.0, 1.0, bins + 1)).to_numpy(dtype=float))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts = pd.cut(exp, edges, include_lowest=True).value_counts(sort=False)
    act_counts = pd.cut(act, edges, include_lowest=True).value_counts(sort=False)
    exp_pct = exp_counts / max(1, int(exp_counts.sum()))
    act_pct = act_counts / max(1, int(act_counts.sum()))
    eps = 1e-6
    return float(((act_pct + eps - exp_pct - eps) * np.log((act_pct + eps) / (exp_pct + eps))).sum())


def _feature_period_drift(membership: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference = membership[membership["period"].isin(["search", "validation"])].copy()
    for feature in features:
        ref = _num(reference, feature)
        ref_mean = float(ref.mean()) if ref.notna().any() else np.nan
        ref_std = _safe_std(ref)
        for period, group in membership.groupby("period", sort=False):
            values = _num(group, feature)
            mean = float(values.mean()) if values.notna().any() else np.nan
            median = float(values.median()) if values.notna().any() else np.nan
            std = _safe_std(values)
            rows.append(
                {
                    "feature": feature,
                    "period": period,
                    "events": int(values.notna().sum()),
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "reference_mean": ref_mean,
                    "reference_std": ref_std,
                    "mean_delta_vs_preholdout": mean - ref_mean if pd.notna(mean) and pd.notna(ref_mean) else np.nan,
                    "std_mean_delta_vs_preholdout": (mean - ref_mean) / ref_std
                    if pd.notna(mean) and pd.notna(ref_mean) and pd.notna(ref_std)
                    else np.nan,
                    "psi_vs_preholdout": _psi(ref, values),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["period", "psi_vs_preholdout"], ascending=[True, False], na_position="last").reset_index(drop=True)


def _cluster_reference_profiles(membership: pd.DataFrame, features: list[str], cfg: V21FConfig) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    reference = membership[membership["period"].isin(["search", "validation"])].copy()
    for cluster_id, group in reference.groupby("state_cluster_id", sort=False):
        if len(group) < cfg.min_cluster_reference_events:
            continue
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for feature in features:
            values = _num(group, feature)
            if not values.notna().any():
                continue
            means[feature] = float(values.mean())
            scale = _safe_iqr(values)
            scales[feature] = float(scale) if pd.notna(scale) else 1.0
        profiles[str(cluster_id)] = {
            "events": int(len(group)),
            "means": means,
            "scales": scales,
        }
    return profiles


def _drift_against_profile(group: pd.DataFrame, profile: dict[str, Any], features: list[str]) -> tuple[float, str, int]:
    scores: list[float] = []
    feature_scores: list[tuple[str, float]] = []
    means = profile.get("means", {})
    scales = profile.get("scales", {})
    for feature in features:
        if feature not in means:
            continue
        value = _num(group, feature).mean()
        scale = scales.get(feature, 1.0) or 1.0
        if pd.isna(value):
            continue
        score = abs(float(value) - float(means[feature])) / max(abs(float(scale)), 1e-9)
        scores.append(score)
        feature_scores.append((feature, score))
    top = ",".join([name for name, _ in sorted(feature_scores, key=lambda item: item[1], reverse=True)[:3]])
    return (float(np.mean(scores)) if scores else np.nan, top, len(scores))


def _cluster_period_drift(membership: pd.DataFrame, features: list[str], cfg: V21FConfig) -> pd.DataFrame:
    profiles = _cluster_reference_profiles(membership, features, cfg)
    rows: list[dict[str, Any]] = []
    for (cluster_id, period), group in membership.groupby(["state_cluster_id", "period"], sort=False):
        profile = profiles.get(str(cluster_id), {})
        drift_score, top_features, feature_count = _drift_against_profile(group, profile, features) if profile else (np.nan, "", 0)
        rows.append(
            {
                "state_cluster_id": cluster_id,
                "period": period,
                "events": int(len(group)),
                "net20_sum": float(_num(group, "net20").sum()),
                "net20_avg": float(_num(group, "net20").mean()) if len(group) else np.nan,
                "hit_rate": float(_num(group, "net20").gt(0).mean()) if len(group) else np.nan,
                "reference_events": int(profile.get("events", 0)) if profile else 0,
                "cluster_feature_drift_score": drift_score,
                "drift_feature_count": feature_count,
                "dominant_drift_features": top_features,
                "dominant_cic_type": str(group["cic_type"].mode().iloc[0]) if "cic_type" in group and not group["cic_type"].mode().empty else "unknown",
                "dominant_btc_state": str(group["btc_state"].mode().iloc[0]) if "btc_state" in group and not group["btc_state"].mode().empty else "unknown",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["period", "net20_sum"], ascending=[True, True]).reset_index(drop=True)


def _event_novelty(event: pd.Series, profile: dict[str, Any], features: list[str]) -> tuple[float, str, int]:
    scores: list[float] = []
    feature_scores: list[tuple[str, float]] = []
    means = profile.get("means", {})
    scales = profile.get("scales", {})
    for feature in features:
        if feature not in means:
            continue
        value = pd.to_numeric(pd.Series([event.get(feature)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        scale = scales.get(feature, 1.0) or 1.0
        score = abs(float(value) - float(means[feature])) / max(abs(float(scale)), 1e-9)
        scores.append(score)
        feature_scores.append((feature, score))
    top = ",".join([name for name, _ in sorted(feature_scores, key=lambda item: item[1], reverse=True)[:3]])
    return (float(np.mean(scores)) if scores else np.nan, top, len(scores))


def _profiles_from_history(history: pd.DataFrame, features: list[str], cfg: V21FConfig) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for cluster_id, group in history.groupby("state_cluster_id", sort=False):
        if len(group) < cfg.min_cluster_reference_events:
            continue
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for feature in features:
            values = _num(group, feature)
            if not values.notna().any():
                continue
            means[feature] = float(values.mean())
            scale = _safe_iqr(values)
            scales[feature] = float(scale) if pd.notna(scale) else 1.0
        profiles[str(cluster_id)] = {"events": int(len(group)), "means": means, "scales": scales}
    return profiles


def _walkforward_novelty(membership: pd.DataFrame, features: list[str], cfg: V21FConfig) -> pd.DataFrame:
    data = membership.copy()
    data["entry_month"] = _entry_month(data)
    months = sorted(data["entry_month"].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    for month in months:
        history = data[data["entry_month"].lt(month)].copy()
        history_months = sorted(history["entry_month"].dropna().astype(str).unique().tolist())
        profiles = _profiles_from_history(history, features, cfg) if len(history_months) >= cfg.min_history_months else {}
        current = data[data["entry_month"].eq(month)].copy()
        for _, event in current.iterrows():
            cluster_id = str(event.get("state_cluster_id", ""))
            profile = profiles.get(cluster_id)
            if profile:
                novelty, top_features, feature_count = _event_novelty(event, profile, features)
                status = "covered"
                reference_events = int(profile["events"])
            else:
                novelty, top_features, feature_count = np.nan, "", 0
                status = "insufficient_history"
                reference_events = 0
            row = {
                "trade_key": event.get("trade_key", event.get("signal_id", "")),
                "symbol": event.get("symbol", ""),
                "candidate": event.get("candidate", ""),
                "entry_time": event.get("entry_time"),
                "entry_month": month,
                "period": event.get("period", ""),
                "state_cluster_id": cluster_id,
                "net20": event.get("net20", np.nan),
                "novelty_status": status,
                "walkforward_state_novelty": novelty,
                "novelty_feature_count": feature_count,
                "dominant_novelty_features": top_features,
                "cluster_reference_events": reference_events,
            }
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    covered = out["walkforward_state_novelty"].notna()
    out["novelty_bucket"] = pd.Series(np.nan, index=out.index, dtype="object")
    if covered.any():
        values = out.loc[covered, "walkforward_state_novelty"]
        labels = [f"q{i + 1}" for i in range(int(cfg.novelty_buckets))]
        try:
            out.loc[covered, "novelty_bucket"] = pd.qcut(values.rank(method="first"), q=int(cfg.novelty_buckets), labels=labels)
        except ValueError:
            out.loc[covered, "novelty_bucket"] = "single_bucket"
    out["novelty_bucket"] = out["novelty_bucket"].astype("object").fillna("insufficient_history")
    return out


def _summary_rows(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame()
    for keys, group in data.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        values = _num(group, "net20")
        row.update(
            {
                "events": int(len(group)),
                "net20_sum": float(values.sum()),
                "net20_avg": float(values.mean()) if len(group) else np.nan,
                "hit_rate": float(values.gt(0).mean()) if len(group) else np.nan,
                "avg_state_novelty": float(_num(group, "walkforward_state_novelty").mean()) if "walkforward_state_novelty" in group else np.nan,
                "cic1_events": int(group.get("candidate", pd.Series("", index=group.index)).astype(str).str.contains("CIC1").sum()),
                "cic2_events": int(group.get("candidate", pd.Series("", index=group.index)).astype(str).str.contains("CIC2").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _novelty_bucket_summary(novelty: pd.DataFrame) -> pd.DataFrame:
    all_rows = _summary_rows(novelty, ["novelty_bucket"])
    all_rows.insert(0, "scope", "full") if not all_rows.empty else None
    covered = novelty[novelty["novelty_status"].eq("covered")].copy() if "novelty_status" in novelty.columns else novelty.iloc[0:0].copy()
    covered_rows = _summary_rows(covered, ["novelty_bucket"])
    if not covered_rows.empty:
        covered_rows.insert(0, "scope", "covered_full")
    period_rows = _summary_rows(novelty, ["period", "novelty_bucket"])
    if not period_rows.empty:
        period_rows.insert(0, "scope", "period")
    covered_period_rows = _summary_rows(covered, ["period", "novelty_bucket"])
    if not covered_period_rows.empty:
        covered_period_rows.insert(0, "scope", "covered_period")
    frames = [frame for frame in (all_rows, covered_rows, period_rows, covered_period_rows) if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _holdout_drift_autopsy(cluster_drift: pd.DataFrame, novelty: pd.DataFrame) -> pd.DataFrame:
    holdout_clusters = cluster_drift[cluster_drift["period"].eq("holdout")].copy()
    novelty_holdout = _summary_rows(novelty[novelty["period"].eq("holdout")], ["state_cluster_id", "novelty_bucket"])
    if novelty_holdout.empty:
        return holdout_clusters
    return holdout_clusters.merge(
        novelty_holdout.rename(
            columns={
                "events": "bucket_events",
                "net20_sum": "bucket_net20_sum",
                "net20_avg": "bucket_net20_avg",
                "hit_rate": "bucket_hit_rate",
                "avg_state_novelty": "bucket_avg_state_novelty",
            }
        ),
        on="state_cluster_id",
        how="left",
    ).sort_values(["net20_sum", "bucket_net20_sum"], ascending=[True, True]).reset_index(drop=True)


def _notes(root: Path, feature_drift: pd.DataFrame, cluster_drift: pd.DataFrame, novelty_summary: pd.DataFrame) -> None:
    lines = [
        "# v2.1F State Drift / Novelty Audit",
        "",
        "Status: offline diagnostic only. No state router, gate, paper-live, or real-live rule is promoted.",
        "",
        "## Largest Holdout Feature Drift",
    ]
    holdout_feature = feature_drift[feature_drift["period"].eq("holdout")].copy()
    if holdout_feature.empty:
        lines.append("- No holdout feature drift rows were produced.")
    else:
        for row in holdout_feature.sort_values("psi_vs_preholdout", ascending=False, na_position="last").head(6).itertuples(index=False):
            lines.append(
                f"- {row.feature}: psi={row.psi_vs_preholdout:.4f}, "
                f"std_mean_delta={row.std_mean_delta_vs_preholdout:.4f}, events={row.events}."
            )
    lines.extend(["", "## Weak Holdout Clusters And Drift"])
    weak = cluster_drift[cluster_drift["period"].eq("holdout")].copy()
    if weak.empty:
        lines.append("- No holdout cluster rows were produced.")
    else:
        for row in weak.sort_values("net20_sum", ascending=True).head(6).itertuples(index=False):
            lines.append(
                f"- {row.state_cluster_id}: net20_sum={row.net20_sum:.4%}, "
                f"events={row.events}, drift_score={row.cluster_feature_drift_score:.4f}, "
                f"top_drift={row.dominant_drift_features or 'n/a'}, "
                f"dominant={row.dominant_cic_type}/{row.dominant_btc_state}."
            )
    lines.extend(["", "## Walk-forward Novelty Buckets"])
    full = novelty_summary[novelty_summary.get("scope", pd.Series("", index=novelty_summary.index)).eq("covered_full")].copy()
    if full.empty:
        full = novelty_summary[novelty_summary.get("scope", pd.Series("", index=novelty_summary.index)).eq("full")].copy()
    if full.empty:
        lines.append("- No novelty bucket summary was produced.")
    else:
        for row in full.sort_values("novelty_bucket").itertuples(index=False):
            lines.append(
                f"- {row.novelty_bucket}: events={row.events}, net20_sum={row.net20_sum:.4%}, "
                f"avg={row.net20_avg:.4%}, novelty={row.avg_state_novelty:.4f}."
            )
    lines.extend(
        [
            "",
            "Decision rule: drift/novelty evidence must become stable in walk-forward validation before it can be converted into a meta-router action.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v21f_state_drift_audit(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V21FConfig = V21FConfig(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg.v21b.v20)
    membership = _prepare_membership(sample, cfg.v21b)
    features = _available_features(membership)
    feature_drift = _feature_period_drift(membership, features)
    cluster_drift = _cluster_period_drift(membership, features, cfg)
    novelty = _walkforward_novelty(membership, features, cfg)
    novelty_summary = _novelty_bucket_summary(novelty)
    holdout_autopsy = _holdout_drift_autopsy(cluster_drift, novelty)

    outputs = {
        "state_feature_period_drift": root / "state_feature_period_drift.csv",
        "state_cluster_period_drift": root / "state_cluster_period_drift.csv",
        "walkforward_state_novelty": root / "walkforward_state_novelty.csv",
        "walkforward_novelty_bucket_summary": root / "walkforward_novelty_bucket_summary.csv",
        "holdout_drift_autopsy": root / "holdout_drift_autopsy.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    feature_drift.to_csv(outputs["state_feature_period_drift"], index=False)
    cluster_drift.to_csv(outputs["state_cluster_period_drift"], index=False)
    novelty.to_csv(outputs["walkforward_state_novelty"], index=False)
    novelty_summary.to_csv(outputs["walkforward_novelty_bucket_summary"], index=False)
    holdout_autopsy.to_csv(outputs["holdout_drift_autopsy"], index=False)
    _notes(root, feature_drift, cluster_drift, novelty_summary)
    return outputs


__all__ = [
    "DRIFT_FEATURES",
    "REPORT_ROOT",
    "V21FConfig",
    "write_v21f_state_drift_audit",
]
