from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class P2CorrelationClusterConfig:
    lookback_days: int = 30
    min_overlap_observations: int = 7 * 24 * 4
    correlation_threshold: float = 0.70
    mutual_top_k: int = 3


def _utc_month(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.to_datetime(parsed.dt.strftime("%Y-%m-01"), utc=True, errors="coerce")


def _components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    remaining = set(adjacency)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = {start}
        while stack:
            source = stack.pop()
            for target in sorted(adjacency.get(source, set())):
                if target in remaining:
                    remaining.remove(target)
                    component.add(target)
                    stack.append(target)
        components.append(sorted(component))
    return sorted(components, key=lambda items: (-len(items), items[0]))


def build_asof_correlation_membership(
    frame: pd.DataFrame,
    cfg: P2CorrelationClusterConfig = P2CorrelationClusterConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build month-frozen clusters using only returns visible before each month."""
    required = {"feature_time", "symbol", "ret_1h"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(), pd.DataFrame()
    data = frame[["feature_time", "symbol", "ret_1h"]].copy()
    data["feature_time"] = pd.to_datetime(data["feature_time"], utc=True, errors="coerce")
    data["symbol"] = data["symbol"].fillna("").astype(str)
    data["ret_1h"] = pd.to_numeric(data["ret_1h"], errors="coerce")
    data = data.dropna(subset=["feature_time"]).loc[lambda item: item["symbol"].ne("")]
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    data["month_start"] = _utc_month(data["feature_time"])
    membership_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for month_start in sorted(data["month_start"].dropna().unique()):
        month = pd.Timestamp(month_start)
        next_month = month + pd.DateOffset(months=1)
        universe = sorted(
            data.loc[
                data["feature_time"].ge(month) & data["feature_time"].lt(next_month),
                "symbol",
            ].unique()
        )
        hist = data[
            data["feature_time"].ge(month - pd.Timedelta(days=cfg.lookback_days))
            & data["feature_time"].lt(month)
            & data["symbol"].isin(universe)
        ]
        pivot = hist.pivot_table(
            index="feature_time", columns="symbol", values="ret_1h", aggfunc="last", observed=True
        )
        observation_counts = pivot.notna().sum() if not pivot.empty else pd.Series(dtype="int64")
        eligible = sorted(
            symbol
            for symbol in universe
            if int(observation_counts.get(symbol, 0)) >= cfg.min_overlap_observations
        )
        adjacency = {symbol: set() for symbol in eligible}
        corr = (
            pivot[eligible].corr(min_periods=cfg.min_overlap_observations)
            if len(eligible) >= 2
            else pd.DataFrame()
        )
        ranked: dict[str, set[str]] = {}
        if not corr.empty:
            for source in eligible:
                scores = corr[source].drop(labels=[source], errors="ignore").dropna()
                scores = scores[scores.ge(cfg.correlation_threshold)].sort_values(ascending=False)
                ranked[source] = set(scores.head(cfg.mutual_top_k).index.astype(str))
            for source, targets in ranked.items():
                for target in targets:
                    if source in ranked.get(target, set()):
                        adjacency[source].add(target)
                        adjacency[target].add(source)
                        if source < target:
                            edge_rows.append(
                                {
                                    "month_start": month,
                                    "source_symbol": source,
                                    "target_symbol": target,
                                    "correlation": float(corr.loc[source, target]),
                                    "lookback_days": cfg.lookback_days,
                                    "min_overlap_observations": cfg.min_overlap_observations,
                                    "correlation_threshold": cfg.correlation_threshold,
                                    "mutual_top_k": cfg.mutual_top_k,
                                }
                            )
        cluster_by_symbol: dict[str, tuple[str, int]] = {}
        for idx, component in enumerate(_components(adjacency), start=1):
            cluster_id = f"{month:%Y-%m}:RC{idx:02d}"
            for symbol in component:
                cluster_by_symbol[symbol] = (cluster_id, len(component))
        for symbol in universe:
            cluster_id, cluster_size = cluster_by_symbol.get(symbol, ("", np.nan))
            membership_rows.append(
                {
                    "month_start": month,
                    "symbol": symbol,
                    "correlation_cluster_id": cluster_id,
                    "correlation_cluster_size": cluster_size,
                    "correlation_cluster_input_covered": bool(cluster_id),
                    "correlation_observations": int(observation_counts.get(symbol, 0)),
                    "history_start": month - pd.Timedelta(days=cfg.lookback_days),
                    "history_end_exclusive": month,
                    "lookback_days": cfg.lookback_days,
                    "min_overlap_observations": cfg.min_overlap_observations,
                    "correlation_threshold": cfg.correlation_threshold,
                    "mutual_top_k": cfg.mutual_top_k,
                }
            )
    return pd.DataFrame(membership_rows), pd.DataFrame(edge_rows)


def attach_asof_correlation_clusters(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["correlation_cluster_id"] = ""
    out["correlation_cluster_size"] = np.nan
    out["correlation_cluster_input_covered"] = False
    if membership.empty or not {"month_start", "symbol"}.issubset(membership.columns):
        return out
    local = membership.copy()
    local["month_start"] = pd.to_datetime(local["month_start"], utc=True, errors="coerce")
    local["symbol"] = local["symbol"].fillna("").astype(str)
    out["_correlation_month_start"] = _utc_month(out["feature_time"])
    merged = out.merge(
        local[
            [
                "month_start",
                "symbol",
                "correlation_cluster_id",
                "correlation_cluster_size",
                "correlation_cluster_input_covered",
            ]
        ],
        left_on=["_correlation_month_start", "symbol"],
        right_on=["month_start", "symbol"],
        how="left",
        suffixes=("", "_membership"),
    )
    for column, default in [
        ("correlation_cluster_id", ""),
        ("correlation_cluster_size", np.nan),
        ("correlation_cluster_input_covered", False),
    ]:
        source = f"{column}_membership"
        if source in merged.columns:
            merged[column] = merged[source].fillna(default)
    return merged.drop(
        columns=[
            "_correlation_month_start",
            "month_start",
            "correlation_cluster_id_membership",
            "correlation_cluster_size_membership",
            "correlation_cluster_input_covered_membership",
        ],
        errors="ignore",
    )


def add_asof_correlation_clusters(
    frame: pd.DataFrame,
    cfg: P2CorrelationClusterConfig = P2CorrelationClusterConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    membership, edges = build_asof_correlation_membership(frame, cfg)
    return attach_asof_correlation_clusters(frame, membership), membership, edges


__all__ = [
    "P2CorrelationClusterConfig",
    "add_asof_correlation_clusters",
    "attach_asof_correlation_clusters",
    "build_asof_correlation_membership",
]
