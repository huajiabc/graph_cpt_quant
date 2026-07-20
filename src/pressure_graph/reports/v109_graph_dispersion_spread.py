"""Market-neutral spread after extreme dispersion inside residual graph communities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    BTC,
    estimate_v106_betas,
)


REPORT_ROOT = Path("reports/v10_9_graph_dispersion_spread")
PANEL_PATH = Path("reports/v10_8_oi_leader_bucket/oi_feature_panel.parquet")
CANDIDATES = ("GDS1_COMMUNITY_CONVERGENCE", "GDS2_COMMUNITY_CONTINUATION")


@dataclass(frozen=True)
class V109Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    min_samples: int = 500
    community_count: int = 8
    min_community_size: int = 6
    dispersion_quantile: float = 0.95
    max_communities: int = 3
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260714


def load_v109_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    columns = ["symbol", "feature_time", "ret_1h", "future_ret_4h"]
    panel = pd.read_parquet(path, columns=columns)
    panel["feature_time"] = pd.to_datetime(
        panel["feature_time"], utc=True, errors="coerce"
    )
    for column in ("ret_1h", "future_ret_4h"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["month_start"] = pd.to_datetime(
        panel["feature_time"].dt.strftime("%Y-%m-01"), utc=True, errors="coerce"
    )
    return panel.drop_duplicates(["symbol", "feature_time"]).sort_values(
        ["feature_time", "symbol"]
    )


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(
        index="feature_time",
        columns="symbol",
        values=column,
        aggfunc="last",
        observed=True,
    ).sort_index()


def build_v109_communities(
    residual_history: pd.DataFrame,
    community_count: int = 8,
    min_samples: int = 500,
) -> tuple[list[list[str]], pd.DataFrame]:
    eligible = sorted(
        symbol
        for symbol in residual_history.columns
        if symbol != BTC
        and int(residual_history[symbol].notna().sum()) >= min_samples
    )
    complete = residual_history[eligible].dropna(how="any") if eligible else pd.DataFrame()
    if len(complete) < min_samples or len(eligible) < community_count:
        return [], pd.DataFrame()
    correlation = complete.corr().fillna(-1.0)
    candidates = sorted(
        (
            (float(correlation.loc[source, target]), source, target)
            for index, source in enumerate(eligible)
            for target in eligible[index + 1 :]
        ),
        reverse=True,
    )
    parent = {symbol: symbol for symbol in eligible}

    def find(symbol: str) -> str:
        while parent[symbol] != symbol:
            parent[symbol] = parent[parent[symbol]]
            symbol = parent[symbol]
        return symbol

    tree = []
    for value, source, target in candidates:
        root_source = find(source)
        root_target = find(target)
        if root_source == root_target:
            continue
        parent[root_target] = root_source
        tree.append((value, source, target))
        if len(tree) == len(eligible) - 1:
            break
    keep_count = max(len(eligible) - community_count, 0)
    kept = sorted(tree, reverse=True)[:keep_count]
    adjacency = {symbol: set() for symbol in eligible}
    for _, source, target in kept:
        adjacency[source].add(target)
        adjacency[target].add(source)
    remaining = set(eligible)
    communities = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component = []
        remaining.remove(start)
        while stack:
            source = stack.pop()
            component.append(source)
            for target in adjacency[source]:
                if target in remaining:
                    remaining.remove(target)
                    stack.append(target)
        communities.append(sorted(component))
    communities = sorted(communities, key=lambda values: (-len(values), values[0]))
    edge_rows = [
        {
            "source_symbol": source,
            "target_symbol": target,
            "correlation": value,
            "tree_edge_retained": (value, source, target) in kept,
        }
        for value, source, target in tree
    ]
    return communities, pd.DataFrame(edge_rows)


def _period(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if month < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def build_v109_contexts(
    panel: pd.DataFrame,
    cfg: V109Config,
) -> tuple[dict[pd.Timestamp, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    memberships = []
    tree_edges = []
    for raw_month in sorted(panel["month_start"].dropna().unique()):
        month = pd.Timestamp(raw_month)
        history = panel[
            panel["feature_time"].ge(month - pd.Timedelta(days=cfg.lookback_days))
            & panel["feature_time"].lt(month)
        ]
        target = panel[panel["month_start"].eq(month)]
        if history.empty or target.empty:
            continue
        hourly = history[history["feature_time"].dt.minute.eq(0)]
        historical_ret = _pivot(hourly, "ret_1h")
        betas = estimate_v106_betas(historical_ret)
        if BTC not in historical_ret.columns or len(betas) < cfg.community_count + 1:
            continue
        residual_hourly = pd.DataFrame(index=historical_ret.index)
        for symbol in historical_ret.columns:
            if symbol == BTC or symbol not in betas.index:
                continue
            residual_hourly[symbol] = historical_ret[symbol] - float(
                betas[symbol]
            ) * historical_ret[BTC]
        communities, edges = build_v109_communities(
            residual_hourly, cfg.community_count, cfg.min_samples
        )
        if not communities:
            continue
        historical_ret_full = _pivot(history, "ret_1h")
        historical_residual = pd.DataFrame(index=historical_ret_full.index)
        for symbol in residual_hourly.columns:
            historical_residual[symbol] = historical_ret_full[symbol] - float(
                betas[symbol]
            ) * historical_ret_full[BTC]
        target_ret = _pivot(target, "ret_1h")
        target_future = _pivot(target, "future_ret_4h")
        if BTC not in target_ret.columns or BTC not in target_future.columns:
            continue
        symbols = sorted(set().union(*communities) & set(target_ret.columns))
        residual_current = pd.DataFrame(index=target_ret.index)
        residual_future = pd.DataFrame(index=target_future.index)
        for symbol in symbols:
            residual_current[symbol] = target_ret[symbol] - float(betas[symbol]) * target_ret[BTC]
            residual_future[symbol] = target_future[symbol] - float(betas[symbol]) * target_future[BTC]
        thresholds = {}
        for index, members in enumerate(communities, start=1):
            community_id = f"{month:%Y-%m}:MST{index:02d}"
            members = [symbol for symbol in members if symbol in residual_current.columns]
            if members:
                dispersion = historical_residual[members].quantile(0.75, axis=1) - historical_residual[members].quantile(0.25, axis=1)
                thresholds[community_id] = float(
                    dispersion.quantile(cfg.dispersion_quantile)
                )
            for symbol in members:
                memberships.append(
                    {
                        "month_start": month,
                        "community_id": community_id,
                        "symbol": symbol,
                        "community_size": len(members),
                        "dispersion_threshold": thresholds.get(community_id, np.nan),
                    }
                )
        if not edges.empty:
            edges = edges.copy()
            edges["month_start"] = month
            tree_edges.append(edges)
        common_times = residual_current.index.intersection(residual_future.index)
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "times": common_times,
            "communities": {
                f"{month:%Y-%m}:MST{index:02d}": [
                    symbol for symbol in members if symbol in residual_current.columns
                ]
                for index, members in enumerate(communities, start=1)
            },
            "thresholds": thresholds,
            "historical_residual": historical_residual,
            "residual_current": residual_current.reindex(common_times),
            "residual_future": residual_future.reindex(common_times),
            "raw_future": target_future.reindex(index=common_times, columns=symbols),
        }
    membership = pd.DataFrame(memberships)
    edges = pd.concat(tree_edges, ignore_index=True) if tree_edges else pd.DataFrame()
    return contexts, membership, edges


def _community_events(
    context: dict[str, Any],
    communities: dict[str, list[str]],
    thresholds: dict[str, float],
    cfg: V109Config,
    signal_shift_bars: int,
) -> pd.DataFrame:
    rows = []
    current: pd.DataFrame = context["residual_current"]
    signal = current.shift(signal_shift_bars) if signal_shift_bars else current
    last_by_community: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for community_id, members in communities.items():
        members = [symbol for symbol in members if symbol in signal.columns]
        threshold = thresholds.get(community_id, np.nan)
        if len(members) < cfg.min_community_size or not np.isfinite(threshold) or threshold <= 0:
            continue
        local = signal[members]
        dispersion = local.quantile(0.75, axis=1) - local.quantile(0.25, axis=1)
        active = dispersion.ge(threshold)
        transition = active & ~active.shift(1, fill_value=False)
        for timestamp in transition.index[transition]:
            last = last_by_community.get(community_id)
            if last is not None and pd.Timestamp(timestamp) - last < cooldown:
                continue
            values = local.loc[timestamp].dropna().sort_values()
            third = len(values) // 3
            if third < 2:
                continue
            bottom = values.head(third).index.tolist()
            top = values.tail(third).index.tolist()
            residual_future: pd.DataFrame = context["residual_future"]
            raw_future: pd.DataFrame = context["raw_future"]
            convergence_residual = 0.5 * (
                residual_future.loc[timestamp, bottom].mean()
                - residual_future.loc[timestamp, top].mean()
            )
            convergence_raw = 0.5 * (
                raw_future.loc[timestamp, bottom].mean()
                - raw_future.loc[timestamp, top].mean()
            )
            if not np.isfinite(convergence_residual) or not np.isfinite(convergence_raw):
                continue
            for candidate, direction in ((CANDIDATES[0], 1.0), (CANDIDATES[1], -1.0)):
                rows.append(
                    {
                        "candidate": candidate,
                        "feature_time": timestamp,
                        "community_id": community_id,
                        "community_size": len(members),
                        "long_symbols": "|".join(bottom if direction > 0 else top),
                        "short_symbols": "|".join(top if direction > 0 else bottom),
                        "dispersion": float(dispersion.loc[timestamp]),
                        "dispersion_ratio": float(dispersion.loc[timestamp] / threshold),
                        "residual_gross_4h": float(direction * convergence_residual),
                        "raw_gross_4h": float(direction * convergence_raw),
                    }
                )
            last_by_community[community_id] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def build_v109_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V109Config,
    community_overrides: dict[pd.Timestamp, dict[str, list[str]]] | None = None,
    signal_shift_bars: int = 0,
) -> pd.DataFrame:
    events = []
    for month, context in sorted(contexts.items()):
        communities = (
            community_overrides.get(month, context["communities"])
            if community_overrides is not None
            else context["communities"]
        )
        thresholds = {}
        for community_id, members in communities.items():
            members = [
                symbol for symbol in members if symbol in context["historical_residual"].columns
            ]
            if members:
                dispersion = context["historical_residual"][members].quantile(0.75, axis=1) - context["historical_residual"][members].quantile(0.25, axis=1)
                thresholds[community_id] = float(
                    dispersion.quantile(cfg.dispersion_quantile)
                )
        local = _community_events(
            context, communities, thresholds, cfg, signal_shift_bars
        )
        if not local.empty:
            local["period"] = context["period"]
            events.append(local)
    all_events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    if all_events.empty:
        return all_events
    rows = []
    for (candidate, timestamp), group in all_events.groupby(
        ["candidate", "feature_time"], sort=True
    ):
        chosen = group.sort_values("dispersion_ratio", ascending=False).head(
            cfg.max_communities
        )
        residual_gross = float(chosen["residual_gross_4h"].mean())
        raw_gross = float(chosen["raw_gross_4h"].mean())
        rows.append(
            {
                "candidate": candidate,
                "feature_time": timestamp,
                "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                "period": str(chosen["period"].iloc[0]),
                "community_sleeves": int(len(chosen)),
                "community_ids": "|".join(chosen["community_id"].astype(str)),
                "mean_dispersion_ratio": float(chosen["dispersion_ratio"].mean()),
                "residual_gross_4h": residual_gross,
                "residual_net_4h_20bp": residual_gross - 0.002,
                "residual_net_4h_30bp": residual_gross - 0.003,
                "residual_net_4h_50bp": residual_gross - 0.005,
                "raw_gross_4h": raw_gross,
                "raw_net_4h_20bp": raw_gross - 0.002,
            }
        )
    return pd.DataFrame(rows)


def summarize_v109(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = portfolios if scope == "all" else portfolios[portfolios["period"].eq(scope)]
        for candidate in CANDIDATES:
            sample = scoped[scoped["candidate"].eq(candidate)]
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "portfolio_observations": int(len(sample)),
                    "active_days": int(sample["entry_day"].nunique()),
                    "active_months": int(sample["entry_month"].nunique()),
                    "mean_community_sleeves": float(sample["community_sleeves"].mean()),
                    "mean_residual_gross_4h": float(sample["residual_gross_4h"].mean()),
                    "mean_residual_net_4h_20bp": float(sample["residual_net_4h_20bp"].mean()),
                    "mean_residual_net_4h_30bp": float(sample["residual_net_4h_30bp"].mean()),
                    "mean_residual_net_4h_50bp": float(sample["residual_net_4h_50bp"].mean()),
                    "mean_raw_net_4h_20bp": float(sample["raw_net_4h_20bp"].mean()),
                }
            )
    return pd.DataFrame(rows)


def random_v109_partitions(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V109Config,
) -> dict[pd.Timestamp, dict[str, list[str]]]:
    output = {}
    for month, context in contexts.items():
        real = context["communities"]
        sizes = [len(members) for members in real.values()]
        symbols = sorted(set().union(*real.values()))
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month.month)
        shuffled = list(rng.permutation(symbols))
        cursor = 0
        communities = {}
        for index, size in enumerate(sizes, start=1):
            community_id = f"{month:%Y-%m}:R{iteration:03d}C{index:02d}"
            communities[community_id] = shuffled[cursor : cursor + size]
            cursor += size
        output[month] = communities
    return output


def random_v109_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]], cfg: V109Config
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        portfolios = build_v109_portfolios(
            contexts, cfg, random_v109_partitions(contexts, iteration, cfg)
        )
        means = {}
        for candidate in CANDIDATES:
            sample = portfolios[portfolios["candidate"].eq(candidate)]
            means[candidate] = float(sample["residual_net_4h_20bp"].mean())
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "portfolio_observations": int(len(sample)),
                    "mean_residual_net_4h_20bp": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "portfolio_observations": int(len(portfolios)),
                "mean_residual_net_4h_20bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(sample: pd.DataFrame, cfg: V109Config) -> tuple[float, float]:
    daily = [
        group["residual_net_4h_20bp"].dropna().to_numpy()
        for _, group in sample.groupby("entry_day")
    ]
    daily = [values for values in daily if len(values)]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _positive_community_share(sample: pd.DataFrame) -> float:
    contributions: dict[str, float] = {}
    for row in sample.itertuples(index=False):
        communities = str(row.community_ids).split("|")
        value = float(row.residual_net_4h_20bp) / len(communities)
        for community in communities:
            contributions[community] = contributions.get(community, 0.0) + value
    positive = pd.Series(contributions, dtype=float).clip(lower=0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf


def audit_v109(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V109Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["candidate"].eq("FAMILY_MAX"), "mean_residual_net_4h_20bp"
    ].dropna()
    rows = []
    for candidate in CANDIDATES:
        lookup = {
            row.scope: row
            for row in summary[summary["candidate"].eq(candidate)].itertuples(index=False)
        }
        sample = real[real["candidate"].eq(candidate)].sort_values("feature_time")
        shifted_mean = float(
            shifted.loc[
                shifted["candidate"].eq(candidate), "residual_net_4h_20bp"
            ].mean()
        )
        ci_low, ci_high = _bootstrap(sample, cfg)
        percentile = float(family.lt(lookup["all"].mean_residual_net_4h_20bp).mean())
        positive_month = sample.groupby("entry_month")["residual_net_4h_20bp"].sum().clip(lower=0)
        month_share = float(positive_month.max() / positive_month.sum()) if positive_month.sum() > 0 else np.inf
        community_share = _positive_community_share(sample)
        chrono = [
            float(sample.iloc[index]["residual_net_4h_20bp"].mean())
            for index in np.array_split(np.arange(len(sample)), 5)
            if len(index)
        ]
        gates = {
            "full_observations_100": lookup["all"].portfolio_observations >= 100,
            "validation_observations_25": lookup["validation"].portfolio_observations >= 25,
            "holdout_observations_25": lookup["holdout"].portfolio_observations >= 25,
            "validation_net20_positive": lookup["validation"].mean_residual_net_4h_20bp > 0,
            "holdout_net20_positive": lookup["holdout"].mean_residual_net_4h_20bp > 0,
            "full_net30_positive": lookup["all"].mean_residual_net_4h_30bp > 0,
            "random_family_p90": percentile >= 0.90,
            "beats_shifted": lookup["all"].mean_residual_net_4h_20bp > shifted_mean,
            "bootstrap_lower_positive": ci_low > 0,
            "five_chrono_nonnegative": bool(chrono) and min(chrono) >= 0,
            "month_share_below_35pct": month_share <= 0.35,
            "community_share_below_35pct": community_share <= 0.35,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "verdict": "graph_dispersion_forward_watch_only"
                if eligible
                else "reject_graph_dispersion_candidate",
                "full_net20": lookup["all"].mean_residual_net_4h_20bp,
                "validation_net20": lookup["validation"].mean_residual_net_4h_20bp,
                "holdout_net20": lookup["holdout"].mean_residual_net_4h_20bp,
                "full_net30": lookup["all"].mean_residual_net_4h_30bp,
                "shifted_net20": shifted_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "chronological_means": "|".join(f"{value:.10f}" for value in chrono),
                "max_positive_month_share": month_share,
                "max_positive_community_share": community_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        )
    family_verdict = (
        "graph_dispersion_forward_watch_only"
        if any(row["eligible"] for row in rows)
        else "reject_graph_dispersion_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def write_v109_graph_dispersion_spread(
    cfg: V109Config = V109Config(),
) -> dict[str, Path]:
    panel = load_v109_panel(cfg.panel_path)
    contexts, membership, edges = build_v109_contexts(panel, cfg)
    real = build_v109_portfolios(contexts, cfg)
    shifted = build_v109_portfolios(contexts, cfg, signal_shift_bars=96)
    summary = summarize_v109(real)
    controls = random_v109_controls(contexts, cfg)
    audit = audit_v109(real, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_community_membership.csv",
        "tree_edges": root / "monthly_mst_edges.csv",
        "portfolios": root / "timestamp_spread_portfolios.parquet",
        "shifted": root / "shifted_signal_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_partition_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["membership"], index=False)
    edges.to_csv(outputs["tree_edges"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v10.9 Residual Graph-Community Dispersion Spread",
        "",
        f"Status: `{audit['family_verdict'].iloc[0]}`.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.candidate}: net20={row.full_net20:.4%}, "
            f"validation={row.validation_net20:.4%}, "
            f"holdout={row.holdout_net20:.4%}, "
            f"random percentile={row.random_family_percentile:.1%}."
        )
    lines.extend(["", "No PaperLive or live permission changed."])
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
