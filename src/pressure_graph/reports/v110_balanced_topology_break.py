"""Balanced residual communities and topology-break bucket-spread audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC, estimate_v106_betas
from pressure_graph.reports.v109_graph_dispersion_spread import (
    PANEL_PATH,
    _bootstrap,
    _period,
    _pivot,
    _positive_community_share,
    load_v109_panel,
)


REPORT_ROOT = Path("reports/v11_0_balanced_topology_break")
CANDIDATES = ("TBR1_TOPOLOGY_REPAIR", "TBR2_BREAK_CONTINUATION")


@dataclass(frozen=True)
class V110Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    min_samples: int = 500
    community_count: int = 8
    coherence_hours: int = 12
    rank_hours: int = 4
    break_quantile: float = 0.05
    max_communities: int = 3
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260714


def _spectral_split(correlation: pd.DataFrame, members: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(members)
    affinity = correlation.loc[ordered, ordered].to_numpy(dtype=float)
    affinity = np.clip(np.nan_to_num(affinity, nan=0.0), 0.0, 1.0)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(len(ordered)) - inverse[:, None] * affinity * inverse[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    fiedler = vectors[:, 1] if len(ordered) > 2 else vectors[:, -1]
    ranked = sorted(zip(fiedler, ordered), key=lambda item: (float(item[0]), item[1]))
    midpoint = len(ranked) // 2
    return sorted(symbol for _, symbol in ranked[:midpoint]), sorted(
        symbol for _, symbol in ranked[midpoint:]
    )


def build_v110_communities(
    residual_history: pd.DataFrame,
    community_count: int = 8,
    min_samples: int = 500,
) -> list[list[str]]:
    eligible = sorted(
        symbol
        for symbol in residual_history.columns
        if symbol != BTC and int(residual_history[symbol].notna().sum()) >= min_samples
    )
    complete = residual_history[eligible].dropna(how="any") if eligible else pd.DataFrame()
    if len(complete) < min_samples or len(eligible) < community_count * 2:
        return []
    correlation = complete.corr().fillna(0.0)
    groups = [eligible]
    while len(groups) < community_count:
        split_index = max(range(len(groups)), key=lambda index: (len(groups[index]), groups[index]))
        source = groups.pop(split_index)
        if len(source) < 4:
            return []
        left, right = _spectral_split(correlation, source)
        groups.extend([left, right])
    return sorted((sorted(group) for group in groups), key=lambda group: (-len(group), group[0]))


def _residualize(returns: pd.DataFrame, betas: pd.Series) -> pd.DataFrame:
    residual = pd.DataFrame(index=returns.index)
    if BTC not in returns.columns:
        return residual
    for symbol in returns.columns:
        if symbol != BTC and symbol in betas.index:
            residual[symbol] = returns[symbol] - float(betas[symbol]) * returns[BTC]
    return residual


def build_v110_contexts(
    panel: pd.DataFrame,
    cfg: V110Config,
) -> tuple[dict[pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    memberships = []
    for raw_month in sorted(panel["month_start"].dropna().unique()):
        month = pd.Timestamp(raw_month)
        history = panel[
            panel["feature_time"].ge(month - pd.Timedelta(days=cfg.lookback_days))
            & panel["feature_time"].lt(month)
            & panel["feature_time"].dt.minute.eq(0)
        ]
        target = panel[
            panel["month_start"].eq(month) & panel["feature_time"].dt.minute.eq(0)
        ]
        if history.empty or target.empty:
            continue
        historical_return = _pivot(history, "ret_1h")
        betas = estimate_v106_betas(historical_return)
        historical_residual = _residualize(historical_return, betas)
        communities = build_v110_communities(
            historical_residual, cfg.community_count, cfg.min_samples
        )
        if not communities:
            continue
        target_return = _pivot(target, "ret_1h")
        target_future = _pivot(target, "future_ret_4h")
        target_residual = _residualize(target_return, betas)
        future_residual = _residualize(target_future, betas)
        symbols = sorted(set().union(*communities) & set(target_residual.columns))
        times = target_residual.index.intersection(future_residual.index)
        named = {}
        for index, members in enumerate(communities, start=1):
            community_id = f"{month:%Y-%m}:BSP{index:02d}"
            local = [symbol for symbol in members if symbol in symbols]
            named[community_id] = local
            for symbol in local:
                memberships.append(
                    {
                        "month_start": month,
                        "community_id": community_id,
                        "symbol": symbol,
                        "community_size": len(local),
                    }
                )
        combined = pd.concat(
            [historical_residual, target_residual.reindex(columns=historical_residual.columns)]
        ).sort_index()
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "times": times,
            "betas": betas,
            "target_return": target_return,
            "communities": named,
            "historical_residual": historical_residual,
            "combined_residual": combined,
            "residual_future": future_residual.reindex(index=times, columns=symbols),
            "raw_future": target_future.reindex(index=times, columns=symbols),
        }
    return contexts, pd.DataFrame(memberships)


def _coherence(
    residual: pd.DataFrame,
    members: list[str],
    scale: pd.Series,
    hours: int,
) -> pd.Series:
    local = residual[members].div(scale[members]).replace([np.inf, -np.inf], np.nan)
    count = local.notna().sum(axis=1)
    total = local.sum(axis=1, skipna=True)
    squared = local.pow(2).sum(axis=1, skipna=True)
    pairwise = (total.pow(2) - squared).div(count * (count - 1))
    pairwise = pairwise.where(count.ge(max(4, len(members) - 1)))
    return pairwise.rolling(hours, min_periods=hours).mean()


def _community_events(
    context: dict[str, Any],
    communities: dict[str, list[str]],
    cfg: V110Config,
    signal_shift_bars: int,
) -> pd.DataFrame:
    rows = []
    history: pd.DataFrame = context["historical_residual"]
    combined: pd.DataFrame = context["combined_residual"]
    target_times = pd.Index(context["times"])
    last_by_community: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for community_id, original_members in communities.items():
        members = [
            symbol
            for symbol in original_members
            if symbol in history.columns and symbol in combined.columns
        ]
        if len(members) < 6:
            continue
        scale = history[members].std(ddof=1).replace(0.0, np.nan)
        history_coherence = _coherence(history, members, scale, cfg.coherence_hours)
        threshold = float(history_coherence.quantile(cfg.break_quantile))
        if not np.isfinite(threshold):
            continue
        coherence = _coherence(combined, members, scale, cfg.coherence_hours)
        rank_signal = combined[members].rolling(
            cfg.rank_hours, min_periods=cfg.rank_hours
        ).sum()
        if signal_shift_bars:
            coherence = coherence.shift(signal_shift_bars)
            rank_signal = rank_signal.shift(signal_shift_bars)
        active = coherence.le(threshold)
        transitions = active & ~active.shift(1, fill_value=False)
        event_times = transitions.index[transitions].intersection(target_times)
        historical_std = float(history_coherence.std(ddof=1))
        for timestamp in event_times:
            last = last_by_community.get(community_id)
            if last is not None and pd.Timestamp(timestamp) - last < cooldown:
                continue
            values = rank_signal.loc[timestamp, members].dropna().sort_values()
            third = len(values) // 3
            if third < 2:
                continue
            bottom = values.head(third).index.tolist()
            top = values.tail(third).index.tolist()
            residual_future: pd.DataFrame = context["residual_future"]
            raw_future: pd.DataFrame = context["raw_future"]
            repair_residual = 0.5 * (
                residual_future.loc[timestamp, bottom].mean()
                - residual_future.loc[timestamp, top].mean()
            )
            repair_raw = 0.5 * (
                raw_future.loc[timestamp, bottom].mean()
                - raw_future.loc[timestamp, top].mean()
            )
            if not np.isfinite(repair_residual) or not np.isfinite(repair_raw):
                continue
            breach = float(
                (threshold - coherence.loc[timestamp]) / historical_std
                if historical_std > 0
                else 0.0
            )
            for candidate, direction in ((CANDIDATES[0], 1.0), (CANDIDATES[1], -1.0)):
                rows.append(
                    {
                        "candidate": candidate,
                        "feature_time": timestamp,
                        "community_id": community_id,
                        "community_size": len(members),
                        "long_symbols": "|".join(bottom if direction > 0 else top),
                        "short_symbols": "|".join(top if direction > 0 else bottom),
                        "coherence": float(coherence.loc[timestamp]),
                        "coherence_threshold": threshold,
                        "break_severity": breach,
                        "residual_gross_4h": float(direction * repair_residual),
                        "raw_gross_4h": float(direction * repair_raw),
                    }
                )
            last_by_community[community_id] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def build_v110_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V110Config,
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
        local = _community_events(context, communities, cfg, signal_shift_bars)
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
        chosen = group.sort_values("break_severity", ascending=False).head(
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
                "mean_break_severity": float(chosen["break_severity"].mean()),
                "residual_gross_4h": residual_gross,
                "residual_net_4h_20bp": residual_gross - 0.002,
                "residual_net_4h_30bp": residual_gross - 0.003,
                "residual_net_4h_50bp": residual_gross - 0.005,
                "raw_gross_4h": raw_gross,
                "raw_net_4h_20bp": raw_gross - 0.002,
            }
        )
    return pd.DataFrame(rows)


def summarize_v110(portfolios: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_residual_net_4h_20bp": float(
                        sample["residual_net_4h_20bp"].mean()
                    ),
                    "mean_residual_net_4h_30bp": float(
                        sample["residual_net_4h_30bp"].mean()
                    ),
                    "mean_residual_net_4h_50bp": float(
                        sample["residual_net_4h_50bp"].mean()
                    ),
                    "mean_raw_net_4h_20bp": float(sample["raw_net_4h_20bp"].mean()),
                }
            )
    return pd.DataFrame(rows)


def random_v110_partitions(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V110Config,
) -> dict[pd.Timestamp, dict[str, list[str]]]:
    output = {}
    for month, context in contexts.items():
        sizes = [len(members) for members in context["communities"].values()]
        symbols = sorted(set().union(*context["communities"].values()))
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


def random_v110_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]], cfg: V110Config
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        portfolios = build_v110_portfolios(
            contexts, cfg, random_v110_partitions(contexts, iteration, cfg)
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


def audit_v110(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V110Config,
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
        positive_month = (
            sample.groupby("entry_month")["residual_net_4h_20bp"].sum().clip(lower=0)
        )
        month_share = float(
            positive_month.max() / positive_month.sum()
            if positive_month.sum() > 0
            else np.inf
        )
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
                "verdict": "topology_break_forward_watch_only"
                if eligible
                else "reject_topology_break_candidate",
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
                "failed_gates": "|".join(name for name, passed in gates.items() if not passed),
            }
        )
    family_verdict = (
        "topology_break_forward_watch_only"
        if any(row["eligible"] for row in rows)
        else "reject_topology_break_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def write_v110_balanced_topology_break(
    cfg: V110Config = V110Config(),
) -> dict[str, Path]:
    panel = load_v109_panel(cfg.panel_path)
    contexts, membership = build_v110_contexts(panel, cfg)
    real = build_v110_portfolios(contexts, cfg)
    shifted = build_v110_portfolios(contexts, cfg, signal_shift_bars=24)
    summary = summarize_v110(real)
    controls = random_v110_controls(contexts, cfg)
    audit = audit_v110(real, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_balanced_membership.csv",
        "portfolios": root / "timestamp_topology_portfolios.parquet",
        "shifted": root / "shifted_signal_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_partition_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["membership"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v11.0 Balanced-Community Topology Break",
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
