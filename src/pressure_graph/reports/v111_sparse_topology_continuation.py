"""Sparse balanced-community topology continuation and horizon audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v109_graph_dispersion_spread import (
    _positive_community_share,
    load_v109_panel,
)
from pressure_graph.reports.v110_balanced_topology_break import (
    PANEL_PATH,
    V110Config,
    _coherence,
    _residualize,
    build_v110_contexts,
    random_v110_partitions,
)


REPORT_ROOT = Path("reports/v11_1_sparse_topology_continuation")
HORIZONS = (1, 2, 4, 8, 12)
CANDIDATE = "STC1_SPARSE_BREAK_CONTINUATION"


@dataclass(frozen=True)
class V111Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    min_samples: int = 500
    community_count: int = 8
    coherence_hours: int = 12
    rank_hours: int = 4
    break_quantile: float = 0.05
    cooldown_hours: int = 4
    severity_quantile: float = 0.80
    min_prior_events: int = 100
    max_communities: int = 3
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260714


def _v110_config(cfg: V111Config) -> V110Config:
    return V110Config(
        panel_path=cfg.panel_path,
        lookback_days=cfg.lookback_days,
        min_samples=cfg.min_samples,
        community_count=cfg.community_count,
        coherence_hours=cfg.coherence_hours,
        rank_hours=cfg.rank_hours,
        break_quantile=cfg.break_quantile,
        cooldown_hours=cfg.cooldown_hours,
        max_communities=cfg.max_communities,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )


def forward_simple_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    output = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)
    for step in range(1, horizon + 1):
        output = output * (1.0 + returns.shift(-step))
    return output - 1.0


def build_v111_contexts(
    panel: pd.DataFrame, cfg: V111Config
) -> tuple[dict[pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    contexts, membership = build_v110_contexts(panel, _v110_config(cfg))
    for context in contexts.values():
        target_return: pd.DataFrame = context["target_return"]
        context["future_raw_by_horizon"] = {}
        context["future_residual_by_horizon"] = {}
        for horizon in HORIZONS:
            raw = forward_simple_returns(target_return, horizon)
            context["future_raw_by_horizon"][horizon] = raw
            context["future_residual_by_horizon"][horizon] = _residualize(
                raw, context["betas"]
            )
    return contexts, membership


def _base_events_for_context(
    context: dict[str, Any],
    communities: dict[str, list[str]],
    cfg: V111Config,
    signal_shift_bars: int,
) -> pd.DataFrame:
    rows = []
    history: pd.DataFrame = context["historical_residual"]
    combined: pd.DataFrame = context["combined_residual"]
    target_times = pd.Index(context["times"])
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    last_by_community: dict[str, pd.Timestamp] = {}
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
        history_std = float(history_coherence.std(ddof=1))
        if not np.isfinite(threshold) or not np.isfinite(history_std) or history_std <= 0:
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
            row: dict[str, Any] = {
                "feature_time": timestamp,
                "month_start": context["month_start"],
                "period": context["period"],
                "community_id": community_id,
                "community_size": len(members),
                "top_symbols": "|".join(top),
                "bottom_symbols": "|".join(bottom),
                "break_severity": float(
                    (threshold - coherence.loc[timestamp]) / history_std
                ),
            }
            for horizon in HORIZONS:
                future: pd.DataFrame = context["future_residual_by_horizon"][horizon]
                if timestamp not in future.index:
                    row[f"top_residual_{horizon}h"] = np.nan
                    row[f"bottom_residual_{horizon}h"] = np.nan
                    row[f"spread_gross_{horizon}h"] = np.nan
                    continue
                top_return = float(future.loc[timestamp, top].mean())
                bottom_return = float(future.loc[timestamp, bottom].mean())
                row[f"top_residual_{horizon}h"] = top_return
                row[f"bottom_residual_{horizon}h"] = bottom_return
                row[f"spread_gross_{horizon}h"] = 0.5 * (top_return - bottom_return)
            rows.append(row)
            last_by_community[community_id] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def build_v111_base_events(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V111Config,
    community_overrides: dict[pd.Timestamp, dict[str, list[str]]] | None = None,
    signal_shift_bars: int = 0,
) -> pd.DataFrame:
    frames = []
    for month, context in sorted(contexts.items()):
        communities = (
            community_overrides.get(month, context["communities"])
            if community_overrides is not None
            else context["communities"]
        )
        local = _base_events_for_context(
            context, communities, cfg, signal_shift_bars
        )
        if not local.empty:
            frames.append(local)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_sparse_events(events: pd.DataFrame, cfg: V111Config) -> pd.DataFrame:
    selected = []
    for raw_month in sorted(events["month_start"].dropna().unique()):
        month = pd.Timestamp(raw_month)
        prior = events.loc[
            events["month_start"].lt(month), "break_severity"
        ].dropna()
        if len(prior) < cfg.min_prior_events:
            continue
        threshold = float(prior.quantile(cfg.severity_quantile))
        local = events[
            events["month_start"].eq(month)
            & events["break_severity"].ge(threshold)
        ].copy()
        local["severity_threshold"] = threshold
        local["prior_event_count"] = len(prior)
        selected.append(local)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def build_v111_portfolios(events: pd.DataFrame, cfg: V111Config) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for timestamp, group in events.groupby("feature_time", sort=True):
        chosen = group.sort_values("break_severity", ascending=False).head(
            cfg.max_communities
        )
        common = {
            "candidate": CANDIDATE,
            "feature_time": timestamp,
            "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
            "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
            "period": str(chosen["period"].iloc[0]),
            "community_sleeves": int(len(chosen)),
            "community_ids": "|".join(chosen["community_id"].astype(str)),
            "mean_break_severity": float(chosen["break_severity"].mean()),
            "severity_threshold": float(chosen["severity_threshold"].max()),
            "prior_event_count": int(chosen["prior_event_count"].min()),
        }
        for horizon in HORIZONS:
            gross = float(chosen[f"spread_gross_{horizon}h"].mean())
            rows.append(
                {
                    **common,
                    "horizon_hours": horizon,
                    "top_residual": float(
                        chosen[f"top_residual_{horizon}h"].mean()
                    ),
                    "bottom_residual": float(
                        chosen[f"bottom_residual_{horizon}h"].mean()
                    ),
                    "spread_gross": gross,
                    "spread_net_20bp": gross - 0.002,
                    "spread_net_30bp": gross - 0.003,
                    "spread_net_50bp": gross - 0.005,
                }
            )
    return pd.DataFrame(rows).dropna(subset=["spread_gross"])


def summarize_v111(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = portfolios if scope == "all" else portfolios[portfolios["period"].eq(scope)]
        for horizon in HORIZONS:
            sample = scoped[scoped["horizon_hours"].eq(horizon)]
            rows.append(
                {
                    "scope": scope,
                    "horizon_hours": horizon,
                    "portfolio_observations": int(len(sample)),
                    "active_days": int(sample["entry_day"].nunique()),
                    "active_months": int(sample["entry_month"].nunique()),
                    "mean_community_sleeves": float(sample["community_sleeves"].mean()),
                    "mean_top_residual": float(sample["top_residual"].mean()),
                    "mean_bottom_residual": float(sample["bottom_residual"].mean()),
                    "mean_spread_gross": float(sample["spread_gross"].mean()),
                    "mean_spread_net_20bp": float(sample["spread_net_20bp"].mean()),
                    "mean_spread_net_30bp": float(sample["spread_net_30bp"].mean()),
                    "mean_spread_net_50bp": float(sample["spread_net_50bp"].mean()),
                }
            )
    return pd.DataFrame(rows)


def random_v111_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]], cfg: V111Config
) -> pd.DataFrame:
    rows = []
    base_cfg = _v110_config(cfg)
    for iteration in range(cfg.random_iterations):
        overrides = random_v110_partitions(contexts, iteration, base_cfg)
        events = build_v111_base_events(contexts, cfg, overrides)
        portfolios = build_v111_portfolios(select_sparse_events(events, cfg), cfg)
        horizon_means = []
        for horizon in HORIZONS:
            sample = portfolios[portfolios["horizon_hours"].eq(horizon)]
            value = float(sample["spread_net_20bp"].mean())
            horizon_means.append(value)
            rows.append(
                {
                    "iteration": iteration,
                    "horizon_hours": horizon,
                    "portfolio_observations": int(len(sample)),
                    "mean_spread_net_20bp": value,
                }
            )
        finite = [value for value in horizon_means if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "horizon_hours": "FAMILY_MAX",
                "portfolio_observations": int(len(portfolios) / len(HORIZONS)),
                "mean_spread_net_20bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(sample: pd.DataFrame, cfg: V111Config) -> tuple[float, float]:
    daily = [
        group["spread_net_20bp"].dropna().to_numpy()
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


def audit_v111(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V111Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["horizon_hours"].eq("FAMILY_MAX"), "mean_spread_net_20bp"
    ].dropna()
    rows = []
    for horizon in HORIZONS:
        lookup = {
            row.scope: row
            for row in summary[summary["horizon_hours"].eq(horizon)].itertuples(
                index=False
            )
        }
        sample = real[real["horizon_hours"].eq(horizon)].sort_values(
            "feature_time"
        )
        shifted_mean = float(
            shifted.loc[
                shifted["horizon_hours"].eq(horizon), "spread_net_20bp"
            ].mean()
        )
        ci_low, ci_high = _bootstrap(sample, cfg)
        percentile = float(family.lt(lookup["all"].mean_spread_net_20bp).mean())
        positive_month = (
            sample.groupby("entry_month")["spread_net_20bp"].sum().clip(lower=0)
        )
        month_share = float(
            positive_month.max() / positive_month.sum()
            if positive_month.sum() > 0
            else np.inf
        )
        community_input = sample.rename(
            columns={"spread_net_20bp": "residual_net_4h_20bp"}
        )
        community_share = _positive_community_share(community_input)
        chrono = [
            float(sample.iloc[index]["spread_net_20bp"].mean())
            for index in np.array_split(np.arange(len(sample)), 5)
            if len(index)
        ]
        gates = {
            "full_observations_100": lookup["all"].portfolio_observations >= 100,
            "validation_observations_25": lookup["validation"].portfolio_observations >= 25,
            "holdout_observations_25": lookup["holdout"].portfolio_observations >= 25,
            "validation_net20_positive": lookup["validation"].mean_spread_net_20bp > 0,
            "holdout_net20_positive": lookup["holdout"].mean_spread_net_20bp > 0,
            "full_net30_positive": lookup["all"].mean_spread_net_30bp > 0,
            "random_family_p90": percentile >= 0.90,
            "beats_shifted": lookup["all"].mean_spread_net_20bp > shifted_mean,
            "bootstrap_lower_positive": ci_low > 0,
            "five_chrono_nonnegative": bool(chrono) and min(chrono) >= 0,
            "month_share_below_35pct": month_share <= 0.35,
            "community_share_below_35pct": community_share <= 0.35,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "horizon_hours": horizon,
                "eligible": eligible,
                "verdict": "retrospective_forward_watch_only"
                if eligible
                else "reject_sparse_horizon",
                "full_gross": lookup["all"].mean_spread_gross,
                "full_net20": lookup["all"].mean_spread_net_20bp,
                "validation_net20": lookup["validation"].mean_spread_net_20bp,
                "holdout_net20": lookup["holdout"].mean_spread_net_20bp,
                "full_net30": lookup["all"].mean_spread_net_30bp,
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
        "retrospective_forward_watch_only"
        if any(row["eligible"] for row in rows)
        else "reject_sparse_topology_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def write_v111_sparse_topology_continuation(
    cfg: V111Config = V111Config(),
) -> dict[str, Path]:
    panel = load_v109_panel(cfg.panel_path)
    contexts, membership = build_v111_contexts(panel, cfg)
    base_events = build_v111_base_events(contexts, cfg)
    sparse_events = select_sparse_events(base_events, cfg)
    real = build_v111_portfolios(sparse_events, cfg)
    shifted_base = build_v111_base_events(contexts, cfg, signal_shift_bars=24)
    shifted = build_v111_portfolios(select_sparse_events(shifted_base, cfg), cfg)
    summary = summarize_v111(real)
    controls = random_v111_controls(contexts, cfg)
    audit = audit_v111(real, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_balanced_membership.csv",
        "base_events": root / "base_topology_events.parquet",
        "sparse_events": root / "sparse_topology_events.parquet",
        "portfolios": root / "horizon_portfolios.parquet",
        "shifted": root / "shifted_horizon_portfolios.parquet",
        "summary": root / "horizon_summary.csv",
        "controls": root / "random_partition_controls.csv",
        "audit": root / "horizon_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["membership"], index=False)
    base_events.to_parquet(outputs["base_events"], index=False)
    sparse_events.to_parquet(outputs["sparse_events"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v11.1 Sparse Topology Continuation",
        "",
        f"Status: `{audit['family_verdict'].iloc[0]}`.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.horizon_hours}h: gross={row.full_gross:.4%}, "
            f"net20={row.full_net20:.4%}, validation={row.validation_net20:.4%}, "
            f"holdout={row.holdout_net20:.4%}, "
            f"random percentile={row.random_family_percentile:.1%}."
        )
    lines.extend(
        [
            "",
            "This is result-informed retrospective research. No PaperLive permission changed.",
        ]
    )
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
