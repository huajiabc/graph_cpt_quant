"""Adaptive 18/24-hour extension of bearish community convergence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v141_directed_taker_flow_graph import (
    load_v141_price_matrices,
)
from pressure_graph.reports.v142_community_volatility_transmission import (
    V142Config,
    load_v142_membership,
)
from pressure_graph.reports.v143_quiet_receiver_convergence import (
    MEMBERSHIP_PATH,
    V143Config,
    _quiet_receivers,
    build_v143_graph_and_contexts,
    randomize_v143_edges,
    reverse_v143_edges,
    source_release_signal,
)


REPORT_ROOT = Path("reports/v14_5_bearish_convergence_horizon_extension")
FINDINGS_PATH = Path(
    "docs/v145_bearish_convergence_horizon_extension_findings_2026_07_15.md"
)
HORIZONS = (18, 24)
CANDIDATES = tuple(
    f"BCH{index}_BEARISH_CONVERGENCE_{horizon}H"
    for index, horizon in enumerate(HORIZONS, start=1)
)


@dataclass(frozen=True)
class V145Config:
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    min_quiet_followers: int = 2
    selected_followers: int = 2
    min_leg_symbols: int = 5
    delayed_signal_hours: int = 24
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def bearish_convergence_return(
    source_return: float,
    receiver_return: float,
) -> float:
    return 0.5 * (source_return - receiver_return)


def build_v145_portfolios(
    contexts: dict[pd.Timestamp, dict],
    edges: pd.DataFrame,
    graph_cfg: V143Config,
    cfg: V145Config,
    signal_shift_hours: int = 0,
) -> pd.DataFrame:
    columns = (
        "candidate",
        "horizon_hours",
        "feature_time",
        "entry_day",
        "entry_month",
        "period",
        "source_community",
        "receiver_communities",
        "source_symbol_count",
        "receiver_symbol_count",
        "raw_gross",
        "raw_net_20bp",
        "raw_net_30bp",
        "residual_gross",
        "residual_net_40bp",
    )
    rows = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        signal = source_release_signal(
            context, graph_cfg, signal_shift_hours=signal_shift_hours
        )
        candidate_times = signal.index[signal.lt(0).any(axis=1)]
        for horizon, candidate in zip(HORIZONS, CANDIDATES, strict=True):
            cooldown = pd.Timedelta(hours=horizon)
            for timestamp in candidate_times:
                timestamp = pd.Timestamp(timestamp)
                last = last_by_candidate.get(candidate)
                if last is not None and timestamp - last < cooldown:
                    continue
                quiet = _quiet_receivers(context, timestamp, graph_cfg)
                opportunities = []
                for source in signal.columns[signal.loc[timestamp].lt(0)]:
                    local = month_edges[
                        month_edges["leader_community"].eq(source)
                        & month_edges["follower_community"].isin(quiet)
                    ].sort_values("edge_weight", ascending=False)
                    if len(local) < cfg.min_quiet_followers:
                        continue
                    selected = local.head(cfg.selected_followers)
                    score = abs(float(signal.loc[timestamp, source])) * float(
                        selected["edge_weight"].mean()
                    )
                    opportunities.append((score, str(source), selected))
                if not opportunities:
                    continue
                _, source, selected_edges = max(
                    opportunities, key=lambda item: item[0]
                )
                receivers = selected_edges["follower_community"].astype(str).tolist()
                source_members = context["community_members"][source]
                receiver_members = sorted(
                    {
                        symbol
                        for receiver in receivers
                        for symbol in context["community_members"][receiver]
                    }
                )
                raw_future = context["raw_future"][horizon]
                residual_future = context["residual_future"][horizon]
                source_raw = pd.to_numeric(
                    raw_future.loc[timestamp, source_members], errors="coerce"
                ).dropna()
                source_residual = pd.to_numeric(
                    residual_future.loc[timestamp, source_members], errors="coerce"
                ).dropna()
                receiver_raw = pd.to_numeric(
                    raw_future.loc[timestamp, receiver_members], errors="coerce"
                ).dropna()
                receiver_residual = pd.to_numeric(
                    residual_future.loc[timestamp, receiver_members], errors="coerce"
                ).dropna()
                source_valid = source_raw.index.intersection(source_residual.index)
                receiver_valid = receiver_raw.index.intersection(
                    receiver_residual.index
                )
                if (
                    len(source_valid) < cfg.min_leg_symbols
                    or len(receiver_valid) < cfg.min_leg_symbols
                ):
                    continue
                raw_gross = bearish_convergence_return(
                    float(source_raw[source_valid].mean()),
                    float(receiver_raw[receiver_valid].mean()),
                )
                residual_gross = bearish_convergence_return(
                    float(source_residual[source_valid].mean()),
                    float(receiver_residual[receiver_valid].mean()),
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "horizon_hours": horizon,
                        "feature_time": timestamp,
                        "entry_day": timestamp.strftime("%Y-%m-%d"),
                        "entry_month": timestamp.strftime("%Y-%m"),
                        "period": context["period"],
                        "source_community": source,
                        "receiver_communities": "|".join(receivers),
                        "source_symbol_count": len(source_valid),
                        "receiver_symbol_count": len(receiver_valid),
                        "raw_gross": raw_gross,
                        "raw_net_20bp": raw_gross - 0.002,
                        "raw_net_30bp": raw_gross - 0.003,
                        "residual_gross": residual_gross,
                        "residual_net_40bp": residual_gross - 0.004,
                    }
                )
                last_by_candidate[candidate] = timestamp
    return pd.DataFrame(rows, columns=columns)


def summarize_v145(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = (
            portfolios if scope == "all" else portfolios[portfolios["period"].eq(scope)]
        )
        for candidate in CANDIDATES:
            sample = scoped[scoped["candidate"].eq(candidate)]
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "observations": len(sample),
                    "active_days": sample["entry_day"].nunique(),
                    "active_months": sample["entry_month"].nunique(),
                    "mean_raw_gross": sample["raw_gross"].mean(),
                    "mean_raw_net20": sample["raw_net_20bp"].mean(),
                    "mean_raw_net30": sample["raw_net_30bp"].mean(),
                    "mean_residual_gross": sample["residual_gross"].mean(),
                    "mean_residual_net40": sample["residual_net_40bp"].mean(),
                }
            )
    return pd.DataFrame(rows)


def random_v145_controls(
    contexts: dict[pd.Timestamp, dict],
    edges: pd.DataFrame,
    graph_cfg: V143Config,
    cfg: V145Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v143_edges(edges, contexts, iteration, graph_cfg)
        portfolios = build_v145_portfolios(
            contexts, randomized, graph_cfg, cfg
        )
        means = {}
        for candidate in CANDIDATES:
            sample = portfolios[portfolios["candidate"].eq(candidate)]
            mean = float(sample["residual_net_40bp"].mean())
            means[candidate] = mean
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "observations": len(sample),
                    "mean_residual_net40": mean,
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "observations": len(portfolios),
                "mean_residual_net40": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(
    sample: pd.DataFrame,
    cfg: V145Config,
) -> tuple[float, float]:
    daily = [
        group["residual_net_40bp"].dropna().to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    daily = [values for values in daily if len(values)]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(
            float(np.mean(np.concatenate([daily[index] for index in chosen])))
        )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _positive_share(values: pd.Series) -> float:
    positive = values.clip(lower=0.0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf


def audit_v145(
    real: pd.DataFrame,
    delayed: pd.DataFrame,
    reversed_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V145Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["candidate"].eq("FAMILY_MAX"), "mean_residual_net40"
    ].dropna()
    rows = []
    for candidate in CANDIDATES:
        lookup = {
            row.scope: row
            for row in summary[summary["candidate"].eq(candidate)].itertuples(
                index=False
            )
        }
        sample = real[real["candidate"].eq(candidate)]
        real_mean = float(lookup["all"].mean_residual_net40)
        delayed_mean = float(
            delayed.loc[
                delayed["candidate"].eq(candidate), "residual_net_40bp"
            ].mean()
        )
        reversed_mean = float(
            reversed_portfolios.loc[
                reversed_portfolios["candidate"].eq(candidate),
                "residual_net_40bp",
            ].mean()
        )
        ci_low, ci_high = _bootstrap_ci(sample, cfg)
        percentile = float(family.lt(real_mean).mean()) if len(family) else np.nan
        month_share = _positive_share(
            sample.groupby("entry_month")["residual_net_40bp"].sum()
        )
        worst_period = float(
            sample.groupby("period")["residual_net_40bp"].mean().min()
        )
        gates = {
            "full_observations_50": lookup["all"].observations >= 50,
            "validation_observations_12": lookup["validation"].observations >= 12,
            "holdout_observations_12": lookup["holdout"].observations >= 12,
            "development_residual_net40_positive": lookup[
                "development"
            ].mean_residual_net40
            > 0,
            "validation_residual_net40_positive": lookup[
                "validation"
            ].mean_residual_net40
            > 0,
            "holdout_residual_net40_positive": lookup["holdout"].mean_residual_net40
            > 0,
            "development_raw_net20_positive": lookup["development"].mean_raw_net20
            > 0,
            "validation_raw_net20_positive": lookup["validation"].mean_raw_net20
            > 0,
            "holdout_raw_net20_positive": lookup["holdout"].mean_raw_net20 > 0,
            "full_raw_net30_positive": lookup["all"].mean_raw_net30 > 0,
            "bootstrap_lower_positive": ci_low > 0,
            "random_family_p95": percentile >= 0.95,
            "beats_reversed": real_mean > reversed_mean,
            "beats_delayed": real_mean > delayed_mean,
            "month_share_below_35pct": month_share <= 0.35,
            "worst_period_above_minus40bp": worst_period >= -0.004,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "verdict": "provisional_forward_shadow_only"
                if eligible
                else "reject_bearish_horizon_candidate",
                "full_residual_net40": real_mean,
                "development_residual_net40": lookup[
                    "development"
                ].mean_residual_net40,
                "validation_residual_net40": lookup[
                    "validation"
                ].mean_residual_net40,
                "holdout_residual_net40": lookup["holdout"].mean_residual_net40,
                "full_raw_net20": lookup["all"].mean_raw_net20,
                "full_raw_net30": lookup["all"].mean_raw_net30,
                "delayed_residual_net40": delayed_mean,
                "reversed_residual_net40": reversed_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "max_positive_month_share": month_share,
                "worst_period_mean": worst_period,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        )
    family_verdict = (
        "provisional_forward_shadow_only"
        if any(row["eligible"] for row in rows)
        else "reject_bearish_horizon_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def _write_findings(
    path: Path,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    edges: pd.DataFrame,
    portfolios: pd.DataFrame,
) -> None:
    lines = [
        "# v14.5 Bearish Convergence Horizon-Extension Findings",
        "",
        f"Verdict: `{audit['family_verdict'].iloc[0]}`.",
        "",
        "This is an adaptive horizon extension and cannot independently establish alpha.",
        "",
        "## Audit",
        "",
        audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Graph months: `{edges['month_start'].nunique()}`; observations: "
        f"`{len(portfolios)}`.",
        "",
        "No PaperLive, leverage, or live-order permission changed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v145_bearish_convergence_horizon_extension(
    cfg: V145Config = V145Config(),
) -> dict[str, Path]:
    membership = load_v142_membership(V142Config(membership_path=cfg.membership_path))
    prices = load_v141_price_matrices()
    graph_cfg = V143Config(
        membership_path=cfg.membership_path,
        return_horizons=HORIZONS,
    )
    edges, contexts = build_v143_graph_and_contexts(prices, membership, graph_cfg)
    real = build_v145_portfolios(contexts, edges, graph_cfg, cfg)
    delayed = build_v145_portfolios(
        contexts,
        edges,
        graph_cfg,
        cfg,
        signal_shift_hours=cfg.delayed_signal_hours,
    )
    reversed_portfolios = build_v145_portfolios(
        contexts,
        reverse_v143_edges(edges, graph_cfg),
        graph_cfg,
        cfg,
    )
    summary = summarize_v145(real)
    controls = random_v145_controls(contexts, edges, graph_cfg, cfg)
    audit = audit_v145(
        real, delayed, reversed_portfolios, summary, controls, cfg
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "edges": root / "volatility_receiver_edges.csv",
        "portfolios": root / "bearish_convergence_portfolios.parquet",
        "delayed": root / "delayed_source_portfolios.parquet",
        "reversed": root / "reversed_edge_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    edges.to_csv(outputs["edges"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    delayed.to_parquet(outputs["delayed"], index=False)
    reversed_portfolios.to_parquet(outputs["reversed"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "adaptive_horizon_extension": True,
                "candidate_family": list(CANDIDATES),
                "observations": len(real),
                "random_iterations": cfg.random_iterations,
                "family_verdict": audit["family_verdict"].iloc[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(cfg.findings_path, audit, summary, edges, real)
    return outputs
