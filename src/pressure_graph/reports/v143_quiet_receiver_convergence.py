"""Adaptive quiet-receiver graph convergence spread audit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    estimate_v106_betas,
)
from pressure_graph.reports.v141_directed_taker_flow_graph import (
    load_v141_price_matrices,
)
from pressure_graph.reports.v142_community_volatility_transmission import (
    MEMBERSHIP_PATH,
    V142Config,
    _node_state_panel,
    _residualize,
    build_v142_node_state,
    load_v142_membership,
)


REPORT_ROOT = Path("reports/v14_3_quiet_receiver_convergence")
FINDINGS_PATH = Path(
    "docs/v143_quiet_receiver_convergence_findings_2026_07_15.md"
)
HORIZONS = (4, 8, 12)
CANDIDATES = tuple(
    f"QRC{index}_QUIET_RECEIVER_CONVERGENCE_{horizon}H"
    for index, horizon in enumerate(HORIZONS, start=1)
)
PORTFOLIO_COLUMNS = (
    "candidate",
    "horizon_hours",
    "feature_time",
    "entry_day",
    "entry_month",
    "period",
    "source_community",
    "receiver_communities",
    "source_direction",
    "source_symbol_count",
    "receiver_symbol_count",
    "raw_gross",
    "raw_net_20bp",
    "raw_net_30bp",
    "residual_gross",
    "residual_net_40bp",
)


@dataclass(frozen=True)
class V143Config:
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    lookback_days: int = 30
    z_lookback_hours: int = 7 * 24
    z_min_periods: int = 5 * 24
    minimum_history_days: int = 28
    min_edge_samples: int = 150
    shrinkage_n: int = 150
    followers_per_leader: int = 3
    return_z_threshold: float = 2.0
    volatility_z_threshold: float = 1.0
    breadth_threshold: float = 0.65
    quiet_return_z_max: float = 0.75
    quiet_volatility_z_max: float = 0.0
    min_node_members: int = 5
    min_quiet_followers: int = 2
    selected_followers: int = 2
    min_leg_symbols: int = 5
    return_horizons: tuple[int, ...] = HORIZONS
    delayed_signal_hours: int = 24
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def _period(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if month < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def build_v143_month_edges(
    source_volatility_z: pd.DataFrame,
    future_realized_volatility: pd.DataFrame,
    month_start: pd.Timestamp,
    cfg: V143Config,
) -> pd.DataFrame:
    common = sorted(
        set(source_volatility_z.columns).intersection(
            future_realized_volatility.columns
        )
    )
    eligible = [
        node
        for node in common
        if int(source_volatility_z[node].notna().sum()) >= cfg.min_edge_samples
        and int(future_realized_volatility[node].notna().sum())
        >= cfg.min_edge_samples
    ]
    if len(eligible) < 4:
        return pd.DataFrame()
    joined = pd.concat(
        {
            "source": source_volatility_z[eligible],
            "target": future_realized_volatility[eligible],
        },
        axis=1,
    ).dropna(how="any")
    if len(joined) < cfg.min_edge_samples:
        return pd.DataFrame()
    source_rank = joined["source"].rank(pct=True, method="average").to_numpy(
        dtype=float
    )
    target_rank = joined["target"].rank(pct=True, method="average").to_numpy(
        dtype=float
    )
    source_std = source_rank.std(axis=0, ddof=1)
    target_std = target_rank.std(axis=0, ddof=1)
    valid = (source_std > 0) & (target_std > 0)
    source_norm = np.zeros_like(source_rank)
    target_norm = np.zeros_like(target_rank)
    source_norm[:, valid] = (
        source_rank[:, valid] - source_rank[:, valid].mean(axis=0)
    ) / source_std[valid]
    target_norm[:, valid] = (
        target_rank[:, valid] - target_rank[:, valid].mean(axis=0)
    ) / target_std[valid]
    n = len(joined)
    correlation = source_norm.T @ target_norm / (n - 1)
    shrinkage = np.sqrt(n / (n + cfg.shrinkage_n))
    rows = []
    for leader_index, leader in enumerate(eligible):
        if not valid[leader_index]:
            continue
        for follower_index, follower in enumerate(eligible):
            if leader == follower or not valid[follower_index]:
                continue
            forward = float(correlation[leader_index, follower_index])
            reverse = float(correlation[follower_index, leader_index])
            advantage = forward - abs(reverse)
            if forward <= 0 or advantage <= 0:
                continue
            rows.append(
                {
                    "month_start": month_start,
                    "leader_community": leader,
                    "follower_community": follower,
                    "sample_n": n,
                    "volatility_spearman": forward,
                    "reverse_spearman": reverse,
                    "magnitude_advantage": advantage,
                    "edge_weight": advantage * shrinkage,
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges = edges.sort_values(
        ["leader_community", "edge_weight"], ascending=[True, False]
    )
    edges["edge_rank"] = (
        edges.groupby("leader_community", sort=False).cumcount() + 1
    )
    return edges[edges["edge_rank"].le(cfg.followers_per_leader)].reset_index(
        drop=True
    )


def _future_node_volatility(node_volatility: pd.DataFrame) -> pd.DataFrame:
    shifted = [node_volatility.shift(-offset) for offset in range(1, 5)]
    total = sum(frame.fillna(0.0) for frame in shifted)
    count = sum(frame.notna().astype(int) for frame in shifted)
    return total.div(count.replace(0, np.nan))


def prepare_v143_future_returns(
    raw_future_all: pd.DataFrame,
    betas: pd.Series,
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residualize with BTC present, then restrict to the tradable symbols."""
    raw = raw_future_all.reindex(columns=symbols)
    residual = _residualize(raw_future_all, betas).reindex(columns=symbols)
    return raw, residual


def build_v143_graph_and_contexts(
    prices: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    cfg: V143Config,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    hourly_return = prices["hourly_return"]
    required_horizons = sorted({4, *cfg.return_horizons})
    future_by_horizon = {
        horizon: prices["close"].shift(-4 * horizon).div(prices["close"]).sub(1.0)
        for horizon in required_horizons
    }
    state_cfg = V142Config(
        z_lookback_hours=cfg.z_lookback_hours,
        z_min_periods=cfg.z_min_periods,
        min_node_members=cfg.min_node_members,
    )
    for raw_month, month_membership in membership.groupby("month_start", sort=True):
        month = pd.Timestamp(raw_month)
        community_members = {
            str(community): sorted(group["symbol"].astype(str).unique())
            for community, group in month_membership.groupby(
                "community_id", sort=True
            )
        }
        symbols = sorted(
            set(month_membership["symbol"].astype(str))
            & set(hourly_return.columns)
        )
        beta_history = hourly_return.loc[
            (hourly_return.index >= month - pd.Timedelta(days=cfg.lookback_days))
            & (hourly_return.index < month)
        ]
        betas = estimate_v106_betas(beta_history)
        symbols = [symbol for symbol in symbols if symbol in betas.index]
        state_start = month - pd.Timedelta(
            days=cfg.lookback_days, hours=cfg.z_lookback_hours
        )
        next_month = month + pd.offsets.MonthBegin(1)
        state_index = hourly_return.index[
            (hourly_return.index >= state_start)
            & (hourly_return.index < next_month)
        ]
        raw_1h = hourly_return.reindex(state_index)
        raw_future_4h = future_by_horizon[4].reindex(state_index)
        residual_1h = _residualize(raw_1h, betas).reindex(columns=symbols)
        residual_future_4h = _residualize(
            raw_future_4h, betas
        ).reindex(columns=symbols)
        available_communities = {
            community: [symbol for symbol in members if symbol in symbols]
            for community, members in community_members.items()
        }
        available_communities = {
            community: members
            for community, members in available_communities.items()
            if len(members) >= cfg.min_node_members
        }
        if len(available_communities) < 4:
            continue
        state = build_v142_node_state(
            residual_1h,
            residual_future_4h,
            available_communities,
            state_cfg,
        )
        future_node_volatility = _future_node_volatility(
            state["node_volatility_1h"]
        )
        history_index = state_index[
            (state_index >= month - pd.Timedelta(days=cfg.lookback_days))
            & (state_index < month - pd.Timedelta(hours=4))
            & (state_index.hour % 4 == 0)
        ]
        if len(history_index) < cfg.min_edge_samples:
            continue
        if history_index.max() - history_index.min() < pd.Timedelta(
            days=cfg.minimum_history_days
        ):
            continue
        edges = build_v143_month_edges(
            state["node_volatility_z"].reindex(history_index),
            future_node_volatility.reindex(history_index),
            month,
            cfg,
        )
        if edges.empty:
            continue
        target_index = state_index[
            (state_index >= month) & (state_index < next_month)
        ]
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "communities": sorted(available_communities),
            "community_members": available_communities,
            **{
                name: frame.reindex(target_index)
                for name, frame in state.items()
            },
            "raw_future": {},
            "residual_future": {},
        }
        for horizon in cfg.return_horizons:
            raw_future, residual_future = prepare_v143_future_returns(
                future_by_horizon[horizon].reindex(index=target_index),
                betas,
                symbols,
            )
            contexts[month]["raw_future"][horizon] = raw_future
            contexts[month]["residual_future"][horizon] = residual_future
        edge_frames.append(edges)
    all_edges = (
        pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    )
    return all_edges, contexts


def source_release_signal(
    context: dict[str, Any],
    cfg: V143Config,
    signal_shift_hours: int = 0,
) -> pd.DataFrame:
    return_z: pd.DataFrame = context["node_return_z"]
    active = (
        return_z.abs().ge(cfg.return_z_threshold)
        & context["node_volatility_z"].ge(cfg.volatility_z_threshold)
        & context["node_breadth"].ge(cfg.breadth_threshold)
        & context["node_member_count"].ge(cfg.min_node_members)
    )
    signal = return_z.abs().sub(cfg.return_z_threshold).clip(lower=0.0).mul(
        np.sign(return_z)
    ).where(active, 0.0)
    return signal.shift(signal_shift_hours) if signal_shift_hours else signal


def _quiet_receivers(
    context: dict[str, Any],
    timestamp: pd.Timestamp,
    cfg: V143Config,
) -> set[str]:
    quiet = (
        context["node_return_z"].loc[timestamp].abs().le(cfg.quiet_return_z_max)
        & context["node_volatility_z"].loc[timestamp].le(
            cfg.quiet_volatility_z_max
        )
        & context["node_member_count"].loc[timestamp].ge(cfg.min_node_members)
    )
    return set(quiet.index[quiet.fillna(False)])


def build_v143_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V143Config,
    signal_shift_hours: int = 0,
) -> pd.DataFrame:
    rows = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        signal = source_release_signal(
            context, cfg, signal_shift_hours=signal_shift_hours
        )
        candidate_times = signal.index[signal.ne(0).any(axis=1)]
        for horizon, candidate in zip(HORIZONS, CANDIDATES, strict=True):
            cooldown = pd.Timedelta(hours=horizon)
            for timestamp in candidate_times:
                timestamp = pd.Timestamp(timestamp)
                last = last_by_candidate.get(candidate)
                if last is not None and timestamp - last < cooldown:
                    continue
                quiet = _quiet_receivers(context, timestamp, cfg)
                opportunities = []
                for source in signal.columns[signal.loc[timestamp].ne(0)]:
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
                )
                source_residual = pd.to_numeric(
                    residual_future.loc[timestamp, source_members], errors="coerce"
                )
                receiver_raw = pd.to_numeric(
                    raw_future.loc[timestamp, receiver_members], errors="coerce"
                )
                receiver_residual = pd.to_numeric(
                    residual_future.loc[timestamp, receiver_members], errors="coerce"
                )
                source_finite = source_raw.notna() & source_residual.notna()
                receiver_finite = receiver_raw.notna() & receiver_residual.notna()
                if (
                    int(source_finite.sum()) < cfg.min_leg_symbols
                    or int(receiver_finite.sum()) < cfg.min_leg_symbols
                ):
                    continue
                direction = float(np.sign(signal.loc[timestamp, source]))
                raw_gross = 0.5 * direction * float(
                    receiver_raw[receiver_finite].mean()
                ) - 0.5 * direction * float(source_raw[source_finite].mean())
                residual_gross = 0.5 * direction * float(
                    receiver_residual[receiver_finite].mean()
                ) - 0.5 * direction * float(
                    source_residual[source_finite].mean()
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
                        "source_direction": direction,
                        "source_symbol_count": int(source_finite.sum()),
                        "receiver_symbol_count": int(receiver_finite.sum()),
                        "raw_gross": raw_gross,
                        "raw_net_20bp": raw_gross - 0.002,
                        "raw_net_30bp": raw_gross - 0.003,
                        "residual_gross": residual_gross,
                        "residual_net_40bp": residual_gross - 0.004,
                    }
                )
                last_by_candidate[candidate] = timestamp
    return pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)


def summarize_v143(portfolios: pd.DataFrame) -> pd.DataFrame:
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


def randomize_v143_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V143Config,
) -> pd.DataFrame:
    rows = []
    for raw_month, month_edges in edges.groupby("month_start", sort=True):
        month = pd.Timestamp(raw_month)
        communities = contexts[month]["communities"]
        month_code = month.year * 12 + month.month
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month_code)
        for leader, local in month_edges.groupby("leader_community", sort=False):
            leader = str(leader)
            choices = [node for node in communities if node != leader]
            source_rows = local.sort_values("edge_rank")
            take = min(len(source_rows), len(choices))
            followers = rng.choice(choices, size=take, replace=False)
            for follower, row in zip(
                followers,
                source_rows.head(take).itertuples(index=False),
                strict=True,
            ):
                payload = row._asdict()
                payload["follower_community"] = str(follower)
                rows.append(payload)
    return pd.DataFrame(rows)


def reverse_v143_edges(edges: pd.DataFrame, cfg: V143Config) -> pd.DataFrame:
    out = edges.copy()
    out[["leader_community", "follower_community"]] = out[
        ["follower_community", "leader_community"]
    ].to_numpy()
    out = out.sort_values(
        ["month_start", "leader_community", "edge_weight"],
        ascending=[True, True, False],
    ).drop_duplicates(
        ["month_start", "leader_community", "follower_community"], keep="first"
    )
    out["edge_rank"] = (
        out.groupby(["month_start", "leader_community"], sort=False).cumcount() + 1
    )
    return out[out["edge_rank"].le(cfg.followers_per_leader)].reset_index(
        drop=True
    )


def random_v143_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V143Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v143_edges(edges, contexts, iteration, cfg)
        portfolios = build_v143_portfolios(contexts, randomized, cfg)
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
    cfg: V143Config,
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


def audit_v143(
    real: pd.DataFrame,
    delayed: pd.DataFrame,
    reversed_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V143Config,
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
            "full_observations_120": lookup["all"].observations >= 120,
            "validation_observations_30": lookup["validation"].observations >= 30,
            "holdout_observations_30": lookup["holdout"].observations >= 30,
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
                else "reject_quiet_receiver_candidate",
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
        else "reject_quiet_receiver_family"
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
        "# v14.3 Quiet-Receiver Graph Convergence Findings",
        "",
        f"Verdict: `{audit['family_verdict'].iloc[0]}`.",
        "",
        "This is an adaptive historical follow-up; a pass can only be provisional.",
        "",
        "## Candidate audit",
        "",
        audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Graph months: `{edges['month_start'].nunique()}`; edges: `{len(edges)}`; "
        f"observations: `{len(portfolios)}`.",
        "",
        "No PaperLive, leverage, or live-order permission changed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v143_quiet_receiver_convergence(
    cfg: V143Config = V143Config(),
) -> dict[str, Path]:
    membership = load_v142_membership(V142Config(membership_path=cfg.membership_path))
    prices = load_v141_price_matrices()
    edges, contexts = build_v143_graph_and_contexts(prices, membership, cfg)
    if edges.empty or not contexts:
        raise RuntimeError("v14.3 produced no causal monthly graph contexts")
    real = build_v143_portfolios(contexts, edges, cfg)
    delayed = build_v143_portfolios(
        contexts,
        edges,
        cfg,
        signal_shift_hours=cfg.delayed_signal_hours,
    )
    reversed_portfolios = build_v143_portfolios(
        contexts, reverse_v143_edges(edges, cfg), cfg
    )
    summary = summarize_v143(real)
    controls = random_v143_controls(contexts, edges, cfg)
    audit = audit_v143(
        real, delayed, reversed_portfolios, summary, controls, cfg
    )
    node_states = _node_state_panel(contexts)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_membership.csv",
        "node_states": root / "community_node_states.parquet",
        "edges": root / "volatility_receiver_edges.csv",
        "portfolios": root / "convergence_spread_portfolios.parquet",
        "delayed": root / "delayed_source_portfolios.parquet",
        "reversed": root / "reversed_edge_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    membership.to_csv(outputs["membership"], index=False)
    node_states.to_parquet(outputs["node_states"], index=False)
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
                "adaptive_followup": True,
                "candidate_family": list(CANDIDATES),
                "graph_months": edges["month_start"].nunique(),
                "edges": len(edges),
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
