"""Semi-high-frequency community volatility-transmission graph audit."""
from __future__ import annotations

import json
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
from pressure_graph.reports.v141_directed_taker_flow_graph import (
    MEMBERSHIP_PATH,
    load_v141_price_matrices,
)


REPORT_ROOT = Path("reports/v14_2_community_volatility_transmission")
FINDINGS_PATH = Path(
    "docs/v142_community_volatility_transmission_findings_2026_07_15.md"
)
CANDIDATES = (
    "CVG1_COMMUNITY_VOL_CONTINUATION",
    "CVG2_COMMUNITY_VOL_REVERSAL",
)


@dataclass(frozen=True)
class V142Config:
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    lookback_days: int = 30
    z_lookback_hours: int = 7 * 24
    z_min_periods: int = 5 * 24
    minimum_history_days: int = 28
    min_edge_samples: int = 150
    shrinkage_n: int = 150
    leaders_per_follower: int = 2
    return_z_threshold: float = 2.0
    volatility_z_threshold: float = 1.0
    breadth_threshold: float = 0.65
    min_node_members: int = 5
    min_follower_communities: int = 2
    max_follower_communities: int = 2
    min_portfolio_symbols: int = 5
    cooldown_hours: int = 4
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


def load_v142_membership(cfg: V142Config = V142Config()) -> pd.DataFrame:
    membership = pd.read_csv(cfg.membership_path)
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    membership["symbol"] = membership["symbol"].astype(str)
    membership["community_id"] = membership["community_id"].astype(str)
    return (
        membership[membership["symbol"].ne(BTC)]
        .dropna(subset=["month_start", "community_id", "symbol"])
        .drop_duplicates(["month_start", "community_id", "symbol"], keep="last")
        .sort_values(["month_start", "community_id", "symbol"])
        .reset_index(drop=True)
    )


def causal_zscore(
    values: pd.Series,
    lookback: int,
    min_periods: int,
) -> pd.Series:
    shifted = values.shift(1)
    prior = shifted.rolling(lookback, min_periods=min_periods)
    return ((values - prior.mean()) / prior.std(ddof=1)).clip(-5.0, 5.0)


def _residualize(
    values: pd.DataFrame,
    betas: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=values.index)
    if BTC not in values.columns:
        return out
    for symbol, beta in betas.items():
        if symbol == BTC or symbol not in values.columns:
            continue
        out[str(symbol)] = values[symbol] - float(beta) * values[BTC]
    return out


def build_v142_node_state(
    residual_1h: pd.DataFrame,
    residual_future_4h: pd.DataFrame,
    community_members: dict[str, list[str]],
    cfg: V142Config,
) -> dict[str, pd.DataFrame]:
    columns = sorted(community_members)
    node_return = pd.DataFrame(index=residual_1h.index, columns=columns, dtype=float)
    node_volatility = pd.DataFrame(
        index=residual_1h.index, columns=columns, dtype=float
    )
    node_breadth = pd.DataFrame(index=residual_1h.index, columns=columns, dtype=float)
    node_members = pd.DataFrame(index=residual_1h.index, columns=columns, dtype=float)
    node_future = pd.DataFrame(
        index=residual_future_4h.index, columns=columns, dtype=float
    )
    for community, members in community_members.items():
        members = [symbol for symbol in members if symbol in residual_1h.columns]
        if not members:
            continue
        member_returns = residual_1h[members]
        median_return = member_returns.median(axis=1, skipna=True)
        positive_breadth = member_returns.gt(0).sum(axis=1).div(
            member_returns.notna().sum(axis=1).replace(0, np.nan)
        )
        negative_breadth = member_returns.lt(0).sum(axis=1).div(
            member_returns.notna().sum(axis=1).replace(0, np.nan)
        )
        node_return[community] = median_return
        node_volatility[community] = member_returns.abs().median(
            axis=1, skipna=True
        )
        node_breadth[community] = positive_breadth.where(
            median_return.ge(0), negative_breadth
        )
        node_members[community] = member_returns.notna().sum(axis=1)
        future_members = [
            symbol for symbol in members if symbol in residual_future_4h.columns
        ]
        if future_members:
            node_future[community] = residual_future_4h[future_members].median(
                axis=1, skipna=True
            )
    return_z = pd.DataFrame(index=node_return.index, columns=columns, dtype=float)
    volatility_z = pd.DataFrame(
        index=node_volatility.index, columns=columns, dtype=float
    )
    for community in columns:
        return_z[community] = causal_zscore(
            node_return[community], cfg.z_lookback_hours, cfg.z_min_periods
        )
        volatility_z[community] = causal_zscore(
            node_volatility[community], cfg.z_lookback_hours, cfg.z_min_periods
        )
    return {
        "node_return_1h": node_return,
        "node_volatility_1h": node_volatility,
        "node_return_z": return_z,
        "node_volatility_z": volatility_z,
        "node_breadth": node_breadth,
        "node_member_count": node_members,
        "node_future_residual_4h": node_future,
    }


def build_v142_month_edges(
    source_z: pd.DataFrame,
    residual_target: pd.DataFrame,
    month_start: pd.Timestamp,
    cfg: V142Config,
) -> pd.DataFrame:
    common = sorted(set(source_z.columns).intersection(residual_target.columns))
    eligible = [
        node
        for node in common
        if int(source_z[node].notna().sum()) >= cfg.min_edge_samples
        and int(residual_target[node].notna().sum()) >= cfg.min_edge_samples
    ]
    if len(eligible) < 4:
        return pd.DataFrame()
    joined = pd.concat(
        {"source": source_z[eligible], "target": residual_target[eligible]},
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
            advantage = abs(forward) - abs(reverse)
            if advantage <= 0 or forward == 0:
                continue
            candidate = CANDIDATES[0] if forward > 0 else CANDIDATES[1]
            rows.append(
                {
                    "month_start": month_start,
                    "candidate": candidate,
                    "leader_community": leader,
                    "follower_community": follower,
                    "relation_sign": 1.0 if forward > 0 else -1.0,
                    "sample_n": n,
                    "source_target_spearman": forward,
                    "reverse_spearman": reverse,
                    "magnitude_advantage": advantage,
                    "edge_weight": advantage * shrinkage,
                }
            )
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges
    edges = edges.sort_values(
        ["candidate", "follower_community", "edge_weight"],
        ascending=[True, True, False],
    )
    edges["edge_rank"] = (
        edges.groupby(["candidate", "follower_community"], sort=False).cumcount()
        + 1
    )
    return edges[edges["edge_rank"].le(cfg.leaders_per_follower)].reset_index(
        drop=True
    )


def build_v142_graph_and_contexts(
    prices: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    cfg: V142Config,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    hourly_return = prices["hourly_return"]
    hourly_future_4h = prices["future_4h"].reindex(hourly_return.index)
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
        if len(symbols) < cfg.min_portfolio_symbols:
            continue
        state_start = month - pd.Timedelta(
            days=cfg.lookback_days, hours=cfg.z_lookback_hours
        )
        next_month = month + pd.offsets.MonthBegin(1)
        state_index = hourly_return.index[
            (hourly_return.index >= state_start)
            & (hourly_return.index < next_month)
        ]
        raw_1h = hourly_return.reindex(state_index)
        raw_future_4h = hourly_future_4h.reindex(state_index)
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
            cfg,
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
        edges = build_v142_month_edges(
            state["node_return_z"].reindex(history_index),
            state["node_future_residual_4h"].reindex(history_index),
            month,
            cfg,
        )
        if edges.empty:
            continue
        target_index = state_index[
            (state_index >= month) & (state_index < next_month)
        ]
        context_state = {
            name: frame.reindex(target_index)
            for name, frame in state.items()
        }
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "communities": sorted(available_communities),
            "community_members": available_communities,
            **context_state,
            "raw_future_4h": raw_future_4h.reindex(
                index=target_index, columns=symbols
            ),
            "residual_future_4h": residual_future_4h.reindex(
                index=target_index, columns=symbols
            ),
        }
        edge_frames.append(edges)
    all_edges = (
        pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    )
    return all_edges, contexts


def community_pressure_matrices(
    context: dict[str, Any],
    edges: pd.DataFrame,
    cfg: V142Config,
    signal_shift_hours: int = 0,
) -> dict[str, pd.DataFrame]:
    communities: list[str] = context["communities"]
    return_z: pd.DataFrame = context["node_return_z"].reindex(
        columns=communities
    )
    volatility_z: pd.DataFrame = context["node_volatility_z"].reindex(
        columns=communities
    )
    breadth: pd.DataFrame = context["node_breadth"].reindex(
        columns=communities
    )
    member_count: pd.DataFrame = context["node_member_count"].reindex(
        columns=communities
    )
    active = (
        return_z.abs().ge(cfg.return_z_threshold)
        & volatility_z.ge(cfg.volatility_z_threshold)
        & breadth.ge(cfg.breadth_threshold)
        & member_count.ge(cfg.min_node_members)
    )
    signed_excess = return_z.abs().sub(cfg.return_z_threshold).clip(
        lower=0.0
    ).mul(np.sign(return_z)).where(active, 0.0)
    if signal_shift_hours:
        signed_excess = signed_excess.shift(signal_shift_hours)
    outputs = {
        candidate: pd.DataFrame(
            0.0, index=return_z.index, columns=communities
        )
        for candidate in CANDIDATES
    }
    for candidate in CANDIDATES:
        candidate_edges = edges[edges["candidate"].eq(candidate)]
        for follower, group in candidate_edges.groupby(
            "follower_community", sort=False
        ):
            follower = str(follower)
            if follower not in communities:
                continue
            local = (
                group[group["leader_community"].astype(str).isin(communities)]
                .sort_values("edge_rank")
                .drop_duplicates("leader_community", keep="first")
            )
            if local.empty:
                continue
            leaders = local["leader_community"].astype(str).tolist()
            weights = pd.to_numeric(local["edge_weight"], errors="coerce").to_numpy(
                dtype=float
            )
            relation = pd.to_numeric(
                local["relation_sign"], errors="coerce"
            ).to_numpy(dtype=float)
            values = signed_excess[leaders].fillna(0.0).to_numpy(dtype=float)
            active_values = values != 0
            denominator = np.where(
                active_values, weights[None, :], 0.0
            ).sum(axis=1)
            pressure = np.divide(
                np.where(
                    active_values,
                    values * weights[None, :] * relation[None, :],
                    0.0,
                ).sum(axis=1),
                denominator,
                out=np.zeros(len(values)),
                where=denominator > 0,
            )
            outputs[candidate][follower] = pressure
    return outputs


def build_v142_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V142Config,
    signal_shift_hours: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        signals = community_pressure_matrices(
            context,
            month_edges,
            cfg,
            signal_shift_hours=signal_shift_hours,
        )
        for candidate, pressure in signals.items():
            eligible_times = pressure.index[
                pressure.ne(0).sum(axis=1).ge(cfg.min_follower_communities)
            ]
            for timestamp in eligible_times:
                timestamp = pd.Timestamp(timestamp)
                last = last_by_candidate.get(candidate)
                if last is not None and timestamp - last < cooldown:
                    continue
                selected_communities = (
                    pressure.loc[timestamp]
                    .loc[lambda values: values.ne(0)]
                    .abs()
                    .sort_values(ascending=False)
                    .head(cfg.max_follower_communities)
                    .index.tolist()
                )
                symbol_directions: dict[str, float] = {}
                for community in selected_communities:
                    direction = float(np.sign(pressure.loc[timestamp, community]))
                    for symbol in context["community_members"][community]:
                        symbol_directions[symbol] = direction
                symbols = sorted(symbol_directions)
                raw = pd.to_numeric(
                    context["raw_future_4h"].loc[timestamp, symbols],
                    errors="coerce",
                )
                residual = pd.to_numeric(
                    context["residual_future_4h"].loc[timestamp, symbols],
                    errors="coerce",
                )
                directions = pd.Series(symbol_directions, dtype=float).reindex(
                    symbols
                )
                finite = raw.notna() & residual.notna() & directions.notna()
                if int(finite.sum()) < cfg.min_portfolio_symbols:
                    continue
                symbols = list(pd.Index(symbols)[finite.to_numpy()])
                signed_raw = raw[finite] * directions[finite]
                signed_residual = residual[finite] * directions[finite]
                raw_gross = float(signed_raw.mean())
                residual_gross = float(signed_residual.mean())
                rows.append(
                    {
                        "candidate": candidate,
                        "feature_time": timestamp,
                        "entry_day": timestamp.strftime("%Y-%m-%d"),
                        "entry_month": timestamp.strftime("%Y-%m"),
                        "period": context["period"],
                        "follower_community_count": len(selected_communities),
                        "follower_communities": "|".join(selected_communities),
                        "portfolio_symbols": "|".join(
                            f"{'+' if symbol_directions[symbol] > 0 else '-'}{symbol}"
                            for symbol in symbols
                        ),
                        "portfolio_symbol_count": len(symbols),
                        "mean_absolute_pressure": float(
                            pressure.loc[timestamp, selected_communities].abs().mean()
                        ),
                        "raw_gross_4h": raw_gross,
                        "raw_net_4h_20bp": raw_gross - 0.002,
                        "raw_net_4h_30bp": raw_gross - 0.003,
                        "residual_gross_4h": residual_gross,
                        "residual_net_4h_40bp": residual_gross - 0.004,
                    }
                )
                last_by_candidate[candidate] = timestamp
    return pd.DataFrame(rows)


def summarize_v142(portfolios: pd.DataFrame) -> pd.DataFrame:
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
                    "portfolio_observations": len(sample),
                    "active_days": sample["entry_day"].nunique(),
                    "active_months": sample["entry_month"].nunique(),
                    "mean_symbol_count": sample["portfolio_symbol_count"].mean(),
                    "mean_raw_gross_4h": sample["raw_gross_4h"].mean(),
                    "mean_raw_net_4h_20bp": sample["raw_net_4h_20bp"].mean(),
                    "mean_raw_net_4h_30bp": sample["raw_net_4h_30bp"].mean(),
                    "mean_residual_gross_4h": sample[
                        "residual_gross_4h"
                    ].mean(),
                    "mean_residual_net_4h_40bp": sample[
                        "residual_net_4h_40bp"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def randomize_v142_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V142Config,
) -> pd.DataFrame:
    rows = []
    for raw_month, month_edges in edges.groupby("month_start", sort=True):
        month = pd.Timestamp(raw_month)
        communities = contexts[month]["communities"]
        month_code = month.year * 12 + month.month
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month_code)
        for (candidate, follower), local in month_edges.groupby(
            ["candidate", "follower_community"], sort=False
        ):
            follower = str(follower)
            choices = [node for node in communities if node != follower]
            source_rows = local.sort_values("edge_rank")
            take = min(len(source_rows), len(choices))
            leaders = rng.choice(choices, size=take, replace=False)
            for leader, row in zip(
                leaders, source_rows.head(take).itertuples(index=False), strict=True
            ):
                payload = row._asdict()
                payload["leader_community"] = str(leader)
                payload["candidate"] = candidate
                rows.append(payload)
    return pd.DataFrame(rows)


def reverse_v142_edges(edges: pd.DataFrame, cfg: V142Config) -> pd.DataFrame:
    out = edges.copy()
    out[["leader_community", "follower_community"]] = out[
        ["follower_community", "leader_community"]
    ].to_numpy()
    out = out.sort_values(
        ["month_start", "candidate", "follower_community", "edge_weight"],
        ascending=[True, True, True, False],
    ).drop_duplicates(
        ["month_start", "candidate", "leader_community", "follower_community"],
        keep="first",
    )
    out["edge_rank"] = (
        out.groupby(
            ["month_start", "candidate", "follower_community"], sort=False
        ).cumcount()
        + 1
    )
    return out[out["edge_rank"].le(cfg.leaders_per_follower)].reset_index(
        drop=True
    )


def random_v142_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V142Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v142_edges(edges, contexts, iteration, cfg)
        portfolios = build_v142_portfolios(contexts, randomized, cfg)
        means = {}
        for candidate in CANDIDATES:
            sample = portfolios[portfolios["candidate"].eq(candidate)]
            mean = float(sample["residual_net_4h_40bp"].mean())
            means[candidate] = mean
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "portfolio_observations": len(sample),
                    "mean_residual_net_4h_40bp": mean,
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "portfolio_observations": len(portfolios),
                "mean_residual_net_4h_40bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(
    sample: pd.DataFrame,
    cfg: V142Config,
) -> tuple[float, float]:
    daily = [
        group["residual_net_4h_40bp"].dropna().to_numpy(dtype=float)
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


def audit_v142(
    real: pd.DataFrame,
    delayed: pd.DataFrame,
    reversed_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V142Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["candidate"].eq("FAMILY_MAX"),
        "mean_residual_net_4h_40bp",
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
        delayed_mean = float(
            delayed.loc[
                delayed["candidate"].eq(candidate), "residual_net_4h_40bp"
            ].mean()
        )
        reversed_mean = float(
            reversed_portfolios.loc[
                reversed_portfolios["candidate"].eq(candidate),
                "residual_net_4h_40bp",
            ].mean()
        )
        ci_low, ci_high = _bootstrap_ci(sample, cfg)
        real_mean = float(lookup["all"].mean_residual_net_4h_40bp)
        percentile = float(family.lt(real_mean).mean()) if len(family) else np.nan
        month_share = _positive_share(
            sample.groupby("entry_month")["residual_net_4h_40bp"].sum()
        )
        worst_period = float(
            sample.groupby("period")["residual_net_4h_40bp"].mean().min()
        )
        gates = {
            "full_observations_150": lookup["all"].portfolio_observations >= 150,
            "validation_observations_40": lookup["validation"].portfolio_observations
            >= 40,
            "holdout_observations_40": lookup["holdout"].portfolio_observations
            >= 40,
            "development_residual_net40_positive": lookup[
                "development"
            ].mean_residual_net_4h_40bp
            > 0,
            "validation_residual_net40_positive": lookup[
                "validation"
            ].mean_residual_net_4h_40bp
            > 0,
            "holdout_residual_net40_positive": lookup[
                "holdout"
            ].mean_residual_net_4h_40bp
            > 0,
            "development_raw_net20_positive": lookup[
                "development"
            ].mean_raw_net_4h_20bp
            > 0,
            "validation_raw_net20_positive": lookup[
                "validation"
            ].mean_raw_net_4h_20bp
            > 0,
            "holdout_raw_net20_positive": lookup["holdout"].mean_raw_net_4h_20bp
            > 0,
            "full_raw_net30_positive": lookup["all"].mean_raw_net_4h_30bp > 0,
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
                "verdict": "community_vol_graph_shadow_candidate"
                if eligible
                else "reject_community_vol_graph_candidate",
                "full_residual_net40": real_mean,
                "development_residual_net40": lookup[
                    "development"
                ].mean_residual_net_4h_40bp,
                "validation_residual_net40": lookup[
                    "validation"
                ].mean_residual_net_4h_40bp,
                "holdout_residual_net40": lookup[
                    "holdout"
                ].mean_residual_net_4h_40bp,
                "full_raw_net20": lookup["all"].mean_raw_net_4h_20bp,
                "full_raw_net30": lookup["all"].mean_raw_net_4h_30bp,
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
        "community_vol_graph_shadow_candidate"
        if any(row["eligible"] for row in rows)
        else "reject_community_vol_graph_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def _node_state_panel(
    contexts: dict[pd.Timestamp, dict[str, Any]],
) -> pd.DataFrame:
    frames = []
    names = (
        "node_return_1h",
        "node_volatility_1h",
        "node_return_z",
        "node_volatility_z",
        "node_breadth",
        "node_member_count",
        "node_future_residual_4h",
    )
    for month, context in contexts.items():
        base = None
        for name in names:
            long = (
                context[name]
                .rename_axis(index="feature_time", columns="community_id")
                .stack(future_stack=True)
                .rename(name)
                .reset_index()
            )
            base = long if base is None else base.merge(
                long, on=["feature_time", "community_id"], how="outer"
            )
        base["month_start"] = month
        frames.append(base)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_findings(
    path: Path,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    edges: pd.DataFrame,
    portfolios: pd.DataFrame,
) -> None:
    lines = [
        "# v14.2 Community Volatility-Transmission Findings",
        "",
        f"Verdict: `{audit['family_verdict'].iloc[0]}`.",
        "",
        "The primary endpoint is four-hour BTC-residual return after 40 bp total cost.",
        "",
        "## Candidate audit",
        "",
        audit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Frozen graph months: `{edges['month_start'].nunique()}`; edges: `{len(edges)}`; "
        f"real observations: `{len(portfolios)}`.",
        "",
        "This retrospective audit grants no PaperLive, leverage, or live-order permission.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v142_community_volatility_transmission(
    cfg: V142Config = V142Config(),
) -> dict[str, Path]:
    membership = load_v142_membership(cfg)
    prices = load_v141_price_matrices()
    edges, contexts = build_v142_graph_and_contexts(prices, membership, cfg)
    if edges.empty or not contexts:
        raise RuntimeError("v14.2 produced no causal monthly graph contexts")
    real = build_v142_portfolios(contexts, edges, cfg)
    delayed = build_v142_portfolios(
        contexts,
        edges,
        cfg,
        signal_shift_hours=cfg.delayed_signal_hours,
    )
    reversed_edges = reverse_v142_edges(edges, cfg)
    reversed_portfolios = build_v142_portfolios(
        contexts, reversed_edges, cfg
    )
    summary = summarize_v142(real)
    controls = random_v142_controls(contexts, edges, cfg)
    audit = audit_v142(
        real, delayed, reversed_portfolios, summary, controls, cfg
    )
    node_states = _node_state_panel(contexts)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_membership.csv",
        "node_states": root / "community_node_states.parquet",
        "edges": root / "community_transmission_edges.csv",
        "portfolios": root / "community_bucket_portfolios.parquet",
        "delayed": root / "delayed_state_portfolios.parquet",
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
                "candidate_family": list(CANDIDATES),
                "graph_months": edges["month_start"].nunique(),
                "edges": len(edges),
                "node_state_rows": len(node_states),
                "portfolio_observations": len(real),
                "random_iterations": cfg.random_iterations,
                "bootstrap_iterations": cfg.bootstrap_iterations,
                "family_verdict": audit["family_verdict"].iloc[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(cfg.findings_path, audit, summary, edges, real)
    return outputs
