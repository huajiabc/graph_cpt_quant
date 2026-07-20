"""Cross-community downside-volatility front and BTC timing audit."""
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
    residualize_v106_returns,
)
from pressure_graph.reports.v113_volatility_transmission_breakout import (
    _period,
    _pivot,
)
from pressure_graph.reports.v116_community_volatility_path import (
    KLINE_ROOT,
    MEMBERSHIP_PATH,
    load_v116_panel,
)


REPORT_ROOT = Path("reports/v11_5_cross_community_volatility_front")
CANDIDATE = "VCF1_ALT_FRONT_LEADS_BTC"


@dataclass(frozen=True)
class V115Config:
    kline_root: Path = KLINE_ROOT
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    min_samples: int = 1000
    community_count: int = 8
    shock_quantile: float = 0.90
    community_density_quantile: float = 0.90
    front_breadth_quantile: float = 0.90
    min_active_communities: int = 3
    btc_lag_quantile: float = 0.25
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def load_v115_panel(
    kline_root: Path = KLINE_ROOT,
    membership_path: Path = MEMBERSHIP_PATH,
) -> pd.DataFrame:
    return load_v116_panel(kline_root, membership_path)


def _downside_shock(
    returns: pd.DataFrame,
    betas: pd.Series,
    scale: pd.Series,
) -> pd.DataFrame:
    residual = residualize_v106_returns(returns, betas)
    return (-residual.div(scale)).clip(lower=0.0).replace(
        [np.inf, -np.inf], np.nan
    )


def _community_density(
    shocks: pd.DataFrame,
    shock_thresholds: pd.Series,
    communities: dict[str, list[str]],
) -> pd.DataFrame:
    density = pd.DataFrame(index=shocks.index)
    for community_id, original_members in communities.items():
        members = [
            symbol
            for symbol in original_members
            if symbol in shocks.columns and symbol in shock_thresholds.index
        ]
        if not members:
            density[community_id] = np.nan
            continue
        density[community_id] = shocks[members].ge(
            shock_thresholds.reindex(members), axis="columns"
        ).mean(axis=1)
    return density


def build_v115_front_state(
    history_shock: pd.DataFrame,
    target_shock: pd.DataFrame,
    history_btc_1h: pd.Series,
    target_btc_1h: pd.Series,
    communities: dict[str, list[str]],
    cfg: V115Config,
) -> pd.DataFrame:
    shock_thresholds = history_shock.quantile(cfg.shock_quantile)
    history_density = _community_density(
        history_shock, shock_thresholds, communities
    )
    target_density = _community_density(
        target_shock, shock_thresholds, communities
    )
    density_thresholds = history_density.quantile(
        cfg.community_density_quantile
    )
    historical_front = history_density.ge(
        density_thresholds, axis="columns"
    ).sum(axis=1)
    target_active = target_density.ge(density_thresholds, axis="columns")
    target_front = target_active.sum(axis=1)
    front_threshold = float(
        historical_front.quantile(cfg.front_breadth_quantile)
    )
    btc_threshold = float(history_btc_1h.quantile(cfg.btc_lag_quantile))
    state = pd.DataFrame(
        {
            "front_breadth": target_front,
            "front_threshold": front_threshold,
            "btc_ret_1h": target_btc_1h.reindex(target_front.index),
            "btc_lag_threshold": btc_threshold,
        },
        index=target_front.index,
    )
    state["active_community_ids"] = [
        "|".join(target_active.columns[row.to_numpy(dtype=bool)].astype(str))
        for _, row in target_active.iterrows()
    ]
    state["event"] = (
        state["front_breadth"].ge(state["front_threshold"])
        & state["front_breadth"].ge(cfg.min_active_communities)
        & state["btc_ret_1h"].ge(state["btc_lag_threshold"])
    )
    return state


def build_v115_global_state(
    history_shock: pd.DataFrame,
    target_shock: pd.DataFrame,
    history_btc_1h: pd.Series,
    target_btc_1h: pd.Series,
    cfg: V115Config,
) -> pd.DataFrame:
    thresholds = history_shock.quantile(cfg.shock_quantile)
    historical_density = history_shock.ge(thresholds, axis="columns").mean(axis=1)
    target_density = target_shock.ge(thresholds, axis="columns").mean(axis=1)
    density_threshold = float(
        historical_density.quantile(cfg.front_breadth_quantile)
    )
    btc_threshold = float(history_btc_1h.quantile(cfg.btc_lag_quantile))
    state = pd.DataFrame(
        {
            "front_breadth": target_density,
            "front_threshold": density_threshold,
            "btc_ret_1h": target_btc_1h.reindex(target_density.index),
            "btc_lag_threshold": btc_threshold,
            "active_community_ids": "GLOBAL_BREADTH",
        },
        index=target_density.index,
    )
    state["event"] = state["front_breadth"].ge(
        state["front_threshold"]
    ) & state["btc_ret_1h"].ge(state["btc_lag_threshold"])
    return state


def build_v115_contexts(
    panel: pd.DataFrame,
    cfg: V115Config,
) -> tuple[dict[pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    membership_rows = []
    frozen_membership = pd.read_csv(cfg.membership_path)
    frozen_membership["month_start"] = pd.to_datetime(
        frozen_membership["month_start"], utc=True, errors="coerce"
    )
    months = sorted(frozen_membership["month_start"].dropna().unique())
    for raw_month in months:
        month = pd.Timestamp(raw_month)
        history = panel[
            panel["feature_time"].ge(
                month - pd.Timedelta(days=cfg.lookback_days)
            )
            & panel["feature_time"].lt(month)
        ]
        target = panel[panel["month_start"].eq(month)]
        if history.empty or target.empty:
            continue
        history_return = _pivot(history, "ret_15m")
        target_return = _pivot(target, "ret_15m")
        history_btc_1h = _pivot(history, "ret_1h").get(BTC, pd.Series(dtype=float))
        target_btc_1h = _pivot(target, "ret_1h").get(BTC, pd.Series(dtype=float))
        target_btc_future = _pivot(target, "future_ret_4h").get(
            BTC, pd.Series(dtype=float)
        )
        if history_btc_1h.empty or target_btc_1h.empty or target_btc_future.empty:
            continue
        betas = estimate_v106_betas(history_return)
        history_residual = residualize_v106_returns(history_return, betas)
        scale = history_residual.std(ddof=1).replace(0.0, np.nan)
        history_shock = _downside_shock(history_return, betas, scale)
        target_shock_all = _downside_shock(target_return, betas, scale)
        local_membership = frozen_membership[
            frozen_membership["month_start"].eq(month)
        ]
        communities = {
            str(community_id).replace("BSP", "VCF"): sorted(
                group["symbol"].astype(str).tolist()
            )
            for community_id, group in local_membership.groupby(
                "community_id", sort=True
            )
        }
        exact_symbols = set().union(*communities.values()) if communities else set()
        exact = (
            len(communities) == cfg.community_count
            and all(len(members) == 9 for members in communities.values())
            and exact_symbols.issubset(history_shock.columns)
            and exact_symbols.issubset(target_shock_all.columns)
        )
        if not exact:
            continue
        for community_id, members in communities.items():
            for symbol in members:
                membership_rows.append(
                    {
                        "month_start": month,
                        "community_id": community_id,
                        "symbol": symbol,
                        "community_size": len(members),
                    }
                )
        target_times = target_shock_all.index[
            target_shock_all.index.minute == 0
        ]
        target_times = target_times.intersection(target_btc_future.index)
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "history_shock": history_shock,
            "target_shock": target_shock_all.reindex(target_times),
            "history_btc_1h": history_btc_1h,
            "target_btc_1h": target_btc_1h.reindex(target_times),
            "target_btc_future_4h": target_btc_future.reindex(target_times),
            "communities": communities,
        }
    return contexts, pd.DataFrame(membership_rows)


def random_v115_communities(
    context: dict[str, Any],
    iteration: int,
    cfg: V115Config,
) -> dict[str, list[str]]:
    sizes = [len(values) for values in context["communities"].values()]
    symbols = sorted(set().union(*context["communities"].values()))
    month = context["month_start"]
    rng = np.random.default_rng(cfg.seed + iteration * 1009 + month.month)
    shuffled = list(rng.permutation(symbols))
    output = {}
    cursor = 0
    for index, size in enumerate(sizes, start=1):
        output[f"{month:%Y-%m}:R{iteration:03d}C{index:02d}"] = shuffled[
            cursor : cursor + size
        ]
        cursor += size
    return output


def build_v115_events(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V115Config,
    community_overrides: dict[pd.Timestamp, dict[str, list[str]]] | None = None,
    signal_shift_hours: int = 0,
    global_breadth: bool = False,
) -> pd.DataFrame:
    rows = []
    last_event: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        communities = (
            community_overrides.get(month, context["communities"])
            if community_overrides is not None
            else context["communities"]
        )
        if global_breadth:
            state = build_v115_global_state(
                context["history_shock"],
                context["target_shock"],
                context["history_btc_1h"],
                context["target_btc_1h"],
                cfg,
            )
        else:
            state = build_v115_front_state(
                context["history_shock"],
                context["target_shock"],
                context["history_btc_1h"],
                context["target_btc_1h"],
                communities,
                cfg,
            )
        if signal_shift_hours:
            state = state.shift(signal_shift_hours)
        for timestamp in state.index[state["event"].eq(True)]:
            if last_event is not None and pd.Timestamp(timestamp) - last_event < cooldown:
                continue
            future = float(context["target_btc_future_4h"].get(timestamp, np.nan))
            if not np.isfinite(future):
                continue
            gross = -future
            rows.append(
                {
                    "candidate": CANDIDATE,
                    "control_kind": "global_breadth" if global_breadth else "real_graph",
                    "feature_time": timestamp,
                    "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                    "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                    "period": context["period"],
                    "front_breadth": float(state.at[timestamp, "front_breadth"]),
                    "front_threshold": float(state.at[timestamp, "front_threshold"]),
                    "active_community_ids": str(
                        state.at[timestamp, "active_community_ids"]
                    ),
                    "btc_ret_1h": float(state.at[timestamp, "btc_ret_1h"]),
                    "btc_lag_threshold": float(
                        state.at[timestamp, "btc_lag_threshold"]
                    ),
                    "btc_short_gross_4h": gross,
                    "btc_short_net_4h_10bp": gross - 0.001,
                    "btc_short_net_4h_20bp": gross - 0.002,
                    "btc_short_net_4h_30bp": gross - 0.003,
                    "btc_short_net_4h_50bp": gross - 0.005,
                }
            )
            last_event = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def summarize_v115(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "scope": scope,
                "portfolio_observations": int(len(sample)),
                "active_days": int(sample["entry_day"].nunique()),
                "active_months": int(sample["entry_month"].nunique()),
                "mean_front_breadth": float(sample["front_breadth"].mean()),
                "mean_btc_short_gross_4h": float(
                    sample["btc_short_gross_4h"].mean()
                ),
                "mean_btc_short_net_4h_10bp": float(
                    sample["btc_short_net_4h_10bp"].mean()
                ),
                "mean_btc_short_net_4h_20bp": float(
                    sample["btc_short_net_4h_20bp"].mean()
                ),
                "mean_btc_short_net_4h_30bp": float(
                    sample["btc_short_net_4h_30bp"].mean()
                ),
                "mean_btc_short_net_4h_50bp": float(
                    sample["btc_short_net_4h_50bp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def random_v115_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V115Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        overrides = {
            month: random_v115_communities(context, iteration, cfg)
            for month, context in contexts.items()
        }
        events = build_v115_events(contexts, cfg, overrides)
        rows.append(
            {
                "iteration": iteration,
                "portfolio_observations": int(len(events)),
                "mean_btc_short_net_4h_20bp": float(
                    events["btc_short_net_4h_20bp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(events: pd.DataFrame, cfg: V115Config) -> tuple[float, float]:
    daily = [
        group["btc_short_net_4h_20bp"].to_numpy(dtype=float)
        for _, group in events.groupby("entry_day")
        if len(group)
    ]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(
            float(np.mean(np.concatenate([daily[index] for index in chosen])))
        )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def audit_v115(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    global_control: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V115Config,
) -> pd.DataFrame:
    lookup = {row.scope: row for row in summary.itertuples(index=False)}
    random_family = controls["mean_btc_short_net_4h_20bp"].dropna()
    percentile = float(
        random_family.lt(lookup["all"].mean_btc_short_net_4h_20bp).mean()
    )
    shifted_mean = float(shifted["btc_short_net_4h_20bp"].mean())
    global_mean = float(global_control["btc_short_net_4h_20bp"].mean())
    ci_low, ci_high = _bootstrap(real, cfg)
    sample = real.sort_values("feature_time")
    chronological = [
        float(sample.iloc[index]["btc_short_net_4h_20bp"].mean())
        for index in np.array_split(np.arange(len(sample)), 5)
        if len(index)
    ]
    month_values = sample.groupby("entry_month")[
        "btc_short_net_4h_20bp"
    ].sum().clip(lower=0.0)
    month_share = float(
        month_values.max() / month_values.sum()
        if month_values.sum() > 0
        else np.inf
    )
    gates = {
        "full_observations_100": lookup["all"].portfolio_observations >= 100,
        "validation_observations_25": lookup[
            "validation"
        ].portfolio_observations
        >= 25,
        "holdout_observations_25": lookup["holdout"].portfolio_observations
        >= 25,
        "validation_net20_positive": lookup[
            "validation"
        ].mean_btc_short_net_4h_20bp
        > 0,
        "holdout_net20_positive": lookup[
            "holdout"
        ].mean_btc_short_net_4h_20bp
        > 0,
        "full_net30_positive": lookup["all"].mean_btc_short_net_4h_30bp > 0,
        "random_family_p90": percentile >= 0.90,
        "beats_shifted": lookup["all"].mean_btc_short_net_4h_20bp
        > shifted_mean,
        "beats_global_breadth": lookup["all"].mean_btc_short_net_4h_20bp
        > global_mean,
        "bootstrap_lower_positive": ci_low > 0,
        "five_chrono_nonnegative": bool(chronological)
        and min(chronological) >= 0,
        "month_share_below_35pct": month_share <= 0.35,
    }
    eligible = all(gates.values())
    return pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "eligible": eligible,
                "verdict": "retrospective_forward_watch_only"
                if eligible
                else "reject_cross_community_volatility_front",
                "full_gross": lookup["all"].mean_btc_short_gross_4h,
                "full_net10": lookup["all"].mean_btc_short_net_4h_10bp,
                "full_net20": lookup["all"].mean_btc_short_net_4h_20bp,
                "validation_net20": lookup[
                    "validation"
                ].mean_btc_short_net_4h_20bp,
                "holdout_net20": lookup["holdout"].mean_btc_short_net_4h_20bp,
                "full_net30": lookup["all"].mean_btc_short_net_4h_30bp,
                "shifted_net20": shifted_mean,
                "global_breadth_net20": global_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "chronological_means": "|".join(
                    f"{value:.10f}" for value in chronological
                ),
                "max_positive_month_share": month_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        ]
    )


def write_v115_cross_community_volatility_front(
    cfg: V115Config = V115Config(),
) -> dict[str, Path]:
    panel = load_v115_panel(cfg.kline_root, cfg.membership_path)
    contexts, membership = build_v115_contexts(panel, cfg)
    real = build_v115_events(contexts, cfg)
    shifted = build_v115_events(contexts, cfg, signal_shift_hours=24)
    global_control = build_v115_events(contexts, cfg, global_breadth=True)
    summary = summarize_v115(real)
    controls = random_v115_controls(contexts, cfg)
    audit = audit_v115(
        real, shifted, global_control, summary, controls, cfg
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_balanced_membership.csv",
        "events": root / "community_front_events.parquet",
        "shifted": root / "shifted_front_events.parquet",
        "global_control": root / "global_breadth_events.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_partition_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["membership"], index=False)
    real.to_parquet(outputs["events"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    global_control.to_parquet(outputs["global_control"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    row = audit.iloc[0]
    lines = [
        "# v11.5 Cross-Community Downside-Volatility Front",
        "",
        f"Status: `{row['verdict']}`.",
        "",
        f"- gross={row['full_gross']:.4%}",
        f"- net20={row['full_net20']:.4%}",
        f"- validation net20={row['validation_net20']:.4%}",
        f"- holdout net20={row['holdout_net20']:.4%}",
        f"- random percentile={row['random_family_percentile']:.1%}",
        "",
        "Research only. No PaperLive or real-order permission changed.",
    ]
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
