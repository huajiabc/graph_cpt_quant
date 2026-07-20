"""Sign-specific residual-semivariance transmission audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    estimate_v106_betas,
    residualize_v106_returns,
)
from pressure_graph.reports.v113_volatility_transmission_breakout import (
    PANEL_PATH,
    V113Config,
    _month_start,
    _period,
    _pivot,
    _rolling_rv,
    _signal_state,
    build_v113_month_edges,
)


REPORT_ROOT = Path("reports/v11_4_semivariance_transmission")
CANDIDATES = {
    "downside": "SVT1_DOWNSIDE_CASCADE",
    "upside": "SVT2_UPSIDE_CASCADE",
}
DIRECTION_SIGN = {"downside": -1.0, "upside": 1.0}


@dataclass(frozen=True)
class V114Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    lags: tuple[int, ...] = (1, 2, 4)
    min_edge_samples: int = 1000
    shrinkage_n: int = 500
    leaders_per_follower: int = 3
    leader_score_quantile: float = 0.95
    leader_shock_quantile: float = 0.90
    compression_quantile: float = 0.50
    breadth_floor: float = 2.0 / 3.0
    gap_rank_floor: float = 0.80
    min_bucket_size: int = 2
    max_bucket_size: int = 5
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def _v113_config(cfg: V114Config) -> V113Config:
    return V113Config(
        panel_path=cfg.panel_path,
        lookback_days=cfg.lookback_days,
        lags=cfg.lags,
        min_edge_samples=cfg.min_edge_samples,
        shrinkage_n=cfg.shrinkage_n,
        leaders_per_follower=cfg.leaders_per_follower,
        leader_score_quantile=cfg.leader_score_quantile,
        leader_shock_quantile=cfg.leader_shock_quantile,
        compression_quantile=cfg.compression_quantile,
        breadth_floor=cfg.breadth_floor,
        gap_rank_floor=cfg.gap_rank_floor,
        min_bucket_size=cfg.min_bucket_size,
        max_bucket_size=cfg.max_bucket_size,
        cooldown_hours=cfg.cooldown_hours,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )


def load_v114_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_parquet(
        path,
        columns=["symbol", "feature_time", "ret_15m", "future_ret_4h"],
    )
    panel["feature_time"] = pd.to_datetime(
        panel["feature_time"], utc=True, errors="coerce"
    )
    for column in ("ret_15m", "future_ret_4h"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["month_start"] = _month_start(panel["feature_time"])
    return (
        panel.dropna(subset=["symbol", "feature_time"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _signed_shock(
    residual: pd.DataFrame,
    scale: pd.Series,
    direction: str,
) -> pd.DataFrame:
    standardized = residual.div(scale).replace([np.inf, -np.inf], np.nan)
    if direction == "upside":
        return standardized.clip(lower=0.0)
    return (-standardized).clip(lower=0.0)


def build_v114_graph_and_contexts(
    panel: pd.DataFrame,
    cfg: V114Config,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    base_cfg = _v113_config(cfg)
    months = sorted(panel["month_start"].dropna().unique())
    for raw_month in months[1:]:
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
        target_future = _pivot(target, "future_ret_4h")
        betas = estimate_v106_betas(history_return)
        history_residual = residualize_v106_returns(history_return, betas)
        target_residual = residualize_v106_returns(target_return, betas)
        future_residual = residualize_v106_returns(target_future, betas)
        scale = history_residual.std(ddof=1).replace(0.0, np.nan)
        target_times = target_residual.index[target_residual.index.minute == 0]
        target_times = target_times.intersection(target_future.index)
        month_context: dict[str, Any] = {
            "month_start": month,
            "period": _period(month),
            "raw_future_4h": target_future.reindex(index=target_times),
            "residual_future_4h": future_residual.reindex(index=target_times),
            "directions": {},
        }
        for direction in CANDIDATES:
            history_shock = _signed_shock(
                history_residual, scale, direction
            )
            edges = build_v113_month_edges(history_shock, month, base_cfg)
            if edges.empty:
                continue
            graph_symbols = sorted(
                (
                    set(edges["leader_symbol"].astype(str))
                    | set(edges["follower_symbol"].astype(str))
                )
                & set(target_residual.columns)
                & set(target_future.columns)
            )
            edges = edges[
                edges["leader_symbol"].astype(str).isin(graph_symbols)
                & edges["follower_symbol"].astype(str).isin(graph_symbols)
            ].copy()
            if edges.empty:
                continue
            edges["direction"] = direction
            edges["candidate"] = CANDIDATES[direction]
            edge_frames.append(edges)
            target_shock_all = _signed_shock(
                target_residual.reindex(columns=graph_symbols),
                scale.reindex(graph_symbols),
                direction,
            )
            combined_shock = pd.concat(
                [
                    history_shock.reindex(columns=graph_symbols),
                    target_shock_all,
                ]
            ).sort_index()
            combined_semivol = _rolling_rv(combined_shock, 4)
            month_context["directions"][direction] = {
                "history_shock": history_shock.reindex(columns=graph_symbols),
                "history_rv_1h": combined_semivol.reindex(
                    index=history_residual.index, columns=graph_symbols
                ),
                "target_shock": target_shock_all.reindex(
                    index=target_times, columns=graph_symbols
                ),
                "target_rv_1h": combined_semivol.reindex(
                    index=target_times, columns=graph_symbols
                ),
            }
        if month_context["directions"]:
            contexts[month] = month_context
    edges = (
        pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame()
    )
    return edges, contexts


def build_v114_events(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V114Config,
    signal_shift_hours: int = 0,
) -> pd.DataFrame:
    rows = []
    last_by_direction: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    base_cfg = _v113_config(cfg)
    for month, context in sorted(contexts.items()):
        for direction, direction_context in context["directions"].items():
            local_edges = edges[
                edges["month_start"].eq(month)
                & edges["direction"].eq(direction)
            ]
            if local_edges.empty:
                continue
            state = _signal_state(direction_context, local_edges, base_cfg)
            if signal_shift_hours:
                for key in state:
                    state[key] = state[key].shift(signal_shift_hours)
            eligible = state["eligible"]
            for timestamp in eligible.index:
                last = last_by_direction.get(direction)
                if last is not None and pd.Timestamp(timestamp) - last < cooldown:
                    continue
                symbols = eligible.columns[eligible.loc[timestamp].eq(True)]
                if len(symbols) < cfg.min_bucket_size:
                    continue
                selected = (
                    state["gap"]
                    .loc[timestamp, symbols]
                    .sort_values(ascending=False)
                    .head(cfg.max_bucket_size)
                    .index.tolist()
                )
                raw = pd.to_numeric(
                    context["raw_future_4h"].loc[timestamp, selected],
                    errors="coerce",
                )
                residual = pd.to_numeric(
                    context["residual_future_4h"].loc[timestamp, selected],
                    errors="coerce",
                )
                finite = raw.notna() & residual.notna()
                if int(finite.sum()) < cfg.min_bucket_size:
                    continue
                traded = list(pd.Index(selected)[finite.to_numpy()])
                sign = DIRECTION_SIGN[direction]
                gross = float(sign * raw[finite].mean())
                residual_gross = float(sign * residual[finite].mean())
                rows.append(
                    {
                        "candidate": CANDIDATES[direction],
                        "direction": direction,
                        "feature_time": timestamp,
                        "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                        "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                        "period": context["period"],
                        "bucket_size": len(traded),
                        "receiver_symbols": "|".join(traded),
                        "mean_leader_score": float(
                            state["score"].loc[timestamp, traded].mean()
                        ),
                        "mean_leader_breadth": float(
                            state["breadth"].loc[timestamp, traded].mean()
                        ),
                        "mean_transmission_gap": float(
                            state["gap"].loc[timestamp, traded].mean()
                        ),
                        "raw_gross_4h": gross,
                        "residual_gross_4h": residual_gross,
                        "raw_net_4h_20bp": gross - 0.002,
                        "raw_net_4h_30bp": gross - 0.003,
                        "raw_net_4h_50bp": gross - 0.005,
                    }
                )
                last_by_direction[direction] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def summarize_v114(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = events if scope == "all" else events[events["period"].eq(scope)]
        for candidate in CANDIDATES.values():
            sample = scoped[scoped["candidate"].eq(candidate)]
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "portfolio_observations": int(len(sample)),
                    "active_days": int(sample["entry_day"].nunique()),
                    "active_months": int(sample["entry_month"].nunique()),
                    "mean_bucket_size": float(sample["bucket_size"].mean()),
                    "mean_raw_gross_4h": float(sample["raw_gross_4h"].mean()),
                    "mean_residual_gross_4h": float(
                        sample["residual_gross_4h"].mean()
                    ),
                    "mean_raw_net_4h_20bp": float(
                        sample["raw_net_4h_20bp"].mean()
                    ),
                    "mean_raw_net_4h_30bp": float(
                        sample["raw_net_4h_30bp"].mean()
                    ),
                    "mean_raw_net_4h_50bp": float(
                        sample["raw_net_4h_50bp"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def randomize_v114_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V114Config,
) -> pd.DataFrame:
    frames = []
    for month, context in contexts.items():
        for direction, direction_context in context["directions"].items():
            local = edges[
                edges["month_start"].eq(month)
                & edges["direction"].eq(direction)
            ].copy()
            if local.empty:
                continue
            symbols = sorted(direction_context["target_shock"].columns)
            rng = np.random.default_rng(
                cfg.seed + iteration * 1009 + month.month * 17 + len(direction)
            )
            for follower, group in local.groupby("follower_symbol", sort=False):
                choices = [symbol for symbol in symbols if symbol != str(follower)]
                sampled = rng.choice(choices, size=len(group), replace=False)
                for index, leader in zip(group.index, sampled):
                    local.at[index, "leader_symbol"] = str(leader)
            frames.append(local)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def random_v114_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V114Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v114_edges(edges, contexts, iteration, cfg)
        events = build_v114_events(contexts, randomized, cfg)
        means = []
        for candidate in CANDIDATES.values():
            sample = events[events["candidate"].eq(candidate)]
            value = float(sample["raw_net_4h_20bp"].mean())
            means.append(value)
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "portfolio_observations": int(len(sample)),
                    "mean_raw_net_4h_20bp": value,
                }
            )
        finite = [value for value in means if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "portfolio_observations": int(len(events)),
                "mean_raw_net_4h_20bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(
    sample: pd.DataFrame,
    cfg: V114Config,
) -> tuple[float, float]:
    daily = [
        group["raw_net_4h_20bp"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day")
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


def _positive_share(sample: pd.DataFrame, column: str) -> float:
    values = sample.groupby(column)["raw_net_4h_20bp"].sum().clip(lower=0.0)
    return float(values.max() / values.sum() if values.sum() > 0 else np.inf)


def _receiver_share(sample: pd.DataFrame) -> float:
    values: dict[str, float] = {}
    for row in sample.itertuples(index=False):
        symbols = [value for value in str(row.receiver_symbols).split("|") if value]
        if not symbols:
            continue
        contribution = float(row.raw_net_4h_20bp) / len(symbols)
        for symbol in symbols:
            values[symbol] = values.get(symbol, 0.0) + contribution
    positive = np.array([max(value, 0.0) for value in values.values()])
    return float(positive.max() / positive.sum() if positive.sum() > 0 else np.inf)


def audit_v114(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V114Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["candidate"].eq("FAMILY_MAX"), "mean_raw_net_4h_20bp"
    ].dropna()
    rows = []
    for candidate in CANDIDATES.values():
        lookup = {
            row.scope: row
            for row in summary[summary["candidate"].eq(candidate)].itertuples(
                index=False
            )
        }
        sample = real[real["candidate"].eq(candidate)].sort_values("feature_time")
        shifted_mean = float(
            shifted.loc[
                shifted["candidate"].eq(candidate), "raw_net_4h_20bp"
            ].mean()
        )
        percentile = float(
            family.lt(lookup["all"].mean_raw_net_4h_20bp).mean()
        )
        ci_low, ci_high = _bootstrap(sample, cfg)
        chronological = [
            float(sample.iloc[index]["raw_net_4h_20bp"].mean())
            for index in np.array_split(np.arange(len(sample)), 5)
            if len(index)
        ]
        month_share = _positive_share(sample, "entry_month")
        receiver_share = _receiver_share(sample)
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
            ].mean_raw_net_4h_20bp
            > 0,
            "holdout_net20_positive": lookup["holdout"].mean_raw_net_4h_20bp
            > 0,
            "full_net30_positive": lookup["all"].mean_raw_net_4h_30bp > 0,
            "random_family_p90": percentile >= 0.90,
            "beats_shifted": lookup["all"].mean_raw_net_4h_20bp > shifted_mean,
            "bootstrap_lower_positive": ci_low > 0,
            "five_chrono_nonnegative": bool(chronological)
            and min(chronological) >= 0,
            "month_share_below_35pct": month_share <= 0.35,
            "receiver_share_below_35pct": receiver_share <= 0.35,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "verdict": "retrospective_forward_watch_only"
                if eligible
                else "reject_semivariance_transmission",
                "full_gross": lookup["all"].mean_raw_gross_4h,
                "full_residual_gross": lookup[
                    "all"
                ].mean_residual_gross_4h,
                "full_net20": lookup["all"].mean_raw_net_4h_20bp,
                "validation_net20": lookup[
                    "validation"
                ].mean_raw_net_4h_20bp,
                "holdout_net20": lookup["holdout"].mean_raw_net_4h_20bp,
                "full_net30": lookup["all"].mean_raw_net_4h_30bp,
                "shifted_net20": shifted_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "chronological_means": "|".join(
                    f"{value:.10f}" for value in chronological
                ),
                "max_positive_month_share": month_share,
                "max_positive_receiver_share": receiver_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        )
    family_verdict = (
        "retrospective_forward_watch_only"
        if any(row["eligible"] for row in rows)
        else "reject_semivariance_transmission_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def write_v114_semivariance_transmission(
    cfg: V114Config = V114Config(),
) -> dict[str, Path]:
    panel = load_v114_panel(cfg.panel_path)
    edges, contexts = build_v114_graph_and_contexts(panel, cfg)
    real = build_v114_events(contexts, edges, cfg)
    shifted = build_v114_events(contexts, edges, cfg, signal_shift_hours=24)
    summary = summarize_v114(real)
    controls = random_v114_controls(contexts, edges, cfg)
    audit = audit_v114(real, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "edges": root / "monthly_semivariance_edges.parquet",
        "events": root / "semivariance_transmission_events.parquet",
        "shifted": root / "shifted_semivariance_events.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    edges.to_parquet(outputs["edges"], index=False)
    real.to_parquet(outputs["events"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v11.4 Sign-Specific Semivariance Transmission",
        "",
        f"Status: `{audit['family_verdict'].iloc[0]}`.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.candidate}: gross={row.full_gross:.4%}, "
            f"net20={row.full_net20:.4%}, validation={row.validation_net20:.4%}, "
            f"holdout={row.holdout_net20:.4%}, "
            f"random percentile={row.random_family_percentile:.1%}."
        )
    lines.extend(
        ["", "Research only. No PaperLive or real-order permission changed."]
    )
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
