"""Coordinated community-volatility path continuation audit."""
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
    _month_start,
    _period,
    _pivot,
)


REPORT_ROOT = Path("reports/v11_6_community_volatility_path")
KLINE_ROOT = Path("data/raw/bybit/klines")
MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
CANDIDATE = "CVP1_EFFICIENT_VOL_CONTINUATION"


@dataclass(frozen=True)
class V116Config:
    kline_root: Path = KLINE_ROOT
    membership_path: Path = MEMBERSHIP_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    min_samples: int = 1000
    community_count: int = 8
    volatility_quantile: float = 0.95
    efficiency_quantile: float = 0.75
    direction_breadth_floor: float = 2.0 / 3.0
    max_communities: int = 3
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def load_v116_panel(
    kline_root: Path = KLINE_ROOT,
    membership_path: Path = MEMBERSHIP_PATH,
) -> pd.DataFrame:
    membership = pd.read_csv(membership_path)
    symbols = sorted(set(membership["symbol"].astype(str)) | {"BTCUSDT"})
    frames = []
    for symbol in symbols:
        path = kline_root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["bar_close_time", "close"])
        frame["feature_time"] = pd.to_datetime(
            frame["bar_close_time"], utc=True, errors="coerce"
        )
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = (
            frame.dropna(subset=["feature_time", "close"])
            .drop_duplicates("feature_time", keep="last")
            .sort_values("feature_time")
        )
        frame["ret_15m"] = frame["close"].pct_change(fill_method=None)
        frame["ret_1h"] = frame["close"].pct_change(4, fill_method=None)
        frame["future_ret_4h"] = frame["close"].shift(-16).div(frame["close"]) - 1.0
        frame["symbol"] = symbol
        frames.append(
            frame[
                [
                    "symbol",
                    "feature_time",
                    "ret_15m",
                    "ret_1h",
                    "future_ret_4h",
                ]
            ]
        )
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    panel["month_start"] = _month_start(panel["feature_time"])
    return (
        panel.dropna(subset=["symbol", "feature_time"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _bucket_path_features(
    residual: pd.DataFrame,
    members: list[str],
) -> pd.DataFrame:
    local = residual[members]
    bucket_15m = local.mean(axis=1)
    bucket_1h = bucket_15m.rolling(4, min_periods=4).sum()
    path_length = bucket_15m.abs().rolling(4, min_periods=4).sum()
    realized_volatility = (
        bucket_15m.pow(2).rolling(4, min_periods=4).sum().pow(0.5)
    )
    efficiency = bucket_1h.abs().div(path_length).where(path_length.gt(0))
    member_1h = local.rolling(4, min_periods=4).sum()
    direction = np.sign(bucket_1h)
    same_sign = member_1h.mul(direction, axis="index").gt(0)
    breadth = same_sign.mean(axis=1).where(direction.ne(0))
    return pd.DataFrame(
        {
            "bucket_return_1h": bucket_1h,
            "realized_volatility_1h": realized_volatility,
            "path_efficiency_1h": efficiency,
            "direction_breadth_1h": breadth,
            "direction": direction,
        }
    )


def build_v116_contexts(
    panel: pd.DataFrame,
    cfg: V116Config,
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
        target_future = _pivot(target, "future_ret_4h")
        betas = estimate_v106_betas(history_return)
        history_residual = residualize_v106_returns(history_return, betas)
        target_residual = residualize_v106_returns(target_return, betas)
        future_residual = residualize_v106_returns(target_future, betas)
        local_membership = frozen_membership[
            frozen_membership["month_start"].eq(month)
        ]
        communities = {
            str(community_id).replace("BSP", "CVP"): sorted(
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
            and exact_symbols.issubset(history_residual.columns)
            and exact_symbols.issubset(target_residual.columns)
            and exact_symbols.issubset(future_residual.columns)
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
        symbols = sorted(set().union(*communities.values()))
        combined_residual = pd.concat(
            [
                history_residual.reindex(columns=symbols),
                target_residual.reindex(columns=symbols),
            ]
        ).sort_index()
        target_times = target_residual.index[target_residual.index.minute == 0]
        target_times = target_times.intersection(target_future.index)
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "history_residual": history_residual.reindex(columns=symbols),
            "combined_residual": combined_residual,
            "target_times": target_times,
            "raw_future_4h": target_future.reindex(
                index=target_times, columns=symbols
            ),
            "residual_future_4h": future_residual.reindex(
                index=target_times, columns=symbols
            ),
            "communities": communities,
        }
    return contexts, pd.DataFrame(membership_rows)


def build_v116_state(
    context: dict[str, Any],
    communities: dict[str, list[str]],
    cfg: V116Config,
) -> dict[str, pd.DataFrame]:
    target_times = context["target_times"]
    keys = (
        "realized_volatility_1h",
        "path_efficiency_1h",
        "direction_breadth_1h",
        "direction",
    )
    target = {
        key: pd.DataFrame(index=target_times, columns=communities, dtype=float)
        for key in keys
    }
    volatility_thresholds = {}
    efficiency_thresholds = {}
    for community_id, members in communities.items():
        historical = _bucket_path_features(context["history_residual"], members)
        combined = _bucket_path_features(context["combined_residual"], members)
        volatility_thresholds[community_id] = float(
            historical["realized_volatility_1h"].quantile(
                cfg.volatility_quantile
            )
        )
        efficiency_thresholds[community_id] = float(
            historical["path_efficiency_1h"].quantile(
                cfg.efficiency_quantile
            )
        )
        for key in keys:
            target[key][community_id] = combined[key].reindex(target_times)
    volatility_threshold = pd.Series(volatility_thresholds, dtype=float)
    efficiency_threshold = pd.Series(efficiency_thresholds, dtype=float)
    volatility_ratio = target["realized_volatility_1h"].div(
        volatility_threshold
    )
    eligible = (
        target["realized_volatility_1h"].ge(volatility_threshold)
        & target["path_efficiency_1h"].ge(efficiency_threshold)
        & target["direction_breadth_1h"].ge(cfg.direction_breadth_floor)
        & target["direction"].ne(0)
    )
    return {
        **target,
        "volatility_ratio": volatility_ratio,
        "eligible": eligible,
    }


def random_v116_communities(
    context: dict[str, Any],
    iteration: int,
    cfg: V116Config,
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


def build_v116_events(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V116Config,
    community_overrides: dict[pd.Timestamp, dict[str, list[str]]] | None = None,
    signal_shift_hours: int = 0,
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
        state = build_v116_state(context, communities, cfg)
        if signal_shift_hours:
            for key in state:
                state[key] = state[key].shift(signal_shift_hours)
        eligible = state["eligible"]
        for timestamp in eligible.index:
            if last_event is not None and pd.Timestamp(timestamp) - last_event < cooldown:
                continue
            community_ids = eligible.columns[eligible.loc[timestamp].eq(True)]
            if not len(community_ids):
                continue
            selected = (
                state["volatility_ratio"]
                .loc[timestamp, community_ids]
                .sort_values(ascending=False)
                .head(cfg.max_communities)
                .index.tolist()
            )
            sleeves = []
            for community_id in selected:
                members = communities[community_id]
                raw = pd.to_numeric(
                    context["raw_future_4h"].loc[timestamp, members],
                    errors="coerce",
                )
                residual = pd.to_numeric(
                    context["residual_future_4h"].loc[timestamp, members],
                    errors="coerce",
                )
                finite = raw.notna() & residual.notna()
                if int(finite.sum()) < max(6, len(members) - 1):
                    continue
                direction = float(state["direction"].at[timestamp, community_id])
                sleeves.append(
                    {
                        "community_id": community_id,
                        "direction": direction,
                        "raw": float(direction * raw[finite].mean()),
                        "residual": float(direction * residual[finite].mean()),
                    }
                )
            if not sleeves:
                continue
            gross = float(np.mean([sleeve["raw"] for sleeve in sleeves]))
            residual_gross = float(
                np.mean([sleeve["residual"] for sleeve in sleeves])
            )
            rows.append(
                {
                    "candidate": CANDIDATE,
                    "feature_time": timestamp,
                    "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                    "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                    "period": context["period"],
                    "community_sleeves": len(sleeves),
                    "community_ids": "|".join(
                        sleeve["community_id"] for sleeve in sleeves
                    ),
                    "long_communities": sum(
                        sleeve["direction"] > 0 for sleeve in sleeves
                    ),
                    "short_communities": sum(
                        sleeve["direction"] < 0 for sleeve in sleeves
                    ),
                    "mean_volatility_ratio": float(
                        state["volatility_ratio"].loc[
                            timestamp,
                            [sleeve["community_id"] for sleeve in sleeves],
                        ].mean()
                    ),
                    "mean_path_efficiency": float(
                        state["path_efficiency_1h"].loc[
                            timestamp,
                            [sleeve["community_id"] for sleeve in sleeves],
                        ].mean()
                    ),
                    "mean_direction_breadth": float(
                        state["direction_breadth_1h"].loc[
                            timestamp,
                            [sleeve["community_id"] for sleeve in sleeves],
                        ].mean()
                    ),
                    "raw_gross_4h": gross,
                    "residual_gross_4h": residual_gross,
                    "raw_net_4h_20bp": gross - 0.002,
                    "raw_net_4h_30bp": gross - 0.003,
                    "raw_net_4h_50bp": gross - 0.005,
                }
            )
            last_event = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def summarize_v116(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        rows.append(
            {
                "scope": scope,
                "portfolio_observations": int(len(sample)),
                "active_days": int(sample["entry_day"].nunique()),
                "active_months": int(sample["entry_month"].nunique()),
                "mean_community_sleeves": float(
                    sample["community_sleeves"].mean()
                ),
                "long_community_share": float(
                    sample["long_communities"].sum()
                    / (
                        sample["long_communities"].sum()
                        + sample["short_communities"].sum()
                    )
                    if sample["long_communities"].sum()
                    + sample["short_communities"].sum()
                    else np.nan
                ),
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


def random_v116_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    cfg: V116Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        overrides = {
            month: random_v116_communities(context, iteration, cfg)
            for month, context in contexts.items()
        }
        events = build_v116_events(contexts, cfg, overrides)
        rows.append(
            {
                "iteration": iteration,
                "portfolio_observations": int(len(events)),
                "mean_raw_net_4h_20bp": float(
                    events["raw_net_4h_20bp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(events: pd.DataFrame, cfg: V116Config) -> tuple[float, float]:
    daily = [
        group["raw_net_4h_20bp"].to_numpy(dtype=float)
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


def _positive_community_share(events: pd.DataFrame) -> float:
    contributions: dict[str, float] = {}
    for row in events.itertuples(index=False):
        communities = [
            value for value in str(row.community_ids).split("|") if value
        ]
        if not communities:
            continue
        amount = float(row.raw_net_4h_20bp) / len(communities)
        for community_id in communities:
            contributions[community_id] = contributions.get(community_id, 0.0) + amount
    positive = np.array([max(value, 0.0) for value in contributions.values()])
    return float(positive.max() / positive.sum() if positive.sum() > 0 else np.inf)


def audit_v116(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V116Config,
) -> pd.DataFrame:
    lookup = {row.scope: row for row in summary.itertuples(index=False)}
    random_family = controls["mean_raw_net_4h_20bp"].dropna()
    percentile = float(
        random_family.lt(lookup["all"].mean_raw_net_4h_20bp).mean()
    )
    shifted_mean = float(shifted["raw_net_4h_20bp"].mean())
    ci_low, ci_high = _bootstrap(real, cfg)
    sample = real.sort_values("feature_time")
    chronological = [
        float(sample.iloc[index]["raw_net_4h_20bp"].mean())
        for index in np.array_split(np.arange(len(sample)), 5)
        if len(index)
    ]
    months = sample.groupby("entry_month")["raw_net_4h_20bp"].sum().clip(
        lower=0.0
    )
    month_share = float(
        months.max() / months.sum() if months.sum() > 0 else np.inf
    )
    community_share = _positive_community_share(sample)
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
        "holdout_net20_positive": lookup["holdout"].mean_raw_net_4h_20bp > 0,
        "full_net30_positive": lookup["all"].mean_raw_net_4h_30bp > 0,
        "random_family_p90": percentile >= 0.90,
        "beats_shifted": lookup["all"].mean_raw_net_4h_20bp > shifted_mean,
        "bootstrap_lower_positive": ci_low > 0,
        "five_chrono_nonnegative": bool(chronological)
        and min(chronological) >= 0,
        "month_share_below_35pct": month_share <= 0.35,
        "community_share_below_35pct": community_share <= 0.35,
    }
    eligible = all(gates.values())
    return pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "eligible": eligible,
                "verdict": "retrospective_forward_watch_only"
                if eligible
                else "reject_community_volatility_path",
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
                "max_positive_community_share": community_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        ]
    )


def write_v116_community_volatility_path(
    cfg: V116Config = V116Config(),
) -> dict[str, Path]:
    panel = load_v116_panel(cfg.kline_root, cfg.membership_path)
    contexts, membership = build_v116_contexts(panel, cfg)
    real = build_v116_events(contexts, cfg)
    shifted = build_v116_events(contexts, cfg, signal_shift_hours=24)
    summary = summarize_v116(real)
    controls = random_v116_controls(contexts, cfg)
    audit = audit_v116(real, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_balanced_membership.csv",
        "events": root / "community_volatility_path_events.parquet",
        "shifted": root / "shifted_path_events.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_partition_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["membership"], index=False)
    real.to_parquet(outputs["events"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    row = audit.iloc[0]
    lines = [
        "# v11.6 Community Volatility Path Efficiency",
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
