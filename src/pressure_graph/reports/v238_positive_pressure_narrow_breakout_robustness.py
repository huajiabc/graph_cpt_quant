"""Post-selection robustness for the positive-pressure narrow BTC breakout."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    V234Config,
    build_v234_month_bootstrap,
    build_v234_random_paths,
    load_v234_inputs,
    simulate_v234_oco,
)


V234_ROOT = Path("reports/v23_4_book_vacuum_oco_breakout")
REPORT_ROOT = Path("reports/v23_8_positive_pressure_narrow_breakout_robustness")
FINDINGS_PATH = Path(
    "docs/v238_positive_pressure_narrow_breakout_robustness_2026_07_17.md"
)
CANDIDATE = "DVB5_POSITIVE_PRESSURE_0625SIGMA_BTC_BREAKOUT"


@dataclass(frozen=True)
class V238Config:
    v234_root: Path = V234_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    primary_sigma_multiple: float = 0.625
    adjacent_sigma_multiple: float = 0.75
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    permutation_iterations: int = 5000
    seed: int = 20260717


def simulate_v238_latency_horizon(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V238Config = V238Config(),
    *,
    horizon_hours: int,
    activation_delay_minutes: int,
) -> pd.DataFrame:
    indexed = bars.set_index("bar_open_time").sort_index()
    rows = []
    for event in features.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        start = entry + pd.Timedelta(minutes=activation_delay_minutes)
        bar_count = (horizon_hours * 60 - activation_delay_minutes) // 15
        times = [start + pd.Timedelta(minutes=15 * offset) for offset in range(bar_count)]
        if not times or any(time not in indexed.index for time in times):
            continue
        path = indexed.loc[times]
        upper = float(event.entry_spot) * np.exp(
            cfg.primary_sigma_multiple * float(event.causal_hourly_sigma)
        )
        lower = float(event.entry_spot) * np.exp(
            -cfg.primary_sigma_multiple * float(event.causal_hourly_sigma)
        )
        exit_spot = float(path.iloc[-1]["close"])
        triggered = False
        ambiguous = False
        gross = 0.0
        trigger_delay = np.nan
        for time, bar in path.iterrows():
            upper_hit = float(bar["high"]) >= upper
            lower_hit = float(bar["low"]) <= lower
            if not upper_hit and not lower_hit:
                continue
            triggered = True
            trigger_delay = (pd.Timestamp(time) - entry).total_seconds() / 60.0
            long_fill = max(upper, float(bar["open"]))
            short_fill = min(lower, float(bar["open"]))
            long_return = exit_spot / long_fill - 1.0
            short_return = 1.0 - exit_spot / short_fill
            if upper_hit and lower_hit:
                ambiguous = True
                gross = min(long_return, short_return)
            elif upper_hit:
                gross = long_return
            else:
                gross = short_return
            break
        rows.append(
            {
                "entry_time": entry,
                "entry_month": event.entry_month,
                "period": event.period,
                "horizon_hours": horizon_hours,
                "activation_delay_minutes": activation_delay_minutes,
                "triggered": triggered,
                "ambiguous_trigger": ambiguous,
                "trigger_delay_minutes": trigger_delay,
                "gross_return": gross,
                "primary_net_return": gross
                - (cfg.primary_cost if triggered else 0.0),
                "stress_net_return": gross
                - (cfg.stress_cost if triggered else 0.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_v238_periods(
    outcomes: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = outcomes if scope == "all" else outcomes[outcomes["period"].eq(scope)]
        rows.append(
            {
                "variant": label,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "triggered_trades": int(local["triggered"].sum()),
                "ambiguous_trades": int(local["ambiguous_trigger"].sum()),
                "mean_primary_net_return_bp": float(
                    local["primary_net_return"].mean() * 10_000
                ),
                "mean_stress_net_return_bp": float(
                    local["stress_net_return"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def _matched_random_by_scope(
    outcomes: pd.DataFrame,
    controls: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V238Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = []
    summaries = []
    for scope in ("all", "development", "validation", "holdout"):
        local = outcomes if scope == "all" else outcomes[outcomes["period"].eq(scope)]
        local_times = set(local["entry_time"])
        local_pools = pools[pools["event_time"].isin(local_times)]
        result = build_v234_random_paths(
            local,
            controls,
            local_pools,
            V234Config(
                random_iterations=cfg.random_iterations,
                seed=cfg.seed,
            ),
        )
        result["scope"] = scope
        paths.append(result)
        event_mean = float(result["event_mean_primary_net"].iloc[0])
        matched_events = int(result["matched_events"].iloc[0])
        summaries.append(
            {
                "scope": scope,
                "events": len(local),
                "matched_events": matched_events,
                "unmatched_events": len(local) - matched_events,
                "event_mean_bp": event_mean * 10_000,
                "matched_random_percentile": float(
                    result["control_mean_primary_net"].le(event_mean).mean() * 100
                ),
                "random_median_bp": float(
                    result["control_mean_primary_net"].median() * 10_000
                ),
            }
        )
    return pd.concat(paths, ignore_index=True), pd.DataFrame(summaries)


def _within_month_sign_permutation(
    all_outcomes: pd.DataFrame,
    cfg: V238Config,
) -> tuple[pd.DataFrame, float]:
    positive = all_outcomes[all_outcomes["signal_direction"].eq(1)]
    negative = all_outcomes[all_outcomes["signal_direction"].eq(-1)]
    observed = float(
        (positive["primary_net_return"].mean() - negative["primary_net_return"].mean())
        * 10_000
    )
    groups = [
        local[["signal_direction", "primary_net_return"]]
        for _, local in all_outcomes.groupby("entry_month", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed + 2)
    rows = []
    for iteration in range(cfg.permutation_iterations):
        positive_values = []
        negative_values = []
        for group in groups:
            labels = rng.permutation(group["signal_direction"].to_numpy())
            values = group["primary_net_return"].to_numpy(dtype=float)
            positive_values.extend(values[labels == 1])
            negative_values.extend(values[labels == -1])
        difference = (np.mean(positive_values) - np.mean(negative_values)) * 10_000
        rows.append(
            {"iteration": iteration, "permuted_sign_difference_bp": difference}
        )
    output = pd.DataFrame(rows)
    p_upper = float(output["permuted_sign_difference_bp"].ge(observed).mean())
    return output, p_upper


def _month_sign_bootstrap(
    all_outcomes: pd.DataFrame,
    cfg: V238Config,
) -> pd.DataFrame:
    groups = {
        month: local
        for month, local in all_outcomes.groupby("entry_month", sort=True)
    }
    months = sorted(groups)
    rng = np.random.default_rng(cfg.seed + 3)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        sampled = rng.choice(months, size=len(months), replace=True)
        local = pd.concat([groups[month] for month in sampled], ignore_index=True)
        difference = (
            local.loc[
                local["signal_direction"].eq(1), "primary_net_return"
            ].mean()
            - local.loc[
                local["signal_direction"].eq(-1), "primary_net_return"
            ].mean()
        ) * 10_000
        rows.append(
            {"iteration": iteration, "sign_difference_bp": float(difference)}
        )
    return pd.DataFrame(rows)


def _leave_one_month_out(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month in sorted(outcomes["entry_month"].unique()):
        local = outcomes[outcomes["entry_month"].ne(month)]
        rows.append(
            {
                "excluded_month": month,
                "remaining_events": len(local),
                "mean_primary_net_return_bp": float(
                    local["primary_net_return"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_v238(gates: pd.DataFrame) -> str:
    absolute = gates.loc[
        gates["gate"].eq("absolute_month_bootstrap_lower_above_zero"), "passed"
    ].iloc[0]
    structural = gates.loc[
        ~gates["gate"].eq("absolute_month_bootstrap_lower_above_zero"), "passed"
    ].all()
    if bool(absolute) and bool(structural):
        return "post_selection_evidence_complete_requires_new_forward_sample"
    if bool(structural):
        return "forward_shadow_candidate_not_statistically_confirmed"
    return "positive_pressure_narrow_breakout_rejected"


def write_v238_positive_pressure_narrow_breakout_robustness(
    cfg: V238Config = V238Config(),
) -> dict[str, Path]:
    features, _, bars = load_v234_inputs(V234Config())
    positive_features = features[features["signal_direction"].eq(1)].copy()
    primary = simulate_v234_oco(
        positive_features,
        bars,
        V234Config(),
        sigma_multiple=cfg.primary_sigma_multiple,
        candidate=CANDIDATE,
    )
    adjacent = simulate_v234_oco(
        positive_features,
        bars,
        V234Config(),
        sigma_multiple=cfg.adjacent_sigma_multiple,
        candidate=CANDIDATE,
    )
    all_primary_width = simulate_v234_oco(
        features,
        bars,
        V234Config(),
        sigma_multiple=cfg.primary_sigma_multiple,
    )
    universe = pd.read_parquet(cfg.v234_root / "causal_control_universe.parquet")
    pools = pd.read_parquet(cfg.v234_root / "matched_control_pools.parquet")
    universe["entry_time"] = pd.to_datetime(universe["entry_time"], utc=True)
    for column in ("event_time", "control_time"):
        pools[column] = pd.to_datetime(pools[column], utc=True)
    positive_times = set(primary["entry_time"])
    positive_pools = pools[pools["event_time"].isin(positive_times)]
    controls = simulate_v234_oco(
        universe,
        bars,
        V234Config(),
        sigma_multiple=cfg.primary_sigma_multiple,
    )
    random_paths, random_summary = _matched_random_by_scope(
        primary, controls, positive_pools, cfg
    )
    absolute_bootstrap = build_v234_month_bootstrap(
        primary,
        V234Config(
            bootstrap_iterations=cfg.bootstrap_iterations,
            seed=cfg.seed,
        ),
    )
    permutation, permutation_p = _within_month_sign_permutation(
        all_primary_width, cfg
    )
    sign_bootstrap = _month_sign_bootstrap(all_primary_width, cfg)
    leave_one_out = _leave_one_month_out(primary)

    base_summary = pd.concat(
        [
            summarize_v238_periods(primary, label="0.625sigma_primary"),
            summarize_v238_periods(adjacent, label="0.75sigma_adjacent"),
        ],
        ignore_index=True,
    )
    sensitivity_frames = []
    for horizon in (3, 4, 6):
        for delay in (0, 15, 30):
            sensitivity = simulate_v238_latency_horizon(
                positive_features,
                bars,
                cfg,
                horizon_hours=horizon,
                activation_delay_minutes=delay,
            )
            sensitivity_frames.append(sensitivity)
    sensitivity_outcomes = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity_summary = []
    for (horizon, delay), local in sensitivity_outcomes.groupby(
        ["horizon_hours", "activation_delay_minutes"], sort=True
    ):
        label = f"0.625sigma_{horizon}h_delay{delay}m"
        sensitivity_summary.append(summarize_v238_periods(local, label=label))
    sensitivity_summary_frame = pd.concat(sensitivity_summary, ignore_index=True)

    primary_rows = base_summary[base_summary["variant"].eq("0.625sigma_primary")]
    adjacent_rows = base_summary[base_summary["variant"].eq("0.75sigma_adjacent")]
    latency15 = sensitivity_summary_frame[
        sensitivity_summary_frame["variant"].eq("0.625sigma_4h_delay15m")
    ]
    horizons_3_4 = sensitivity_summary_frame[
        sensitivity_summary_frame["variant"].isin(
            ["0.625sigma_3h_delay0m", "0.625sigma_4h_delay0m"]
        )
    ]
    absolute_lower = float(
        absolute_bootstrap["mean_primary_net_return"].quantile(0.025) * 10_000
    )
    sign_lower = float(sign_bootstrap["sign_difference_bp"].quantile(0.025))
    gates = {
        "primary_positive_all_temporal_scopes": bool(
            primary_rows["mean_primary_net_return_bp"].gt(0).all()
        ),
        "adjacent_width_positive_all_temporal_scopes": bool(
            adjacent_rows["mean_primary_net_return_bp"].gt(0).all()
        ),
        "primary_stress_positive_full_sample": float(
            primary_rows.loc[primary_rows["scope"].eq("all"), "mean_stress_net_return_bp"].iloc[0]
        )
        > 0,
        "fifteen_minute_latency_primary_positive_all_scopes": bool(
            latency15["mean_primary_net_return_bp"].gt(0).all()
        ),
        "three_and_four_hour_primary_positive_all_scopes": bool(
            horizons_3_4["mean_primary_net_return_bp"].gt(0).all()
        ),
        "matched_random_percentile_at_least_90_all_scopes": bool(
            random_summary["matched_random_percentile"].ge(90).all()
        ),
        "within_month_sign_permutation_upper_p_at_most_1pct": permutation_p <= 0.01,
        "month_bootstrap_sign_difference_lower_above_zero": sign_lower > 0,
        "leave_one_month_out_minimum_above_zero": float(
            leave_one_out["mean_primary_net_return_bp"].min()
        )
        > 0,
        "absolute_month_bootstrap_lower_above_zero": absolute_lower > 0,
    }
    observed = [
        float(primary_rows["mean_primary_net_return_bp"].min()),
        float(adjacent_rows["mean_primary_net_return_bp"].min()),
        float(
            primary_rows.loc[
                primary_rows["scope"].eq("all"), "mean_stress_net_return_bp"
            ].iloc[0]
        ),
        float(latency15["mean_primary_net_return_bp"].min()),
        float(horizons_3_4["mean_primary_net_return_bp"].min()),
        float(random_summary["matched_random_percentile"].min()),
        permutation_p,
        sign_lower,
        float(leave_one_out["mean_primary_net_return_bp"].min()),
        absolute_lower,
    ]
    gate_frame = pd.DataFrame(
        {"gate": list(gates), "passed": list(gates.values()), "observed": observed}
    )
    verdict = classify_v238(gate_frame)

    root = ensure_dir(cfg.report_root)
    paths = {
        "primary": root / "primary_positive_pressure_outcomes.parquet",
        "adjacent": root / "adjacent_width_outcomes.parquet",
        "sensitivity_outcomes": root / "latency_horizon_outcomes.parquet",
        "base_summary": root / "base_summary.csv",
        "sensitivity_summary": root / "latency_horizon_summary.csv",
        "random_paths": root / "matched_random_paths.parquet",
        "random_summary": root / "matched_random_summary.csv",
        "absolute_bootstrap": root / "absolute_month_bootstrap.parquet",
        "permutation": root / "within_month_sign_permutations.parquet",
        "sign_bootstrap": root / "month_sign_difference_bootstrap.parquet",
        "leave_one_out": root / "leave_one_month_out.csv",
        "gates": root / "evidence_gates.csv",
        "config": root / "analysis_config.json",
        "findings": cfg.findings_path,
    }
    primary.to_parquet(paths["primary"], index=False)
    adjacent.to_parquet(paths["adjacent"], index=False)
    sensitivity_outcomes.to_parquet(paths["sensitivity_outcomes"], index=False)
    base_summary.to_csv(paths["base_summary"], index=False)
    sensitivity_summary_frame.to_csv(paths["sensitivity_summary"], index=False)
    random_paths.to_parquet(paths["random_paths"], index=False)
    random_summary.to_csv(paths["random_summary"], index=False)
    absolute_bootstrap.to_parquet(paths["absolute_bootstrap"], index=False)
    permutation.to_parquet(paths["permutation"], index=False)
    sign_bootstrap.to_parquet(paths["sign_bootstrap"], index=False)
    leave_one_out.to_csv(paths["leave_one_out"], index=False)
    gate_frame.to_csv(paths["gates"], index=False)
    paths["config"].write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.8 Positive-Pressure Narrow-Breakout Robustness",
                "",
                f"Verdict: `{verdict}`.",
                "",
                "This is explicitly post-selection robustness, not a new untouched",
                "holdout test. The 0.625-sigma width and positive-pressure filter",
                "were identified after inspecting v23.4--v23.7 outcomes.",
                "",
                base_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                random_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                gate_frame.to_markdown(index=False, floatfmt=".4f"),
                "",
                f"Within-month sign permutation upper-tail p: {permutation_p:.6f}.",
                f"Month-bootstrap sign-difference 2.5% lower: {sign_lower:.4f} bp.",
                f"Absolute strategy month-bootstrap 2.5% lower: {absolute_lower:.4f} bp.",
                "",
                "The candidate is suitable only for new forward shadow observation.",
                "It is not statistically confirmed and has no PaperLive/live permission.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V238Config",
    "classify_v238",
    "simulate_v238_latency_horizon",
    "summarize_v238_periods",
    "write_v238_positive_pressure_narrow_breakout_robustness",
]
