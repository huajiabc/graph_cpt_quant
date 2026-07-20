"""Preregistered payoff reveal for alt-first volatility ignition."""

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
    simulate_v234_oco,
)
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    build_v233_hourly_context,
    load_v233_btc_15m,
)
from pressure_graph.reports.v2320_alt_first_volatility_ignition_feature_audit import (
    REPORT_ROOT as V2320_ROOT,
    feature_hash_v2320,
)


REPORT_ROOT = Path("reports/v23_21_alt_first_volatility_ignition_breakout")
FINDINGS_PATH = Path(
    "docs/v2321_alt_first_volatility_ignition_breakout_findings_2026_07_17.md"
)
CANDIDATE = "AVI1_ALT_FIRST_VOLATILITY_IGNITION_BTC_BREAKOUT"
CONTROL = "AVI1_BTC_QUIET_WITHOUT_BROAD_ALT_SHOCK_CONTROL"
FROZEN_FEATURE_HASH = "C4F814ADD57330518B98C6ABFA2CCC98A7A4BC7EC3814D2B8DB7F9118A478B6F"


@dataclass(frozen=True)
class V2321Config:
    v2320_root: Path = V2320_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    primary_sigma_multiple: float = 0.75
    adjacent_sigma_multiples: tuple[float, ...] = (0.625, 1.0)
    path_hours: int = 4
    bar_minutes: int = 15
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    event_exclusion_hours: int = 8
    nearest_controls: int = 10
    minimum_controls: int = 4
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    seed: int = 20260717


def load_v2321_inputs(
    cfg: V2321Config = V2321Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(cfg.v2320_root / "alt_first_ignition_features.parquet")
    states = pd.read_parquet(cfg.v2320_root / "hourly_ignition_states.parquet")
    for frame, columns in (
        (features, ("feature_time", "entry_time")),
        (states, ("decision_time",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    bars = load_v233_btc_15m(V233Config())
    return features, states, bars


def _simulation_config(cfg: V2321Config) -> V234Config:
    return V234Config(
        primary_sigma_multiple=cfg.primary_sigma_multiple,
        path_hours=cfg.path_hours,
        bar_minutes=cfg.bar_minutes,
        primary_cost=cfg.primary_cost,
        stress_cost=cfg.stress_cost,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )


def build_v2321_control_universe(
    features: pd.DataFrame,
    states: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V2321Config = V2321Config(),
) -> pd.DataFrame:
    hourly = build_v233_hourly_context(bars, V233Config())
    universe = hourly.merge(
        states.rename(columns={"decision_time": "entry_time"}),
        on="entry_time",
        how="inner",
        validate="one_to_one",
    )
    universe["entry_month"] = universe["entry_time"].dt.strftime("%Y-%m")
    universe["utc_hour"] = universe["entry_time"].dt.hour
    universe["period"] = np.select(
        [
            universe["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            universe["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    available = set(bars["bar_open_time"])
    expected = cfg.path_hours * 60 // cfg.bar_minutes
    universe["path_timestamp_count"] = [
        sum(
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset) in available
            for offset in range(expected)
        )
        for entry in universe["entry_time"]
    ]
    excluded = {
        pd.Timestamp(entry) + pd.Timedelta(hours=offset)
        for entry in features["entry_time"]
        for offset in range(-cfg.event_exclusion_hours, cfg.event_exclusion_hours + 1)
    }
    first_month = features["entry_time"].min().floor("D").replace(day=1)
    last_month = (
        features["entry_time"].max().floor("D").replace(day=1)
        + pd.offsets.MonthEnd(1)
        + pd.Timedelta(hours=23)
    )
    ready = (
        universe["entry_time"].between(first_month, last_month)
        & universe["btc_still_quiet"]
        & ~universe["alt_shock_ready"]
        & universe["entry_spot"].gt(0)
        & universe["causal_hourly_sigma"].gt(0)
        & universe["path_timestamp_count"].eq(expected)
        & ~universe["entry_time"].isin(excluded)
    )
    keep = [
        "entry_time",
        "entry_month",
        "period",
        "utc_hour",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "causal_hourly_sigma",
        "btc_abs_move_z",
        "path_timestamp_count",
    ]
    return universe.loc[ready, keep].sort_values("entry_time").reset_index(drop=True)


def build_v2321_control_pools(
    features: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: V2321Config = V2321Config(),
) -> pd.DataFrame:
    rows = []
    for event in features.sort_values("entry_time").itertuples(index=False):
        local = universe.loc[
            universe["entry_month"].eq(event.entry_month)
            & universe["utc_hour"].eq(event.entry_time.hour)
        ].copy()
        local["sigma_distance"] = np.log(
            local["causal_hourly_sigma"] / float(event.causal_hourly_sigma)
        ).abs()
        local["btc_shock_distance"] = (
            local["btc_abs_move_z"] - float(event.btc_abs_move_z)
        ).abs()
        local["match_distance"] = local["sigma_distance"] + local[
            "btc_shock_distance"
        ]
        local = local.sort_values(["match_distance", "entry_time"]).head(
            cfg.nearest_controls
        )
        if len(local) < cfg.minimum_controls:
            continue
        for rank, control in enumerate(local.itertuples(index=False), start=1):
            rows.append(
                {
                    "event_time": event.entry_time,
                    "event_period": event.period,
                    "event_month": event.entry_month,
                    "control_time": control.entry_time,
                    "match_rank": rank,
                    "match_distance": control.match_distance,
                    "sigma_distance": control.sigma_distance,
                    "btc_shock_distance": control.btc_shock_distance,
                }
            )
    return pd.DataFrame(rows).sort_values(["event_time", "match_rank"]).reset_index(
        drop=True
    )


def summarize_v2321(outcomes: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = outcomes if scope == "all" else outcomes.loc[outcomes["period"].eq(scope)]
        rows.append(
            {
                "variant": variant,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "triggered_trades": int(local["triggered"].sum()),
                "ambiguous_trades": int(local["ambiguous_trigger"].sum()),
                "mean_gross_return_bp": float(local["gross_return"].mean() * 10_000),
                "mean_primary_net_return_bp": float(
                    local["primary_net_return"].mean() * 10_000
                ),
                "mean_stress_net_return_bp": float(
                    local["stress_net_return"].mean() * 10_000
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v2321_random_by_scope(
    outcomes: pd.DataFrame,
    controls: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V2321Config = V2321Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = []
    summaries = []
    for scope in ("all", "development", "validation", "holdout"):
        local = outcomes if scope == "all" else outcomes.loc[outcomes["period"].eq(scope)]
        local_pools = pools.loc[pools["event_time"].isin(set(local["entry_time"]))]
        result = build_v234_random_paths(
            local,
            controls,
            local_pools,
            V234Config(random_iterations=cfg.random_iterations, seed=cfg.seed),
        )
        result["scope"] = scope
        paths.append(result)
        event_mean = float(result["event_mean_primary_net"].iloc[0])
        matched = int(result["matched_events"].iloc[0])
        summaries.append(
            {
                "scope": scope,
                "events": len(local),
                "matched_events": matched,
                "unmatched_events": len(local) - matched,
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


def build_v2321_leave_one_month_out(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month in sorted(outcomes["entry_month"].unique()):
        local = outcomes.loc[outcomes["entry_month"].ne(month)]
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


def decide_v2321(
    summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    variants: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leaveout: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    primary = summary.set_index("scope")
    period_rows = primary.loc[["development", "validation", "holdout"]]
    trigger_margin = min(
        int(primary.loc["all", "triggered_trades"]) - 80,
        int(period_rows["triggered_trades"].min()) - 15,
    )
    bootstrap_low = float(
        bootstrap["mean_primary_net_return"].quantile(0.025) * 10_000
    )
    adjacent = variants.loc[~np.isclose(variants["sigma_multiple"], 0.75)]
    rows = [
        ("minimum_trigger_count", trigger_margin >= 0, float(trigger_margin)),
        (
            "primary_positive_all_scopes",
            primary["mean_primary_net_return_bp"].gt(0).all(),
            float(primary["mean_primary_net_return_bp"].min()),
        ),
        (
            "stress_positive_all_scopes",
            primary["mean_stress_net_return_bp"].gt(0).all(),
            float(primary["mean_stress_net_return_bp"].min()),
        ),
        (
            "absolute_month_bootstrap_lower_above_zero",
            bootstrap_low > 0,
            bootstrap_low,
        ),
        (
            "matched_random_percentile_at_least_90_all_scopes",
            random_summary["matched_random_percentile"].ge(90).all(),
            float(random_summary["matched_random_percentile"].min()),
        ),
        (
            "every_event_has_at_least_four_controls",
            random_summary["unmatched_events"].eq(0).all(),
            float(random_summary["unmatched_events"].max()),
        ),
        (
            "same_bar_ambiguity_at_most_10pct",
            primary.loc["all", "ambiguous_trades"]
            / primary.loc["all", "triggered_trades"]
            <= 0.10,
            float(
                primary.loc["all", "ambiguous_trades"]
                / primary.loc["all", "triggered_trades"]
            ),
        ),
        (
            "leave_one_month_out_minimum_above_zero",
            leaveout["mean_primary_net_return_bp"].min() > 0,
            float(leaveout["mean_primary_net_return_bp"].min()),
        ),
        (
            "adjacent_widths_positive_all_scopes",
            adjacent["mean_primary_net_return_bp"].gt(0).all(),
            float(adjacent["mean_primary_net_return_bp"].min()),
        ),
    ]
    gates = pd.DataFrame(
        [
            {"gate": gate, "passed": bool(passed), "observed": observed}
            for gate, passed, observed in rows
        ]
    )
    verdict = (
        "alt_first_ignition_research_candidate_requires_forward"
        if bool(gates["passed"].all())
        else "alt_first_ignition_breakout_rejected"
    )
    return gates, verdict


def write_v2321_alt_first_volatility_ignition_breakout(
    cfg: V2321Config = V2321Config(),
) -> dict[str, Path]:
    features, states, bars = load_v2321_inputs(cfg)
    if feature_hash_v2320(features) != FROZEN_FEATURE_HASH:
        raise RuntimeError("v23.20 feature hash differs from preregistration")
    sim_cfg = _simulation_config(cfg)
    universe = build_v2321_control_universe(features, states, bars, cfg)
    pools = build_v2321_control_pools(features, universe, cfg)
    outcomes = simulate_v234_oco(
        features,
        bars,
        sim_cfg,
        sigma_multiple=cfg.primary_sigma_multiple,
        candidate=CANDIDATE,
    )
    controls = simulate_v234_oco(
        universe,
        bars,
        sim_cfg,
        sigma_multiple=cfg.primary_sigma_multiple,
        candidate=CONTROL,
    )
    variant_frames = []
    variant_summaries = []
    for multiple in (cfg.primary_sigma_multiple, *cfg.adjacent_sigma_multiples):
        local = simulate_v234_oco(
            features,
            bars,
            sim_cfg,
            sigma_multiple=multiple,
            candidate=CANDIDATE,
        )
        variant_frames.append(local)
        summary = summarize_v2321(local, variant=f"{multiple:g}sigma")
        summary["sigma_multiple"] = multiple
        variant_summaries.append(summary)
    all_variants = pd.concat(variant_frames, ignore_index=True)
    variant_summary = pd.concat(variant_summaries, ignore_index=True)
    primary_summary = summarize_v2321(outcomes, variant="0.75sigma")
    random_paths, random_summary = build_v2321_random_by_scope(
        outcomes,
        controls,
        pools,
        cfg,
    )
    bootstrap = build_v234_month_bootstrap(outcomes, sim_cfg)
    leaveout = build_v2321_leave_one_month_out(outcomes)
    gates, verdict = decide_v2321(
        primary_summary,
        random_summary,
        variant_summary,
        bootstrap,
        leaveout,
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "outcomes": root / "event_outcomes.parquet",
        "control_universe": root / "control_universe.parquet",
        "control_pools": root / "matched_control_pools.parquet",
        "control_outcomes": root / "control_outcomes.parquet",
        "variants": root / "barrier_variant_outcomes.parquet",
        "summary": root / "result_summary.csv",
        "variant_summary": root / "barrier_variant_summary.csv",
        "random_paths": root / "matched_random_paths.parquet",
        "random_summary": root / "matched_random_summary.csv",
        "bootstrap": root / "absolute_month_bootstrap.parquet",
        "leaveout": root / "leave_one_month_out.csv",
        "gates": root / "evidence_gates.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    outcomes.to_parquet(paths["outcomes"], index=False)
    universe.to_parquet(paths["control_universe"], index=False)
    pools.to_parquet(paths["control_pools"], index=False)
    controls.to_parquet(paths["control_outcomes"], index=False)
    all_variants.to_parquet(paths["variants"], index=False)
    primary_summary.to_csv(paths["summary"], index=False)
    variant_summary.to_csv(paths["variant_summary"], index=False)
    random_paths.to_parquet(paths["random_paths"], index=False)
    random_summary.to_csv(paths["random_summary"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    leaveout.to_csv(paths["leaveout"], index=False)
    gates.to_csv(paths["gates"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "verdict": verdict,
                "all_gates_passed": bool(gates["passed"].all()),
                "feature_hash": feature_hash_v2320(features),
                "config": {
                    **asdict(cfg),
                    "v2320_root": str(cfg.v2320_root),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                },
                "scope": "research_only_no_live_authorization",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.21 Alt-First Volatility Ignition Breakout Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                primary_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                random_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                gates.to_markdown(index=False, floatfmt=".4f"),
                "",
                "This result uses the frozen price-only alt-first state and a quiet-BTC",
                "matched control. It does not authorize live or PaperLive execution.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2321Config",
    "build_v2321_control_pools",
    "build_v2321_control_universe",
    "build_v2321_leave_one_month_out",
    "build_v2321_random_by_scope",
    "decide_v2321",
    "load_v2321_inputs",
    "summarize_v2321",
    "write_v2321_alt_first_volatility_ignition_breakout",
]
