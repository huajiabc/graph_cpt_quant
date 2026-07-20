"""Frozen outcome reveal for q90 pressure plus broad taker-buy confirmation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2323_q90_broad_taker_confirmation_feature_audit import (
    REPORT_ROOT as V2323_ROOT,
    V2323Config,
    feature_hash_v2323,
    load_v2323_inputs,
)
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    V233Config,
    build_v233_hourly_context,
    load_v233_btc_15m,
)
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    V234Config,
    build_v234_random_paths,
)


REPORT_ROOT = Path("reports/v23_24_q90_broad_taker_confirmation")
FINDINGS_PATH = Path(
    "docs/v2324_q90_broad_taker_confirmation_findings_2026_07_17.md"
)
CANDIDATE = "BTF1_Q90_POSITIVE_PRESSURE_BROAD_TAKER_BUY"
CONTROL = "BTF1_BROAD_TAKER_BUY_WITHOUT_Q90_CONTROL"
FROZEN_FEATURE_HASH = "7C33473808AC4C2CFE3DBB81FD71642C2D6C3814D405D9925EE1D495B6A1C7DD"


@dataclass(frozen=True)
class V2324Config:
    v2323_root: Path = V2323_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_buy_symbols: int = 9
    holding_hours: int = 4
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    delay_minutes: int = 15
    event_exclusion_hours: int = 8
    nearest_controls: int = 10
    minimum_controls: int = 5
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    permutation_iterations: int = 5000
    seed: int = 20260717


def load_v2324_inputs(
    cfg: V2324Config = V2324Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    confirmed = pd.read_parquet(cfg.v2323_root / "broad_taker_confirmed_features.parquet")
    context = pd.read_parquet(cfg.v2323_root / "positive_q90_taker_context.parquet")
    for frame in (confirmed, context):
        for column in ("feature_time", "entry_time", "metric_feature_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    _, metrics = load_v2323_inputs(V2323Config(report_root=cfg.v2323_root))
    bars = load_v233_btc_15m(V233Config())
    return confirmed, context, metrics, bars


def _attach_btc_context(features: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if {"entry_spot", "causal_hourly_sigma"}.issubset(features.columns):
        return features.copy()
    hourly = build_v233_hourly_context(bars, V233Config())
    return features.merge(hourly, on="entry_time", how="left", validate="one_to_one")


def price_v2324_long(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V2324Config = V2324Config(),
    *,
    delay_minutes: int = 0,
    candidate: str = CANDIDATE,
) -> pd.DataFrame:
    indexed = bars.set_index("bar_open_time").sort_index()
    attached = _attach_btc_context(features, bars)
    rows = []
    for event in attached.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        fill_time = entry + pd.Timedelta(minutes=delay_minutes)
        exit_bar_time = entry + pd.Timedelta(hours=cfg.holding_hours) - pd.Timedelta(
            minutes=15
        )
        if fill_time not in indexed.index or exit_bar_time not in indexed.index:
            continue
        fill_bar = indexed.loc[fill_time]
        exit_bar = indexed.loc[exit_bar_time]
        fill = max(float(event.entry_spot), float(fill_bar["open"]))
        exit_spot = float(exit_bar["close"])
        gross = exit_spot / fill - 1.0
        row = event._asdict()
        row.update(
            {
                "candidate": candidate,
                "fill_time": fill_time,
                "delay_minutes": delay_minutes,
                "fill_price": fill,
                "exit_time": entry + pd.Timedelta(hours=cfg.holding_hours),
                "exit_spot": exit_spot,
                "gross_return": gross,
                "primary_net_return": gross - cfg.primary_cost,
                "stress_net_return": gross - cfg.stress_cost,
                "reversed_primary_net_return": -gross - cfg.primary_cost,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def build_v2324_control_universe(
    positive_q90: pd.DataFrame,
    metrics: dict[str, pd.Series],
    bars: pd.DataFrame,
    cfg: V2324Config = V2324Config(),
) -> pd.DataFrame:
    first = positive_q90["entry_time"].min().floor("D").replace(day=1)
    last = (
        positive_q90["entry_time"].max().floor("D").replace(day=1)
        + pd.offsets.MonthEnd(1)
        + pd.Timedelta(hours=23)
    )
    reference = next(iter(metrics.values()))
    times = reference.index[
        (reference.index.minute == 0)
        & (reference.index >= first)
        & (reference.index <= last)
    ]
    rows = []
    for time in times:
        values = [
            float(np.log(series.at[time]))
            for series in metrics.values()
            if time in series.index and np.isfinite(series.at[time]) and series.at[time] > 0
        ]
        if len(values) == 16 and sum(value > 0 for value in values) >= cfg.minimum_buy_symbols:
            rows.append(
                {
                    "entry_time": time,
                    "entry_month": time.strftime("%Y-%m"),
                    "utc_hour": time.hour,
                    "metric_symbol_count": len(values),
                    "taker_buy_symbol_count": sum(value > 0 for value in values),
                    "median_log_taker_ratio": float(np.median(values)),
                }
            )
    universe = pd.DataFrame(rows)
    hourly = build_v233_hourly_context(bars, V233Config())
    universe = universe.merge(hourly, on="entry_time", how="inner", validate="one_to_one")
    universe["period"] = np.select(
        [
            universe["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            universe["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    available = set(bars["bar_open_time"])
    exit_offset = pd.Timedelta(hours=cfg.holding_hours) - pd.Timedelta(minutes=15)
    universe = universe.loc[
        universe["entry_spot"].gt(0)
        & universe["causal_hourly_sigma"].gt(0)
        & universe["entry_time"].map(lambda time: time in available)
        & universe["entry_time"].map(lambda time: time + exit_offset in available)
    ].copy()
    excluded = {
        pd.Timestamp(time) + pd.Timedelta(hours=offset)
        for time in positive_q90["entry_time"]
        for offset in range(-cfg.event_exclusion_hours, cfg.event_exclusion_hours + 1)
    }
    return universe.loc[~universe["entry_time"].isin(excluded)].reset_index(drop=True)


def build_v2324_control_pools(
    confirmed: pd.DataFrame,
    universe: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V2324Config = V2324Config(),
) -> pd.DataFrame:
    events = _attach_btc_context(confirmed, bars)
    rows = []
    for event in events.itertuples(index=False):
        local = universe.loc[
            universe["entry_month"].eq(event.entry_month)
            & universe["utc_hour"].eq(event.entry_time.hour)
        ].copy()
        local["sigma_distance"] = np.log(
            local["causal_hourly_sigma"] / float(event.causal_hourly_sigma)
        ).abs()
        local["flow_distance"] = (
            local["median_log_taker_ratio"] - float(event.median_log_taker_ratio)
        ).abs()
        local["match_distance"] = local["sigma_distance"] + local["flow_distance"]
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
                    "flow_distance": control.flow_distance,
                }
            )
    return pd.DataFrame(rows).sort_values(["event_time", "match_rank"]).reset_index(
        drop=True
    )


def summarize_v2324(outcomes: pd.DataFrame, *, label: str) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = outcomes if scope == "all" else outcomes.loc[outcomes["period"].eq(scope)]
        rows.append(
            {
                "variant": label,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "mean_gross_return_bp": float(local["gross_return"].mean() * 10_000),
                "mean_primary_net_return_bp": float(
                    local["primary_net_return"].mean() * 10_000
                ),
                "mean_stress_net_return_bp": float(
                    local["stress_net_return"].mean() * 10_000
                ),
                "win_rate_primary": float(local["primary_net_return"].gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_v2324_random_by_scope(
    outcomes: pd.DataFrame,
    controls: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V2324Config = V2324Config(),
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


def build_v2324_bootstrap(
    outcomes: pd.DataFrame,
    cfg: V2324Config = V2324Config(),
) -> pd.DataFrame:
    groups = {
        month: local["primary_net_return"].to_numpy(float)
        for month, local in outcomes.groupby("entry_month", sort=True)
    }
    months = sorted(groups)
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        sampled = rng.choice(months, size=len(months), replace=True)
        mean = float(np.concatenate([groups[month] for month in sampled]).mean())
        rows.append({"iteration": iteration, "mean_primary_net_return": mean})
    return pd.DataFrame(rows)


def build_v2324_leaveout(outcomes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "excluded_month": month,
                "remaining_events": int(outcomes["entry_month"].ne(month).sum()),
                "mean_primary_net_return_bp": float(
                    outcomes.loc[
                        outcomes["entry_month"].ne(month), "primary_net_return"
                    ].mean()
                    * 10_000
                ),
            }
            for month in sorted(outcomes["entry_month"].unique())
        ]
    )


def build_v2324_label_permutation(
    confirmed: pd.DataFrame,
    unconfirmed: pd.DataFrame,
    cfg: V2324Config = V2324Config(),
) -> tuple[pd.DataFrame, float]:
    combined = pd.concat(
        [
            confirmed.assign(confirmed_label=True),
            unconfirmed.assign(confirmed_label=False),
        ],
        ignore_index=True,
    )
    observed = float(
        confirmed["primary_net_return"].mean()
        - unconfirmed["primary_net_return"].mean()
    )
    groups = [group for _, group in combined.groupby("entry_month", sort=True)]
    rng = np.random.default_rng(cfg.seed + 2)
    rows = []
    for iteration in range(cfg.permutation_iterations):
        yes = []
        no = []
        for group in groups:
            labels = rng.permutation(group["confirmed_label"].to_numpy(bool))
            values = group["primary_net_return"].to_numpy(float)
            yes.extend(values[labels])
            no.extend(values[~labels])
        difference = float(np.mean(yes) - np.mean(no))
        rows.append({"iteration": iteration, "difference": difference})
    output = pd.DataFrame(rows)
    return output, float(output["difference"].ge(observed).mean())


def decide_v2324(
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    unconfirmed_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leaveout: pd.DataFrame,
    outcomes: pd.DataFrame,
    permutation_p: float,
) -> tuple[pd.DataFrame, str]:
    primary = summary.set_index("scope")
    delayed = delayed_summary.set_index("scope")
    unconfirmed = unconfirmed_summary.set_index("scope")
    bootstrap_low = float(
        bootstrap["mean_primary_net_return"].quantile(0.025) * 10_000
    )
    differences = primary["mean_primary_net_return_bp"] - unconfirmed[
        "mean_primary_net_return_bp"
    ]
    monthly = outcomes.groupby("entry_month", observed=True)["primary_net_return"].sum()
    positive = monthly.loc[monthly.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    rows = [
        (
            "minimum_events_and_period_coverage",
            len(outcomes) >= 24
            and outcomes["period"].value_counts().reindex(
                ["development", "validation", "holdout"], fill_value=0
            ).ge(7).all(),
            float(outcomes["period"].value_counts().min()),
        ),
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
            "leave_one_month_out_minimum_above_zero",
            leaveout["mean_primary_net_return_bp"].min() > 0,
            float(leaveout["mean_primary_net_return_bp"].min()),
        ),
        (
            "matched_random_percentile_at_least_90_all_scopes",
            random_summary["matched_random_percentile"].ge(90).all(),
            float(random_summary["matched_random_percentile"].min()),
        ),
        (
            "every_event_has_at_least_five_controls",
            random_summary["unmatched_events"].eq(0).all(),
            float(random_summary["unmatched_events"].max()),
        ),
        (
            "confirmed_beats_unconfirmed_all_scopes",
            differences.gt(0).all(),
            float(differences.min()),
        ),
        (
            "within_month_label_permutation_p_at_most_010",
            permutation_p <= 0.10,
            permutation_p,
        ),
        (
            "fifteen_minute_delay_positive_all_scopes",
            delayed["mean_primary_net_return_bp"].gt(0).all(),
            float(delayed["mean_primary_net_return_bp"].min()),
        ),
        (
            "long_beats_sign_reversed_short",
            outcomes["primary_net_return"].mean()
            > outcomes["reversed_primary_net_return"].mean(),
            float(
                (
                    outcomes["primary_net_return"].mean()
                    - outcomes["reversed_primary_net_return"].mean()
                )
                * 10_000
            ),
        ),
        (
            "positive_month_concentration_at_most_050",
            concentration <= 0.50,
            concentration,
        ),
    ]
    gates = pd.DataFrame(
        [
            {"gate": gate, "passed": bool(passed), "observed": observed}
            for gate, passed, observed in rows
        ]
    )
    verdict = (
        "broad_taker_confirmation_research_candidate_requires_forward"
        if bool(gates["passed"].all())
        else "broad_taker_confirmation_rejected"
    )
    return gates, verdict


def write_v2324_q90_broad_taker_confirmation(
    cfg: V2324Config = V2324Config(),
) -> dict[str, Path]:
    confirmed, context, metrics, bars = load_v2324_inputs(cfg)
    if feature_hash_v2323(confirmed) != FROZEN_FEATURE_HASH:
        raise RuntimeError("v23.23 feature hash differs from preregistration")
    unconfirmed_features = context.loc[
        context["taker_buy_symbol_count"].lt(cfg.minimum_buy_symbols)
    ].copy()
    outcomes = price_v2324_long(confirmed, bars, cfg)
    delayed = price_v2324_long(
        confirmed,
        bars,
        cfg,
        delay_minutes=cfg.delay_minutes,
        candidate=f"{CANDIDATE}_DELAYED_15M",
    )
    unconfirmed = price_v2324_long(
        unconfirmed_features,
        bars,
        cfg,
        candidate=f"{CANDIDATE}_UNCONFIRMED_Q90",
    )
    universe = build_v2324_control_universe(context, metrics, bars, cfg)
    pools = build_v2324_control_pools(confirmed, universe, bars, cfg)
    controls = price_v2324_long(universe, bars, cfg, candidate=CONTROL)
    summary = summarize_v2324(outcomes, label="confirmed")
    delayed_summary = summarize_v2324(delayed, label="delay_15m")
    unconfirmed_summary = summarize_v2324(unconfirmed, label="unconfirmed")
    random_paths, random_summary = build_v2324_random_by_scope(
        outcomes,
        controls,
        pools,
        cfg,
    )
    bootstrap = build_v2324_bootstrap(outcomes, cfg)
    leaveout = build_v2324_leaveout(outcomes)
    permutations, permutation_p = build_v2324_label_permutation(
        outcomes,
        unconfirmed,
        cfg,
    )
    gates, verdict = decide_v2324(
        summary,
        delayed_summary,
        unconfirmed_summary,
        random_summary,
        bootstrap,
        leaveout,
        outcomes,
        permutation_p,
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "outcomes": root / "confirmed_event_outcomes.parquet",
        "delayed": root / "delayed_event_outcomes.parquet",
        "unconfirmed": root / "unconfirmed_event_outcomes.parquet",
        "control_universe": root / "control_universe.parquet",
        "control_pools": root / "matched_control_pools.parquet",
        "controls": root / "control_outcomes.parquet",
        "summary": root / "result_summary.csv",
        "delayed_summary": root / "delayed_summary.csv",
        "unconfirmed_summary": root / "unconfirmed_summary.csv",
        "random_paths": root / "matched_random_paths.parquet",
        "random_summary": root / "matched_random_summary.csv",
        "bootstrap": root / "absolute_month_bootstrap.parquet",
        "leaveout": root / "leave_one_month_out.csv",
        "permutations": root / "within_month_label_permutations.parquet",
        "gates": root / "evidence_gates.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    frames = {
        "outcomes": outcomes,
        "delayed": delayed,
        "unconfirmed": unconfirmed,
        "control_universe": universe,
        "control_pools": pools,
        "controls": controls,
        "random_paths": random_paths,
        "bootstrap": bootstrap,
        "permutations": permutations,
    }
    for key, frame in frames.items():
        frame.to_parquet(paths[key], index=False)
    for key, frame in (
        ("summary", summary),
        ("delayed_summary", delayed_summary),
        ("unconfirmed_summary", unconfirmed_summary),
        ("random_summary", random_summary),
        ("leaveout", leaveout),
        ("gates", gates),
    ):
        frame.to_csv(paths[key], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "verdict": verdict,
                "all_gates_passed": bool(gates["passed"].all()),
                "feature_hash": feature_hash_v2323(confirmed),
                "permutation_p": permutation_p,
                "config": {
                    **asdict(cfg),
                    "v2323_root": str(cfg.v2323_root),
                    "report_root": str(cfg.report_root),
                    "findings_path": str(cfg.findings_path),
                },
                "scope": "research_only_post_selected_ancestor",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.24 q90 Broad-Taker Confirmation Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                random_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                gates.to_markdown(index=False, floatfmt=".4f"),
                "",
                "This is a second-stage attribution test of a post-selected q90",
                "ancestor and cannot independently authorize deployment.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2324Config",
    "build_v2324_bootstrap",
    "build_v2324_control_pools",
    "build_v2324_control_universe",
    "build_v2324_label_permutation",
    "build_v2324_leaveout",
    "build_v2324_random_by_scope",
    "decide_v2324",
    "load_v2324_inputs",
    "price_v2324_long",
    "summarize_v2324",
    "write_v2324_q90_broad_taker_confirmation",
]
