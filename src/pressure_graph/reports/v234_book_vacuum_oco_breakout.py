"""Preregistered BTC OCO breakout test after v22.4 book-vacuum events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v233_book_vacuum_oco_breakout_feature_audit import (
    EVENT_PATH as RAW_EVENT_PATH,
    V233Config,
    build_v233_hourly_context,
    load_v233_btc_15m,
)


FEATURE_PATH = Path(
    "reports/v23_3_book_vacuum_oco_breakout_feature_audit/oco_breakout_features.parquet"
)
REPORT_ROOT = Path("reports/v23_4_book_vacuum_oco_breakout")
FINDINGS_PATH = Path("docs/v234_book_vacuum_oco_breakout_findings_2026_07_17.md")
PREREG_PATH = Path("docs/v234_book_vacuum_oco_breakout_prereg_2026_07_17.md")
CANDIDATE = "DVB3_BOOK_VACUUM_BTC_OCO_BREAKOUT"
MATCHED_CONTROL = "DVB3_MATCHED_NON_EVENT_BTC_OCO_BREAKOUT"
FEATURE_SHA256 = "20E29DC3BCF5E8702E46AE3B21B8F900BD26C42A6E43F7D7B488C847250DD828"


@dataclass(frozen=True)
class V234Config:
    feature_path: Path = FEATURE_PATH
    raw_event_path: Path = RAW_EVENT_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    primary_sigma_multiple: float = 1.0
    secondary_sigma_multiples: tuple[float, ...] = (0.75, 1.25)
    path_hours: int = 4
    bar_minutes: int = 15
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    event_exclusion_hours: int = 8
    nearest_controls: int = 10
    minimum_controls: int = 5
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v234_inputs(
    cfg: V234Config = V234Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(cfg.feature_path)
    raw_events = pd.read_parquet(cfg.raw_event_path)
    for frame in (features, raw_events):
        for column in ("feature_time", "entry_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    bars = load_v233_btc_15m(V233Config())
    return features, raw_events, bars


def build_v234_control_universe(
    event_features: pd.DataFrame,
    raw_events: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V234Config = V234Config(),
) -> pd.DataFrame:
    hourly = build_v233_hourly_context(bars, V233Config())
    first_month = event_features["entry_time"].min().floor("D").replace(day=1)
    last_month = event_features["entry_time"].max().floor("D").replace(day=1)
    last_month = last_month + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    universe = hourly[hourly["entry_time"].between(first_month, last_month)].copy()
    universe["entry_month"] = universe["entry_time"].dt.strftime("%Y-%m")
    universe["utc_hour"] = universe["entry_time"].dt.hour
    expected = cfg.path_hours * 60 // cfg.bar_minutes
    available = set(bars["bar_open_time"])
    universe["path_timestamp_count"] = [
        sum(
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset) in available
            for offset in range(expected)
        )
        for entry in universe["entry_time"]
    ]
    excluded = {
        pd.Timestamp(time) + pd.Timedelta(hours=offset)
        for time in raw_events["entry_time"]
        for offset in range(-cfg.event_exclusion_hours, cfg.event_exclusion_hours + 1)
    }
    ready = (
        universe["entry_spot"].gt(0)
        & universe["causal_hourly_sigma"].gt(0)
        & universe["path_timestamp_count"].eq(expected)
        & ~universe["entry_time"].isin(excluded)
    )
    keep = [
        "entry_time",
        "entry_month",
        "utc_hour",
        "entry_spot",
        "prior_24h_sum_squared_log_move",
        "causal_hourly_sigma",
        "path_timestamp_count",
    ]
    return universe.loc[ready, keep].sort_values("entry_time").reset_index(drop=True)


def build_v234_matched_control_pools(
    event_features: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: V234Config = V234Config(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in event_features.sort_values("entry_time").itertuples(index=False):
        local = universe[
            universe["entry_month"].eq(event.entry_month)
            & universe["utc_hour"].eq(event.entry_time.hour)
        ].copy()
        local["match_distance"] = np.log(
            local["causal_hourly_sigma"] / float(event.causal_hourly_sigma)
        ).abs()
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
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["event_time", "match_rank"]
    ).reset_index(drop=True)


def simulate_v234_oco(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V234Config = V234Config(),
    *,
    sigma_multiple: float | None = None,
    candidate: str = CANDIDATE,
) -> pd.DataFrame:
    multiple = (
        cfg.primary_sigma_multiple if sigma_multiple is None else sigma_multiple
    )
    indexed = bars.set_index("bar_open_time").sort_index()
    bar_count = cfg.path_hours * 60 // cfg.bar_minutes
    rows: list[dict[str, object]] = []
    for feature in features.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(feature.entry_time)
        times = [
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset)
            for offset in range(bar_count)
        ]
        if any(time not in indexed.index for time in times):
            continue
        path = indexed.loc[times]
        spot = float(feature.entry_spot)
        sigma = float(feature.causal_hourly_sigma)
        upper = spot * np.exp(multiple * sigma)
        lower = spot * np.exp(-multiple * sigma)
        exit_spot = float(path.iloc[-1]["close"])
        triggered = False
        ambiguous = False
        direction = 0
        trigger_time = pd.NaT
        fill_price = np.nan
        gross = 0.0
        reversed_gross = 0.0
        for time, bar in path.iterrows():
            upper_hit = float(bar["high"]) >= upper
            lower_hit = float(bar["low"]) <= lower
            if not upper_hit and not lower_hit:
                continue
            triggered = True
            trigger_time = pd.Timestamp(time)
            long_fill = max(upper, float(bar["open"]))
            short_fill = min(lower, float(bar["open"]))
            long_gross = exit_spot / long_fill - 1.0
            short_gross = 1.0 - exit_spot / short_fill
            if upper_hit and lower_hit:
                ambiguous = True
                if long_gross <= short_gross:
                    direction, fill_price, gross = 1, long_fill, long_gross
                else:
                    direction, fill_price, gross = -1, short_fill, short_gross
                reversed_gross = min(
                    1.0 - exit_spot / long_fill,
                    exit_spot / short_fill - 1.0,
                )
            elif upper_hit:
                direction, fill_price, gross = 1, long_fill, long_gross
                reversed_gross = 1.0 - exit_spot / long_fill
            else:
                direction, fill_price, gross = -1, short_fill, short_gross
                reversed_gross = exit_spot / short_fill - 1.0
            break
        cost = cfg.primary_cost if triggered else 0.0
        stress_cost = cfg.stress_cost if triggered else 0.0
        row = feature._asdict()
        row.update(
            {
                "candidate": candidate,
                "sigma_multiple": multiple,
                "upper_stop_price": upper,
                "lower_stop_price": lower,
                "exit_time": entry + pd.Timedelta(hours=cfg.path_hours),
                "exit_spot": exit_spot,
                "triggered": triggered,
                "ambiguous_trigger": ambiguous,
                "trigger_time": trigger_time,
                "trigger_delay_minutes": (
                    (trigger_time - entry).total_seconds() / 60.0
                    if triggered
                    else np.nan
                ),
                "trade_direction": direction,
                "fill_price": fill_price,
                "gross_return": gross,
                "primary_net_return": gross - cost,
                "stress_net_return": gross - stress_cost,
                "reversed_primary_net_return": reversed_gross - cost,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def build_v234_random_paths(
    event_outcomes: pd.DataFrame,
    control_outcomes: pd.DataFrame,
    pools: pd.DataFrame,
    cfg: V234Config = V234Config(),
) -> pd.DataFrame:
    event_lookup = event_outcomes.set_index("entry_time")["primary_net_return"]
    control_lookup = control_outcomes.set_index("entry_time")["primary_net_return"]
    grouped = {
        event: local["control_time"].tolist()
        for event, local in pools.groupby("event_time", sort=True)
        if event in event_lookup.index
    }
    event_times = sorted(grouped)
    event_mean = float(event_lookup.loc[event_times].mean())
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for iteration in range(cfg.random_iterations):
        sampled = [
            controls[int(rng.integers(0, len(controls)))]
            for controls in grouped.values()
        ]
        control_mean = float(control_lookup.loc[sampled].mean())
        rows.append(
            {
                "iteration": iteration,
                "matched_events": len(event_times),
                "event_mean_primary_net": event_mean,
                "control_mean_primary_net": control_mean,
                "event_minus_control": event_mean - control_mean,
            }
        )
    return pd.DataFrame(rows)


def build_v234_month_bootstrap(
    outcomes: pd.DataFrame,
    cfg: V234Config = V234Config(),
) -> pd.DataFrame:
    groups = {
        month: local["primary_net_return"].to_numpy(dtype=float)
        for month, local in outcomes.groupby("entry_month", sort=True)
    }
    months = sorted(groups)
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for iteration in range(cfg.bootstrap_iterations):
        sampled = rng.choice(months, size=len(months), replace=True)
        mean = float(np.concatenate([groups[month] for month in sampled]).mean())
        rows.append(
            {"iteration": iteration, "mean_primary_net_return": mean}
        )
    return pd.DataFrame(rows)


def summarize_v234(outcomes: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("all", outcomes),
        ("development", outcomes[outcomes["period"].eq("development")]),
        ("validation", outcomes[outcomes["period"].eq("validation")]),
        ("holdout", outcomes[outcomes["period"].eq("holdout")]),
        ("positive_pressure", outcomes[outcomes["signal_direction"].eq(1)]),
        ("negative_pressure", outcomes[outcomes["signal_direction"].eq(-1)]),
    ]
    rows: list[dict[str, object]] = []
    for scope, local in scopes:
        trades = local[local["triggered"]]
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": len(local),
                "active_months": local["entry_month"].nunique(),
                "triggered_trades": len(trades),
                "trade_rate": float(local["triggered"].mean()),
                "long_trades": int(trades["trade_direction"].eq(1).sum()),
                "short_trades": int(trades["trade_direction"].eq(-1).sum()),
                "ambiguous_trades": int(trades["ambiguous_trigger"].sum()),
                "ambiguous_trade_fraction": float(
                    trades["ambiguous_trigger"].mean() if len(trades) else np.nan
                ),
                "mean_gross_return_per_event_bp": float(
                    local["gross_return"].mean() * 10_000
                ),
                "mean_primary_net_return_per_event_bp": float(
                    local["primary_net_return"].mean() * 10_000
                ),
                "mean_stress_net_return_per_event_bp": float(
                    local["stress_net_return"].mean() * 10_000
                ),
                "mean_primary_net_return_per_trade_bp": float(
                    trades["primary_net_return"].mean() * 10_000
                ),
                "mean_reversed_primary_net_return_per_event_bp": float(
                    local["reversed_primary_net_return"].mean() * 10_000
                ),
                "median_trigger_delay_minutes": float(
                    trades["trigger_delay_minutes"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_v234_variants(variants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for multiple, local in variants.groupby("sigma_multiple", sort=True):
        for scope in ("all", "development", "validation", "holdout"):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
            rows.append(
                {
                    "sigma_multiple": multiple,
                    "scope": scope,
                    "events": len(sample),
                    "triggered_trades": int(sample["triggered"].sum()),
                    "mean_primary_net_return_per_event_bp": float(
                        sample["primary_net_return"].mean() * 10_000
                    ),
                }
            )
    return pd.DataFrame(rows)


def decide_v234(
    summary: pd.DataFrame,
    random_paths: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    indexed = summary.set_index("scope")
    all_row = indexed.loc["all"]
    period_rows = indexed.loc[["development", "validation", "holdout"]]
    lower = float(bootstrap["mean_primary_net_return"].quantile(0.025) * 10_000)
    event_mean = float(random_paths["event_mean_primary_net"].iloc[0])
    percentile = float(
        random_paths["control_mean_primary_net"].le(event_mean).mean() * 100
    )
    gates = {
        "minimum_80_triggers_and_20_each_period": int(all_row["triggered_trades"])
        >= 80
        and bool(period_rows["triggered_trades"].ge(20).all()),
        "overall_primary_net_positive": float(
            all_row["mean_primary_net_return_per_event_bp"]
        )
        > 0,
        "development_validation_holdout_primary_net_positive": bool(
            period_rows["mean_primary_net_return_per_event_bp"].gt(0).all()
        ),
        "month_block_bootstrap_lower_above_zero": lower > 0,
        "matched_random_percentile_at_least_90": percentile >= 90.0,
        "ambiguous_trigger_fraction_at_most_10pct": float(
            all_row["ambiguous_trade_fraction"]
        )
        <= 0.10,
        "primary_beats_reversed_direction": float(
            all_row["mean_primary_net_return_per_event_bp"]
        )
        > float(all_row["mean_reversed_primary_net_return_per_event_bp"]),
    }
    observed = [
        min(
            int(all_row["triggered_trades"]) - 80,
            int(period_rows["triggered_trades"].min()) - 20,
        ),
        float(all_row["mean_primary_net_return_per_event_bp"]),
        float(period_rows["mean_primary_net_return_per_event_bp"].min()),
        lower,
        percentile,
        float(all_row["ambiguous_trade_fraction"]),
        float(all_row["mean_primary_net_return_per_event_bp"])
        - float(all_row["mean_reversed_primary_net_return_per_event_bp"]),
    ]
    decision = pd.DataFrame(
        {"gate": list(gates), "passed": list(gates.values()), "observed": observed}
    )
    verdict = (
        "research_only_oco_breakout_supported"
        if bool(decision["passed"].all())
        else "oco_breakout_rejected"
    )
    return decision, verdict


def write_v234_book_vacuum_oco_breakout(
    cfg: V234Config = V234Config(),
) -> dict[str, Path]:
    if _sha256(cfg.feature_path) != FEATURE_SHA256:
        raise RuntimeError("v23.3 feature hash differs from preregistration")
    features, raw_events, bars = load_v234_inputs(cfg)
    universe = build_v234_control_universe(features, raw_events, bars, cfg)
    pools = build_v234_matched_control_pools(features, universe, cfg)
    outcomes = simulate_v234_oco(features, bars, cfg)
    control_outcomes = simulate_v234_oco(
        universe, bars, cfg, candidate=MATCHED_CONTROL
    )
    variant_frames = [outcomes]
    for multiple in cfg.secondary_sigma_multiples:
        variant_frames.append(
            simulate_v234_oco(features, bars, cfg, sigma_multiple=multiple)
        )
    variants = pd.concat(variant_frames, ignore_index=True)
    random_paths = build_v234_random_paths(outcomes, control_outcomes, pools, cfg)
    bootstrap = build_v234_month_bootstrap(outcomes, cfg)
    summary = summarize_v234(outcomes)
    variant_summary = summarize_v234_variants(variants)
    decision, verdict = decide_v234(summary, random_paths, bootstrap)

    root = ensure_dir(cfg.report_root)
    paths = {
        "outcomes": root / "oco_event_outcomes.parquet",
        "control_universe": root / "causal_control_universe.parquet",
        "control_pools": root / "matched_control_pools.parquet",
        "control_outcomes": root / "oco_control_outcomes.parquet",
        "variants": root / "barrier_variant_outcomes.parquet",
        "random_paths": root / "matched_random_paths.parquet",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "summary": root / "result_summary.csv",
        "variant_summary": root / "barrier_variant_summary.csv",
        "decision": root / "decision_gates.csv",
        "config": root / "frozen_config.json",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    outcomes.to_parquet(paths["outcomes"], index=False)
    universe.to_parquet(paths["control_universe"], index=False)
    pools.to_parquet(paths["control_pools"], index=False)
    control_outcomes.to_parquet(paths["control_outcomes"], index=False)
    variants.to_parquet(paths["variants"], index=False)
    random_paths.to_parquet(paths["random_paths"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    summary.to_csv(paths["summary"], index=False)
    variant_summary.to_csv(paths["variant_summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    paths["config"].write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8"
    )
    paths["hashes"].write_text(
        "input,sha256\n"
        + "\n".join(
            f"{path},{_sha256(path)}"
            for path in (cfg.feature_path, cfg.raw_event_path)
        )
        + "\n",
        encoding="utf-8",
    )
    percentile = float(decision.loc[
        decision["gate"].eq("matched_random_percentile_at_least_90"), "observed"
    ].iloc[0])
    lower = float(decision.loc[
        decision["gate"].eq("month_block_bootstrap_lower_above_zero"), "observed"
    ].iloc[0])
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.4 Book-Vacuum BTC OCO Breakout Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Barrier variants:",
                "",
                variant_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                f"Matched random-time percentile: {percentile:.2f}.",
                f"Month-block bootstrap 2.5% lower bound: {lower:.4f} bp.",
                "",
                "The result uses pessimistic same-bar ambiguity and gap fills.",
                "It remains a historical 15-minute bar simulation, not a guarantee",
                "of stop execution or queue position.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V234Config",
    "build_v234_control_universe",
    "build_v234_matched_control_pools",
    "build_v234_month_bootstrap",
    "build_v234_random_paths",
    "decide_v234",
    "simulate_v234_oco",
    "summarize_v234",
    "summarize_v234_variants",
    "write_v234_book_vacuum_oco_breakout",
]
