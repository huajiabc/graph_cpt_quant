"""Preregistered result for the denser positive-q80 BTC breakout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v224_alt_book_vacuum_pressure_feature_audit import (
    V224Config,
    select_v224_events,
)
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    V234Config,
    build_v234_control_universe,
    build_v234_matched_control_pools,
    build_v234_month_bootstrap,
    load_v234_inputs,
    simulate_v234_oco,
)
from pressure_graph.reports.v238_positive_pressure_narrow_breakout_robustness import (
    V238Config,
    _leave_one_month_out,
    _matched_random_by_scope,
    summarize_v238_periods,
)


V2311_ROOT = Path("reports/v23_11_positive_q80_vacuum_breakout_feature_audit")
FEATURE_PATH = V2311_ROOT / "positive_q80_breakout_features.parquet"
BUCKET_STATE_PATH = V2311_ROOT / "hourly_q80_bucket_states.parquet"
REPORT_ROOT = Path("reports/v23_12_positive_q80_vacuum_breakout")
FINDINGS_PATH = Path(
    "docs/v2312_positive_q80_vacuum_breakout_findings_2026_07_17.md"
)
PREREG_PATH = Path("docs/v2312_positive_q80_vacuum_breakout_prereg_2026_07_17.md")
CANDIDATE = "DVB6_POSITIVE_Q80_VACUUM_0625SIGMA_BREAKOUT"
CONTROL = "DVB6_MATCHED_NON_EVENT_0625SIGMA_BREAKOUT"
FEATURE_SHA256 = "163972748E6CC095BD086414CADC8C8A9F7535082B1733A6C7E2933EEF848B93"


@dataclass(frozen=True)
class V2312Config:
    feature_path: Path = FEATURE_PATH
    bucket_state_path: Path = BUCKET_STATE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    pressure_quantile: float = 0.80
    primary_sigma_multiple: float = 0.625
    adjacent_sigma_multiple: float = 0.75
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    minimum_total_triggers: int = 80
    minimum_period_triggers: int = 20
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _all_q80_events(cfg: V2312Config) -> pd.DataFrame:
    states = pd.read_parquet(cfg.bucket_state_path)
    states["decision_time"] = pd.to_datetime(
        states["decision_time"], utc=True, errors="coerce"
    )
    v224_cfg = replace(
        V224Config(),
        pressure_quantile=cfg.pressure_quantile,
        minimum_events=0,
        minimum_period_events=0,
        minimum_direction_period_events=0,
        minimum_active_months=0,
    )
    return select_v224_events(states, v224_cfg)


def decide_v2312(
    primary_summary: pd.DataFrame,
    adjacent_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    cfg: V2312Config = V2312Config(),
) -> tuple[pd.DataFrame, str]:
    primary = primary_summary.set_index("scope")
    adjacent = adjacent_summary.set_index("scope")
    all_row = primary.loc["all"]
    periods = primary.loc[["development", "validation", "holdout"]]
    adjacent_scopes = adjacent.loc[
        ["all", "development", "validation", "holdout"]
    ]
    lower = float(bootstrap["mean_primary_net_return"].quantile(0.025) * 10_000)
    gates = {
        "minimum_total_and_each_period_triggers": int(all_row["triggered_trades"])
        >= cfg.minimum_total_triggers
        and bool(
            periods["triggered_trades"].ge(cfg.minimum_period_triggers).all()
        ),
        "primary_positive_all_temporal_scopes": float(
            all_row["mean_primary_net_return_bp"]
        )
        > 0
        and bool(periods["mean_primary_net_return_bp"].gt(0).all()),
        "stress_positive_full_sample": float(all_row["mean_stress_net_return_bp"])
        > 0,
        "absolute_month_bootstrap_lower_above_zero": lower > 0,
        "matched_random_percentile_at_least_90_all_scopes": bool(
            random_summary["matched_random_percentile"].ge(90).all()
        ),
        "every_event_has_at_least_five_matched_controls": bool(
            random_summary["unmatched_events"].eq(0).all()
        ),
        "same_bar_ambiguity_at_most_10pct": int(all_row["ambiguous_trades"])
        / max(int(all_row["triggered_trades"]), 1)
        <= 0.10,
        "leave_one_month_out_minimum_above_zero": float(
            leave_one_out["mean_primary_net_return_bp"].min()
        )
        > 0,
        "adjacent_width_positive_all_temporal_scopes": bool(
            adjacent_scopes["mean_primary_net_return_bp"].gt(0).all()
        ),
    }
    observed = [
        min(
            int(all_row["triggered_trades"]) - cfg.minimum_total_triggers,
            int(periods["triggered_trades"].min()) - cfg.minimum_period_triggers,
        ),
        min(
            float(all_row["mean_primary_net_return_bp"]),
            float(periods["mean_primary_net_return_bp"].min()),
        ),
        float(all_row["mean_stress_net_return_bp"]),
        lower,
        float(random_summary["matched_random_percentile"].min()),
        int(random_summary["unmatched_events"].max()),
        int(all_row["ambiguous_trades"])
        / max(int(all_row["triggered_trades"]), 1),
        float(leave_one_out["mean_primary_net_return_bp"].min()),
        float(adjacent_scopes["mean_primary_net_return_bp"].min()),
    ]
    decision = pd.DataFrame(
        {"gate": list(gates), "passed": list(gates.values()), "observed": observed}
    )
    verdict = (
        "research_only_positive_q80_density_extension_supported"
        if bool(decision["passed"].all())
        else "positive_q80_density_extension_rejected"
    )
    return decision, verdict


def write_v2312_positive_q80_vacuum_breakout(
    cfg: V2312Config = V2312Config(),
) -> dict[str, Path]:
    if _sha256(cfg.feature_path) != FEATURE_SHA256:
        raise RuntimeError("v23.11 feature hash differs from preregistration")
    features = pd.read_parquet(cfg.feature_path)
    for column in ("feature_time", "entry_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="coerce")
    _, _, bars = load_v234_inputs(V234Config())
    all_q80_events = _all_q80_events(cfg)
    sim_cfg = V234Config(
        primary_cost=cfg.primary_cost,
        stress_cost=cfg.stress_cost,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )
    universe = build_v234_control_universe(
        features, all_q80_events, bars, sim_cfg
    )
    pools = build_v234_matched_control_pools(features, universe, sim_cfg)
    primary = simulate_v234_oco(
        features,
        bars,
        sim_cfg,
        sigma_multiple=cfg.primary_sigma_multiple,
        candidate=CANDIDATE,
    )
    adjacent = simulate_v234_oco(
        features,
        bars,
        sim_cfg,
        sigma_multiple=cfg.adjacent_sigma_multiple,
        candidate=CANDIDATE,
    )
    controls = simulate_v234_oco(
        universe,
        bars,
        sim_cfg,
        sigma_multiple=cfg.primary_sigma_multiple,
        candidate=CONTROL,
    )
    random_paths, random_summary = _matched_random_by_scope(
        primary, controls, pools, replace(
            V238Config(),
            random_iterations=cfg.random_iterations,
            seed=cfg.seed,
        )
    )
    bootstrap = build_v234_month_bootstrap(primary, sim_cfg)
    leave_one_out = _leave_one_month_out(primary)
    primary_summary = summarize_v238_periods(primary, label="q80_0.625sigma")
    adjacent_summary = summarize_v238_periods(adjacent, label="q80_0.75sigma")
    decision, verdict = decide_v2312(
        primary_summary,
        adjacent_summary,
        random_summary,
        bootstrap,
        leave_one_out,
        cfg,
    )

    root = ensure_dir(cfg.report_root)
    paths = {
        "primary": root / "primary_event_outcomes.parquet",
        "adjacent": root / "adjacent_event_outcomes.parquet",
        "control_universe": root / "causal_control_universe.parquet",
        "control_pools": root / "matched_control_pools.parquet",
        "controls": root / "control_outcomes.parquet",
        "random_paths": root / "matched_random_paths.parquet",
        "random_summary": root / "matched_random_summary.csv",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "leave_one_out": root / "leave_one_month_out.csv",
        "primary_summary": root / "primary_summary.csv",
        "adjacent_summary": root / "adjacent_summary.csv",
        "decision": root / "decision_gates.csv",
        "config": root / "frozen_config.json",
        "findings": cfg.findings_path,
    }
    primary.to_parquet(paths["primary"], index=False)
    adjacent.to_parquet(paths["adjacent"], index=False)
    universe.to_parquet(paths["control_universe"], index=False)
    pools.to_parquet(paths["control_pools"], index=False)
    controls.to_parquet(paths["controls"], index=False)
    random_paths.to_parquet(paths["random_paths"], index=False)
    random_summary.to_csv(paths["random_summary"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    leave_one_out.to_csv(paths["leave_one_out"], index=False)
    primary_summary.to_csv(paths["primary_summary"], index=False)
    adjacent_summary.to_csv(paths["adjacent_summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    paths["config"].write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.12 Positive-q80 Vacuum Breakout Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                primary_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                random_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                decision.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The q80 outcomes were revealed only after the v23.11 feature hash",
                "and v23.12 gates were frozen. This remains a research extension",
                "of a post-selected ancestor, not live authorization.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2312Config",
    "decide_v2312",
    "write_v2312_positive_q80_vacuum_breakout",
]
