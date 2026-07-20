"""Preregistered single-interpolation q85 breakout result."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
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
from pressure_graph.reports.v2312_positive_q80_vacuum_breakout import (
    V2312Config,
    _all_q80_events,
    decide_v2312,
)


V2314_ROOT = Path("reports/v23_14_positive_q85_vacuum_breakout_feature_audit")
FEATURE_PATH = V2314_ROOT / "positive_q85_breakout_features.parquet"
BUCKET_STATE_PATH = V2314_ROOT / "hourly_q85_bucket_states.parquet"
REPORT_ROOT = Path("reports/v23_15_positive_q85_vacuum_breakout")
FINDINGS_PATH = Path(
    "docs/v2315_positive_q85_vacuum_breakout_findings_2026_07_17.md"
)
PREREG_PATH = Path("docs/v2315_positive_q85_vacuum_breakout_prereg_2026_07_17.md")
CANDIDATE = "DVB7_POSITIVE_Q85_VACUUM_0625SIGMA_BREAKOUT"
CONTROL = "DVB7_MATCHED_NON_EVENT_0625SIGMA_BREAKOUT"
FEATURE_SHA256 = "F14AB2F30433594501EA3FC9284F8E087768BF3B48C58056CB4EFEEEB65BEE28"


@dataclass(frozen=True)
class V2315Config:
    feature_path: Path = FEATURE_PATH
    bucket_state_path: Path = BUCKET_STATE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    pressure_quantile: float = 0.85
    primary_sigma_multiple: float = 0.625
    adjacent_sigma_multiple: float = 0.75
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    minimum_total_triggers: int = 70
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


def _decision_config(cfg: V2315Config) -> V2312Config:
    return V2312Config(
        feature_path=cfg.feature_path,
        bucket_state_path=cfg.bucket_state_path,
        pressure_quantile=cfg.pressure_quantile,
        primary_sigma_multiple=cfg.primary_sigma_multiple,
        adjacent_sigma_multiple=cfg.adjacent_sigma_multiple,
        primary_cost=cfg.primary_cost,
        stress_cost=cfg.stress_cost,
        minimum_total_triggers=cfg.minimum_total_triggers,
        minimum_period_triggers=cfg.minimum_period_triggers,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )


def write_v2315_positive_q85_vacuum_breakout(
    cfg: V2315Config = V2315Config(),
) -> dict[str, Path]:
    if _sha256(cfg.feature_path) != FEATURE_SHA256:
        raise RuntimeError("v23.14 feature hash differs from preregistration")
    features = pd.read_parquet(cfg.feature_path)
    for column in ("feature_time", "entry_time"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="coerce")
    _, _, bars = load_v234_inputs(V234Config())
    decision_cfg = _decision_config(cfg)
    all_q85_events = _all_q80_events(decision_cfg)
    sim_cfg = V234Config(
        primary_cost=cfg.primary_cost,
        stress_cost=cfg.stress_cost,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )
    universe = build_v234_control_universe(
        features, all_q85_events, bars, sim_cfg
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
        primary,
        controls,
        pools,
        replace(
            V238Config(),
            random_iterations=cfg.random_iterations,
            seed=cfg.seed,
        ),
    )
    bootstrap = build_v234_month_bootstrap(primary, sim_cfg)
    leave = _leave_one_month_out(primary)
    primary_summary = summarize_v238_periods(primary, label="q85_0.625sigma")
    adjacent_summary = summarize_v238_periods(adjacent, label="q85_0.75sigma")
    decision, _ = decide_v2312(
        primary_summary,
        adjacent_summary,
        random_summary,
        bootstrap,
        leave,
        decision_cfg,
    )
    verdict = (
        "research_only_positive_q85_interpolation_supported"
        if bool(decision["passed"].all())
        else "positive_q85_interpolation_rejected"
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "primary": root / "primary_event_outcomes.parquet",
        "adjacent": root / "adjacent_event_outcomes.parquet",
        "universe": root / "causal_control_universe.parquet",
        "pools": root / "matched_control_pools.parquet",
        "controls": root / "control_outcomes.parquet",
        "random_paths": root / "matched_random_paths.parquet",
        "random_summary": root / "matched_random_summary.csv",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "leave": root / "leave_one_month_out.csv",
        "primary_summary": root / "primary_summary.csv",
        "adjacent_summary": root / "adjacent_summary.csv",
        "decision": root / "decision_gates.csv",
        "config": root / "frozen_config.json",
        "findings": cfg.findings_path,
    }
    primary.to_parquet(paths["primary"], index=False)
    adjacent.to_parquet(paths["adjacent"], index=False)
    universe.to_parquet(paths["universe"], index=False)
    pools.to_parquet(paths["pools"], index=False)
    controls.to_parquet(paths["controls"], index=False)
    random_paths.to_parquet(paths["random_paths"], index=False)
    random_summary.to_csv(paths["random_summary"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    leave.to_csv(paths["leave"], index=False)
    primary_summary.to_csv(paths["primary_summary"], index=False)
    adjacent_summary.to_csv(paths["adjacent_summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    paths["config"].write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.15 Positive-q85 Vacuum Breakout Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                primary_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                random_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                decision.to_markdown(index=False, floatfmt=".4f"),
                "",
                "q85 was the sole frozen interpolation between q80 and q90.",
                "No further pressure-quantile search is authorized by this round.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = ["V2315Config", "write_v2315_positive_q85_vacuum_breakout"]
