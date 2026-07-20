"""Temporal holdout confirmation for the selected two-sigma BTC OCO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v234_book_vacuum_oco_breakout import (
    FEATURE_SHA256,
    V234Config,
    build_v234_month_bootstrap,
    build_v234_random_paths,
    load_v234_inputs,
    simulate_v234_oco,
    summarize_v234,
)


FEATURE_PATH = Path(
    "reports/v23_3_book_vacuum_oco_breakout_feature_audit/oco_breakout_features.parquet"
)
V234_ROOT = Path("reports/v23_4_book_vacuum_oco_breakout")
REPORT_ROOT = Path("reports/v23_6_two_sigma_oco_temporal_confirmation")
FINDINGS_PATH = Path(
    "docs/v236_two_sigma_oco_temporal_confirmation_findings_2026_07_17.md"
)
PREREG_PATH = Path(
    "docs/v236_two_sigma_oco_temporal_confirmation_prereg_2026_07_17.md"
)
CANDIDATE = "DVB4_TWO_SIGMA_BTC_OCO_BREAKOUT"
CONTROL = "DVB4_MATCHED_NON_EVENT_TWO_SIGMA_BTC_OCO"


@dataclass(frozen=True)
class V236Config:
    feature_path: Path = FEATURE_PATH
    v234_root: Path = V234_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    prereg_path: Path = PREREG_PATH
    sigma_multiple: float = 2.0
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    minimum_holdout_triggers: int = 20
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _random_percentile(paths: pd.DataFrame) -> float:
    event_mean = float(paths["event_mean_primary_net"].iloc[0])
    return float(paths["control_mean_primary_net"].le(event_mean).mean() * 100.0)


def decide_v236(
    summary: pd.DataFrame,
    full_random: pd.DataFrame,
    holdout_random: pd.DataFrame,
    bootstrap: pd.DataFrame,
    cfg: V236Config = V236Config(),
) -> tuple[pd.DataFrame, str]:
    indexed = summary.set_index("scope")
    all_row = indexed.loc["all"]
    holdout = indexed.loc["holdout"]
    lower = float(bootstrap["mean_primary_net_return"].quantile(0.025) * 10_000)
    full_percentile = _random_percentile(full_random)
    holdout_percentile = _random_percentile(holdout_random)
    gates = {
        "minimum_20_holdout_triggers": int(holdout["triggered_trades"])
        >= cfg.minimum_holdout_triggers,
        "holdout_primary_net_positive": float(
            holdout["mean_primary_net_return_per_event_bp"]
        )
        > 0,
        "full_sample_primary_net_positive": float(
            all_row["mean_primary_net_return_per_event_bp"]
        )
        > 0,
        "month_block_bootstrap_lower_above_zero": lower > 0,
        "holdout_matched_random_percentile_at_least_90": holdout_percentile >= 90.0,
        "full_matched_random_percentile_at_least_90": full_percentile >= 90.0,
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
        int(holdout["triggered_trades"]),
        float(holdout["mean_primary_net_return_per_event_bp"]),
        float(all_row["mean_primary_net_return_per_event_bp"]),
        lower,
        holdout_percentile,
        full_percentile,
        float(all_row["ambiguous_trade_fraction"]),
        float(all_row["mean_primary_net_return_per_event_bp"])
        - float(all_row["mean_reversed_primary_net_return_per_event_bp"]),
    ]
    decision = pd.DataFrame(
        {"gate": list(gates), "passed": list(gates.values()), "observed": observed}
    )
    verdict = (
        "research_only_two_sigma_oco_supported"
        if bool(decision["passed"].all())
        else "two_sigma_oco_rejected"
    )
    return decision, verdict


def write_v236_two_sigma_oco_temporal_confirmation(
    cfg: V236Config = V236Config(),
) -> dict[str, Path]:
    if _sha256(cfg.feature_path) != FEATURE_SHA256:
        raise RuntimeError("v23.3 feature hash differs from preregistration")
    features, _, bars = load_v234_inputs(V234Config())
    universe = pd.read_parquet(cfg.v234_root / "causal_control_universe.parquet")
    pools = pd.read_parquet(cfg.v234_root / "matched_control_pools.parquet")
    for frame, columns in (
        (universe, ("entry_time",)),
        (pools, ("event_time", "control_time")),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    sim_cfg = V234Config(
        primary_cost=cfg.primary_cost,
        stress_cost=cfg.stress_cost,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )
    outcomes = simulate_v234_oco(
        features,
        bars,
        sim_cfg,
        sigma_multiple=cfg.sigma_multiple,
        candidate=CANDIDATE,
    )
    controls = simulate_v234_oco(
        universe,
        bars,
        sim_cfg,
        sigma_multiple=cfg.sigma_multiple,
        candidate=CONTROL,
    )
    full_random = build_v234_random_paths(outcomes, controls, pools, sim_cfg)
    holdout_times = set(outcomes.loc[outcomes["period"].eq("holdout"), "entry_time"])
    holdout_outcomes = outcomes[outcomes["entry_time"].isin(holdout_times)]
    holdout_pools = pools[pools["event_time"].isin(holdout_times)]
    holdout_random = build_v234_random_paths(
        holdout_outcomes, controls, holdout_pools, sim_cfg
    )
    bootstrap = build_v234_month_bootstrap(outcomes, sim_cfg)
    summary = summarize_v234(outcomes)
    summary["candidate"] = CANDIDATE
    decision, verdict = decide_v236(
        summary, full_random, holdout_random, bootstrap, cfg
    )

    root = ensure_dir(cfg.report_root)
    paths = {
        "outcomes": root / "two_sigma_event_outcomes.parquet",
        "control_outcomes": root / "two_sigma_control_outcomes.parquet",
        "full_random": root / "full_matched_random_paths.parquet",
        "holdout_random": root / "holdout_matched_random_paths.parquet",
        "bootstrap": root / "month_block_bootstrap.parquet",
        "summary": root / "result_summary.csv",
        "decision": root / "decision_gates.csv",
        "config": root / "frozen_config.json",
        "findings": cfg.findings_path,
    }
    outcomes.to_parquet(paths["outcomes"], index=False)
    controls.to_parquet(paths["control_outcomes"], index=False)
    full_random.to_parquet(paths["full_random"], index=False)
    holdout_random.to_parquet(paths["holdout_random"], index=False)
    bootstrap.to_parquet(paths["bootstrap"], index=False)
    summary.to_csv(paths["summary"], index=False)
    decision.to_csv(paths["decision"], index=False)
    paths["config"].write_text(
        json.dumps(asdict(cfg), default=str, indent=2), encoding="utf-8"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.6 Two-Sigma OCO Temporal Confirmation Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                decision.to_markdown(index=False, floatfmt=".4f"),
                "",
                "Development and validation are selection-period diagnostics; the",
                "2.00-sigma holdout is the new temporal confirmation evidence.",
                "The result remains a bar-based research simulation.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V236Config",
    "decide_v236",
    "write_v236_two_sigma_oco_temporal_confirmation",
]
