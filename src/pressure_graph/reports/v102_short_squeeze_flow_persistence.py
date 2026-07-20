"""Third-look short-squeeze exact-flow persistence audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v101_exact_flow_persistence import (
    CANDIDATE,
    V101Config,
    _load_btc,
    _load_minute_data,
    attach_v101_btc,
    random_v101_controls,
)


REPORT_ROOT = Path("reports/v10_2_short_squeeze_flow_persistence")
SOURCE_ROOT = Path("reports/v10_0_exact_taker_flow_alpha")
BTC_PATH = Path("data/raw/bybit/klines/BTCUSDT.parquet")
CACHE_ROOT = Path("data/processed/v100_exact_taker_flow_1m")


@dataclass(frozen=True)
class V102Config:
    source_root: Path = SOURCE_ROOT
    btc_path: Path = BTC_PATH
    cache_root: Path = CACHE_ROOT
    report_root: Path = REPORT_ROOT
    random_iterations: int = 500
    bootstrap_iterations: int = 2000
    seed: int = 20260817


def summarize_v102(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = panel if scope == "all" else panel[panel["period"].eq(scope)]
        selected = scoped[scoped[CANDIDATE].fillna(False).astype(bool)]
        other = scoped[~scoped[CANDIDATE].fillna(False).astype(bool)]
        hedged = selected[
            selected["symbol"].ne("BTCUSDT")
            & selected["hedged_net_return_240m_40bp"].notna()
        ]
        rows.append(
            {
                "scope": scope,
                "trades": int(len(selected)),
                "symbols": int(selected["symbol"].nunique()),
                "active_days": int(selected["entry_day"].nunique()),
                "mean_raw_net10": float(selected["net_return_240m_10bp"].mean()),
                "mean_raw_net20": float(selected["net_return_240m_20bp"].mean()),
                "mean_raw_net30": float(selected["net_return_240m_30bp"].mean()),
                "other_trades": int(len(other)),
                "other_mean_raw_net20": float(other["net_return_240m_20bp"].mean()),
                "selected_minus_other_net20": float(
                    selected["net_return_240m_20bp"].mean()
                    - other["net_return_240m_20bp"].mean()
                ),
                "hedged_trades": int(len(hedged)),
                "mean_hedged_net40": float(
                    hedged["hedged_net_return_240m_40bp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_v102(panel: pd.DataFrame, cfg: V102Config) -> dict[str, float]:
    day_groups = [group for _, group in panel.groupby("entry_day", sort=True)]
    rng = np.random.default_rng(cfg.seed)
    raw_means = []
    lifts = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(day_groups), len(day_groups))
        sample = pd.concat([day_groups[index] for index in chosen], ignore_index=True)
        selected = sample[sample[CANDIDATE].fillna(False).astype(bool)]
        other = sample[~sample[CANDIDATE].fillna(False).astype(bool)]
        raw_means.append(float(selected["net_return_240m_20bp"].mean()))
        lifts.append(
            float(
                selected["net_return_240m_20bp"].mean()
                - other["net_return_240m_20bp"].mean()
            )
        )
    selected = panel[panel[CANDIDATE].fillna(False).astype(bool)]
    day_sum = selected.groupby("entry_day")["net_return_240m_20bp"].sum()
    positive = day_sum.clip(lower=0)
    return {
        "raw_bootstrap_low": float(np.quantile(raw_means, 0.025)),
        "raw_bootstrap_high": float(np.quantile(raw_means, 0.975)),
        "lift_bootstrap_low": float(np.quantile(lifts, 0.025)),
        "lift_bootstrap_high": float(np.quantile(lifts, 0.975)),
        "max_positive_day_share": float(positive.max() / positive.sum()),
    }


def audit_v102(
    panel: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V102Config,
) -> pd.DataFrame:
    rows = {row.scope: row for row in summary.itertuples(index=False)}
    full = rows["all"]
    shifted_full = summarize_v102(shifted).query("scope == 'all'").iloc[0]
    stability = bootstrap_v102(panel, cfg)
    raw_percentile = float(controls["mean_raw_net20"].lt(full.mean_raw_net20).mean())
    hedged_percentile = float(
        controls["mean_hedged_net40"].lt(full.mean_hedged_net40).mean()
    )
    gates = {
        "full_n_50": (full.trades >= 50, full.trades),
        "validation_n_15": (rows["validation"].trades >= 15, rows["validation"].trades),
        "holdout_n_12": (rows["holdout"].trades >= 12, rows["holdout"].trades),
        "all_splits_raw_net30_positive": (
            min(rows[scope].mean_raw_net30 for scope in rows) > 0,
            min(rows[scope].mean_raw_net30 for scope in rows),
        ),
        "raw_bootstrap_lower_positive": (
            stability["raw_bootstrap_low"] > 0,
            stability["raw_bootstrap_low"],
        ),
        "all_splits_selection_lift_positive": (
            min(rows[scope].selected_minus_other_net20 for scope in rows) > 0,
            min(rows[scope].selected_minus_other_net20 for scope in rows),
        ),
        "lift_bootstrap_lower_positive": (
            stability["lift_bootstrap_low"] > 0,
            stability["lift_bootstrap_low"],
        ),
        "path_random_p95": (raw_percentile >= 0.95, raw_percentile),
        "beats_shifted_placebo": (
            full.mean_raw_net20 > shifted_full.mean_raw_net20,
            full.mean_raw_net20 - shifted_full.mean_raw_net20,
        ),
        "positive_day_share_below_35pct": (
            stability["max_positive_day_share"] <= 0.35,
            stability["max_positive_day_share"],
        ),
        "validation_holdout_hedged_nonnegative": (
            min(rows["validation"].mean_hedged_net40, rows["holdout"].mean_hedged_net40)
            >= 0,
            min(rows["validation"].mean_hedged_net40, rows["holdout"].mean_hedged_net40),
        ),
        "hedged_random_p90": (hedged_percentile >= 0.90, hedged_percentile),
    }
    eligible = all(bool(passed) for passed, _ in gates.values())
    verdict = (
        "post_discovery_forward_watch_clue"
        if eligible
        else "reject_single_venue_flow_persistence_branch"
    )
    return pd.DataFrame(
        [
            {
                "check": check,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
                "verdict": verdict,
                **stability,
            }
            for check, (passed, value) in gates.items()
        ]
    )


def write_v102_short_squeeze_flow_persistence(
    cfg: V102Config = V102Config(),
) -> dict[str, Path]:
    panel = pd.read_parquet(cfg.source_root / "event_panel.parquet")
    shifted = pd.read_parquet(cfg.source_root / "shifted_60m_panel.parquet")
    panel = panel[panel["path_name"].eq("short_squeeze")].copy()
    shifted = shifted[shifted["path_name"].eq("short_squeeze")].copy()
    btc = _load_btc(cfg.btc_path)
    panel = attach_v101_btc(panel, btc, tolerance_minutes=15)
    shifted = attach_v101_btc(shifted, btc, tolerance_minutes=15)
    summary = summarize_v102(panel)
    minute_data = _load_minute_data(cfg.cache_root)
    control_cfg = V101Config(
        btc_path=cfg.btc_path,
        random_iterations=cfg.random_iterations,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
        btc_tolerance_minutes=15,
    )
    controls = random_v101_controls(panel, minute_data, btc, control_cfg)
    audit = audit_v102(panel, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "summary": root / "path_summary.csv",
        "random_controls": root / "path_random_controls.csv",
        "audit": root / "path_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["random_controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v10.2 Short-Squeeze Exact-Flow Persistence",
        "",
        f"Status: `{audit['verdict'].iloc[0]}`. Third-look offline audit only.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.scope}: n={row.trades}, raw_net20={row.mean_raw_net20:.4%}, "
            f"selection_lift={row.selected_minus_other_net20:.4%}, "
            f"hedged_net40={row.mean_hedged_net40:.4%}."
        )
    lines.extend(["", "No paper or live permission changed."])
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
