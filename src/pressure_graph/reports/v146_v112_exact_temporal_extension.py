"""Exact temporal extension of the frozen v11.2 topology-break strategy."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v109_graph_dispersion_spread import load_v109_panel
from pressure_graph.reports.v111_sparse_topology_continuation import (
    HORIZONS,
    audit_v111,
    build_v111_base_events,
    build_v111_contexts,
    build_v111_portfolios,
    select_sparse_events,
    summarize_v111,
)
from pressure_graph.reports.v112_high_vol_topology_continuation import (
    V112Config,
    apply_v112_volatility_gate,
    build_v112_volatility_state,
    random_v112_controls,
)
from pressure_graph.reports.v141_directed_taker_flow_graph import (
    load_v141_price_matrices,
)


CANONICAL_PANEL_PATH = Path(
    "reports/v10_8_oi_leader_bucket/oi_feature_panel.parquet"
)
CANONICAL_PORTFOLIO_PATH = Path(
    "reports/v11_2_high_vol_topology_continuation/"
    "high_vol_horizon_portfolios.parquet"
)
REPORT_ROOT = Path("reports/v14_6_v112_exact_temporal_extension")
FINDINGS_PATH = Path(
    "docs/v146_v112_exact_temporal_extension_findings_2026_07_15.md"
)


@dataclass(frozen=True)
class V146Config:
    canonical_panel_path: Path = CANONICAL_PANEL_PATH
    canonical_portfolio_path: Path = CANONICAL_PORTFOLIO_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def build_v146_extended_panel(
    canonical: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    canonical = canonical.copy()
    canonical["feature_time"] = pd.to_datetime(
        canonical["feature_time"], utc=True, errors="coerce"
    )
    cutoff = canonical["feature_time"].max()
    symbols = sorted(set(canonical["symbol"].astype(str)) & set(close.columns))
    local_close = close.reindex(columns=symbols)
    trailing_1h = local_close.div(local_close.shift(4)).sub(1.0)
    future_4h = local_close.shift(-16).div(local_close).sub(1.0)
    tail_index = local_close.index[local_close.index > cutoff]
    ret_long = (
        trailing_1h.reindex(tail_index)
        .rename_axis(index="feature_time", columns="symbol")
        .stack(future_stack=True)
        .rename("ret_1h")
    )
    future_long = (
        future_4h.reindex(tail_index)
        .rename_axis(index="feature_time", columns="symbol")
        .stack(future_stack=True)
        .rename("future_ret_4h")
    )
    tail = pd.concat([ret_long, future_long], axis=1).reset_index()
    tail = tail.dropna(subset=["ret_1h"])
    tail["month_start"] = pd.to_datetime(
        tail["feature_time"].dt.strftime("%Y-%m-01"),
        utc=True,
        errors="coerce",
    )
    return (
        pd.concat([canonical, tail], ignore_index=True)
        .drop_duplicates(["symbol", "feature_time"], keep="first")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def audit_v146_parity(
    rebuilt: pd.DataFrame,
    canonical: pd.DataFrame,
    tolerance: float = 1e-12,
) -> dict[str, float | int | bool]:
    keys = ["candidate", "feature_time", "horizon_hours"]
    columns = [
        "spread_gross",
        "spread_net_20bp",
        "spread_net_30bp",
        "spread_net_50bp",
    ]
    left = canonical[keys + columns].copy()
    right = rebuilt[keys + columns].copy()
    for frame in (left, right):
        frame["feature_time"] = pd.to_datetime(
            frame["feature_time"], utc=True, errors="coerce"
        )
    merged = left.merge(
        right,
        on=keys,
        how="left",
        suffixes=("_canonical", "_rebuilt"),
        validate="one_to_one",
        indicator=True,
    )
    matched = int(merged["_merge"].eq("both").sum())
    differences = []
    for column in columns:
        differences.append(
            (
                merged[f"{column}_canonical"]
                - merged[f"{column}_rebuilt"]
            ).abs()
        )
    max_difference = float(pd.concat(differences, axis=1).max().max())
    passed = matched == len(canonical) and max_difference <= tolerance
    return {
        "passed": passed,
        "canonical_rows": len(canonical),
        "matched_rows": matched,
        "max_abs_return_difference": max_difference,
    }


def summarize_v146_forward(
    forward: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        sample = forward[forward["horizon_hours"].eq(horizon)]
        rows.append(
            {
                "horizon_hours": horizon,
                "observations": len(sample),
                "active_days": sample["entry_day"].nunique(),
                "active_months": sample["entry_month"].nunique(),
                "mean_gross": sample["spread_gross"].mean(),
                "mean_net20": sample["spread_net_20bp"].mean(),
                "mean_net30": sample["spread_net_30bp"].mean(),
                "mean_net50": sample["spread_net_50bp"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_forward(
    sample: pd.DataFrame,
    cfg: V146Config,
) -> tuple[float, float]:
    daily = [
        group["spread_net_20bp"].dropna().to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    daily = [values for values in daily if len(values)]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(
            float(np.mean(np.concatenate([daily[index] for index in chosen])))
        )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def audit_v146_extension(
    forward: pd.DataFrame,
    forward_summary: pd.DataFrame,
    full_audit: pd.DataFrame,
    parity: dict[str, float | int | bool],
    cfg: V146Config,
) -> pd.DataFrame:
    primary = forward_summary[forward_summary["horizon_hours"].eq(4)].iloc[0]
    sample = forward[forward["horizon_hours"].eq(4)]
    full = full_audit[full_audit["horizon_hours"].eq(4)].iloc[0]
    ci_low, ci_high = _bootstrap_forward(sample, cfg)
    gates = {
        "canonical_parity": bool(parity["passed"]),
        "new_observations_10": int(primary["observations"]) >= 10,
        "new_months_2": int(primary["active_months"]) >= 2,
        "new_net20_positive": float(primary["mean_net20"]) > 0,
        "new_net50_positive": float(primary["mean_net50"]) > 0,
        "new_bootstrap_lower_positive": ci_low > 0,
        "full_random_family_p95": float(full["random_family_percentile"]) >= 0.95,
        "full_beats_shifted": float(full["full_net20"]) > float(full["shifted_net20"]),
    }
    confirmed = all(gates.values())
    return pd.DataFrame(
        [
            {
                "primary_horizon_hours": 4,
                "confirmed": confirmed,
                "verdict": "continue_exact_v112_forward_shadow"
                if confirmed
                else "insufficient_exact_v112_forward_confirmation",
                "new_observations": int(primary["observations"]),
                "new_active_months": int(primary["active_months"]),
                "new_mean_gross": float(primary["mean_gross"]),
                "new_mean_net20": float(primary["mean_net20"]),
                "new_mean_net50": float(primary["mean_net50"]),
                "new_bootstrap_ci_low": ci_low,
                "new_bootstrap_ci_high": ci_high,
                "full_net20": float(full["full_net20"]),
                "full_random_family_percentile": float(
                    full["random_family_percentile"]
                ),
                "parity_max_abs_difference": float(
                    parity["max_abs_return_difference"]
                ),
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        ]
    )


def _write_findings(
    path: Path,
    extension_audit: pd.DataFrame,
    forward_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    parity: dict[str, float | int | bool],
) -> None:
    lines = [
        "# v14.6 Exact v11.2 Temporal-Extension Findings",
        "",
        f"Verdict: `{extension_audit['verdict'].iloc[0]}`.",
        "",
        f"Canonical parity: `{parity['passed']}`; matched "
        f"`{parity['matched_rows']}/{parity['canonical_rows']}` rows; maximum absolute "
        f"return difference `{parity['max_abs_return_difference']:.3e}`.",
        "",
        "## New entries after the canonical last event",
        "",
        forward_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Full extended sample",
        "",
        full_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "No PaperLive, leverage, or live-order permission changed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v146_v112_exact_temporal_extension(
    cfg: V146Config = V146Config(),
) -> dict[str, Path]:
    canonical_panel = load_v109_panel(cfg.canonical_panel_path)
    prices = load_v141_price_matrices()
    extended_panel = build_v146_extended_panel(canonical_panel, prices["close"])
    strategy_cfg = V112Config()
    contexts, membership = build_v111_contexts(extended_panel, strategy_cfg)
    state = build_v112_volatility_state(
        extended_panel, list(contexts), strategy_cfg
    )
    base = build_v111_base_events(contexts, strategy_cfg)
    sparse = select_sparse_events(base, strategy_cfg)
    real = apply_v112_volatility_gate(
        build_v111_portfolios(sparse, strategy_cfg), state
    )
    shifted_base = build_v111_base_events(
        contexts, strategy_cfg, signal_shift_bars=24
    )
    shifted = apply_v112_volatility_gate(
        build_v111_portfolios(
            select_sparse_events(shifted_base, strategy_cfg), strategy_cfg
        ),
        state,
    )
    controls = random_v112_controls(contexts, state, strategy_cfg)
    full_summary = summarize_v111(real)
    full_audit = audit_v111(
        real, shifted, full_summary, controls, strategy_cfg
    )
    canonical_portfolio = pd.read_parquet(cfg.canonical_portfolio_path)
    parity = audit_v146_parity(real, canonical_portfolio)
    if not parity["passed"]:
        raise RuntimeError(f"v14.6 canonical parity failed: {parity}")
    canonical_last = pd.to_datetime(
        canonical_portfolio["feature_time"], utc=True, errors="coerce"
    ).max()
    forward = real[real["feature_time"].gt(canonical_last)].copy()
    forward_summary = summarize_v146_forward(forward)
    extension_audit = audit_v146_extension(
        forward, forward_summary, full_audit, parity, cfg
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "extended_panel": root / "extended_price_feature_panel.parquet",
        "membership": root / "monthly_balanced_membership.csv",
        "volatility_state": root / "btc_volatility_state.parquet",
        "sparse_events": root / "sparse_topology_events.parquet",
        "portfolios": root / "extended_horizon_portfolios.parquet",
        "forward_portfolios": root / "new_forward_portfolios.parquet",
        "shifted": root / "shifted_high_vol_portfolios.parquet",
        "controls": root / "random_partition_controls.csv",
        "full_summary": root / "full_horizon_summary.csv",
        "forward_summary": root / "forward_horizon_summary.csv",
        "full_audit": root / "full_horizon_audit.csv",
        "extension_audit": root / "extension_audit.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    extended_panel.to_parquet(outputs["extended_panel"], index=False)
    membership.to_csv(outputs["membership"], index=False)
    state.to_parquet(outputs["volatility_state"], index=False)
    sparse.to_parquet(outputs["sparse_events"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    forward.to_parquet(outputs["forward_portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    full_summary.to_csv(outputs["full_summary"], index=False)
    forward_summary.to_csv(outputs["forward_summary"], index=False)
    full_audit.to_csv(outputs["full_audit"], index=False)
    extension_audit.to_csv(outputs["extension_audit"], index=False)
    outputs["metadata"].write_text(
        json.dumps(
            {
                "canonical_last_entry": canonical_last.isoformat(),
                "canonical_panel_last_time": canonical_panel[
                    "feature_time"
                ].max().isoformat(),
                "extended_panel_last_time": extended_panel[
                    "feature_time"
                ].max().isoformat(),
                "parity": parity,
                "new_portfolio_rows": len(forward),
                "verdict": extension_audit["verdict"].iloc[0],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_findings(
        cfg.findings_path,
        extension_audit,
        forward_summary,
        full_summary,
        parity,
    )
    return outputs
