"""High-BTC-volatility gate for sparse topology continuation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v109_graph_dispersion_spread import load_v109_panel
from pressure_graph.reports.v110_balanced_topology_break import random_v110_partitions
from pressure_graph.reports.v111_sparse_topology_continuation import (
    HORIZONS,
    V111Config,
    _v110_config,
    audit_v111,
    build_v111_base_events,
    build_v111_contexts,
    build_v111_portfolios,
    select_sparse_events,
    summarize_v111,
)


REPORT_ROOT = Path("reports/v11_2_high_vol_topology_continuation")


@dataclass(frozen=True)
class V112Config(V111Config):
    report_root: Path = REPORT_ROOT
    volatility_hours: int = 24
    volatility_quantile: float = 0.75


def build_v112_volatility_state(
    panel: pd.DataFrame,
    months: list[pd.Timestamp],
    cfg: V112Config,
) -> pd.DataFrame:
    btc = (
        panel[
            panel["symbol"].eq(BTC)
            & panel["feature_time"].dt.minute.eq(0)
        ]
        .drop_duplicates("feature_time", keep="last")
        .sort_values("feature_time")
        .set_index("feature_time")["ret_1h"]
        .astype(float)
    )
    volatility = btc.rolling(
        cfg.volatility_hours, min_periods=cfg.volatility_hours
    ).std(ddof=1)
    frames = []
    for month in sorted(months):
        history = volatility[
            (volatility.index >= month - pd.Timedelta(days=cfg.lookback_days))
            & (volatility.index < month)
        ]
        threshold = float(history.quantile(cfg.volatility_quantile))
        target = volatility[
            (volatility.index >= month)
            & (volatility.index < month + pd.offsets.MonthBegin(1))
        ]
        frames.append(
            pd.DataFrame(
                {
                    "feature_time": target.index,
                    "month_start": month,
                    "btc_volatility_24h": target.to_numpy(),
                    "btc_volatility_threshold": threshold,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_v112_volatility_gate(
    portfolios: pd.DataFrame, state: pd.DataFrame
) -> pd.DataFrame:
    if portfolios.empty:
        return portfolios.copy()
    output = portfolios.merge(state, on="feature_time", how="left")
    return output[
        output["btc_volatility_24h"].ge(output["btc_volatility_threshold"])
    ].reset_index(drop=True)


def random_v112_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    state: pd.DataFrame,
    cfg: V112Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        overrides = random_v110_partitions(contexts, iteration, _v110_config(cfg))
        events = build_v111_base_events(contexts, cfg, overrides)
        portfolios = build_v111_portfolios(select_sparse_events(events, cfg), cfg)
        portfolios = apply_v112_volatility_gate(portfolios, state)
        horizon_means = []
        for horizon in HORIZONS:
            sample = portfolios[portfolios["horizon_hours"].eq(horizon)]
            value = float(sample["spread_net_20bp"].mean())
            horizon_means.append(value)
            rows.append(
                {
                    "iteration": iteration,
                    "horizon_hours": horizon,
                    "portfolio_observations": int(len(sample)),
                    "mean_spread_net_20bp": value,
                }
            )
        finite = [value for value in horizon_means if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "horizon_hours": "FAMILY_MAX",
                "portfolio_observations": int(len(portfolios) / len(HORIZONS)),
                "mean_spread_net_20bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_v112_high_vol_topology_continuation(
    cfg: V112Config = V112Config(),
) -> dict[str, Path]:
    panel = load_v109_panel(cfg.panel_path)
    contexts, membership = build_v111_contexts(panel, cfg)
    state = build_v112_volatility_state(panel, list(contexts), cfg)
    base = build_v111_base_events(contexts, cfg)
    sparse = select_sparse_events(base, cfg)
    real = apply_v112_volatility_gate(build_v111_portfolios(sparse, cfg), state)
    shifted_base = build_v111_base_events(contexts, cfg, signal_shift_bars=24)
    shifted_sparse = select_sparse_events(shifted_base, cfg)
    shifted = apply_v112_volatility_gate(
        build_v111_portfolios(shifted_sparse, cfg), state
    )
    summary = summarize_v111(real)
    controls = random_v112_controls(contexts, state, cfg)
    audit = audit_v111(real, shifted, summary, controls, cfg)
    audit["verdict"] = np.where(
        audit["eligible"],
        "retrospective_forward_watch_only",
        "reject_high_vol_horizon",
    )
    family_verdict = (
        "retrospective_forward_watch_only"
        if bool(audit["eligible"].any())
        else "reject_high_vol_topology_family"
    )
    audit["family_verdict"] = family_verdict
    root = ensure_dir(cfg.report_root)
    outputs = {
        "membership": root / "monthly_balanced_membership.csv",
        "volatility_state": root / "btc_volatility_state.parquet",
        "sparse_events": root / "sparse_topology_events.parquet",
        "portfolios": root / "high_vol_horizon_portfolios.parquet",
        "shifted": root / "shifted_high_vol_portfolios.parquet",
        "summary": root / "horizon_summary.csv",
        "controls": root / "random_partition_controls.csv",
        "audit": root / "horizon_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    membership.to_csv(outputs["membership"], index=False)
    state.to_parquet(outputs["volatility_state"], index=False)
    sparse.to_parquet(outputs["sparse_events"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v11.2 High-Volatility Topology Continuation",
        "",
        f"Status: `{family_verdict}`.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.horizon_hours}h: gross={row.full_gross:.4%}, "
            f"net20={row.full_net20:.4%}, validation={row.validation_net20:.4%}, "
            f"holdout={row.holdout_net20:.4%}, "
            f"random percentile={row.random_family_percentile:.1%}."
        )
    lines.extend(
        [
            "",
            "This is result-informed retrospective research. No PaperLive permission changed.",
        ]
    )
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
