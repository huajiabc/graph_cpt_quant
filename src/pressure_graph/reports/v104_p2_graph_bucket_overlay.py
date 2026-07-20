"""P2 portfolio overlay using as-of continuous graph-bucket returns."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v91_p2_portfolio_risk_shadows import (
    P2_EW,
    P2RiskShadowConfig,
    _candidate_pool,
    _select_arm,
)
from pressure_graph.reports.v103_graph_bucket_return_diffusion import shifted_v103_panel


REPORT_ROOT = Path("reports/v10_4_p2_graph_bucket_overlay")
P2_PATH = Path("reports/v0_7d2_cic_mir1_replay/paper_portfolio_trades.parquet")
BUCKET_PATH = Path("reports/v10_3_graph_bucket_return_diffusion/bucket_feature_panel.parquet")
FEATURE_COLUMNS = [
    "bucket_ret_1h",
    "bucket_ret_1h_rank",
    "bucket_positive_breadth_1h",
    "bucket_excess_ret_1h",
    "target_lag_gap_1h",
]


@dataclass(frozen=True)
class V104Config:
    p2_path: Path = P2_PATH
    bucket_path: Path = BUCKET_PATH
    report_root: Path = REPORT_ROOT
    random_iterations: int = 500
    bootstrap_iterations: int = 2000
    max_positions: int = 8
    seed: int = 20260818


def load_v104_p2_pool(path: Path = P2_PATH) -> pd.DataFrame:
    pool = pd.read_parquet(path)
    pool = pool[
        pool["candidate"].astype(str).isin(["CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"])
    ].copy()
    pool = _candidate_pool(pool)
    pool["entry_time"] = pd.to_datetime(pool["entry_time"], utc=True, errors="coerce")
    pool["exit_time"] = pd.to_datetime(pool["exit_time"], utc=True, errors="coerce")
    pool["gross_return"] = pd.to_numeric(pool["gross_return"], errors="coerce")
    pool["net_rt20"] = pool["gross_return"] - 0.002
    pool["net_rt40"] = pool["gross_return"] - 0.004
    pool["net_rt60"] = pool["gross_return"] - 0.006
    pool["net_return_10bp"] = pool["net_rt20"]
    pool["net_return_20bp"] = pool["net_rt40"]
    pool["net_return_30bp"] = pool["net_rt60"]
    pool["entry_day"] = pool["entry_time"].dt.strftime("%Y-%m-%d")
    pool["entry_month"] = pool["entry_time"].dt.strftime("%Y-%m")
    pool["period"] = np.select(
        [
            pool["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            pool["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    return pool.sort_values(["entry_time", "symbol", "candidate_priority"]).reset_index(drop=True)


def load_v104_bucket_panel(path: Path = BUCKET_PATH) -> pd.DataFrame:
    columns = ["symbol", "feature_time", *FEATURE_COLUMNS]
    panel = pd.read_parquet(path, columns=columns)
    panel["feature_time"] = pd.to_datetime(panel["feature_time"], utc=True, errors="coerce")
    return panel.drop_duplicates(["symbol", "feature_time"]).sort_values(
        ["symbol", "feature_time"]
    )


def attach_v104_bucket_context(
    pool: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    for symbol, trades in pool.groupby("symbol", sort=False):
        context = panel[panel["symbol"].astype(str).eq(str(symbol))].drop(
            columns="symbol"
        )
        local = trades.drop(columns=FEATURE_COLUMNS, errors="ignore").sort_values("entry_time")
        if context.empty:
            for column in FEATURE_COLUMNS:
                local[column] = np.nan
            local["bucket_feature_time"] = pd.NaT
            pieces.append(local)
            continue
        context = context.rename(columns={"feature_time": "bucket_feature_time"})
        pieces.append(
            pd.merge_asof(
                local,
                context.sort_values("bucket_feature_time"),
                left_on="entry_time",
                right_on="bucket_feature_time",
                direction="backward",
                tolerance=pd.Timedelta(minutes=15),
            )
        )
    out = pd.concat(pieces, ignore_index=True) if pieces else pool.copy()
    out["graph_covered"] = out["bucket_ret_1h"].notna()
    return out.sort_values(["entry_time", "symbol", "candidate_priority"]).reset_index(drop=True)


def add_v104_overlay_flags(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    strong = (
        out["bucket_ret_1h"].ge(0.005)
        & out["bucket_ret_1h_rank"].ge(0.80)
        & out["bucket_positive_breadth_1h"].ge(0.60)
        & out["bucket_excess_ret_1h"].ge(0.002)
    )
    out["strong_bucket_laggard"] = out["graph_covered"] & strong & out[
        "target_lag_gap_1h"
    ].ge(0.003)
    out["overlay_keep"] = ~out["strong_bucket_laggard"]
    return out


def _select_v104(pool: pd.DataFrame, keep_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = pool[pool[keep_column].fillna(False).astype(bool)].copy()
    selected, capacity_skipped = _select_arm(
        filtered,
        P2_EW,
        P2RiskShadowConfig(max_positions=8),
    )
    return selected, capacity_skipped


def _scope_summary(
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped_pool = pool if scope == "all" else pool[pool["period"].eq(scope)]
        scoped = selected if scope == "all" else selected[selected["period"].eq(scope)]
        blocked = scoped_pool[~scoped_pool["overlay_keep"]]
        rows.append(
            {
                "policy": policy,
                "scope": scope,
                "candidate_pool": int(len(scoped_pool)),
                "graph_covered": int(scoped_pool["graph_covered"].sum()),
                "overlay_blocked": int(len(blocked)) if policy != "P2_BASELINE" else 0,
                "selected_trades": int(len(scoped)),
                "mean_net_rt20": float(scoped["net_rt20"].mean()),
                "mean_net_rt40": float(scoped["net_rt40"].mean()),
                "mean_net_rt60": float(scoped["net_rt60"].mean()),
                "portfolio_net_rt40_over_8": float(scoped["net_rt40"].sum() / 8.0),
                "blocked_mean_net_rt40": float(blocked["net_rt40"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_v104(
    pool: pd.DataFrame,
    baseline_selected: pd.DataFrame,
    overlay_selected: pd.DataFrame,
) -> pd.DataFrame:
    baseline_pool = pool.copy()
    baseline_pool["overlay_keep"] = True
    return pd.concat(
        [
            _scope_summary(baseline_pool, baseline_selected, "P2_BASELINE"),
            _scope_summary(pool, overlay_selected, "P2_AVOID_STRONG_BUCKET_LAGGARD"),
        ],
        ignore_index=True,
    )


def random_v104_controls(pool: pd.DataFrame, cfg: V104Config) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + iteration)
        randomized = pool.copy()
        for _, indices in randomized.groupby("entry_month", sort=False).groups.items():
            indices = np.asarray(list(indices), dtype=int)
            shuffled = rng.permutation(indices)
            randomized.loc[indices, FEATURE_COLUMNS] = randomized.loc[
                shuffled, FEATURE_COLUMNS
            ].to_numpy()
        randomized["graph_covered"] = randomized["bucket_ret_1h"].notna()
        randomized = add_v104_overlay_flags(randomized)
        selected, _ = _select_v104(randomized, "overlay_keep")
        rows.append(
            {
                "iteration": iteration,
                "selected_trades": int(len(selected)),
                "portfolio_net_rt40_over_8": float(selected["net_rt40"].sum() / 8.0),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_delta(
    baseline: pd.DataFrame,
    overlay: pd.DataFrame,
    cfg: V104Config,
) -> tuple[float, float]:
    baseline_day = baseline.groupby("entry_day")["net_rt40"].sum()
    overlay_day = overlay.groupby("entry_day")["net_rt40"].sum()
    days = sorted(set(baseline_day.index) | set(overlay_day.index))
    delta = np.asarray(
        [float(overlay_day.get(day, 0.0) - baseline_day.get(day, 0.0)) for day in days]
    )
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(delta), len(delta))
        boot.append(float(delta[chosen].mean()))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _positive_share(frame: pd.DataFrame, column: str) -> float:
    values = frame.groupby(column)["net_rt40"].sum().clip(lower=0)
    return float(values.max() / values.sum()) if values.sum() > 0 else np.inf


def audit_v104(
    pool: pd.DataFrame,
    baseline: pd.DataFrame,
    overlay: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V104Config,
) -> pd.DataFrame:
    base = {
        row.scope: row
        for row in summary[summary["policy"].eq("P2_BASELINE")].itertuples(index=False)
    }
    active = {
        row.scope: row
        for row in summary[
            summary["policy"].eq("P2_AVOID_STRONG_BUCKET_LAGGARD")
        ].itertuples(index=False)
    }
    full_lift = active["all"].portfolio_net_rt40_over_8 - base["all"].portfolio_net_rt40_over_8
    shifted_lift = float(shifted["net_rt40"].sum() / 8.0 - base["all"].portfolio_net_rt40_over_8)
    random_lifts = controls["portfolio_net_rt40_over_8"] - base["all"].portfolio_net_rt40_over_8
    random_percentile = float(random_lifts.lt(full_lift).mean())
    ci_low, ci_high = _bootstrap_delta(baseline, overlay, cfg)
    blocked = pool[~pool["overlay_keep"]]
    gates: dict[str, tuple[bool, float]] = {
        "covered_candidates_120": (pool["graph_covered"].sum() >= 120, pool["graph_covered"].sum()),
        "overlay_selected_100": (len(overlay) >= 100, len(overlay)),
        "validation_selected_25": (active["validation"].selected_trades >= 25, active["validation"].selected_trades),
        "holdout_selected_25": (active["holdout"].selected_trades >= 25, active["holdout"].selected_trades),
        "full_portfolio_positive": (active["all"].portfolio_net_rt40_over_8 > 0, active["all"].portfolio_net_rt40_over_8),
        "validation_portfolio_positive": (active["validation"].portfolio_net_rt40_over_8 > 0, active["validation"].portfolio_net_rt40_over_8),
        "holdout_portfolio_positive": (active["holdout"].portfolio_net_rt40_over_8 > 0, active["holdout"].portfolio_net_rt40_over_8),
        "validation_lift_positive": (
            active["validation"].portfolio_net_rt40_over_8 - base["validation"].portfolio_net_rt40_over_8 > 0,
            active["validation"].portfolio_net_rt40_over_8 - base["validation"].portfolio_net_rt40_over_8,
        ),
        "holdout_lift_positive": (
            active["holdout"].portfolio_net_rt40_over_8 - base["holdout"].portfolio_net_rt40_over_8 > 0,
            active["holdout"].portfolio_net_rt40_over_8 - base["holdout"].portfolio_net_rt40_over_8,
        ),
        "blocked_mean_negative": (blocked["net_rt40"].mean() < 0, blocked["net_rt40"].mean()),
        "permutation_p90": (random_percentile >= 0.90, random_percentile),
        "beats_shifted_placebo": (full_lift > shifted_lift, full_lift - shifted_lift),
        "bootstrap_lower_positive": (ci_low > 0, ci_low),
        "retains_70pct_pool": (pool["overlay_keep"].mean() >= 0.70, pool["overlay_keep"].mean()),
        "month_share_below_35pct": (_positive_share(overlay, "entry_month") <= 0.35, _positive_share(overlay, "entry_month")),
        "symbol_share_below_35pct": (_positive_share(overlay, "symbol") <= 0.35, _positive_share(overlay, "symbol")),
    }
    eligible = all(bool(passed) for passed, _ in gates.values())
    verdict = (
        "p2_graph_bucket_forward_watch_only"
        if eligible
        else "reject_p2_graph_bucket_overlay"
    )
    return pd.DataFrame(
        [
            {
                "check": check,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
                "verdict": verdict,
                "full_lift": full_lift,
                "shifted_lift": shifted_lift,
                "random_percentile": random_percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
            for check, (passed, value) in gates.items()
        ]
    )


def write_v104_p2_graph_bucket_overlay(
    cfg: V104Config = V104Config(),
) -> dict[str, Path]:
    raw_pool = load_v104_p2_pool(cfg.p2_path)
    panel = load_v104_bucket_panel(cfg.bucket_path)
    pool = add_v104_overlay_flags(attach_v104_bucket_context(raw_pool, panel))
    baseline_pool = pool.copy()
    baseline_pool["baseline_keep"] = True
    baseline, _ = _select_v104(baseline_pool, "baseline_keep")
    overlay, capacity_skipped = _select_v104(pool, "overlay_keep")
    summary = summarize_v104(pool, baseline, overlay)
    shifted_panel = shifted_v103_panel(
        pd.read_parquet(cfg.bucket_path)
    )[["symbol", "feature_time", *FEATURE_COLUMNS]]
    shifted_pool = add_v104_overlay_flags(
        attach_v104_bucket_context(raw_pool, shifted_panel)
    )
    shifted, _ = _select_v104(shifted_pool, "overlay_keep")
    controls = random_v104_controls(pool, cfg)
    audit = audit_v104(pool, baseline, overlay, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "pool": root / "p2_pool_bucket_context.parquet",
        "baseline": root / "baseline_selected.parquet",
        "overlay": root / "overlay_selected.parquet",
        "capacity_skipped": root / "overlay_capacity_skipped.parquet",
        "summary": root / "overlay_summary.csv",
        "random_controls": root / "permutation_controls.csv",
        "audit": root / "overlay_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    pool.to_parquet(outputs["pool"], index=False)
    baseline.to_parquet(outputs["baseline"], index=False)
    overlay.to_parquet(outputs["overlay"], index=False)
    capacity_skipped.to_parquet(outputs["capacity_skipped"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["random_controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v10.4 P2 Graph-Bucket Overlay",
        "",
        f"Status: `{audit['verdict'].iloc[0]}`. Historical overlay audit only.",
        "",
    ]
    for row in summary.itertuples(index=False):
        if row.scope in {"all", "validation", "holdout"}:
            lines.append(
                f"- {row.policy}/{row.scope}: selected={row.selected_trades}, "
                f"portfolio_net_rt40_over_8={row.portfolio_net_rt40_over_8:.4%}."
            )
    lines.extend(["", "P2 and all live permissions remain unchanged."])
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
