from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, read_parquet


SEARCH_END = pd.Timestamp("2026-02-01T00:00:00Z")
VALIDATION_END = pd.Timestamp("2026-05-01T00:00:00Z")
P2_PORTFOLIO_ID = "P2_MAX8_BASELINE"
REPORT_ROOT = Path("reports/v9_0_p2_continuation_strength")
REPLAY_LEDGER = Path("reports/v0_7d2_cic_mir1_replay/checkpoint_trade_ledger.parquet")
FORWARD_LEDGER = Path("reports/v0_7d2_cic_mir1_paper_live/forward/checkpoint_trades.parquet")
SCORE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
SCORE_LABELS = ["00_20", "20_40", "40_60", "60_80", "80_100"]


@dataclass(frozen=True)
class V90Config:
    replay_ledger: Path = REPLAY_LEDGER
    forward_ledger: Path = FORWARD_LEDGER
    report_root: Path = REPORT_ROOT
    bootstrap_samples: int = 2000
    permutation_samples: int = 2000
    seed: int = 20260711


def _clip_scale(values: pd.Series, low: float, width: float) -> pd.Series:
    return ((pd.to_numeric(values, errors="coerce") - low) / width).clip(0.0, 1.0)


def add_continuation_strength(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["strength_market_breadth"] = _clip_scale(
        out.get("volume_impulse_density_at_signal", pd.Series(np.nan, index=out.index)), 0.10, 0.20
    )
    out["strength_beta"] = _clip_scale(
        out.get("beta_extension_score_at_signal", pd.Series(np.nan, index=out.index)), 80.0, 20.0
    )
    out["strength_local_shock"] = _clip_scale(
        out.get("local_volume_shock_strength_at_signal", pd.Series(np.nan, index=out.index)), 2.0, 4.0
    )
    pullback = pd.to_datetime(out.get("pullback_time"), utc=True, errors="coerce")
    reclaim = pd.to_datetime(out.get("reclaim_time"), utc=True, errors="coerce")
    reclaim_minutes = (reclaim - pullback).dt.total_seconds() / 60.0
    out["strength_reclaim_speed"] = (1.0 - reclaim_minutes / 120.0).clip(0.0, 1.0)
    components = [
        "strength_market_breadth",
        "strength_beta",
        "strength_local_shock",
        "strength_reclaim_speed",
    ]
    out["strength_components_present"] = out[components].notna().sum(axis=1)
    out["continuation_strength"] = out[components].mean(axis=1, skipna=True)
    out.loc[out["strength_components_present"] < 3, "continuation_strength"] = np.nan
    out["strength_bin"] = pd.cut(
        out["continuation_strength"], bins=SCORE_BINS, labels=SCORE_LABELS, include_lowest=True, right=False
    )
    return out


def _period(entry: pd.Series) -> pd.Series:
    out = pd.Series("legacy_holdout", index=entry.index, dtype="string")
    out.loc[entry < SEARCH_END] = "search_reference"
    out.loc[(entry >= SEARCH_END) & (entry < VALIDATION_END)] = "validation_reference"
    return out


def prepare_p2_core(frame: pd.DataFrame, *, forward_only: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out = out[out.get("portfolio_id", "").astype(str).eq(P2_PORTFOLIO_ID)]
    if "is_overflow" in out.columns:
        out = out[~out["is_overflow"].fillna(False).astype(bool)]
    if forward_only:
        timely = out.get("timely_forward_observation", pd.Series(False, index=out.index))
        out = out[timely.fillna(False).astype(bool)]
    out = out.drop_duplicates("trade_id", keep="last")
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["period"] = "timely_forward" if forward_only else _period(out["entry_time"])
    out = add_continuation_strength(out)
    out["net20"] = pd.to_numeric(
        out.get("net_return_20bp", out.get("effective_net_return_20bp")), errors="coerce"
    )
    return out.dropna(subset=["trade_id", "burst_id", "entry_time", "continuation_strength", "net20"])


def burst_level(core: pd.DataFrame) -> pd.DataFrame:
    if core.empty:
        return pd.DataFrame(
            columns=["period", "burst_id", "month", "burst_score", "burst_net20", "trades"]
        )
    data = core.copy()
    data["month"] = data["entry_time"].dt.strftime("%Y-%m")
    return (
        data.groupby(["period", "burst_id", "month"], observed=True, dropna=False)
        .agg(
            burst_score=("continuation_strength", "mean"),
            burst_net20=("net20", lambda value: float(pd.to_numeric(value, errors="coerce").sum()) / 8.0),
            trades=("trade_id", "nunique"),
        )
        .reset_index()
    )


def _slope(sample: pd.DataFrame) -> float:
    clean = sample.dropna(subset=["burst_score", "burst_net20"])
    if len(clean) < 3 or clean["burst_score"].nunique() < 2:
        return np.nan
    return float(np.polyfit(clean["burst_score"], clean["burst_net20"], 1)[0])


def _stability_row(
    period: str,
    sample: pd.DataFrame,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, object]:
    clean = sample.dropna(subset=["burst_score", "burst_net20"]).reset_index(drop=True)
    observed_slope = _slope(clean)
    spearman = clean["burst_score"].corr(clean["burst_net20"], method="spearman") if len(clean) >= 3 else np.nan
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    if len(clean) >= 3:
        for _ in range(bootstrap_samples):
            sampled = clean.iloc[rng.integers(0, len(clean), len(clean))]
            value = _slope(sampled)
            if np.isfinite(value):
                bootstrap.append(value)
    null_slopes: list[float] = []
    if len(clean) >= 3 and np.isfinite(observed_slope):
        for _ in range(permutation_samples):
            shuffled = clean.copy()
            shuffled["burst_score"] = rng.permutation(shuffled["burst_score"].to_numpy())
            value = _slope(shuffled)
            if np.isfinite(value):
                null_slopes.append(value)
    return {
        "period": period,
        "bursts": int(len(clean)),
        "trades": int(pd.to_numeric(clean.get("trades"), errors="coerce").fillna(0).sum()),
        "mean_burst_net20": float(clean["burst_net20"].mean()) if not clean.empty else np.nan,
        "slope": observed_slope,
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "slope_ci_low": float(np.quantile(bootstrap, 0.025)) if bootstrap else np.nan,
        "slope_ci_high": float(np.quantile(bootstrap, 0.975)) if bootstrap else np.nan,
        "permutation_p_two_sided": (
            float(np.mean(np.abs(null_slopes) >= abs(observed_slope))) if null_slopes else np.nan
        ),
        "decision_sample_ready": bool(period == "timely_forward" and len(clean) >= 30),
    }


def summarize_strength(
    core: pd.DataFrame,
    bursts: pd.DataFrame,
    cfg: V90Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if core.empty:
        bins = pd.DataFrame(columns=["period", "strength_bin", "trades", "mean_net20"])
    else:
        bins = (
            core.groupby(["period", "strength_bin"], observed=False, dropna=False)
            .agg(trades=("trade_id", "nunique"), mean_net20=("net20", "mean"))
            .reset_index()
        )
    stability = pd.DataFrame(
        [
            _stability_row(
                str(period),
                sample,
                bootstrap_samples=cfg.bootstrap_samples,
                permutation_samples=cfg.permutation_samples,
                seed=cfg.seed + idx,
            )
            for idx, (period, sample) in enumerate(bursts.groupby("period", observed=True, sort=True))
        ]
    )
    return bins, stability


def _read_optional(path: Path) -> pd.DataFrame:
    return read_parquet(path) if path.exists() else pd.DataFrame()


def write_v90_p2_continuation_strength(cfg: V90Config = V90Config()) -> dict[str, Path]:
    replay = prepare_p2_core(_read_optional(cfg.replay_ledger), forward_only=False)
    forward = prepare_p2_core(_read_optional(cfg.forward_ledger), forward_only=True)
    core = pd.concat([replay, forward], ignore_index=True, sort=False)
    bursts = burst_level(core)
    bins, stability = summarize_strength(core, bursts, cfg)

    root = ensure_dir(cfg.report_root)
    outputs = {
        "trade_scores": root / "trade_scores.csv",
        "burst_scores": root / "burst_scores.csv",
        "score_bins": root / "score_bins.csv",
        "stability": root / "stability.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    core.to_csv(outputs["trade_scores"], index=False)
    bursts.to_csv(outputs["burst_scores"], index=False)
    bins.to_csv(outputs["score_bins"], index=False)
    stability.to_csv(outputs["stability"], index=False)
    forward_trades = int(len(forward))
    forward_bursts = int(forward["burst_id"].nunique()) if not forward.empty else 0
    lines = [
        "# v9.0 P2 Continuation Strength",
        "",
        "Status: pre-registered diagnostic only; no live action or threshold tuning.",
        "",
        f"- timely_forward_trades: {forward_trades}",
        f"- timely_forward_bursts: {forward_bursts}",
        f"- decision_sample_ready: {forward_trades >= 100 and forward_bursts >= 30}",
        "- legacy holdout is descriptive only and cannot approve the hypothesis.",
        "- next decision requires 100 timely trades and 30 timely bursts.",
    ]
    outputs["candidate_notes"].write_text("\n".join(lines), encoding="utf-8")
    return outputs


__all__ = [
    "V90Config",
    "add_continuation_strength",
    "burst_level",
    "prepare_p2_core",
    "summarize_strength",
    "write_v90_p2_continuation_strength",
]
