"""Post-discovery 240-minute persistence audit for frozen v10.0 exact flow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v100_exact_taker_flow_alpha import (
    CANDIDATES,
    SEED,
    _extract_event,
    _price_at_or_after,
)


REPORT_ROOT = Path("reports/v10_1_exact_flow_persistence")
SOURCE_ROOT = Path("reports/v10_0_exact_taker_flow_alpha")
CACHE_ROOT = Path("data/processed/v100_exact_taker_flow_1m")
BTC_PATH = Path("data/raw/bybit/klines_1m_execution_public/BTCUSDT.parquet")
CANDIDATE = CANDIDATES[0]


@dataclass(frozen=True)
class V101Config:
    source_root: Path = SOURCE_ROOT
    cache_root: Path = CACHE_ROOT
    btc_path: Path = BTC_PATH
    report_root: Path = REPORT_ROOT
    random_iterations: int = 500
    bootstrap_iterations: int = 2000
    seed: int = SEED + 101
    btc_tolerance_minutes: int = 2


def _load_minute_data(root: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path).sort_values("bar_open_time").reset_index(drop=True)
        frame["bar_open_time"] = pd.to_datetime(frame["bar_open_time"], utc=True)
        frames[path.stem] = frame
    return frames


def _load_btc(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("bar_open_time").reset_index(drop=True)
    frame["bar_open_time"] = pd.to_datetime(frame["bar_open_time"], utc=True)
    return frame


def _btc_return(
    btc: pd.DataFrame,
    signal_time: pd.Timestamp,
    tolerance_minutes: int = 2,
) -> float:
    entry = _price_at_or_after(
        btc, pd.Timestamp(signal_time), tolerance_minutes=tolerance_minutes
    )
    exit_price = _price_at_or_after(
        btc,
        pd.Timestamp(signal_time) + pd.Timedelta(minutes=240),
        tolerance_minutes=tolerance_minutes,
    )
    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0:
        return np.nan
    return float(exit_price / entry - 1.0)


def attach_v101_btc(
    panel: pd.DataFrame,
    btc: pd.DataFrame,
    tolerance_minutes: int = 2,
) -> pd.DataFrame:
    out = panel.copy()
    out["btc_gross_return_240m"] = [
        _btc_return(btc, timestamp, tolerance_minutes) for timestamp in out["signal_time"]
    ]
    token = pd.to_numeric(out["gross_return_240m"], errors="coerce")
    benchmark = pd.to_numeric(out["btc_gross_return_240m"], errors="coerce")
    out["relative_gross_return_240m"] = token - benchmark
    out["hedged_net_return_240m_40bp"] = (
        out["relative_gross_return_240m"] - 0.004
    )
    return out


def summarize_v101(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", panel)]
    scopes.extend(
        (period, panel[panel["period"].eq(period)])
        for period in ("development", "validation", "holdout")
    )
    for scope, scoped in scopes:
        for path in ("all", "short_squeeze", "momentum_ignition"):
            path_data = scoped if path == "all" else scoped[scoped["path_name"].eq(path)]
            sample = path_data[path_data[CANDIDATE].fillna(False).astype(bool)]
            hedged = sample[
                sample["symbol"].ne("BTCUSDT")
                & sample["hedged_net_return_240m_40bp"].notna()
            ]
            rows.append(
                {
                    "scope": scope,
                    "path_name": path,
                    "candidate": f"{CANDIDATE}_240M",
                    "trades": int(len(sample)),
                    "symbols": int(sample["symbol"].nunique()),
                    "active_days": int(sample["entry_day"].nunique()),
                    "mean_raw_net10": float(sample["net_return_240m_10bp"].mean()),
                    "mean_raw_net20": float(sample["net_return_240m_20bp"].mean()),
                    "mean_raw_net30": float(sample["net_return_240m_30bp"].mean()),
                    "median_raw_net20": float(sample["net_return_240m_20bp"].median()),
                    "win_rate_raw_net20": float(sample["net_return_240m_20bp"].gt(0).mean()),
                    "hedged_trades": int(len(hedged)),
                    "mean_hedged_net40": float(
                        hedged["hedged_net_return_240m_40bp"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _is_of1(row: dict[str, Any]) -> bool:
    return bool(
        row["imbalance_5m"] >= 0.10
        and row["imbalance_15m"] >= 0.0
        and row["return_5m"] > 0.0
        and row["turnover_acceleration"] >= 1.0
    )


def _event_random_values(
    event: pd.Series,
    minute_data: dict[str, pd.DataFrame],
    btc: pd.DataFrame,
    btc_tolerance_minutes: int,
) -> list[tuple[float, float]]:
    day = pd.Timestamp(event["signal_time"]).floor("D")
    anchors = pd.date_range(
        day + pd.Timedelta(minutes=15),
        day + pd.Timedelta(hours=23, minutes=45),
        freq="15min",
    )
    values = []
    for anchor in anchors:
        temporary = event.copy()
        temporary["signal_time"] = anchor
        row = _extract_event(temporary, minute_data)
        if row is None:
            continue
        raw = np.nan
        hedged = np.nan
        if _is_of1(row):
            raw = float(row["net_return_240m_20bp"])
            btc_return = _btc_return(btc, anchor, btc_tolerance_minutes)
            if event["symbol"] != "BTCUSDT" and np.isfinite(btc_return):
                hedged = float(row["gross_return_240m"] - btc_return - 0.004)
        values.append((raw, hedged))
    return values


def random_v101_controls(
    panel: pd.DataFrame,
    minute_data: dict[str, pd.DataFrame],
    btc: pd.DataFrame,
    cfg: V101Config,
) -> pd.DataFrame:
    event_columns = [
        "event_id",
        "exchange",
        "symbol",
        "path_name",
        "original_signal_time",
        "period",
    ]
    events = panel[event_columns].copy()
    events = events.rename(columns={"original_signal_time": "signal_time"})
    pools = [
        _event_random_values(event, minute_data, btc, cfg.btc_tolerance_minutes)
        for _, event in events.iterrows()
    ]
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + iteration)
        raw_values = []
        hedged_values = []
        for pool in pools:
            if not pool:
                continue
            raw, hedged = pool[int(rng.integers(0, len(pool)))]
            if np.isfinite(raw):
                raw_values.append(raw)
            if np.isfinite(hedged):
                hedged_values.append(hedged)
        rows.append(
            {
                "iteration": iteration,
                "raw_trades": len(raw_values),
                "mean_raw_net20": float(np.mean(raw_values)) if raw_values else np.nan,
                "hedged_trades": len(hedged_values),
                "mean_hedged_net40": (
                    float(np.mean(hedged_values)) if hedged_values else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _stability(panel: pd.DataFrame, cfg: V101Config) -> dict[str, float]:
    sample = panel[panel[CANDIDATE].fillna(False).astype(bool)].copy()
    daily = [
        group["net_return_240m_20bp"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    day_sum = sample.groupby("entry_day")["net_return_240m_20bp"].sum()
    positive = day_sum.clip(lower=0)
    return {
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "max_positive_day_share": float(positive.max() / positive.sum()),
        "worst_day_sum": float(day_sum.min()),
    }


def audit_v101(
    panel: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V101Config,
) -> pd.DataFrame:
    full = summary[(summary["scope"] == "all") & (summary["path_name"] == "all")].iloc[0]
    validation = summary[
        (summary["scope"] == "validation") & (summary["path_name"] == "all")
    ].iloc[0]
    holdout = summary[
        (summary["scope"] == "holdout") & (summary["path_name"] == "all")
    ].iloc[0]
    shifted_summary = summarize_v101(shifted)
    shifted_full = shifted_summary[
        (shifted_summary["scope"] == "all")
        & (shifted_summary["path_name"] == "all")
    ].iloc[0]
    stability = _stability(panel, cfg)
    raw_percentile = float(
        random_controls["mean_raw_net20"].lt(full.mean_raw_net20).mean()
    )
    hedged_percentile = float(
        random_controls["mean_hedged_net40"].lt(full.mean_hedged_net40).mean()
    )
    path_rows = summary[
        (summary["scope"] == "all") & summary["path_name"].ne("all")
    ]
    gates = {
        "full_trades_at_least_80": (full.trades >= 80, full.trades),
        "validation_trades_at_least_25": (validation.trades >= 25, validation.trades),
        "holdout_trades_at_least_25": (holdout.trades >= 25, holdout.trades),
        "six_symbols": (full.symbols >= 6, full.symbols),
        "ten_active_days": (full.active_days >= 10, full.active_days),
        "validation_raw_net20_positive": (
            validation.mean_raw_net20 > 0,
            validation.mean_raw_net20,
        ),
        "holdout_raw_net20_positive": (holdout.mean_raw_net20 > 0, holdout.mean_raw_net20),
        "full_raw_net30_positive": (full.mean_raw_net30 > 0, full.mean_raw_net30),
        "bootstrap_lower_positive": (
            stability["bootstrap_ci_low"] > 0,
            stability["bootstrap_ci_low"],
        ),
        "raw_random_p90": (raw_percentile >= 0.90, raw_percentile),
        "beats_shifted_placebo": (
            full.mean_raw_net20 > shifted_full.mean_raw_net20,
            full.mean_raw_net20 - shifted_full.mean_raw_net20,
        ),
        "both_paths_nonnegative": (
            path_rows["mean_raw_net20"].min() >= 0,
            path_rows["mean_raw_net20"].min(),
        ),
        "positive_day_share_below_35pct": (
            stability["max_positive_day_share"] <= 0.35,
            stability["max_positive_day_share"],
        ),
        "validation_hedged_net40_positive": (
            validation.mean_hedged_net40 > 0,
            validation.mean_hedged_net40,
        ),
        "holdout_hedged_net40_positive": (
            holdout.mean_hedged_net40 > 0,
            holdout.mean_hedged_net40,
        ),
        "hedged_random_p90": (hedged_percentile >= 0.90, hedged_percentile),
    }
    eligible = all(bool(passed) for passed, _ in gates.values())
    verdict = (
        "post_discovery_persistence_clue_only"
        if eligible
        else "reject_exact_flow_240m_persistence"
    )
    return pd.DataFrame(
        [
            {
                "candidate": f"{CANDIDATE}_240M",
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


def write_v101_exact_flow_persistence(
    cfg: V101Config = V101Config(),
) -> dict[str, Path]:
    panel = pd.read_parquet(cfg.source_root / "event_panel.parquet")
    shifted = pd.read_parquet(cfg.source_root / "shifted_60m_panel.parquet")
    btc = _load_btc(cfg.btc_path)
    panel = attach_v101_btc(panel, btc, cfg.btc_tolerance_minutes)
    shifted = attach_v101_btc(shifted, btc, cfg.btc_tolerance_minutes)
    summary = summarize_v101(panel)
    minute_data = _load_minute_data(cfg.cache_root)
    controls = random_v101_controls(panel, minute_data, btc, cfg)
    audit = audit_v101(panel, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "event_panel": root / "event_panel_btc_attributed.parquet",
        "summary": root / "persistence_summary.csv",
        "random_controls": root / "random_time_controls.csv",
        "audit": root / "persistence_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    panel.to_parquet(outputs["event_panel"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["random_controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    focal = summary[(summary["path_name"] == "all")]
    lines = [
        "# v10.1 Exact-Flow 240m Persistence",
        "",
        f"Status: `{audit['verdict'].iloc[0]}`. Post-discovery offline audit only.",
        "",
    ]
    for row in focal.itertuples(index=False):
        lines.append(
            f"- {row.scope}: raw_n={row.trades}, raw_net20={row.mean_raw_net20:.4%}, "
            f"hedged_n={row.hedged_trades}, hedged_net40={row.mean_hedged_net40:.4%}."
        )
    lines.extend(["", "P2 and all live permissions remain unchanged."])
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
