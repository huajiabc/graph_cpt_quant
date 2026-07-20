"""Direct BTCDVOL futures basis-convergence audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v11_7_direct_dvol_carry")
DVOL_PATH = Path("data/external/orthogonal_volatility/deribit_dvol_1h/BTC.parquet")
FUTURES_ROOT = Path("data/external/orthogonal_volatility/deribit_dvol_futures_1h")
CANDIDATES = ("DVC1_ALL_BASIS_CONVERGENCE", "DVC2_ABS2_BASIS_CONVERGENCE")


@dataclass(frozen=True)
class V117Config:
    dvol_path: Path = DVOL_PATH
    futures_root: Path = FUTURES_ROOT
    report_root: Path = REPORT_ROOT
    timing_days: int = 14
    basis_floor: float = 2.0
    random_iterations: int = 2000
    bootstrap_iterations: int = 5000
    seed: int = 20260715


def _period(expiration_time: pd.Timestamp) -> str:
    if expiration_time < pd.Timestamp("2025-01-01", tz="UTC"):
        return "development"
    if expiration_time < pd.Timestamp("2025-07-01", tz="UTC"):
        return "validation"
    return "holdout"


def load_v117_dvol(path: Path = DVOL_PATH) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["dvol_time", "close"])
    frame["dvol_time"] = pd.to_datetime(frame["dvol_time"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return (
        frame.dropna(subset=["dvol_time", "close"])
        .drop_duplicates("dvol_time", keep="last")
        .sort_values("dvol_time")
        .reset_index(drop=True)
    )


def _latest_dvol(dvol: pd.DataFrame, timestamp: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    times = dvol["dvol_time"].to_numpy(dtype="datetime64[ns]")
    target = np.datetime64(pd.Timestamp(timestamp).tz_convert("UTC").tz_localize(None))
    index = int(np.searchsorted(times, target, side="right")) - 1
    if index < 0:
        return None
    row = dvol.iloc[index]
    dvol_time = pd.Timestamp(row["dvol_time"])
    if timestamp - dvol_time > pd.Timedelta(hours=2):
        return None
    return dvol_time, float(row["close"])


def build_v117_contract_trade(
    futures: pd.DataFrame,
    dvol: pd.DataFrame,
    timing_days: int = 14,
) -> dict[str, Any] | None:
    frame = futures.copy()
    frame["bar_open_time"] = pd.to_datetime(
        frame["bar_open_time"], utc=True, errors="coerce"
    )
    frame["expiration_time"] = pd.to_datetime(
        frame["expiration_time"], utc=True, errors="coerce"
    )
    for column in ("open", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["bar_open_time", "expiration_time", "open", "close", "volume"]
    ).sort_values("bar_open_time")
    if frame.empty:
        return None
    expiration_time = pd.Timestamp(frame["expiration_time"].iloc[0])
    target_time = expiration_time - pd.Timedelta(days=timing_days)
    signal_rows = frame[
        frame["bar_open_time"].le(target_time)
        & frame["bar_open_time"].ge(target_time - pd.Timedelta(hours=48))
        & frame["volume"].gt(0)
    ]
    if signal_rows.empty:
        return None
    signal = signal_rows.iloc[-1]
    signal_time = pd.Timestamp(signal["bar_open_time"])
    dvol_point = _latest_dvol(dvol, signal_time)
    if dvol_point is None:
        return None
    dvol_time, dvol_close = dvol_point
    entry_rows = frame[
        frame["bar_open_time"].gt(signal_time)
        & frame["bar_open_time"].le(signal_time + pd.Timedelta(hours=24))
        & frame["volume"].gt(0)
    ]
    settlement_rows = frame[
        frame["bar_open_time"].ge(expiration_time)
        & frame["bar_open_time"].le(expiration_time + pd.Timedelta(hours=1))
    ]
    if entry_rows.empty or settlement_rows.empty:
        return None
    entry = entry_rows.iloc[0]
    settlement = settlement_rows.iloc[0]
    basis = float(signal["close"] - dvol_close)
    if not np.isfinite(basis) or basis == 0:
        return None
    direction = -float(np.sign(basis))
    entry_price = float(entry["open"])
    settlement_price = float(settlement["close"])
    if entry_price <= 0 or settlement_price <= 0:
        return None
    gross_return = direction * (settlement_price - entry_price) / entry_price
    return {
        "instrument_name": str(frame["instrument_name"].iloc[0]),
        "expiration_time": expiration_time,
        "period": _period(expiration_time),
        "timing_days": timing_days,
        "target_time": target_time,
        "signal_time": signal_time,
        "dvol_time": dvol_time,
        "entry_time": pd.Timestamp(entry["bar_open_time"]),
        "settlement_time": pd.Timestamp(settlement["bar_open_time"]),
        "signal_future_close": float(signal["close"]),
        "signal_dvol_close": dvol_close,
        "signal_basis_points": basis,
        "direction": direction,
        "entry_price": entry_price,
        "settlement_price": settlement_price,
        "gross_return": gross_return,
        "net10": gross_return - 0.0010,
        "net30": gross_return - 0.0030,
        "net50": gross_return - 0.0050,
    }


def build_v117_trades(
    dvol: pd.DataFrame,
    futures_root: Path = FUTURES_ROOT,
    timing_days: int = 14,
) -> pd.DataFrame:
    rows = []
    maximum_dvol_time = dvol["dvol_time"].max()
    for path in sorted(futures_root.glob("BTCDVOL_USDC-*.parquet")):
        frame = pd.read_parquet(path)
        expiration = pd.to_datetime(
            frame.get("expiration_time"), utc=True, errors="coerce"
        ).dropna()
        if expiration.empty or expiration.iloc[0] > maximum_dvol_time:
            continue
        trade = build_v117_contract_trade(frame, dvol, timing_days)
        if trade is not None:
            rows.append(trade)
    return pd.DataFrame(rows).sort_values("expiration_time").reset_index(drop=True)


def _candidate_sample(trades: pd.DataFrame, candidate: str, basis_floor: float) -> pd.DataFrame:
    if candidate == CANDIDATES[0]:
        return trades.copy()
    return trades[trades["signal_basis_points"].abs().ge(basis_floor)].copy()


def summarize_v117(trades: pd.DataFrame, cfg: V117Config) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        candidate_sample = _candidate_sample(trades, candidate, cfg.basis_floor)
        for period in ("full", "development", "validation", "holdout"):
            sample = (
                candidate_sample
                if period == "full"
                else candidate_sample[candidate_sample["period"].eq(period)]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "period": period,
                    "contracts": len(sample),
                    "mean_basis_points": float(sample["signal_basis_points"].mean())
                    if len(sample)
                    else np.nan,
                    "gross_return": float(sample["gross_return"].mean())
                    if len(sample)
                    else np.nan,
                    "net10": float(sample["net10"].mean()) if len(sample) else np.nan,
                    "net30": float(sample["net30"].mean()) if len(sample) else np.nan,
                    "net50": float(sample["net50"].mean()) if len(sample) else np.nan,
                    "win_rate_net10": float(sample["net10"].gt(0).mean())
                    if len(sample)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_interval(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if not len(values):
        return np.nan, np.nan
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def audit_v117(
    trades: pd.DataFrame,
    timing_controls: dict[int, pd.DataFrame],
    cfg: V117Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    controls = []
    decisions = []
    for candidate in CANDIDATES:
        sample = _candidate_sample(trades, candidate, cfg.basis_floor)
        values = sample["net10"].to_numpy(dtype=float)
        lower, upper = _bootstrap_interval(values, cfg.bootstrap_iterations, rng)
        move = (
            (sample["settlement_price"] - sample["entry_price"])
            .div(sample["entry_price"])
            .to_numpy(dtype=float)
        )
        random_means = (
            rng.choice(np.array([-1.0, 1.0]), size=(cfg.random_iterations, len(move)))
            * move[None, :]
            - 0.0010
        ).mean(axis=1) if len(move) else np.array([])
        real_mean = float(sample["net10"].mean()) if len(sample) else np.nan
        percentile = (
            float((random_means <= real_mean).mean() * 100.0)
            if len(random_means)
            else np.nan
        )
        reverse = float((-sample["gross_return"] - 0.0010).mean()) if len(sample) else np.nan
        positive = sample.loc[sample["net10"].gt(0), "net10"]
        concentration = (
            float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else np.nan
        )
        controls.extend(
            [
                {"candidate": candidate, "control": "bootstrap_low", "value": lower},
                {"candidate": candidate, "control": "bootstrap_high", "value": upper},
                {"candidate": candidate, "control": "random_sign_percentile", "value": percentile},
                {"candidate": candidate, "control": "reverse_direction_net10", "value": reverse},
                {"candidate": candidate, "control": "max_positive_contract_share", "value": concentration},
            ]
        )
        timing_values = {}
        for timing, frame in sorted(timing_controls.items()):
            timing_sample = _candidate_sample(frame, candidate, cfg.basis_floor)
            value = float(timing_sample["net10"].mean()) if len(timing_sample) else np.nan
            timing_values[timing] = value
            controls.append(
                {"candidate": candidate, "control": f"timing_{timing}d_net10", "value": value}
            )
        validation = sample[sample["period"].eq("validation")]
        holdout = sample[sample["period"].eq("holdout")]
        gates = {
            "validation_n_at_least_6": len(validation) >= 6,
            "holdout_n_at_least_6": len(holdout) >= 6,
            "full_net10_positive": real_mean > 0,
            "validation_net10_positive": float(validation["net10"].mean()) > 0
            if len(validation)
            else False,
            "holdout_net10_positive": float(holdout["net10"].mean()) > 0
            if len(holdout)
            else False,
            "full_net30_positive": float(sample["net30"].mean()) > 0 if len(sample) else False,
            "holdout_net30_positive": float(holdout["net30"].mean()) > 0
            if len(holdout)
            else False,
            "bootstrap_low_positive": lower > 0,
            "random_percentile_at_least_95": percentile >= 95,
            "beats_reversed_direction": real_mean > reverse,
            "positive_contract_share_below_35pct": concentration <= 0.35,
        }
        decisions.append(
            {
                "candidate": candidate,
                "verdict": "eligible_live_spread_validation" if all(gates.values()) else "reject",
                **gates,
            }
        )
    return pd.DataFrame(controls), pd.DataFrame(decisions)


def _markdown(summary: pd.DataFrame, controls: pd.DataFrame, decision: pd.DataFrame) -> str:
    lines = [
        "# v11.7 Direct DVOL Carry Audit",
        "",
        "This report is generated from preregistered direct BTCDVOL futures rules.",
        "Historical bid/ask is unavailable, so even a statistical pass would require live spread validation.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Controls",
        "",
        controls.to_markdown(index=False),
        "",
        "## Decision",
        "",
        decision.to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def write_v117_direct_dvol_carry(cfg: V117Config = V117Config()) -> dict[str, Path]:
    dvol = load_v117_dvol(cfg.dvol_path)
    trades = build_v117_trades(dvol, cfg.futures_root, cfg.timing_days)
    timing_controls = {
        timing: build_v117_trades(dvol, cfg.futures_root, timing) for timing in (7, 21)
    }
    summary = summarize_v117(trades, cfg)
    controls, decision = audit_v117(trades, timing_controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "trades": root / "contract_trades.parquet",
        "summary": root / "summary.csv",
        "controls": root / "controls.csv",
        "decision": root / "decision.csv",
        "report": root / "report.md",
    }
    trades.to_parquet(outputs["trades"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    decision.to_csv(outputs["decision"], index=False)
    outputs["report"].write_text(_markdown(summary, controls, decision), encoding="utf-8")
    return outputs


__all__ = [
    "CANDIDATES",
    "V117Config",
    "audit_v117",
    "build_v117_contract_trade",
    "build_v117_trades",
    "load_v117_dvol",
    "summarize_v117",
    "write_v117_direct_dvol_carry",
]
