"""BTC perpetual direction from causal 25-delta option skew extremes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v167_alt_bucket_vol_front_btc_straddle import _valid_quote_rows


DATA_ROOT = Path("data/external/binance_option_vol_front")
OPTION_PATH = DATA_ROOT / "option_eoh_hour0.parquet"
FEATURE_PATH = Path(
    "reports/v16_7_alt_bucket_vol_front_btc_straddle/daily_volatility_features.parquet"
)
REPORT_ROOT = Path("reports/v17_0_option_skew_directional_btc")
FINDINGS_PATH = Path("docs/v170_option_skew_directional_btc_findings_2026_07_16.md")


def select_25_delta_skew(snapshot: pd.DataFrame, target_dte: float = 30.0) -> dict[str, object] | None:
    local = _valid_quote_rows(snapshot)
    local["dte"] = (
        pd.to_datetime(local["expiration_time"], utc=True)
        - pd.to_datetime(local["snapshot_time"], utc=True)
    ).dt.total_seconds() / 86_400
    local["mark_iv"] = pd.to_numeric(local["mark_iv"], errors="coerce")
    local["delta"] = pd.to_numeric(local["delta"], errors="coerce")
    local = local[
        local["dte"].between(21, 45)
        & local["mark_iv"].gt(0)
        & local["delta"].notna()
    ].copy()
    if local.empty:
        return None
    expiries = local[["expiration_time", "dte"]].drop_duplicates()
    expiries["target_gap"] = (expiries["dte"] - target_dte).abs()
    expiry = expiries.sort_values(["target_gap", "expiration_time"]).iloc[0][
        "expiration_time"
    ]
    local = local[local["expiration_time"].eq(expiry)].copy()
    calls = local[local["option_type"].eq("C") & local["delta"].between(0.10, 0.40)].copy()
    puts = local[local["option_type"].eq("P") & local["delta"].between(-0.40, -0.10)].copy()
    if calls.empty or puts.empty:
        return None
    calls["delta_gap"] = (calls["delta"] - 0.25).abs()
    puts["delta_gap"] = (puts["delta"] + 0.25).abs()
    call = calls.sort_values(["delta_gap", "strike_price"]).iloc[0]
    put = puts.sort_values(["delta_gap", "strike_price"]).iloc[0]
    return {
        "snapshot_time": call["snapshot_time"],
        "expiration_time": expiry,
        "dte": float(call["dte"]),
        "call_symbol": call["symbol"],
        "put_symbol": put["symbol"],
        "call_strike": float(call["strike_price"]),
        "put_strike": float(put["strike_price"]),
        "call_delta": float(call["delta"]),
        "put_delta": float(put["delta"]),
        "call_mark_iv": float(call["mark_iv"]),
        "put_mark_iv": float(put["mark_iv"]),
        "skew": float(put["mark_iv"] - call["mark_iv"]),
    }


def add_causal_skew_zscore(
    surface: pd.DataFrame,
    lookback: int = 30,
    maximum_span_days: int = 45,
) -> pd.DataFrame:
    result = surface.sort_values("snapshot_time").reset_index(drop=True).copy()
    result["skew_zscore"] = np.nan
    for position in range(lookback, len(result)):
        history = result.iloc[position - lookback : position]
        span = history["snapshot_time"].iloc[-1] - history["snapshot_time"].iloc[0]
        values = history["skew"].to_numpy(dtype=float)
        standard_deviation = float(np.std(values, ddof=0))
        if span <= pd.Timedelta(days=maximum_span_days) and standard_deviation > 0:
            current = float(result.loc[position, "skew"])
            result.loc[position, "skew_zscore"] = (
                current - float(np.mean(values))
            ) / standard_deviation
    return result


def build_skew_surface(options: pd.DataFrame) -> pd.DataFrame:
    frame = options.copy()
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], utc=True)
    frame["expiration_time"] = pd.to_datetime(frame["expiration_time"], utc=True)
    rows = []
    for _, snapshot in frame.groupby("snapshot_time"):
        row = select_25_delta_skew(snapshot)
        if row is not None:
            rows.append(row)
    return add_causal_skew_zscore(pd.DataFrame(rows))


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2023-08-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2023-09-15", tz="UTC"):
        return "validation"
    return "holdout"


def build_skew_trades(surface: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_frame = features.copy()
    feature_frame["snapshot_time"] = pd.to_datetime(feature_frame["snapshot_time"], utc=True)
    prices = feature_frame.set_index("snapshot_time")["btc_price"]
    rows = []
    for row in surface.itertuples(index=False):
        zscore = float(row.skew_zscore)
        if not np.isfinite(zscore):
            continue
        entry_time = pd.Timestamp(row.snapshot_time)
        exit_time = entry_time + pd.Timedelta(days=1)
        if entry_time not in prices.index or exit_time not in prices.index:
            continue
        direction = -float(np.sign(zscore)) if abs(zscore) >= 1.0 else 0.0
        entry_btc = float(prices.loc[entry_time])
        exit_btc = float(prices.loc[exit_time])
        gross = direction * (exit_btc / entry_btc - 1)
        rows.append(
            {
                **row._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "period": _period(entry_time),
                "direction": direction,
                "entry_btc": entry_btc,
                "exit_btc": exit_btc,
                "btc_return": exit_btc / entry_btc - 1,
                "gross_return": gross,
                "primary_net_return": gross - (0.001 if direction else 0.0),
                "stress_net_return": gross - (0.002 if direction else 0.0),
                "reversed_primary_net_return": -gross - (0.001 if direction else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def _bootstrap(values: np.ndarray, draws: int = 5_000, seed: int = 17_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _positive_concentration(frame: pd.DataFrame) -> tuple[float, float]:
    positive = frame[frame["primary_net_return"].gt(0)].copy()
    total = float(positive["primary_net_return"].sum())
    if total <= 0:
        return float("inf"), float("inf")
    month = positive["entry_time"].dt.strftime("%Y-%m")
    monthly = positive.assign(month=month).groupby("month")["primary_net_return"].sum()
    return float(monthly.max() / total), float(positive["primary_net_return"].max() / total)


def _summary(frame: pd.DataFrame, scope: str) -> dict[str, object]:
    return {
        "scope": scope,
        "trades": len(frame),
        "short_fraction": float(frame["direction"].lt(0).mean()) if len(frame) else np.nan,
        "mean_gross_bp": float(frame["gross_return"].mean() * 10_000) if len(frame) else np.nan,
        "mean_primary_net_bp": (
            float(frame["primary_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "mean_stress_net_bp": (
            float(frame["stress_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "win_rate": float(frame["primary_net_return"].gt(0).mean()) if len(frame) else np.nan,
        "mean_abs_skew_zscore": (
            float(frame["skew_zscore"].abs().mean()) if len(frame) else np.nan
        ),
    }


def evaluate_v170(
    all_rows: pd.DataFrame,
    features: pd.DataFrame,
    circular_draws: int = 2_000,
    circular_seed: int = 17_001,
) -> dict[str, pd.DataFrame]:
    calendar = all_rows.copy().sort_values("entry_time").reset_index(drop=True)
    candidate = calendar[calendar["direction"].ne(0)].copy()
    period_rows = [_summary(candidate, "all")]
    for period in ("development", "validation", "holdout"):
        period_rows.append(_summary(candidate[candidate["period"].eq(period)], period))
    period_summary = pd.DataFrame(period_rows)
    by_period = period_summary.set_index("scope")

    bootstrap_low = np.nan
    bootstrap_high = np.nan
    if len(candidate):
        bootstrap_low, bootstrap_high = _bootstrap(
            candidate["primary_net_return"].to_numpy(dtype=float)
        )
    month_concentration, trade_concentration = _positive_concentration(candidate)
    worst_trade = (
        float(candidate["primary_net_return"].min()) if len(candidate) else np.nan
    )

    feature_frame = features.copy()
    feature_frame["snapshot_time"] = pd.to_datetime(feature_frame["snapshot_time"], utc=True)
    prices = feature_frame.set_index("snapshot_time")["btc_price"]
    delayed_rows = []
    for row in candidate.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time) + pd.Timedelta(days=1)
        exit_time = entry_time + pd.Timedelta(days=1)
        if entry_time not in prices.index or exit_time not in prices.index:
            continue
        btc_return = float(prices.loc[exit_time] / prices.loc[entry_time] - 1)
        delayed_rows.append(
            {
                "entry_time": entry_time,
                "direction": row.direction,
                "primary_net_return": row.direction * btc_return - 0.001,
            }
        )
    delayed = pd.DataFrame(delayed_rows)
    delayed_mean = (
        float(delayed["primary_net_return"].mean()) if len(delayed) else np.nan
    )
    reversed_mean = (
        float(candidate["reversed_primary_net_return"].mean()) if len(candidate) else np.nan
    )

    directions = calendar["direction"].to_numpy(dtype=float)
    btc_returns = calendar["btc_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(circular_seed)
    shifts = rng.integers(1, len(calendar), size=circular_draws)
    circular_rows = []
    for draw, shift in enumerate(shifts):
        shifted = np.roll(directions, int(shift))
        active = shifted != 0
        values = shifted[active] * btc_returns[active] - 0.001
        circular_rows.append(
            {
                "draw": draw,
                "shift": int(shift),
                "trades": len(values),
                "mean_primary_net_return": float(values.mean()) if len(values) else np.nan,
            }
        )
    circular = pd.DataFrame(circular_rows)
    real_mean = (
        float(candidate["primary_net_return"].mean()) if len(candidate) else np.nan
    )
    control_means = circular["mean_primary_net_return"].dropna()
    circular_percentile = (
        float((control_means <= real_mean).mean()) if len(control_means) else np.nan
    )
    gates = {
        "trades_30": len(candidate) >= 30,
        "development_trades_8": int(by_period.loc["development", "trades"]) >= 8,
        "validation_trades_8": int(by_period.loc["validation", "trades"]) >= 8,
        "holdout_trades_8": int(by_period.loc["holdout", "trades"]) >= 8,
        "development_primary_positive": by_period.loc["development", "mean_primary_net_bp"] > 0,
        "validation_primary_positive": by_period.loc["validation", "mean_primary_net_bp"] > 0,
        "holdout_primary_positive": by_period.loc["holdout", "mean_primary_net_bp"] > 0,
        "full_stress_positive": by_period.loc["all", "mean_stress_net_bp"] > 0,
        "bootstrap_lower_positive": bootstrap_low > 0,
        "circular_percentile_95": circular_percentile >= 0.95,
        "beats_delayed": real_mean > delayed_mean,
        "beats_reversed": real_mean > reversed_mean,
        "month_concentration_50": month_concentration <= 0.50,
        "trade_concentration_35": trade_concentration <= 0.35,
        "worst_trade_above_minus_500bp": worst_trade >= -0.05,
    }
    gate_frame = pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in gates.items()]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": "OSD1_25D_SKEW_FOLLOW_BTC",
                "surface_days_with_zscore": int(calendar["skew_zscore"].notna().sum()),
                "candidate_trades": len(candidate),
                "primary_net_bp": real_mean * 10_000,
                "stress_net_bp": float(candidate["stress_net_return"].mean() * 10_000)
                if len(candidate)
                else np.nan,
                "bootstrap_95_low_bp": bootstrap_low * 10_000,
                "bootstrap_95_high_bp": bootstrap_high * 10_000,
                "circular_percentile": circular_percentile,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "positive_month_concentration": month_concentration,
                "positive_trade_concentration": trade_concentration,
                "worst_trade_bp": worst_trade * 10_000,
                "promote_research_followup": bool(all(gates.values())),
                "failed_gates": "|".join(gate for gate, passed in gates.items() if not passed),
            }
        ]
    )
    return {
        "candidate": candidate,
        "delayed": delayed,
        "period_summary": period_summary,
        "circular_controls": circular,
        "gates": gate_frame,
        "outcome": outcome,
    }


def _write_findings(results: dict[str, pd.DataFrame], path: Path) -> None:
    outcome = results["outcome"].iloc[0]
    verdict = (
        "promote_option_skew_directional_audit"
        if bool(outcome["promote_research_followup"])
        else "reject_option_skew_directional_alpha"
    )
    text = [
        "# v17.0 BTC Option-Skew Directional Alpha Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        results["outcome"].to_markdown(index=False, floatfmt=".4f"),
        "",
        results["period_summary"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "The option surface is information only; the traded leg is BTC perpetual at",
        "10/20 bp total round-trip cost. The short 2023 archive grants no PaperLive,",
        "leverage, remote-change, or real-order permission.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v170_option_skew_directional_btc(
    option_path: Path = OPTION_PATH,
    feature_path: Path = FEATURE_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    options = pd.read_parquet(option_path)
    features = pd.read_parquet(feature_path)
    surface = build_skew_surface(options)
    all_rows = build_skew_trades(surface, features)
    results = evaluate_v170(all_rows, features)
    root = ensure_dir(report_root)
    outputs = {
        "surface": root / "daily_25d_skew_surface.parquet",
        "all_rows": root / "all_skew_signal_rows.parquet",
        "candidate_trades": root / "candidate_trades.parquet",
        "delayed_trades": root / "delayed_trades.parquet",
        "period_summary": root / "period_summary.csv",
        "circular_controls": root / "circular_controls.parquet",
        "gates": root / "gates.csv",
        "outcome": root / "summary.csv",
        "findings": findings_path,
    }
    surface.to_parquet(outputs["surface"], index=False)
    all_rows.to_parquet(outputs["all_rows"], index=False)
    results["candidate"].to_parquet(outputs["candidate_trades"], index=False)
    results["delayed"].to_parquet(outputs["delayed_trades"], index=False)
    results["period_summary"].to_csv(outputs["period_summary"], index=False)
    results["circular_controls"].to_parquet(outputs["circular_controls"], index=False)
    results["gates"].to_csv(outputs["gates"], index=False)
    results["outcome"].to_csv(outputs["outcome"], index=False)
    _write_findings(results, findings_path)
    return outputs
