"""BTC perpetual direction from causal innovations in 25-delta option skew."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


SURFACE_PATH = Path("reports/v17_0_option_skew_directional_btc/daily_25d_skew_surface.parquet")
FEATURE_PATH = Path(
    "reports/v16_7_alt_bucket_vol_front_btc_straddle/daily_volatility_features.parquet"
)
REPORT_ROOT = Path("reports/v17_1_option_skew_innovation_btc")
FINDINGS_PATH = Path("docs/v171_option_skew_innovation_btc_findings_2026_07_16.md")


def add_causal_innovation_zscore(
    surface: pd.DataFrame,
    lookback: int = 30,
    maximum_gap_days: int = 3,
    maximum_history_span_days: int = 45,
) -> pd.DataFrame:
    result = surface.sort_values("snapshot_time").reset_index(drop=True).copy()
    result["skew_innovation"] = np.nan
    for position in range(1, len(result)):
        gap = result.loc[position, "snapshot_time"] - result.loc[position - 1, "snapshot_time"]
        if gap <= pd.Timedelta(days=maximum_gap_days):
            result.loc[position, "skew_innovation"] = (
                result.loc[position, "skew"] - result.loc[position - 1, "skew"]
            )
    result["innovation_zscore"] = np.nan
    for position in range(len(result)):
        history = result.iloc[:position].dropna(subset=["skew_innovation"]).tail(lookback)
        if len(history) != lookback:
            continue
        span = history["snapshot_time"].iloc[-1] - history["snapshot_time"].iloc[0]
        current = result.loc[position, "skew_innovation"]
        standard_deviation = float(np.std(history["skew_innovation"], ddof=0))
        if (
            pd.notna(current)
            and span <= pd.Timedelta(days=maximum_history_span_days)
            and standard_deviation > 0
        ):
            result.loc[position, "innovation_zscore"] = (
                float(current) - float(history["skew_innovation"].mean())
            ) / standard_deviation
    return result


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2023-08-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2023-09-15", tz="UTC"):
        return "validation"
    return "holdout"


def build_innovation_trades(surface: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    feature_frame = features.copy()
    feature_frame["snapshot_time"] = pd.to_datetime(feature_frame["snapshot_time"], utc=True)
    prices = feature_frame.set_index("snapshot_time")["btc_price"]
    rows = []
    for row in surface.itertuples(index=False):
        zscore = float(row.innovation_zscore)
        if not np.isfinite(zscore):
            continue
        entry_time = pd.Timestamp(row.snapshot_time)
        exit_time = entry_time + pd.Timedelta(days=1)
        if entry_time not in prices.index or exit_time not in prices.index:
            continue
        direction = -float(np.sign(zscore)) if abs(zscore) >= 1.0 else 0.0
        entry_btc = float(prices.loc[entry_time])
        exit_btc = float(prices.loc[exit_time])
        btc_return = exit_btc / entry_btc - 1
        gross = direction * btc_return
        rows.append(
            {
                **row._asdict(),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "period": _period(entry_time),
                "direction": direction,
                "entry_btc": entry_btc,
                "exit_btc": exit_btc,
                "btc_return": btc_return,
                "gross_return": gross,
                "primary_net_return": gross - (0.001 if direction else 0.0),
                "stress_net_return": gross - (0.002 if direction else 0.0),
                "reversed_primary_net_return": -gross - (0.001 if direction else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


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
    }


def _bootstrap(values: np.ndarray, draws: int = 5_000, seed: int = 17_100) -> tuple[float, float]:
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


def evaluate_v171(
    all_rows: pd.DataFrame,
    features: pd.DataFrame,
    circular_draws: int = 2_000,
    circular_seed: int = 17_101,
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
    controls = circular["mean_primary_net_return"].dropna()
    circular_percentile = float((controls <= real_mean).mean()) if len(controls) else np.nan
    development_short_fraction = float(by_period.loc["development", "short_fraction"])
    validation_short_fraction = float(by_period.loc["validation", "short_fraction"])
    gates = {
        "trades_25": len(candidate) >= 25,
        "development_trades_5": int(by_period.loc["development", "trades"]) >= 5,
        "validation_trades_5": int(by_period.loc["validation", "trades"]) >= 5,
        "holdout_trades_5": int(by_period.loc["holdout", "trades"]) >= 5,
        "development_primary_positive": by_period.loc["development", "mean_primary_net_bp"] > 0,
        "validation_primary_positive": by_period.loc["validation", "mean_primary_net_bp"] > 0,
        "holdout_primary_positive": by_period.loc["holdout", "mean_primary_net_bp"] > 0,
        "full_stress_positive": by_period.loc["all", "mean_stress_net_bp"] > 0,
        "bootstrap_lower_positive": bootstrap_low > 0,
        "circular_percentile_95": circular_percentile >= 0.95,
        "beats_delayed": real_mean > delayed_mean,
        "beats_reversed": real_mean > reversed_mean,
        "development_direction_balance": 0.25 <= development_short_fraction <= 0.75,
        "validation_direction_balance": 0.25 <= validation_short_fraction <= 0.75,
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
                "candidate": "OSD2_25D_SKEW_INNOVATION_FOLLOW_BTC",
                "innovation_days_with_zscore": int(calendar["innovation_zscore"].notna().sum()),
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
        "promote_skew_innovation_audit"
        if bool(outcome["promote_research_followup"])
        else "reject_option_skew_innovation_alpha"
    )
    text = [
        "# v17.1 BTC Option-Skew Innovation Alpha Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        results["outcome"].to_markdown(index=False, floatfmt=".4f"),
        "",
        results["period_summary"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "The option surface is used only as a causal signal; BTC perpetual is the",
        "traded leg at 10/20 bp total cost. This adaptive short-history study changes",
        "no PaperLive, leverage, remote, or real-order permission.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v171_option_skew_innovation_btc(
    surface_path: Path = SURFACE_PATH,
    feature_path: Path = FEATURE_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    surface = pd.read_parquet(surface_path)
    surface["snapshot_time"] = pd.to_datetime(surface["snapshot_time"], utc=True)
    innovation_surface = add_causal_innovation_zscore(surface)
    features = pd.read_parquet(feature_path)
    all_rows = build_innovation_trades(innovation_surface, features)
    results = evaluate_v171(all_rows, features)
    root = ensure_dir(report_root)
    outputs = {
        "surface": root / "daily_skew_innovation_surface.parquet",
        "all_rows": root / "all_innovation_signal_rows.parquet",
        "candidate_trades": root / "candidate_trades.parquet",
        "delayed_trades": root / "delayed_trades.parquet",
        "period_summary": root / "period_summary.csv",
        "circular_controls": root / "circular_controls.parquet",
        "gates": root / "gates.csv",
        "outcome": root / "summary.csv",
        "findings": findings_path,
    }
    innovation_surface.to_parquet(outputs["surface"], index=False)
    all_rows.to_parquet(outputs["all_rows"], index=False)
    results["candidate"].to_parquet(outputs["candidate_trades"], index=False)
    results["delayed"].to_parquet(outputs["delayed_trades"], index=False)
    results["period_summary"].to_csv(outputs["period_summary"], index=False)
    results["circular_controls"].to_parquet(outputs["circular_controls"], index=False)
    results["gates"].to_csv(outputs["gates"], index=False)
    results["outcome"].to_csv(outputs["outcome"], index=False)
    _write_findings(results, findings_path)
    return outputs
