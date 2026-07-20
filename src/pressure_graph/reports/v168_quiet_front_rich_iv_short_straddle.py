"""Executable rich-IV short straddle gated by a quiet alt-volatility front."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


V167_ROOT = Path("reports/v16_7_alt_bucket_vol_front_btc_straddle")
TRADE_PATH = V167_ROOT / "all_eligible_daily_straddles.parquet"
REPORT_ROOT = Path("reports/v16_8_quiet_front_rich_iv_short_straddle")
FINDINGS_PATH = Path("docs/v168_quiet_front_rich_iv_short_straddle_findings_2026_07_16.md")


def _bootstrap_mean(values: np.ndarray, draws: int = 5_000, seed: int = 16_800) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _positive_concentration(frame: pd.DataFrame) -> tuple[float, float]:
    positive = frame[frame["short_primary_net_return"].gt(0)].copy()
    total = float(positive["short_primary_net_return"].sum())
    if total <= 0:
        return float("inf"), float("inf")
    monthly = positive.assign(month=positive["entry_time"].dt.strftime("%Y-%m")).groupby("month")[
        "short_primary_net_return"
    ].sum()
    return float(monthly.max() / total), float(positive["short_primary_net_return"].max() / total)


def _tail_metrics(values: pd.Series) -> tuple[float, float]:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if ordered.empty:
        return np.nan, np.nan
    tail_count = max(1, int(np.ceil(len(ordered) * 0.05)))
    return float(ordered.iloc[0]), float(ordered.iloc[:tail_count].mean())


def _summary_row(frame: pd.DataFrame, scope: str) -> dict[str, object]:
    return {
        "scope": scope,
        "trades": len(frame),
        "active_months": int(frame["entry_time"].dt.strftime("%Y-%m").nunique())
        if len(frame)
        else 0,
        "mean_short_gross_bp": (
            float(frame["short_gross_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "mean_short_primary_net_bp": (
            float(frame["short_primary_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "mean_short_stress_net_bp": (
            float(frame["short_stress_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "mean_identical_long_primary_net_bp": (
            float(frame["primary_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "win_rate": (
            float(frame["short_primary_net_return"].gt(0).mean()) if len(frame) else np.nan
        ),
        "mean_iv_rv_spread_vol_points": (
            float(frame["iv_rv_spread"].mean() * 100) if len(frame) else np.nan
        ),
    }


def evaluate_v168(
    all_trades: pd.DataFrame,
    circular_draws: int = 2_000,
    circular_seed: int = 16_801,
) -> dict[str, pd.DataFrame]:
    frame = all_trades.copy().sort_values("entry_time").reset_index(drop=True)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["rich_iv"] = frame["iv_rv_spread"].ge(0.10)
    frame["quiet_front"] = frame["front_gap"].le(0) & frame["alt_high_vol_breadth"].le(1 / 3)
    frame["candidate_entry_v168"] = frame["rich_iv"] & frame["quiet_front"]
    candidate = frame[frame["candidate_entry_v168"]].copy()
    iv_only = frame[frame["rich_iv"]].copy()

    quiet_times = set(frame.loc[frame["quiet_front"], "entry_time"] + pd.Timedelta(days=1))
    delayed = frame[frame["rich_iv"] & frame["entry_time"].isin(quiet_times)].copy()

    period_rows = [_summary_row(candidate, "all")]
    for period in ("development", "validation", "holdout"):
        period_rows.append(_summary_row(candidate[candidate["period"].eq(period)], period))
    period_summary = pd.DataFrame(period_rows)
    by_period = period_summary.set_index("scope")

    bootstrap_low = np.nan
    bootstrap_high = np.nan
    if len(candidate):
        bootstrap_low, bootstrap_high = _bootstrap_mean(
            candidate["short_primary_net_return"].to_numpy(dtype=float)
        )
    worst_trade, expected_shortfall_5 = _tail_metrics(candidate["short_primary_net_return"])
    month_concentration, trade_concentration = _positive_concentration(candidate)

    rich_mask = frame["rich_iv"].to_numpy(dtype=bool)
    quiet_mask = frame["quiet_front"].to_numpy(dtype=bool)
    short_returns = frame["short_primary_net_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(circular_seed)
    shifts = rng.integers(1, len(frame), size=circular_draws)
    circular_rows = []
    for draw, shift in enumerate(shifts):
        shifted_candidate = rich_mask & np.roll(quiet_mask, int(shift))
        values = short_returns[shifted_candidate]
        circular_rows.append(
            {
                "draw": draw,
                "shift": int(shift),
                "trades": len(values),
                "mean_short_primary_net_return": float(values.mean()) if len(values) else np.nan,
            }
        )
    circular = pd.DataFrame(circular_rows)
    real_mean = (
        float(candidate["short_primary_net_return"].mean()) if len(candidate) else np.nan
    )
    controls = circular["mean_short_primary_net_return"].dropna()
    circular_percentile = float((controls <= real_mean).mean()) if len(controls) else np.nan
    iv_only_mean = (
        float(iv_only["short_primary_net_return"].mean()) if len(iv_only) else np.nan
    )
    delayed_mean = (
        float(delayed["short_primary_net_return"].mean()) if len(delayed) else np.nan
    )
    identical_long_mean = (
        float(candidate["primary_net_return"].mean()) if len(candidate) else np.nan
    )

    gates = {
        "trades_20": len(candidate) >= 20,
        "validation_trades_5": int(by_period.loc["validation", "trades"]) >= 5,
        "holdout_trades_5": int(by_period.loc["holdout", "trades"]) >= 5,
        "development_primary_positive": by_period.loc[
            "development", "mean_short_primary_net_bp"
        ]
        > 0,
        "validation_primary_positive": by_period.loc[
            "validation", "mean_short_primary_net_bp"
        ]
        > 0,
        "holdout_primary_positive": by_period.loc["holdout", "mean_short_primary_net_bp"] > 0,
        "full_stress_positive": by_period.loc["all", "mean_short_stress_net_bp"] > 0,
        "bootstrap_lower_positive": bootstrap_low > 0,
        "circular_percentile_95": circular_percentile >= 0.95,
        "beats_iv_only": real_mean > iv_only_mean,
        "beats_delayed": real_mean > delayed_mean,
        "identical_long_negative": identical_long_mean < 0,
        "worst_trade_above_minus_500bp": worst_trade >= -0.05,
        "expected_shortfall_5_above_minus_300bp": expected_shortfall_5 >= -0.03,
        "month_concentration_50": month_concentration <= 0.50,
        "trade_concentration_35": trade_concentration <= 0.35,
    }
    gate_frame = pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in gates.items()]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": "OVS1_RICH_IV_QUIET_FRONT_SHORT_STRADDLE",
                "eligible_daily_straddles": len(frame),
                "rich_iv_days": len(iv_only),
                "candidate_trades": len(candidate),
                "candidate_primary_net_bp": real_mean * 10_000,
                "candidate_stress_net_bp": float(
                    candidate["short_stress_net_return"].mean() * 10_000
                )
                if len(candidate)
                else np.nan,
                "bootstrap_95_low_bp": bootstrap_low * 10_000,
                "bootstrap_95_high_bp": bootstrap_high * 10_000,
                "circular_percentile": circular_percentile,
                "iv_only_primary_net_bp": iv_only_mean * 10_000,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "identical_long_primary_net_bp": identical_long_mean * 10_000,
                "worst_trade_bp": worst_trade * 10_000,
                "expected_shortfall_5_bp": expected_shortfall_5 * 10_000,
                "positive_month_concentration": month_concentration,
                "positive_trade_concentration": trade_concentration,
                "promote_research_followup": bool(all(gates.values())),
                "failed_gates": "|".join(gate for gate, passed in gates.items() if not passed),
            }
        ]
    )
    return {
        "candidate": candidate,
        "iv_only": iv_only,
        "delayed": delayed,
        "period_summary": period_summary,
        "circular_controls": circular,
        "gates": gate_frame,
        "outcome": outcome,
    }


def _write_findings(results: dict[str, pd.DataFrame], path: Path) -> None:
    outcome = results["outcome"].iloc[0]
    verdict = (
        "promote_longer_executable_options_research"
        if bool(outcome["promote_research_followup"])
        else "reject_rich_iv_quiet_front_short_straddle"
    )
    text = [
        "# v16.8 Quiet Alt Front + Rich IV Short-Straddle Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        results["outcome"].to_markdown(index=False, floatfmt=".4f"),
        "",
        results["period_summary"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Short entries use archived bids and exits use archived asks; the result is not",
        "the negative of v16.7 long PnL. Static BTC delta hedging and option/hedge fees",
        "are included. The study is adaptive and the 2023 archive is short, so no",
        "PaperLive, leverage, remote-change, or real-order permission is granted.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v168_quiet_front_rich_iv_short_straddle(
    trade_path: Path = TRADE_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    all_trades = pd.read_parquet(trade_path)
    results = evaluate_v168(all_trades)
    root = ensure_dir(report_root)
    outputs = {
        "candidate_trades": root / "candidate_trades.parquet",
        "iv_only_trades": root / "iv_only_trades.parquet",
        "delayed_trades": root / "delayed_trades.parquet",
        "period_summary": root / "period_summary.csv",
        "circular_controls": root / "circular_controls.parquet",
        "gates": root / "gates.csv",
        "outcome": root / "summary.csv",
        "findings": findings_path,
    }
    results["candidate"].to_parquet(outputs["candidate_trades"], index=False)
    results["iv_only"].to_parquet(outputs["iv_only_trades"], index=False)
    results["delayed"].to_parquet(outputs["delayed_trades"], index=False)
    results["period_summary"].to_csv(outputs["period_summary"], index=False)
    results["circular_controls"].to_parquet(outputs["circular_controls"], index=False)
    results["gates"].to_csv(outputs["gates"], index=False)
    results["outcome"].to_csv(outputs["outcome"], index=False)
    _write_findings(results, findings_path)
    return outputs
