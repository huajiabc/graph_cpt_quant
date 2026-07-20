"""Weekly executable straddles with daily archived-delta BTC hedging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v167_alt_bucket_vol_front_btc_straddle import (
    PRIMARY_HEDGE_FEE_RATE_PER_SIDE,
    PRIMARY_OPTION_FEE_RATE,
    STRESS_HEDGE_FEE_RATE_PER_SIDE,
    STRESS_OPTION_FEE_RATE,
    _option_fee,
    _valid_quote_rows,
    select_atm_straddle,
)


DATA_ROOT = Path("data/external/binance_option_vol_front")
OPTION_PATH = DATA_ROOT / "option_eoh_hour0.parquet"
FEATURE_PATH = Path(
    "reports/v16_7_alt_bucket_vol_front_btc_straddle/daily_volatility_features.parquet"
)
REPORT_ROOT = Path("reports/v16_9_weekly_delta_hedged_quiet_front_short_straddle")
FINDINGS_PATH = Path(
    "docs/v169_weekly_delta_hedged_quiet_front_short_straddle_findings_2026_07_16.md"
)


def calculate_weekly_straddle_trade(
    option_path: list[pd.DataFrame],
    btc_prices: list[float],
) -> dict[str, float]:
    """Calculate short and long seven-day paths with daily static-between-close hedges."""
    if len(option_path) != 8 or len(btc_prices) != 8:
        raise ValueError("Weekly option and BTC paths must both contain eight snapshots")
    indexed = [pair.set_index("option_type") for pair in option_path]
    entry = indexed[0]
    exit_ = indexed[-1]
    deltas = np.array(
        [float(pair.loc[["C", "P"], "delta"].sum()) for pair in indexed[:-1]],
        dtype=float,
    )
    prices = np.asarray(btc_prices, dtype=float)
    if not np.isfinite(deltas).all() or not np.isfinite(prices).all():
        raise ValueError("Weekly delta and BTC paths must be finite")

    short_option_pnl = float(
        entry.loc[["C", "P"], "best_bid_price"].sum()
        - exit_.loc[["C", "P"], "best_ask_price"].sum()
    )
    long_option_pnl = float(
        exit_.loc[["C", "P"], "best_bid_price"].sum()
        - entry.loc[["C", "P"], "best_ask_price"].sum()
    )
    short_hedge_pnl = float(np.sum(deltas * np.diff(prices)))
    long_hedge_pnl = -short_hedge_pnl
    hedge_turnover_notional = float(
        abs(deltas[0]) * prices[0]
        + np.sum(np.abs(np.diff(deltas)) * prices[1:-1])
        + abs(deltas[-1]) * prices[-1]
    )

    short_primary_option_fees = sum(
        _option_fee(
            PRIMARY_OPTION_FEE_RATE,
            prices[0],
            float(entry.loc[side, "best_bid_price"]),
        )
        + _option_fee(
            PRIMARY_OPTION_FEE_RATE,
            prices[-1],
            float(exit_.loc[side, "best_ask_price"]),
        )
        for side in ("C", "P")
    )
    short_stress_option_fees = sum(
        _option_fee(
            STRESS_OPTION_FEE_RATE,
            prices[0],
            float(entry.loc[side, "best_bid_price"]),
        )
        + _option_fee(
            STRESS_OPTION_FEE_RATE,
            prices[-1],
            float(exit_.loc[side, "best_ask_price"]),
        )
        for side in ("C", "P")
    )
    long_primary_option_fees = sum(
        _option_fee(
            PRIMARY_OPTION_FEE_RATE,
            prices[0],
            float(entry.loc[side, "best_ask_price"]),
        )
        + _option_fee(
            PRIMARY_OPTION_FEE_RATE,
            prices[-1],
            float(exit_.loc[side, "best_bid_price"]),
        )
        for side in ("C", "P")
    )
    long_stress_option_fees = sum(
        _option_fee(
            STRESS_OPTION_FEE_RATE,
            prices[0],
            float(entry.loc[side, "best_ask_price"]),
        )
        + _option_fee(
            STRESS_OPTION_FEE_RATE,
            prices[-1],
            float(exit_.loc[side, "best_bid_price"]),
        )
        for side in ("C", "P")
    )
    primary_hedge_fees = PRIMARY_HEDGE_FEE_RATE_PER_SIDE * hedge_turnover_notional
    stress_hedge_fees = STRESS_HEDGE_FEE_RATE_PER_SIDE * hedge_turnover_notional
    short_gross_pnl = short_option_pnl + short_hedge_pnl
    long_gross_pnl = long_option_pnl + long_hedge_pnl
    short_primary = short_gross_pnl - short_primary_option_fees - primary_hedge_fees
    short_stress = short_gross_pnl - short_stress_option_fees - stress_hedge_fees
    long_primary = long_gross_pnl - long_primary_option_fees - primary_hedge_fees
    long_stress = long_gross_pnl - long_stress_option_fees - stress_hedge_fees
    return {
        "entry_credit": float(entry.loc[["C", "P"], "best_bid_price"].sum()),
        "entry_debit": float(entry.loc[["C", "P"], "best_ask_price"].sum()),
        "exit_credit": float(exit_.loc[["C", "P"], "best_bid_price"].sum()),
        "exit_debit": float(exit_.loc[["C", "P"], "best_ask_price"].sum()),
        "hedge_turnover_notional": hedge_turnover_notional,
        "short_option_pnl": short_option_pnl,
        "short_hedge_pnl": short_hedge_pnl,
        "short_gross_return": short_gross_pnl / prices[0],
        "short_primary_net_return": short_primary / prices[0],
        "short_stress_net_return": short_stress / prices[0],
        "long_option_pnl": long_option_pnl,
        "long_hedge_pnl": long_hedge_pnl,
        "long_gross_return": long_gross_pnl / prices[0],
        "long_primary_net_return": long_primary / prices[0],
        "long_stress_net_return": long_stress / prices[0],
    }


def build_weekly_straddle_returns(
    options: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    option_frame = options.copy()
    option_frame["snapshot_time"] = pd.to_datetime(option_frame["snapshot_time"], utc=True)
    option_frame["expiration_time"] = pd.to_datetime(
        option_frame["expiration_time"], utc=True
    )
    feature_frame = features.copy()
    feature_frame["snapshot_time"] = pd.to_datetime(feature_frame["snapshot_time"], utc=True)
    feature_index = feature_frame.set_index("snapshot_time")
    grouped = {timestamp: frame for timestamp, frame in option_frame.groupby("snapshot_time")}
    rows: list[dict[str, object]] = []
    for entry_time in sorted(grouped):
        path_times = [entry_time + pd.Timedelta(days=offset) for offset in range(8)]
        if any(timestamp not in grouped or timestamp not in feature_index.index for timestamp in path_times):
            continue
        entry_btc = float(feature_index.loc[entry_time, "btc_price"])
        entry_pair = select_atm_straddle(grouped[entry_time], entry_btc)
        if entry_pair.empty:
            continue
        symbols = set(entry_pair["symbol"].astype(str))
        pairs: list[pd.DataFrame] = []
        prices: list[float] = []
        path_valid = True
        for position, timestamp in enumerate(path_times):
            snapshot = _valid_quote_rows(grouped[timestamp])
            pair = snapshot[snapshot["symbol"].astype(str).isin(symbols)].copy()
            pair = pair.sort_values("option_type").drop_duplicates("option_type", keep="last")
            needs_delta = position < 7
            deltas_valid = pd.to_numeric(pair.get("delta"), errors="coerce").notna().all()
            if set(pair["option_type"]) != {"C", "P"} or (needs_delta and not deltas_valid):
                path_valid = False
                break
            pairs.append(pair.reset_index(drop=True))
            prices.append(float(feature_index.loc[timestamp, "btc_price"]))
        if not path_valid:
            continue
        trade = calculate_weekly_straddle_trade(pairs, prices)
        feature = feature_index.loc[entry_time]
        entry_atm_mark_iv = float(
            pd.to_numeric(entry_pair["mark_iv"], errors="coerce").mean()
        )
        annualized_btc_rv = float(feature["btc_realized_vol_24h"] * np.sqrt(365))
        call = entry_pair.set_index("option_type").loc["C"]
        put = entry_pair.set_index("option_type").loc["P"]
        rows.append(
            {
                "entry_time": entry_time,
                "exit_time": path_times[-1],
                "period": feature["period"],
                "expiration_time": call["expiration_time"],
                "dte": float(call["dte"]),
                "strike_price": float(call["strike_price"]),
                "call_symbol": call["symbol"],
                "put_symbol": put["symbol"],
                "entry_btc": prices[0],
                "exit_btc": prices[-1],
                "entry_atm_mark_iv": entry_atm_mark_iv,
                "annualized_btc_realized_vol": annualized_btc_rv,
                "iv_rv_spread": entry_atm_mark_iv - annualized_btc_rv,
                "front_gap": float(feature["front_gap"]),
                "alt_high_vol_breadth": float(feature["alt_high_vol_breadth"]),
                **trade,
            }
        )
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def _apply_nonoverlap(frame: pd.DataFrame, condition: pd.Series) -> pd.DataFrame:
    selected: list[int] = []
    next_available: pd.Timestamp | None = None
    for index, row in frame.loc[condition].sort_values("entry_time").iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        if next_available is None or entry_time >= next_available:
            selected.append(index)
            next_available = pd.Timestamp(row["exit_time"])
    return frame.loc[selected].sort_values("entry_time").reset_index(drop=True)


def _bootstrap_mean(values: np.ndarray, draws: int = 5_000, seed: int = 16_900) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _tail_metrics(values: pd.Series) -> tuple[float, float]:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    if ordered.empty:
        return np.nan, np.nan
    count = max(1, int(np.ceil(len(ordered) * 0.05)))
    return float(ordered.iloc[0]), float(ordered.iloc[:count].mean())


def _positive_concentration(frame: pd.DataFrame) -> tuple[float, float]:
    positive = frame[frame["short_primary_net_return"].gt(0)].copy()
    total = float(positive["short_primary_net_return"].sum())
    if total <= 0:
        return float("inf"), float("inf")
    month = positive["entry_time"].dt.strftime("%Y-%m")
    monthly = positive.assign(month=month).groupby("month")["short_primary_net_return"].sum()
    return float(monthly.max() / total), float(positive["short_primary_net_return"].max() / total)


def _summary(frame: pd.DataFrame, scope: str) -> dict[str, object]:
    return {
        "scope": scope,
        "trades": len(frame),
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
            float(frame["long_primary_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "win_rate": (
            float(frame["short_primary_net_return"].gt(0).mean()) if len(frame) else np.nan
        ),
        "mean_hedge_turnover_x_btc": (
            float((frame["hedge_turnover_notional"] / frame["entry_btc"]).mean())
            if len(frame)
            else np.nan
        ),
    }


def evaluate_v169(
    weekly: pd.DataFrame,
    features: pd.DataFrame,
    circular_draws: int = 2_000,
    circular_seed: int = 16_901,
) -> dict[str, pd.DataFrame]:
    frame = weekly.copy().sort_values("entry_time").reset_index(drop=True)
    frame["rich_iv"] = frame["iv_rv_spread"].ge(0.10)
    frame["quiet_front"] = frame["front_gap"].le(0) & frame["alt_high_vol_breadth"].le(1 / 3)
    candidate = _apply_nonoverlap(frame, frame["rich_iv"] & frame["quiet_front"])
    iv_only = _apply_nonoverlap(frame, frame["rich_iv"])

    feature_frame = features.copy()
    feature_frame["snapshot_time"] = pd.to_datetime(feature_frame["snapshot_time"], utc=True)
    feature_quiet = feature_frame["front_gap"].le(0) & feature_frame[
        "alt_high_vol_breadth"
    ].le(1 / 3)
    delayed_times = set(
        feature_frame.loc[feature_quiet, "snapshot_time"] + pd.Timedelta(days=1)
    )
    delayed_condition = frame["rich_iv"] & frame["entry_time"].isin(delayed_times)
    delayed = _apply_nonoverlap(frame, delayed_condition)

    period_rows = [_summary(candidate, "all")]
    for period in ("development", "validation", "holdout"):
        period_rows.append(_summary(candidate[candidate["period"].eq(period)], period))
    period_summary = pd.DataFrame(period_rows)
    by_period = period_summary.set_index("scope")

    bootstrap_low = np.nan
    bootstrap_high = np.nan
    if len(candidate):
        bootstrap_low, bootstrap_high = _bootstrap_mean(
            candidate["short_primary_net_return"].to_numpy(dtype=float)
        )
    worst_trade, expected_shortfall = _tail_metrics(candidate["short_primary_net_return"])
    month_concentration, trade_concentration = _positive_concentration(candidate)

    rich = frame["rich_iv"].to_numpy(dtype=bool)
    quiet = frame["quiet_front"].to_numpy(dtype=bool)
    rng = np.random.default_rng(circular_seed)
    shifts = rng.integers(1, len(frame), size=circular_draws)
    circular_rows = []
    for draw, shift in enumerate(shifts):
        condition = pd.Series(rich & np.roll(quiet, int(shift)), index=frame.index)
        selected = _apply_nonoverlap(frame, condition)
        circular_rows.append(
            {
                "draw": draw,
                "shift": int(shift),
                "trades": len(selected),
                "mean_short_primary_net_return": float(
                    selected["short_primary_net_return"].mean()
                )
                if len(selected)
                else np.nan,
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
    long_mean = (
        float(candidate["long_primary_net_return"].mean()) if len(candidate) else np.nan
    )
    gates = {
        "trades_10": len(candidate) >= 10,
        "validation_trades_2": int(by_period.loc["validation", "trades"]) >= 2,
        "holdout_trades_2": int(by_period.loc["holdout", "trades"]) >= 2,
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
        "identical_long_negative": long_mean < 0,
        "worst_trade_above_minus_1000bp": worst_trade >= -0.10,
        "expected_shortfall_above_minus_600bp": expected_shortfall >= -0.06,
        "month_concentration_50": month_concentration <= 0.50,
        "trade_concentration_50": trade_concentration <= 0.50,
    }
    gate_frame = pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in gates.items()]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": "OVS2_WEEKLY_RICH_IV_QUIET_FRONT_SHORT_STRADDLE",
                "eligible_weekly_paths": len(frame),
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
                "identical_long_primary_net_bp": long_mean * 10_000,
                "worst_trade_bp": worst_trade * 10_000,
                "expected_shortfall_5_bp": expected_shortfall * 10_000,
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
        "promote_weekly_options_followup_audit"
        if bool(outcome["promote_research_followup"])
        else "reject_weekly_quiet_front_short_straddle"
    )
    text = [
        "# v16.9 Weekly Delta-Hedged Quiet-Front Short-Straddle Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        results["outcome"].to_markdown(index=False, floatfmt=".4f"),
        "",
        results["period_summary"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "All eight option snapshots are exact archived quotes. Entry sells at bids,",
        "day-seven exit buys at asks, and every daily BTC delta adjustment pays turnover",
        "cost. This adaptive short 2023 study changes no PaperLive, leverage, remote,",
        "or real-order permission.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v169_weekly_delta_hedged_quiet_front_short_straddle(
    option_path: Path = OPTION_PATH,
    feature_path: Path = FEATURE_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    options = pd.read_parquet(option_path)
    features = pd.read_parquet(feature_path)
    weekly = build_weekly_straddle_returns(options, features)
    results = evaluate_v169(weekly, features)
    root = ensure_dir(report_root)
    outputs = {
        "all_weekly_paths": root / "all_eligible_weekly_paths.parquet",
        "candidate_trades": root / "candidate_trades.parquet",
        "iv_only_trades": root / "iv_only_trades.parquet",
        "delayed_trades": root / "delayed_trades.parquet",
        "period_summary": root / "period_summary.csv",
        "circular_controls": root / "circular_controls.parquet",
        "gates": root / "gates.csv",
        "outcome": root / "summary.csv",
        "findings": findings_path,
    }
    weekly.to_parquet(outputs["all_weekly_paths"], index=False)
    results["candidate"].to_parquet(outputs["candidate_trades"], index=False)
    results["iv_only"].to_parquet(outputs["iv_only_trades"], index=False)
    results["delayed"].to_parquet(outputs["delayed_trades"], index=False)
    results["period_summary"].to_csv(outputs["period_summary"], index=False)
    results["circular_controls"].to_parquet(outputs["circular_controls"], index=False)
    results["gates"].to_csv(outputs["gates"], index=False)
    results["outcome"].to_csv(outputs["outcome"], index=False)
    _write_findings(results, findings_path)
    return outputs
