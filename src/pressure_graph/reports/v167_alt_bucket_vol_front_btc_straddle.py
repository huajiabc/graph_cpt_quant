"""Alt-bucket volatility front monetized with executable BTC straddles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


DATA_ROOT = Path("data/external/binance_option_vol_front")
OPTION_PATH = DATA_ROOT / "option_eoh_hour0.parquet"
PRICE_ROOT = DATA_ROOT / "um_klines_1h"
REPORT_ROOT = Path("reports/v16_7_alt_bucket_vol_front_btc_straddle")
FINDINGS_PATH = Path("docs/v167_alt_bucket_vol_front_btc_straddle_findings_2026_07_16.md")

BTC_SYMBOL = "BTCUSDT"
ALT_SYMBOLS = (
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "BCHUSDT",
    "MATICUSDT",
)
ALL_SYMBOLS = (BTC_SYMBOL, *ALT_SYMBOLS)
LOOKBACK_DAYS = 30
MINIMUM_ALTS = 10
PRIMARY_OPTION_FEE_RATE = 0.0003
STRESS_OPTION_FEE_RATE = 0.0006
OPTION_FEE_PREMIUM_CAP = 0.10
PRIMARY_HEDGE_FEE_RATE_PER_SIDE = 0.0004
STRESS_HEDGE_FEE_RATE_PER_SIDE = 0.0008


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2023-08-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2023-09-15", tz="UTC"):
        return "validation"
    return "holdout"


def causal_percentile(values: pd.Series, lookback: int = LOOKBACK_DAYS) -> pd.Series:
    """Percentile of the current value against exactly the preceding observations."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    array = numeric.to_numpy(dtype=float)
    for position in range(lookback, len(array)):
        history = array[position - lookback : position]
        current = array[position]
        if np.isfinite(current) and np.isfinite(history).all():
            result.iloc[position] = float(np.mean(history <= current))
    return result


def load_hourly_close_panel(price_root: Path = PRICE_ROOT) -> pd.DataFrame:
    series = []
    for symbol in ALL_SYMBOLS:
        frame = pd.read_parquet(
            price_root / f"{symbol}.parquet",
            columns=["bar_open_time", "close"],
        )
        frame["price_time"] = pd.to_datetime(frame["bar_open_time"], utc=True) + pd.Timedelta(
            hours=1
        )
        values = pd.to_numeric(frame["close"], errors="coerce")
        series.append(pd.Series(values.to_numpy(), index=frame["price_time"], name=symbol))
    return pd.concat(series, axis=1).sort_index()


def build_daily_volatility_features(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Build causal daily bucket states on the frozen 01:00 UTC grid."""
    close = close_panel.loc[:, ALL_SYMBOLS].astype(float).sort_index()
    hourly_log_return = np.log(close).diff()
    realized_24h = hourly_log_return.pow(2).rolling(24, min_periods=24).sum().pow(0.5)
    daily = realized_24h[realized_24h.index.hour == 1].copy()
    percentiles = daily.apply(causal_percentile)
    valid_alt_count = percentiles.loc[:, ALT_SYMBOLS].notna().sum(axis=1)
    alt_high_breadth = percentiles.loc[:, ALT_SYMBOLS].ge(0.80).sum(axis=1) / valid_alt_count
    alt_bucket_percentile = percentiles.loc[:, ALT_SYMBOLS].median(axis=1, skipna=True)
    btc_percentile = percentiles[BTC_SYMBOL]
    front_gap = alt_bucket_percentile - btc_percentile
    active = (
        valid_alt_count.ge(MINIMUM_ALTS)
        & alt_high_breadth.ge(1 / 3)
        & btc_percentile.le(0.70)
        & front_gap.ge(0.25)
    )
    entry = active & ~active.shift(1, fill_value=False)
    result = pd.DataFrame(
        {
            "snapshot_time": daily.index,
            "btc_price": close.reindex(daily.index)[BTC_SYMBOL].to_numpy(),
            "btc_realized_vol_24h": daily[BTC_SYMBOL].to_numpy(),
            "btc_vol_percentile": btc_percentile.to_numpy(),
            "valid_alt_count": valid_alt_count.to_numpy(),
            "alt_high_vol_breadth": alt_high_breadth.to_numpy(),
            "alt_bucket_vol_percentile": alt_bucket_percentile.to_numpy(),
            "front_gap": front_gap.to_numpy(),
            "front_active": active.to_numpy(),
            "candidate_entry": entry.to_numpy(),
        }
    )
    result["period"] = result["snapshot_time"].map(_period)
    return result


def _valid_quote_rows(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = (
        frame["best_bid_price"].gt(0)
        & frame["best_ask_price"].gt(0)
        & frame["best_ask_price"].ge(frame["best_bid_price"])
        & frame["best_bid_qty"].gt(0)
        & frame["best_ask_qty"].gt(0)
    )
    return frame[numeric].copy()


def select_atm_straddle(
    snapshot: pd.DataFrame,
    btc_price: float,
    target_dte: float = 30.0,
) -> pd.DataFrame:
    """Select one deterministic executable call-put pair from an EOH snapshot."""
    local = _valid_quote_rows(snapshot)
    local["dte"] = (
        pd.to_datetime(local["expiration_time"], utc=True)
        - pd.to_datetime(local["snapshot_time"], utc=True)
    ).dt.total_seconds() / 86_400
    local = local[local["dte"].between(21, 45)].copy()
    if local.empty:
        return pd.DataFrame()
    expiry_table = local[["expiration_time", "dte"]].drop_duplicates()
    expiry_table["target_gap"] = (expiry_table["dte"] - target_dte).abs()
    chosen_expiry = expiry_table.sort_values(["target_gap", "expiration_time"]).iloc[0][
        "expiration_time"
    ]
    local = local[local["expiration_time"].eq(chosen_expiry)].copy()
    pair_counts = local.groupby("strike_price")["option_type"].nunique()
    paired_strikes = pair_counts[pair_counts.eq(2)].index.to_numpy(dtype=float)
    if not len(paired_strikes):
        return pd.DataFrame()
    chosen_strike = sorted(paired_strikes, key=lambda strike: (abs(strike - btc_price), strike))[0]
    pair = local[local["strike_price"].eq(chosen_strike)].copy()
    pair = pair.sort_values("option_type").drop_duplicates("option_type", keep="last")
    if set(pair["option_type"]) != {"C", "P"}:
        return pd.DataFrame()
    return pair.reset_index(drop=True)


def _option_fee(rate: float, spot: float, premium: float) -> float:
    return min(rate * spot, OPTION_FEE_PREMIUM_CAP * premium)


def calculate_straddle_trade(
    entry_pair: pd.DataFrame,
    exit_pair: pd.DataFrame,
    entry_btc: float,
    exit_btc: float,
) -> dict[str, float]:
    """Calculate bid/ask-crossed, statically delta-hedged one-day PnL."""
    entry = entry_pair.set_index("option_type")
    exit_ = exit_pair.set_index("option_type")
    entry_premium = float(entry.loc[["C", "P"], "best_ask_price"].sum())
    exit_value = float(exit_.loc[["C", "P"], "best_bid_price"].sum())
    option_pnl = exit_value - entry_premium
    entry_delta = float(entry.loc[["C", "P"], "delta"].sum())
    hedge_quantity = -entry_delta
    hedge_pnl = hedge_quantity * (exit_btc - entry_btc)
    gross_pnl = option_pnl + hedge_pnl
    primary_option_fees = sum(
        _option_fee(PRIMARY_OPTION_FEE_RATE, entry_btc, float(entry.loc[side, "best_ask_price"]))
        + _option_fee(PRIMARY_OPTION_FEE_RATE, exit_btc, float(exit_.loc[side, "best_bid_price"]))
        for side in ("C", "P")
    )
    stress_option_fees = sum(
        _option_fee(STRESS_OPTION_FEE_RATE, entry_btc, float(entry.loc[side, "best_ask_price"]))
        + _option_fee(STRESS_OPTION_FEE_RATE, exit_btc, float(exit_.loc[side, "best_bid_price"]))
        for side in ("C", "P")
    )
    primary_hedge_fees = PRIMARY_HEDGE_FEE_RATE_PER_SIDE * abs(hedge_quantity) * (
        entry_btc + exit_btc
    )
    stress_hedge_fees = STRESS_HEDGE_FEE_RATE_PER_SIDE * abs(hedge_quantity) * (
        entry_btc + exit_btc
    )
    primary_pnl = gross_pnl - primary_option_fees - primary_hedge_fees
    stress_pnl = gross_pnl - stress_option_fees - stress_hedge_fees
    return {
        "entry_premium": entry_premium,
        "exit_value": exit_value,
        "entry_delta": entry_delta,
        "hedge_quantity": hedge_quantity,
        "option_pnl": option_pnl,
        "hedge_pnl": hedge_pnl,
        "gross_return": gross_pnl / entry_btc,
        "unhedged_option_return": option_pnl / entry_btc,
        "primary_fee_return": (primary_option_fees + primary_hedge_fees) / entry_btc,
        "stress_fee_return": (stress_option_fees + stress_hedge_fees) / entry_btc,
        "primary_net_return": primary_pnl / entry_btc,
        "stress_net_return": stress_pnl / entry_btc,
        "primary_premium_return": primary_pnl / entry_premium,
    }


def calculate_short_straddle_trade(
    entry_pair: pd.DataFrame,
    exit_pair: pd.DataFrame,
    entry_btc: float,
    exit_btc: float,
) -> dict[str, float]:
    """Calculate an independently executable short straddle, never negative long PnL."""
    entry = entry_pair.set_index("option_type")
    exit_ = exit_pair.set_index("option_type")
    entry_credit = float(entry.loc[["C", "P"], "best_bid_price"].sum())
    exit_cost = float(exit_.loc[["C", "P"], "best_ask_price"].sum())
    option_pnl = entry_credit - exit_cost
    entry_option_delta = float(entry.loc[["C", "P"], "delta"].sum())
    hedge_quantity = entry_option_delta
    hedge_pnl = hedge_quantity * (exit_btc - entry_btc)
    gross_pnl = option_pnl + hedge_pnl
    primary_option_fees = sum(
        _option_fee(PRIMARY_OPTION_FEE_RATE, entry_btc, float(entry.loc[side, "best_bid_price"]))
        + _option_fee(PRIMARY_OPTION_FEE_RATE, exit_btc, float(exit_.loc[side, "best_ask_price"]))
        for side in ("C", "P")
    )
    stress_option_fees = sum(
        _option_fee(STRESS_OPTION_FEE_RATE, entry_btc, float(entry.loc[side, "best_bid_price"]))
        + _option_fee(STRESS_OPTION_FEE_RATE, exit_btc, float(exit_.loc[side, "best_ask_price"]))
        for side in ("C", "P")
    )
    primary_hedge_fees = PRIMARY_HEDGE_FEE_RATE_PER_SIDE * abs(hedge_quantity) * (
        entry_btc + exit_btc
    )
    stress_hedge_fees = STRESS_HEDGE_FEE_RATE_PER_SIDE * abs(hedge_quantity) * (
        entry_btc + exit_btc
    )
    primary_pnl = gross_pnl - primary_option_fees - primary_hedge_fees
    stress_pnl = gross_pnl - stress_option_fees - stress_hedge_fees
    return {
        "short_entry_credit": entry_credit,
        "short_exit_cost": exit_cost,
        "short_entry_option_delta": entry_option_delta,
        "short_hedge_quantity": hedge_quantity,
        "short_option_pnl": option_pnl,
        "short_hedge_pnl": hedge_pnl,
        "short_gross_return": gross_pnl / entry_btc,
        "short_primary_fee_return": (primary_option_fees + primary_hedge_fees) / entry_btc,
        "short_stress_fee_return": (stress_option_fees + stress_hedge_fees) / entry_btc,
        "short_primary_net_return": primary_pnl / entry_btc,
        "short_stress_net_return": stress_pnl / entry_btc,
        "short_primary_credit_return": primary_pnl / entry_credit,
    }


def build_daily_straddle_returns(
    options: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Construct every eligible non-stale daily straddle before signal filtering."""
    option_frame = options.copy()
    for column in ("snapshot_time", "expiration_time"):
        option_frame[column] = pd.to_datetime(option_frame[column], utc=True)
    features = feature_frame.set_index("snapshot_time")
    grouped = {timestamp: frame for timestamp, frame in option_frame.groupby("snapshot_time")}
    rows: list[dict[str, object]] = []
    for entry_time in sorted(grouped):
        exit_time = entry_time + pd.Timedelta(days=1)
        if entry_time not in features.index or exit_time not in features.index:
            continue
        if exit_time not in grouped:
            continue
        entry_btc = float(features.loc[entry_time, "btc_price"])
        exit_btc = float(features.loc[exit_time, "btc_price"])
        if not (np.isfinite(entry_btc) and np.isfinite(exit_btc) and entry_btc > 0):
            continue
        entry_pair = select_atm_straddle(grouped[entry_time], entry_btc)
        if entry_pair.empty:
            continue
        symbols = set(entry_pair["symbol"].astype(str))
        exit_snapshot = _valid_quote_rows(grouped[exit_time])
        exit_pair = exit_snapshot[exit_snapshot["symbol"].astype(str).isin(symbols)].copy()
        exit_pair = exit_pair.sort_values("option_type").drop_duplicates("option_type", keep="last")
        if set(exit_pair["option_type"]) != {"C", "P"}:
            continue
        if not pd.to_numeric(entry_pair["delta"], errors="coerce").notna().all():
            continue
        trade = calculate_straddle_trade(entry_pair, exit_pair, entry_btc, exit_btc)
        short_trade = calculate_short_straddle_trade(
            entry_pair, exit_pair, entry_btc, exit_btc
        )
        call = entry_pair.set_index("option_type").loc["C"]
        put = entry_pair.set_index("option_type").loc["P"]
        feature = features.loc[entry_time]
        entry_atm_mark_iv = float(
            pd.to_numeric(entry_pair["mark_iv"], errors="coerce").mean()
        )
        annualized_btc_realized_vol = float(feature["btc_realized_vol_24h"] * np.sqrt(365))
        rows.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "period": feature["period"],
                "expiration_time": call["expiration_time"],
                "dte": float(call["dte"]),
                "strike_price": float(call["strike_price"]),
                "call_symbol": call["symbol"],
                "put_symbol": put["symbol"],
                "entry_btc": entry_btc,
                "exit_btc": exit_btc,
                "btc_vol_percentile": float(feature["btc_vol_percentile"]),
                "alt_high_vol_breadth": float(feature["alt_high_vol_breadth"]),
                "alt_bucket_vol_percentile": float(feature["alt_bucket_vol_percentile"]),
                "front_gap": float(feature["front_gap"]),
                "front_active": bool(feature["front_active"]),
                "candidate_entry": bool(feature["candidate_entry"]),
                "entry_atm_mark_iv": entry_atm_mark_iv,
                "annualized_btc_realized_vol": annualized_btc_realized_vol,
                "iv_rv_spread": entry_atm_mark_iv - annualized_btc_realized_vol,
                **trade,
                **short_trade,
            }
        )
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def _bootstrap_mean(values: np.ndarray, draws: int = 5_000, seed: int = 16_700) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _positive_concentration(frame: pd.DataFrame) -> tuple[float, float]:
    positive = frame[frame["primary_net_return"].gt(0)].copy()
    total = float(positive["primary_net_return"].sum())
    if total <= 0:
        return float("inf"), float("inf")
    month = positive.assign(month=positive["entry_time"].dt.to_period("M")).groupby("month")[
        "primary_net_return"
    ].sum()
    return float(month.max() / total), float(positive["primary_net_return"].max() / total)


def _period_summary(frame: pd.DataFrame, scope: str) -> dict[str, object]:
    return {
        "scope": scope,
        "trades": len(frame),
        "active_days": int(frame["entry_time"].dt.floor("D").nunique()) if len(frame) else 0,
        "mean_gross_bp": float(frame["gross_return"].mean() * 10_000) if len(frame) else np.nan,
        "mean_primary_net_bp": (
            float(frame["primary_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "mean_stress_net_bp": (
            float(frame["stress_net_return"].mean() * 10_000) if len(frame) else np.nan
        ),
        "mean_premium_return_pct": (
            float(frame["primary_premium_return"].mean() * 100) if len(frame) else np.nan
        ),
        "win_rate": float(frame["primary_net_return"].gt(0).mean()) if len(frame) else np.nan,
    }


def evaluate_v167(
    feature_frame: pd.DataFrame,
    all_trades: pd.DataFrame,
    circular_draws: int = 2_000,
    circular_seed: int = 16_701,
) -> dict[str, pd.DataFrame]:
    candidate = all_trades[all_trades["candidate_entry"]].copy()
    compression = all_trades[all_trades["btc_vol_percentile"].le(0.70)].copy()
    feature_entries = feature_frame.set_index("snapshot_time")["candidate_entry"]
    delayed_times = set(feature_entries[feature_entries].index + pd.Timedelta(days=1))
    delayed = all_trades[all_trades["entry_time"].isin(delayed_times)].copy()

    period_rows = [_period_summary(candidate, "all")]
    for period in ("development", "validation", "holdout"):
        period_rows.append(_period_summary(candidate[candidate["period"].eq(period)], period))
    period_summary = pd.DataFrame(period_rows)

    bootstrap_low = np.nan
    bootstrap_high = np.nan
    if len(candidate):
        bootstrap_low, bootstrap_high = _bootstrap_mean(
            candidate["primary_net_return"].to_numpy(dtype=float)
        )
    month_concentration, trade_concentration = _positive_concentration(candidate)

    event_mask = all_trades["candidate_entry"].to_numpy(dtype=bool)
    returns = all_trades["primary_net_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(circular_seed)
    shifts = rng.integers(1, len(all_trades), size=circular_draws)
    control_rows = []
    for draw, shift in enumerate(shifts):
        shifted = np.roll(event_mask, int(shift))
        values = returns[shifted]
        control_rows.append(
            {
                "draw": draw,
                "shift": int(shift),
                "trades": len(values),
                "mean_primary_net_return": float(values.mean()) if len(values) else np.nan,
            }
        )
    circular = pd.DataFrame(control_rows)
    finite_controls = circular["mean_primary_net_return"].dropna()
    real_mean = float(candidate["primary_net_return"].mean()) if len(candidate) else np.nan
    circular_percentile = (
        float((finite_controls <= real_mean).mean()) if len(finite_controls) else np.nan
    )
    delayed_mean = float(delayed["primary_net_return"].mean()) if len(delayed) else np.nan
    compression_mean = (
        float(compression["primary_net_return"].mean()) if len(compression) else np.nan
    )

    by_period = period_summary.set_index("scope")
    gates = {
        "trades_20": len(candidate) >= 20,
        "validation_trades_5": int(by_period.loc["validation", "trades"]) >= 5,
        "holdout_trades_5": int(by_period.loc["holdout", "trades"]) >= 5,
        "development_primary_positive": by_period.loc["development", "mean_primary_net_bp"] > 0,
        "validation_primary_positive": by_period.loc["validation", "mean_primary_net_bp"] > 0,
        "holdout_primary_positive": by_period.loc["holdout", "mean_primary_net_bp"] > 0,
        "full_stress_positive": by_period.loc["all", "mean_stress_net_bp"] > 0,
        "bootstrap_lower_positive": bootstrap_low > 0,
        "circular_percentile_95": circular_percentile >= 0.95,
        "beats_delayed": real_mean > delayed_mean,
        "beats_btc_compression": real_mean > compression_mean,
        "month_concentration_50": month_concentration <= 0.50,
        "trade_concentration_35": trade_concentration <= 0.35,
    }
    gate_frame = pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in gates.items()]
    )
    outcome = pd.DataFrame(
        [
            {
                "candidate": "OVT1_ALT_VOL_FRONT_LONG_BTC_STRADDLE",
                "eligible_daily_straddles": len(all_trades),
                "candidate_trades": len(candidate),
                "candidate_primary_net_bp": real_mean * 10_000,
                "candidate_stress_net_bp": float(candidate["stress_net_return"].mean() * 10_000)
                if len(candidate)
                else np.nan,
                "bootstrap_95_low_bp": bootstrap_low * 10_000,
                "bootstrap_95_high_bp": bootstrap_high * 10_000,
                "circular_percentile": circular_percentile,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "btc_compression_primary_net_bp": compression_mean * 10_000,
                "positive_month_concentration": month_concentration,
                "positive_trade_concentration": trade_concentration,
                "promote_research_followup": bool(all(gates.values())),
                "failed_gates": "|".join(gate for gate, passed in gates.items() if not passed),
            }
        ]
    )
    return {
        "candidate": candidate,
        "compression": compression,
        "delayed": delayed,
        "period_summary": period_summary,
        "circular_controls": circular,
        "gates": gate_frame,
        "outcome": outcome,
    }


def _write_findings(results: dict[str, pd.DataFrame], path: Path) -> None:
    outcome = results["outcome"].iloc[0]
    verdict = (
        "promote_longer_options_history_research"
        if bool(outcome["promote_research_followup"])
        else "reject_frozen_alt_front_straddle"
    )
    text = [
        "# v16.7 Alt-Bucket Volatility Front -> BTC Straddle Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        results["outcome"].to_markdown(index=False, floatfmt=".4f"),
        "",
        results["period_summary"].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Every option entry uses the archived ask and every exit uses the archived bid.",
        "The static BTC delta hedge and frozen primary/stress fee schedules are included.",
        "Missing option days were not filled. This short 2023 archive cannot authorize",
        "PaperLive, leverage, remote changes, or real orders.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v167_alt_bucket_vol_front_btc_straddle(
    option_path: Path = OPTION_PATH,
    price_root: Path = PRICE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    close_panel = load_hourly_close_panel(price_root)
    features = build_daily_volatility_features(close_panel)
    options = pd.read_parquet(option_path)
    all_trades = build_daily_straddle_returns(options, features)
    results = evaluate_v167(features, all_trades)
    root = ensure_dir(report_root)
    outputs = {
        "daily_features": root / "daily_volatility_features.parquet",
        "all_straddles": root / "all_eligible_daily_straddles.parquet",
        "candidate_trades": root / "candidate_trades.parquet",
        "period_summary": root / "period_summary.csv",
        "circular_controls": root / "circular_controls.parquet",
        "gates": root / "gates.csv",
        "outcome": root / "summary.csv",
        "findings": findings_path,
    }
    features.to_parquet(outputs["daily_features"], index=False)
    all_trades.to_parquet(outputs["all_straddles"], index=False)
    results["candidate"].to_parquet(outputs["candidate_trades"], index=False)
    results["period_summary"].to_csv(outputs["period_summary"], index=False)
    results["circular_controls"].to_parquet(outputs["circular_controls"], index=False)
    results["gates"].to_csv(outputs["gates"], index=False)
    results["outcome"].to_csv(outputs["outcome"], index=False)
    _write_findings(results, findings_path)
    return outputs
