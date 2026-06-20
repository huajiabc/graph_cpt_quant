"""v5.1 Funding Extreme + OI Low RV Audit.

This report audits the only v5.0 crowding pocket with a thin positive
short/BTC relative-value edge.  It is an audit candidate, not an alpha
candidate: price PnL, funding-carry proxy, hedge choice, timing, concentration,
matched random baselines, and cost stress are separated explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v50_perp_crowding_atlas import (
    DEFAULT_FEATURE_PATH,
    UNIVERSE_COL,
    _available_columns,
    _bool,
    _max_contribution,
    _month_cap35,
    _num,
    _pct01,
)


REPORT_ROOT = Path("reports/v5_1_funding_extreme_oi_low_rv_audit")


@dataclass(frozen=True)
class V51Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = DEFAULT_FEATURE_PATH
    universe_col: str = UNIVERSE_COL
    funding_extreme: float = 0.90
    oi_low: float = 0.20
    cost_bps: float = 20.0
    random_seed: int = 51


FEATURE_COLUMNS = (
    "symbol",
    "feature_time",
    "bar_close_time",
    "close",
    "warmup_complete",
    UNIVERSE_COL,
    "universe_static_current_top30",
    "funding_time",
    "funding_rate_settled",
    "funding_interval_minutes",
    "funding_age_minutes",
    "funding_percentile",
    "oi_value_delta_4h_percentile",
    "oi_delta_4h_percentile",
    "ret_4h",
    "ret_4h_percentile",
    "btc_market_state",
    "btc_ret_4h",
    "future_ret_4h",
    "future_ret_12h",
)

HORIZONS = {"4h": 4.0, "12h": 12.0, "24h": 24.0}
HEDGE_LEGS = ("BTC", "ETH", "BTC_ETH", "MARKET")


def _read_feature_subset(cfg: V51Config) -> pd.DataFrame:
    if not cfg.feature_path.exists():
        return pd.DataFrame()
    cols = _available_columns(cfg.feature_path, tuple(dict.fromkeys([*FEATURE_COLUMNS, cfg.universe_col])), cfg.universe_col)
    pf = pq.ParquetFile(cfg.feature_path)
    frames: list[pd.DataFrame] = []
    for idx in range(pf.num_row_groups):
        chunk = pf.read_row_group(idx, columns=cols).to_pandas()
        if cfg.universe_col in chunk.columns:
            chunk = chunk[_bool(chunk, cfg.universe_col)].copy()
        elif "universe_static_current_top30" in chunk.columns:
            chunk = chunk[_bool(chunk, "universe_static_current_top30")].copy()
        if "warmup_complete" in chunk.columns:
            chunk = chunk[_bool(chunk, "warmup_complete", True)].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["feature_time"] = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    out["bar_close_time"] = pd.to_datetime(out.get("bar_close_time", out["feature_time"]), utc=True, errors="coerce")
    out["funding_time"] = pd.to_datetime(out.get("funding_time"), utc=True, errors="coerce")
    out = out.dropna(subset=["symbol", "feature_time"]).sort_values(["symbol", "feature_time"]).reset_index(drop=True)
    return out


def _add_future_24h(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "close" not in frame.columns:
        frame["future_ret_24h"] = np.nan
        return frame
    out = frame.sort_values(["symbol", "feature_time"]).copy()
    future_close = out.groupby("symbol", sort=False)["close"].shift(-96)
    future_time = out.groupby("symbol", sort=False)["feature_time"].shift(-96)
    valid = (pd.to_datetime(future_time, utc=True, errors="coerce") - out["feature_time"]).eq(pd.Timedelta(hours=24))
    out["future_ret_24h"] = np.where(valid, pd.to_numeric(future_close, errors="coerce") / _num(out, "close") - 1.0, np.nan)
    return out


def _funding_carry(frame: pd.DataFrame, horizon_hours: float) -> pd.Series:
    interval_hours = (_num(frame, "funding_interval_minutes", 480.0) / 60.0).replace(0, np.nan).fillna(8.0)
    return _num(frame, "funding_rate_settled", 0.0).fillna(0.0) * float(horizon_hours) / interval_hours


def _add_candidate_context(frame: pd.DataFrame, cfg: V51Config) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = _add_future_24h(frame)
    out["month"] = out["feature_time"].dt.strftime("%Y-%m")
    out["funding_pct01"] = _pct01(_num(out, "funding_percentile"))
    out["oi_pct01"] = _pct01(_num(out, "oi_value_delta_4h_percentile").combine_first(_num(out, "oi_delta_4h_percentile")))
    out["ret_4h_pct01"] = _pct01(_num(out, "ret_4h_percentile"))
    out["candidate_funding_extreme_oi_low"] = out["funding_pct01"].ge(cfg.funding_extreme) & out["oi_pct01"].le(cfg.oi_low)
    out["funding_bucket"] = np.select(
        [out["funding_pct01"].ge(0.90), out["funding_pct01"].ge(0.80), out["funding_pct01"].le(0.20)],
        ["funding_extreme", "funding_high", "funding_low"],
        default="funding_mid",
    )
    out["oi_bucket"] = np.select(
        [out["oi_pct01"].ge(0.90), out["oi_pct01"].ge(0.80), out["oi_pct01"].le(0.20)],
        ["oi_extreme", "oi_high", "oi_low"],
        default="oi_mid",
    )
    out["ret_bucket"] = np.select(
        [out["ret_4h_pct01"].ge(0.75), out["ret_4h_pct01"].le(0.25)],
        ["ret_high", "ret_low"],
        default="ret_mid",
    )
    age = _num(out, "funding_age_minutes")
    interval = _num(out, "funding_interval_minutes", 480.0).replace(0, np.nan).fillna(480.0)
    age_mod = np.mod(age.fillna(np.nan), interval)
    time_to_next = interval - age_mod
    out["funding_age_mod_minutes"] = age_mod
    out["time_to_next_funding_minutes"] = time_to_next
    out["funding_timing_bucket"] = np.select(
        [
            time_to_next.le(60),
            time_to_next.le(240),
            age_mod.le(60),
            age_mod.le(240),
        ],
        ["pre_funding_0_1h", "pre_funding_1_4h", "post_funding_0_1h", "post_funding_1_4h"],
        default="mid_cycle",
    )
    return out


def _hedge_frame(frame: pd.DataFrame, symbol: str, prefix: str) -> pd.DataFrame:
    local = frame[frame["symbol"].astype(str).eq(symbol)].copy()
    cols = ["feature_time", "funding_rate_settled", "funding_interval_minutes", *[f"future_ret_{h}" for h in HORIZONS]]
    local = local[[col for col in cols if col in local.columns]].copy()
    return local.rename(columns={col: f"{prefix}_{col}" for col in local.columns if col != "feature_time"})


def _add_hedge_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for prefix, symbol in [("btc", "BTCUSDT"), ("eth", "ETHUSDT")]:
        hedge = _hedge_frame(out, symbol, prefix)
        out = out.merge(hedge, on="feature_time", how="left")
    market_cols = []
    for h in HORIZONS:
        carry_col = f"symbol_funding_carry_{h}"
        out[carry_col] = _funding_carry(out, HORIZONS[h])
        market_cols.append(carry_col)
    market = out.groupby("feature_time", sort=False).agg(
        **{f"market_future_ret_{h}": (f"future_ret_{h}", "mean") for h in HORIZONS},
        **{f"market_long_funding_cost_{h}": (f"symbol_funding_carry_{h}", "mean") for h in HORIZONS},
    ).reset_index()
    out = out.merge(market, on="feature_time", how="left")
    for h, hours in HORIZONS.items():
        out[f"btc_funding_carry_{h}"] = (
            _num(out, "btc_funding_rate_settled", 0.0).fillna(0.0)
            * hours
            / (_num(out, "btc_funding_interval_minutes", 480.0).replace(0, np.nan).fillna(480.0) / 60.0)
        )
        out[f"eth_funding_carry_{h}"] = (
            _num(out, "eth_funding_rate_settled", 0.0).fillna(0.0)
            * hours
            / (_num(out, "eth_funding_interval_minutes", 480.0).replace(0, np.nan).fillna(480.0) / 60.0)
        )
        out[f"btc_eth_future_ret_{h}"] = pd.concat([_num(out, f"btc_future_ret_{h}"), _num(out, f"eth_future_ret_{h}")], axis=1).mean(axis=1)
        out[f"btc_eth_funding_carry_{h}"] = pd.concat([_num(out, f"btc_funding_carry_{h}"), _num(out, f"eth_funding_carry_{h}")], axis=1).mean(axis=1)
    return out


def _pair_components(frame: pd.DataFrame, hedge: str, horizon: str, cost_bps: float) -> pd.DataFrame:
    out = frame.copy()
    if hedge == "BTC":
        hedge_ret = _num(out, f"btc_future_ret_{horizon}")
        hedge_carry = _num(out, f"btc_funding_carry_{horizon}")
    elif hedge == "ETH":
        hedge_ret = _num(out, f"eth_future_ret_{horizon}")
        hedge_carry = _num(out, f"eth_funding_carry_{horizon}")
    elif hedge == "BTC_ETH":
        hedge_ret = _num(out, f"btc_eth_future_ret_{horizon}")
        hedge_carry = _num(out, f"btc_eth_funding_carry_{horizon}")
    elif hedge == "MARKET":
        hedge_ret = _num(out, f"market_future_ret_{horizon}")
        hedge_carry = _num(out, f"market_long_funding_cost_{horizon}")
    else:
        raise ValueError(f"unsupported hedge leg: {hedge}")
    out["hedge_leg"] = hedge
    out["horizon"] = horizon
    out["pair_price_pnl"] = hedge_ret - _num(out, f"future_ret_{horizon}")
    out["funding_carry_pnl"] = _num(out, f"symbol_funding_carry_{horizon}") - hedge_carry
    out["total_pair_gross_pnl"] = out["pair_price_pnl"] + out["funding_carry_pnl"]
    out["total_pair_net20_legacy_cost"] = out["total_pair_gross_pnl"] - 2.0 * cost_bps / 10_000.0
    out["total_pair_net20_pair_cost"] = out["total_pair_gross_pnl"] - 4.0 * cost_bps / 10_000.0
    return out


def _candidate_ledger(features: pd.DataFrame, cfg: V51Config) -> pd.DataFrame:
    candidates = features[features["candidate_funding_extreme_oi_low"].fillna(False)].copy()
    if candidates.empty:
        return candidates
    ledgers = []
    for hedge in HEDGE_LEGS:
        ledgers.append(_pair_components(candidates, hedge, "12h", cfg.cost_bps))
    ledger = pd.concat(ledgers, ignore_index=True)
    keep = [
        "symbol",
        "feature_time",
        "month",
        "hedge_leg",
        "horizon",
        "funding_rate_settled",
        "funding_pct01",
        "oi_pct01",
        "ret_4h_pct01",
        "btc_market_state",
        "funding_timing_bucket",
        "pair_price_pnl",
        "funding_carry_pnl",
        "total_pair_gross_pnl",
        "total_pair_net20_legacy_cost",
        "total_pair_net20_pair_cost",
    ]
    return ledger[[col for col in keep if col in ledger.columns]].copy()


def _summary(frame: pd.DataFrame, value_col: str, label_cols: dict[str, object] | None = None) -> dict[str, object]:
    label_cols = label_cols or {}
    if frame.empty:
        return {
            **label_cols,
            "events": 0,
            "mean": np.nan,
            "hit_rate": np.nan,
            "month_cap35": np.nan,
            "worst_month": np.nan,
            "max_symbol_contribution": np.nan,
        }
    monthly = _num(frame, value_col).groupby(frame["month"]).mean()
    return {
        **label_cols,
        "events": int(len(frame)),
        "mean": float(_num(frame, value_col).mean()),
        "hit_rate": float(_num(frame, value_col).gt(0).mean()),
        "month_cap35": _month_cap35(frame, value_col),
        "worst_month": float(monthly.min()) if len(monthly) else np.nan,
        "max_symbol_contribution": _max_contribution(frame, "symbol", value_col),
    }


def _decomposition(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hedge, horizon), group in ledger.groupby(["hedge_leg", "horizon"], sort=False):
        row = {"hedge_leg": hedge, "horizon": horizon, "events": int(len(group))}
        for col in ["pair_price_pnl", "funding_carry_pnl", "total_pair_gross_pnl", "total_pair_net20_legacy_cost", "total_pair_net20_pair_cost"]:
            row[f"{col}_mean"] = float(_num(group, col).mean()) if len(group) else np.nan
            row[f"{col}_month_cap35"] = _month_cap35(group, col)
        rows.append(row)
    return pd.DataFrame(rows)


def _time_to_funding_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    btc = ledger[ledger["hedge_leg"].eq("BTC")].copy()
    rows = []
    for bucket, group in btc.groupby("funding_timing_bucket", sort=False, dropna=False):
        rows.append(_summary(group, "total_pair_net20_pair_cost", {"funding_timing_bucket": bucket}))
    return pd.DataFrame(rows)


def _hedge_leg_comparison(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hedge, group in ledger.groupby("hedge_leg", sort=False):
        rows.append(_summary(group, "total_pair_net20_pair_cost", {"hedge_leg": hedge}))
    return pd.DataFrame(rows)


def _month_symbol_stability(ledger: pd.DataFrame) -> pd.DataFrame:
    btc = ledger[ledger["hedge_leg"].eq("BTC")].copy()
    rows = []
    base = float(_num(btc, "total_pair_net20_pair_cost").mean()) if len(btc) else np.nan
    for month in sorted(btc["month"].dropna().astype(str).unique()):
        sample = btc[~btc["month"].astype(str).eq(month)]
        row = _summary(sample, "total_pair_net20_pair_cost", {"stability_type": "leave_one_month", "removed": month})
        row["delta_vs_full_mean"] = row["mean"] - base if np.isfinite(base) and np.isfinite(row["mean"]) else np.nan
        rows.append(row)
    top_symbols = (
        _num(btc, "total_pair_net20_pair_cost")
        .groupby(btc["symbol"].astype(str))
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(50)
        .index
        .tolist()
    )
    for symbol in top_symbols:
        sample = btc[~btc["symbol"].astype(str).eq(symbol)]
        row = _summary(sample, "total_pair_net20_pair_cost", {"stability_type": "leave_one_symbol", "removed": symbol})
        row["delta_vs_full_mean"] = row["mean"] - base if np.isfinite(base) and np.isfinite(row["mean"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _matched_random_baseline(features: pd.DataFrame, cfg: V51Config) -> pd.DataFrame:
    candidate = features[features["candidate_funding_extreme_oi_low"].fillna(False)].copy()
    if candidate.empty:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(cfg.random_seed)
    match_specs = {
        "same_month_btc_state_ret_bucket": ["month", "btc_market_state", "ret_bucket"],
        "same_funding_extreme": ["month", "funding_bucket"],
        "same_oi_low": ["month", "oi_bucket"],
        "same_month_only": ["month"],
    }
    for name, keys in match_specs.items():
        sampled = []
        pool = features[~features["candidate_funding_extreme_oi_low"].fillna(False)].copy()
        for key_values, group in candidate.groupby(keys, sort=False, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            mask = pd.Series(True, index=pool.index)
            for key, value in zip(keys, key_values):
                mask &= pool[key].astype(str).eq(str(value))
            eligible = pool[mask]
            if eligible.empty:
                continue
            take = min(len(group), len(eligible))
            sampled.append(eligible.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
        sample = pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()
        if not sample.empty:
            sample_ledger = _pair_components(sample, "BTC", "12h", cfg.cost_bps)
        else:
            sample_ledger = sample
        rows.append(_summary(sample_ledger, "total_pair_net20_pair_cost", {"baseline": name}))
    candidate_ledger = _pair_components(candidate, "BTC", "12h", cfg.cost_bps)
    rows.insert(0, _summary(candidate_ledger, "total_pair_net20_pair_cost", {"baseline": "real_funding_extreme_oi_low"}))
    return pd.DataFrame(rows)


def _cost_stress(features: pd.DataFrame) -> pd.DataFrame:
    candidate = features[features["candidate_funding_extreme_oi_low"].fillna(False)].copy()
    rows = []
    for hedge in HEDGE_LEGS:
        base = _pair_components(candidate, hedge, "12h", 0.0)
        for cost in [10.0, 20.0, 30.0, 50.0]:
            local = base.copy()
            local["net_legacy_cost"] = _num(local, "total_pair_gross_pnl") - 2.0 * cost / 10_000.0
            local["net_pair_cost"] = _num(local, "total_pair_gross_pnl") - 4.0 * cost / 10_000.0
            rows.append(
                {
                    "hedge_leg": hedge,
                    "cost_bps": cost,
                    "events": int(len(local)),
                    "total_pair_gross_pnl": float(_num(local, "total_pair_gross_pnl").mean()) if len(local) else np.nan,
                    "net_legacy_cost": float(_num(local, "net_legacy_cost").mean()) if len(local) else np.nan,
                    "net_pair_cost": float(_num(local, "net_pair_cost").mean()) if len(local) else np.nan,
                    "month_cap35_net_pair_cost": _month_cap35(local, "net_pair_cost"),
                }
            )
    return pd.DataFrame(rows)


def _write_notes(
    path: Path,
    decomposition: pd.DataFrame,
    hedge: pd.DataFrame,
    randoms: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    btc = decomposition[decomposition["hedge_leg"].eq("BTC")]
    btc_row = btc.iloc[0] if not btc.empty else None
    real = randoms[randoms["baseline"].eq("real_funding_extreme_oi_low")]
    real_row = real.iloc[0] if not real.empty else None
    leave_month = stability[stability["stability_type"].eq("leave_one_month")] if not stability.empty else pd.DataFrame()
    lines = [
        "# v5.1 Funding Extreme + OI Low RV Audit",
        "",
        "Status: audit candidate only. No live, shadow, selector, or real-live permission is changed.",
        "",
        "Carry convention: funding carry is a proxy using as-of settled funding rate linearly scaled by horizon. It is not a true settlement replay.",
        "Cost convention: `legacy_cost` subtracts one round-trip pair proxy; `pair_cost` subtracts two legs round-trip and is the stricter audit view.",
        "",
        "Core BTC hedge decomposition:",
    ]
    if btc_row is not None:
        lines.append(
            f"- events={int(btc_row['events'])}, price={btc_row['pair_price_pnl_mean']:.4%}, "
            f"carry={btc_row['funding_carry_pnl_mean']:.4%}, gross={btc_row['total_pair_gross_pnl_mean']:.4%}, "
            f"strict_pair_net20={btc_row['total_pair_net20_pair_cost_mean']:.4%}."
        )
    if real_row is not None:
        lines.append(f"- matched baseline real strict net20={real_row['mean']:.4%}, month_cap={real_row['month_cap35']:.4%}.")
    if not hedge.empty:
        best = hedge.sort_values("mean", ascending=False).head(1).iloc[0]
        lines.append(f"- best hedge leg by strict pair cost: {best['hedge_leg']} mean={best['mean']:.4%}, month_cap={best['month_cap35']:.4%}.")
    if not leave_month.empty:
        worst = leave_month.sort_values("mean").head(1).iloc[0]
        lines.append(f"- weakest leave-one-month result: remove {worst['removed']} mean={worst['mean']:.4%}.")
    lines.extend(
        [
            "",
            "Decision guardrails:",
            "- If strict pair-cost net is not positive, this cannot become an RV alpha candidate.",
            "- If matched random is comparable, the pocket remains diagnostic.",
            "- If the result is mostly funding carry, future work must treat it as carry timing rather than price-reversal alpha.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v51_funding_extreme_oi_low_rv_audit(cfg: V51Config | None = None) -> dict[str, Path]:
    cfg = cfg or V51Config()
    report_root = ensure_dir(cfg.report_root)
    features = _add_hedge_features(_add_candidate_context(_read_feature_subset(cfg), cfg))
    ledger = _candidate_ledger(features, cfg)
    decomposition = _decomposition(ledger)
    timing = _time_to_funding_summary(ledger)
    hedge = _hedge_leg_comparison(ledger)
    stability = _month_symbol_stability(ledger)
    randoms = _matched_random_baseline(features, cfg)
    costs = _cost_stress(features)

    outputs = {
        "rv_candidate_ledger": report_root / "rv_candidate_ledger.csv",
        "price_funding_pnl_decomposition": report_root / "price_funding_pnl_decomposition.csv",
        "time_to_funding_bucket_summary": report_root / "time_to_funding_bucket_summary.csv",
        "hedge_leg_comparison": report_root / "hedge_leg_comparison.csv",
        "month_symbol_stability": report_root / "month_symbol_stability.csv",
        "matched_random_baseline": report_root / "matched_random_baseline.csv",
        "cost_stress": report_root / "cost_stress.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    ledger.to_csv(outputs["rv_candidate_ledger"], index=False)
    decomposition.to_csv(outputs["price_funding_pnl_decomposition"], index=False)
    timing.to_csv(outputs["time_to_funding_bucket_summary"], index=False)
    hedge.to_csv(outputs["hedge_leg_comparison"], index=False)
    stability.to_csv(outputs["month_symbol_stability"], index=False)
    randoms.to_csv(outputs["matched_random_baseline"], index=False)
    costs.to_csv(outputs["cost_stress"], index=False)
    _write_notes(outputs["candidate_notes"], decomposition, hedge, randoms, stability)
    return outputs

