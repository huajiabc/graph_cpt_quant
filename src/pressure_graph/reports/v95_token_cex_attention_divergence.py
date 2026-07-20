"""v9.5 token DEX attention -> CEX underreaction alpha audit.

The module evaluates one frozen, independently timed entry hypothesis.  It is
historical research only: no result here changes paper/live permissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v50_perp_crowding_atlas import _max_contribution, _month_cap35


REPORT_ROOT = Path("reports/v9_5_token_cex_attention_divergence")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
EVENT_PATH = Path("reports/v6_3_token_pool_dex_attention/token_pool_attention_events.csv")
P2_TRADE_PATH = Path("reports/v1_3a_checkpoint_robustness/_v09b_trades_tmp.csv")

TAD0 = "TAD0_ALL_ATTENTION"
TAD1 = "TAD1_UNDERREACTION_RECLAIM"
TAD2 = "TAD2_CEX_CONFIRMATION"
NEG = "NEG_UNDERREACTION_DOWN_BAR"
CANDIDATES = (TAD0, TAD1, TAD2, NEG)


@dataclass(frozen=True)
class V95Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = FEATURE_PATH
    event_path: Path = EVENT_PATH
    p2_trade_path: Path = P2_TRADE_PATH
    exchange: str = "bybit"
    event_sources: tuple[str, ...] = (
        "dexpaprika_pool_ohlcv_1h",
        "geckoterminal_pool_ohlcv",
    )
    mapping_confidence: tuple[str, ...] = ("A", "B")
    cooldown_hours: int = 24
    entry_tolerance_minutes: int = 30
    universe_top_n: int = 50
    universe_lookback_days: int = 30
    min_universe_history_days: int = 7
    search_end: str = "2026-02-01T00:00:00Z"
    validation_end: str = "2026-05-01T00:00:00Z"
    cost_grid_bps: tuple[float, ...] = (10.0, 20.0, 30.0, 50.0)
    bootstrap_trials: int = 2_000
    random_trials: int = 200
    random_token_trials: int = 100
    random_seed: int = 20260713


FEATURE_COLUMNS = (
    "exchange",
    "symbol",
    "feature_time",
    "turnover",
    "warmup_complete",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "volume_z_1h",
    "future_ret_12h",
)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _bool(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[col].fillna(default).astype(bool)


def _read_events(cfg: V95Config) -> pd.DataFrame:
    if not cfg.event_path.exists():
        return pd.DataFrame()
    events = pd.read_csv(cfg.event_path, low_memory=False)
    required = {"event_id", "cex_symbol", "event_available_time", "source"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame()
    out = events.copy()
    out["cex_symbol"] = out["cex_symbol"].fillna("").astype(str).str.upper()
    out["source"] = out["source"].fillna("").astype(str)
    out["mapping_confidence"] = out.get("mapping_confidence", "").fillna("").astype(str)
    out["event_time"] = pd.to_datetime(out.get("event_time"), utc=True, errors="coerce")
    out["event_available_time"] = pd.to_datetime(out["event_available_time"], utc=True, errors="coerce")
    out = out[
        out["source"].isin(cfg.event_sources)
        & out["mapping_confidence"].isin(cfg.mapping_confidence)
    ].copy()
    out = out.dropna(subset=["event_available_time"]).drop_duplicates("event_id")
    return _deoverlap_events(out, cfg.cooldown_hours)


def _deoverlap_events(events: pd.DataFrame, cooldown_hours: int) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    local = events.sort_values(["cex_symbol", "event_available_time", "event_id"]).copy()
    cooldown = pd.Timedelta(hours=cooldown_hours)
    keep: list[int] = []
    last: dict[str, pd.Timestamp] = {}
    for idx, row in local.iterrows():
        symbol = str(row["cex_symbol"])
        timestamp = pd.Timestamp(row["event_available_time"])
        if symbol not in last or timestamp - last[symbol] >= cooldown:
            keep.append(idx)
            last[symbol] = timestamp
    return local.loc[keep].sort_values(["event_available_time", "cex_symbol"]).reset_index(drop=True)


def _read_feature_data(
    cfg: V95Config,
    event_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return event-symbol context rows and compact daily turnover totals."""
    if not cfg.feature_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    parquet = pq.ParquetFile(cfg.feature_path)
    available = set(parquet.schema.names)
    columns = [col for col in FEATURE_COLUMNS if col in available]
    context_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    for idx in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(idx, columns=columns).to_pandas()
        if chunk.empty:
            continue
        if "exchange" in chunk.columns:
            chunk = chunk[chunk["exchange"].fillna("").astype(str).str.lower().eq(cfg.exchange.lower())]
        chunk["symbol"] = chunk["symbol"].fillna("").astype(str).str.upper()
        chunk["feature_time"] = pd.to_datetime(chunk["feature_time"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["feature_time", "symbol"])
        if chunk.empty:
            continue
        daily = chunk[["symbol", "feature_time"]].copy()
        daily["turnover"] = _num(chunk, "turnover").fillna(0.0).to_numpy()
        daily["day"] = daily["feature_time"].dt.floor("D")
        daily_frames.append(daily.groupby(["day", "symbol"], as_index=False)["turnover"].sum())
        context = chunk[chunk["symbol"].isin(event_symbols) & _bool(chunk, "warmup_complete", True)].copy()
        if not context.empty:
            context_frames.append(context)
    context = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    if not daily.empty:
        daily = daily.groupby(["day", "symbol"], as_index=False)["turnover"].sum()
    if not context.empty:
        context = context.sort_values(["symbol", "feature_time"]).drop_duplicates(
            ["symbol", "feature_time"], keep="last"
        )
    return context.reset_index(drop=True), daily


def _monthly_top_symbols(daily: pd.DataFrame, cfg: V95Config) -> dict[str, set[str]]:
    if daily.empty:
        return {}
    days = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    first_month = days.min().floor("D").replace(day=1)
    last_month = days.max().floor("D").replace(day=1)
    months = pd.date_range(first_month, last_month, freq="MS", tz="UTC")
    result: dict[str, set[str]] = {}
    for month in months:
        start = month - pd.Timedelta(days=cfg.universe_lookback_days)
        mask = days.ge(start) & days.lt(month)
        hist = daily[mask]
        if hist["day"].nunique() < cfg.min_universe_history_days:
            result[month.strftime("%Y-%m")] = set()
            continue
        ranked = hist.groupby("symbol")["turnover"].sum().nlargest(cfg.universe_top_n)
        result[month.strftime("%Y-%m")] = set(ranked.index.astype(str))
    return result


def _apply_universe(context: pd.DataFrame, top_by_month: dict[str, set[str]]) -> pd.DataFrame:
    if context.empty:
        return context.copy()
    out = context.copy()
    out["month"] = out["feature_time"].dt.strftime("%Y-%m")
    out["universe_top50"] = [
        symbol in top_by_month.get(month, set())
        for symbol, month in zip(out["symbol"].astype(str), out["month"].astype(str), strict=False)
    ]
    return out


def _attach_entries(events: pd.DataFrame, context: pd.DataFrame, cfg: V95Config) -> pd.DataFrame:
    if events.empty or context.empty:
        return pd.DataFrame()
    eligible = context[_bool(context, "universe_top50")].copy()
    rows: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(minutes=cfg.entry_tolerance_minutes)
    for symbol, group in events.groupby("cex_symbol", sort=False):
        features = eligible[eligible["symbol"].eq(str(symbol))].sort_values("feature_time")
        if features.empty:
            continue
        right = features.drop(columns=["symbol"], errors="ignore")
        joined = pd.merge_asof(
            group.sort_values("event_available_time"),
            right,
            left_on="event_available_time",
            right_on="feature_time",
            direction="forward",
            allow_exact_matches=False,
            tolerance=tolerance,
        )
        joined["symbol"] = str(symbol)
        rows.append(joined)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).dropna(subset=["feature_time", "future_ret_12h"])
    out["entry_time"] = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    out["entry_delay_minutes"] = (
        out["entry_time"] - pd.to_datetime(out["event_available_time"], utc=True, errors="coerce")
    ).dt.total_seconds() / 60.0
    return _add_candidate_fields(out, cfg)


def _add_candidate_fields(frame: pd.DataFrame, cfg: V95Config) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    ret15 = _num(out, "ret_15m")
    ret1h = _num(out, "ret_1h")
    ret4h = _num(out, "ret_4h")
    volz = _num(out, "volume_z_1h")
    underreaction = ret4h.abs().le(0.01) & volz.lt(1.0)
    out[TAD0] = True
    out[TAD1] = underreaction & ret15.gt(0.0)
    out[TAD2] = ret1h.gt(0.0) & volz.ge(1.0) & ret4h.le(0.04)
    out[NEG] = underreaction & ret15.le(0.0)
    out["gross_12h"] = _num(out, "future_ret_12h")
    for cost in cfg.cost_grid_bps:
        out[f"net{int(cost)}_12h"] = out["gross_12h"] - 2.0 * cost / 10_000.0
    search_end = pd.Timestamp(cfg.search_end)
    validation_end = pd.Timestamp(cfg.validation_end)
    out["split"] = np.select(
        [out["entry_time"].lt(search_end), out["entry_time"].lt(validation_end)],
        ["search", "validation"],
        default="holdout",
    )
    return out


def _day_block_bootstrap(
    sample: pd.DataFrame,
    value_col: str,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    if sample.empty or value_col not in sample.columns:
        return np.nan, np.nan
    local = sample.dropna(subset=[value_col, "entry_time"]).copy()
    local["entry_day"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce").dt.floor("D")
    blocks = [group[value_col].to_numpy(dtype=float) for _, group in local.groupby("entry_day")]
    if len(blocks) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(trials, dtype=float)
    for trial in range(trials):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        values = np.concatenate([blocks[idx] for idx in selected])
        means[trial] = float(np.nanmean(values))
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def _metrics(
    sample: pd.DataFrame,
    cfg: V95Config,
    seed_offset: int = 0,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    if sample.empty:
        return {
            "trades": 0,
            "symbols": 0,
            "active_months": 0,
            "gross_12h": np.nan,
            "net10_12h": np.nan,
            "net20_12h": np.nan,
            "net30_12h": np.nan,
            "net50_12h": np.nan,
            "hit_rate_net20": np.nan,
            "month_cap35_net20": np.nan,
            "max_month_contribution": np.nan,
            "max_symbol_contribution": np.nan,
            "bootstrap95_low_net20": np.nan,
            "bootstrap95_high_net20": np.nan,
        }
    low, high = (
        _day_block_bootstrap(
            sample,
            "net20_12h",
            cfg.bootstrap_trials,
            cfg.random_seed + seed_offset,
        )
        if bootstrap
        else (np.nan, np.nan)
    )
    return {
        "trades": int(len(sample)),
        "symbols": int(sample["symbol"].nunique()),
        "active_months": int(sample["month"].nunique()),
        "gross_12h": float(_num(sample, "gross_12h").mean()),
        "net10_12h": float(_num(sample, "net10_12h").mean()),
        "net20_12h": float(_num(sample, "net20_12h").mean()),
        "net30_12h": float(_num(sample, "net30_12h").mean()),
        "net50_12h": float(_num(sample, "net50_12h").mean()),
        "hit_rate_net20": float(_num(sample, "net20_12h").gt(0).mean()),
        "month_cap35_net20": _month_cap35(sample, "net20_12h"),
        "max_month_contribution": _max_contribution(sample, "month", "net20_12h"),
        "max_symbol_contribution": _max_contribution(sample, "symbol", "net20_12h"),
        "bootstrap95_low_net20": low,
        "bootstrap95_high_net20": high,
    }


def _candidate_summary(ledger: pd.DataFrame, cfg: V95Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_idx, candidate in enumerate(CANDIDATES):
        selected = ledger[_bool(ledger, candidate)]
        for split_idx, split in enumerate(("full", "search", "validation", "holdout")):
            sample = selected if split == "full" else selected[selected["split"].eq(split)]
            rows.append(
                {
                    "candidate": candidate,
                    "split": split,
                    **_metrics(sample, cfg, seed_offset=100 * candidate_idx + split_idx),
                }
            )
    return pd.DataFrame(rows)


def _group_summary(ledger: pd.DataFrame, cfg: V95Config, key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        selected = ledger[_bool(ledger, candidate)]
        for value, group in selected.groupby(key, dropna=False, sort=False):
            rows.append({"candidate": candidate, key: value, **_metrics(group, cfg, bootstrap=False)})
    return pd.DataFrame(rows)


def _leave_one_month(ledger: pd.DataFrame, cfg: V95Config) -> pd.DataFrame:
    selected = ledger[_bool(ledger, TAD1)]
    rows = []
    for month in sorted(selected["month"].dropna().astype(str).unique()):
        sample = selected[selected["month"].ne(month)]
        rows.append(
            {"candidate": TAD1, "removed_month": month, **_metrics(sample, cfg, bootstrap=False)}
        )
    return pd.DataFrame(rows)


def _add_p2_overlap(ledger: pd.DataFrame, cfg: V95Config) -> pd.DataFrame:
    out = ledger.copy()
    out["p2_overlap_60m"] = False
    if out.empty or not cfg.p2_trade_path.exists():
        return out
    trades = pd.read_csv(cfg.p2_trade_path, low_memory=False)
    if trades.empty or not {"candidate", "symbol", "entry_time"}.issubset(trades.columns):
        return out
    trades = trades[trades["candidate"].astype(str).isin(["CIC1_beta_extreme", "CIC2_beta_broad"])].copy()
    if "cost_single_side_bps" in trades.columns:
        cost = _num(trades, "cost_single_side_bps")
        trades = trades[cost.eq(20.0) | cost.isna()].copy()
    trades["symbol"] = trades["symbol"].fillna("").astype(str).str.upper()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time"]).drop_duplicates(["symbol", "entry_time"])
    by_symbol = {
        symbol: group["entry_time"].sort_values().astype("int64").to_numpy()
        for symbol, group in trades.groupby("symbol", sort=False)
    }
    tolerance = int(pd.Timedelta(minutes=60).value)
    flags = []
    for row in out[["symbol", "entry_time"]].itertuples(index=False):
        values = by_symbol.get(str(row.symbol), np.array([], dtype=np.int64))
        timestamp = int(pd.Timestamp(row.entry_time).value)
        pos = int(np.searchsorted(values, timestamp))
        distances = []
        if pos < len(values):
            distances.append(abs(int(values[pos]) - timestamp))
        if pos > 0:
            distances.append(abs(int(values[pos - 1]) - timestamp))
        flags.append(bool(distances and min(distances) <= tolerance))
    out["p2_overlap_60m"] = flags
    return out


def _p2_overlap_summary(ledger: pd.DataFrame, cfg: V95Config) -> pd.DataFrame:
    rows = []
    for candidate in (TAD1, TAD2):
        selected = ledger[_bool(ledger, candidate)]
        rows.extend(
            [
                {"candidate": candidate, "sample": "all", **_metrics(selected, cfg, bootstrap=False)},
                {
                    "candidate": candidate,
                    "sample": "p2_overlap",
                    **_metrics(selected[_bool(selected, "p2_overlap_60m")], cfg, bootstrap=False),
                },
                {
                    "candidate": candidate,
                    "sample": "p2_nonoverlap",
                    **_metrics(selected[~_bool(selected, "p2_overlap_60m")], cfg, bootstrap=False),
                },
            ]
        )
    return pd.DataFrame(rows)


def _same_token_random_controls(
    ledger: pd.DataFrame,
    feature_pool: pd.DataFrame,
    cfg: V95Config,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.random_seed + 10)
    rows: list[dict[str, Any]] = []
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for candidate in (TAD1, TAD2):
        real = ledger[_bool(ledger, candidate)]
        pool = feature_pool[_bool(feature_pool, candidate)].dropna(subset=["future_ret_12h"]).copy()
        grouped = {
            (symbol, month): group.reset_index(drop=True)
            for (symbol, month), group in pool.groupby(["symbol", "month"], sort=False)
        }
        for trial in range(cfg.random_trials):
            picks = []
            for row in real[["symbol", "month", "event_available_time"]].itertuples(index=False):
                candidates = grouped.get((str(row.symbol), str(row.month)), pd.DataFrame())
                if candidates.empty:
                    continue
                far = candidates[
                    (pd.to_datetime(candidates["entry_time"], utc=True) - pd.Timestamp(row.event_available_time))
                    .abs()
                    .ge(cooldown)
                ]
                if far.empty:
                    continue
                picks.append(far.iloc[int(rng.integers(0, len(far)))])
            sample = pd.DataFrame(picks)
            rows.append(
                {
                    "candidate": candidate,
                    "control": "same_token_same_month_random_time",
                    "trial": trial,
                    **_metrics(sample, cfg, bootstrap=False),
                }
            )
    return rows


def _same_chain_random_controls(
    events: pd.DataFrame,
    context: pd.DataFrame,
    cfg: V95Config,
) -> list[dict[str, Any]]:
    if events.empty or "chain" not in events.columns:
        return []
    symbol_chain = (
        events.dropna(subset=["chain"])
        .groupby("cex_symbol")["chain"]
        .agg(lambda values: values.astype(str).mode().iloc[0])
        .to_dict()
    )
    symbols_by_chain: dict[str, list[str]] = {}
    for symbol, chain in symbol_chain.items():
        symbols_by_chain.setdefault(str(chain), []).append(str(symbol))
    rng = np.random.default_rng(cfg.random_seed + 20)
    rows: list[dict[str, Any]] = []
    for trial in range(cfg.random_token_trials):
        randomized = events.copy()
        targets = []
        for source_symbol in randomized["cex_symbol"].astype(str):
            chain = str(symbol_chain.get(source_symbol, ""))
            choices = sorted(symbol for symbol in symbols_by_chain.get(chain, []) if symbol != source_symbol)
            targets.append(choices[int(rng.integers(0, len(choices)))] if choices else "")
        randomized["source_cex_symbol"] = randomized["cex_symbol"]
        randomized["cex_symbol"] = targets
        randomized = randomized[randomized["cex_symbol"].ne("")].copy()
        randomized["event_id"] = randomized["event_id"].astype(str) + f"|random_chain_{trial}"
        randomized = _deoverlap_events(randomized, cfg.cooldown_hours)
        attached = _attach_entries(randomized, context, cfg)
        for candidate in (TAD1, TAD2):
            sample = attached[_bool(attached, candidate)]
            rows.append(
                {
                    "candidate": candidate,
                    "control": "same_chain_random_token_same_time",
                    "trial": trial,
                    **_metrics(sample, cfg, bootstrap=False),
                }
            )
    return rows


def _control_summary(
    events: pd.DataFrame,
    ledger: pd.DataFrame,
    context: pd.DataFrame,
    feature_pool: pd.DataFrame,
    cfg: V95Config,
) -> pd.DataFrame:
    rows = _same_token_random_controls(ledger, feature_pool, cfg)
    rows.extend(_same_chain_random_controls(events, context, cfg))
    shifted = events.copy()
    shifted["event_available_time"] = shifted["event_available_time"] + pd.Timedelta(days=7)
    shifted["event_time"] = shifted["event_time"] + pd.Timedelta(days=7)
    shifted["event_id"] = shifted["event_id"].astype(str) + "|shift_7d"
    placebo = _attach_entries(_deoverlap_events(shifted, cfg.cooldown_hours), context, cfg)
    for candidate in (TAD1, TAD2):
        rows.append(
            {
                "candidate": candidate,
                "control": "shifted_event_plus_7d",
                "trial": 0,
                **_metrics(placebo[_bool(placebo, candidate)], cfg, bootstrap=False),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    overlap: pd.DataFrame,
    candidate: str,
) -> tuple[pd.DataFrame, str]:
    primary = summary[summary["candidate"].eq(candidate)].set_index("split")
    full = primary.loc["full"]
    validation = primary.loc["validation"]
    holdout = primary.loc["holdout"]
    candidate_controls = controls[controls["candidate"].eq(candidate)]
    same_token = candidate_controls[
        candidate_controls["control"].eq("same_token_same_month_random_time")
    ]["net20_12h"].dropna()
    same_chain = candidate_controls[
        candidate_controls["control"].eq("same_chain_random_token_same_time")
    ]["net20_12h"].dropna()
    shifted = candidate_controls[candidate_controls["control"].eq("shifted_event_plus_7d")]
    shifted_net = float(shifted["net20_12h"].iloc[0]) if not shifted.empty else np.nan
    nonoverlap = overlap[
        overlap["candidate"].eq(candidate) & overlap["sample"].eq("p2_nonoverlap")
    ].iloc[0]
    rules = [
        ("full_net20_positive", float(full["net20_12h"]), 0.0, float(full["net20_12h"]) > 0),
        ("validation_net20_positive", float(validation["net20_12h"]), 0.0, float(validation["net20_12h"]) > 0),
        ("holdout_net20_positive", float(holdout["net20_12h"]), 0.0, float(holdout["net20_12h"]) > 0),
        ("validation_trades", int(validation["trades"]), 20, int(validation["trades"]) >= 20),
        ("holdout_trades", int(holdout["trades"]), 20, int(holdout["trades"]) >= 20),
        ("validation_active_months", int(validation["active_months"]), 2, int(validation["active_months"]) >= 2),
        ("holdout_active_months", int(holdout["active_months"]), 2, int(holdout["active_months"]) >= 2),
        ("month_cap35_net20_positive", float(full["month_cap35_net20"]), 0.0, float(full["month_cap35_net20"]) > 0),
        ("max_month_contribution", float(full["max_month_contribution"]), 0.35, float(full["max_month_contribution"]) <= 0.35),
        ("bootstrap95_low_positive", float(full["bootstrap95_low_net20"]), 0.0, float(full["bootstrap95_low_net20"]) > 0),
        (
            "beats_same_token_random_p90",
            float(full["net20_12h"] - same_token.quantile(0.90)) if len(same_token) else np.nan,
            0.0,
            bool(len(same_token) and full["net20_12h"] > same_token.quantile(0.90)),
        ),
        (
            "beats_same_chain_random_median",
            float(full["net20_12h"] - same_chain.median()) if len(same_chain) else np.nan,
            0.0,
            bool(len(same_chain) and full["net20_12h"] > same_chain.median()),
        ),
        ("beats_shifted_7d", float(full["net20_12h"] - shifted_net), 0.0, bool(full["net20_12h"] > shifted_net)),
        (
            "p2_nonoverlap_count",
            int(nonoverlap["trades"]),
            max(20, int(np.ceil(0.5 * full["trades"]))),
            int(nonoverlap["trades"]) >= max(20, int(np.ceil(0.5 * full["trades"]))),
        ),
        ("p2_nonoverlap_net20_positive", float(nonoverlap["net20_12h"]), 0.0, float(nonoverlap["net20_12h"]) > 0),
        ("net30_positive", float(full["net30_12h"]), 0.0, float(full["net30_12h"]) > 0),
    ]
    table = pd.DataFrame(rules, columns=["rule", "observed", "required", "passed"])
    sample_rules = {
        "validation_trades",
        "holdout_trades",
        "validation_active_months",
        "holdout_active_months",
    }
    if bool(table["passed"].all()):
        verdict = "FORWARD_COUNTERFACTUAL_ELIGIBLE"
    elif bool(table[~table["rule"].isin(sample_rules)]["passed"].all()):
        verdict = "INSUFFICIENT_SAMPLE"
    else:
        verdict = "FAIL_HISTORICAL_SCREEN"
    table.insert(0, "candidate", candidate)
    table["verdict"] = verdict
    return table, verdict


def _coverage(
    events: pd.DataFrame,
    context: pd.DataFrame,
    ledger: pd.DataFrame,
    top_by_month: dict[str, set[str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stage": "eligible_hourly_events_after_24h_cooldown", "rows": len(events), "symbols": events.get("cex_symbol", pd.Series(dtype=str)).nunique()},
            {"stage": "cex_context_rows_for_event_symbols", "rows": len(context), "symbols": context.get("symbol", pd.Series(dtype=str)).nunique()},
            {"stage": "months_with_frozen_top50", "rows": sum(bool(value) for value in top_by_month.values()), "symbols": np.nan},
            {"stage": "matched_top50_entries_with_12h_label", "rows": len(ledger), "symbols": ledger.get("symbol", pd.Series(dtype=str)).nunique()},
            {"stage": "primary_TAD1_entries", "rows": int(_bool(ledger, TAD1).sum()), "symbols": ledger.loc[_bool(ledger, TAD1), "symbol"].nunique() if not ledger.empty else 0},
        ]
    )


def _write_notes(
    path: Path,
    summary: pd.DataFrame,
    decision: pd.DataFrame,
    verdicts: dict[str, str],
) -> None:
    lines = [
        "# v9.5 Token DEX Attention -> CEX Underreaction",
        "",
        "Status: historical alpha audit only. P2 paper/live configuration is unchanged.",
    ]
    for candidate in (TAD1, TAD2):
        selected = summary[(summary["candidate"].eq(candidate)) & (summary["split"].eq("full"))]
        if selected.empty:
            continue
        row = selected.iloc[0]
        failed = decision.loc[
            decision["candidate"].eq(candidate) & ~decision["passed"], "rule"
        ].astype(str).tolist()
        lines.extend(
            [
                f"- {candidate}: verdict=`{verdicts[candidate]}`, trades={int(row['trades'])}, "
                f"net20_12h={row['net20_12h']:.4%}, net30_12h={row['net30_12h']:.4%}, "
                f"bootstrap95=[{row['bootstrap95_low_net20']:.4%}, "
                f"{row['bootstrap95_high_net20']:.4%}].",
                f"  Failed rules: {', '.join(failed) if failed else 'none'}.",
            ]
        )
    lines.extend(
        [
            "- A positive search result alone cannot enable forward or live action.",
            "- Thresholds, source filters, symbols, and exits remain frozen after this run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v95_token_cex_attention_divergence(cfg: V95Config | None = None) -> dict[str, Path]:
    cfg = cfg or V95Config()
    report_root = ensure_dir(cfg.report_root)
    events = _read_events(cfg)
    context, daily = _read_feature_data(cfg, set(events.get("cex_symbol", pd.Series(dtype=str)).astype(str)))
    top_by_month = _monthly_top_symbols(daily, cfg)
    context = _apply_universe(context, top_by_month)
    ledger = _attach_entries(events, context, cfg)
    ledger = _add_p2_overlap(ledger, cfg)

    feature_pool = context[_bool(context, "universe_top50")].copy()
    if not feature_pool.empty:
        feature_pool["entry_time"] = feature_pool["feature_time"]
        feature_pool = _add_candidate_fields(feature_pool, cfg)

    summary = _candidate_summary(ledger, cfg)
    source_summary = _group_summary(ledger, cfg, "source")
    month_summary = _group_summary(ledger, cfg, "month")
    symbol_summary = _group_summary(ledger, cfg, "symbol")
    leave_month = _leave_one_month(ledger, cfg)
    overlap = _p2_overlap_summary(ledger, cfg)
    controls = _control_summary(events, ledger, context, feature_pool, cfg)
    decision_frames = []
    verdicts = {}
    for candidate in (TAD1, TAD2):
        candidate_decision, candidate_verdict = _decision_table(
            summary, controls, overlap, candidate
        )
        decision_frames.append(candidate_decision)
        verdicts[candidate] = candidate_verdict
    decision = pd.concat(decision_frames, ignore_index=True)
    coverage = _coverage(events, context, ledger, top_by_month)

    outputs = {
        "coverage": report_root / "coverage.csv",
        "event_entry_ledger": report_root / "event_entry_ledger.csv",
        "candidate_summary": report_root / "candidate_summary.csv",
        "source_summary": report_root / "source_summary.csv",
        "month_summary": report_root / "month_summary.csv",
        "symbol_summary": report_root / "symbol_summary.csv",
        "leave_one_month": report_root / "leave_one_month.csv",
        "negative_controls": report_root / "negative_controls.csv",
        "p2_overlap_summary": report_root / "p2_overlap_summary.csv",
        "decision_table": report_root / "decision_table.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["coverage"], index=False)
    ledger.to_csv(outputs["event_entry_ledger"], index=False)
    summary.to_csv(outputs["candidate_summary"], index=False)
    source_summary.to_csv(outputs["source_summary"], index=False)
    month_summary.to_csv(outputs["month_summary"], index=False)
    symbol_summary.to_csv(outputs["symbol_summary"], index=False)
    leave_month.to_csv(outputs["leave_one_month"], index=False)
    controls.to_csv(outputs["negative_controls"], index=False)
    overlap.to_csv(outputs["p2_overlap_summary"], index=False)
    decision.to_csv(outputs["decision_table"], index=False)
    _write_notes(
        outputs["candidate_notes"],
        summary,
        decision,
        verdicts,
    )
    return outputs
