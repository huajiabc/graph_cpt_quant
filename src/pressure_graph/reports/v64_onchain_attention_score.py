"""v6.4 On-chain Attention Score.

Composite score audit over existing v6.2 market-level and v6.3 token/pool-level
attention artifacts.  This is a diagnostic layer, not a selector or gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _max_contribution, _month_cap_expectancy
from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config, _read_p2_trades


REPORT_ROOT = Path("reports/v6_4_onchain_attention_score")
DEFAULT_MARKET_ATTENTION_DAYS = Path("reports/v6_2_onchain_attention_attribution/onchain_attention_days.csv")
DEFAULT_TOKEN_EVENTS = Path("reports/v6_3_token_pool_dex_attention/token_pool_attention_events.csv")
DEFAULT_TOKEN_MAPPING = Path("reports/v6_3_token_pool_dex_attention/token_pool_mapping.csv")


@dataclass(frozen=True)
class V64Config:
    report_root: Path = REPORT_ROOT
    market_attention_days_path: Path = DEFAULT_MARKET_ATTENTION_DAYS
    token_events_path: Path = DEFAULT_TOKEN_EVENTS
    token_mapping_path: Path = DEFAULT_TOKEN_MAPPING
    v61: V61Config = V61Config()
    lookback_windows: tuple[str, ...] = ("12h", "24h", "48h")
    high_score_quantile: float = 0.75
    random_trials: int = 100
    random_seed: int = 640


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _read_market_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    if out.empty or "event_available_time" not in out.columns:
        return pd.DataFrame()
    out["event_available_time"] = pd.to_datetime(out["event_available_time"], utc=True, errors="coerce")
    out["event_date"] = pd.to_datetime(out.get("event_date"), utc=True, errors="coerce")
    raw = _num(out, "attention_intensity")
    if raw.isna().all():
        raw = _num(out, "max_zscore").clip(lower=0) + _num(out, "max_percentile").fillna(0)
    out["market_attention_raw"] = raw.fillna(0).clip(lower=0)
    out["market_attention_score"] = out["market_attention_raw"].rank(pct=True).fillna(0)
    out["primary_event_type"] = out.get("primary_event_type", "unknown").astype(str)
    return out.dropna(subset=["event_available_time"]).sort_values("event_available_time").reset_index(drop=True)


def _read_token_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    out = pd.read_csv(path)
    if out.empty or "event_available_time" not in out.columns or "cex_symbol" not in out.columns:
        return pd.DataFrame()
    out["event_available_time"] = pd.to_datetime(out["event_available_time"], utc=True, errors="coerce")
    raw = _num(out, "zscore").clip(lower=0).fillna(0) + _num(out, "percentile").fillna(0)
    out["token_attention_raw"] = raw
    out["token_attention_score"] = out.groupby("cex_symbol", sort=False)["token_attention_raw"].rank(pct=True).fillna(0)
    out["event_type"] = out.get("event_type", "unknown").astype(str)
    return out.dropna(subset=["event_available_time", "cex_symbol"]).sort_values(["cex_symbol", "event_available_time"]).reset_index(drop=True)


def _read_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["net20"] = _num(out, "net20") if "net20" in out.columns else _num(out, "net_return")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    if "trade_id" not in out.columns:
        out["trade_id"] = (
            out.get("signal_id", out["symbol"].astype(str) + "|" + out["entry_time"].astype(str) + "|" + out["candidate"].astype(str))
            .astype(str)
        )
    out = out.dropna(subset=["entry_time", "net20"]).sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)
    try:
        return _add_asof_burst_phase(out, "1h")
    except Exception:  # noqa: BLE001 - burst phase is diagnostic only
        out["burst_count_so_far"] = np.nan
        return out


def _parse_window(value: str) -> pd.Timedelta:
    return pd.Timedelta(value)


def _event_type_join(values: pd.Series, limit: int = 4) -> str:
    unique = [str(item) for item in values.dropna().astype(str).unique() if str(item)]
    return "|".join(unique[:limit])


def _score_one_window(
    trades: pd.DataFrame,
    market_events: pd.DataFrame,
    token_events: pd.DataFrame,
    window: pd.Timedelta,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    market_times = market_events.sort_values("event_available_time") if not market_events.empty else pd.DataFrame()
    token_by_symbol = {symbol: group.sort_values("event_available_time") for symbol, group in token_events.groupby("cex_symbol", sort=False)} if not token_events.empty else {}
    for trade in trades.itertuples(index=False):
        entry = pd.Timestamp(trade.entry_time)
        symbol = str(trade.symbol)
        market_prior = pd.DataFrame()
        if not market_times.empty:
            market_prior = market_times[
                market_times["event_available_time"].le(entry)
                & market_times["event_available_time"].gt(entry - window)
            ]
        token_prior = pd.DataFrame()
        if symbol in token_by_symbol:
            local = token_by_symbol[symbol]
            token_prior = local[
                local["event_available_time"].le(entry)
                & local["event_available_time"].gt(entry - window)
            ]
        market_score = float(_num(market_prior, "market_attention_score").max()) if len(market_prior) else 0.0
        token_score = float(_num(token_prior, "token_attention_score").max()) if len(token_prior) else 0.0
        combined_score = 0.6 * market_score + 0.4 * token_score if token_score > 0 else market_score
        rows.append(
            {
                **trade._asdict(),
                "lookback_window": str(window),
                "market_event_count": int(len(market_prior)),
                "token_event_count": int(len(token_prior)),
                "market_attention_score": market_score,
                "token_attention_score": token_score,
                "combined_attention_score": combined_score,
                "market_event_types": _event_type_join(market_prior.get("primary_event_type", pd.Series(dtype=str))),
                "token_event_types": _event_type_join(token_prior.get("event_type", pd.Series(dtype=str))),
            }
        )
    return pd.DataFrame(rows)


def _score_trades(trades: pd.DataFrame, market_events: pd.DataFrame, token_events: pd.DataFrame, cfg: V64Config) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frames = [_score_one_window(trades, market_events, token_events, _parse_window(window)) for window in cfg.lookback_windows]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bucket_scores(frame: pd.DataFrame, score_col: str) -> pd.Series:
    score = _num(frame, score_col)
    bucket = pd.Series("no_attention", index=frame.index, dtype="object")
    positive = score.gt(0) & score.notna()
    if positive.sum() < 4:
        bucket.loc[positive] = "attention_present"
        return bucket
    ranks = score[positive].rank(method="first")
    try:
        labels = pd.qcut(ranks, q=4, labels=["q1_low", "q2_mid", "q3_high", "q4_top"])
        bucket.loc[positive] = labels.astype(str).to_numpy()
    except ValueError:
        bucket.loc[positive] = "attention_present"
    return bucket


def _sample_metrics(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {"trades": 0, "net20": np.nan, "hit_rate": np.nan, "month_cap35_net20": np.nan, "max_symbol_contribution": np.nan}
    local = sample.copy()
    local["net_return"] = _num(local, "net20")
    return {
        "trades": int(len(local)),
        "net20": float(_num(local, "net20").mean()),
        "hit_rate": float(_num(local, "net20").gt(0).mean()),
        "month_cap35_net20": _month_cap_expectancy(local),
        "max_symbol_contribution": _max_contribution(local, "symbol"),
    }


def _modules(scored: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    modules = [
        ("P2_all", scored),
        ("CIC1", scored[scored["candidate"].astype(str).eq("CIC1_beta_extreme")]),
        ("CIC2", scored[scored["candidate"].astype(str).eq("CIC2_beta_broad")]),
    ]
    if "burst_count_so_far" in scored.columns:
        modules.append(("O6_late9_candidates", scored[_num(scored, "burst_count_so_far").ge(9)]))
    return modules


def _score_bucket_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame([{"module": "P2_all", "score": "none", "bucket": "no_scored_trades", "trades": 0}])
    rows = []
    for window, window_frame in scored.groupby("lookback_window", sort=False):
        for score_col in ("market_attention_score", "token_attention_score", "combined_attention_score"):
            local = window_frame.copy()
            local["score_bucket"] = _bucket_scores(local, score_col)
            for module, sample in _modules(local):
                if sample.empty:
                    rows.append({"lookback_window": window, "module": module, "score": score_col, "bucket": "empty", "trades": 0})
                    continue
                for bucket, group in sample.groupby("score_bucket", sort=False):
                    rows.append({"lookback_window": window, "module": module, "score": score_col, "bucket": bucket, **_sample_metrics(group)})
    return pd.DataFrame(rows)


def _score_component_summary(scored: pd.DataFrame, cfg: V64Config) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for window, window_frame in scored.groupby("lookback_window", sort=False):
        threshold = _num(window_frame, "combined_attention_score").quantile(cfg.high_score_quantile)
        for module, sample in _modules(window_frame):
            if sample.empty:
                continue
            buckets = {
                "market_prior": sample[_num(sample, "market_attention_score").gt(0)],
                "no_market_prior": sample[_num(sample, "market_attention_score").le(0)],
                "token_prior": sample[_num(sample, "token_attention_score").gt(0)],
                "no_token_prior": sample[_num(sample, "token_attention_score").le(0)],
                "combined_high": sample[_num(sample, "combined_attention_score").ge(threshold)],
                "combined_not_high": sample[_num(sample, "combined_attention_score").lt(threshold)],
            }
            for bucket, group in buckets.items():
                rows.append(
                    {
                        "lookback_window": window,
                        "module": module,
                        "bucket": bucket,
                        "combined_high_threshold": float(threshold) if pd.notna(threshold) else np.nan,
                        **_sample_metrics(group),
                    }
                )
    return pd.DataFrame(rows)


def _random_score_control(scored: pd.DataFrame, cfg: V64Config) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for window, window_frame in scored.groupby("lookback_window", sort=False):
        score = _num(window_frame, "combined_attention_score")
        threshold = score.quantile(cfg.high_score_quantile)
        actual = window_frame[score.ge(threshold)]
        actual_metrics = _sample_metrics(actual)
        random_nets = []
        random_caps = []
        for _ in range(cfg.random_trials):
            local = window_frame.copy()
            shuffled = []
            for _, group in local.groupby("month", sort=False):
                values = _num(group, "combined_attention_score").to_numpy()
                rng.shuffle(values)
                shuffled.extend(values)
            local["random_score"] = shuffled
            selected = local[_num(local, "random_score").ge(threshold)]
            metrics = _sample_metrics(selected)
            random_nets.append(metrics["net20"])
            random_caps.append(metrics["month_cap35_net20"])
        net_arr = np.asarray(random_nets, dtype=float)
        net_arr = net_arr[np.isfinite(net_arr)]
        cap_arr = np.asarray(random_caps, dtype=float)
        cap_arr = cap_arr[np.isfinite(cap_arr)]
        actual_net = actual_metrics["net20"]
        rows.append(
            {
                "lookback_window": window,
                "score": "combined_attention_score",
                "actual_high_trades": actual_metrics["trades"],
                "actual_high_net20": actual_net,
                "actual_high_month_cap35_net20": actual_metrics["month_cap35_net20"],
                "random_trials": cfg.random_trials,
                "random_median_net20": float(np.nanmedian(net_arr)) if len(net_arr) else np.nan,
                "random_p75_net20": float(np.nanpercentile(net_arr, 75)) if len(net_arr) else np.nan,
                "random_p90_net20": float(np.nanpercentile(net_arr, 90)) if len(net_arr) else np.nan,
                "random_median_month_cap35_net20": float(np.nanmedian(cap_arr)) if len(cap_arr) else np.nan,
                "actual_percentile": float((net_arr <= actual_net).mean()) if len(net_arr) and pd.notna(actual_net) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _coverage(market_events: pd.DataFrame, token_events: pd.DataFrame, mapping: pd.DataFrame, trades: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": "market_attention_days", "rows": int(len(market_events)), "status": "ok" if len(market_events) else "missing_or_empty"},
            {"dataset": "token_pool_attention_events", "rows": int(len(token_events)), "status": "ok" if len(token_events) else "missing_or_empty"},
            {
                "dataset": "token_mapping_A_B",
                "rows": int(mapping["mapping_confidence"].astype(str).isin(["A", "B"]).sum()) if not mapping.empty and "mapping_confidence" in mapping.columns else 0,
                "status": "usable_mapping",
            },
            {"dataset": "p2_trades", "rows": int(len(trades)), "status": "ok" if len(trades) else "missing_or_empty"},
            {"dataset": "scored_trade_rows", "rows": int(len(scored)), "status": "ok" if len(scored) else "empty"},
            {"dataset": "token_score_symbols", "rows": int(token_events["cex_symbol"].nunique()) if not token_events.empty else 0, "status": "coverage"},
        ]
    )


def _write_notes(path: Path, coverage: pd.DataFrame, component: pd.DataFrame, random_control: pd.DataFrame) -> None:
    lines = [
        "# v6.4 On-chain Attention Score",
        "",
        "Status: diagnostic score only. No strategy, gate, selector, shadow, or real-live permission is changed.",
        "",
    ]
    cov = coverage.set_index("dataset") if not coverage.empty else pd.DataFrame()
    for dataset in ("market_attention_days", "token_pool_attention_events", "token_mapping_A_B", "p2_trades"):
        if dataset in cov.index:
            lines.append(f"- {dataset}: rows={int(cov.loc[dataset, 'rows'])}, status={cov.loc[dataset, 'status']}.")
    token_prior = pd.DataFrame()
    if not component.empty and "bucket" in component.columns:
        token_prior = component[component["bucket"].astype(str).eq("token_prior")]
    if not token_prior.empty:
        token_trades = int(pd.to_numeric(token_prior["trades"], errors="coerce").fillna(0).max())
        lines.append(f"- max token-prior P2/CIC trades in score audit: {token_trades}.")
    best = pd.DataFrame()
    if not component.empty and "bucket" in component.columns:
        best = component[component["bucket"].astype(str).eq("combined_high")].dropna(subset=["net20"])
    if not best.empty:
        row = best.sort_values("net20", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                f"Best combined-high bucket: {row['module']} {row['lookback_window']} net20={row['net20']:.4%}, trades={int(row['trades'])}.",
            ]
        )
    if not random_control.empty:
        row = random_control.sort_values("actual_high_net20", ascending=False).iloc[0]
        lines.append(
            f"Best random-control row: {row['lookback_window']} actual_high_net20={row['actual_high_net20']:.4%}, "
            f"random_p75={row['random_p75_net20']:.4%}, percentile={row['actual_percentile']:.2f}."
        )
        if pd.notna(row.get("random_p90_net20", np.nan)) and row["actual_high_net20"] <= row["random_p90_net20"]:
            lines.append("Random-control status: not passed at p90; score remains diagnostic.")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Token-level score is coverage-limited until symbol-to-pool mapping expands.",
            "- Market-level score cannot be interpreted as token-specific DEX alpha.",
            "- A score can only become a shadow context after token coverage and random controls improve.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v64_onchain_attention_score(cfg: V64Config | None = None) -> dict[str, Path]:
    cfg = cfg or V64Config()
    report_root = ensure_dir(cfg.report_root)
    market_events = _read_market_events(cfg.market_attention_days_path)
    token_events = _read_token_events(cfg.token_events_path)
    mapping = _read_mapping(cfg.token_mapping_path)
    trades = _prepare_trades(_read_p2_trades(cfg.v61.trade_cache_path))
    scored = _score_trades(trades, market_events, token_events, cfg)
    coverage = _coverage(market_events, token_events, mapping, trades, scored)
    bucket = _score_bucket_summary(scored)
    component = _score_component_summary(scored, cfg)
    random_control = _random_score_control(scored, cfg)

    outputs = {
        "score_coverage": report_root / "score_coverage.csv",
        "scored_trade_ledger": report_root / "scored_trade_ledger.csv",
        "score_bucket_summary": report_root / "score_bucket_summary.csv",
        "score_component_summary": report_root / "score_component_summary.csv",
        "score_random_control": report_root / "score_random_control.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["score_coverage"], index=False)
    scored.to_csv(outputs["scored_trade_ledger"], index=False)
    bucket.to_csv(outputs["score_bucket_summary"], index=False)
    component.to_csv(outputs["score_component_summary"], index=False)
    random_control.to_csv(outputs["score_random_control"], index=False)
    _write_notes(outputs["candidate_notes"], coverage, component, random_control)
    return outputs
