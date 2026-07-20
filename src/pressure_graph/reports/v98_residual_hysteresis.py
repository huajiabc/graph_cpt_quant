"""BTC-beta-neutral residual alpha with frozen low-turnover portfolio policies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v97_direct_ml_alpha import (
    FEATURE_PATH,
    RANK_SOURCES,
    REGIME_FEATURES,
    V97Config,
    _fit_predict,
    _max_drawdown,
    _period,
    load_v97_source,
)


REPORT_ROOT = Path("reports/v9_8_residual_hysteresis")
SEED = 20260713
DERIVED_RANK_SOURCES = ["beta_30d", "trailing_residual_4h"]
V98_RANK_SOURCES = [*RANK_SOURCES, *DERIVED_RANK_SOURCES]
V98_RANK_FEATURES = [f"xrank_{col}" for col in V98_RANK_SOURCES]
V98_MODEL_FEATURES = [*V98_RANK_FEATURES, *REGIME_FEATURES]
POLICIES: dict[str, tuple[int, int]] = {
    "refresh": (5, 1),
    "sticky_10": (10, 1),
    "sticky_10_min2": (10, 2),
}
BOOKS = ("long_top5", "long_short_5x5")


@dataclass(frozen=True)
class V98Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = FEATURE_PATH
    sample_start: str = "2025-07-01"
    sample_end: str = "2026-06-01"
    label_col: str = "future_ret_4h"
    horizon_hours: int = 4
    eval_months: tuple[str, ...] = (
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
    )
    min_train_months: int = 6
    min_cross_section: int = 20
    beta_window_hours: int = 720
    beta_min_obs: int = 240
    beta_clip: tuple[float, float] = (-1.0, 3.0)
    top_k: int = 5
    costs_bps: tuple[int, ...] = (10, 20, 30, 50)
    random_iterations: int = 500
    seed: int = SEED


def load_v98_source(cfg: V98Config = V98Config()) -> pd.DataFrame:
    return load_v97_source(
        V97Config(
            feature_path=cfg.feature_path,
            label_col=cfg.label_col,
            horizon_hours=cfg.horizon_hours,
        )
    )


def _rolling_beta(group: pd.DataFrame, cfg: V98Config) -> pd.Series:
    symbol_return = pd.to_numeric(group["ret_4h"], errors="coerce")
    btc_return = pd.to_numeric(group["btc_ret_4h"], errors="coerce")
    rolling = symbol_return.rolling(cfg.beta_window_hours, min_periods=cfg.beta_min_obs)
    covariance = rolling.cov(btc_return)
    variance = btc_return.rolling(
        cfg.beta_window_hours, min_periods=cfg.beta_min_obs
    ).var()
    beta = covariance / variance.where(variance.abs().gt(1e-12))
    return beta.clip(*cfg.beta_clip)


def prepare_v98_dataset(
    frame: pd.DataFrame,
    cfg: V98Config = V98Config(),
) -> pd.DataFrame:
    data = frame.copy()
    data["feature_time"] = pd.to_datetime(data["feature_time"], utc=True, errors="coerce")
    start = pd.Timestamp(cfg.sample_start, tz="UTC")
    end = pd.Timestamp(cfg.sample_end, tz="UTC")
    warmup_start = start - pd.Timedelta(hours=cfg.beta_window_hours)
    mask = data["feature_time"].ge(warmup_start) & data["feature_time"].lt(end)
    mask &= data["feature_time"].dt.minute.eq(0)
    if "universe_dynamic_monthly_top30" in data:
        mask &= data["universe_dynamic_monthly_top30"].fillna(False).astype(bool)
    if "warmup_complete" in data:
        mask &= data["warmup_complete"].fillna(False).astype(bool)
    hourly = data.loc[mask].sort_values(["symbol", "feature_time"]).copy()
    hourly["future_return"] = pd.to_numeric(hourly[cfg.label_col], errors="coerce")
    hourly["ret_4h"] = pd.to_numeric(hourly["ret_4h"], errors="coerce")
    hourly["btc_ret_4h"] = pd.to_numeric(hourly["btc_ret_4h"], errors="coerce")
    beta = pd.Series(np.nan, index=hourly.index, dtype=float)
    for _, indices in hourly.groupby("symbol", sort=False).groups.items():
        idx = list(indices)
        beta.loc[idx] = _rolling_beta(hourly.loc[idx], cfg).to_numpy(dtype=float)
    hourly["beta_30d"] = beta
    hourly["trailing_residual_4h"] = (
        hourly["ret_4h"] - hourly["beta_30d"] * hourly["btc_ret_4h"]
    )

    anchor_mask = hourly["feature_time"].ge(start)
    anchor_mask &= hourly["feature_time"].dt.hour.mod(cfg.horizon_hours).eq(0)
    anchors = hourly.loc[anchor_mask].copy()
    btc = (
        anchors.loc[anchors["symbol"].astype(str).eq("BTCUSDT"), ["feature_time", "future_return"]]
        .drop_duplicates("feature_time")
        .rename(columns={"future_return": "future_btc_return"})
    )
    anchors = anchors.merge(btc, on="feature_time", how="left", validate="many_to_one")
    anchors["future_residual"] = (
        anchors["future_return"]
        - anchors["beta_30d"] * anchors["future_btc_return"]
    )
    anchors = anchors.dropna(
        subset=[
            "feature_time",
            "symbol",
            "future_return",
            "future_btc_return",
            "beta_30d",
            "future_residual",
        ]
    )
    counts = anchors.groupby("feature_time")["symbol"].transform("nunique")
    anchors = anchors[counts.ge(cfg.min_cross_section)].copy()
    anchors["log_turnover"] = np.log1p(
        pd.to_numeric(anchors["turnover"], errors="coerce").clip(lower=0)
    )
    for col in V98_RANK_SOURCES:
        numeric = pd.to_numeric(anchors.get(col), errors="coerce")
        anchors[f"xrank_{col}"] = (
            numeric.groupby(anchors["feature_time"]).rank(pct=True) - 0.5
        )
    for col in REGIME_FEATURES:
        anchors[col] = pd.to_numeric(anchors.get(col), errors="coerce")
    raw_market = anchors.groupby("feature_time")["future_return"].transform("mean")
    residual_market = anchors.groupby("feature_time")["future_residual"].transform("mean")
    anchors["target_direct_relative"] = anchors["future_return"] - raw_market
    anchors["target_residual_relative"] = anchors["future_residual"] - residual_market
    anchors["entry_month"] = anchors["feature_time"].dt.strftime("%Y-%m")
    anchors["period"] = anchors["entry_month"].map(_period)
    return anchors.sort_values(["feature_time", "symbol"]).reset_index(drop=True)


def _fit_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    kind: str,
    seed: int,
    shuffled: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    fit_train = train.copy()
    fit_train["target_relative"] = pd.to_numeric(fit_train[target_col], errors="coerce")
    return _fit_predict(
        fit_train,
        test,
        V98_MODEL_FEATURES,
        kind,
        seed,
        shuffled=shuffled,
    )


def walkforward_v98_predictions(
    data: pd.DataFrame,
    cfg: V98Config = V98Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants: dict[str, tuple[str, str, bool]] = {
        "ridge_residual": ("ridge", "target_residual_relative", False),
        "xgb_residual": ("xgb", "target_residual_relative", False),
        "xgb_residual_shuffled": ("xgb", "target_residual_relative", True),
        "ridge_direct_control": ("ridge", "target_direct_relative", False),
        "xgb_direct_control": ("xgb", "target_direct_relative", False),
    }
    base_cols = [
        "symbol",
        "feature_time",
        "entry_month",
        "period",
        "future_return",
        "future_btc_return",
        "future_residual",
        "target_direct_relative",
        "target_residual_relative",
        "beta_30d",
        *REGIME_FEATURES,
    ]
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for eval_month in cfg.eval_months:
        month_start = pd.Timestamp(f"{eval_month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        train = data[
            data["feature_time"].lt(month_start - pd.Timedelta(hours=cfg.horizon_hours))
            & data["entry_month"].lt(eval_month)
        ].copy()
        test = data[
            data["feature_time"].ge(month_start) & data["feature_time"].lt(month_end)
        ].copy()
        if test.empty or train["entry_month"].nunique() < cfg.min_train_months:
            continue
        momentum = test[base_cols].copy()
        momentum["model"] = "residual_momentum"
        momentum["score"] = pd.to_numeric(
            test["xrank_trailing_residual_4h"], errors="coerce"
        )
        prediction_frames.append(momentum)
        seed = cfg.seed + int(eval_month.replace("-", ""))
        for model_name, (kind, target, shuffled) in variants.items():
            score, importance = _fit_target(
                train,
                test,
                target,
                kind,
                seed,
                shuffled=shuffled,
            )
            current = test[base_cols].copy()
            current["model"] = model_name
            current["score"] = score
            prediction_frames.append(current)
            importance["model"] = model_name
            importance["eval_month"] = eval_month
            importance_frames.append(importance)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame()
    )
    return predictions, importance


def _select_ranked(
    ranked_symbols: list[str],
    previous_ages: dict[str, int],
    top_k: int,
    exit_size: int,
    min_hold: int,
) -> tuple[list[str], dict[str, int]]:
    available = set(ranked_symbols)
    rank = {symbol: i + 1 for i, symbol in enumerate(ranked_symbols)}
    retained = [
        symbol
        for symbol, age in previous_ages.items()
        if symbol in available and (age < min_hold or rank[symbol] <= exit_size)
    ]
    retained.sort(key=rank.__getitem__)
    selected = retained[:top_k]
    for symbol in ranked_symbols:
        if len(selected) >= top_k:
            break
        if symbol not in selected:
            selected.append(symbol)
    ages = {
        symbol: previous_ages.get(symbol, 0) + 1
        for symbol in selected
    }
    return selected, ages


def _turnover(previous: set[str], current: set[str]) -> float:
    if not current:
        return 0.0
    if not previous:
        return 1.0
    return 1.0 - len(previous & current) / len(current)


def build_v98_portfolio_ledger(
    predictions: pd.DataFrame,
    cfg: V98Config = V98Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    for model, model_data in predictions.groupby("model", sort=False):
        groups = list(
            model_data.sort_values("feature_time").groupby("feature_time", sort=True)
        )
        for policy, (exit_size, min_hold) in POLICIES.items():
            for book in BOOKS:
                long_ages: dict[str, int] = {}
                short_ages: dict[str, int] = {}
                for timestamp, group in groups:
                    ranked = group.dropna(subset=["score"]).sort_values(
                        ["score", "symbol"], ascending=[False, True]
                    )
                    if len(ranked) < cfg.top_k * 2:
                        continue
                    long_ranked = ranked["symbol"].astype(str).tolist()
                    short_ranked = list(reversed(long_ranked))
                    previous_long = set(long_ages)
                    previous_short = set(short_ages)
                    long_names, long_ages = _select_ranked(
                        long_ranked,
                        long_ages,
                        cfg.top_k,
                        exit_size,
                        min_hold,
                    )
                    if book == "long_short_5x5":
                        short_names, short_ages = _select_ranked(
                            short_ranked,
                            short_ages,
                            cfg.top_k,
                            exit_size,
                            min_hold,
                        )
                    else:
                        short_names = []
                        short_ages = {}
                    indexed = group.set_index(group["symbol"].astype(str), drop=False)
                    long_data = indexed.loc[long_names]
                    residual_market = float(
                        pd.to_numeric(group["future_residual"], errors="coerce").mean()
                    )
                    long_return = float(
                        pd.to_numeric(long_data["future_residual"], errors="coerce").mean()
                    )
                    long_turnover = _turnover(previous_long, set(long_names))
                    if book == "long_short_5x5":
                        short_data = indexed.loc[short_names]
                        short_return = float(
                            pd.to_numeric(
                                short_data["future_residual"], errors="coerce"
                            ).mean()
                        )
                        short_turnover = _turnover(previous_short, set(short_names))
                        gross_excess = 0.5 * (long_return - short_return)
                        turnover = 0.5 * (long_turnover + short_turnover)
                    else:
                        short_data = pd.DataFrame()
                        short_return = np.nan
                        short_turnover = 0.0
                        gross_excess = long_return - residual_market
                        turnover = long_turnover
                    score = pd.to_numeric(group["score"], errors="coerce")
                    target = pd.to_numeric(
                        group["target_residual_relative"], errors="coerce"
                    )
                    ic = (
                        float(score.corr(target, method="spearman"))
                        if score.nunique() > 1
                        else np.nan
                    )
                    row: dict[str, Any] = {
                        "model": model,
                        "book": book,
                        "policy": policy,
                        "feature_time": timestamp,
                        "entry_month": str(group["entry_month"].iloc[0]),
                        "period": str(group["period"].iloc[0]),
                        "cross_section": int(group["symbol"].nunique()),
                        "long_symbols": ";".join(sorted(long_names)),
                        "short_symbols": ";".join(sorted(short_names)),
                        "long_turnover": long_turnover,
                        "short_turnover": short_turnover,
                        "turnover": turnover,
                        "rank_ic": ic,
                        "long_residual_return": long_return,
                        "short_residual_return": short_return,
                        "residual_market_return": residual_market,
                        "gross_excess": gross_excess,
                    }
                    for cost in cfg.costs_bps:
                        paid = turnover * cost / 10_000.0
                        row[f"cost_{cost}"] = paid
                        row[f"net_excess_{cost}"] = gross_excess - paid
                    rows.append(row)
                    side_frames = [("long", long_data)]
                    if book == "long_short_5x5":
                        side_frames.append(("short", short_data))
                    for side, selected in side_frames:
                        sign = 1.0 if side == "long" else -1.0
                        gross_weight = 1.0 if book == "long_top5" else 0.5
                        for item in selected.itertuples(index=False):
                            if book == "long_top5":
                                contribution = (
                                    float(item.future_residual) - residual_market
                                ) / len(selected)
                            else:
                                contribution = (
                                    sign
                                    * gross_weight
                                    * float(item.future_residual)
                                    / len(selected)
                                )
                            holdings.append(
                                {
                                    "model": model,
                                    "book": book,
                                    "policy": policy,
                                    "feature_time": timestamp,
                                    "entry_month": row["entry_month"],
                                    "period": row["period"],
                                    "symbol": str(item.symbol),
                                    "side": side,
                                    "score": float(item.score),
                                    "beta_30d": float(item.beta_30d),
                                    "future_residual": float(item.future_residual),
                                    "gross_excess_contribution": contribution,
                                    "net_excess_contribution_20": (
                                        contribution
                                        - row["cost_20"]
                                        / (len(long_data) + len(short_data))
                                    ),
                                }
                            )
    return pd.DataFrame(rows), pd.DataFrame(holdings)


def summarize_v98_ledger(
    ledger: pd.DataFrame,
    costs_bps: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", ledger)]
    scopes.extend(
        (period, ledger[ledger["period"].eq(period)])
        for period in ("development", "validation", "holdout")
    )
    for scope, scoped in scopes:
        keys = ["model", "book", "policy"]
        for (model, book, policy), group in scoped.groupby(keys, sort=False):
            row: dict[str, Any] = {
                "scope": scope,
                "model": model,
                "book": book,
                "policy": policy,
                "periods": int(len(group)),
                "months": int(group["entry_month"].nunique()),
                "mean_rank_ic": float(
                    pd.to_numeric(group["rank_ic"], errors="coerce").mean()
                ),
                "gross_excess_sum": float(
                    pd.to_numeric(group["gross_excess"], errors="coerce").sum()
                ),
                "average_turnover": float(
                    pd.to_numeric(group["turnover"], errors="coerce").mean()
                ),
            }
            for cost in costs_bps:
                values = pd.to_numeric(group[f"net_excess_{cost}"], errors="coerce")
                row[f"net_excess_{cost}_sum"] = float(values.sum())
                row[f"net_excess_{cost}_max_drawdown"] = _max_drawdown(values)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_v98_monthly(
    ledger: pd.DataFrame,
    costs_bps: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model", "book", "policy", "entry_month"]
    for (model, book, policy, month), group in ledger.groupby(keys, sort=False):
        row: dict[str, Any] = {
            "model": model,
            "book": book,
            "policy": policy,
            "entry_month": month,
            "period": str(group["period"].iloc[0]),
            "periods": int(len(group)),
            "mean_rank_ic": float(
                pd.to_numeric(group["rank_ic"], errors="coerce").mean()
            ),
            "gross_excess_sum": float(
                pd.to_numeric(group["gross_excess"], errors="coerce").sum()
            ),
            "average_turnover": float(
                pd.to_numeric(group["turnover"], errors="coerce").mean()
            ),
        }
        for cost in costs_bps:
            row[f"net_excess_{cost}_sum"] = float(
                pd.to_numeric(group[f"net_excess_{cost}"], errors="coerce").sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def random_v98_controls(
    predictions: pd.DataFrame,
    cfg: V98Config,
) -> pd.DataFrame:
    source = predictions[predictions["model"].eq("residual_momentum")].copy()
    groups = []
    for timestamp, group in source.sort_values("feature_time").groupby(
        "feature_time", sort=True
    ):
        groups.append(
            (
                timestamp,
                group["symbol"].astype(str).to_numpy(),
                pd.to_numeric(group["future_residual"], errors="coerce").to_numpy(
                    dtype=float
                ),
                float(
                    pd.to_numeric(group["future_residual"], errors="coerce").mean()
                ),
            )
        )
    rows: list[dict[str, Any]] = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + iteration)
        state = {
            policy: {"long": {}, "short": {}}
            for policy in POLICIES
        }
        totals = {
            (book, policy): 0.0
            for book in BOOKS
            for policy in POLICIES
        }
        for _, symbols, residuals, market_residual in groups:
            permutation = rng.permutation(len(symbols))
            long_ranked = symbols[permutation].tolist()
            short_ranked = list(reversed(long_ranked))
            residual_map = dict(zip(symbols.tolist(), residuals.tolist(), strict=True))
            for policy, (exit_size, min_hold) in POLICIES.items():
                previous_long = set(state[policy]["long"])
                previous_short = set(state[policy]["short"])
                long_names, state[policy]["long"] = _select_ranked(
                    long_ranked,
                    state[policy]["long"],
                    cfg.top_k,
                    exit_size,
                    min_hold,
                )
                short_names, state[policy]["short"] = _select_ranked(
                    short_ranked,
                    state[policy]["short"],
                    cfg.top_k,
                    exit_size,
                    min_hold,
                )
                long_return = float(np.mean([residual_map[x] for x in long_names]))
                short_return = float(np.mean([residual_map[x] for x in short_names]))
                long_turnover = _turnover(previous_long, set(long_names))
                short_turnover = _turnover(previous_short, set(short_names))
                totals[("long_top5", policy)] += (
                    long_return - market_residual - long_turnover * 20 / 10_000.0
                )
                totals[("long_short_5x5", policy)] += (
                    0.5 * (long_return - short_return)
                    - 0.5 * (long_turnover + short_turnover) * 20 / 10_000.0
                )
        for (book, policy), value in totals.items():
            rows.append(
                {
                    "iteration": iteration,
                    "book": book,
                    "policy": policy,
                    "net_excess_20_sum": value,
                }
            )
    return pd.DataFrame(rows)


def v98_score_buckets(predictions: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    data = predictions.copy()
    percentile = data.groupby(["model", "feature_time"])["score"].rank(
        pct=True, method="first"
    )
    data["score_bucket"] = np.ceil(percentile * buckets).clip(1, buckets).astype(int)
    rows: list[dict[str, Any]] = []
    scopes = [("all", data)]
    scopes.extend(
        (period, data[data["period"].eq(period)])
        for period in ("development", "validation", "holdout")
    )
    for scope, scoped in scopes:
        for (model, bucket), group in scoped.groupby(
            ["model", "score_bucket"], sort=False
        ):
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "score_bucket": int(bucket),
                    "rows": int(len(group)),
                    "target_residual_mean": float(
                        pd.to_numeric(
                            group["target_residual_relative"], errors="coerce"
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def v98_break_even_cost(ledger: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "book", "policy"]
    rows: list[dict[str, Any]] = []
    for (model, book, policy), group in ledger.groupby(keys, sort=False):
        gross = float(pd.to_numeric(group["gross_excess"], errors="coerce").sum())
        turnover = float(pd.to_numeric(group["turnover"], errors="coerce").sum())
        rows.append(
            {
                "model": model,
                "book": book,
                "policy": policy,
                "gross_excess_sum": gross,
                "turnover_sum": turnover,
                "break_even_full_turnover_cost_bps": (
                    gross / turnover * 10_000.0 if turnover > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric(
    summary: pd.DataFrame,
    model: str,
    book: str,
    policy: str,
    scope: str,
    column: str,
) -> float:
    sample = summary[
        summary["model"].eq(model)
        & summary["book"].eq(book)
        & summary["policy"].eq(policy)
        & summary["scope"].eq(scope)
    ]
    return float(sample.iloc[0][column]) if not sample.empty else np.nan


def _bucket_monotonic(buckets: pd.DataFrame, model: str) -> bool:
    values = (
        buckets[buckets["scope"].eq("all") & buckets["model"].eq(model)]
        .sort_values("score_bucket")["target_residual_mean"]
        .to_numpy(dtype=float)
    )
    return bool(
        len(values) == 5
        and np.all(np.diff(values) >= 0)
        and values[-1] > values[0]
    )


def audit_v98(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    random_controls: pd.DataFrame,
    buckets: pd.DataFrame,
) -> pd.DataFrame:
    candidates = ("ridge_residual", "xgb_residual")
    direct_map = {
        "ridge_residual": "ridge_direct_control",
        "xgb_residual": "xgb_direct_control",
    }

    def base_gates(model: str, book: str, policy: str) -> dict[str, tuple[bool, float]]:
        validation = _metric(
            summary, model, book, policy, "validation", "net_excess_20_sum"
        )
        holdout = _metric(
            summary, model, book, policy, "holdout", "net_excess_20_sum"
        )
        full30 = _metric(summary, model, book, policy, "all", "net_excess_30_sum")
        turnover = _metric(summary, model, book, policy, "all", "average_turnover")
        model_monthly = monthly[
            monthly["model"].eq(model)
            & monthly["book"].eq(book)
            & monthly["policy"].eq(policy)
        ]
        month_values = pd.to_numeric(
            model_monthly["net_excess_20_sum"], errors="coerce"
        )
        positive_months = int(month_values.gt(0).sum())
        positive = month_values.clip(lower=0)
        month_share = (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
        )
        full20 = _metric(summary, model, book, policy, "all", "net_excess_20_sum")
        matched_random = random_controls[
            random_controls["book"].eq(book)
            & random_controls["policy"].eq(policy)
        ]["net_excess_20_sum"]
        random_percentile = (
            float(pd.to_numeric(matched_random, errors="coerce").lt(full20).mean())
            if len(matched_random)
            else np.nan
        )
        direct = direct_map[model]
        direct_full = _metric(
            summary, direct, book, policy, "all", "net_excess_20_sum"
        )
        direct_validation = _metric(
            summary, direct, book, policy, "validation", "net_excess_20_sum"
        )
        direct_holdout = _metric(
            summary, direct, book, policy, "holdout", "net_excess_20_sum"
        )
        return {
            "validation_net20_positive": (validation > 0, validation),
            "holdout_net20_positive": (holdout > 0, holdout),
            "full_net30_positive": (full30 > 0, full30),
            "three_positive_months": (positive_months >= 3, float(positive_months)),
            "month_contribution_below_35pct": (month_share <= 0.35, month_share),
            "average_turnover_at_most_35pct": (turnover <= 0.35, turnover),
            "matched_random_p90": (random_percentile >= 0.90, random_percentile),
            "beats_direct_full": (full20 > direct_full, full20 - direct_full),
            "beats_direct_validation": (
                validation > direct_validation,
                validation - direct_validation,
            ),
            "beats_direct_holdout": (
                holdout > direct_holdout,
                holdout - direct_holdout,
            ),
        }

    gate_cache: dict[tuple[str, str, str], dict[str, tuple[bool, float]]] = {}
    for model in candidates:
        for book in BOOKS:
            for policy in POLICIES:
                gate_cache[(model, book, policy)] = base_gates(model, book, policy)

    eligibility: dict[tuple[str, str, str], bool] = {}
    rows: list[dict[str, Any]] = []
    for model in candidates:
        for book in BOOKS:
            for policy in POLICIES:
                gates = dict(gate_cache[(model, book, policy)])
                if model == "xgb_residual":
                    ridge_eligible = all(
                        passed
                        for passed, _ in gate_cache[("ridge_residual", book, policy)].values()
                    )
                    xgb_validation = _metric(
                        summary, model, book, policy, "validation", "net_excess_20_sum"
                    )
                    xgb_holdout = _metric(
                        summary, model, book, policy, "holdout", "net_excess_20_sum"
                    )
                    ridge_validation = _metric(
                        summary,
                        "ridge_residual",
                        book,
                        policy,
                        "validation",
                        "net_excess_20_sum",
                    )
                    ridge_holdout = _metric(
                        summary,
                        "ridge_residual",
                        book,
                        policy,
                        "holdout",
                        "net_excess_20_sum",
                    )
                    shuffled_validation = _metric(
                        summary,
                        "xgb_residual_shuffled",
                        book,
                        policy,
                        "validation",
                        "net_excess_20_sum",
                    )
                    shuffled_holdout = _metric(
                        summary,
                        "xgb_residual_shuffled",
                        book,
                        policy,
                        "holdout",
                        "net_excess_20_sum",
                    )
                    monotonic = _bucket_monotonic(buckets, model)
                    gates.update(
                        {
                            "ridge_baseline_eligible": (
                                ridge_eligible,
                                float(ridge_eligible),
                            ),
                            "beats_ridge_validation": (
                                xgb_validation > ridge_validation,
                                xgb_validation - ridge_validation,
                            ),
                            "beats_ridge_holdout": (
                                xgb_holdout > ridge_holdout,
                                xgb_holdout - ridge_holdout,
                            ),
                            "beats_shuffle_validation": (
                                xgb_validation > shuffled_validation,
                                xgb_validation - shuffled_validation,
                            ),
                            "beats_shuffle_holdout": (
                                xgb_holdout > shuffled_holdout,
                                xgb_holdout - shuffled_holdout,
                            ),
                            "score_buckets_monotonic": (
                                monotonic,
                                float(monotonic),
                            ),
                        }
                    )
                eligible = all(passed for passed, _ in gates.values())
                eligibility[(model, book, policy)] = eligible
                for check, (passed, value) in gates.items():
                    rows.append(
                        {
                            "model": model,
                            "book": book,
                            "policy": policy,
                            "check": check,
                            "passed": bool(passed),
                            "value": value,
                            "eligible": eligible,
                        }
                    )
    ridge_any = any(
        value for (model, _, _), value in eligibility.items() if model == "ridge_residual"
    )
    xgb_any = any(
        value for (model, _, _), value in eligibility.items() if model == "xgb_residual"
    )
    if ridge_any and xgb_any:
        verdict = "complexity_candidate_research_only"
    elif ridge_any:
        verdict = "residual_baseline_candidate_no_complexity"
    else:
        verdict = "reject_residual_hysteresis_no_complexity_case"
    audit = pd.DataFrame(rows)
    audit["verdict"] = verdict
    return audit


def v98_symbol_contribution(holdings: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "book", "policy", "symbol", "side"]
    return (
        holdings.groupby(keys, as_index=False)["net_excess_contribution_20"]
        .sum()
        .sort_values(["model", "book", "policy", "net_excess_contribution_20"],
                     ascending=[True, True, True, False])
    )


def _write_notes(
    root: Path,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    verdict = str(audit["verdict"].iloc[0]) if not audit.empty else "not_run"
    lines = [
        "# v9.8 Market-Neutral Residual Alpha and Hysteresis",
        "",
        f"Status: `{verdict}`. Offline research only; no trading permission changes.",
        "",
        (
            f"Prepared rows: {len(data)}; decision timestamps: "
            f"{data['feature_time'].nunique()}; symbols: {data['symbol'].nunique()}."
        ),
        "",
        "## Residual model and turnover ladder",
    ]
    focal = summary[
        summary["scope"].isin(["all", "validation", "holdout"])
        & summary["model"].isin(["ridge_residual", "xgb_residual"])
    ].copy()
    for row in focal.sort_values(["model", "book", "policy", "scope"]).itertuples(
        index=False
    ):
        lines.append(
            f"- {row.model}/{row.book}/{row.policy}/{row.scope}: "
            f"net20={row.net_excess_20_sum:.4%}, net30={row.net_excess_30_sum:.4%}, "
            f"turnover={row.average_turnover:.3f}, IC={row.mean_rank_ic:.4f}."
        )
    eligible = audit[audit["eligible"].eq(True)][
        ["model", "book", "policy"]
    ].drop_duplicates()
    lines.extend(
        [
            "",
            "## Decision",
            (
                "- Eligible combinations: none."
                if eligible.empty
                else "- Eligible combinations: "
                + ", ".join("/".join(row) for row in eligible.itertuples(index=False, name=None))
                + "."
            ),
            "- Passing combinations would remain research-only because May is the only complete holdout month.",
            "- P2 and all live permissions remain unchanged.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v98_residual_hysteresis(
    cfg: V98Config = V98Config(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    data = prepare_v98_dataset(load_v98_source(cfg), cfg)
    predictions, importance = walkforward_v98_predictions(data, cfg)
    ledger, holdings = build_v98_portfolio_ledger(predictions, cfg)
    summary = summarize_v98_ledger(ledger, cfg.costs_bps)
    monthly = summarize_v98_monthly(ledger, cfg.costs_bps)
    random_controls = random_v98_controls(predictions, cfg)
    buckets = v98_score_buckets(predictions)
    break_even = v98_break_even_cost(ledger)
    audit = audit_v98(summary, monthly, random_controls, buckets)
    symbol_contribution = v98_symbol_contribution(holdings)
    dataset_summary = pd.DataFrame(
        [
            {
                "rows": len(data),
                "decision_timestamps": data["feature_time"].nunique(),
                "symbols": data["symbol"].nunique(),
                "start": data["feature_time"].min(),
                "end": data["feature_time"].max(),
                "beta_non_null_rate": data["beta_30d"].notna().mean(),
                "beta_mean": data["beta_30d"].mean(),
                "beta_median": data["beta_30d"].median(),
            }
        ]
    )
    outputs = {
        "dataset_summary": root / "dataset_summary.csv",
        "oos_predictions": root / "oos_predictions.parquet",
        "portfolio_ledger": root / "portfolio_ledger.csv",
        "selected_holdings": root / "selected_holdings.parquet",
        "model_summary": root / "model_summary.csv",
        "monthly_summary": root / "monthly_summary.csv",
        "random_controls": root / "random_portfolio_controls.csv",
        "score_buckets": root / "score_bucket_summary.csv",
        "break_even_cost": root / "break_even_cost_summary.csv",
        "feature_importance": root / "feature_importance.csv",
        "symbol_contribution": root / "symbol_contribution_summary.csv",
        "audit": root / "candidate_audit.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    dataset_summary.to_csv(outputs["dataset_summary"], index=False)
    predictions.to_parquet(outputs["oos_predictions"], index=False)
    ledger.to_csv(outputs["portfolio_ledger"], index=False)
    holdings.to_parquet(outputs["selected_holdings"], index=False)
    summary.to_csv(outputs["model_summary"], index=False)
    monthly.to_csv(outputs["monthly_summary"], index=False)
    random_controls.to_csv(outputs["random_controls"], index=False)
    buckets.to_csv(outputs["score_buckets"], index=False)
    break_even.to_csv(outputs["break_even_cost"], index=False)
    importance.to_csv(outputs["feature_importance"], index=False)
    symbol_contribution.to_csv(outputs["symbol_contribution"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    _write_notes(root, data, summary, audit)
    return outputs
