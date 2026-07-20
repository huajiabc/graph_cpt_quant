"""Direct cross-sectional alpha model ladder with strict monthly walk-forward QA."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v9_7_direct_ml_alpha")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
SEED = 20260713

MOMENTUM_FEATURES = [
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "volatility_1h",
    "volatility_4h",
    "volume_z_1h",
    "volume_z_4h",
    "ret_4h_percentile",
    "volume_1h_percentile",
    "volume_4h_percentile",
    "log_turnover",
]
CROWDING_FEATURES = [
    "funding_z",
    "funding_percentile",
    "oi_value_delta_z_1h",
    "oi_value_delta_z_4h",
    "oi_value_delta_1h_percentile",
    "oi_value_delta_4h_percentile",
]
REGIME_FEATURES = [
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_volatility_4h",
    "btc_volatility_percentile",
]
RANK_SOURCES = [*MOMENTUM_FEATURES, *CROWDING_FEATURES]
RANK_FEATURES = [f"xrank_{col}" for col in RANK_SOURCES]
MODEL_FEATURES = [*RANK_FEATURES, *REGIME_FEATURES]


@dataclass(frozen=True)
class V97Config:
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
    top_k: int = 5
    costs_bps: tuple[int, ...] = (10, 20, 30, 50)
    random_iterations: int = 200
    seed: int = SEED


def _period(month: str) -> str:
    if month <= "2026-02":
        return "development"
    if month <= "2026-04":
        return "validation"
    return "holdout"


def _source_columns(cfg: V97Config) -> list[str]:
    return list(
        dict.fromkeys(
            [
                "symbol",
                "feature_time",
                "universe_dynamic_monthly_top30",
                "warmup_complete",
                cfg.label_col,
                "turnover",
                *[col for col in MOMENTUM_FEATURES if col != "log_turnover"],
                *CROWDING_FEATURES,
                *REGIME_FEATURES,
            ]
        )
    )


def load_v97_source(cfg: V97Config = V97Config()) -> pd.DataFrame:
    return pd.read_parquet(
        cfg.feature_path,
        columns=_source_columns(cfg),
        filters=[("universe_dynamic_monthly_top30", "=", True)],
    )


def prepare_v97_dataset(frame: pd.DataFrame, cfg: V97Config = V97Config()) -> pd.DataFrame:
    data = frame.copy()
    data["feature_time"] = pd.to_datetime(data["feature_time"], utc=True, errors="coerce")
    start = pd.Timestamp(cfg.sample_start, tz="UTC")
    end = pd.Timestamp(cfg.sample_end, tz="UTC")
    mask = data["feature_time"].ge(start) & data["feature_time"].lt(end)
    mask &= data["feature_time"].dt.minute.eq(0)
    mask &= data["feature_time"].dt.hour.mod(cfg.horizon_hours).eq(0)
    if "universe_dynamic_monthly_top30" in data:
        mask &= data["universe_dynamic_monthly_top30"].fillna(False).astype(bool)
    if "warmup_complete" in data:
        mask &= data["warmup_complete"].fillna(False).astype(bool)
    data = data.loc[mask].copy()
    data["future_return"] = pd.to_numeric(data[cfg.label_col], errors="coerce")
    data = data.dropna(subset=["feature_time", "symbol", "future_return"])
    counts = data.groupby("feature_time")["symbol"].transform("nunique")
    data = data[counts.ge(cfg.min_cross_section)].copy()
    data["log_turnover"] = np.log1p(pd.to_numeric(data["turnover"], errors="coerce").clip(lower=0))
    for col in RANK_SOURCES:
        numeric = pd.to_numeric(data.get(col), errors="coerce")
        data[f"xrank_{col}"] = numeric.groupby(data["feature_time"]).rank(pct=True) - 0.5
    for col in REGIME_FEATURES:
        data[col] = pd.to_numeric(data.get(col), errors="coerce")
    market = data.groupby("feature_time")["future_return"].transform("mean")
    data["target_relative"] = data["future_return"] - market
    data["entry_month"] = data["feature_time"].dt.strftime("%Y-%m")
    data["period"] = data["entry_month"].map(_period)
    return data.sort_values(["feature_time", "symbol"]).reset_index(drop=True)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    kind: str,
    seed: int,
    shuffled: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    from sklearn.impute import SimpleImputer

    y = train["target_relative"].to_numpy(dtype=float).copy()
    if shuffled:
        rng = np.random.default_rng(seed)
        for _, idx in train.groupby("entry_month", sort=False).indices.items():
            y[idx] = rng.permutation(y[idx])
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x_train = imputer.fit_transform(train[feature_cols])
    x_test = imputer.transform(test[feature_cols])
    encoded = imputer.get_feature_names_out(feature_cols)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = Ridge(alpha=10.0)
        model.fit(x_train, y)
        score = model.predict(x_test)
        importance = np.abs(np.asarray(model.coef_, dtype=float))
    else:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=240,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=50,
            reg_alpha=1.0,
            reg_lambda=10.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
        )
        model.fit(x_train, y, verbose=False)
        score = model.predict(x_test)
        importance = np.asarray(model.feature_importances_, dtype=float)
    detail = pd.DataFrame({"feature": encoded, "importance": importance})
    return np.asarray(score, dtype=float), detail


def walkforward_predictions(
    data: pd.DataFrame,
    cfg: V97Config = V97Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants: dict[str, tuple[str, list[str], bool]] = {
        "ridge": ("ridge", MODEL_FEATURES, False),
        "xgb_shallow": ("xgb", MODEL_FEATURES, False),
        "xgb_shuffled": ("xgb", MODEL_FEATURES, True),
        "xgb_no_momentum": (
            "xgb",
            [col for col in MODEL_FEATURES if col not in {f"xrank_{x}" for x in MOMENTUM_FEATURES}],
            False,
        ),
        "xgb_no_crowding": (
            "xgb",
            [col for col in MODEL_FEATURES if col not in {f"xrank_{x}" for x in CROWDING_FEATURES}],
            False,
        ),
        "xgb_no_regime": ("xgb", [col for col in MODEL_FEATURES if col not in REGIME_FEATURES], False),
    }
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    base_cols = [
        "symbol",
        "feature_time",
        "entry_month",
        "period",
        "future_return",
        "target_relative",
        *REGIME_FEATURES,
    ]
    for eval_month in cfg.eval_months:
        month_start = pd.Timestamp(f"{eval_month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthBegin(1)
        train = data[
            data["feature_time"].lt(month_start - pd.Timedelta(hours=cfg.horizon_hours))
            & data["entry_month"].lt(eval_month)
        ].copy()
        test = data[data["feature_time"].ge(month_start) & data["feature_time"].lt(month_end)].copy()
        if test.empty or train["entry_month"].nunique() < cfg.min_train_months:
            continue
        momentum = test[base_cols].copy()
        momentum["model"] = "momentum_rank"
        momentum["score"] = pd.to_numeric(test["xrank_ret_4h_percentile"], errors="coerce")
        prediction_frames.append(momentum)
        for model_name, (kind, features, shuffled) in variants.items():
            score, importance = _fit_predict(
                train,
                test,
                features,
                kind,
                cfg.seed + int(eval_month.replace("-", "")),
                shuffled=shuffled,
            )
            current = test[base_cols].copy()
            current["model"] = model_name
            current["score"] = score
            prediction_frames.append(current)
            importance["model"] = model_name
            importance["eval_month"] = eval_month
            importance_frames.append(importance)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return predictions, importance


def build_portfolio_ledger(
    predictions: pd.DataFrame,
    top_k: int = 5,
    costs_bps: tuple[int, ...] = (10, 20, 30, 50),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    holdings: list[dict[str, Any]] = []
    for model, model_data in predictions.groupby("model", sort=False):
        previous: set[str] = set()
        for timestamp, group in model_data.sort_values("feature_time").groupby("feature_time", sort=True):
            ranked = group.dropna(subset=["score"]).sort_values(
                ["score", "symbol"], ascending=[False, True]
            )
            selected = ranked.head(min(top_k, len(ranked))).copy()
            if selected.empty:
                continue
            names = set(selected["symbol"].astype(str))
            turnover = 1.0 if not previous else 1.0 - len(previous & names) / max(len(names), 1)
            previous = names
            market_ret = float(pd.to_numeric(group["future_return"], errors="coerce").mean())
            gross_ret = float(pd.to_numeric(selected["future_return"], errors="coerce").mean())
            target = pd.to_numeric(group["target_relative"], errors="coerce")
            score = pd.to_numeric(group["score"], errors="coerce")
            ic = float(score.corr(target, method="spearman")) if score.nunique() > 1 else np.nan
            row: dict[str, Any] = {
                "model": model,
                "feature_time": timestamp,
                "entry_month": str(group["entry_month"].iloc[0]),
                "period": str(group["period"].iloc[0]),
                "cross_section": int(group["symbol"].nunique()),
                "selected_count": int(len(selected)),
                "selected_symbols": ";".join(sorted(names)),
                "turnover": float(turnover),
                "rank_ic": ic,
                "gross_ret": gross_ret,
                "market_ret": market_ret,
                "gross_excess": gross_ret - market_ret,
            }
            for cost in costs_bps:
                paid = turnover * cost / 10_000.0
                row[f"cost_{cost}"] = paid
                row[f"net_ret_{cost}"] = gross_ret - paid
                row[f"net_excess_{cost}"] = gross_ret - market_ret - paid
            rows.append(row)
            weight = 1.0 / len(selected)
            for item in selected.itertuples(index=False):
                holding = {
                    "model": model,
                    "feature_time": timestamp,
                    "entry_month": row["entry_month"],
                    "period": row["period"],
                    "symbol": str(item.symbol),
                    "score": float(item.score),
                    "future_return": float(item.future_return),
                    "market_ret": market_ret,
                    "weight": weight,
                    "turnover": float(turnover),
                }
                for cost in costs_bps:
                    holding[f"net_excess_contribution_{cost}"] = (
                        (float(item.future_return) - market_ret) * weight
                        - row[f"cost_{cost}"] * weight
                    )
                holdings.append(holding)
    return pd.DataFrame(rows), pd.DataFrame(holdings)


def _max_drawdown(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).clip(lower=-0.99)
    equity = (1.0 + values).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else np.nan


def summarize_ledger(ledger: pd.DataFrame, costs_bps: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("all", ledger)]
    scopes.extend((period, ledger[ledger["period"].eq(period)]) for period in ("development", "validation", "holdout"))
    for scope, scoped in scopes:
        for model, group in scoped.groupby("model", sort=False):
            row: dict[str, Any] = {
                "scope": scope,
                "model": model,
                "periods": int(len(group)),
                "months": int(group["entry_month"].nunique()),
                "mean_rank_ic": float(pd.to_numeric(group["rank_ic"], errors="coerce").mean()),
                "median_rank_ic": float(pd.to_numeric(group["rank_ic"], errors="coerce").median()),
                "ic_positive_rate": float(pd.to_numeric(group["rank_ic"], errors="coerce").gt(0).mean()),
                "gross_ret_sum": float(pd.to_numeric(group["gross_ret"], errors="coerce").sum()),
                "market_ret_sum": float(pd.to_numeric(group["market_ret"], errors="coerce").sum()),
                "gross_excess_sum": float(pd.to_numeric(group["gross_excess"], errors="coerce").sum()),
                "average_turnover": float(pd.to_numeric(group["turnover"], errors="coerce").mean()),
            }
            for cost in costs_bps:
                row[f"net_ret_{cost}_sum"] = float(pd.to_numeric(group[f"net_ret_{cost}"], errors="coerce").sum())
                row[f"net_excess_{cost}_sum"] = float(pd.to_numeric(group[f"net_excess_{cost}"], errors="coerce").sum())
                row[f"net_ret_{cost}_max_drawdown"] = _max_drawdown(group[f"net_ret_{cost}"])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_monthly(ledger: pd.DataFrame, costs_bps: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for (model, month), group in ledger.groupby(["model", "entry_month"], sort=False):
        row: dict[str, Any] = {
            "model": model,
            "entry_month": month,
            "period": str(group["period"].iloc[0]),
            "periods": int(len(group)),
            "mean_rank_ic": float(pd.to_numeric(group["rank_ic"], errors="coerce").mean()),
            "gross_excess_sum": float(pd.to_numeric(group["gross_excess"], errors="coerce").sum()),
            "average_turnover": float(pd.to_numeric(group["turnover"], errors="coerce").mean()),
        }
        for cost in costs_bps:
            row[f"net_excess_{cost}_sum"] = float(pd.to_numeric(group[f"net_excess_{cost}"], errors="coerce").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def random_topk_controls(
    predictions: pd.DataFrame,
    cfg: V97Config,
) -> pd.DataFrame:
    source = predictions[predictions["model"].eq("momentum_rank")].copy()
    groups = []
    for timestamp, group in source.sort_values("feature_time").groupby("feature_time", sort=True):
        groups.append(
            (
                timestamp,
                group["symbol"].astype(str).to_numpy(),
                pd.to_numeric(group["future_return"], errors="coerce").to_numpy(dtype=float),
                float(pd.to_numeric(group["future_return"], errors="coerce").mean()),
            )
        )
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + iteration)
        previous: set[str] = set()
        total = 0.0
        for _, symbols, returns, market_ret in groups:
            count = min(cfg.top_k, len(symbols))
            idx = rng.choice(len(symbols), size=count, replace=False)
            selected = set(symbols[idx])
            turnover = 1.0 if not previous else 1.0 - len(previous & selected) / max(count, 1)
            previous = selected
            total += float(np.nanmean(returns[idx])) - market_ret - turnover * 20 / 10_000.0
        rows.append({"iteration": iteration, "net_excess_20_sum": total})
    return pd.DataFrame(rows)


def score_bucket_summary(predictions: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    data = predictions.copy()
    percentile = data.groupby(["model", "feature_time"])["score"].rank(
        pct=True, method="first"
    )
    data["score_bucket"] = np.ceil(percentile * buckets).clip(1, buckets).astype(int)
    rows = []
    scopes = [("all", data)]
    scopes.extend(
        (period, data[data["period"].eq(period)])
        for period in ("development", "validation", "holdout")
    )
    for scope, scoped in scopes:
        for (model, bucket), group in scoped.groupby(["model", "score_bucket"], sort=False):
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "score_bucket": int(bucket),
                    "rows": int(len(group)),
                    "timestamps": int(group["feature_time"].nunique()),
                    "target_relative_mean": float(
                        pd.to_numeric(group["target_relative"], errors="coerce").mean()
                    ),
                    "future_return_mean": float(
                        pd.to_numeric(group["future_return"], errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def break_even_cost_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("all", ledger)]
    scopes.extend(
        (period, ledger[ledger["period"].eq(period)])
        for period in ("development", "validation", "holdout")
    )
    for scope, scoped in scopes:
        for model, group in scoped.groupby("model", sort=False):
            gross = float(pd.to_numeric(group["gross_excess"], errors="coerce").sum())
            turnover = float(pd.to_numeric(group["turnover"], errors="coerce").sum())
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "gross_excess_sum": gross,
                    "turnover_sum": turnover,
                    "break_even_full_turnover_cost_bps": (
                        gross / turnover * 10_000.0 if turnover > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def regime_summary(predictions: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    context = predictions[
        ["feature_time", "btc_ret_4h", "btc_volatility_percentile"]
    ].drop_duplicates("feature_time")
    direction = np.where(
        pd.to_numeric(context["btc_ret_4h"], errors="coerce").ge(0),
        "btc_up",
        "btc_down",
    )
    volatility = np.where(
        pd.to_numeric(context["btc_volatility_percentile"], errors="coerce").ge(75),
        "highvol",
        "lowvol",
    )
    context["market_regime"] = pd.Series(direction, index=context.index).str.cat(
        pd.Series(volatility, index=context.index), sep="_"
    )
    data = ledger.merge(
        context[["feature_time", "market_regime"]],
        on="feature_time",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for (model, regime), group in data.groupby(["model", "market_regime"], sort=False):
        rows.append(
            {
                "model": model,
                "market_regime": regime,
                "periods": int(len(group)),
                "months": int(group["entry_month"].nunique()),
                "mean_rank_ic": float(pd.to_numeric(group["rank_ic"], errors="coerce").mean()),
                "gross_excess_sum": float(
                    pd.to_numeric(group["gross_excess"], errors="coerce").sum()
                ),
                "net_excess_20_sum": float(
                    pd.to_numeric(group["net_excess_20"], errors="coerce").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def symbol_contribution_summary(holdings: pd.DataFrame) -> pd.DataFrame:
    data = (
        holdings.groupby(["model", "symbol"], as_index=False)["net_excess_contribution_20"]
        .sum()
        .rename(columns={"net_excess_contribution_20": "net_excess_20_contribution"})
    )
    data["positive_contribution"] = data["net_excess_20_contribution"].clip(lower=0)
    total_positive = data.groupby("model")["positive_contribution"].transform("sum")
    data["positive_contribution_share"] = np.where(
        total_positive.gt(0), data["positive_contribution"] / total_positive, np.nan
    )
    data["contribution_rank"] = data.groupby("model")["net_excess_20_contribution"].rank(
        method="first", ascending=False
    )
    return data.sort_values(["model", "contribution_rank"])


def complexity_audit(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    random_controls: pd.DataFrame,
) -> pd.DataFrame:
    def metric(model: str, scope: str, col: str) -> float:
        sample = summary[summary["model"].eq(model) & summary["scope"].eq(scope)]
        return float(sample.iloc[0][col]) if not sample.empty else np.nan

    xgb_all = metric("xgb_shallow", "all", "net_excess_20_sum")
    ridge_all = metric("ridge", "all", "net_excess_20_sum")
    random_percentile = float(random_controls["net_excess_20_sum"].lt(xgb_all).mean()) if len(random_controls) else np.nan
    xgb_monthly = monthly[monthly["model"].eq("xgb_shallow")]
    positive_months = int(pd.to_numeric(xgb_monthly["net_excess_20_sum"], errors="coerce").gt(0).sum())
    total = float(pd.to_numeric(xgb_monthly["net_excess_20_sum"], errors="coerce").sum())
    best = float(pd.to_numeric(xgb_monthly["net_excess_20_sum"], errors="coerce").clip(lower=0).max()) if len(xgb_monthly) else np.nan
    month_share = best / total if total > 0 else np.inf
    ablation_values = [
        metric(model, "all", "net_excess_20_sum")
        for model in ("xgb_no_momentum", "xgb_no_crowding", "xgb_no_regime")
    ]
    gates = {
        "validation_net20_positive": metric("xgb_shallow", "validation", "net_excess_20_sum") > 0,
        "holdout_net20_positive": metric("xgb_shallow", "holdout", "net_excess_20_sum") > 0,
        "validation_beats_ridge_net20": metric("xgb_shallow", "validation", "net_excess_20_sum") > metric("ridge", "validation", "net_excess_20_sum"),
        "holdout_beats_ridge_net20": metric("xgb_shallow", "holdout", "net_excess_20_sum") > metric("ridge", "holdout", "net_excess_20_sum"),
        "validation_beats_ridge_ic": metric("xgb_shallow", "validation", "mean_rank_ic") > metric("ridge", "validation", "mean_rank_ic"),
        "holdout_beats_ridge_ic": metric("xgb_shallow", "holdout", "mean_rank_ic") > metric("ridge", "holdout", "mean_rank_ic"),
        "full_net30_positive": metric("xgb_shallow", "all", "net_excess_30_sum") > 0,
        "three_positive_months": positive_months >= 3,
        "month_contribution_below_35pct": month_share <= 0.35,
        "random_p90_pass": random_percentile >= 0.90,
        "beats_shuffled_label": xgb_all > metric("xgb_shuffled", "all", "net_excess_20_sum"),
        "all_ablations_remain_positive": all(value > 0 for value in ablation_values),
    }
    passed = sum(bool(value) for value in gates.values())
    if xgb_all <= 0 and ridge_all <= 0:
        verdict = "reject_direct_alpha_after_cost"
    elif xgb_all <= ridge_all:
        verdict = "no_case_for_added_complexity"
    elif passed == len(gates):
        verdict = "complexity_candidate_research_only"
    else:
        verdict = "weak_complexity_evidence_not_eligible"
    rows = [
        {"check": name, "passed": bool(value), "value": np.nan, "verdict": verdict}
        for name, value in gates.items()
    ]
    rows.extend(
        [
            {"check": "xgb_full_net20", "passed": xgb_all > 0, "value": xgb_all, "verdict": verdict},
            {"check": "ridge_full_net20", "passed": ridge_all > 0, "value": ridge_all, "verdict": verdict},
            {"check": "xgb_random_percentile", "passed": random_percentile >= 0.90, "value": random_percentile, "verdict": verdict},
            {"check": "xgb_positive_months", "passed": positive_months >= 3, "value": positive_months, "verdict": verdict},
            {"check": "xgb_best_month_share", "passed": month_share <= 0.35, "value": month_share, "verdict": verdict},
        ]
    )
    return pd.DataFrame(rows)


def _notes(
    root: Path,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    verdict = str(audit["verdict"].iloc[0]) if not audit.empty else "not_run"
    lines = [
        "# v9.7 Direct Cross-Sectional ML Alpha",
        "",
        f"Status: `{verdict}`. Offline research only; no trading permission changes.",
        "",
        f"Prepared rows: {len(data)}; decision timestamps: {data['feature_time'].nunique()}; symbols: {data['symbol'].nunique()}.",
        "",
        "## Model ladder",
    ]
    focal = summary[summary["scope"].isin(["all", "validation", "holdout"])].copy()
    for row in focal.sort_values(["model", "scope"]).itertuples(index=False):
        lines.append(
            f"- {row.model}/{row.scope}: periods={row.periods}, mean_ic={row.mean_rank_ic:.4f}, "
            f"net20_excess={row.net_excess_20_sum:.4%}, net30_excess={row.net_excess_30_sum:.4%}, "
            f"turnover={row.average_turnover:.3f}."
        )
    failed = audit[audit["passed"].eq(False)]["check"].astype(str).tolist()
    lines.extend(
        [
            "",
            "## Complexity decision",
            f"- Failed gates: {', '.join(failed) if failed else 'none'}.",
            "- One complete holdout month is insufficient for promotion even if all numeric gates pass.",
            "- P2 and all live permissions remain unchanged.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v97_direct_ml_alpha(cfg: V97Config = V97Config()) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    data = prepare_v97_dataset(load_v97_source(cfg), cfg)
    predictions, importance = walkforward_predictions(data, cfg)
    ledger, holdings = build_portfolio_ledger(predictions, cfg.top_k, cfg.costs_bps)
    summary = summarize_ledger(ledger, cfg.costs_bps)
    monthly = summarize_monthly(ledger, cfg.costs_bps)
    random_controls = random_topk_controls(predictions, cfg)
    buckets = score_bucket_summary(predictions)
    break_even = break_even_cost_summary(ledger)
    regimes = regime_summary(predictions, ledger)
    symbol_contribution = symbol_contribution_summary(holdings)
    audit = complexity_audit(summary, monthly, random_controls)
    dataset_summary = pd.DataFrame(
        [
            {
                "rows": len(data),
                "decision_timestamps": data["feature_time"].nunique(),
                "symbols": data["symbol"].nunique(),
                "start": data["feature_time"].min(),
                "end": data["feature_time"].max(),
                "min_cross_section": data.groupby("feature_time")["symbol"].nunique().min(),
                "max_cross_section": data.groupby("feature_time")["symbol"].nunique().max(),
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
        "random_controls": root / "random_top5_controls.csv",
        "score_buckets": root / "score_bucket_summary.csv",
        "break_even_cost": root / "break_even_cost_summary.csv",
        "regime_summary": root / "regime_summary.csv",
        "symbol_contribution": root / "symbol_contribution_summary.csv",
        "feature_importance": root / "feature_importance.csv",
        "complexity_audit": root / "complexity_audit.csv",
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
    regimes.to_csv(outputs["regime_summary"], index=False)
    symbol_contribution.to_csv(outputs["symbol_contribution"], index=False)
    importance.to_csv(outputs["feature_importance"], index=False)
    audit.to_csv(outputs["complexity_audit"], index=False)
    _notes(root, data, summary, audit)
    return outputs


__all__ = [
    "V97Config",
    "build_portfolio_ledger",
    "complexity_audit",
    "prepare_v97_dataset",
    "walkforward_predictions",
    "write_v97_direct_ml_alpha",
]
