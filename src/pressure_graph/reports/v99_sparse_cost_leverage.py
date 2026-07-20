"""Sparse cost-aware residual portfolios with frozen leverage stress tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v9_9_sparse_cost_leverage")
PREDICTION_ROOT = Path("reports/v9_8_residual_hysteresis_4h")
SEED = 20260714
MODELS = ("ridge_residual", "xgb_residual", "xgb_residual_shuffled")
REAL_MODELS = ("ridge_residual", "xgb_residual")
BOOKS = ("long_sparse", "long_short_sparse")
HURDLES_BPS = (0.0, 2.5, 5.0, 10.0, 20.0)
LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class V99Config:
    prediction_root: Path = PREDICTION_ROOT
    report_root: Path = REPORT_ROOT
    horizon_hours: int = 4
    top_k: int = 5
    replacement_hurdle_bps: float = 20.0
    hurdles_bps: tuple[float, ...] = HURDLES_BPS
    costs_bps: tuple[int, ...] = (10, 20, 30, 50)
    leverages: tuple[float, ...] = LEVERAGES
    random_iterations: int = 500
    carry_bps_per_8h: float = 2.0
    seed: int = SEED


def load_v99_predictions(cfg: V99Config = V99Config()) -> pd.DataFrame:
    path = cfg.prediction_root / "oos_predictions.parquet"
    data = pd.read_parquet(path)
    data["feature_time"] = pd.to_datetime(data["feature_time"], utc=True, errors="coerce")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["target_residual_relative"] = pd.to_numeric(
        data["target_residual_relative"], errors="coerce"
    )
    return data[data["model"].isin(MODELS)].sort_values(
        ["model", "feature_time", "symbol"]
    )


def _select_sparse_side(
    scores: dict[str, float],
    previous: set[str],
    side: str,
    entry_hurdle: float,
    replacement_hurdle: float,
    top_k: int,
) -> set[str]:
    if side not in {"long", "short"}:
        raise ValueError(f"unknown side: {side}")
    finite = {symbol: score for symbol, score in scores.items() if np.isfinite(score)}
    if side == "long":
        selected = {
            symbol
            for symbol in previous
            if symbol in finite and finite[symbol] >= 0
        }
        ranked = sorted(finite, key=lambda symbol: (-finite[symbol], symbol))
        eligible = [symbol for symbol in ranked if finite[symbol] >= entry_hurdle]
    else:
        selected = {
            symbol
            for symbol in previous
            if symbol in finite and finite[symbol] <= 0
        }
        ranked = sorted(finite, key=lambda symbol: (finite[symbol], symbol))
        eligible = [symbol for symbol in ranked if finite[symbol] <= -entry_hurdle]
    if len(selected) > top_k:
        selected = set([symbol for symbol in ranked if symbol in selected][:top_k])
    for symbol in eligible:
        if len(selected) >= top_k:
            break
        selected.add(symbol)

    while len(selected) == top_k:
        outside = [symbol for symbol in eligible if symbol not in selected]
        if not outside:
            break
        if side == "long":
            incumbent = min(selected, key=lambda symbol: (finite[symbol], symbol))
            challenger = outside[0]
            improvement = finite[challenger] - finite[incumbent]
        else:
            incumbent = max(selected, key=lambda symbol: (finite[symbol], symbol))
            challenger = outside[0]
            improvement = finite[incumbent] - finite[challenger]
        if improvement < replacement_hurdle:
            break
        selected.remove(incumbent)
        selected.add(challenger)
    return selected


def _side_turnover(previous: set[str], current: set[str], top_k: int) -> float:
    entries = len(current - previous)
    exits = len(previous - current)
    return max(entries, exits) / top_k


def simulate_sparse_policy(
    predictions: pd.DataFrame,
    model: str,
    book: str,
    hurdle_bps: float,
    cfg: V99Config = V99Config(),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if book not in BOOKS:
        raise ValueError(f"unknown book: {book}")
    data = predictions[predictions["model"].eq(model)].copy()
    if start is not None:
        data = data[data["feature_time"].ge(pd.Timestamp(start, tz="UTC"))]
    if end is not None:
        data = data[data["feature_time"].lt(pd.Timestamp(end, tz="UTC"))]
    entry_hurdle = hurdle_bps / 10_000.0
    replacement_hurdle = cfg.replacement_hurdle_bps / 10_000.0
    previous_long: set[str] = set()
    previous_short: set[str] = set()
    rows: list[dict[str, Any]] = []
    for timestamp, group in data.groupby("feature_time", sort=True):
        scores = dict(
            zip(
                group["symbol"].astype(str),
                pd.to_numeric(group["score"], errors="coerce"),
                strict=True,
            )
        )
        targets = dict(
            zip(
                group["symbol"].astype(str),
                pd.to_numeric(group["target_residual_relative"], errors="coerce"),
                strict=True,
            )
        )
        current_long = _select_sparse_side(
            scores,
            previous_long,
            "long",
            entry_hurdle,
            replacement_hurdle,
            cfg.top_k,
        )
        long_turnover = _side_turnover(previous_long, current_long, cfg.top_k)
        if book == "long_short_sparse":
            current_short = _select_sparse_side(
                scores,
                previous_short,
                "short",
                entry_hurdle,
                replacement_hurdle,
                cfg.top_k,
            )
            short_turnover = _side_turnover(
                previous_short, current_short, cfg.top_k
            )
            gross_excess = 0.5 * (
                sum(targets[symbol] for symbol in current_long) / cfg.top_k
                - sum(targets[symbol] for symbol in current_short) / cfg.top_k
            )
            turnover = 0.5 * (long_turnover + short_turnover)
            gross_exposure = 0.5 * (
                len(current_long) + len(current_short)
            ) / cfg.top_k
        else:
            current_short = set()
            short_turnover = 0.0
            gross_excess = sum(targets[symbol] for symbol in current_long) / cfg.top_k
            turnover = long_turnover
            gross_exposure = len(current_long) / cfg.top_k
        row: dict[str, Any] = {
            "model": model,
            "book": book,
            "hurdle_bps": hurdle_bps,
            "feature_time": timestamp,
            "entry_month": str(group["entry_month"].iloc[0]),
            "period": str(group["period"].iloc[0]),
            "long_count": len(current_long),
            "short_count": len(current_short),
            "long_symbols": ";".join(sorted(current_long)),
            "short_symbols": ";".join(sorted(current_short)),
            "long_turnover": long_turnover,
            "short_turnover": short_turnover,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
            "gross_excess": gross_excess,
        }
        for cost in cfg.costs_bps:
            paid = turnover * cost / 10_000.0
            row[f"cost_{cost}"] = paid
            row[f"net_excess_{cost}"] = gross_excess - paid
        rows.append(row)
        previous_long = current_long
        previous_short = current_short
    return pd.DataFrame(rows)


def _ledger_summary(ledger: pd.DataFrame, costs_bps: tuple[int, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "periods": int(len(ledger)),
        "trade_periods": int(pd.to_numeric(ledger["turnover"], errors="coerce").gt(0).sum()),
        "average_turnover": float(
            pd.to_numeric(ledger["turnover"], errors="coerce").mean()
        ),
        "average_gross_exposure": float(
            pd.to_numeric(ledger["gross_exposure"], errors="coerce").mean()
        ),
        "gross_excess_sum": float(
            pd.to_numeric(ledger["gross_excess"], errors="coerce").sum()
        ),
    }
    for cost in costs_bps:
        row[f"net_excess_{cost}_sum"] = float(
            pd.to_numeric(ledger[f"net_excess_{cost}"], errors="coerce").sum()
        )
    return row


def select_development_hurdles(
    predictions: pd.DataFrame,
    cfg: V99Config = V99Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for book in BOOKS:
            candidates = []
            for hurdle in cfg.hurdles_bps:
                ledger = simulate_sparse_policy(
                    predictions,
                    model,
                    book,
                    hurdle,
                    cfg,
                    start="2026-01-01",
                    end="2026-03-01",
                )
                result = {
                    "model": model,
                    "book": book,
                    "hurdle_bps": hurdle,
                    **_ledger_summary(ledger, cfg.costs_bps),
                }
                candidates.append(result)
                selection_rows.append(result)
            ranked = sorted(
                candidates,
                key=lambda row: (
                    -row["net_excess_20_sum"],
                    row["average_turnover"],
                    -row["hurdle_bps"],
                ),
            )
            best = dict(ranked[0])
            best["development_selected_off"] = best["net_excess_20_sum"] <= 0
            selected_rows.append(best)
    selection = pd.DataFrame(selection_rows)
    selected = pd.DataFrame(selected_rows)
    selection = selection.merge(
        selected[["model", "book", "hurdle_bps"]].assign(selected_active_hurdle=True),
        on=["model", "book", "hurdle_bps"],
        how="left",
    )
    selection["selected_active_hurdle"] = selection["selected_active_hurdle"].eq(True)
    return selection, selected


def evaluate_selected_hurdles(
    predictions: pd.DataFrame,
    selected: pd.DataFrame,
    cfg: V99Config = V99Config(),
) -> pd.DataFrame:
    ledgers = []
    for row in selected.itertuples(index=False):
        ledger = simulate_sparse_policy(
            predictions,
            row.model,
            row.book,
            float(row.hurdle_bps),
            cfg,
            start="2026-03-01",
            end="2026-06-01",
        )
        ledger["development_selected_off"] = bool(row.development_selected_off)
        ledgers.append(ledger)
    return pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()


def summarize_evaluation(
    ledger: pd.DataFrame,
    cfg: V99Config = V99Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    keys = ["model", "book", "hurdle_bps", "development_selected_off"]
    for key, group in ledger.groupby(keys, sort=False):
        model, book, hurdle, selected_off = key
        scopes = [("evaluation", group)]
        scopes.extend(
            (period, group[group["period"].eq(period)])
            for period in ("validation", "holdout")
        )
        for scope, scoped in scopes:
            summary_rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "book": book,
                    "hurdle_bps": hurdle,
                    "development_selected_off": selected_off,
                    **_ledger_summary(scoped, cfg.costs_bps),
                }
            )
        for month, scoped in group.groupby("entry_month", sort=True):
            monthly_rows.append(
                {
                    "model": model,
                    "book": book,
                    "hurdle_bps": hurdle,
                    "development_selected_off": selected_off,
                    "entry_month": month,
                    "period": str(scoped["period"].iloc[0]),
                    **_ledger_summary(scoped, cfg.costs_bps),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)


def _random_group_records(data: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for timestamp, group in data.groupby("feature_time", sort=True):
        records.append(
            {
                "feature_time": timestamp,
                "period": str(group["period"].iloc[0]),
                "month": str(group["entry_month"].iloc[0]),
                "symbols": group["symbol"].astype(str).to_numpy(),
                "scores": pd.to_numeric(group["score"], errors="coerce").to_numpy(dtype=float),
                "targets": pd.to_numeric(
                    group["target_residual_relative"], errors="coerce"
                ).to_numpy(dtype=float),
            }
        )
    return records


def _update_random_state(
    record: dict[str, Any],
    shuffled_scores: np.ndarray,
    previous_long: set[str],
    previous_short: set[str],
    hurdle_bps: float,
    cfg: V99Config,
) -> tuple[set[str], set[str], dict[str, float]]:
    scores = dict(zip(record["symbols"].tolist(), shuffled_scores.tolist(), strict=True))
    targets = dict(zip(record["symbols"].tolist(), record["targets"].tolist(), strict=True))
    current_long = _select_sparse_side(
        scores,
        previous_long,
        "long",
        hurdle_bps / 10_000.0,
        cfg.replacement_hurdle_bps / 10_000.0,
        cfg.top_k,
    )
    current_short = _select_sparse_side(
        scores,
        previous_short,
        "short",
        hurdle_bps / 10_000.0,
        cfg.replacement_hurdle_bps / 10_000.0,
        cfg.top_k,
    )
    long_turnover = _side_turnover(previous_long, current_long, cfg.top_k)
    short_turnover = _side_turnover(previous_short, current_short, cfg.top_k)
    long_gross = sum(targets[symbol] for symbol in current_long) / cfg.top_k
    long_short_gross = 0.5 * (
        long_gross - sum(targets[symbol] for symbol in current_short) / cfg.top_k
    )
    metrics = {
        "long_sparse_net20": long_gross - long_turnover * 20 / 10_000.0,
        "long_sparse_turnover": long_turnover,
        "long_short_sparse_net20": long_short_gross
        - 0.5 * (long_turnover + short_turnover) * 20 / 10_000.0,
        "long_short_sparse_turnover": 0.5 * (long_turnover + short_turnover),
    }
    return current_long, current_short, metrics


def random_sparse_controls(
    predictions: pd.DataFrame,
    cfg: V99Config = V99Config(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in REAL_MODELS:
        records = _random_group_records(predictions[predictions["model"].eq(model)])
        development = [record for record in records if record["period"] == "development"]
        evaluation = [record for record in records if record["period"] != "development"]
        for iteration in range(cfg.random_iterations):
            rng = np.random.default_rng(cfg.seed + iteration)
            development_stats = {
                (book, hurdle): {"net20": 0.0, "turnover": 0.0, "periods": 0}
                for book in BOOKS
                for hurdle in cfg.hurdles_bps
            }
            dev_states = {
                hurdle: {"long": set(), "short": set()}
                for hurdle in cfg.hurdles_bps
            }
            for record in development:
                shuffled = rng.permutation(record["scores"])
                for hurdle in cfg.hurdles_bps:
                    state = dev_states[hurdle]
                    new_long, new_short, metrics = _update_random_state(
                        record,
                        shuffled,
                        state["long"],
                        state["short"],
                        hurdle,
                        cfg,
                    )
                    state["long"] = new_long
                    state["short"] = new_short
                    for book in BOOKS:
                        stats = development_stats[(book, hurdle)]
                        stats["net20"] += metrics[f"{book}_net20"]
                        stats["turnover"] += metrics[f"{book}_turnover"]
                        stats["periods"] += 1
            selected_hurdles: dict[str, float] = {}
            selected_active: dict[str, bool] = {}
            for book in BOOKS:
                ranked = sorted(
                    cfg.hurdles_bps,
                    key=lambda hurdle: (
                        -development_stats[(book, hurdle)]["net20"],
                        development_stats[(book, hurdle)]["turnover"]
                        / max(development_stats[(book, hurdle)]["periods"], 1),
                        -hurdle,
                    ),
                )
                selected_hurdles[book] = ranked[0]
                selected_active[book] = (
                    development_stats[(book, ranked[0])]["net20"] > 0
                )

            eval_states = {
                book: {"long": set(), "short": set()}
                for book in BOOKS
            }
            eval_stats = {
                book: {
                    "net20": 0.0,
                    "validation_net20": 0.0,
                    "holdout_net20": 0.0,
                    "turnover": 0.0,
                    "periods": 0,
                }
                for book in BOOKS
            }
            for record in evaluation:
                shuffled = rng.permutation(record["scores"])
                for book in BOOKS:
                    state = eval_states[book]
                    new_long, new_short, metrics = _update_random_state(
                        record,
                        shuffled,
                        state["long"],
                        state["short"],
                        selected_hurdles[book],
                        cfg,
                    )
                    state["long"] = new_long
                    state["short"] = new_short if book == "long_short_sparse" else set()
                    value = metrics[f"{book}_net20"]
                    eval_stats[book]["net20"] += value
                    eval_stats[book][f"{record['period']}_net20"] += value
                    eval_stats[book]["turnover"] += metrics[f"{book}_turnover"]
                    eval_stats[book]["periods"] += 1
            for book in BOOKS:
                stats = eval_stats[book]
                rows.append(
                    {
                        "model": model,
                        "iteration": iteration,
                        "book": book,
                        "selected_hurdle_bps": selected_hurdles[book],
                        "development_selected_active": selected_active[book],
                        "diagnostic_evaluation_net20": stats["net20"],
                        "honest_evaluation_net20": (
                            stats["net20"] if selected_active[book] else 0.0
                        ),
                        "validation_net20": stats["validation_net20"],
                        "holdout_net20": stats["holdout_net20"],
                        "average_turnover": stats["turnover"]
                        / max(stats["periods"], 1),
                    }
                )
    return pd.DataFrame(rows)


def audit_sparse_candidates(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    selected: pd.DataFrame,
    random_controls: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def metric(model: str, book: str, scope: str, column: str) -> float:
        sample = summary[
            summary["model"].eq(model)
            & summary["book"].eq(book)
            & summary["scope"].eq(scope)
        ]
        return float(sample.iloc[0][column]) if not sample.empty else np.nan

    for model in REAL_MODELS:
        for book in BOOKS:
            selection = selected[
                selected["model"].eq(model) & selected["book"].eq(book)
            ].iloc[0]
            validation = metric(model, book, "validation", "net_excess_20_sum")
            holdout = metric(model, book, "holdout", "net_excess_20_sum")
            evaluation30 = metric(model, book, "evaluation", "net_excess_30_sum")
            evaluation20 = metric(model, book, "evaluation", "net_excess_20_sum")
            turnover = metric(model, book, "evaluation", "average_turnover")
            model_monthly = monthly[
                monthly["model"].eq(model) & monthly["book"].eq(book)
            ]
            month_values = pd.to_numeric(
                model_monthly["net_excess_20_sum"], errors="coerce"
            )
            positive_months = int(month_values.gt(0).sum())
            positive = month_values.clip(lower=0)
            concentration = (
                float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
            )
            random = random_controls[
                random_controls["model"].eq(model)
                & random_controls["book"].eq(book)
            ]["honest_evaluation_net20"]
            random_percentile = float(
                pd.to_numeric(random, errors="coerce").lt(evaluation20).mean()
            )
            shuffled = metric(
                "xgb_residual_shuffled", book, "evaluation", "net_excess_20_sum"
            )
            gates = {
                "development_selected_active": (
                    not bool(selection.development_selected_off),
                    float(not bool(selection.development_selected_off)),
                ),
                "validation_net20_positive": (validation > 0, validation),
                "holdout_net20_positive": (holdout > 0, holdout),
                "evaluation_net30_positive": (evaluation30 > 0, evaluation30),
                "two_positive_evaluation_months": (
                    positive_months >= 2,
                    float(positive_months),
                ),
                "average_turnover_at_most_10pct": (turnover <= 0.10, turnover),
                "matched_random_p90": (
                    random_percentile >= 0.90,
                    random_percentile,
                ),
                "beats_shuffled_label": (evaluation20 > shuffled, evaluation20 - shuffled),
                "month_contribution_below_60pct": (
                    concentration <= 0.60,
                    concentration,
                ),
            }
            eligible = all(passed for passed, _ in gates.values())
            for check, (passed, value) in gates.items():
                rows.append(
                    {
                        "model": model,
                        "book": book,
                        "hurdle_bps": float(selection.hurdle_bps),
                        "check": check,
                        "passed": bool(passed),
                        "value": value,
                        "eligible": eligible,
                    }
                )
    audit = pd.DataFrame(rows)
    audit["verdict"] = (
        "sparse_research_candidate_only"
        if audit["eligible"].any()
        else "reject_sparse_cost_aware_alpha"
    )
    return audit


def _compound_metrics(returns: pd.Series, periods_per_year: float) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    ruin = False
    for value in values:
        if value <= -1.0:
            equity = 0.0
            ruin = True
        elif not ruin:
            equity *= 1.0 + value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0 if peak > 0 else -1.0)
    quantile = float(np.quantile(values, 0.05)) if len(values) else np.nan
    tail = values[values <= quantile] if len(values) else np.array([])
    return {
        "periods": len(values),
        "additive_return": float(values.sum()),
        "equity_multiple": float(equity),
        "annualized_volatility": float(
            np.std(values, ddof=1) * np.sqrt(periods_per_year)
        ) if len(values) > 1 else np.nan,
        "max_drawdown": float(max_drawdown),
        "worst_period": float(np.min(values)) if len(values) else np.nan,
        "expected_shortfall_5pct": float(np.mean(tail)) if len(tail) else np.nan,
        "periods_below_minus_20pct": int(np.sum(values <= -0.20)),
        "periods_below_minus_50pct": int(np.sum(values <= -0.50)),
        "periods_below_minus_80pct": int(np.sum(values <= -0.80)),
        "periods_below_minus_100pct": int(np.sum(values <= -1.00)),
        "ruin": ruin,
    }


def leverage_stress(
    evaluation_ledger: pd.DataFrame,
    audit: pd.DataFrame,
    cfg: V99Config = V99Config(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods_per_year = 365.0 * 24.0 / cfg.horizon_hours
    candidate_keys = set(
        audit[audit["eligible"].eq(True)][["model", "book"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for (model, book), ledger in evaluation_ledger.groupby(["model", "book"], sort=False):
        base_candidate = (model, book) in candidate_keys
        for leverage in cfg.leverages:
            transaction_only = leverage * pd.to_numeric(
                ledger["net_excess_20"], errors="coerce"
            )
            carry = (
                leverage
                * pd.to_numeric(ledger["gross_exposure"], errors="coerce")
                * cfg.carry_bps_per_8h
                / 10_000.0
                * cfg.horizon_hours
                / 8.0
            )
            for case, returns in (
                ("transaction_only_upper_bound", transaction_only),
                ("carry_stress", transaction_only - carry),
            ):
                metrics = _compound_metrics(returns, periods_per_year)
                rows.append(
                    {
                        "model": model,
                        "book": book,
                        "hurdle_bps": float(ledger["hurdle_bps"].iloc[0]),
                        "development_selected_off": bool(
                            ledger["development_selected_off"].iloc[0]
                        ),
                        "leverage": leverage,
                        "case": case,
                        "base_unlevered_candidate": base_candidate,
                        **metrics,
                    }
                )
    result = pd.DataFrame(rows)
    result["leverage_eligible"] = False
    for (model, book), group in result.groupby(["model", "book"]):
        base_candidate = bool(group["base_unlevered_candidate"].iloc[0])
        for leverage in cfg.leverages:
            stress = group[
                group["leverage"].eq(leverage) & group["case"].eq("carry_stress")
            ].iloc[0]
            if leverage == 5.0:
                eligible = False
            elif leverage == 2.0:
                eligible = (
                    base_candidate
                    and stress.max_drawdown >= -0.30
                    and stress.periods_below_minus_50pct == 0
                )
            elif leverage == 3.0:
                eligible = (
                    base_candidate
                    and stress.max_drawdown >= -0.45
                    and stress.periods_below_minus_50pct == 0
                )
            else:
                eligible = base_candidate
            mask = (
                result["model"].eq(model)
                & result["book"].eq(book)
                & result["leverage"].eq(leverage)
            )
            result.loc[mask, "leverage_eligible"] = eligible
    return result


def _write_notes(
    root: Path,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
    leverage: pd.DataFrame,
) -> None:
    verdict = str(audit["verdict"].iloc[0]) if not audit.empty else "not_run"
    lines = [
        "# v9.9 Sparse Cost-Aware Residual Trading and Leverage",
        "",
        f"Status: `{verdict}`. Offline research only; no live permission changes.",
        "",
        "## Development selection and evaluation",
    ]
    for row in selected.sort_values(["model", "book"]).itertuples(index=False):
        evaluation = summary[
            summary["model"].eq(row.model)
            & summary["book"].eq(row.book)
            & summary["scope"].eq("evaluation")
        ].iloc[0]
        lines.append(
            f"- {row.model}/{row.book}: hurdle={row.hurdle_bps:g}bp, "
            f"development_off={bool(row.development_selected_off)}, "
            f"Mar-May net20={evaluation.net_excess_20_sum:.4%}, "
            f"net30={evaluation.net_excess_30_sum:.4%}, "
            f"turnover={evaluation.average_turnover:.3f}, "
            f"exposure={evaluation.average_gross_exposure:.3f}."
        )
    eligible = audit[audit["eligible"].eq(True)][["model", "book"]].drop_duplicates()
    lines.extend(
        [
            "",
            "## Leverage decision",
            (
                "- No unlevered combination passed; every leverage level is ineligible."
                if eligible.empty
                else "- Unlevered eligible combinations: "
                + ", ".join("/".join(row) for row in eligible.itertuples(index=False, name=None))
                + "."
            ),
        ]
    )
    focal = leverage[
        leverage["model"].isin(REAL_MODELS)
        & leverage["case"].eq("carry_stress")
        & leverage["leverage"].isin([1.0, 2.0, 3.0, 5.0])
    ]
    for row in focal.sort_values(["model", "book", "leverage"]).itertuples(index=False):
        lines.append(
            f"- {row.model}/{row.book}/{row.leverage:g}x: "
            f"additive={row.additive_return:.4%}, equity={row.equity_multiple:.3f}, "
            f"max_drawdown={row.max_drawdown:.2%}, worst={row.worst_period:.2%}, "
            f"ruin={bool(row.ruin)}."
        )
    lines.extend(
        [
            "- `5x` is tail-risk stress only and cannot be recommended from this sample.",
            "- P2 and all live permissions remain unchanged.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v99_sparse_cost_leverage(
    cfg: V99Config = V99Config(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    predictions = load_v99_predictions(cfg)
    threshold_grid, selected = select_development_hurdles(predictions, cfg)
    evaluation_ledger = evaluate_selected_hurdles(predictions, selected, cfg)
    summary, monthly = summarize_evaluation(evaluation_ledger, cfg)
    random_controls = random_sparse_controls(predictions, cfg)
    audit = audit_sparse_candidates(
        summary, monthly, selected, random_controls
    )
    leverage = leverage_stress(evaluation_ledger, audit, cfg)
    outputs = {
        "threshold_grid": root / "development_threshold_grid.csv",
        "selected_hurdles": root / "selected_hurdles.csv",
        "evaluation_ledger": root / "evaluation_ledger.csv",
        "evaluation_summary": root / "evaluation_summary.csv",
        "monthly_summary": root / "monthly_summary.csv",
        "random_controls": root / "random_sparse_controls.csv",
        "candidate_audit": root / "candidate_audit.csv",
        "leverage_stress": root / "leverage_stress.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    threshold_grid.to_csv(outputs["threshold_grid"], index=False)
    selected.to_csv(outputs["selected_hurdles"], index=False)
    evaluation_ledger.to_csv(outputs["evaluation_ledger"], index=False)
    summary.to_csv(outputs["evaluation_summary"], index=False)
    monthly.to_csv(outputs["monthly_summary"], index=False)
    random_controls.to_csv(outputs["random_controls"], index=False)
    audit.to_csv(outputs["candidate_audit"], index=False)
    leverage.to_csv(outputs["leverage_stress"], index=False)
    _write_notes(root, selected, summary, audit, leverage)
    return outputs
