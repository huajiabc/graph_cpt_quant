"""Independent raw-data and statistical audit for v14.9 FSS3."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v132_tg1_forward_temporal_extension import (
    RAW_BYBIT_ROOT,
    RECENT_ROOT,
    hourly_bybit_prices,
    load_v132_bybit_klines,
)


PANEL_PATH = Path(
    "reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet"
)
REPORT_ROOT = Path("reports/v14_9_funding_sign_turnover_cap")
PORTFOLIO_PATH = REPORT_ROOT / "weekly_portfolio.parquet"
AUDIT_DOC = Path("docs/v150_fss3_independent_audit_findings_2026_07_15.md")


@dataclass(frozen=True)
class V150AuditConfig:
    panel_path: Path = PANEL_PATH
    portfolio_path: Path = PORTFOLIO_PATH
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    minimum_side_breadth: int = 4
    transition_turnover_cap: float = 0.70
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    bootstrap_iterations: int = 10_000
    null_iterations: int = 5_000
    bootstrap_block_weeks: int = 4
    bisection_iterations: int = 64
    seed: int = 20260718
    structural_tolerance: float = 5e-12


def _raw_funding_arrays(symbol: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frames = []
    for path in (
        RAW_BYBIT_ROOT / "funding" / f"{symbol}.parquet",
        RECENT_ROOT / "bybit_funding" / f"{symbol}.parquet",
    ):
        if path.exists():
            frames.append(
                pd.read_parquet(
                    path, columns=["funding_time", "funding_rate_settled"]
                )
            )
    if not frames:
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    funding = pd.concat(frames, ignore_index=True)
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True, errors="coerce"
    ).dt.floor("s")
    funding["funding_rate_settled"] = pd.to_numeric(
        funding["funding_rate_settled"], errors="coerce"
    )
    funding = (
        funding.dropna(subset=["funding_time", "funding_rate_settled"])
        .drop_duplicates("funding_time", keep="last")
        .sort_values("funding_time")
    )
    return (
        pd.DatetimeIndex(funding["funding_time"]),
        funding["funding_rate_settled"].to_numpy(dtype=float),
    )


def _window_sum(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    *,
    left_side: str,
    right_side: str,
) -> float:
    left = int(times.searchsorted(start, side=left_side))
    right = int(times.searchsorted(stop, side=right_side))
    return float(values[left:right].sum())


def _audit_turnover(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        sum(
            abs(left.get(symbol, 0.0) - right.get(symbol, 0.0))
            for symbol in set(left) | set(right)
        )
    )


def _audit_neutralize(
    local: pd.DataFrame,
    alt_weights: dict[str, float],
) -> dict[str, float]:
    indexed = local.set_index("symbol")
    alt = {
        symbol: float(weight)
        for symbol, weight in alt_weights.items()
        if symbol in indexed.index and abs(weight) > 1e-16
    }
    hedge = -float(
        sum(weight * float(indexed.at[symbol, "btc_beta"]) for symbol, weight in alt.items())
    )
    gross = float(sum(abs(weight) for weight in alt.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in alt.items()}
    weights[BTC] = hedge / gross
    return weights


def _audit_components(
    local: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, float]:
    indexed = local.set_index("symbol")
    alt_symbols = [symbol for symbol in weights if symbol != BTC]
    price = float(
        sum(
            weights[symbol] * float(indexed.at[symbol, "price_return"])
            for symbol in alt_symbols
        )
        + weights[BTC] * float(local.iloc[0]["btc_return"])
    )
    funding = float(
        sum(
            -weights[symbol] * float(indexed.at[symbol, "future_funding"])
            for symbol in alt_symbols
        )
        - weights[BTC] * float(local.iloc[0]["btc_future_funding"])
    )
    residual_beta = float(
        sum(
            weights[symbol] * float(indexed.at[symbol, "btc_beta"])
            for symbol in alt_symbols
        )
        + weights[BTC]
    )
    return {
        "price_return_audit": price,
        "funding_return_audit": funding,
        "gross_return_audit": price + funding,
        "gross_notional_audit": float(sum(abs(weight) for weight in weights.values())),
        "residual_btc_beta_audit": residual_beta,
    }


def _audit_capped_step(
    local: pd.DataFrame,
    previous: dict[str, float] | None,
    target: dict[str, float],
    cap: float,
    bisection_iterations: int,
) -> tuple[dict[str, float], float, float, float]:
    if previous is None:
        return target, 1.0, _audit_turnover({}, target), 0.0
    direct_turnover = _audit_turnover(previous, target)
    if direct_turnover <= cap + 1e-14:
        return target, 1.0, direct_turnover, 0.0
    current_symbols = set(local["symbol"].astype(str))
    previous_alt = {
        symbol: weight
        for symbol, weight in previous.items()
        if symbol != BTC and symbol in current_symbols
    }
    target_alt = {symbol: weight for symbol, weight in target.items() if symbol != BTC}

    def point(fraction: float) -> tuple[dict[str, float], float]:
        raw = {
            symbol: (1.0 - fraction) * previous_alt.get(symbol, 0.0)
            + fraction * target_alt.get(symbol, 0.0)
            for symbol in set(previous_alt) | set(target_alt)
        }
        weights = _audit_neutralize(local, raw)
        return weights, _audit_turnover(previous, weights)

    base, base_turnover = point(0.0)
    if base_turnover >= cap:
        return base, 0.0, base_turnover, max(0.0, base_turnover - cap)
    low = 0.0
    high = 1.0
    best = base
    best_turnover = base_turnover
    for _ in range(bisection_iterations):
        middle = (low + high) / 2.0
        weights, turnover = point(middle)
        if turnover <= cap:
            low = middle
            best = weights
            best_turnover = turnover
        else:
            high = middle
    return best, low, best_turnover, max(0.0, best_turnover - cap)


def _alternate_block_bootstrap(
    values: np.ndarray,
    cfg: V150AuditConfig,
) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed)
    block_count = int(np.ceil(len(values) / cfg.bootstrap_block_weeks))
    offsets = np.arange(cfg.bootstrap_block_weeks)
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draws[iteration] = values[indices.ravel()[: len(values)]].mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _array_neutralize(alt: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, float]:
    hedge = -float(np.dot(alt, beta))
    gross = float(np.abs(alt).sum() + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return np.zeros_like(alt), 0.0
    return alt / gross, hedge / gross


def _array_capped_step(
    previous_alt: np.ndarray | None,
    previous_btc: float,
    target_alt: np.ndarray,
    target_btc: float,
    beta: np.ndarray,
    mask: np.ndarray,
    cfg: V150AuditConfig,
) -> tuple[np.ndarray, float, float]:
    if previous_alt is None:
        turnover = float(np.abs(target_alt).sum() + abs(target_btc))
        return target_alt, target_btc, turnover

    def turnover(alt: np.ndarray, btc: float) -> float:
        return float(np.abs(alt - previous_alt).sum() + abs(btc - previous_btc))

    direct = turnover(target_alt, target_btc)
    if direct <= cfg.transition_turnover_cap + 1e-14:
        return target_alt, target_btc, direct
    retained = previous_alt * mask

    def point(fraction: float) -> tuple[np.ndarray, float, float]:
        alt, btc = _array_neutralize(
            (1.0 - fraction) * retained + fraction * target_alt, beta
        )
        return alt, btc, turnover(alt, btc)

    base_alt, base_btc, base_turnover = point(0.0)
    if base_turnover >= cfg.transition_turnover_cap:
        return base_alt, base_btc, base_turnover
    low = 0.0
    high = 1.0
    best_alt = base_alt
    best_btc = base_btc
    best_turnover = base_turnover
    for _ in range(cfg.bisection_iterations):
        middle = (low + high) / 2.0
        alt, btc, current_turnover = point(middle)
        if current_turnover <= cfg.transition_turnover_cap:
            low = middle
            best_alt = alt
            best_btc = btc
            best_turnover = current_turnover
        else:
            high = middle
    return best_alt, best_btc, best_turnover


def _alternate_null(
    panel: pd.DataFrame,
    portfolio: pd.DataFrame,
    cfg: V150AuditConfig,
) -> tuple[float, float, float, float]:
    all_symbols = sorted(panel["symbol"].astype(str).unique())
    symbol_to_index = {symbol: index for index, symbol in enumerate(all_symbols)}
    groups = {
        pd.Timestamp(entry): local
        for entry, local in panel.groupby("entry_time", sort=True, observed=True)
    }
    weeks = []
    for row in portfolio.sort_values("entry_time").itertuples(index=False):
        local = groups[pd.Timestamp(row.entry_time)]
        usable = local.dropna(subset=["price_return", "future_funding", "btc_beta"])
        indexed = usable.set_index("symbol")
        symbols = sorted(usable["symbol"].astype(str).unique())
        indices = np.asarray([symbol_to_index[symbol] for symbol in symbols], dtype=int)
        beta = np.zeros(len(all_symbols), dtype=float)
        price = np.zeros(len(all_symbols), dtype=float)
        funding = np.zeros(len(all_symbols), dtype=float)
        beta[indices] = indexed.loc[symbols, "btc_beta"].to_numpy(dtype=float)
        price[indices] = indexed.loc[symbols, "price_return"].to_numpy(dtype=float)
        funding[indices] = indexed.loc[symbols, "future_funding"].to_numpy(dtype=float)
        mask = np.zeros(len(all_symbols), dtype=bool)
        mask[indices] = True
        weeks.append(
            (
                indices,
                beta,
                price,
                funding,
                mask,
                float(usable.iloc[0]["btc_return"]),
                float(usable.iloc[0]["btc_future_funding"]),
                int(row.negative_breadth),
                int(row.positive_breadth),
            )
        )
    rng = np.random.default_rng(cfg.seed + 1)
    null_means = np.empty(cfg.null_iterations, dtype=float)
    for iteration in range(cfg.null_iterations):
        previous_alt: np.ndarray | None = None
        previous_btc = 0.0
        gross_returns = []
        turnovers = []
        for (
            indices,
            beta,
            price,
            funding,
            mask,
            btc_return,
            btc_funding,
            long_count,
            short_count,
        ) in weeks:
            chosen = indices[
                rng.choice(len(indices), long_count + short_count, replace=False)
            ]
            raw = np.zeros(len(all_symbols), dtype=float)
            raw[chosen[:long_count]] = 0.5 / long_count
            raw[chosen[long_count:]] = -0.5 / short_count
            target_alt, target_btc = _array_neutralize(raw, beta)
            alt, btc, current_turnover = _array_capped_step(
                previous_alt,
                previous_btc,
                target_alt,
                target_btc,
                beta,
                mask,
                cfg,
            )
            gross_returns.append(
                float(np.dot(alt, price) + btc * btc_return - np.dot(alt, funding) - btc * btc_funding)
            )
            turnovers.append(current_turnover)
            previous_alt = alt
            previous_btc = btc
        if previous_alt is not None:
            turnovers[-1] += float(np.abs(previous_alt).sum() + abs(previous_btc))
        null_means[iteration] = float(
            np.mean(np.asarray(gross_returns) - cfg.one_way_cost * np.asarray(turnovers))
        )
    observed = float(portfolio["primary_net_return"].mean())
    return (
        float(100 * np.mean(null_means <= observed)),
        float(null_means.mean()),
        float(np.quantile(null_means, 0.95)),
        float(np.quantile(null_means, 0.99)),
    )


def run_v150_fss3_independent_audit(
    cfg: V150AuditConfig = V150AuditConfig(),
) -> dict[str, Path]:
    panel = pd.read_parquet(cfg.panel_path)
    portfolio = pd.read_parquet(cfg.portfolio_path)
    for frame in (panel, portfolio):
        for column in ("entry_time", "exit_time", "month_start"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    panel = panel.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    portfolio = portfolio.sort_values("entry_time").reset_index(drop=True)
    groups = {
        pd.Timestamp(entry): local.copy()
        for entry, local in panel.groupby("entry_time", sort=True, observed=True)
    }

    prices = hourly_bybit_prices(load_v132_bybit_klines()).pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    )
    funding_cache = {
        symbol: _raw_funding_arrays(symbol)
        for symbol in sorted(set(panel["symbol"].astype(str)) | {BTC})
    }
    raw_diffs = {
        "score": [],
        "future_funding": [],
        "price": [],
        "btc_price": [],
        "btc_funding": [],
    }
    btc_times, btc_rates = funding_cache[BTC]
    for row in panel.itertuples(index=False):
        times, rates = funding_cache[str(row.symbol)]
        score = _window_sum(
            times,
            rates,
            row.entry_time - pd.Timedelta(days=7),
            row.entry_time,
            left_side="left",
            right_side="left",
        )
        future_funding = _window_sum(
            times,
            rates,
            row.entry_time,
            row.exit_time,
            left_side="right",
            right_side="right",
        )
        price_return = float(
            prices.at[row.exit_time, row.symbol] / prices.at[row.entry_time, row.symbol] - 1.0
        )
        btc_price = float(
            prices.at[row.exit_time, BTC] / prices.at[row.entry_time, BTC] - 1.0
        )
        btc_funding = _window_sum(
            btc_times,
            btc_rates,
            row.entry_time,
            row.exit_time,
            left_side="right",
            right_side="right",
        )
        raw_diffs["score"].append(abs(score - float(row.score_7d)))
        raw_diffs["future_funding"].append(
            abs(future_funding - float(row.future_funding))
        )
        raw_diffs["price"].append(abs(price_return - float(row.price_return)))
        raw_diffs["btc_price"].append(abs(btc_price - float(row.btc_return)))
        raw_diffs["btc_funding"].append(
            abs(btc_funding - float(row.btc_future_funding))
        )

    reconstructed = []
    contributions: dict[str, float] = {}
    previous: dict[str, float] | None = None
    previous_entry: pd.Timestamp | None = None
    for saved in portfolio.itertuples(index=False):
        entry = pd.Timestamp(saved.entry_time)
        local = groups[entry]
        eligible = local.dropna(
            subset=["score_7d", "price_return", "future_funding", "btc_beta"]
        )
        negative = sorted(
            eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique()
        )
        positive = sorted(
            eligible.loc[eligible["score_7d"].gt(0), "symbol"].astype(str).unique()
        )
        if (
            len(negative) < cfg.minimum_side_breadth
            or len(positive) < cfg.minimum_side_breadth
        ):
            raise RuntimeError(f"Insufficient independent breadth at {entry}")
        raw = {symbol: 0.5 / len(negative) for symbol in negative}
        raw.update({symbol: -0.5 / len(positive) for symbol in positive})
        target = _audit_neutralize(local, raw)
        if previous_entry is not None and entry - previous_entry > pd.Timedelta(
            days=7, minutes=1
        ):
            raise RuntimeError("Audit path contains an unhandled weekly gap")
        weights, fraction, rebalance_turnover, cap_breach = _audit_capped_step(
            local,
            previous,
            target,
            cfg.transition_turnover_cap,
            cfg.bisection_iterations,
        )
        components = _audit_components(local, weights)
        selected_long = "|".join(
            sorted(symbol for symbol, weight in weights.items() if symbol != BTC and weight > 0)
        )
        selected_short = "|".join(
            sorted(symbol for symbol, weight in weights.items() if symbol != BTC and weight < 0)
        )
        if selected_long != saved.selected_long_symbols or selected_short != saved.selected_short_symbols:
            raise RuntimeError(f"Executed selection mismatch at {entry}")
        indexed = local.set_index("symbol")
        for symbol, weight in weights.items():
            if symbol == BTC:
                contribution = weight * float(
                    local.iloc[0]["btc_return"] - local.iloc[0]["btc_future_funding"]
                )
            else:
                contribution = weight * float(
                    indexed.at[symbol, "price_return"]
                    - indexed.at[symbol, "future_funding"]
                )
            contributions[symbol] = contributions.get(symbol, 0.0) + contribution
        reconstructed.append(
            {
                "entry_time": entry,
                "executed_target_fraction_audit": fraction,
                "target_tracking_l1_audit": _audit_turnover(weights, target),
                "rebalance_turnover_audit": rebalance_turnover,
                "cap_breach_audit": cap_breach,
                "realized_turnover_audit": rebalance_turnover,
                **components,
            }
        )
        previous = weights
        previous_entry = entry
    recomputed = pd.DataFrame(reconstructed)
    recomputed.loc[recomputed.index[-1], "realized_turnover_audit"] += sum(
        abs(weight) for weight in previous.values()
    )
    recomputed["primary_net_return_audit"] = (
        recomputed["gross_return_audit"]
        - cfg.one_way_cost * recomputed["realized_turnover_audit"]
    )
    recomputed["stress_net_return_audit"] = (
        recomputed["gross_return_audit"]
        - cfg.stress_one_way_cost * recomputed["realized_turnover_audit"]
    )
    merged = portfolio.merge(recomputed, on="entry_time", how="inner")
    compare_fields = {
        "executed_target_fraction": "executed_target_fraction_audit",
        "target_tracking_l1": "target_tracking_l1_audit",
        "rebalance_turnover": "rebalance_turnover_audit",
        "cap_breach": "cap_breach_audit",
        "realized_turnover": "realized_turnover_audit",
        "price_return": "price_return_audit",
        "funding_return": "funding_return_audit",
        "gross_return": "gross_return_audit",
        "primary_net_return": "primary_net_return_audit",
        "stress_net_return": "stress_net_return_audit",
        "gross_notional": "gross_notional_audit",
        "residual_btc_beta": "residual_btc_beta_audit",
    }
    field_diffs = {
        field: float((merged[field] - merged[audit_field]).abs().max())
        for field, audit_field in compare_fields.items()
    }
    raw_max = {name: float(max(values)) for name, values in raw_diffs.items()}
    structural_max = max(*field_diffs.values(), *raw_max.values())
    structural_pass = bool(structural_max <= cfg.structural_tolerance)

    bootstrap_low, bootstrap_high = _alternate_block_bootstrap(
        portfolio["primary_net_return"].to_numpy(dtype=float), cfg
    )
    null_percentile, null_mean, null_95, null_99 = _alternate_null(
        panel, portfolio, cfg
    )
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    states = portfolio.groupby("breadth_state", observed=True)["primary_net_return"].mean()
    months = portfolio.groupby("month_start", observed=True)["primary_net_return"].sum()
    positive_months = months[months.gt(0)]
    month_concentration = float(positive_months.max() / positive_months.sum())
    leave_one_month_out = [
        float(portfolio.loc[portfolio["month_start"].ne(month), "primary_net_return"].mean())
        for month in months.index
    ]
    recent_six = portfolio.tail(6)["primary_net_return"]
    pnl_curve = portfolio["primary_net_return"].cumsum()
    drawdown = pnl_curve - pnl_curve.cummax().clip(lower=0)
    btc_weekly = np.asarray(
        [groups[pd.Timestamp(entry)]["btc_return"].iloc[0] for entry in portfolio["entry_time"]],
        dtype=float,
    )
    price_values = portfolio["price_return"].to_numpy(dtype=float)
    primary_values = portfolio["primary_net_return"].to_numpy(dtype=float)
    btc_variance = float(np.var(btc_weekly, ddof=1))
    positive_contributions = {symbol: value for symbol, value in contributions.items() if value > 0}
    positive_coins = {
        symbol: value for symbol, value in positive_contributions.items() if symbol != BTC
    }
    gate_recheck = bool(
        len(portfolio) >= 45
        and portfolio["month_start"].nunique() >= 11
        and portfolio["period"].value_counts().get("validation", 0) >= 10
        and portfolio["period"].value_counts().get("holdout", 0) >= 10
        and portfolio["realized_turnover"].mean() <= 0.75
        and portfolio.loc[portfolio["cap_applicable"], "rebalance_turnover"].max()
        <= cfg.transition_turnover_cap + 1e-10
        and portfolio["cap_breach"].max() <= 1e-10
        and portfolio["funding_return"].mean() > 0
        and portfolio["stress_net_return"].mean() > 0
        and periods.min() > 0
        and states.min() > 0
        and bootstrap_low > 0
        and null_percentile >= 95
        and month_concentration <= 0.35
        and periods.min() * 10_000 >= -40
        and portfolio["residual_btc_beta"].abs().max() <= 1e-12
        and (portfolio["gross_notional"] - 1.0).abs().max() <= 1e-12
    )
    audit_pass = bool(structural_pass and gate_recheck)
    audit = {
        "audit_pass": audit_pass,
        "structural_pass": structural_pass,
        "frozen_gate_recheck_pass": gate_recheck,
        "weeks": len(portfolio),
        "structural_tolerance": cfg.structural_tolerance,
        "max_structural_difference": structural_max,
        **{f"raw_max_abs_{name}_diff": value for name, value in raw_max.items()},
        **{f"max_abs_{name}_diff": value for name, value in field_diffs.items()},
        "alternate_bootstrap_95_low_bp": bootstrap_low * 10_000,
        "alternate_bootstrap_95_high_bp": bootstrap_high * 10_000,
        "alternate_null_percentile": null_percentile,
        "alternate_null_mean_bp": null_mean * 10_000,
        "alternate_null_95pct_bp": null_95 * 10_000,
        "alternate_null_99pct_bp": null_99 * 10_000,
        "minimum_leave_one_month_out_mean_bp": min(leave_one_month_out) * 10_000,
        "recent_six_mean_bp": float(recent_six.mean() * 10_000),
        "recent_six_positive_weeks": int(recent_six.gt(0).sum()),
        "primary_funding_only_after_cost_bp": float(
            (
                portfolio["funding_return"]
                - cfg.one_way_cost * portfolio["realized_turnover"]
            ).mean()
            * 10_000
        ),
        "stress_funding_only_after_cost_bp": float(
            (
                portfolio["funding_return"]
                - cfg.stress_one_way_cost * portfolio["realized_turnover"]
            ).mean()
            * 10_000
        ),
        "realized_weekly_price_beta_to_btc": float(
            np.cov(price_values, btc_weekly, ddof=1)[0, 1] / btc_variance
        ),
        "realized_weekly_primary_beta_to_btc": float(
            np.cov(primary_values, btc_weekly, ddof=1)[0, 1] / btc_variance
        ),
        "weekly_price_btc_correlation": float(np.corrcoef(price_values, btc_weekly)[0, 1]),
        "btc_up_primary_mean_bp": float(primary_values[btc_weekly > 0].mean() * 10_000),
        "btc_down_primary_mean_bp": float(primary_values[btc_weekly < 0].mean() * 10_000),
        "worst_week_bp": float(portfolio["primary_net_return"].min() * 10_000),
        "additive_max_drawdown_bp": float(drawdown.min() * 10_000),
        "positive_symbol_concentration": float(
            max(positive_contributions.values()) / sum(positive_contributions.values())
        ),
        "largest_positive_symbol": max(positive_contributions, key=positive_contributions.get),
        "positive_coin_concentration_ex_btc": float(
            max(positive_coins.values()) / sum(positive_coins.values())
        ),
        "largest_positive_coin": max(positive_coins, key=positive_coins.get),
    }
    root = ensure_dir(cfg.report_root)
    paths = {
        "audit": root / "independent_audit.json",
        "recomputed": root / "independent_audit_recomputed_weekly.csv",
        "symbol_contributions": root / "independent_audit_symbol_contributions.csv",
        "findings": cfg.audit_doc,
    }
    paths["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    recomputed.to_csv(paths["recomputed"], index=False)
    pd.DataFrame(
        sorted(contributions.items(), key=lambda item: item[1], reverse=True),
        columns=["symbol", "gross_contribution"],
    ).to_csv(paths["symbol_contributions"], index=False)
    paths["findings"].write_text(
        "\n".join(
            [
                "# v15.0 FSS3 Independent Audit Findings",
                "",
                f"Verdict: `{'pass' if audit_pass else 'fail'}`.",
                "",
                pd.DataFrame([audit]).to_markdown(index=False, floatfmt=".6f"),
                "",
                "Raw funding windows, raw price endpoints, current-sign targets,",
                "turnover-capped holdings, beta hedge, gross-one weights, funding and",
                "price PnL, and costs were independently reconstructed. Bootstrap and",
                "full-universe nulls use a different seed and independent implementation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
