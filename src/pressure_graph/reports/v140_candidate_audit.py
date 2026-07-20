"""Independent structural and statistical audit for the promoted v14.0 candidate."""

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


REPORT_ROOT = Path("reports/v14_0_equal_weight_negative_funding_state")
PANEL_PATH = Path("reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet")
PORTFOLIO_PATH = REPORT_ROOT / "weekly_portfolio.parquet"
AUDIT_DOC = Path("docs/v140_equal_weight_negative_funding_state_audit_2026_07_15.md")


@dataclass(frozen=True)
class V140AuditConfig:
    panel_path: Path = PANEL_PATH
    portfolio_path: Path = PORTFOLIO_PATH
    report_root: Path = REPORT_ROOT
    one_way_cost: float = 0.002
    bootstrap_iterations: int = 10_000
    null_iterations: int = 5_000
    bootstrap_block_weeks: int = 4
    seed: int = 20260716


def _funding_arrays(symbol: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    frames = []
    for path in (
        RAW_BYBIT_ROOT / "funding" / f"{symbol}.parquet",
        RECENT_ROOT / "bybit_funding" / f"{symbol}.parquet",
    ):
        if path.exists():
            frame = pd.read_parquet(path, columns=["funding_time", "funding_rate_settled"])
            frames.append(frame)
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
    return pd.DatetimeIndex(funding["funding_time"]), funding["funding_rate_settled"].to_numpy(
        dtype=float
    )


def _sum_window(
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


def _alternate_block_bootstrap(
    values: np.ndarray,
    cfg: V140AuditConfig,
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


def _alternate_null_percentile(
    panel_groups: dict[pd.Timestamp, pd.DataFrame],
    portfolio: pd.DataFrame,
    cfg: V140AuditConfig,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(cfg.seed + 1)
    null_sum = np.zeros(cfg.null_iterations, dtype=float)
    for row in portfolio.itertuples(index=False):
        local = panel_groups[pd.Timestamp(row.entry_time)].dropna(
            subset=["price_return", "future_funding", "btc_beta"]
        )
        beta = local["btc_beta"].to_numpy(dtype=float)
        price = local["price_return"].to_numpy(dtype=float)
        funding = local["future_funding"].to_numpy(dtype=float)
        breadth = int(row.eligible_negative_names)
        choices = np.asarray(
            [
                rng.choice(len(local), size=breadth, replace=False)
                for _ in range(cfg.null_iterations)
            ]
        )
        mean_beta = beta[choices].mean(axis=1)
        long_total = 1.0 / (1.0 + mean_beta)
        btc_short = mean_beta / (1.0 + mean_beta)
        gross = (
            long_total * price[choices].mean(axis=1)
            - btc_short * float(local["btc_return"].iloc[0])
            - long_total * funding[choices].mean(axis=1)
            + btc_short * float(local["btc_future_funding"].iloc[0])
        )
        null_sum += gross - cfg.one_way_cost * float(row.realized_turnover)
    null_means = null_sum / len(portfolio)
    observed = float(portfolio["primary_net_return"].mean())
    percentile = float(100 * np.mean(null_means <= observed))
    return percentile, float(null_means.mean()), float(np.quantile(null_means, 0.99))


def run_v140_candidate_audit(
    cfg: V140AuditConfig = V140AuditConfig(),
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
        pd.Timestamp(entry): frame.copy()
        for entry, frame in panel.groupby("entry_time", sort=True, observed=True)
    }

    prices = hourly_bybit_prices(load_v132_bybit_klines()).pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    )
    funding_cache = {
        symbol: _funding_arrays(symbol)
        for symbol in sorted(set(panel["symbol"].astype(str)) | {BTC})
    }

    score_diffs = []
    future_funding_diffs = []
    price_diffs = []
    btc_price_diffs = []
    btc_funding_diffs = []
    for row in panel.itertuples(index=False):
        times, rates = funding_cache[str(row.symbol)]
        score = _sum_window(
            times,
            rates,
            row.entry_time - pd.Timedelta(days=7),
            row.entry_time,
            left_side="left",
            right_side="left",
        )
        future_funding = _sum_window(
            times,
            rates,
            row.entry_time,
            row.exit_time,
            left_side="right",
            right_side="right",
        )
        raw_price = float(
            prices.at[row.exit_time, row.symbol] / prices.at[row.entry_time, row.symbol] - 1.0
        )
        btc_price = float(prices.at[row.exit_time, BTC] / prices.at[row.entry_time, BTC] - 1.0)
        btc_times, btc_rates = funding_cache[BTC]
        btc_funding = _sum_window(
            btc_times,
            btc_rates,
            row.entry_time,
            row.exit_time,
            left_side="right",
            right_side="right",
        )
        score_diffs.append(abs(score - float(row.score_7d)))
        future_funding_diffs.append(abs(future_funding - float(row.future_funding)))
        price_diffs.append(abs(raw_price - float(row.price_return)))
        btc_price_diffs.append(abs(btc_price - float(row.btc_return)))
        btc_funding_diffs.append(abs(btc_funding - float(row.btc_future_funding)))

    recomputed = []
    symbol_contributions: dict[str, float] = {}
    previous_weights: dict[str, float] | None = None
    for index, row in portfolio.iterrows():
        local = groups[pd.Timestamp(row["entry_time"])].set_index("symbol")
        selected = sorted(local.loc[local["score_7d"].lt(0)].index.astype(str).tolist())
        recorded = str(row["selected_symbols"]).split("|")
        if selected != recorded:
            raise RuntimeError(f"Selection mismatch at {row['entry_time']}")
        mean_beta = float(local.loc[selected, "btc_beta"].mean())
        long_total = 1.0 / (1.0 + mean_beta)
        btc_short = mean_beta / (1.0 + mean_beta)
        weights = {symbol: long_total / len(selected) for symbol in selected}
        weights[BTC] = -btc_short
        if previous_weights is None:
            turnover = sum(abs(weight) for weight in weights.values())
        else:
            turnover = sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in set(weights) | set(previous_weights)
            )
        if index == len(portfolio) - 1:
            turnover += sum(abs(weight) for weight in weights.values())
        long_price = float(
            sum(weights[symbol] * local.at[symbol, "price_return"] for symbol in selected)
        )
        btc_price = float(-btc_short * local.iloc[0]["btc_return"])
        coin_funding = float(
            sum(-weights[symbol] * local.at[symbol, "future_funding"] for symbol in selected)
        )
        btc_funding = float(btc_short * local.iloc[0]["btc_future_funding"])
        gross = long_price + btc_price + coin_funding + btc_funding
        primary = gross - cfg.one_way_cost * turnover
        residual_beta = float(
            sum(weights[symbol] * local.at[symbol, "btc_beta"] for symbol in selected) - btc_short
        )
        recomputed.append(
            {
                "entry_time": row["entry_time"],
                "gross_notional": sum(abs(weight) for weight in weights.values()),
                "residual_beta": residual_beta,
                "turnover": turnover,
                "gross_return": gross,
                "primary_net_return": primary,
            }
        )
        for symbol in selected:
            contribution = weights[symbol] * (
                local.at[symbol, "price_return"] - local.at[symbol, "future_funding"]
            )
            symbol_contributions[symbol] = symbol_contributions.get(symbol, 0.0) + float(
                contribution
            )
        btc_contribution = btc_short * (
            local.iloc[0]["btc_future_funding"] - local.iloc[0]["btc_return"]
        )
        symbol_contributions[BTC] = symbol_contributions.get(BTC, 0.0) + float(btc_contribution)
        previous_weights = weights

    audit_frame = pd.DataFrame(recomputed)
    merged = portfolio.merge(audit_frame, on="entry_time", suffixes=("", "_audit"))
    turnover_diff = float((merged["realized_turnover"] - merged["turnover"]).abs().max())
    gross_diff = float((merged["gross_return"] - merged["gross_return_audit"]).abs().max())
    primary_diff = float(
        (merged["primary_net_return"] - merged["primary_net_return_audit"]).abs().max()
    )
    bootstrap_low, bootstrap_high = _alternate_block_bootstrap(
        portfolio["primary_net_return"].to_numpy(dtype=float), cfg
    )
    null_percentile, null_mean, null_99 = _alternate_null_percentile(groups, portfolio, cfg)
    monthly = portfolio.groupby("month_start")["primary_net_return"].sum()
    leave_one_month_out = {
        str(month): float(
            portfolio.loc[portfolio["month_start"].ne(month), "primary_net_return"].mean()
        )
        for month in monthly.index
    }
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
    realized_price_beta = float(np.cov(price_values, btc_weekly, ddof=1)[0, 1] / btc_variance)
    realized_primary_beta = float(np.cov(primary_values, btc_weekly, ddof=1)[0, 1] / btc_variance)
    btc_up = btc_weekly > 0
    btc_down = btc_weekly < 0
    positive_symbol = {symbol: value for symbol, value in symbol_contributions.items() if value > 0}
    symbol_concentration = float(max(positive_symbol.values()) / sum(positive_symbol.values()))
    positive_coin = {symbol: value for symbol, value in positive_symbol.items() if symbol != BTC}
    coin_concentration = float(max(positive_coin.values()) / sum(positive_coin.values()))
    structural_tolerance = 1e-12
    structural_pass = bool(
        max(
            max(score_diffs),
            max(future_funding_diffs),
            max(price_diffs),
            max(btc_price_diffs),
            max(btc_funding_diffs),
            turnover_diff,
            gross_diff,
            primary_diff,
            float(audit_frame["gross_notional"].sub(1.0).abs().max()),
            float(audit_frame["residual_beta"].abs().max()),
        )
        <= structural_tolerance
    )
    audit_pass = bool(structural_pass and bootstrap_low > 0 and null_percentile >= 90)
    audit = {
        "audit_pass": audit_pass,
        "structural_pass": structural_pass,
        "weeks": len(portfolio),
        "raw_max_abs_score_diff": max(score_diffs),
        "raw_max_abs_future_funding_diff": max(future_funding_diffs),
        "raw_max_abs_price_diff": max(price_diffs),
        "raw_max_abs_btc_price_diff": max(btc_price_diffs),
        "raw_max_abs_btc_funding_diff": max(btc_funding_diffs),
        "max_abs_turnover_diff": turnover_diff,
        "max_abs_gross_return_diff": gross_diff,
        "max_abs_primary_return_diff": primary_diff,
        "max_abs_gross_notional_diff": float(audit_frame["gross_notional"].sub(1.0).abs().max()),
        "max_abs_residual_beta": float(audit_frame["residual_beta"].abs().max()),
        "alternate_bootstrap_95_low_bp": bootstrap_low * 10_000,
        "alternate_bootstrap_95_high_bp": bootstrap_high * 10_000,
        "alternate_null_percentile": null_percentile,
        "alternate_null_mean_bp": null_mean * 10_000,
        "alternate_null_99pct_bp": null_99 * 10_000,
        "recent_six_mean_bp": float(recent_six.mean() * 10_000),
        "recent_six_positive_weeks": int(recent_six.gt(0).sum()),
        "minimum_leave_one_month_out_mean_bp": min(leave_one_month_out.values()) * 10_000,
        "primary_funding_only_after_cost_bp": float(
            (portfolio["funding_return"] - cfg.one_way_cost * portfolio["realized_turnover"]).mean()
            * 10_000
        ),
        "stress_funding_only_after_cost_bp": float(
            (
                portfolio["funding_return"] - 2 * cfg.one_way_cost * portfolio["realized_turnover"]
            ).mean()
            * 10_000
        ),
        "realized_weekly_price_beta_to_btc": realized_price_beta,
        "realized_weekly_primary_beta_to_btc": realized_primary_beta,
        "weekly_price_btc_correlation": float(np.corrcoef(price_values, btc_weekly)[0, 1]),
        "btc_up_primary_mean_bp": float(primary_values[btc_up].mean() * 10_000),
        "btc_down_primary_mean_bp": float(primary_values[btc_down].mean() * 10_000),
        "worst_week_bp": float(portfolio["primary_net_return"].min() * 10_000),
        "additive_max_drawdown_bp": float(drawdown.min() * 10_000),
        "positive_symbol_concentration": symbol_concentration,
        "largest_positive_symbol": max(positive_symbol, key=positive_symbol.get),
        "positive_coin_concentration_ex_btc": coin_concentration,
        "largest_positive_coin": max(positive_coin, key=positive_coin.get),
    }
    root = ensure_dir(cfg.report_root)
    paths = {
        "audit": root / "audit.json",
        "recomputed": root / "audit_recomputed_weekly.csv",
        "symbol_contributions": root / "audit_symbol_contributions.csv",
        "findings": AUDIT_DOC,
    }
    paths["audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit_frame.to_csv(paths["recomputed"], index=False)
    pd.DataFrame(
        sorted(symbol_contributions.items(), key=lambda item: item[1], reverse=True),
        columns=["symbol", "gross_contribution"],
    ).to_csv(paths["symbol_contributions"], index=False)
    paths["findings"].write_text(
        "\n".join(
            [
                "# v14.0 Candidate Independent Audit",
                "",
                f"Verdict: `{'pass' if audit_pass else 'fail'}`.",
                "",
                pd.DataFrame([audit]).to_markdown(index=False, floatfmt=".6f"),
                "",
                "Raw funding windows, raw price endpoints, holdings, gross-one weights,",
                "BTC-beta neutrality, signed turnover, gross PnL, and primary-cost PnL",
                "were independently reconstructed. Alternate bootstrap and full-universe",
                "random controls use a different seed from the preregistered run.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
