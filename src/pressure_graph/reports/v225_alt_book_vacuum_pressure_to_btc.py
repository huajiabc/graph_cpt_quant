"""Preregistered BTC propagation test from synchronized alt-book pressure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
    load_v155_hourly_prices,
)


V224_ROOT = Path("reports/v22_4_alt_book_vacuum_pressure_feature_audit")
EVENT_PATH = V224_ROOT / "candidate_feature_events.parquet"
BUCKET_STATE_PATH = V224_ROOT / "hourly_bucket_states.parquet"
REPORT_ROOT = Path("reports/v22_5_alt_book_vacuum_pressure_to_btc")
FINDINGS_PATH = Path(
    "docs/v225_alt_book_vacuum_pressure_to_btc_findings_2026_07_17.md"
)
CANDIDATE = "DVB1_ALT_BOOK_VACUUM_PRESSURE_TO_BTC"
REVERSED_CONTROL = "DVB1_REVERSED_PRESSURE_DIRECTION"
DELAYED_CONTROL = "DVB1_ONE_HOUR_DELAYED_ENTRY"
NO_VACUUM_CONTROL = "DVB1_PRESSURE_WITHOUT_BROAD_WITHDRAWAL"
FEATURE_SHA256 = "A6495D01FD26E05D1762531590A886176CCFDF3AFAE870559072A0132C07D43F"


@dataclass(frozen=True)
class V225Config:
    event_path: Path = EVENT_PATH
    bucket_state_path: Path = BUCKET_STATE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_directional_symbols: int = 11
    minimum_withdrawing_symbols: int = 5
    cooldown_hours: int = 4
    btc_primary_cost: float = 0.0010
    btc_stress_cost: float = 0.0020
    alt_primary_cost: float = 0.0020
    alt_stress_cost: float = 0.0030
    random_iterations: int = 1000
    bootstrap_iterations: int = 2000
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v225_inputs(
    cfg: V225Config = V225Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(cfg.event_path)
    states = pd.read_parquet(cfg.bucket_state_path)
    prices = load_v155_hourly_prices()
    for frame, columns in (
        (events, ("feature_time", "entry_time")),
        (states, ("decision_time",)),
        (prices, ("feature_time",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return events, states, prices


def build_v225_no_vacuum_events(
    states: pd.DataFrame,
    cfg: V225Config = V225Config(),
) -> pd.DataFrame:
    base_pressure = (
        states["bucket_pressure"].abs().ge(states["prior_abs_pressure_threshold"])
        & states["directional_symbol_count"].ge(cfg.minimum_directional_symbols)
        & states["withdrawing_symbol_count"].lt(cfg.minimum_withdrawing_symbols)
    )
    starts = states[base_pressure & ~base_pressure.shift(1, fill_value=False)].copy()
    selected: list[int] = []
    last_time: pd.Timestamp | None = None
    for index, row in starts.iterrows():
        decision_time = pd.Timestamp(row["decision_time"])
        if last_time is None or decision_time - last_time >= pd.Timedelta(
            hours=cfg.cooldown_hours
        ):
            selected.append(index)
            last_time = decision_time
    output = starts.loc[selected].copy()
    output["candidate"] = NO_VACUUM_CONTROL
    output["entry_time"] = output["decision_time"]
    output["signal_direction"] = output["direction"].astype(int)
    output["entry_month"] = output["entry_time"].dt.strftime("%Y-%m")
    output["period"] = np.select(
        [
            output["entry_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            output["entry_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    return output.reset_index(drop=True)


def _price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()


def price_v225_events(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V225Config = V225Config(),
    *,
    candidate: str = CANDIDATE,
) -> pd.DataFrame:
    matrix = _price_matrix(prices)
    required_symbols = [BTC, *FROZEN_SYMBOLS]
    rows: list[dict[str, object]] = []
    for event in events.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(event.entry_time)
        times = [entry + pd.Timedelta(hours=offset) for offset in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5)]
        if any(time not in matrix.index for time in times):
            continue
        price_slice = matrix.loc[times, required_symbols]
        if not bool(np.isfinite(price_slice.to_numpy(dtype=float)).all()):
            continue
        direction = int(event.signal_direction)
        btc = price_slice[BTC]
        btc_gross_1h = direction * (float(btc.loc[entry + pd.Timedelta(hours=1)]) / float(btc.loc[entry]) - 1.0)
        btc_gross_4h = direction * (float(btc.loc[entry + pd.Timedelta(hours=4)]) / float(btc.loc[entry]) - 1.0)
        btc_delayed_gross_4h = direction * (
            float(btc.loc[entry + pd.Timedelta(hours=5)])
            / float(btc.loc[entry + pd.Timedelta(hours=1)])
            - 1.0
        )
        alt_returns = (
            price_slice.loc[entry + pd.Timedelta(hours=4), list(FROZEN_SYMBOLS)]
            / price_slice.loc[entry, list(FROZEN_SYMBOLS)]
            - 1.0
        )
        alt_bucket_gross_4h = direction * float(alt_returns.mean())
        btc_returns = btc.pct_change(fill_method=None)
        prior_variance = float(
            np.square(btc_returns.loc[entry - pd.Timedelta(hours=3) : entry]).sum()
        )
        future_variance = float(
            np.square(
                btc_returns.loc[
                    entry + pd.Timedelta(hours=1) : entry + pd.Timedelta(hours=4)
                ]
            ).sum()
        )
        rows.append(
            {
                "candidate": candidate,
                "feature_time": getattr(event, "feature_time", entry),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "entry_day": entry.floor("D"),
                "entry_month": event.entry_month,
                "period": event.period,
                "signal_direction": direction,
                "bucket_pressure": float(event.bucket_pressure),
                "directional_symbol_count": int(event.directional_symbol_count),
                "withdrawing_symbol_count": int(event.withdrawing_symbol_count),
                "btc_gross_return_1h": btc_gross_1h,
                "btc_primary_net_return_1h": btc_gross_1h - cfg.btc_primary_cost,
                "btc_gross_return_4h": btc_gross_4h,
                "btc_primary_net_return_4h": btc_gross_4h - cfg.btc_primary_cost,
                "btc_stress_net_return_4h": btc_gross_4h - cfg.btc_stress_cost,
                "reversed_primary_net_return_4h": -btc_gross_4h
                - cfg.btc_primary_cost,
                "delayed_gross_return_4h": btc_delayed_gross_4h,
                "delayed_primary_net_return_4h": btc_delayed_gross_4h
                - cfg.btc_primary_cost,
                "alt_bucket_gross_return_4h": alt_bucket_gross_4h,
                "alt_bucket_primary_net_return_4h": alt_bucket_gross_4h
                - cfg.alt_primary_cost,
                "alt_bucket_stress_net_return_4h": alt_bucket_gross_4h
                - cfg.alt_stress_cost,
                "prior_btc_realized_variance_4h": prior_variance,
                "future_btc_realized_variance_4h": future_variance,
                "future_to_prior_btc_variance_ratio": future_variance / prior_variance
                if prior_variance > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def build_v225_random_controls(
    events: pd.DataFrame,
    states: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: V225Config = V225Config(),
) -> pd.DataFrame:
    matrix = _price_matrix(prices)
    candidate_times = set(events["entry_time"])
    pool = states[
        states["prior_abs_pressure_threshold"].notna()
        & ~states["decision_time"].isin(candidate_times)
    ].copy()
    pool["entry_time"] = pool["decision_time"]
    pool["entry_month"] = pool["entry_time"].dt.strftime("%Y-%m")
    pool["signal_direction"] = pool["direction"].astype(int)
    valid_times = [
        time
        for time in pool["entry_time"]
        if time in matrix.index and time + pd.Timedelta(hours=4) in matrix.index
    ]
    pool = pool[pool["entry_time"].isin(valid_times)].copy()
    btc = matrix[BTC]
    pool["signed_btc_gross_4h"] = pool["signal_direction"] * (
        pool["entry_time"].map(btc.shift(-4)) / pool["entry_time"].map(btc) - 1.0
    )
    pools = {
        key: local["signed_btc_gross_4h"].to_numpy(dtype=float)
        for key, local in pool.groupby(
            ["entry_month", "signal_direction"], sort=False, observed=True
        )
    }
    requested = (
        events.groupby(["entry_month", "signal_direction"], observed=True)
        .size()
        .to_dict()
    )
    rows = []
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        values = []
        for key, count in requested.items():
            available = pools[key]
            values.extend(rng.choice(available, size=int(count), replace=True))
        gross = np.asarray(values, dtype=float)
        rows.append(
            {
                "iteration": iteration,
                "control": "same_month_same_direction_random_non_event_hours",
                "events": len(gross),
                "mean_gross_return_4h": float(gross.mean()),
                "mean_primary_net_return_4h": float(
                    gross.mean() - cfg.btc_primary_cost
                ),
            }
        )
    return pd.DataFrame(rows)


def _day_cluster_bootstrap(
    outcomes: pd.DataFrame,
    cfg: V225Config,
) -> tuple[float, float]:
    groups = [
        local["btc_primary_net_return_4h"].to_numpy(dtype=float)
        for _, local in outcomes.groupby("entry_day", sort=False, observed=True)
    ]
    rng = np.random.default_rng(cfg.seed + 2)
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(groups), size=len(groups))
        draws[iteration] = float(np.concatenate([groups[index] for index in chosen]).mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_v225(
    outcomes: pd.DataFrame,
    no_vacuum: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V225Config = V225Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bootstrap_low, bootstrap_high = _day_cluster_bootstrap(outcomes, cfg)
    primary = outcomes["btc_primary_net_return_4h"]
    periods = outcomes.groupby("period", observed=True).agg(
        events=("entry_time", "size"),
        mean_gross_bp=("btc_gross_return_4h", lambda x: float(x.mean() * 10_000)),
        mean_primary_net_bp=("btc_primary_net_return_4h", lambda x: float(x.mean() * 10_000)),
        mean_stress_net_bp=("btc_stress_net_return_4h", lambda x: float(x.mean() * 10_000)),
        variance_ratio=("future_to_prior_btc_variance_ratio", "mean"),
    ).reset_index()
    directions = outcomes.groupby("signal_direction", observed=True).agg(
        events=("entry_time", "size"),
        mean_gross_bp=("btc_gross_return_4h", lambda x: float(x.mean() * 10_000)),
        mean_primary_net_bp=("btc_primary_net_return_4h", lambda x: float(x.mean() * 10_000)),
        variance_ratio=("future_to_prior_btc_variance_ratio", "mean"),
    ).reset_index()
    period_primary = outcomes.groupby("period", observed=True)[
        "btc_primary_net_return_4h"
    ].mean()
    period_variance = outcomes.groupby("period", observed=True)[
        "future_to_prior_btc_variance_ratio"
    ].mean()
    direction_primary = outcomes.groupby("signal_direction", observed=True)[
        "btc_primary_net_return_4h"
    ].mean()
    month_pnl = outcomes.groupby("entry_month", observed=True)[
        "btc_primary_net_return_4h"
    ].sum()
    day_pnl = outcomes.groupby("entry_day", observed=True)[
        "btc_primary_net_return_4h"
    ].sum()
    positive_months = month_pnl[month_pnl.gt(0)]
    positive_days = day_pnl[day_pnl.gt(0)]
    month_concentration = (
        float(positive_months.max() / positive_months.sum())
        if positive_months.sum() > 0
        else np.inf
    )
    day_concentration = (
        float(positive_days.max() / positive_days.sum())
        if positive_days.sum() > 0
        else np.inf
    )
    observed_mean = float(primary.mean())
    counts = outcomes["period"].value_counts()
    cross = pd.crosstab(outcomes["period"], outcomes["signal_direction"])
    row: dict[str, object] = {
        "candidate": CANDIDATE,
        "events": len(outcomes),
        "active_days": outcomes["entry_day"].nunique(),
        "active_months": outcomes["entry_month"].nunique(),
        "development_events": int(counts.get("development", 0)),
        "validation_events": int(counts.get("validation", 0)),
        "holdout_events": int(counts.get("holdout", 0)),
        "minimum_direction_period_events": int(cross.min().min()),
        "mean_btc_gross_1h_bp": float(outcomes["btc_gross_return_1h"].mean() * 10_000),
        "mean_btc_primary_net_1h_bp": float(outcomes["btc_primary_net_return_1h"].mean() * 10_000),
        "mean_btc_gross_4h_bp": float(outcomes["btc_gross_return_4h"].mean() * 10_000),
        "mean_btc_primary_net_4h_bp": observed_mean * 10_000,
        "mean_btc_stress_net_4h_bp": float(outcomes["btc_stress_net_return_4h"].mean() * 10_000),
        "development_primary_net_4h_bp": float(period_primary.get("development", np.nan) * 10_000),
        "validation_primary_net_4h_bp": float(period_primary.get("validation", np.nan) * 10_000),
        "holdout_primary_net_4h_bp": float(period_primary.get("holdout", np.nan) * 10_000),
        "long_primary_net_4h_bp": float(direction_primary.get(1, np.nan) * 10_000),
        "short_primary_net_4h_bp": float(direction_primary.get(-1, np.nan) * 10_000),
        "mean_alt_bucket_gross_4h_bp": float(outcomes["alt_bucket_gross_return_4h"].mean() * 10_000),
        "mean_alt_bucket_primary_net_4h_bp": float(outcomes["alt_bucket_primary_net_return_4h"].mean() * 10_000),
        "mean_variance_ratio": float(outcomes["future_to_prior_btc_variance_ratio"].mean()),
        "validation_variance_ratio": float(period_variance.get("validation", np.nan)),
        "holdout_variance_ratio": float(period_variance.get("holdout", np.nan)),
        "bootstrap_95_low_primary_bp": bootstrap_low * 10_000,
        "bootstrap_95_high_primary_bp": bootstrap_high * 10_000,
        "random_time_percentile": float(
            100
            * random_controls["mean_primary_net_return_4h"].le(observed_mean).mean()
        ),
        "reversed_primary_net_4h_bp": float(outcomes["reversed_primary_net_return_4h"].mean() * 10_000),
        "delayed_primary_net_4h_bp": float(outcomes["delayed_primary_net_return_4h"].mean() * 10_000),
        "no_vacuum_primary_net_4h_bp": float(no_vacuum["btc_primary_net_return_4h"].mean() * 10_000),
        "positive_month_concentration": month_concentration,
        "positive_day_concentration": day_concentration,
    }
    row["promote"] = bool(
        row["events"] >= 150
        and row["active_months"] >= 11
        and row["development_events"] >= 45
        and row["validation_events"] >= 45
        and row["holdout_events"] >= 45
        and row["minimum_direction_period_events"] >= 15
        and all(
            float(row[key]) > 0
            for key in (
                "mean_btc_gross_1h_bp",
                "mean_btc_gross_4h_bp",
                "mean_btc_primary_net_4h_bp",
                "mean_btc_stress_net_4h_bp",
                "development_primary_net_4h_bp",
                "validation_primary_net_4h_bp",
                "holdout_primary_net_4h_bp",
                "long_primary_net_4h_bp",
                "short_primary_net_4h_bp",
                "bootstrap_95_low_primary_bp",
            )
        )
        and row["random_time_percentile"] >= 95
        and row["mean_btc_primary_net_4h_bp"] > row["reversed_primary_net_4h_bp"]
        and row["mean_btc_primary_net_4h_bp"] > row["delayed_primary_net_4h_bp"]
        and row["mean_btc_primary_net_4h_bp"] > row["no_vacuum_primary_net_4h_bp"]
        and row["mean_variance_ratio"] > 1
        and row["validation_variance_ratio"] > 1
        and row["holdout_variance_ratio"] > 1
        and row["positive_month_concentration"] <= 0.35
        and row["positive_day_concentration"] <= 0.20
    )
    return pd.DataFrame([row]), periods, directions


def build_v225_cost_frontier(outcomes: pd.DataFrame) -> pd.DataFrame:
    gross = float(outcomes["btc_gross_return_4h"].mean())
    return pd.DataFrame(
        [
            {
                "round_trip_cost_bp": cost_bp,
                "mean_net_return_bp": gross * 10_000 - cost_bp,
            }
            for cost_bp in (0, 5, 10, 15, 20, 30)
        ]
    )


def write_v225_alt_book_vacuum_pressure_to_btc(
    cfg: V225Config = V225Config(),
) -> dict[str, Path]:
    events, states, prices = load_v225_inputs(cfg)
    no_vacuum_features = build_v225_no_vacuum_events(states, cfg)
    outcomes = price_v225_events(events, prices, cfg, candidate=CANDIDATE)
    no_vacuum = price_v225_events(
        no_vacuum_features, prices, cfg, candidate=NO_VACUUM_CONTROL
    )
    random_controls = build_v225_random_controls(outcomes, states, prices, cfg)
    summary, periods, directions = summarize_v225(
        outcomes, no_vacuum, random_controls, cfg
    )
    cost_frontier = build_v225_cost_frontier(outcomes)
    root = ensure_dir(cfg.report_root)
    paths = {
        "events": root / "candidate_events.parquet",
        "no_vacuum": root / "no_vacuum_control_events.parquet",
        "random": root / "random_time_controls.csv",
        "summary": root / "candidate_outcome.csv",
        "periods": root / "period_summary.csv",
        "directions": root / "direction_summary.csv",
        "cost_frontier": root / "cost_frontier.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    outcomes.to_parquet(paths["events"], index=False)
    no_vacuum.to_parquet(paths["no_vacuum"], index=False)
    random_controls.to_csv(paths["random"], index=False)
    summary.to_csv(paths["summary"], index=False)
    periods.to_csv(paths["periods"], index=False)
    directions.to_csv(paths["directions"], index=False)
    cost_frontier.to_csv(paths["cost_frontier"], index=False)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "feature_sha256": _sha256(cfg.event_path),
                "preregistered_feature_sha256": FEATURE_SHA256,
                "promoted": summary.loc[summary["promote"], "candidate"].tolist(),
                "config": serialized,
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = (
        "retain_new_research_candidate"
        if bool(summary.iloc[0]["promote"])
        else "reject_alt_book_vacuum_pressure_to_btc"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v22.5 Alt-Book Vacuum Pressure to BTC Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Chronological periods",
                "",
                periods.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Signal direction",
                "",
                directions.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The single preregistered four-hour BTC endpoint was evaluated.",
                "The one-hour and alt-bucket outcomes are secondary and cannot rescue it.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
