"""Stateful record-only forward shadow for the frozen v14.9 FSS3 strategy."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pressure_graph.io import ensure_dir, write_parquet


BTC = "BTCUSDT"
CANDIDATE = "FSS3_CURRENT_SIGN_070_TURNOVER_CAP"


DECISION_COLUMNS = [
    "strategy_id",
    "decision_time",
    "exit_time",
    "month_start",
    "decision_status",
    "first_observed_at_utc",
    "first_observation_lag_minutes",
    "timely_forward_decision",
    "evidence_eligible",
    "state_source",
    "previous_state_hash",
    "membership_hash",
    "funding_snapshot_hash",
    "beta_price_snapshot_hash",
    "decision_input_hash",
    "negative_breadth",
    "positive_breadth",
    "selected_long_symbols",
    "selected_short_symbols",
    "executed_target_fraction",
    "target_tracking_l1",
    "cap_binding",
    "rebalance_turnover",
    "cap_breach",
    "gross_notional",
    "residual_btc_beta",
    "executed_weight_count",
    "reason",
    "mode",
    "push_policy",
    "real_orders_allowed",
    "leverage_allowed",
]

WEIGHT_COLUMNS = [
    "strategy_id",
    "decision_time",
    "symbol",
    "weight",
    "is_btc_hedge",
    "mode",
    "real_orders_allowed",
]

PERFORMANCE_COLUMNS = [
    "strategy_id",
    "decision_time",
    "scheduled_exit_time",
    "mark_time",
    "status",
    "timely_forward_decision",
    "evidence_eligible",
    "price_return",
    "funding_return",
    "gross_return",
    "rebalance_turnover",
    "primary_cost",
    "stress_cost",
    "primary_net_return",
    "stress_net_return",
    "mode",
    "real_orders_allowed",
]


@dataclass(frozen=True)
class FSS3LiveConfig:
    base_config: Path
    live_root: Path
    membership_path: Path
    seed_weights_path: Path
    report_root: Path
    history_days: int = 55
    beta_lookback_days: int = 30
    funding_lookback_days: int = 7
    funding_max_age_hours: int = 12
    price_interval: str = "1h"
    price_stale_after_hours: int = 2
    refresh_interval_minutes: int = 360
    timely_lag_minutes: int = 60
    incomplete_input_grace_minutes: int = 60
    minimum_names_per_side: int = 4
    transition_turnover_cap: float = 0.70
    bisection_iterations: int = 48
    primary_one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    mode: str = "live_shadow"
    push_policy: str = "record_only"
    enabled: bool = True
    real_orders_allowed: bool = False
    leverage_allowed: bool = False
    snapshot_inputs: bool = True
    allow_historical_state_catchup: bool = True


@dataclass(frozen=True)
class FSS3LiveStatus:
    status: str
    observed_at_utc: str
    strategy_id: str
    lifecycle_status: str
    scope: str
    enabled: bool
    push_policy: str
    real_orders_allowed: bool
    leverage_allowed: bool
    latest_price_time: str
    data_stale: bool
    seed_entry_time: str
    latest_decision_time: str
    latest_decision_status: str
    latest_decision_timely: bool
    total_recorded_decisions: int
    timely_recorded_decisions: int
    completed_timely_weeks: int
    open_timely_weeks: int
    current_weight_count: int
    current_gross_notional: float
    mean_completed_primary_net: float | None
    mean_completed_stress_net: float | None
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _estimate_betas(history_returns: pd.DataFrame) -> pd.Series:
    """Exact frozen v14.9 monthly beta estimator, copied into the live boundary."""
    if BTC not in history_returns.columns:
        return pd.Series(dtype=float)
    btc_values = pd.to_numeric(history_returns[BTC], errors="coerce")
    variance = float(btc_values.var(ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        return pd.Series(dtype=float)
    betas: dict[str, float] = {}
    for symbol in history_returns.columns:
        values = pd.to_numeric(history_returns[symbol], errors="coerce")
        valid = values.notna() & btc_values.notna()
        if int(valid.sum()) < 100:
            continue
        covariance = float(
            np.mean(
                (values[valid] - values[valid].mean())
                * (btc_values[valid] - btc_values[valid].mean())
            )
        )
        betas[str(symbol)] = covariance / variance
    return pd.Series(betas, dtype=float)


def _weight_turnover(
    left: dict[str, float], right: dict[str, float]
) -> float:
    return float(
        sum(
            abs(left.get(symbol, 0.0) - right.get(symbol, 0.0))
            for symbol in set(left) | set(right)
        )
    )


def _neutralize(
    local: pd.DataFrame,
    raw_alt_weights: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    if not raw_alt_weights:
        return {}, 0.0, 0.0
    indexed = local.set_index("symbol")
    alt_beta = float(
        sum(
            raw_alt_weights[symbol] * float(indexed.at[symbol, "btc_beta"])
            for symbol in raw_alt_weights
        )
    )
    unscaled = dict(raw_alt_weights)
    unscaled[BTC] = -alt_beta
    gross = float(sum(abs(weight) for weight in unscaled.values()))
    if not np.isfinite(gross) or gross <= 0:
        return {}, 0.0, 0.0
    weights = {symbol: weight / gross for symbol, weight in unscaled.items()}
    residual_beta = float(
        sum(
            weights[symbol] * float(indexed.at[symbol, "btc_beta"])
            for symbol in raw_alt_weights
        )
        + weights[BTC]
    )
    return weights, float(sum(abs(weight) for weight in weights.values())), residual_beta


def _funding_sign_target(
    local: pd.DataFrame,
    minimum_side_breadth: int,
) -> tuple[dict[str, float], int, int]:
    eligible = local.dropna(subset=["score_7d", "btc_beta"])
    negative = sorted(
        eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique()
    )
    positive = sorted(
        eligible.loc[eligible["score_7d"].gt(0), "symbol"].astype(str).unique()
    )
    if (
        len(negative) < minimum_side_breadth
        or len(positive) < minimum_side_breadth
    ):
        return {}, len(negative), len(positive)
    raw = {symbol: 0.5 / len(negative) for symbol in negative}
    raw.update({symbol: -0.5 / len(positive) for symbol in positive})
    target, _, _ = _neutralize(local, raw)
    return target, len(negative), len(positive)


def _blend_and_neutralize(
    local: pd.DataFrame,
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    fraction: float,
) -> tuple[dict[str, float], float, float]:
    current_symbols = set(local["symbol"].astype(str))
    previous_alt = {
        symbol: weight
        for symbol, weight in previous_weights.items()
        if symbol != BTC and symbol in current_symbols
    }
    target_alt = {
        symbol: weight
        for symbol, weight in target_weights.items()
        if symbol != BTC
    }
    alt = {
        symbol: (1.0 - fraction) * previous_alt.get(symbol, 0.0)
        + fraction * target_alt.get(symbol, 0.0)
        for symbol in set(previous_alt) | set(target_alt)
    }
    alt = {
        symbol: weight for symbol, weight in alt.items() if abs(weight) > 1e-16
    }
    return _neutralize(local, alt)


def _execute_capped_transition(
    local: pd.DataFrame,
    previous_weights: dict[str, float],
    target_weights: dict[str, float],
    cap: float,
    bisection_iterations: int,
) -> tuple[dict[str, float], float, float, float, float, float]:
    if not previous_weights:
        indexed = local.set_index("symbol")
        residual = float(
            sum(
                weight * float(indexed.at[symbol, "btc_beta"])
                for symbol, weight in target_weights.items()
                if symbol != BTC
            )
            + target_weights.get(BTC, 0.0)
        )
        gross = float(sum(abs(weight) for weight in target_weights.values()))
        return target_weights, 1.0, gross, 0.0, gross, residual
    target_turnover = _weight_turnover(previous_weights, target_weights)
    if target_turnover <= cap + 1e-14:
        weights, gross, residual = _blend_and_neutralize(
            local, previous_weights, target_weights, 1.0
        )
        return weights, 1.0, target_turnover, 0.0, gross, residual

    base_weights, base_gross, base_residual = _blend_and_neutralize(
        local, previous_weights, target_weights, 0.0
    )
    base_turnover = _weight_turnover(previous_weights, base_weights)
    if base_turnover >= cap:
        return (
            base_weights,
            0.0,
            base_turnover,
            max(0.0, base_turnover - cap),
            base_gross,
            base_residual,
        )

    low = 0.0
    high = 1.0
    best_weights = base_weights
    best_turnover = base_turnover
    best_gross = base_gross
    best_residual = base_residual
    for _ in range(bisection_iterations):
        middle = 0.5 * (low + high)
        weights, gross, residual = _blend_and_neutralize(
            local, previous_weights, target_weights, middle
        )
        turnover = _weight_turnover(previous_weights, weights)
        if turnover <= cap:
            low = middle
            best_weights = weights
            best_turnover = turnover
            best_gross = gross
            best_residual = residual
        else:
            high = middle
    return best_weights, low, best_turnover, 0.0, best_gross, best_residual


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def latest_monday(value: object) -> pd.Timestamp:
    timestamp = _utc(value)
    return timestamp.normalize() - pd.Timedelta(days=timestamp.weekday())


def load_fss3_live_config(path: str | Path) -> FSS3LiveConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    application = payload["application"]
    data = payload["data"]
    decision = payload["decision"]
    execution = payload["execution"]
    forward = payload["forward"]
    cfg = FSS3LiveConfig(
        base_config=Path(data["base_config"]),
        live_root=Path(data["live_root"]),
        membership_path=Path(data["membership_path"]),
        seed_weights_path=Path(data["seed_weights_path"]),
        report_root=Path(forward["report_root"]),
        history_days=int(data["history_days"]),
        beta_lookback_days=int(data["beta_lookback_days"]),
        funding_lookback_days=int(data["funding_lookback_days"]),
        funding_max_age_hours=int(data["funding_max_age_hours"]),
        price_interval=str(data["price_interval"]),
        price_stale_after_hours=int(data["price_stale_after_hours"]),
        refresh_interval_minutes=int(data["refresh_interval_minutes"]),
        timely_lag_minutes=int(decision["timely_lag_minutes"]),
        incomplete_input_grace_minutes=int(
            decision["incomplete_input_grace_minutes"]
        ),
        minimum_names_per_side=int(decision["minimum_names_per_side"]),
        transition_turnover_cap=float(execution["transition_turnover_cap"]),
        bisection_iterations=int(execution["bisection_iterations"]),
        primary_one_way_cost=float(execution["primary_one_way_bps"]) / 10_000,
        stress_one_way_cost=float(execution["stress_one_way_bps"]) / 10_000,
        mode=str(application["scope"]),
        push_policy=str(application["push_policy"]),
        enabled=bool(application["enabled"]),
        real_orders_allowed=bool(application["real_orders_allowed"]),
        leverage_allowed=bool(application["leverage_allowed"]),
        snapshot_inputs=bool(forward["snapshot_inputs"]),
        allow_historical_state_catchup=bool(
            forward["allow_historical_state_catchup"]
        ),
    )
    if cfg.mode != "live_shadow" or cfg.push_policy != "record_only":
        raise ValueError("FSS3 forward shadow must be live_shadow/record_only")
    if cfg.real_orders_allowed or cfg.leverage_allowed:
        raise ValueError("FSS3 forward shadow cannot enable orders or leverage")
    return cfg


def load_fss3_membership(path: str | Path) -> pd.DataFrame:
    membership = pd.read_csv(path)
    required = {"month_start", "symbol"}
    missing = required - set(membership.columns)
    if missing:
        raise ValueError(f"missing membership columns: {sorted(missing)}")
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    membership["symbol"] = membership["symbol"].astype(str)
    return (
        membership.dropna(subset=["month_start", "symbol"])
        .drop_duplicates(["month_start", "symbol"], keep="last")
        .sort_values(["month_start", "symbol"])
        .reset_index(drop=True)
    )


def build_fss3_hourly_prices(klines: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "bar_close_time", "close"}
    missing = required - set(klines.columns)
    if missing:
        raise ValueError(f"missing kline columns: {sorted(missing)}")
    prices = klines[["symbol", "bar_close_time", "close"]].rename(
        columns={"bar_close_time": "feature_time"}
    )
    prices["feature_time"] = pd.to_datetime(
        prices["feature_time"], utc=True, errors="coerce"
    )
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices[
        prices["feature_time"].dt.minute.eq(0)
        & prices["feature_time"].dt.second.eq(0)
    ]
    return (
        prices.dropna(subset=["symbol", "feature_time", "close"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _canonical_frame_hash(frame: pd.DataFrame, sort_by: list[str]) -> str:
    if frame.empty:
        payload = b""
    else:
        data = frame.copy()
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = pd.to_datetime(
                    data[column], utc=True, errors="coerce"
                ).map(lambda value: "" if pd.isna(value) else value.isoformat())
        data = data.sort_values(sort_by).reset_index(drop=True)
        payload = data.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weights_hash(weights: dict[str, float]) -> str:
    frame = pd.DataFrame(
        [{"symbol": symbol, "weight": weight} for symbol, weight in weights.items()]
    )
    return _canonical_frame_hash(frame, ["symbol"])


def build_fss3_decision_input(
    funding: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    decision_time: object,
    cfg: FSS3LiveConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision = _utc(decision_time)
    month = pd.Timestamp(
        year=decision.year, month=decision.month, day=1, tz="UTC"
    )
    members = sorted(
        membership.loc[membership["month_start"].eq(month), "symbol"]
        .astype(str)
        .unique()
    )
    reasons: list[str] = []
    if not members:
        reasons.append("membership_missing_for_month")
    if len(members) != len(set(members)):
        reasons.append("membership_duplicate_symbols")

    price_data = prices.copy()
    price_data["feature_time"] = pd.to_datetime(
        price_data["feature_time"], utc=True, errors="coerce"
    )
    price_data["close"] = pd.to_numeric(price_data["close"], errors="coerce")
    price_data = price_data[
        price_data["symbol"].isin([*members, BTC])
        & price_data["feature_time"].le(decision)
    ].copy()
    close = price_data.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    beta_start = month - pd.Timedelta(days=cfg.beta_lookback_days)
    beta_returns = returns[
        (returns.index >= beta_start) & (returns.index < month)
    ]
    betas = _estimate_betas(beta_returns)

    funding_data = funding.copy()
    funding_data["funding_time"] = pd.to_datetime(
        funding_data["funding_time"], utc=True, errors="coerce"
    )
    funding_data["funding_rate_settled"] = pd.to_numeric(
        funding_data["funding_rate_settled"], errors="coerce"
    )
    funding_start = decision - pd.Timedelta(days=cfg.funding_lookback_days)
    funding_snapshot = funding_data[
        funding_data["symbol"].isin(members)
        & funding_data["funding_time"].ge(funding_start)
        & funding_data["funding_time"].lt(decision)
    ].copy()

    missing_beta = sorted(set(members) - set(betas.dropna().index.astype(str)))
    if missing_beta:
        reasons.append("missing_beta:" + "|".join(missing_beta))
    if decision not in close.index or BTC not in close.columns or pd.isna(
        close.at[decision, BTC]
    ):
        reasons.append("missing_btc_entry_price")

    rows: list[dict[str, Any]] = []
    missing_funding: list[str] = []
    stale_funding: list[str] = []
    missing_entry_price: list[str] = []
    for symbol in members:
        local_funding = funding_snapshot[funding_snapshot["symbol"].eq(symbol)]
        if local_funding.empty:
            missing_funding.append(symbol)
            continue
        latest_funding = local_funding["funding_time"].max()
        if decision - latest_funding > pd.Timedelta(
            hours=cfg.funding_max_age_hours
        ):
            stale_funding.append(symbol)
        if (
            decision not in close.index
            or symbol not in close.columns
            or pd.isna(close.at[decision, symbol])
        ):
            missing_entry_price.append(symbol)
        rows.append(
            {
                "entry_time": decision,
                "exit_time": decision + pd.Timedelta(days=7),
                "month_start": month,
                "symbol": symbol,
                "score_7d": float(local_funding["funding_rate_settled"].sum()),
                "btc_beta": float(betas.get(symbol, np.nan)),
                # The frozen research target helper requires these columns but
                # they never affect the sign target or beta-neutral weights.
                "price_return": 0.0,
                "future_funding": 0.0,
                "btc_return": 0.0,
                "btc_future_funding": 0.0,
            }
        )
    if missing_funding:
        reasons.append("missing_funding:" + "|".join(missing_funding))
    if stale_funding:
        reasons.append("stale_funding:" + "|".join(stale_funding))
    if missing_entry_price:
        reasons.append("missing_entry_price:" + "|".join(missing_entry_price))

    local = pd.DataFrame(rows)
    membership_snapshot = membership[
        membership["month_start"].eq(month) & membership["symbol"].isin(members)
    ].copy()
    beta_price_snapshot = price_data[
        price_data["feature_time"].ge(beta_start - pd.Timedelta(hours=1))
        & price_data["feature_time"].le(decision)
    ].copy()
    metadata: dict[str, Any] = {
        "decision_time": decision,
        "month_start": month,
        "members": tuple(members),
        "membership_snapshot": membership_snapshot,
        "funding_snapshot": funding_snapshot,
        "beta_price_snapshot": beta_price_snapshot,
        "membership_hash": _canonical_frame_hash(
            membership_snapshot, ["month_start", "symbol"]
        ),
        "funding_snapshot_hash": _canonical_frame_hash(
            funding_snapshot, ["funding_time", "symbol"]
        ),
        "beta_price_snapshot_hash": _canonical_frame_hash(
            beta_price_snapshot, ["feature_time", "symbol"]
        ),
        "decision_input_hash": _canonical_frame_hash(
            local, ["entry_time", "symbol"]
        ),
        "reasons": tuple(reasons),
    }
    return local, metadata


def build_fss3_weights(
    local: pd.DataFrame,
    previous_weights: dict[str, float],
    cfg: FSS3LiveConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    target, negative_breadth, positive_breadth = _funding_sign_target(
        local, cfg.minimum_names_per_side
    )
    if not target:
        turnover = float(sum(abs(weight) for weight in previous_weights.values()))
        return {}, {
            "decision_status": "no_signal_fail_closed",
            "negative_breadth": negative_breadth,
            "positive_breadth": positive_breadth,
            "executed_target_fraction": 0.0,
            "target_tracking_l1": 0.0,
            "cap_binding": False,
            "rebalance_turnover": turnover,
            "cap_breach": 0.0,
            "gross_notional": 0.0,
            "residual_btc_beta": 0.0,
            "target_weights_hash": _weights_hash({}),
        }
    (
        weights,
        fraction,
        turnover,
        cap_breach,
        gross_notional,
        residual_btc_beta,
    ) = _execute_capped_transition(
        local,
        previous_weights,
        target,
        cfg.transition_turnover_cap,
        cfg.bisection_iterations,
    )
    return weights, {
        "decision_status": "recorded",
        "negative_breadth": negative_breadth,
        "positive_breadth": positive_breadth,
        "executed_target_fraction": fraction,
        "target_tracking_l1": _weight_turnover(weights, target),
        "cap_binding": bool(fraction < 1.0 - 1e-10),
        "rebalance_turnover": turnover,
        "cap_breach": cap_breach,
        "gross_notional": gross_notional,
        "residual_btc_beta": residual_btc_beta,
        "target_weights_hash": _weights_hash(target),
    }


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_ledger(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return _empty(columns)
    data = pd.read_parquet(path)
    for column in ("decision_time", "exit_time", "first_observed_at_utc"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    return data


def _seed_state(path: Path) -> tuple[pd.Timestamp, dict[str, float], str]:
    seed = pd.read_parquet(path)
    required = {"entry_time", "symbol", "weight"}
    missing = required - set(seed.columns)
    if missing:
        raise ValueError(f"seed weights missing columns: {sorted(missing)}")
    seed["entry_time"] = pd.to_datetime(seed["entry_time"], utc=True, errors="coerce")
    entry = seed["entry_time"].max()
    current = seed[seed["entry_time"].eq(entry)]
    weights = {
        str(row.symbol): float(row.weight)
        for row in current.itertuples(index=False)
    }
    return pd.Timestamp(entry), weights, _weights_hash(weights)


def _write_snapshot(
    cfg: FSS3LiveConfig,
    metadata: dict[str, Any],
) -> Path:
    decision = pd.Timestamp(metadata["decision_time"])
    root = ensure_dir(
        cfg.report_root / "snapshots" / decision.strftime("%Y%m%dT%H%M%SZ")
    )
    frames = {
        "membership": metadata["membership_snapshot"],
        "funding_lookback": metadata["funding_snapshot"],
        "beta_prices": metadata["beta_price_snapshot"],
    }
    expected = {
        "membership": metadata["membership_hash"],
        "funding_lookback": metadata["funding_snapshot_hash"],
        "beta_prices": metadata["beta_price_snapshot_hash"],
    }
    for name, frame in frames.items():
        path = root / f"{name}.parquet"
        if not path.exists():
            write_parquet(frame, path)
        existing = pd.read_parquet(path)
        sort_by = {
            "membership": ["month_start", "symbol"],
            "funding_lookback": ["funding_time", "symbol"],
            "beta_prices": ["feature_time", "symbol"],
        }[name]
        actual = _canonical_frame_hash(existing, sort_by)
        if actual != expected[name]:
            raise RuntimeError(f"immutable FSS3 snapshot drifted: {path}")
    metadata_path = root / "metadata.json"
    payload = {
        "decision_time": decision.isoformat(),
        "membership_hash": metadata["membership_hash"],
        "funding_snapshot_hash": metadata["funding_snapshot_hash"],
        "beta_price_snapshot_hash": metadata["beta_price_snapshot_hash"],
        "decision_input_hash": metadata["decision_input_hash"],
        "real_orders_allowed": False,
    }
    if metadata_path.exists():
        existing_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_payload != payload:
            raise RuntimeError(f"immutable FSS3 metadata drifted: {metadata_path}")
    else:
        metadata_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return root


def _performance_ledger(
    decisions: pd.DataFrame,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    observed_at: pd.Timestamp,
    cfg: FSS3LiveConfig,
) -> pd.DataFrame:
    if decisions.empty:
        return _empty(PERFORMANCE_COLUMNS)
    price_data = prices.copy()
    price_data["feature_time"] = pd.to_datetime(
        price_data["feature_time"], utc=True, errors="coerce"
    )
    close = price_data.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    funding_data = funding.copy()
    funding_data["funding_time"] = pd.to_datetime(
        funding_data["funding_time"], utc=True, errors="coerce"
    )
    funding_data["funding_rate_settled"] = pd.to_numeric(
        funding_data["funding_rate_settled"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for decision_row in decisions.sort_values("decision_time").itertuples(index=False):
        decision = pd.Timestamp(decision_row.decision_time)
        scheduled_exit = decision + pd.Timedelta(days=7)
        local_weights = weights[weights["decision_time"].eq(decision)]
        weight_map = {
            str(row.symbol): float(row.weight)
            for row in local_weights.itertuples(index=False)
        }
        symbols = sorted(weight_map)
        complete = observed_at >= scheduled_exit
        mark_time = scheduled_exit if complete else pd.NaT
        status = "completed" if complete else "open"
        if symbols:
            available_symbols = [symbol for symbol in symbols if symbol in close.columns]
            if len(available_symbols) != len(symbols):
                status = "missing_price"
            else:
                common = close[symbols].dropna(how="any")
                common = common[
                    (common.index >= decision)
                    & (common.index <= min(observed_at, scheduled_exit))
                ]
                if decision not in common.index:
                    status = "missing_entry_price"
                elif complete and scheduled_exit not in common.index:
                    status = "missing_exit_price"
                elif not complete and not common.empty:
                    mark_time = common.index.max()
        else:
            mark_time = min(observed_at.floor("h"), scheduled_exit)
        price_return = np.nan
        funding_return = np.nan
        if status in {"completed", "open"} and not pd.isna(mark_time):
            if weight_map:
                price_return = float(
                    sum(
                        weight
                        * (
                            float(close.at[mark_time, symbol])
                            / float(close.at[decision, symbol])
                            - 1.0
                        )
                        for symbol, weight in weight_map.items()
                    )
                )
                funding_return = float(
                    sum(
                        -weight
                        * funding_data.loc[
                            funding_data["symbol"].eq(symbol)
                            & funding_data["funding_time"].gt(decision)
                            & funding_data["funding_time"].le(mark_time),
                            "funding_rate_settled",
                        ].sum()
                        for symbol, weight in weight_map.items()
                    )
                )
            else:
                price_return = 0.0
                funding_return = 0.0
        gross = (
            price_return + funding_return
            if np.isfinite(price_return) and np.isfinite(funding_return)
            else np.nan
        )
        primary_cost = cfg.primary_one_way_cost * float(
            decision_row.rebalance_turnover
        )
        stress_cost = cfg.stress_one_way_cost * float(
            decision_row.rebalance_turnover
        )
        evidence_eligible = bool(
            complete
            and status == "completed"
            and decision_row.timely_forward_decision
            and decision_row.decision_status == "recorded"
        )
        rows.append(
            {
                "strategy_id": CANDIDATE,
                "decision_time": decision,
                "scheduled_exit_time": scheduled_exit,
                "mark_time": mark_time,
                "status": status,
                "timely_forward_decision": bool(
                    decision_row.timely_forward_decision
                ),
                "evidence_eligible": evidence_eligible,
                "price_return": price_return,
                "funding_return": funding_return,
                "gross_return": gross,
                "rebalance_turnover": float(decision_row.rebalance_turnover),
                "primary_cost": primary_cost,
                "stress_cost": stress_cost,
                "primary_net_return": gross - primary_cost
                if np.isfinite(gross)
                else np.nan,
                "stress_net_return": gross - stress_cost
                if np.isfinite(gross)
                else np.nan,
                "mode": cfg.mode,
                "real_orders_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_status(
    root: Path, status: FSS3LiveStatus
) -> tuple[Path, Path]:
    json_path = root / "live_status.json"
    markdown_path = root / "live_status.md"
    json_path.write_text(
        json.dumps(asdict(status), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# v14.9 FSS3 Forward-Shadow Status",
        "",
        f"- status: `{status.status}`",
        f"- observed_at_utc: {status.observed_at_utc}",
        f"- lifecycle_status: `{status.lifecycle_status}`",
        f"- application: scope=`{status.scope}`, enabled=`{status.enabled}`, push_policy=`{status.push_policy}`",
        f"- real_orders_allowed: `{status.real_orders_allowed}`",
        f"- leverage_allowed: `{status.leverage_allowed}`",
        f"- latest_price_time: {status.latest_price_time}",
        f"- data_stale: {status.data_stale}",
        f"- seed_entry_time: {status.seed_entry_time}",
        f"- latest_decision_time: {status.latest_decision_time}",
        f"- latest_decision_status: `{status.latest_decision_status}`",
        f"- latest_decision_timely: `{status.latest_decision_timely}`",
        f"- total_recorded_decisions: {status.total_recorded_decisions}",
        f"- timely_recorded_decisions: {status.timely_recorded_decisions}",
        f"- completed_timely_weeks: {status.completed_timely_weeks}",
        f"- open_timely_weeks: {status.open_timely_weeks}",
        f"- current_weight_count: {status.current_weight_count}",
        f"- current_gross_notional: {status.current_gross_notional:.12f}",
        f"- mean_completed_primary_net: {status.mean_completed_primary_net}",
        f"- mean_completed_stress_net: {status.mean_completed_stress_net}",
        f"- reasons: {','.join(status.reasons) if status.reasons else 'none'}",
        f"- warnings: {','.join(status.warnings) if status.warnings else 'none'}",
        "",
        "This application records immutable decisions, weights and virtual PnL only.",
        "It has no order, push or leverage route.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def write_fss3_forward_shadow(
    funding: pd.DataFrame,
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: FSS3LiveConfig,
    *,
    observed_at: object | None = None,
) -> dict[str, Path]:
    observed = (
        pd.Timestamp.now(tz="UTC") if observed_at is None else _utc(observed_at)
    )
    root = ensure_dir(cfg.report_root)
    forward_root = ensure_dir(root / "forward")
    decisions_path = forward_root / "decisions.parquet"
    weights_path = forward_root / "executed_weights.parquet"
    performance_path = forward_root / "weekly_performance.parquet"
    decisions = _read_ledger(decisions_path, DECISION_COLUMNS)
    weights = _read_ledger(weights_path, WEIGHT_COLUMNS)
    if not weights.empty:
        weights["decision_time"] = pd.to_datetime(
            weights["decision_time"], utc=True, errors="coerce"
        )
    seed_entry, seed_weights, seed_hash = _seed_state(cfg.seed_weights_path)
    latest_due = latest_monday(observed)
    due = pd.date_range(
        seed_entry + pd.Timedelta(days=7), latest_due, freq="7D", tz="UTC"
    )
    reasons: list[str] = []
    warnings: list[str] = []

    for decision in due:
        decision = pd.Timestamp(decision)
        if not decisions.empty and decisions["decision_time"].eq(decision).any():
            continue
        prior = decisions[decisions["decision_time"].lt(decision)].sort_values(
            "decision_time"
        )
        if prior.empty:
            previous_weights = seed_weights
            state_source = f"seed:{seed_entry.isoformat()}"
            previous_state_hash = seed_hash
        else:
            prior_row = prior.iloc[-1]
            prior_time = pd.Timestamp(prior_row["decision_time"])
            local_prior = weights[weights["decision_time"].eq(prior_time)]
            expected_count = int(prior_row["executed_weight_count"])
            if len(local_prior) != expected_count:
                reasons.append("previous_executed_state_ambiguous")
                break
            previous_weights = {
                str(row.symbol): float(row.weight)
                for row in local_prior.itertuples(index=False)
            }
            state_source = f"decision:{prior_time.isoformat()}"
            previous_state_hash = _weights_hash(previous_weights)

        local, metadata = build_fss3_decision_input(
            funding, prices, membership, decision, cfg
        )
        input_reasons = list(metadata["reasons"])
        decision_reason = ""
        if input_reasons:
            grace_end = decision + pd.Timedelta(
                minutes=cfg.incomplete_input_grace_minutes
            )
            if observed <= grace_end:
                reasons.extend(input_reasons)
                break
            if cfg.snapshot_inputs:
                _write_snapshot(cfg, metadata)
            executed = {}
            details = {
                "decision_status": "input_fail_closed",
                "negative_breadth": int(
                    pd.to_numeric(
                        local.get("score_7d", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .lt(0)
                    .sum()
                ),
                "positive_breadth": int(
                    pd.to_numeric(
                        local.get("score_7d", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    .gt(0)
                    .sum()
                ),
                "executed_target_fraction": 0.0,
                "target_tracking_l1": 0.0,
                "cap_binding": False,
                "rebalance_turnover": float(
                    sum(abs(weight) for weight in previous_weights.values())
                ),
                "cap_breach": 0.0,
                "gross_notional": 0.0,
                "residual_btc_beta": 0.0,
            }
            decision_reason = ";".join(input_reasons)
            warnings.append(f"input_fail_closed:{decision.isoformat()}")
        else:
            if cfg.snapshot_inputs:
                _write_snapshot(cfg, metadata)
            executed, details = build_fss3_weights(local, previous_weights, cfg)
            if details["decision_status"] != "recorded":
                decision_reason = "minimum_side_breadth_not_met"
        lag_minutes = (observed - decision).total_seconds() / 60
        timely = bool(0 <= lag_minutes <= cfg.timely_lag_minutes)
        if not timely:
            warnings.append(f"non_timely_state_catchup:{decision.isoformat()}")
            if not cfg.allow_historical_state_catchup:
                reasons.append("historical_state_catchup_disabled")
                break
        decision_row = {
            "strategy_id": CANDIDATE,
            "decision_time": decision,
            "exit_time": decision + pd.Timedelta(days=7),
            "month_start": metadata["month_start"],
            "decision_status": details["decision_status"],
            "first_observed_at_utc": observed,
            "first_observation_lag_minutes": lag_minutes,
            "timely_forward_decision": timely,
            "evidence_eligible": bool(
                timely and details["decision_status"] == "recorded"
            ),
            "state_source": state_source,
            "previous_state_hash": previous_state_hash,
            "membership_hash": metadata["membership_hash"],
            "funding_snapshot_hash": metadata["funding_snapshot_hash"],
            "beta_price_snapshot_hash": metadata["beta_price_snapshot_hash"],
            "decision_input_hash": metadata["decision_input_hash"],
            "negative_breadth": details["negative_breadth"],
            "positive_breadth": details["positive_breadth"],
            "selected_long_symbols": "|".join(
                sorted(
                    symbol
                    for symbol, weight in executed.items()
                    if symbol != BTC and weight > 0
                )
            ),
            "selected_short_symbols": "|".join(
                sorted(
                    symbol
                    for symbol, weight in executed.items()
                    if symbol != BTC and weight < 0
                )
            ),
            "executed_target_fraction": details["executed_target_fraction"],
            "target_tracking_l1": details["target_tracking_l1"],
            "cap_binding": details["cap_binding"],
            "rebalance_turnover": details["rebalance_turnover"],
            "cap_breach": details["cap_breach"],
            "gross_notional": details["gross_notional"],
            "residual_btc_beta": details["residual_btc_beta"],
            "executed_weight_count": len(executed),
            "reason": decision_reason,
            "mode": cfg.mode,
            "push_policy": cfg.push_policy,
            "real_orders_allowed": False,
            "leverage_allowed": False,
        }
        new_decision = pd.DataFrame([decision_row], columns=DECISION_COLUMNS)
        decisions = (
            new_decision
            if decisions.empty
            else pd.concat([decisions, new_decision], ignore_index=True)
        )
        new_weight_rows = pd.DataFrame(
            [
                {
                    "strategy_id": CANDIDATE,
                    "decision_time": decision,
                    "symbol": symbol,
                    "weight": value,
                    "is_btc_hedge": symbol == BTC,
                    "mode": cfg.mode,
                    "real_orders_allowed": False,
                }
                for symbol, value in executed.items()
            ],
            columns=WEIGHT_COLUMNS,
        )
        weights = (
            new_weight_rows
            if weights.empty
            else pd.concat([weights, new_weight_rows], ignore_index=True)
        )

    decisions = decisions.sort_values("decision_time").drop_duplicates(
        "decision_time", keep="first"
    )
    weights = weights.sort_values(["decision_time", "symbol"]).drop_duplicates(
        ["decision_time", "symbol"], keep="first"
    )
    performance = _performance_ledger(
        decisions, weights, prices, funding, observed, cfg
    )
    write_parquet(decisions, decisions_path)
    write_parquet(weights, weights_path)
    write_parquet(performance, performance_path)

    price_latest = (
        pd.to_datetime(prices["feature_time"], utc=True, errors="coerce").max()
        if not prices.empty
        else pd.NaT
    )
    data_stale = bool(
        pd.isna(price_latest)
        or observed - pd.Timestamp(price_latest)
        > pd.Timedelta(hours=cfg.price_stale_after_hours)
    )
    if data_stale:
        reasons.append("price_data_stale")
    latest_row = decisions.iloc[-1] if not decisions.empty else None
    latest_time = (
        pd.Timestamp(latest_row["decision_time"]) if latest_row is not None else pd.NaT
    )
    latest_weights = (
        weights[weights["decision_time"].eq(latest_time)]
        if not pd.isna(latest_time)
        else _empty(WEIGHT_COLUMNS)
    )
    eligible = performance[performance["evidence_eligible"].fillna(False)]
    completed = eligible[eligible["status"].eq("completed")]
    opened = eligible[eligible["status"].eq("open")]
    if latest_row is not None and latest_row["decision_status"] != "recorded":
        reasons.append(f"latest_decision_{latest_row['decision_status']}")
    if latest_row is not None and not bool(
        latest_row["timely_forward_decision"]
    ):
        warnings.append(
            f"non_timely_state_catchup:{pd.Timestamp(latest_row['decision_time']).isoformat()}"
        )
    if not performance.empty:
        latest_performance_status = str(
            performance.sort_values("decision_time").iloc[-1]["status"]
        )
        if latest_performance_status.startswith("missing_"):
            reasons.append(
                f"latest_performance_{latest_performance_status}"
            )
    status = FSS3LiveStatus(
        status="READY_RECORD_ONLY" if not reasons else "BLOCKED_RECORD_ONLY",
        observed_at_utc=observed.isoformat(),
        strategy_id=CANDIDATE,
        lifecycle_status="LIVE_RECORD_ONLY",
        scope=cfg.mode,
        enabled=cfg.enabled,
        push_policy=cfg.push_policy,
        real_orders_allowed=False,
        leverage_allowed=False,
        latest_price_time=""
        if pd.isna(price_latest)
        else pd.Timestamp(price_latest).isoformat(),
        data_stale=data_stale,
        seed_entry_time=seed_entry.isoformat(),
        latest_decision_time=""
        if pd.isna(latest_time)
        else latest_time.isoformat(),
        latest_decision_status=""
        if latest_row is None
        else str(latest_row["decision_status"]),
        latest_decision_timely=False
        if latest_row is None
        else bool(latest_row["timely_forward_decision"]),
        total_recorded_decisions=len(decisions),
        timely_recorded_decisions=int(
            decisions["timely_forward_decision"]
            .astype("boolean")
            .fillna(False)
            .sum()
        )
        if not decisions.empty
        else 0,
        completed_timely_weeks=len(completed),
        open_timely_weeks=len(opened),
        current_weight_count=len(latest_weights),
        current_gross_notional=float(latest_weights["weight"].abs().sum())
        if not latest_weights.empty
        else 0.0,
        mean_completed_primary_net=float(completed["primary_net_return"].mean())
        if not completed.empty
        else None,
        mean_completed_stress_net=float(completed["stress_net_return"].mean())
        if not completed.empty
        else None,
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
    status_json, status_md = _write_status(root, status)
    manifest_path = forward_root / "run_manifest.csv"
    _append_manifest(
        manifest_path,
        {
            "observed_at_utc": observed.isoformat(),
            "status": status.status,
            "latest_price_time": status.latest_price_time,
            "latest_decision_time": status.latest_decision_time,
            "latest_decision_timely": status.latest_decision_timely,
            "total_recorded_decisions": status.total_recorded_decisions,
            "completed_timely_weeks": status.completed_timely_weeks,
            "open_timely_weeks": status.open_timely_weeks,
            "real_orders_allowed": False,
        },
    )
    return {
        "decisions": decisions_path,
        "weights": weights_path,
        "performance": performance_path,
        "status_json": status_json,
        "status_md": status_md,
        "run_manifest": manifest_path,
    }


__all__ = [
    "DECISION_COLUMNS",
    "FSS3LiveConfig",
    "FSS3LiveStatus",
    "PERFORMANCE_COLUMNS",
    "WEIGHT_COLUMNS",
    "build_fss3_decision_input",
    "build_fss3_hourly_prices",
    "build_fss3_weights",
    "latest_monday",
    "load_fss3_live_config",
    "load_fss3_membership",
    "write_fss3_forward_shadow",
]
