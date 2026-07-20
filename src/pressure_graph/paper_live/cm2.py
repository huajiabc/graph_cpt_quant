"""Stateful record-only TG1 sleeve and fixed 80/20 CM2 forward shadow."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pressure_graph.io import ensure_dir, write_parquet


TG1 = "TG1_FORWARD_EXTENDED_TO_2026_07"
CM2 = "CM2_FIXED_80_FSS3_20_TG1"


@dataclass(frozen=True)
class CM2LiveConfig:
    base_config: Path
    live_root: Path
    membership_path: Path
    tg1_seed_portfolio_path: Path
    fss3_report_root: Path
    report_root: Path
    history_days: int = 40
    funding_lookback_days: int = 30
    funding_max_age_hours: int = 12
    refresh_interval_minutes: int = 360
    price_stale_after_hours: int = 2
    timely_lag_minutes: int = 60
    incomplete_input_grace_minutes: int = 60
    bucket_size: int = 9
    hold_rank: int = 18
    primary_one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    fss3_weight: float = 0.80
    tg1_weight: float = 0.20
    scope: str = "live_shadow"
    enabled: bool = True
    push_policy: str = "record_only"
    real_orders_allowed: bool = False
    leverage_allowed: bool = False
    snapshot_inputs: bool = True
    allow_historical_state_catchup: bool = True


@dataclass(frozen=True)
class CM2LiveStatus:
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
    latest_decision_time: str
    latest_decision_status: str
    latest_decision_timely: bool
    tg1_total_decisions: int
    tg1_timely_decisions: int
    cm2_aligned_weeks: int
    cm2_completed_evidence_weeks: int
    cm2_open_evidence_weeks: int
    current_tg1_names: int
    current_tg1_gross_notional: float
    mean_completed_primary_net: float | None
    mean_completed_stress_net: float | None
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


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
    "bybit_funding_hash",
    "binance_funding_hash",
    "bybit_price_hash",
    "binance_price_hash",
    "decision_input_hash",
    "coverage",
    "positive_spread_names",
    "selected_symbols",
    "retained_names",
    "rebalance_turnover",
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
    "rank_order",
    "venue_long",
    "venue_short",
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
    "bybit_return",
    "binance_return",
    "price_basis_return",
    "funding_spread_return",
    "gross_return",
    "rebalance_turnover",
    "primary_cost",
    "stress_cost",
    "primary_net_return",
    "stress_net_return",
    "mode",
    "real_orders_allowed",
]

CM2_COLUMNS = [
    "strategy_id",
    "decision_time",
    "scheduled_exit_time",
    "mark_time",
    "status",
    "timely_forward_decision",
    "evidence_eligible",
    "fss3_weight",
    "tg1_weight",
    "fss3_price_return",
    "tg1_price_return",
    "price_return",
    "fss3_funding_return",
    "tg1_funding_return",
    "funding_return",
    "fss3_primary_net_return",
    "tg1_primary_net_return",
    "primary_net_return",
    "fss3_stress_net_return",
    "tg1_stress_net_return",
    "stress_net_return",
    "mode",
    "real_orders_allowed",
]


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


def _hash_frame(frame: pd.DataFrame, sort_by: list[str]) -> str:
    data = frame.copy()
    if not data.empty:
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = pd.to_datetime(
                    data[column], utc=True, errors="coerce"
                ).map(lambda value: "" if pd.isna(value) else value.isoformat())
        data = data.sort_values(sort_by).reset_index(drop=True)
    payload = data.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_hash(symbols: list[str]) -> str:
    return hashlib.sha256("|".join(symbols).encode("utf-8")).hexdigest()


def load_cm2_live_config(path: str | Path) -> CM2LiveConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    application = payload["application"]
    data = payload["data"]
    decision = payload["decision"]
    execution = payload["execution"]
    construction = payload["construction"]
    forward = payload["forward"]
    cfg = CM2LiveConfig(
        base_config=Path(data["base_config"]),
        live_root=Path(data["live_root"]),
        membership_path=Path(data["membership_path"]),
        tg1_seed_portfolio_path=Path(data["tg1_seed_portfolio_path"]),
        fss3_report_root=Path(data["fss3_report_root"]),
        report_root=Path(forward["report_root"]),
        history_days=int(data["history_days"]),
        funding_lookback_days=int(data["funding_lookback_days"]),
        funding_max_age_hours=int(data["funding_max_age_hours"]),
        refresh_interval_minutes=int(data["refresh_interval_minutes"]),
        price_stale_after_hours=int(data["price_stale_after_hours"]),
        timely_lag_minutes=int(decision["timely_lag_minutes"]),
        incomplete_input_grace_minutes=int(
            decision["incomplete_input_grace_minutes"]
        ),
        bucket_size=int(decision["bucket_size"]),
        hold_rank=int(decision["hold_rank"]),
        primary_one_way_cost=float(execution["primary_one_way_bps"]) / 10_000,
        stress_one_way_cost=float(execution["stress_one_way_bps"]) / 10_000,
        fss3_weight=float(construction["fss3_weight"]),
        tg1_weight=float(construction["tg1_weight"]),
        scope=str(application["scope"]),
        enabled=bool(application["enabled"]),
        push_policy=str(application["push_policy"]),
        real_orders_allowed=bool(application["real_orders_allowed"]),
        leverage_allowed=bool(application["leverage_allowed"]),
        snapshot_inputs=bool(forward["snapshot_inputs"]),
        allow_historical_state_catchup=bool(
            forward["allow_historical_state_catchup"]
        ),
    )
    if cfg.scope != "live_shadow" or cfg.push_policy != "record_only":
        raise ValueError("CM2 forward shadow must be live_shadow/record_only")
    if cfg.real_orders_allowed or cfg.leverage_allowed:
        raise ValueError("CM2 forward shadow cannot enable orders or leverage")
    if abs(cfg.fss3_weight - 0.80) > 1e-12 or abs(cfg.tg1_weight - 0.20) > 1e-12:
        raise ValueError("CM2 allocation must remain exactly 80/20")
    return cfg


def load_membership(path: str | Path) -> pd.DataFrame:
    membership = pd.read_csv(path)
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


def _normalize_price(
    frame: pd.DataFrame, value_column: str, output_column: str
) -> pd.DataFrame:
    data = frame[["symbol", "feature_time", value_column]].rename(
        columns={value_column: output_column}
    )
    data["feature_time"] = pd.to_datetime(
        data["feature_time"], utc=True, errors="coerce"
    )
    data[output_column] = pd.to_numeric(data[output_column], errors="coerce")
    return (
        data.dropna(subset=["symbol", "feature_time", output_column])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _normalize_funding(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[["symbol", "funding_time", "funding_rate_settled"]].copy()
    data["funding_time"] = pd.to_datetime(
        data["funding_time"], utc=True, errors="coerce"
    ).dt.floor("s")
    data["funding_rate_settled"] = pd.to_numeric(
        data["funding_rate_settled"], errors="coerce"
    )
    return (
        data.dropna(subset=["symbol", "funding_time", "funding_rate_settled"])
        .drop_duplicates(["symbol", "funding_time"], keep="last")
        .sort_values(["funding_time", "symbol"])
        .reset_index(drop=True)
    )


def build_tg1_decision_input(
    bybit_funding: pd.DataFrame,
    binance_funding: pd.DataFrame,
    bybit_prices: pd.DataFrame,
    binance_prices: pd.DataFrame,
    membership: pd.DataFrame,
    decision_time: object,
    cfg: CM2LiveConfig,
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

    bybit_price = _normalize_price(bybit_prices, "close", "bybit_close")
    binance_price = _normalize_price(
        binance_prices, "binance_close", "binance_close"
    )
    bybit_fund = _normalize_funding(bybit_funding)
    binance_fund = _normalize_funding(binance_funding)
    start = decision - pd.Timedelta(days=cfg.funding_lookback_days)
    bybit_window = bybit_fund[
        bybit_fund["symbol"].isin(members)
        & bybit_fund["funding_time"].ge(start)
        & bybit_fund["funding_time"].lt(decision)
    ].copy()
    binance_window = binance_fund[
        binance_fund["symbol"].isin(members)
        & binance_fund["funding_time"].ge(start)
        & binance_fund["funding_time"].lt(decision)
    ].copy()
    bybit_entry = bybit_price[
        bybit_price["symbol"].isin(members)
        & bybit_price["feature_time"].eq(decision)
    ].copy()
    binance_entry = binance_price[
        binance_price["symbol"].isin(members)
        & binance_price["feature_time"].eq(decision)
    ].copy()
    bybit_group = {
        str(symbol): local
        for symbol, local in bybit_window.groupby("symbol", observed=True)
    }
    binance_group = {
        str(symbol): local
        for symbol, local in binance_window.groupby("symbol", observed=True)
    }
    bybit_entry_map = bybit_entry.set_index("symbol")["bybit_close"].to_dict()
    binance_entry_map = binance_entry.set_index("symbol")[
        "binance_close"
    ].to_dict()
    rows: list[dict[str, Any]] = []
    for symbol in members:
        left = bybit_group.get(symbol)
        right = binance_group.get(symbol)
        if left is None or right is None:
            continue
        latest_left = left["funding_time"].max()
        latest_right = right["funding_time"].max()
        if (
            decision - latest_left
            > pd.Timedelta(hours=cfg.funding_max_age_hours)
            or decision - latest_right
            > pd.Timedelta(hours=cfg.funding_max_age_hours)
        ):
            continue
        if symbol not in bybit_entry_map or symbol not in binance_entry_map:
            continue
        score = float(
            right["funding_rate_settled"].sum()
            - left["funding_rate_settled"].sum()
        )
        rows.append(
            {
                "entry_time": decision,
                "exit_time": decision + pd.Timedelta(days=7),
                "month_start": month,
                "symbol": symbol,
                "score_30d": score,
                "pair_gross_return": 0.0,
                "bybit_entry": float(bybit_entry_map[symbol]),
                "binance_entry": float(binance_entry_map[symbol]),
            }
        )
    local = pd.DataFrame(rows)
    if len(local) < cfg.bucket_size:
        reasons.append(f"eligible_names_below_{cfg.bucket_size}")
    positive = (
        int(pd.to_numeric(local["score_30d"], errors="coerce").gt(0).sum())
        if not local.empty
        else 0
    )
    if positive < cfg.bucket_size:
        reasons.append(f"positive_spread_names_below_{cfg.bucket_size}")

    membership_snapshot = membership[
        membership["month_start"].eq(month)
    ].copy()
    metadata = {
        "decision_time": decision,
        "month_start": month,
        "membership_snapshot": membership_snapshot,
        "bybit_funding_snapshot": bybit_window,
        "binance_funding_snapshot": binance_window,
        "bybit_price_snapshot": bybit_entry,
        "binance_price_snapshot": binance_entry,
        "membership_hash": _hash_frame(
            membership_snapshot, ["month_start", "symbol"]
        ),
        "bybit_funding_hash": _hash_frame(
            bybit_window, ["funding_time", "symbol"]
        ),
        "binance_funding_hash": _hash_frame(
            binance_window, ["funding_time", "symbol"]
        ),
        "bybit_price_hash": _hash_frame(
            bybit_entry, ["feature_time", "symbol"]
        ),
        "binance_price_hash": _hash_frame(
            binance_entry, ["feature_time", "symbol"]
        ),
        "decision_input_hash": _hash_frame(local, ["entry_time", "symbol"]),
        "coverage": len(local),
        "positive_spread_names": positive,
        "reasons": tuple(reasons),
    }
    return local, metadata


def build_tg1_weights(
    local: pd.DataFrame,
    previous_symbols: list[str],
    cfg: CM2LiveConfig,
) -> tuple[list[str], dict[str, float], dict[str, Any]]:
    ranked = local.dropna(subset=["score_30d", "pair_gross_return"])
    ranked = ranked[ranked["score_30d"].gt(0)].sort_values(
        ["score_30d", "symbol"], ascending=[False, True]
    )
    ranks = {
        str(symbol): rank
        for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)
    }
    retained = [
        symbol
        for symbol in previous_symbols
        if symbol in ranks and ranks[symbol] <= cfg.hold_rank
    ]
    selected = retained[: cfg.bucket_size]
    for symbol in ranked["symbol"].astype(str):
        if len(selected) >= cfg.bucket_size:
            break
        if symbol not in selected:
            selected.append(symbol)
    if len(selected) < cfg.bucket_size:
        selected = []
    if not selected:
        return [], {}, {
            "decision_status": "no_signal_fail_closed",
            "retained_names": 0,
            "rebalance_turnover": 1.0 if previous_symbols else 0.0,
        }
    weights = {symbol: 1.0 / cfg.bucket_size for symbol in selected}
    previous = {
        symbol: 1.0 / cfg.bucket_size for symbol in previous_symbols
    }
    turnover = float(
        sum(
            abs(weights.get(symbol, 0.0) - previous.get(symbol, 0.0))
            for symbol in set(weights) | set(previous)
        )
    )
    return selected, weights, {
        "decision_status": "recorded",
        "retained_names": len(set(selected) & set(previous_symbols)),
        "rebalance_turnover": turnover,
    }


def _seed_state(path: Path) -> tuple[pd.Timestamp, list[str], str]:
    portfolio = pd.read_parquet(path)
    portfolio["entry_time"] = pd.to_datetime(
        portfolio["entry_time"], utc=True, errors="coerce"
    )
    latest = pd.Timestamp(portfolio["entry_time"].max())
    row = portfolio[portfolio["entry_time"].eq(latest)].iloc[-1]
    symbols = [item for item in str(row["selected_symbols"]).split("|") if item]
    return latest, symbols, _state_hash(symbols)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(path)
    for column in (
        "decision_time",
        "exit_time",
        "first_observed_at_utc",
        "scheduled_exit_time",
        "mark_time",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(
                frame[column], utc=True, errors="coerce"
            )
    return frame


def _write_snapshot(cfg: CM2LiveConfig, metadata: dict[str, Any]) -> None:
    decision = pd.Timestamp(metadata["decision_time"])
    root = ensure_dir(
        cfg.report_root
        / "tg1"
        / "snapshots"
        / decision.strftime("%Y%m%dT%H%M%SZ")
    )
    frames = {
        "membership": metadata["membership_snapshot"],
        "bybit_funding": metadata["bybit_funding_snapshot"],
        "binance_funding": metadata["binance_funding_snapshot"],
        "bybit_entry_prices": metadata["bybit_price_snapshot"],
        "binance_entry_prices": metadata["binance_price_snapshot"],
    }
    hashes = {
        "membership": metadata["membership_hash"],
        "bybit_funding": metadata["bybit_funding_hash"],
        "binance_funding": metadata["binance_funding_hash"],
        "bybit_entry_prices": metadata["bybit_price_hash"],
        "binance_entry_prices": metadata["binance_price_hash"],
    }
    sorts = {
        "membership": ["month_start", "symbol"],
        "bybit_funding": ["funding_time", "symbol"],
        "binance_funding": ["funding_time", "symbol"],
        "bybit_entry_prices": ["feature_time", "symbol"],
        "binance_entry_prices": ["feature_time", "symbol"],
    }
    for name, frame in frames.items():
        path = root / f"{name}.parquet"
        if not path.exists():
            write_parquet(frame, path)
        existing = pd.read_parquet(path)
        if _hash_frame(existing, sorts[name]) != hashes[name]:
            raise RuntimeError(f"immutable TG1 snapshot drifted: {path}")
    payload = {
        "decision_time": decision.isoformat(),
        **{f"{name}_hash": value for name, value in hashes.items()},
        "decision_input_hash": metadata["decision_input_hash"],
        "real_orders_allowed": False,
    }
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"immutable TG1 metadata drifted: {metadata_path}")
    else:
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tg1_performance(
    decisions: pd.DataFrame,
    weights: pd.DataFrame,
    bybit_funding: pd.DataFrame,
    binance_funding: pd.DataFrame,
    bybit_prices: pd.DataFrame,
    binance_prices: pd.DataFrame,
    observed: pd.Timestamp,
    cfg: CM2LiveConfig,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)
    bybit_price = _normalize_price(bybit_prices, "close", "bybit_close")
    binance_price = _normalize_price(
        binance_prices, "binance_close", "binance_close"
    )
    bybit_close = bybit_price.pivot_table(
        index="feature_time",
        columns="symbol",
        values="bybit_close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    binance_close = binance_price.pivot_table(
        index="feature_time",
        columns="symbol",
        values="binance_close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    bybit_fund = _normalize_funding(bybit_funding)
    binance_fund = _normalize_funding(binance_funding)
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
        complete = observed >= scheduled_exit
        mark_time = scheduled_exit if complete else pd.NaT
        status = "completed" if complete else "open"
        common = pd.DatetimeIndex([], tz="UTC")
        if symbols:
            if not set(symbols).issubset(bybit_close.columns) or not set(
                symbols
            ).issubset(binance_close.columns):
                status = "missing_price"
            else:
                common = bybit_close[symbols].dropna(how="any").index.intersection(
                    binance_close[symbols].dropna(how="any").index
                )
                common = common[
                    (common >= decision)
                    & (common <= min(observed.floor("h"), scheduled_exit))
                ]
                if decision not in common:
                    status = "missing_entry_price"
                elif complete and scheduled_exit not in common:
                    status = "missing_exit_price"
                elif not complete and len(common):
                    mark_time = common.max()
        else:
            mark_time = min(observed.floor("h"), scheduled_exit)

        bybit_return = np.nan
        binance_return = np.nan
        funding_spread = np.nan
        if status in {"completed", "open"} and not pd.isna(mark_time):
            if weight_map:
                bybit_return = float(
                    sum(
                        weight
                        * (
                            float(bybit_close.at[mark_time, symbol])
                            / float(bybit_close.at[decision, symbol])
                            - 1.0
                        )
                        for symbol, weight in weight_map.items()
                    )
                )
                binance_return = float(
                    sum(
                        weight
                        * (
                            float(binance_close.at[mark_time, symbol])
                            / float(binance_close.at[decision, symbol])
                            - 1.0
                        )
                        for symbol, weight in weight_map.items()
                    )
                )
                funding_spread = float(
                    sum(
                        weight
                        * (
                            binance_fund.loc[
                                binance_fund["symbol"].eq(symbol)
                                & binance_fund["funding_time"].gt(decision)
                                & binance_fund["funding_time"].le(mark_time),
                                "funding_rate_settled",
                            ].sum()
                            - bybit_fund.loc[
                                bybit_fund["symbol"].eq(symbol)
                                & bybit_fund["funding_time"].gt(decision)
                                & bybit_fund["funding_time"].le(mark_time),
                                "funding_rate_settled",
                            ].sum()
                        )
                        for symbol, weight in weight_map.items()
                    )
                )
            else:
                bybit_return = binance_return = funding_spread = 0.0
        price_basis = (
            bybit_return - binance_return
            if np.isfinite(bybit_return) and np.isfinite(binance_return)
            else np.nan
        )
        gross = (
            price_basis + funding_spread
            if np.isfinite(price_basis) and np.isfinite(funding_spread)
            else np.nan
        )
        primary_cost = cfg.primary_one_way_cost * float(
            decision_row.rebalance_turnover
        )
        stress_cost = cfg.stress_one_way_cost * float(
            decision_row.rebalance_turnover
        )
        eligible = bool(
            complete
            and status == "completed"
            and decision_row.timely_forward_decision
            and decision_row.decision_status == "recorded"
        )
        rows.append(
            {
                "strategy_id": TG1,
                "decision_time": decision,
                "scheduled_exit_time": scheduled_exit,
                "mark_time": mark_time,
                "status": status,
                "timely_forward_decision": bool(
                    decision_row.timely_forward_decision
                ),
                "evidence_eligible": eligible,
                "bybit_return": bybit_return,
                "binance_return": binance_return,
                "price_basis_return": price_basis,
                "funding_spread_return": funding_spread,
                "gross_return": gross,
                "rebalance_turnover": float(decision_row.rebalance_turnover),
                "primary_cost": primary_cost,
                "stress_cost": stress_cost,
                "primary_net_return": (
                    gross - primary_cost if np.isfinite(gross) else np.nan
                ),
                "stress_net_return": (
                    gross - stress_cost if np.isfinite(gross) else np.nan
                ),
                "mode": cfg.scope,
                "real_orders_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)


def _build_cm2_performance(
    tg1: pd.DataFrame,
    fss3_report_root: Path,
    cfg: CM2LiveConfig,
) -> tuple[pd.DataFrame, list[str]]:
    fss3_path = fss3_report_root / "forward" / "weekly_performance.parquet"
    if not fss3_path.exists():
        return pd.DataFrame(columns=CM2_COLUMNS), ["fss3_performance_missing"]
    fss3 = pd.read_parquet(fss3_path)
    for column in ("decision_time", "scheduled_exit_time", "mark_time"):
        fss3[column] = pd.to_datetime(fss3[column], utc=True, errors="coerce")
    merged = fss3.merge(
        tg1,
        on="decision_time",
        how="inner",
        suffixes=("_fss3", "_tg1"),
        validate="one_to_one",
    )
    missing = sorted(
        set(tg1["decision_time"]) - set(merged["decision_time"])
    )
    reasons = (
        ["fss3_calendar_missing:" + "|".join(item.isoformat() for item in missing)]
        if missing
        else []
    )
    rows: list[dict[str, Any]] = []
    for row in merged.sort_values("decision_time").itertuples(index=False):
        aligned_exit = (
            pd.Timestamp(row.scheduled_exit_time_fss3)
            == pd.Timestamp(row.scheduled_exit_time_tg1)
        )
        status = (
            "completed"
            if aligned_exit
            and row.status_fss3 == "completed"
            and row.status_tg1 == "completed"
            else "open"
            if aligned_exit
            and row.status_fss3 == "open"
            and row.status_tg1 == "open"
            else "fail_closed_alignment"
        )
        timely = bool(
            row.timely_forward_decision_fss3
            and row.timely_forward_decision_tg1
        )
        eligible = bool(
            status == "completed"
            and row.evidence_eligible_fss3
            and row.evidence_eligible_tg1
        )
        finite = all(
            np.isfinite(value)
            for value in (
                row.price_return,
                row.price_basis_return,
                row.funding_return,
                row.funding_spread_return,
                row.primary_net_return_fss3,
                row.primary_net_return_tg1,
                row.stress_net_return_fss3,
                row.stress_net_return_tg1,
            )
        )
        if not finite:
            status = "fail_closed_missing_return"
            eligible = False
        rows.append(
            {
                "strategy_id": CM2,
                "decision_time": row.decision_time,
                "scheduled_exit_time": row.scheduled_exit_time_tg1,
                "mark_time": min(
                    pd.Timestamp(row.mark_time_fss3),
                    pd.Timestamp(row.mark_time_tg1),
                )
                if not pd.isna(row.mark_time_fss3)
                and not pd.isna(row.mark_time_tg1)
                else pd.NaT,
                "status": status,
                "timely_forward_decision": timely,
                "evidence_eligible": eligible,
                "fss3_weight": cfg.fss3_weight,
                "tg1_weight": cfg.tg1_weight,
                "fss3_price_return": row.price_return,
                "tg1_price_return": row.price_basis_return,
                "price_return": cfg.fss3_weight * row.price_return
                + cfg.tg1_weight * row.price_basis_return,
                "fss3_funding_return": row.funding_return,
                "tg1_funding_return": row.funding_spread_return,
                "funding_return": cfg.fss3_weight * row.funding_return
                + cfg.tg1_weight * row.funding_spread_return,
                "fss3_primary_net_return": row.primary_net_return_fss3,
                "tg1_primary_net_return": row.primary_net_return_tg1,
                "primary_net_return": cfg.fss3_weight
                * row.primary_net_return_fss3
                + cfg.tg1_weight * row.primary_net_return_tg1,
                "fss3_stress_net_return": row.stress_net_return_fss3,
                "tg1_stress_net_return": row.stress_net_return_tg1,
                "stress_net_return": cfg.fss3_weight
                * row.stress_net_return_fss3
                + cfg.tg1_weight * row.stress_net_return_tg1,
                "mode": cfg.scope,
                "real_orders_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=CM2_COLUMNS), reasons


def _write_status(root: Path, status: CM2LiveStatus) -> tuple[Path, Path]:
    json_path = root / "live_status.json"
    markdown_path = root / "live_status.md"
    json_path.write_text(
        json.dumps(asdict(status), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# v16.5 CM2 Forward-Shadow Status",
        "",
        f"- status: `{status.status}`",
        f"- observed_at_utc: {status.observed_at_utc}",
        f"- lifecycle_status: `{status.lifecycle_status}`",
        f"- application: scope=`{status.scope}`, enabled=`{status.enabled}`, push_policy=`{status.push_policy}`",
        f"- real_orders_allowed: `{status.real_orders_allowed}`",
        f"- leverage_allowed: `{status.leverage_allowed}`",
        f"- latest_price_time: {status.latest_price_time}",
        f"- latest_decision_time: {status.latest_decision_time}",
        f"- latest_decision_status: `{status.latest_decision_status}`",
        f"- latest_decision_timely: `{status.latest_decision_timely}`",
        f"- tg1_total_decisions: {status.tg1_total_decisions}",
        f"- tg1_timely_decisions: {status.tg1_timely_decisions}",
        f"- cm2_aligned_weeks: {status.cm2_aligned_weeks}",
        f"- cm2_completed_evidence_weeks: {status.cm2_completed_evidence_weeks}",
        f"- cm2_open_evidence_weeks: {status.cm2_open_evidence_weeks}",
        f"- current_tg1_names: {status.current_tg1_names}",
        f"- current_tg1_gross_notional: {status.current_tg1_gross_notional:.12f}",
        f"- mean_completed_primary_net: {status.mean_completed_primary_net}",
        f"- mean_completed_stress_net: {status.mean_completed_stress_net}",
        f"- reasons: {','.join(status.reasons) if status.reasons else 'none'}",
        f"- warnings: {','.join(status.warnings) if status.warnings else 'none'}",
        "",
        "CM2 is the exact 80% FSS3 / 20% TG1 fixed allocation.",
        "TG1 remains an internal reference sleeve, not a standalone promoted alpha.",
        "All outputs are virtual records; no order, push, leverage, or cross-sleeve netting path exists.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def write_cm2_forward_shadow(
    bybit_funding: pd.DataFrame,
    binance_funding: pd.DataFrame,
    bybit_prices: pd.DataFrame,
    binance_prices: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: CM2LiveConfig,
    *,
    observed_at: object | None = None,
) -> dict[str, Path]:
    observed = (
        pd.Timestamp.now(tz="UTC") if observed_at is None else _utc(observed_at)
    )
    root = ensure_dir(cfg.report_root)
    tg1_root = ensure_dir(root / "tg1" / "forward")
    decisions_path = tg1_root / "decisions.parquet"
    weights_path = tg1_root / "executed_weights.parquet"
    performance_path = tg1_root / "weekly_performance.parquet"
    cm2_path = ensure_dir(root / "forward") / "weekly_performance.parquet"
    decisions = _read(decisions_path, DECISION_COLUMNS)
    weights = _read(weights_path, WEIGHT_COLUMNS)
    seed_time, seed_symbols, seed_hash = _seed_state(
        cfg.tg1_seed_portfolio_path
    )
    due = pd.date_range(
        seed_time + pd.Timedelta(days=7),
        latest_monday(observed),
        freq="7D",
        tz="UTC",
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
            previous_symbols = seed_symbols
            state_source = f"seed:{seed_time.isoformat()}"
            previous_state_hash = seed_hash
        else:
            prior_row = prior.iloc[-1]
            previous_symbols = [
                item
                for item in str(prior_row["selected_symbols"]).split("|")
                if item
            ]
            if len(previous_symbols) != int(prior_row["executed_weight_count"]):
                reasons.append("previous_tg1_state_ambiguous")
                break
            state_source = f"decision:{pd.Timestamp(prior_row['decision_time']).isoformat()}"
            previous_state_hash = _state_hash(previous_symbols)
        local, metadata = build_tg1_decision_input(
            bybit_funding,
            binance_funding,
            bybit_prices,
            binance_prices,
            membership,
            decision,
            cfg,
        )
        input_reasons = list(metadata["reasons"])
        if input_reasons and observed <= decision + pd.Timedelta(
            minutes=cfg.incomplete_input_grace_minutes
        ):
            reasons.extend(input_reasons)
            break
        if cfg.snapshot_inputs:
            _write_snapshot(cfg, metadata)
        if input_reasons:
            selected: list[str] = []
            selected_weights: dict[str, float] = {}
            details = {
                "decision_status": "input_fail_closed",
                "retained_names": 0,
                "rebalance_turnover": 1.0 if previous_symbols else 0.0,
            }
            decision_reason = ";".join(input_reasons)
            warnings.append(f"input_fail_closed:{decision.isoformat()}")
        else:
            selected, selected_weights, details = build_tg1_weights(
                local, previous_symbols, cfg
            )
            decision_reason = (
                ""
                if details["decision_status"] == "recorded"
                else "minimum_positive_spread_breadth_not_met"
            )
        lag = (observed - decision).total_seconds() / 60
        timely = bool(0 <= lag <= cfg.timely_lag_minutes)
        if not timely:
            warnings.append(f"non_timely_state_catchup:{decision.isoformat()}")
            if not cfg.allow_historical_state_catchup:
                reasons.append("historical_state_catchup_disabled")
                break
        row = {
            "strategy_id": TG1,
            "decision_time": decision,
            "exit_time": decision + pd.Timedelta(days=7),
            "month_start": metadata["month_start"],
            "decision_status": details["decision_status"],
            "first_observed_at_utc": observed,
            "first_observation_lag_minutes": lag,
            "timely_forward_decision": timely,
            "evidence_eligible": bool(
                timely and details["decision_status"] == "recorded"
            ),
            "state_source": state_source,
            "previous_state_hash": previous_state_hash,
            "membership_hash": metadata["membership_hash"],
            "bybit_funding_hash": metadata["bybit_funding_hash"],
            "binance_funding_hash": metadata["binance_funding_hash"],
            "bybit_price_hash": metadata["bybit_price_hash"],
            "binance_price_hash": metadata["binance_price_hash"],
            "decision_input_hash": metadata["decision_input_hash"],
            "coverage": metadata["coverage"],
            "positive_spread_names": metadata["positive_spread_names"],
            "selected_symbols": "|".join(selected),
            "retained_names": details["retained_names"],
            "rebalance_turnover": details["rebalance_turnover"],
            "executed_weight_count": len(selected_weights),
            "reason": decision_reason,
            "mode": cfg.scope,
            "push_policy": cfg.push_policy,
            "real_orders_allowed": False,
            "leverage_allowed": False,
        }
        decisions = pd.concat(
            [decisions, pd.DataFrame([row], columns=DECISION_COLUMNS)],
            ignore_index=True,
        )
        new_weights = pd.DataFrame(
            [
                {
                    "strategy_id": TG1,
                    "decision_time": decision,
                    "symbol": symbol,
                    "weight": selected_weights[symbol],
                    "rank_order": rank,
                    "venue_long": "bybit_linear_perpetual",
                    "venue_short": "binance_usdm_perpetual",
                    "mode": cfg.scope,
                    "real_orders_allowed": False,
                }
                for rank, symbol in enumerate(selected, start=1)
            ],
            columns=WEIGHT_COLUMNS,
        )
        weights = pd.concat([weights, new_weights], ignore_index=True)
    decisions = decisions.sort_values("decision_time").drop_duplicates(
        "decision_time", keep="first"
    )
    weights = weights.sort_values(
        ["decision_time", "rank_order", "symbol"]
    ).drop_duplicates(["decision_time", "symbol"], keep="first")
    performance = _tg1_performance(
        decisions,
        weights,
        bybit_funding,
        binance_funding,
        bybit_prices,
        binance_prices,
        observed,
        cfg,
    )
    cm2, alignment_reasons = _build_cm2_performance(
        performance, cfg.fss3_report_root, cfg
    )
    reasons.extend(alignment_reasons)
    write_parquet(decisions, decisions_path)
    write_parquet(weights, weights_path)
    write_parquet(performance, performance_path)
    write_parquet(cm2, cm2_path)

    latest_bybit = (
        pd.to_datetime(bybit_prices["feature_time"], utc=True, errors="coerce").max()
        if not bybit_prices.empty
        else pd.NaT
    )
    latest_binance = (
        pd.to_datetime(
            binance_prices["feature_time"], utc=True, errors="coerce"
        ).max()
        if not binance_prices.empty
        else pd.NaT
    )
    latest_price = min(latest_bybit, latest_binance)
    if pd.isna(latest_price) or observed - latest_price > pd.Timedelta(
        hours=cfg.price_stale_after_hours
    ):
        reasons.append("price_data_stale")
    latest_row = decisions.iloc[-1] if not decisions.empty else None
    latest_time = (
        pd.Timestamp(latest_row["decision_time"])
        if latest_row is not None
        else pd.NaT
    )
    latest_weights = (
        weights[weights["decision_time"].eq(latest_time)]
        if not pd.isna(latest_time)
        else pd.DataFrame(columns=WEIGHT_COLUMNS)
    )
    completed = cm2[cm2["evidence_eligible"].fillna(False)]
    opened = cm2[
        cm2["timely_forward_decision"].fillna(False)
        & cm2["status"].eq("open")
    ]
    if latest_row is not None and latest_row["decision_status"] != "recorded":
        reasons.append(f"latest_tg1_{latest_row['decision_status']}")
    status = CM2LiveStatus(
        status="READY_RECORD_ONLY" if not reasons else "BLOCKED_RECORD_ONLY",
        observed_at_utc=observed.isoformat(),
        strategy_id=CM2,
        lifecycle_status="LIVE_RECORD_ONLY",
        scope=cfg.scope,
        enabled=cfg.enabled,
        push_policy=cfg.push_policy,
        real_orders_allowed=False,
        leverage_allowed=False,
        latest_price_time=(
            "" if pd.isna(latest_price) else latest_price.isoformat()
        ),
        latest_decision_time=(
            "" if pd.isna(latest_time) else latest_time.isoformat()
        ),
        latest_decision_status=(
            "" if latest_row is None else str(latest_row["decision_status"])
        ),
        latest_decision_timely=(
            False
            if latest_row is None
            else bool(latest_row["timely_forward_decision"])
        ),
        tg1_total_decisions=len(decisions),
        tg1_timely_decisions=int(
            decisions["timely_forward_decision"]
            .astype("boolean")
            .fillna(False)
            .sum()
        )
        if not decisions.empty
        else 0,
        cm2_aligned_weeks=len(cm2),
        cm2_completed_evidence_weeks=len(completed),
        cm2_open_evidence_weeks=len(opened),
        current_tg1_names=len(latest_weights),
        current_tg1_gross_notional=float(latest_weights["weight"].abs().sum())
        if not latest_weights.empty
        else 0.0,
        mean_completed_primary_net=float(
            completed["primary_net_return"].mean()
        )
        if not completed.empty
        else None,
        mean_completed_stress_net=float(
            completed["stress_net_return"].mean()
        )
        if not completed.empty
        else None,
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
    status_json, status_md = _write_status(root, status)
    return {
        "tg1_decisions": decisions_path,
        "tg1_weights": weights_path,
        "tg1_performance": performance_path,
        "cm2_performance": cm2_path,
        "status_json": status_json,
        "status_md": status_md,
    }


__all__ = [
    "CM2",
    "TG1",
    "CM2LiveConfig",
    "build_tg1_decision_input",
    "build_tg1_weights",
    "latest_monday",
    "load_cm2_live_config",
    "load_membership",
    "write_cm2_forward_shadow",
]
