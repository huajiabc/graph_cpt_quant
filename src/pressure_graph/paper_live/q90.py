"""Delayed-archive forward ledger for the frozen v23.8 q90 OCO candidate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pressure_graph.io import ensure_dir, write_parquet


CANDIDATE = "DVB5_POSITIVE_PRESSURE_0625SIGMA_BTC_BREAKOUT"
FROZEN_SYMBOLS = (
    "SOLUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",
    "WIFUSDT",
    "ETHUSDT",
    "ENAUSDT",
    "HBARUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ONDOUSDT",
    "XRPUSDT",
    "XLMUSDT",
    "FARTCOINUSDT",
    "WLDUSDT",
    "SEIUSDT",
    "TIAUSDT",
)


@dataclass(frozen=True)
class Q90LiveConfig:
    base_config: Path
    historical_feature_root: Path
    forward_root: Path
    report_root: Path
    historical_cutoff: pd.Timestamp
    first_forward_book_day: pd.Timestamp
    archive_publication_lag_days: int = 1
    maximum_days_per_run: int = 1
    download_workers: int = 8
    sigma_multiple: float = 0.625
    horizon_hours: int = 4
    bar_minutes: int = 15
    primary_cost: float = 0.001
    stress_cost: float = 0.002
    scope: str = "live_shadow"
    enabled: bool = True
    push_policy: str = "record_only"
    real_orders_allowed: bool = False
    leverage_allowed: bool = False
    timely_execution_eligible: bool = False
    forward_research_evidence_eligible: bool = True


def load_q90_live_config(path: str | Path) -> Q90LiveConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    application = payload["application"]
    data = payload["data"]
    signal = payload["signal"]
    forward = payload["forward"]
    cfg = Q90LiveConfig(
        base_config=Path(data["base_config"]),
        historical_feature_root=Path(data["historical_feature_root"]),
        forward_root=Path(data["forward_root"]),
        report_root=Path(data["report_root"]),
        historical_cutoff=pd.Timestamp(data["historical_cutoff"]),
        first_forward_book_day=pd.Timestamp(data["first_forward_book_day"]),
        archive_publication_lag_days=int(
            data["archive_publication_lag_days"]
        ),
        maximum_days_per_run=int(data["maximum_days_per_run"]),
        download_workers=int(data["download_workers"]),
        sigma_multiple=float(signal["barrier_sigma_multiple"]),
        horizon_hours=int(signal["horizon_hours"]),
        bar_minutes=int(signal["bar_minutes"]),
        primary_cost=float(signal["primary_round_trip_bps"]) / 10_000,
        stress_cost=float(signal["stress_round_trip_bps"]) / 10_000,
        scope=str(application["scope"]),
        enabled=bool(application["enabled"]),
        push_policy=str(application["push_policy"]),
        real_orders_allowed=bool(application["real_orders_allowed"]),
        leverage_allowed=bool(application["leverage_allowed"]),
        timely_execution_eligible=bool(
            forward["timely_execution_eligible"]
        ),
        forward_research_evidence_eligible=bool(
            forward["forward_research_evidence_eligible"]
        ),
    )
    if cfg.scope != "live_shadow" or cfg.push_policy != "record_only":
        raise ValueError("q90 shadow must be live_shadow/record_only")
    if cfg.real_orders_allowed or cfg.leverage_allowed:
        raise ValueError("q90 shadow cannot enable orders or leverage")
    if cfg.timely_execution_eligible:
        raise ValueError("delayed q90 archives cannot be execution-timely")
    if abs(cfg.sigma_multiple - 0.625) > 1e-12:
        raise ValueError("q90 barrier must remain frozen at 0.625 sigma")
    return cfg


def _load_book_features(root: Path) -> pd.DataFrame:
    columns = [
        "decision_time",
        "source_day",
        "symbol",
        "notional_imbalance_1p0_median",
        "notional_imbalance_1p0_valid_snapshots",
        "notional_imbalance_5p0_median",
        "total_notional_1p0_median",
        "total_notional_5p0_median",
        "archive_sha256",
    ]
    frames = [
        pd.read_parquet(root / f"{symbol}.parquet", columns=columns)
        for symbol in FROZEN_SYMBOLS
    ]
    data = pd.concat(frames, ignore_index=True)
    for column in ("decision_time", "source_day"):
        data[column] = pd.to_datetime(
            data[column], utc=True, errors="coerce"
        )
    return (
        data.dropna(subset=["decision_time", "symbol"])
        .drop_duplicates(["decision_time", "symbol"], keep="last")
        .sort_values(["symbol", "decision_time"])
        .reset_index(drop=True)
    )


def _q90_symbol_states(features: pd.DataFrame) -> pd.DataFrame:
    output = features.sort_values(["symbol", "decision_time"]).copy()
    output = output[
        output["notional_imbalance_1p0_valid_snapshots"].ge(90)
        & output["total_notional_1p0_median"].gt(0)
    ].copy()
    grouped = output.groupby("symbol", sort=False, observed=True)
    previous_depth = grouped["total_notional_1p0_median"].shift(1)
    output["depth_log_change"] = np.log(
        output["total_notional_1p0_median"] / previous_depth
    )
    output["imbalance_prior_mean"] = grouped[
        "notional_imbalance_1p0_median"
    ].transform(
        lambda values: values.shift(1)
        .rolling(720, min_periods=480)
        .mean()
    )
    output["imbalance_prior_std"] = grouped[
        "notional_imbalance_1p0_median"
    ].transform(
        lambda values: values.shift(1)
        .rolling(720, min_periods=480)
        .std()
    )
    output["imbalance_z"] = (
        output["notional_imbalance_1p0_median"]
        - output["imbalance_prior_mean"]
    ) / output["imbalance_prior_std"].replace(0, np.nan)
    output["prior_withdrawal_threshold"] = output.groupby(
        "symbol", sort=False, observed=True
    )["depth_log_change"].transform(
        lambda values: values.shift(1)
        .rolling(720, min_periods=480)
        .quantile(0.20)
    )
    output["depth_withdrawal_state"] = output["depth_log_change"].le(
        output["prior_withdrawal_threshold"]
    )
    output["feature_ready"] = np.isfinite(
        output[
            ["imbalance_z", "prior_withdrawal_threshold", "depth_log_change"]
        ]
    ).all(axis=1)
    return output.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def _q90_bucket_states(symbol_states: pd.DataFrame) -> pd.DataFrame:
    ready = symbol_states[symbol_states["feature_ready"]]
    rows: list[dict[str, object]] = []
    for decision_time, local in ready.groupby(
        "decision_time", sort=True, observed=True
    ):
        if local["symbol"].nunique() < 15:
            continue
        pressure = float(local["imbalance_z"].median())
        direction = 1 if pressure >= 0 else -1
        directional = local["imbalance_z"].mul(direction).gt(0)
        withdrawing = local["depth_withdrawal_state"]
        rows.append(
            {
                "decision_time": pd.Timestamp(decision_time),
                "covered_symbols": int(local["symbol"].nunique()),
                "bucket_pressure": pressure,
                "direction": direction,
                "directional_symbol_count": int(directional.sum()),
                "directional_breadth": float(directional.mean()),
                "withdrawing_symbol_count": int(withdrawing.sum()),
                "withdrawal_breadth": float(withdrawing.mean()),
                "directional_symbols": "|".join(
                    sorted(local.loc[directional, "symbol"].astype(str))
                ),
                "withdrawing_symbols": "|".join(
                    sorted(local.loc[withdrawing, "symbol"].astype(str))
                ),
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output = output.sort_values("decision_time").reset_index(drop=True)
    output["prior_abs_pressure_threshold"] = (
        output["bucket_pressure"]
        .abs()
        .shift(1)
        .rolling(720, min_periods=480)
        .quantile(0.90)
    )
    output["candidate_state"] = (
        output["bucket_pressure"]
        .abs()
        .ge(output["prior_abs_pressure_threshold"])
        & output["directional_symbol_count"].ge(11)
        & output["withdrawing_symbol_count"].ge(5)
    )
    return output


def _q90_events(bucket_states: pd.DataFrame) -> pd.DataFrame:
    if bucket_states.empty:
        return pd.DataFrame()
    state = bucket_states["candidate_state"]
    starts = bucket_states[state & ~state.shift(1, fill_value=False)].copy()
    selected: list[int] = []
    last_time: pd.Timestamp | None = None
    for index, row in starts.iterrows():
        decision = pd.Timestamp(row["decision_time"])
        if last_time is None or decision - last_time >= pd.Timedelta(hours=4):
            selected.append(index)
            last_time = decision
    events = starts.loc[selected].copy()
    events["candidate"] = CANDIDATE
    events["feature_time"] = events["decision_time"]
    events["entry_time"] = events["decision_time"]
    events["signal_direction"] = events["direction"].astype(int)
    events["entry_month"] = events["entry_time"].dt.strftime("%Y-%m")
    events["period"] = "holdout"
    columns = [
        "candidate",
        "feature_time",
        "entry_time",
        "entry_month",
        "period",
        "signal_direction",
        "bucket_pressure",
        "prior_abs_pressure_threshold",
        "covered_symbols",
        "directional_symbol_count",
        "directional_breadth",
        "withdrawing_symbol_count",
        "withdrawal_breadth",
        "directional_symbols",
        "withdrawing_symbols",
    ]
    return events[columns].sort_values("entry_time").reset_index(drop=True)


def _run_q90_feature_update(
    cfg: Q90LiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    historical = _load_book_features(cfg.historical_feature_root)
    forward = _load_book_features(
        cfg.forward_root / "book_depth" / "hourly_features"
    )
    cutoff = pd.Timestamp(historical["decision_time"].max())
    if cutoff != cfg.historical_cutoff:
        raise RuntimeError(
            f"q90 historical cutoff drifted: {cutoff} != {cfg.historical_cutoff}"
        )
    combined = (
        pd.concat([historical, forward], ignore_index=True)
        .drop_duplicates(["symbol", "decision_time"], keep="last")
        .sort_values(["symbol", "decision_time"])
        .reset_index(drop=True)
    )
    states = _q90_bucket_states(_q90_symbol_states(combined))
    events = _q90_events(states)
    forward_events = events[events["entry_time"].gt(cutoff)].copy()
    forward_events["forward_candidate_eligible"] = forward_events[
        "signal_direction"
    ].eq(1)
    manifest = json.loads(
        (cfg.forward_root / "forward_collection_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    forward_hours = int(forward["decision_time"].nunique())
    metadata = {
        **manifest,
        "candidate": CANDIDATE,
        "historical_cutoff": cutoff.isoformat(),
        "forward_first_decision": pd.Timestamp(
            forward["decision_time"].min()
        ).isoformat(),
        "forward_last_decision": pd.Timestamp(
            forward["decision_time"].max()
        ).isoformat(),
        "forward_symbol_hours": len(forward),
        "forward_hours": forward_hours,
        "forward_days": forward_hours / 24,
        "forward_bucket_hours": int(
            states.loc[
                states["decision_time"].gt(cutoff), "decision_time"
            ].nunique()
        ),
        "strict_v224_forward_events": len(forward_events),
        "positive_pressure_forward_events": int(
            forward_events["forward_candidate_eligible"].sum()
        ),
        "outcomes_loaded": False,
    }
    per_symbol = forward.groupby("symbol")["decision_time"].nunique()
    first = pd.Timestamp(metadata["forward_first_decision"])
    expected_times = pd.date_range(
        first, periods=forward_hours, freq="h", tz="UTC"
    )
    actual_times = pd.DatetimeIndex(
        forward["decision_time"].dropna().unique()
    ).sort_values()
    checks = pd.DataFrame(
        {
            "check": [
                "forward_panel_has_16_symbols",
                "each_symbol_has_all_forward_hours",
                "forward_hours_are_contiguous",
                "forward_starts_one_hour_after_cutoff",
                "forward_keys_unique",
                "forward_bucket_has_all_complete_hours",
                "candidate_requires_positive_pressure",
                "no_forward_outcomes_loaded",
            ],
            "passed": [
                forward["symbol"].nunique() == len(FROZEN_SYMBOLS),
                bool(per_symbol.eq(forward_hours).all()),
                actual_times.equals(expected_times),
                first == cutoff + pd.Timedelta(hours=1),
                not forward.duplicated(["symbol", "decision_time"]).any(),
                metadata["forward_bucket_hours"] == forward_hours,
                bool(
                    forward_events.loc[
                        forward_events["forward_candidate_eligible"],
                        "signal_direction",
                    ]
                    .eq(1)
                    .all()
                ),
                metadata["outcomes_loaded"] is False,
            ],
        }
    )
    if not bool(checks["passed"].all()):
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"q90 forward feature audit failed: {failed}")
    return forward, states, forward_events, {**metadata, "_checks": checks}


def _q90_event_features(
    events: pd.DataFrame, bars: pd.DataFrame, cfg: Q90LiveConfig
) -> pd.DataFrame:
    local_bars = bars.copy()
    for column in ("bar_open_time", "bar_close_time"):
        local_bars[column] = pd.to_datetime(
            local_bars[column], utc=True, errors="coerce"
        )
    for column in ("open", "high", "low", "close"):
        local_bars[column] = pd.to_numeric(
            local_bars[column], errors="coerce"
        )
    local_bars = local_bars.dropna(
        subset=[
            "bar_open_time",
            "bar_close_time",
            "open",
            "high",
            "low",
            "close",
        ]
    )
    hourly = local_bars[
        local_bars["bar_close_time"].dt.minute.eq(0)
        & local_bars["bar_close_time"].dt.second.eq(0)
    ][["bar_close_time", "close"]].rename(
        columns={"bar_close_time": "entry_time", "close": "entry_spot"}
    )
    hourly = hourly.drop_duplicates("entry_time", keep="last").sort_values(
        "entry_time"
    )
    log_move = np.log(hourly["entry_spot"] / hourly["entry_spot"].shift(1))
    hourly["prior_24h_sum_squared_log_move"] = log_move.rolling(
        24, min_periods=24
    ).apply(lambda values: float(np.square(values).sum()), raw=True)
    hourly["causal_hourly_sigma"] = np.sqrt(
        hourly["prior_24h_sum_squared_log_move"] / 24
    )
    output = events.merge(hourly, on="entry_time", how="left", validate="one_to_one")
    available = set(local_bars["bar_open_time"])
    count = cfg.horizon_hours * 60 // cfg.bar_minutes
    output["path_timestamp_count"] = [
        sum(
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset)
            in available
            for offset in range(count)
        )
        for entry in output["entry_time"]
    ]
    return output[
        output["entry_spot"].gt(0)
        & output["causal_hourly_sigma"].gt(0)
        & output["path_timestamp_count"].eq(count)
    ].copy()


def _simulate_q90_oco(
    features: pd.DataFrame, bars: pd.DataFrame, cfg: Q90LiveConfig
) -> pd.DataFrame:
    indexed = bars.copy()
    indexed["bar_open_time"] = pd.to_datetime(
        indexed["bar_open_time"], utc=True, errors="coerce"
    )
    indexed = indexed.drop_duplicates("bar_open_time", keep="last").set_index(
        "bar_open_time"
    ).sort_index()
    bar_count = cfg.horizon_hours * 60 // cfg.bar_minutes
    rows: list[dict[str, object]] = []
    for feature in features.sort_values("entry_time").itertuples(index=False):
        entry = pd.Timestamp(feature.entry_time)
        times = [
            entry + pd.Timedelta(minutes=cfg.bar_minutes * offset)
            for offset in range(bar_count)
        ]
        if any(time not in indexed.index for time in times):
            continue
        path = indexed.loc[times]
        spot = float(feature.entry_spot)
        sigma = float(feature.causal_hourly_sigma)
        upper = spot * np.exp(cfg.sigma_multiple * sigma)
        lower = spot * np.exp(-cfg.sigma_multiple * sigma)
        exit_spot = float(path.iloc[-1]["close"])
        triggered = False
        ambiguous = False
        direction = 0
        trigger_time = pd.NaT
        fill = np.nan
        gross = 0.0
        for time, bar in path.iterrows():
            upper_hit = float(bar["high"]) >= upper
            lower_hit = float(bar["low"]) <= lower
            if not upper_hit and not lower_hit:
                continue
            triggered = True
            trigger_time = pd.Timestamp(time)
            long_fill = max(upper, float(bar["open"]))
            short_fill = min(lower, float(bar["open"]))
            long_gross = exit_spot / long_fill - 1.0
            short_gross = 1.0 - exit_spot / short_fill
            if upper_hit and lower_hit:
                ambiguous = True
                direction, fill, gross = (
                    (1, long_fill, long_gross)
                    if long_gross <= short_gross
                    else (-1, short_fill, short_gross)
                )
            elif upper_hit:
                direction, fill, gross = 1, long_fill, long_gross
            else:
                direction, fill, gross = -1, short_fill, short_gross
            break
        row = feature._asdict()
        row.update(
            {
                "candidate": CANDIDATE,
                "sigma_multiple": cfg.sigma_multiple,
                "upper_stop_price": upper,
                "lower_stop_price": lower,
                "exit_time": entry + pd.Timedelta(hours=cfg.horizon_hours),
                "exit_spot": exit_spot,
                "triggered": triggered,
                "ambiguous_trigger": ambiguous,
                "trigger_time": trigger_time,
                "trigger_delay_minutes": (
                    (trigger_time - entry).total_seconds() / 60
                    if triggered
                    else np.nan
                ),
                "trade_direction": direction,
                "fill_price": fill,
                "gross_return": gross,
                "primary_net_return": gross
                - (cfg.primary_cost if triggered else 0.0),
                "stress_net_return": gross
                - (cfg.stress_cost if triggered else 0.0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def next_q90_book_day(
    cfg: Q90LiveConfig, observed_at: object
) -> pd.Timestamp | None:
    observed = pd.Timestamp(observed_at)
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    root = cfg.forward_root / "book_depth" / "daily_features"
    latest: pd.Timestamp | None = None
    if root.exists():
        days: list[pd.Timestamp] = []
        for path in root.glob("*.parquet"):
            frame = pd.read_parquet(path, columns=["source_day"])
            values = pd.to_datetime(
                frame["source_day"], utc=True, errors="coerce"
            ).dropna()
            if len(values):
                days.append(pd.Timestamp(values.max()).normalize())
        if days:
            latest = min(days)
    next_day = (
        cfg.first_forward_book_day.normalize()
        if latest is None
        else latest + pd.Timedelta(days=1)
    )
    latest_publishable = observed.normalize() - pd.Timedelta(
        days=cfg.archive_publication_lag_days
    )
    return next_day if next_day <= latest_publishable else None


def write_q90_ledgers(
    cfg: Q90LiveConfig,
    *,
    observed_at: object,
    btc_bars: pd.DataFrame | None = None,
) -> dict[str, Path]:
    observed = pd.Timestamp(observed_at)
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    forward, states, events, metadata = _run_q90_feature_update(cfg)
    checks = metadata.pop("_checks")
    positive = events[events["forward_candidate_eligible"]].copy()
    root = ensure_dir(cfg.report_root)
    feature_root = ensure_dir(root / "feature_update")
    write_parquet(
        forward,
        feature_root / "isolated_forward_symbol_features.parquet",
    )
    cutoff = pd.Timestamp(metadata["historical_cutoff"])
    write_parquet(
        states[states["decision_time"].gt(cutoff)],
        feature_root / "forward_bucket_states.parquet",
    )
    write_parquet(events, feature_root / "new_forward_events.parquet")
    checks.to_csv(feature_root / "data_quality_checks.csv", index=False)
    (feature_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    forward_root = ensure_dir(root / "forward")
    decisions_path = forward_root / "event_decisions.parquet"
    outcomes_path = forward_root / "event_outcomes.parquet"
    existing = (
        pd.read_parquet(decisions_path)
        if decisions_path.exists()
        else pd.DataFrame()
    )
    if not existing.empty:
        existing["entry_time"] = pd.to_datetime(
            existing["entry_time"], utc=True, errors="coerce"
        )
    known = set(existing.get("entry_time", pd.Series(dtype="datetime64[ns, UTC]")))
    new = positive[~positive["entry_time"].isin(known)].copy()
    if not new.empty:
        new["strategy_id"] = CANDIDATE
        new["first_feature_observed_at_utc"] = observed
        new["archive_observation_lag_minutes"] = (
            observed - new["entry_time"]
        ).dt.total_seconds() / 60
        new["decision_status"] = "recorded_delayed_archive"
        new["timely_execution_eligible"] = False
        new["forward_research_evidence_eligible"] = (
            cfg.forward_research_evidence_eligible
        )
        new["mode"] = cfg.scope
        new["push_policy"] = cfg.push_policy
        new["real_orders_allowed"] = False
        existing = pd.concat([existing, new], ignore_index=True, sort=False)
    decisions = (
        existing.sort_values("entry_time")
        .drop_duplicates("entry_time", keep="first")
        .reset_index(drop=True)
        if not existing.empty
        else pd.DataFrame(
            columns=[
                *positive.columns,
                "strategy_id",
                "first_feature_observed_at_utc",
                "archive_observation_lag_minutes",
                "decision_status",
                "timely_execution_eligible",
                "forward_research_evidence_eligible",
                "mode",
                "push_policy",
                "real_orders_allowed",
            ]
        )
    )
    write_parquet(decisions, decisions_path)

    existing_outcomes = (
        pd.read_parquet(outcomes_path)
        if outcomes_path.exists()
        else pd.DataFrame()
    )
    if not existing_outcomes.empty:
        existing_outcomes["entry_time"] = pd.to_datetime(
            existing_outcomes["entry_time"], utc=True, errors="coerce"
        )
    outcome_times = set(
        existing_outcomes.get(
            "entry_time", pd.Series(dtype="datetime64[ns, UTC]")
        )
    )
    pending = (
        decisions[
            ~decisions["entry_time"].isin(outcome_times)
            & pd.to_datetime(
                decisions["entry_time"], utc=True, errors="coerce"
            ).le(observed - pd.Timedelta(hours=cfg.horizon_hours))
        ].copy()
        if not decisions.empty
        else decisions.copy()
    )
    if not pending.empty and btc_bars is not None and not btc_bars.empty:
        event_columns = [
            "candidate",
            "feature_time",
            "entry_time",
            "entry_month",
            "period",
            "signal_direction",
            "bucket_pressure",
            "prior_abs_pressure_threshold",
            "covered_symbols",
            "directional_symbol_count",
            "directional_breadth",
            "withdrawing_symbol_count",
            "withdrawal_breadth",
        ]
        event_features = _q90_event_features(
            pending[event_columns], btc_bars, cfg
        )
        outcomes = _simulate_q90_oco(event_features, btc_bars, cfg)
        if not outcomes.empty:
            observation = decisions[
                [
                    "entry_time",
                    "first_feature_observed_at_utc",
                    "archive_observation_lag_minutes",
                    "timely_execution_eligible",
                    "forward_research_evidence_eligible",
                ]
            ]
            outcomes = outcomes.merge(
                observation,
                on="entry_time",
                how="left",
                validate="one_to_one",
            )
            outcomes["outcome_recorded_at_utc"] = observed
            outcomes["mode"] = cfg.scope
            outcomes["real_orders_allowed"] = False
            existing_outcomes = pd.concat(
                [existing_outcomes, outcomes], ignore_index=True, sort=False
            )
    completed_outcomes = (
        existing_outcomes.sort_values("entry_time")
        .drop_duplicates("entry_time", keep="first")
        .reset_index(drop=True)
        if not existing_outcomes.empty
        else existing_outcomes
    )
    write_parquet(completed_outcomes, outcomes_path)

    status_payload = {
        "status": "READY_DELAYED_FORWARD_RECORD_ONLY",
        "observed_at_utc": observed.isoformat(),
        "strategy_id": CANDIDATE,
        "lifecycle_status": "CANDIDATE_WATCH",
        "scope": cfg.scope,
        "enabled": cfg.enabled,
        "push_policy": cfg.push_policy,
        "real_orders_allowed": False,
        "leverage_allowed": False,
        "timely_execution_eligible": False,
        "forward_research_evidence_eligible": (
            cfg.forward_research_evidence_eligible
        ),
        "historical_cutoff": metadata["historical_cutoff"],
        "forward_first_decision": metadata["forward_first_decision"],
        "forward_last_decision": metadata["forward_last_decision"],
        "forward_hours": metadata["forward_hours"],
        "strict_forward_events": metadata["strict_v224_forward_events"],
        "positive_q90_forward_events": len(decisions),
        "completed_virtual_outcomes": len(completed_outcomes),
        "pending_virtual_outcomes": max(
            0, len(decisions) - len(completed_outcomes)
        ),
        "archive_delay_limitation": (
            "signals are reconstructed after daily archive publication and "
            "cannot count as timely execution evidence"
        ),
    }
    status_json = root / "live_status.json"
    status_md = root / "live_status.md"
    status_json.write_text(
        json.dumps(status_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status_md.write_text(
        "\n".join(
            [
                "# v23.8 q90 Delayed Forward-Shadow Status",
                "",
                f"- status: `{status_payload['status']}`",
                f"- observed_at_utc: {status_payload['observed_at_utc']}",
                "- lifecycle_status: `CANDIDATE_WATCH`",
                "- application: scope=`live_shadow`, enabled=`True`, push_policy=`record_only`",
                "- real_orders_allowed: `False`",
                "- leverage_allowed: `False`",
                "- timely_execution_eligible: `False`",
                f"- forward_hours: {status_payload['forward_hours']}",
                f"- positive_q90_forward_events: {len(decisions)}",
                f"- completed_virtual_outcomes: {len(completed_outcomes)}",
                "",
                "Daily Binance book-depth archives are delayed. This ledger is valid",
                "for untouched forward-research evidence, but not for live execution",
                "latency or fill evidence. It contains no order route.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "decisions": decisions_path,
        "outcomes": outcomes_path,
        "status_json": status_json,
        "status_md": status_md,
    }


__all__ = [
    "FROZEN_SYMBOLS",
    "Q90LiveConfig",
    "load_q90_live_config",
    "next_q90_book_day",
    "write_q90_ledgers",
]
