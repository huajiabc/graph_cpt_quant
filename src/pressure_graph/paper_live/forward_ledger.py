from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir, write_parquet


@dataclass(frozen=True)
class ForwardLedgerConfig:
    timely_lag_minutes: int = 30


def _utc_timestamp(value: object | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _key_frame(frame: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="string")
    missing = [col for col in key_cols if col not in frame.columns]
    if missing:
        raise KeyError(f"forward ledger key columns missing: {missing}")
    return frame[key_cols].fillna("").astype(str).agg("|".join, axis=1)


def _first_observation_fields(
    frame: pd.DataFrame,
    observed_at: pd.Timestamp,
    event_time_col: str | None,
    timely_lag_minutes: int,
) -> pd.DataFrame:
    out = frame.copy()
    out["first_observed_at_utc"] = observed_at
    out["last_observed_at_utc"] = observed_at
    if event_time_col and event_time_col in out.columns:
        event_time = pd.to_datetime(out[event_time_col], utc=True, errors="coerce")
        lag = (observed_at - event_time).dt.total_seconds()
        out["first_observation_lag_seconds"] = lag
        out["timely_forward_observation"] = lag.ge(0.0) & lag.le(float(timely_lag_minutes) * 60.0)
    else:
        out["first_observation_lag_seconds"] = np.nan
        out["timely_forward_observation"] = False
    return out


def merge_cumulative_ledger(
    current: pd.DataFrame,
    path: Path,
    *,
    key_cols: list[str],
    observed_at: pd.Timestamp,
    event_time_col: str | None,
    timely_lag_minutes: int,
    eligibility_by_signal: pd.Series | None = None,
) -> pd.DataFrame:
    """Merge a rolling view into a cumulative ledger without losing first-seen evidence."""
    current = current.copy()
    if current.empty:
        if path.exists():
            return pd.read_parquet(path)
        write_parquet(current, path)
        return current

    current = _first_observation_fields(current, observed_at, event_time_col, timely_lag_minutes)
    current["_forward_key"] = _key_frame(current, key_cols)
    current = current.drop_duplicates("_forward_key", keep="last")

    if eligibility_by_signal is not None and "signal_id" in current.columns:
        mapped = current["signal_id"].astype(str).map(eligibility_by_signal)
        current["timely_forward_observation"] = mapped.astype("boolean").fillna(False).astype(bool)

    if path.exists():
        existing = pd.read_parquet(path)
    else:
        existing = pd.DataFrame()
    if existing.empty:
        merged = current
    else:
        existing = existing.copy()
        existing["_forward_key"] = _key_frame(existing, key_cols)
        existing = existing.drop_duplicates("_forward_key", keep="last")
        preserved = existing.set_index("_forward_key")[
            [
                "first_observed_at_utc",
                "first_observation_lag_seconds",
                "timely_forward_observation",
            ]
        ]
        overlap = current["_forward_key"].isin(preserved.index)
        for col in preserved.columns:
            current.loc[overlap, col] = current.loc[overlap, "_forward_key"].map(preserved[col])
        existing_only = existing[~existing["_forward_key"].isin(set(current["_forward_key"]))]
        merged = pd.concat([existing_only, current], ignore_index=True, sort=False)

    merged = merged.drop(columns=["_forward_key"], errors="ignore")
    sort_cols = [col for col in [event_time_col, *key_cols] if col and col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    write_parquet(merged, path)
    return merged


def _append_manifest_row(manifest_path: Path, manifest_row: pd.DataFrame) -> None:
    """Append while migrating legacy manifests whose schema grew between deployments."""
    current_columns = list(manifest_row.columns)
    intermediate_columns = [
        column
        for column in current_columns
        if column
        not in {
            "rolling_risk_shadow_skipped",
            "rolling_token_context",
            "cumulative_risk_shadow_skipped",
            "cumulative_token_context",
        }
    ]
    records: list[dict[str, object]] = []
    extra_columns: list[str] = []
    if manifest_path.exists() and manifest_path.stat().st_size:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if rows:
            legacy_columns = rows[0]
            extra_columns = [col for col in legacy_columns if col not in current_columns]
            for values in rows[1:]:
                if not values:
                    continue
                if len(values) == len(legacy_columns):
                    records.append(dict(zip(legacy_columns, values, strict=True)))
                elif len(values) == len(intermediate_columns):
                    records.append(dict(zip(intermediate_columns, values, strict=True)))
                elif len(values) == len(current_columns):
                    # Recover a row appended by a newer writer beneath a legacy header.
                    records.append(dict(zip(current_columns, values, strict=True)))
                else:
                    raise ValueError(
                        f"unsupported forward manifest row width {len(values)}; "
                        f"expected {len(legacy_columns)}, {len(intermediate_columns)}, "
                        f"or {len(current_columns)}"
                    )
    columns = [*current_columns, *extra_columns]
    combined = pd.concat([pd.DataFrame(records), manifest_row], ignore_index=True, sort=False)
    combined = combined.reindex(columns=columns)
    temp_path = manifest_path.with_suffix(".csv.tmp")
    combined.to_csv(temp_path, index=False)
    temp_path.replace(manifest_path)


def write_forward_run(
    *,
    report_root: Path,
    prepared: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    portfolio_trades: pd.DataFrame,
    overflow_trades: pd.DataFrame,
    checkpoint_trades: pd.DataFrame,
    data_stale: bool,
    risk_shadow_trades: pd.DataFrame | None = None,
    risk_shadow_skipped: pd.DataFrame | None = None,
    token_context: pd.DataFrame | None = None,
    observed_at: object | None = None,
    config: ForwardLedgerConfig = ForwardLedgerConfig(),
) -> dict[str, Path]:
    """Persist one live observation plus cumulative, deduplicated forward ledgers."""
    observed = _utc_timestamp(observed_at)
    risk_shadow_trades = risk_shadow_trades if risk_shadow_trades is not None else pd.DataFrame()
    risk_shadow_skipped = risk_shadow_skipped if risk_shadow_skipped is not None else pd.DataFrame()
    token_context = token_context if token_context is not None else pd.DataFrame()
    root = ensure_dir(Path(report_root) / "forward")
    run_id = observed.strftime("%Y%m%dT%H%M%S%fZ")
    run_root = ensure_dir(root / "runs" / run_id)

    snapshot_paths = {
        "signals_snapshot": run_root / "signals.parquet",
        "trades_snapshot": run_root / "trades.parquet",
        "baseline_trades_snapshot": run_root / "baseline_trades.parquet",
        "portfolio_trades_snapshot": run_root / "portfolio_trades.parquet",
        "overflow_trades_snapshot": run_root / "overflow_trades.parquet",
        "checkpoint_trades_snapshot": run_root / "checkpoint_trades.parquet",
        "risk_shadow_trades_snapshot": run_root / "risk_shadow_trades.parquet",
        "risk_shadow_skipped_snapshot": run_root / "risk_shadow_skipped.parquet",
        "token_context_snapshot": run_root / "token_context.parquet",
    }
    write_parquet(signals, snapshot_paths["signals_snapshot"])
    write_parquet(trades, snapshot_paths["trades_snapshot"])
    write_parquet(baseline_trades, snapshot_paths["baseline_trades_snapshot"])
    write_parquet(portfolio_trades, snapshot_paths["portfolio_trades_snapshot"])
    write_parquet(overflow_trades, snapshot_paths["overflow_trades_snapshot"])
    write_parquet(checkpoint_trades, snapshot_paths["checkpoint_trades_snapshot"])
    write_parquet(risk_shadow_trades, snapshot_paths["risk_shadow_trades_snapshot"])
    write_parquet(risk_shadow_skipped, snapshot_paths["risk_shadow_skipped_snapshot"])
    write_parquet(token_context, snapshot_paths["token_context_snapshot"])

    cumulative_paths = {
        "signals": root / "signals.parquet",
        "trades": root / "trades.parquet",
        "baseline_trades": root / "baseline_trades.parquet",
        "portfolio_trades": root / "portfolio_trades.parquet",
        "overflow_trades": root / "overflow_trades.parquet",
        "checkpoint_trades": root / "checkpoint_trades.parquet",
        "risk_shadow_trades": root / "risk_shadow_trades.parquet",
        "risk_shadow_skipped": root / "risk_shadow_skipped.parquet",
        "token_context": root / "token_context.parquet",
    }
    cumulative_signals = merge_cumulative_ledger(
        signals,
        cumulative_paths["signals"],
        key_cols=["signal_id"],
        observed_at=observed,
        event_time_col="feature_time",
        timely_lag_minutes=config.timely_lag_minutes,
    )
    signal_eligibility = (
        cumulative_signals.assign(signal_id=cumulative_signals.get("signal_id", "").astype(str))
        .drop_duplicates("signal_id", keep="last")
        .set_index("signal_id")["timely_forward_observation"]
        if not cumulative_signals.empty and "signal_id" in cumulative_signals.columns
        else pd.Series(dtype=bool)
    )
    cumulative_trades = merge_cumulative_ledger(
        trades,
        cumulative_paths["trades"],
        key_cols=["trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )
    cumulative_baselines = merge_cumulative_ledger(
        baseline_trades,
        cumulative_paths["baseline_trades"],
        key_cols=["trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
    )
    cumulative_portfolios = merge_cumulative_ledger(
        portfolio_trades,
        cumulative_paths["portfolio_trades"],
        key_cols=["portfolio_id", "trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )
    cumulative_overflow = merge_cumulative_ledger(
        overflow_trades,
        cumulative_paths["overflow_trades"],
        key_cols=["portfolio_id", "trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )
    cumulative_checkpoints = merge_cumulative_ledger(
        checkpoint_trades,
        cumulative_paths["checkpoint_trades"],
        key_cols=["portfolio_id", "trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )
    cumulative_risk_shadows = merge_cumulative_ledger(
        risk_shadow_trades,
        cumulative_paths["risk_shadow_trades"],
        key_cols=["risk_shadow_arm", "trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )
    cumulative_risk_shadow_skipped = merge_cumulative_ledger(
        risk_shadow_skipped,
        cumulative_paths["risk_shadow_skipped"],
        key_cols=["risk_shadow_arm", "trade_id"],
        observed_at=observed,
        event_time_col="local_volume_shock_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )
    cumulative_token_context = merge_cumulative_ledger(
        token_context,
        cumulative_paths["token_context"],
        key_cols=["trade_id"],
        observed_at=observed,
        event_time_col="entry_time",
        timely_lag_minutes=config.timely_lag_minutes,
        eligibility_by_signal=signal_eligibility,
    )

    feature_time = pd.to_datetime(prepared.get("feature_time"), utc=True, errors="coerce")
    manifest_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "observed_at_utc": observed,
                "latest_feature_time": feature_time.max() if not feature_time.empty else pd.NaT,
                "data_stale": bool(data_stale),
                "rolling_signals": int(len(signals)),
                "rolling_trades": int(len(trades)),
                "rolling_baseline_trades": int(len(baseline_trades)),
                "rolling_portfolio_trades": int(len(portfolio_trades)),
                "rolling_overflow_trades": int(len(overflow_trades)),
                "rolling_checkpoint_trades": int(len(checkpoint_trades)),
                "rolling_risk_shadow_trades": int(len(risk_shadow_trades)),
                "rolling_risk_shadow_skipped": int(len(risk_shadow_skipped)),
                "rolling_token_context": int(len(token_context)),
                "cumulative_signals": int(len(cumulative_signals)),
                "cumulative_trades": int(len(cumulative_trades)),
                "cumulative_baseline_trades": int(len(cumulative_baselines)),
                "cumulative_portfolio_trades": int(len(cumulative_portfolios)),
                "cumulative_overflow_trades": int(len(cumulative_overflow)),
                "cumulative_checkpoint_trades": int(len(cumulative_checkpoints)),
                "cumulative_risk_shadow_trades": int(len(cumulative_risk_shadows)),
                "cumulative_risk_shadow_skipped": int(len(cumulative_risk_shadow_skipped)),
                "cumulative_token_context": int(len(cumulative_token_context)),
                "timely_signals": int(
                    cumulative_signals.get("timely_forward_observation", pd.Series(dtype=bool))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
            }
        ]
    )
    manifest_path = root / "run_manifest.csv"
    _append_manifest_row(manifest_path, manifest_row)
    return {**snapshot_paths, **cumulative_paths, "run_manifest": manifest_path}


__all__ = [
    "ForwardLedgerConfig",
    "merge_cumulative_ledger",
    "write_forward_run",
]
