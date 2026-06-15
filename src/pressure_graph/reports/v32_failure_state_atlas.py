"""v3.2 Failure State Atlas.

This report converts short/failure motifs into action-label evidence for the
long book. It does not search a short strategy. It labels each failure event by
whether future same-symbol CIC longs should have been allowed or skipped, and
whether any currently open long should have been kept or exited.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig
from pressure_graph.reports.v13e_cp60_beta_protection_stability import _prepare_sample_at_cost
from pressure_graph.reports.v30_symbol_risk_off_overlay import FOCAL_COST_BPS
from pressure_graph.reports.v31_failure_position_management import (
    V31Config,
    _load_or_stream_events,
)


REPORT_ROOT = Path("reports/v3_2_failure_state_atlas")
FUTURE_BARS = 48
BAR = pd.Timedelta(minutes=15)
LABEL_THRESHOLD = 0.001

STATE_COLUMNS = (
    "symbol",
    "feature_time",
    "ret_1h",
    "ret_4h",
    "volume_z_1h",
    "volume_z_4h",
    "funding_percentile",
    "oi_delta_1h_percentile",
    "oi_delta_4h_percentile",
    "ret_4h_percentile",
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_market_state",
    "btc_vol_regime",
)


@dataclass(frozen=True)
class V32Config:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    v31: V31Config = V31Config()
    future_bars: int = FUTURE_BARS
    label_threshold: float = LABEL_THRESHOLD
    position_ledger_path: Path = Path("reports/v3_1_failure_position_management/failure_position_ledger.csv")


def _event_id(events: pd.DataFrame) -> pd.Series:
    key = (
        events["symbol"].astype(str)
        + "|"
        + pd.to_datetime(events["feature_time"], utc=True, errors="coerce").astype(str)
        + "|"
        + events["motif"].astype(str)
    )
    return key + "|" + events.groupby(key, sort=False).cumcount().astype(str)


def _load_event_state_features(feature_path: Path, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    available = pd.read_parquet(feature_path, columns=list(STATE_COLUMNS))
    available["feature_time"] = pd.to_datetime(available["feature_time"], utc=True, errors="coerce")
    keys = events[["symbol", "feature_time"]].drop_duplicates().copy()
    keys["feature_time"] = pd.to_datetime(keys["feature_time"], utc=True, errors="coerce")
    state = keys.merge(available, on=["symbol", "feature_time"], how="left")
    return state


def _state_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ret_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    volume_z = pd.to_numeric(out.get("volume_z_1h"), errors="coerce")
    funding = pd.to_numeric(out.get("funding_percentile"), errors="coerce")
    out["symbol_momentum_bucket"] = np.select(
        [ret_pct.ge(0.8), ret_pct.le(0.2)],
        ["high_momentum", "low_momentum"],
        default="mid_momentum",
    )
    out["volume_state_bucket"] = np.select(
        [volume_z.ge(2.0), volume_z.le(0.0)],
        ["volume_shock", "quiet_volume"],
        default="normal_volume",
    )
    out["funding_state_bucket"] = np.select(
        [funding.ge(0.8), funding.le(0.2)],
        ["high_funding", "low_funding"],
        default="mid_funding",
    )
    out["btc_market_state"] = out.get("btc_market_state", pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)
    out["btc_vol_regime"] = out.get("btc_vol_regime", pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)
    return out


def _future_long_labels(events: pd.DataFrame, sample: pd.DataFrame, cfg: V32Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample = sample.copy()
    sample["signal_time"] = pd.to_datetime(sample["signal_time"], utc=True, errors="coerce")
    sample_net = pd.to_numeric(sample["net_return_at_cost"], errors="coerce")
    sample_by_symbol = {str(sym): group.copy() for sym, group in sample.groupby("symbol", sort=False, dropna=False)}
    threshold = float(cfg.label_threshold)
    window = int(cfg.future_bars) * BAR
    for row in events.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        ts = pd.Timestamp(getattr(row, "feature_time"))
        group = sample_by_symbol.get(symbol, pd.DataFrame())
        if group.empty:
            future = group
        else:
            sig = pd.to_datetime(group["signal_time"], utc=True, errors="coerce")
            future = group[sig.gt(ts) & sig.le(ts + window)].copy()
        values = pd.to_numeric(future.get("net_return_at_cost", pd.Series(dtype=float)), errors="coerce")
        avg = float(values.mean()) if len(values) else np.nan
        total = float(values.sum()) if len(values) else 0.0
        if not len(values):
            label = "no_future_long_sample"
        elif avg < -threshold:
            label = "no_long_48_best"
        elif avg > threshold:
            label = "allow_long_best"
        else:
            label = "future_long_neutral"
        rows.append(
            {
                "failure_event_id": getattr(row, "failure_event_id"),
                "future_long_count_48": int(len(values)),
                "future_long_avg_net20_48": avg,
                "future_long_sum_net20_48": total,
                "future_long_loss_share_48": float(values.lt(0).mean()) if len(values) else np.nan,
                "future_long_label": label,
            }
        )
    return pd.DataFrame(rows)


def _position_labels(events: pd.DataFrame, cfg: V32Config) -> pd.DataFrame:
    rows = []
    if not cfg.position_ledger_path.exists():
        return pd.DataFrame(columns=["failure_event_id", "open_position_count", "position_delta_vs_keep_avg", "position_action_label"])
    ledger = pd.read_csv(cfg.position_ledger_path)
    if ledger.empty:
        return pd.DataFrame(columns=["failure_event_id", "open_position_count", "position_delta_vs_keep_avg", "position_action_label"])
    ledger = ledger[ledger.get("position_management_rule", pd.Series(dtype=str)).astype(str).eq("B3_no_failure_exit")].copy()
    if ledger.empty:
        return pd.DataFrame(columns=["failure_event_id", "open_position_count", "position_delta_vs_keep_avg", "position_action_label"])
    ledger["failure_event_time"] = pd.to_datetime(ledger.get("failure_event_time"), utc=True, errors="coerce")
    ledger = ledger[ledger.get("failure_event_during_position", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)].copy()
    threshold = float(cfg.label_threshold)
    for row in events.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        ts = pd.Timestamp(getattr(row, "feature_time"))
        local = ledger[ledger["symbol"].astype(str).eq(symbol) & ledger["failure_event_time"].eq(ts)]
        values = pd.to_numeric(local.get("failure_event_delta_vs_keep", pd.Series(dtype=float)), errors="coerce")
        avg = float(values.mean()) if len(values) else np.nan
        if not len(values):
            label = "no_open_long"
        elif avg > threshold:
            label = "exit_long_best"
        elif avg < -threshold:
            label = "keep_long_best"
        else:
            label = "position_neutral"
        rows.append(
            {
                "failure_event_id": getattr(row, "failure_event_id"),
                "open_position_count": int(len(values)),
                "position_delta_vs_keep_avg": avg,
                "position_delta_vs_keep_sum": float(values.sum()) if len(values) else 0.0,
                "position_action_label": label,
            }
        )
    return pd.DataFrame(rows)


def _best_action(row: pd.Series) -> str:
    future = str(row.get("future_long_label", ""))
    position = str(row.get("position_action_label", ""))
    if position == "exit_long_best" and future == "no_long_48_best":
        return "exit_existing_and_no_long_48"
    if position == "exit_long_best":
        return "exit_existing_long"
    if position == "keep_long_best" and future == "no_long_48_best":
        return "keep_existing_but_no_new_long_48"
    if future == "no_long_48_best":
        return "no_long_48"
    if future == "allow_long_best":
        return "allow_long"
    return "no_action_or_neutral"


def _event_action_labels(events: pd.DataFrame, state: pd.DataFrame, future: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    out = events.merge(state, on=["symbol", "feature_time"], how="left")
    out = out.merge(future, on="failure_event_id", how="left")
    out = out.merge(position, on="failure_event_id", how="left")
    out["best_action_label"] = out.apply(_best_action, axis=1)
    return _state_buckets(out)


def _action_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for col in ("motif", "future_long_label", "position_action_label", "best_action_label"):
        group = labels.groupby(col, sort=False, dropna=False).agg(
            events=("failure_event_id", "count"),
            future_long_count_48=("future_long_count_48", "sum"),
            future_long_avg_net20_48=("future_long_avg_net20_48", "mean"),
            position_delta_vs_keep_avg=("position_delta_vs_keep_avg", "mean"),
        )
        group = group.reset_index().rename(columns={col: "bucket"})
        group.insert(0, "group_col", col)
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def _state_atlas(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    group_cols = ["motif", "btc_market_state", "symbol_momentum_bucket", "volume_state_bucket"]
    def _conditional_rate(values: pd.Series, target: str, counts: pd.Series) -> float:
        local = pd.DataFrame({"value": values.astype(str), "count": pd.to_numeric(counts, errors="coerce")})
        local = local[local["count"].gt(0)]
        return float(local["value"].eq(target).mean()) if len(local) else np.nan

    atlas = labels.groupby(group_cols, sort=False, dropna=False).agg(
        events=("failure_event_id", "count"),
        future_long_events=("future_long_count_48", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
        future_long_avg_net20_48=("future_long_avg_net20_48", "mean"),
        no_long_48_rate=("future_long_label", lambda x: float(pd.Series(x).astype(str).eq("no_long_48_best").mean())),
        allow_long_rate=("future_long_label", lambda x: float(pd.Series(x).astype(str).eq("allow_long_best").mean())),
        exit_long_rate=("position_action_label", lambda x: float(pd.Series(x).astype(str).eq("exit_long_best").mean())),
        keep_long_rate=("position_action_label", lambda x: float(pd.Series(x).astype(str).eq("keep_long_best").mean())),
    )
    atlas = atlas.reset_index()
    conditional_rows = []
    for key, group in labels.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        conditional_rows.append(
            {
                **dict(zip(group_cols, key, strict=False)),
                "no_long_48_conditional_rate": _conditional_rate(
                    group["future_long_label"], "no_long_48_best", group["future_long_count_48"]
                ),
                "allow_long_conditional_rate": _conditional_rate(
                    group["future_long_label"], "allow_long_best", group["future_long_count_48"]
                ),
            }
        )
    conditional = pd.DataFrame(conditional_rows)
    if not conditional.empty:
        atlas = atlas.merge(conditional, on=group_cols, how="left")
    return atlas.sort_values(["events", "future_long_avg_net20_48"], ascending=[False, True])


def _write_notes(root: Path, labels: pd.DataFrame, atlas: pd.DataFrame) -> None:
    lines = [
        "# v3.2 Failure State Atlas",
        "",
        "Purpose: label failure motifs by best long-book action, rather than treating them as standalone shorts.",
        "Status: offline atlas only. No live / paper-live / real-live changes.",
        "",
        "## Action Distribution",
    ]
    if labels.empty:
        lines.append("- No labels.")
    else:
        for label, count in labels["best_action_label"].value_counts().items():
            lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## State Buckets"])
    if atlas.empty:
        lines.append("- No atlas buckets.")
    else:
        head = atlas.head(8)
        for row in head.itertuples(index=False):
            lines.append(
                f"- motif={row.motif}, btc={row.btc_market_state}, momentum={row.symbol_momentum_bucket}, "
                f"volume={row.volume_state_bucket}: events={row.events}, "
                f"future_avg={row.future_long_avg_net20_48:.4%}, no_long_rate={row.no_long_48_rate:.2%}."
            )
    if not labels.empty:
        future_events = labels[pd.to_numeric(labels["future_long_count_48"], errors="coerce").gt(0)]
        lines.extend(["", "## Label Sparsity"])
        lines.append(
            f"- failure events with any future same-symbol CIC long in 48 bars: "
            f"{len(future_events)} / {len(labels)}."
        )
        if len(future_events):
            lines.append(
                f"- conditional no-long labels: "
                f"{future_events['future_long_label'].astype(str).eq('no_long_48_best').mean():.2%}; "
                f"conditional allow-long labels: "
                f"{future_events['future_long_label'].astype(str).eq('allow_long_best').mean():.2%}."
            )
    lines.extend(
        [
            "",
            "## Discipline",
            "- Labels are realized counterfactuals and must not be used as live features.",
            "- `no_long_48_best` means future same-symbol CIC candidates in the 48-bar window were poor on average.",
            "- Position labels compare failure-event exit vs keeping the existing B3 managed long.",
            "- True short action remains unproven and is intentionally not promoted here.",
        ]
    )
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v32_failure_state_atlas(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V32Config = V32Config(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, FOCAL_COST_BPS)
    events = _load_or_stream_events(feature_path, instruments, config, cfg.v31)
    events = events.copy()
    events["feature_time"] = pd.to_datetime(events["feature_time"], utc=True, errors="coerce")
    events = events.dropna(subset=["symbol", "feature_time", "motif"]).sort_values(["feature_time", "symbol", "motif"]).reset_index(drop=True)
    events["failure_event_id"] = _event_id(events)
    state = _load_event_state_features(feature_path, events)
    future = _future_long_labels(events, sample, cfg)
    position = _position_labels(events, cfg)
    labels = _event_action_labels(events, state, future, position)
    summary = _action_summary(labels)
    atlas = _state_atlas(labels)
    outputs = {
        "failure_event_action_labels": root / "failure_event_action_labels.csv",
        "failure_action_summary": root / "failure_action_summary.csv",
        "failure_state_atlas": root / "failure_state_atlas.csv",
        "candidate_notes": root / "candidate_notes.md",
    }
    labels.to_csv(outputs["failure_event_action_labels"], index=False)
    summary.to_csv(outputs["failure_action_summary"], index=False)
    atlas.to_csv(outputs["failure_state_atlas"], index=False)
    _write_notes(root, labels, atlas)
    return outputs


__all__ = [
    "REPORT_ROOT",
    "V32Config",
    "write_v32_failure_state_atlas",
]
