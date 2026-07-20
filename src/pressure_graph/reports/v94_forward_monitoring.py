from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


P2_CANDIDATES = ("CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1")


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].astype("boolean").fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _timely(frame: pd.DataFrame) -> pd.Series:
    return _bool(frame, "timely_forward_observation")


def _event_key(frame: pd.DataFrame) -> pd.Series:
    event_time = frame.get("local_volume_shock_time", frame.get("feature_time", pd.Series("", index=frame.index)))
    return (
        frame.get("exchange", pd.Series("", index=frame.index)).fillna("").astype(str)
        + "|"
        + frame.get("symbol", pd.Series("", index=frame.index)).fillna("").astype(str)
        + "|"
        + pd.to_datetime(event_time, utc=True, errors="coerce").astype(str)
    )


def _time_present(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return pd.to_datetime(frame[column], utc=True, errors="coerce").notna()


def build_forward_funnel(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = signals[
        _timely(signals)
        & signals.get("candidate", pd.Series("", index=signals.index)).astype(str).isin(P2_CANDIDATES)
    ].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    skip = data.get("skip_reason", pd.Series("", index=data.index)).fillna("").astype(str)
    data["_event_key"] = _event_key(data)
    data["observed"] = True
    data["market_gate_passed"] = ~skip.str.contains("market_gate_off", regex=False)
    data["pullback_seen"] = _time_present(data, "pullback_time")
    data["entry_created"] = _time_present(data, "entry_time")
    data["portfolio_accepted"] = _bool(data, "portfolio_accepted")
    data["trade_completed"] = _time_present(data, "exit_time")
    stages = [
        "observed",
        "market_gate_passed",
        "pullback_seen",
        "entry_created",
        "portfolio_accepted",
        "trade_completed",
    ]
    opportunities = data.groupby("_event_key", sort=False)[stages].max()
    rows: list[dict[str, object]] = []
    previous = np.nan
    for stage in stages:
        count = int(opportunities[stage].sum())
        rows.append(
            {
                "stage": stage,
                "signal_rows": int(data[stage].sum()),
                "unique_opportunities": count,
                "conversion_from_previous": count / previous if np.isfinite(previous) and previous > 0 else np.nan,
            }
        )
        previous = float(count)
    skip_summary = (
        data.assign(skip_reason=skip.replace("", "not_skipped"))
        .groupby(["candidate", "skip_reason"], as_index=False, dropna=False)
        .agg(signal_rows=("signal_id", "nunique"), unique_opportunities=("_event_key", "nunique"))
        .sort_values(["unique_opportunities", "candidate"], ascending=[False, True])
    )
    return pd.DataFrame(rows), skip_summary


def _completed_timely(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "exit_time" in frame.columns:
        completed = _time_present(frame, "exit_time")
    elif "net20_later" in frame.columns:
        completed = pd.to_numeric(frame["net20_later"], errors="coerce").notna()
    else:
        completed = pd.Series(True, index=frame.index, dtype=bool)
    return frame[_timely(frame) & completed].copy()


def _progress_row(metric: str, observed: int, target: int, evidence: str) -> dict[str, object]:
    return {
        "metric": metric,
        "observed": int(observed),
        "target": int(target),
        "remaining": max(0, int(target) - int(observed)),
        "progress_ratio": min(1.0, int(observed) / int(target)) if target > 0 else np.nan,
        "ready": int(observed) >= int(target),
        "evidence": evidence,
    }


def build_sample_progress(
    *,
    checkpoint_trades: pd.DataFrame,
    overflow_trades: pd.DataFrame,
    risk_shadow_trades: pd.DataFrame,
    risk_shadow_skipped: pd.DataFrame,
    token_context: pd.DataFrame,
    primary_portfolio_id: str,
) -> pd.DataFrame:
    checkpoint = _completed_timely(checkpoint_trades)
    primary = checkpoint[
        checkpoint.get("portfolio_id", pd.Series("", index=checkpoint.index)).astype(str).eq(primary_portfolio_id)
    ]
    overflow = _completed_timely(overflow_trades)
    cp60 = checkpoint[
        checkpoint.get("portfolio_id", pd.Series("", index=checkpoint.index)).astype(str).isin(
            ["P2_MAX8_CP60", "P2_MAX8_CP60_PLUS_O6"]
        )
        & checkpoint.get("exit_reason", pd.Series("", index=checkpoint.index)).astype(str).str.contains(
            "checkpoint|cp60", case=False, regex=True
        )
    ]
    protected = checkpoint[
        checkpoint.get("portfolio_id", pd.Series("", index=checkpoint.index)).astype(str).isin(
            ["P2_MAX8_CP60_PROTECT_A_CAP2", "P2_MAX8_CP60_PROTECT_A_CAP2_PLUS_O6"]
        )
        & _bool(checkpoint, "protection_applied")
    ]
    risk = _completed_timely(risk_shadow_trades)
    risk_skipped = risk_shadow_skipped[_timely(risk_shadow_skipped)].copy() if not risk_shadow_skipped.empty else risk_shadow_skipped.copy()
    vol = risk[risk.get("risk_shadow_arm", pd.Series("", index=risk.index)).astype(str).eq("P2_VOL")]
    vol_decisions = int(_num(vol, "position_size").sub(1.0).abs().gt(1e-12).sum())
    if not risk_skipped.empty:
        vol_decisions += int(
            risk_skipped.get("risk_shadow_arm", pd.Series("", index=risk_skipped.index)).astype(str).eq("P2_VOL").sum()
        )
    corr_decisions = (
        int(
            (
                risk_skipped.get("risk_shadow_arm", pd.Series("", index=risk_skipped.index))
                .astype(str)
                .eq("P2_CORR")
                & risk_skipped.get("skip_reason", pd.Series("", index=risk_skipped.index))
                .astype(str)
                .eq("correlation_cluster_cap")
            ).sum()
        )
        if not risk_skipped.empty
        else 0
    )
    token = _completed_timely(token_context)
    token = token[token.get("candidate", pd.Series("", index=token.index)).astype(str).isin(P2_CANDIDATES)]
    token_prior = int(_bool(token, "token_prior_24h").sum())
    rows = [
        _progress_row("P2 initial review", len(primary), 100, "timely completed P2_MAX8_BASELINE trades"),
        _progress_row("P2 stronger review", len(primary), 200, "timely completed P2_MAX8_BASELINE trades"),
        _progress_row("CP60 initial review", len(cp60), 50, "timely checkpoint exits"),
        _progress_row("O6 initial review", len(overflow), 30, "timely completed overflow trades"),
        _progress_row("Protect_A initial review", len(protected), 30, "timely protected checkpoint exits"),
        _progress_row("P2_VOL completed trades", len(vol), 100, "timely completed P2_VOL shadow trades"),
        _progress_row("P2_VOL constrained decisions", vol_decisions, 30, "non-unit sizing or capacity skips"),
        _progress_row("P2_CORR constrained decisions", corr_decisions, 30, "correlation_cluster_cap skips"),
        _progress_row("Token P2 trades", len(token), 100, "timely P2 token-context trades"),
        _progress_row("Token prior-24h trades", token_prior, 30, "timely P2 trades with token prior-24h"),
    ]
    return pd.DataFrame(rows)


def write_forward_monitoring(
    report_root: Path,
    *,
    primary_portfolio_id: str,
    observed_at: object | None = None,
) -> dict[str, Path]:
    root = ensure_dir(Path(report_root) / "forward")
    signals = _read_parquet(root / "signals.parquet")
    funnel, skips = build_forward_funnel(signals)
    progress = build_sample_progress(
        checkpoint_trades=_read_parquet(root / "checkpoint_trades.parquet"),
        overflow_trades=_read_parquet(root / "overflow_trades.parquet"),
        risk_shadow_trades=_read_parquet(root / "risk_shadow_trades.parquet"),
        risk_shadow_skipped=_read_parquet(root / "risk_shadow_skipped.parquet"),
        token_context=_read_parquet(root / "token_context.parquet"),
        primary_portfolio_id=primary_portfolio_id,
    )
    outputs = {
        "funnel": root / "forward_funnel.csv",
        "skip_reasons": root / "forward_skip_reasons.csv",
        "sample_progress": root / "sample_progress.csv",
        "status": root / "forward_monitoring_status.md",
    }
    funnel.to_csv(outputs["funnel"], index=False)
    skips.to_csv(outputs["skip_reasons"], index=False)
    progress.to_csv(outputs["sample_progress"], index=False)
    stamp = pd.Timestamp.now(tz="UTC") if observed_at is None else pd.Timestamp(observed_at)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    lines = ["# Forward Monitoring", "", f"- observed_at_utc: {stamp.isoformat()}", "", "## Funnel"]
    if funnel.empty:
        lines.append("- no timely P2 opportunities yet")
    else:
        for row in funnel.to_dict("records"):
            lines.append(f"- {row['stage']}: {int(row['unique_opportunities'])} unique opportunities")
    lines.extend(["", "## Pre-Registered Sample Gates"])
    for row in progress.to_dict("records"):
        lines.append(
            f"- {row['metric']}: {int(row['observed'])}/{int(row['target'])}, "
            f"remaining={int(row['remaining'])}, ready={str(bool(row['ready'])).lower()}"
        )
    outputs["status"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


__all__ = ["build_forward_funnel", "build_sample_progress", "write_forward_monitoring"]
