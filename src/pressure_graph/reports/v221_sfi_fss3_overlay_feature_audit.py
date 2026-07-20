"""Feature-only audit for an SFI rank tilt inside weekly FSS3 targets."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v218_spot_perp_flow_inventory_feature_audit import (
    REPORT_ROOT as V218_REPORT_ROOT,
)


PANEL_PATH = Path("reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet")
SFI_PATH = V218_REPORT_ROOT / "decision_symbol_features.parquet"
REPORT_ROOT = Path("reports/v22_1_sfi_fss3_overlay_feature_audit")
FINDINGS_PATH = Path("docs/v221_sfi_fss3_overlay_feature_audit_2026_07_17.md")
CANDIDATE = "SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT"


@dataclass(frozen=True)
class V221Config:
    minimum_source_lag_hours: int = 12
    maximum_source_lag_hours: int = 36
    rank_tilt: float = 0.50
    minimum_sfi_coverage: int = 30
    minimum_side_breadth: int = 4
    # Frozen from feature availability before any return/PnL reveal: the causal
    # 12-36h source window supplies 35 weeks, with the smallest period at 8 weeks.
    minimum_active_weeks: int = 35
    minimum_period_weeks: int = 8
    minimum_active_months: int = 9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_v221_overlay_features(
    panel: pd.DataFrame,
    sfi: pd.DataFrame,
    cfg: V221Config = V221Config(),
) -> pd.DataFrame:
    sfi_groups = {
        pd.Timestamp(time): local[local["feature_eligible"]]
        .set_index("symbol")["spot_minus_perp_flow_z"]
        .astype(float)
        for time, local in sfi.groupby("feature_time", sort=False)
    }
    rows: list[dict[str, object]] = []
    for entry, local in panel.groupby("entry_time", sort=True):
        entry = pd.Timestamp(entry)
        eligible_times = [
            time
            for time, values in sfi_groups.items()
            if entry - pd.Timedelta(hours=cfg.maximum_source_lag_hours)
            <= time
            <= entry - pd.Timedelta(hours=cfg.minimum_source_lag_hours)
            and len(values) >= cfg.minimum_sfi_coverage
        ]
        if not eligible_times:
            continue
        source_time = max(eligible_times)
        scores = sfi_groups[source_time]
        usable = local.dropna(subset=["score_7d", "btc_beta"]).copy()
        longs = usable[usable["score_7d"].lt(0)].copy()
        shorts = usable[usable["score_7d"].gt(0)].copy()
        if len(longs) < cfg.minimum_side_breadth or len(shorts) < cfg.minimum_side_breadth:
            continue
        for side, group in (("long", longs), ("short", shorts)):
            group = group.copy()
            group["sfi_score"] = group["symbol"].map(scores)
            benefit = group["sfi_score"] if side == "long" else -group["sfi_score"]
            percentile = benefit.rank(method="average", pct=True)
            group["benefit_percentile"] = percentile
            group["tilt_multiplier"] = 1.0
            available = group["sfi_score"].notna()
            group.loc[available, "tilt_multiplier"] = 1.0 + cfg.rank_tilt * (
                group.loc[available, "benefit_percentile"] - 0.5
            )
            base_abs = 0.5 / len(group)
            signed_base = base_abs if side == "long" else -base_abs
            raw_abs = base_abs * group["tilt_multiplier"]
            normalized_abs = 0.5 * raw_abs / raw_abs.sum()
            signed_overlay = normalized_abs if side == "long" else -normalized_abs
            for position, item in enumerate(group.itertuples(index=False)):
                rows.append(
                    {
                        "candidate": CANDIDATE,
                        "entry_time": entry,
                        "sfi_feature_time": source_time,
                        "month_start": item.month_start,
                        "period": item.period,
                        "entry_month": entry.strftime("%Y-%m"),
                        "symbol": item.symbol,
                        "side": side,
                        "funding_score_7d": float(item.score_7d),
                        "btc_beta": float(item.btc_beta),
                        "sfi_score": float(item.sfi_score)
                        if pd.notna(item.sfi_score)
                        else math.nan,
                        "sfi_available": bool(pd.notna(item.sfi_score)),
                        "benefit_percentile": (
                            float(item.benefit_percentile)
                            if pd.notna(item.benefit_percentile)
                            else math.nan
                        ),
                        "tilt_multiplier": float(item.tilt_multiplier),
                        "base_raw_weight": signed_base,
                        "overlay_raw_weight": float(signed_overlay.iloc[position]),
                        "sfi_cross_section_coverage": len(scores),
                        "information_lag_hours": (entry - source_time).total_seconds() / 3600.0,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["entry_time", "side", "symbol"]).reset_index(drop=True)


def summarize_v221_coverage(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features[features["period"].eq(scope)]
        local_weeks = local.drop_duplicates("entry_time")
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "weeks": local_weeks["entry_time"].nunique(),
                "months": local_weeks["entry_month"].nunique(),
                "mean_symbols": (
                    float(local.groupby("entry_time").size().mean()) if len(local) else math.nan
                ),
                "mean_sfi_coverage": (
                    float(local_weeks["sfi_cross_section_coverage"].mean())
                    if len(local_weeks)
                    else math.nan
                ),
                "mean_sfi_symbol_availability": (
                    float(local["sfi_available"].mean()) if len(local) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v221_features(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    cfg: V221Config = V221Config(),
) -> pd.DataFrame:
    grouped = features.groupby(["entry_time", "side"])["overlay_raw_weight"].sum()
    all_row = coverage[coverage["scope"].eq("all")].iloc[0]
    period_rows = coverage[coverage["scope"].ne("all")]
    columns = " ".join(features.columns).lower()
    checks = {
        "feature_symbol_keys_unique": not features.duplicated(["entry_time", "symbol"]).any(),
        "sfi_precedes_entry_by_frozen_12_to_36h": bool(
            features["information_lag_hours"]
            .between(cfg.minimum_source_lag_hours, cfg.maximum_source_lag_hours)
            .all()
        ),
        "weekly_schedule_unchanged_monday_00": bool(
            features["entry_time"].dt.dayofweek.eq(0).all()
            and features["entry_time"].dt.hour.eq(0).all()
        ),
        "sfi_coverage_floor": bool(
            features["sfi_cross_section_coverage"].ge(cfg.minimum_sfi_coverage).all()
        ),
        "funding_sign_direction_preserved": bool(
            features.loc[features["side"].eq("long"), "funding_score_7d"].lt(0).all()
            and features.loc[features["side"].eq("short"), "funding_score_7d"].gt(0).all()
        ),
        "overlay_weight_sign_preserved": bool(
            features.loc[features["side"].eq("long"), "overlay_raw_weight"].gt(0).all()
            and features.loc[features["side"].eq("short"), "overlay_raw_weight"].lt(0).all()
        ),
        "long_short_raw_notional_half_each": bool(
            grouped.xs("long", level="side").sub(0.5).abs().lt(1e-12).all()
            and grouped.xs("short", level="side").add(0.5).abs().lt(1e-12).all()
        ),
        "tilt_multiplier_frozen_range": bool(features["tilt_multiplier"].between(0.75, 1.25).all()),
        "missing_sfi_names_remain_neutral": bool(
            features.loc[~features["sfi_available"], "tilt_multiplier"].eq(1.0).all()
        ),
        "minimum_active_weeks": int(all_row["weeks"]) >= cfg.minimum_active_weeks,
        "minimum_each_period_weeks": bool(period_rows["weeks"].ge(cfg.minimum_period_weeks).all()),
        "minimum_active_months": int(all_row["months"]) >= cfg.minimum_active_months,
        "no_future_outcome_columns": not any(
            token in columns for token in ("future", "return", "gross", "net", "pnl", "exit")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def _write_findings(
    checks: pd.DataFrame, coverage: pd.DataFrame, hashes: pd.DataFrame, path: Path
) -> None:
    verdict = (
        "feature_viable_freeze_sfi_fss3_overlay"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    text = [
        "# v22.1 SFI-on-FSS3 Overlay Feature Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The overlay uses the latest 00/12 UTC SFI snapshot with at least 30 eligible "
        "symbols between 12 and 36 hours before the unchanged Monday 00:00 FSS3 "
        "rebalance. Within each original funding-sign "
        "side, a frozen 0.50 rank tilt produces multipliers in [0.75, 1.25]; missing "
        "SFI names retain multiplier 1.0. Each side is renormalized to 0.5 raw "
        "notional, so no sign, name, or decision-time filter is introduced.",
        "",
        "No future price, funding, return, PnL, or turnover outcome was calculated.",
        "",
        hashes.to_markdown(index=False),
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v221_feature_audit(
    panel_path: Path = PANEL_PATH,
    sfi_path: Path = SFI_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V221Config = V221Config(),
) -> dict[str, Path]:
    panel = pd.read_parquet(
        panel_path,
        columns=["entry_time", "month_start", "period", "symbol", "score_7d", "btc_beta"],
    )
    sfi = pd.read_parquet(
        sfi_path,
        columns=["feature_time", "symbol", "spot_minus_perp_flow_z", "feature_eligible"],
    )
    for frame, column in ((panel, "entry_time"), (panel, "month_start"), (sfi, "feature_time")):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    features = build_v221_overlay_features(panel, sfi, cfg)
    coverage = summarize_v221_coverage(features)
    checks = audit_v221_features(features, coverage, cfg)
    hashes = pd.DataFrame(
        [
            {"input": str(panel_path), "sha256": _sha256(panel_path)},
            {"input": str(sfi_path), "sha256": _sha256(sfi_path)},
        ]
    )
    root = ensure_dir(report_root)
    outputs = {
        "features": root / "weekly_symbol_overlay_features.parquet",
        "coverage": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": findings_path,
    }
    features.to_parquet(outputs["features"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    checks.to_csv(outputs["checks"], index=False)
    hashes.to_csv(outputs["hashes"], index=False)
    _write_findings(checks, coverage, hashes, findings_path)
    return outputs


__all__ = [
    "CANDIDATE",
    "V221Config",
    "audit_v221_features",
    "build_v221_overlay_features",
    "write_v221_feature_audit",
]
