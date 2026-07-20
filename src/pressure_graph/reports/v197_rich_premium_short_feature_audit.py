"""Feature-only audit for one-sided graph-premium sleeves and BTC hedges."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    REPORT_ROOT as V195_REPORT_ROOT,
    V195FeatureConfig,
    _neutralize_weights,
    _turnover,
)


REPORT_ROOT = Path("reports/v19_7_rich_premium_short_feature_audit")
FINDINGS_PATH = Path("docs/v197_rich_premium_short_feature_audit_2026_07_17.md")
FEATURE_PANEL_PATH = V195_REPORT_ROOT / "daily_symbol_feature_panel.parquet"
GLOBAL_RICH_SHORT = "GLOBAL_RICH_PREMIUM_SHORT"
GLOBAL_CHEAP_LONG = "GLOBAL_CHEAP_PREMIUM_LONG"
COMMUNITY_RICH_SHORT = "COMMUNITY_RICH_PREMIUM_SHORT"
COMMUNITY_CHEAP_LONG = "COMMUNITY_CHEAP_PREMIUM_LONG"
DIRECTION_FAMILY = (
    GLOBAL_RICH_SHORT,
    GLOBAL_CHEAP_LONG,
    COMMUNITY_RICH_SHORT,
    COMMUNITY_CHEAP_LONG,
)


@dataclass(frozen=True)
class V197FeatureConfig(V195FeatureConfig):
    feature_panel_path: Path = FEATURE_PANEL_PATH


def build_v197_target(
    local: pd.DataFrame,
    family: str,
    cfg: V197FeatureConfig = V197FeatureConfig(),
) -> tuple[dict[str, float], list[str]]:
    eligible = local.dropna(subset=["peer_premium_z", "btc_beta"]).copy()
    selected: list[str] = []
    direction = -1.0 if "RICH_PREMIUM_SHORT" in family else 1.0
    if family.startswith("GLOBAL_"):
        if len(eligible) < cfg.minimum_global_cross_section:
            return {}, []
        ranked = eligible.sort_values(["peer_premium_z", "symbol"])
        chosen = (
            ranked.tail(cfg.global_bucket_size)
            if direction < 0
            else ranked.head(cfg.global_bucket_size)
        )
        selected = chosen["symbol"].astype(str).tolist()
    elif family.startswith("COMMUNITY_"):
        for _, group in eligible.groupby("community_id", sort=True):
            ranked = group.sort_values(["peer_premium_z", "symbol"])
            if len(ranked) < cfg.minimum_community_size:
                continue
            selected.append(
                str(ranked.iloc[-1 if direction < 0 else 0]["symbol"])
            )
    else:
        raise ValueError(f"unknown family: {family}")
    if not selected:
        return {}, []
    raw = {symbol: direction / len(selected) for symbol in selected}
    beta = eligible.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
    return _neutralize_weights(raw, beta), selected


def build_v197_weekly_targets(
    feature_panel: pd.DataFrame,
    cfg: V197FeatureConfig = V197FeatureConfig(),
) -> pd.DataFrame:
    panel = feature_panel.copy()
    panel["entry_time"] = pd.to_datetime(panel["entry_time"], utc=True)
    rows = []
    for entry, local in panel.groupby("entry_time", sort=True):
        entry = pd.Timestamp(entry)
        if entry.weekday() != 0 or entry.hour != 0 or entry.minute != 0:
            continue
        funding_sign = -np.sign(local.set_index("symbol")["funding_7d"])
        for family in DIRECTION_FAMILY:
            weights, selected = build_v197_target(local, family, cfg)
            if not weights:
                continue
            direction = -1.0 if "RICH_PREMIUM_SHORT" in family else 1.0
            aligned = [
                direction == funding_sign.get(symbol, 0.0)
                for symbol in selected
                if funding_sign.get(symbol, 0.0) != 0
            ]
            alt_notional = sum(
                abs(weight) for symbol, weight in weights.items() if symbol != BTC
            )
            rows.append(
                {
                    "entry_time": entry,
                    "period": str(local["period"].iloc[0]),
                    "family": family,
                    "selected_symbols": "|".join(sorted(selected)),
                    "selected_count": len(selected),
                    "eligible_symbols": len(local),
                    "eligible_communities": local["community_id"].nunique(),
                    "alt_notional": alt_notional,
                    "btc_hedge_notional": abs(weights.get(BTC, 0.0)),
                    "funding_sign_alignment": float(np.mean(aligned)) if aligned else np.nan,
                    "residual_btc_beta": float(
                        sum(
                            weight
                            * (
                                1.0
                                if symbol == BTC
                                else float(
                                    local.set_index("symbol")["btc_beta"].get(symbol, np.nan)
                                )
                            )
                            for symbol, weight in weights.items()
                        )
                    ),
                    "gross_notional": float(sum(abs(value) for value in weights.values())),
                    "weights": weights,
                }
            )
    return pd.DataFrame(rows)


def summarize_v197_targets(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in DIRECTION_FAMILY:
        sample = targets[targets["family"].eq(family)].sort_values("entry_time")
        previous: dict[str, float] = {}
        previous_selected: set[str] | None = None
        turnovers = []
        jaccards = []
        for row in sample.itertuples(index=False):
            current = dict(row.weights)
            selected = set(str(row.selected_symbols).split("|"))
            turnovers.append(_turnover(previous, current))
            if previous_selected is not None:
                jaccards.append(
                    len(previous_selected & selected)
                    / max(1, len(previous_selected | selected))
                )
            previous = current
            previous_selected = selected
        counts = sample["period"].value_counts()
        rows.append(
            {
                "family": family,
                "weekly_targets": len(sample),
                "active_months": sample["entry_time"].dt.strftime("%Y-%m").nunique(),
                "development_targets": int(counts.get("development", 0)),
                "validation_targets": int(counts.get("validation", 0)),
                "holdout_targets": int(counts.get("holdout", 0)),
                "median_selected_count": float(sample["selected_count"].median()),
                "mean_transition_turnover": float(np.mean(turnovers)),
                "median_transition_turnover": float(np.median(turnovers)),
                "mean_selected_jaccard": float(np.mean(jaccards)),
                "median_alt_notional": float(sample["alt_notional"].median()),
                "median_btc_hedge_notional": float(
                    sample["btc_hedge_notional"].median()
                ),
                "mean_funding_sign_alignment": float(
                    sample["funding_sign_alignment"].mean()
                ),
                "max_abs_residual_btc_beta": float(
                    sample["residual_btc_beta"].abs().max()
                ),
                "max_gross_notional_drift": float(
                    (sample["gross_notional"] - 1.0).abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_findings(summary: pd.DataFrame, path: Path) -> None:
    text = [
        "# v19.7 Rich-Premium Short Feature-Only Audit",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "This is a post-v19.6 attribution follow-up. All four global/community and",
        "rich-short/cheap-long directions are retained as one multiplicity family.",
        "Only as-of selections, BTC hedges, and target turnover were calculated; no",
        "future price or funding return was inspected in this audit.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v197_feature_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V197FeatureConfig = V197FeatureConfig(),
) -> dict[str, Path]:
    feature_panel = pd.read_parquet(cfg.feature_panel_path)
    targets = build_v197_weekly_targets(feature_panel, cfg)
    summary = summarize_v197_targets(targets)
    root = ensure_dir(report_root)
    outputs = {
        "targets": root / "weekly_target_weights.parquet",
        "summary": root / "target_coverage_summary.csv",
        "findings": findings_path,
    }
    serial = targets.copy()
    serial["weights"] = serial["weights"].map(
        lambda value: "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))
    )
    serial.to_parquet(outputs["targets"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    _write_findings(summary, findings_path)
    return outputs


__all__ = [
    "COMMUNITY_CHEAP_LONG",
    "COMMUNITY_RICH_SHORT",
    "DIRECTION_FAMILY",
    "GLOBAL_CHEAP_LONG",
    "GLOBAL_RICH_SHORT",
    "V197FeatureConfig",
    "build_v197_target",
    "build_v197_weekly_targets",
    "summarize_v197_targets",
    "write_v197_feature_audit",
]
