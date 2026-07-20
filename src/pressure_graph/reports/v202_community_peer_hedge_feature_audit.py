"""Feature-only audit for peer-hedging community trade-overshoot buckets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    build_v187_monthly_risk,
)
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    MEMBERSHIP_PATH,
    load_v195_membership,
)
from pressure_graph.reports.v200_reference_price_transmission_feature_audit import (
    REPORT_ROOT as V200_REPORT_ROOT,
)
from pressure_graph.reports.v201_reference_price_transmission import (
    COMMUNITY_TRADE,
    select_v201_feature_events,
)


REPORT_ROOT = Path("reports/v20_2_community_peer_hedge_feature_audit")
FINDINGS_PATH = Path(
    "docs/v202_community_peer_hedge_feature_audit_2026_07_17.md"
)
FEATURE_EVENTS_PATH = V200_REPORT_ROOT / "candidate_feature_events.parquet"
CANDIDATE = "RPH1_COMMUNITY_TRADE_OVERSHOOT_PEER_HEDGE"


@dataclass(frozen=True)
class V202FeatureConfig:
    minimum_peers: int = 2
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000


def build_peer_hedged_weights(
    selected: list[str],
    peers: list[str],
    source_sign: float,
) -> dict[str, float]:
    if not selected or not peers:
        return {}
    selected_direction = -float(source_sign)
    peer_direction = float(source_sign)
    weights = {
        symbol: 0.5 * selected_direction / len(selected) for symbol in selected
    }
    weights.update(
        {symbol: 0.5 * peer_direction / len(peers) for symbol in peers}
    )
    return weights


def build_v202_peer_targets(
    feature_events: pd.DataFrame,
    membership: pd.DataFrame,
    risk: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V202FeatureConfig = V202FeatureConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_events = select_v201_feature_events(feature_events, COMMUNITY_TRADE)
    membership_lookup = {
        (pd.Timestamp(month), str(community)): sorted(
            (set(group["symbol"].astype(str)) & set(close.columns)) - {BTC}
        )
        for (month, community), group in membership.groupby(
            ["month_start", "community_id"], sort=True
        )
    }
    risk_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]
    rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    for item in selected_events.itertuples(index=False):
        timestamp = pd.Timestamp(item.feature_time)
        selected = [name for name in str(item.receivers).split("|") if name]
        community_members = membership_lookup.get(
            (_month(timestamp), str(item.community_id)), []
        )
        peers = [name for name in community_members if name not in set(selected)]
        beta = pd.Series(
            {
                symbol: risk_lookup.get((_month(timestamp), symbol), np.nan)
                for symbol in [*selected, *peers]
            },
            dtype=float,
        )
        exclusion = None
        if len(peers) < cfg.minimum_peers:
            exclusion = "fewer_than_two_unselected_peers"
        elif beta.isna().any():
            exclusion = "missing_prior_month_beta"
        if exclusion:
            excluded_rows.append(
                {
                    "source_event_id": item.source_event_id,
                    "feature_time": timestamp,
                    "period": item.period,
                    "community_id": item.community_id,
                    "selected_count": len(selected),
                    "peer_count": len(peers),
                    "reason": exclusion,
                }
            )
            continue
        weights = build_peer_hedged_weights(
            selected, peers, float(item.source_sign)
        )
        residual_beta = float(
            sum(weights[symbol] * beta[symbol] for symbol in weights)
        )
        rows.append(
            {
                **item._asdict(),
                "candidate": CANDIDATE,
                "selected_symbols": "|".join(selected),
                "peer_symbols": "|".join(peers),
                "selected_count": len(selected),
                "peer_count": len(peers),
                "selected_direction": -float(item.source_sign),
                "peer_direction": float(item.source_sign),
                "net_dollar_exposure": float(sum(weights.values())),
                "gross_notional": float(sum(abs(value) for value in weights.values())),
                "prior_btc_beta_exposure": residual_beta,
                "weights": weights,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(excluded_rows)


def summarize_v202_targets(
    targets: pd.DataFrame,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    counts = targets["period"].value_counts()
    original = len(targets) + len(excluded)
    summary = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "original_feature_events": original,
                "constructible_events": len(targets),
                "excluded_events": len(excluded),
                "constructible_fraction": len(targets) / original if original else 0.0,
                "development_events": int(counts.get("development", 0)),
                "validation_events": int(counts.get("validation", 0)),
                "holdout_events": int(counts.get("holdout", 0)),
                "active_days": targets["entry_day"].nunique(),
                "active_months": targets["entry_month"].nunique(),
                "median_selected_count": float(targets["selected_count"].median()),
                "median_peer_count": float(targets["peer_count"].median()),
                "minimum_peer_count": int(targets["peer_count"].min()),
                "median_abs_prior_btc_beta_exposure": float(
                    targets["prior_btc_beta_exposure"].abs().median()
                ),
                "p90_abs_prior_btc_beta_exposure": float(
                    targets["prior_btc_beta_exposure"].abs().quantile(0.90)
                ),
                "max_abs_net_dollar_exposure": float(
                    targets["net_dollar_exposure"].abs().max()
                ),
                "max_gross_notional_drift": float(
                    targets["gross_notional"].sub(1.0).abs().max()
                ),
            }
        ]
    )
    row = summary.iloc[0]
    summary["feature_viable"] = bool(
        row["constructible_events"] >= 200
        and row["validation_events"] >= 50
        and row["holdout_events"] >= 60
        and row["active_months"] >= 10
        and row["minimum_peer_count"] >= 2
        and row["max_abs_net_dollar_exposure"] <= 1e-12
        and row["max_gross_notional_drift"] <= 1e-12
    )
    return summary


def _write_findings(
    summary: pd.DataFrame,
    excluded: pd.DataFrame,
    path: Path,
) -> None:
    reasons = (
        excluded["reason"].value_counts().rename_axis("reason").reset_index(name="events")
        if not excluded.empty
        else pd.DataFrame(columns=["reason", "events"])
    )
    text = [
        "# v20.2 Community Peer-Hedge Feature-Only Audit",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Excluded-event reasons:",
        "",
        reasons.to_markdown(index=False) if not reasons.empty else "No exclusions.",
        "",
        "The construction reuses only the frozen RPT4 community trade-overshoot "
        "events. Selected over-shooting receivers retain the preregistered fade "
        "direction; every other available member of the same frozen community forms "
        "the opposite peer sleeve. Each sleeve has 0.5 absolute notional, so the "
        "book is exactly dollar neutral without introducing a BTC leg.",
        "",
        "Prior BTC beta is reported as a risk diagnostic, not neutralized. No future "
        "return or candidate PnL was calculated or inspected in this audit. The "
        "branch is explicitly posthoc because v20.1 sleeve attribution motivated "
        "the hedge redesign.",
        "",
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v202_feature_audit(
    feature_events_path: Path = FEATURE_EVENTS_PATH,
    membership_path: Path = MEMBERSHIP_PATH,
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V202FeatureConfig = V202FeatureConfig(),
) -> dict[str, Path]:
    feature_events = pd.read_parquet(feature_events_path)
    feature_events["feature_time"] = pd.to_datetime(
        feature_events["feature_time"], utc=True
    )
    membership = load_v195_membership(membership_path)
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    risk = build_v187_monthly_risk(
        close.pct_change(fill_method=None),
        feature_events["feature_time"].min(),
        feature_events["feature_time"].max(),
        cfg,  # type: ignore[arg-type]
    )
    targets, excluded = build_v202_peer_targets(
        feature_events, membership, risk, close, cfg
    )
    summary = summarize_v202_targets(targets, excluded)
    root = ensure_dir(report_root)
    outputs = {
        "targets": root / "peer_hedged_targets.parquet",
        "excluded": root / "excluded_feature_events.csv",
        "summary": root / "feature_coverage_summary.csv",
        "risk": root / "monthly_btc_risk.parquet",
        "findings": findings_path,
    }
    serial = targets.copy()
    serial["weights"] = serial["weights"].map(
        lambda value: "|".join(
            f"{key}:{value[key]:.12g}" for key in sorted(value)
        )
    )
    serial.to_parquet(outputs["targets"], index=False)
    excluded.to_csv(outputs["excluded"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    risk.to_parquet(outputs["risk"], index=False)
    _write_findings(summary, excluded, findings_path)
    return outputs


__all__ = [
    "CANDIDATE",
    "V202FeatureConfig",
    "build_peer_hedged_weights",
    "build_v202_peer_targets",
    "summarize_v202_targets",
    "write_v202_feature_audit",
]
