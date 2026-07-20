"""Feature-only audit for DEX-conditioned community laggard/leader spreads."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    KLINE_ROOT,
    load_v178_market_data,
)
from pressure_graph.reports.v212_dex_community_propagation_feature_audit import (
    ALL_PEERS,
    REPORT_ROOT as V212_REPORT_ROOT,
)


REPORT_ROOT = Path("reports/v21_5_dex_community_relative_spread_feature_audit")
FINDINGS_PATH = Path("docs/v215_dex_community_relative_spread_feature_audit_2026_07_17.md")
EVENT_FEATURES_PATH = V212_REPORT_ROOT / "dex_community_event_features.parquet"
RELATIVE_SPREAD = "DAP3_DEX_COMMUNITY_RELATIVE_CATCHUP_SPREAD"


@dataclass(frozen=True)
class V215FeatureConfig:
    minimum_leg_size: int = 2
    minimum_events: int = 200
    minimum_period_events: int = 30
    minimum_active_months: int = 8
    minimum_source_symbols: int = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _symbols(value: object) -> list[str]:
    return [symbol for symbol in str(value).split("|") if symbol]


def build_v215_spread_features(
    event_features: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V215FeatureConfig = V215FeatureConfig(),
) -> pd.DataFrame:
    return_1h = close.pct_change(4, fill_method=None)
    rows: list[dict[str, object]] = []
    eligible = event_features[event_features[ALL_PEERS].eq(True)]
    for item in eligible.itertuples(index=False):
        feature_time = pd.Timestamp(item.feature_time)
        peers = [
            symbol
            for symbol in _symbols(item.community_peer_symbols)
            if symbol in return_1h.columns
        ]
        if feature_time not in return_1h.index:
            continue
        peer_returns = return_1h.loc[feature_time, peers].dropna()
        leg_size = len(peer_returns) // 2
        if leg_size < cfg.minimum_leg_size:
            continue
        direction = float(item.source_direction)
        ranked = (direction * peer_returns).sort_values(kind="stable")
        laggards = ranked.head(leg_size).index.astype(str).tolist()
        leaders = ranked.tail(leg_size).index.astype(str).tolist()
        if set(laggards) & set(leaders):
            continue
        rows.append(
            {
                "candidate": RELATIVE_SPREAD,
                "event_id": item.event_id,
                "source_symbol": item.source_symbol,
                "source_vendor": item.source_vendor,
                "chain": item.chain,
                "event_time": item.event_time,
                "event_available_time": item.event_available_time,
                "feature_time": feature_time,
                "entry_time": item.entry_time,
                "period": item.period,
                "entry_month": item.entry_month,
                "community_id": item.community_id,
                "source_return_z": item.source_return_z,
                "source_direction": int(direction),
                "community_peer_count": len(peer_returns),
                "leg_size": leg_size,
                "laggard_symbols": "|".join(sorted(laggards)),
                "leader_symbols": "|".join(sorted(leaders)),
                "median_laggard_signed_return_1h": float(ranked.reindex(laggards).median()),
                "median_leader_signed_return_1h": float(ranked.reindex(leaders).median()),
                "ranking_window_end": feature_time,
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["feature_time", "community_id", "source_symbol", "event_id"])
        .reset_index(drop=True)
    )


def summarize_v215_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in (
        "all",
        "development",
        "source_transition",
        "validation",
        "holdout",
    ):
        sample = candidates if scope == "all" else candidates[candidates["period"].eq(scope)]
        rows.append(
            {
                "candidate": RELATIVE_SPREAD,
                "scope": scope,
                "events": len(sample),
                "active_days": (sample["entry_time"].dt.floor("D").nunique() if len(sample) else 0),
                "active_months": sample["entry_month"].nunique() if len(sample) else 0,
                "source_symbols": sample["source_symbol"].nunique() if len(sample) else 0,
                "communities": sample["community_id"].nunique() if len(sample) else 0,
                "mean_leg_size": (float(sample["leg_size"].mean()) if len(sample) else math.nan),
                "median_feature_rank_gap_bp": (
                    float(
                        (
                            sample["median_leader_signed_return_1h"]
                            - sample["median_laggard_signed_return_1h"]
                        ).median()
                        * 10_000
                    )
                    if len(sample)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v215_features(
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    cfg: V215FeatureConfig = V215FeatureConfig(),
) -> pd.DataFrame:
    all_row = coverage[coverage["scope"].eq("all")].iloc[0]
    periods = coverage[coverage["scope"].isin(["development", "validation", "holdout"])]
    columns = " ".join(candidates.columns).lower()
    checks = {
        "candidate_event_keys_unique": not candidates["event_id"].duplicated().any(),
        "feature_precedes_entry_by_full_bar": bool(
            candidates["entry_time"]
            .sub(candidates["feature_time"])
            .eq(pd.Timedelta(minutes=15))
            .all()
        ),
        "ranking_window_ends_at_feature": bool(
            candidates["ranking_window_end"].eq(candidates["feature_time"]).all()
        ),
        "laggard_and_leader_legs_disjoint": bool(
            all(
                not set(_symbols(row.laggard_symbols)) & set(_symbols(row.leader_symbols))
                for row in candidates.itertuples(index=False)
            )
        ),
        "legs_have_equal_size": bool(
            all(
                len(_symbols(row.laggard_symbols))
                == len(_symbols(row.leader_symbols))
                == int(row.leg_size)
                for row in candidates.itertuples(index=False)
            )
        ),
        "minimum_two_names_per_leg": bool(candidates["leg_size"].ge(cfg.minimum_leg_size).all()),
        "laggard_rank_below_leader_rank": bool(
            candidates["median_laggard_signed_return_1h"]
            .le(candidates["median_leader_signed_return_1h"])
            .all()
        ),
        "minimum_total_events": int(all_row["events"]) >= cfg.minimum_events,
        "minimum_each_eligible_period_events": bool(
            periods["events"].ge(cfg.minimum_period_events).all()
        ),
        "minimum_active_months": int(all_row["active_months"]) >= cfg.minimum_active_months,
        "minimum_source_diversification": int(all_row["source_symbols"])
        >= cfg.minimum_source_symbols,
        "vendor_transition_visible": bool(
            coverage.loc[coverage["scope"].eq("source_transition"), "events"].iloc[0] > 0
        ),
        "no_post_event_outcome_columns": not any(
            token in columns for token in ("future", "gross", "net_return", "pnl", "exit_price")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def _write_findings(
    checks: pd.DataFrame,
    coverage: pd.DataFrame,
    source_hash: str,
    path: Path,
) -> None:
    verdict = (
        "feature_viable_freeze_relative_spread_diagnostic"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    text = [
        "# v21.5 DEX Community Relative-Spread Feature Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "At each already frozen v21.2 source event, peers are ranked using only "
        "their one-hour returns observed by the feature close. The slowest and "
        "fastest equal-sized halves form disjoint laggard and leader legs; an odd "
        "middle name is discarded. Each leg requires at least two names.",
        "",
        "This candidate is explicitly second-stage and was motivated by the v21.3 "
        "reveal. Same-history results can assess economic magnitude but cannot "
        "provide independent promotion evidence.",
        "",
        f"Frozen v21.2 event-feature SHA256: `{source_hash}`.",
        "",
        "No post-event return was calculated or inspected in this feature audit. "
        "No live, PaperLive, application, leverage, remote, or order state changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v215_feature_audit(
    event_features_path: Path = EVENT_FEATURES_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V215FeatureConfig = V215FeatureConfig(),
) -> dict[str, Path]:
    event_features = pd.read_parquet(event_features_path)
    for column in ("event_time", "event_available_time", "feature_time", "entry_time"):
        event_features[column] = pd.to_datetime(event_features[column], utc=True)
    close, _ = load_v178_market_data(kline_root)
    candidates = build_v215_spread_features(event_features, close, cfg)
    coverage = summarize_v215_coverage(candidates)
    checks = audit_v215_features(candidates, coverage, cfg)
    source_hash = _sha256(event_features_path)
    root = ensure_dir(report_root)
    outputs = {
        "candidates": root / "relative_spread_feature_events.parquet",
        "coverage": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "source_hash": root / "source_feature_hash.txt",
        "findings": findings_path,
    }
    candidates.to_parquet(outputs["candidates"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    checks.to_csv(outputs["checks"], index=False)
    outputs["source_hash"].write_text(source_hash + "\n", encoding="utf-8")
    _write_findings(checks, coverage, source_hash, findings_path)
    return outputs


__all__ = [
    "RELATIVE_SPREAD",
    "V215FeatureConfig",
    "audit_v215_features",
    "build_v215_spread_features",
    "summarize_v215_coverage",
    "write_v215_feature_audit",
]
