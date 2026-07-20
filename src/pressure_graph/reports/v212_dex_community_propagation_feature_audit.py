"""Feature-only audit for DEX attention propagated into graph-community peers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import (
    KLINE_ROOT,
    load_v178_market_data,
)
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    MEMBERSHIP_PATH,
    load_v195_membership,
)
from pressure_graph.reports.v95_token_cex_attention_divergence import (
    EVENT_PATH,
    _deoverlap_events,
)


REPORT_ROOT = Path("reports/v21_2_dex_community_propagation_feature_audit")
FINDINGS_PATH = Path("docs/v212_dex_community_propagation_feature_audit_2026_07_17.md")
ALL_PEERS = "DAP1_CONFIRMED_DEX_COMMUNITY_PROPAGATION"
LAGGARD_PEERS = "DAP2_CONFIRMED_DEX_COMMUNITY_LAGGARDS"
CANDIDATES = (ALL_PEERS, LAGGARD_PEERS)


@dataclass(frozen=True)
class V212FeatureConfig:
    event_sources: tuple[str, ...] = (
        "dexpaprika_pool_ohlcv_1h",
        "geckoterminal_pool_ohlcv",
    )
    mapping_confidence: tuple[str, ...] = ("A", "B")
    first_feature_time: pd.Timestamp = pd.Timestamp("2025-08-01", tz="UTC")
    source_cooldown_hours: int = 24
    community_cooldown_hours: int = 4
    source_lookback_bars: int = 30 * 96
    source_minimum_bars: int = 20 * 96
    source_z_threshold: float = 1.0
    maximum_feature_delay_minutes: int = 15
    entry_delay_bars: int = 1
    minimum_all_peers: int = 4
    minimum_laggard_peers: int = 3
    minimum_candidate_events: int = 100
    minimum_period_events: int = 15
    minimum_source_symbols: int = 10
    minimum_active_months: int = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v212_events(
    path: Path = EVENT_PATH,
    cfg: V212FeatureConfig = V212FeatureConfig(),
) -> pd.DataFrame:
    events = pd.read_csv(path, low_memory=False)
    required = {
        "event_id",
        "cex_symbol",
        "event_time",
        "event_available_time",
        "source",
        "mapping_confidence",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"DEX event file missing columns: {sorted(missing)}")
    events["cex_symbol"] = events["cex_symbol"].fillna("").astype(str).str.upper()
    events["source"] = events["source"].fillna("").astype(str)
    events["mapping_confidence"] = events["mapping_confidence"].fillna("").astype(str)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
    events["event_available_time"] = pd.to_datetime(
        events["event_available_time"], utc=True, errors="coerce"
    )
    for column in ("zscore", "percentile", "raw_value"):
        events[column] = pd.to_numeric(events.get(column), errors="coerce")
    events = events[
        events["source"].isin(cfg.event_sources)
        & events["mapping_confidence"].isin(cfg.mapping_confidence)
    ].copy()
    events = (
        events.dropna(subset=["event_id", "cex_symbol", "event_time", "event_available_time"])
        .drop_duplicates("event_id", keep="last")
        .sort_values(["cex_symbol", "event_available_time", "event_id"])
        .reset_index(drop=True)
    )
    return _deoverlap_events(events, cfg.source_cooldown_hours)


def build_v212_source_context(
    close: pd.DataFrame,
    cfg: V212FeatureConfig = V212FeatureConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return_1h = close.pct_change(4, fill_method=None)
    prior = return_1h.shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_minimum_bars
    )
    prior_mean = prior.mean()
    prior_scale = prior.std(ddof=1)
    source_z = (return_1h - prior_mean).div(prior_scale.where(prior_scale.gt(0)))
    return return_1h, prior_mean, prior_scale, source_z


def _month(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(timestamp).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _v212_period(timestamp: pd.Timestamp) -> str:
    """Coverage-frozen splits around the DEX vendor transition."""
    if timestamp < pd.Timestamp("2025-12-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2026-03-01", tz="UTC"):
        return "source_transition"
    if timestamp < pd.Timestamp("2026-05-01", tz="UTC"):
        return "validation"
    return "holdout"


def _join_symbols(symbols: list[str]) -> str:
    return "|".join(sorted(set(symbols)))


def _community_cooldown_mask(events: pd.DataFrame, cooldown_hours: int) -> pd.Series:
    accepted = pd.Series(False, index=events.index, dtype=bool)
    cooldown = pd.Timedelta(hours=cooldown_hours)
    eligible = events[events["source_confirmed"].eq(True) & events["community_id"].notna()].copy()
    eligible["absolute_source_z"] = eligible["source_return_z"].abs()
    eligible = eligible.sort_values(
        ["community_id", "feature_time", "absolute_source_z", "event_id"],
        ascending=[True, True, False, True],
    )
    for _, local in eligible.groupby("community_id", sort=True):
        last: pd.Timestamp | None = None
        for index, row in local.iterrows():
            timestamp = pd.Timestamp(row["feature_time"])
            if last is None or timestamp - last >= cooldown:
                accepted.loc[index] = True
                last = timestamp
    return accepted


def build_v212_event_features(
    events: pd.DataFrame,
    close: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: V212FeatureConfig = V212FeatureConfig(),
) -> pd.DataFrame:
    returns, prior_mean, prior_scale, source_z = build_v212_source_context(close, cfg)
    membership_groups = {
        (pd.Timestamp(month), str(community)): sorted(group["symbol"].astype(str))
        for (month, community), group in membership.groupby(
            ["month_start", "community_id"], sort=True
        )
    }
    symbol_community = {
        (pd.Timestamp(row.month_start), str(row.symbol)): str(row.community_id)
        for row in membership.itertuples(index=False)
    }
    index = pd.DatetimeIndex(close.index)
    rows: list[dict[str, object]] = []
    for event in events.itertuples(index=False):
        available = pd.Timestamp(event.event_available_time)
        position = int(index.searchsorted(available, side="right"))
        if position >= len(index):
            continue
        feature_time = pd.Timestamp(index[position])
        if feature_time < cfg.first_feature_time:
            continue
        feature_delay = (feature_time - available).total_seconds() / 60.0
        if feature_delay > cfg.maximum_feature_delay_minutes:
            continue
        entry_position = position + cfg.entry_delay_bars
        if entry_position >= len(index):
            continue
        entry_time = pd.Timestamp(index[entry_position])
        source = str(event.cex_symbol)
        if source not in close.columns:
            continue
        month = _month(feature_time)
        community = symbol_community.get((month, source))
        if community is None:
            continue
        peer_members = [
            symbol
            for symbol in membership_groups.get((month, community), [])
            if symbol != source and symbol in close.columns
        ]
        source_return = returns.at[feature_time, source]
        mean = prior_mean.at[feature_time, source]
        scale = prior_scale.at[feature_time, source]
        zscore = source_z.at[feature_time, source]
        peer_returns = returns.loc[feature_time, peer_members].dropna()
        available_peers = peer_returns.index.astype(str).tolist()
        direction = float(np.sign(source_return)) if pd.notna(source_return) else math.nan
        if np.isfinite(direction) and direction != 0 and len(peer_returns):
            signed_peer_returns = (direction * peer_returns).sort_values(kind="stable")
            laggard_size = max(
                cfg.minimum_laggard_peers,
                int(math.ceil(len(signed_peer_returns) / 2.0)),
            )
            laggards = signed_peer_returns.head(laggard_size).index.astype(str).tolist()
        else:
            laggards = []
        confirmed = bool(
            pd.notna(zscore)
            and abs(float(zscore)) >= cfg.source_z_threshold
            and np.isfinite(direction)
            and direction != 0
        )
        rows.append(
            {
                "event_id": str(event.event_id),
                "source_symbol": source,
                "chain": str(getattr(event, "chain", "")),
                "source_vendor": str(event.source),
                "mapping_confidence": str(event.mapping_confidence),
                "event_time": pd.Timestamp(event.event_time),
                "event_available_time": available,
                "feature_time": feature_time,
                "entry_time": entry_time,
                "feature_delay_minutes": feature_delay,
                "period": _v212_period(feature_time),
                "entry_month": feature_time.strftime("%Y-%m"),
                "month_start": month,
                "community_id": community,
                "dex_attention_zscore": float(event.zscore),
                "dex_attention_percentile": float(event.percentile),
                "source_return_1h": float(source_return),
                "source_prior_mean_1h": float(mean),
                "source_prior_scale_1h": float(scale),
                "source_return_z": float(zscore),
                "source_direction": int(direction) if np.isfinite(direction) else 0,
                "source_confirmed": confirmed,
                "source_return_window_end": feature_time,
                "source_scale_history_end": feature_time - pd.Timedelta(minutes=15),
                "community_peer_count": len(available_peers),
                "community_peer_symbols": _join_symbols(available_peers),
                "laggard_peer_count": len(laggards),
                "laggard_peer_symbols": _join_symbols(laggards),
            }
        )
    features = pd.DataFrame(rows)
    if features.empty:
        return features
    features["community_event_accepted"] = _community_cooldown_mask(
        features, cfg.community_cooldown_hours
    )
    features[ALL_PEERS] = (
        features["source_confirmed"]
        & features["community_event_accepted"]
        & features["community_peer_count"].ge(cfg.minimum_all_peers)
    )
    features[LAGGARD_PEERS] = features[ALL_PEERS] & features["laggard_peer_count"].ge(
        cfg.minimum_laggard_peers
    )
    return features.sort_values(
        ["feature_time", "community_id", "source_symbol", "event_id"]
    ).reset_index(drop=True)


def build_v212_candidate_features(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in features.itertuples(index=False):
        rules = (
            (ALL_PEERS, bool(getattr(row, ALL_PEERS)), row.community_peer_symbols),
            (
                LAGGARD_PEERS,
                bool(getattr(row, LAGGARD_PEERS)),
                row.laggard_peer_symbols,
            ),
        )
        for candidate, selected, symbols in rules:
            if not selected:
                continue
            selection = [symbol for symbol in str(symbols).split("|") if symbol]
            rows.append(
                {
                    "candidate": candidate,
                    "event_id": row.event_id,
                    "source_symbol": row.source_symbol,
                    "source_vendor": row.source_vendor,
                    "chain": row.chain,
                    "event_time": row.event_time,
                    "event_available_time": row.event_available_time,
                    "feature_time": row.feature_time,
                    "entry_time": row.entry_time,
                    "period": row.period,
                    "entry_month": row.entry_month,
                    "community_id": row.community_id,
                    "source_return_z": row.source_return_z,
                    "source_direction": row.source_direction,
                    "selection_count": len(selection),
                    "selection_symbols": _join_symbols(selection),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["feature_time", "candidate", "community_id", "source_symbol"])
        .reset_index(drop=True)
    )


def summarize_v212_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        local = candidates[candidates["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = local if scope == "all" else local[local["period"].eq(scope)]
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": len(sample),
                    "active_days": (
                        sample["entry_time"].dt.floor("D").nunique() if len(sample) else 0
                    ),
                    "active_months": (sample["entry_month"].nunique() if len(sample) else 0),
                    "source_symbols": (sample["source_symbol"].nunique() if len(sample) else 0),
                    "communities": (sample["community_id"].nunique() if len(sample) else 0),
                    "mean_selection_count": (
                        float(sample["selection_count"].mean()) if len(sample) else math.nan
                    ),
                    "median_abs_source_z": (
                        float(sample["source_return_z"].abs().median()) if len(sample) else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_v212_sources(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby(["candidate", "period", "source_vendor"], as_index=False)
        .agg(
            events=("event_id", "size"),
            source_symbols=("source_symbol", "nunique"),
            active_months=("entry_month", "nunique"),
        )
        .sort_values(["candidate", "period", "source_vendor"])
        .reset_index(drop=True)
    )


def _minimum_source_spacing_ok(events: pd.DataFrame, hours: int) -> bool:
    for _, local in events.groupby("cex_symbol", sort=True):
        delta = local.sort_values("event_available_time")["event_available_time"].diff().dropna()
        if not delta.ge(pd.Timedelta(hours=hours)).all():
            return False
    return True


def _minimum_community_spacing_ok(features: pd.DataFrame, hours: int) -> bool:
    accepted = features[features["community_event_accepted"]]
    for _, local in accepted.groupby("community_id", sort=True):
        delta = local.sort_values("feature_time")["feature_time"].diff().dropna()
        if not delta.ge(pd.Timedelta(hours=hours)).all():
            return False
    return True


def audit_v212_features(
    events: pd.DataFrame,
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: V212FeatureConfig = V212FeatureConfig(),
) -> pd.DataFrame:
    all_coverage = coverage[coverage["scope"].eq("all")]
    period_coverage = coverage[coverage["scope"].ne("all")]
    membership_lookup = {
        (pd.Timestamp(row.month_start), str(row.symbol)): str(row.community_id)
        for row in membership.itertuples(index=False)
    }
    exact_peer_membership = True
    for row in candidates.itertuples(index=False):
        for symbol in str(row.selection_symbols).split("|"):
            if (
                not symbol
                or symbol == row.source_symbol
                or membership_lookup.get((_month(row.feature_time), symbol)) != row.community_id
            ):
                exact_peer_membership = False
                break
    candidate_columns = " ".join(candidates.columns).lower()
    checks = {
        "source_event_ids_unique": not events["event_id"].duplicated().any(),
        "event_information_available_after_event": bool(
            events["event_available_time"].gt(events["event_time"]).all()
        ),
        "source_cooldown_exact_or_longer": _minimum_source_spacing_ok(
            events, cfg.source_cooldown_hours
        ),
        "feature_close_strictly_after_availability": bool(
            features["feature_time"].gt(features["event_available_time"]).all()
        ),
        "feature_delay_within_15_minutes": bool(
            features["feature_delay_minutes"].gt(0).all()
            and features["feature_delay_minutes"].le(cfg.maximum_feature_delay_minutes).all()
        ),
        "entry_one_full_bar_after_feature": bool(
            candidates["entry_time"]
            .sub(candidates["feature_time"])
            .eq(pd.Timedelta(minutes=15 * cfg.entry_delay_bars))
            .all()
        ),
        "source_return_window_observed_by_feature": bool(
            features["source_return_window_end"].eq(features["feature_time"]).all()
        ),
        "source_scale_history_strictly_prior": bool(
            features["source_scale_history_end"].lt(features["feature_time"]).all()
        ),
        "candidate_source_confirmation_threshold": bool(
            candidates["source_return_z"].abs().ge(cfg.source_z_threshold).all()
        ),
        "community_cooldown_exact_or_longer": _minimum_community_spacing_ok(
            features, cfg.community_cooldown_hours
        ),
        "selected_peers_match_same_month_community": exact_peer_membership,
        "selected_peers_exclude_source": bool(
            all(
                row.source_symbol not in str(row.selection_symbols).split("|")
                for row in candidates.itertuples(index=False)
            )
        ),
        "candidate_minimum_bucket_sizes": bool(
            candidates.loc[candidates["candidate"].eq(ALL_PEERS), "selection_count"]
            .ge(cfg.minimum_all_peers)
            .all()
            and candidates.loc[candidates["candidate"].eq(LAGGARD_PEERS), "selection_count"]
            .ge(cfg.minimum_laggard_peers)
            .all()
        ),
        "candidate_keys_unique": not candidates.duplicated(["candidate", "event_id"]).any(),
        "candidate_all_sample_coverage": bool(
            all_coverage["events"].ge(cfg.minimum_candidate_events).all()
        ),
        "candidate_each_period_coverage": bool(
            period_coverage["events"].ge(cfg.minimum_period_events).all()
        ),
        "candidate_source_diversification": bool(
            all_coverage["source_symbols"].ge(cfg.minimum_source_symbols).all()
        ),
        "candidate_month_diversification": bool(
            all_coverage["active_months"].ge(cfg.minimum_active_months).all()
        ),
        "both_dex_vendors_represented": set(candidates["source_vendor"].unique())
        == set(cfg.event_sources),
        "no_post_event_outcome_columns": not any(
            token in candidate_columns
            for token in ("future", "gross", "net_return", "pnl", "exit_price")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def _write_findings(
    checks: pd.DataFrame,
    coverage: pd.DataFrame,
    source_coverage: pd.DataFrame,
    event_count: int,
    feature_count: int,
    hashes: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "feature_viable_freeze_dex_community_propagation"
        if bool(checks["passed"].all())
        else "feature_audit_failed_do_not_reveal"
    )
    failed = checks.loc[~checks["passed"], "check"].astype(str).tolist()
    text = [
        "# v21.2 DEX Community-Propagation Feature Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"After 24-hour same-source de-overlap, {event_count:,} DEX attention "
        f"events were available; {feature_count:,} mapped to a causal 15-minute "
        "source feature and a forward-frozen monthly graph community.",
        "",
        coverage[coverage["scope"].eq("all")].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Frozen feature candidates:",
        "",
        "- DAP1: a DEX volume-attention event is followed by an absolute CEX "
        "source return innovation of at least 1.0 prior sigma; after a four-hour "
        "community cooldown, select every other available member of the source's "
        "monthly graph community, requiring at least four peers.",
        "- DAP2: apply the same source rule, rank peers by their observed one-hour "
        "return in the source direction, and select the slowest half (at least "
        "three names). This is a relative laggard rule, not an outcome filter.",
        "- DEX data are available at the recorded event availability time. The "
        "source feature is the first 15-minute close strictly afterward; proposed "
        "execution waits one additional complete 15-minute bar.",
        "- Source-return normalization uses only the preceding 20-30 days. No "
        "post-event portfolio return was calculated or inspected in this audit.",
        "- Coverage-only chronology freezes Aug-Nov 2025 as development, "
        "Dec 2025-Feb 2026 as a visible but excluded vendor-transition interval, "
        "Mar-Apr 2026 as validation, and May 2026 onward as holdout.",
        "",
        "Vendor/period coverage:",
        "",
        source_coverage.to_markdown(index=False),
        "",
        "Input hashes:",
        "",
        hashes.to_markdown(index=False),
        "",
    ]
    if failed:
        text.extend(["Failed checks:", "", *[f"- {item}" for item in failed], ""])
    text.extend(
        [
            "No live, PaperLive, application, leverage, remote, or order state was "
            "read or changed.",
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v212_feature_audit(
    event_path: Path = EVENT_PATH,
    membership_path: Path = MEMBERSHIP_PATH,
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V212FeatureConfig = V212FeatureConfig(),
) -> dict[str, Path]:
    close, _ = load_v178_market_data(kline_root)
    membership = load_v195_membership(membership_path)
    events = load_v212_events(event_path, cfg)
    features = build_v212_event_features(events, close, membership, cfg)
    candidates = build_v212_candidate_features(features)
    coverage = summarize_v212_coverage(candidates)
    source_coverage = summarize_v212_sources(candidates)
    checks = audit_v212_features(events, features, candidates, coverage, membership, cfg)
    hashes = pd.DataFrame(
        [
            {"input": str(event_path), "sha256": _sha256(event_path)},
            {"input": str(membership_path), "sha256": _sha256(membership_path)},
        ]
    )
    root = ensure_dir(report_root)
    outputs = {
        "event_features": root / "dex_community_event_features.parquet",
        "candidates": root / "candidate_feature_events.parquet",
        "coverage": root / "feature_coverage_summary.csv",
        "source_coverage": root / "source_period_coverage.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": findings_path,
    }
    features.to_parquet(outputs["event_features"], index=False)
    candidates.to_parquet(outputs["candidates"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    source_coverage.to_csv(outputs["source_coverage"], index=False)
    checks.to_csv(outputs["checks"], index=False)
    hashes.to_csv(outputs["hashes"], index=False)
    _write_findings(
        checks,
        coverage,
        source_coverage,
        len(events),
        len(features),
        hashes,
        findings_path,
    )
    return outputs


__all__ = [
    "ALL_PEERS",
    "CANDIDATES",
    "LAGGARD_PEERS",
    "V212FeatureConfig",
    "audit_v212_features",
    "build_v212_candidate_features",
    "build_v212_event_features",
    "build_v212_source_context",
    "load_v212_events",
    "summarize_v212_coverage",
    "write_v212_feature_audit",
]
