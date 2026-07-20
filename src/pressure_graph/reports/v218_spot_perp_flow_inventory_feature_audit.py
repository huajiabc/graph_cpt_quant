"""Feature-only audit for Binance spot-versus-perpetual taker-flow inventory."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _period
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    MEMBERSHIP_PATH,
    load_v195_membership,
)


SPOT_ROOT = Path("data/external/binance_spot_1h")
PERP_ROOT = Path("data/external/binance_um_carry/klines_1h")
REPORT_ROOT = Path("reports/v21_8_spot_perp_flow_inventory_feature_audit")
FINDINGS_PATH = Path("docs/v218_spot_perp_flow_inventory_feature_audit_2026_07_17.md")
GLOBAL_SPREAD = "SFI1_GLOBAL_SPOT_PERP_FLOW_DIVERGENCE"
COMMUNITY_SPREAD = "SFI2_COMMUNITY_SPOT_PERP_FLOW_DIVERGENCE"
CANDIDATES = (GLOBAL_SPREAD, COMMUNITY_SPREAD)


@dataclass(frozen=True)
class V218FeatureConfig:
    first_feature_time: pd.Timestamp = pd.Timestamp("2025-09-01", tz="UTC")
    decision_hours: tuple[int, ...] = (0, 12)
    normalization_lookback_hours: int = 30 * 24
    normalization_minimum_hours: int = 20 * 24
    activity_lookback_hours: int = 7 * 24
    activity_minimum_hours: int = 5 * 24
    minimum_activity_ratio: float = 0.5
    global_score_threshold: float = 1.0
    global_bucket_size: int = 8
    global_minimum_leg: int = 5
    community_score_threshold: float = 0.5
    community_gap_threshold: float = 1.5
    community_minimum_cross_section: int = 4
    community_minimum_pairs: int = 4
    entry_delay_hours: int = 1
    proposed_holding_hours: int = 12
    expected_common_symbols: int = 61
    minimum_candidate_events: int = 100
    minimum_period_events: int = 20
    minimum_active_months: int = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v218_venue_panel(
    root: Path,
    symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quote_frames: list[pd.Series] = []
    taker_frames: list[pd.Series] = []
    for symbol in sorted(symbols):
        path = root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=["feature_time", "quote_volume", "taker_buy_quote_volume"],
        )
        frame["feature_time"] = pd.to_datetime(frame["feature_time"], utc=True, errors="coerce")
        frame["quote_volume"] = pd.to_numeric(frame["quote_volume"], errors="coerce")
        frame["taker_buy_quote_volume"] = pd.to_numeric(
            frame["taker_buy_quote_volume"], errors="coerce"
        )
        frame = (
            frame.dropna(subset=["feature_time", "quote_volume", "taker_buy_quote_volume"])
            .drop_duplicates("feature_time", keep="last")
            .sort_values("feature_time")
            .set_index("feature_time")
        )
        quote_frames.append(frame["quote_volume"].rename(symbol))
        taker_frames.append(frame["taker_buy_quote_volume"].rename(symbol))
    if not quote_frames:
        return pd.DataFrame(), pd.DataFrame()
    quote = pd.concat(quote_frames, axis=1).sort_index()
    taker = pd.concat(taker_frames, axis=1).reindex(quote.index)
    quote.index.name = "feature_time"
    taker.index.name = "feature_time"
    return quote, taker


def build_v218_flow_features(
    quote: pd.DataFrame,
    taker_buy_quote: pd.DataFrame,
    cfg: V218FeatureConfig = V218FeatureConfig(),
) -> dict[str, pd.DataFrame]:
    imbalance = 2.0 * taker_buy_quote.div(quote.where(quote.gt(0))) - 1.0
    prior = imbalance.shift(1).rolling(
        cfg.normalization_lookback_hours,
        min_periods=cfg.normalization_minimum_hours,
    )
    prior_mean = prior.mean()
    prior_scale = prior.std(ddof=1)
    zscore = (imbalance - prior_mean).div(prior_scale.where(prior_scale.gt(0)))
    prior_activity = (
        quote.shift(1)
        .rolling(
            cfg.activity_lookback_hours,
            min_periods=cfg.activity_minimum_hours,
        )
        .median()
    )
    activity_ratio = quote.div(prior_activity.where(prior_activity.gt(0)))
    return {
        "imbalance": imbalance,
        "prior_mean": prior_mean,
        "prior_scale": prior_scale,
        "zscore": zscore,
        "activity_ratio": activity_ratio,
    }


def build_v218_decision_features(
    spot_quote: pd.DataFrame,
    spot_taker: pd.DataFrame,
    perp_quote: pd.DataFrame,
    perp_taker: pd.DataFrame,
    membership: pd.DataFrame,
    cfg: V218FeatureConfig = V218FeatureConfig(),
) -> pd.DataFrame:
    common = sorted(set(spot_quote.columns) & set(perp_quote.columns))
    common_index = spot_quote.index.intersection(perp_quote.index)
    spot_quote = spot_quote.reindex(index=common_index, columns=common)
    spot_taker = spot_taker.reindex(index=common_index, columns=common)
    perp_quote = perp_quote.reindex(index=common_index, columns=common)
    perp_taker = perp_taker.reindex(index=common_index, columns=common)
    spot = build_v218_flow_features(spot_quote, spot_taker, cfg)
    perp = build_v218_flow_features(perp_quote, perp_taker, cfg)
    membership_lookup = {
        (pd.Timestamp(row.month_start), str(row.symbol)): str(row.community_id)
        for row in membership.itertuples(index=False)
    }
    decisions = common_index[
        (common_index >= cfg.first_feature_time)
        & common_index.hour.isin(cfg.decision_hours)
        & (common_index.minute == 0)
    ]
    rows: list[dict[str, object]] = []
    for timestamp in decisions:
        month = pd.Timestamp(timestamp).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        for symbol in common:
            community = membership_lookup.get((month, symbol))
            if community is None:
                continue
            spot_z = spot["zscore"].at[timestamp, symbol]
            perp_z = perp["zscore"].at[timestamp, symbol]
            spot_activity = spot["activity_ratio"].at[timestamp, symbol]
            perp_activity = perp["activity_ratio"].at[timestamp, symbol]
            score = spot_z - perp_z
            eligible = bool(
                np.isfinite(score)
                and np.isfinite(spot_activity)
                and np.isfinite(perp_activity)
                and spot_activity >= cfg.minimum_activity_ratio
                and perp_activity >= cfg.minimum_activity_ratio
            )
            rows.append(
                {
                    "feature_time": timestamp,
                    "entry_time": timestamp + pd.Timedelta(hours=cfg.entry_delay_hours),
                    "period": _period(timestamp),
                    "entry_day": timestamp.floor("D"),
                    "entry_month": timestamp.strftime("%Y-%m"),
                    "month_start": month,
                    "community_id": community,
                    "symbol": symbol,
                    "spot_imbalance": float(spot["imbalance"].at[timestamp, symbol]),
                    "perp_imbalance": float(perp["imbalance"].at[timestamp, symbol]),
                    "spot_flow_z": float(spot_z),
                    "perp_flow_z": float(perp_z),
                    "spot_activity_ratio": float(spot_activity),
                    "perp_activity_ratio": float(perp_activity),
                    "spot_minus_perp_flow_z": float(score),
                    "feature_eligible": eligible,
                    "normalization_history_end": timestamp - pd.Timedelta(hours=1),
                    "activity_history_end": timestamp - pd.Timedelta(hours=1),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["feature_time", "community_id", "symbol"])
        .reset_index(drop=True)
    )


def _join(symbols: list[str]) -> str:
    return "|".join(sorted(symbols))


def build_v218_candidate_features(
    decisions: pd.DataFrame,
    cfg: V218FeatureConfig = V218FeatureConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp, local in decisions.groupby("feature_time", sort=True):
        eligible = local[local["feature_eligible"]].sort_values(
            ["spot_minus_perp_flow_z", "symbol"]
        )
        base = {
            "feature_time": timestamp,
            "entry_time": eligible["entry_time"].iloc[0] if len(eligible) else pd.NaT,
            "period": eligible["period"].iloc[0] if len(eligible) else "",
            "entry_day": eligible["entry_day"].iloc[0] if len(eligible) else pd.NaT,
            "entry_month": eligible["entry_month"].iloc[0] if len(eligible) else "",
            "eligible_symbols": len(eligible),
        }
        positive = eligible[eligible["spot_minus_perp_flow_z"].ge(cfg.global_score_threshold)].tail(
            cfg.global_bucket_size
        )
        negative = eligible[
            eligible["spot_minus_perp_flow_z"].le(-cfg.global_score_threshold)
        ].head(cfg.global_bucket_size)
        if len(positive) >= cfg.global_minimum_leg and len(negative) >= cfg.global_minimum_leg:
            rows.append(
                {
                    **base,
                    "candidate": GLOBAL_SPREAD,
                    "long_count": len(positive),
                    "short_count": len(negative),
                    "pair_count": 0,
                    "long_symbols": _join(positive["symbol"].astype(str).tolist()),
                    "short_symbols": _join(negative["symbol"].astype(str).tolist()),
                    "community_pairs": "",
                    "mean_long_score": float(positive["spot_minus_perp_flow_z"].mean()),
                    "mean_short_score": float(negative["spot_minus_perp_flow_z"].mean()),
                    "minimum_pair_gap": math.nan,
                }
            )
        pairs: list[tuple[str, str, float]] = []
        for _, group in eligible.groupby("community_id", sort=True):
            ranked = group.sort_values(["spot_minus_perp_flow_z", "symbol"])
            if len(ranked) < cfg.community_minimum_cross_section:
                continue
            low = ranked.iloc[0]
            high = ranked.iloc[-1]
            gap = float(high["spot_minus_perp_flow_z"] - low["spot_minus_perp_flow_z"])
            if (
                float(high["spot_minus_perp_flow_z"]) >= cfg.community_score_threshold
                and float(low["spot_minus_perp_flow_z"]) <= -cfg.community_score_threshold
                and gap >= cfg.community_gap_threshold
            ):
                pairs.append((str(high["symbol"]), str(low["symbol"]), gap))
        if len(pairs) >= cfg.community_minimum_pairs:
            rows.append(
                {
                    **base,
                    "candidate": COMMUNITY_SPREAD,
                    "long_count": len(pairs),
                    "short_count": len(pairs),
                    "pair_count": len(pairs),
                    "long_symbols": _join([high for high, _, _ in pairs]),
                    "short_symbols": _join([low for _, low, _ in pairs]),
                    "community_pairs": "|".join(f"{high}>{low}" for high, low, _ in sorted(pairs)),
                    "mean_long_score": float(
                        eligible.set_index("symbol")
                        .loc[[high for high, _, _ in pairs], "spot_minus_perp_flow_z"]
                        .mean()
                    ),
                    "mean_short_score": float(
                        eligible.set_index("symbol")
                        .loc[[low for _, low, _ in pairs], "spot_minus_perp_flow_z"]
                        .mean()
                    ),
                    "minimum_pair_gap": min(gap for _, _, gap in pairs),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature_time", "candidate"]).reset_index(drop=True)


def summarize_v218_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
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
                    "active_days": sample["entry_day"].nunique() if len(sample) else 0,
                    "active_months": sample["entry_month"].nunique() if len(sample) else 0,
                    "mean_eligible_symbols": (
                        float(sample["eligible_symbols"].mean()) if len(sample) else math.nan
                    ),
                    "mean_long_count": (
                        float(sample["long_count"].mean()) if len(sample) else math.nan
                    ),
                    "mean_short_count": (
                        float(sample["short_count"].mean()) if len(sample) else math.nan
                    ),
                    "mean_score_spread": (
                        float((sample["mean_long_score"] - sample["mean_short_score"]).mean())
                        if len(sample)
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_v218_features(
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    common_symbols: set[str],
    cfg: V218FeatureConfig = V218FeatureConfig(),
) -> pd.DataFrame:
    all_coverage = coverage[coverage["scope"].eq("all")]
    period_coverage = coverage[coverage["scope"].ne("all")]
    candidate_columns = " ".join(candidates.columns).lower()
    checks = {
        "common_universe_exact_61_symbols": len(common_symbols) == cfg.expected_common_symbols,
        "decision_symbol_keys_unique": not decisions.duplicated(["feature_time", "symbol"]).any(),
        "decision_hours_only_00_12": set(decisions["feature_time"].dt.hour.unique())
        == set(cfg.decision_hours),
        "normalization_history_strictly_prior": bool(
            decisions["normalization_history_end"].lt(decisions["feature_time"]).all()
        ),
        "activity_history_strictly_prior": bool(
            decisions["activity_history_end"].lt(decisions["feature_time"]).all()
        ),
        "eligible_rows_meet_activity_floor": bool(
            decisions.loc[decisions["feature_eligible"], "spot_activity_ratio"]
            .ge(cfg.minimum_activity_ratio)
            .all()
            and decisions.loc[decisions["feature_eligible"], "perp_activity_ratio"]
            .ge(cfg.minimum_activity_ratio)
            .all()
        ),
        "entry_waits_one_complete_hour": bool(
            candidates["entry_time"]
            .sub(candidates["feature_time"])
            .eq(pd.Timedelta(hours=cfg.entry_delay_hours))
            .all()
        ),
        "candidate_keys_unique": not candidates.duplicated(["candidate", "feature_time"]).any(),
        "long_short_legs_disjoint": bool(
            all(
                not set(str(row.long_symbols).split("|")) & set(str(row.short_symbols).split("|"))
                for row in candidates.itertuples(index=False)
            )
        ),
        "long_scores_positive_short_scores_negative": bool(
            candidates["mean_long_score"].gt(0).all() and candidates["mean_short_score"].lt(0).all()
        ),
        "global_minimum_leg_sizes": bool(
            candidates.loc[candidates["candidate"].eq(GLOBAL_SPREAD), "long_count"]
            .ge(cfg.global_minimum_leg)
            .all()
            and candidates.loc[candidates["candidate"].eq(GLOBAL_SPREAD), "short_count"]
            .ge(cfg.global_minimum_leg)
            .all()
        ),
        "community_minimum_pair_count": bool(
            candidates.loc[candidates["candidate"].eq(COMMUNITY_SPREAD), "pair_count"]
            .ge(cfg.community_minimum_pairs)
            .all()
        ),
        "community_minimum_pair_gap": bool(
            candidates.loc[candidates["candidate"].eq(COMMUNITY_SPREAD), "minimum_pair_gap"]
            .ge(cfg.community_gap_threshold)
            .all()
        ),
        "candidate_all_sample_coverage": bool(
            all_coverage["events"].ge(cfg.minimum_candidate_events).all()
        ),
        "candidate_each_period_coverage": bool(
            period_coverage["events"].ge(cfg.minimum_period_events).all()
        ),
        "candidate_month_diversification": bool(
            all_coverage["active_months"].ge(cfg.minimum_active_months).all()
        ),
        "no_post_event_outcome_columns": not any(
            token in candidate_columns
            for token in ("future", "gross", "net_return", "pnl", "exit_price")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def _write_findings(
    checks: pd.DataFrame,
    coverage: pd.DataFrame,
    decision_count: int,
    common_symbol_count: int,
    hashes: pd.DataFrame,
    path: Path,
) -> None:
    verdict = (
        "feature_viable_freeze_spot_perp_flow_inventory"
        if bool(checks["passed"].all())
        else "feature_audit_failed_do_not_reveal"
    )
    failed = checks.loc[~checks["passed"], "check"].astype(str).tolist()
    text = [
        "# v21.8 Spot-Perpetual Flow-Inventory Feature Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"The exact spot/perpetual/member intersection contains {common_symbol_count} "
        f"symbols. The audit built {decision_count:,} symbol-hour feature rows at "
        "00/12 UTC using only then-available data.",
        "",
        coverage[coverage["scope"].eq("all")].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Frozen feature candidates:",
        "",
        "- SFI1 ranks the causally standardized spot-minus-perpetual taker imbalance "
        "gap globally, taking up to eight scores above +1 and eight below -1, with "
        "at least five names per leg.",
        "- SFI2 takes the highest and lowest gap within each forward-frozen monthly "
        "graph community when both signs and a 1.5-sigma within-community gap are "
        "present; at least four community pairs are required.",
        "- Both spot and perpetual hourly quote activity must be at least half their "
        "strictly prior seven-day median. Flow z-scores use only the prior 20-30 days.",
        "- The feature is observed at the hourly close and proposed execution waits "
        "one additional complete hour. No post-event return was calculated.",
        "",
        "Input manifest hashes:",
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


def write_v218_feature_audit(
    spot_root: Path = SPOT_ROOT,
    perp_root: Path = PERP_ROOT,
    membership_path: Path = MEMBERSHIP_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V218FeatureConfig = V218FeatureConfig(),
) -> dict[str, Path]:
    membership = load_v195_membership(membership_path)
    membership_symbols = set(membership["symbol"].astype(str))
    spot_symbols = {path.stem for path in spot_root.glob("*.parquet")}
    perp_symbols = {path.stem for path in perp_root.glob("*.parquet")}
    common_symbols = membership_symbols & spot_symbols & perp_symbols
    spot_quote, spot_taker = load_v218_venue_panel(spot_root, common_symbols)
    perp_quote, perp_taker = load_v218_venue_panel(perp_root, common_symbols)
    decisions = build_v218_decision_features(
        spot_quote,
        spot_taker,
        perp_quote,
        perp_taker,
        membership,
        cfg,
    )
    candidates = build_v218_candidate_features(decisions, cfg)
    coverage = summarize_v218_coverage(candidates)
    checks = audit_v218_features(decisions, candidates, coverage, common_symbols, cfg)
    hashes = pd.DataFrame(
        [
            {
                "input": str(spot_root / "manifest.csv"),
                "sha256": _sha256(spot_root / "manifest.csv"),
            },
            {
                "input": str(perp_root.parent / "manifest.csv"),
                "sha256": _sha256(perp_root.parent / "manifest.csv"),
            },
            {"input": str(membership_path), "sha256": _sha256(membership_path)},
        ]
    )
    root = ensure_dir(report_root)
    outputs = {
        "decisions": root / "decision_symbol_features.parquet",
        "candidates": root / "candidate_feature_events.parquet",
        "coverage": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": findings_path,
    }
    decisions.to_parquet(outputs["decisions"], index=False)
    candidates.to_parquet(outputs["candidates"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    checks.to_csv(outputs["checks"], index=False)
    hashes.to_csv(outputs["hashes"], index=False)
    _write_findings(
        checks,
        coverage,
        len(decisions),
        len(common_symbols),
        hashes,
        findings_path,
    )
    return outputs


__all__ = [
    "CANDIDATES",
    "COMMUNITY_SPREAD",
    "GLOBAL_SPREAD",
    "V218FeatureConfig",
    "audit_v218_features",
    "build_v218_candidate_features",
    "build_v218_decision_features",
    "build_v218_flow_features",
    "load_v218_venue_panel",
    "summarize_v218_coverage",
    "write_v218_feature_audit",
]
