"""Feature-only rank audit inside frozen broad book-vacuum events."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir


V224_ROOT = Path("reports/v22_4_alt_book_vacuum_pressure_feature_audit")
EVENT_PATH = V224_ROOT / "candidate_feature_events.parquet"
SYMBOL_STATE_PATH = V224_ROOT / "symbol_feature_states.parquet"
REPORT_ROOT = Path("reports/v22_7_vacuum_pressure_cross_section_feature_audit")
FINDINGS_PATH = Path(
    "docs/v227_vacuum_pressure_cross_section_feature_audit_2026_07_17.md"
)
CANDIDATE = "DVS1_VACUUM_PRESSURE_TOP4_MINUS_BOTTOM4"


@dataclass(frozen=True)
class V227Config:
    event_path: Path = EVENT_PATH
    symbol_state_path: Path = SYMBOL_STATE_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    side_count: int = 4
    minimum_events: int = 150
    minimum_period_events: int = 45
    minimum_active_months: int = 11


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_v227_rank_features(
    events: pd.DataFrame,
    symbol_states: pd.DataFrame,
    cfg: V227Config = V227Config(),
) -> pd.DataFrame:
    event_columns = [
        "entry_time",
        "entry_month",
        "period",
        "bucket_pressure",
        "signal_direction",
        "directional_symbol_count",
        "withdrawing_symbol_count",
    ]
    local_events = events[event_columns].rename(
        columns={"entry_time": "decision_time"}
    )
    local = local_events.merge(
        symbol_states[
            [
                "decision_time",
                "symbol",
                "imbalance_z",
                "depth_log_change",
                "prior_withdrawal_threshold",
                "depth_withdrawal_state",
                "feature_ready",
            ]
        ],
        on="decision_time",
        how="inner",
        validate="one_to_many",
    )
    rows: list[dict[str, object]] = []
    for decision_time, group in local.groupby(
        "decision_time", sort=True, observed=True
    ):
        ready = group[group["feature_ready"]].sort_values(
            ["imbalance_z", "symbol"], ascending=[False, True]
        )
        if len(ready) < 2 * cfg.side_count:
            continue
        longs = ready.head(cfg.side_count)
        shorts = ready.tail(cfg.side_count)
        for side, selected, sign in (
            ("long", longs, 1.0),
            ("short", shorts, -1.0),
        ):
            for rank, item in enumerate(selected.itertuples(index=False), start=1):
                rows.append(
                    {
                        "candidate": CANDIDATE,
                        "feature_time": pd.Timestamp(decision_time),
                        "entry_time": pd.Timestamp(decision_time),
                        "entry_month": item.entry_month,
                        "period": item.period,
                        "bucket_pressure": float(item.bucket_pressure),
                        "aggregate_signal_direction": int(item.signal_direction),
                        "directional_symbol_count": int(item.directional_symbol_count),
                        "withdrawing_symbol_count": int(item.withdrawing_symbol_count),
                        "symbol": item.symbol,
                        "side": side,
                        "side_rank": rank,
                        "imbalance_z": float(item.imbalance_z),
                        "depth_log_change": float(item.depth_log_change),
                        "prior_withdrawal_threshold": float(
                            item.prior_withdrawal_threshold
                        ),
                        "depth_withdrawal_state": bool(item.depth_withdrawal_state),
                        "raw_weight": sign * 0.5 / cfg.side_count,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "side", "side_rank", "symbol"]
    ).reset_index(drop=True)


def summarize_v227(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        local = features if scope == "all" else features[features["period"].eq(scope)]
        event_rows = local.drop_duplicates("entry_time")
        score_gap = local.groupby(["entry_time", "side"], observed=True)[
            "imbalance_z"
        ].mean().unstack("side")
        rows.append(
            {
                "candidate": CANDIDATE,
                "scope": scope,
                "events": event_rows["entry_time"].nunique(),
                "active_months": event_rows["entry_month"].nunique(),
                "mean_score_gap": float((score_gap["long"] - score_gap["short"]).mean()),
                "long_withdrawal_rate": float(
                    local.loc[local["side"].eq("long"), "depth_withdrawal_state"].mean()
                ),
                "short_withdrawal_rate": float(
                    local.loc[local["side"].eq("short"), "depth_withdrawal_state"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_v227_features(
    features: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: V227Config = V227Config(),
) -> pd.DataFrame:
    event_side = features.groupby(["entry_time", "side"], observed=True).agg(
        names=("symbol", "nunique"),
        weight=("raw_weight", "sum"),
        minimum_score=("imbalance_z", "min"),
        maximum_score=("imbalance_z", "max"),
    )
    long = event_side.xs("long", level="side")
    short = event_side.xs("short", level="side")
    all_row = summary[summary["scope"].eq("all")].iloc[0]
    periods = summary[summary["scope"].ne("all")]
    columns = " ".join(features.columns).lower()
    checks = {
        "feature_week_symbol_keys_unique": not features.duplicated(
            ["entry_time", "symbol"]
        ).any(),
        "exactly_four_names_per_side": bool(
            event_side["names"].eq(cfg.side_count).all()
        ),
        "raw_half_notional_per_side": bool(
            long["weight"].sub(0.5).abs().lt(1e-12).all()
            and short["weight"].add(0.5).abs().lt(1e-12).all()
        ),
        "top4_scores_strictly_above_bottom4": bool(
            long["minimum_score"].gt(short["maximum_score"]).all()
        ),
        "long_short_weight_signs": bool(
            features.loc[features["side"].eq("long"), "raw_weight"].gt(0).all()
            and features.loc[features["side"].eq("short"), "raw_weight"].lt(0).all()
        ),
        "entry_equals_completed_feature_time": bool(
            features["entry_time"].eq(features["feature_time"]).all()
        ),
        "four_hour_event_cooldown": bool(
            features["entry_time"]
            .drop_duplicates()
            .sort_values()
            .diff()
            .dropna()
            .ge(pd.Timedelta(hours=4))
            .all()
        ),
        "minimum_total_events": int(all_row["events"]) >= cfg.minimum_events,
        "minimum_each_period_events": bool(
            periods["events"].ge(cfg.minimum_period_events).all()
        ),
        "minimum_active_months": int(all_row["active_months"])
        >= cfg.minimum_active_months,
        "no_future_outcome_columns": not any(
            token in columns
            for token in ("future", "return", "pnl", "gross", "net", "exit", "price")
        ),
    }
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})


def write_v227_vacuum_pressure_cross_section_feature_audit(
    cfg: V227Config = V227Config(),
) -> dict[str, Path]:
    events = pd.read_parquet(cfg.event_path)
    states = pd.read_parquet(cfg.symbol_state_path)
    for frame, columns in (
        (events, ("entry_time", "feature_time")),
        (states, ("decision_time",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    features = build_v227_rank_features(events, states, cfg)
    summary = summarize_v227(features)
    checks = audit_v227_features(features, summary, cfg)
    hashes = pd.DataFrame(
        [
            {"input": str(cfg.event_path), "sha256": _sha256(cfg.event_path)},
            {
                "input": str(cfg.symbol_state_path),
                "sha256": _sha256(cfg.symbol_state_path),
            },
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "ranked_symbol_features.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    features.to_parquet(paths["features"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    hashes.to_csv(paths["hashes"], index=False)
    verdict = (
        "feature_viable_freeze_vacuum_pressure_spread"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v22.7 Vacuum-Pressure Cross-Section Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "At each frozen v22.4 broad book-vacuum event, the sole candidate",
                "is long the four highest one-percent imbalance z-scores and short",
                "the four lowest, with 0.5 raw notional per side. No depth severity",
                "multiplier, threshold grid, or outcome-conditioned rank was used.",
                "",
                "No future price, return, PnL, beta, turnover, or outcome was loaded.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
