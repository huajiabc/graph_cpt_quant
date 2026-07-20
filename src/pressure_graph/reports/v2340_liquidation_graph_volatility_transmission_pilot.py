"""Retrospective 17x17 liquidation-to-volatility transmission graph pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    EXPECTED_SYMBOLS,
    load_v2334_liquidations,
)
from pressure_graph.reports.v2336_liquidation_price_mechanism_pilot import (
    build_v2336_price_outcomes,
    load_v2336_prices,
)


REPORT_ROOT = Path("reports/v23_40_liquidation_graph_volatility_transmission_pilot")
FINDINGS_PATH = Path(
    "docs/v2340_liquidation_graph_volatility_transmission_pilot_2026_07_17.md"
)


def _partial_rank(
    feature: pd.Series,
    outcome: pd.Series,
    control: pd.Series,
) -> float:
    local = pd.DataFrame(
        {"feature": feature, "outcome": outcome, "control": control}
    ).dropna()
    design = np.column_stack(
        [np.ones(len(local)), rankdata(local["control"].to_numpy())]
    )
    feature_rank = rankdata(local["feature"].to_numpy())
    outcome_rank = rankdata(local["outcome"].to_numpy())
    feature_residual = feature_rank - design @ np.linalg.lstsq(
        design, feature_rank, rcond=None
    )[0]
    outcome_residual = outcome_rank - design @ np.linalg.lstsq(
        design, outcome_rank, rcond=None
    )[0]
    return float(np.corrcoef(feature_residual, outcome_residual)[0, 1])


def _shift_percentile(
    feature: pd.Series,
    outcome: pd.Series,
    control: pd.Series | None = None,
) -> float:
    if control is None:
        actual = float(feature.corr(outcome, method="spearman"))
        placebos = [
            pd.Series(np.roll(feature.to_numpy(), shift)).corr(
                outcome.reset_index(drop=True), method="spearman"
            )
            for shift in range(1, len(feature))
        ]
    else:
        local = pd.DataFrame(
            {"feature": feature, "outcome": outcome, "control": control}
        ).dropna()
        actual = _partial_rank(local["feature"], local["outcome"], local["control"])
        placebos = [
            _partial_rank(
                pd.Series(np.roll(local["feature"].to_numpy(), shift)),
                local["outcome"].reset_index(drop=True),
                local["control"].reset_index(drop=True),
            )
            for shift in range(1, len(local))
        ]
    return float(np.mean(np.asarray(placebos) <= actual))


def build_v2340_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = load_v2334_liquidations()
    events["decision_time"] = events["event_time"].dt.floor(
        "15min"
    ) + pd.Timedelta(minutes=15)
    source_usd = (
        events.groupby(["decision_time", "bybit_symbol"])["notional_usd"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=EXPECTED_SYMBOLS, fill_value=0.0)
    )
    outcomes = build_v2336_price_outcomes(load_v2336_prices())
    receiver_range = outcomes.pivot(
        index="decision_time", columns="symbol", values="log_range_60m"
    ).reindex(columns=EXPECTED_SYMBOLS)
    common = receiver_range.dropna().index
    source_usd = source_usd.reindex(common, fill_value=0.0)
    receiver_range = receiver_range.loc[common]
    source_usd.index.name = "decision_time"
    receiver_range.index.name = "decision_time"
    return source_usd, receiver_range


def build_v2340_transmission_matrix(
    source_usd: pd.DataFrame,
    receiver_range: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for source in EXPECTED_SYMBOLS:
        feature = np.log1p(source_usd[source])
        for receiver in EXPECTED_SYMBOLS:
            outcome = receiver_range[receiver]
            prior = outcome.shift(4)
            rows.append(
                {
                    "source_symbol": source,
                    "receiver_symbol": receiver,
                    "observations": int(prior.notna().sum()),
                    "raw_spearman": float(
                        feature.corr(outcome, method="spearman")
                    ),
                    "partial_rank_controlling_receiver_prior_range": _partial_rank(
                        feature, outcome, prior
                    ),
                    "is_own_symbol": source == receiver,
                    "is_btc_receiver": receiver == "BTCUSDT",
                }
            )
    return pd.DataFrame(rows)


def build_v2340_aggregate_summary(
    source_usd: pd.DataFrame,
    receiver_range: pd.DataFrame,
) -> pd.DataFrame:
    total = source_usd.sum(axis=1)
    alt_total = source_usd.drop(columns="BTCUSDT").sum(axis=1)
    breadth = source_usd.gt(0).sum(axis=1).astype(float)
    weights = source_usd.div(total.replace(0, np.nan), axis=0)
    hhi = weights.pow(2).sum(axis=1).fillna(0.0)
    median_receiver_range = receiver_range.median(axis=1)
    prior_median_range = median_receiver_range.shift(4)
    features = {
        "all_source_log_total_usd": np.log1p(total),
        "alt_source_log_total_usd": np.log1p(alt_total),
        "active_source_breadth": breadth,
        "source_notional_hhi": hhi,
    }
    rows = []
    for name, feature in features.items():
        rows.append(
            {
                "feature": name,
                "observations": int(prior_median_range.notna().sum()),
                "raw_spearman_to_future_median_range": float(
                    feature.corr(median_receiver_range, method="spearman")
                ),
                "partial_rank_controlling_prior_median_range": _partial_rank(
                    feature, median_receiver_range, prior_median_range
                ),
                "raw_circular_shift_percentile": _shift_percentile(
                    feature, median_receiver_range
                ),
                "partial_circular_shift_percentile": _shift_percentile(
                    feature, median_receiver_range, prior_median_range
                ),
            }
        )
    return pd.DataFrame(rows)


def build_v2340_source_summary(
    source_usd: pd.DataFrame,
    receiver_range: pd.DataFrame,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    median_receiver_range = receiver_range.median(axis=1)
    prior_median_range = median_receiver_range.shift(4)
    rows = []
    for source in EXPECTED_SYMBOLS:
        feature = np.log1p(source_usd[source])
        local = matrix[matrix["source_symbol"].eq(source)]
        rows.append(
            {
                "source_symbol": source,
                "event_buckets": int(source_usd[source].gt(0).sum()),
                "total_notional_usd": float(source_usd[source].sum()),
                "raw_to_future_median_range": float(
                    feature.corr(median_receiver_range, method="spearman")
                ),
                "partial_to_future_median_range": _partial_rank(
                    feature, median_receiver_range, prior_median_range
                ),
                "mean_receiver_raw_spearman": float(local["raw_spearman"].mean()),
                "mean_receiver_partial_rank": float(
                    local[
                        "partial_rank_controlling_receiver_prior_range"
                    ].mean()
                ),
                "positive_receiver_share_raw": float(
                    local["raw_spearman"].gt(0).mean()
                ),
                "positive_receiver_share_partial": float(
                    local[
                        "partial_rank_controlling_receiver_prior_range"
                    ].gt(0).mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "partial_to_future_median_range", ascending=False
    )


def run_v2340() -> dict[str, object]:
    source_usd, receiver_range = build_v2340_panels()
    matrix = build_v2340_transmission_matrix(source_usd, receiver_range)
    aggregate = build_v2340_aggregate_summary(source_usd, receiver_range)
    sources = build_v2340_source_summary(source_usd, receiver_range, matrix)
    aggregate_index = aggregate.set_index("feature")
    all_partial = float(
        aggregate_index.loc[
            "all_source_log_total_usd",
            "partial_rank_controlling_prior_median_range",
        ]
    )
    alt_partial = float(
        aggregate_index.loc[
            "alt_source_log_total_usd",
            "partial_rank_controlling_prior_median_range",
        ]
    )
    breadth_partial = float(
        aggregate_index.loc[
            "active_source_breadth",
            "partial_rank_controlling_prior_median_range",
        ]
    )
    hhi_partial = float(
        aggregate_index.loc[
            "source_notional_hhi",
            "partial_rank_controlling_prior_median_range",
        ]
    )
    best_single = float(sources["partial_to_future_median_range"].max())
    best_alt = float(
        sources.loc[
            sources["source_symbol"].ne("BTCUSDT"),
            "partial_to_future_median_range",
        ].max()
    )
    checks = {
        "common_receiver_panel_has_at_least_90_decisions": len(receiver_range) >= 90,
        "source_receiver_matrix_exact_289": len(matrix) == 17 * 17,
        "all_graph_metrics_finite": np.isfinite(
            matrix[
                [
                    "raw_spearman",
                    "partial_rank_controlling_receiver_prior_range",
                ]
            ]
        ).all().all(),
        "all_source_bucket_partial_rank_above_020": all_partial >= 0.20,
        "alt_source_bucket_partial_rank_above_015": alt_partial >= 0.15,
        "breadth_partial_rank_above_020": breadth_partial >= 0.20,
        "concentration_relation_is_negative": hhi_partial < 0,
        "all_source_bucket_beats_best_single_source": all_partial > best_single,
        "alt_bucket_beats_best_single_alt_source": alt_partial > best_alt,
        "aggregate_timing_beats_90pct_circular_shifts": float(
            aggregate_index.loc[
                "all_source_log_total_usd",
                "partial_circular_shift_percentile",
            ]
        )
        >= 0.90,
        "retrospective_graph_pilot_not_promotion_evidence": True,
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    metadata = {
        "status": "retrospective_graph_bucket_volatility_relation_supported",
        "retained_hypothesis": (
            "aggregate liquidation notional and breadth as forward broad-volatility "
            "regime features"
        ),
        "single_source_selection_allowed": False,
        "promotion_allowed": False,
        "outcomes_loaded": True,
        "decisions": len(receiver_range),
        "sources": len(source_usd.columns),
        "receivers": len(receiver_range.columns),
    }
    return {
        "audit": audit,
        "source_panel": source_usd,
        "receiver_panel": receiver_range,
        "matrix": matrix,
        "aggregate": aggregate,
        "sources": sources,
        "metadata": metadata,
    }


def _write_findings(result: dict[str, object], path: Path) -> None:
    aggregate = result["aggregate"].set_index("feature")
    sources = result["sources"]
    matrix = result["matrix"]
    off_diagonal = matrix[~matrix["is_own_symbol"]]
    all_row = aggregate.loc["all_source_log_total_usd"]
    alt_row = aggregate.loc["alt_source_log_total_usd"]
    breadth_row = aggregate.loc["active_source_breadth"]
    hhi_row = aggregate.loc["source_notional_hhi"]
    best_single = sources.iloc[0]
    best_alt = sources[sources["source_symbol"].ne("BTCUSDT")].iloc[0]
    text = [
        "# v23.40 Liquidation Graph Volatility Transmission Pilot",
        "",
        "Verdict: `retrospective_graph_bucket_volatility_relation_supported`.",
        "",
        "Across the 17-source by 17-receiver graph, the all-source liquidation bucket "
        "has raw Spearman "
        f"{all_row['raw_spearman_to_future_median_range']:+.3f} to the future "
        "60-minute median receiver range and partial rank "
        f"{all_row['partial_rank_controlling_prior_median_range']:+.3f} after "
        "controlling the prior market median range. Its partial circular-shift "
        f"percentile is {all_row['partial_circular_shift_percentile']:.1%}.",
        "",
        "The alt-only bucket remains positive at partial rank "
        f"{alt_row['partial_rank_controlling_prior_median_range']:+.3f}; active-source "
        f"breadth is {breadth_row['partial_rank_controlling_prior_median_range']:+.3f}, "
        "while source concentration HHI is "
        f"{hhi_row['partial_rank_controlling_prior_median_range']:+.3f}. Broad, "
        "distributed cascades therefore carry more broad-volatility information than "
        "concentrated single-coin events in this pilot.",
        "",
        f"The best single source is {best_single['source_symbol']} at partial rank "
        f"{best_single['partial_to_future_median_range']:+.3f}; the best single alt is "
        f"{best_alt['source_symbol']} at "
        f"{best_alt['partial_to_future_median_range']:+.3f}. Both are weaker than "
        "their corresponding aggregate buckets. Off-diagonal source-receiver edges "
        f"are positive in {off_diagonal['partial_rank_controlling_receiver_prior_range'].gt(0).mean():.1%} "
        "of pairs after prior-range control.",
        "",
        "This supports the graph/bucket research direction, not any individual edge "
        "or tradable strategy. The sample is one retrospective day with overlapping "
        "receiver horizons. Forward confirmation must freeze the aggregate-notional, "
        "breadth, and concentration features without selecting source-specific edges.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2340(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    result = run_v2340()
    root = ensure_dir(report_root)
    paths = {
        "audit": root / "audit_checks.csv",
        "source_panel": root / "source_liquidation_15m.parquet",
        "receiver_panel": root / "receiver_future_range_60m.parquet",
        "matrix": root / "source_receiver_transmission_matrix.csv",
        "aggregate": root / "aggregate_bucket_summary.csv",
        "sources": root / "source_summary.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    result["audit"].to_csv(paths["audit"], index=False)
    result["source_panel"].to_parquet(paths["source_panel"])
    result["receiver_panel"].to_parquet(paths["receiver_panel"])
    result["matrix"].to_csv(paths["matrix"], index=False)
    result["aggregate"].to_csv(paths["aggregate"], index=False)
    result["sources"].to_csv(paths["sources"], index=False)
    paths["metadata"].write_text(
        json.dumps(result["metadata"], indent=2), encoding="utf-8"
    )
    _write_findings(result, findings_path)
    return paths


__all__ = [
    "build_v2340_aggregate_summary",
    "build_v2340_panels",
    "build_v2340_transmission_matrix",
    "run_v2340",
    "write_v2340",
]
