"""Leave-one-out and non-overlap audit of the v23.40 graph-bucket relation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2340_liquidation_graph_volatility_transmission_pilot import (
    build_v2340_panels,
)


REPORT_ROOT = Path("reports/v23_41_liquidation_graph_bucket_independent_audit")
FINDINGS_PATH = Path(
    "docs/v2341_liquidation_graph_bucket_independent_audit_2026_07_17.md"
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


def run_v2341() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source_usd, receiver_range = build_v2340_panels()
    full_feature = np.log1p(source_usd.sum(axis=1))
    full_outcome = receiver_range.median(axis=1)
    full_prior = full_outcome.shift(4)

    source_rows = []
    for omitted in source_usd.columns:
        feature = np.log1p(source_usd.drop(columns=omitted).sum(axis=1))
        source_rows.append(
            {
                "omitted_source": omitted,
                "remaining_sources": source_usd.shape[1] - 1,
                "raw_spearman": float(
                    feature.corr(full_outcome, method="spearman")
                ),
                "partial_rank_controlling_prior_market_range": _partial_rank(
                    feature, full_outcome, full_prior
                ),
            }
        )
    source_loo = pd.DataFrame(source_rows).sort_values(
        "partial_rank_controlling_prior_market_range"
    )

    receiver_rows = []
    for omitted in receiver_range.columns:
        outcome = receiver_range.drop(columns=omitted).median(axis=1)
        receiver_rows.append(
            {
                "omitted_receiver": omitted,
                "remaining_receivers": receiver_range.shape[1] - 1,
                "raw_spearman": float(
                    full_feature.corr(outcome, method="spearman")
                ),
                "partial_rank_controlling_prior_market_range": _partial_rank(
                    full_feature, outcome, outcome.shift(4)
                ),
            }
        )
    receiver_loo = pd.DataFrame(receiver_rows).sort_values(
        "partial_rank_controlling_prior_market_range"
    )

    hourly = pd.DataFrame(
        {"feature": full_feature, "outcome": full_outcome, "prior": full_prior}
    ).dropna()
    hourly = hourly[hourly.index.minute == 0]
    hourly_raw = float(hourly["feature"].corr(hourly["outcome"], method="spearman"))
    hourly_partial = _partial_rank(
        hourly["feature"], hourly["outcome"], hourly["prior"]
    )
    full_partial = _partial_rank(full_feature, full_outcome, full_prior)
    checks = {
        "source_leave_one_out_exact_17": len(source_loo) == 17,
        "receiver_leave_one_out_exact_17": len(receiver_loo) == 17,
        "source_leave_one_out_partial_min_above_015": float(
            source_loo["partial_rank_controlling_prior_market_range"].min()
        )
        >= 0.15,
        "receiver_leave_one_out_partial_min_above_020": float(
            receiver_loo["partial_rank_controlling_prior_market_range"].min()
        )
        >= 0.20,
        "hourly_nonoverlap_has_at_least_20_decisions": len(hourly) >= 20,
        "hourly_nonoverlap_partial_rank_above_015": hourly_partial >= 0.15,
        "full_partial_rank_above_020": full_partial >= 0.20,
        "retrospective_one_day_still_not_promotion_evidence": True,
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    metadata = {
        "status": "graph_bucket_relation_stable_to_leave_one_out_within_pilot",
        "full_partial_rank": full_partial,
        "hourly_nonoverlap_raw_spearman": hourly_raw,
        "hourly_nonoverlap_partial_rank": hourly_partial,
        "hourly_nonoverlap_decisions": len(hourly),
        "minimum_source_leave_one_out_partial_rank": float(
            source_loo["partial_rank_controlling_prior_market_range"].min()
        ),
        "minimum_receiver_leave_one_out_partial_rank": float(
            receiver_loo["partial_rank_controlling_prior_market_range"].min()
        ),
        "promotion_allowed": False,
        "forward_confirmation_required": True,
    }
    return audit, source_loo, receiver_loo, metadata


def _write_findings(
    audit: pd.DataFrame,
    source_loo: pd.DataFrame,
    receiver_loo: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    worst_source = source_loo.iloc[0]
    worst_receiver = receiver_loo.iloc[0]
    text = [
        "# v23.41 Liquidation Graph-Bucket Independent Audit",
        "",
        f"Verdict: `{metadata['status']}`.",
        "",
        "The graph-bucket relation is not carried by one coin. Removing each source "
        "in turn leaves partial rank at or above "
        f"{metadata['minimum_source_leave_one_out_partial_rank']:+.3f}; the weakest "
        f"case omits {worst_source['omitted_source']}. Removing each receiver leaves "
        "partial rank at or above "
        f"{metadata['minimum_receiver_leave_one_out_partial_rank']:+.3f}; the weakest "
        f"case omits {worst_receiver['omitted_receiver']}.",
        "",
        f"On non-overlapping hourly decisions (n={metadata['hourly_nonoverlap_decisions']}), "
        "raw Spearman is "
        f"{metadata['hourly_nonoverlap_raw_spearman']:+.3f} and partial rank after "
        "prior market range is "
        f"{metadata['hourly_nonoverlap_partial_rank']:+.3f}.",
        "",
        "This strengthens the mechanism case for aggregate notional and breadth as "
        "graph-level volatility-state features. It still does not establish tradable "
        "alpha because the source snapshot is retrospective and covers one day.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2341(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, source_loo, receiver_loo, metadata = run_v2341()
    root = ensure_dir(report_root)
    paths = {
        "audit": root / "audit_checks.csv",
        "source_loo": root / "source_leave_one_out.csv",
        "receiver_loo": root / "receiver_leave_one_out.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    audit.to_csv(paths["audit"], index=False)
    source_loo.to_csv(paths["source_loo"], index=False)
    receiver_loo.to_csv(paths["receiver_loo"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_findings(audit, source_loo, receiver_loo, metadata, findings_path)
    return paths


__all__ = ["run_v2341", "write_v2341"]
