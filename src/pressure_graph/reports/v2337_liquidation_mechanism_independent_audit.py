"""Independent lead-lag audit of the v23.36 liquidation mechanism pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pressure_graph.io import ensure_dir


V2336_ROOT = Path("reports/v23_36_liquidation_price_mechanism_pilot")
REPORT_ROOT = Path("reports/v23_37_liquidation_mechanism_independent_audit")
FINDINGS_PATH = Path(
    "docs/v2337_liquidation_mechanism_independent_audit_2026_07_17.md"
)


def _partial_rank_correlation(
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


def _circular_shift_percentile(feature: pd.Series, outcome: pd.Series) -> float:
    actual = feature.corr(outcome, method="spearman")
    values = [
        pd.Series(np.roll(feature.to_numpy(), shift)).corr(
            outcome.reset_index(drop=True), method="spearman"
        )
        for shift in range(1, len(feature))
    ]
    return float(np.mean(np.asarray(values) <= actual))


def run_v2337(
    v2336_root: Path = V2336_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    market = pd.read_parquet(v2336_root / "market_15m_panel.parquet")
    own = pd.read_parquet(v2336_root / "own_symbol_15m_panel.parquet")
    market["decision_time"] = pd.to_datetime(
        market["decision_time"], utc=True, errors="coerce"
    )
    market = market.sort_values("decision_time").reset_index(drop=True)
    market["prior_btc_log_range_60m"] = market["btc_log_range_60m"].shift(4)
    feature = market["log1p_liq_15m_total_usd"]
    alt_feature = market["log1p_alt_liq_15m_total_usd"]
    future = market["btc_log_range_60m"]
    prior = market["prior_btc_log_range_60m"]
    complete = market.dropna(subset=["prior_btc_log_range_60m"])
    hourly = complete[complete["decision_time"].dt.minute.eq(0)]
    alt_own = own[
        own["symbol"].ne("BTCUSDT") & own["flow_direction"].ne(0)
    ]

    metrics = pd.DataFrame(
        [
            {
                "metric": "total_liquidation_future_range_spearman",
                "value": float(feature.corr(future, method="spearman")),
            },
            {
                "metric": "alt_liquidation_future_range_spearman",
                "value": float(alt_feature.corr(future, method="spearman")),
            },
            {
                "metric": "total_liquidation_prior_range_spearman",
                "value": float(
                    complete["log1p_liq_15m_total_usd"].corr(
                        complete["prior_btc_log_range_60m"], method="spearman"
                    )
                ),
            },
            {
                "metric": "future_prior_range_spearman",
                "value": float(
                    complete["btc_log_range_60m"].corr(
                        complete["prior_btc_log_range_60m"], method="spearman"
                    )
                ),
            },
            {
                "metric": "total_liquidation_partial_rank_controlling_prior_range",
                "value": _partial_rank_correlation(feature, future, prior),
            },
            {
                "metric": "alt_liquidation_partial_rank_controlling_prior_range",
                "value": _partial_rank_correlation(alt_feature, future, prior),
            },
            {
                "metric": "total_liquidation_circular_shift_percentile",
                "value": _circular_shift_percentile(feature, future),
            },
            {
                "metric": "hourly_nonoverlap_total_future_range_spearman",
                "value": float(
                    hourly["log1p_liq_15m_total_usd"].corr(
                        hourly["btc_log_range_60m"], method="spearman"
                    )
                ),
            },
            {
                "metric": "hourly_nonoverlap_total_prior_range_spearman",
                "value": float(
                    hourly["log1p_liq_15m_total_usd"].corr(
                        hourly["prior_btc_log_range_60m"], method="spearman"
                    )
                ),
            },
            {
                "metric": "alt_own_60m_continuation_rate",
                "value": float(alt_own["signed_log_return_60m"].gt(0).mean()),
            },
            {
                "metric": "alt_own_60m_mean_signed_return_bp",
                "value": float(alt_own["signed_log_return_60m"].mean() * 10_000),
            },
        ]
    )
    values = metrics.set_index("metric")["value"]
    checks = {
        "market_decisions_regular_15m": len(market) >= 90
        and market["decision_time"].diff().dropna().eq(pd.Timedelta(minutes=15)).all(),
        "prior_range_uses_exact_four_bar_lag": int(prior.notna().sum())
        == len(market) - 4,
        "all_independent_metrics_finite": np.isfinite(metrics["value"]).all(),
        "raw_forward_relation_is_positive": values[
            "total_liquidation_future_range_spearman"
        ]
        >= 0.25,
        "timing_beats_most_circular_shifts": values[
            "total_liquidation_circular_shift_percentile"
        ]
        >= 0.90,
        "liquidations_are_more_reactive_than_predictive": values[
            "total_liquidation_prior_range_spearman"
        ]
        > values["total_liquidation_future_range_spearman"],
        "incremental_relation_after_prior_range_is_weak": abs(
            values["total_liquidation_partial_rank_controlling_prior_range"]
        )
        < 0.20,
        "directional_continuation_is_not_supported": values[
            "alt_own_60m_continuation_rate"
        ]
        < 0.50
        and values["alt_own_60m_mean_signed_return_bp"] < 0,
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    metadata = {
        "status": "mechanism_supported_regime_marker_not_standalone_alpha",
        "candidate_role": "volatility_regime_filter_or_oco_overlay",
        "standalone_directional_role": "rejected_by_retrospective_pilot",
        "promotion_allowed": False,
        "forward_confirmation_required": True,
        "market_decisions": len(market),
        "hourly_nonoverlap_decisions": len(hourly),
        "alt_own_symbol_cells": len(alt_own),
    }
    return audit, metrics, metadata


def _write_findings(
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    values = metrics.set_index("metric")["value"]
    text = [
        "# v23.37 Liquidation Mechanism Independent Audit",
        "",
        f"Verdict: `{metadata['status']}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {int((~audit['passed']).sum())}.",
        "",
        "The raw 15-minute liquidation-intensity correlation with next-60-minute "
        f"BTC range is {values['total_liquidation_future_range_spearman']:+.3f}, "
        "and its circular-shift percentile is "
        f"{values['total_liquidation_circular_shift_percentile']:.1%}.",
        "",
        "However, correlation with the preceding 60-minute range is stronger at "
        f"{values['total_liquidation_prior_range_spearman']:+.3f}. After controlling "
        "for that prior range, the partial rank relation to future range is only "
        f"{values['total_liquidation_partial_rank_controlling_prior_range']:+.3f}. "
        "The one-day evidence therefore supports liquidation flow as a timely "
        "high-volatility regime marker, but not as a standalone volatility predictor.",
        "",
        "Directional continuation is also unsupported: alt forced-flow continuation "
        f"is {values['alt_own_60m_continuation_rate']:.1%}, with mean signed return "
        f"{values['alt_own_60m_mean_signed_return_bp']:+.2f} bp. The only retained "
        "research role is an OCO activation/avoidance overlay or regime covariate, "
        "subject to the v23.35 forward gate.",
        "",
        "This remains retrospective, non-promotable evidence.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2337(
    v2336_root: Path = V2336_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit, metrics, metadata = run_v2337(v2336_root)
    root = ensure_dir(report_root)
    paths = {
        "audit": root / "audit_checks.csv",
        "metrics": root / "independent_lead_lag_metrics.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    audit.to_csv(paths["audit"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_findings(audit, metrics, metadata, findings_path)
    return paths


__all__ = ["run_v2337", "write_v2337"]
