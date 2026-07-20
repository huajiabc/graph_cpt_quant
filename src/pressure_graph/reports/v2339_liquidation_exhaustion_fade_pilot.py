"""Retrospective fade test after liquidation-conditioned BTC barrier touches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2338_liquidation_filtered_oco_execution_pilot import (
    BTC_PATH,
    V2336_ROOT,
    V2338Config,
    build_v2338_causal_sigma,
    load_v2338_inputs,
)


REPORT_ROOT = Path("reports/v23_39_liquidation_exhaustion_fade_pilot")
FINDINGS_PATH = Path(
    "docs/v2339_liquidation_exhaustion_fade_pilot_2026_07_17.md"
)


@dataclass(frozen=True)
class V2339Config:
    sigma_multiple: float = 0.625
    horizons_minutes: tuple[int, ...] = (60, 240)
    optimistic_cost: float = 0.0005
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020


def simulate_v2339_fades(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V2339Config = V2339Config(),
) -> pd.DataFrame:
    sigma = build_v2338_causal_sigma(bars, V2338Config())
    ready = pd.merge_asof(
        features.sort_values("decision_time"),
        sigma[["sigma_time", "causal_hourly_sigma"]],
        left_on="decision_time",
        right_on="sigma_time",
        direction="backward",
        allow_exact_matches=True,
    )
    total_q25 = float(ready["liq_15m_total_usd"].quantile(0.25))
    total_q75 = float(ready["liq_15m_total_usd"].quantile(0.75))
    alt_q75 = float(ready["alt_liq_15m_total_usd"].quantile(0.75))
    indexed = bars.set_index("bar_open_time").sort_index()
    rows = []
    for feature in ready.itertuples(index=False):
        entry = pd.Timestamp(feature.decision_time)
        if not np.isfinite(feature.causal_hourly_sigma) or entry not in indexed.index:
            continue
        spot = float(indexed.loc[entry, "open"])
        width = cfg.sigma_multiple * float(feature.causal_hourly_sigma)
        upper = spot * np.exp(width)
        lower = spot * np.exp(-width)
        for horizon in cfg.horizons_minutes:
            count = horizon // 15
            times = [entry + pd.Timedelta(minutes=15 * offset) for offset in range(count)]
            if any(time not in indexed.index for time in times):
                continue
            path = indexed.loc[times]
            exit_spot = float(path.iloc[-1]["close"])
            triggered = False
            ambiguous = False
            breakout_gross = 0.0
            fade_gross = 0.0
            trigger_delay = np.nan
            for time, bar in path.iterrows():
                upper_hit = float(bar["high"]) >= upper
                lower_hit = float(bar["low"]) <= lower
                if not upper_hit and not lower_hit:
                    continue
                triggered = True
                trigger_delay = (pd.Timestamp(time) - entry).total_seconds() / 60.0
                long_fill = max(upper, float(bar["open"]))
                short_fill = min(lower, float(bar["open"]))
                long_breakout = exit_spot / long_fill - 1.0
                short_breakout = 1.0 - exit_spot / short_fill
                upper_fade = 1.0 - exit_spot / long_fill
                lower_fade = exit_spot / short_fill - 1.0
                if upper_hit and lower_hit:
                    ambiguous = True
                    breakout_gross = min(long_breakout, short_breakout)
                    fade_gross = min(upper_fade, lower_fade)
                elif upper_hit:
                    breakout_gross = long_breakout
                    fade_gross = upper_fade
                else:
                    breakout_gross = short_breakout
                    fade_gross = lower_fade
                break
            row = feature._asdict()
            row.update(
                {
                    "horizon_minutes": horizon,
                    "entry_spot": spot,
                    "upper_stop_price": upper,
                    "lower_stop_price": lower,
                    "exit_spot": exit_spot,
                    "triggered": triggered,
                    "ambiguous_trigger": ambiguous,
                    "trigger_delay_minutes": trigger_delay,
                    "breakout_gross_return": breakout_gross,
                    "fade_gross_return": fade_gross,
                    "fade_optimistic_net_return": fade_gross
                    - (cfg.optimistic_cost if triggered else 0.0),
                    "fade_primary_net_return": fade_gross
                    - (cfg.primary_cost if triggered else 0.0),
                    "fade_stress_net_return": fade_gross
                    - (cfg.stress_cost if triggered else 0.0),
                    "total_liquidation_quartile": (
                        "bottom"
                        if feature.liq_15m_total_usd <= total_q25
                        else "top"
                        if feature.liq_15m_total_usd >= total_q75
                        else "middle"
                    ),
                    "alt_top_quartile": feature.alt_liq_15m_total_usd >= alt_q75,
                    "total_q25": total_q25,
                    "total_q75": total_q75,
                    "alt_q75": alt_q75,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["horizon_minutes", "decision_time"]
    ).reset_index(drop=True)


def summarize_v2339(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in sorted(outcomes["horizon_minutes"].unique()):
        horizon_data = outcomes[outcomes["horizon_minutes"].eq(horizon)]
        scopes = (
            ("all", horizon_data),
            (
                "total_top_quartile",
                horizon_data[
                    horizon_data["total_liquidation_quartile"].eq("top")
                ],
            ),
            (
                "total_bottom_quartile",
                horizon_data[
                    horizon_data["total_liquidation_quartile"].eq("bottom")
                ],
            ),
            ("alt_top_quartile", horizon_data[horizon_data["alt_top_quartile"]]),
        )
        for scope, local in scopes:
            triggered = local[local["triggered"]]
            rows.append(
                {
                    "horizon_minutes": horizon,
                    "scope": scope,
                    "decisions": len(local),
                    "triggered_trades": len(triggered),
                    "trigger_rate": float(local["triggered"].mean()),
                    "mean_fade_gross_bp_per_decision": float(
                        local["fade_gross_return"].mean() * 10_000
                    ),
                    "mean_fade_optimistic_bp_per_decision": float(
                        local["fade_optimistic_net_return"].mean() * 10_000
                    ),
                    "mean_fade_primary_bp_per_decision": float(
                        local["fade_primary_net_return"].mean() * 10_000
                    ),
                    "mean_fade_stress_bp_per_decision": float(
                        local["fade_stress_net_return"].mean() * 10_000
                    ),
                    "mean_fade_primary_bp_per_trade": float(
                        triggered["fade_primary_net_return"].mean() * 10_000
                    ),
                    "positive_primary_decision_rate": float(
                        local["fade_primary_net_return"].gt(0).mean()
                    ),
                    "ambiguous_trades": int(local["ambiguous_trigger"].sum()),
                }
            )
    return pd.DataFrame(rows)


def run_v2339(
    v2336_root: Path = V2336_ROOT,
    btc_path: Path = BTC_PATH,
    cfg: V2339Config = V2339Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features, bars = load_v2338_inputs(v2336_root, btc_path, V2338Config())
    outcomes = simulate_v2339_fades(features, bars, cfg)
    summary = summarize_v2339(outcomes)
    indexed = summary.set_index(["horizon_minutes", "scope"])
    top_60 = indexed.loc[(60, "total_top_quartile")]
    top_240 = indexed.loc[(240, "total_top_quartile")]
    checks = {
        "both_horizons_have_at_least_75_decisions": all(
            len(outcomes[outcomes["horizon_minutes"].eq(horizon)]) >= 75
            for horizon in cfg.horizons_minutes
        ),
        "quartile_groups_have_at_least_18_decisions": int(
            top_60["decisions"]
        )
        >= 18
        and int(top_240["decisions"]) >= 18,
        "one_hour_top_quartile_fade_primary_positive": top_60[
            "mean_fade_primary_bp_per_decision"
        ]
        > 0,
        "four_hour_top_quartile_fade_primary_positive": top_240[
            "mean_fade_primary_bp_per_decision"
        ]
        > 0,
        "four_hour_top_quartile_fade_stress_positive": top_240[
            "mean_fade_stress_bp_per_decision"
        ]
        > 0,
        "retrospective_threshold_not_promotion_evidence": True,
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    if checks["four_hour_top_quartile_fade_stress_positive"]:
        verdict = "retrospective_fade_candidate_requires_forward_confirmation"
    elif checks["four_hour_top_quartile_fade_primary_positive"]:
        verdict = "fade_only_primary_cost_positive_not_stress_robust"
    elif top_240["mean_fade_optimistic_bp_per_decision"] > 0:
        verdict = "fade_only_optimistic_cost_positive"
    else:
        verdict = "liquidation_exhaustion_fade_not_economic"
    metadata = {
        "status": verdict,
        "sigma_multiple": cfg.sigma_multiple,
        "horizons_minutes": list(cfg.horizons_minutes),
        "optimistic_cost_bp": cfg.optimistic_cost * 10_000,
        "primary_cost_bp": cfg.primary_cost * 10_000,
        "stress_cost_bp": cfg.stress_cost * 10_000,
        "promotion_allowed": False,
        "outcomes_loaded": True,
    }
    return audit, outcomes, summary, metadata


def _write_findings(
    summary: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    indexed = summary.set_index(["horizon_minutes", "scope"])
    top_60 = indexed.loc[(60, "total_top_quartile")]
    top_240 = indexed.loc[(240, "total_top_quartile")]
    alt_240 = indexed.loc[(240, "alt_top_quartile")]
    text = [
        "# v23.39 Liquidation Exhaustion Fade Pilot",
        "",
        f"Verdict: `{metadata['status']}`.",
        "",
        "After a 0.625-sigma barrier touch in the top total-liquidation quartile, "
        f"the one-hour fade earns {top_60['mean_fade_primary_bp_per_decision']:+.2f} "
        "bp/decision at 10 bp cost and "
        f"{top_60['mean_fade_stress_bp_per_decision']:+.2f} bp at 20 bp cost.",
        "",
        "At four hours, the same total-liquidation fade earns "
        f"{top_240['mean_fade_optimistic_bp_per_decision']:+.2f} bp at 5 bp cost, "
        f"{top_240['mean_fade_primary_bp_per_decision']:+.2f} bp at 10 bp cost, and "
        f"{top_240['mean_fade_stress_bp_per_decision']:+.2f} bp at 20 bp cost. "
        "The alt-only top-quartile four-hour primary result is "
        f"{alt_240['mean_fade_primary_bp_per_decision']:+.2f} bp/decision.",
        "",
        "This remains a retrospectively ranked one-day pilot with overlapping paths. "
        "It cannot be promoted or levered. Only a positive, cost-stressed forward "
        "result under the v23.35 knowledge-time contract could retain the fade idea.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2339(
    v2336_root: Path = V2336_ROOT,
    btc_path: Path = BTC_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V2339Config = V2339Config(),
) -> dict[str, Path]:
    audit, outcomes, summary, metadata = run_v2339(v2336_root, btc_path, cfg)
    root = ensure_dir(report_root)
    paths = {
        "audit": root / "audit_checks.csv",
        "outcomes": root / "fade_outcomes.parquet",
        "summary": root / "fade_summary.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    audit.to_csv(paths["audit"], index=False)
    outcomes.to_parquet(paths["outcomes"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_findings(summary, metadata, findings_path)
    return paths


__all__ = ["V2339Config", "run_v2339", "simulate_v2339_fades", "write_v2339"]
