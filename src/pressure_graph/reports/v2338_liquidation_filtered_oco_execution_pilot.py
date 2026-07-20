"""Retrospective execution pilot for liquidation-filtered BTC OCO breakouts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


V2336_ROOT = Path("reports/v23_36_liquidation_price_mechanism_pilot")
BTC_PATH = Path(
    "data/external/okx_liquidation_forward/oco_context/"
    "bybit_klines_15m/BTCUSDT.parquet"
)
REPORT_ROOT = Path("reports/v23_38_liquidation_filtered_oco_execution_pilot")
FINDINGS_PATH = Path(
    "docs/v2338_liquidation_filtered_oco_execution_pilot_2026_07_17.md"
)


@dataclass(frozen=True)
class V2338Config:
    sigma_multiple: float = 0.625
    sigma_hours: int = 24
    path_bars: int = 4
    primary_cost: float = 0.0010
    stress_cost: float = 0.0020
    closed_price_end: pd.Timestamp = pd.Timestamp("2026-07-17T04:30:00Z")


def load_v2338_inputs(
    v2336_root: Path = V2336_ROOT,
    btc_path: Path = BTC_PATH,
    cfg: V2338Config = V2338Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(v2336_root / "market_15m_panel.parquet")
    features["decision_time"] = pd.to_datetime(
        features["decision_time"], utc=True, errors="coerce"
    )
    bars = pd.read_parquet(btc_path)
    for column in ("bar_open_time", "bar_close_time"):
        bars[column] = pd.to_datetime(bars[column], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars[bars["bar_close_time"].le(cfg.closed_price_end)].copy()
    return features.sort_values("decision_time"), bars.sort_values("bar_open_time")


def build_v2338_causal_sigma(
    bars: pd.DataFrame,
    cfg: V2338Config = V2338Config(),
) -> pd.DataFrame:
    hourly = bars[
        bars["bar_close_time"].dt.minute.eq(0)
        & bars["bar_close_time"].dt.second.eq(0)
    ][["bar_close_time", "close"]].rename(
        columns={"bar_close_time": "sigma_time", "close": "hourly_close"}
    )
    hourly = hourly.drop_duplicates("sigma_time", keep="last").sort_values(
        "sigma_time"
    )
    hourly["hourly_log_move"] = np.log(
        hourly["hourly_close"] / hourly["hourly_close"].shift(1)
    )
    hourly["causal_hourly_sigma"] = np.sqrt(
        hourly["hourly_log_move"]
        .pow(2)
        .rolling(cfg.sigma_hours, min_periods=cfg.sigma_hours)
        .mean()
    )
    return hourly.dropna(subset=["causal_hourly_sigma"]).reset_index(drop=True)


def simulate_v2338(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: V2338Config = V2338Config(),
) -> pd.DataFrame:
    sigma = build_v2338_causal_sigma(bars, cfg)
    ready = pd.merge_asof(
        features.sort_values("decision_time"),
        sigma[["sigma_time", "causal_hourly_sigma"]],
        left_on="decision_time",
        right_on="sigma_time",
        direction="backward",
        allow_exact_matches=True,
    )
    indexed = bars.set_index("bar_open_time").sort_index()
    rows = []
    for feature in ready.itertuples(index=False):
        entry = pd.Timestamp(feature.decision_time)
        times = [entry + pd.Timedelta(minutes=15 * offset) for offset in range(cfg.path_bars)]
        if not np.isfinite(feature.causal_hourly_sigma) or any(
            time not in indexed.index for time in times
        ):
            continue
        path = indexed.loc[times]
        spot = float(path.iloc[0]["open"])
        width = cfg.sigma_multiple * float(feature.causal_hourly_sigma)
        upper = spot * np.exp(width)
        lower = spot * np.exp(-width)
        exit_spot = float(path.iloc[-1]["close"])
        triggered = False
        ambiguous = False
        direction = 0
        gross = 0.0
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
            long_gross = exit_spot / long_fill - 1.0
            short_gross = 1.0 - exit_spot / short_fill
            if upper_hit and lower_hit:
                ambiguous = True
                if long_gross <= short_gross:
                    direction, gross = 1, long_gross
                else:
                    direction, gross = -1, short_gross
            elif upper_hit:
                direction, gross = 1, long_gross
            else:
                direction, gross = -1, short_gross
            break
        row = feature._asdict()
        row.update(
            {
                "entry_spot": spot,
                "upper_stop_price": upper,
                "lower_stop_price": lower,
                "exit_spot": exit_spot,
                "triggered": triggered,
                "ambiguous_trigger": ambiguous,
                "trade_direction": direction,
                "trigger_delay_minutes": trigger_delay,
                "gross_return": gross,
                "primary_net_return": gross
                - (cfg.primary_cost if triggered else 0.0),
                "stress_net_return": gross
                - (cfg.stress_cost if triggered else 0.0),
            }
        )
        rows.append(row)
    outcomes = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    total_q25 = float(outcomes["liq_15m_total_usd"].quantile(0.25))
    total_q75 = float(outcomes["liq_15m_total_usd"].quantile(0.75))
    alt_q75 = float(outcomes["alt_liq_15m_total_usd"].quantile(0.75))
    outcomes["total_liquidation_quartile"] = np.select(
        [
            outcomes["liq_15m_total_usd"].le(total_q25),
            outcomes["liq_15m_total_usd"].ge(total_q75),
        ],
        ["bottom", "top"],
        default="middle",
    )
    outcomes["alt_top_quartile"] = outcomes["alt_liq_15m_total_usd"].ge(alt_q75)
    outcomes["total_q25"] = total_q25
    outcomes["total_q75"] = total_q75
    outcomes["alt_q75"] = alt_q75
    return outcomes


def summarize_v2338(outcomes: pd.DataFrame) -> pd.DataFrame:
    scopes = [
        ("all", outcomes),
        ("total_top_quartile", outcomes[outcomes["total_liquidation_quartile"].eq("top")]),
        (
            "total_bottom_quartile",
            outcomes[outcomes["total_liquidation_quartile"].eq("bottom")],
        ),
        ("alt_top_quartile", outcomes[outcomes["alt_top_quartile"]]),
    ]
    rows = []
    for scope, local in scopes:
        triggered = local[local["triggered"]]
        rows.append(
            {
                "scope": scope,
                "decisions": len(local),
                "triggered_trades": len(triggered),
                "trigger_rate": float(local["triggered"].mean()),
                "ambiguous_trades": int(local["ambiguous_trigger"].sum()),
                "mean_gross_bp_per_decision": float(local["gross_return"].mean() * 10_000),
                "mean_primary_bp_per_decision": float(
                    local["primary_net_return"].mean() * 10_000
                ),
                "mean_stress_bp_per_decision": float(
                    local["stress_net_return"].mean() * 10_000
                ),
                "mean_primary_bp_per_trade": float(
                    triggered["primary_net_return"].mean() * 10_000
                ),
                "positive_primary_decision_rate": float(
                    local["primary_net_return"].gt(0).mean()
                ),
                "median_trigger_delay_minutes": float(
                    triggered["trigger_delay_minutes"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_v2338(
    v2336_root: Path = V2336_ROOT,
    btc_path: Path = BTC_PATH,
    cfg: V2338Config = V2338Config(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features, bars = load_v2338_inputs(v2336_root, btc_path, cfg)
    outcomes = simulate_v2338(features, bars, cfg)
    summary = summarize_v2338(outcomes)
    indexed = summary.set_index("scope")
    top = indexed.loc["total_top_quartile"]
    bottom = indexed.loc["total_bottom_quartile"]
    checks = {
        "causal_sigma_available_for_at_least_85_decisions": len(outcomes) >= 85,
        "all_paths_have_finite_prices_and_sigma": np.isfinite(
            outcomes[
                [
                    "entry_spot",
                    "upper_stop_price",
                    "lower_stop_price",
                    "exit_spot",
                    "causal_hourly_sigma",
                ]
            ]
        ).all().all(),
        "quartile_groups_have_at_least_20_decisions": int(
            indexed.loc["total_top_quartile", "decisions"]
        )
        >= 20
        and int(indexed.loc["total_bottom_quartile", "decisions"]) >= 20,
        "high_liquidation_increases_trigger_rate": top["trigger_rate"]
        > bottom["trigger_rate"],
        "high_liquidation_primary_net_is_positive": top[
            "mean_primary_bp_per_decision"
        ]
        > 0,
        "high_liquidation_stress_net_is_positive": top[
            "mean_stress_bp_per_decision"
        ]
        > 0,
        "retrospective_quartile_not_promotion_evidence": True,
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    if bool(
        checks["high_liquidation_primary_net_is_positive"]
        and checks["high_liquidation_stress_net_is_positive"]
    ):
        verdict = "retrospective_execution_signal_forward_test_required"
    elif checks["high_liquidation_increases_trigger_rate"]:
        verdict = "higher_trigger_rate_but_not_cost_robust"
    else:
        verdict = "no_execution_value_in_retrospective_pilot"
    metadata = {
        "status": verdict,
        "sigma_multiple": cfg.sigma_multiple,
        "sigma_hours": cfg.sigma_hours,
        "path_minutes": cfg.path_bars * 15,
        "primary_cost_bp": cfg.primary_cost * 10_000,
        "stress_cost_bp": cfg.stress_cost * 10_000,
        "promotion_allowed": False,
        "outcomes_loaded": True,
    }
    return audit, outcomes, summary, metadata


def _write_findings(
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    indexed = summary.set_index("scope")
    top = indexed.loc["total_top_quartile"]
    bottom = indexed.loc["total_bottom_quartile"]
    alt = indexed.loc["alt_top_quartile"]
    text = [
        "# v23.38 Liquidation-Filtered OCO Execution Pilot",
        "",
        f"Verdict: `{metadata['status']}`.",
        "",
        "The retrospective top liquidation quartile changes BTC 0.625-sigma OCO "
        f"trigger rate from {bottom['trigger_rate']:.1%} in the bottom quartile to "
        f"{top['trigger_rate']:.1%}. Top-quartile mean net is "
        f"{top['mean_primary_bp_per_decision']:+.2f} bp/decision at 10 bp cost and "
        f"{top['mean_stress_bp_per_decision']:+.2f} bp/decision at 20 bp cost.",
        "",
        "Using alt-only liquidation intensity, the top-quartile result is "
        f"{alt['mean_primary_bp_per_decision']:+.2f} bp/decision primary and "
        f"{alt['mean_stress_bp_per_decision']:+.2f} bp/decision stress.",
        "",
        "This is a one-day, retrospectively ranked, overlapping-path mechanism test. "
        "It cannot promote a strategy. Any retained execution hypothesis must use a "
        "causal expanding threshold in the v23.35 forward ledger and must survive "
        "chronological cost-stressed evaluation.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2338(
    v2336_root: Path = V2336_ROOT,
    btc_path: Path = BTC_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V2338Config = V2338Config(),
) -> dict[str, Path]:
    audit, outcomes, summary, metadata = run_v2338(v2336_root, btc_path, cfg)
    root = ensure_dir(report_root)
    paths = {
        "audit": root / "audit_checks.csv",
        "outcomes": root / "oco_outcomes.parquet",
        "summary": root / "oco_summary.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    audit.to_csv(paths["audit"], index=False)
    outcomes.to_parquet(paths["outcomes"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_findings(audit, summary, metadata, findings_path)
    return paths


__all__ = [
    "V2338Config",
    "build_v2338_causal_sigma",
    "run_v2338",
    "simulate_v2338",
    "write_v2338",
]
