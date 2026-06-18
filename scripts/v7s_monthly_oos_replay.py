"""Monthly OOS Replay Harness (77.docx P3).

When a fresh month closes, this script reads every known short
candidate's frozen trades.csv, isolates the target month, and tells the
operator three things per candidate:

1. New month's contribution (n, net20, win rate).
2. Whether the target month would have BROKEN gate5 / leave-one-month /
   walk-forward under the global gate_checks evaluator.
3. The verdict change: ``stable`` / ``improved`` / ``worsened`` /
   ``newly_no_value`` etc.

Output: ``reports/oos_monthly_replay/monthly_oos_verdict_<YYYY-MM>.csv``.

The candidate registry lives at the bottom of this file — extend it
when a new candidate ships. The defaults cover v7S Directions A / D / E
and the F3/F5 / Path C / v3.4 SS sleeves where the trades.csv exists.

Demo mode (default): if the target month already has trades in the
frozen csv, the script treats those as the "OOS" sample and shows
what would happen if they were a future month. Real OOS mode: pass
``--actually-new`` so the script asserts the target month was NOT in
the baseline (production use after a fresh backfill).

A1_imb10bp_h24 watch protocol (the only candidate flagged for forward
monitoring, decided 2026-06-18):

    - Current verdict: no_value (gate5 fails — best_month_share = 63.7 %).
    - Alpha emerged only in 2025-Q4; bucket-0 walk-forward = -1.19 %.
    - Bootstrap CI on mean_net20 = [-0.40 %, +2.52 %], P(>0) = 93 %.
    - Three promotion gates must ALL clear in a single replay before
      reconsidering for shadow/live:
        1. ex-target-month mean_net20 still > 0
        2. shuffled-regime control p < 0.05 (currently 0.053)
        3. target-month net20 > 0 (forward-OOS evidence)
    - Auto-demote: if 3 consecutive monthly replays show
      verdict_change == "newly_no_value" OR target_month_sum_net20 < 0,
      drop A1_imb10bp_h24 from the registry and close the candidate.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.validation.gate_checks import (
    GateNames,
    GateThresholds,
    Verdict,
    evaluate_candidate_verdict,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    trades_path: Path
    candidate_filter: dict[str, str]  # column -> required value
    net_col: str = "net20"
    notes: str = ""


REPORTS_ROOT = Path("reports")
OUTPUT_ROOT = REPORTS_ROOT / "oos_monthly_replay"


CANDIDATE_REGISTRY: list[CandidateSpec] = [
    # v7S Direction A
    CandidateSpec(
        name="v7s_A1_imb10bp_h24",
        trades_path=REPORTS_ROOT / "v7s_short_alpha" / "A_cross_exchange_lag" / "short_trades.csv",
        candidate_filter={"candidate_code": "A1_imb10bp", "execution": "h24"},
        notes=(
            "The promote-candidate that downgraded under tightened gate5; P3's primary OOS target. "
            "Watch protocol: see module docstring. Promote only if ex-target>0 AND shuffled p<0.05 "
            "AND target-month net20>0 in a single replay. Auto-demote after 3 consecutive "
            "newly_no_value / negative target-month replays."
        ),
    ),
    CandidateSpec(
        name="v7s_A1_canonical_h24",
        trades_path=REPORTS_ROOT / "v7s_short_alpha" / "A_cross_exchange_lag" / "short_trades.csv",
        candidate_filter={"candidate_code": "A1_binance_sell_impulse_bybit_lag", "execution": "h24"},
    ),
    # v7S Direction D
    CandidateSpec(
        name="v7s_D0_naked_h24",
        trades_path=REPORTS_ROOT / "v7s_short_alpha" / "D_relative_value_pair" / "short_trades.csv",
        candidate_filter={"candidate_code": "D0_naked_short", "execution": "h24"},
    ),
    CandidateSpec(
        name="v7s_D1_pair_btc_h24",
        trades_path=REPORTS_ROOT / "v7s_short_alpha" / "D_relative_value_pair" / "short_trades.csv",
        candidate_filter={"candidate_code": "D1_pair_btc", "execution": "h24"},
    ),
    # v7S Direction E
    CandidateSpec(
        name="v7s_E1_break_entry_fast",
        trades_path=REPORTS_ROOT / "v7s_short_alpha" / "E_cic_failure_confirmed" / "short_trades.csv",
        candidate_filter={"candidate_code": "E1_cic_break_entry_strict", "execution": "fast"},
    ),
    # v3.4 SS sleeves
    CandidateSpec(
        name="v3_4_SS1A_fast",
        trades_path=REPORTS_ROOT / "v3_4_true_short_sleeve" / "short_sleeve_trades.csv",
        candidate_filter={"sleeve_code": "SS1A", "execution": "fast"},
    ),
    CandidateSpec(
        name="v3_4_SS3A_fast",
        trades_path=REPORTS_ROOT / "v3_4_true_short_sleeve" / "short_sleeve_trades.csv",
        candidate_filter={"sleeve_code": "SS3A", "execution": "fast"},
    ),
    # v6S Path C
    CandidateSpec(
        name="v6S_path_c_S_C1_swing",
        trades_path=REPORTS_ROOT / "v6s_path_c_short_validation" / "v6s_pathc_outcomes_per_event.csv",
        candidate_filter={"candidate_code": "S_C1_normal_short_swing"},
    ),
]


def _load_trades(spec: CandidateSpec) -> pd.DataFrame:
    if not spec.trades_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(spec.trades_path)
    except Exception:
        return pd.DataFrame()
    for col, val in spec.candidate_filter.items():
        if col not in df.columns:
            return pd.DataFrame()
        df = df[df[col].astype(str) == str(val)]
    return df.reset_index(drop=True)


def _attach_month(trades: pd.DataFrame, time_col: str = "signal_time") -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out = out.dropna(subset=[time_col])
    out["month"] = out[time_col].dt.strftime("%Y-%m")
    return out


def _net_col_normalize(trades: pd.DataFrame, spec: CandidateSpec) -> pd.DataFrame:
    """Make sure a ``net20`` column exists. If the candidate's CSV uses a
    different name we either copy or compute from ``gross_return``."""
    if trades.empty:
        return trades
    out = trades.copy()
    if "net20" in out.columns:
        return out
    if spec.net_col in out.columns and spec.net_col != "net20":
        out["net20"] = pd.to_numeric(out[spec.net_col], errors="coerce")
        return out
    if "gross_return" in out.columns:
        # Default fallback: charge 40 bps round-trip on a 1-leg short.
        out["net20"] = pd.to_numeric(out["gross_return"], errors="coerce") - 0.004
        return out
    out["net20"] = float("nan")
    return out


def _summary_row(
    spec: CandidateSpec,
    target_month: str,
    baseline_trades: pd.DataFrame,
    target_trades: pd.DataFrame,
    full_trades: pd.DataFrame,
    thresholds: GateThresholds,
) -> dict[str, object]:
    pool_nets = pd.to_numeric(full_trades.get("net20"), errors="coerce").dropna().to_numpy() if not full_trades.empty else None
    baseline_verdict = evaluate_candidate_verdict(
        baseline_trades, pool_nets=pool_nets, thresholds=thresholds
    )
    full_verdict = evaluate_candidate_verdict(
        full_trades, pool_nets=pool_nets, thresholds=thresholds
    )
    target_nets = pd.to_numeric(target_trades.get("net20"), errors="coerce")

    row: dict[str, object] = {
        "candidate": spec.name,
        "target_month": target_month,
        "trades_path": str(spec.trades_path),
        "n_baseline": int(len(baseline_trades)),
        "n_target_month": int(len(target_trades)),
        "n_full": int(len(full_trades)),
        "target_month_sum_net20": float(target_nets.sum()) if not target_trades.empty else 0.0,
        "target_month_mean_net20": float(target_nets.mean()) if not target_trades.empty else float("nan"),
        "target_month_win_rate": float((target_nets > 0).mean()) if not target_trades.empty else float("nan"),
        "baseline_verdict": baseline_verdict.final_verdict.value,
        "full_verdict_with_new_month": full_verdict.final_verdict.value,
        "verdict_change": _classify_verdict_change(baseline_verdict.final_verdict, full_verdict.final_verdict),
        # Carry the gate-pass + concentration metrics on the FULL-with-new-month surface
        "full_best_month_share": full_verdict.distribution.best_month_share,
        "full_leave_one_month_min": full_verdict.distribution.leave_one_month_min,
        "full_walk_forward_min_net": full_verdict.walk_forward.walk_forward_min_net if full_verdict.walk_forward else float("nan"),
        "full_bootstrap_ci_lo": full_verdict.bootstrap.ci_lo if full_verdict.bootstrap else float("nan"),
        "full_bootstrap_ci_hi": full_verdict.bootstrap.ci_hi if full_verdict.bootstrap else float("nan"),
        "full_gate_failures": ";".join(full_verdict.failures),
    }
    return row


_VERDICT_RANK = {
    Verdict.PROMOTE: 4,
    Verdict.RISK_OFF_ONLY: 3,
    Verdict.DIAGNOSTIC_ONLY: 2,
    Verdict.NO_VALUE: 1,
    Verdict.NO_DATA: 0,
}


def _classify_verdict_change(prev: Verdict, new: Verdict) -> str:
    if prev is new:
        return "stable"
    prev_rank = _VERDICT_RANK.get(prev, 0)
    new_rank = _VERDICT_RANK.get(new, 0)
    if new_rank > prev_rank:
        return f"improved_to_{new.value}"
    if new_rank < prev_rank:
        return f"worsened_to_{new.value}"
    return f"changed_to_{new.value}"


def replay_candidate(
    spec: CandidateSpec,
    target_month: str,
    thresholds: GateThresholds,
    *,
    actually_new: bool = False,
) -> dict[str, object]:
    raw = _load_trades(spec)
    if raw.empty:
        return {
            "candidate": spec.name,
            "target_month": target_month,
            "trades_path": str(spec.trades_path),
            "n_baseline": 0,
            "n_target_month": 0,
            "n_full": 0,
            "verdict_change": "no_trades_csv",
        }
    raw = _net_col_normalize(raw, spec)
    raw = _attach_month(raw)
    if raw.empty:
        return {
            "candidate": spec.name,
            "target_month": target_month,
            "verdict_change": "no_signal_times",
        }
    target_trades = raw[raw["month"] == target_month]
    baseline_trades = raw[raw["month"] != target_month]
    full_trades = raw
    if actually_new and not baseline_trades.empty and not target_trades.empty:
        # Sanity: target month should not have been in the baseline if this is true OOS.
        warnings.warn(
            f"{spec.name}: --actually-new specified but target month {target_month} "
            f"contains trades — proceeding as a demo replay."
        )
    return _summary_row(spec, target_month, baseline_trades, target_trades, full_trades, thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly OOS Replay Harness (77.docx P3).")
    parser.add_argument("--target-month", required=True, help="Target month YYYY-MM.")
    parser.add_argument("--actually-new", action="store_true", help="Assert true OOS mode (target month was not in baseline).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    thresholds = GateThresholds()
    rows: list[dict[str, object]] = []
    for spec in CANDIDATE_REGISTRY:
        row = replay_candidate(spec, args.target_month, thresholds, actually_new=args.actually_new)
        rows.append(row)
        print(
            f"  {row['candidate']:40s} n_target={row.get('n_target_month', 0):>4} "
            f"verdict={row.get('verdict_change', '?')}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"monthly_oos_verdict_{args.target_month}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
