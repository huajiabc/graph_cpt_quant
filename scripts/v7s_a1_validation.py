"""Validation harness for the v7S Direction A1_imb10bp h24 PROMOTE candidate.

Runs three independent checks on the candidate's 56 events:

1. **Walk-forward over three disjoint thirds** of the event stream
   (by signal_time). For each third we report N, mean_net20, win_rate,
   so we can see whether the alpha persists across regimes.

2. **Bootstrap 95 % CI** on mean_net20 using 5000 resamples with
   replacement. The closure doc's reopen criteria are based on
   mean-level passes; a wide CI containing zero would weaken the
   `promote` verdict regardless of the gate evaluation.

3. **Per-month breakdown** with the cumulative net stacked so a reader
   can see month-to-month dispersion (gate5 evaluator passes at the
   month-capped level; the raw per-month dispersion is informative).

Outputs:
    reports/v7s_short_alpha/A_cross_exchange_lag/a1_imb10bp_validation.csv
    reports/v7s_short_alpha/A_cross_exchange_lag/a1_imb10bp_validation.md

This script ASSUMES the canonical run has already produced
``short_trades.csv`` under ``reports/v7s_short_alpha/A_cross_exchange_lag/``.
Run the once-script first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORT_ROOT = Path("reports/v7s_short_alpha/A_cross_exchange_lag")
TRADE_PATH = REPORT_ROOT / "short_trades.csv"
CANDIDATE_CODE = "A1_imb10bp"
EXECUTION = "h24"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260617
N_WALK_FORWARD_BUCKETS = 3


def _load_a1_imb10bp_h24() -> pd.DataFrame:
    if not TRADE_PATH.exists():
        raise FileNotFoundError(f"trades CSV missing at {TRADE_PATH}; run the once-script first")
    trades = pd.read_csv(TRADE_PATH)
    sub = trades[
        (trades["candidate_code"].astype(str) == CANDIDATE_CODE)
        & (trades["execution"].astype(str) == EXECUTION)
    ].copy()
    if sub.empty:
        raise RuntimeError(f"no trades for {CANDIDATE_CODE} @ {EXECUTION} in {TRADE_PATH}")
    sub["signal_time"] = pd.to_datetime(sub["signal_time"], utc=True, errors="coerce")
    sub = sub.dropna(subset=["signal_time"]).sort_values("signal_time").reset_index(drop=True)
    return sub


def _walk_forward(sub: pd.DataFrame, buckets: int) -> pd.DataFrame:
    """Split events into N disjoint thirds by signal_time order."""
    sub = sub.copy()
    sub["bucket"] = np.linspace(0, buckets, num=len(sub), endpoint=False).astype(int)
    rows: list[dict[str, object]] = []
    for b, group in sub.groupby("bucket"):
        nets = pd.to_numeric(group["net20"], errors="coerce")
        start_ts = group["signal_time"].min()
        end_ts = group["signal_time"].max()
        rows.append(
            {
                "bucket": int(b),
                "n": int(len(group)),
                "start_signal_time": str(start_ts),
                "end_signal_time": str(end_ts),
                "mean_gross": float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
                "mean_net20": float(nets.mean()),
                "win_rate_net20": float((nets > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(sub: pd.DataFrame, n_draws: int, seed: int) -> dict[str, float]:
    nets = pd.to_numeric(sub["net20"], errors="coerce").dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(nets)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    draws = np.array([float(rng.choice(nets, size=n, replace=True).mean()) for _ in range(n_draws)])
    return {
        "n": int(n),
        "mean": float(nets.mean()),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "p_positive": float((draws > 0).mean()),
    }


def _per_month(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["month"] = sub["signal_time"].dt.strftime("%Y-%m")
    nets = pd.to_numeric(sub["net20"], errors="coerce")
    df = pd.DataFrame(
        {
            "month": sub["month"],
            "net20": nets,
        }
    )
    agg = (
        df.groupby("month")
        .agg(
            n=("net20", "size"),
            sum_net20=("net20", "sum"),
            mean_net20=("net20", "mean"),
            win_rate=("net20", lambda s: float((s > 0).mean())),
        )
        .reset_index()
    )
    return agg.sort_values("month").reset_index(drop=True)


def _format_markdown(walk_df: pd.DataFrame, ci: dict, monthly: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# A1_imb10bp h24 — validation report\n")
    lines.append(
        f"Candidate code: `{CANDIDATE_CODE}` execution: `{EXECUTION}`\n"
    )
    lines.append("## Walk-forward over disjoint thirds\n")
    lines.append("| bucket | window | N | mean_gross | mean_net20 | win_rate |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in walk_df.iterrows():
        lines.append(
            f"| {row['bucket']} | {row['start_signal_time'][:10]} → {row['end_signal_time'][:10]} | "
            f"{row['n']} | {row['mean_gross']:.4f} | {row['mean_net20']:.4f} | {row['win_rate_net20']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"All-bucket positive: "
        f"{'YES' if (walk_df['mean_net20'] > 0).all() else 'NO'}"
    )
    lines.append("")

    lines.append("## Bootstrap (5000 resamples, seed=20260617)\n")
    lines.append(
        f"- N = {ci['n']}, mean net20 = {ci['mean']:.4f}"
    )
    lines.append(
        f"- 95% CI: [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]"
    )
    lines.append(
        f"- P(mean > 0) = {ci['p_positive']:.3f}"
    )
    if ci["ci_lo"] > 0:
        lines.append("- CI excludes zero on the positive side → robust positive expectancy.")
    elif ci["ci_hi"] < 0:
        lines.append("- CI excludes zero on the negative side → robust negative expectancy.")
    else:
        lines.append("- CI straddles zero → expectancy not significantly distinguishable from zero.")
    lines.append("")

    lines.append("## Per-month breakdown\n")
    lines.append("| month | N | sum_net20 | mean_net20 | win_rate |")
    lines.append("|---|---|---|---|---|")
    for _, row in monthly.iterrows():
        lines.append(
            f"| {row['month']} | {row['n']} | {row['sum_net20']:.4f} | "
            f"{row['mean_net20']:.4f} | {row['win_rate']:.3f} |"
        )
    lines.append("")
    pos_months = int((monthly["sum_net20"] > 0).sum())
    total_months = int(len(monthly))
    lines.append(f"Positive months: {pos_months}/{total_months}")
    if total_months > 0:
        worst_month = monthly.loc[monthly["sum_net20"].idxmin(), "month"]
        best_month = monthly.loc[monthly["sum_net20"].idxmax(), "month"]
        worst_share = float(monthly["sum_net20"].min() / monthly["sum_net20"].sum()) if monthly["sum_net20"].sum() != 0 else float("nan")
        best_share = float(monthly["sum_net20"].max() / monthly["sum_net20"].sum()) if monthly["sum_net20"].sum() != 0 else float("nan")
        lines.append(f"Best month: {best_month} ({best_share:.3f} of total)")
        lines.append(f"Worst month: {worst_month} ({worst_share:.3f} of total)")
    return "\n".join(lines)


def main() -> None:
    sub = _load_a1_imb10bp_h24()
    print(f"Loaded {len(sub)} A1_imb10bp h24 trades", flush=True)
    walk_df = _walk_forward(sub, N_WALK_FORWARD_BUCKETS)
    print(f"Walk-forward buckets: {len(walk_df)}", flush=True)
    ci = _bootstrap_ci(sub, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    print(
        f"Bootstrap: N={ci['n']} mean={ci['mean']:.4f} CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] "
        f"P(>0)={ci['p_positive']:.3f}",
        flush=True,
    )
    monthly = _per_month(sub)
    print(f"Months: {len(monthly)}", flush=True)

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ROOT / "a1_imb10bp_validation.csv"
    walk_df.assign(view="walk_forward").to_csv(csv_path, index=False)
    monthly.assign(view="per_month").to_csv(csv_path, mode="a", index=False, header=True)
    pd.DataFrame([{**ci, "view": "bootstrap_ci"}]).to_csv(csv_path, mode="a", index=False, header=True)
    print(f"wrote: {csv_path}", flush=True)

    md_path = REPORT_ROOT / "a1_imb10bp_validation.md"
    md_path.write_text(_format_markdown(walk_df, ci, monthly), encoding="utf-8")
    print(f"wrote: {md_path}", flush=True)


if __name__ == "__main__":
    main()
