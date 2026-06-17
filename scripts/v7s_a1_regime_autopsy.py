"""A1 Regime Autopsy — NOT an upgrade (per 77.docx P2).

The A1_imb10bp h24 cell looked like it cleared the v7S 10-gate battery
under the first cut of the evaluator, but validation surfaced that
2025-10 carried 63.7 % of the alpha and walk-forward bucket 0 was
negative. 77 docx P2 directs: research the regime, do NOT promote
it. This script is the regime autopsy. Three outputs:

1. ``a1_regime_autopsy.csv`` — per-(month) regime feature averages
   so a reader can see what was different about 2025-10 vs the
   negative months.
2. ``ex_2025_10_summary.csv`` — A1 metrics with 2025-10 excluded.
   If the candidate still pays without that one month, the regime is
   incidental; if it collapses, the regime IS the candidate.
3. ``regime_control_summary.csv`` — shuffled-regime control. For each
   A1 trade, re-pair its outcome with a random entry_time from the
   same month-pool and recompute the regime features. If the
   distinction between productive vs unproductive months survives the
   shuffle, the "regime" is just calendar noise, not a real gate.

The autopsy DOES NOT decide A1 is tradable. It produces evidence for
or against the regime-gate hypothesis. The candidate stays
`diagnostic_only` until OOS replay on fresh months confirms the
regime is forward-stable.

Reads the canonical run's ``short_trades.csv`` under
``reports/v7s_short_alpha/A_cross_exchange_lag/``. Reads the v0.3
feature parquet for the regime overlays.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_ROOT = Path("reports/v7s_short_alpha/A_cross_exchange_lag")
TRADES_CSV = REPORT_ROOT / "short_trades.csv"
AUTOPSY_CSV = REPORT_ROOT / "a1_regime_autopsy.csv"
EX_MONTH_CSV = REPORT_ROOT / "ex_2025_10_summary.csv"
REGIME_CONTROL_CSV = REPORT_ROOT / "regime_control_summary.csv"

CANDIDATE_CODE = "A1_imb10bp"
EXECUTION = "h24"
TARGET_MONTH = "2025-10"

# Regime features available in the v0.3 feature parquet.
REGIME_COLS: tuple[str, ...] = (
    "btc_ret_4h",
    "btc_ret_1h",
    "btc_volatility_4h",
    "btc_volatility_percentile",
    "btc_market_state",
    "btc_vol_regime",
    "funding_percentile",
    "funding_rate_settled",
    "oi_value_delta_4h_percentile",
    "oi_delta_1h_percentile",
    "ret_4h_percentile",
    "volume_z_4h",
)


def _load_a1_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        raise FileNotFoundError(f"trades CSV missing at {TRADES_CSV}")
    trades = pd.read_csv(TRADES_CSV)
    sub = trades[
        (trades["candidate_code"].astype(str) == CANDIDATE_CODE)
        & (trades["execution"].astype(str) == EXECUTION)
    ].copy()
    sub["signal_time"] = pd.to_datetime(sub["signal_time"], utc=True, errors="coerce")
    sub["entry_time"] = pd.to_datetime(sub["entry_time"], utc=True, errors="coerce")
    sub = sub.dropna(subset=["signal_time"]).sort_values("signal_time").reset_index(drop=True)
    return sub


def _attach_regime_features(trades: pd.DataFrame, feature_path: Path) -> pd.DataFrame:
    """Per trade, look up its entry-bar regime features in the v0.3 parquet."""
    if not feature_path.exists():
        print(f"warning: feature parquet missing at {feature_path}; regime cols will be NaN", flush=True)
        for col in REGIME_COLS:
            trades[col] = float("nan")
        return trades

    symbols = sorted(trades["symbol"].astype(str).unique())
    cols_to_load = ("symbol", "bar_open_time") + tuple(c for c in REGIME_COLS if c)
    out_rows: list[dict[str, object]] = []
    for symbol in symbols:
        symbol_trades = trades[trades["symbol"] == symbol]
        try:
            sym_df = pd.read_parquet(feature_path, filters=[("symbol", "=", symbol)], columns=list(cols_to_load))
        except Exception:
            # Some parquet engines don't support filters; fall back to full read once.
            full = pd.read_parquet(feature_path, columns=list(cols_to_load))
            sym_df = full[full["symbol"].astype(str) == symbol]
        if sym_df.empty:
            for _, row in symbol_trades.iterrows():
                out_rows.append({**row.to_dict(), **{c: float("nan") for c in REGIME_COLS}})
            continue
        sym_df["bar_open_time"] = pd.to_datetime(sym_df["bar_open_time"], utc=True, errors="coerce")
        sym_df = sym_df.dropna(subset=["bar_open_time"]).sort_values("bar_open_time").reset_index(drop=True)
        ns_to_idx: dict[int, int] = {int(t.value): i for i, t in enumerate(sym_df["bar_open_time"])}
        for _, row in symbol_trades.iterrows():
            entry_time = pd.Timestamp(row.get("entry_time"))
            if pd.isna(entry_time):
                idx = -1
            else:
                idx = ns_to_idx.get(int(entry_time.value), -1)
            enriched = row.to_dict()
            if idx >= 0:
                feature_row = sym_df.iloc[idx]
                for col in REGIME_COLS:
                    enriched[col] = feature_row.get(col, float("nan"))
            else:
                for col in REGIME_COLS:
                    enriched[col] = float("nan")
            out_rows.append(enriched)
    return pd.DataFrame(out_rows)


def _per_month_summary(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["month"] = trades["signal_time"].dt.strftime("%Y-%m")
    nets = pd.to_numeric(trades["net20"], errors="coerce")
    rows: list[dict[str, object]] = []
    for month, group in trades.groupby("month"):
        row: dict[str, object] = {
            "month": month,
            "n": int(len(group)),
            "sum_net20": float(pd.to_numeric(group["net20"], errors="coerce").sum()),
            "mean_net20": float(pd.to_numeric(group["net20"], errors="coerce").mean()),
            "win_rate": float((pd.to_numeric(group["net20"], errors="coerce") > 0).mean()),
        }
        for col in REGIME_COLS:
            vals = pd.to_numeric(group.get(col), errors="coerce")
            if vals.notna().any():
                row[f"mean_{col}"] = float(vals.mean())
            else:
                # Non-numeric like btc_market_state — take mode.
                if col in group.columns:
                    sub_vals = group[col].astype(str)
                    if len(sub_vals):
                        row[f"mode_{col}"] = str(sub_vals.value_counts().idxmax())
                    else:
                        row[f"mode_{col}"] = ""
                else:
                    row[f"mode_{col}"] = ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _ex_target_month(trades: pd.DataFrame, target_month: str) -> pd.DataFrame:
    trades = trades.copy()
    trades["month"] = trades["signal_time"].dt.strftime("%Y-%m")
    ex = trades[trades["month"] != target_month]
    full = trades
    rows: list[dict[str, object]] = []
    for label, sub in (("full_sample", full), (f"ex_{target_month}", ex)):
        nets = pd.to_numeric(sub["net20"], errors="coerce")
        rows.append(
            {
                "label": label,
                "n": int(len(sub)),
                "sum_net20": float(nets.sum()),
                "mean_net20": float(nets.mean()),
                "win_rate": float((nets > 0).mean()),
                "best_month_share": _best_month_share(sub),
            }
        )
    return pd.DataFrame(rows)


def _best_month_share(trades: pd.DataFrame) -> float:
    if trades.empty or "signal_time" not in trades.columns:
        return float("nan")
    sub = trades.copy()
    sub["month"] = pd.to_datetime(sub["signal_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    nets = pd.to_numeric(sub["net20"], errors="coerce")
    monthly = nets.groupby(sub["month"]).sum()
    if monthly.empty:
        return float("nan")
    total = float(monthly.sum())
    if total == 0:
        return float("nan")
    return float(monthly.max() / total)


def _regime_control(trades: pd.DataFrame, n_shuffles: int, seed: int) -> pd.DataFrame:
    """Shuffled-regime control: re-pair each trade's outcome with a random
    entry_time from the same month's POOL of trades and recompute the
    monthly best-month-share + mean. If the distinction between productive
    and unproductive months survives the shuffle, the "regime" is calendar
    noise."""
    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades["signal_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    nets = pd.to_numeric(trades["net20"], errors="coerce").fillna(0.0).to_numpy()
    months = trades["month"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    target_shares: list[float] = []
    mean_shares: list[float] = []
    n_shuffles = max(int(n_shuffles), 1)
    for _ in range(n_shuffles):
        permuted_months = rng.permutation(months)
        df_perm = pd.DataFrame({"net20": nets, "month": permuted_months})
        per_month = df_perm.groupby("month")["net20"].sum()
        total = float(per_month.sum())
        if total == 0 or len(per_month) == 0:
            continue
        target_share = float(per_month.get(TARGET_MONTH, 0.0) / total)
        target_shares.append(target_share)
        mean_shares.append(float(per_month.abs().max() / per_month.abs().sum()))
    # Observed share of target month with the real data.
    observed = pd.DataFrame({"net20": nets, "month": months}).groupby("month")["net20"].sum()
    observed_target_share = float(observed.get(TARGET_MONTH, 0.0) / max(observed.sum(), 1e-12))

    df = pd.DataFrame(
        [
            {
                "metric": f"target_month_share_{TARGET_MONTH}",
                "observed": observed_target_share,
                "control_mean": float(np.mean(target_shares)) if target_shares else float("nan"),
                "control_p95": float(np.percentile(target_shares, 95)) if target_shares else float("nan"),
                "p_more_extreme": float(np.mean(np.array(target_shares) >= observed_target_share))
                if target_shares
                else float("nan"),
            },
            {
                "metric": "max_month_share_any",
                "observed": float(observed.abs().max() / observed.abs().sum()),
                "control_mean": float(np.mean(mean_shares)) if mean_shares else float("nan"),
                "control_p95": float(np.percentile(mean_shares, 95)) if mean_shares else float("nan"),
                "p_more_extreme": float("nan"),
            },
        ]
    )
    return df


def _write_notes(autopsy: pd.DataFrame, ex_month: pd.DataFrame, control: pd.DataFrame, target_month: str) -> Path:
    path = REPORT_ROOT / "a1_regime_autopsy_notes.md"
    lines: list[str] = []
    lines.append(f"# A1 Regime Autopsy — {target_month}\n")
    lines.append("> A1 is NOT a candidate. This document records regime evidence so a future regime detector can be designed WITHOUT overfitting to {target_month}.\n".format(target_month=target_month))
    lines.append(f"## Per-month regime profile\n")
    if not autopsy.empty:
        cols = [c for c in autopsy.columns if c.startswith("mean_") or c == "month" or c == "n" or c == "sum_net20" or c == "mean_net20" or c == "win_rate" or c.startswith("mode_")]
        lines.append(autopsy[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Sample with target month excluded\n")
    if not ex_month.empty:
        lines.append(ex_month.to_markdown(index=False))
    lines.append("")
    lines.append(f"## Shuffled-regime control (target = {target_month})\n")
    if not control.empty:
        lines.append(control.to_markdown(index=False))
    lines.append("")
    lines.append("## Verdict guidance\n")
    if not ex_month.empty and "mean_net20" in ex_month.columns:
        full = ex_month[ex_month["label"] == "full_sample"]
        ex = ex_month[ex_month["label"] != "full_sample"]
        if not ex.empty:
            full_mean = float(full["mean_net20"].iloc[0]) if not full.empty else float("nan")
            ex_mean = float(ex["mean_net20"].iloc[0])
            if ex_mean > 0:
                lines.append(f"- ex-{target_month} mean_net20 = {ex_mean:.4f} > 0: regime is incidental, not load-bearing.")
            else:
                lines.append(f"- ex-{target_month} mean_net20 = {ex_mean:.4f} ≤ 0: the regime IS the candidate. Do NOT promote A1 without it.")
            lines.append(f"- full_sample mean_net20 = {full_mean:.4f}.")
    lines.append("")
    lines.append("Per 77 docx, A1 STAYS in diagnostic_only / regime-event research until:")
    lines.append("- ex-target month mean is positive, AND")
    lines.append("- OOS replay on fresh months stays positive, AND")
    lines.append("- shuffled-regime control fails to reproduce the target month's contribution.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 Regime Autopsy — research only, not an upgrade.")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet"),
        help="v0.3 features parquet (per-symbol bars).",
    )
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

    trades = _load_a1_trades()
    print(f"Loaded {len(trades)} A1_imb10bp h24 trades", flush=True)
    trades_with_regime = _attach_regime_features(trades, args.feature_path)
    print("Regime features attached", flush=True)

    autopsy = _per_month_summary(trades_with_regime)
    autopsy.to_csv(AUTOPSY_CSV, index=False)
    print(f"wrote: {AUTOPSY_CSV}", flush=True)

    ex_month = _ex_target_month(trades_with_regime, TARGET_MONTH)
    ex_month.to_csv(EX_MONTH_CSV, index=False)
    print(f"wrote: {EX_MONTH_CSV}", flush=True)

    control = _regime_control(trades_with_regime, args.n_shuffles, args.seed)
    control.to_csv(REGIME_CONTROL_CSV, index=False)
    print(f"wrote: {REGIME_CONTROL_CSV}", flush=True)

    notes = _write_notes(autopsy, ex_month, control, TARGET_MONTH)
    print(f"wrote: {notes}", flush=True)


if __name__ == "__main__":
    main()
