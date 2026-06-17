"""Diagnostic: count how many candidates each gate of Direction E filters out.

Counts per (symbol, candidate_code) and totals:
- cic_longs_seen
- breakdowns_found
- after_cp60
- after_no_protect_a
- after_beta_high_gone
- after_sell_flow

Writes a CSV at reports/v7s_short_alpha/E_cic_failure_confirmed/gate_cascade_counts.csv
so we can see WHICH gate is killing the population. No verdict — diagnostic only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import load_config
from pressure_graph.io import ensure_dir, read_parquet
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _rank_inputs
from pressure_graph.reports.v10a_cic_basket_portfolio import _focus_pool
from pressure_graph.reports.v3_4_true_short_sleeve import (
    _build_cic_long_index,
    _find_breakdown,
    _gate_cp60_would_exit,
    _gate_no_protect_a,
)
from pressure_graph.reports.v12s_short_motif_atlas import _f
from pressure_graph.reports.v7s_short_alpha import (
    CANDIDATE_E1,
    CANDIDATE_E2,
    V7SConfig,
    _build_orderflow_lookup,
    _gate_beta_high_gone,
    _gate_sell_flow_confirms,
)


def main() -> None:
    config_path = Path("configs/v0_3.yaml")
    config = load_config(config_path)
    cfg = V7SConfig()
    feature_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    trade_cache_path = cfg.trade_cache_path

    instruments = pd.DataFrame()  # not needed for rank_inputs
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    trade_cache = read_parquet(trade_cache_path)
    pool = _focus_pool(trade_cache, cfg.long_pool_name) if not trade_cache.empty else pd.DataFrame()
    cic_long_index = _build_cic_long_index(pool, cfg.v34_cfg) if not pool.empty else {}
    print(f"CIC long index size: {len(cic_long_index)}", flush=True)
    orderflow_lookup = _build_orderflow_lookup(cfg.orderflow_event_path)
    print(f"orderflow lookup loaded: {orderflow_lookup is not None}", flush=True)

    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= cfg.top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    rows: list[dict[str, object]] = []
    columns_seen: dict[str, int] = {}
    for i, symbol in enumerate(symbols, start=1):
        group = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if group.empty:
            continue
        group = group.sort_values("bar_open_time").reset_index(drop=True)
        for col in ("gate_beta_already_extended", "c2_bucket_beta_extreme_overextended", "gate_Protect_A", "feature_time"):
            columns_seen[col] = columns_seen.get(col, 0) + (1 if col in group.columns else 0)

        feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
        feature_ns = feature_time.astype("int64").to_numpy()
        n = len(group)

        for code in (CANDIDATE_E1, CANDIDATE_E2):
            breakdown_reference = "entry" if code == CANDIDATE_E1 else "pullback_low"
            cic_long_count = 0
            breakdown_count = 0
            after_cp60 = 0
            after_no_protect = 0
            after_beta_gone = 0
            after_sell_flow = 0
            sell_flow_audit_buckets: dict[str, int] = {}
            for (sym_key, ts_key), payload in cic_long_index.items():
                if sym_key != symbol:
                    continue
                cic_long_count += 1
                anchor_ns = int(pd.Timestamp(ts_key).value)
                confirmation_idx = int(np.searchsorted(feature_ns, anchor_ns, side="left"))
                if confirmation_idx >= n - 1:
                    continue
                if breakdown_reference == "entry":
                    close = _f(group, "close")
                    level = float(close[confirmation_idx]) if confirmation_idx < len(close) else float("nan")
                else:
                    ref_low = float(payload.get("pullback_low", float("nan")))
                    level = ref_low if np.isfinite(ref_low) else (
                        float(np.nanmin(_f(group, "low")[confirmation_idx : confirmation_idx + 4]))
                    )
                break_idx = _find_breakdown(group, confirmation_idx, level, cfg.e_breakdown_valid_bars)
                if break_idx < 0:
                    continue
                breakdown_count += 1
                if not _gate_cp60_would_exit(group, break_idx, cfg.v34_cfg):
                    continue
                after_cp60 += 1
                if not _gate_no_protect_a(group, break_idx, cfg.v34_cfg):
                    continue
                after_no_protect += 1
                if not _gate_beta_high_gone(group, break_idx, cfg):
                    continue
                after_beta_gone += 1
                passed, reason = _gate_sell_flow_confirms(group, break_idx, cfg, orderflow_lookup)
                sell_flow_audit_buckets[reason] = sell_flow_audit_buckets.get(reason, 0) + 1
                if passed:
                    after_sell_flow += 1

            rows.append(
                {
                    "symbol": symbol,
                    "candidate_code": code,
                    "cic_long_count": cic_long_count,
                    "breakdown_count": breakdown_count,
                    "after_cp60": after_cp60,
                    "after_no_protect_a": after_no_protect,
                    "after_beta_high_gone": after_beta_gone,
                    "after_sell_flow": after_sell_flow,
                    "sell_flow_audit": ";".join(f"{k}={v}" for k, v in sorted(sell_flow_audit_buckets.items())),
                }
            )
        if i % 25 == 0:
            print(f"cascade: {i}/{len(symbols)} symbols", flush=True)

    out_root = ensure_dir(Path("reports/v7s_short_alpha/E_cic_failure_confirmed"))
    out_path = out_root / "gate_cascade_counts.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nwrote: {out_path}", flush=True)

    # Summary
    if not df.empty:
        totals = df.groupby("candidate_code")[
            ["cic_long_count", "breakdown_count", "after_cp60", "after_no_protect_a", "after_beta_high_gone", "after_sell_flow"]
        ].sum()
        print("\n=== totals per candidate ===", flush=True)
        print(totals.to_string(), flush=True)

    print("\n=== feature column presence (out of total symbols) ===", flush=True)
    for k, v in columns_seen.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
