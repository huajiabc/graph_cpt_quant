"""v7.0 Narrative / Sector Rotation Graph.

This report tests semantic sector breadth/leader-beta context as a diagnostic
layer.  It deliberately avoids promoting a selector: any positive sector motif
must later pass random-sector controls and forward logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v50_perp_crowding_atlas import DEFAULT_FEATURE_PATH, UNIVERSE_COL, _available_columns, _bool, _max_contribution, _month_cap35, _num


REPORT_ROOT = Path("reports/v7_0_narrative_sector_rotation")
DEFAULT_BINANCE_INSTRUMENTS = Path("data/raw/binance/instruments.parquet")
DEFAULT_TRADE_CACHE = Path("reports/v1_3a_checkpoint_robustness/_v09b_trades_tmp.csv")


MANUAL_NARRATIVES: dict[str, str] = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH_L2",
    "SOLUSDT": "SOLANA",
    "JUPUSDT": "SOLANA",
    "RAYDIUMUSDT": "SOLANA",
    "WIFUSDT": "MEME",
    "DOGEUSDT": "MEME",
    "1000PEPEUSDT": "MEME",
    "FARTCOINUSDT": "MEME",
    "POPCATUSDT": "MEME",
    "PENGUUSDT": "MEME",
    "TRUMPUSDT": "MEME",
    "MOODENGUSDT": "MEME",
    "WLDUSDT": "AI",
    "TAOUSDT": "AI",
    "VIRTUALUSDT": "AI",
    "RENDERUSDT": "AI",
    "NEARUSDT": "AI",
    "FETUSDT": "AI",
    "ONDOUSDT": "RWA",
    "ENAUSDT": "RWA",
    "LINKUSDT": "ORACLE_RWA",
    "AAVEUSDT": "DEFI",
    "UNIUSDT": "DEFI",
    "LDOUSDT": "DEFI",
    "PENDLEUSDT": "DEFI",
    "CRVUSDT": "DEFI",
    "INJUSDT": "DEFI",
    "OPUSDT": "ETH_L2",
    "ARBUSDT": "ETH_L2",
    "STRKUSDT": "ETH_L2",
    "SUIUSDT": "L1",
    "APTUSDT": "L1",
    "AVAXUSDT": "L1",
    "SEIUSDT": "L1",
    "TIAUSDT": "MODULAR",
    "TONUSDT": "L1",
    "XRPUSDT": "PAYMENT",
    "XLMUSDT": "PAYMENT",
    "LTCUSDT": "PAYMENT",
    "BCHUSDT": "PAYMENT",
    "ZECUSDT": "PRIVACY",
    "XAUTUSDT": "GOLD",
}


@dataclass(frozen=True)
class V70Config:
    report_root: Path = REPORT_ROOT
    feature_path: Path = DEFAULT_FEATURE_PATH
    binance_instruments_path: Path = DEFAULT_BINANCE_INSTRUMENTS
    trade_cache_path: Path = DEFAULT_TRADE_CACHE
    universe_col: str = UNIVERSE_COL
    cost_bps: float = 20.0
    impulse_ret_1h: float = 0.004
    impulse_volume_z: float = 1.5
    breadth_high: float = 0.25
    random_seed: int = 70


FEATURE_COLUMNS = (
    "symbol",
    "feature_time",
    "warmup_complete",
    UNIVERSE_COL,
    "universe_static_current_top30",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "ret_4h_percentile",
    "volume_z_1h",
    "volume_z_4h",
    "future_ret_4h",
    "future_ret_12h",
    "btc_market_state",
)


def _read_features(cfg: V70Config) -> pd.DataFrame:
    if not cfg.feature_path.exists():
        return pd.DataFrame()
    cols = _available_columns(cfg.feature_path, tuple(dict.fromkeys([*FEATURE_COLUMNS, cfg.universe_col])), cfg.universe_col)
    pf = pq.ParquetFile(cfg.feature_path)
    frames: list[pd.DataFrame] = []
    for idx in range(pf.num_row_groups):
        chunk = pf.read_row_group(idx, columns=cols).to_pandas()
        if cfg.universe_col in chunk.columns:
            chunk = chunk[_bool(chunk, cfg.universe_col)].copy()
        elif "universe_static_current_top30" in chunk.columns:
            chunk = chunk[_bool(chunk, "universe_static_current_top30")].copy()
        if "warmup_complete" in chunk.columns:
            chunk = chunk[_bool(chunk, "warmup_complete", True)].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["feature_time"] = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    return out.dropna(subset=["symbol", "feature_time"]).sort_values(["symbol", "feature_time"]).reset_index(drop=True)


def _parse_subtype(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0]).upper().replace(" ", "_")
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        text = text.strip("[]").replace("'", "").split(",")[0].strip()
    return text.upper().replace(" ", "_") if text and text != "nan" else ""


def _taxonomy(features: pd.DataFrame, cfg: V70Config) -> pd.DataFrame:
    symbols = sorted(features["symbol"].dropna().astype(str).unique().tolist()) if not features.empty else []
    subtype = {}
    if cfg.binance_instruments_path.exists():
        inst = pd.read_parquet(cfg.binance_instruments_path, columns=["symbol", "underlyingSubType"])
        for row in inst.itertuples(index=False):
            subtype[str(row.symbol)] = _parse_subtype(getattr(row, "underlyingSubType"))
    rows = []
    for symbol in symbols:
        narrative = MANUAL_NARRATIVES.get(symbol) or subtype.get(symbol) or "UNCATEGORIZED"
        rows.append(
            {
                "symbol": symbol,
                "narrative": narrative,
                "source": "manual" if symbol in MANUAL_NARRATIVES else ("binance_underlyingSubType" if subtype.get(symbol) else "uncategorized"),
            }
        )
    return pd.DataFrame(rows)


def _decorate(features: pd.DataFrame, taxonomy: pd.DataFrame, cfg: V70Config) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    out = features.merge(taxonomy[["symbol", "narrative"]], on="symbol", how="left")
    out["narrative"] = out["narrative"].fillna("UNCATEGORIZED")
    out["local_impulse"] = _num(out, "ret_1h").ge(cfg.impulse_ret_1h) & _num(out, "volume_z_1h").ge(cfg.impulse_volume_z)
    out["local_positive_volume_shock"] = _num(out, "ret_15m").gt(0) & (_num(out, "volume_z_1h").ge(cfg.impulse_volume_z) | _num(out, "volume_z_4h").ge(cfg.impulse_volume_z))
    out["target_lagging"] = _num(out, "ret_4h_percentile").le(0.70)
    out["net20_12h"] = _num(out, "future_ret_12h") - 2.0 * cfg.cost_bps / 10_000.0
    out["month"] = out["feature_time"].dt.strftime("%Y-%m")
    out = out.sort_values(["symbol", "feature_time"]).copy()
    out["self_past_impulse_1h"] = (
        out.groupby("symbol", sort=False)["local_impulse"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
        .fillna(0.0)
    )
    sector = out.groupby(["feature_time", "narrative"], sort=False).agg(
        sector_symbols=("symbol", "nunique"),
        sector_impulse_count=("local_impulse", "sum"),
        sector_past_impulse_count=("self_past_impulse_1h", "sum"),
    ).reset_index()
    sector["sector_breadth"] = sector["sector_impulse_count"] / sector["sector_symbols"].replace(0, np.nan)
    sector["sector_past_breadth"] = sector["sector_past_impulse_count"] / sector["sector_symbols"].replace(0, np.nan)
    out = out.merge(sector, on=["feature_time", "narrative"], how="left")
    out["other_past_impulse_count"] = _num(out, "sector_past_impulse_count") - _num(out, "self_past_impulse_1h")
    out["sector_leader_prior_1h"] = out["other_past_impulse_count"].gt(0)
    out["sector_breadth_high"] = _num(out, "sector_past_breadth").ge(cfg.breadth_high)
    return out


def _summary(frame: pd.DataFrame, label: dict[str, object]) -> dict[str, object]:
    if frame.empty:
        return {**label, "events": 0, "net20_12h": np.nan, "hit_rate": np.nan, "month_cap35_net20": np.nan, "max_symbol_contribution": np.nan}
    return {
        **label,
        "events": int(len(frame)),
        "net20_12h": float(_num(frame, "net20_12h").mean()),
        "hit_rate": float(_num(frame, "net20_12h").gt(0).mean()),
        "month_cap35_net20": _month_cap35(frame, "net20_12h"),
        "max_symbol_contribution": _max_contribution(frame, "symbol", "net20_12h"),
    }


def _taxonomy_coverage(taxonomy: pd.DataFrame) -> pd.DataFrame:
    if taxonomy.empty:
        return pd.DataFrame([{"narrative": "no_taxonomy", "symbols": 0}])
    return taxonomy.groupby(["narrative", "source"], as_index=False, sort=False).agg(symbols=("symbol", "nunique"))


def _sector_breadth_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for narrative, group in frame.groupby("narrative", sort=False):
        rows.append(
            {
                "narrative": narrative,
                "rows": int(len(group)),
                "symbols": int(group["symbol"].nunique()),
                "avg_sector_breadth": float(_num(group, "sector_breadth").mean()),
                "avg_sector_past_breadth": float(_num(group, "sector_past_breadth").mean()),
                "local_impulse_rate": float(group["local_impulse"].mean()),
                "future_net20_12h": float(_num(group, "net20_12h").mean()),
            }
        )
    return pd.DataFrame(rows)


def _leader_beta_summary(frame: pd.DataFrame, cfg: V70Config) -> pd.DataFrame:
    masks = {
        "target_local_shock_only": frame["local_positive_volume_shock"],
        "sector_leader_prior_1h": frame["sector_leader_prior_1h"] & frame["target_lagging"],
        "sector_leader_prior_1h_plus_local_shock": frame["sector_leader_prior_1h"] & frame["target_lagging"] & frame["local_positive_volume_shock"],
        "sector_breadth_high_plus_local_shock": frame["sector_breadth_high"] & frame["local_positive_volume_shock"],
        "sector_leader_prior_1h_no_local_shock": frame["sector_leader_prior_1h"] & frame["target_lagging"] & ~frame["local_positive_volume_shock"],
    }
    return pd.DataFrame([_summary(frame[mask.fillna(False)], {"candidate": name}) for name, mask in masks.items()])


def _random_sector_control(frame: pd.DataFrame, taxonomy: pd.DataFrame, cfg: V70Config) -> pd.DataFrame:
    if frame.empty or taxonomy.empty:
        return pd.DataFrame()
    shuffled = taxonomy.copy()
    rng = np.random.default_rng(cfg.random_seed)
    shuffled["narrative"] = rng.permutation(shuffled["narrative"].to_numpy())
    derived = {
        "narrative",
        "local_impulse",
        "local_positive_volume_shock",
        "target_lagging",
        "net20_12h",
        "month",
        "self_past_impulse_1h",
        "sector_symbols",
        "sector_impulse_count",
        "sector_past_impulse_count",
        "sector_breadth",
        "sector_past_breadth",
        "other_past_impulse_count",
        "sector_leader_prior_1h",
        "sector_breadth_high",
    }
    base = frame.drop(columns=[col for col in derived if col in frame.columns], errors="ignore")
    random_frame = _decorate(base, shuffled, cfg)
    real = _leader_beta_summary(frame, cfg)
    real["control"] = "real_narrative"
    rand = _leader_beta_summary(random_frame, cfg)
    rand["control"] = "random_shuffled_narrative"
    out = pd.concat([real, rand], ignore_index=True)
    piv = out.pivot_table(index="candidate", columns="control", values="net20_12h", aggfunc="first")
    if {"real_narrative", "random_shuffled_narrative"}.issubset(piv.columns):
        piv["real_vs_random_lift"] = piv["real_narrative"] - piv["random_shuffled_narrative"]
    return out.merge(piv.reset_index(), on="candidate", how="left")


def _load_p2_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    trades = pd.read_csv(path, low_memory=False)
    if trades.empty or "candidate" not in trades.columns:
        return pd.DataFrame()
    out = trades[trades["candidate"].astype(str).isin(["CIC1_beta_extreme", "CIC2_beta_broad"])].copy()
    if out.empty:
        return out
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    out["net20"] = _num(out, "net_return")
    return out


def _cic_sector_interaction(frame: pd.DataFrame, cfg: V70Config) -> pd.DataFrame:
    trades = _load_p2_trades(cfg.trade_cache_path)
    if trades.empty or frame.empty:
        return pd.DataFrame([{"bucket": "no_p2_trade_cache", "trades": 0}])
    context_cols = ["symbol", "feature_time", "narrative", "sector_leader_prior_1h", "sector_breadth_high", "sector_past_breadth"]
    context = frame[context_cols].sort_values(["symbol", "feature_time"]).copy()
    rows = []
    for symbol, local in trades.groupby("symbol", sort=False):
        ctx = context[context["symbol"].astype(str).eq(str(symbol))]
        if ctx.empty:
            continue
        rows.append(
            pd.merge_asof(
                local.sort_values("entry_time"),
                ctx,
                left_on="entry_time",
                right_on="feature_time",
                by="symbol",
                direction="backward",
                tolerance=pd.Timedelta(minutes=30),
            )
        )
    merged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if merged.empty:
        return pd.DataFrame([{"bucket": "no_aligned_p2_context", "trades": 0}])
    leader_context = merged["sector_leader_prior_1h"].astype("boolean").fillna(False).astype(bool)
    breadth_context = merged["sector_breadth_high"].astype("boolean").fillna(False).astype(bool)
    merged["sector_context"] = np.select(
        [leader_context, breadth_context],
        ["sector_leader_prior_1h", "sector_breadth_high"],
        default="no_sector_context",
    )
    out = []
    for key in ["candidate", "sector_context", "narrative"]:
        for bucket, group in merged.groupby(key, sort=False, dropna=False):
            out.append(
                {
                    "bucket_type": key,
                    "bucket": bucket,
                    "trades": int(len(group)),
                    "net20": float(_num(group, "net20").mean()) if len(group) else np.nan,
                    "hit_rate": float(_num(group, "net20").gt(0).mean()) if len(group) else np.nan,
                    "month_cap35_net20": _month_cap35(group, "net20"),
                    "max_symbol_contribution": _max_contribution(group, "symbol", "net20"),
                }
            )
    return pd.DataFrame(out)


def _write_notes(path: Path, summary: pd.DataFrame, controls: pd.DataFrame) -> None:
    best = summary.sort_values("net20_12h", ascending=False).head(1) if not summary.empty else pd.DataFrame()
    lines = [
        "# v7.0 Narrative / Sector Rotation Graph",
        "",
        "Status: diagnostic atlas only. No sector gate, selector, or live permission is changed.",
    ]
    if not best.empty:
        row = best.iloc[0]
        lines.append(f"- Best first-pass candidate: `{row['candidate']}` events={int(row['events'])}, net20_12h={row['net20_12h']:.4%}.")
    if not controls.empty and "real_vs_random_lift" in controls.columns:
        ctrl = controls.dropna(subset=["real_vs_random_lift"]).sort_values("real_vs_random_lift", ascending=False).head(1)
        if not ctrl.empty:
            row = ctrl.iloc[0]
            lines.append(f"- Best real-vs-random lift: `{row['candidate']}` lift={row['real_vs_random_lift']:.4%}.")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Static narrative labels are seed taxonomy, not proof of semantic graph alpha.",
            "- Any positive motif must beat random sector assignment and survive month/symbol concentration before promotion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v70_narrative_sector_rotation(cfg: V70Config | None = None) -> dict[str, Path]:
    cfg = cfg or V70Config()
    report_root = ensure_dir(cfg.report_root)
    features = _read_features(cfg)
    taxonomy = _taxonomy(features, cfg)
    frame = _decorate(features, taxonomy, cfg)
    coverage = _taxonomy_coverage(taxonomy)
    breadth = _sector_breadth_summary(frame)
    summary = _leader_beta_summary(frame, cfg)
    controls = _random_sector_control(frame, taxonomy, cfg)
    cic = _cic_sector_interaction(frame, cfg)

    outputs = {
        "taxonomy_coverage": report_root / "taxonomy_coverage.csv",
        "sector_breadth_summary": report_root / "sector_breadth_summary.csv",
        "sector_leader_beta_summary": report_root / "sector_leader_beta_summary.csv",
        "random_sector_control": report_root / "random_sector_control.csv",
        "cic_sector_interaction": report_root / "cic_sector_interaction.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    coverage.to_csv(outputs["taxonomy_coverage"], index=False)
    breadth.to_csv(outputs["sector_breadth_summary"], index=False)
    summary.to_csv(outputs["sector_leader_beta_summary"], index=False)
    controls.to_csv(outputs["random_sector_control"], index=False)
    cic.to_csv(outputs["cic_sector_interaction"], index=False)
    _write_notes(outputs["candidate_notes"], summary, controls)
    return outputs
