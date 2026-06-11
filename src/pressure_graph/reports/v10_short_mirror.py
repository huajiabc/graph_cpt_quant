from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.io import raw_path
from pressure_graph.paper_live.v07d2 import prepare_v07d2_features_from_history
from pressure_graph.reports.v03 import _event_flags


REPORT_ROOT = Path("reports/v1_0_short_mirror_failure")
PAPER_LIVE_ROOT = Path("reports/v1_0_short_paper_live")
TOP_N = 50
COST_BPS = [10.0, 20.0, 30.0, 50.0]


@dataclass(frozen=True)
class ShortCandidate:
    candidate: str
    family: str
    gate_col: str
    event_col: str
    entry_kind: str
    strict_entry_gate: bool
    require_gate_lost_at_entry: bool = False
    bounce_pct: float = 0.010
    valid_bars: int = 8


CANDIDATES = [
    ShortCandidate(
        "S1_BEAR_MIRROR_D10_B05",
        "bearish_mirror",
        "s1_bear_density10_beta05",
        "bearish_volume_shock_event",
        "bounce_reject",
        True,
    ),
    ShortCandidate(
        "S1_BEAR_MIRROR_D10_B10",
        "bearish_mirror",
        "s1_bear_density10_beta10",
        "bearish_volume_shock_event",
        "bounce_reject",
        True,
    ),
    ShortCandidate(
        "S1_BEAR_MIRROR_D08_B10",
        "bearish_mirror",
        "s1_bear_density08_beta10",
        "bearish_volume_shock_event",
        "bounce_reject",
        True,
    ),
    ShortCandidate(
        "S1_BEAR_MIRROR_D10_B20",
        "bearish_mirror",
        "s1_bear_density10_beta20",
        "bearish_volume_shock_event",
        "bounce_reject",
        True,
    ),
    ShortCandidate(
        "S2_FAIL_MIR1_BREAK_LOW",
        "long_failure",
        "c2_mir1_raw",
        "bullish_volume_shock_event",
        "break_signal_low",
        False,
    ),
    ShortCandidate(
        "S2_FAIL_CIC2_BREAK_LOW",
        "long_failure",
        "c2_beta_continuation",
        "bullish_volume_shock_event",
        "break_signal_low",
        False,
    ),
    ShortCandidate(
        "S2_FAIL_RELAX_D08_B90_BREAK_LOW",
        "long_failure",
        "c2_relax_density08_beta90",
        "bullish_volume_shock_event",
        "break_signal_low",
        False,
    ),
    ShortCandidate(
        "S2_FAIL_MIR1_NO_RECLAIM_BREAK_LOW",
        "long_failure_clean",
        "c2_mir1_raw",
        "bullish_volume_shock_event",
        "pullback_no_reclaim_break_low",
        False,
    ),
    ShortCandidate(
        "S2_FAIL_CIC2_NO_RECLAIM_BREAK_LOW",
        "long_failure_clean",
        "c2_beta_continuation",
        "bullish_volume_shock_event",
        "pullback_no_reclaim_break_low",
        False,
    ),
    ShortCandidate(
        "S2_FAIL_CIC2_GATE_LOST_BREAK_LOW",
        "long_failure_gate_lost",
        "c2_beta_continuation",
        "bullish_volume_shock_event",
        "break_signal_low",
        False,
        True,
    ),
]

PAPER_LIVE_CANDIDATES = [
    candidate
    for candidate in CANDIDATES
    if candidate.candidate
    in {
        "S1_BEAR_MIRROR_D10_B05",
        "S1_BEAR_MIRROR_D10_B10",
        "S1_BEAR_MIRROR_D08_B10",
    }
]


def _safe_float(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) else default


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _vol_regime_rule_short(row: pd.Series) -> ExecutionRule:
    """Asymmetric SL/TP profile for short side.

    Shorts have a hard ceiling on profit (-100%) and unbounded loss risk on
    short-squeeze, so the short profile uses a tighter SL than the long
    mirror. Calibration (low/mid/high vol):
        low  vol_pct < 40 -> tp=0.025 sl=0.015
        mid  vol_pct < 80 -> tp=0.035 sl=0.020
        high vol_pct >=80 -> tp=0.045 sl=0.025
    """
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.025, sl=0.015, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.035, sl=0.020, max_hold_bars=16)
    return ExecutionRule(tp=0.045, sl=0.025, max_hold_bars=16)


# Threshold below which an 8-hour funding rate is "too negative" for a short
# (i.e. shorts are crowded and paying longs). -0.03%/8h ~= -32% annualized.
FUNDING_SHORT_BLOCK_THRESHOLD = -0.0003


def _funding_block_short(row: pd.Series) -> bool:
    """Block a short entry when 8-hour funding is below FUNDING_SHORT_BLOCK_THRESHOLD.

    Fail-open semantics: missing funding data (NaN / not in row) returns False
    so we do not silently block entries when the data feed is incomplete.
    Callers can compose stricter funding policies on top.
    """
    funding = pd.to_numeric(row.get("funding_rate_settled"), errors="coerce")
    if pd.isna(funding):
        return False
    return bool(funding < FUNDING_SHORT_BLOCK_THRESHOLD)


def add_short_mirror_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)
    rank = pd.to_numeric(out.get("dynamic_all_rank"), errors="coerce")
    top50 = rank <= TOP_N
    volume_z = pd.to_numeric(out.get("volume_z_4h"), errors="coerce")
    ret_4h = pd.to_numeric(out.get("ret_4h"), errors="coerce")
    ret_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")

    out["bearish_volume_shock_state"] = warmup & top50 & (volume_z > 2.0) & (ret_4h < 0)
    out["bearish_volume_shock_event"] = (
        out.sort_values(["exchange", "symbol", "bar_open_time"])
        .groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[
            "bearish_volume_shock_state"
        ]
        .apply(lambda series: _event_flags(series, 16))
        .reindex(out.index)
        .fillna(False)
        .astype(bool)
    )

    density_inputs = pd.DataFrame(
        {
            "feature_time": pd.to_datetime(out["feature_time"], utc=True, errors="coerce"),
            "top50": top50.fillna(False).astype(bool),
            "bearish": out["bearish_volume_shock_state"].fillna(False).astype(bool),
        },
        index=out.index,
    )
    grouped = density_inputs.groupby("feature_time", sort=False)[["top50", "bearish"]].sum()
    density_map = (grouped["bearish"] / grouped["top50"].replace(0, np.nan)).to_dict()
    out["bearish_volume_impulse_density"] = density_inputs["feature_time"].map(density_map)

    out["s1_bear_density10_beta05"] = (out["bearish_volume_impulse_density"] >= 0.10) & (
        ret_pct <= 5
    )
    out["s1_bear_density10_beta10"] = (out["bearish_volume_impulse_density"] >= 0.10) & (
        ret_pct <= 10
    )
    out["s1_bear_density08_beta10"] = (out["bearish_volume_impulse_density"] >= 0.08) & (
        ret_pct <= 10
    )
    out["s1_bear_density10_beta20"] = (out["bearish_volume_impulse_density"] >= 0.10) & (
        ret_pct <= 20
    )
    return out


def _entry_bounce_reject(
    group: pd.DataFrame,
    signal_idx: int,
    bounce_pct: float,
    valid_bars: int,
) -> tuple[int, float, str] | None:
    signal = group.iloc[signal_idx]
    signal_close = _safe_float(signal.get("close"))
    trigger = signal_close * (1.0 + bounce_pct)
    saw_bounce = False
    end = min(signal_idx + valid_bars, len(group) - 1)
    for idx in range(signal_idx + 1, end + 1):
        row = group.iloc[idx]
        if not saw_bounce and _safe_float(row.get("high")) >= trigger:
            saw_bounce = True
        if saw_bounce and _safe_float(row.get("close")) <= signal_close:
            entry_idx = idx + 1
            if entry_idx >= len(group):
                return None
            return entry_idx, _safe_float(group.iloc[entry_idx].get("open")), "bounce_reject_next_open"
    return None


def _entry_break_signal_low(
    group: pd.DataFrame,
    signal_idx: int,
    valid_bars: int,
) -> tuple[int, float, str] | None:
    signal_low = _safe_float(group.iloc[signal_idx].get("low"))
    end = min(signal_idx + valid_bars, len(group) - 1)
    for idx in range(signal_idx + 1, end + 1):
        if _safe_float(group.iloc[idx].get("close")) <= signal_low:
            entry_idx = idx + 1
            if entry_idx >= len(group):
                return None
            return entry_idx, _safe_float(group.iloc[entry_idx].get("open")), "break_signal_low_next_open"
    return None


def _entry_pullback_no_reclaim_break_low(
    group: pd.DataFrame,
    signal_idx: int,
    pullback_pct: float,
    valid_bars: int,
) -> tuple[int, float, str] | None:
    signal = group.iloc[signal_idx]
    signal_close = _safe_float(signal.get("close"))
    signal_low = _safe_float(signal.get("low"))
    pullback = signal_close * (1.0 - pullback_pct)
    saw_pullback = False
    end = min(signal_idx + valid_bars, len(group) - 1)
    for idx in range(signal_idx + 1, end + 1):
        row = group.iloc[idx]
        if not saw_pullback and _safe_float(row.get("low")) <= pullback:
            saw_pullback = True
        if saw_pullback and _safe_float(row.get("close")) >= signal_close:
            return None
        if saw_pullback and _safe_float(row.get("close")) <= signal_low:
            entry_idx = idx + 1
            if entry_idx >= len(group):
                return None
            return entry_idx, _safe_float(group.iloc[entry_idx].get("open")), "no_reclaim_break_low_next_open"
    return None


def _entry_for_short_candidate(
    group: pd.DataFrame,
    signal_idx: int,
    candidate: ShortCandidate,
) -> tuple[int, float, str] | None:
    if candidate.entry_kind == "bounce_reject":
        return _entry_bounce_reject(group, signal_idx, candidate.bounce_pct, candidate.valid_bars)
    if candidate.entry_kind == "break_signal_low":
        return _entry_break_signal_low(group, signal_idx, candidate.valid_bars)
    if candidate.entry_kind == "pullback_no_reclaim_break_low":
        return _entry_pullback_no_reclaim_break_low(
            group, signal_idx, candidate.bounce_pct, candidate.valid_bars
        )
    raise KeyError(candidate.entry_kind)


def _short_exit(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    rule: ExecutionRule,
) -> dict[str, object]:
    tp_price = entry_price * (1.0 - rule.tp)
    sl_price = entry_price * (1.0 + rule.sl)
    max_idx = min(entry_idx + rule.max_hold_bars - 1, len(group) - 1)
    exit_idx = max_idx
    exit_price = _safe_float(group.iloc[max_idx].get("close"))
    exit_reason = "max_hold"
    ambiguous = False
    for idx in range(entry_idx, max_idx + 1):
        row = group.iloc[idx]
        tp_hit = _safe_float(row.get("low")) <= tp_price
        sl_hit = _safe_float(row.get("high")) >= sl_price
        if tp_hit and sl_hit:
            exit_idx = idx
            exit_price = sl_price
            exit_reason = "sl_ambiguous"
            ambiguous = True
            break
        if sl_hit:
            exit_idx = idx
            exit_price = sl_price
            exit_reason = "sl"
            break
        if tp_hit:
            exit_idx = idx
            exit_price = tp_price
            exit_reason = "tp"
            break
    holding = group.iloc[entry_idx : exit_idx + 1]
    return {
        "exit_idx": exit_idx,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return": 1.0 - exit_price / entry_price,
        "holding_bars": exit_idx - entry_idx + 1,
        "holding_minutes": (
            pd.Timestamp(group.iloc[exit_idx]["bar_close_time"])
            - pd.Timestamp(group.iloc[entry_idx]["bar_open_time"])
        ).total_seconds()
        / 60.0,
        "mfe": 1.0 - _safe_float(holding["low"].min()) / entry_price,
        "mae": 1.0 - _safe_float(holding["high"].max()) / entry_price,
        "same_bar_ambiguity": ambiguous,
    }


def simulate_short_candidate(
    data: pd.DataFrame,
    candidate: ShortCandidate,
    *,
    rule_factory: Callable[[pd.Series], ExecutionRule] | None = None,
    funding_blocker: Callable[[pd.Series], bool] | None = None,
) -> pd.DataFrame:
    """Simulate fills for ``candidate`` over ``data``.

    Parameters
    ----------
    rule_factory:
        Optional override for the per-entry execution rule. Defaults to
        ``_vol_regime_rule`` (symmetric long-mirror). For S2 paper-live use
        ``_vol_regime_rule_short`` to apply the asymmetric short profile.
    funding_blocker:
        Optional callable that receives the entry-bar row and returns True to
        skip the entry. None means no funding check.
    """
    pick_rule = rule_factory or _vol_regime_rule
    rows: list[dict[str, object]] = []
    sort_cols = ["exchange", "symbol", "bar_open_time"]
    for _, group in data.sort_values(sort_cols).groupby(["exchange", "symbol"], sort=False, observed=True):
        group = group.reset_index(drop=True)
        active_until = -1
        signal_mask = (
            group.get(candidate.gate_col, pd.Series(False, index=group.index)).fillna(False).astype(bool)
            & group.get(candidate.event_col, pd.Series(False, index=group.index)).fillna(False).astype(bool)
            & (pd.to_numeric(group.get("dynamic_all_rank"), errors="coerce") <= TOP_N)
        )
        for signal_idx, is_signal in enumerate(signal_mask):
            if not is_signal or signal_idx <= active_until:
                continue
            entry = _entry_for_short_candidate(group, signal_idx, candidate)
            if entry is None:
                continue
            entry_idx, entry_price, entry_reason = entry
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            entry_row = group.iloc[entry_idx]
            entry_gate = bool(entry_row.get(candidate.gate_col, False))
            if candidate.strict_entry_gate and not entry_gate:
                continue
            if candidate.require_gate_lost_at_entry and entry_gate:
                continue
            if funding_blocker is not None and funding_blocker(entry_row):
                continue
            rule = pick_rule(entry_row)
            exit_data = _short_exit(group, entry_idx, entry_price, rule)
            active_until = int(exit_data["exit_idx"])
            gross = float(exit_data["gross_return"])
            row = {
                "candidate": candidate.candidate,
                "family": candidate.family,
                "exchange": str(group.iloc[entry_idx]["exchange"]),
                "symbol": str(group.iloc[entry_idx]["symbol"]),
                "signal_time": pd.Timestamp(group.iloc[signal_idx]["feature_time"]),
                "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
                "exit_time": pd.Timestamp(group.iloc[int(exit_data["exit_idx"])]["bar_close_time"]),
                "entry_price": entry_price,
                "exit_price": exit_data["exit_price"],
                "entry_reason": entry_reason,
                "exit_reason": exit_data["exit_reason"],
                "gross_return": gross,
                "holding_bars": exit_data["holding_bars"],
                "holding_minutes": exit_data["holding_minutes"],
                "mfe": exit_data["mfe"],
                "mae": exit_data["mae"],
                "same_bar_ambiguity": exit_data["same_bar_ambiguity"],
                "market_gate_at_signal": True,
                "market_gate_at_entry": entry_gate,
                "btc_state_at_signal": str(group.iloc[signal_idx].get("btc_market_state", "unknown")),
                "btc_state_at_entry": str(group.iloc[entry_idx].get("btc_market_state", "unknown")),
                "volume_impulse_density_at_signal": _safe_float(
                    group.iloc[signal_idx].get("volume_impulse_density")
                ),
                "bearish_volume_impulse_density_at_signal": _safe_float(
                    group.iloc[signal_idx].get("bearish_volume_impulse_density")
                ),
                "ret_4h_percentile_at_signal": _safe_float(group.iloc[signal_idx].get("ret_4h_percentile")),
            }
            for cost in COST_BPS:
                row[f"net{int(cost)}"] = gross - 2.0 * cost / 10_000.0
            rows.append(row)
    return pd.DataFrame(rows)


def _month_cap_net(trades: pd.DataFrame, column: str = "net20", cap: float = 0.35) -> float:
    if trades.empty or column not in trades.columns:
        return np.nan
    month = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    monthly = trades.groupby(month)[column].sum()
    total = monthly.sum()
    if not np.isfinite(total) or total <= 0:
        return float(trades[column].mean())
    capped = monthly.clip(upper=total * cap)
    return float(capped.sum() / len(trades))


def _summary_row(candidate: ShortCandidate, data: pd.DataFrame, trades: pd.DataFrame) -> dict[str, object]:
    signal_count = int(
        (
            data.get(candidate.gate_col, pd.Series(False, index=data.index)).fillna(False).astype(bool)
            & data.get(candidate.event_col, pd.Series(False, index=data.index)).fillna(False).astype(bool)
            & (pd.to_numeric(data.get("dynamic_all_rank"), errors="coerce") <= TOP_N)
        ).sum()
    )
    exits = trades["exit_reason"].astype(str) if not trades.empty else pd.Series(dtype=str)
    month = pd.to_datetime(trades.get("entry_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True, errors="coerce").dt.strftime(
        "%Y-%m"
    )
    month_pnl = trades.groupby(month)["net20"].sum() if not trades.empty else pd.Series(dtype=float)
    symbol_pnl = trades.groupby("symbol")["net20"].sum() if not trades.empty else pd.Series(dtype=float)
    return {
        "candidate": candidate.candidate,
        "family": candidate.family,
        "signals": signal_count,
        "trades": int(len(trades)),
        "fill_rate": float(len(trades) / signal_count) if signal_count else np.nan,
        "gross": _safe_float(trades.get("gross_return", pd.Series(dtype=float)).mean()),
        "net10": _safe_float(trades.get("net10", pd.Series(dtype=float)).mean()),
        "net20": _safe_float(trades.get("net20", pd.Series(dtype=float)).mean()),
        "net30": _safe_float(trades.get("net30", pd.Series(dtype=float)).mean()),
        "net50": _safe_float(trades.get("net50", pd.Series(dtype=float)).mean()),
        "tp_rate": float(exits.str.startswith("tp").mean()) if not exits.empty else np.nan,
        "sl_rate": float(exits.str.startswith("sl").mean()) if not exits.empty else np.nan,
        "timeout_rate": float(exits.eq("max_hold").mean()) if not exits.empty else np.nan,
        "same_bar_ambiguity_rate": _safe_float(
            trades.get("same_bar_ambiguity", pd.Series(dtype=bool)).mean()
        ),
        "median_mfe": _safe_float(trades.get("mfe", pd.Series(dtype=float)).median()),
        "median_mae": _safe_float(trades.get("mae", pd.Series(dtype=float)).median()),
        "hit_down_10pct": _safe_float((trades.get("mfe", pd.Series(dtype=float)) >= 0.10).mean()),
        "month_cap35_net20": _month_cap_net(trades, "net20", 0.35),
        "max_month_contribution_abs20": _safe_float(month_pnl.abs().max() / month_pnl.abs().sum())
        if not month_pnl.empty and month_pnl.abs().sum() > 0
        else np.nan,
        "max_symbol_contribution_abs20": _safe_float(symbol_pnl.abs().max() / symbol_pnl.abs().sum())
        if not symbol_pnl.empty and symbol_pnl.abs().sum() > 0
        else np.nan,
    }


def write_v10_short_mirror_failure(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    paper_config = load_v07a2_config("configs/v0_7d2_cic_mir1_paper_live.yaml")
    prepared = prepare_v07d2_features_from_history(feature_path, instruments, config, paper_config, days=None)
    prepared = add_short_mirror_columns(prepared)

    report_root.mkdir(parents=True, exist_ok=True)
    frames = [simulate_short_candidate(prepared, candidate) for candidate in CANDIDATES]
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(
        [
            _summary_row(
                candidate,
                prepared,
                trades[trades["candidate"].eq(candidate.candidate)] if not trades.empty else trades,
            )
            for candidate in CANDIDATES
        ]
    ).sort_values(["net20", "trades"], ascending=[False, False])

    regime = (
        trades.groupby(["candidate", "btc_state_at_signal"], dropna=False)[["net10", "net20", "net50"]]
        .agg(["count", "mean"])
        .reset_index()
        if not trades.empty
        else pd.DataFrame()
    )

    notes = [
        "# v1.0 Short Mirror & Long-Failure Short",
        "",
        "This report tests frozen short-side analogues only. It does not alter paper-live.",
        "",
        "- S1 family: bearish mirror, bearish shock -> 1pct bounce -> rejection -> short.",
        "- S2 family: long-signal failure, bullish shock setup -> break signal low -> short.",
        "- Returns are short net returns after round-trip fee/slippage assumptions.",
    ]
    outputs = {
        "candidate_summary": report_root / "candidate_summary.csv",
        "short_trades": report_root / "short_trades.parquet",
        "regime_split": report_root / "regime_split.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["candidate_summary"], index=False)
    if not trades.empty:
        trades.to_parquet(outputs["short_trades"], index=False)
    else:
        pd.DataFrame().to_parquet(outputs["short_trades"], index=False)
    regime.to_csv(outputs["regime_split"], index=False)
    outputs["candidate_notes"].write_text("\n".join(notes), encoding="utf-8")
    return outputs


def _write_current_status(
    prepared: pd.DataFrame,
    summary: pd.DataFrame,
    signal_days: int,
    report_root: Path,
) -> Path:
    latest_time = pd.to_datetime(prepared.get("feature_time"), utc=True, errors="coerce").max()
    latest = prepared[pd.to_datetime(prepared.get("feature_time"), utc=True, errors="coerce").eq(latest_time)]
    btc_state = str(latest.get("btc_market_state", pd.Series(["unknown"])).iloc[0]) if not latest.empty else "unknown"
    bearish_density = _safe_float(latest.get("bearish_volume_impulse_density", pd.Series(dtype=float)).max())
    rows = [
        "# v1.0 Short Mirror Paper-Live Shadow",
        "",
        "- mode: paper_live_shadow_only",
        "- real_live_allowed: False",
        f"- latest_feature_time: {latest_time}",
        f"- btc_state: {btc_state}",
        f"- bearish_volume_impulse_density: {bearish_density if np.isfinite(bearish_density) else 'n/a'}",
        f"- signal_days: {signal_days}",
        "",
        "## Candidates",
    ]
    for row in summary.to_dict("records"):
        net20 = row.get("net20")
        net20_text = "n/a" if pd.isna(net20) else f"{float(net20):.4%}"
        rows.append(
            f"- {row['candidate']}: trades={int(row['trades'])}, net20={net20_text}, "
            f"sample_status={'insufficient' if int(row['trades']) < 30 else 'watch_only'}"
        )
    path = report_root / "current_status.md"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def write_v10_short_paper_live(
    prepared: pd.DataFrame,
    signal_days: int = 7,
    report_root: Path = PAPER_LIVE_ROOT,
) -> dict[str, Path]:
    data = add_short_mirror_columns(prepared)
    latest = pd.to_datetime(data["feature_time"], utc=True, errors="coerce").max()
    signal_start = latest - pd.Timedelta(days=signal_days)

    frames = [simulate_short_candidate(data, candidate) for candidate in PAPER_LIVE_CANDIDATES]
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not trades.empty:
        trades = trades[pd.to_datetime(trades["signal_time"], utc=True, errors="coerce") >= signal_start].copy()
    window_data = data[pd.to_datetime(data["feature_time"], utc=True, errors="coerce") >= signal_start].copy()
    summary = pd.DataFrame(
        [
            _summary_row(
                candidate,
                window_data,
                trades[trades["candidate"].eq(candidate.candidate)] if not trades.empty else trades,
            )
            for candidate in PAPER_LIVE_CANDIDATES
        ]
    ).sort_values(["net20", "trades"], ascending=[False, False])

    report_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidate_summary": report_root / "candidate_summary.csv",
        "paper_trades": report_root / "paper_trades.parquet",
        "current_status": report_root / "current_status.md",
    }
    summary.to_csv(outputs["candidate_summary"], index=False)
    if trades.empty:
        pd.DataFrame().to_parquet(outputs["paper_trades"], index=False)
    else:
        trades.to_parquet(outputs["paper_trades"], index=False)
    outputs["current_status"] = _write_current_status(data, summary, signal_days, report_root)
    return outputs


def run_v10_from_config(config_path: str | Path = "configs/v0_3.yaml") -> dict[str, Path]:
    from pressure_graph.config import load_config

    config = load_config(config_path)
    feature_path = Path(config.paths.data_root) / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = pd.read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v10_short_mirror_failure(feature_path, instruments, config)
