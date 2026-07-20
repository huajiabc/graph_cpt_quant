"""Directed residual-volatility transmission and OCO breakout audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    estimate_v106_betas,
    residualize_v106_returns,
)


REPORT_ROOT = Path("reports/v11_3_volatility_transmission_breakout")
PANEL_PATH = Path(
    "reports/v10_3_graph_bucket_return_diffusion/bucket_feature_panel.parquet"
)
KLINE_ROOT = Path("data/raw/bybit/klines")
CANDIDATE = "VTB1_VOL_RECEIVER_OCO"


@dataclass(frozen=True)
class V113Config:
    panel_path: Path = PANEL_PATH
    kline_root: Path = KLINE_ROOT
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    lags: tuple[int, ...] = (1, 2, 4)
    min_edge_samples: int = 1000
    shrinkage_n: int = 500
    leaders_per_follower: int = 3
    leader_score_quantile: float = 0.95
    leader_shock_quantile: float = 0.90
    compression_quantile: float = 0.50
    breadth_floor: float = 2.0 / 3.0
    gap_rank_floor: float = 0.80
    min_bucket_size: int = 2
    max_bucket_size: int = 5
    cooldown_hours: int = 4
    entry_window_bars: int = 4
    exit_horizon_bars: int = 16
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def _month_start(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.to_datetime(
        parsed.dt.strftime("%Y-%m-01"), utc=True, errors="coerce"
    )


def _period(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if month < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(
        index="feature_time",
        columns="symbol",
        values=column,
        aggfunc="last",
        observed=True,
    ).sort_index()


def load_v113_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_parquet(
        path,
        columns=["symbol", "feature_time", "ret_15m"],
    )
    panel["feature_time"] = pd.to_datetime(
        panel["feature_time"], utc=True, errors="coerce"
    )
    panel["ret_15m"] = pd.to_numeric(panel["ret_15m"], errors="coerce")
    panel["month_start"] = _month_start(panel["feature_time"])
    return (
        panel.dropna(subset=["symbol", "feature_time"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def load_v113_ohlc(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    root: Path = KLINE_ROOT,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    use_columns = ["bar_close_time", "high", "low", "close"]
    for symbol in sorted(set(symbols)):
        path = root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=use_columns)
        frame["bar_close_time"] = pd.to_datetime(
            frame["bar_close_time"], utc=True, errors="coerce"
        )
        for column in ("high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame[
                frame["bar_close_time"].ge(start)
                & frame["bar_close_time"].le(end)
            ]
            .dropna(subset=["bar_close_time", "high", "low", "close"])
            .drop_duplicates("bar_close_time", keep="last")
            .sort_values("bar_close_time")
            .set_index("bar_close_time")
        )
        if not frame.empty:
            output[symbol] = frame
    return output


def _standardized_absolute_residual(
    residual: pd.DataFrame,
    scale: pd.Series,
) -> pd.DataFrame:
    return residual.abs().div(scale).replace([np.inf, -np.inf], np.nan)


def _rolling_rv(values: pd.DataFrame, bars: int) -> pd.DataFrame:
    return values.pow(2).rolling(bars, min_periods=bars).sum().pow(0.5)


def _forward_rv(values: pd.DataFrame, bars: int) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=values.index, columns=values.columns)
    count = pd.DataFrame(0, index=values.index, columns=values.columns, dtype=int)
    for step in range(1, bars + 1):
        shifted = values.shift(-step)
        finite = shifted.notna()
        total = total.add(shifted.pow(2).fillna(0.0), fill_value=0.0)
        count = count.add(finite.astype(int), fill_value=0)
    return total.pow(0.5).where(count.eq(bars))


def build_v113_month_edges(
    absolute_shock_history: pd.DataFrame,
    month_start: pd.Timestamp,
    cfg: V113Config,
) -> pd.DataFrame:
    eligible = [
        str(column)
        for column in absolute_shock_history.columns
        if int(absolute_shock_history[column].notna().sum())
        >= cfg.min_edge_samples
    ]
    complete = (
        absolute_shock_history[eligible].dropna(how="any")
        if eligible
        else pd.DataFrame()
    )
    if len(complete) < cfg.min_edge_samples or len(eligible) < 4:
        return pd.DataFrame()
    values = complete.to_numpy(dtype=float)
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for lag in cfg.lags:
        if len(values) - lag < cfg.min_edge_samples:
            continue
        leader = values[:-lag]
        follower = values[lag:]
        leader_std = leader.std(axis=0, ddof=1)
        follower_std = follower.std(axis=0, ddof=1)
        valid = (leader_std > 0) & (follower_std > 0)
        left = np.zeros_like(leader)
        right = np.zeros_like(follower)
        left[:, valid] = (
            leader[:, valid] - leader[:, valid].mean(axis=0)
        ) / leader_std[valid]
        right[:, valid] = (
            follower[:, valid] - follower[:, valid].mean(axis=0)
        ) / follower_std[valid]
        sample_n = len(leader)
        correlation = left.T @ right / (sample_n - 1)
        shrinkage = np.sqrt(sample_n / (sample_n + cfg.shrinkage_n))
        for leader_index, leader_symbol in enumerate(eligible):
            if not valid[leader_index]:
                continue
            for follower_index, follower_symbol in enumerate(eligible):
                if leader_index == follower_index or not valid[follower_index]:
                    continue
                forward = float(correlation[leader_index, follower_index])
                reverse = float(correlation[follower_index, leader_index])
                advantage = forward - reverse
                if forward <= 0 or advantage <= 0:
                    continue
                weight = advantage * shrinkage
                key = (leader_symbol, follower_symbol)
                if key not in best or weight > float(best[key]["edge_weight"]):
                    best[key] = {
                        "month_start": month_start,
                        "leader_symbol": leader_symbol,
                        "follower_symbol": follower_symbol,
                        "lag_bars": int(lag),
                        "lag_minutes": int(lag * 15),
                        "sample_n": int(sample_n),
                        "lag_correlation": forward,
                        "reverse_correlation": reverse,
                        "direction_advantage": advantage,
                        "edge_weight": weight,
                    }
    edges = pd.DataFrame(best.values())
    if edges.empty:
        return edges
    edges = edges.sort_values(
        ["follower_symbol", "edge_weight"], ascending=[True, False]
    )
    edges["edge_rank"] = edges.groupby(
        "follower_symbol", sort=False
    ).cumcount() + 1
    return edges[edges["edge_rank"].le(cfg.leaders_per_follower)].reset_index(
        drop=True
    )


def _score_matrices(
    shocks: pd.DataFrame,
    edges: pd.DataFrame,
    individual_thresholds: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    followers = sorted(set(edges["follower_symbol"].astype(str)))
    score = pd.DataFrame(np.nan, index=shocks.index, columns=followers)
    breadth = pd.DataFrame(np.nan, index=shocks.index, columns=followers)
    for follower, group in edges.groupby("follower_symbol", sort=False):
        follower = str(follower)
        local = group[
            group["leader_symbol"].astype(str).isin(shocks.columns)
        ].copy()
        if local.empty:
            continue
        leaders = local["leader_symbol"].astype(str).tolist()
        weights = pd.to_numeric(
            local["edge_weight"], errors="coerce"
        ).to_numpy(dtype=float)
        values = shocks[leaders].to_numpy(dtype=float)
        finite = np.isfinite(values)
        weighted = np.where(finite, values * weights[None, :], 0.0)
        denominators = np.where(finite, weights[None, :], 0.0).sum(axis=1)
        score[follower] = np.divide(
            weighted.sum(axis=1),
            denominators,
            out=np.full(len(values), np.nan),
            where=denominators > 0,
        )
        thresholds = individual_thresholds.reindex(leaders).to_numpy(dtype=float)
        available = finite & np.isfinite(thresholds)[None, :]
        breadth[follower] = np.divide(
            ((values >= thresholds[None, :]) & available).sum(axis=1),
            available.sum(axis=1),
            out=np.full(len(values), np.nan),
            where=available.sum(axis=1) > 0,
        )
    return score, breadth


def _signal_state(
    context: dict[str, Any],
    edges: pd.DataFrame,
    cfg: V113Config,
) -> dict[str, pd.DataFrame]:
    history_shock: pd.DataFrame = context["history_shock"]
    target_shock: pd.DataFrame = context["target_shock"]
    shock_thresholds = history_shock.quantile(cfg.leader_shock_quantile)
    historical_score, _ = _score_matrices(
        history_shock, edges, shock_thresholds
    )
    target_score, target_breadth = _score_matrices(
        target_shock, edges, shock_thresholds
    )
    followers = target_score.columns.intersection(
        context["target_rv_1h"].columns
    )
    historical_score = historical_score.reindex(columns=followers)
    target_score = target_score.reindex(columns=followers)
    target_breadth = target_breadth.reindex(columns=followers)
    score_threshold = historical_score.quantile(cfg.leader_score_quantile)
    compression_threshold = context["history_rv_1h"].reindex(
        columns=followers
    ).quantile(cfg.compression_quantile)
    own_rv = context["target_rv_1h"].reindex(columns=followers)
    score_ratio = target_score.div(score_threshold).replace(
        [np.inf, -np.inf], np.nan
    )
    compression_ratio = own_rv.div(compression_threshold).replace(
        [np.inf, -np.inf], np.nan
    )
    gap = score_ratio - compression_ratio
    gap_rank = gap.rank(axis=1, pct=True, method="average")
    eligible = (
        target_score.ge(score_threshold)
        & target_breadth.ge(cfg.breadth_floor)
        & own_rv.le(compression_threshold)
        & gap_rank.ge(cfg.gap_rank_floor)
    )
    return {
        "score": target_score,
        "breadth": target_breadth,
        "own_rv_1h": own_rv,
        "score_threshold": pd.DataFrame(
            np.tile(score_threshold.to_numpy(), (len(target_score), 1)),
            index=target_score.index,
            columns=followers,
        ),
        "compression_threshold": pd.DataFrame(
            np.tile(compression_threshold.to_numpy(), (len(target_score), 1)),
            index=target_score.index,
            columns=followers,
        ),
        "gap": gap,
        "eligible": eligible,
    }


def build_v113_graph_and_contexts(
    panel: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame],
    cfg: V113Config,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    months = sorted(panel["month_start"].dropna().unique())
    for raw_month in months[1:]:
        month = pd.Timestamp(raw_month)
        history = panel[
            panel["feature_time"].ge(
                month - pd.Timedelta(days=cfg.lookback_days)
            )
            & panel["feature_time"].lt(month)
        ]
        target = panel[panel["month_start"].eq(month)]
        if history.empty or target.empty:
            continue
        history_return = _pivot(history, "ret_15m")
        target_return = _pivot(target, "ret_15m")
        betas = estimate_v106_betas(history_return)
        history_residual = residualize_v106_returns(history_return, betas)
        target_residual = residualize_v106_returns(target_return, betas)
        scale = history_residual.std(ddof=1).replace(0.0, np.nan)
        history_shock = _standardized_absolute_residual(
            history_residual, scale
        )
        edges = build_v113_month_edges(history_shock, month, cfg)
        if edges.empty:
            continue
        graph_symbols = sorted(
            (
                set(edges["leader_symbol"].astype(str))
                | set(edges["follower_symbol"].astype(str))
            )
            & set(target_residual.columns)
            & set(ohlc)
        )
        edges = edges[
            edges["leader_symbol"].astype(str).isin(graph_symbols)
            & edges["follower_symbol"].astype(str).isin(graph_symbols)
        ].reset_index(drop=True)
        if edges.empty:
            continue
        edge_frames.append(edges)
        combined_residual = pd.concat(
            [
                history_residual.reindex(columns=graph_symbols),
                target_residual.reindex(columns=graph_symbols),
            ]
        ).sort_index()
        combined_shock = _standardized_absolute_residual(
            combined_residual, scale.reindex(graph_symbols)
        )
        combined_rv_1h = _rolling_rv(combined_residual, 4)
        target_times = target_residual.index[
            target_residual.index.minute == 0
        ]
        target_times = target_times.intersection(combined_residual.index)
        target_residual_local = target_residual.reindex(
            index=target_residual.index, columns=graph_symbols
        )
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "history_shock": history_shock.reindex(columns=graph_symbols),
            "history_rv_1h": combined_rv_1h.reindex(
                index=history_residual.index, columns=graph_symbols
            ),
            "target_shock": combined_shock.reindex(
                index=target_times, columns=graph_symbols
            ),
            "target_rv_1h": combined_rv_1h.reindex(
                index=target_times, columns=graph_symbols
            ),
            "prior_rv_4h": _rolling_rv(target_residual_local, 16).reindex(
                index=target_times
            ),
            "future_rv_4h": _forward_rv(target_residual_local, 16).reindex(
                index=target_times
            ),
            "ohlc": ohlc,
            "oco_cache": {},
        }
    all_edges = (
        pd.concat(edge_frames, ignore_index=True)
        if edge_frames
        else pd.DataFrame()
    )
    return all_edges, contexts


def simulate_v113_oco_leg(
    ohlc: pd.DataFrame,
    signal_time: pd.Timestamp,
    cfg: V113Config,
) -> dict[str, Any]:
    signal_time = pd.Timestamp(signal_time)
    step = pd.Timedelta(minutes=15)
    reference_times = pd.date_range(
        signal_time - step * 3,
        signal_time,
        freq=step,
    )
    reference = ohlc.reindex(reference_times)
    if len(reference) != 4 or reference[["high", "low"]].isna().any().any():
        return {"filled": False, "ambiguous": False, "reason": "missing_reference"}
    reference_high = float(reference["high"].max())
    reference_low = float(reference["low"].min())
    entry_times = pd.date_range(
        signal_time + step,
        signal_time + step * cfg.entry_window_bars,
        freq=step,
    )
    entry_window = ohlc.reindex(entry_times)
    exit_time = signal_time + step * cfg.exit_horizon_bars
    if exit_time not in ohlc.index or not np.isfinite(ohlc.at[exit_time, "close"]):
        return {"filled": False, "ambiguous": False, "reason": "missing_exit"}
    exit_price = float(ohlc.at[exit_time, "close"])
    for timestamp, bar in entry_window.iterrows():
        if not np.isfinite(bar.get("high", np.nan)) or not np.isfinite(
            bar.get("low", np.nan)
        ):
            continue
        long_hit = float(bar["high"]) >= reference_high
        short_hit = float(bar["low"]) <= reference_low
        if long_hit and short_hit:
            return {
                "filled": False,
                "ambiguous": True,
                "reason": "same_bar_dual_trigger",
                "entry_time": timestamp,
            }
        if long_hit:
            return {
                "filled": True,
                "ambiguous": False,
                "reason": "long_break",
                "side": "long",
                "entry_time": timestamp,
                "entry_price": reference_high,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "gross_return": exit_price / reference_high - 1.0,
            }
        if short_hit:
            return {
                "filled": True,
                "ambiguous": False,
                "reason": "short_break",
                "side": "short",
                "entry_time": timestamp,
                "entry_price": reference_low,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "gross_return": 1.0 - exit_price / reference_low,
            }
    return {"filled": False, "ambiguous": False, "reason": "unfilled"}


def _cached_oco(
    context: dict[str, Any],
    symbol: str,
    timestamp: pd.Timestamp,
    cfg: V113Config,
) -> dict[str, Any]:
    key = (symbol, pd.Timestamp(timestamp))
    cache: dict[tuple[str, pd.Timestamp], dict[str, Any]] = context["oco_cache"]
    if key not in cache:
        cache[key] = simulate_v113_oco_leg(
            context["ohlc"][symbol], pd.Timestamp(timestamp), cfg
        )
    return cache[key]


def build_v113_events(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V113Config,
    signal_shift_hours: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    last_event: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        local_edges = edges[edges["month_start"].eq(month)]
        if local_edges.empty:
            continue
        state = _signal_state(context, local_edges, cfg)
        if signal_shift_hours:
            for key in state:
                state[key] = state[key].shift(signal_shift_hours)
        eligible = state["eligible"]
        for timestamp in eligible.index:
            if last_event is not None and pd.Timestamp(timestamp) - last_event < cooldown:
                continue
            symbols = eligible.columns[eligible.loc[timestamp].eq(True)]
            if len(symbols) < cfg.min_bucket_size:
                continue
            selected = (
                state["gap"]
                .loc[timestamp, symbols]
                .sort_values(ascending=False)
                .head(cfg.max_bucket_size)
                .index.tolist()
            )
            leg_rows = []
            for symbol in selected:
                prior_rv = float(context["prior_rv_4h"].at[timestamp, symbol])
                future_rv = float(context["future_rv_4h"].at[timestamp, symbol])
                expansion = (
                    future_rv / prior_rv
                    if np.isfinite(prior_rv)
                    and prior_rv > 0
                    and np.isfinite(future_rv)
                    else np.nan
                )
                result = _cached_oco(context, symbol, timestamp, cfg)
                leg_rows.append(
                    {
                        "symbol": symbol,
                        "future_rv_expansion_4h": expansion,
                        **result,
                    }
                )
            filled = [row for row in leg_rows if row.get("filled")]
            ambiguous = [row for row in leg_rows if row.get("ambiguous")]
            gross = (
                float(np.mean([row["gross_return"] for row in filled]))
                if len(filled) >= cfg.min_bucket_size
                else np.nan
            )
            expansion_values = [
                float(row["future_rv_expansion_4h"])
                for row in leg_rows
                if np.isfinite(row["future_rv_expansion_4h"])
            ]
            rows.append(
                {
                    "candidate": CANDIDATE,
                    "feature_time": timestamp,
                    "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                    "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                    "period": context["period"],
                    "selected_size": len(selected),
                    "selected_symbols": "|".join(selected),
                    "filled_size": len(filled),
                    "filled_symbols": "|".join(
                        str(row["symbol"]) for row in filled
                    ),
                    "long_legs": sum(row.get("side") == "long" for row in filled),
                    "short_legs": sum(row.get("side") == "short" for row in filled),
                    "ambiguous_legs": len(ambiguous),
                    "unfilled_legs": len(selected) - len(filled) - len(ambiguous),
                    "mean_leader_score": float(
                        state["score"].loc[timestamp, selected].mean()
                    ),
                    "mean_leader_breadth": float(
                        state["breadth"].loc[timestamp, selected].mean()
                    ),
                    "mean_transmission_gap": float(
                        state["gap"].loc[timestamp, selected].mean()
                    ),
                    "mean_future_rv_expansion_4h": float(
                        np.mean(expansion_values)
                        if expansion_values
                        else np.nan
                    ),
                    "breakout_gross_4h": gross,
                    "breakout_net_4h_20bp": gross - 0.002,
                    "breakout_net_4h_30bp": gross - 0.003,
                    "breakout_net_4h_50bp": gross - 0.005,
                }
            )
            last_event = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def summarize_v113(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        sample = events if scope == "all" else events[events["period"].eq(scope)]
        traded = sample.dropna(subset=["breakout_gross_4h"])
        selected_legs = int(sample["selected_size"].sum())
        rows.append(
            {
                "scope": scope,
                "event_observations": int(len(sample)),
                "traded_portfolios": int(len(traded)),
                "active_days": int(sample["entry_day"].nunique()),
                "active_months": int(sample["entry_month"].nunique()),
                "mean_selected_size": float(sample["selected_size"].mean()),
                "oco_fill_rate": float(
                    sample["filled_size"].sum() / selected_legs
                    if selected_legs
                    else np.nan
                ),
                "ambiguity_rate": float(
                    sample["ambiguous_legs"].sum() / selected_legs
                    if selected_legs
                    else np.nan
                ),
                "long_share": float(
                    sample["long_legs"].sum()
                    / (sample["long_legs"].sum() + sample["short_legs"].sum())
                    if sample["long_legs"].sum() + sample["short_legs"].sum()
                    else np.nan
                ),
                "mean_future_rv_expansion_4h": float(
                    sample["mean_future_rv_expansion_4h"].mean()
                ),
                "mean_breakout_gross_4h": float(
                    traded["breakout_gross_4h"].mean()
                ),
                "mean_breakout_net_4h_20bp": float(
                    traded["breakout_net_4h_20bp"].mean()
                ),
                "mean_breakout_net_4h_30bp": float(
                    traded["breakout_net_4h_30bp"].mean()
                ),
                "mean_breakout_net_4h_50bp": float(
                    traded["breakout_net_4h_50bp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def randomize_v113_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V113Config,
) -> pd.DataFrame:
    frames = []
    for month, context in contexts.items():
        local = edges[edges["month_start"].eq(month)].copy()
        if local.empty:
            continue
        symbols = sorted(context["target_shock"].columns)
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month.month)
        leaders = []
        for follower, group in local.groupby("follower_symbol", sort=False):
            choices = [symbol for symbol in symbols if symbol != str(follower)]
            sampled = rng.choice(
                choices,
                size=len(group),
                replace=len(choices) < len(group),
            )
            leaders.extend(zip(group.index, sampled))
        for index, leader in leaders:
            local.at[index, "leader_symbol"] = str(leader)
        frames.append(local)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def random_v113_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V113Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v113_edges(edges, contexts, iteration, cfg)
        events = build_v113_events(contexts, randomized, cfg)
        traded = events.dropna(subset=["breakout_net_4h_20bp"])
        rows.append(
            {
                "iteration": iteration,
                "event_observations": int(len(events)),
                "traded_portfolios": int(len(traded)),
                "mean_future_rv_expansion_4h": float(
                    events["mean_future_rv_expansion_4h"].mean()
                ),
                "mean_breakout_net_4h_20bp": float(
                    traded["breakout_net_4h_20bp"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(events: pd.DataFrame, cfg: V113Config) -> tuple[float, float]:
    sample = events.dropna(subset=["breakout_net_4h_20bp"])
    daily = [
        group["breakout_net_4h_20bp"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day")
        if len(group)
    ]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(
            float(np.mean(np.concatenate([daily[index] for index in chosen])))
        )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _positive_month_share(events: pd.DataFrame) -> float:
    sample = events.dropna(subset=["breakout_net_4h_20bp"])
    values = sample.groupby("entry_month")["breakout_net_4h_20bp"].sum()
    positive = values.clip(lower=0.0)
    return float(
        positive.max() / positive.sum() if positive.sum() > 0 else np.inf
    )


def _positive_receiver_share(events: pd.DataFrame) -> float:
    contributions: dict[str, float] = {}
    for row in events.dropna(subset=["breakout_net_4h_20bp"]).itertuples(
        index=False
    ):
        symbols = [value for value in str(row.filled_symbols).split("|") if value]
        if not symbols:
            continue
        amount = float(row.breakout_net_4h_20bp) / len(symbols)
        for symbol in symbols:
            contributions[symbol] = contributions.get(symbol, 0.0) + amount
    positive = np.array([max(value, 0.0) for value in contributions.values()])
    return float(positive.max() / positive.sum() if positive.sum() > 0 else np.inf)


def audit_v113(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V113Config,
) -> pd.DataFrame:
    lookup = {row.scope: row for row in summary.itertuples(index=False)}
    traded = real.dropna(subset=["breakout_net_4h_20bp"]).sort_values(
        "feature_time"
    )
    shifted_mean = float(shifted["breakout_net_4h_20bp"].mean())
    family = controls["mean_breakout_net_4h_20bp"].dropna()
    percentile = float(
        family.lt(lookup["all"].mean_breakout_net_4h_20bp).mean()
    )
    ci_low, ci_high = _bootstrap(real, cfg)
    chronological = [
        float(traded.iloc[index]["breakout_net_4h_20bp"].mean())
        for index in np.array_split(np.arange(len(traded)), 5)
        if len(index)
    ]
    month_share = _positive_month_share(real)
    receiver_share = _positive_receiver_share(real)
    gates = {
        "full_observations_100": lookup["all"].traded_portfolios >= 100,
        "validation_observations_25": lookup["validation"].traded_portfolios >= 25,
        "holdout_observations_25": lookup["holdout"].traded_portfolios >= 25,
        "validation_net20_positive": lookup[
            "validation"
        ].mean_breakout_net_4h_20bp
        > 0,
        "holdout_net20_positive": lookup["holdout"].mean_breakout_net_4h_20bp
        > 0,
        "full_net30_positive": lookup["all"].mean_breakout_net_4h_30bp > 0,
        "validation_vol_expansion_positive": lookup[
            "validation"
        ].mean_future_rv_expansion_4h
        > 1,
        "holdout_vol_expansion_positive": lookup[
            "holdout"
        ].mean_future_rv_expansion_4h
        > 1,
        "random_family_p90": percentile >= 0.90,
        "beats_shifted": lookup["all"].mean_breakout_net_4h_20bp
        > shifted_mean,
        "bootstrap_lower_positive": ci_low > 0,
        "five_chrono_nonnegative": bool(chronological)
        and min(chronological) >= 0,
        "month_share_below_35pct": month_share <= 0.35,
        "receiver_share_below_35pct": receiver_share <= 0.35,
    }
    eligible = all(gates.values())
    return pd.DataFrame(
        [
            {
                "candidate": CANDIDATE,
                "eligible": eligible,
                "verdict": "retrospective_forward_watch_only"
                if eligible
                else "reject_volatility_transmission_breakout",
                "full_net20": lookup["all"].mean_breakout_net_4h_20bp,
                "validation_net20": lookup[
                    "validation"
                ].mean_breakout_net_4h_20bp,
                "holdout_net20": lookup["holdout"].mean_breakout_net_4h_20bp,
                "full_net30": lookup["all"].mean_breakout_net_4h_30bp,
                "full_vol_expansion": lookup[
                    "all"
                ].mean_future_rv_expansion_4h,
                "validation_vol_expansion": lookup[
                    "validation"
                ].mean_future_rv_expansion_4h,
                "holdout_vol_expansion": lookup[
                    "holdout"
                ].mean_future_rv_expansion_4h,
                "shifted_net20": shifted_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "chronological_means": "|".join(
                    f"{value:.10f}" for value in chronological
                ),
                "max_positive_month_share": month_share,
                "max_positive_receiver_share": receiver_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        ]
    )


def write_v113_volatility_transmission_breakout(
    cfg: V113Config = V113Config(),
) -> dict[str, Path]:
    panel = load_v113_panel(cfg.panel_path)
    start = pd.Timestamp(panel["feature_time"].min()) - pd.Timedelta(hours=2)
    end = pd.Timestamp(panel["feature_time"].max()) + pd.Timedelta(hours=5)
    ohlc = load_v113_ohlc(
        panel["symbol"].astype(str).unique().tolist(), start, end, cfg.kline_root
    )
    edges, contexts = build_v113_graph_and_contexts(panel, ohlc, cfg)
    real = build_v113_events(contexts, edges, cfg)
    shifted = build_v113_events(
        contexts, edges, cfg, signal_shift_hours=24
    )
    summary = summarize_v113(real)
    controls = random_v113_controls(contexts, edges, cfg)
    audit = audit_v113(real, shifted, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "edges": root / "monthly_directed_volatility_edges.parquet",
        "events": root / "volatility_transmission_events.parquet",
        "portfolios": root / "breakout_portfolios.parquet",
        "shifted": root / "shifted_breakout_events.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    edges.to_parquet(outputs["edges"], index=False)
    real.to_parquet(outputs["events"], index=False)
    real.dropna(subset=["breakout_gross_4h"]).to_parquet(
        outputs["portfolios"], index=False
    )
    shifted.to_parquet(outputs["shifted"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    row = audit.iloc[0]
    lines = [
        "# v11.3 Directed Volatility-Transmission Breakout",
        "",
        f"Status: `{row['verdict']}`.",
        "",
        f"- traded portfolios: {int(summary.loc[summary['scope'].eq('all'), 'traded_portfolios'].iloc[0])}",
        f"- future 4h RV expansion: {row['full_vol_expansion']:.4f}x",
        f"- net20: {row['full_net20']:.4%}",
        f"- validation net20: {row['validation_net20']:.4%}",
        f"- holdout net20: {row['holdout_net20']:.4%}",
        f"- random percentile: {row['random_family_percentile']:.1%}",
        "",
        "Research only. No PaperLive, leverage, or real-order permission changed.",
    ]
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
