"""BTC price/active-flow shocks propagated into 15-minute alt laggards."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


KLINE_ROOT = Path("data/raw/binance/klines")
REPORT_ROOT = Path("reports/v17_8_btc_confirmed_flow_laggard")
FINDINGS_PATH = Path("docs/v178_btc_confirmed_flow_laggard_findings_2026_07_16.md")
BTC = "BTCUSDT"
EXCLUDED_RECEIVERS = {BTC, "XAUTUSDT"}
RAW_CANDIDATE = "BFR1_CONFIRMED_BTC_LAGGARD_CATCHUP"
NEUTRAL_CANDIDATE = "BFR2_BTC_NEUTRAL_LAGGARD_CATCHUP"
CANDIDATES = (RAW_CANDIDATE, NEUTRAL_CANDIDATE)


@dataclass(frozen=True)
class V178Config:
    source_lookback_bars: int = 30 * 96
    source_min_bars: int = 20 * 96
    source_return_quantile: float = 0.975
    source_flow_quantile: float = 0.80
    source_turnover_quantile: float = 0.75
    cooldown_bars: int = 4
    graph_lookback_days: int = 30
    graph_min_samples: int = 2_000
    receiver_pool_size: int = 10
    max_laggards: int = 5
    min_laggards: int = 3
    holding_bars: int = 2
    raw_primary_cost: float = 0.0020
    raw_stress_cost: float = 0.0030
    neutral_primary_cost: float = 0.0030
    neutral_stress_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 17_800


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2026-03-01", tz="UTC"):
        return "validation"
    return "holdout"


def load_v178_market_data(
    root: Path = KLINE_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_frames: list[pd.Series] = []
    btc_source: pd.DataFrame | None = None
    for path in sorted(root.glob("*.parquet")):
        symbol = path.stem.upper()
        if symbol in EXCLUDED_RECEIVERS - {BTC}:
            continue
        columns = ["bar_close_time", "close"]
        if symbol == BTC:
            columns.extend(["turnover", "taker_buy_quote"])
        frame = pd.read_parquet(path, columns=columns)
        frame["bar_close_time"] = pd.to_datetime(
            frame["bar_close_time"], utc=True, errors="coerce"
        )
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = (
            frame.dropna(subset=["bar_close_time", "close"])
            .drop_duplicates("bar_close_time", keep="last")
            .sort_values("bar_close_time")
            .set_index("bar_close_time")
        )
        close_frames.append(frame["close"].rename(symbol))
        if symbol == BTC:
            for column in ("turnover", "taker_buy_quote"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            btc_source = frame[["close", "turnover", "taker_buy_quote"]].copy()
    if btc_source is None:
        raise FileNotFoundError("BTCUSDT 15-minute source bars are missing")
    close = pd.concat(close_frames, axis=1).sort_index()
    regular = pd.date_range(close.index.min(), close.index.max(), freq="15min", tz="UTC")
    close = close.reindex(regular)
    close.index.name = "feature_time"
    btc_source = btc_source.reindex(regular)
    btc_source.index.name = "feature_time"
    return close, btc_source


def build_btc_source_signals(
    btc_source: pd.DataFrame,
    cfg: V178Config = V178Config(),
    return_quantile: float | None = None,
    shift_bars: int = 0,
) -> pd.DataFrame:
    frame = btc_source.copy()
    frame.index.name = "feature_time"
    frame["btc_return_15m"] = frame["close"].pct_change(fill_method=None)
    frame["taker_imbalance"] = (
        2.0 * frame["taker_buy_quote"] / frame["turnover"] - 1.0
    )
    absolute_return = frame["btc_return_15m"].abs()
    absolute_flow = frame["taker_imbalance"].abs()
    history_return = absolute_return.shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    )
    history_flow = absolute_flow.shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    )
    history_turnover = frame["turnover"].shift(1).rolling(
        cfg.source_lookback_bars, min_periods=cfg.source_min_bars
    )
    cutoff = cfg.source_return_quantile if return_quantile is None else return_quantile
    frame["return_threshold"] = history_return.quantile(cutoff)
    frame["flow_threshold"] = history_flow.quantile(cfg.source_flow_quantile)
    frame["turnover_threshold"] = history_turnover.quantile(
        cfg.source_turnover_quantile
    )
    frame["direction"] = np.sign(frame["btc_return_15m"])
    frame["signed_flow"] = frame["direction"] * frame["taker_imbalance"]
    frame["source_eligible"] = (
        absolute_return.ge(frame["return_threshold"])
        & frame["signed_flow"].ge(frame["flow_threshold"])
        & frame["turnover"].ge(frame["turnover_threshold"])
        & frame["direction"].ne(0)
    )
    accepted: list[pd.Timestamp] = []
    last_time: pd.Timestamp | None = None
    cooldown = pd.Timedelta(minutes=15 * cfg.cooldown_bars)
    for timestamp in frame.index[frame["source_eligible"].eq(True)]:
        timestamp = pd.Timestamp(timestamp)
        if last_time is None or timestamp - last_time >= cooldown:
            accepted.append(timestamp)
            last_time = timestamp
    result = frame.loc[accepted].reset_index()
    result["source_feature_time"] = result["feature_time"]
    if shift_bars:
        result["feature_time"] += pd.Timedelta(minutes=15 * shift_bars)
    result["return_quantile"] = cutoff
    result["signal_shift_bars"] = shift_bars
    return result


def build_monthly_btc_receiver_graph(
    returns: pd.DataFrame,
    first_month: pd.Timestamp,
    last_month: pd.Timestamp,
    cfg: V178Config = V178Config(),
) -> pd.DataFrame:
    months = pd.date_range(
        pd.Timestamp(first_month).tz_convert("UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ),
        pd.Timestamp(last_month).tz_convert("UTC").replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ),
        freq="MS",
    )
    alts = sorted(set(returns.columns) - EXCLUDED_RECEIVERS)
    rows: list[dict[str, object]] = []
    for month in months:
        history = returns[
            (returns.index >= month - pd.Timedelta(days=cfg.graph_lookback_days))
            & (returns.index < month)
        ]
        local: list[dict[str, object]] = []
        for alt in alts:
            beta_pair = history[[BTC, alt]].dropna()
            if len(beta_pair) < cfg.graph_min_samples or beta_pair[BTC].var(ddof=1) <= 0:
                continue
            beta = float(beta_pair[alt].cov(beta_pair[BTC]) / beta_pair[BTC].var(ddof=1))
            residual = history[alt] - beta * history[BTC]
            pair = pd.DataFrame(
                {
                    "btc": history[BTC],
                    "alt": history[alt],
                    "residual_next": residual.shift(-1),
                    "btc_next": history[BTC].shift(-1),
                }
            ).dropna()
            sample_n = len(pair)
            if sample_n < cfg.graph_min_samples:
                continue
            signed_forward = float(pair["btc"].corr(pair["residual_next"], method="spearman"))
            absolute_forward = float(
                pair["btc"].abs().corr(pair["residual_next"].abs(), method="spearman")
            )
            absolute_reverse = float(
                pair["alt"].abs().corr(pair["btc_next"].abs(), method="spearman")
            )
            advantage = absolute_forward - absolute_reverse
            local.append(
                {
                    "graph_month": month,
                    "receiver": alt,
                    "sample_n": sample_n,
                    "btc_beta": beta,
                    "signed_forward_correlation": signed_forward,
                    "absolute_forward_correlation": absolute_forward,
                    "absolute_reverse_correlation": absolute_reverse,
                    "absolute_direction_advantage": advantage,
                    "receiver_score": signed_forward + 0.5 * advantage,
                }
            )
        local_frame = pd.DataFrame(local)
        if local_frame.empty:
            continue
        selected = local_frame[
            local_frame["signed_forward_correlation"].gt(0)
        ].sort_values(
            ["receiver_score", "signed_forward_correlation"], ascending=False
        ).head(cfg.receiver_pool_size)
        rank = {
            receiver: index + 1
            for index, receiver in enumerate(selected["receiver"].astype(str))
        }
        for row in local:
            receiver = str(row["receiver"])
            row["selected"] = receiver in rank
            row["receiver_rank"] = rank.get(receiver, math.nan)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["graph_month", "selected", "receiver_rank", "receiver"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def _month(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(timestamp).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


def _receiver_rows(graph: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    return graph[
        graph["graph_month"].eq(_month(timestamp)) & graph["selected"].eq(True)
    ].sort_values("receiver_rank")


def build_v178_events(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: V178Config = V178Config(),
    holding_bars: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon = cfg.holding_bars if holding_bars is None else holding_bars
    event_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        entry_time = pd.Timestamp(signal.feature_time)
        exit_time = entry_time + pd.Timedelta(minutes=15 * horizon)
        local = _receiver_rows(graph, entry_time)
        receivers = local["receiver"].astype(str).tolist()
        if (
            len(receivers) < cfg.min_laggards
            or entry_time not in close.index
            or exit_time not in close.index
        ):
            continue
        beta = local.set_index("receiver")["btc_beta"].astype(float)
        current = returns.reindex(index=[entry_time], columns=[BTC, *receivers]).iloc[0]
        residual = current[receivers] - beta.reindex(receivers) * float(current[BTC])
        directed_residual = float(signal.direction) * residual
        laggards = directed_residual[directed_residual.le(0)].sort_values().head(
            cfg.max_laggards
        )
        if len(laggards) < cfg.min_laggards:
            continue
        chosen = laggards.index.astype(str).tolist()
        required = [BTC, *chosen]
        if close.loc[[entry_time, exit_time], required].isna().any().any():
            continue
        future = close.loc[exit_time, required] / close.loc[entry_time, required] - 1.0
        direction = float(signal.direction)
        raw_gross = direction * float(future[chosen].mean())
        mean_beta = float(beta.reindex(chosen).mean())
        neutral_gross = direction * (
            float(future[chosen].mean()) - mean_beta * float(future[BTC])
        ) / (1.0 + abs(mean_beta))
        common = {
            **signal._asdict(),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "holding_bars": horizon,
            "period": _period(entry_time),
            "entry_day": entry_time.strftime("%Y-%m-%d"),
            "entry_month": entry_time.strftime("%Y-%m"),
            "receiver_pool_size": len(receivers),
            "laggard_count": len(chosen),
            "laggards": "|".join(chosen),
            "mean_directed_residual_15m": float(laggards.mean()),
            "mean_laggard_beta": mean_beta,
            "btc_future_return": float(future[BTC]),
            "mean_laggard_future_return": float(future[chosen].mean()),
        }
        event_rows.extend(
            [
                {
                    **common,
                    "candidate": RAW_CANDIDATE,
                    "gross_return": raw_gross,
                    "primary_net_return": raw_gross - cfg.raw_primary_cost,
                    "stress_net_return": raw_gross - cfg.raw_stress_cost,
                },
                {
                    **common,
                    "candidate": NEUTRAL_CANDIDATE,
                    "gross_return": neutral_gross,
                    "primary_net_return": neutral_gross - cfg.neutral_primary_cost,
                    "stress_net_return": neutral_gross - cfg.neutral_stress_cost,
                },
            ]
        )
        for receiver in chosen:
            selected_rows.append(
                {
                    "entry_time": entry_time,
                    "receiver": receiver,
                    "direction": direction,
                    "btc_beta": float(beta[receiver]),
                    "directed_residual_15m": float(directed_residual[receiver]),
                    "future_return": float(future[receiver]),
                }
            )
    events = pd.DataFrame(event_rows)
    selected = pd.DataFrame(selected_rows)
    if not events.empty:
        events = events.sort_values(["entry_time", "candidate"]).reset_index(drop=True)
    return events, selected


def summarize_v178(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        candidate_rows = events[events["candidate"].eq(candidate)]
        for scope in ("all", "development", "validation", "holdout"):
            sample = (
                candidate_rows
                if scope == "all"
                else candidate_rows[candidate_rows["period"].eq(scope)]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "events": int(len(sample)),
                    "active_days": int(sample["entry_day"].nunique()),
                    "active_months": int(sample["entry_month"].nunique()),
                    "mean_laggards": float(sample["laggard_count"].mean())
                    if len(sample)
                    else math.nan,
                    "mean_gross_bp": float(sample["gross_return"].mean() * 10_000)
                    if len(sample)
                    else math.nan,
                    "mean_primary_net_bp": float(
                        sample["primary_net_return"].mean() * 10_000
                    )
                    if len(sample)
                    else math.nan,
                    "mean_stress_net_bp": float(
                        sample["stress_net_return"].mean() * 10_000
                    )
                    if len(sample)
                    else math.nan,
                    "win_rate_primary": float(sample["primary_net_return"].gt(0).mean())
                    if len(sample)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _random_controls(
    signals: pd.DataFrame,
    graph: pd.DataFrame,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: V178Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.random_iterations):
        values = {candidate: [] for candidate in CANDIDATES}
        for signal in signals.itertuples(index=False):
            entry_time = pd.Timestamp(signal.feature_time)
            exit_time = entry_time + pd.Timedelta(minutes=15 * cfg.holding_bars)
            if entry_time not in close.index or exit_time not in close.index:
                continue
            local = graph[
                graph["graph_month"].eq(_month(entry_time))
                & graph["signed_forward_correlation"].gt(0)
            ]
            if len(local) < cfg.receiver_pool_size:
                continue
            sampled_names = rng.choice(
                local["receiver"].astype(str).to_numpy(),
                size=cfg.receiver_pool_size,
                replace=False,
            ).tolist()
            sampled = local.set_index("receiver").loc[sampled_names]
            beta = sampled["btc_beta"].astype(float)
            current = returns.reindex(
                index=[entry_time], columns=[BTC, *sampled_names]
            ).iloc[0]
            residual = current[sampled_names] - beta * float(current[BTC])
            directed = float(signal.direction) * residual
            laggards = directed[directed.le(0)].sort_values().head(cfg.max_laggards)
            if len(laggards) < cfg.min_laggards:
                continue
            chosen = laggards.index.astype(str).tolist()
            required = [BTC, *chosen]
            if close.loc[[entry_time, exit_time], required].isna().any().any():
                continue
            future = close.loc[exit_time, required] / close.loc[entry_time, required] - 1.0
            direction = float(signal.direction)
            raw = direction * float(future[chosen].mean()) - cfg.raw_primary_cost
            mean_beta = float(beta.reindex(chosen).mean())
            neutral = direction * (
                float(future[chosen].mean()) - mean_beta * float(future[BTC])
            ) / (1.0 + abs(mean_beta)) - cfg.neutral_primary_cost
            values[RAW_CANDIDATE].append(raw)
            values[NEUTRAL_CANDIDATE].append(neutral)
        means: dict[str, float] = {}
        for candidate in CANDIDATES:
            means[candidate] = (
                float(np.mean(values[candidate])) if values[candidate] else math.nan
            )
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "events": len(values[candidate]),
                    "mean_primary_net_return": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "events": max(len(values[candidate]) for candidate in CANDIDATES),
                "mean_primary_net_return": max(finite) if finite else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(sample: pd.DataFrame, cfg: V178Config, offset: int) -> tuple[float, float]:
    daily = [
        group["primary_net_return"].to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    if not daily:
        return math.nan, math.nan
    rng = np.random.default_rng(cfg.seed + offset)
    means = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        means.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _month_share(sample: pd.DataFrame) -> float:
    monthly = sample.groupby("entry_month")["primary_net_return"].sum().clip(lower=0)
    return float(monthly.max() / monthly.sum()) if monthly.sum() > 0 else math.inf


def audit_v178(
    events: pd.DataFrame,
    summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V178Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = random_controls.loc[
        random_controls["candidate"].eq("FAMILY_MAX"), "mean_primary_net_return"
    ].dropna()
    gate_rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(CANDIDATES):
        candidate_summary = summary[summary["candidate"].eq(candidate)].set_index("scope")
        sample = events[events["candidate"].eq(candidate)]
        low, high = _bootstrap(sample, cfg, candidate_index)
        real_mean = float(sample["primary_net_return"].mean()) if len(sample) else math.nan
        reversed_mean = float((-sample["gross_return"] - (
            cfg.raw_primary_cost if candidate == RAW_CANDIDATE else cfg.neutral_primary_cost
        )).mean()) if len(sample) else math.nan
        delayed = delayed_summary[
            delayed_summary["candidate"].eq(candidate)
            & delayed_summary["scope"].eq("all")
        ]
        delayed_mean = (
            float(delayed["mean_primary_net_bp"].iloc[0]) / 10_000
            if not delayed.empty
            else math.nan
        )
        local_sensitivity = sensitivity[
            sensitivity["candidate"].eq(candidate) & sensitivity["scope"].eq("all")
        ].set_index("return_quantile")
        local_horizon = horizon_summary[
            horizon_summary["candidate"].eq(candidate)
            & horizon_summary["scope"].eq("all")
        ].set_index("holding_bars")
        random_percentile = float(family.le(real_mean).mean()) if len(family) else math.nan
        month_share = _month_share(sample)
        checks: dict[str, tuple[bool, float]] = {
            "full_events_100": (
                int(candidate_summary.loc["all", "events"]) >= 100,
                float(candidate_summary.loc["all", "events"]),
            ),
            "validation_events_20": (
                int(candidate_summary.loc["validation", "events"]) >= 20,
                float(candidate_summary.loc["validation", "events"]),
            ),
            "holdout_events_25": (
                int(candidate_summary.loc["holdout", "events"]) >= 25,
                float(candidate_summary.loc["holdout", "events"]),
            ),
            "development_primary_positive": (
                candidate_summary.loc["development", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["development", "mean_primary_net_bp"]),
            ),
            "validation_primary_positive": (
                candidate_summary.loc["validation", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["validation", "mean_primary_net_bp"]),
            ),
            "holdout_primary_positive": (
                candidate_summary.loc["holdout", "mean_primary_net_bp"] > 0,
                float(candidate_summary.loc["holdout", "mean_primary_net_bp"]),
            ),
            "full_stress_positive": (
                candidate_summary.loc["all", "mean_stress_net_bp"] > 0,
                float(candidate_summary.loc["all", "mean_stress_net_bp"]),
            ),
            "bootstrap_lower_positive": (low > 0, low * 10_000),
            "random_family_percentile_95": (
                random_percentile >= 0.95,
                random_percentile,
            ),
            "beats_one_bar_delay": (
                real_mean > delayed_mean,
                (real_mean - delayed_mean) * 10_000,
            ),
            "beats_reversed_direction": (
                real_mean > reversed_mean,
                (real_mean - reversed_mean) * 10_000,
            ),
            "source_q95_positive": (
                0.95 in local_sensitivity.index
                and local_sensitivity.loc[0.95, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[0.95, "mean_primary_net_bp"])
                if 0.95 in local_sensitivity.index
                else math.nan,
            ),
            "source_q99_positive": (
                0.99 in local_sensitivity.index
                and local_sensitivity.loc[0.99, "mean_primary_net_bp"] > 0,
                float(local_sensitivity.loc[0.99, "mean_primary_net_bp"])
                if 0.99 in local_sensitivity.index
                else math.nan,
            ),
            "holding_15m_positive": (
                1 in local_horizon.index
                and local_horizon.loc[1, "mean_primary_net_bp"] > 0,
                float(local_horizon.loc[1, "mean_primary_net_bp"])
                if 1 in local_horizon.index
                else math.nan,
            ),
            "holding_60m_positive": (
                4 in local_horizon.index
                and local_horizon.loc[4, "mean_primary_net_bp"] > 0,
                float(local_horizon.loc[4, "mean_primary_net_bp"])
                if 4 in local_horizon.index
                else math.nan,
            ),
            "positive_month_share_35": (month_share <= 0.35, month_share),
        }
        eligible = all(passed for passed, _ in checks.values())
        gate_rows.extend(
            {
                "candidate": candidate,
                "check": check,
                "passed": bool(passed),
                "value": float(value),
                "eligible": eligible,
            }
            for check, (passed, value) in checks.items()
        )
        outcomes.append(
            {
                "candidate": candidate,
                "events": len(sample),
                "mean_primary_net_bp": real_mean * 10_000,
                "bootstrap_95_low_bp": low * 10_000,
                "bootstrap_95_high_bp": high * 10_000,
                "random_family_percentile": random_percentile,
                "delayed_primary_net_bp": delayed_mean * 10_000,
                "reversed_primary_net_bp": reversed_mean * 10_000,
                "positive_month_share": month_share,
                "eligible": eligible,
                "failed_gates": "|".join(
                    check for check, (passed, _) in checks.items() if not passed
                ),
            }
        )
    outcome = pd.DataFrame(outcomes)
    outcome["verdict"] = np.where(
        outcome["eligible"],
        "offline_research_candidate_only",
        "reject_btc_confirmed_flow_laggard",
    )
    return pd.DataFrame(gate_rows), outcome


def _write_findings(outcome: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    verdict = (
        "offline_research_candidate_only"
        if bool(outcome["eligible"].any())
        else "reject_btc_confirmed_flow_laggard"
    )
    text = [
        "# v17.8 BTC Confirmed-Flow Laggard Findings",
        "",
        f"Verdict: `{verdict}`.",
        "",
        outcome.to_markdown(index=False, floatfmt=".4f"),
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Signals use only completed Binance USD-M 15m bars. Graphs are frozen from",
        "prior-month history and laggards are selected at the closed signal bar.",
        "No PaperLive, application, leverage, remote, or real-order permission changes.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v178_btc_confirmed_flow_laggard(
    kline_root: Path = KLINE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V178Config = V178Config(),
) -> dict[str, Path]:
    close, btc_source = load_v178_market_data(kline_root)
    returns = close.pct_change(fill_method=None)
    signals = build_btc_source_signals(btc_source, cfg)
    graph = build_monthly_btc_receiver_graph(
        returns,
        signals["feature_time"].min(),
        signals["feature_time"].max(),
        cfg,
    )
    events, selected = build_v178_events(signals, graph, close, returns, cfg)
    summary = summarize_v178(events)

    delayed_signals = build_btc_source_signals(btc_source, cfg, shift_bars=1)
    delayed_events, _ = build_v178_events(
        delayed_signals, graph, close, returns, cfg
    )
    delayed_summary = summarize_v178(delayed_events)

    sensitivity_frames: list[pd.DataFrame] = []
    for quantile in (0.95, 0.99):
        local_signals = build_btc_source_signals(
            btc_source, cfg, return_quantile=quantile
        )
        local_events, _ = build_v178_events(local_signals, graph, close, returns, cfg)
        local_summary = summarize_v178(local_events)
        local_summary["return_quantile"] = quantile
        sensitivity_frames.append(local_summary)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)

    horizon_frames: list[pd.DataFrame] = []
    for holding_bars in (1, 4):
        local_events, _ = build_v178_events(
            signals, graph, close, returns, cfg, holding_bars=holding_bars
        )
        local_summary = summarize_v178(local_events)
        local_summary["holding_bars"] = holding_bars
        horizon_frames.append(local_summary)
    horizon_summary = pd.concat(horizon_frames, ignore_index=True)
    random_controls = _random_controls(signals, graph, close, returns, cfg)
    gates, outcome = audit_v178(
        events,
        summary,
        delayed_summary,
        sensitivity,
        horizon_summary,
        random_controls,
        cfg,
    )
    root = ensure_dir(report_root)
    paths = {
        "signals": root / "btc_source_signals.parquet",
        "graph": root / "monthly_btc_receiver_graph.parquet",
        "events": root / "candidate_events.parquet",
        "selected": root / "selected_laggards.parquet",
        "summary": root / "period_summary.csv",
        "delayed_events": root / "delayed_candidate_events.parquet",
        "delayed_summary": root / "delayed_period_summary.csv",
        "sensitivity": root / "source_threshold_sensitivity.csv",
        "horizons": root / "holding_horizon_summary.csv",
        "random_controls": root / "random_receiver_controls.parquet",
        "gates": root / "candidate_gates.csv",
        "outcome": root / "candidate_outcome.csv",
        "findings": findings_path,
    }
    signals.to_parquet(paths["signals"], index=False)
    graph.to_parquet(paths["graph"], index=False)
    events.to_parquet(paths["events"], index=False)
    selected.to_parquet(paths["selected"], index=False)
    summary.to_csv(paths["summary"], index=False)
    delayed_events.to_parquet(paths["delayed_events"], index=False)
    delayed_summary.to_csv(paths["delayed_summary"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    horizon_summary.to_csv(paths["horizons"], index=False)
    random_controls.to_parquet(paths["random_controls"], index=False)
    gates.to_csv(paths["gates"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    _write_findings(outcome, summary, findings_path)
    return paths


__all__ = [
    "V178Config",
    "build_btc_source_signals",
    "build_monthly_btc_receiver_graph",
    "build_v178_events",
    "load_v178_market_data",
    "summarize_v178",
    "write_v178_btc_confirmed_flow_laggard",
]
