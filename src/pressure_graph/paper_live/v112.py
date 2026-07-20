"""Paper-live shadow observer for the frozen v11.2 topology signal."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pressure_graph.io import ensure_dir, write_parquet


BTC = "BTCUSDT"


@dataclass(frozen=True)
class V112LiveConfig:
    live_root: Path
    history_days: int
    base_config: Path
    seed_base_events: Path
    report_root: Path
    symbols: tuple[str, ...]
    fallback_candidates: tuple[str, ...] = ()
    lookback_days: int = 30
    min_samples: int = 500
    community_count: int = 8
    expected_community_size: int = 9
    coherence_hours: int = 12
    rank_hours: int = 4
    break_quantile: float = 0.05
    cooldown_hours: int = 4
    severity_quantile: float = 0.80
    min_prior_events: int = 100
    volatility_hours: int = 24
    volatility_quantile: float = 0.75
    max_communities: int = 3
    horizon_hours: int = 4
    round_trip_cost_bps: float = 20.0
    timely_lag_minutes: int = 60
    stale_after_minutes: int = 45
    mode: str = "paper_live_shadow_only"
    real_orders_allowed: bool = False


@dataclass(frozen=True)
class V112LiveStatus:
    status: str
    observed_at_utc: str
    latest_feature_time: str
    data_stale: bool
    mode: str
    real_orders_allowed: bool
    exact_universe_ready: bool
    active_universe_mode: str
    universe_replacements: tuple[str, ...]
    configured_symbols: int
    eligible_symbols: int
    community_count: int
    community_sizes: tuple[int, ...]
    prior_event_count: int
    severity_threshold: float | None
    btc_volatility_threshold: float | None
    rolling_base_events: int
    rolling_selected_signals: int
    cumulative_signals: int
    timely_signals: int
    completed_portfolios: int
    open_portfolios: int
    mean_completed_net20: float | None
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def load_v112_live_config(path: str | Path) -> V112LiveConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = payload["data"]
    graph = payload["graph"]
    signal = payload["signal"]
    forward = payload["forward"]
    experiment = payload["experiment"]
    symbols = tuple(dict.fromkeys(str(value) for value in payload["universe"]["symbols"]))
    fallback_candidates = tuple(
        dict.fromkeys(
            str(value) for value in payload["universe"].get("fallback_candidates", [])
        )
    )
    return V112LiveConfig(
        live_root=Path(data["live_root"]),
        history_days=int(data["history_days"]),
        base_config=Path(data["base_config"]),
        seed_base_events=Path(data["seed_base_events"]),
        report_root=Path(forward["report_root"]),
        symbols=symbols,
        fallback_candidates=fallback_candidates,
        lookback_days=int(graph["lookback_days"]),
        min_samples=int(graph["min_samples"]),
        community_count=int(graph["community_count"]),
        expected_community_size=int(graph["expected_community_size"]),
        coherence_hours=int(graph["coherence_hours"]),
        rank_hours=int(graph["rank_hours"]),
        break_quantile=float(graph["break_quantile"]),
        cooldown_hours=int(graph["cooldown_hours"]),
        severity_quantile=float(signal["severity_quantile"]),
        min_prior_events=int(signal["min_prior_events"]),
        volatility_hours=int(signal["volatility_hours"]),
        volatility_quantile=float(signal["volatility_quantile"]),
        max_communities=int(signal["max_communities"]),
        horizon_hours=int(forward["horizon_hours"]),
        round_trip_cost_bps=float(forward["round_trip_cost_bps"]),
        timely_lag_minutes=int(forward["timely_lag_minutes"]),
        stale_after_minutes=int(forward["stale_after_minutes"]),
        mode=str(experiment["mode"]),
        real_orders_allowed=bool(experiment["real_orders_allowed"]),
    )


def build_v112_return_panel(klines: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "bar_open_time", "bar_close_time", "close"}
    missing = required - set(klines.columns)
    if missing:
        raise ValueError(f"missing kline columns: {sorted(missing)}")
    optional = ["turnover"] if "turnover" in klines.columns else []
    data = klines[[*required, *optional]].copy()
    data["bar_open_time"] = pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce")
    data["feature_time"] = pd.to_datetime(data["bar_close_time"], utc=True, errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    if "turnover" in data.columns:
        data["turnover"] = pd.to_numeric(data["turnover"], errors="coerce").fillna(0.0)
    else:
        data["turnover"] = 0.0
    data = data.dropna(subset=["symbol", "bar_open_time", "feature_time", "close"])
    data = data.sort_values(["symbol", "bar_open_time"]).drop_duplicates(
        ["symbol", "bar_open_time"], keep="last"
    )
    grouped = data.groupby("symbol", sort=False)
    previous_close = grouped["close"].shift(4)
    previous_time = grouped["bar_open_time"].shift(4)
    data["ret_1h"] = data["close"].div(previous_close).sub(1.0)
    data["turnover_1h"] = grouped["turnover"].rolling(4, min_periods=4).sum().reset_index(
        level=0, drop=True
    )
    data.loc[data["bar_open_time"].sub(previous_time).ne(pd.Timedelta(hours=1)), "ret_1h"] = np.nan
    return data[["symbol", "feature_time", "ret_1h", "turnover_1h"]].sort_values(
        ["feature_time", "symbol"]
    )


def _pivot(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot_table(
        index="feature_time", columns="symbol", values="ret_1h", aggfunc="last"
    ).sort_index()


def _estimate_betas(returns: pd.DataFrame) -> pd.Series:
    if BTC not in returns.columns:
        return pd.Series(dtype=float)
    btc = pd.to_numeric(returns[BTC], errors="coerce")
    variance = float(btc.var(ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        return pd.Series(dtype=float)
    betas = {}
    for symbol in returns.columns:
        values = pd.to_numeric(returns[symbol], errors="coerce")
        valid = values.notna() & btc.notna()
        if int(valid.sum()) < 100:
            continue
        covariance = float(
            np.mean(
                (values[valid] - values[valid].mean())
                * (btc[valid] - btc[valid].mean())
            )
        )
        betas[str(symbol)] = covariance / variance
    return pd.Series(betas, dtype=float)


def _residualize(returns: pd.DataFrame, betas: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame(index=returns.index)
    if BTC not in returns.columns:
        return output
    for symbol, beta in betas.items():
        if symbol != BTC and symbol in returns.columns:
            output[str(symbol)] = returns[symbol] - float(beta) * returns[BTC]
    return output


def _spectral_split(
    correlation: pd.DataFrame, members: list[str]
) -> tuple[list[str], list[str]]:
    ordered = sorted(members)
    affinity = correlation.loc[ordered, ordered].to_numpy(dtype=float)
    affinity = np.clip(np.nan_to_num(affinity, nan=0.0), 0.0, 1.0)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(len(ordered)) - inverse[:, None] * affinity * inverse[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    fiedler = vectors[:, 1] if len(ordered) > 2 else vectors[:, -1]
    ranked = sorted(zip(fiedler, ordered), key=lambda item: (float(item[0]), item[1]))
    midpoint = len(ranked) // 2
    return sorted(symbol for _, symbol in ranked[:midpoint]), sorted(
        symbol for _, symbol in ranked[midpoint:]
    )


def build_v112_live_communities(
    residual_history: pd.DataFrame,
    community_count: int,
    min_samples: int,
) -> list[list[str]]:
    eligible = sorted(
        symbol
        for symbol in residual_history.columns
        if symbol != BTC and int(residual_history[symbol].notna().sum()) >= min_samples
    )
    complete = residual_history[eligible].dropna(how="any") if eligible else pd.DataFrame()
    if len(complete) < min_samples or len(eligible) < community_count * 2:
        return []
    correlation = complete.corr().fillna(0.0)
    groups = [eligible]
    while len(groups) < community_count:
        split_index = max(
            range(len(groups)), key=lambda index: (len(groups[index]), groups[index])
        )
        source = groups.pop(split_index)
        if len(source) < 4:
            return []
        groups.extend(_spectral_split(correlation, source))
    return sorted(
        (sorted(group) for group in groups), key=lambda group: (-len(group), group[0])
    )


def _coherence(
    residual: pd.DataFrame, members: list[str], scale: pd.Series, hours: int
) -> pd.Series:
    local = residual[members].div(scale[members]).replace([np.inf, -np.inf], np.nan)
    count = local.notna().sum(axis=1)
    total = local.sum(axis=1, skipna=True)
    squared = local.pow(2).sum(axis=1, skipna=True)
    pairwise = (total.pow(2) - squared).div(count * (count - 1))
    pairwise = pairwise.where(count.ge(max(4, len(members) - 1)))
    return pairwise.rolling(hours, min_periods=hours).mean()


def build_v112_live_month(
    panel: pd.DataFrame, cfg: V112LiveConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = panel.copy()
    data["feature_time"] = pd.to_datetime(data["feature_time"], utc=True, errors="coerce")
    source_latest = data["feature_time"].max()
    allowed_symbols = set(cfg.symbols) | set(cfg.fallback_candidates)
    data = data[data["symbol"].isin(allowed_symbols)].copy()
    latest = data.loc[data["feature_time"].dt.minute.eq(0), "feature_time"].max()
    if pd.isna(latest):
        return pd.DataFrame(), pd.DataFrame(), {
            "latest_feature_time": pd.NaT,
            "source_latest_feature_time": source_latest,
        }
    month = pd.Timestamp(year=latest.year, month=latest.month, day=1, tz="UTC")
    hourly = data[data["feature_time"].dt.minute.eq(0)].copy()
    history_all = hourly[
        hourly["feature_time"].ge(month - pd.Timedelta(days=cfg.lookback_days))
        & hourly["feature_time"].lt(month)
    ]
    exact_members = [symbol for symbol in cfg.symbols if symbol != BTC]
    counts = history_all.groupby("symbol")["ret_1h"].count()
    exact_eligible = [symbol for symbol in exact_members if int(counts.get(symbol, 0)) >= cfg.min_samples]
    invalid_exact = sorted(set(exact_members) - set(exact_eligible))
    fallback_eligible = [
        symbol
        for symbol in cfg.fallback_candidates
        if int(counts.get(symbol, 0)) >= cfg.min_samples and symbol not in exact_members
    ]
    if "turnover_1h" in history_all.columns:
        turnover = history_all.groupby("symbol")["turnover_1h"].sum(min_count=1)
        fallback_eligible = sorted(
            fallback_eligible,
            key=lambda symbol: (-float(turnover.get(symbol, 0.0)), symbol),
        )
    replacements = list(zip(invalid_exact, fallback_eligible[: len(invalid_exact)]))
    active_members = [symbol for symbol in exact_members if symbol not in invalid_exact]
    active_members.extend(replacement for _, replacement in replacements)
    active_symbols = [BTC, *active_members]
    data = hourly[hourly["symbol"].isin(active_symbols)].copy()
    history = data[
        data["feature_time"].ge(month - pd.Timedelta(days=cfg.lookback_days))
        & data["feature_time"].lt(month)
    ]
    target = data[data["feature_time"].ge(month)]
    historical_return = _pivot(history)
    target_return = _pivot(target)
    betas = _estimate_betas(historical_return)
    historical_residual = _residualize(historical_return, betas)
    communities = build_v112_live_communities(
        historical_residual, cfg.community_count, cfg.min_samples
    )
    membership_rows = []
    for index, members in enumerate(communities, start=1):
        community_id = f"{month:%Y-%m}:BSP{index:02d}"
        for symbol in members:
            membership_rows.append(
                {
                    "month_start": month,
                    "community_id": community_id,
                    "symbol": symbol,
                    "community_size": len(members),
                    "beta": float(betas.get(symbol, np.nan)),
                }
            )
    membership = pd.DataFrame(membership_rows)
    metadata: dict[str, Any] = {
        "latest_feature_time": latest,
        "source_latest_feature_time": source_latest,
        "month_start": month,
        "exact_eligible_symbols": len(exact_eligible),
        "exact_universe_ready": not invalid_exact,
        "invalid_exact_symbols": tuple(invalid_exact),
        "replacements": tuple(f"{source}->{target}" for source, target in replacements),
        "active_universe_mode": "exact" if not replacements else "adapted_fallback",
        "eligible_symbols": int(len(set().union(*communities))) if communities else 0,
        "community_count": len(communities),
        "community_sizes": tuple(len(group) for group in communities),
        "betas": betas,
        "hourly_returns": pd.concat([historical_return, target_return]).sort_index(),
    }
    if not communities or target_return.empty:
        return pd.DataFrame(), membership, metadata
    target_residual = _residualize(target_return, betas)
    combined = pd.concat(
        [historical_residual, target_residual.reindex(columns=historical_residual.columns)]
    ).sort_index()
    btc_volatility = pd.concat([historical_return, target_return]).sort_index()[BTC].rolling(
        cfg.volatility_hours, min_periods=cfg.volatility_hours
    ).std(ddof=1)
    historical_volatility = btc_volatility[btc_volatility.index < month]
    volatility_threshold = float(
        historical_volatility.quantile(cfg.volatility_quantile)
    )
    metadata["btc_volatility_threshold"] = volatility_threshold
    rows = []
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for index, members in enumerate(communities, start=1):
        community_id = f"{month:%Y-%m}:BSP{index:02d}"
        scale = historical_residual[members].std(ddof=1).replace(0.0, np.nan)
        historical_coherence = _coherence(
            historical_residual, members, scale, cfg.coherence_hours
        )
        threshold = float(historical_coherence.quantile(cfg.break_quantile))
        history_std = float(historical_coherence.std(ddof=1))
        if not np.isfinite(threshold) or not np.isfinite(history_std) or history_std <= 0:
            continue
        coherence = _coherence(combined, members, scale, cfg.coherence_hours)
        rank_signal = combined[members].rolling(
            cfg.rank_hours, min_periods=cfg.rank_hours
        ).sum()
        active = coherence.le(threshold)
        transitions = active & ~active.shift(1, fill_value=False)
        times = transitions.index[transitions].intersection(target_return.index)
        last: pd.Timestamp | None = None
        for timestamp in times:
            timestamp = pd.Timestamp(timestamp)
            if last is not None and timestamp - last < cooldown:
                continue
            values = rank_signal.loc[timestamp, members].dropna().sort_values()
            third = len(values) // 3
            if third < 2:
                continue
            bottom = values.head(third).index.tolist()
            top = values.tail(third).index.tolist()
            severity = float((threshold - coherence.loc[timestamp]) / history_std)
            rows.append(
                {
                    "event_id": f"v112-base|{timestamp.isoformat()}|{community_id}",
                    "feature_time": timestamp,
                    "month_start": month,
                    "community_id": community_id,
                    "community_size": len(members),
                    "top_symbols": "|".join(top),
                    "bottom_symbols": "|".join(bottom),
                    "top_beta": float(betas[top].mean()),
                    "bottom_beta": float(betas[bottom].mean()),
                    "break_severity": severity,
                    "btc_volatility_24h": float(btc_volatility.get(timestamp, np.nan)),
                    "btc_volatility_threshold": volatility_threshold,
                    "active_universe_mode": metadata["active_universe_mode"],
                    "universe_replacements": "|".join(metadata["replacements"]),
                }
            )
            last = timestamp
    return pd.DataFrame(rows), membership, metadata


def _merge_by_key(existing: pd.DataFrame, current: pd.DataFrame, key: str) -> pd.DataFrame:
    frames = [frame for frame in (existing, current) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(key, keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )


def select_v112_live_signals(
    current_events: pd.DataFrame,
    historical_events: pd.DataFrame,
    cfg: V112LiveConfig,
) -> tuple[pd.DataFrame, float | None, int]:
    if current_events.empty:
        prior = historical_events.get("break_severity", pd.Series(dtype=float)).dropna()
        threshold = (
            float(prior.quantile(cfg.severity_quantile))
            if len(prior) >= cfg.min_prior_events
            else None
        )
        return pd.DataFrame(), threshold, int(len(prior))
    month = pd.Timestamp(current_events["month_start"].iloc[0])
    prior = historical_events.loc[
        pd.to_datetime(historical_events["month_start"], utc=True, errors="coerce").lt(month),
        "break_severity",
    ].dropna()
    if len(prior) < cfg.min_prior_events:
        return pd.DataFrame(), None, int(len(prior))
    threshold = float(prior.quantile(cfg.severity_quantile))
    eligible = current_events[
        current_events["break_severity"].ge(threshold)
        & current_events["btc_volatility_24h"].ge(
            current_events["btc_volatility_threshold"]
        )
    ].copy()
    chosen = []
    for timestamp, group in eligible.groupby("feature_time", sort=True):
        local = group.sort_values("break_severity", ascending=False).head(
            cfg.max_communities
        )
        local["selected_rank"] = np.arange(1, len(local) + 1)
        local["portfolio_id"] = f"v112|{pd.Timestamp(timestamp).isoformat()}"
        chosen.append(local)
    selected = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()
    return selected, threshold, int(len(prior))


def merge_v112_signal_ledger(
    existing: pd.DataFrame,
    selected: pd.DataFrame,
    observed_at: pd.Timestamp,
    timely_lag_minutes: int,
) -> pd.DataFrame:
    current = selected.copy()
    if not current.empty:
        current["signal_id"] = current["event_id"].str.replace(
            "v112-base|", "v112-signal|", regex=False
        )
        current["first_observed_at_utc"] = observed_at
    merged = _merge_by_key(existing, current, "signal_id")
    if merged.empty:
        return merged
    if not existing.empty:
        preserved = existing.drop_duplicates("signal_id", keep="last").set_index("signal_id")
        mapped = merged["signal_id"].map(preserved["first_observed_at_utc"])
        merged["first_observed_at_utc"] = mapped.fillna(merged["first_observed_at_utc"])
    merged["feature_time"] = pd.to_datetime(merged["feature_time"], utc=True, errors="coerce")
    merged["first_observed_at_utc"] = pd.to_datetime(
        merged["first_observed_at_utc"], utc=True, errors="coerce"
    )
    lag = merged["first_observed_at_utc"].sub(merged["feature_time"]).dt.total_seconds() / 60
    merged["first_observation_lag_minutes"] = lag
    merged["timely_forward_observation"] = lag.ge(0) & lag.le(timely_lag_minutes)
    merged["mode"] = "paper_live_shadow_only"
    merged["real_orders_allowed"] = False
    return merged.sort_values(["feature_time", "signal_id"]).reset_index(drop=True)


def _compound_forward(
    returns: pd.DataFrame, timestamp: pd.Timestamp, symbols: list[str], horizon: int
) -> pd.Series:
    values = pd.Series(1.0, index=symbols, dtype=float)
    for step in range(1, horizon + 1):
        future_time = timestamp + pd.Timedelta(hours=step)
        if future_time not in returns.index:
            return pd.Series(np.nan, index=symbols, dtype=float)
        values = values * (1.0 + returns.loc[future_time, symbols])
    return values - 1.0


def build_v112_portfolio_ledger(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: V112LiveConfig,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    sleeve_rows = []
    latest = returns.index.max()
    for row in signals.itertuples(index=False):
        timestamp = pd.Timestamp(row.feature_time)
        top = str(row.top_symbols).split("|")
        bottom = str(row.bottom_symbols).split("|")
        symbols = sorted(set(top + bottom + [BTC]))
        future = _compound_forward(returns, timestamp, symbols, cfg.horizon_hours)
        completed = bool(future.notna().all())
        gross = np.nan
        if completed:
            top_residual = float(future[top].mean() - float(row.top_beta) * future[BTC])
            bottom_residual = float(
                future[bottom].mean() - float(row.bottom_beta) * future[BTC]
            )
            gross = 0.5 * (top_residual - bottom_residual)
        sleeve_rows.append(
            {
                "portfolio_id": row.portfolio_id,
                "signal_id": row.signal_id,
                "feature_time": timestamp,
                "timely_forward_observation": bool(row.timely_forward_observation),
                "sleeve_completed": completed,
                "sleeve_gross_return_4h": gross,
                "latest_return_time": latest,
            }
        )
    sleeves = pd.DataFrame(sleeve_rows)
    rows = []
    for portfolio_id, group in sleeves.groupby("portfolio_id", sort=True):
        completed = bool(group["sleeve_completed"].all())
        gross = float(group["sleeve_gross_return_4h"].mean()) if completed else np.nan
        rows.append(
            {
                "portfolio_id": portfolio_id,
                "feature_time": group["feature_time"].iloc[0],
                "sleeves": int(len(group)),
                "timely_forward_observation": bool(
                    group["timely_forward_observation"].all()
                ),
                "status": "completed" if completed else "open",
                "gross_return_4h": gross,
                "net_return_20bp": gross - cfg.round_trip_cost_bps / 10_000
                if completed
                else np.nan,
                "latest_return_time": latest,
            }
        )
    return pd.DataFrame(rows).sort_values("feature_time").reset_index(drop=True)


def _append_manifest(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_status(root: Path, status: V112LiveStatus) -> tuple[Path, Path]:
    json_path = root / "live_status.json"
    markdown_path = root / "live_status.md"
    json_path.write_text(
        json.dumps(asdict(status), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# v11.2 Topology PaperLive Status",
        "",
        f"- status: `{status.status}`",
        f"- observed_at_utc: {status.observed_at_utc}",
        f"- latest_feature_time: {status.latest_feature_time}",
        f"- data_stale: {status.data_stale}",
        f"- mode: `{status.mode}`",
        f"- real_orders_allowed: `{status.real_orders_allowed}`",
        f"- exact_universe_ready: `{status.exact_universe_ready}`",
        f"- active_universe_mode: `{status.active_universe_mode}`",
        f"- universe_replacements: {list(status.universe_replacements)}",
        f"- eligible_symbols: {status.eligible_symbols}/{status.configured_symbols - 1}",
        f"- communities: {status.community_count} {list(status.community_sizes)}",
        f"- prior_event_count: {status.prior_event_count}",
        f"- severity_threshold: {status.severity_threshold}",
        f"- btc_volatility_threshold: {status.btc_volatility_threshold}",
        f"- rolling_base_events: {status.rolling_base_events}",
        f"- rolling_selected_signals: {status.rolling_selected_signals}",
        f"- cumulative_signals: {status.cumulative_signals}",
        f"- timely_signals: {status.timely_signals}",
        f"- completed_portfolios: {status.completed_portfolios}",
        f"- open_portfolios: {status.open_portfolios}",
        f"- mean_completed_net20: {status.mean_completed_net20}",
        f"- reasons: {','.join(status.reasons) if status.reasons else 'none'}",
        f"- warnings: {','.join(status.warnings) if status.warnings else 'none'}",
        "",
        "This observer writes virtual signals and returns only. It contains no order route.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def write_v112_paper_live(
    panel: pd.DataFrame,
    cfg: V112LiveConfig,
    *,
    observed_at: object | None = None,
) -> dict[str, Path]:
    observed = pd.Timestamp.now(tz="UTC") if observed_at is None else pd.Timestamp(observed_at)
    observed = observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
    root = ensure_dir(cfg.report_root)
    forward_root = ensure_dir(root / "forward")
    current_events, membership, metadata = build_v112_live_month(panel, cfg)
    seed = pd.read_parquet(cfg.seed_base_events)
    base_path = forward_root / "base_events.parquet"
    existing_base = pd.read_parquet(base_path) if base_path.exists() else pd.DataFrame()
    cumulative_base = _merge_by_key(existing_base, current_events, "event_id")
    historical_events = pd.concat([seed, cumulative_base], ignore_index=True, sort=False)
    selected, severity_threshold, prior_count = select_v112_live_signals(
        current_events, historical_events, cfg
    )
    signals_path = forward_root / "signals.parquet"
    existing_signals = pd.read_parquet(signals_path) if signals_path.exists() else pd.DataFrame()
    signals = merge_v112_signal_ledger(
        existing_signals, selected, observed, cfg.timely_lag_minutes
    )
    returns: pd.DataFrame = metadata.get("hourly_returns", pd.DataFrame())
    portfolios = build_v112_portfolio_ledger(signals, returns, cfg)
    portfolio_path = forward_root / "portfolio_trades.parquet"
    membership_path = root / "monthly_membership.csv"
    write_parquet(cumulative_base, base_path)
    write_parquet(signals, signals_path)
    write_parquet(portfolios, portfolio_path)
    membership.to_csv(membership_path, index=False)
    latest = metadata.get("source_latest_feature_time", pd.NaT)
    data_stale = bool(
        pd.isna(latest)
        or observed - pd.Timestamp(latest) > pd.Timedelta(minutes=cfg.stale_after_minutes)
    )
    reasons = []
    warnings = []
    sizes = tuple(metadata.get("community_sizes", ()))
    if metadata.get("eligible_symbols", 0) != len(cfg.symbols) - 1:
        reasons.append("universe_incomplete")
    if len(sizes) != cfg.community_count or any(
        size != cfg.expected_community_size for size in sizes
    ):
        reasons.append("community_shape_invalid")
    if prior_count < cfg.min_prior_events:
        reasons.append("insufficient_prior_events")
    if data_stale:
        reasons.append("data_stale")
    if not metadata.get("exact_universe_ready", False):
        warnings.append("exact_universe_unavailable")
    if metadata.get("replacements"):
        warnings.append("adapted_universe_not_exact_v112_forward_evidence")
    completed = portfolios[portfolios.get("status", "").eq("completed")] if not portfolios.empty else portfolios
    timely = signals.get("timely_forward_observation", pd.Series(dtype=bool))
    status = V112LiveStatus(
        status=(
            "READY_ADAPTED_SHADOW"
            if not reasons and metadata.get("replacements")
            else "READY_EXACT_SHADOW"
            if not reasons
            else "BLOCKED_SHADOW"
        ),
        observed_at_utc=observed.isoformat(),
        latest_feature_time="" if pd.isna(latest) else pd.Timestamp(latest).isoformat(),
        data_stale=data_stale,
        mode=cfg.mode,
        real_orders_allowed=cfg.real_orders_allowed,
        exact_universe_ready=bool(metadata.get("exact_universe_ready", False)),
        active_universe_mode=str(metadata.get("active_universe_mode", "unavailable")),
        universe_replacements=tuple(metadata.get("replacements", ())),
        configured_symbols=len(cfg.symbols),
        eligible_symbols=int(metadata.get("eligible_symbols", 0)),
        community_count=int(metadata.get("community_count", 0)),
        community_sizes=sizes,
        prior_event_count=prior_count,
        severity_threshold=severity_threshold,
        btc_volatility_threshold=metadata.get("btc_volatility_threshold"),
        rolling_base_events=int(len(current_events)),
        rolling_selected_signals=int(len(selected)),
        cumulative_signals=int(len(signals)),
        timely_signals=int(timely.fillna(False).astype(bool).sum()),
        completed_portfolios=int(len(completed)),
        open_portfolios=int((portfolios.get("status", pd.Series(dtype=str)) == "open").sum()),
        mean_completed_net20=float(completed["net_return_20bp"].mean())
        if not completed.empty
        else None,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
    status_json, status_md = _write_status(root, status)
    manifest_path = forward_root / "run_manifest.csv"
    _append_manifest(
        manifest_path,
        {
            "observed_at_utc": observed.isoformat(),
            "latest_feature_time": status.latest_feature_time,
            "status": status.status,
            "data_stale": status.data_stale,
            "rolling_base_events": status.rolling_base_events,
            "rolling_selected_signals": status.rolling_selected_signals,
            "cumulative_signals": status.cumulative_signals,
            "timely_signals": status.timely_signals,
            "completed_portfolios": status.completed_portfolios,
            "open_portfolios": status.open_portfolios,
            "real_orders_allowed": status.real_orders_allowed,
        },
    )
    return {
        "base_events": base_path,
        "signals": signals_path,
        "portfolio_trades": portfolio_path,
        "membership": membership_path,
        "status_json": status_json,
        "status_md": status_md,
        "run_manifest": manifest_path,
    }


__all__ = [
    "V112LiveConfig",
    "V112LiveStatus",
    "build_v112_live_communities",
    "build_v112_live_month",
    "build_v112_portfolio_ledger",
    "build_v112_return_panel",
    "load_v112_live_config",
    "merge_v112_signal_ledger",
    "select_v112_live_signals",
    "write_v112_paper_live",
]
