"""Directed BTC-residual lead-lag graph and downstream bucket audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v10_6_directed_residual_bucket")
PANEL_PATH = Path("reports/v10_3_graph_bucket_return_diffusion/bucket_feature_panel.parquet")
BTC = "BTCUSDT"
CANDIDATES = ("DRB1_DIRECTED_PROPAGATION", "DRB2_DIRECTED_LAGGARD")


@dataclass(frozen=True)
class V106Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    lags: tuple[int, ...] = (1, 2, 4)
    min_edge_samples: int = 1000
    shrinkage_n: int = 500
    leaders_per_follower: int = 3
    min_bucket_size: int = 3
    max_bucket_size: int = 5
    cooldown_hours: int = 4
    predicted_floor: float = 0.003
    predicted_rank_floor: float = 0.80
    breadth_floor: float = 2.0 / 3.0
    lag_gap_floor: float = 0.002
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260714


def _month_start(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.to_datetime(parsed.dt.strftime("%Y-%m-01"), utc=True, errors="coerce")


def load_v106_panel(path: Path = PANEL_PATH) -> pd.DataFrame:
    columns = [
        "symbol",
        "feature_time",
        "ret_15m",
        "ret_1h",
        "future_ret_4h",
    ]
    panel = pd.read_parquet(path, columns=columns)
    panel["feature_time"] = pd.to_datetime(panel["feature_time"], utc=True, errors="coerce")
    for column in ("ret_15m", "ret_1h", "future_ret_4h"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["month_start"] = _month_start(panel["feature_time"])
    return (
        panel.dropna(subset=["symbol", "feature_time"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def estimate_v106_betas(history_returns: pd.DataFrame, btc: str = BTC) -> pd.Series:
    if btc not in history_returns.columns:
        return pd.Series(dtype=float)
    btc_values = pd.to_numeric(history_returns[btc], errors="coerce")
    variance = float(btc_values.var(ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        return pd.Series(dtype=float)
    betas = {}
    for symbol in history_returns.columns:
        values = pd.to_numeric(history_returns[symbol], errors="coerce")
        valid = values.notna() & btc_values.notna()
        if int(valid.sum()) < 100:
            continue
        covariance = float(
            np.mean(
                (values[valid] - values[valid].mean())
                * (btc_values[valid] - btc_values[valid].mean())
            )
        )
        betas[str(symbol)] = covariance / variance
    return pd.Series(betas, dtype=float)


def residualize_v106_returns(
    returns: pd.DataFrame, betas: pd.Series, btc: str = BTC
) -> pd.DataFrame:
    if btc not in returns.columns:
        return pd.DataFrame(index=returns.index)
    out = pd.DataFrame(index=returns.index)
    btc_values = pd.to_numeric(returns[btc], errors="coerce")
    for symbol, beta in betas.items():
        if symbol == btc or symbol not in returns.columns:
            continue
        out[str(symbol)] = pd.to_numeric(returns[symbol], errors="coerce") - float(
            beta
        ) * btc_values
    return out


def build_v106_month_edges(
    residual_history: pd.DataFrame,
    month_start: pd.Timestamp,
    cfg: V106Config,
) -> pd.DataFrame:
    eligible = [
        str(column)
        for column in residual_history.columns
        if int(residual_history[column].notna().sum()) >= cfg.min_edge_samples
    ]
    complete = residual_history[eligible].dropna(how="any") if eligible else pd.DataFrame()
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
        valid_columns = (leader_std > 0) & (follower_std > 0)
        standardized_leader = np.zeros_like(leader)
        standardized_follower = np.zeros_like(follower)
        standardized_leader[:, valid_columns] = (
            leader[:, valid_columns] - leader[:, valid_columns].mean(axis=0)
        ) / leader_std[valid_columns]
        standardized_follower[:, valid_columns] = (
            follower[:, valid_columns] - follower[:, valid_columns].mean(axis=0)
        ) / follower_std[valid_columns]
        n = len(leader)
        correlation = standardized_leader.T @ standardized_follower / (n - 1)
        shrinkage = np.sqrt(n / (n + cfg.shrinkage_n))
        for leader_index, leader_symbol in enumerate(eligible):
            if not valid_columns[leader_index]:
                continue
            for follower_index, follower_symbol in enumerate(eligible):
                if leader_index == follower_index or not valid_columns[follower_index]:
                    continue
                lag_corr = float(correlation[leader_index, follower_index])
                reverse_corr = float(correlation[follower_index, leader_index])
                advantage = lag_corr - reverse_corr
                if lag_corr <= 0 or advantage <= 0:
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
                        "sample_n": int(n),
                        "lag_correlation": lag_corr,
                        "reverse_correlation": reverse_corr,
                        "direction_advantage": advantage,
                        "edge_weight": weight,
                    }
    edges = pd.DataFrame(best.values())
    if edges.empty:
        return edges
    edges = edges.sort_values(
        ["follower_symbol", "edge_weight"], ascending=[True, False]
    )
    edges["edge_rank"] = (
        edges.groupby("follower_symbol", sort=False).cumcount() + 1
    )
    return edges[edges["edge_rank"].le(cfg.leaders_per_follower)].reset_index(
        drop=True
    )


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(
        index="feature_time",
        columns="symbol",
        values=column,
        aggfunc="last",
        observed=True,
    ).sort_index()


def _period(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if month < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def build_v106_graph_and_contexts(
    panel: pd.DataFrame, cfg: V106Config
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    months = sorted(panel["month_start"].dropna().unique())
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    for raw_month in months[1:]:
        month = pd.Timestamp(raw_month)
        history_start = month - pd.Timedelta(days=cfg.lookback_days)
        history = panel[
            panel["feature_time"].ge(history_start)
            & panel["feature_time"].lt(month)
        ]
        target = panel[panel["month_start"].eq(month)]
        if history.empty or target.empty:
            continue
        historical_returns = _pivot(history, "ret_15m")
        betas = estimate_v106_betas(historical_returns)
        residual_history = residualize_v106_returns(historical_returns, betas)
        edges = build_v106_month_edges(residual_history, month, cfg)
        if edges.empty:
            continue
        edge_frames.append(edges)
        target_ret_1h = _pivot(target, "ret_1h")
        target_future = _pivot(target, "future_ret_4h")
        if BTC not in target_ret_1h.columns or BTC not in target_future.columns:
            continue
        graph_symbols = sorted(
            set(edges["leader_symbol"].astype(str))
            | set(edges["follower_symbol"].astype(str))
        )
        graph_symbols = [
            symbol
            for symbol in graph_symbols
            if symbol in target_ret_1h.columns
            and symbol in target_future.columns
            and symbol in betas.index
        ]
        residual_1h = pd.DataFrame(index=target_ret_1h.index)
        residual_future = pd.DataFrame(index=target_future.index)
        for symbol in graph_symbols:
            beta = float(betas[symbol])
            residual_1h[symbol] = target_ret_1h[symbol] - beta * target_ret_1h[BTC]
            residual_future[symbol] = target_future[symbol] - beta * target_future[BTC]
        raw_future = target_future.reindex(columns=graph_symbols)
        common_times = residual_1h.index.intersection(raw_future.index)
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "times": common_times,
            "symbols": graph_symbols,
            "residual_1h": residual_1h.reindex(common_times),
            "raw_future_4h": raw_future.reindex(common_times),
            "residual_future_4h": residual_future.reindex(common_times),
            "betas": betas.reindex(graph_symbols),
        }
    all_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    return all_edges, contexts


def _mapping_signal_matrices(
    context: dict[str, Any], edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols: list[str] = context["symbols"]
    residual: pd.DataFrame = context["residual_1h"].reindex(columns=symbols)
    predicted = pd.DataFrame(np.nan, index=residual.index, columns=symbols)
    breadth = pd.DataFrame(np.nan, index=residual.index, columns=symbols)
    for follower, group in edges.groupby("follower_symbol", sort=False):
        follower = str(follower)
        if follower not in predicted.columns:
            continue
        local = group[group["leader_symbol"].astype(str).isin(symbols)].copy()
        if local.empty:
            continue
        leaders = local["leader_symbol"].astype(str).tolist()
        weights = pd.to_numeric(local["edge_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        values = residual[leaders].to_numpy(dtype=float)
        finite = np.isfinite(values)
        weighted = np.where(finite, values * weights[None, :], 0.0)
        denominators = np.where(finite, weights[None, :], 0.0).sum(axis=1)
        predicted[follower] = np.divide(
            weighted.sum(axis=1),
            denominators,
            out=np.full(len(values), np.nan),
            where=denominators > 0,
        )
        available = finite.sum(axis=1)
        breadth[follower] = np.divide(
            ((values > 0) & finite).sum(axis=1),
            available,
            out=np.full(len(values), np.nan),
            where=available > 0,
        )
    return predicted, breadth


def build_v106_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V106Config,
    signal_shift_bars: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        predicted, breadth = _mapping_signal_matrices(context, month_edges)
        own = context["residual_1h"].reindex(columns=predicted.columns)
        if signal_shift_bars:
            predicted = predicted.shift(signal_shift_bars)
            breadth = breadth.shift(signal_shift_bars)
            own = own.shift(signal_shift_bars)
        rank = predicted.rank(axis=1, pct=True, method="average")
        base = (
            predicted.ge(cfg.predicted_floor)
            & rank.ge(cfg.predicted_rank_floor)
            & breadth.ge(cfg.breadth_floor)
        )
        states = {
            CANDIDATES[0]: base,
            CANDIDATES[1]: base & predicted.sub(own).ge(cfg.lag_gap_floor),
        }
        raw_future = context["raw_future_4h"].reindex(columns=predicted.columns)
        residual_future = context["residual_future_4h"].reindex(
            columns=predicted.columns
        )
        for timestamp in predicted.index:
            for candidate, state in states.items():
                last = last_by_candidate.get(candidate)
                if last is not None and pd.Timestamp(timestamp) - last < cooldown:
                    continue
                eligible = state.columns[state.loc[timestamp].fillna(False)]
                if len(eligible) < cfg.min_bucket_size:
                    continue
                ordered = (
                    predicted.loc[timestamp, eligible]
                    .sort_values(ascending=False)
                    .head(cfg.max_bucket_size)
                    .index.tolist()
                )
                raw = pd.to_numeric(raw_future.loc[timestamp, ordered], errors="coerce")
                residual_outcome = pd.to_numeric(
                    residual_future.loc[timestamp, ordered], errors="coerce"
                )
                finite = raw.notna() & residual_outcome.notna()
                if int(finite.sum()) < cfg.min_bucket_size:
                    continue
                selected = list(pd.Index(ordered)[finite.to_numpy()])
                raw_gross = float(raw[finite].mean())
                residual_gross = float(residual_outcome[finite].mean())
                row = {
                    "candidate": candidate,
                    "feature_time": timestamp,
                    "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                    "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                    "period": context["period"],
                    "bucket_size": int(len(selected)),
                    "follower_symbols": "|".join(selected),
                    "mean_predicted_residual_1h": float(
                        predicted.loc[timestamp, selected].mean()
                    ),
                    "mean_leader_positive_breadth": float(
                        breadth.loc[timestamp, selected].mean()
                    ),
                    "mean_lag_gap_1h": float(
                        predicted.loc[timestamp, selected].sub(
                            own.loc[timestamp, selected]
                        ).mean()
                    ),
                    "raw_gross_4h": raw_gross,
                    "residual_gross_4h": residual_gross,
                    "raw_net_4h_20bp": raw_gross - 0.002,
                    "raw_net_4h_30bp": raw_gross - 0.003,
                    "raw_net_4h_50bp": raw_gross - 0.005,
                    "residual_net_4h_40bp": residual_gross - 0.004,
                }
                rows.append(row)
                last_by_candidate[candidate] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def summarize_v106(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        scoped = portfolios if scope == "all" else portfolios[portfolios["period"].eq(scope)]
        for candidate in CANDIDATES:
            sample = scoped[scoped["candidate"].eq(candidate)]
            rows.append(
                {
                    "scope": scope,
                    "candidate": candidate,
                    "portfolio_observations": int(len(sample)),
                    "active_days": int(sample["entry_day"].nunique()),
                    "active_months": int(sample["entry_month"].nunique()),
                    "mean_bucket_size": float(sample["bucket_size"].mean()),
                    "mean_raw_gross_4h": float(sample["raw_gross_4h"].mean()),
                    "mean_raw_net_4h_20bp": float(sample["raw_net_4h_20bp"].mean()),
                    "mean_raw_net_4h_30bp": float(sample["raw_net_4h_30bp"].mean()),
                    "mean_raw_net_4h_50bp": float(sample["raw_net_4h_50bp"].mean()),
                    "mean_residual_gross_4h": float(
                        sample["residual_gross_4h"].mean()
                    ),
                    "mean_residual_net_4h_40bp": float(
                        sample["residual_net_4h_40bp"].mean()
                    ),
                    "win_rate_residual_net40": float(
                        sample["residual_net_4h_40bp"].gt(0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def randomize_v106_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V106Config,
) -> pd.DataFrame:
    frames = []
    for month, group in edges.groupby("month_start", sort=True):
        month = pd.Timestamp(month)
        symbols = contexts[month]["symbols"]
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month.month)
        local_rows = []
        for follower, follower_edges in group.groupby("follower_symbol", sort=False):
            follower = str(follower)
            choices = [symbol for symbol in symbols if symbol != follower]
            take = min(len(follower_edges), len(choices))
            leaders = rng.choice(choices, size=take, replace=False)
            source_rows = follower_edges.sort_values("edge_rank").head(take)
            for leader, row in zip(leaders, source_rows.itertuples(index=False), strict=True):
                payload = row._asdict()
                payload["leader_symbol"] = str(leader)
                local_rows.append(payload)
        frames.append(pd.DataFrame(local_rows))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reverse_v106_edges(edges: pd.DataFrame, cfg: V106Config) -> pd.DataFrame:
    out = edges.copy()
    out[["leader_symbol", "follower_symbol"]] = out[
        ["follower_symbol", "leader_symbol"]
    ].to_numpy()
    out = out.sort_values(
        ["month_start", "follower_symbol", "edge_weight"],
        ascending=[True, True, False],
    )
    out = out.drop_duplicates(
        ["month_start", "leader_symbol", "follower_symbol"], keep="first"
    )
    out["edge_rank"] = (
        out.groupby(["month_start", "follower_symbol"], sort=False).cumcount() + 1
    )
    return out[out["edge_rank"].le(cfg.leaders_per_follower)].reset_index(drop=True)


def random_v106_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V106Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v106_edges(edges, contexts, iteration, cfg)
        portfolios = build_v106_portfolios(contexts, randomized, cfg)
        means = {}
        for candidate in CANDIDATES:
            sample = portfolios[portfolios["candidate"].eq(candidate)]
            means[candidate] = float(sample["residual_net_4h_40bp"].mean())
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "portfolio_observations": int(len(sample)),
                    "mean_residual_net_4h_40bp": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "portfolio_observations": int(len(portfolios)),
                "mean_residual_net_4h_40bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(sample: pd.DataFrame, cfg: V106Config) -> tuple[float, float]:
    daily = [
        group["residual_net_4h_40bp"].dropna().to_numpy(dtype=float)
        for _, group in sample.groupby("entry_day", sort=True)
    ]
    daily = [values for values in daily if len(values)]
    if not daily:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(daily), len(daily))
        boot.append(float(np.mean(np.concatenate([daily[index] for index in chosen]))))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _positive_month_share(sample: pd.DataFrame) -> float:
    values = sample.groupby("entry_month")["residual_net_4h_40bp"].sum().clip(lower=0)
    return float(values.max() / values.sum()) if values.sum() > 0 else np.inf


def audit_v106(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    reversed_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V106Config,
) -> pd.DataFrame:
    family = pd.to_numeric(
        controls.loc[
            controls["candidate"].eq("FAMILY_MAX"),
            "mean_residual_net_4h_40bp",
        ],
        errors="coerce",
    ).dropna()
    rows = []
    for candidate in CANDIDATES:
        lookup = {
            row.scope: row
            for row in summary[summary["candidate"].eq(candidate)].itertuples(index=False)
        }
        sample = real[real["candidate"].eq(candidate)]
        shifted_mean = float(
            shifted.loc[
                shifted["candidate"].eq(candidate), "residual_net_4h_40bp"
            ].mean()
        )
        reversed_mean = float(
            reversed_portfolios.loc[
                reversed_portfolios["candidate"].eq(candidate),
                "residual_net_4h_40bp",
            ].mean()
        )
        ci_low, ci_high = _bootstrap_ci(sample, cfg)
        percentile = float(family.lt(lookup["all"].mean_residual_net_4h_40bp).mean())
        month_share = _positive_month_share(sample)
        gates = {
            "full_observations_100": lookup["all"].portfolio_observations >= 100,
            "validation_observations_25": lookup["validation"].portfolio_observations >= 25,
            "holdout_observations_25": lookup["holdout"].portfolio_observations >= 25,
            "full_residual_net40_positive": lookup["all"].mean_residual_net_4h_40bp > 0,
            "validation_residual_net40_positive": lookup["validation"].mean_residual_net_4h_40bp > 0,
            "holdout_residual_net40_positive": lookup["holdout"].mean_residual_net_4h_40bp > 0,
            "validation_raw_net20_positive": lookup["validation"].mean_raw_net_4h_20bp > 0,
            "holdout_raw_net20_positive": lookup["holdout"].mean_raw_net_4h_20bp > 0,
            "random_family_p90": percentile >= 0.90,
            "beats_shifted": lookup["all"].mean_residual_net_4h_40bp > shifted_mean,
            "beats_reversed": lookup["all"].mean_residual_net_4h_40bp > reversed_mean,
            "bootstrap_lower_positive": ci_low > 0,
            "month_share_below_35pct": month_share <= 0.35,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "verdict": "directed_residual_bucket_forward_watch_only"
                if eligible
                else "reject_directed_residual_bucket_candidate",
                "full_residual_net40": lookup["all"].mean_residual_net_4h_40bp,
                "validation_residual_net40": lookup["validation"].mean_residual_net_4h_40bp,
                "holdout_residual_net40": lookup["holdout"].mean_residual_net_4h_40bp,
                "validation_raw_net20": lookup["validation"].mean_raw_net_4h_20bp,
                "holdout_raw_net20": lookup["holdout"].mean_raw_net_4h_20bp,
                "shifted_residual_net40": shifted_mean,
                "reversed_residual_net40": reversed_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "max_positive_month_share": month_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        )
    family_verdict = (
        "directed_residual_bucket_forward_watch_only"
        if any(row["eligible"] for row in rows)
        else "reject_directed_residual_bucket_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def write_v106_directed_residual_bucket(
    cfg: V106Config = V106Config(),
) -> dict[str, Path]:
    panel = load_v106_panel(cfg.panel_path)
    edges, contexts = build_v106_graph_and_contexts(panel, cfg)
    real = build_v106_portfolios(contexts, edges, cfg)
    shifted = build_v106_portfolios(contexts, edges, cfg, signal_shift_bars=96)
    reversed_edges = reverse_v106_edges(edges, cfg)
    reversed_portfolios = build_v106_portfolios(contexts, reversed_edges, cfg)
    summary = summarize_v106(real)
    controls = random_v106_controls(contexts, edges, cfg)
    audit = audit_v106(
        real, shifted, reversed_portfolios, summary, controls, cfg
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "edges": root / "directed_residual_edges.csv",
        "portfolios": root / "timestamp_bucket_portfolios.parquet",
        "shifted": root / "shifted_signal_portfolios.parquet",
        "reversed": root / "reversed_edge_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    edges.to_csv(outputs["edges"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    reversed_portfolios.to_parquet(outputs["reversed"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v10.6 Directed Residual Graph-Bucket",
        "",
        f"Status: `{audit['family_verdict'].iloc[0]}`.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.candidate}: residual net40={row.full_residual_net40:.4%}, "
            f"validation={row.validation_residual_net40:.4%}, "
            f"holdout={row.holdout_residual_net40:.4%}, "
            f"random percentile={row.random_family_percentile:.1%}."
        )
    lines.extend(["", "No PaperLive or live permission changed."])
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
