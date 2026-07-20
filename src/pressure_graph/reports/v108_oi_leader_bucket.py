"""Directed OI-shock graph and downstream multi-coin bucket audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    BTC,
    estimate_v106_betas,
)


REPORT_ROOT = Path("reports/v10_8_oi_leader_bucket")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
UNIVERSE_EDGE_PATH = Path("reports/v0_7b_neighbor_graph/neighbor_graph_edges.csv")
CANDIDATES = ("OIG1_POSITIVE_OI_PROPAGATION", "OIG2_NEGATIVE_OI_PROPAGATION")


@dataclass(frozen=True)
class V108Config:
    feature_path: Path = FEATURE_PATH
    universe_edge_path: Path = UNIVERSE_EDGE_PATH
    report_root: Path = REPORT_ROOT
    lookback_days: int = 30
    minimum_history_days: int = 28
    min_edge_samples: int = 500
    shrinkage_n: int = 500
    leaders_per_follower: int = 3
    oi_z_threshold: float = 2.0
    min_bucket_size: int = 3
    max_bucket_size: int = 5
    cooldown_hours: int = 4
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    seed: int = 20260714


def _universe(path: Path) -> set[str]:
    edges = pd.read_csv(path, usecols=["source_symbol", "neighbor_symbol"])
    return set(edges["source_symbol"].astype(str)) | set(
        edges["neighbor_symbol"].astype(str)
    )


def load_v108_features(cfg: V108Config = V108Config()) -> pd.DataFrame:
    symbols = _universe(cfg.universe_edge_path)
    columns = [
        "symbol",
        "feature_time",
        "ret_1h",
        "future_ret_4h",
        "oi_value_delta_z_1h",
        "warmup_complete",
    ]
    parquet = pq.ParquetFile(cfg.feature_path)
    frames = []
    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(index, columns=columns).to_pandas()
        chunk = chunk[chunk["symbol"].astype(str).isin(symbols)]
        chunk = chunk[chunk["warmup_complete"].fillna(False).astype(bool)]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["feature_time"] = pd.to_datetime(
        out["feature_time"], utc=True, errors="coerce"
    )
    for column in ("ret_1h", "future_ret_4h", "oi_value_delta_z_1h"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["month_start"] = pd.to_datetime(
        out["feature_time"].dt.strftime("%Y-%m-01"), utc=True, errors="coerce"
    )
    return (
        out.dropna(subset=["symbol", "feature_time"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def _pivot(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.pivot_table(
        index="feature_time",
        columns="symbol",
        values=column,
        aggfunc="last",
        observed=True,
    ).sort_index()


def build_v108_month_edges(
    oi_source: pd.DataFrame,
    residual_target: pd.DataFrame,
    month_start: pd.Timestamp,
    cfg: V108Config,
) -> pd.DataFrame:
    common = sorted(
        set(oi_source.columns).intersection(residual_target.columns) - {BTC}
    )
    eligible = [
        symbol
        for symbol in common
        if int(oi_source[symbol].notna().sum()) >= cfg.min_edge_samples
        and int(residual_target[symbol].notna().sum()) >= cfg.min_edge_samples
    ]
    if len(eligible) < 4:
        return pd.DataFrame()
    joined = pd.concat(
        {
            "source": oi_source[eligible].clip(-5, 5),
            "target": residual_target[eligible],
        },
        axis=1,
    ).dropna(how="any")
    if len(joined) < cfg.min_edge_samples:
        return pd.DataFrame()
    source_rank = joined["source"].rank(pct=True, method="average").to_numpy(
        dtype=float
    )
    target_rank = joined["target"].rank(pct=True, method="average").to_numpy(
        dtype=float
    )
    source_std = source_rank.std(axis=0, ddof=1)
    target_std = target_rank.std(axis=0, ddof=1)
    valid = (source_std > 0) & (target_std > 0)
    source_z = np.zeros_like(source_rank)
    target_z = np.zeros_like(target_rank)
    source_z[:, valid] = (
        source_rank[:, valid] - source_rank[:, valid].mean(axis=0)
    ) / source_std[valid]
    target_z[:, valid] = (
        target_rank[:, valid] - target_rank[:, valid].mean(axis=0)
    ) / target_std[valid]
    n = len(joined)
    correlation = source_z.T @ target_z / (n - 1)
    shrinkage = np.sqrt(n / (n + cfg.shrinkage_n))
    rows = []
    for leader_index, leader in enumerate(eligible):
        if not valid[leader_index]:
            continue
        for follower_index, follower in enumerate(eligible):
            if leader == follower or not valid[follower_index]:
                continue
            forward = float(correlation[leader_index, follower_index])
            reverse = float(correlation[follower_index, leader_index])
            advantage = forward - reverse
            if forward <= 0 or advantage <= 0:
                continue
            rows.append(
                {
                    "month_start": month_start,
                    "leader_symbol": leader,
                    "follower_symbol": follower,
                    "sample_n": n,
                    "source_target_spearman": forward,
                    "reverse_spearman": reverse,
                    "direction_advantage": advantage,
                    "edge_weight": advantage * shrinkage,
                }
            )
    edges = pd.DataFrame(rows)
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


def _period(month: pd.Timestamp) -> str:
    if month < pd.Timestamp("2026-01-01", tz="UTC"):
        return "development"
    if month < pd.Timestamp("2026-04-01", tz="UTC"):
        return "validation"
    return "holdout"


def build_v108_graph_and_contexts(
    panel: pd.DataFrame,
    cfg: V108Config,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]]]:
    months = sorted(panel["month_start"].dropna().unique())
    edge_frames = []
    contexts: dict[pd.Timestamp, dict[str, Any]] = {}
    for raw_month in months:
        month = pd.Timestamp(raw_month)
        history_start = month - pd.Timedelta(days=cfg.lookback_days)
        history_end = month - pd.Timedelta(hours=4)
        history = panel[
            panel["feature_time"].ge(history_start)
            & panel["feature_time"].lt(history_end)
        ]
        if history.empty:
            continue
        history_span = history["feature_time"].max() - history["feature_time"].min()
        if history_span < pd.Timedelta(days=cfg.minimum_history_days):
            continue
        hourly = history[history["feature_time"].dt.minute.eq(0)]
        historical_ret = _pivot(hourly, "ret_1h")
        historical_future = _pivot(hourly, "future_ret_4h")
        historical_oi = _pivot(hourly, "oi_value_delta_z_1h")
        betas = estimate_v106_betas(historical_ret)
        if BTC not in historical_future.columns:
            continue
        residual_target = pd.DataFrame(index=historical_future.index)
        for symbol in historical_future.columns:
            if symbol == BTC or symbol not in betas.index:
                continue
            residual_target[str(symbol)] = historical_future[symbol] - float(
                betas[symbol]
            ) * historical_future[BTC]
        edges = build_v108_month_edges(historical_oi, residual_target, month, cfg)
        if edges.empty:
            continue
        target = panel[panel["month_start"].eq(month)]
        if target.empty:
            continue
        oi = _pivot(target, "oi_value_delta_z_1h")
        ret = _pivot(target, "ret_1h")
        future = _pivot(target, "future_ret_4h")
        if BTC not in future.columns:
            continue
        graph_symbols = sorted(
            (
                set(edges["leader_symbol"].astype(str))
                | set(edges["follower_symbol"].astype(str))
            )
            & set(oi.columns)
            & set(ret.columns)
            & set(future.columns)
            & set(betas.index.astype(str))
        )
        residual_future = pd.DataFrame(index=future.index)
        for symbol in graph_symbols:
            residual_future[symbol] = future[symbol] - float(betas[symbol]) * future[BTC]
        common_times = oi.index.intersection(ret.index).intersection(future.index)
        contexts[month] = {
            "month_start": month,
            "period": _period(month),
            "times": common_times,
            "symbols": graph_symbols,
            "oi_z_1h": oi.reindex(index=common_times, columns=graph_symbols),
            "ret_1h": ret.reindex(index=common_times, columns=graph_symbols),
            "raw_future_4h": future.reindex(index=common_times, columns=graph_symbols),
            "residual_future_4h": residual_future.reindex(
                index=common_times, columns=graph_symbols
            ),
        }
        edge_frames.append(edges)
    all_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    return all_edges, contexts


def _pressure_matrices(
    context: dict[str, Any],
    edges: pd.DataFrame,
    cfg: V108Config,
    signal_shift_bars: int,
) -> dict[str, pd.DataFrame]:
    symbols: list[str] = context["symbols"]
    oi: pd.DataFrame = context["oi_z_1h"].reindex(columns=symbols)
    ret: pd.DataFrame = context["ret_1h"].reindex(columns=symbols)
    positive_source = oi.sub(cfg.oi_z_threshold).clip(lower=0).where(ret.gt(0), 0.0)
    negative_source = oi.sub(cfg.oi_z_threshold).clip(lower=0).where(ret.lt(0), 0.0)
    if signal_shift_bars:
        positive_source = positive_source.shift(signal_shift_bars)
        negative_source = negative_source.shift(signal_shift_bars)
    outputs = {
        candidate: pd.DataFrame(0.0, index=oi.index, columns=symbols)
        for candidate in CANDIDATES
    }
    source_by_candidate = {
        CANDIDATES[0]: positive_source,
        CANDIDATES[1]: negative_source,
    }
    for follower, group in edges.groupby("follower_symbol", sort=False):
        follower = str(follower)
        if follower not in symbols:
            continue
        local = group[group["leader_symbol"].astype(str).isin(symbols)].copy()
        if local.empty:
            continue
        leaders = local["leader_symbol"].astype(str).tolist()
        weights = pd.to_numeric(local["edge_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        for candidate, source in source_by_candidate.items():
            values = source[leaders].to_numpy(dtype=float)
            finite = np.isfinite(values)
            denominator = np.where(finite, weights[None, :], 0.0).sum(axis=1)
            outputs[candidate][follower] = np.divide(
                np.where(finite, values * weights[None, :], 0.0).sum(axis=1),
                denominator,
                out=np.zeros(len(values)),
                where=denominator > 0,
            )
    return outputs


def build_v108_portfolios(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V108Config,
    signal_shift_bars: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    last_by_candidate: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, context in sorted(contexts.items()):
        month_edges = edges[edges["month_start"].eq(month)]
        if month_edges.empty:
            continue
        pressures = _pressure_matrices(
            context, month_edges, cfg, signal_shift_bars
        )
        raw_future: pd.DataFrame = context["raw_future_4h"]
        residual_future: pd.DataFrame = context["residual_future_4h"]
        for candidate, pressure in pressures.items():
            direction = 1.0 if candidate == CANDIDATES[0] else -1.0
            eligible_times = pressure.index[pressure.gt(0).sum(axis=1).ge(cfg.min_bucket_size)]
            for timestamp in eligible_times:
                last = last_by_candidate.get(candidate)
                if last is not None and pd.Timestamp(timestamp) - last < cooldown:
                    continue
                selected = (
                    pressure.loc[timestamp]
                    .loc[lambda values: values.gt(0)]
                    .sort_values(ascending=False)
                    .head(cfg.max_bucket_size)
                    .index.tolist()
                )
                raw = direction * pd.to_numeric(
                    raw_future.loc[timestamp, selected], errors="coerce"
                )
                residual = direction * pd.to_numeric(
                    residual_future.loc[timestamp, selected], errors="coerce"
                )
                finite = raw.notna() & residual.notna()
                if int(finite.sum()) < cfg.min_bucket_size:
                    continue
                selected = list(pd.Index(selected)[finite.to_numpy()])
                raw_gross = float(raw[finite].mean())
                residual_gross = float(residual[finite].mean())
                rows.append(
                    {
                        "candidate": candidate,
                        "feature_time": timestamp,
                        "entry_day": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                        "entry_month": pd.Timestamp(timestamp).strftime("%Y-%m"),
                        "period": context["period"],
                        "direction": direction,
                        "bucket_size": int(len(selected)),
                        "follower_symbols": "|".join(selected),
                        "mean_oi_pressure": float(
                            pressure.loc[timestamp, selected].mean()
                        ),
                        "raw_gross_4h": raw_gross,
                        "raw_net_4h_20bp": raw_gross - 0.002,
                        "raw_net_4h_30bp": raw_gross - 0.003,
                        "raw_net_4h_50bp": raw_gross - 0.005,
                        "residual_gross_4h": residual_gross,
                        "residual_net_4h_40bp": residual_gross - 0.004,
                    }
                )
                last_by_candidate[candidate] = pd.Timestamp(timestamp)
    return pd.DataFrame(rows)


def summarize_v108(portfolios: pd.DataFrame) -> pd.DataFrame:
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
                }
            )
    return pd.DataFrame(rows)


def randomize_v108_edges(
    edges: pd.DataFrame,
    contexts: dict[pd.Timestamp, dict[str, Any]],
    iteration: int,
    cfg: V108Config,
) -> pd.DataFrame:
    rows = []
    for month, group in edges.groupby("month_start", sort=True):
        month = pd.Timestamp(month)
        symbols = contexts[month]["symbols"]
        rng = np.random.default_rng(cfg.seed + iteration * 1009 + month.month)
        for follower, local in group.groupby("follower_symbol", sort=False):
            follower = str(follower)
            choices = [symbol for symbol in symbols if symbol != follower]
            source_rows = local.sort_values("edge_rank")
            take = min(len(source_rows), len(choices))
            leaders = rng.choice(choices, size=take, replace=False)
            for leader, row in zip(
                leaders, source_rows.head(take).itertuples(index=False), strict=True
            ):
                payload = row._asdict()
                payload["leader_symbol"] = str(leader)
                rows.append(payload)
    return pd.DataFrame(rows)


def reverse_v108_edges(edges: pd.DataFrame, cfg: V108Config) -> pd.DataFrame:
    out = edges.copy()
    out[["leader_symbol", "follower_symbol"]] = out[
        ["follower_symbol", "leader_symbol"]
    ].to_numpy()
    out = out.sort_values(
        ["month_start", "follower_symbol", "edge_weight"],
        ascending=[True, True, False],
    ).drop_duplicates(
        ["month_start", "leader_symbol", "follower_symbol"], keep="first"
    )
    out["edge_rank"] = (
        out.groupby(["month_start", "follower_symbol"], sort=False).cumcount() + 1
    )
    return out[out["edge_rank"].le(cfg.leaders_per_follower)].reset_index(drop=True)


def random_v108_controls(
    contexts: dict[pd.Timestamp, dict[str, Any]],
    edges: pd.DataFrame,
    cfg: V108Config,
) -> pd.DataFrame:
    rows = []
    for iteration in range(cfg.random_iterations):
        randomized = randomize_v108_edges(edges, contexts, iteration, cfg)
        portfolios = build_v108_portfolios(contexts, randomized, cfg)
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


def _bootstrap_ci(sample: pd.DataFrame, cfg: V108Config) -> tuple[float, float]:
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


def _positive_share(values: pd.Series) -> float:
    positive = values.clip(lower=0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf


def _symbol_share(sample: pd.DataFrame) -> float:
    contributions: dict[str, float] = {}
    for row in sample.itertuples(index=False):
        symbols = str(row.follower_symbols).split("|")
        value = float(row.residual_net_4h_40bp) / len(symbols)
        for symbol in symbols:
            contributions[symbol] = contributions.get(symbol, 0.0) + value
    return _positive_share(pd.Series(contributions, dtype=float))


def audit_v108(
    real: pd.DataFrame,
    shifted: pd.DataFrame,
    reversed_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V108Config,
) -> pd.DataFrame:
    family = controls.loc[
        controls["candidate"].eq("FAMILY_MAX"), "mean_residual_net_4h_40bp"
    ].dropna()
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
        month_share = _positive_share(
            sample.groupby("entry_month")["residual_net_4h_40bp"].sum()
        )
        symbol_share = _symbol_share(sample)
        gates = {
            "full_observations_100": lookup["all"].portfolio_observations >= 100,
            "validation_observations_25": lookup["validation"].portfolio_observations >= 25,
            "holdout_observations_25": lookup["holdout"].portfolio_observations >= 25,
            "validation_residual_net40_positive": lookup["validation"].mean_residual_net_4h_40bp > 0,
            "holdout_residual_net40_positive": lookup["holdout"].mean_residual_net_4h_40bp > 0,
            "validation_raw_net20_positive": lookup["validation"].mean_raw_net_4h_20bp > 0,
            "holdout_raw_net20_positive": lookup["holdout"].mean_raw_net_4h_20bp > 0,
            "full_raw_net30_positive": lookup["all"].mean_raw_net_4h_30bp > 0,
            "random_family_p90": percentile >= 0.90,
            "beats_shifted": lookup["all"].mean_residual_net_4h_40bp > shifted_mean,
            "beats_reversed": lookup["all"].mean_residual_net_4h_40bp > reversed_mean,
            "bootstrap_lower_positive": ci_low > 0,
            "month_share_below_35pct": month_share <= 0.35,
            "symbol_share_below_35pct": symbol_share <= 0.35,
        }
        eligible = all(gates.values())
        rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "verdict": "oi_graph_bucket_forward_watch_only"
                if eligible
                else "reject_oi_graph_bucket_candidate",
                "full_residual_net40": lookup["all"].mean_residual_net_4h_40bp,
                "validation_residual_net40": lookup["validation"].mean_residual_net_4h_40bp,
                "holdout_residual_net40": lookup["holdout"].mean_residual_net_4h_40bp,
                "validation_raw_net20": lookup["validation"].mean_raw_net_4h_20bp,
                "holdout_raw_net20": lookup["holdout"].mean_raw_net_4h_20bp,
                "full_raw_net30": lookup["all"].mean_raw_net_4h_30bp,
                "shifted_residual_net40": shifted_mean,
                "reversed_residual_net40": reversed_mean,
                "random_family_percentile": percentile,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "max_positive_month_share": month_share,
                "max_positive_symbol_share": symbol_share,
                "failed_gates": "|".join(
                    name for name, passed in gates.items() if not passed
                ),
            }
        )
    family_verdict = (
        "oi_graph_bucket_forward_watch_only"
        if any(row["eligible"] for row in rows)
        else "reject_oi_graph_bucket_family"
    )
    for row in rows:
        row["family_verdict"] = family_verdict
    return pd.DataFrame(rows)


def write_v108_oi_leader_bucket(
    cfg: V108Config = V108Config(),
) -> dict[str, Path]:
    panel = load_v108_features(cfg)
    edges, contexts = build_v108_graph_and_contexts(panel, cfg)
    real = build_v108_portfolios(contexts, edges, cfg)
    shifted = build_v108_portfolios(contexts, edges, cfg, signal_shift_bars=96)
    reversed_edges = reverse_v108_edges(edges, cfg)
    reversed_portfolios = build_v108_portfolios(contexts, reversed_edges, cfg)
    summary = summarize_v108(real)
    controls = random_v108_controls(contexts, edges, cfg)
    audit = audit_v108(
        real, shifted, reversed_portfolios, summary, controls, cfg
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "feature_panel": root / "oi_feature_panel.parquet",
        "edges": root / "oi_leader_edges.csv",
        "portfolios": root / "timestamp_bucket_portfolios.parquet",
        "shifted": root / "shifted_signal_portfolios.parquet",
        "reversed": root / "reversed_edge_portfolios.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    panel.to_parquet(outputs["feature_panel"], index=False)
    edges.to_csv(outputs["edges"], index=False)
    real.to_parquet(outputs["portfolios"], index=False)
    shifted.to_parquet(outputs["shifted"], index=False)
    reversed_portfolios.to_parquet(outputs["reversed"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = [
        "# v10.8 OI-Leader Downstream Bucket",
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
