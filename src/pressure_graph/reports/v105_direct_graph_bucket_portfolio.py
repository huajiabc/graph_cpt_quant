"""Directly trade graph-neighbor buckets instead of target-coin catch-up."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v103_graph_bucket_return_diffusion import (
    EDGE_PATH,
    _prepare_control_months,
    _row_percentile,
    load_v103_edges,
    random_v103_neighbor_map,
    real_v103_neighbor_map,
)


REPORT_ROOT = Path("reports/v10_5_direct_graph_bucket_portfolio")
PANEL_PATH = Path("reports/v10_3_graph_bucket_return_diffusion/bucket_feature_panel.parquet")
CANDIDATES = ("GBM1_BROAD_BUCKET_CONTINUATION", "GBM2_BUCKET_TURN_REVERSAL")
SIGNAL_FEATURES = (
    "bucket_ret_15m",
    "bucket_ret_1h",
    "bucket_positive_breadth_1h",
    "bucket_ret_1h_rank",
    "bucket_excess_ret_1h",
)


@dataclass(frozen=True)
class V105Config:
    panel_path: Path = PANEL_PATH
    edge_path: Path = EDGE_PATH
    report_root: Path = REPORT_ROOT
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    min_neighbors: int = 3
    max_buckets: int = 3
    cooldown_hours: int = 4
    seed: int = 20260714


def add_v105_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    base = (
        out["bucket_ret_1h"].ge(0.005)
        & out["bucket_ret_1h_rank"].ge(0.80)
        & out["bucket_positive_breadth_1h"].ge(0.60)
        & out["bucket_excess_ret_1h"].ge(0.002)
    )
    out[CANDIDATES[0]] = base
    out[CANDIDATES[1]] = base & out["bucket_ret_15m"].le(0)
    return out


def _transition_events(
    panel: pd.DataFrame, candidate: str, cooldown_hours: int
) -> pd.DataFrame:
    ordered = panel.sort_values(["symbol", "feature_time"]).copy()
    active = ordered[candidate].fillna(False).astype(bool)
    previous = active.groupby(ordered["symbol"], sort=False).shift(1, fill_value=False)
    transitions = ordered[active & ~previous].copy()
    keep: list[bool] = []
    last_by_source: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cooldown_hours)
    for row in transitions.itertuples(index=False):
        source = str(row.symbol)
        timestamp = pd.Timestamp(row.feature_time)
        last = last_by_source.get(source)
        accepted = last is None or timestamp - last >= cooldown
        keep.append(accepted)
        if accepted:
            last_by_source[source] = timestamp
    out = transitions.loc[keep].copy()
    out["candidate"] = candidate
    return out


def build_v105_portfolios(
    panel: pd.DataFrame, cfg: V105Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.concat(
        [_transition_events(panel, candidate, cfg.cooldown_hours) for candidate in CANDIDATES],
        ignore_index=True,
    )
    selected_frames = []
    rows: list[dict[str, Any]] = []
    for (candidate, timestamp), group in events.groupby(
        ["candidate", "feature_time"], sort=True
    ):
        chosen = group.sort_values("bucket_excess_ret_1h", ascending=False).head(
            cfg.max_buckets
        ).copy()
        direction = 1.0 if candidate == CANDIDATES[0] else -1.0
        gross = direction * pd.to_numeric(
            chosen["bucket_future_ret_4h"], errors="coerce"
        )
        finite = gross.dropna()
        if finite.empty:
            continue
        chosen["direction"] = direction
        chosen["sleeve_gross_4h"] = gross
        selected_frames.append(chosen)
        row: dict[str, Any] = {
            "candidate": candidate,
            "feature_time": timestamp,
            "period": str(chosen["period"].iloc[0]),
            "entry_day": str(chosen["entry_day"].iloc[0]),
            "entry_month": str(chosen["entry_month"].iloc[0]),
            "bucket_sleeves": int(len(finite)),
            "source_symbols": "|".join(chosen["symbol"].astype(str)),
            "gross_4h": float(finite.mean()),
        }
        for cost in (20, 30, 50):
            row[f"net_4h_{cost}bp"] = row["gross_4h"] - cost / 10_000.0
        rows.append(row)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    return pd.DataFrame(rows), selected


def summarize_v105(portfolios: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_bucket_sleeves": float(sample["bucket_sleeves"].mean()),
                    "mean_gross_4h": float(sample["gross_4h"].mean()),
                    "mean_net_4h_20bp": float(sample["net_4h_20bp"].mean()),
                    "mean_net_4h_30bp": float(sample["net_4h_30bp"].mean()),
                    "mean_net_4h_50bp": float(sample["net_4h_50bp"].mean()),
                    "win_rate_net_4h_20bp": float(sample["net_4h_20bp"].gt(0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _fast_random_means(
    prepared: dict[pd.Timestamp, dict[str, Any]],
    neighbor_map: dict[tuple[pd.Timestamp, str], list[str]],
    cfg: V105Config,
) -> dict[str, tuple[int, float]]:
    outcomes: dict[str, list[float]] = {candidate: [] for candidate in CANDIDATES}
    counts = {candidate: 0 for candidate in CANDIDATES}
    previous: dict[tuple[str, str], bool] = {}
    last_time: dict[tuple[str, str], pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, data in prepared.items():
        targets: list[str] = data["targets"]
        symbols: list[str] = data["symbols"]
        times: pd.DatetimeIndex = data["times"]
        matrices: dict[str, np.ndarray] = data["matrices"]
        symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
        shape = (len(times), len(targets))
        bucket_ret_15m = np.full(shape, np.nan)
        bucket_ret_1h = np.full(shape, np.nan)
        bucket_breadth = np.full(shape, np.nan)
        bucket_future = np.full(shape, np.nan)
        neighbor_counts = np.zeros(shape, dtype=int)
        for column, target in enumerate(targets):
            indices = [
                symbol_index[symbol]
                for symbol in neighbor_map.get((month, target), [])
                if symbol in symbol_index
            ]
            if len(indices) < cfg.min_neighbors:
                continue
            current = matrices["ret_1h"][:, indices]
            count = np.isfinite(current).sum(axis=1)
            neighbor_counts[:, column] = count
            bucket_ret_1h[:, column] = np.divide(
                np.nansum(current, axis=1), count,
                out=np.full(len(times), np.nan), where=count > 0,
            )
            bucket_breadth[:, column] = np.sum(current > 0, axis=1) / np.where(
                count > 0, count, np.nan
            )
            for matrix_name, destination in (
                ("ret_15m", bucket_ret_15m),
                ("future_ret_4h", bucket_future),
            ):
                values = matrices[matrix_name][:, indices]
                valid = np.isfinite(values).sum(axis=1)
                destination[:, column] = np.divide(
                    np.nansum(values, axis=1), valid,
                    out=np.full(len(times), np.nan), where=valid > 0,
                )
        rank_input = bucket_ret_1h.copy()
        rank_input[neighbor_counts < cfg.min_neighbors] = np.nan
        rank = _row_percentile(rank_input)
        excess = bucket_ret_1h - data["market_median"][:, None]
        base = (
            (neighbor_counts >= cfg.min_neighbors)
            & (bucket_ret_1h >= 0.005)
            & (rank >= 0.80)
            & (bucket_breadth >= 0.60)
            & (excess >= 0.002)
        )
        states = {CANDIDATES[0]: base, CANDIDATES[1]: base & (bucket_ret_15m <= 0)}
        for candidate, state in states.items():
            accepted = np.zeros_like(state, dtype=bool)
            for column, target in enumerate(targets):
                valid = np.flatnonzero(np.isfinite(bucket_ret_1h[:, column]))
                local = state[valid, column]
                transition = local.copy()
                prior = previous.get((candidate, target), False)
                if len(local):
                    transition[1:] &= ~local[:-1]
                    transition[0] &= ~prior
                    previous[(candidate, target)] = bool(local[-1])
                for index in valid[np.flatnonzero(transition)]:
                    timestamp = pd.Timestamp(times[index])
                    last = last_time.get((candidate, target))
                    if last is None or timestamp - last >= cooldown:
                        accepted[index, column] = True
                        last_time[(candidate, target)] = timestamp
            direction = 1.0 if candidate == CANDIDATES[0] else -1.0
            for index in np.flatnonzero(accepted.any(axis=1)):
                columns = np.flatnonzero(accepted[index])
                ordered = columns[np.argsort(excess[index, columns])[::-1]][: cfg.max_buckets]
                values = direction * bucket_future[index, ordered]
                finite = values[np.isfinite(values)]
                if len(finite):
                    counts[candidate] += 1
                    outcomes[candidate].append(float(np.mean(finite) - 0.002))
    return {
        candidate: (counts[candidate], float(np.mean(outcomes[candidate])))
        if outcomes[candidate]
        else (counts[candidate], np.nan)
        for candidate in CANDIDATES
    }


def random_v105_controls(panel: pd.DataFrame, cfg: V105Config) -> pd.DataFrame:
    edges = load_v103_edges(cfg.edge_path)
    real_map = real_v103_neighbor_map(edges)
    raw = panel.drop_duplicates(["symbol", "feature_time"])[
        ["symbol", "feature_time", "month_start", "ret_15m", "ret_1h", "ret_4h", "future_ret_4h", "future_ret_12h"]
    ]
    prepared = _prepare_control_months(raw, real_map)
    rows = []
    for iteration in range(cfg.random_iterations):
        mapping = random_v103_neighbor_map(real_map, iteration, cfg.seed)
        results = _fast_random_means(prepared, mapping, cfg)
        finite = []
        for candidate, (count, mean) in results.items():
            finite.append(mean)
            rows.append(
                {"iteration": iteration, "candidate": candidate,
                 "portfolio_observations": count, "mean_net_4h_20bp": mean}
            )
        rows.append(
            {"iteration": iteration, "candidate": "FAMILY_MAX",
             "portfolio_observations": sum(value[0] for value in results.values()),
             "mean_net_4h_20bp": float(np.nanmax(finite))}
        )
    return pd.DataFrame(rows)


def shifted_v105_panel(panel: pd.DataFrame, bars: int = 96) -> pd.DataFrame:
    out = panel.sort_values(["symbol", "feature_time"]).copy()
    for column in SIGNAL_FEATURES:
        out[column] = out.groupby("symbol", sort=False)[column].shift(bars)
    return add_v105_states(out)


def _bootstrap_ci(sample: pd.DataFrame, cfg: V105Config) -> tuple[float, float]:
    days = [group["net_4h_20bp"].dropna().to_numpy() for _, group in sample.groupby("entry_day")]
    days = [values for values in days if len(values)]
    if not days:
        return np.nan, np.nan
    rng = np.random.default_rng(cfg.seed)
    boot = []
    for _ in range(cfg.bootstrap_iterations):
        chosen = rng.integers(0, len(days), len(days))
        boot.append(float(np.mean(np.concatenate([days[index] for index in chosen]))))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def audit_v105(
    portfolios: pd.DataFrame,
    shifted: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    cfg: V105Config,
) -> pd.DataFrame:
    family = controls.loc[controls["candidate"].eq("FAMILY_MAX"), "mean_net_4h_20bp"]
    rows = []
    eligible_any = False
    for candidate in CANDIDATES:
        lookup = {row.scope: row for row in summary[summary["candidate"].eq(candidate)].itertuples(index=False)}
        sample = portfolios[portfolios["candidate"].eq(candidate)]
        shifted_mean = float(shifted.loc[shifted["candidate"].eq(candidate), "net_4h_20bp"].mean())
        ci_low, ci_high = _bootstrap_ci(sample, cfg)
        positive_month = sample.groupby("entry_month")["net_4h_20bp"].sum().clip(lower=0)
        month_share = float(positive_month.max() / positive_month.sum()) if positive_month.sum() > 0 else np.inf
        percentile = float(family.lt(lookup["all"].mean_net_4h_20bp).mean())
        gates = {
            "full_observations_100": lookup["all"].portfolio_observations >= 100,
            "validation_observations_25": lookup["validation"].portfolio_observations >= 25,
            "holdout_observations_25": lookup["holdout"].portfolio_observations >= 25,
            "full_positive": lookup["all"].mean_net_4h_20bp > 0,
            "validation_positive": lookup["validation"].mean_net_4h_20bp > 0,
            "holdout_positive": lookup["holdout"].mean_net_4h_20bp > 0,
            "random_family_p90": percentile >= 0.90,
            "beats_shifted": lookup["all"].mean_net_4h_20bp > shifted_mean,
            "bootstrap_lower_positive": ci_low > 0,
            "month_share_below_35pct": month_share <= 0.35,
        }
        eligible = all(gates.values())
        eligible_any |= eligible
        rows.append(
            {"candidate": candidate, "eligible": eligible,
             "verdict": "direct_bucket_forward_watch_only" if eligible else "reject_direct_bucket_candidate",
             "full_mean_net_4h_20bp": lookup["all"].mean_net_4h_20bp,
             "validation_mean_net_4h_20bp": lookup["validation"].mean_net_4h_20bp,
             "holdout_mean_net_4h_20bp": lookup["holdout"].mean_net_4h_20bp,
             "shifted_mean_net_4h_20bp": shifted_mean, "random_family_percentile": percentile,
             "bootstrap_ci_low": ci_low, "bootstrap_ci_high": ci_high,
             "max_positive_month_share": month_share,
             "failed_gates": "|".join(name for name, passed in gates.items() if not passed)}
        )
    verdict = "direct_bucket_forward_watch_only" if eligible_any else "reject_direct_graph_bucket_family"
    for row in rows:
        row["family_verdict"] = verdict
    return pd.DataFrame(rows)


def write_v105_direct_graph_bucket_portfolio(cfg: V105Config = V105Config()) -> dict[str, Path]:
    panel = pd.read_parquet(cfg.panel_path)
    panel = add_v105_states(panel)
    portfolios, selected = build_v105_portfolios(panel, cfg)
    shifted_portfolios, _ = build_v105_portfolios(shifted_v105_panel(panel), cfg)
    summary = summarize_v105(portfolios)
    controls = random_v105_controls(panel, cfg)
    audit = audit_v105(portfolios, shifted_portfolios, summary, controls, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "portfolios": root / "timestamp_bucket_portfolios.parquet",
        "selected": root / "selected_bucket_sleeves.parquet",
        "summary": root / "candidate_summary.csv",
        "controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    portfolios.to_parquet(outputs["portfolios"], index=False)
    selected.to_parquet(outputs["selected"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    lines = ["# v10.5 Direct Graph-Bucket Portfolio", "", f"Status: `{audit['family_verdict'].iloc[0]}`.", ""]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.candidate}: net20={row.full_mean_net_4h_20bp:.4%}, "
            f"validation={row.validation_mean_net_4h_20bp:.4%}, holdout={row.holdout_mean_net_4h_20bp:.4%}, "
            f"random percentile={row.random_family_percentile:.1%}."
        )
    lines.extend(["", "No PaperLive or live permission changed."])
    outputs["notes"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs
