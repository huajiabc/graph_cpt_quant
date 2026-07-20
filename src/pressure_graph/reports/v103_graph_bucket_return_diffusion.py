"""Graph-native continuous neighbor-bucket return diffusion audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v10_3_graph_bucket_return_diffusion")
FEATURE_PATH = Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet")
EDGE_PATH = Path("reports/v0_7b_neighbor_graph/neighbor_graph_edges.csv")
SEED = 20260714
CANDIDATES = (
    "GBR1_BROAD_LAG_CATCHUP",
    "GBR2_LAG_NO_TURN",
    "GBR3_COIMPULSE_CONTINUATION",
)
BUCKET_FEATURES = (
    "bucket_ret_15m",
    "bucket_ret_1h",
    "bucket_ret_4h",
    "bucket_positive_breadth_1h",
    "bucket_dispersion_1h",
    "bucket_ret_1h_rank",
    "bucket_excess_ret_1h",
)


@dataclass(frozen=True)
class V103Config:
    feature_path: Path = FEATURE_PATH
    edge_path: Path = EDGE_PATH
    report_root: Path = REPORT_ROOT
    random_iterations: int = 50
    bootstrap_iterations: int = 2000
    min_neighbors: int = 3
    max_positions: int = 3
    cooldown_hours: int = 4
    seed: int = SEED


def _month_start(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.to_datetime(parsed.dt.strftime("%Y-%m-01"), utc=True, errors="coerce")


def load_v103_edges(path: Path = EDGE_PATH) -> pd.DataFrame:
    edges = pd.read_csv(path)
    edges = edges[edges["edge_type"].astype(str).eq("return_corr_30d")].copy()
    edges["month_start"] = pd.to_datetime(edges["month_start"], utc=True, errors="coerce")
    edges["edge_rank"] = pd.to_numeric(edges["edge_rank"], errors="coerce")
    edges = edges.dropna(subset=["month_start", "source_symbol", "neighbor_symbol"])
    return edges.sort_values(["month_start", "source_symbol", "edge_rank"]).reset_index(drop=True)


def _allowed_month_symbols(edges: pd.DataFrame) -> pd.DataFrame:
    source = edges[["month_start", "source_symbol"]].rename(
        columns={"source_symbol": "symbol"}
    )
    neighbor = edges[["month_start", "neighbor_symbol"]].rename(
        columns={"neighbor_symbol": "symbol"}
    )
    return pd.concat([source, neighbor], ignore_index=True).drop_duplicates()


def load_v103_features(
    feature_path: Path,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    allowed = _allowed_month_symbols(edges)
    columns = [
        "symbol",
        "feature_time",
        "entry_time",
        "ret_15m",
        "ret_1h",
        "ret_4h",
        "future_ret_4h",
        "future_ret_12h",
        "warmup_complete",
    ]
    parquet = pq.ParquetFile(feature_path)
    frames = []
    for index in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(index, columns=columns).to_pandas()
        chunk["feature_time"] = pd.to_datetime(chunk["feature_time"], utc=True, errors="coerce")
        chunk["entry_time"] = pd.to_datetime(chunk["entry_time"], utc=True, errors="coerce")
        chunk["month_start"] = _month_start(chunk["feature_time"])
        chunk = chunk.merge(allowed, on=["month_start", "symbol"], how="inner")
        if "warmup_complete" in chunk.columns:
            chunk = chunk[chunk["warmup_complete"].fillna(False).astype(bool)]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for column in [
        "ret_15m",
        "ret_1h",
        "ret_4h",
        "future_ret_4h",
        "future_ret_12h",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return (
        out.dropna(subset=["symbol", "feature_time", "entry_time"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["month_start", "symbol", "feature_time"])
        .reset_index(drop=True)
    )


def real_v103_neighbor_map(edges: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], list[str]]:
    mapping = {}
    for (month, source), group in edges.groupby(
        ["month_start", "source_symbol"], sort=False
    ):
        mapping[(pd.Timestamp(month), str(source))] = (
            group.sort_values("edge_rank")["neighbor_symbol"].astype(str).tolist()
        )
    return mapping


def random_v103_neighbor_map(
    real_map: dict[tuple[pd.Timestamp, str], list[str]],
    iteration: int,
    seed: int = SEED,
) -> dict[tuple[pd.Timestamp, str], list[str]]:
    universes: dict[pd.Timestamp, list[str]] = {}
    for month, source in real_map:
        universes.setdefault(month, []).append(source)
    universes = {month: sorted(set(symbols)) for month, symbols in universes.items()}
    out = {}
    for (month, source), neighbors in sorted(real_map.items()):
        choices = [symbol for symbol in universes[month] if symbol != source]
        local_seed = seed + iteration * 100_003 + sum(ord(char) for char in f"{month}{source}")
        rng = np.random.default_rng(local_seed)
        take = min(len(neighbors), len(choices))
        out[(month, source)] = (
            list(rng.choice(choices, size=take, replace=False)) if take else []
        )
    return out


def _month_pivots(month_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    value_columns = [
        "ret_15m",
        "ret_1h",
        "ret_4h",
        "future_ret_4h",
        "future_ret_12h",
    ]
    return {
        column: month_frame.pivot_table(
            index="feature_time",
            columns="symbol",
            values=column,
            aggfunc="last",
            observed=True,
        ).sort_index()
        for column in value_columns
    }


def build_v103_bucket_panel(
    frame: pd.DataFrame,
    neighbor_map: dict[tuple[pd.Timestamp, str], list[str]],
    min_neighbors: int = 3,
) -> pd.DataFrame:
    rows = []
    for month, month_frame in frame.groupby("month_start", sort=True):
        month = pd.Timestamp(month)
        pivots = _month_pivots(month_frame)
        market_median = pivots["ret_1h"].median(axis=1)
        targets = sorted(source for key_month, source in neighbor_map if key_month == month)
        for target in targets:
            base = month_frame[month_frame["symbol"].astype(str).eq(target)].copy()
            if base.empty:
                continue
            base = base.set_index("feature_time", drop=False).sort_index()
            neighbors = [
                symbol
                for symbol in neighbor_map.get((month, target), [])
                if symbol in pivots["ret_1h"].columns
            ]
            if len(neighbors) < min_neighbors:
                continue
            aligned = pivots["ret_1h"].reindex(base.index)[neighbors]
            count = aligned.notna().sum(axis=1)
            base["neighbor_count"] = count
            base["bucket_ret_1h"] = aligned.mean(axis=1)
            base["bucket_positive_breadth_1h"] = aligned.gt(0).sum(axis=1) / count.replace(0, np.nan)
            base["bucket_dispersion_1h"] = aligned.std(axis=1, ddof=0)
            for column, output in [
                ("ret_15m", "bucket_ret_15m"),
                ("ret_4h", "bucket_ret_4h"),
                ("future_ret_4h", "bucket_future_ret_4h"),
                ("future_ret_12h", "bucket_future_ret_12h"),
            ]:
                base[output] = pivots[column].reindex(base.index)[neighbors].mean(axis=1)
            base["market_median_ret_1h"] = market_median.reindex(base.index)
            base["neighbor_symbols"] = "|".join(neighbors)
            rows.append(base.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[pd.to_numeric(out["neighbor_count"], errors="coerce").ge(min_neighbors)].copy()
    out["bucket_ret_1h_rank"] = out.groupby("feature_time", sort=False)[
        "bucket_ret_1h"
    ].rank(pct=True, method="average")
    out["bucket_excess_ret_1h"] = out["bucket_ret_1h"] - out["market_median_ret_1h"]
    out["target_lag_gap_1h"] = out["bucket_ret_1h"] - out["ret_1h"]
    out["raw_gross_4h"] = out["future_ret_4h"]
    out["raw_gross_12h"] = out["future_ret_12h"]
    out["catchup_gross_4h"] = out["future_ret_4h"] - out["bucket_future_ret_4h"]
    out["catchup_gross_12h"] = out["future_ret_12h"] - out["bucket_future_ret_12h"]
    for cost in (20, 30, 50):
        out[f"raw_net_4h_{cost}bp"] = out["raw_gross_4h"] - cost / 10_000.0
        out[f"raw_net_12h_{cost}bp"] = out["raw_gross_12h"] - cost / 10_000.0
    for cost in (40, 60, 100):
        out[f"catchup_net_4h_{cost}bp"] = out["catchup_gross_4h"] - cost / 10_000.0
        out[f"catchup_net_12h_{cost}bp"] = out["catchup_gross_12h"] - cost / 10_000.0
    out["entry_day"] = out["feature_time"].dt.strftime("%Y-%m-%d")
    out["entry_month"] = out["feature_time"].dt.strftime("%Y-%m")
    out["period"] = np.select(
        [
            out["feature_time"].lt(pd.Timestamp("2026-01-01", tz="UTC")),
            out["feature_time"].lt(pd.Timestamp("2026-04-01", tz="UTC")),
        ],
        ["development", "validation"],
        default="holdout",
    )
    return add_v103_states(out)


def add_v103_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    bucket = (
        out["bucket_ret_1h"].ge(0.005)
        & out["bucket_ret_1h_rank"].ge(0.80)
        & out["bucket_positive_breadth_1h"].ge(0.60)
        & out["bucket_excess_ret_1h"].ge(0.002)
    )
    lag = out["target_lag_gap_1h"].ge(0.003)
    out["GBR1_BROAD_LAG_CATCHUP"] = bucket & lag & out["ret_15m"].gt(0)
    out["GBR2_LAG_NO_TURN"] = bucket & lag & out["ret_15m"].le(0)
    out["GBR3_COIMPULSE_CONTINUATION"] = (
        bucket
        & out["ret_1h"].gt(0)
        & out["target_lag_gap_1h"].le(0.001)
        & out["ret_15m"].gt(0)
    )
    return out


def _transition_and_cooldown(
    panel: pd.DataFrame,
    candidate: str,
    cooldown_hours: int,
) -> pd.DataFrame:
    ordered = panel.sort_values(["symbol", "feature_time"]).copy()
    active = ordered[candidate].fillna(False).astype(bool)
    previous = active.groupby(ordered["symbol"], sort=False).shift(1, fill_value=False)
    transitions = ordered[active & ~previous].copy()
    keep = []
    last_by_symbol: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cooldown_hours)
    for row in transitions.itertuples(index=False):
        last = last_by_symbol.get(str(row.symbol))
        accepted = last is None or pd.Timestamp(row.feature_time) - last >= cooldown
        keep.append(accepted)
        if accepted:
            last_by_symbol[str(row.symbol)] = pd.Timestamp(row.feature_time)
    out = transitions.loc[keep].copy()
    out["candidate"] = candidate
    return out


def build_v103_events(
    panel: pd.DataFrame,
    cooldown_hours: int = 4,
) -> pd.DataFrame:
    frames = [
        _transition_and_cooldown(panel, candidate, cooldown_hours)
        for candidate in CANDIDATES
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_v103_portfolios(
    events: pd.DataFrame,
    max_positions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected_frames = []
    portfolio_rows = []
    metric_columns = [
        "raw_gross_4h",
        "raw_gross_12h",
        "catchup_gross_4h",
        "catchup_gross_12h",
        *[f"raw_net_{horizon}h_{cost}bp" for horizon in (4, 12) for cost in (20, 30, 50)],
        *[
            f"catchup_net_{horizon}h_{cost}bp"
            for horizon in (4, 12)
            for cost in (40, 60, 100)
        ],
    ]
    for (candidate, feature_time), group in events.groupby(
        ["candidate", "feature_time"], sort=True
    ):
        rank_column = (
            "ret_1h" if candidate == "GBR3_COIMPULSE_CONTINUATION" else "target_lag_gap_1h"
        )
        chosen = group.sort_values(rank_column, ascending=False).head(max_positions).copy()
        chosen["portfolio_weight"] = 1.0 / len(chosen)
        selected_frames.append(chosen)
        payload: dict[str, Any] = {
            "candidate": candidate,
            "feature_time": feature_time,
            "period": str(chosen["period"].iloc[0]),
            "entry_day": str(chosen["entry_day"].iloc[0]),
            "entry_month": str(chosen["entry_month"].iloc[0]),
            "positions": int(len(chosen)),
            "symbols": "|".join(chosen["symbol"].astype(str)),
        }
        for column in metric_columns:
            payload[column] = float(pd.to_numeric(chosen[column], errors="coerce").mean())
        portfolio_rows.append(payload)
    return pd.DataFrame(portfolio_rows), pd.concat(selected_frames, ignore_index=True)


def summarize_v103(portfolios: pd.DataFrame) -> pd.DataFrame:
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
                    "mean_positions": float(sample["positions"].mean()),
                    "mean_raw_net_4h_20bp": float(sample["raw_net_4h_20bp"].mean()),
                    "mean_raw_net_4h_30bp": float(sample["raw_net_4h_30bp"].mean()),
                    "mean_raw_net_4h_50bp": float(sample["raw_net_4h_50bp"].mean()),
                    "mean_catchup_net_4h_40bp": float(
                        sample["catchup_net_4h_40bp"].mean()
                    ),
                    "mean_catchup_net_4h_60bp": float(
                        sample["catchup_net_4h_60bp"].mean()
                    ),
                    "mean_catchup_net_4h_100bp": float(
                        sample["catchup_net_4h_100bp"].mean()
                    ),
                    "mean_raw_net_12h_20bp": float(sample["raw_net_12h_20bp"].mean()),
                    "mean_catchup_net_12h_40bp": float(
                        sample["catchup_net_12h_40bp"].mean()
                    ),
                    "win_rate_catchup_net_4h_40bp": float(
                        sample["catchup_net_4h_40bp"].gt(0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def shifted_v103_panel(panel: pd.DataFrame, bars: int = 96) -> pd.DataFrame:
    out = panel.sort_values(["symbol", "feature_time"]).copy()
    for column in BUCKET_FEATURES:
        out[column] = out.groupby("symbol", sort=False)[column].shift(bars)
    out["target_lag_gap_1h"] = out["bucket_ret_1h"] - out["ret_1h"]
    return add_v103_states(out)


def _prepare_control_months(
    frame: pd.DataFrame,
    real_map: dict[tuple[pd.Timestamp, str], list[str]],
) -> dict[pd.Timestamp, dict[str, Any]]:
    prepared = {}
    for month, month_frame in frame.groupby("month_start", sort=True):
        month = pd.Timestamp(month)
        targets = sorted(source for key_month, source in real_map if key_month == month)
        pivots = _month_pivots(month_frame)
        symbols = sorted(pivots["ret_1h"].columns.astype(str))
        times = pivots["ret_1h"].index
        prepared[month] = {
            "targets": targets,
            "symbols": symbols,
            "times": times,
            "matrices": {
                column: pivot.reindex(index=times, columns=symbols).to_numpy(dtype=float)
                for column, pivot in pivots.items()
            },
            "market_median": pivots["ret_1h"].reindex(times).median(axis=1).to_numpy(dtype=float),
        }
    return prepared


def _row_percentile(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=1, pct=True, method="average").to_numpy(dtype=float)


def _fast_control_means(
    prepared: dict[pd.Timestamp, dict[str, Any]],
    neighbor_map: dict[tuple[pd.Timestamp, str], list[str]],
    cfg: V103Config,
) -> dict[str, tuple[int, float]]:
    values: dict[str, list[float]] = {candidate: [] for candidate in CANDIDATES}
    counts = {candidate: 0 for candidate in CANDIDATES}
    last_time: dict[tuple[str, str], pd.Timestamp] = {}
    previous_state: dict[tuple[str, str], bool] = {}
    cooldown = pd.Timedelta(hours=cfg.cooldown_hours)
    for month, data in prepared.items():
        targets: list[str] = data["targets"]
        symbols: list[str] = data["symbols"]
        times: pd.DatetimeIndex = data["times"]
        matrices: dict[str, np.ndarray] = data["matrices"]
        symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
        target_indices = [symbol_index.get(target, -1) for target in targets]
        valid_targets = [index >= 0 for index in target_indices]
        shape = (len(times), len(targets))
        bucket_ret_1h = np.full(shape, np.nan)
        bucket_breadth = np.full(shape, np.nan)
        bucket_future_4h = np.full(shape, np.nan)
        neighbor_counts = np.zeros(shape, dtype=int)
        for column, target in enumerate(targets):
            neighbor_indices = [
                symbol_index[symbol]
                for symbol in neighbor_map.get((month, target), [])
                if symbol in symbol_index
            ]
            if len(neighbor_indices) < cfg.min_neighbors:
                continue
            current = matrices["ret_1h"][:, neighbor_indices]
            count = np.isfinite(current).sum(axis=1)
            neighbor_counts[:, column] = count
            bucket_ret_1h[:, column] = np.divide(
                np.nansum(current, axis=1),
                count,
                out=np.full(len(times), np.nan),
                where=count > 0,
            )
            bucket_breadth[:, column] = np.sum(current > 0, axis=1) / np.where(
                count > 0, count, np.nan
            )
            future = matrices["future_ret_4h"][:, neighbor_indices]
            future_count = np.isfinite(future).sum(axis=1)
            bucket_future_4h[:, column] = np.divide(
                np.nansum(future, axis=1),
                future_count,
                out=np.full(len(times), np.nan),
                where=future_count > 0,
            )
        target_ret_1h = np.full(shape, np.nan)
        target_ret_15m = np.full(shape, np.nan)
        target_future_4h = np.full(shape, np.nan)
        for column, index in enumerate(target_indices):
            if valid_targets[column]:
                target_ret_1h[:, column] = matrices["ret_1h"][:, index]
                target_ret_15m[:, column] = matrices["ret_15m"][:, index]
                target_future_4h[:, column] = matrices["future_ret_4h"][:, index]
        rank_input = bucket_ret_1h.copy()
        rank_input[(neighbor_counts < cfg.min_neighbors) | ~np.isfinite(target_ret_1h)] = np.nan
        rank = _row_percentile(rank_input)
        excess = bucket_ret_1h - data["market_median"][:, None]
        lag = bucket_ret_1h - target_ret_1h
        base = (
            (neighbor_counts >= cfg.min_neighbors)
            & (bucket_ret_1h >= 0.005)
            & (rank >= 0.80)
            & (bucket_breadth >= 0.60)
            & (excess >= 0.002)
        )
        states = {
            "GBR1_BROAD_LAG_CATCHUP": base & (lag >= 0.003) & (target_ret_15m > 0),
            "GBR2_LAG_NO_TURN": base & (lag >= 0.003) & (target_ret_15m <= 0),
            "GBR3_COIMPULSE_CONTINUATION": (
                base & (target_ret_1h > 0) & (lag <= 0.001) & (target_ret_15m > 0)
            ),
        }
        catchup = target_future_4h - bucket_future_4h - 0.004
        for candidate, state in states.items():
            accepted = np.zeros_like(state, dtype=bool)
            for column, target in enumerate(targets):
                prior = previous_state.get((candidate, target), False)
                valid = np.flatnonzero(
                    np.isfinite(target_ret_1h[:, column])
                    & np.isfinite(target_ret_15m[:, column])
                )
                local_state = state[valid, column]
                transition = local_state.copy()
                if len(local_state):
                    transition[1:] &= ~local_state[:-1]
                    transition[0] &= ~prior
                    previous_state[(candidate, target)] = bool(local_state[-1])
                for index in valid[np.flatnonzero(transition)]:
                    timestamp = pd.Timestamp(times[index])
                    last = last_time.get((candidate, target))
                    if last is None or timestamp - last >= cooldown:
                        accepted[index, column] = True
                        last_time[(candidate, target)] = timestamp
            score = target_ret_1h if candidate == "GBR3_COIMPULSE_CONTINUATION" else lag
            for index in np.flatnonzero(accepted.any(axis=1)):
                counts[candidate] += 1
                columns = np.flatnonzero(accepted[index])
                order = columns[np.argsort(score[index, columns])[::-1]]
                chosen = order[: cfg.max_positions]
                outcomes = catchup[index, chosen]
                finite = outcomes[np.isfinite(outcomes)]
                if len(finite):
                    values[candidate].append(float(np.mean(finite)))
    return {
        candidate: (
            counts[candidate],
            float(np.mean(candidate_values)) if candidate_values else np.nan,
        )
        for candidate, candidate_values in values.items()
    }


def random_v103_controls(
    frame: pd.DataFrame,
    real_map: dict[tuple[pd.Timestamp, str], list[str]],
    cfg: V103Config,
) -> pd.DataFrame:
    prepared = _prepare_control_months(frame, real_map)
    rows = []
    for iteration in range(cfg.random_iterations):
        mapping = random_v103_neighbor_map(real_map, iteration, cfg.seed)
        results = _fast_control_means(prepared, mapping, cfg)
        means = {}
        for candidate in CANDIDATES:
            count, means[candidate] = results[candidate]
            rows.append(
                {
                    "iteration": iteration,
                    "candidate": candidate,
                    "portfolio_observations": count,
                    "mean_catchup_net_4h_40bp": means[candidate],
                }
            )
        finite = [value for value in means.values() if np.isfinite(value)]
        rows.append(
            {
                "iteration": iteration,
                "candidate": "FAMILY_MAX",
                "portfolio_observations": int(sum(results[item][0] for item in CANDIDATES)),
                "mean_catchup_net_4h_40bp": max(finite) if finite else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _stability_and_concentration(
    portfolios: pd.DataFrame,
    selected: pd.DataFrame,
    candidate: str,
    cfg: V103Config,
) -> dict[str, Any]:
    sample = portfolios[portfolios["candidate"].eq(candidate)].sort_values("feature_time")
    daily = []
    for _, group in sample.groupby("entry_day", sort=True):
        values = pd.to_numeric(
            group["catchup_net_4h_40bp"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if len(values):
            daily.append(values)
    rng = np.random.default_rng(cfg.seed + CANDIDATES.index(candidate))
    boot = []
    if daily:
        for _ in range(cfg.bootstrap_iterations):
            chosen = rng.integers(0, len(daily), len(daily))
            values = np.concatenate([daily[index] for index in chosen])
            boot.append(float(np.mean(values)))
    positive_month = sample.groupby("entry_month")["catchup_net_4h_40bp"].sum().clip(lower=0)
    positive_day = sample.groupby("entry_day")["catchup_net_4h_40bp"].sum().clip(lower=0)
    chosen = selected[selected["candidate"].eq(candidate)].copy()
    chosen["weighted_catchup"] = chosen["catchup_net_4h_40bp"] * chosen["portfolio_weight"]
    positive_symbol = chosen.groupby("symbol")["weighted_catchup"].sum().clip(lower=0)

    def share(values: pd.Series) -> float:
        return float(values.max() / values.sum()) if values.sum() > 0 else np.inf

    chrono_indices = np.array_split(np.arange(len(sample)), 5) if len(sample) >= 5 else []
    chrono_means = [
        float(sample.iloc[indices]["catchup_net_4h_40bp"].mean())
        for indices in chrono_indices
    ]
    return {
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
        "max_positive_month_share": share(positive_month),
        "max_positive_day_share": share(positive_day),
        "max_positive_symbol_share": share(positive_symbol),
        "chronological_bucket_means": "|".join(f"{value:.10f}" for value in chrono_means),
        "chronological_min": min(chrono_means) if chrono_means else np.nan,
        "target_symbols": int(chosen["symbol"].nunique()),
    }


def audit_v103(
    portfolios: pd.DataFrame,
    selected: pd.DataFrame,
    shifted_portfolios: pd.DataFrame,
    summary: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V103Config,
) -> pd.DataFrame:
    family = pd.to_numeric(
        random_controls.loc[
            random_controls["candidate"].eq("FAMILY_MAX"),
            "mean_catchup_net_4h_40bp",
        ],
        errors="coerce",
    )
    rows = []
    eligibility = {}
    for candidate in CANDIDATES:
        lookup = {
            row.scope: row
            for row in summary[summary["candidate"].eq(candidate)].itertuples(index=False)
        }
        full = lookup["all"]
        validation = lookup["validation"]
        holdout = lookup["holdout"]
        stability = _stability_and_concentration(portfolios, selected, candidate, cfg)
        shifted_mean = float(
            shifted_portfolios.loc[
                shifted_portfolios["candidate"].eq(candidate),
                "catchup_net_4h_40bp",
            ].mean()
        )
        real_mean = float(full.mean_catchup_net_4h_40bp)
        random_percentile = float(family.lt(real_mean).mean())
        gates = {
            "full_n_200": (full.portfolio_observations >= 200, full.portfolio_observations),
            "validation_n_60": (
                validation.portfolio_observations >= 60,
                validation.portfolio_observations,
            ),
            "holdout_n_60": (
                holdout.portfolio_observations >= 60,
                holdout.portfolio_observations,
            ),
            "ten_target_symbols": (stability["target_symbols"] >= 10, stability["target_symbols"]),
            "eight_active_months": (full.active_months >= 8, full.active_months),
            "thirty_active_days": (full.active_days >= 30, full.active_days),
            "validation_catchup_net40_positive": (
                validation.mean_catchup_net_4h_40bp > 0,
                validation.mean_catchup_net_4h_40bp,
            ),
            "holdout_catchup_net40_positive": (
                holdout.mean_catchup_net_4h_40bp > 0,
                holdout.mean_catchup_net_4h_40bp,
            ),
            "validation_raw_net20_positive": (
                validation.mean_raw_net_4h_20bp > 0,
                validation.mean_raw_net_4h_20bp,
            ),
            "holdout_raw_net20_positive": (
                holdout.mean_raw_net_4h_20bp > 0,
                holdout.mean_raw_net_4h_20bp,
            ),
            "full_catchup_net60_positive": (
                full.mean_catchup_net_4h_60bp > 0,
                full.mean_catchup_net_4h_60bp,
            ),
            "bootstrap_lower_positive": (
                stability["bootstrap_ci_low"] > 0,
                stability["bootstrap_ci_low"],
            ),
            "random_graph_family_p90": (random_percentile >= 0.90, random_percentile),
            "beats_shifted_placebo": (real_mean > shifted_mean, real_mean - shifted_mean),
            "month_share_below_35pct": (
                stability["max_positive_month_share"] <= 0.35,
                stability["max_positive_month_share"],
            ),
            "day_share_below_35pct": (
                stability["max_positive_day_share"] <= 0.35,
                stability["max_positive_day_share"],
            ),
            "symbol_share_below_35pct": (
                stability["max_positive_symbol_share"] <= 0.35,
                stability["max_positive_symbol_share"],
            ),
            "five_chronological_buckets_nonnegative": (
                stability["chronological_min"] >= 0,
                stability["chronological_min"],
            ),
        }
        eligible = all(bool(passed) for passed, _ in gates.values())
        eligibility[candidate] = eligible
        for check, (passed, value) in gates.items():
            rows.append(
                {
                    "candidate": candidate,
                    "check": check,
                    "passed": bool(passed),
                    "value": float(value),
                    "eligible": eligible,
                    "random_graph_percentile": random_percentile,
                    "shifted_mean_catchup_net40": shifted_mean,
                    **stability,
                }
            )
    audit = pd.DataFrame(rows)
    audit["verdict"] = (
        "graph_bucket_research_candidate_only"
        if any(eligibility.values())
        else "reject_fixed_top5_graph_bucket_diffusion"
    )
    return audit


def _write_notes(
    path: Path,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    verdict = str(audit["verdict"].iloc[0]) if not audit.empty else "not_run"
    lines = [
        "# v10.3 Graph Bucket-Return Diffusion",
        "",
        f"Status: `{verdict}`. Offline graph-native research only.",
        "",
    ]
    focal = summary[summary["scope"].isin(["all", "validation", "holdout"])]
    for row in focal.sort_values(["candidate", "scope"]).itertuples(index=False):
        lines.append(
            f"- {row.candidate}/{row.scope}: n={row.portfolio_observations}, "
            f"raw_net20={row.mean_raw_net_4h_20bp:.4%}, "
            f"catchup_net40={row.mean_catchup_net_4h_40bp:.4%}."
        )
    lines.extend(
        [
            "",
            "Cost labels are total round-trip basis points.",
            "P2 and all live permissions remain unchanged.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v103_graph_bucket_return_diffusion(
    cfg: V103Config = V103Config(),
) -> dict[str, Path]:
    edges = load_v103_edges(cfg.edge_path)
    frame = load_v103_features(cfg.feature_path, edges)
    real_map = real_v103_neighbor_map(edges)
    panel = build_v103_bucket_panel(frame, real_map, cfg.min_neighbors)
    events = build_v103_events(panel, cfg.cooldown_hours)
    portfolios, selected = build_v103_portfolios(events, cfg.max_positions)
    summary = summarize_v103(portfolios)
    shifted_panel = shifted_v103_panel(panel)
    shifted_events = build_v103_events(shifted_panel, cfg.cooldown_hours)
    shifted_portfolios, _ = build_v103_portfolios(shifted_events, cfg.max_positions)
    random_controls = random_v103_controls(frame, real_map, cfg)
    audit = audit_v103(
        portfolios,
        selected,
        shifted_portfolios,
        summary,
        random_controls,
        cfg,
    )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "bucket_panel": root / "bucket_feature_panel.parquet",
        "selected_events": root / "selected_target_events.parquet",
        "portfolios": root / "timestamp_portfolios.csv",
        "summary": root / "candidate_summary.csv",
        "random_controls": root / "random_graph_controls.csv",
        "audit": root / "candidate_audit.csv",
        "notes": root / "candidate_notes.md",
    }
    panel.to_parquet(outputs["bucket_panel"], index=False)
    selected.to_parquet(outputs["selected_events"], index=False)
    portfolios.to_csv(outputs["portfolios"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    random_controls.to_csv(outputs["random_controls"], index=False)
    audit.to_csv(outputs["audit"], index=False)
    _write_notes(outputs["notes"], summary, audit)
    return outputs
