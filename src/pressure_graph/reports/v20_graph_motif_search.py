"""v2.0 Graph Motif Search framework.

The search space is deliberately grammar-bound. ACO samples motif paths, GA
combines path/portfolio components, and SA performs local architecture
refinement around top GA candidates. All outputs are offline diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import math
import random
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v10a_cic_basket_portfolio import V10AConfig
from pressure_graph.reports.v13d_cp60_context_protection import (
    PROTECTION_COST_BPS,
    _apply_protected_checkpoint,
)
from pressure_graph.reports.v13e_cp60_beta_protection_stability import _prepare_sample_at_cost


REPORT_ROOT = Path("reports/v2_0_graph_motif_search")
SEED = 20260613
SEARCH_END = pd.Timestamp("2026-02-01T00:00:00Z")
VALIDATION_END = pd.Timestamp("2026-05-01T00:00:00Z")
PROTECT_A_BETA_HIGH_THRESHOLD = 99.97866821289062


@dataclass(frozen=True)
class V20Config:
    report_root: Path = REPORT_ROOT
    v10a: V10AConfig = V10AConfig()
    seed: int = SEED
    aco_iterations: int = 12
    aco_ants: int = 30
    ga_population: int = 48
    ga_generations: int = 14
    sa_steps: int = 120


@dataclass(frozen=True)
class MotifSpec:
    market_context: str = "market_impulse_density_high"
    cluster_context: str = "none"
    local_state: str = "cic1_cic2"
    execution: str = "reclaim_1pct"
    exit_type: str = "vol_regime_fast"
    max_positions: int = 8
    overflow_rule: str = "none"
    overflow_trigger: int = 9
    overflow_slots: int = 0
    cic1_overflow_size: float = 0.0
    cic2_overflow_size: float = 0.0
    checkpoint_rule: str = "none"
    protect_cap: int = 0


GRAMMAR: dict[str, list[Any]] = {
    "market_context": ["market_impulse_density_high", "BTC_up", "BTC_chop"],
    "cluster_context": ["none", "cluster_impulse_high"],
    "local_state": ["cic1_cic2", "cic1_extreme", "cic2_broad"],
    "execution": ["reclaim_1pct"],
    "exit_type": ["vol_regime_fast"],
    "max_positions": [5, 8, 10, 12],
    "overflow_rule": ["none", "O6_late9", "O6_late15"],
    "checkpoint_rule": ["none", "CP60", "Protect_A_cap1", "Protect_A_cap2"],
}


BENCHMARKS: dict[str, MotifSpec] = {
    "B0_P2_max8": MotifSpec(max_positions=8),
    "B1_P2_max8_O6": MotifSpec(
        max_positions=8,
        overflow_rule="O6_late9",
        overflow_trigger=9,
        overflow_slots=4,
        cic1_overflow_size=0.50,
        cic2_overflow_size=0.25,
    ),
    "B2_P2_max8_CP60": MotifSpec(max_positions=8, checkpoint_rule="CP60"),
    "B3_P2_max8_CP60_O6": MotifSpec(
        max_positions=8,
        checkpoint_rule="CP60",
        overflow_rule="O6_late9",
        overflow_trigger=9,
        overflow_slots=4,
        cic1_overflow_size=0.50,
        cic2_overflow_size=0.25,
    ),
    "B4_P2_max8_ProtectA_cap2_O6": MotifSpec(
        max_positions=8,
        checkpoint_rule="Protect_A_cap2",
        protect_cap=2,
        overflow_rule="O6_late9",
        overflow_trigger=9,
        overflow_slots=4,
        cic1_overflow_size=0.50,
        cic2_overflow_size=0.25,
    ),
}


def _spec_key(spec: MotifSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))


def _spec_nodes(spec: MotifSpec) -> str:
    nodes = [
        spec.market_context,
        spec.cluster_context,
        spec.local_state,
        spec.execution,
        spec.exit_type,
        f"max{spec.max_positions}",
        spec.overflow_rule,
        spec.checkpoint_rule,
    ]
    if spec.checkpoint_rule.startswith("Protect_A"):
        nodes.append(f"protect_cap{spec.protect_cap}")
    return " -> ".join([node for node in nodes if node and node != "none"])


def _grammar_table() -> pd.DataFrame:
    rows = []
    for layer, values in GRAMMAR.items():
        for value in values:
            rows.append({"layer": layer, "node": value})
    return pd.DataFrame(rows)


def _prepare_sample(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    root: Path,
    cfg: V20Config,
) -> pd.DataFrame:
    sample = _prepare_sample_at_cost(feature_path, instruments, config, root, cfg.v10a, PROTECTION_COST_BPS)
    if sample.empty:
        raise ValueError("No P2 CIC sample available for v2.0 graph motif search.")
    sample = sample.copy()
    for col in ("signal_time", "entry_time", "exit_time", "checkpoint_time"):
        if col in sample.columns:
            sample[col] = pd.to_datetime(sample[col], utc=True, errors="coerce")
    if "month" not in sample.columns:
        sample["month"] = sample["entry_time"].dt.strftime("%Y-%m")
    return sample.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _period_sample(sample: pd.DataFrame, period: str) -> pd.DataFrame:
    entry = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce")
    if period == "search":
        return sample[entry.lt(SEARCH_END)].copy()
    if period == "validation":
        return sample[entry.ge(SEARCH_END) & entry.lt(VALIDATION_END)].copy()
    if period == "holdout":
        return sample[entry.ge(VALIDATION_END)].copy()
    if period == "full":
        return sample.copy()
    raise KeyError(period)


def _candidate_filter(sample: pd.DataFrame, spec: MotifSpec) -> pd.DataFrame:
    out = sample.copy()
    if spec.market_context == "BTC_up" and "btc_state_at_entry" in out.columns:
        out = out[out["btc_state_at_entry"].astype(str).eq("BTC_up")]
    elif spec.market_context == "BTC_chop" and "btc_state_at_entry" in out.columns:
        out = out[out["btc_state_at_entry"].astype(str).eq("BTC_chop")]
    elif spec.market_context == "market_impulse_density_high":
        pass

    if spec.cluster_context == "cluster_impulse_high":
        cluster = pd.to_numeric(out.get("cluster_density", out.get("cluster_impulse_density")), errors="coerce")
        if cluster.notna().any():
            threshold = float(cluster.quantile(0.8))
            out = out[cluster.ge(threshold)]

    candidate = out["candidate"].astype(str)
    if spec.local_state == "cic1_extreme":
        out = out[candidate.eq("CIC1_beta_extreme")]
    elif spec.local_state == "cic2_broad":
        out = out[candidate.eq("CIC2_beta_broad")]
    return out.sort_values(["entry_time", "symbol", "candidate"]).reset_index(drop=True)


def _no_checkpoint_sample(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    out["checkpoint_rule"] = "none"
    out["checkpoint_early_exit"] = False
    out["kept_due_to_protection"] = False
    out["effective_exit_time"] = out["exit_time"]
    out["effective_net_return"] = pd.to_numeric(out["net_return_at_cost"], errors="coerce")
    return out


def _cp60_would_exit(sample: pd.DataFrame) -> pd.Series:
    checkpoint = pd.to_datetime(sample.get("checkpoint_time"), utc=True, errors="coerce")
    exit_time = pd.to_datetime(sample.get("exit_time"), utc=True, errors="coerce")
    covered = sample.get("checkpoint_price_covered", pd.Series(False, index=sample.index)).fillna(False).astype(bool)
    net = pd.to_numeric(sample.get("checkpoint_net_at_cost"), errors="coerce")
    return covered & checkpoint.lt(exit_time) & net.le(0.0)


def _beta_high_mask(sample: pd.DataFrame) -> pd.Series:
    if "beta_extreme_strength_high" in sample.columns:
        return sample["beta_extreme_strength_high"].fillna(False).astype(bool)
    for col in ("beta_extreme_strength", "c2_beta_extension_score", "beta_extension_score_at_signal"):
        if col in sample.columns:
            return pd.to_numeric(sample[col], errors="coerce").ge(PROTECT_A_BETA_HIGH_THRESHOLD)
    return pd.Series(False, index=sample.index)


def _protect_a_cap_mask(sample: pd.DataFrame, cap: int) -> pd.Series:
    eligible = sample[_cp60_would_exit(sample) & _beta_high_mask(sample)].copy()
    protected = pd.Series(False, index=sample.index)
    if eligible.empty:
        return protected
    eligible["_checkpoint_sort"] = pd.to_datetime(eligible["checkpoint_time"], utc=True, errors="coerce")
    eligible["_entry_sort"] = pd.to_datetime(eligible["entry_time"], utc=True, errors="coerce")
    sort_cols = ["burst_id", "_checkpoint_sort", "_entry_sort", "symbol"] if "burst_id" in eligible.columns else ["_checkpoint_sort", "_entry_sort", "symbol"]
    eligible = eligible.sort_values(sort_cols)
    group_key = "burst_id" if "burst_id" in eligible.columns else pd.Series("unknown", index=eligible.index)
    protected.loc[eligible.groupby(group_key, sort=False, dropna=False).head(int(cap)).index] = True
    return protected


def _apply_management(sample: pd.DataFrame, spec: MotifSpec) -> pd.DataFrame:
    if sample.empty:
        return sample.copy()
    if spec.checkpoint_rule == "none":
        return _no_checkpoint_sample(sample)
    if spec.checkpoint_rule == "CP60":
        return _apply_protected_checkpoint(sample, pd.Series(False, index=sample.index), "CP60_all")
    if spec.checkpoint_rule == "Protect_A_cap1":
        return _apply_protected_checkpoint(sample, _protect_a_cap_mask(sample, 1), "Protect_A_cap1")
    if spec.checkpoint_rule == "Protect_A_cap2":
        return _apply_protected_checkpoint(sample, _protect_a_cap_mask(sample, 2), "Protect_A_cap2")
    raise KeyError(spec.checkpoint_rule)


def _overflow_size(row: pd.Series, spec: MotifSpec) -> float:
    candidate = str(row.get("candidate", ""))
    if candidate == "CIC1_beta_extreme":
        return float(spec.cic1_overflow_size)
    if candidate == "CIC2_beta_broad":
        return float(spec.cic2_overflow_size)
    return 0.0


def _simulate_portfolio(sample: pd.DataFrame, spec: MotifSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = _apply_management(_candidate_filter(sample, spec), spec)
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    active_core: list[dict[str, Any]] = []
    active_overflow: list[dict[str, Any]] = []
    for _, row in pool.sort_values(["entry_time", "symbol", "candidate"]).iterrows():
        entry = pd.Timestamp(row["entry_time"])
        active_core = [item for item in active_core if pd.Timestamp(item["exit_time"]) > entry]
        active_overflow = [item for item in active_overflow if pd.Timestamp(item["exit_time"]) > entry]
        active_symbols = {str(item["symbol"]) for item in [*active_core, *active_overflow]}
        payload = row.to_dict()
        payload["exposure_weight"] = 0.0
        payload["sleeve"] = "skipped"
        payload["selection_status"] = "skipped"
        payload["skip_reason"] = ""
        payload["weighted_return"] = 0.0
        if str(row["symbol"]) in active_symbols:
            payload["skip_reason"] = "symbol_already_active"
            skipped.append(payload)
            continue
        if len(active_core) < int(spec.max_positions):
            payload["exposure_weight"] = 1.0
            payload["sleeve"] = "core"
            payload["selection_status"] = "selected"
            payload["weighted_return"] = float(row["effective_net_return"])
            selected.append(payload)
            active_core.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"]})
            continue
        overflow_size = _overflow_size(row, spec)
        overflow_allowed = (
            spec.overflow_rule != "none"
            and int(row.get("burst_count_so_far", 0)) >= int(spec.overflow_trigger)
            and overflow_size > 0
        )
        if overflow_allowed and len(active_overflow) < int(spec.overflow_slots):
            payload["exposure_weight"] = overflow_size
            payload["sleeve"] = "overflow"
            payload["selection_status"] = "selected"
            payload["weighted_return"] = float(row["effective_net_return"]) * overflow_size
            selected.append(payload)
            active_overflow.append({"symbol": str(row["symbol"]), "exit_time": row["effective_exit_time"]})
            continue
        payload["skip_reason"] = "overflow_full" if overflow_allowed else "portfolio_full_not_overflow_eligible"
        skipped.append(payload)
    return pd.DataFrame(selected), pd.DataFrame(skipped)


def _period_returns(ledger: pd.DataFrame, key_col: str, denominator: int) -> pd.Series:
    if ledger.empty or key_col not in ledger.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(ledger["weighted_return"], errors="coerce").groupby(
        ledger[key_col].astype(str), sort=False, dropna=False
    ).sum() / max(1, denominator)


def _month_cap(contrib: pd.Series, cap: float = 0.35) -> float:
    if contrib.empty:
        return np.nan
    total = float(contrib.sum())
    cap_value = total * cap if total > 0 else 0.0
    values = []
    for value in contrib:
        values.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(values))


def _max_contribution_ratio(contrib: pd.Series) -> float:
    positive = contrib[contrib > 0]
    total = float(positive.sum())
    return float(positive.max() / total) if total > 0 and len(positive) else np.nan


def _drawdown(contrib: pd.Series) -> float:
    if contrib.empty:
        return np.nan
    equity = contrib.cumsum()
    return float((equity - equity.cummax()).min())


def _period_hours(ledger: pd.DataFrame) -> float:
    if ledger.empty:
        return np.nan
    start = pd.to_datetime(ledger["entry_time"], utc=True, errors="coerce").min()
    end = pd.to_datetime(ledger["effective_exit_time"], utc=True, errors="coerce").max()
    if pd.isna(start) or pd.isna(end) or end <= start:
        return np.nan
    return float((end - start).total_seconds() / 3600.0)


def _complexity(spec: MotifSpec) -> int:
    value = 4  # market, local, execution, exit
    value += int(spec.market_context != "market_impulse_density_high")
    value += int(spec.cluster_context != "none")
    value += int(spec.overflow_rule != "none")
    value += int(spec.checkpoint_rule != "none")
    value += int(spec.checkpoint_rule.startswith("Protect_A"))
    value += int(spec.max_positions != 8)
    return value


def _metrics_for_sample(sample: pd.DataFrame, spec: MotifSpec) -> dict[str, Any]:
    ledger, skipped = _simulate_portfolio(sample, spec)
    denominator = int(spec.max_positions)
    if ledger.empty:
        return {
            "trades": 0,
            "skipped_trades": int(len(skipped)),
            "portfolio_net20": 0.0,
            "fitness": -999.0,
            "complexity": _complexity(spec),
        }
    weighted = pd.to_numeric(ledger["weighted_return"], errors="coerce")
    contrib = weighted / max(1, denominator)
    month_returns = _period_returns(ledger, "month", denominator)
    burst_returns = _period_returns(ledger, "burst_id", denominator)
    symbol_returns = _period_returns(ledger, "symbol", denominator)
    top_month = month_returns.max() if len(month_returns) else np.nan
    portfolio_net = float(contrib.sum())
    ex_top_month = float(portfolio_net - top_month) if pd.notna(top_month) else np.nan
    period_hours = _period_hours(ledger)
    holding_hours = (
        (
            pd.to_datetime(ledger["effective_exit_time"], utc=True, errors="coerce")
            - pd.to_datetime(ledger["entry_time"], utc=True, errors="coerce")
        ).dt.total_seconds()
        / 3600.0
        * pd.to_numeric(ledger["exposure_weight"], errors="coerce").fillna(1.0)
    ).sum()
    capital_utilization = float(holding_hours / (period_hours * denominator)) if period_hours and period_hours > 0 else np.nan
    return_per_capital_day = float(portfolio_net / period_hours * 24.0) if period_hours and period_hours > 0 else np.nan
    max_month_contribution = _max_contribution_ratio(month_returns)
    worst_burst = float(burst_returns.min()) if len(burst_returns) else np.nan
    worst_month = float(month_returns.min()) if len(month_returns) else np.nan
    month_cap = _month_cap(month_returns)
    complexity = _complexity(spec)
    fitness = (
        0.30 * portfolio_net
        + 0.25 * (month_cap if pd.notna(month_cap) else 0.0)
        + 0.20 * (ex_top_month if pd.notna(ex_top_month) else 0.0)
        + 0.15 * (return_per_capital_day if pd.notna(return_per_capital_day) else 0.0)
        - 0.25 * max(0.0, -(worst_burst if pd.notna(worst_burst) else 0.0))
        - 0.20 * max(0.0, -(worst_month if pd.notna(worst_month) else 0.0))
        - 0.10 * max(0.0, (max_month_contribution if pd.notna(max_month_contribution) else 0.0) - 0.35)
        - 0.002 * complexity
    )
    return {
        "trades": int(len(ledger)),
        "skipped_trades": int(len(skipped)),
        "core_trades": int(ledger["sleeve"].astype(str).eq("core").sum()),
        "overflow_trades": int(ledger["sleeve"].astype(str).eq("overflow").sum()),
        "early_exit_trades": int(ledger.get("checkpoint_early_exit", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "protected_exits": int(ledger.get("kept_due_to_protection", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "portfolio_net20": portfolio_net,
        "month_cap35_net20": month_cap,
        "ex_top_month_net20": ex_top_month,
        "worst_burst_net20": worst_burst,
        "worst_month_net20": worst_month,
        "max_month_contribution": max_month_contribution,
        "max_symbol_contribution": _max_contribution_ratio(symbol_returns),
        "max_drawdown_proxy": _drawdown(contrib),
        "return_per_capital_day": return_per_capital_day,
        "capital_utilization": capital_utilization,
        "complexity": complexity,
        "fitness": fitness,
    }


def _evaluation_row(spec: MotifSpec, sample: pd.DataFrame, *, candidate_id: str, source: str) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source": source,
        "nodes": _spec_nodes(spec),
        **asdict(spec),
    }
    for period in ("search", "validation", "holdout", "full"):
        metrics = _metrics_for_sample(_period_sample(sample, period), spec)
        for key, value in metrics.items():
            rows[f"{period}_{key}"] = value
    return rows


def _normalize_spec(spec: MotifSpec) -> MotifSpec:
    if spec.overflow_rule == "none":
        return replace(spec, overflow_trigger=999_999, overflow_slots=0, cic1_overflow_size=0.0, cic2_overflow_size=0.0)
    if spec.overflow_rule == "O6_late9":
        return replace(spec, overflow_trigger=9, overflow_slots=4, cic1_overflow_size=0.50, cic2_overflow_size=0.25)
    if spec.overflow_rule == "O6_late15":
        return replace(spec, overflow_trigger=15, overflow_slots=2, cic1_overflow_size=0.50, cic2_overflow_size=0.25)
    return spec


def _random_spec(rng: random.Random) -> MotifSpec:
    checkpoint = rng.choice(GRAMMAR["checkpoint_rule"])
    cap = 0
    if checkpoint == "Protect_A_cap1":
        cap = 1
    elif checkpoint == "Protect_A_cap2":
        cap = 2
    return _normalize_spec(
        MotifSpec(
            market_context=rng.choice(GRAMMAR["market_context"]),
            cluster_context=rng.choice(GRAMMAR["cluster_context"]),
            local_state=rng.choice(GRAMMAR["local_state"]),
            max_positions=int(rng.choice(GRAMMAR["max_positions"])),
            overflow_rule=rng.choice(GRAMMAR["overflow_rule"]),
            checkpoint_rule=checkpoint,
            protect_cap=cap,
        )
    )


def _aco_path_miner(sample: pd.DataFrame, cfg: V20Config) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    pheromone = {layer: {str(node): 1.0 for node in values} for layer, values in GRAMMAR.items()}
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    def choose(layer: str) -> Any:
        values = GRAMMAR[layer]
        weights = [pheromone[layer][str(value)] for value in values]
        return rng.choices(values, weights=weights, k=1)[0]

    for iteration in range(cfg.aco_iterations):
        batch: list[tuple[MotifSpec, dict[str, Any]]] = []
        for ant in range(cfg.aco_ants):
            checkpoint = choose("checkpoint_rule")
            cap = 1 if checkpoint == "Protect_A_cap1" else 2 if checkpoint == "Protect_A_cap2" else 0
            spec = _normalize_spec(
                MotifSpec(
                    market_context=choose("market_context"),
                    cluster_context=choose("cluster_context"),
                    local_state=choose("local_state"),
                    max_positions=int(choose("max_positions")),
                    overflow_rule=choose("overflow_rule"),
                    checkpoint_rule=checkpoint,
                    protect_cap=cap,
                )
            )
            key = _spec_key(spec)
            metrics = cache.get(key)
            if metrics is None:
                metrics = _metrics_for_sample(_period_sample(sample, "search"), spec)
                cache[key] = metrics
            batch.append((spec, metrics))
            rows.append(
                {
                    "iteration": iteration,
                    "ant": ant,
                    "path_id": key,
                    "nodes": _spec_nodes(spec),
                    **asdict(spec),
                    **metrics,
                }
            )
        for layer in pheromone:
            for node in pheromone[layer]:
                pheromone[layer][node] *= 0.85
                pheromone[layer][node] = max(0.05, pheromone[layer][node])
        for spec, metrics in sorted(batch, key=lambda item: item[1].get("fitness", -999), reverse=True)[: max(2, cfg.aco_ants // 5)]:
            reward = max(0.01, float(metrics.get("fitness", 0.0)) + 0.05)
            for layer, value in asdict(spec).items():
                if layer in pheromone and str(value) in pheromone[layer]:
                    pheromone[layer][str(value)] += reward
    return pd.DataFrame(rows).sort_values("fitness", ascending=False).drop_duplicates("path_id").reset_index(drop=True)


def _crossover(left: MotifSpec, right: MotifSpec, rng: random.Random) -> MotifSpec:
    data = {}
    ldict = asdict(left)
    rdict = asdict(right)
    for key in ldict:
        data[key] = ldict[key] if rng.random() < 0.5 else rdict[key]
    return _normalize_spec(MotifSpec(**data))


def _mutate(spec: MotifSpec, rng: random.Random, rate: float = 0.16) -> MotifSpec:
    data = asdict(spec)
    for key in ("market_context", "cluster_context", "local_state", "max_positions", "overflow_rule", "checkpoint_rule"):
        if rng.random() < rate:
            data[key] = rng.choice(GRAMMAR[key])
    checkpoint = data["checkpoint_rule"]
    data["protect_cap"] = 1 if checkpoint == "Protect_A_cap1" else 2 if checkpoint == "Protect_A_cap2" else 0
    return _normalize_spec(MotifSpec(**data))


def _ga_architecture_search(sample: pd.DataFrame, cfg: V20Config, seeds: list[MotifSpec]) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 101)
    population = list(seeds)
    while len(population) < cfg.ga_population:
        population.append(_random_spec(rng))
    cache: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    def fitness(spec: MotifSpec) -> float:
        key = _spec_key(spec)
        if key not in cache:
            cache[key] = _metrics_for_sample(_period_sample(sample, "search"), spec)
        return float(cache[key].get("fitness", -999.0))

    for generation in range(cfg.ga_generations):
        scored = sorted([(spec, fitness(spec)) for spec in population], key=lambda item: item[1], reverse=True)
        for rank, (spec, score) in enumerate(scored):
            all_rows.append({"generation": generation, "rank": rank, "fitness": score, "path_id": _spec_key(spec), "nodes": _spec_nodes(spec), **asdict(spec)})
        elites = [spec for spec, _ in scored[: max(4, cfg.ga_population // 8)]]
        next_pop = elites.copy()
        while len(next_pop) < cfg.ga_population:
            parent_pool = rng.sample(scored[: max(12, cfg.ga_population // 2)], k=4)
            p1 = max(parent_pool[:2], key=lambda item: item[1])[0]
            p2 = max(parent_pool[2:], key=lambda item: item[1])[0]
            next_pop.append(_mutate(_crossover(p1, p2, rng), rng))
        population = next_pop
    frame = pd.DataFrame(all_rows)
    return frame.sort_values("fitness", ascending=False).drop_duplicates("path_id").reset_index(drop=True)


def _sa_neighbors(spec: MotifSpec) -> list[MotifSpec]:
    neighbors: list[MotifSpec] = []
    for max_pos in sorted(set([spec.max_positions - 1, spec.max_positions + 1, 5, 8, 10, 12])):
        if 3 <= max_pos <= 15 and max_pos != spec.max_positions:
            neighbors.append(replace(spec, max_positions=max_pos))
    if spec.overflow_rule != "none":
        for trigger in (8, 9, 10, 11, 15):
            if trigger != spec.overflow_trigger:
                neighbors.append(replace(spec, overflow_trigger=trigger))
        for slots in (2, 4, 6):
            if slots != spec.overflow_slots:
                neighbors.append(replace(spec, overflow_slots=slots))
        for c1, c2 in ((0.25, 0.125), (0.50, 0.25), (0.75, 0.375)):
            if (c1, c2) != (spec.cic1_overflow_size, spec.cic2_overflow_size):
                neighbors.append(replace(spec, cic1_overflow_size=c1, cic2_overflow_size=c2))
    if spec.checkpoint_rule.startswith("Protect_A"):
        alt = "Protect_A_cap1" if spec.checkpoint_rule == "Protect_A_cap2" else "Protect_A_cap2"
        cap = 1 if alt == "Protect_A_cap1" else 2
        neighbors.append(replace(spec, checkpoint_rule=alt, protect_cap=cap))
    elif spec.checkpoint_rule == "CP60":
        neighbors.append(replace(spec, checkpoint_rule="Protect_A_cap2", protect_cap=2))
    return [_normalize_spec(item) for item in neighbors]


def _sa_refinement(sample: pd.DataFrame, cfg: V20Config, starts: list[MotifSpec]) -> pd.DataFrame:
    rng = random.Random(cfg.seed + 202)
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}

    def score(spec: MotifSpec) -> float:
        key = _spec_key(spec)
        if key not in cache:
            cache[key] = _metrics_for_sample(_period_sample(sample, "search"), spec)
        return float(cache[key].get("fitness", -999.0))

    for start_idx, start in enumerate(starts):
        current = start
        current_score = score(current)
        best = current
        best_score = current_score
        visited: dict[str, MotifSpec] = {_spec_key(current): current}
        for step in range(cfg.sa_steps):
            neighbors = _sa_neighbors(current)
            if not neighbors:
                break
            proposal = rng.choice(neighbors)
            proposal_score = score(proposal)
            temp = max(0.002, 0.06 * (1.0 - step / max(1, cfg.sa_steps)))
            accept = proposal_score >= current_score or rng.random() < math.exp((proposal_score - current_score) / temp)
            if accept:
                current = proposal
                current_score = proposal_score
                visited[_spec_key(current)] = current
                if current_score > best_score:
                    best = current
                    best_score = current_score
        neighbor_scores = [score(n) for n in _sa_neighbors(best)]
        neighbor_avg = float(np.mean(neighbor_scores)) if neighbor_scores else np.nan
        rows.append(
            {
                "start_index": start_idx,
                "best_path_id": _spec_key(best),
                "best_nodes": _spec_nodes(best),
                "best_score": best_score,
                "neighbor_avg_score": neighbor_avg,
                "stability_score": neighbor_avg / best_score if best_score and pd.notna(neighbor_avg) else np.nan,
                "visited_specs": len(visited),
                **asdict(best),
            }
        )
    return pd.DataFrame(rows).sort_values("best_score", ascending=False).reset_index(drop=True)


def _spec_from_row(row: pd.Series) -> MotifSpec:
    data = {field: row[field] for field in asdict(MotifSpec()).keys() if field in row.index}
    data["max_positions"] = int(data.get("max_positions", 8))
    data["overflow_trigger"] = int(data.get("overflow_trigger", 999_999))
    data["overflow_slots"] = int(data.get("overflow_slots", 0))
    data["protect_cap"] = int(data.get("protect_cap", 0))
    data["cic1_overflow_size"] = float(data.get("cic1_overflow_size", 0.0))
    data["cic2_overflow_size"] = float(data.get("cic2_overflow_size", 0.0))
    return _normalize_spec(MotifSpec(**data))


def _write_notes(
    root: Path,
    benchmarks: pd.DataFrame,
    aco: pd.DataFrame,
    ga: pd.DataFrame,
    sa: pd.DataFrame,
    stable: pd.DataFrame,
) -> None:
    lines = [
        "# v2.0 Graph Motif Search",
        "",
        "Status: offline search framework only. No paper-live or real-live rule changes.",
        "",
        "## Search Discipline",
        f"- search_period: entry_time < {SEARCH_END.isoformat()}",
        f"- validation_period: {SEARCH_END.isoformat()} <= entry_time < {VALIDATION_END.isoformat()}",
        f"- holdout_period: entry_time >= {VALIDATION_END.isoformat()}",
        "- ACO/GA optimize search fitness only; validation and holdout are reported separately.",
        "- Grammar is restricted to already-audited building blocks.",
        "",
        "## Benchmarks",
    ]
    for row in benchmarks.itertuples(index=False):
        lines.append(
            f"- {row.candidate_id}: search_net={row.search_portfolio_net20:.4%}, "
            f"validation_net={row.validation_portfolio_net20:.4%}, holdout_net={row.holdout_portfolio_net20:.4%}."
        )
    if not aco.empty:
        top = aco.iloc[0]
        lines.extend(["", "## ACO Top Path", f"- {top.nodes}: fitness={top.fitness:.6f}, net20={top.portfolio_net20:.4%}."])
    if not ga.empty:
        top = ga.iloc[0]
        lines.extend(["", "## GA Top Architecture", f"- {top.nodes}: search_fitness={top.fitness:.6f}."])
    if not sa.empty:
        top = sa.iloc[0]
        lines.extend(
            [
                "",
                "## SA Best Stable Candidate",
                f"- {top.best_nodes}: best_score={top.best_score:.6f}, neighbor_avg={top.neighbor_avg_score:.6f}, "
                f"stability={top.stability_score:.4f}.",
            ]
        )
    benchmark_b4 = benchmarks.loc[benchmarks["candidate_id"] == "B4_P2_max8_ProtectA_cap2_O6"].iloc[0]
    searched = stable.loc[stable["source"] != "benchmark"].copy()
    if not searched.empty:
        validation_best = searched.sort_values(
            ["validation_portfolio_net20", "full_complexity", "search_fitness"],
            ascending=[False, True, False],
        ).iloc[0]
        search_best = searched.sort_values("search_fitness", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Promotion Decision",
                (
                    f"- Current benchmark B4: validation_net={benchmark_b4.validation_portfolio_net20:.4%}, "
                    f"holdout_net={benchmark_b4.holdout_portfolio_net20:.4%}, "
                    f"full_net={benchmark_b4.full_portfolio_net20:.4%}."
                ),
                (
                    f"- Best validation searched candidate: {validation_best.nodes}, "
                    f"validation_net={validation_best.validation_portfolio_net20:.4%}, "
                    f"holdout_net={validation_best.holdout_portfolio_net20:.4%}, "
                    f"full_net={validation_best.full_portfolio_net20:.4%}."
                ),
                (
                    f"- Best search-fitness candidate: {search_best.nodes}, "
                    f"validation_net={search_best.validation_portfolio_net20:.4%}, "
                    f"holdout_net={search_best.holdout_portfolio_net20:.4%}, "
                    f"full_net={search_best.full_portfolio_net20:.4%}."
                ),
                (
                    "- Decision: no v2.0 candidate is promoted. The best searched candidates improve search/full "
                    "or validation metrics, but they do not beat B4 on holdout/worst-window behavior."
                ),
            ]
        )
    lines.extend(["", "Next: keep v2.0 as an offline motif-search framework; future shadow candidates must beat B4 on validation and holdout."])
    (root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v20_graph_motif_search(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    cfg: V20Config = V20Config(),
) -> dict[str, Path]:
    root = ensure_dir(cfg.report_root)
    sample = _prepare_sample(feature_path, instruments, config, root, cfg)
    benchmarks = pd.DataFrame(
        [_evaluation_row(spec, sample, candidate_id=name, source="benchmark") for name, spec in BENCHMARKS.items()]
    )
    aco = _aco_path_miner(sample, cfg)
    aco_specs = [_spec_from_row(row) for _, row in aco.head(12).iterrows()]
    ga_seed_specs = list(BENCHMARKS.values()) + aco_specs
    ga = _ga_architecture_search(sample, cfg, ga_seed_specs)
    ga_top_specs = [_spec_from_row(row) for _, row in ga.head(5).iterrows()]
    sa = _sa_refinement(sample, cfg, ga_top_specs)

    top_specs: list[tuple[str, str, MotifSpec]] = []
    for name, spec in BENCHMARKS.items():
        top_specs.append((name, "benchmark", spec))
    for idx, row in aco.head(10).iterrows():
        top_specs.append((f"ACO_{idx:03d}", "aco", _spec_from_row(row)))
    for idx, row in ga.head(10).iterrows():
        top_specs.append((f"GA_{idx:03d}", "ga", _spec_from_row(row)))
    for idx, row in sa.head(5).iterrows():
        top_specs.append((f"SA_{idx:03d}", "sa", _spec_from_row(row)))
    seen: set[str] = set()
    eval_rows = []
    for candidate_id, source, spec in top_specs:
        key = _spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        eval_rows.append(_evaluation_row(spec, sample, candidate_id=candidate_id, source=source))
    stable = pd.DataFrame(eval_rows).sort_values("search_fitness", ascending=False).reset_index(drop=True)

    outputs = {
        "motif_dsl": root / "motif_dsl.csv",
        "benchmarks": root / "benchmarks.csv",
        "top_paths": root / "top_paths.csv",
        "candidate_architectures": root / "candidate_architectures.csv",
        "stable_candidates": root / "stable_candidates.csv",
        "sa_refinement": root / "sa_refinement.csv",
        "search_config": root / "search_config.json",
        "candidate_notes": root / "candidate_notes.md",
    }
    _grammar_table().to_csv(outputs["motif_dsl"], index=False)
    benchmarks.to_csv(outputs["benchmarks"], index=False)
    aco.to_csv(outputs["top_paths"], index=False)
    ga.to_csv(outputs["candidate_architectures"], index=False)
    stable.to_csv(outputs["stable_candidates"], index=False)
    sa.to_csv(outputs["sa_refinement"], index=False)
    outputs["search_config"].write_text(
        json.dumps(
            {
                "seed": cfg.seed,
                "aco_iterations": cfg.aco_iterations,
                "aco_ants": cfg.aco_ants,
                "ga_population": cfg.ga_population,
                "ga_generations": cfg.ga_generations,
                "sa_steps": cfg.sa_steps,
                "search_end": SEARCH_END.isoformat(),
                "validation_end": VALIDATION_END.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_notes(root, benchmarks, aco, ga, sa, stable)
    return outputs


__all__ = [
    "GRAMMAR",
    "MotifSpec",
    "REPORT_ROOT",
    "V20Config",
    "write_v20_graph_motif_search",
]
