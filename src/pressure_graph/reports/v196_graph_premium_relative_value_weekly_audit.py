"""Independent audit for v19.6 weekly graph-premium relative value."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v190_binance_premium_index_audit import (
    load_v190_premium_panel,
)
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    FUNDING_ROOTS,
    REPORT_ROOT as V195_REPORT_ROOT,
    load_v195_funding,
)
from pressure_graph.reports.v196_graph_premium_relative_value_weekly import (
    CANDIDATES,
    GLOBAL_CANDIDATE,
    REPORT_ROOT,
    V196Config,
)


AUDIT_ROOT = Path("reports/v19_6_graph_premium_relative_value_weekly_audit")
FINDINGS_PATH = Path(
    "docs/v196_graph_premium_relative_value_weekly_audit_2026_07_17.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def _parse_map(value: str) -> dict[str, float]:
    if not value:
        return {}
    return {
        item.rsplit(":", 1)[0]: float(item.rsplit(":", 1)[1])
        for item in str(value).split("|")
    }


def _max_abs(values: list[float] | np.ndarray | pd.Series) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmax(np.abs(array))) if array.size else 0.0


def _turnover(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        sum(
            abs(left.get(symbol, 0.0) - right.get(symbol, 0.0))
            for symbol in set(left) | set(right)
        )
    )


def _funding_sum(
    funding: pd.DataFrame,
    symbol: str,
    entry: pd.Timestamp,
    exit_time: pd.Timestamp,
) -> float:
    local = funding[funding["symbol"].eq(symbol)]
    return float(
        local.loc[
            local["funding_time"].gt(entry)
            & local["funding_time"].le(exit_time),
            "funding_rate_settled",
        ].sum()
    )


def audit_v196_independently(
    report_root: Path = REPORT_ROOT,
    cfg: V196Config = V196Config(),
) -> pd.DataFrame:
    portfolio = pd.read_parquet(report_root / "weekly_portfolio.parquet")
    delayed = pd.read_parquet(report_root / "delayed_weekly_portfolio.parquet")
    controls = pd.read_parquet(report_root / "control_weekly_portfolios.parquet")
    random = pd.read_parquet(report_root / "random_controls.parquet")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    features = pd.read_parquet(
        V195_REPORT_ROOT / "daily_symbol_feature_panel.parquet"
    )
    close, _ = load_v184_exact_panels()
    premium = load_v190_premium_panel().reindex(
        index=close.index, columns=close.columns
    )
    funding = load_v195_funding(set(close.columns) | {BTC}, FUNDING_ROOTS)
    fss3 = pd.read_parquet(cfg.fss3_path)

    for frame in (portfolio, delayed, controls):
        for column in ("entry_time", "exit_time", "target_feature_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frame["weights_map"] = frame["weights"].map(_parse_map)
        frame["contribution_map"] = frame["symbol_contributions"].map(_parse_map)
    features["entry_time"] = pd.to_datetime(features["entry_time"], utc=True)
    funding["funding_time"] = pd.to_datetime(funding["funding_time"], utc=True)
    fss3["entry_time"] = pd.to_datetime(fss3["entry_time"], utc=True)

    rows: list[dict[str, object]] = []
    _check(rows, "candidate_domain_exact", set(portfolio["candidate"]) == set(CANDIDATES),
        "|".join(sorted(portfolio["candidate"].unique())))
    _check(rows, "weekly_keys_unique", not portfolio.duplicated(
        ["candidate", "entry_time"]
    ).any(), len(portfolio))
    _check(rows, "all_entries_monday_0000", portfolio["entry_time"].dt.weekday.eq(0).all()
        and portfolio["entry_time"].dt.hour.eq(0).all()
        and portfolio["entry_time"].dt.minute.eq(0).all(), len(portfolio))
    holding = portfolio["exit_time"] - portfolio["entry_time"]
    _check(rows, "holding_exact_seven_days", holding.eq(pd.Timedelta(days=7)).all(),
        int(holding.ne(pd.Timedelta(days=7)).sum()))
    _check(rows, "primary_target_is_current_completed_score", portfolio[
        "target_feature_time"
    ].eq(portfolio["entry_time"]).all(), int(portfolio["target_feature_time"].ne(
        portfolio["entry_time"]
    ).sum()))

    prior_mean = premium.shift(1).rolling(
        cfg.lookback_bars, min_periods=cfg.minimum_bars
    ).mean()
    prior_scale = premium.shift(1).rolling(
        cfg.lookback_bars, min_periods=cfg.minimum_bars
    ).std(ddof=1)
    premium_z = (premium - prior_mean).div(prior_scale.where(prior_scale.gt(0)))
    weekly_features = features[features["entry_time"].isin(portfolio["entry_time"])]
    expected_z = np.asarray(
        [premium_z.at[row.entry_time, row.symbol] for row in weekly_features.itertuples()],
        dtype=float,
    )
    z_error = _max_abs(weekly_features["premium_z"].to_numpy(dtype=float) - expected_z)
    _check(rows, "premium_z_uses_shifted_prior_history", z_error <= 1e-12, z_error)
    peer_expected = weekly_features["premium_z"] - weekly_features.groupby(
        ["entry_time", "community_id"]
    )["premium_z"].transform("median")
    peer_error = _max_abs(weekly_features["peer_premium_z"] - peer_expected)
    _check(rows, "graph_peer_residual_exact", peer_error <= 1e-12, peer_error)

    selection_errors = 0
    beta_errors = []
    gross_errors = []
    for event in portfolio.itertuples(index=False):
        local = features[features["entry_time"].eq(event.entry_time)].copy()
        weights = dict(event.weights_map)
        alt = {symbol: weight for symbol, weight in weights.items() if symbol != BTC}
        long_names = sorted(symbol for symbol, weight in alt.items() if weight > 0)
        short_names = sorted(symbol for symbol, weight in alt.items() if weight < 0)
        if event.candidate == GLOBAL_CANDIDATE:
            ranked = local.sort_values(["peer_premium_z", "symbol"])
            expected_long = sorted(ranked.head(cfg.global_bucket_size)["symbol"].astype(str))
            expected_short = sorted(ranked.tail(cfg.global_bucket_size)["symbol"].astype(str))
            selection_errors += int(long_names != expected_long or short_names != expected_short)
        else:
            expected_long = []
            expected_short = []
            for _, group in local.groupby("community_id", sort=True):
                ranked = group.sort_values(["peer_premium_z", "symbol"])
                if len(ranked) < cfg.minimum_community_size:
                    continue
                expected_long.append(str(ranked.iloc[0]["symbol"]))
                expected_short.append(str(ranked.iloc[-1]["symbol"]))
            selection_errors += int(
                long_names != sorted(expected_long) or short_names != sorted(expected_short)
            )
        beta = local.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
        residual_beta = sum(
            weight * (1.0 if symbol == BTC else float(beta[symbol]))
            for symbol, weight in weights.items()
        )
        beta_errors.extend(
            [residual_beta, sum(abs(value) for value in weights.values()) - 1.0]
        )
        price = close.loc[event.exit_time, list(weights)] / close.loc[
            event.entry_time, list(weights)
        ] - 1.0
        funding_values = {
            symbol: _funding_sum(funding, symbol, event.entry_time, event.exit_time)
            for symbol in weights
        }
        price_value = sum(weights[symbol] * float(price[symbol]) for symbol in weights)
        funding_value = sum(
            -weights[symbol] * funding_values[symbol] for symbol in weights
        )
        gross = price_value + funding_value
        gross_errors.extend(
            [
                price_value - float(event.price_return),
                funding_value - float(event.funding_return),
                gross - float(event.gross_return),
            ]
        )
    _check(rows, "frozen_global_and_community_selections_exact", selection_errors == 0,
        selection_errors)
    beta_error = _max_abs(beta_errors)
    _check(
        rows,
        "beta_neutral_and_gross_one_exact",
        beta_error <= 2e-12,
        beta_error,
    )
    gross_error = _max_abs(gross_errors)
    _check(rows, "price_and_entry_exclusive_funding_pnl_exact", gross_error <= 1e-12,
        gross_error)

    turnover_errors = []
    formula_errors = []
    for candidate in CANDIDATES:
        sample = portfolio[portfolio["candidate"].eq(candidate)].sort_values("entry_time")
        previous: dict[str, float] = {}
        expected_turnovers = []
        for event in sample.itertuples(index=False):
            current = dict(event.weights_map)
            expected_turnovers.append(_turnover(previous, current))
            previous = current
        expected_turnovers[-1] += sum(abs(value) for value in previous.values())
        turnover_errors.extend(
            sample["realized_turnover"].to_numpy(dtype=float)
            - np.asarray(expected_turnovers)
        )
        formula_errors.extend(
            (
                sample["primary_net_return"]
                - (sample["gross_return"] - cfg.one_way_cost * sample["realized_turnover"])
            ).tolist()
        )
        formula_errors.extend(
            (
                sample["stress_net_return"]
                - (
                    sample["gross_return"]
                    - cfg.stress_one_way_cost * sample["realized_turnover"]
                )
            ).tolist()
        )
        formula_errors.extend(
            (
                sample["reversed_primary_net_return"]
                - (-sample["gross_return"] - cfg.one_way_cost * sample["realized_turnover"])
            ).tolist()
        )
    turnover_error = _max_abs(turnover_errors)
    _check(
        rows,
        "initial_transition_and_terminal_turnover_exact",
        turnover_error <= 2e-12,
        turnover_error,
    )
    formula_error = _max_abs(formula_errors)
    _check(rows, "primary_stress_reverse_cost_formulas_exact", formula_error <= 1e-12,
        formula_error)

    delayed_offset = delayed["entry_time"] - delayed["target_feature_time"]
    _check(rows, "delay_control_exact_one_week", delayed_offset.eq(
        pd.Timedelta(days=7)
    ).all(), int(delayed_offset.ne(pd.Timedelta(days=7)).sum()))
    _check(rows, "delay_control_drops_only_first_week", all(
        len(delayed[delayed["candidate"].eq(candidate)])
        == len(portfolio[portfolio["candidate"].eq(candidate)]) - 1
        for candidate in CANDIDATES
    ), len(delayed))
    global_delay = delayed[delayed["candidate"].eq(GLOBAL_CANDIDATE)]
    delay_periods = global_delay.groupby("period")["primary_net_return"].mean()
    _check(rows, "unexpected_delay_control_not_stable", delay_periods.get(
        "validation", 0.0
    ) < 0, float(delay_periods.get("validation", np.nan) * 10_000))

    _check(rows, "random_iterations_complete", random["iteration"].nunique()
        == cfg.null_iterations, int(random["iteration"].nunique()))
    random_wide = random.pivot(
        index="iteration", columns="candidate", values="mean_primary_net_return"
    )
    family_error = _max_abs(
        random_wide["FAMILY_MAX"] - random_wide[list(CANDIDATES)].max(axis=1)
    )
    _check(rows, "random_family_max_exact", family_error <= 1e-12, family_error)
    percentile_errors = []
    family = random_wide["FAMILY_MAX"]
    for row in outcome.itertuples(index=False):
        mean = portfolio.loc[portfolio["candidate"].eq(row.candidate),
            "primary_net_return"].mean()
        percentile_errors.append(
            abs(float(family.le(mean).mean()) - float(row.random_family_percentile))
        )
    percentile_error = max(percentile_errors)
    _check(rows, "outcome_random_percentiles_exact", percentile_error <= 1e-12,
        percentile_error)

    summary_errors = []
    correlation_errors = []
    for row in outcome.itertuples(index=False):
        sample = portfolio[portfolio["candidate"].eq(row.candidate)]
        summary_errors.extend(
            [
                len(sample) - int(row.weeks),
                sample["gross_return"].mean() * 10_000 - float(row.mean_gross_bp),
                sample["primary_net_return"].mean() * 10_000
                - float(row.mean_primary_net_bp),
            ]
        )
        comparison = sample[["entry_time", "primary_net_return"]].merge(
            fss3[["entry_time", "primary_net_return"]].rename(
                columns={"primary_net_return": "fss3"}
            ),
            on="entry_time",
        )
        correlation = comparison[["primary_net_return", "fss3"]].corr().iloc[0, 1]
        correlation_errors.append(correlation - float(row.fss3_primary_return_correlation))
    summary_error = _max_abs(summary_errors)
    _check(rows, "outcome_core_summaries_exact", summary_error <= 1e-10,
        summary_error)
    correlation_error = _max_abs(correlation_errors)
    _check(rows, "fss3_correlations_exact", correlation_error <= 1e-12,
        correlation_error)
    _check(rows, "fss3_independence_gate_passes", outcome[
        "fss3_primary_return_correlation"
    ].abs().le(0.60).all(), float(outcome["fss3_primary_return_correlation"].abs().max()))
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(rows, "rejection_verdict_exact", outcome["verdict"].eq(
        "reject_graph_premium_relative_value_weekly"
    ).all(), "|".join(outcome["verdict"].astype(str)))
    _check(rows, "both_candidates_net_negative", outcome[
        "mean_primary_net_bp"
    ].lt(0).all(), float(outcome["mean_primary_net_bp"].max()))
    _check(rows, "random_family_percentiles_below_half", outcome[
        "random_family_percentile"
    ].lt(0.50).all(), float(outcome["random_family_percentile"].max()))
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v196" in path.name.lower() or "graph_premium" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))

    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_graph_premium_weekly_rejected_short_sleeve_observation_only",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v19.6 Graph-Premium Weekly Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "Both frozen portfolios are rejected. The rich-premium short sleeve is an",
        "observed attribution only and receives no promotion from this audit.",
        "No live, PaperLive, leverage, application, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v196_independent_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v196_independently()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v196_independently", "write_v196_independent_audit"]
