"""Independent audit for v19.8 one-sided rich-premium short portfolios."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    FUNDING_ROOTS,
    REPORT_ROOT as V195_REPORT_ROOT,
    load_v195_funding,
)
from pressure_graph.reports.v198_rich_graph_premium_short_weekly import (
    CANDIDATES,
    GLOBAL_CANDIDATE,
    REPORT_ROOT,
    V198Config,
    _parse_weights,
)


AUDIT_ROOT = Path("reports/v19_8_rich_graph_premium_short_weekly_audit")
FINDINGS_PATH = Path(
    "docs/v198_rich_graph_premium_short_weekly_audit_2026_07_17.md"
)


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


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


def audit_v198_independently(
    report_root: Path = REPORT_ROOT,
    cfg: V198Config = V198Config(),
) -> pd.DataFrame:
    portfolio = pd.read_parquet(report_root / "weekly_short_portfolio.parquet")
    delayed = pd.read_parquet(report_root / "delayed_short_portfolio.parquet")
    cheap = pd.read_parquet(report_root / "cheap_long_diagnostic.parquet")
    orthogonal = pd.read_parquet(
        report_root / "funding_orthogonal_short_diagnostic.parquet"
    )
    random = pd.read_parquet(report_root / "four_direction_random_controls.parquet")
    gates = pd.read_csv(report_root / "candidate_gates.csv")
    outcome = pd.read_csv(report_root / "candidate_outcome.csv")
    features = pd.read_parquet(
        V195_REPORT_ROOT / "daily_symbol_feature_panel.parquet"
    )
    close, _ = load_v184_exact_panels()
    funding = load_v195_funding(set(close.columns) | {BTC}, FUNDING_ROOTS)
    for frame in (portfolio, delayed, cheap, orthogonal):
        for column in ("entry_time", "exit_time", "target_feature_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frame["weights_map"] = frame["weights"].map(_parse_weights)
    features["entry_time"] = pd.to_datetime(features["entry_time"], utc=True)
    funding["funding_time"] = pd.to_datetime(funding["funding_time"], utc=True)

    rows: list[dict[str, object]] = []
    _check(rows, "candidate_domain_exact", set(portfolio["candidate"]) == set(CANDIDATES),
        "|".join(sorted(portfolio["candidate"].unique())))
    _check(rows, "weekly_keys_unique", not portfolio.duplicated(
        ["candidate", "entry_time"]
    ).any(), len(portfolio))
    _check(rows, "all_entries_monday_0000", portfolio["entry_time"].dt.weekday.eq(0).all()
        and portfolio["entry_time"].dt.hour.eq(0).all(), len(portfolio))
    holding = portfolio["exit_time"] - portfolio["entry_time"]
    _check(rows, "holding_exact_seven_days", holding.eq(pd.Timedelta(days=7)).all(),
        int(holding.ne(pd.Timedelta(days=7)).sum()))
    _check(rows, "current_completed_score_used", portfolio["target_feature_time"].eq(
        portfolio["entry_time"]
    ).all(), int(portfolio["target_feature_time"].ne(portfolio["entry_time"]).sum()))

    selection_errors = 0
    sign_errors = 0
    beta_errors = []
    pnl_errors = []
    for event in portfolio.itertuples(index=False):
        local = features[features["entry_time"].eq(event.entry_time)].copy()
        weights = dict(event.weights_map)
        alt = {symbol: weight for symbol, weight in weights.items() if symbol != BTC}
        sign_errors += int(not all(weight < 0 for weight in alt.values()))
        if event.candidate == GLOBAL_CANDIDATE:
            expected = set(
                local.sort_values(["peer_premium_z", "symbol"])
                .tail(cfg.global_bucket_size)["symbol"]
                .astype(str)
            )
        else:
            expected = set()
            for _, group in local.groupby("community_id", sort=True):
                ranked = group.sort_values(["peer_premium_z", "symbol"])
                if len(ranked) >= cfg.minimum_community_size:
                    expected.add(str(ranked.iloc[-1]["symbol"]))
        selection_errors += int(set(alt) != expected)
        beta = local.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
        residual = weights[BTC] + sum(
            weight * float(beta[symbol]) for symbol, weight in alt.items()
        )
        beta_errors.extend(
            [residual, sum(abs(value) for value in weights.values()) - 1.0]
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
        short_gross = sum(
            weights[symbol] * float(price[symbol])
            - weights[symbol] * funding_values[symbol]
            for symbol in alt
        )
        btc_gross = (
            weights[BTC] * float(price[BTC])
            - weights[BTC] * funding_values[BTC]
        )
        pnl_errors.extend(
            [
                price_value - float(event.price_return),
                funding_value - float(event.funding_return),
                gross - float(event.gross_return),
                short_gross - float(event.short_gross_return),
                btc_gross - float(event.btc_gross_return),
            ]
        )
    _check(rows, "richest_global_and_community_selections_exact", selection_errors == 0,
        selection_errors)
    _check(rows, "all_alt_positions_short_and_btc_hedge_long", sign_errors == 0
        and portfolio["weights_map"].map(lambda value: value[BTC] > 0).all(), sign_errors)
    beta_error = _max_abs(beta_errors)
    _check(rows, "serialized_beta_neutral_and_gross_one", beta_error <= 2e-12,
        beta_error)
    pnl_error = _max_abs(pnl_errors)
    _check(rows, "price_funding_short_btc_attribution_exact", pnl_error <= 1e-12,
        pnl_error)

    turnover_errors = []
    formula_errors = []
    for candidate in CANDIDATES:
        sample = portfolio[portfolio["candidate"].eq(candidate)].sort_values("entry_time")
        previous: dict[str, float] = {}
        expected_turnover = []
        for event in sample.itertuples(index=False):
            current = dict(event.weights_map)
            expected_turnover.append(_turnover(previous, current))
            previous = current
        expected_turnover[-1] += sum(abs(value) for value in previous.values())
        turnover_errors.extend(
            sample["realized_turnover"].to_numpy(dtype=float)
            - np.asarray(expected_turnover)
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
    _check(rows, "initial_transition_terminal_turnover_exact", turnover_error <= 2e-12,
        turnover_error)
    formula_error = _max_abs(formula_errors)
    _check(rows, "primary_stress_reverse_cost_formulas_exact", formula_error <= 1e-12,
        formula_error)

    delay_offset = delayed["entry_time"] - delayed["target_feature_time"]
    _check(rows, "delay_control_exact_one_week", delay_offset.eq(
        pd.Timedelta(days=7)
    ).all(), int(delay_offset.ne(pd.Timedelta(days=7)).sum()))
    _check(rows, "cheap_long_and_orthogonal_controls_complete", cheap.groupby(
        "candidate"
    ).size().eq(43).all() and orthogonal.groupby("candidate").size().eq(43).all(),
        len(cheap) + len(orthogonal))
    _check(rows, "random_iterations_complete", random["iteration"].nunique()
        == cfg.null_iterations, int(random["iteration"].nunique()))
    variants = [
        "GLOBAL_RICH_PREMIUM_SHORT",
        "GLOBAL_CHEAP_PREMIUM_LONG",
        "COMMUNITY_RICH_PREMIUM_SHORT",
        "COMMUNITY_CHEAP_PREMIUM_LONG",
    ]
    random_wide = random.pivot(
        index="iteration", columns="variant", values="mean_primary_net_return"
    )
    family_error = _max_abs(
        random_wide["FOUR_DIRECTION_FAMILY_MAX"]
        - random_wide[variants].max(axis=1)
    )
    _check(rows, "four_direction_family_max_exact", family_error <= 1e-12,
        family_error)
    family = random_wide["FOUR_DIRECTION_FAMILY_MAX"]
    percentile_errors = []
    for row in outcome.itertuples(index=False):
        mean = portfolio.loc[portfolio["candidate"].eq(row.candidate),
            "primary_net_return"].mean()
        percentile_errors.append(
            abs(float(family.le(mean).mean())
                - float(row.random_four_direction_family_percentile))
        )
    percentile_error = max(percentile_errors)
    _check(rows, "outcome_family_percentiles_exact", percentile_error <= 1e-12,
        percentile_error)

    summary_errors = []
    for row in outcome.itertuples(index=False):
        sample = portfolio[portfolio["candidate"].eq(row.candidate)]
        summary_errors.extend(
            [
                len(sample) - int(row.weeks),
                sample["gross_return"].mean() * 10_000 - float(row.mean_gross_bp),
                sample["primary_net_return"].mean() * 10_000
                - float(row.mean_primary_net_bp),
                sample["short_gross_return"].mean() * 10_000
                - float(row.short_alt_gross_bp),
                sample["btc_gross_return"].mean() * 10_000
                - float(row.btc_hedge_gross_bp),
            ]
        )
    summary_error = _max_abs(summary_errors)
    _check(rows, "outcome_core_and_attribution_summaries_exact", summary_error <= 1e-10,
        summary_error)
    _check(rows, "all_gate_rows_ineligible", not gates["eligible"].any(), len(gates))
    _check(rows, "no_candidate_eligible", not outcome["eligible"].any(), len(outcome))
    _check(rows, "rejection_verdict_exact", outcome["verdict"].eq(
        "reject_rich_graph_premium_short_weekly"
    ).all(), "|".join(outcome["verdict"].astype(str)))
    _check(rows, "both_short_portfolios_negative_gross", outcome["mean_gross_bp"].lt(0).all(),
        float(outcome["mean_gross_bp"].max()))
    _check(rows, "short_sleeves_positive_but_btc_hedges_dominate", (
        outcome["short_alt_gross_bp"].gt(0)
        & outcome["btc_hedge_gross_bp"].lt(0)
        & (outcome["btc_hedge_gross_bp"].abs() > outcome["short_alt_gross_bp"])
    ).all(), float((outcome["btc_hedge_gross_bp"].abs()
        - outcome["short_alt_gross_bp"]).min()))
    _check(rows, "four_direction_family_percentiles_zero", outcome[
        "random_four_direction_family_percentile"
    ].eq(0).all(), float(outcome["random_four_direction_family_percentile"].max()))
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if "v198" in path.name.lower() or "rich_graph_premium" in path.name.lower()
    ]
    _check(rows, "no_live_config_created", not config_hits, "|".join(config_hits))

    audit = pd.DataFrame(rows)
    audit["round_verdict"] = np.where(
        audit["passed"].all(),
        "audit_pass_rich_premium_short_rejected_hedge_dependency_proven",
        "audit_failure_requires_investigation",
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    failed = audit[~audit["passed"]]
    text = [
        "# v19.8 Rich Graph-Premium Short Independent Audit",
        "",
        f"Verdict: `{audit['round_verdict'].iloc[0]}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; "
        f"failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "The positive alt-short attribution cannot survive as an independent",
        "beta-neutral short portfolio because the required long-BTC hedge dominates.",
        "No live, PaperLive, leverage, application, remote, or order scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v198_independent_audit(
    report_root: Path = AUDIT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v198_independently()
    root = ensure_dir(report_root)
    outputs = {"audit": root / "audit_checks.csv", "findings": findings_path}
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v198_independently", "write_v198_independent_audit"]
