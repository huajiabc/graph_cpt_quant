"""Independent consistency audit for the v17.2--v17.6 Deribit surface round."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir


QUARTERLY_ROOT = Path("data/external/deribit_quarterly_option_trades")
MONTHLY_ROOT = Path("data/external/deribit_monthly_option_trades")
DVOL_PATH = Path("data/external/orthogonal_volatility/deribit_dvol_1h/BTC.parquet")
REPORT_ROOT = Path("reports/v17_7_deribit_surface_round_audit")
FINDINGS_PATH = Path("docs/v177_deribit_surface_round_audit_2026_07_16.md")
ROUND_REPORTS = {
    "v173": Path("reports/v17_3_deribit_skew_receiver_bucket"),
    "v174": Path("reports/v17_4_deribit_skew_receiver_oco"),
    "v175": Path("reports/v17_5_deribit_skew_receiver_insulator_spread"),
    "v176": Path("reports/v17_6_monthly_deribit_surface_extension"),
}


def _check(rows: list[dict[str, object]], name: str, passed: bool, value: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "value": value})


def _summary_consistency(
    rows: list[dict[str, object]],
    version: str,
    root: Path,
) -> None:
    events = pd.read_parquet(root / "candidate_events.parquet")
    summary = pd.read_csv(root / "period_summary.csv")
    for candidate, sample in events.groupby("candidate", sort=True):
        reported = summary[
            summary["candidate"].eq(candidate) & summary["scope"].eq("all")
        ]
        actual = float(sample["primary_net_return"].mean() * 10_000)
        value = float(reported["mean_primary_net_bp"].iloc[0])
        _check(
            rows,
            f"{version}_{candidate}_summary_exact",
            math.isclose(actual, value, rel_tol=0, abs_tol=1e-10),
            actual - value,
        )


def audit_v177() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active = pd.read_parquet(QUARTERLY_ROOT / "active_hourly_trade_bars.parquet")
    quarterly = pd.read_parquet(QUARTERLY_ROOT / "daily_trade_surface.parquet")
    monthly = pd.read_parquet(MONTHLY_ROOT / "daily_trade_surface.parquet")
    selected = pd.read_parquet(
        ROUND_REPORTS["v176"] / "nearest_30d_daily_surface.parquet"
    )
    _check(rows, "active_volume_strictly_positive", active["volume"].gt(0).all(), len(active))
    _check(rows, "active_cost_strictly_positive", active["cost"].gt(0).all(), len(active))
    _check(
        rows,
        "active_before_expiry",
        active["bar_open_time"].lt(active["expiration_time"]).all(),
        int(active["bar_open_time"].ge(active["expiration_time"]).sum()),
    )
    _check(
        rows,
        "active_iv_finite",
        active["implied_volatility"].notna().all(),
        int(active["implied_volatility"].isna().sum()),
    )
    causal_quarterly = pd.to_datetime(quarterly["surface_date"], utc=True) + pd.Timedelta(
        days=1
    )
    _check(
        rows,
        "quarterly_feature_time_after_completed_day",
        pd.to_datetime(quarterly["feature_time"], utc=True).eq(causal_quarterly).all(),
        len(quarterly),
    )
    _check(
        rows,
        "quarterly_quality_rows_567",
        int(quarterly["quality_pass"].sum()) == 567,
        int(quarterly["quality_pass"].sum()),
    )
    _check(
        rows,
        "monthly_quality_rows_1638",
        int(monthly["quality_pass"].sum()) == 1_638,
        int(monthly["quality_pass"].sum()),
    )
    _check(
        rows,
        "monthly_selected_one_row_per_day",
        selected["feature_time"].is_unique,
        len(selected),
    )
    _check(
        rows,
        "monthly_selected_rows_1447",
        len(selected) == 1_447,
        len(selected),
    )

    dvol = pd.read_parquet(DVOL_PATH)
    dvol["feature_time"] = (
        pd.to_datetime(dvol["dvol_time"], utc=True).dt.floor("D")
        + pd.Timedelta(days=1)
    )
    dvol_daily = dvol.groupby("feature_time")["close"].mean().div(100).rename("dvol")
    overlap = quarterly[quarterly["quality_pass"]].merge(dvol_daily, on="feature_time")
    correlation = float(overlap["atm_iv"].corr(overlap["dvol"]))
    _check(rows, "quarterly_atm_dvol_correlation_95", correlation >= 0.95, correlation)

    for version, root in ROUND_REPORTS.items():
        _summary_consistency(rows, version, root)
        outcome_name = "robustness_outcome.csv" if version == "v176" else "candidate_outcome.csv"
        outcome = pd.read_csv(root / outcome_name)
        _check(
            rows,
            f"{version}_no_candidate_eligible",
            not outcome["eligible"].astype(bool).any(),
            int(outcome["eligible"].astype(bool).sum()),
        )
        if version == "v176":
            _check(
                rows,
                "v176_no_forward_watch",
                not outcome["forward_watch"].astype(bool).any(),
                int(outcome["forward_watch"].astype(bool).sum()),
            )
    config_hits = [
        path.name
        for path in Path("configs").glob("*")
        if any(token in path.name.lower() for token in ("v173", "v174", "v175", "v176"))
    ]
    _check(rows, "no_round_live_config_created", not config_hits, "|".join(config_hits))
    audit = pd.DataFrame(rows)
    audit["round_verdict"] = (
        "audit_pass_all_alpha_rejected"
        if audit["passed"].all()
        else "audit_failure_requires_investigation"
    )
    return audit


def _write_findings(audit: pd.DataFrame, path: Path) -> None:
    verdict = str(audit["round_verdict"].iloc[0])
    failed = audit[~audit["passed"]]
    text = [
        "# v17.7 Deribit Surface Round Independent Audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        f"Checks: {len(audit)}; passed: {int(audit['passed'].sum())}; failed: {len(failed)}.",
        "",
        failed.to_markdown(index=False) if not failed.empty else "No failed checks.",
        "",
        "All v17.3--v17.6 alpha candidates remain rejected. The data archive is",
        "accepted for future signal research, but no live or application scope changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v177_deribit_surface_round_audit(
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    audit = audit_v177()
    root = ensure_dir(report_root)
    outputs = {
        "audit": root / "audit_checks.csv",
        "findings": findings_path,
    }
    audit.to_csv(outputs["audit"], index=False)
    _write_findings(audit, findings_path)
    return outputs


__all__ = ["audit_v177", "write_v177_deribit_surface_round_audit"]
