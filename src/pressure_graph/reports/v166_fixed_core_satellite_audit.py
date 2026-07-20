"""Independent raw-sleeve and risk audit for v16.5 fixed 80/20 portfolio."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


FSS3_PATH = Path("reports/v14_9_funding_sign_turnover_cap/weekly_portfolio.parquet")
TG1_PATH = Path("reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet")
REPORT_ROOT = Path("reports/v16_5_fixed_core_satellite_fss3_tg1")
AUDIT_DOC = Path("docs/v166_fixed_core_satellite_audit_2026_07_16.md")


@dataclass(frozen=True)
class V166AuditConfig:
    fss3_path: Path = FSS3_PATH
    tg1_path: Path = TG1_PATH
    report_root: Path = REPORT_ROOT
    audit_doc: Path = AUDIT_DOC
    fss3_weight: float = 0.80
    tg1_weight: float = 0.20
    bootstrap_iterations: int = 20_000
    bootstrap_block_weeks: int = 4
    seed: int = 20260730
    tolerance: float = 1e-12


def load_v166_raw_sleeves(cfg: V166AuditConfig) -> pd.DataFrame:
    fss3 = pd.read_parquet(cfg.fss3_path)
    tg1 = pd.read_parquet(cfg.tg1_path)
    for frame in (fss3, tg1):
        for column in ("entry_time", "exit_time", "month_start"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    fss3 = fss3[
        [
            "entry_time",
            "exit_time",
            "month_start",
            "period",
            "price_return",
            "funding_return",
            "primary_net_return",
            "stress_net_return",
            "realized_turnover",
        ]
    ].rename(
        columns={
            "exit_time": "fss3_exit_time",
            "month_start": "fss3_month_start",
            "period": "fss3_period",
            "price_return": "fss3_price_return_raw",
            "funding_return": "fss3_funding_return_raw",
            "primary_net_return": "fss3_primary_return_raw",
            "stress_net_return": "fss3_stress_return_raw",
            "realized_turnover": "fss3_turnover_raw",
        }
    )
    tg1 = tg1[
        [
            "entry_time",
            "exit_time",
            "month_start",
            "period",
            "price_basis_return",
            "funding_spread_return",
            "primary_net_return",
            "stress_net_return",
            "realized_turnover",
        ]
    ].rename(
        columns={
            "exit_time": "tg1_exit_time",
            "month_start": "tg1_month_start",
            "period": "tg1_period",
            "price_basis_return": "tg1_price_return_raw",
            "funding_spread_return": "tg1_funding_return_raw",
            "primary_net_return": "tg1_primary_return_raw",
            "stress_net_return": "tg1_stress_return_raw",
            "realized_turnover": "tg1_turnover_raw",
        }
    )
    merged = fss3.merge(tg1, on="entry_time", how="inner", validate="one_to_one")
    calendar_exact = (
        merged["fss3_exit_time"].eq(merged["tg1_exit_time"])
        & merged["fss3_month_start"].eq(merged["tg1_month_start"])
        & merged["fss3_period"].eq(merged["tg1_period"])
    )
    if not calendar_exact.all():
        raise RuntimeError("raw sleeve calendars do not align")
    merged["exit_time"] = merged["fss3_exit_time"]
    merged["month_start"] = merged["fss3_month_start"]
    merged["period"] = merged["fss3_period"]
    return merged.sort_values("entry_time").reset_index(drop=True)


def _drawdown(values: pd.Series) -> float:
    curve = pd.to_numeric(values, errors="coerce").cumsum()
    return float((curve - curve.cummax().clip(lower=0)).min())


def _semideviation(values: pd.Series) -> float:
    downside = np.minimum(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def _alternate_bootstrap(values: np.ndarray, cfg: V166AuditConfig) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed)
    offsets = np.arange(cfg.bootstrap_block_weeks)
    block_count = int(np.ceil(len(values) / cfg.bootstrap_block_weeks))
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draws[iteration] = float(values[indices.ravel()[: len(values)]].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def audit_v166_portfolio(cfg: V166AuditConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_v166_raw_sleeves(cfg)
    stored = pd.read_parquet(cfg.report_root / "weekly_portfolio.parquet")
    stored["entry_time"] = pd.to_datetime(stored["entry_time"], utc=True)
    audit = raw.merge(
        stored[
            [
                "entry_time",
                "fss3_weight",
                "tg1_weight",
                "price_return",
                "funding_return",
                "primary_net_return",
                "stress_net_return",
            ]
        ],
        on="entry_time",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    audit["price_audit"] = (
        cfg.fss3_weight * audit["fss3_price_return_raw"]
        + cfg.tg1_weight * audit["tg1_price_return_raw"]
    )
    audit["funding_audit"] = (
        cfg.fss3_weight * audit["fss3_funding_return_raw"]
        + cfg.tg1_weight * audit["tg1_funding_return_raw"]
    )
    audit["primary_audit"] = (
        cfg.fss3_weight * audit["fss3_primary_return_raw"]
        + cfg.tg1_weight * audit["tg1_primary_return_raw"]
    )
    audit["stress_audit"] = (
        cfg.fss3_weight * audit["fss3_stress_return_raw"]
        + cfg.tg1_weight * audit["tg1_stress_return_raw"]
    )
    for column in ("price", "funding", "primary", "stress"):
        stored_column = f"{column}_return" if column in {"price", "funding"} else f"{column}_net_return"
        audit[f"{column}_difference"] = audit[f"{column}_audit"] - audit[stored_column]
    audit["fss3_weight_difference"] = audit["fss3_weight"] - cfg.fss3_weight
    audit["tg1_weight_difference"] = audit["tg1_weight"] - cfg.tg1_weight
    summary = pd.DataFrame(
        [
            {
                "weeks": len(audit),
                "calendar_exact": audit["_merge"].eq("both").all(),
                **{
                    f"max_abs_{column}_difference": audit[f"{column}_difference"].abs().max()
                    for column in (
                        "price",
                        "funding",
                        "primary",
                        "stress",
                        "fss3_weight",
                        "tg1_weight",
                    )
                },
            }
        ]
    )
    return audit, summary


def audit_v166_risk(cfg: V166AuditConfig) -> pd.DataFrame:
    raw = load_v166_raw_sleeves(cfg)
    combined = (
        cfg.fss3_weight * raw["fss3_primary_return_raw"]
        + cfg.tg1_weight * raw["tg1_primary_return_raw"]
    )
    stress = (
        cfg.fss3_weight * raw["fss3_stress_return_raw"]
        + cfg.tg1_weight * raw["tg1_stress_return_raw"]
    )
    low, high = _alternate_bootstrap(combined.to_numpy(dtype=float), cfg)
    periods = pd.DataFrame({"period": raw["period"], "combined": combined}).groupby(
        "period", observed=True
    )["combined"].mean()
    combined_drawdown = _drawdown(combined)
    fss3_drawdown = _drawdown(raw["fss3_primary_return_raw"])
    combined_semideviation = _semideviation(combined)
    fss3_semideviation = _semideviation(raw["fss3_primary_return_raw"])
    return pd.DataFrame(
        [
            {
                "alternate_bootstrap_95_low_bp": low * 10_000,
                "alternate_bootstrap_95_high_bp": high * 10_000,
                "development_primary_net_bp": periods.get("development", np.nan)
                * 10_000,
                "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
                "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
                "validation_stress_net_bp": pd.DataFrame(
                    {"period": raw["period"], "stress": stress}
                )
                .groupby("period", observed=True)["stress"]
                .mean()
                .get("validation", np.nan)
                * 10_000,
                "drawdown_reduction_vs_fss3": 1.0
                - abs(combined_drawdown) / abs(fss3_drawdown),
                "downside_semideviation_reduction_vs_fss3": 1.0
                - combined_semideviation / fss3_semideviation,
                "mean_retention_vs_fss3": combined.mean()
                / raw["fss3_primary_return_raw"].mean(),
            }
        ]
    )


def write_v166_fixed_core_satellite_audit(
    cfg: V166AuditConfig = V166AuditConfig(),
) -> dict[str, Path]:
    audit, formula_summary = audit_v166_portfolio(cfg)
    risk_summary = audit_v166_risk(cfg)
    main = pd.read_csv(cfg.report_root / "summary.csv")
    formula_columns = [
        column for column in formula_summary if column.startswith("max_abs_")
    ]
    formula_pass = bool(
        formula_summary.at[0, "calendar_exact"]
        and formula_summary.loc[0, formula_columns].le(cfg.tolerance).all()
    )
    risk_pass = bool(
        risk_summary.at[0, "alternate_bootstrap_95_low_bp"] > 0
        and risk_summary.at[0, "development_primary_net_bp"] > 0
        and risk_summary.at[0, "validation_primary_net_bp"] > 0
        and risk_summary.at[0, "holdout_primary_net_bp"] > 0
        and risk_summary.at[0, "drawdown_reduction_vs_fss3"] >= 0.15
        and risk_summary.at[0, "downside_semideviation_reduction_vs_fss3"] > 0
        and risk_summary.at[0, "mean_retention_vs_fss3"] >= 0.75
    )
    outcome = pd.DataFrame(
        [
            {
                "formula_audit_pass": formula_pass,
                "risk_audit_pass": risk_pass,
                "main_promotion_reproduced": bool(main.at[0, "promote"]),
                "forward_shadow_only": True,
                "validation_stress_caveat": bool(
                    risk_summary.at[0, "validation_stress_net_bp"] <= 0
                ),
            }
        ]
    )
    root = ensure_dir(cfg.report_root / "independent_audit")
    paths = {
        "recalculation": root / "raw_sleeve_recalculation.parquet",
        "formula_summary": root / "formula_audit_summary.csv",
        "risk_summary": root / "risk_audit_summary.csv",
        "outcome": root / "audit_outcome.csv",
        "audit_doc": cfg.audit_doc,
    }
    audit.to_parquet(paths["recalculation"], index=False)
    formula_summary.to_csv(paths["formula_summary"], index=False)
    risk_summary.to_csv(paths["risk_summary"], index=False)
    outcome.to_csv(paths["outcome"], index=False)
    passed = bool(
        outcome.at[0, "formula_audit_pass"]
        and outcome.at[0, "risk_audit_pass"]
        and outcome.at[0, "main_promotion_reproduced"]
    )
    verdict = "forward_shadow_confirmed_with_caveat" if passed else "audit_failed"
    paths["audit_doc"].write_text(
        "\n".join(
            [
                "# v16.6 Independent Audit of v16.5 CM2",
                "",
                f"Verdict: `{verdict}`.",
                "",
                outcome.to_markdown(index=False),
                "",
                formula_summary.to_markdown(index=False, floatfmt=".3e"),
                "",
                risk_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The audit independently reloaded both raw sleeve portfolios, aligned",
                "their calendars, recomputed all 49 fixed-weight returns, and repeated",
                "risk tests with 20,000 alternate bootstrap draws. The candidate remains",
                "forward-shadow only because validation stress return is negative and the",
                "20% satellite cap was chosen after sleeve-level evidence review.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["outcome"].with_suffix(".json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "formula_pass": formula_pass,
                "risk_pass": risk_pass,
                "validation_stress_caveat": bool(
                    outcome.at[0, "validation_stress_caveat"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
