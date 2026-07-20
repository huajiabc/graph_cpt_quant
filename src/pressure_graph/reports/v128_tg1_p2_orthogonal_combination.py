"""Fixed 50/50 combination of TG1 carry and the frozen P2 max8 core."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


TG1_PATH = Path(
    "reports/v12_6_turnover_governed_cross_venue_carry/weekly_portfolio.parquet"
)
P2_PATH = Path("reports/v0_7d2_cic_mir1_replay/paper_portfolio_trades.parquet")
REPORT_ROOT = Path("reports/v12_8_tg1_p2_orthogonal_combination")
P2_ID = "P2_CIC_COMBINED_BASKET_MAX8"
CANDIDATE = "CM1_50_50_TG1_P2_MAX8"


@dataclass(frozen=True)
class V128Config:
    tg1_path: Path = TG1_PATH
    p2_path: Path = P2_PATH
    report_root: Path = REPORT_ROOT
    p2_slots: int = 8
    tg1_weight: float = 0.5
    p2_weight: float = 0.5
    bootstrap_iterations: int = 2000
    seed: int = 20260715


def build_v128_weekly_panel(cfg: V128Config = V128Config()) -> pd.DataFrame:
    tg1 = pd.read_parquet(cfg.tg1_path)
    tg1["entry_time"] = pd.to_datetime(tg1["entry_time"], utc=True)
    tg1["exit_time"] = pd.to_datetime(tg1["exit_time"], utc=True)
    tg1["month_start"] = pd.to_datetime(tg1["month_start"], utc=True)
    p2 = pd.read_parquet(
        cfg.p2_path,
        columns=[
            "portfolio_id",
            "selected",
            "trade_id",
            "entry_time",
            "net_return_20bp",
        ],
    )
    p2["entry_time"] = pd.to_datetime(p2["entry_time"], utc=True, errors="coerce")
    p2["net_return_20bp"] = pd.to_numeric(
        p2["net_return_20bp"], errors="coerce"
    )
    p2 = p2[
        p2["portfolio_id"].eq(P2_ID)
        & p2["selected"].fillna(False).astype(bool)
    ].dropna(subset=["trade_id", "entry_time", "net_return_20bp"])
    p2 = p2.drop_duplicates("trade_id", keep="last")
    rows = []
    for item in tg1.sort_values("entry_time").itertuples(index=False):
        local = p2[
            p2["entry_time"].ge(item.entry_time)
            & p2["entry_time"].lt(item.exit_time)
        ]
        p2_return = float(local["net_return_20bp"].sum() / cfg.p2_slots)
        tg1_return = float(item.primary_net_return)
        rows.append(
            {
                "candidate": CANDIDATE,
                "entry_time": item.entry_time,
                "exit_time": item.exit_time,
                "month_start": item.month_start,
                "period": item.period,
                "tg1_return": tg1_return,
                "p2_return": p2_return,
                "p2_trades": len(local),
                "combined_return": cfg.tg1_weight * tg1_return
                + cfg.p2_weight * p2_return,
            }
        )
    return pd.DataFrame(rows)


def summarize_v128(
    panel: pd.DataFrame, cfg: V128Config = V128Config()
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    values = panel["combined_return"].to_numpy(dtype=float)
    draws = rng.choice(
        values, size=(cfg.bootstrap_iterations, len(values)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
    periods = panel.groupby("period", observed=True)["combined_return"].mean()
    months = panel.groupby("month_start", observed=True)["combined_return"].sum()
    positive = months[months.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.nan
    )
    counts = panel["period"].value_counts()
    shifted = panel["p2_return"].shift(1).fillna(panel["p2_return"].iloc[-1])
    shifted_combined = cfg.tg1_weight * panel["tg1_return"] + cfg.p2_weight * shifted
    correlation = float(panel[["tg1_return", "p2_return"]].corr().iloc[0, 1])
    row = {
        "candidate": CANDIDATE,
        "weeks": len(panel),
        "months": int(panel["month_start"].nunique()),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "p2_trades": int(panel["p2_trades"].sum()),
        "active_p2_weeks": int(panel["p2_trades"].gt(0).sum()),
        "sleeve_correlation": correlation,
        "mean_tg1_bp": float(panel["tg1_return"].mean() * 10_000),
        "mean_p2_bp": float(panel["p2_return"].mean() * 10_000),
        "mean_combined_bp": float(panel["combined_return"].mean() * 10_000),
        "development_combined_bp": float(
            periods.get("development", np.nan) * 10_000
        ),
        "validation_combined_bp": float(
            periods.get("validation", np.nan) * 10_000
        ),
        "holdout_combined_bp": float(periods.get("holdout", np.nan) * 10_000),
        "bootstrap_95_low_bp": float(ci_low * 10_000),
        "bootstrap_95_high_bp": float(ci_high * 10_000),
        "positive_month_concentration": concentration,
        "worst_period_bp": float(periods.min() * 10_000),
        "shifted_p2_mean_bp": float(shifted_combined.mean() * 10_000),
    }
    row["promote"] = bool(
        row["weeks"] >= 40
        and row["months"] >= 10
        and row["validation_weeks"] >= 10
        and row["holdout_weeks"] >= 8
        and row["mean_tg1_bp"] > 0
        and row["mean_p2_bp"] > 0
        and abs(row["sleeve_correlation"]) <= 0.50
        and all(
            row[key] > 0
            for key in (
                "development_combined_bp",
                "validation_combined_bp",
                "holdout_combined_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["positive_month_concentration"] <= 0.35
        and row["worst_period_bp"] >= -40
    )
    return pd.DataFrame([row])


def write_v128_tg1_p2_orthogonal_combination(
    cfg: V128Config = V128Config(),
) -> dict[str, Path]:
    panel = build_v128_weekly_panel(cfg)
    summary = summarize_v128(panel, cfg)
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_combination.parquet",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v128_tg1_p2_orthogonal_combination_findings_2026_07_15.md"
        ),
    }
    panel.to_parquet(paths["panel"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "weeks": len(panel),
                "p2_trades": int(panel["p2_trades"].sum()),
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_portfolio_shadow" if promoted else "reject_combination"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v12.8 TG1 + Frozen P2 Orthogonal Combination Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The 50/50 capital weight and exact frozen sleeves were registered before "
                "alignment. No existing PaperLive strategy was changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
