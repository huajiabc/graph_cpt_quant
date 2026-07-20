"""Exact TG1 rerun with the preregistered July 2025 temporal extension."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v123_cross_sectional_funding_carry import (
    load_v123_funding,
    load_v123_prices,
)
from pressure_graph.reports.v125_cross_venue_perpetual_carry import (
    V125Config,
    build_v125_weekly_panel,
    load_v125_binance_funding,
    load_v125_binance_prices,
)
from pressure_graph.reports.v126_turnover_governed_cross_venue_carry import (
    V126Config,
    build_v126_nulls,
    build_v126_portfolio,
    summarize_v126,
)


REPORT_ROOT = Path("reports/v13_1_tg1_temporal_extension")
CANDIDATE = "TG1_EXTENDED_FROM_2025_07"


def write_v131_tg1_temporal_extension() -> dict[str, Path]:
    source_cfg = V125Config(first_entry=pd.Timestamp("2025-07-07", tz="UTC"))
    panel = build_v125_weekly_panel(
        load_v123_funding(source_cfg),
        load_v125_binance_funding(source_cfg),
        load_v123_prices(source_cfg),
        load_v125_binance_prices(source_cfg),
        source_cfg,
    )
    tg_cfg = V126Config(
        panel_path=REPORT_ROOT / "weekly_symbol_panel.parquet",
        report_root=REPORT_ROOT,
    )
    portfolio = build_v126_portfolio(panel, tg_cfg)
    portfolio["candidate"] = CANDIDATE
    nulls = build_v126_nulls(panel, portfolio, tg_cfg)
    summary = summarize_v126(portfolio, nulls, replace(tg_cfg, seed=tg_cfg.seed + 20))
    summary["candidate"] = CANDIDATE
    root = ensure_dir(REPORT_ROOT)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path("docs/v131_tg1_temporal_extension_findings_2026_07_15.md"),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    paths["metadata"].write_text(
        json.dumps(
            {
                "panel_rows": len(panel),
                "weeks": len(portfolio),
                "first_entry": portfolio["entry_time"].min().isoformat(),
                "promoted": [CANDIDATE] if promoted else [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_candidate" if promoted else "reject_as_tradable_alpha"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v13.1 Exact TG1 Temporal-Extension Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The added July weeks use the unchanged TG1 score, path, costs, null, and "
                "gates. No existing PaperLive strategy was changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
