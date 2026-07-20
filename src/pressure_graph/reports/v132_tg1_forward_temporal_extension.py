"""Exact TG1 extension through the last complete July 2026 week."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import (
    BTC,
    estimate_v106_betas,
)
from pressure_graph.reports.v110_balanced_topology_break import (
    _residualize,
    build_v110_communities,
)
from pressure_graph.reports.v125_cross_venue_perpetual_carry import (
    V125Config,
    build_v125_weekly_panel,
)
from pressure_graph.reports.v126_turnover_governed_cross_venue_carry import (
    V126Config,
    build_v126_nulls,
    build_v126_portfolio,
    summarize_v126,
)


RAW_BYBIT_ROOT = Path("data/raw/bybit")
BINANCE_ROOT = Path("data/external/binance_um_carry")
RECENT_ROOT = Path("data/external/recent_perp_carry")
BASE_MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)
CANONICAL_TG1_PATH = Path(
    "reports/v12_6_turnover_governed_cross_venue_carry/weekly_portfolio.parquet"
)
REPORT_ROOT = Path("reports/v13_2_tg1_forward_temporal_extension")
CANDIDATE = "TG1_FORWARD_EXTENDED_TO_2026_07"


def load_v132_bybit_klines() -> pd.DataFrame:
    symbols = sorted(
        set(pd.read_csv(BASE_MEMBERSHIP_PATH)["symbol"].astype(str)) | {BTC}
    )
    frames = []
    for symbol in symbols:
        paths = [
            RAW_BYBIT_ROOT / "klines" / f"{symbol}.parquet",
            RECENT_ROOT / "bybit_klines_15m" / f"{symbol}.parquet",
        ]
        for path in paths:
            if not path.exists():
                continue
            frame = pd.read_parquet(
                path, columns=["symbol", "bar_open_time", "bar_close_time", "close"]
            )
            frames.append(frame)
    klines = pd.concat(frames, ignore_index=True)
    for column in ("bar_open_time", "bar_close_time"):
        klines[column] = pd.to_datetime(klines[column], utc=True, errors="coerce")
    klines["close"] = pd.to_numeric(klines["close"], errors="coerce")
    return (
        klines.dropna(subset=["symbol", "bar_open_time", "bar_close_time", "close"])
        .drop_duplicates(["symbol", "bar_open_time"], keep="last")
        .sort_values(["bar_close_time", "symbol"])
        .reset_index(drop=True)
    )


def hourly_bybit_prices(klines: pd.DataFrame) -> pd.DataFrame:
    hourly = klines[
        klines["bar_close_time"].dt.minute.eq(0)
        & klines["bar_close_time"].dt.second.eq(0)
    ][["symbol", "bar_close_time", "close"]].rename(
        columns={"bar_close_time": "feature_time"}
    )
    return (
        hourly.drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def build_v132_july_membership(
    hourly_prices: pd.DataFrame,
    min_samples: int = 500,
) -> pd.DataFrame:
    july = pd.Timestamp("2026-07-01", tz="UTC")
    history = hourly_prices[
        hourly_prices["feature_time"].ge(july - pd.Timedelta(days=30))
        & hourly_prices["feature_time"].lt(july)
    ]
    close = history.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    betas = estimate_v106_betas(returns)
    residual = _residualize(returns, betas)
    communities = build_v110_communities(
        residual, community_count=8, min_samples=min_samples
    )
    if len(communities) != 8:
        raise RuntimeError(f"July membership failed: expected 8 groups, got {len(communities)}")
    rows = []
    for index, members in enumerate(communities, start=1):
        community_id = f"2026-07:BSP{index:02d}"
        for symbol in members:
            rows.append(
                {
                    "month_start": july,
                    "community_id": community_id,
                    "symbol": symbol,
                    "community_size": len(members),
                }
            )
    return pd.DataFrame(rows)


def _combined_funding(
    historical_root: Path,
    recent_root: Path,
    symbol_column: str,
) -> pd.DataFrame:
    symbols = sorted(pd.read_csv(BASE_MEMBERSHIP_PATH)["symbol"].astype(str).unique())
    frames = []
    for symbol in symbols:
        for path in (
            historical_root / f"{symbol}.parquet",
            recent_root / f"{symbol}.parquet",
        ):
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if symbol_column in frame.columns and symbol_column != "symbol":
                frame = frame.rename(columns={symbol_column: "symbol"})
            frame["symbol"] = symbol
            frames.append(frame[["symbol", "funding_time", "funding_rate_settled"]])
    funding = pd.concat(frames, ignore_index=True)
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True, errors="coerce"
    ).dt.floor("s")
    funding["funding_rate_settled"] = pd.to_numeric(
        funding["funding_rate_settled"], errors="coerce"
    )
    return (
        funding.dropna(subset=["symbol", "funding_time", "funding_rate_settled"])
        .drop_duplicates(["symbol", "funding_time"], keep="last")
        .sort_values(["funding_time", "symbol"])
        .reset_index(drop=True)
    )


def load_v132_binance_prices() -> pd.DataFrame:
    symbols = sorted(pd.read_csv(BASE_MEMBERSHIP_PATH)["symbol"].astype(str).unique())
    frames = []
    for symbol in symbols:
        for path in (
            BINANCE_ROOT / "klines_1h" / f"{symbol}.parquet",
            RECENT_ROOT / "binance_klines_1h" / f"{symbol}.parquet",
        ):
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            if "bybit_symbol" in frame.columns:
                frame = frame.rename(columns={"bybit_symbol": "symbol"})
            frame["symbol"] = symbol
            frames.append(frame[["symbol", "feature_time", "close"]])
    prices = pd.concat(frames, ignore_index=True)
    prices["feature_time"] = pd.to_datetime(
        prices["feature_time"], utc=True, errors="coerce"
    )
    prices["binance_close"] = pd.to_numeric(prices["close"], errors="coerce")
    return (
        prices.dropna(subset=["symbol", "feature_time", "binance_close"])
        .drop_duplicates(["symbol", "feature_time"], keep="last")
        .sort_values(["feature_time", "symbol"])
        [["symbol", "feature_time", "binance_close"]]
        .reset_index(drop=True)
    )


def _history_consistency(portfolio: pd.DataFrame) -> float:
    canonical = pd.read_parquet(CANONICAL_TG1_PATH)[
        ["entry_time", "primary_net_return"]
    ].rename(columns={"primary_net_return": "canonical_return"})
    merged = portfolio.merge(canonical, on="entry_time", how="inner")
    if len(merged) != len(canonical):
        raise RuntimeError(
            f"Historical overlap mismatch: {len(merged)} vs {len(canonical)} weeks"
        )
    # The canonical sample charged a full terminal close on its last week.
    # Continuing the same path rolls that 20 bp one-way close to the new end.
    old_terminal = canonical["entry_time"].max()
    canonical.loc[
        canonical["entry_time"].eq(old_terminal), "canonical_return"
    ] += 0.002
    merged = portfolio.merge(canonical, on="entry_time", how="inner")
    difference = (merged["primary_net_return"] - merged["canonical_return"]).abs()
    maximum = float(difference.max())
    if maximum > 1e-12:
        raise RuntimeError(f"Historical TG1 return drifted by {maximum:.3e}")
    return maximum


def write_v132_tg1_forward_temporal_extension() -> dict[str, Path]:
    klines = load_v132_bybit_klines()
    bybit_prices = hourly_bybit_prices(klines)
    july_membership = build_v132_july_membership(bybit_prices)
    membership = pd.read_csv(BASE_MEMBERSHIP_PATH)
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    membership = pd.concat([membership, july_membership], ignore_index=True)
    root = ensure_dir(REPORT_ROOT)
    membership_path = root / "monthly_balanced_membership_extended.csv"
    membership.to_csv(membership_path, index=False)
    bybit_funding = _combined_funding(
        RAW_BYBIT_ROOT / "funding",
        RECENT_ROOT / "bybit_funding",
        "symbol",
    )
    binance_funding = _combined_funding(
        BINANCE_ROOT / "funding",
        RECENT_ROOT / "binance_funding",
        "bybit_symbol",
    )
    binance_prices = load_v132_binance_prices()
    panel_cfg = V125Config(membership_path=membership_path)
    panel = build_v125_weekly_panel(
        bybit_funding, binance_funding, bybit_prices, binance_prices, panel_cfg
    )
    tg_cfg = V126Config(
        panel_path=root / "weekly_symbol_panel.parquet",
        report_root=root,
    )
    portfolio = build_v126_portfolio(panel, tg_cfg)
    historical_max_diff = _history_consistency(portfolio)
    portfolio["candidate"] = CANDIDATE
    nulls = build_v126_nulls(panel, portfolio, tg_cfg)
    summary = summarize_v126(portfolio, nulls, replace(tg_cfg, seed=tg_cfg.seed + 30))
    summary["candidate"] = CANDIDATE
    paths = {
        "membership": membership_path,
        "panel": root / "weekly_symbol_panel.parquet",
        "portfolio": root / "weekly_portfolio.parquet",
        "nulls": root / "null_distributions.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": Path(
            "docs/v132_tg1_forward_temporal_extension_findings_2026_07_15.md"
        ),
    }
    panel.to_parquet(paths["panel"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    summary.to_csv(paths["summary"], index=False)
    promoted = bool(summary.loc[0, "promote"])
    new_weeks = portfolio[portfolio["entry_time"].gt(pd.Timestamp("2026-05-25", tz="UTC"))]
    paths["metadata"].write_text(
        json.dumps(
            {
                "july_members": int(july_membership["symbol"].nunique()),
                "july_community_sizes": sorted(
                    july_membership.groupby("community_id").size().astype(int).tolist()
                ),
                "weeks": len(portfolio),
                "new_weeks": len(new_weeks),
                "last_entry": portfolio["entry_time"].max().isoformat(),
                "historical_max_abs_return_diff": historical_max_diff,
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
                "# v13.2 Exact TG1 Forward Temporal-Extension Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Newly added weeks",
                "",
                new_weeks[
                    [
                        "entry_time",
                        "selected_symbols",
                        "funding_spread_return",
                        "price_basis_return",
                        "realized_turnover",
                        "primary_net_return",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
                f"Historical overlap maximum absolute return difference: "
                f"`{historical_max_diff:.3e}`. PaperLive was not changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
