"""Retrospective one-day mechanism pilot for liquidation-to-price transmission."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    DATA_ROOT,
    EXPECTED_SYMBOLS,
    load_v2334_liquidations,
)


PRICE_ROOT = DATA_ROOT / "mechanism_prices"
REPORT_ROOT = Path("reports/v23_36_liquidation_price_mechanism_pilot")
FINDINGS_PATH = Path(
    "docs/v2336_liquidation_price_mechanism_pilot_2026_07_17.md"
)
PRIMARY_FEATURES = (
    "log1p_liq_15m_total_usd",
    "liq_15m_net_forced_buy_share",
    "liq_15m_active_symbols",
    "liq_15m_symbol_notional_hhi",
    "liq_15m_to_60m_notional_share",
    "log1p_alt_liq_15m_total_usd",
    "alt_liq_15m_net_forced_buy_share",
    "alt_liq_15m_active_symbols",
    "alt_liq_15m_symbol_notional_hhi",
)
PRIMARY_OUTCOMES = (
    "btc_log_return_15m",
    "btc_abs_log_return_15m",
    "btc_log_range_15m",
    "btc_log_return_60m",
    "btc_abs_log_return_60m",
    "btc_log_range_60m",
)


def load_v2336_prices(price_root: Path = PRICE_ROOT) -> pd.DataFrame:
    coverage = json.loads((price_root / "coverage.json").read_text(encoding="utf-8"))
    requested_end = pd.Timestamp(coverage["end"])
    rows = []
    for symbol in EXPECTED_SYMBOLS:
        path = price_root / "bybit_klines_15m" / f"{symbol}.parquet"
        frame = pd.read_parquet(path)
        frame["bar_open_time"] = pd.to_datetime(
            frame["bar_open_time"], utc=True, errors="coerce"
        )
        frame["bar_close_time"] = pd.to_datetime(
            frame["bar_close_time"], utc=True, errors="coerce"
        )
        frame["symbol"] = symbol
        rows.append(frame[frame["bar_open_time"].lt(requested_end)])
    prices = pd.concat(rows, ignore_index=True)
    for column in ("open", "high", "low", "close"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    return prices.sort_values(["symbol", "bar_open_time"]).reset_index(drop=True)


def build_v2336_price_outcomes(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, local in prices.groupby("symbol", sort=True):
        local = local.sort_values("bar_open_time").reset_index(drop=True)
        for index, bar in local.iterrows():
            decision = pd.Timestamp(bar["bar_open_time"])
            row: dict[str, object] = {
                "symbol": symbol,
                "decision_time": decision,
            }
            complete = True
            for horizon, count in ((15, 1), (60, 4)):
                path = local.iloc[index : index + count]
                expected = pd.date_range(
                    decision, periods=count, freq="15min", tz="UTC"
                )
                if len(path) != count or not pd.DatetimeIndex(
                    path["bar_open_time"]
                ).equals(expected):
                    complete = False
                    break
                entry = float(path.iloc[0]["open"])
                exit_price = float(path.iloc[-1]["close"])
                high = float(path["high"].max())
                low = float(path["low"].min())
                log_return = float(np.log(exit_price / entry))
                row[f"log_return_{horizon}m"] = log_return
                row[f"abs_log_return_{horizon}m"] = abs(log_return)
                row[f"log_range_{horizon}m"] = float(np.log(high / low))
            if complete:
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["symbol", "decision_time"]).reset_index(
        drop=True
    )


def _liquidation_metrics(events: pd.DataFrame) -> dict[str, float]:
    sell = events["position_side"].eq("long")
    buy = events["position_side"].eq("short")
    sell_usd = float(events.loc[sell, "notional_usd"].sum())
    buy_usd = float(events.loc[buy, "notional_usd"].sum())
    total = sell_usd + buy_usd
    symbol_usd = events.groupby("bybit_symbol")["notional_usd"].sum()
    return {
        "event_count": float(len(events)),
        "forced_sell_usd": sell_usd,
        "forced_buy_usd": buy_usd,
        "total_usd": total,
        "net_forced_buy_share": (buy_usd - sell_usd) / total if total else 0.0,
        "active_symbols": float(events["bybit_symbol"].nunique()),
        "symbol_notional_hhi": (
            float(np.square(symbol_usd / total).sum()) if total else 0.0
        ),
        "btc_notional_share": (
            float(
                events.loc[
                    events["bybit_symbol"].eq("BTCUSDT"), "notional_usd"
                ].sum()
            )
            / total
            if total
            else 0.0
        ),
    }


def build_v2336_market_panel(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    btc = outcomes[outcomes["symbol"].eq("BTCUSDT")].copy()
    btc = btc.rename(
        columns={
            column: f"btc_{column}"
            for column in btc.columns
            if column not in {"symbol", "decision_time"}
        }
    ).drop(columns="symbol")
    rows = []
    for decision in btc["decision_time"]:
        row: dict[str, object] = {"decision_time": decision}
        for prefix, universe in (
            ("liq", events),
            ("alt_liq", events[events["bybit_symbol"].ne("BTCUSDT")]),
        ):
            for window in (15, 60):
                start = decision - pd.Timedelta(minutes=window)
                local = universe[
                    universe["event_time"].ge(start)
                    & universe["event_time"].lt(decision)
                ]
                for name, value in _liquidation_metrics(local).items():
                    row[f"{prefix}_{window}m_{name}"] = value
            row[f"{prefix}_15m_to_60m_notional_share"] = row[
                f"{prefix}_15m_total_usd"
            ] / max(float(row[f"{prefix}_60m_total_usd"]), 1.0)
            row[f"log1p_{prefix}_15m_total_usd"] = float(
                np.log1p(row[f"{prefix}_15m_total_usd"])
            )
        rows.append(row)
    features = pd.DataFrame(rows)
    return features.merge(btc, on="decision_time", how="inner", validate="one_to_one")


def build_v2336_own_symbol_panel(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    local = events.copy()
    local["decision_time"] = local["event_time"].dt.floor("15min") + pd.Timedelta(
        minutes=15
    )
    rows = []
    for (symbol, decision), group in local.groupby(
        ["bybit_symbol", "decision_time"], sort=True
    ):
        metrics = _liquidation_metrics(group)
        rows.append(
            {
                "symbol": symbol,
                "decision_time": decision,
                **metrics,
                "flow_direction": float(np.sign(metrics["net_forced_buy_share"])),
            }
        )
    panel = pd.DataFrame(rows).merge(
        outcomes, on=["symbol", "decision_time"], how="inner", validate="one_to_one"
    )
    for horizon in (15, 60):
        panel[f"signed_log_return_{horizon}m"] = (
            panel["flow_direction"] * panel[f"log_return_{horizon}m"]
        )
    return panel.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def summarize_v2336_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in PRIMARY_FEATURES:
        for outcome in PRIMARY_OUTCOMES:
            local = panel[[feature, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "observations": len(local),
                    "spearman": float(local[feature].corr(local[outcome], method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def summarize_v2336_quartiles(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ("liq_15m_total_usd", "alt_liq_15m_total_usd"):
        low = float(panel[feature].quantile(0.25))
        high = float(panel[feature].quantile(0.75))
        for outcome in (
            "btc_abs_log_return_15m",
            "btc_log_range_15m",
            "btc_abs_log_return_60m",
            "btc_log_range_60m",
        ):
            bottom = panel.loc[panel[feature].le(low), outcome]
            top = panel.loc[panel[feature].ge(high), outcome]
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "q25": low,
                    "q75": high,
                    "bottom_observations": len(bottom),
                    "top_observations": len(top),
                    "bottom_mean": float(bottom.mean()),
                    "top_mean": float(top.mean()),
                    "top_minus_bottom": float(top.mean() - bottom.mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_v2336_own_symbol(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, local in (
        ("all", panel),
        ("alts_only", panel[panel["symbol"].ne("BTCUSDT")]),
        ("btc_only", panel[panel["symbol"].eq("BTCUSDT")]),
    ):
        directed = local[local["flow_direction"].ne(0)]
        for horizon in (15, 60):
            signed = directed[f"signed_log_return_{horizon}m"]
            rows.append(
                {
                    "scope": scope,
                    "horizon_minutes": horizon,
                    "observations": len(directed),
                    "symbols": directed["symbol"].nunique(),
                    "continuation_rate": float(signed.gt(0).mean()),
                    "mean_signed_return_bp": float(signed.mean() * 10_000),
                    "median_signed_return_bp": float(signed.median() * 10_000),
                    "notional_abs_move_spearman": float(
                        np.log1p(directed["total_usd"]).corr(
                            directed[f"abs_log_return_{horizon}m"], method="spearman"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_v2336(
    data_root: Path = DATA_ROOT,
    price_root: Path = PRICE_ROOT,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    events = load_v2334_liquidations(data_root)
    prices = load_v2336_prices(price_root)
    outcomes = build_v2336_price_outcomes(prices)
    market = build_v2336_market_panel(events, outcomes)
    own = build_v2336_own_symbol_panel(events, outcomes)
    correlations = summarize_v2336_correlations(market)
    quartiles = summarize_v2336_quartiles(market)
    own_summary = summarize_v2336_own_symbol(own)
    manifest = pd.read_csv(price_root / "manifest.csv")
    checks = {
        "price_manifest_exact_17": len(manifest) == 17,
        "all_bybit_price_downloads_succeeded": manifest["error"].isna().all()
        and manifest["bybit_kline_rows"].gt(0).all(),
        "price_symbols_exact": set(prices["symbol"]) == set(EXPECTED_SYMBOLS),
        "price_keys_unique": not prices.duplicated(
            ["symbol", "bar_open_time"]
        ).any(),
        "price_ohlc_positive_finite": np.isfinite(
            prices[["open", "high", "low", "close"]]
        ).all().all()
        and prices[["open", "high", "low", "close"]].gt(0).all().all(),
        "market_panel_has_at_least_90_decisions": len(market) >= 90,
        "own_symbol_panel_has_at_least_350_active_cells": len(own) >= 350,
        "primary_correlation_matrix_complete": len(correlations)
        == len(PRIMARY_FEATURES) * len(PRIMARY_OUTCOMES)
        and correlations["spearman"].notna().all(),
        "retrospective_only_not_promotion_evidence": True,
    }
    audit = pd.DataFrame(
        {"check": list(checks), "passed": list(checks.values())}
    )
    metadata = {
        "status": "retrospective_mechanism_pilot_not_alpha_confirmation",
        "events": len(events),
        "market_decisions": len(market),
        "own_symbol_active_cells": len(own),
        "price_symbols": prices["symbol"].nunique(),
        "outcomes_loaded": True,
        "promotion_allowed": False,
        "reason": (
            "initial liquidation snapshot was retrieved after most event outcomes; "
            "results may guide mechanism hypotheses but cannot confirm alpha"
        ),
    }
    return {
        "audit": audit,
        "market_panel": market,
        "own_symbol_panel": own,
        "correlations": correlations,
        "quartiles": quartiles,
        "own_summary": own_summary,
        "metadata": metadata,
    }


def _lookup_correlation(
    correlations: pd.DataFrame, feature: str, outcome: str
) -> float:
    return float(
        correlations.loc[
            correlations["feature"].eq(feature)
            & correlations["outcome"].eq(outcome),
            "spearman",
        ].iloc[0]
    )


def _write_findings(result: dict[str, object], path: Path) -> None:
    correlations = result["correlations"]
    quartiles = result["quartiles"]
    own = result["own_summary"]
    metadata = result["metadata"]
    total_to_range = _lookup_correlation(
        correlations, "log1p_liq_15m_total_usd", "btc_log_range_60m"
    )
    alt_to_range = _lookup_correlation(
        correlations, "log1p_alt_liq_15m_total_usd", "btc_log_range_60m"
    )
    direction_to_return = _lookup_correlation(
        correlations, "liq_15m_net_forced_buy_share", "btc_log_return_60m"
    )
    quartile = quartiles[
        quartiles["feature"].eq("liq_15m_total_usd")
        & quartiles["outcome"].eq("btc_log_range_60m")
    ].iloc[0]
    own_alt = own[
        own["scope"].eq("alts_only") & own["horizon_minutes"].eq(60)
    ].iloc[0]
    text = [
        "# v23.36 Liquidation-to-Price Mechanism Pilot",
        "",
        "Verdict: `retrospective_mechanism_pilot_not_alpha_confirmation`.",
        "",
        f"Market decisions: {metadata['market_decisions']}; active symbol-buckets: "
        f"{metadata['own_symbol_active_cells']}.",
        "",
        "Primary descriptive readings:",
        "",
        f"- log 15m total liquidation versus next-60m BTC log range: "
        f"Spearman {total_to_range:+.3f}.",
        f"- alt-only liquidation versus next-60m BTC log range: "
        f"Spearman {alt_to_range:+.3f}.",
        f"- forced-buy share versus next-60m BTC return: "
        f"Spearman {direction_to_return:+.3f}.",
        f"- top-versus-bottom liquidation quartile next-60m BTC range difference: "
        f"{float(quartile['top_minus_bottom']) * 10_000:+.2f} bp.",
        f"- alt own-symbol forced-flow continuation over 60m: "
        f"{float(own_alt['mean_signed_return_bp']):+.2f} bp mean, "
        f"{float(own_alt['continuation_rate']):.1%} positive.",
        "",
        "These results are mechanism diagnostics only. The initial OKX snapshot was "
        "retrieved after most outcomes occurred, the window is roughly one day, and "
        "15-minute observations overlap at the 60-minute horizon. No threshold, "
        "candidate, PaperLive, live, leverage, remote, application, or order state "
        "may be changed from this pilot.",
        "",
        "Confirmatory evidence must come from the v23.35 forward contract using "
        "`first_seen_at <= decision_time`.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v2336(
    data_root: Path = DATA_ROOT,
    price_root: Path = PRICE_ROOT,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
) -> dict[str, Path]:
    result = run_v2336(data_root, price_root)
    root = ensure_dir(report_root)
    paths = {
        "audit": root / "audit_checks.csv",
        "market_panel": root / "market_15m_panel.parquet",
        "own_symbol_panel": root / "own_symbol_15m_panel.parquet",
        "correlations": root / "primary_correlations.csv",
        "quartiles": root / "fixed_quartile_comparisons.csv",
        "own_summary": root / "own_symbol_summary.csv",
        "metadata": root / "metadata.json",
        "findings": findings_path,
    }
    for name in (
        "audit",
        "correlations",
        "quartiles",
        "own_summary",
    ):
        result[name].to_csv(paths[name], index=False)
    result["market_panel"].to_parquet(paths["market_panel"], index=False)
    result["own_symbol_panel"].to_parquet(paths["own_symbol_panel"], index=False)
    paths["metadata"].write_text(
        json.dumps(result["metadata"], indent=2), encoding="utf-8"
    )
    _write_findings(result, findings_path)
    return paths


__all__ = [
    "build_v2336_market_panel",
    "build_v2336_own_symbol_panel",
    "build_v2336_price_outcomes",
    "load_v2336_prices",
    "run_v2336",
    "write_v2336",
]
