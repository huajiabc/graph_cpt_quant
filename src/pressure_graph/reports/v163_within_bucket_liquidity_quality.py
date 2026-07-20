"""Weekly within-bucket relative value from displayed-depth liquidity quality."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
    beta_neutral_v155_weights,
    load_v155_hourly_prices,
)
from pressure_graph.reports.v159_hourly_cross_venue_depth_imbalance import (
    V159Config,
    rolling_v159_betas,
)


FEATURE_ROOT = Path("data/external/binance_um_book_depth/hourly_features")
BINANCE_ROOT = Path("data/external/binance_um_carry/klines_1h")
RECENT_BINANCE_ROOT = Path("data/external/recent_perp_carry/binance_klines_1h")
REPORT_ROOT = Path("reports/v16_3_within_bucket_liquidity_quality")
FINDINGS_PATH = Path("docs/v163_within_bucket_liquidity_quality_findings_2026_07_16.md")
CANDIDATE = "LQ1_WITHIN_BUCKET_LIQUIDITY_QUALITY"
REVERSED_CONTROL = "LQ1_REVERSED_LIQUIDITY_FRAGILITY_PREMIUM"
STALE_CONTROL = "LQ1_ONE_WEEK_STALE_QUALITY"
RAW_DEPTH_CONTROL = "LQ1_RAW_DEPTH_ONLY"
FROZEN_PAIRS = {
    "BSP01": ("SOLUSDT", "DOGEUSDT"),
    "BSP02": ("1000PEPEUSDT", "WIFUSDT"),
    "BSP03": ("ETHUSDT", "ENAUSDT"),
    "BSP04": ("HBARUSDT", "AVAXUSDT"),
    "BSP05": ("LINKUSDT", "ONDOUSDT"),
    "BSP06": ("XRPUSDT", "XLMUSDT"),
    "BSP07": ("FARTCOINUSDT", "WLDUSDT"),
    "BSP08": ("SEIUSDT", "TIAUSDT"),
}


@dataclass(frozen=True)
class V163Config:
    feature_root: Path = FEATURE_ROOT
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_weekly_hours: int = 150
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    random_iterations: int = 1000
    bootstrap_iterations: int = 5000
    bootstrap_block_weeks: int = 4
    seed: int = 20260726


def load_v163_depth(root: Path = FEATURE_ROOT) -> pd.DataFrame:
    columns = [
        "decision_time",
        "symbol",
        "total_notional_1p0_median",
        "total_notional_5p0_median",
    ]
    frames = [
        pd.read_parquet(root / f"{symbol}.parquet", columns=columns)
        for symbol in FROZEN_SYMBOLS
    ]
    depth = pd.concat(frames, ignore_index=True)
    depth["decision_time"] = pd.to_datetime(depth["decision_time"], utc=True)
    return depth.sort_values(["decision_time", "symbol"]).reset_index(drop=True)


def load_v163_binance_volume() -> pd.DataFrame:
    frames = []
    for symbol in FROZEN_SYMBOLS:
        for root in (BINANCE_ROOT, RECENT_BINANCE_ROOT):
            path = root / f"{symbol}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path, columns=["feature_time", "quote_volume"])
            frame["symbol"] = symbol
            frames.append(frame)
    volume = pd.concat(frames, ignore_index=True)
    volume["feature_time"] = pd.to_datetime(volume["feature_time"], utc=True)
    volume["quote_volume"] = pd.to_numeric(volume["quote_volume"], errors="coerce")
    return (
        volume.dropna(subset=["feature_time", "symbol", "quote_volume"])
        .drop_duplicates(["feature_time", "symbol"], keep="last")
        .sort_values(["feature_time", "symbol"])
        .reset_index(drop=True)
    )


def build_v163_weekly_panel(
    depth: pd.DataFrame,
    volume: pd.DataFrame,
    hourly_prices: pd.DataFrame,
    cfg: V163Config = V163Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = hourly_prices.pivot_table(
        index="feature_time",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=True,
    ).sort_index()
    hourly_return = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    betas = rolling_v159_betas(hourly_return, V159Config())
    required = list(FROZEN_SYMBOLS) + [BTC]
    mondays = pd.date_range(
        "2025-07-07", "2026-07-13", freq="W-MON", tz="UTC"
    )
    feature_rows = []
    coverage_rows = []
    for decision in mondays:
        start = decision - pd.Timedelta(days=7)
        local_depth = depth[
            depth["decision_time"].gt(start) & depth["decision_time"].le(decision)
        ]
        local_volume = volume[
            volume["feature_time"].gt(start) & volume["feature_time"].le(decision)
        ]
        depth_summary = local_depth.groupby("symbol", observed=True).agg(
            depth_hours=("total_notional_1p0_median", "count"),
            depth_1pct=("total_notional_1p0_median", "median"),
            depth_5pct=("total_notional_5p0_median", "median"),
        )
        volume_summary = local_volume.groupby("symbol", observed=True).agg(
            volume_hours=("quote_volume", "count"),
            mean_hourly_quote_volume=("quote_volume", "mean"),
        )
        weekly = depth_summary.join(volume_summary, how="outer")
        exact_universe = set(weekly.index.astype(str)) == set(FROZEN_SYMBOLS)
        hours_ready = bool(
            exact_universe
            and weekly[["depth_hours", "volume_hours"]]
            .ge(cfg.minimum_weekly_hours)
            .all()
            .all()
        )
        price_ready = bool(
            decision in close.index
            and decision + pd.Timedelta(days=7) in close.index
            and close.loc[decision, required].notna().all()
            and close.loc[decision + pd.Timedelta(days=7), required].notna().all()
        )
        beta_ready = bool(
            decision in betas.index
            and betas.loc[decision, list(FROZEN_SYMBOLS)].notna().all()
        )
        usable = exact_universe and hours_ready and price_ready and beta_ready
        coverage_rows.append(
            {
                "decision_time": decision,
                "symbols": len(weekly),
                "minimum_depth_hours": int(weekly["depth_hours"].min())
                if len(weekly)
                else 0,
                "minimum_volume_hours": int(weekly["volume_hours"].min())
                if len(weekly)
                else 0,
                "exact_universe": exact_universe,
                "hours_ready": hours_ready,
                "price_ready": price_ready,
                "beta_ready": beta_ready,
                "usable": usable,
            }
        )
        if not usable:
            continue
        future_return = close.loc[decision + pd.Timedelta(days=7), required] / close.loc[
            decision, required
        ] - 1.0
        period = (
            "development"
            if decision <= pd.Timestamp("2025-12-29", tz="UTC")
            else "validation"
            if decision <= pd.Timestamp("2026-03-30", tz="UTC")
            else "holdout"
        )
        for symbol in FROZEN_SYMBOLS:
            row = weekly.loc[symbol]
            quality_1pct = float(np.log(row["depth_1pct"] / row["mean_hourly_quote_volume"]))
            quality_5pct = float(np.log(row["depth_5pct"] / row["mean_hourly_quote_volume"]))
            feature_rows.append(
                {
                    "decision_time": decision,
                    "period": period,
                    "symbol": symbol,
                    "depth_hours": int(row["depth_hours"]),
                    "volume_hours": int(row["volume_hours"]),
                    "depth_1pct": float(row["depth_1pct"]),
                    "depth_5pct": float(row["depth_5pct"]),
                    "mean_hourly_quote_volume": float(row["mean_hourly_quote_volume"]),
                    "quality_1pct": quality_1pct,
                    "quality_5pct": quality_5pct,
                    "btc_beta": float(betas.at[decision, symbol]),
                    "price_return": float(future_return[symbol]),
                    "btc_return": float(future_return[BTC]),
                }
            )
    panel = pd.DataFrame(feature_rows)
    if not panel.empty:
        stale = panel[["decision_time", "symbol", "quality_1pct"]].copy()
        stale["decision_time"] += pd.Timedelta(days=7)
        stale = stale.rename(columns={"quality_1pct": "stale_quality_1pct"})
        panel = panel.merge(
            stale,
            on=["decision_time", "symbol"],
            how="left",
            validate="one_to_one",
        )
    return (
        panel.sort_values(["decision_time", "symbol"]).reset_index(drop=True),
        pd.DataFrame(coverage_rows).sort_values("decision_time").reset_index(drop=True),
    )


def select_v163_pairs(
    local: pd.DataFrame,
    score_column: str,
    *,
    reverse: bool = False,
) -> tuple[list[str], list[str]]:
    indexed = local.set_index("symbol")
    longs = []
    shorts = []
    for pair in FROZEN_PAIRS.values():
        ordered = sorted(
            pair,
            key=lambda symbol: (-float(indexed.at[symbol, score_column]), symbol),
        )
        high, low = ordered
        if reverse:
            high, low = low, high
        longs.append(high)
        shorts.append(low)
    return longs, shorts


def _turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    return float(
        sum(
            abs(previous.get(symbol, 0.0) - current.get(symbol, 0.0))
            for symbol in set(previous) | set(current)
        )
    )


def build_v163_portfolio(
    panel: pd.DataFrame,
    cfg: V163Config = V163Config(),
    *,
    score_column: str = "quality_1pct",
    candidate: str = CANDIDATE,
    reverse: bool = False,
) -> pd.DataFrame:
    rows = []
    previous_weights: dict[str, float] = {}
    previous_time: pd.Timestamp | None = None
    for raw_time, local in panel.groupby("decision_time", sort=True, observed=True):
        decision = pd.Timestamp(raw_time)
        if previous_time is not None and decision - previous_time > pd.Timedelta(days=7):
            forced_close = float(sum(abs(weight) for weight in previous_weights.values()))
            rows[-1]["forced_close_turnover"] = forced_close
            rows[-1]["realized_turnover"] = float(rows[-1]["entry_turnover"]) + forced_close
            rows[-1]["primary_net_return"] = float(rows[-1]["gross_return"]) - (
                cfg.one_way_cost * float(rows[-1]["realized_turnover"])
            )
            rows[-1]["stress_net_return"] = float(rows[-1]["gross_return"]) - (
                cfg.stress_one_way_cost * float(rows[-1]["realized_turnover"])
            )
            previous_weights = {}
        local = local.sort_values("symbol").reset_index(drop=True)
        longs, shorts = select_v163_pairs(local, score_column, reverse=reverse)
        betas = local.set_index("symbol")["btc_beta"].astype(float).to_dict()
        weights = beta_neutral_v155_weights(longs, shorts, betas)
        indexed = local.set_index("symbol")
        gross_return = float(
            sum(
                weights[symbol] * float(indexed.at[symbol, "price_return"])
                for symbol in longs + shorts
            )
            + weights[BTC] * float(local.iloc[0]["btc_return"])
        )
        residual_beta = float(
            sum(weights[symbol] * betas[symbol] for symbol in longs + shorts)
            + weights[BTC]
        )
        gross_notional = float(sum(abs(weight) for weight in weights.values()))
        entry_turnover = _turnover(previous_weights, weights)
        rows.append(
            {
                "candidate": candidate,
                "decision_time": decision,
                "period": local.iloc[0]["period"],
                "long_symbols": "|".join(longs),
                "short_symbols": "|".join(shorts),
                "weights_json": json.dumps(weights, sort_keys=True),
                "btc_hedge_weight": weights[BTC],
                "entry_turnover": entry_turnover,
                "forced_close_turnover": 0.0,
                "realized_turnover": entry_turnover,
                "gross_notional": gross_notional,
                "residual_btc_beta": residual_beta,
                "gross_return": gross_return,
                "primary_net_return": gross_return - cfg.one_way_cost * entry_turnover,
                "stress_net_return": gross_return
                - cfg.stress_one_way_cost * entry_turnover,
            }
        )
        previous_weights = weights
        previous_time = decision
    return pd.DataFrame(rows)


def build_v163_random_controls(
    panel: pd.DataFrame,
    cfg: V163Config = V163Config(),
) -> pd.DataFrame:
    rows = []
    grouped = [
        (pd.Timestamp(time), local.sort_values("symbol").reset_index(drop=True))
        for time, local in panel.groupby("decision_time", sort=True, observed=True)
    ]
    for iteration in range(cfg.random_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        previous_weights: dict[str, float] = {}
        total_net = 0.0
        total_turnover = 0.0
        previous_time: pd.Timestamp | None = None
        for decision, local in grouped:
            if previous_time is not None and decision - previous_time > pd.Timedelta(days=7):
                forced_close = float(sum(abs(weight) for weight in previous_weights.values()))
                total_net -= cfg.one_way_cost * forced_close
                total_turnover += forced_close
                previous_weights = {}
            shuffled = local.copy()
            shuffled["random_quality"] = rng.permutation(
                shuffled["quality_1pct"].to_numpy(dtype=float)
            )
            longs, shorts = select_v163_pairs(shuffled, "random_quality")
            betas = local.set_index("symbol")["btc_beta"].astype(float).to_dict()
            weights = beta_neutral_v155_weights(longs, shorts, betas)
            indexed = local.set_index("symbol")
            gross_return = float(
                sum(
                    weights[symbol] * float(indexed.at[symbol, "price_return"])
                    for symbol in longs + shorts
                )
                + weights[BTC] * float(local.iloc[0]["btc_return"])
            )
            turnover = _turnover(previous_weights, weights)
            total_net += gross_return - cfg.one_way_cost * turnover
            total_turnover += turnover
            previous_weights = weights
            previous_time = decision
        rows.append(
            {
                "iteration": iteration,
                "mean_primary_net_return": total_net / len(grouped),
                "mean_turnover": total_turnover / len(grouped),
            }
        )
    return pd.DataFrame(rows)


def _moving_block_bootstrap(values: np.ndarray, cfg: V163Config) -> tuple[float, float]:
    rng = np.random.default_rng(cfg.seed + 2)
    offsets = np.arange(cfg.bootstrap_block_weeks)
    block_count = int(np.ceil(len(values) / cfg.bootstrap_block_weeks))
    draws = np.empty(cfg.bootstrap_iterations, dtype=float)
    for iteration in range(cfg.bootstrap_iterations):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draws[iteration] = float(values[indices.ravel()[: len(values)]].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_v163(
    portfolio: pd.DataFrame,
    reversed_control: pd.DataFrame,
    stale_control: pd.DataFrame,
    raw_depth_control: pd.DataFrame,
    random_controls: pd.DataFrame,
    cfg: V163Config = V163Config(),
) -> pd.DataFrame:
    bootstrap_low, bootstrap_high = _moving_block_bootstrap(
        portfolio["primary_net_return"].to_numpy(dtype=float), cfg
    )
    periods = portfolio.groupby("period", observed=True)["primary_net_return"].mean()
    counts = portfolio["period"].value_counts()
    monthly = portfolio.assign(month=portfolio["decision_time"].dt.strftime("%Y-%m")).groupby(
        "month", observed=True
    )["primary_net_return"].sum()
    positive = monthly[monthly.gt(0)]
    concentration = float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    observed_mean = float(portfolio["primary_net_return"].mean())
    stale_times = set(stale_control["decision_time"])
    primary_stale_overlap = portfolio[portfolio["decision_time"].isin(stale_times)]
    row = {
        "candidate": CANDIDATE,
        "weeks": len(portfolio),
        "months": portfolio["decision_time"].dt.strftime("%Y-%m").nunique(),
        "validation_weeks": int(counts.get("validation", 0)),
        "holdout_weeks": int(counts.get("holdout", 0)),
        "mean_gross_bp": portfolio["gross_return"].mean() * 10_000,
        "mean_primary_net_bp": observed_mean * 10_000,
        "mean_stress_net_bp": portfolio["stress_net_return"].mean() * 10_000,
        "development_primary_net_bp": periods.get("development", np.nan) * 10_000,
        "validation_primary_net_bp": periods.get("validation", np.nan) * 10_000,
        "holdout_primary_net_bp": periods.get("holdout", np.nan) * 10_000,
        "bootstrap_95_low_bp": bootstrap_low * 10_000,
        "bootstrap_95_high_bp": bootstrap_high * 10_000,
        "random_quality_pairing_percentile": 100
        * random_controls["mean_primary_net_return"].le(observed_mean).mean(),
        "positive_month_concentration": concentration,
        "mean_one_way_turnover": portfolio["realized_turnover"].mean(),
        "reversed_control_mean_bp": reversed_control["primary_net_return"].mean()
        * 10_000,
        "primary_stale_overlap_mean_bp": primary_stale_overlap[
            "primary_net_return"
        ].mean()
        * 10_000,
        "stale_control_mean_bp": stale_control["primary_net_return"].mean() * 10_000,
        "raw_depth_control_mean_bp": raw_depth_control["primary_net_return"].mean()
        * 10_000,
        "max_abs_residual_btc_beta": portfolio["residual_btc_beta"].abs().max(),
        "max_gross_notional_drift": (portfolio["gross_notional"] - 1.0).abs().max(),
    }
    row["promote"] = bool(
        row["weeks"] >= 48
        and row["months"] >= 11
        and row["validation_weeks"] >= 12
        and row["holdout_weeks"] >= 13
        and all(
            row[key] > 0
            for key in (
                "mean_primary_net_bp",
                "mean_stress_net_bp",
                "development_primary_net_bp",
                "validation_primary_net_bp",
                "holdout_primary_net_bp",
                "bootstrap_95_low_bp",
            )
        )
        and row["random_quality_pairing_percentile"] >= 99
        and row["positive_month_concentration"] <= 0.35
        and row["mean_one_way_turnover"] <= 0.35
        and row["mean_primary_net_bp"] > row["reversed_control_mean_bp"]
        and row["primary_stale_overlap_mean_bp"] > row["stale_control_mean_bp"]
        and row["mean_primary_net_bp"] > row["raw_depth_control_mean_bp"]
        and row["max_abs_residual_btc_beta"] <= 1e-10
        and row["max_gross_notional_drift"] <= 1e-10
    )
    return pd.DataFrame([row])


def write_v163_within_bucket_liquidity_quality(
    cfg: V163Config = V163Config(),
) -> dict[str, Path]:
    panel, coverage = build_v163_weekly_panel(
        load_v163_depth(cfg.feature_root),
        load_v163_binance_volume(),
        load_v155_hourly_prices(),
        cfg,
    )
    portfolio = build_v163_portfolio(panel, cfg)
    reversed_control = build_v163_portfolio(
        panel, cfg, candidate=REVERSED_CONTROL, reverse=True
    )
    stale_ready = panel.groupby("decision_time", observed=True)["stale_quality_1pct"].count()
    stale_times = stale_ready.index[stale_ready.eq(len(FROZEN_SYMBOLS))]
    stale_panel = panel[panel["decision_time"].isin(stale_times)]
    stale_control = build_v163_portfolio(
        stale_panel, cfg, score_column="stale_quality_1pct", candidate=STALE_CONTROL
    )
    raw_depth_control = build_v163_portfolio(
        panel, cfg, score_column="depth_1pct", candidate=RAW_DEPTH_CONTROL
    )
    diagnostic_5pct = build_v163_portfolio(
        panel, cfg, score_column="quality_5pct", candidate="LQ1_5PCT_DIAGNOSTIC_ONLY"
    )
    random_controls = build_v163_random_controls(panel, cfg)
    summary = summarize_v163(
        portfolio,
        reversed_control,
        stale_control,
        raw_depth_control,
        random_controls,
        cfg,
    )
    controls = pd.DataFrame(
        [
            {
                "control": name,
                "weeks": len(frame),
                "mean_primary_net_bp": frame["primary_net_return"].mean() * 10_000,
            }
            for name, frame in (
                (REVERSED_CONTROL, reversed_control),
                (STALE_CONTROL, stale_control),
                (RAW_DEPTH_CONTROL, raw_depth_control),
                ("LQ1_5PCT_DIAGNOSTIC_ONLY_NON_PROMOTABLE", diagnostic_5pct),
            )
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "panel": root / "weekly_symbol_panel.parquet",
        "coverage": root / "coverage.csv",
        "portfolio": root / "weekly_portfolio.parquet",
        "reversed": root / "reversed_control.parquet",
        "stale": root / "stale_control.parquet",
        "raw_depth": root / "raw_depth_control.parquet",
        "diagnostic_5pct": root / "five_percent_diagnostic.parquet",
        "random": root / "random_quality_pairings.csv",
        "summary": root / "summary.csv",
        "controls": root / "control_summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    panel.to_parquet(paths["panel"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    portfolio.to_parquet(paths["portfolio"], index=False)
    reversed_control.to_parquet(paths["reversed"], index=False)
    stale_control.to_parquet(paths["stale"], index=False)
    raw_depth_control.to_parquet(paths["raw_depth"], index=False)
    diagnostic_5pct.to_parquet(paths["diagnostic_5pct"], index=False)
    random_controls.to_csv(paths["random"], index=False)
    summary.to_csv(paths["summary"], index=False)
    controls.to_csv(paths["controls"], index=False)
    promoted = summary.loc[summary["promote"], "candidate"].tolist()
    serialized_config = {
        **asdict(cfg),
        "feature_root": str(cfg.feature_root),
        "report_root": str(cfg.report_root),
        "findings_path": str(cfg.findings_path),
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "promoted": promoted,
                "config": serialized_config,
                "frozen_pairs": FROZEN_PAIRS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = "promote_forward_shadow_candidate" if promoted else "reject_candidate"
    paths["findings"].write_text(
        "\n".join(
            [
                "# v16.3 Within-Bucket Liquidity Quality Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "## Frozen controls",
                "",
                controls.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The graph pairs, seven-day depth/volume ratio, direction, beta hedge,",
                "costs and gates were frozen before returns. The 5% row is diagnostic-only.",
                "PaperLive and remote state are unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
