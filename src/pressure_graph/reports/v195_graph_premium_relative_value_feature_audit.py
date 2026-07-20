"""Feature-only audit for slow graph-relative premium-index value states."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _month, _period
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    build_v187_monthly_risk,
)
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_panel,
)


REPORT_ROOT = Path("reports/v19_5_graph_premium_relative_value_feature_audit")
FINDINGS_PATH = Path(
    "docs/v195_graph_premium_relative_value_feature_audit_2026_07_17.md"
)
MEMBERSHIP_PATH = Path(
    "reports/v13_2_tg1_forward_temporal_extension/"
    "monthly_balanced_membership_extended.csv"
)
FUNDING_ROOTS = (
    Path("data/external/binance_um_long_history/funding"),
    Path("data/external/binance_um_carry/funding"),
    Path("data/external/recent_perp_carry/binance_funding"),
)
SCORE_FAMILIES = (
    "GLOBAL_GRAPH_PEER_PREMIUM",
    "GLOBAL_FUNDING_ORTHOGONAL_PREMIUM",
    "COMMUNITY_GRAPH_PEER_PREMIUM",
)


@dataclass(frozen=True)
class V195FeatureConfig:
    lookback_bars: int = 30 * 96
    minimum_bars: int = 20 * 96
    funding_lookback_days: int = 7
    first_entry: pd.Timestamp = pd.Timestamp("2025-08-04", tz="UTC")
    global_bucket_size: int = 8
    minimum_global_cross_section: int = 32
    minimum_community_size: int = 4
    risk_lookback_days: int = 30
    risk_min_samples: int = 2_000
    receiver_bucket_size: int = 8
    min_receiver_bucket: int = 5
    primary_cost: float = 0.0030
    stress_cost: float = 0.0040
    random_iterations: int = 500
    bootstrap_iterations: int = 2_000
    seed: int = 19_500


def load_v195_membership(path: Path = MEMBERSHIP_PATH) -> pd.DataFrame:
    membership = pd.read_csv(path)
    membership["month_start"] = pd.to_datetime(
        membership["month_start"], utc=True, errors="coerce"
    )
    return (
        membership.dropna(subset=["month_start", "community_id", "symbol"])
        .drop_duplicates(["month_start", "symbol"], keep="last")
        .sort_values(["month_start", "community_id", "symbol"])
        .reset_index(drop=True)
    )


def load_v195_funding(
    symbols: list[str] | set[str],
    roots: tuple[Path, ...] = FUNDING_ROOTS,
) -> pd.DataFrame:
    frames = []
    for symbol in sorted(set(symbols)):
        for root in roots:
            path = root / f"{symbol}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(
                path, columns=["funding_time", "funding_rate_settled"]
            )
            frame["symbol"] = symbol
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "funding_time", "funding_rate_settled"])
    funding = pd.concat(frames, ignore_index=True)
    funding["funding_time"] = pd.to_datetime(
        funding["funding_time"], utc=True, errors="coerce"
    )
    funding["funding_rate_settled"] = pd.to_numeric(
        funding["funding_rate_settled"], errors="coerce"
    )
    return (
        funding.dropna(subset=["symbol", "funding_time", "funding_rate_settled"])
        .drop_duplicates(["symbol", "funding_time"], keep="last")
        .sort_values(["funding_time", "symbol"])
        .reset_index(drop=True)
    )


def build_v195_premium_z(
    premium: pd.DataFrame,
    close: pd.DataFrame,
    cfg: V195FeatureConfig = V195FeatureConfig(),
) -> pd.DataFrame:
    exact = premium.reindex(index=close.index, columns=close.columns)
    prior_mean = (
        exact.shift(1)
        .rolling(cfg.lookback_bars, min_periods=cfg.minimum_bars)
        .mean()
    )
    prior_scale = (
        exact.shift(1)
        .rolling(cfg.lookback_bars, min_periods=cfg.minimum_bars)
        .std(ddof=1)
    )
    return (exact - prior_mean).div(prior_scale.where(prior_scale.gt(0)))


def _funding_lookup(
    funding: pd.DataFrame,
    entries: pd.DatetimeIndex,
    lookback_days: int,
) -> dict[tuple[pd.Timestamp, str], float]:
    lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for symbol, local in funding.groupby("symbol", sort=True):
        time = pd.DatetimeIndex(local["funding_time"])
        values = local["funding_rate_settled"].to_numpy(dtype=float)
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        raw = time.view("int64")
        for entry in entries:
            right = int(np.searchsorted(raw, entry.value, side="left"))
            left_time = entry - pd.Timedelta(days=lookback_days)
            left = int(np.searchsorted(raw, left_time.value, side="left"))
            lookup[(entry, str(symbol))] = float(cumulative[right] - cumulative[left])
    return lookup


def _funding_orthogonal_residual(
    peer_score: pd.Series, funding_score: pd.Series
) -> pd.Series:
    valid = peer_score.notna() & funding_score.notna()
    output = pd.Series(np.nan, index=peer_score.index, dtype=float)
    if valid.sum() < 3 or funding_score[valid].var(ddof=1) <= 0:
        output.loc[valid] = peer_score.loc[valid] - peer_score.loc[valid].mean()
        return output
    x = np.column_stack(
        [np.ones(int(valid.sum())), funding_score.loc[valid].to_numpy(dtype=float)]
    )
    y = peer_score.loc[valid].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    output.loc[valid] = y - x @ coefficients
    return output


def _neutralize_weights(
    raw: dict[str, float], beta: pd.Series
) -> dict[str, float]:
    hedge = -float(
        sum(weight * float(beta.get(symbol, np.nan)) for symbol, weight in raw.items())
    )
    if not np.isfinite(hedge):
        return {}
    gross = float(sum(abs(weight) for weight in raw.values()) + abs(hedge))
    if not np.isfinite(gross) or gross <= 0:
        return {}
    weights = {symbol: weight / gross for symbol, weight in raw.items()}
    weights[BTC] = hedge / gross
    return weights


def _global_target(
    local: pd.DataFrame, score_column: str, cfg: V195FeatureConfig
) -> tuple[dict[str, float], list[str], list[str]]:
    ranked = local.dropna(subset=[score_column, "btc_beta"]).sort_values(
        [score_column, "symbol"]
    )
    size = cfg.global_bucket_size
    if len(ranked) < cfg.minimum_global_cross_section or len(ranked) < 2 * size:
        return {}, [], []
    long_names = ranked.head(size)["symbol"].astype(str).tolist()
    short_names = ranked.tail(size)["symbol"].astype(str).tolist()
    if set(long_names) & set(short_names):
        return {}, [], []
    raw = {symbol: 0.5 / size for symbol in long_names}
    raw.update({symbol: -0.5 / size for symbol in short_names})
    weights = _neutralize_weights(raw, ranked.set_index("symbol")["btc_beta"])
    return weights, long_names, short_names


def _community_target(
    local: pd.DataFrame, cfg: V195FeatureConfig
) -> tuple[dict[str, float], list[str], list[str]]:
    pairs: list[tuple[str, str]] = []
    for _, group in local.groupby("community_id", sort=True):
        ranked = group.dropna(subset=["peer_premium_z", "btc_beta"]).sort_values(
            ["peer_premium_z", "symbol"]
        )
        if len(ranked) < cfg.minimum_community_size:
            continue
        low = str(ranked.iloc[0]["symbol"])
        high = str(ranked.iloc[-1]["symbol"])
        if low != high:
            pairs.append((low, high))
    if not pairs:
        return {}, [], []
    pair_weight = 0.5 / len(pairs)
    raw: dict[str, float] = {}
    for low, high in pairs:
        raw[low] = raw.get(low, 0.0) + pair_weight
        raw[high] = raw.get(high, 0.0) - pair_weight
    beta = local.drop_duplicates("symbol").set_index("symbol")["btc_beta"]
    weights = _neutralize_weights(raw, beta)
    return weights, [low for low, _ in pairs], [high for _, high in pairs]


def _turnover(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        sum(
            abs(left.get(symbol, 0.0) - right.get(symbol, 0.0))
            for symbol in set(left) | set(right)
        )
    )


def build_v195_feature_panel_and_targets(
    close: pd.DataFrame,
    premium: pd.DataFrame,
    membership: pd.DataFrame,
    funding: pd.DataFrame,
    cfg: V195FeatureConfig = V195FeatureConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    premium_z = build_v195_premium_z(premium, close, cfg)
    last_entry = min(close.index.max(), premium.index.max()).floor("D")
    entries = pd.date_range(cfg.first_entry, last_entry, freq="1D", tz="UTC")
    funding_scores = _funding_lookup(funding, entries, cfg.funding_lookback_days)
    returns = close.pct_change(fill_method=None)
    risk_cfg = cfg
    risk = build_v187_monthly_risk(
        returns, entries.min(), entries.max(), risk_cfg  # type: ignore[arg-type]
    )
    beta_lookup = risk.set_index(["risk_month", "receiver"])["btc_beta"]

    feature_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for entry in entries:
        if entry not in premium_z.index:
            continue
        month = _month(entry)
        local_membership = membership[membership["month_start"].eq(month)]
        if local_membership.empty:
            continue
        local_rows = []
        for item in local_membership.itertuples(index=False):
            symbol = str(item.symbol)
            if symbol == BTC or symbol not in premium_z.columns:
                continue
            own_z = float(premium_z.at[entry, symbol])
            beta = beta_lookup.get((month, symbol), np.nan)
            funding_7d = funding_scores.get((entry, symbol), np.nan)
            if not np.isfinite(own_z) or not np.isfinite(beta) or not np.isfinite(
                funding_7d
            ):
                continue
            local_rows.append(
                {
                    "entry_time": entry,
                    "month_start": month,
                    "period": _period(entry),
                    "community_id": str(item.community_id),
                    "symbol": symbol,
                    "premium_z": own_z,
                    "funding_7d": float(funding_7d),
                    "btc_beta": float(beta),
                }
            )
        local = pd.DataFrame(local_rows)
        if local.empty:
            continue
        local["community_premium_median_z"] = local.groupby(
            "community_id"
        )["premium_z"].transform("median")
        local["peer_premium_z"] = (
            local["premium_z"] - local["community_premium_median_z"]
        )
        funding_std = float(local["funding_7d"].std(ddof=1))
        local["funding_7d_z"] = (
            (local["funding_7d"] - local["funding_7d"].mean()) / funding_std
            if funding_std > 0
            else 0.0
        )
        local["funding_orthogonal_premium_z"] = _funding_orthogonal_residual(
            local["peer_premium_z"], local["funding_7d_z"]
        )
        feature_rows.extend(local.to_dict("records"))
        correlation_rows.append(
            {
                "entry_time": entry,
                "period": _period(entry),
                "symbols": len(local),
                "communities": local["community_id"].nunique(),
                "peer_premium_funding_correlation": local[
                    ["peer_premium_z", "funding_7d_z"]
                ].corr().iloc[0, 1],
                "orthogonal_premium_funding_correlation": local[
                    ["funding_orthogonal_premium_z", "funding_7d_z"]
                ].corr().iloc[0, 1],
                "peer_premium_dispersion": local["peer_premium_z"].std(ddof=1),
                "orthogonal_premium_dispersion": local[
                    "funding_orthogonal_premium_z"
                ].std(ddof=1),
            }
        )
        targets = {
            "GLOBAL_GRAPH_PEER_PREMIUM": _global_target(
                local, "peer_premium_z", cfg
            ),
            "GLOBAL_FUNDING_ORTHOGONAL_PREMIUM": _global_target(
                local, "funding_orthogonal_premium_z", cfg
            ),
            "COMMUNITY_GRAPH_PEER_PREMIUM": _community_target(local, cfg),
        }
        funding_sign = -np.sign(local.set_index("symbol")["funding_7d"])
        for family, (weights, long_names, short_names) in targets.items():
            if not weights:
                continue
            alt_weights = {symbol: weight for symbol, weight in weights.items() if symbol != BTC}
            aligned = [
                np.sign(weight) == funding_sign.get(symbol, 0.0)
                for symbol, weight in alt_weights.items()
                if funding_sign.get(symbol, 0.0) != 0
            ]
            target_rows.append(
                {
                    "entry_time": entry,
                    "period": _period(entry),
                    "family": family,
                    "eligible_symbols": len(local),
                    "eligible_communities": local["community_id"].nunique(),
                    "long_symbols": "|".join(sorted(long_names)),
                    "short_symbols": "|".join(sorted(short_names)),
                    "long_count": len(long_names),
                    "short_count": len(short_names),
                    "funding_sign_alignment": float(np.mean(aligned)) if aligned else np.nan,
                    "residual_btc_beta": float(
                        sum(
                            weight * local.set_index("symbol")["btc_beta"].get(symbol, 0.0)
                            for symbol, weight in alt_weights.items()
                        )
                        + weights.get(BTC, 0.0)
                    ),
                    "gross_notional": float(sum(abs(weight) for weight in weights.values())),
                    "weights": weights,
                }
            )
    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(target_rows),
        pd.DataFrame(correlation_rows),
    )


def summarize_v195_target_coverage(
    targets: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for family in SCORE_FAMILIES:
        daily = targets[targets["family"].eq(family)].sort_values("entry_time")
        for cadence_days in (1, 2, 7):
            sample = daily.iloc[::cadence_days].copy()
            if sample.empty:
                continue
            previous: dict[str, float] = {}
            turnovers = []
            long_jaccard = []
            short_jaccard = []
            previous_long: set[str] | None = None
            previous_short: set[str] | None = None
            for row in sample.itertuples(index=False):
                current = dict(row.weights)
                turnovers.append(_turnover(previous, current))
                current_long = set(str(row.long_symbols).split("|"))
                current_short = set(str(row.short_symbols).split("|"))
                if previous_long is not None and previous_short is not None:
                    long_jaccard.append(
                        len(previous_long & current_long)
                        / max(1, len(previous_long | current_long))
                    )
                    short_jaccard.append(
                        len(previous_short & current_short)
                        / max(1, len(previous_short | current_short))
                    )
                previous = current
                previous_long = current_long
                previous_short = current_short
            rows.append(
                {
                    "family": family,
                    "cadence_days": cadence_days,
                    "decisions": len(sample),
                    "development_decisions": int(
                        sample["period"].eq("development").sum()
                    ),
                    "validation_decisions": int(
                        sample["period"].eq("validation").sum()
                    ),
                    "holdout_decisions": int(sample["period"].eq("holdout").sum()),
                    "active_months": sample["entry_time"].dt.strftime("%Y-%m").nunique(),
                    "median_eligible_symbols": float(sample["eligible_symbols"].median()),
                    "median_eligible_communities": float(
                        sample["eligible_communities"].median()
                    ),
                    "mean_target_turnover": float(np.mean(turnovers)),
                    "median_target_turnover": float(np.median(turnovers)),
                    "mean_long_jaccard": float(np.mean(long_jaccard)),
                    "mean_short_jaccard": float(np.mean(short_jaccard)),
                    "mean_funding_sign_alignment": float(
                        sample["funding_sign_alignment"].mean()
                    ),
                    "max_abs_residual_btc_beta": float(
                        sample["residual_btc_beta"].abs().max()
                    ),
                    "max_gross_notional_drift": float(
                        (sample["gross_notional"] - 1.0).abs().max()
                    ),
                    "median_peer_premium_funding_correlation": float(
                        correlations["peer_premium_funding_correlation"].median()
                    ),
                    "median_orthogonal_premium_funding_correlation": float(
                        correlations[
                            "orthogonal_premium_funding_correlation"
                        ].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_findings(summary: pd.DataFrame, path: Path) -> None:
    text = [
        "# v19.5 Graph-Premium Relative-Value Feature-Only Audit",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Premium level z-scores use shifted prior-30-day moments. Funding scores",
        "use Binance settlements strictly before each decision. Graph peer medians,",
        "funding orthogonalization, target weights, and turnover use only as-of data.",
        "No future price or funding return was calculated or inspected.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v195_feature_audit(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    membership_path: Path = MEMBERSHIP_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V195FeatureConfig = V195FeatureConfig(),
) -> dict[str, Path]:
    close, _ = load_v184_exact_panels(metrics_root, kline_root)
    premium = load_v190_premium_panel(premium_root)
    membership = load_v195_membership(membership_path)
    symbols = sorted(set(close.columns) & set(membership["symbol"]))
    funding = load_v195_funding(symbols)
    feature_panel, targets, correlations = build_v195_feature_panel_and_targets(
        close, premium, membership, funding, cfg
    )
    summary = summarize_v195_target_coverage(targets, correlations)
    root = ensure_dir(report_root)
    outputs = {
        "features": root / "daily_symbol_feature_panel.parquet",
        "targets": root / "daily_target_weights.parquet",
        "correlations": root / "daily_score_correlations.csv",
        "summary": root / "target_coverage_summary.csv",
        "findings": findings_path,
    }
    feature_panel.to_parquet(outputs["features"], index=False)
    target_output = targets.copy()
    target_output["weights"] = target_output["weights"].map(
        lambda value: "|".join(f"{key}:{value[key]:.12g}" for key in sorted(value))
    )
    target_output.to_parquet(outputs["targets"], index=False)
    correlations.to_csv(outputs["correlations"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    _write_findings(summary, findings_path)
    return outputs


__all__ = [
    "FUNDING_ROOTS",
    "MEMBERSHIP_PATH",
    "SCORE_FAMILIES",
    "V195FeatureConfig",
    "build_v195_feature_panel_and_targets",
    "build_v195_premium_z",
    "load_v195_funding",
    "load_v195_membership",
    "summarize_v195_target_coverage",
    "write_v195_feature_audit",
]
