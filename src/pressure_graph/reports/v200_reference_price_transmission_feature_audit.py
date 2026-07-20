"""Feature-only audit for graph-bucket reference-price transmission states."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v178_btc_confirmed_flow_laggard import _period
from pressure_graph.reports.v184_btc_inclusive_metrics_audit import (
    KLINE_ROOT,
    METRICS_ROOT,
    load_v184_exact_panels,
)
from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v190_binance_premium_index_audit import (
    PREMIUM_ROOT,
    load_v190_premium_panel,
)
from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    MEMBERSHIP_PATH,
    load_v195_membership,
)
from pressure_graph.reports.v199_binance_reference_price_audit import (
    INDEX_ROOT,
    MARK_ROOT,
    load_reference_ohlc_panels,
)


REPORT_ROOT = Path("reports/v20_0_reference_price_transmission_feature_audit")
FINDINGS_PATH = Path(
    "docs/v200_reference_price_transmission_feature_audit_2026_07_17.md"
)
REFERENCE_LAG = "REFERENCE_RESIDUAL_INDEX_LEAD_CATCHUP"
TRADE_OVERSHOOT = "TRADE_VS_MARK_OVERSHOOT_FADE"
FAMILIES = (REFERENCE_LAG, TRADE_OVERSHOOT)
GLOBAL = "GLOBAL_BTC_INDEX_SHOCK"
COMMUNITY = "COMMUNITY_COHERENT_INDEX_SHOCK"


@dataclass(frozen=True)
class V200FeatureConfig:
    lookback_bars: int = 30 * 96
    minimum_bars: int = 20 * 96
    global_source_quantiles: tuple[float, ...] = (0.85, 0.90)
    community_source_z: tuple[float, ...] = (1.5, 2.0)
    receiver_z_thresholds: tuple[float, ...] = (1.0, 1.5, 2.0)
    receiver_alignment_z: float = 0.25
    receiver_bucket_size: int = 8
    cooldown_bars: int = 4
    minimum_community_members: int = 3


def _rolling_scale(
    values: pd.DataFrame,
    cfg: V200FeatureConfig,
) -> pd.DataFrame:
    return (
        values.shift(1)
        .rolling(cfg.lookback_bars, min_periods=cfg.minimum_bars)
        .std(ddof=1)
    )


def causal_rolling_residual_z(
    dependent: pd.DataFrame,
    regressor: pd.DataFrame,
    cfg: V200FeatureConfig = V200FeatureConfig(),
) -> pd.DataFrame:
    """Current residual using only shifted history for regression moments."""
    output = pd.DataFrame(index=dependent.index, columns=dependent.columns, dtype=float)
    for symbol in dependent.columns:
        y = dependent[symbol]
        x = regressor[symbol]
        y_prior = y.shift(1)
        x_prior = x.shift(1)
        y_mean = y_prior.rolling(
            cfg.lookback_bars, min_periods=cfg.minimum_bars
        ).mean()
        x_mean = x_prior.rolling(
            cfg.lookback_bars, min_periods=cfg.minimum_bars
        ).mean()
        y_var = y_prior.rolling(
            cfg.lookback_bars, min_periods=cfg.minimum_bars
        ).var(ddof=1)
        x_var = x_prior.rolling(
            cfg.lookback_bars, min_periods=cfg.minimum_bars
        ).var(ddof=1)
        covariance = y_prior.rolling(
            cfg.lookback_bars, min_periods=cfg.minimum_bars
        ).cov(x_prior)
        valid_x_var = x_var.where(x_var.gt(0))
        beta = covariance.div(valid_x_var)
        residual = (y - y_mean) - beta * (x - x_mean)
        residual_variance = (
            y_var - covariance.pow(2).div(valid_x_var)
        ).clip(lower=0)
        output[symbol] = residual.div(
            np.sqrt(residual_variance).where(residual_variance.gt(0))
        )
    output.index.name = dependent.index.name
    return output


def cross_sectional_residual_z(
    dependent: pd.DataFrame,
    regressor: pd.DataFrame,
    minimum_symbols: int = 20,
) -> pd.DataFrame:
    """Neutralize each completed-bar cross section and standardize its residual."""
    output = pd.DataFrame(index=dependent.index, columns=dependent.columns, dtype=float)
    for timestamp in dependent.index:
        local = pd.concat(
            {"dependent": dependent.loc[timestamp], "regressor": regressor.loc[timestamp]},
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(local) < minimum_symbols or local["regressor"].var(ddof=1) <= 0:
            continue
        x = np.column_stack(
            [np.ones(len(local)), local["regressor"].to_numpy(dtype=float)]
        )
        y = local["dependent"].to_numpy(dtype=float)
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        scale = float(np.std(residual, ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            continue
        output.loc[timestamp, local.index] = residual / scale
    output.index.name = dependent.index.name
    return output


def build_v200_transmission_features(
    futures_close: pd.DataFrame,
    mark_close: pd.DataFrame,
    index_close: pd.DataFrame,
    premium_close: pd.DataFrame,
    cfg: V200FeatureConfig = V200FeatureConfig(),
) -> dict[str, pd.DataFrame]:
    symbols = sorted(
        set(futures_close.columns)
        & set(mark_close.columns)
        & set(index_close.columns)
        & set(premium_close.columns)
    )
    times = futures_close.index
    futures = futures_close.reindex(index=times, columns=symbols)
    mark = mark_close.reindex(index=times, columns=symbols)
    index = index_close.reindex(index=times, columns=symbols)
    premium = premium_close.reindex(index=times, columns=symbols)
    futures_return = np.log(futures).diff()
    mark_return = np.log(mark).diff()
    index_return = np.log(index).diff()
    premium_innovation = premium.diff()
    reference_gap = mark_return - index_return
    trade_gap = futures_return - mark_return

    reference_gap_z = reference_gap.div(_rolling_scale(reference_gap, cfg))
    trade_gap_z = trade_gap.div(_rolling_scale(trade_gap, cfg))
    premium_innovation_z = premium_innovation.div(
        _rolling_scale(premium_innovation, cfg)
    )
    index_return_z = index_return.div(_rolling_scale(index_return, cfg))
    mark_return_z = mark_return.div(_rolling_scale(mark_return, cfg))
    futures_return_z = futures_return.div(_rolling_scale(futures_return, cfg))
    reference_time_residual_z = causal_rolling_residual_z(
        reference_gap, premium_innovation, cfg
    )
    reference_residual_z = cross_sectional_residual_z(
        reference_time_residual_z, premium_innovation_z
    )
    return {
        "futures_return": futures_return,
        "mark_return": mark_return,
        "index_return": index_return,
        "premium_innovation": premium_innovation,
        "reference_gap": reference_gap,
        "trade_gap": trade_gap,
        "futures_return_z": futures_return_z,
        "mark_return_z": mark_return_z,
        "index_return_z": index_return_z,
        "premium_innovation_z": premium_innovation_z,
        "reference_gap_z": reference_gap_z,
        "reference_time_residual_z": reference_time_residual_z,
        "reference_residual_z": reference_residual_z,
        "trade_gap_z": trade_gap_z,
    }


def summarize_v200_feature_correlations(
    features: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    symbols = list(features["reference_gap_z"].columns)
    for symbol in symbols:
        local = pd.concat(
            {
                "reference_raw": features["reference_gap_z"][symbol],
                "reference_residual": features["reference_residual_z"][symbol],
                "trade_gap": features["trade_gap_z"][symbol],
                "premium": features["premium_innovation_z"][symbol],
            },
            axis=1,
        ).replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "symbol": symbol,
                "valid_reference_residual_points": int(
                    local[["reference_residual", "premium"]].dropna().shape[0]
                ),
                "raw_reference_premium_correlation": local[
                    "reference_raw"
                ].corr(local["premium"]),
                "residual_reference_premium_correlation": local[
                    "reference_residual"
                ].corr(local["premium"]),
                "trade_gap_premium_correlation": local["trade_gap"].corr(
                    local["premium"]
                ),
                "residual_reference_trade_gap_correlation": local[
                    "reference_residual"
                ].corr(local["trade_gap"]),
                "reference_residual_abs_q99": float(
                    local["reference_residual"].abs().quantile(0.99)
                ),
                "trade_gap_abs_q99": float(
                    local["trade_gap"].abs().quantile(0.99)
                ),
            }
        )
    by_symbol = pd.DataFrame(rows)
    cross_sectional_rows = []
    for timestamp in features["reference_gap_z"].index:
        local = pd.concat(
            {
                "raw": features["reference_gap_z"].loc[timestamp],
                "residual": features["reference_residual_z"].loc[timestamp],
                "premium": features["premium_innovation_z"].loc[timestamp],
            },
            axis=1,
        ).replace([np.inf, -np.inf], np.nan)
        cross_sectional_rows.append(
            {
                "raw": local["raw"].corr(local["premium"]),
                "residual": local["residual"].corr(local["premium"]),
            }
        )
    cross_sectional = pd.DataFrame(cross_sectional_rows)
    aggregate = pd.DataFrame(
        [
            {
                "symbols": len(by_symbol),
                "minimum_valid_reference_residual_points": int(
                    by_symbol["valid_reference_residual_points"].min()
                ),
                "median_raw_reference_premium_correlation": float(
                    by_symbol["raw_reference_premium_correlation"].median()
                ),
                "median_abs_raw_reference_premium_correlation": float(
                    by_symbol["raw_reference_premium_correlation"].abs().median()
                ),
                "median_residual_reference_premium_correlation": float(
                    by_symbol["residual_reference_premium_correlation"].median()
                ),
                "median_abs_residual_reference_premium_correlation": float(
                    by_symbol[
                        "residual_reference_premium_correlation"
                    ].abs().median()
                ),
                "max_abs_residual_reference_premium_correlation": float(
                    by_symbol[
                        "residual_reference_premium_correlation"
                    ].abs().max()
                ),
                "median_abs_cross_sectional_raw_premium_correlation": float(
                    cross_sectional["raw"].abs().median()
                ),
                "median_abs_cross_sectional_residual_premium_correlation": float(
                    cross_sectional["residual"].abs().median()
                ),
                "max_abs_cross_sectional_residual_premium_correlation": float(
                    cross_sectional["residual"].abs().max()
                ),
                "median_abs_trade_gap_premium_correlation": float(
                    by_symbol["trade_gap_premium_correlation"].abs().median()
                ),
                "median_abs_residual_reference_trade_gap_correlation": float(
                    by_symbol[
                        "residual_reference_trade_gap_correlation"
                    ].abs().median()
                ),
                "median_reference_residual_abs_q99": float(
                    by_symbol["reference_residual_abs_q99"].median()
                ),
                "median_trade_gap_abs_q99": float(
                    by_symbol["trade_gap_abs_q99"].median()
                ),
            }
        ]
    )
    return aggregate, by_symbol


def _cooldown_times(
    eligible: pd.Series,
    cooldown_bars: int,
) -> list[pd.Timestamp]:
    accepted: list[pd.Timestamp] = []
    last: pd.Timestamp | None = None
    cooldown = pd.Timedelta(minutes=15 * cooldown_bars)
    for timestamp in eligible.index[eligible.fillna(False)]:
        timestamp = pd.Timestamp(timestamp)
        if last is None or timestamp - last >= cooldown:
            accepted.append(timestamp)
            last = timestamp
    return accepted


def _receiver_state(
    timestamp: pd.Timestamp,
    names: list[str],
    source_sign: float,
    family: str,
    features: dict[str, pd.DataFrame],
    cfg: V200FeatureConfig,
) -> pd.DataFrame:
    if family == REFERENCE_LAG:
        score = -source_sign * features["reference_residual_z"].loc[
            timestamp, names
        ]
        aligned = source_sign * features["index_return_z"].loc[timestamp, names]
    elif family == TRADE_OVERSHOOT:
        score = source_sign * features["trade_gap_z"].loc[timestamp, names]
        aligned = source_sign * features["mark_return_z"].loc[timestamp, names]
    else:
        raise ValueError(f"unknown family: {family}")
    premium_abs = features["premium_innovation_z"].loc[timestamp, names].abs()
    return (
        pd.DataFrame(
            {
                "score": score,
                "aligned_return_z": aligned,
                "abs_premium_innovation_z": premium_abs,
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .query("aligned_return_z >= @cfg.receiver_alignment_z")
    )


def _event_rows(
    timestamp: pd.Timestamp,
    source_scope: str,
    source_setting: str,
    source_score: float,
    source_sign: float,
    community_id: str,
    names: list[str],
    features: dict[str, pd.DataFrame],
    cfg: V200FeatureConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_id = f"{source_scope}|{community_id}|{timestamp.isoformat()}"
    for family in FAMILIES:
        state = _receiver_state(
            timestamp, names, source_sign, family, features, cfg
        )
        for threshold in cfg.receiver_z_thresholds:
            selected = (
                state[state["score"].ge(threshold)]
                .sort_values(["score"], ascending=False)
                .head(cfg.receiver_bucket_size)
            )
            rows.append(
                {
                    "source_event_id": source_id,
                    "feature_time": timestamp,
                    "period": _period(timestamp),
                    "entry_day": timestamp.strftime("%Y-%m-%d"),
                    "entry_month": timestamp.strftime("%Y-%m"),
                    "source_scope": source_scope,
                    "source_setting": source_setting,
                    "community_id": community_id,
                    "source_score": source_score,
                    "source_sign": source_sign,
                    "family": family,
                    "receiver_z_threshold": threshold,
                    "eligible_members": len(names),
                    "receiver_count": len(selected),
                    "receivers": "|".join(selected.index.astype(str)),
                    "median_receiver_score": float(selected["score"].median()),
                    "mean_receiver_score": float(selected["score"].mean()),
                    "mean_abs_premium_innovation_z": float(
                        selected["abs_premium_innovation_z"].mean()
                    ),
                    "premium_shock_overlap": float(
                        selected["abs_premium_innovation_z"].ge(1.0).mean()
                    )
                    if len(selected)
                    else np.nan,
                }
            )
    return rows


def build_v200_feature_events(
    features: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    cfg: V200FeatureConfig = V200FeatureConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    times = features["index_return"].index
    symbols = set(features["index_return"].columns)
    alt_names = sorted(symbols - {BTC})
    btc_abs_return = features["index_return"][BTC].abs()
    for quantile in cfg.global_source_quantiles:
        threshold = (
            btc_abs_return.shift(1)
            .rolling(cfg.lookback_bars, min_periods=cfg.minimum_bars)
            .quantile(quantile)
        )
        eligible = btc_abs_return.ge(threshold) & threshold.notna()
        for timestamp in _cooldown_times(eligible, cfg.cooldown_bars):
            source_return = float(features["index_return"].at[timestamp, BTC])
            source_sign = float(np.sign(source_return))
            if source_sign == 0:
                continue
            rows.extend(
                _event_rows(
                    timestamp,
                    GLOBAL,
                    f"q{int(round(quantile * 100))}",
                    float(abs(source_return) / threshold.at[timestamp]),
                    source_sign,
                    "GLOBAL",
                    alt_names,
                    features,
                    cfg,
                )
            )

    for month, monthly_membership in membership.groupby("month_start", sort=True):
        month = pd.Timestamp(month)
        month_end = month + pd.offsets.MonthBegin(1)
        month_times = times[(times >= month) & (times < month_end)]
        if month_times.empty:
            continue
        for community_id, group in monthly_membership.groupby(
            "community_id", sort=True
        ):
            names = sorted((set(group["symbol"].astype(str)) & symbols) - {BTC})
            if len(names) < cfg.minimum_community_members:
                continue
            coherent = features["index_return_z"].loc[
                month_times, names
            ].median(axis=1)
            for threshold in cfg.community_source_z:
                eligible = coherent.abs().ge(threshold) & coherent.notna()
                for timestamp in _cooldown_times(eligible, cfg.cooldown_bars):
                    source_score = float(coherent.at[timestamp])
                    source_sign = float(np.sign(source_score))
                    if source_sign == 0:
                        continue
                    rows.extend(
                        _event_rows(
                            timestamp,
                            COMMUNITY,
                            f"z{threshold:.1f}",
                            abs(source_score),
                            source_sign,
                            str(community_id),
                            names,
                            features,
                            cfg,
                        )
                    )
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(
            [
                "feature_time",
                "source_scope",
                "community_id",
                "family",
                "receiver_z_threshold",
            ]
        ).reset_index(drop=True)
    return events


def summarize_v200_feature_events(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [
        "source_scope",
        "family",
        "source_setting",
        "receiver_z_threshold",
    ]
    for keys, local in events.groupby(group_columns, sort=True):
        source_scope, family, source_setting, receiver_threshold = keys
        any_receiver = local[local["receiver_count"].gt(0)]
        bucket3 = local[local["receiver_count"].ge(3)]
        bucket5 = local[local["receiver_count"].ge(5)]
        canonical_receiver_threshold = (
            1.0
            if source_scope == COMMUNITY and family == REFERENCE_LAG
            else 1.5
        )
        canonical = (
            receiver_threshold == canonical_receiver_threshold
            and (
                (source_scope == GLOBAL and source_setting == "q90")
                or (source_scope == COMMUNITY and source_setting == "z2.0")
            )
        )
        rows.append(
            {
                "source_scope": source_scope,
                "family": family,
                "source_setting": source_setting,
                "receiver_z_threshold": receiver_threshold,
                "canonical": canonical,
                "source_events": len(local),
                "events_any_receiver": len(any_receiver),
                "bucket_events_min3": len(bucket3),
                "bucket_events_min5": len(bucket5),
                "bucket3_development_events": int(
                    bucket3["period"].eq("development").sum()
                ),
                "bucket3_validation_events": int(
                    bucket3["period"].eq("validation").sum()
                ),
                "bucket3_holdout_events": int(
                    bucket3["period"].eq("holdout").sum()
                ),
                "bucket3_active_days": bucket3["entry_day"].nunique(),
                "bucket3_active_months": bucket3["entry_month"].nunique(),
                "median_receiver_count_when_active": float(
                    any_receiver["receiver_count"].median()
                ),
                "median_source_score": float(local["source_score"].median()),
                "median_receiver_score_bucket3": float(
                    bucket3["median_receiver_score"].median()
                ),
                "mean_premium_shock_overlap_bucket3": float(
                    bucket3["premium_shock_overlap"].mean()
                ),
                "feature_viable": bool(
                    canonical
                    and len(bucket3) >= 100
                    and bucket3["period"].eq("validation").sum() >= 20
                    and bucket3["period"].eq("holdout").sum() >= 25
                    and bucket3["entry_month"].nunique() >= 8
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_findings(
    correlations: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    canonical = summary[summary["canonical"]]
    text = [
        "# v20.0 Reference-Price Transmission Feature-Only Audit",
        "",
        "## Orthogonality",
        "",
        correlations.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Canonical pre-reveal configurations",
        "",
        canonical.to_markdown(index=False, floatfmt=".4f"),
        "",
        "`REFERENCE_RESIDUAL_INDEX_LEAD_CATCHUP` first removes the official premium "
        "innovation with a shifted prior-30-day rolling regression, then removes "
        "the remaining contemporaneous cross-sectional premium component before "
        "ranking receivers. `TRADE_VS_MARK_OVERSHOOT_FADE` uses the last-trade/mark "
        "return gap, a separate execution-price layer.",
        "",
        "Global events require a causal BTC index-return tail. Community events "
        "require a coherent median standardized index move inside the frozen "
        "monthly graph community. Receiver scores, buckets, cooldowns, and overlap "
        "diagnostics use only the completed feature bar.",
        "The canonical community reference-lag threshold is 1.0 because the final "
        "double-orthogonal score is cross-sectionally standardized before being "
        "restricted to small communities; all other canonical receiver thresholds "
        "remain 1.5. This coverage choice was made without future-return inspection.",
        "",
        "No future price, funding, candidate PnL, live, PaperLive, application, "
        "leverage, remote, or order state was read or changed.",
        "",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(text), encoding="utf-8")


def write_v200_feature_audit(
    metrics_root: Path = METRICS_ROOT,
    kline_root: Path = KLINE_ROOT,
    mark_root: Path = MARK_ROOT,
    index_root: Path = INDEX_ROOT,
    premium_root: Path = PREMIUM_ROOT,
    membership_path: Path = MEMBERSHIP_PATH,
    report_root: Path = REPORT_ROOT,
    findings_path: Path = FINDINGS_PATH,
    cfg: V200FeatureConfig = V200FeatureConfig(),
) -> dict[str, Path]:
    futures, _ = load_v184_exact_panels(metrics_root, kline_root)
    mark = load_reference_ohlc_panels(mark_root)["close"]
    index = load_reference_ohlc_panels(index_root)["close"]
    premium = load_v190_premium_panel(premium_root)
    membership = load_v195_membership(membership_path)
    features = build_v200_transmission_features(
        futures, mark, index, premium, cfg
    )
    correlations, by_symbol = summarize_v200_feature_correlations(features)
    events = build_v200_feature_events(features, membership, cfg)
    summary = summarize_v200_feature_events(events)
    root = ensure_dir(report_root)
    outputs = {
        "events": root / "candidate_feature_events.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "correlations": root / "feature_correlation_summary.csv",
        "by_symbol": root / "feature_correlation_by_symbol.csv",
        "reference_residual_z": root / "reference_residual_z.parquet",
        "trade_gap_z": root / "trade_gap_z.parquet",
        "findings": findings_path,
    }
    events.to_parquet(outputs["events"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    correlations.to_csv(outputs["correlations"], index=False)
    by_symbol.to_csv(outputs["by_symbol"], index=False)
    features["reference_residual_z"].to_parquet(outputs["reference_residual_z"])
    features["trade_gap_z"].to_parquet(outputs["trade_gap_z"])
    _write_findings(correlations, summary, findings_path)
    return outputs


__all__ = [
    "COMMUNITY",
    "FAMILIES",
    "GLOBAL",
    "REFERENCE_LAG",
    "TRADE_OVERSHOOT",
    "V200FeatureConfig",
    "build_v200_feature_events",
    "build_v200_transmission_features",
    "causal_rolling_residual_z",
    "cross_sectional_residual_z",
    "summarize_v200_feature_correlations",
    "summarize_v200_feature_events",
    "write_v200_feature_audit",
]
