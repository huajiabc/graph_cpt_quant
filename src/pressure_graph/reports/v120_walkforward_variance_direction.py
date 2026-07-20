"""Purged walk-forward ridge model for monthly variance direction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v12_0_walkforward_variance_direction")
PANEL_PATH = Path("reports/v11_9_variance_risk_premium/monthly_panel.csv")
FEATURES = (
    "dvol",
    "trailing_rv_30d",
    "iv_minus_trailing_rv",
    "iv_to_trailing_ratio",
    "dvol_change_1m",
    "trailing_rv_change_1m",
    "dvol_z_12m",
)


@dataclass(frozen=True)
class V120Config:
    panel_path: Path = PANEL_PATH
    report_root: Path = REPORT_ROOT
    ridge_alpha: float = 10.0
    minimum_training_months: int = 24
    random_iterations: int = 2000
    bootstrap_iterations: int = 5000
    seed: int = 20260715


def _directional_payoff(iv: float, future_rv: float, direction: float, cost_points: float) -> float:
    iv_decimal = iv / 100.0
    if iv_decimal <= 0:
        return np.nan
    rv_decimal = future_rv / 100.0
    if direction > 0:
        strike = max(iv - cost_points, 0.0) / 100.0
        return (strike**2 - rv_decimal**2) / (iv_decimal**2)
    strike = (iv + cost_points) / 100.0
    return (rv_decimal**2 - strike**2) / (iv_decimal**2)


def load_v120_features(path: Path = PANEL_PATH) -> pd.DataFrame:
    panel = pd.read_csv(path)
    panel["signal_time"] = pd.to_datetime(panel["signal_time"], utc=True, errors="coerce")
    numeric = [
        "dvol",
        "trailing_rv_30d",
        "future_rv_30d",
        "iv_minus_trailing_rv",
        "gross_variance_payoff",
        "net_1vol",
        "net_2vol",
    ]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["signal_time", *numeric]).sort_values("signal_time")
    panel["iv_to_trailing_ratio"] = (
        panel["dvol"].div(panel["trailing_rv_30d"]) - 1.0
    )
    panel["dvol_change_1m"] = panel["dvol"].pct_change(fill_method=None)
    panel["trailing_rv_change_1m"] = panel["trailing_rv_30d"].pct_change(
        fill_method=None
    )
    history = panel["dvol"].shift(1)
    mean = history.rolling(12, min_periods=12).mean()
    std = history.rolling(12, min_periods=12).std(ddof=1)
    panel["dvol_z_12m"] = panel["dvol"].sub(mean).div(std.where(std.gt(0))).clip(-5, 5)
    return panel.reset_index(drop=True)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=1)
    std = np.where(std > 0, std, 1.0)
    x = (train_x - mean) / std
    target = (predict_x - mean) / std
    y_mean = float(train_y.mean())
    centered = train_y - y_mean
    penalty = np.eye(x.shape[1]) * alpha
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ centered)
    return float(y_mean + target @ coefficients), coefficients / std


def build_v120_predictions(panel: pd.DataFrame, cfg: V120Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coefficient_rows = []
    for index, current in panel.iterrows():
        if current[list(FEATURES)].isna().any():
            continue
        signal_time = pd.Timestamp(current["signal_time"])
        resolved = panel[
            panel["signal_time"].add(pd.Timedelta(days=30)).le(signal_time)
        ].dropna(subset=[*FEATURES, "gross_variance_payoff"])
        if len(resolved) < cfg.minimum_training_months:
            continue
        train_x = resolved.loc[:, FEATURES].to_numpy(dtype=float)
        train_y = resolved["gross_variance_payoff"].to_numpy(dtype=float)
        predict_x = current.loc[list(FEATURES)].to_numpy(dtype=float)
        prediction, coefficients = _ridge_predict(
            train_x, train_y, predict_x, cfg.ridge_alpha
        )
        direction = 1.0 if prediction >= 0 else -1.0
        iv = float(current["dvol"])
        future_rv = float(current["future_rv_30d"])
        rows.append(
            {
                "signal_time": signal_time,
                "period": str(current["period"]),
                "training_months": len(resolved),
                "predicted_short_variance_payoff": prediction,
                "direction": direction,
                "direction_label": "short_variance" if direction > 0 else "long_variance",
                "dvol": iv,
                "future_rv_30d": future_rv,
                "gross_payoff": _directional_payoff(iv, future_rv, direction, 0.0),
                "net_1vol": _directional_payoff(iv, future_rv, direction, 1.0),
                "net_2vol": _directional_payoff(iv, future_rv, direction, 2.0),
                "always_short_net_1vol": _directional_payoff(iv, future_rv, 1.0, 1.0),
                "spread_rule_direction": 1.0
                if float(current["iv_minus_trailing_rv"]) >= 0
                else -1.0,
            }
        )
        for feature, coefficient in zip(FEATURES, coefficients, strict=True):
            coefficient_rows.append(
                {
                    "signal_time": signal_time,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "training_months": len(resolved),
                }
            )
    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions["spread_rule_net_1vol"] = [
            _directional_payoff(
                float(row.dvol),
                float(row.future_rv_30d),
                float(row.spread_rule_direction),
                1.0,
            )
            for row in predictions.itertuples(index=False)
        ]
    return predictions, pd.DataFrame(coefficient_rows)


def summarize_v120(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ("full", "development", "validation", "holdout"):
        sample = (
            predictions if period == "full" else predictions[predictions["period"].eq(period)]
        )
        rows.append(
            {
                "period": period,
                "predictions": len(sample),
                "long_variance_share": float(sample["direction"].lt(0).mean())
                if len(sample)
                else np.nan,
                "gross_payoff": float(sample["gross_payoff"].mean()) if len(sample) else np.nan,
                "net_1vol": float(sample["net_1vol"].mean()) if len(sample) else np.nan,
                "net_2vol": float(sample["net_2vol"].mean()) if len(sample) else np.nan,
                "always_short_net_1vol": float(sample["always_short_net_1vol"].mean())
                if len(sample)
                else np.nan,
                "spread_rule_net_1vol": float(sample["spread_rule_net_1vol"].mean())
                if len(sample)
                else np.nan,
                "win_rate_net_1vol": float(sample["net_1vol"].gt(0).mean())
                if len(sample)
                else np.nan,
                "worst_net_1vol": float(sample["net_1vol"].min()) if len(sample) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def audit_v120(
    predictions: pd.DataFrame, cfg: V120Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    values = predictions["net_1vol"].to_numpy(dtype=float)
    lower, upper = _bootstrap(values, cfg.bootstrap_iterations, rng)
    short = predictions["always_short_net_1vol"].to_numpy(dtype=float)
    long = np.array(
        [
            _directional_payoff(float(row.dvol), float(row.future_rv_30d), -1.0, 1.0)
            for row in predictions.itertuples(index=False)
        ]
    )
    random_means = []
    for _ in range(cfg.random_iterations):
        choose_short = rng.integers(0, 2, size=len(predictions)).astype(bool)
        random_means.append(float(np.where(choose_short, short, long).mean()))
    random_values = np.asarray(random_means)
    real = float(predictions["net_1vol"].mean())
    percentile = float((random_values <= real).mean() * 100.0)
    cumulative = predictions["net_1vol"].cumsum()
    drawdown = float((cumulative.cummax() - cumulative).max())
    controls = pd.DataFrame(
        [
            {"control": "bootstrap_low", "value": lower},
            {"control": "bootstrap_high", "value": upper},
            {"control": "random_direction_percentile", "value": percentile},
            {"control": "random_direction_mean", "value": float(random_values.mean())},
            {"control": "normalized_max_drawdown", "value": drawdown},
        ]
    )
    slices = []
    boundaries = np.linspace(0, len(predictions), 6, dtype=int)
    for index in range(5):
        fifth = predictions.iloc[boundaries[index] : boundaries[index + 1]]
        slices.append(
            {
                "chronological_fifth": index + 1,
                "predictions": len(fifth),
                "net_1vol": float(fifth["net_1vol"].mean()) if len(fifth) else np.nan,
            }
        )
    validation = predictions[predictions["period"].eq("validation")]
    holdout = predictions[predictions["period"].eq("holdout")]
    gates = {
        "full_n_at_least_24": len(predictions) >= 24,
        "validation_n_at_least_6": len(validation) >= 6,
        "holdout_n_at_least_8": len(holdout) >= 8,
        "full_net_1vol_positive": real > 0,
        "validation_net_1vol_positive": float(validation["net_1vol"].mean()) > 0
        if len(validation)
        else False,
        "holdout_net_1vol_positive": float(holdout["net_1vol"].mean()) > 0
        if len(holdout)
        else False,
        "full_net_2vol_positive": float(predictions["net_2vol"].mean()) > 0,
        "holdout_net_2vol_positive": float(holdout["net_2vol"].mean()) > 0
        if len(holdout)
        else False,
        "bootstrap_low_positive": lower > 0,
        "random_percentile_at_least_95": percentile >= 95,
        "beats_always_short_full": real
        > float(predictions["always_short_net_1vol"].mean()),
        "beats_always_short_holdout": float(holdout["net_1vol"].mean())
        > float(holdout["always_short_net_1vol"].mean())
        if len(holdout)
        else False,
    }
    decision = pd.DataFrame(
        [
            {
                "candidate": "VMD1_PURGED_WALKFORWARD_RIDGE",
                "verdict": "forward_only_model_candidate" if all(gates.values()) else "reject",
                **gates,
            }
        ]
    )
    return controls, pd.DataFrame(slices), decision


def write_v120_walkforward_variance_direction(
    cfg: V120Config = V120Config(),
) -> dict[str, Path]:
    panel = load_v120_features(cfg.panel_path)
    predictions, coefficients = build_v120_predictions(panel, cfg)
    summary = summarize_v120(predictions)
    controls, slices, decision = audit_v120(predictions, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "predictions": root / "predictions.csv",
        "coefficients": root / "coefficients.csv",
        "summary": root / "summary.csv",
        "controls": root / "controls.csv",
        "slices": root / "chronological_fifths.csv",
        "decision": root / "decision.csv",
    }
    predictions.to_csv(outputs["predictions"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    slices.to_csv(outputs["slices"], index=False)
    decision.to_csv(outputs["decision"], index=False)
    return outputs


__all__ = [
    "FEATURES",
    "V120Config",
    "audit_v120",
    "build_v120_predictions",
    "load_v120_features",
    "summarize_v120",
    "write_v120_walkforward_variance_direction",
]
