"""Monthly BTC implied-versus-realized variance risk-premium audit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


REPORT_ROOT = Path("reports/v11_9_variance_risk_premium")
DVOL_PATH = Path("data/external/orthogonal_volatility/deribit_dvol_1h/BTC.parquet")
PRICE_PATH = Path(
    "data/external/orthogonal_volatility/deribit_perpetual_1h/BTC-PERPETUAL.parquet"
)
CANDIDATES = ("VRP1_MONTHLY_SHORT_VARIANCE", "VRP2_RICH_IV_SHORT_VARIANCE")


@dataclass(frozen=True)
class V119Config:
    dvol_path: Path = DVOL_PATH
    price_path: Path = PRICE_PATH
    report_root: Path = REPORT_ROOT
    minimum_hourly_observations: int = 700
    rich_iv_spread_points: float = 5.0
    random_iterations: int = 2000
    bootstrap_iterations: int = 5000
    seed: int = 20260715


def _period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2025-01-01", tz="UTC"):
        return "development"
    if timestamp < pd.Timestamp("2025-07-01", tz="UTC"):
        return "validation"
    return "holdout"


def _annualized_rv(log_returns: pd.Series) -> float:
    values = pd.to_numeric(log_returns, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.square(values).sum() * 365.0 / 30.0) * 100.0)


def _normalized_payoff(iv: float, future_rv: float, strike_cost_points: float) -> float:
    iv_decimal = iv / 100.0
    effective = max(iv - strike_cost_points, 0.0) / 100.0
    rv_decimal = future_rv / 100.0
    return (effective**2 - rv_decimal**2) / (iv_decimal**2) if iv_decimal > 0 else np.nan


def build_v119_monthly_panel(
    dvol_path: Path = DVOL_PATH,
    price_path: Path = PRICE_PATH,
    minimum_hourly_observations: int = 700,
) -> pd.DataFrame:
    dvol = pd.read_parquet(dvol_path, columns=["dvol_time", "close"])
    dvol["dvol_time"] = pd.to_datetime(dvol["dvol_time"], utc=True, errors="coerce")
    dvol["close"] = pd.to_numeric(dvol["close"], errors="coerce")
    dvol = dvol.dropna().drop_duplicates("dvol_time", keep="last").sort_values("dvol_time")
    price = pd.read_parquet(price_path, columns=["bar_open_time", "close"])
    price["bar_open_time"] = pd.to_datetime(
        price["bar_open_time"], utc=True, errors="coerce"
    )
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna().drop_duplicates("bar_open_time", keep="last").sort_values(
        "bar_open_time"
    )
    price["log_return"] = np.log(price["close"]).diff()
    start = max(dvol["dvol_time"].min(), price["bar_open_time"].min()) + pd.Timedelta(days=31)
    end = min(dvol["dvol_time"].max(), price["bar_open_time"].max()) - pd.Timedelta(days=30)
    anchors = pd.date_range(start=start.floor("D").replace(day=1), end=end, freq="MS")
    anchors = anchors + pd.Timedelta(hours=8)
    dvol_times = dvol["dvol_time"].to_numpy(dtype="datetime64[ns]")
    rows = []
    for anchor in anchors:
        target = np.datetime64(pd.Timestamp(anchor).tz_convert("UTC").tz_localize(None))
        dvol_index = int(np.searchsorted(dvol_times, target, side="right")) - 1
        if dvol_index < 0:
            continue
        dvol_row = dvol.iloc[dvol_index]
        dvol_time = pd.Timestamp(dvol_row["dvol_time"])
        if anchor - dvol_time > pd.Timedelta(hours=2):
            continue
        trailing = price[
            price["bar_open_time"].gt(anchor - pd.Timedelta(days=30))
            & price["bar_open_time"].le(anchor)
        ]["log_return"]
        future = price[
            price["bar_open_time"].gt(anchor)
            & price["bar_open_time"].le(anchor + pd.Timedelta(days=30))
        ]["log_return"]
        if (
            trailing.notna().sum() < minimum_hourly_observations
            or future.notna().sum() < minimum_hourly_observations
        ):
            continue
        iv = float(dvol_row["close"])
        trailing_rv = _annualized_rv(trailing)
        future_rv = _annualized_rv(future)
        rows.append(
            {
                "signal_time": anchor,
                "period": _period(anchor),
                "dvol_time": dvol_time,
                "dvol": iv,
                "trailing_rv_30d": trailing_rv,
                "future_rv_30d": future_rv,
                "iv_minus_trailing_rv": iv - trailing_rv,
                "gross_variance_payoff": _normalized_payoff(iv, future_rv, 0.0),
                "net_1vol": _normalized_payoff(iv, future_rv, 1.0),
                "net_2vol": _normalized_payoff(iv, future_rv, 2.0),
                "trailing_hours": int(trailing.notna().sum()),
                "future_hours": int(future.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)


def _sample(panel: pd.DataFrame, candidate: str, cfg: V119Config) -> pd.DataFrame:
    if candidate == CANDIDATES[0]:
        return panel.copy()
    return panel[panel["iv_minus_trailing_rv"].ge(cfg.rich_iv_spread_points)].copy()


def summarize_v119(panel: pd.DataFrame, cfg: V119Config) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        candidate_panel = _sample(panel, candidate, cfg)
        for period in ("full", "development", "validation", "holdout"):
            sample = (
                candidate_panel
                if period == "full"
                else candidate_panel[candidate_panel["period"].eq(period)]
            )
            rows.append(
                {
                    "candidate": candidate,
                    "period": period,
                    "months": len(sample),
                    "mean_dvol": float(sample["dvol"].mean()) if len(sample) else np.nan,
                    "mean_future_rv": float(sample["future_rv_30d"].mean())
                    if len(sample)
                    else np.nan,
                    "gross_variance_payoff": float(sample["gross_variance_payoff"].mean())
                    if len(sample)
                    else np.nan,
                    "net_1vol": float(sample["net_1vol"].mean()) if len(sample) else np.nan,
                    "net_2vol": float(sample["net_2vol"].mean()) if len(sample) else np.nan,
                    "win_rate_net_1vol": float(sample["net_1vol"].gt(0).mean())
                    if len(sample)
                    else np.nan,
                    "worst_net_1vol": float(sample["net_1vol"].min()) if len(sample) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    if not len(values):
        return np.nan, np.nan
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _max_drawdown(values: pd.Series) -> float:
    cumulative = values.fillna(0.0).cumsum()
    drawdown = cumulative.cummax() - cumulative
    return float(drawdown.max()) if len(drawdown) else np.nan


def audit_v119(panel: pd.DataFrame, cfg: V119Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    controls = []
    slices = []
    decisions = []
    for candidate in CANDIDATES:
        sample = _sample(panel, candidate, cfg).reset_index(drop=True)
        validation = sample[sample["period"].eq("validation")]
        holdout = sample[sample["period"].eq("holdout")]
        values = sample["net_1vol"].to_numpy(dtype=float)
        lower, upper = _bootstrap(values, cfg.bootstrap_iterations, rng)
        random_means = []
        iv = sample["dvol"].to_numpy(dtype=float)
        future_rv = sample["future_rv_30d"].to_numpy(dtype=float)
        for _ in range(cfg.random_iterations):
            shuffled = rng.permutation(iv)
            payoffs = [
                _normalized_payoff(float(shuffled[index]), float(future_rv[index]), 1.0)
                for index in range(len(sample))
            ]
            random_means.append(float(np.mean(payoffs)) if payoffs else np.nan)
        real = float(sample["net_1vol"].mean()) if len(sample) else np.nan
        random_values = np.asarray(random_means, dtype=float)
        percentile = (
            float(np.nanmean(random_values <= real) * 100.0) if len(random_values) else np.nan
        )
        stale = sample["dvol"].shift(1)
        stale_payoff = pd.Series(
            [
                _normalized_payoff(float(stale.iloc[index]), float(future_rv[index]), 1.0)
                if pd.notna(stale.iloc[index])
                else np.nan
                for index in range(len(sample))
            ]
        )
        positive = sample.loc[sample["net_1vol"].gt(0), "net_1vol"]
        positive_share = (
            float(positive.max() / positive.sum()) if len(positive) and positive.sum() > 0 else np.nan
        )
        controls.extend(
            [
                {"candidate": candidate, "control": "bootstrap_low", "value": lower},
                {"candidate": candidate, "control": "bootstrap_high", "value": upper},
                {"candidate": candidate, "control": "random_iv_percentile", "value": percentile},
                {
                    "candidate": candidate,
                    "control": "random_iv_mean",
                    "value": float(np.nanmean(random_values)) if len(random_values) else np.nan,
                },
                {"candidate": candidate, "control": "long_variance_net_1vol", "value": -real},
                {
                    "candidate": candidate,
                    "control": "stale_one_month_dvol_net_1vol",
                    "value": float(stale_payoff.mean()),
                },
                {
                    "candidate": candidate,
                    "control": "normalized_max_drawdown",
                    "value": _max_drawdown(sample["net_1vol"]),
                },
                {"candidate": candidate, "control": "max_positive_month_share", "value": positive_share},
            ]
        )
        if len(sample):
            boundaries = np.linspace(0, len(sample), 6, dtype=int)
            for index in range(5):
                fifth = sample.iloc[boundaries[index] : boundaries[index + 1]]
                slices.append(
                    {
                        "candidate": candidate,
                        "chronological_fifth": index + 1,
                        "months": len(fifth),
                        "net_1vol": float(fifth["net_1vol"].mean()),
                    }
                )
        minimum_full = 24 if candidate == CANDIDATES[0] else 12
        minimum_validation = 6 if candidate == CANDIDATES[0] else 4
        minimum_holdout = 8 if candidate == CANDIDATES[0] else 4
        gates = {
            "full_n_gate": len(sample) >= minimum_full,
            "validation_n_gate": len(validation) >= minimum_validation,
            "holdout_n_gate": len(holdout) >= minimum_holdout,
            "full_net_1vol_positive": real > 0,
            "validation_net_1vol_positive": float(validation["net_1vol"].mean()) > 0
            if len(validation)
            else False,
            "holdout_net_1vol_positive": float(holdout["net_1vol"].mean()) > 0
            if len(holdout)
            else False,
            "full_net_2vol_positive": float(sample["net_2vol"].mean()) > 0
            if len(sample)
            else False,
            "holdout_net_2vol_positive": float(holdout["net_2vol"].mean()) > 0
            if len(holdout)
            else False,
            "bootstrap_low_positive": lower > 0,
            "random_percentile_at_least_95": percentile >= 95,
            "worst_month_above_minus_100pct": float(sample["net_1vol"].min()) > -1.0
            if len(sample)
            else False,
            "positive_month_share_below_35pct": positive_share <= 0.35,
        }
        decisions.append(
            {
                "candidate": candidate,
                "verdict": "eligible_executable_options_reconstruction"
                if all(gates.values())
                else "reject",
                **gates,
            }
        )
    return pd.DataFrame(controls), pd.DataFrame(slices), pd.DataFrame(decisions)


def write_v119_variance_risk_premium(cfg: V119Config = V119Config()) -> dict[str, Path]:
    panel = build_v119_monthly_panel(
        cfg.dvol_path, cfg.price_path, cfg.minimum_hourly_observations
    )
    summary = summarize_v119(panel, cfg)
    controls, slices, decisions = audit_v119(panel, cfg)
    root = ensure_dir(cfg.report_root)
    outputs = {
        "panel": root / "monthly_panel.csv",
        "summary": root / "summary.csv",
        "controls": root / "controls.csv",
        "slices": root / "chronological_fifths.csv",
        "decision": root / "decision.csv",
    }
    panel.to_csv(outputs["panel"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    slices.to_csv(outputs["slices"], index=False)
    decisions.to_csv(outputs["decision"], index=False)
    return outputs


__all__ = [
    "CANDIDATES",
    "V119Config",
    "audit_v119",
    "build_v119_monthly_panel",
    "summarize_v119",
    "write_v119_variance_risk_premium",
]
