from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    "momentum_rank",
    "ridge",
    "xgb_shallow",
    "xgb_shuffled",
    "xgb_no_crowding",
]


def _context(predictions: pd.DataFrame) -> pd.DataFrame:
    context = predictions[
        ["feature_time", "btc_ret_4h", "btc_volatility_percentile"]
    ].drop_duplicates("feature_time")
    context["posthoc_up_lowvol"] = (
        pd.to_numeric(context["btc_ret_4h"], errors="coerce").ge(0)
        & pd.to_numeric(context["btc_volatility_percentile"], errors="coerce").lt(75)
    )
    return context


def run(root: Path, iterations: int, seed: int) -> dict[str, Path]:
    predictions = pd.read_parquet(root / "oos_predictions.parquet")
    ledger = pd.read_csv(root / "portfolio_ledger.csv")
    predictions["feature_time"] = pd.to_datetime(
        predictions["feature_time"], utc=True, errors="coerce"
    )
    ledger["feature_time"] = pd.to_datetime(ledger["feature_time"], utc=True, errors="coerce")
    context = _context(predictions)
    local_ledger = ledger.merge(context, on="feature_time", how="left", validate="many_to_one")
    local_ledger = local_ledger[local_ledger["posthoc_up_lowvol"].eq(True)].copy()
    monthly = (
        local_ledger[local_ledger["model"].isin(MODELS)]
        .groupby(["model", "entry_month", "period"], as_index=False)
        .agg(
            periods=("feature_time", "size"),
            mean_rank_ic=("rank_ic", "mean"),
            gross_excess_sum=("gross_excess", "sum"),
            net_excess_20_sum=("net_excess_20", "sum"),
            average_turnover=("turnover", "mean"),
        )
    )
    source = predictions[predictions["model"].eq("momentum_rank")].merge(
        context, on="feature_time", how="left", validate="many_to_one"
    )
    source = source[source["posthoc_up_lowvol"].eq(True)].copy()
    groups = []
    for _, group in source.sort_values("feature_time").groupby("feature_time", sort=True):
        groups.append(
            (
                group["symbol"].astype(str).to_numpy(),
                pd.to_numeric(group["future_return"], errors="coerce").to_numpy(dtype=float),
                float(pd.to_numeric(group["future_return"], errors="coerce").mean()),
            )
        )
    random_rows = []
    for iteration in range(iterations):
        rng = np.random.default_rng(seed + iteration)
        previous: set[str] = set()
        total = 0.0
        for symbols, returns, market_ret in groups:
            idx = rng.choice(len(symbols), size=5, replace=False)
            selected = set(symbols[idx])
            turnover = 1.0 if not previous else 1.0 - len(previous & selected) / 5.0
            previous = selected
            total += float(np.nanmean(returns[idx])) - market_ret - turnover * 20 / 10_000.0
        random_rows.append({"iteration": iteration, "net_excess_20_sum": total})
    random_control = pd.DataFrame(random_rows)
    model_rows = []
    for model in MODELS:
        value = float(
            pd.to_numeric(
                local_ledger.loc[local_ledger["model"].eq(model), "net_excess_20"],
                errors="coerce",
            ).sum()
        )
        model_rows.append(
            {
                "model": model,
                "posthoc_up_lowvol_periods": int(
                    local_ledger.loc[local_ledger["model"].eq(model), "feature_time"].nunique()
                ),
                "net_excess_20_sum": value,
                "random_percentile": float(random_control["net_excess_20_sum"].lt(value).mean()),
            }
        )
    model_control = pd.DataFrame(model_rows)
    outputs = {
        "monthly": root / "posthoc_up_lowvol_monthly.csv",
        "random": root / "posthoc_up_lowvol_random_controls.csv",
        "model_control": root / "posthoc_up_lowvol_model_control.csv",
    }
    monthly.to_csv(outputs["monthly"], index=False)
    random_control.to_csv(outputs["random"], index=False)
    model_control.to_csv(outputs["model_control"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the post-hoc 12h BTC-up/low-vol slice.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("reports/v9_7_1_direct_ml_alpha_12h"),
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    for name, path in run(args.root, args.iterations, args.seed).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
