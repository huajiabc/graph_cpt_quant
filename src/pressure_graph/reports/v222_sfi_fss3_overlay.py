"""Outcome reveal for the preregistered causal SFI tilt inside weekly FSS3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v106_directed_residual_bucket import BTC
from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    _moving_block_means,
)
from pressure_graph.reports.v147_funding_sign_spread import (
    V147Config,
    beta_neutral_components,
    load_v147_panel,
)
from pressure_graph.reports.v149_funding_sign_turnover_cap import (
    _execute_capped_array_transition,
    _neutralize_alt_array,
    execute_capped_transition,
    funding_sign_target,
    weight_turnover,
)
from pressure_graph.reports.v151_causal_risk_parity_fss3_tg1 import (
    _additive_max_drawdown,
)


PANEL_PATH = Path(
    "reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet"
)
FEATURE_PATH = Path(
    "reports/v22_1_sfi_fss3_overlay_feature_audit/"
    "weekly_symbol_overlay_features.parquet"
)
FSS3_PATH = Path("reports/v14_9_funding_sign_turnover_cap/weekly_portfolio.parquet")
TG1_PATH = Path("reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet")
CM2_PATH = Path(
    "reports/v16_5_fixed_core_satellite_fss3_tg1/weekly_portfolio.parquet"
)
REPORT_ROOT = Path("reports/v22_2_sfi_fss3_overlay")
FINDINGS_PATH = Path("docs/v222_sfi_fss3_overlay_findings_2026_07_17.md")
CANDIDATE = "SFO1_FSS3_WITH_CAUSAL_SFI_RANK_TILT"
REVERSED_CONTROL = "SFO1_REVERSED_SFI_RANK_TILT"
BASELINE_CONTROL = "SFO1_ZERO_TILT_FSS3_RECONSTRUCTION"
CM2_CANDIDATE = "CM2_FIXED_80_SFO1_20_TG1"


@dataclass(frozen=True)
class V222Config:
    panel_path: Path = PANEL_PATH
    feature_path: Path = FEATURE_PATH
    fss3_path: Path = FSS3_PATH
    tg1_path: Path = TG1_PATH
    cm2_path: Path = CM2_PATH
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    minimum_side_breadth: int = 4
    transition_turnover_cap: float = 0.70
    one_way_cost: float = 0.002
    stress_one_way_cost: float = 0.004
    fss3_weight: float = 0.80
    tg1_weight: float = 0.20
    null_iterations: int = 1000
    bootstrap_iterations: int = 2000
    bootstrap_block_weeks: int = 4
    bisection_iterations: int = 48
    seed: int = 20260717


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_v222_inputs(
    cfg: V222Config = V222Config(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_v147_panel(V147Config(panel_path=cfg.panel_path))
    features = pd.read_parquet(cfg.feature_path)
    for column in ("entry_time", "sfi_feature_time", "month_start"):
        features[column] = pd.to_datetime(features[column], utc=True, errors="coerce")
    return panel, features.sort_values(["entry_time", "side", "symbol"]).reset_index(
        drop=True
    )


def _eligible_sign_sides(
    local: pd.DataFrame,
    minimum_side_breadth: int,
) -> tuple[list[str], list[str]]:
    eligible = local.dropna(
        subset=["score_7d", "price_return", "future_funding", "btc_beta"]
    )
    longs = sorted(
        eligible.loc[eligible["score_7d"].lt(0), "symbol"].astype(str).unique()
    )
    shorts = sorted(
        eligible.loc[eligible["score_7d"].gt(0), "symbol"].astype(str).unique()
    )
    if len(longs) < minimum_side_breadth or len(shorts) < minimum_side_breadth:
        return [], []
    return longs, shorts


def overlay_raw_target(
    local: pd.DataFrame,
    week_features: pd.DataFrame,
    cfg: V222Config = V222Config(),
    *,
    mode: str = "observed",
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Build the frozen within-side tilt; modes only change rank assignment."""
    if mode not in {"observed", "reversed", "random"}:
        raise ValueError(f"unsupported overlay mode: {mode}")
    longs, shorts = _eligible_sign_sides(local, cfg.minimum_side_breadth)
    if not longs or not shorts:
        return {}
    indexed = week_features.set_index("symbol", verify_integrity=True)
    raw: dict[str, float] = {}
    for side, symbols, sign in (("long", longs, 1.0), ("short", shorts, -1.0)):
        multipliers = pd.Series(1.0, index=symbols, dtype=float)
        present = [symbol for symbol in symbols if symbol in indexed.index]
        if present:
            source = indexed.loc[present]
            if not bool(source["side"].eq(side).all()):
                raise ValueError("feature side differs from the FSS3 funding-sign side")
            available = source[source["sfi_available"]].index.astype(str).tolist()
            if available:
                observed = source.loc[available, "tilt_multiplier"].to_numpy(dtype=float)
                if mode == "observed":
                    assigned = observed
                elif mode == "reversed":
                    assigned = np.sort(observed)[::-1]
                    order = np.argsort(observed, kind="stable")
                    mapped = np.empty_like(assigned)
                    mapped[order] = assigned
                    assigned = mapped
                else:
                    if rng is None:
                        raise ValueError("random overlay requires an RNG")
                    assigned = rng.permutation(observed)
                multipliers.loc[available] = assigned
        magnitude = multipliers / float(multipliers.sum()) * 0.5
        raw.update({symbol: sign * float(magnitude.at[symbol]) for symbol in symbols})
    return raw


def build_v222_path(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    cfg: V222Config = V222Config(),
    *,
    mode: str = "observed",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if mode not in {"baseline", "observed", "reversed", "random"}:
        raise ValueError(f"unsupported path mode: {mode}")
    feature_groups = {
        pd.Timestamp(entry): local
        for entry, local in features.groupby("entry_time", sort=False, observed=True)
    }
    labels = {
        "baseline": BASELINE_CONTROL,
        "observed": CANDIDATE,
        "reversed": REVERSED_CONTROL,
        "random": "SFO1_RANDOM_WITHIN_SIDE_RANK_TILT",
    }
    rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] | None = None
    previous_entry: pd.Timestamp | None = None
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        entry = pd.Timestamp(entry)
        active = entry in feature_groups
        base_target, _, _, negative_breadth, positive_breadth = funding_sign_target(
            local, cfg.minimum_side_breadth
        )
        target = base_target
        source_time = pd.NaT
        if active:
            source_time = pd.Timestamp(feature_groups[entry]["sfi_feature_time"].iloc[0])
        if active and mode != "baseline":
            raw = overlay_raw_target(
                local,
                feature_groups[entry],
                cfg,
                mode=mode,
                rng=rng,
            )
            target, _ = beta_neutral_components(local, raw)
        has_gap = previous_entry is not None and entry - previous_entry > pd.Timedelta(
            days=7, minutes=1
        )
        if has_gap and previous_weights is not None:
            rows[-1]["realized_turnover"] = float(rows[-1]["realized_turnover"]) + sum(
                abs(weight) for weight in previous_weights.values()
            )
            previous_weights = None
        if not target:
            if previous_weights is not None:
                rows[-1]["realized_turnover"] = float(
                    rows[-1]["realized_turnover"]
                ) + sum(abs(weight) for weight in previous_weights.values())
                previous_weights = None
            previous_entry = entry
            continue
        weights, components, fraction, turnover, breach = execute_capped_transition(
            local,
            previous_weights,
            target,
            cfg.transition_turnover_cap,
            cfg.bisection_iterations,
        )
        rows.append(
            {
                "candidate": labels[mode],
                "entry_time": entry,
                "exit_time": local["exit_time"].iloc[0],
                "month_start": local["month_start"].iloc[0],
                "period": local["period"].iloc[0],
                "overlay_active": active,
                "sfi_feature_time": source_time,
                "coverage": len(local),
                "negative_breadth": negative_breadth,
                "positive_breadth": positive_breadth,
                "executed_target_fraction": fraction,
                "target_tracking_l1": weight_turnover(weights, target),
                "cap_applicable": previous_weights is not None,
                "cap_binding": previous_weights is not None and fraction < 1.0 - 1e-10,
                "rebalance_turnover": turnover,
                "cap_breach": breach,
                "realized_turnover": turnover,
                "_weights": weights,
                **components,
            }
        )
        previous_weights = weights
        previous_entry = entry
    if rows and previous_weights is not None:
        rows[-1]["realized_turnover"] = float(rows[-1]["realized_turnover"]) + sum(
            abs(weight) for weight in previous_weights.values()
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["primary_net_return"] = (
        output["gross_return"] - cfg.one_way_cost * output["realized_turnover"]
    )
    output["stress_net_return"] = (
        output["gross_return"]
        - cfg.stress_one_way_cost * output["realized_turnover"]
    )
    return output


def build_v222_cm2(
    overlay: pd.DataFrame,
    cfg: V222Config = V222Config(),
) -> pd.DataFrame:
    tg1 = pd.read_parquet(cfg.tg1_path)
    for column in ("entry_time", "exit_time", "month_start"):
        tg1[column] = pd.to_datetime(tg1[column], utc=True, errors="coerce")
    tg1 = tg1[
        [
            "entry_time",
            "price_basis_return",
            "funding_spread_return",
            "primary_net_return",
            "stress_net_return",
        ]
    ].rename(
        columns={
            "price_basis_return": "tg1_price_return",
            "funding_spread_return": "tg1_funding_return",
            "primary_net_return": "tg1_primary_return",
            "stress_net_return": "tg1_stress_return",
        }
    )
    keep = overlay[
        [
            "entry_time",
            "exit_time",
            "month_start",
            "period",
            "overlay_active",
            "price_return",
            "funding_return",
            "primary_net_return",
            "stress_net_return",
        ]
    ].rename(
        columns={
            "price_return": "fss3_price_return",
            "funding_return": "fss3_funding_return",
            "primary_net_return": "fss3_primary_return",
            "stress_net_return": "fss3_stress_return",
        }
    )
    output = keep.merge(tg1, on="entry_time", how="inner", validate="one_to_one")
    output["candidate"] = CM2_CANDIDATE
    output["fss3_weight"] = cfg.fss3_weight
    output["tg1_weight"] = cfg.tg1_weight
    output["price_return"] = (
        cfg.fss3_weight * output["fss3_price_return"]
        + cfg.tg1_weight * output["tg1_price_return"]
    )
    output["funding_return"] = (
        cfg.fss3_weight * output["fss3_funding_return"]
        + cfg.tg1_weight * output["tg1_funding_return"]
    )
    output["primary_net_return"] = (
        cfg.fss3_weight * output["fss3_primary_return"]
        + cfg.tg1_weight * output["tg1_primary_return"]
    )
    output["stress_net_return"] = (
        cfg.fss3_weight * output["fss3_stress_return"]
        + cfg.tg1_weight * output["tg1_stress_return"]
    )
    return output


def _load_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for column in ("entry_time", "exit_time", "month_start"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def baseline_reconstruction_errors(
    reconstructed: pd.DataFrame,
    cfg: V222Config = V222Config(),
) -> pd.DataFrame:
    saved = _load_reference(cfg.fss3_path)
    columns = [
        "realized_turnover",
        "price_return",
        "funding_return",
        "gross_return",
        "primary_net_return",
        "stress_net_return",
        "gross_notional",
        "residual_btc_beta",
    ]
    merged = reconstructed[["entry_time", *columns]].merge(
        saved[["entry_time", *columns]],
        on="entry_time",
        how="outer",
        suffixes=("_reconstructed", "_saved"),
        indicator=True,
        validate="one_to_one",
    )
    rows = [
        {
            "check": "entry_key_set",
            "maximum_absolute_error": 0.0
            if bool(merged["_merge"].eq("both").all())
            else np.inf,
        }
    ]
    for column in columns:
        error = (
            merged[f"{column}_reconstructed"] - merged[f"{column}_saved"]
        ).abs()
        rows.append(
            {"check": column, "maximum_absolute_error": float(error.max())}
        )
    return pd.DataFrame(rows)


def build_v222_comparison(
    overlay: pd.DataFrame,
    baseline: pd.DataFrame,
    reversed_control: pd.DataFrame,
    cm2_overlay: pd.DataFrame,
    cfg: V222Config = V222Config(),
) -> pd.DataFrame:
    saved_cm2 = _load_reference(cfg.cm2_path)[
        ["entry_time", "primary_net_return", "stress_net_return"]
    ].rename(
        columns={
            "primary_net_return": "baseline_cm2_primary_return",
            "stress_net_return": "baseline_cm2_stress_return",
        }
    )
    frame = overlay[
        [
            "entry_time",
            "month_start",
            "period",
            "overlay_active",
            "price_return",
            "funding_return",
            "primary_net_return",
            "stress_net_return",
            "realized_turnover",
        ]
    ].rename(
        columns={column: f"overlay_{column}"
                 for column in (
                     "price_return",
                     "funding_return",
                     "primary_net_return",
                     "stress_net_return",
                     "realized_turnover",
                 )}
    )
    base = baseline[
        [
            "entry_time",
            "price_return",
            "funding_return",
            "primary_net_return",
            "stress_net_return",
            "realized_turnover",
        ]
    ].rename(
        columns={column: f"baseline_{column}"
                 for column in (
                     "price_return",
                     "funding_return",
                     "primary_net_return",
                     "stress_net_return",
                     "realized_turnover",
                 )}
    )
    reverse = reversed_control[["entry_time", "primary_net_return"]].rename(
        columns={"primary_net_return": "reversed_primary_net_return"}
    )
    cm2 = cm2_overlay[
        ["entry_time", "primary_net_return", "stress_net_return"]
    ].rename(
        columns={
            "primary_net_return": "overlay_cm2_primary_return",
            "stress_net_return": "overlay_cm2_stress_return",
        }
    )
    frame = (
        frame.merge(base, on="entry_time", validate="one_to_one")
        .merge(reverse, on="entry_time", validate="one_to_one")
        .merge(cm2, on="entry_time", validate="one_to_one")
        .merge(saved_cm2, on="entry_time", validate="one_to_one")
    )
    for outcome in ("price_return", "funding_return", "primary_net_return", "stress_net_return", "realized_turnover"):
        frame[f"incremental_{outcome}"] = (
            frame[f"overlay_{outcome}"] - frame[f"baseline_{outcome}"]
        )
    frame["reversed_incremental_primary_return"] = (
        frame["reversed_primary_net_return"] - frame["baseline_primary_net_return"]
    )
    frame["incremental_cm2_primary_return"] = (
        frame["overlay_cm2_primary_return"] - frame["baseline_cm2_primary_return"]
    )
    frame["incremental_cm2_stress_return"] = (
        frame["overlay_cm2_stress_return"] - frame["baseline_cm2_stress_return"]
    )
    return frame


def build_v222_nulls(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    baseline: pd.DataFrame,
    cfg: V222Config = V222Config(),
) -> pd.DataFrame:
    all_symbols = sorted(panel["symbol"].astype(str).unique())
    symbol_to_index = {symbol: index for index, symbol in enumerate(all_symbols)}
    feature_groups = {
        pd.Timestamp(entry): local.set_index("symbol", verify_integrity=True)
        for entry, local in features.groupby("entry_time", sort=False, observed=True)
    }
    weeks: list[dict[str, object]] = []
    for entry, local in panel.groupby("entry_time", sort=True, observed=True):
        entry = pd.Timestamp(entry)
        eligible = local.dropna(
            subset=["score_7d", "price_return", "future_funding", "btc_beta"]
        )
        longs = sorted(
            eligible.loc[eligible["score_7d"].lt(0), "symbol"]
            .astype(str)
            .unique()
        )
        shorts = sorted(
            eligible.loc[eligible["score_7d"].gt(0), "symbol"]
            .astype(str)
            .unique()
        )
        indexed = eligible.set_index("symbol")
        symbols = sorted(eligible["symbol"].astype(str).unique())
        indices = np.asarray([symbol_to_index[symbol] for symbol in symbols], dtype=int)
        beta = np.zeros(len(all_symbols), dtype=float)
        price = np.zeros(len(all_symbols), dtype=float)
        funding = np.zeros(len(all_symbols), dtype=float)
        beta[indices] = indexed.loc[symbols, "btc_beta"].to_numpy(dtype=float)
        price[indices] = indexed.loc[symbols, "price_return"].to_numpy(dtype=float)
        funding[indices] = indexed.loc[symbols, "future_funding"].to_numpy(dtype=float)
        mask = np.zeros(len(all_symbols), dtype=bool)
        mask[indices] = True
        side_specs: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        if entry in feature_groups:
            feature = feature_groups[entry]
            for side, side_symbols, sign in (
                ("long", longs, 1.0),
                ("short", shorts, -1.0),
            ):
                side_indices = np.asarray(
                    [symbol_to_index[symbol] for symbol in side_symbols], dtype=int
                )
                present = [symbol for symbol in side_symbols if symbol in feature.index]
                source = feature.loc[present]
                if not bool(source["side"].eq(side).all()):
                    raise ValueError("feature side differs from the FSS3 funding-sign side")
                available = source[source["sfi_available"]]
                available_indices = np.asarray(
                    [symbol_to_index[symbol] for symbol in available.index], dtype=int
                )
                observed_multipliers = available["tilt_multiplier"].to_numpy(
                    dtype=float
                )
                side_specs.append(
                    (side_indices, available_indices, sign, observed_multipliers)
                )
        weeks.append(
            {
                "entry_time": entry,
                "active": entry in feature_groups,
                "long_indices": np.asarray(
                    [symbol_to_index[symbol] for symbol in longs], dtype=int
                ),
                "short_indices": np.asarray(
                    [symbol_to_index[symbol] for symbol in shorts], dtype=int
                ),
                "side_specs": side_specs,
                "beta": beta,
                "price": price,
                "funding": funding,
                "mask": mask,
                "btc_return": float(eligible.iloc[0]["btc_return"]),
                "btc_funding": float(eligible.iloc[0]["btc_future_funding"]),
            }
        )
    baseline_indexed = baseline.set_index("entry_time")
    baseline_values = baseline_indexed.loc[
        [week["entry_time"] for week in weeks], "primary_net_return"
    ].to_numpy(dtype=float)
    active_mask = np.asarray([bool(week["active"]) for week in weeks], dtype=bool)
    rows: list[dict[str, object]] = []
    for iteration in range(cfg.null_iterations):
        rng = np.random.default_rng(cfg.seed + 1 + iteration * 1009)
        previous_alt: np.ndarray | None = None
        previous_btc = 0.0
        gross_returns: list[float] = []
        turnovers: list[float] = []
        for week in weeks:
            raw = np.zeros(len(all_symbols), dtype=float)
            if week["active"]:
                for side_indices, available_indices, sign, observed in week[
                    "side_specs"
                ]:
                    multipliers = np.ones(len(side_indices), dtype=float)
                    position = {
                        int(symbol_index): index
                        for index, symbol_index in enumerate(side_indices)
                    }
                    assigned = rng.permutation(observed)
                    for symbol_index, multiplier in zip(
                        available_indices, assigned, strict=True
                    ):
                        multipliers[position[int(symbol_index)]] = multiplier
                    raw[side_indices] = sign * 0.5 * multipliers / multipliers.sum()
            else:
                long_indices = week["long_indices"]
                short_indices = week["short_indices"]
                raw[long_indices] = 0.5 / len(long_indices)
                raw[short_indices] = -0.5 / len(short_indices)
            target_alt, target_btc = _neutralize_alt_array(raw, week["beta"])
            alt, btc, turnover = _execute_capped_array_transition(
                previous_alt,
                previous_btc,
                target_alt,
                target_btc,
                week["beta"],
                week["mask"],
                cfg.transition_turnover_cap,
                cfg.bisection_iterations,
            )
            gross_returns.append(
                float(
                    np.dot(alt, week["price"])
                    + btc * week["btc_return"]
                    - np.dot(alt, week["funding"])
                    - btc * week["btc_funding"]
                )
            )
            turnovers.append(turnover)
            previous_alt = alt
            previous_btc = btc
        if turnovers and previous_alt is not None:
            turnovers[-1] += float(np.abs(previous_alt).sum() + abs(previous_btc))
        primary = np.asarray(gross_returns) - cfg.one_way_cost * np.asarray(turnovers)
        increment = primary[active_mask] - baseline_values[active_mask]
        rows.append(
            {
                "iteration": iteration,
                "null_type": "within_week_within_side_random_sfi_rank",
                "active_weeks": int(active_mask.sum()),
                "mean_active_primary_increment": float(increment.mean()),
                "mean_full_path_primary_return": float(primary.mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_v222(
    overlay: pd.DataFrame,
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    cm2_overlay: pd.DataFrame,
    nulls: pd.DataFrame,
    reconstruction_errors: pd.DataFrame,
    cfg: V222Config = V222Config(),
) -> pd.DataFrame:
    active = comparison[comparison["overlay_active"]].sort_values("entry_time")
    increment = active["incremental_primary_net_return"].to_numpy(dtype=float)
    draws = _moving_block_means(
        increment,
        cfg.bootstrap_iterations,
        cfg.bootstrap_block_weeks,
        np.random.default_rng(cfg.seed + 2),
    )
    bootstrap_low, bootstrap_high = np.quantile(draws, [0.025, 0.975])
    periods = active.groupby("period", observed=True)[
        "incremental_primary_net_return"
    ].mean()
    counts = active["period"].value_counts()
    monthly = active.groupby("month_start", observed=True)[
        "incremental_primary_net_return"
    ].sum()
    positive = monthly[monthly.gt(0)]
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else np.inf
    )
    leave_one_month_out = [
        float(
            active.loc[
                active["month_start"].ne(month), "incremental_primary_net_return"
            ].mean()
        )
        for month in monthly.index
    ]
    cm2_active = cm2_overlay[cm2_overlay["overlay_active"]].sort_values("entry_time")
    saved_cm2 = _load_reference(cfg.cm2_path).set_index("entry_time")
    saved_cm2_active = saved_cm2.loc[cm2_active["entry_time"]]
    overlay_drawdown = _additive_max_drawdown(cm2_active["primary_net_return"])
    baseline_drawdown = _additive_max_drawdown(
        saved_cm2_active["primary_net_return"]
    )
    applicable = overlay[overlay["cap_applicable"]]
    observed_mean = float(increment.mean())
    row: dict[str, object] = {
        "candidate": CANDIDATE,
        "path_weeks": len(overlay),
        "active_weeks": len(active),
        "active_months": active["month_start"].nunique(),
        "development_active_weeks": int(counts.get("development", 0)),
        "validation_active_weeks": int(counts.get("validation", 0)),
        "holdout_active_weeks": int(counts.get("holdout", 0)),
        "baseline_reconstruction_max_error": float(
            reconstruction_errors["maximum_absolute_error"].max()
        ),
        "full_path_overlay_fss3_primary_bp": float(
            overlay["primary_net_return"].mean() * 10_000
        ),
        "full_path_baseline_fss3_primary_bp": float(
            baseline["primary_net_return"].mean() * 10_000
        ),
        "full_path_fss3_increment_bp": float(
            comparison["incremental_primary_net_return"].mean() * 10_000
        ),
        "active_fss3_primary_increment_bp": observed_mean * 10_000,
        "active_fss3_stress_increment_bp": float(
            active["incremental_stress_net_return"].mean() * 10_000
        ),
        "active_cm2_primary_increment_bp": float(
            active["incremental_cm2_primary_return"].mean() * 10_000
        ),
        "active_cm2_stress_increment_bp": float(
            active["incremental_cm2_stress_return"].mean() * 10_000
        ),
        "active_price_increment_bp": float(
            active["incremental_price_return"].mean() * 10_000
        ),
        "active_funding_increment_bp": float(
            active["incremental_funding_return"].mean() * 10_000
        ),
        "development_primary_increment_bp": float(
            periods.get("development", np.nan) * 10_000
        ),
        "validation_primary_increment_bp": float(
            periods.get("validation", np.nan) * 10_000
        ),
        "holdout_primary_increment_bp": float(
            periods.get("holdout", np.nan) * 10_000
        ),
        "bootstrap_95_low_increment_bp": float(bootstrap_low * 10_000),
        "bootstrap_95_high_increment_bp": float(bootstrap_high * 10_000),
        "random_null_percentile": float(
            100 * nulls["mean_active_primary_increment"].le(observed_mean).mean()
        ),
        "reversed_active_primary_increment_bp": float(
            active["reversed_incremental_primary_return"].mean() * 10_000
        ),
        "mean_overlay_turnover": float(overlay["realized_turnover"].mean()),
        "mean_baseline_turnover": float(baseline["realized_turnover"].mean()),
        "mean_turnover_increment": float(
            overlay["realized_turnover"].mean()
            - baseline["realized_turnover"].mean()
        ),
        "max_capped_transition_turnover": float(
            applicable["rebalance_turnover"].max()
        ),
        "max_cap_breach": float(overlay["cap_breach"].max()),
        "max_abs_residual_btc_beta": float(
            overlay["residual_btc_beta"].abs().max()
        ),
        "max_gross_notional_drift": float(
            (overlay["gross_notional"] - 1.0).abs().max()
        ),
        "positive_month_increment_concentration": concentration,
        "minimum_leave_one_month_out_increment_bp": float(
            min(leave_one_month_out) * 10_000
        ),
        "overlay_cm2_active_drawdown_bp": float(overlay_drawdown * 10_000),
        "baseline_cm2_active_drawdown_bp": float(baseline_drawdown * 10_000),
        "cm2_active_drawdown_worsening_bp": float(
            (abs(overlay_drawdown) - abs(baseline_drawdown)) * 10_000
        ),
    }
    row["promote"] = bool(
        row["path_weeks"] == 49
        and row["active_weeks"] == 35
        and row["active_months"] == 9
        and row["development_active_weeks"] == 16
        and row["validation_active_weeks"] == 11
        and row["holdout_active_weeks"] == 8
        and row["baseline_reconstruction_max_error"] <= 1e-12
        and all(
            float(row[key]) > 0
            for key in (
                "active_fss3_primary_increment_bp",
                "active_fss3_stress_increment_bp",
                "active_cm2_primary_increment_bp",
                "active_cm2_stress_increment_bp",
                "development_primary_increment_bp",
                "validation_primary_increment_bp",
                "holdout_primary_increment_bp",
                "bootstrap_95_low_increment_bp",
                "minimum_leave_one_month_out_increment_bp",
            )
        )
        and row["random_null_percentile"] >= 95
        and row["active_fss3_primary_increment_bp"]
        > row["reversed_active_primary_increment_bp"]
        and row["mean_turnover_increment"] <= 0.10
        and row["max_capped_transition_turnover"]
        <= cfg.transition_turnover_cap + 1e-10
        and row["max_cap_breach"] <= 1e-10
        and row["max_abs_residual_btc_beta"] <= 1e-12
        and row["max_gross_notional_drift"] <= 1e-12
        and row["positive_month_increment_concentration"] <= 0.35
        and row["cm2_active_drawdown_worsening_bp"] <= 200
    )
    return pd.DataFrame([row])


def write_v222_sfi_fss3_overlay(
    cfg: V222Config = V222Config(),
) -> dict[str, Path]:
    panel, features = load_v222_inputs(cfg)
    baseline = build_v222_path(panel, features, cfg, mode="baseline")
    overlay = build_v222_path(panel, features, cfg, mode="observed")
    reversed_control = build_v222_path(panel, features, cfg, mode="reversed")
    cm2 = build_v222_cm2(overlay, cfg)
    reconstruction = baseline_reconstruction_errors(baseline, cfg)
    comparison = build_v222_comparison(
        overlay, baseline, reversed_control, cm2, cfg
    )
    nulls = build_v222_nulls(panel, features, baseline, cfg)
    summary = summarize_v222(
        overlay, baseline, comparison, cm2, nulls, reconstruction, cfg
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "portfolio": root / "weekly_overlay_fss3.parquet",
        "weights": root / "weekly_overlay_weights.parquet",
        "baseline": root / "weekly_zero_tilt_reconstruction.parquet",
        "reversed": root / "weekly_reversed_control.parquet",
        "cm2": root / "weekly_overlay_cm2.parquet",
        "comparison": root / "weekly_increment_comparison.parquet",
        "nulls": root / "random_rank_nulls.csv",
        "reconstruction": root / "baseline_reconstruction_errors.csv",
        "summary": root / "summary.csv",
        "metadata": root / "metadata.json",
        "findings": cfg.findings_path,
    }
    weight_rows = [
        {
            "entry_time": row["entry_time"],
            "symbol": symbol,
            "weight": weight,
            "is_btc_hedge": symbol == BTC,
            "overlay_active": row["overlay_active"],
        }
        for _, row in overlay.iterrows()
        for symbol, weight in row["_weights"].items()
    ]
    pd.DataFrame(weight_rows).to_parquet(paths["weights"], index=False)
    overlay.drop(columns="_weights").to_parquet(paths["portfolio"], index=False)
    baseline.drop(columns="_weights").to_parquet(paths["baseline"], index=False)
    reversed_control.drop(columns="_weights").to_parquet(paths["reversed"], index=False)
    cm2.to_parquet(paths["cm2"], index=False)
    comparison.to_parquet(paths["comparison"], index=False)
    nulls.to_csv(paths["nulls"], index=False)
    reconstruction.to_csv(paths["reconstruction"], index=False)
    summary.to_csv(paths["summary"], index=False)
    serialized = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }
    paths["metadata"].write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "feature_sha256": _sha256(cfg.feature_path),
                "promoted": summary.loc[summary["promote"], "candidate"].tolist(),
                "config": serialized,
                "permissions_changed": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict = (
        "promote_overlay_research_candidate"
        if bool(summary.iloc[0]["promote"])
        else "reject_strategy_overlay"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v22.2 SFI-on-FSS3 Overlay Findings",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "The sole preregistered 0.50 within-side rank tilt was evaluated; no ",
                "tilt grid, return-conditioned fallback, or extra trade cycle was used.",
                "",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths
