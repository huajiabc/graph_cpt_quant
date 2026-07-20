from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.paper_live.fss3 import (
    FSS3LiveConfig,
    build_fss3_decision_input,
    build_fss3_weights,
    write_fss3_forward_shadow,
)


def _cfg(tmp_path: Path) -> FSS3LiveConfig:
    return FSS3LiveConfig(
        base_config=Path("configs/v0_3.yaml"),
        live_root=tmp_path / "data",
        membership_path=tmp_path / "membership.csv",
        seed_weights_path=tmp_path / "seed.parquet",
        report_root=tmp_path / "reports",
    )


def test_fss3_live_execution_reproduces_last_saved_research_transition(
    tmp_path: Path,
) -> None:
    panel = pd.read_parquet(
        "reports/v13_4_negative_funding_beta_neutral_rebound/weekly_symbol_panel.parquet"
    )
    panel["entry_time"] = pd.to_datetime(panel["entry_time"], utc=True)
    current_time = panel["entry_time"].max()
    previous_time = current_time - pd.Timedelta(days=7)
    local = panel[panel["entry_time"].eq(current_time)].copy()
    saved = pd.read_parquet(
        "reports/v14_9_funding_sign_turnover_cap/weekly_weights.parquet"
    )
    saved["entry_time"] = pd.to_datetime(saved["entry_time"], utc=True)
    previous = {
        str(row.symbol): float(row.weight)
        for row in saved[saved["entry_time"].eq(previous_time)].itertuples(index=False)
    }
    expected = {
        str(row.symbol): float(row.weight)
        for row in saved[saved["entry_time"].eq(current_time)].itertuples(index=False)
    }
    actual, details = build_fss3_weights(local, previous, _cfg(tmp_path))
    assert set(actual) == set(expected)
    assert max(abs(actual[symbol] - expected[symbol]) for symbol in expected) < 1e-12
    assert np.isclose(details["rebalance_turnover"], 0.70)
    assert details["cap_binding"]
    assert abs(details["residual_btc_beta"]) < 1e-12


def test_fss3_decision_input_excludes_decision_time_funding(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    decision = pd.Timestamp("2026-02-02 00:00", tz="UTC")
    month = pd.Timestamp("2026-02-01", tz="UTC")
    symbols = [f"S{i}" for i in range(8)]
    membership = pd.DataFrame(
        {"month_start": month, "symbol": symbols}
    )
    times = pd.date_range(
        pd.Timestamp("2026-01-01 00:00", tz="UTC"), decision, freq="h"
    )
    rng = np.random.default_rng(7)
    btc_returns = rng.normal(0, 0.003, len(times))
    frames = [
        pd.DataFrame(
            {
                "symbol": "BTCUSDT",
                "feature_time": times,
                "close": 100 * np.cumprod(1 + btc_returns),
            }
        )
    ]
    for index, symbol in enumerate(symbols):
        residual = rng.normal(0, 0.001, len(times))
        returns = (0.8 + index / 20) * btc_returns + residual
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "feature_time": times,
                    "close": 20 * np.cumprod(1 + returns),
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    funding_rows = []
    for index, symbol in enumerate(symbols):
        sign = -1.0 if index < 4 else 1.0
        funding_rows.extend(
            [
                {
                    "symbol": symbol,
                    "funding_time": decision - pd.Timedelta(hours=8),
                    "funding_rate_settled": sign * 0.001,
                },
                {
                    "symbol": symbol,
                    "funding_time": decision,
                    "funding_rate_settled": -sign * 100.0,
                },
            ]
        )
    local, metadata = build_fss3_decision_input(
        pd.DataFrame(funding_rows), prices, membership, decision, cfg
    )
    assert metadata["reasons"] == ()
    assert (local.sort_values("symbol")["score_7d"].iloc[:4] < 0).all()
    assert (local.sort_values("symbol")["score_7d"].iloc[4:] > 0).all()
    assert local["score_7d"].abs().max() == 0.001


def test_fss3_record_only_config_rejects_order_permission(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.mode == "live_shadow"
    assert cfg.push_policy == "record_only"
    assert cfg.real_orders_allowed is False
    assert cfg.leverage_allowed is False


def test_fss3_forward_shadow_is_idempotent_and_preserves_first_observation(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    decision = pd.Timestamp("2026-02-02 00:00", tz="UTC")
    seed_time = decision - pd.Timedelta(days=7)
    month = pd.Timestamp("2026-02-01", tz="UTC")
    symbols = [f"S{i}" for i in range(8)]
    membership = pd.DataFrame({"month_start": month, "symbol": symbols})
    membership.to_csv(cfg.membership_path, index=False)
    pd.DataFrame(
        [
            {
                "entry_time": seed_time,
                "symbol": symbol,
                "weight": (0.5 / 4) if index < 4 else (-0.5 / 4),
                "is_btc_hedge": False,
            }
            for index, symbol in enumerate(symbols)
        ]
        + [
            {
                "entry_time": seed_time,
                "symbol": "BTCUSDT",
                "weight": 0.0,
                "is_btc_hedge": True,
            }
        ]
    ).to_parquet(cfg.seed_weights_path, index=False)

    times = pd.date_range(
        pd.Timestamp("2026-01-01 00:00", tz="UTC"), decision, freq="h"
    )
    rng = np.random.default_rng(11)
    btc_returns = rng.normal(0, 0.003, len(times))
    price_frames = [
        pd.DataFrame(
            {
                "symbol": "BTCUSDT",
                "feature_time": times,
                "close": 100 * np.cumprod(1 + btc_returns),
            }
        )
    ]
    for index, symbol in enumerate(symbols):
        returns = (0.7 + index / 20) * btc_returns + rng.normal(
            0, 0.001, len(times)
        )
        price_frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "feature_time": times,
                    "close": 20 * np.cumprod(1 + returns),
                }
            )
        )
    prices = pd.concat(price_frames, ignore_index=True)
    funding_times = pd.date_range(
        decision - pd.Timedelta(days=7),
        decision - pd.Timedelta(hours=8),
        freq="8h",
        tz="UTC",
    )
    funding = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "funding_time": funding_times,
                    "funding_rate_settled": (
                        -0.0001 if index < 4 else 0.0001
                    ),
                }
            )
            for index, symbol in enumerate(symbols)
        ]
        + [
            pd.DataFrame(
                {
                    "symbol": "BTCUSDT",
                    "funding_time": funding_times,
                    "funding_rate_settled": 0.00005,
                }
            )
        ],
        ignore_index=True,
    )
    first_observed = decision + pd.Timedelta(minutes=30)
    write_fss3_forward_shadow(
        funding, prices, membership, cfg, observed_at=first_observed
    )
    write_fss3_forward_shadow(
        funding,
        prices,
        membership,
        cfg,
        observed_at=decision + pd.Timedelta(hours=1),
    )
    decisions = pd.read_parquet(cfg.report_root / "forward" / "decisions.parquet")
    weights = pd.read_parquet(
        cfg.report_root / "forward" / "executed_weights.parquet"
    )
    status = pd.read_json(cfg.report_root / "live_status.json", typ="series")
    assert len(decisions) == 1
    assert pd.Timestamp(decisions.loc[0, "first_observed_at_utc"]) == first_observed
    assert bool(decisions.loc[0, "timely_forward_decision"])
    assert np.isclose(weights["weight"].abs().sum(), 1.0)
    assert status["push_policy"] == "record_only"
    assert not bool(status["real_orders_allowed"])
    assert not list(cfg.report_root.rglob("*order*"))
