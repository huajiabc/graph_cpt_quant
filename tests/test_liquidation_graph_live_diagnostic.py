from pathlib import Path

from pressure_graph.paper_live.liquidation_graph import (
    LiquidationGraphLiveConfig,
)


def test_liquidation_graph_live_diagnostic_has_no_strategy_permissions(
    tmp_path: Path,
) -> None:
    cfg = LiquidationGraphLiveConfig(
        live_root=tmp_path / "live",
        source_root=tmp_path / "source",
        contract_root=tmp_path / "contract",
        report_root=tmp_path / "report",
        ledger_path=tmp_path / "ledger.parquet",
    )
    assert cfg.scope == "live_shadow"
    assert cfg.push_policy == "record_only"
    assert not cfg.real_orders_allowed
    assert not cfg.leverage_allowed
    assert cfg.minimum_hourly_decisions == 336
    assert cfg.minimum_utc_days == 14
