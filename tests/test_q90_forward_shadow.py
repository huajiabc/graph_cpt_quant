from pathlib import Path

import pandas as pd

from pressure_graph.paper_live.q90 import (
    Q90LiveConfig,
    next_q90_book_day,
)


def _cfg(tmp_path: Path) -> Q90LiveConfig:
    return Q90LiveConfig(
        base_config=Path("configs/v0_3.yaml"),
        historical_feature_root=tmp_path / "historical",
        forward_root=tmp_path / "forward",
        report_root=tmp_path / "report",
        historical_cutoff=pd.Timestamp("2026-07-15", tz="UTC"),
        first_forward_book_day=pd.Timestamp("2026-07-15", tz="UTC"),
    )


def test_q90_first_archive_day_is_due_after_publication_lag(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    assert next_q90_book_day(
        cfg, pd.Timestamp("2026-07-16 01:00", tz="UTC")
    ) == pd.Timestamp("2026-07-15", tz="UTC")


def test_q90_archive_shadow_is_never_execution_timely(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    assert not cfg.timely_execution_eligible
    assert cfg.forward_research_evidence_eligible
    assert not cfg.real_orders_allowed
    assert not cfg.leverage_allowed
