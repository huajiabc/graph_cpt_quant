"""Feature-only interpolation audit at the single frozen q85 pressure threshold."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)
from pressure_graph.reports.v2311_positive_q80_vacuum_breakout_feature_audit import (
    V2311Config,
    audit_v2311_features,
    build_v2311_features,
    summarize_v2311,
)


REPORT_ROOT = Path("reports/v23_14_positive_q85_vacuum_breakout_feature_audit")
FINDINGS_PATH = Path(
    "docs/v2314_positive_q85_vacuum_breakout_feature_audit_2026_07_17.md"
)
CANDIDATE = "DVB7_POSITIVE_Q85_VACUUM_0625SIGMA_BREAKOUT"


@dataclass(frozen=True)
class V2314Config:
    report_root: Path = REPORT_ROOT
    findings_path: Path = FINDINGS_PATH
    pressure_quantile: float = 0.85
    minimum_events: int = 70
    minimum_period_events: int = 20
    minimum_active_months: int = 12


def _base_config(cfg: V2314Config) -> V2311Config:
    return V2311Config(
        report_root=cfg.report_root,
        findings_path=cfg.findings_path,
        candidate=CANDIDATE,
        pressure_quantile=cfg.pressure_quantile,
        minimum_events=cfg.minimum_events,
        minimum_period_events=cfg.minimum_period_events,
        minimum_active_months=cfg.minimum_active_months,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_v2314_positive_q85_vacuum_breakout_feature_audit(
    cfg: V2314Config = V2314Config(),
) -> dict[str, Path]:
    base = _base_config(cfg)
    _, states, features = build_v2311_features(base)
    summary = summarize_v2311(features, base)
    checks = audit_v2311_features(states, features, summary, base)
    hashes = pd.DataFrame(
        [
            {
                "input": str(base.feature_root / f"{symbol}.parquet"),
                "sha256": _sha256(base.feature_root / f"{symbol}.parquet"),
            }
            for symbol in FROZEN_SYMBOLS
        ]
    )
    root = ensure_dir(cfg.report_root)
    paths = {
        "features": root / "positive_q85_breakout_features.parquet",
        "bucket_states": root / "hourly_q85_bucket_states.parquet",
        "summary": root / "feature_coverage_summary.csv",
        "checks": root / "data_quality_checks.csv",
        "hashes": root / "input_hashes.csv",
        "findings": cfg.findings_path,
    }
    features.to_parquet(paths["features"], index=False)
    states.to_parquet(paths["bucket_states"], index=False)
    summary.to_csv(paths["summary"], index=False)
    checks.to_csv(paths["checks"], index=False)
    hashes.to_csv(paths["hashes"], index=False)
    verdict = (
        "feature_viable_freeze_single_q85_interpolation"
        if bool(checks["passed"].all())
        else "feature_audit_failed"
    )
    paths["findings"].write_text(
        "\n".join(
            [
                "# v23.14 Positive-q85 Vacuum Breakout Feature Audit",
                "",
                f"Verdict: `{verdict}`.",
                "",
                summary.to_markdown(index=False, floatfmt=".4f"),
                "",
                "q85 is the sole predeclared interpolation between the rejected",
                "q80 density extension and the post-selected q90 tail candidate.",
                "All other breadth, withdrawal, transition, cooldown, BTC sigma,",
                "barrier, and path-coverage rules are unchanged.",
                "",
                "No post-entry trigger, fill, high/low path, or return was used.",
                "No live, PaperLive, leverage, remote, application, or order state changed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V2314Config",
    "write_v2314_positive_q85_vacuum_breakout_feature_audit",
]
