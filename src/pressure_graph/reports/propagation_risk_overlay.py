"""Propagation-graph-node risk overlay (consumer of Phase 3 graph node).

Phase 3 registered ``bybit:DOGEUSDT → binance_um:DOGEUSDT @ E1`` as a
diagnostic propagation graph node (NOT a tradeable short). The goal docx
prescribes four use cases for that node — this module implements the first
of them:

    "Binance DOGE long risk warning"

Mechanism: for each registered node, watch the SOURCE-venue CVD for the
node's event class. When an event fires AND a long position is held on the
TARGET venue/symbol within a configurable window, emit a risk flag whose
severity is tied to the node's ``status_statistically_real`` field:

- node statistically_real == True  → severity "elevated"
- node statistically_real == False → severity "informational"  (regime_event)
- node tradeable_after_cost == True → severity "elevated" trumps to "actionable"

This module **does not** veto entries or change position sizes. It writes a
log artefact the operator can correlate with realised long-side P&L during
stress regimes (e.g. 2025-10). The wiring matches the docx's caveat that
even a Phase-3-validated edge stays "diagnostic graph node, not a real
short" until every gate independently passes.

Architecture is intentionally symmetric to ``sell_pressure_propagation.py``:

- The event detector is the same ``detect_events`` from Phase 1; no
  duplicate detection logic.
- The "is the target long?" check is a thin probe over a positions frame
  (``symbol, position_open_time, position_close_time`` rows) so the operator
  can plug in a real-positions SQLite export OR a synthetic always-long
  frame for footprint analysis.
- The output is long-format DataFrame + markdown so existing CSV-consuming
  tools see the same shape Phase 3 produces.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir
from pressure_graph.reports.sell_pressure_propagation import (
    PropagationConfig,
    detect_events,
    load_continuous_cvd,
)


SEVERITY_INFORMATIONAL = "informational"
SEVERITY_ELEVATED = "elevated"
SEVERITY_ACTIONABLE = "actionable"


@dataclass(frozen=True)
class PropagationOverlayConfig:
    """Knobs the operator can tune without re-running Phase 3."""

    # How long a source event stays "active" as a risk flag — matches the
    # node's published lag windows (5/15/30 min). At 5min bars this is 6 bars.
    flag_lookforward_bars: int = 6

    # An "always-long" synthetic positions frame is the default for footprint
    # analysis. Set this False to require a real positions frame.
    treat_as_always_long: bool = True

    # The PropagationConfig used at event-detection time. Should match the
    # config the graph node was built with so the event count is comparable.
    cfg: PropagationConfig = PropagationConfig()


@dataclass(frozen=True)
class RiskFlag:
    """One emitted risk flag — long-form for easy concat / CSV."""

    edge_id: str
    source_label: str
    target_label: str
    source_symbol: str
    target_symbol: str
    event_type: str
    bar_open_time: pd.Timestamp
    severity: str
    statistically_real: bool
    tradeable_after_cost: bool
    target_long_active: bool
    lag_minutes: tuple[int, ...]


# -- Node parsing ------------------------------------------------------------------


def load_propagation_nodes(path: Path) -> list[dict]:
    """Read the graph_nodes.json artefact written by Phase 3.

    The on-disk schema is a top-level list of node dicts. Missing files
    return ``[]`` rather than raising — the operator may run the overlay
    before Phase 3 has produced its artefact yet.
    """
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    payload = json.loads(raw)
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def node_event_type(node: dict) -> str:
    """Parse the event_type out of the node's edge_id (``...->...@E1`` → ``E1``)."""
    edge_id = str(node.get("edge_id", ""))
    if "@" in edge_id:
        return edge_id.rsplit("@", 1)[-1]
    return "E1"


def node_severity(node: dict) -> str:
    """Map the node's verdict to a risk severity per the docx grade ladder."""
    real = bool(node.get("status_statistically_real", False))
    tradeable = bool(node.get("status_tradeable_after_cost", False))
    if tradeable:
        return SEVERITY_ACTIONABLE
    if real:
        return SEVERITY_ELEVATED
    return SEVERITY_INFORMATIONAL


def _split_label(label: str) -> tuple[str, str]:
    """Split a 'venue:symbol' label; default to ('', label) if no colon."""
    if ":" not in label:
        return "", label
    venue, symbol = label.split(":", 1)
    return venue, symbol


# -- Positions probe ---------------------------------------------------------------


def _normalize_positions(positions: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce a positions frame to the columns the overlay expects.

    Expected schema:
        symbol (str), position_open_time (UTC datetime), position_close_time
        (UTC datetime, NaT for open positions)

    Missing frames or empty frames → empty result.
    """
    if positions is None or positions.empty:
        return pd.DataFrame(
            columns=["symbol", "position_open_time", "position_close_time"]
        )
    out = positions.copy()
    if "symbol" not in out.columns:
        raise ValueError("positions frame missing 'symbol' column")
    if "position_open_time" not in out.columns:
        raise ValueError("positions frame missing 'position_open_time' column")
    out["position_open_time"] = pd.to_datetime(out["position_open_time"], utc=True)
    if "position_close_time" not in out.columns:
        out["position_close_time"] = pd.NaT
    else:
        out["position_close_time"] = pd.to_datetime(
            out["position_close_time"], utc=True
        )
    return out[["symbol", "position_open_time", "position_close_time"]]


def is_long_at(positions: pd.DataFrame, symbol: str, ts: pd.Timestamp) -> bool:
    """True if ``positions`` shows an open long on ``symbol`` at ``ts``."""
    if positions.empty:
        return False
    relevant = positions[positions["symbol"] == symbol]
    if relevant.empty:
        return False
    open_match = relevant["position_open_time"] <= ts
    close_match = relevant["position_close_time"].isna() | (
        relevant["position_close_time"] > ts
    )
    return bool((open_match & close_match).any())


# -- Overlay driver ----------------------------------------------------------------


def emit_flags_for_node(
    node: dict,
    source_cvd: pd.DataFrame,
    positions: pd.DataFrame,
    overlay_cfg: PropagationOverlayConfig,
) -> list[RiskFlag]:
    """Detect source events and emit one RiskFlag per (event, lookforward-bar)
    where a long is held on the target.

    If ``overlay_cfg.treat_as_always_long`` is True, the long check is bypassed
    and every event emits one flag at the event's own bar (so the operator
    sees the raw footprint — the lookforward expansion is only useful when
    cross-referencing real position windows).
    """
    src_label = str(node.get("edge", "").split("->", 1)[0]).strip() or str(
        node.get("source_label", "")
    )
    tgt_label = str(node.get("edge", "").split("->", 1)[-1]).strip() or str(
        node.get("target_label", "")
    )
    src_venue, src_symbol = _split_label(src_label)
    tgt_venue, tgt_symbol = _split_label(tgt_label)
    event_type = node_event_type(node)
    severity = node_severity(node)
    statistically_real = bool(node.get("status_statistically_real", False))
    tradeable_after_cost = bool(node.get("status_tradeable_after_cost", False))
    lag_minutes = tuple(int(x) for x in node.get("lag_minutes", []))

    events = detect_events(source_cvd, overlay_cfg.cfg)
    if event_type not in events.columns:
        return []
    event_mask = events[event_type].to_numpy()
    bar_times = pd.to_datetime(source_cvd["bar_open_time"]).reset_index(drop=True)

    flags: list[RiskFlag] = []
    n = len(bar_times)
    look = max(0, int(overlay_cfg.flag_lookforward_bars))
    for i in range(n):
        if not event_mask[i]:
            continue
        if overlay_cfg.treat_as_always_long:
            flags.append(
                RiskFlag(
                    edge_id=str(node.get("edge_id", f"{src_label}->{tgt_label}@{event_type}")),
                    source_label=src_label,
                    target_label=tgt_label,
                    source_symbol=src_symbol,
                    target_symbol=tgt_symbol,
                    event_type=event_type,
                    bar_open_time=bar_times.iloc[i],
                    severity=severity,
                    statistically_real=statistically_real,
                    tradeable_after_cost=tradeable_after_cost,
                    target_long_active=True,
                    lag_minutes=lag_minutes,
                )
            )
            continue
        # Real positions: emit one flag per lookforward bar where target is long
        for k in range(look + 1):
            j = i + k
            if j >= n:
                break
            ts = bar_times.iloc[j]
            if not is_long_at(positions, tgt_symbol, ts):
                continue
            flags.append(
                RiskFlag(
                    edge_id=str(node.get("edge_id", f"{src_label}->{tgt_label}@{event_type}")),
                    source_label=src_label,
                    target_label=tgt_label,
                    source_symbol=src_symbol,
                    target_symbol=tgt_symbol,
                    event_type=event_type,
                    bar_open_time=ts,
                    severity=severity,
                    statistically_real=statistically_real,
                    tradeable_after_cost=tradeable_after_cost,
                    target_long_active=True,
                    lag_minutes=lag_minutes,
                )
            )
    return flags


def flags_to_dataframe(flags: list[RiskFlag]) -> pd.DataFrame:
    """Materialize a list of RiskFlag to a long-format DataFrame."""
    if not flags:
        return pd.DataFrame(
            columns=[
                "edge_id",
                "source_label",
                "target_label",
                "source_symbol",
                "target_symbol",
                "event_type",
                "bar_open_time",
                "severity",
                "statistically_real",
                "tradeable_after_cost",
                "target_long_active",
                "lag_minutes",
            ]
        )
    rows = [
        {
            "edge_id": f.edge_id,
            "source_label": f.source_label,
            "target_label": f.target_label,
            "source_symbol": f.source_symbol,
            "target_symbol": f.target_symbol,
            "event_type": f.event_type,
            "bar_open_time": f.bar_open_time,
            "severity": f.severity,
            "statistically_real": f.statistically_real,
            "tradeable_after_cost": f.tradeable_after_cost,
            "target_long_active": f.target_long_active,
            "lag_minutes": ",".join(str(x) for x in f.lag_minutes),
        }
        for f in flags
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class OverlayInputs:
    """All the inputs the overlay needs in one frozen object."""

    nodes: tuple[dict, ...]
    source_cvd_by_label: dict[str, pd.DataFrame]
    positions: pd.DataFrame | None = None
    overlay_cfg: PropagationOverlayConfig = PropagationOverlayConfig()


def run_overlay(inputs: OverlayInputs) -> pd.DataFrame:
    """Top-level driver. Iterates every node, joins to its source CVD, emits
    risk flags, returns the consolidated long-format DataFrame.
    """
    positions = _normalize_positions(inputs.positions)
    all_flags: list[RiskFlag] = []
    for node in inputs.nodes:
        src_label = (
            str(node.get("edge", "").split("->", 1)[0]).strip()
            or str(node.get("source_label", ""))
        )
        if src_label not in inputs.source_cvd_by_label:
            continue
        source_cvd = inputs.source_cvd_by_label[src_label]
        flags = emit_flags_for_node(
            node=node,
            source_cvd=source_cvd,
            positions=positions,
            overlay_cfg=inputs.overlay_cfg,
        )
        all_flags.extend(flags)
    return flags_to_dataframe(all_flags)


# -- Aggregation + write -----------------------------------------------------------


def summarize_overlay(log_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate flags by (edge_id, year-month, severity) for the markdown
    summary. Empty input → empty frame."""
    if log_df.empty:
        return pd.DataFrame()
    out = log_df.copy()
    ts = pd.to_datetime(out["bar_open_time"], utc=True)
    out["month"] = ts.dt.strftime("%Y-%m")
    grouped = (
        out.groupby(["edge_id", "month", "severity"], dropna=False)
        .size()
        .reset_index(name="n_flags")
    )
    return grouped.sort_values(["edge_id", "month", "severity"]).reset_index(drop=True)


def write_overlay_report(
    log_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    nodes: tuple[dict, ...],
    report_root: Path,
) -> dict[str, Path]:
    """Write the log + per-month aggregation + markdown summary."""
    ensure_dir(report_root)
    log_path = report_root / "propagation_risk_log.csv"
    summary_path = report_root / "propagation_risk_monthly.csv"
    md_path = report_root / "summary.md"
    log_df.to_csv(log_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    md_lines: list[str] = [
        "# Propagation-graph-node Risk Overlay (shadow)",
        "",
        "Consumer of `reports/sell_pressure_propagation_phase3*/graph_nodes.json`. "
        "Implements the docx-prescribed use case **\"Binance DOGE long risk warning\"** "
        "by emitting a per-event risk flag stream on the target venue/symbol. "
        "Severity is bound to the node verdict — `informational` for regime_event, "
        "`elevated` for statistically_real, `actionable` for tradeable_after_cost. "
        "The overlay does **not** veto entries or modify sizing; it is a logging-only "
        "shadow used to correlate node firings with realised target-side P&L.",
        "",
        f"- Total flags emitted: **{len(log_df)}**",
        f"- Nodes consumed: **{len(nodes)}**",
        "",
    ]
    for node in nodes:
        sev = node_severity(node)
        md_lines.append(
            f"- `{node.get('edge_id', '?')}` → {sev} "
            f"(statistically_real={node.get('status_statistically_real')}, "
            f"tradeable_after_cost={node.get('status_tradeable_after_cost')})"
        )
    md_lines.append("")
    if summary_df.empty:
        md_lines.append("_No flags emitted — overlay returned an empty log._")
    else:
        md_lines += [
            "## Flags by (edge, month, severity)",
            "",
            summary_df.to_markdown(index=False),
        ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "propagation_risk_log_csv": log_path,
        "propagation_risk_monthly_csv": summary_path,
        "summary_md": md_path,
    }


def run_overlay_from_disk(
    graph_nodes_path: Path,
    cvd_roots: dict[str, Path],
    bar_size_minutes: int,
    report_root: Path,
    positions: pd.DataFrame | None = None,
    overlay_cfg: PropagationOverlayConfig | None = None,
) -> dict[str, Path]:
    """End-to-end disk-driven overlay run.

    ``cvd_roots`` maps a venue tag (e.g. ``"bybit"``) to its continuous-CVD
    root. Source CVD is loaded for each node's source label using the
    matching venue root and the node's source symbol.
    """
    overlay_cfg = overlay_cfg or PropagationOverlayConfig()
    nodes = load_propagation_nodes(graph_nodes_path)
    if not nodes:
        # Preserve the long-form schema in the empty CSVs so downstream
        # read_csv calls don't choke on a zero-column file.
        return write_overlay_report(
            flags_to_dataframe([]), pd.DataFrame(), tuple(), report_root
        )
    source_cvd_by_label: dict[str, pd.DataFrame] = {}
    for node in nodes:
        src_label = (
            str(node.get("edge", "").split("->", 1)[0]).strip()
            or str(node.get("source_label", ""))
        )
        if src_label in source_cvd_by_label:
            continue
        venue, symbol = _split_label(src_label)
        if venue not in cvd_roots:
            continue
        try:
            source_cvd_by_label[src_label] = load_continuous_cvd(
                cvd_roots[venue], symbol, bar_size_minutes
            )
        except FileNotFoundError:
            continue
    log_df = run_overlay(
        OverlayInputs(
            nodes=tuple(nodes),
            source_cvd_by_label=source_cvd_by_label,
            positions=positions,
            overlay_cfg=overlay_cfg,
        )
    )
    summary_df = summarize_overlay(log_df)
    return write_overlay_report(log_df, summary_df, tuple(nodes), report_root)
