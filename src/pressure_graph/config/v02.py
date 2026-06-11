from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FrozenCandidate:
    candidate: str
    path_name: str
    signal_col: str
    entry_policy: str
    execution_rule: str
    priority: str
    rationale: str


@dataclass(frozen=True)
class FillPolicy:
    name: str
    touch_buffer_bps: float
    mode: str


@dataclass(frozen=True)
class V02Config:
    name: str
    source_config: Path
    universe_col: str
    final_holdout_only: bool
    candidates: list[FrozenCandidate]
    fill_policies: list[FillPolicy]
    cost_bps: list[float]
    max_positions: list[int]
    rankings: list[str]


def load_v02_config(path: str | Path = "configs/v0_2_frozen_candidates.yaml") -> V02Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    experiment = payload["experiment"]
    candidates = [FrozenCandidate(**item) for item in payload["candidates"]]
    fill_policies = [
        FillPolicy(name=name, **settings) for name, settings in payload["fill_policies"].items()
    ]
    return V02Config(
        name=experiment["name"],
        source_config=Path(experiment["source_config"]),
        universe_col=experiment["universe_col"],
        final_holdout_only=bool(experiment.get("final_holdout_only", True)),
        candidates=candidates,
        fill_policies=fill_policies,
        cost_bps=[float(item) for item in payload["cost_bps"]],
        max_positions=[int(item) for item in payload["max_positions"]],
        rankings=list(payload.get("rankings", [])),
    )
