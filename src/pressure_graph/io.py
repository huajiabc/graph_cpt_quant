from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def raw_path(data_root: Path, exchange: str, dataset: str, symbol: str | None = None) -> Path:
    base = data_root / "raw" / exchange / dataset
    if symbol is None:
        return base.with_suffix(".parquet")
    return base / f"{symbol}.parquet"


def processed_path(data_root: Path, experiment_name: str, filename: str) -> Path:
    return data_root / "processed" / experiment_name / filename


def report_path(report_root: Path, filename: str) -> Path:
    return report_root / filename
