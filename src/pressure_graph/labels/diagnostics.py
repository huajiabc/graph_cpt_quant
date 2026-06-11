from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["exchange", "symbol"]


def _touch_diagnostics_for_group(
    group: pd.DataFrame,
    window_bars: int,
    suffix: str,
    up_threshold: float = 0.03,
    down_threshold: float = 0.02,
) -> pd.DataFrame:
    out = group.sort_values("bar_open_time").copy()
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)

    first_touch: list[str] = []
    bars_to_hit = np.full(len(out), np.nan)
    bars_to_dd = np.full(len(out), np.nan)
    bars_to_first = np.full(len(out), np.nan)

    for idx, price in enumerate(close):
        if not np.isfinite(price) or price <= 0:
            first_touch.append("neither")
            continue
        up_price = price * (1.0 + up_threshold)
        down_price = price * (1.0 - down_threshold)
        hit_bar: int | None = None
        dd_bar: int | None = None
        same_bar = False
        for offset, lookahead in enumerate(
            range(idx + 1, min(idx + window_bars + 1, len(out))), start=1
        ):
            up_hit = np.isfinite(high[lookahead]) and high[lookahead] >= up_price
            down_hit = np.isfinite(low[lookahead]) and low[lookahead] <= down_price
            if up_hit and hit_bar is None:
                hit_bar = offset
            if down_hit and dd_bar is None:
                dd_bar = offset
            if up_hit and down_hit and hit_bar == dd_bar == offset:
                same_bar = True
            if hit_bar is not None or dd_bar is not None:
                if hit_bar is not None and dd_bar is not None:
                    break

        if hit_bar is not None:
            bars_to_hit[idx] = hit_bar
        if dd_bar is not None:
            bars_to_dd[idx] = dd_bar
        candidates = [bar for bar in [hit_bar, dd_bar] if bar is not None]
        if candidates:
            bars_to_first[idx] = min(candidates)

        if hit_bar is None and dd_bar is None:
            first_touch.append("neither")
        elif hit_bar is not None and dd_bar is None:
            first_touch.append("tp_first")
        elif hit_bar is None and dd_bar is not None:
            first_touch.append("sl_first")
        elif hit_bar == dd_bar and same_bar:
            first_touch.append("both_same_bar")
        elif hit_bar < dd_bar:
            first_touch.append("tp_first")
        else:
            first_touch.append("sl_first")

    first_touch_series = pd.Series(first_touch, index=out.index)
    hit = np.isfinite(bars_to_hit)
    dd = np.isfinite(bars_to_dd)
    clean_hit = first_touch_series.eq("tp_first") & hit
    dirty_hit = hit & ~clean_hit

    out[f"first_touch_3up_2down_{suffix}"] = first_touch_series
    out[f"bars_to_hit_3pct_{suffix}"] = bars_to_hit
    out[f"bars_to_dd_2pct_{suffix}"] = bars_to_dd
    out[f"bars_to_first_touch_{suffix}"] = bars_to_first
    out[f"clean_hit_3pct_{suffix}"] = clean_hit
    out[f"dirty_hit_3pct_{suffix}"] = dirty_hit
    out[f"bad_drop_no_hit_{suffix}"] = (~hit) & dd
    out[f"no_event_{suffix}"] = (~hit) & (~dd)
    return out


def add_touch_diagnostics(
    df: pd.DataFrame,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = {"4h": 16}
    frames = []
    for _, group in df.groupby(KEYS, sort=False):
        enriched = group
        for suffix, bars in windows.items():
            enriched = _touch_diagnostics_for_group(enriched, bars, suffix)
        frames.append(enriched)
    if not frames:
        return df.copy()
    return pd.concat(frames, ignore_index=True)
