"""v9 retest entry predicate, pure-function port.

Authoritative source: ``src/short_cpt_quant/v9_sprint.py`` (P1 plan refs
lines 1989-1994). The v9 retest condition is:

    low <= level * (1.0 + retest_buffer)
    and close > level
    and close > ema20

Interpretation: the bar dipped at or just above the setup level, then closed
back above both the level and the fast EMA. The graph-side caller supplies
``level = signal_close`` and uses the existing per-bar ``ema20`` feature
column. The predicate is intentionally a 4-scalar comparison — we did NOT
port v9's column production (setup_*, retest_*) or its sizing dependencies.
"""

from __future__ import annotations


DEFAULT_RETEST_BUFFER: float = 0.003


def passes_v9_retest(
    low: float,
    close: float,
    level: float,
    ema20: float,
    retest_buffer: float = DEFAULT_RETEST_BUFFER,
) -> bool:
    """Return True iff the bar satisfies the v9 retest entry condition.

    Parameters
    ----------
    low, close:
        OHLC values from the candidate bar.
    level:
        Reference price the bar must retest, then close above. For graph this
        is typically the signal_close.
    ema20:
        Fast moving average value computed on the candidate timeframe. The
        bar must close above it to confirm the trend remains intact.
    retest_buffer:
        Tolerance for "near level": ``low`` may be up to
        ``level * (1 + retest_buffer)``. Defaults to ``DEFAULT_RETEST_BUFFER``
        (0.003) matching the v9 default ``retest_buffer``.
    """
    return (
        (low <= level * (1.0 + retest_buffer))
        and (close > level)
        and (close > ema20)
    )
