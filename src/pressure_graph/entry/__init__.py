"""Entry-policy predicates ported from sibling systems.

The submodules here are intentionally tiny, self-contained pure functions
that v07a2 / v07d2 paper-live ledgers can call. We do not pull in upstream
data structures (e.g. v9's watchlist) — only the *predicate*.
"""

from pressure_graph.entry.retest import DEFAULT_RETEST_BUFFER, passes_v9_retest

__all__ = ["DEFAULT_RETEST_BUFFER", "passes_v9_retest"]
