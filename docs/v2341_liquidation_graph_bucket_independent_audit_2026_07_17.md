# v23.41 Liquidation Graph-Bucket Independent Audit

Verdict: `graph_bucket_relation_stable_to_leave_one_out_within_pilot`.

The graph-bucket relation is not carried by one coin. Removing each source in turn leaves partial rank at or above +0.220; the weakest case omits BTCUSDT. Removing each receiver leaves partial rank at or above +0.252; the weakest case omits AVAXUSDT.

On non-overlapping hourly decisions (n=22), raw Spearman is +0.315 and partial rank after prior market range is +0.211.

This strengthens the mechanism case for aggregate notional and breadth as graph-level volatility-state features. It still does not establish tradable alpha because the source snapshot is retrospective and covers one day.
