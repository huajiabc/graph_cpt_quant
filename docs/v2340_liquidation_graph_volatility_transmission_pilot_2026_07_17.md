# v23.40 Liquidation Graph Volatility Transmission Pilot

Verdict: `retrospective_graph_bucket_volatility_relation_supported`.

Across the 17-source by 17-receiver graph, the all-source liquidation bucket has raw Spearman +0.414 to the future 60-minute median receiver range and partial rank +0.285 after controlling the prior market median range. Its partial circular-shift percentile is 93.3%.

The alt-only bucket remains positive at partial rank +0.220; active-source breadth is +0.256, while source concentration HHI is -0.093. Broad, distributed cascades therefore carry more broad-volatility information than concentrated single-coin events in this pilot.

The best single source is BTCUSDT at partial rank +0.241; the best single alt is 1000PEPEUSDT at +0.148. Both are weaker than their corresponding aggregate buckets. Off-diagonal source-receiver edges are positive in 64.0% of pairs after prior-range control.

This supports the graph/bucket research direction, not any individual edge or tradable strategy. The sample is one retrospective day with overlapping receiver horizons. Forward confirmation must freeze the aggregate-notional, breadth, and concentration features without selecting source-specific edges.
