# v15.4 Binance Near-Touch Book-Depth Findings

Verdict: `reject_before_return_test_insufficient_feature_history`.

The frozen primary field required signed Binance book-depth rows at `-0.2%` and
`+0.2%`. Historical archives through late 2025 contain only the signed 1%, 2%,
3%, 4%, and 5% bands. The 0.2% rows were introduced by symbol at different dates
during 2026. Across the downloaded panel only 45.6% of symbol-days had a finite
0.2% feature, and no possible complete-universe sample can satisfy the frozen
minimum of 300 decision days.

The candidate is therefore rejected on its pre-return coverage gate. No v15.4
portfolio return, sign alternative, depth-band alternative, or threshold tuning was
inspected. Raw archives and hashes remain under
`data/external/binance_um_book_depth` for audit. PaperLive and remote state are
unchanged.
