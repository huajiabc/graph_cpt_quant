# v23.36 Liquidation-to-Price Mechanism Pilot

Verdict: `retrospective_mechanism_pilot_not_alpha_confirmation`.

Market decisions: 94; active symbol-buckets: 404.

Primary descriptive readings:

- log 15m total liquidation versus next-60m BTC log range: Spearman +0.374.
- alt-only liquidation versus next-60m BTC log range: Spearman +0.338.
- forced-buy share versus next-60m BTC return: Spearman -0.185.
- top-versus-bottom liquidation quartile next-60m BTC range difference: +23.28 bp.
- alt own-symbol forced-flow continuation over 60m: -3.28 bp mean, 45.4% positive.

These results are mechanism diagnostics only. The initial OKX snapshot was retrieved after most outcomes occurred, the window is roughly one day, and 15-minute observations overlap at the 60-minute horizon. No threshold, candidate, PaperLive, live, leverage, remote, application, or order state may be changed from this pilot.

Confirmatory evidence must come from the v23.35 forward contract using `first_seen_at <= decision_time`.
