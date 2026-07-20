# v21.2-v21.4 DEX Community-Propagation Round Summary

Verdict: `directional_community_propagation_rejected_relative_rank_residual_only`.

## Design

This round separated the information source from the traded assets.  A DEX volume
attention event on one token was only usable at its recorded availability time;
the first subsequent 15-minute close supplied a causal CEX confirmation, and
execution waited one further complete bar.  The traded names were other members
of the source token's forward-frozen monthly graph community.

- DAP1 traded all available community peers in the source return direction.
- DAP2 traded the slowest half of peers ranked by their already observed one-hour
  return in that direction.
- Both used a causal monthly BTC beta hedge, unit gross, a 12-hour holding period,
  and 20/40 bp round-trip costs.

The feature audit passed 20/20 checks.  Each rule had 274 feature events across 10
months, 15 source symbols, and 32 communities.  Four events in the DEX vendor
transition interval were excluded from the frozen eligibility estimand; 267
events per candidate had complete risk and 12-hour price endpoints.

## Reveal

| Candidate | Events | Gross bp | Net at 20 bp | Random percentile | Bootstrap lower 95%, net bp |
|---|---:|---:|---:|---:|---:|
| DAP1 all peers | 267 | -1.8487 | -21.8487 | 0.6620 | -31.4170 |
| DAP2 relative laggards | 267 | +1.4058 | -18.5942 | 0.9080 | -28.8206 |

DAP2 gross results by development/validation/holdout were -2.6123, +9.5254,
and +18.4954 bp.  The recent strengthening remained below the frozen 20 bp cost
hurdle, while delayed entry yielded only +3.0829 bp gross.  DAP1 and DAP2 both
failed chronological net, random-control, bootstrap, and concentration gates.
The +24-hour placebo correctly lost the effect, and source-token-only response was
worse, but those attribution successes cannot offset the absent economic edge.

The independent v21.4 audit passed 25/25 checks, including exact source-file hash,
strictly prior beta estimates, timing, weights, symbol contributions, costs, all
1,000 random paths, block bootstrap, concentration, and rejection decisions.

## Residual research implication

DAP2 exceeded DAP1 gross return by approximately 2.56, 7.67, and 3.35 bp in the
three chronological periods.  This is a post-reveal observation, not independent
alpha evidence.  It motivates one bounded diagnostic of a dollar-neutral
laggard-versus-leader community spread.  Any favorable result on the same history
must remain research-only until genuinely new natural-forward data arrive.

No live, PaperLive, application, leverage, remote, or order state was read or
changed.
