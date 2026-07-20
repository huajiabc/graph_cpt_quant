# Volatility Alpha Exploration — Consolidated Findings

Date: 2026-07-15

Status: four preregistered retrospective families completed. No deployment or permission change.
The remote v11.2 PaperLive shadow observer continues unchanged.

## What was learned

| Version | Mechanism | Observations | Main result | Graph attribution | Verdict |
|---|---|---:|---|---|---|
| v11.3 | Directed absolute-volatility receiver + OCO | 174 traded | future RV 1.2456x; net20 -21.51 bp | RV expansion at random 50th percentile | reject |
| v11.4 | Down/up semivariance transmission | 220 down / 211 up | downside gross +18.91 bp, net20 -1.09 bp; upside net20 -24.80 bp | downside random-family 78th percentile | reject |
| v11.5 | Exact 8x9 downside community front -> BTC short | 378 | gross +6.16 bp; net20 -13.84 bp | random-partition 68th percentile | reject |
| v11.6 | Exact 8x9 high-vol, efficient-path continuation | 412 | gross +4.15 bp; net20 -15.85 bp | random-partition 8th percentile | reject |

## Core conclusion

Price-only data does contain volatility structure:

- compressed coins commonly show later realized-volatility expansion;
- downside shock states contain more directional information than upside states;
- one-sided high-efficiency paths are more informative than a one-day displacement.

But none of those facts became stable executable alpha. The receiver-volatility effect was
identical under random graphs. Downside transmission was mostly continued BTC beta rather than
follower residual return. Cross-community breadth did not lead BTC economically, and efficient
source-community paths were worse than random partitions.

This establishes an information boundary, not a reason to stop research: increasingly complex
models built only from the same returns, realized volatility, and correlation graph are unlikely
to manufacture the missing direction. A GNN, transformer, or boosted tree could fit interactions
inside these samples, but every tested causal primitive is below cost or lacks graph attribution;
model complexity would mainly increase selection risk.

## Where alpha can still come from

1. **Forward synchronized aggressor flow.** Binance/Bybit taker-flow and OI/funding shocks can
   supply a genuinely orthogonal edge variable. This remains data-age gated, not conceptually
   rejected.
2. **Volatility instruments.** A realized-volatility forecast can be monetized directly only with
   options, variance-like exposure, or a demonstrable volatility-risk premium. The current
   futures-only workspace has no historical implied-volatility surface, so that lane is data
   blocked.
3. **Execution/holding overlays.** v11.6 path efficiency may be tested against independently valid
   P2/v11.2 entries as a hold/exit or sizing context. That would be portfolio improvement, not new
   entry alpha, and requires a separate preregistration.
4. **Continue v11.2 forward observation.** It remains the strongest graph-specific rare-state
   mechanism, but its retrospective uncertainty and sparse forward count still prohibit leverage
   or promotion.

## Decision

Do not deploy v11.3-v11.6, do not add leverage, and do not tune stronger retrospective thresholds.
Keep the code and reports as negative evidence. The next alpha iteration should wait for an
orthogonal data source or explicitly change the tradable instrument, rather than re-encoding the
same price graph with a more complex model.
