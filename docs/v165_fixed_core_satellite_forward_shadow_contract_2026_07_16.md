# v16.5 CM2 Fixed Core-Satellite Forward-Shadow Contract

`CM2_FIXED_80_FSS3_20_TG1` is registered locally as a portfolio-layer
`CANDIDATE_WATCH`. It is not a new raw alpha, a PaperLive strategy, or an
authorization for remote changes, leverage, or real orders.

## Frozen construction

At each Monday 00:00 UTC decision, allocate exactly 80% of capital to the exact
v14.9 `FSS3_CURRENT_SIGN_070_TURNOVER_CAP` sleeve and 20% to the exact v13.2
`TG1_FORWARD_EXTENDED_TO_2026_07` sleeve. Do not optimize, volatility-scale,
regime-switch, or drift the 80/20 allocation. Each sleeve retains its own signal,
membership, transition, venue, turnover, and 20/40 bp cost treatment.

The combined price, funding, primary-net, and stress-net return is the linear
80/20 sum of the two complete sleeve returns. No cross-sleeve netting benefit is
claimed. No additional allocation-turnover charge is required while the capital
split stays fixed; any future transfer or dynamic allocation rule requires a new
preregistration and execution-cost audit.

Fail closed for the week if either sleeve cannot be reproduced exactly with
decision-time-available data, if either prior executed state is missing, or if
entry/exit calendars are not identical. A failed sleeve may not be silently
replaced by a 100% allocation to the other sleeve.

## Evidence and limitations

The frozen 49-week research sample produced 78.98 bp/week primary net and
67.01 bp/week stress net. The independent v16.6 rebuild reproduced every weekly
component and weight with zero difference; its alternate four-week-block
bootstrap lower bound was +27.28 bp/week. Relative to standalone FSS3, the fixed
combination retained 83.87% of mean return while reducing additive drawdown by
25.35% and downside semideviation by 22.41%.

The evidence is not sufficient for deployment. Validation primary return was
only +4.93 bp/week and validation stress return was -6.32 bp/week. The 20%
satellite cap was selected after sleeve summaries were known, and TG1 was rejected
as a standalone tradable alpha because its positive-month concentration exceeded
35%. The construction is therefore a diversification hypothesis awaiting natural
forward evidence, not a claim that TG1 has independently graduated.

## Required forward telemetry and review gate

For each natural Monday-to-Monday week, record both sleeve target/executed
weights, data cutoffs and hashes, primary/stress costs, price and funding PnL,
turnover, missing-data decisions, combined 80/20 returns, and the same-week FSS3
benchmark. The shadow implementation must prove that both sleeve states are
isolated and that no automatic order or push path is reachable.

Do not review the candidate for PaperLive before at least 12 new complete natural
weeks. A review requires positive forward primary and stress means, a relative
risk comparison with FSS3, exact data/state and execution audits, and a fresh
independent report. Passing those checks still does not authorize deployment;
remote PaperLive requires a separate explicit request. The frozen candidate file
is `configs/v16_5_fixed_core_satellite_fss3_tg1_candidate.yaml`.
