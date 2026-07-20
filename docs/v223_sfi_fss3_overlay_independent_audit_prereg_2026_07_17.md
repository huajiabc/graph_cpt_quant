# v22.3 SFI-on-FSS3 Independent Audit Preregistration

Date frozen: 2026-07-17, after the v22.2 outcome reveal and before the
independent audit calculations.

The audit must read the saved v22.1/v22.2 artifacts and raw weekly panel, not
accept v22.2 summary fields as evidence. It will independently reconstruct
weekly positions' price, funding, gross return, full-L1 turnover, primary and
stress costs, BTC beta, gross notional, target tracking, fixed 80/20 CM2
arithmetic, paired bootstrap interval, random-null percentile, gate decisions,
artifact hashes, and permission metadata.

All audited numerical identities use tolerance `1e-12`, except the frozen
turnover cap breach tolerance of `1e-10`. The audit passes only if every check
passes. Passing validates the negative v22.2 result; it cannot promote the
rejected overlay or change PaperLive, leverage, live, remote, application, or
order state.
