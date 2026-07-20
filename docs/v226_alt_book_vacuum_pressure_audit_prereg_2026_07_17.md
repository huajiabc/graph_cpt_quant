# v22.6 Alt-Book Vacuum Pressure Independent Audit Preregistration

Date frozen: 2026-07-17, after v22.5 reveal and before audit calculations.

The audit will independently rebuild exact `t`, `t+1h`, `t+4h`, and `t+5h`
returns from the raw combined Bybit hourly close panel; reprice BTC, reversed,
delayed and 16-alt bucket outcomes; recompute prior/future BTC variance; rebuild
the 307-event no-vacuum feature control; replay all 1,000 same-month,
same-direction random-time paths from explicit timestamp lookups; reproduce the
entry-day bootstrap and every promotion gate; and verify hashes and permission
metadata.

Numerical identities must agree within `1e-12`, every structural check must
pass, and the promotion flag must equal the conjunction of frozen v22.5 gates.
Audit passage validates the v22.5 rejection; it cannot promote the candidate or
change PaperLive, live, leverage, remote, application, or order state.
