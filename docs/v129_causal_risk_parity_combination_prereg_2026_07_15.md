# v12.9 Causal Risk-Parity TG1 + P2 Combination Preregistration

Date frozen: 2026-07-15, before inspecting dynamic-weight returns.

The exact v12.8 aligned TG1 and frozen P2 sleeve series are reused. For the first
eight weeks the allocation is 50/50. Thereafter, each Monday uses only the prior
eight completed weekly sleeve returns to set inverse-volatility weights. The TG1
weight is clipped to `[0.25, 0.75]`; P2 receives the complement. A zero or
undefined trailing volatility keeps the preceding weight. No return mean,
correlation forecast, regime, month, or alternative window is used.

Combined return is the weighted sum of the two already costed sleeve returns,
minus an additional 20 bp one-way times the absolute weekly change in TG1 capital
weight. The initial 50/50 allocation and each sleeve's terminal costs already sit
inside the sleeve return series and are not charged twice.

Promotion gates are the v12.8 gates: at least 40 weeks, ten months, ten validation
weeks, eight holdout weeks; both sleeve means positive; absolute sleeve
correlation no greater than 0.50; combined development, validation, and holdout
means positive; paired 2,000-draw week bootstrap lower bound positive;
positive-month concentration no greater than 35%; and worst period no worse than
-40 bp/week. Mean TG1 weight and allocation turnover are descriptive only.

Passing means forward portfolio-shadow candidacy. PaperLive remains unchanged.
