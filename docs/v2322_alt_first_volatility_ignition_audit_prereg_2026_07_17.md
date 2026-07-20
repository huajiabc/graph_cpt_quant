# v23.22 Alt-First Volatility Ignition Audit Preregistration

The audit will rebuild v23.20 from the hourly price source and BTC causal
context, then rebuild every v23.21 control, OCO path, width variant, random
path, month bootstrap, leave-one-month-out row, gate, and verdict. All saved
artifacts must agree within `1e-12` and the feature hash must remain frozen.

The audit validates implementation and rejection; it cannot rescue a failed
economic result. No live, PaperLive, leverage, remote, application, or order
state is changed.
