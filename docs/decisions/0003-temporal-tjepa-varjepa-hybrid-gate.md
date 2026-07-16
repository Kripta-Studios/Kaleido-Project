# Decision 0003: Temporal T-JEPA, Var-JEPA and hybrid gate

> Superseded as the primary demonstration decision by
> [Decision 0004](0004-aligned-public-benchmark-pivot.md). This record remains the
> auditable negative result for the warehouse/JEPA line.

Date: 2026-07-16

Status: accepted for `smoke_only`

## Hypothesis

A more faithful temporal JEPA protocol—non-overlapping future targets, a stopped
EMA teacher and validation-selected collapse control—will improve Event-JEPA.
A variational formulation will add useful uncertainty, and either representation
will add incremental value when appended to the winning boosting features.

## Decision

Keep raw quantile boosting as the selected predictor. Retain Temporal T-JEPA and
Var-Event-JEPA as diagnostic research implementations, but do not promote either
representation, their hybrid, or a world-model claim. The representation gate is
closed because validation selects raw features and the selected hybrid does not
beat raw boosting on test in any seed.

## Evidence

Dataset/export: Warehouse Outbound Event Log 2025, export 1; SHA-256
`2e6767ddd724304fe3716b0c768233e4a8db520f25780ce5ccd3795811348416`.
Split: `chronological_future_grouped_by_operation`, 14,000/3,000/3,000
operations and 42,000/9,000/9,000 prefix rows. Seeds: 11, 42 and 73.
Regularizer and hybrid selection used validation only; test influenced no choice.
Claim state: `smoke_only`.

| Experiment | Test MAE min | Gate interpretation |
|---|---:|---|
| Existing quantile boosting | 734.51 | frozen product floor |
| Raw boosting rerun, 3 seeds | 734.38 ± 0.23 | selected overall by validation |
| Existing Event-JEPA | 763.51 ± 2.84 | earlier shared-encoder reference |
| Temporal T-JEPA, multi-VISReg | 759.38 ± 1.15 | improves Event-JEPA, not boosting |
| Temporal T-JEPA, completion-only | 761.43 ± 1.65 | multi-horizon advantage not seed-robust |
| Temporal T-JEPA, shuffled future | 762.91 ± 3.18 | correct future wins in 3/3 seeds |
| Var-Event-JEPA | 760.41 ± 6.25 | does not improve T-JEPA or boosting |
| Raw + Var-Event-JEPA, selected hybrid | 738.98 ± 0.60 | +4.60 min vs raw; rejected |

## Interpreting the roughly 734 minutes

The metric is remaining-time MAE: the mean of the absolute gap between predicted
and actual remaining minutes at each held-out prefix. It is not predicted
duration, delay, saving or business value. In the single-seed M3 comparison,
boosting reduces global-median MAE from 789.08 to 734.51 minutes (54.57 minutes,
6.9%) and activity-median MAE from 779.34 to 734.51 (44.83 minutes, 5.8%).

That comparison supports `best_tested_mean_error_floor`, not operational
usefulness. Boosting median AE is 363.98 minutes, slightly above the global
median baseline's 353.15; P90 interval width is 2,459.90 minutes and the worst
last-activity group reaches 1,045.82 minutes. Kaleido must pre-agree tolerances
by decision horizon, lead time, interval width and false-alert cost before the
absolute error can be called useful. The presentation line is: “boosting wins
the public benchmark; Kaleido usefulness remains unknown.”

Temporal T-JEPA selected `multi_visreg` from SiGReg, VISReg-style variance and a
register-token candidate using mean validation pinball loss. Its target blocks
contain only events after the cutoff, and its target encoder is a stopped-gradient
EMA copy. Correct temporal pairing beat the fixed shuffled control in every seed,
but multi-horizon prediction did not beat completion-only in every seed.

Var-Event-JEPA adds Gaussian context, auxiliary and future variables, a learned
conditional future prior, observation decoders and KL annealing. Its mean latent
uncertainty/error Spearman correlation is 0.045; two of three seeds are negative.
The uncertainty is therefore diagnostic and is not calibrated in minutes.

The hybrid experiment fitted raw, raw+T-JEPA, raw+Var-JEPA and raw+both quantile
models with the same features, split and seeds. Validation selected `raw`; among
hybrids it selected `raw_var_jepa`. Test was reported only after those choices.

## Why boosting wins here

- traces are short: median 4 events, p95 6 and maximum 8;
- the vocabulary has only 9 activities and the process is comparatively regular;
- elapsed time, prefix progress, current activity and recent timing expose most of
  the supervised signal directly to trees;
- all 42,000 training prefixes already have labels, so self-supervised pretraining
  does not unlock a larger unlabeled corpus;
- compressing a short prefix into a 32-dimensional latent can discard exact timing
  while adding estimation noise;
- the public log has no verified actions, immutable plan revisions, object graph or
  rich context—the conditions where a sequential world model could add structure.

This does not show that JEPA is generally inferior to boosting. It shows that the
additional representation did not earn its complexity on this frozen dataset.

## Tests

- `uv run ruff check .`
- `uv run mypy src`
- `uv run pytest -q` (53 tests after dashboard utility integration)
- three-seed training runs for Temporal T-JEPA, Var-Event-JEPA and hybrid boosting
- dashboard HTTP check: six evidence stages, `KEEP RAW BOOSTING`, `smoke_only`

## Limitations

The temporal models are event-sequence adaptations, not exact reproductions of
T-JEPA or Var-T-JEPA. Training is CPU-budgeted (3/4 pretraining epochs), the source
is public non-port data and latent uncertainty is not time-calibrated. There is no
Kaleido accuracy, action-value, causal, ROI, savings or deployment evidence.

## Next falsifiable step

Freeze a versioned Kaleido export with operator-reviewed field roles, immutable
plan revisions, censoring, object relationships and real controllable actions.
Pre-register the same raw, sequential, T-JEPA, Var-JEPA and hybrid gates. Promote a
world-model component only if it beats raw boosting or a separately agreed
calibration, robustness, data-efficiency or action-value criterion across seeds.
