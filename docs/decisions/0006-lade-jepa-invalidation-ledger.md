# Decision 0006: LaDe JEPA invalidation and non-promotion ledger

Date: 2026-07-17  
Status: accepted; results retained only for audit and case-selection learning

## Hypothesis

LaDe Jilin was evaluated as a public surrogate for a dispatch world model and
remaining-route-time prediction. The investigation showed that its engineered
position, time and progress features make it primarily a tabular forecasting
problem, not a defensible flagship JEPA use case for Kaleido.

## Changes and invalidations

Three development generations exist locally and must not be mixed:

1. **v1 downstream JEPA invalid.** Embeddings were extracted from a shuffled
   training loader and concatenated with raw rows in original order. Full-label
   JEPA/boosting metrics were misaligned and are withdrawn. Fixed-order
   validation/test transition extraction and the separately extracted sparse
   benchmark were not affected by this specific bug.
2. **v2 downstream rows aligned, action semantics invalid for product.** The
   action channel chose tasks according to their eventual delivery order. A task
   may have been accepted at the cutoff, but knowing it would be delivered next
   is future information unless a versioned dispatch plan records that order.
   Any correct-vs-shuffled transition gain is therefore an oracle upper bound,
   not evidence of an observable operator action.
3. **v3 FIFO observable but incomplete and not promoted.** The code replaces
   eventual delivery order with a deterministic FIFO policy over tasks already
   accepted and pending at the cutoff. Training was stopped when the research
   question moved to AIS port-call dynamics.

The builder now fails the leakage audit unless the observable
`accepted_pending_fifo` policy is used. The oracle policy remains available only
for explicitly labelled diagnostics.

## Tests and evidence

Dataset/export: `lade_delivery_jilin_2022`, export version 1; local source
SHA-256 `12e2cf4664dd5b4475d39dddee8872f5a03b3082f08f0eece7f103baee6c6e73`.
Split: chronological May--August train, September validation, October test,
grouped by courier-day. October has already influenced development choices.

Tests added cover grouped splits, cutoff-visible FIFO actions, deterministic
route-level label masks, non-shuffled embedding extraction and modality masks.
Ruff, mypy and the focused unit tests passed before this decision.

Feature-availability diagnostics on the previously opened period showed that
continuous location was a dominant shortcut: removing current GPS increased a
raw boosting MAE from roughly 68 to 120 minutes. These figures are diagnostic
and use the already opened partition; they are not a Kaleido accuracy claim.

## Limitations

- LaDe is last-mile delivery in Jilin, not port operations.
- It has no immutable dispatcher plan or verified operator action.
- Earlier downstream results are affected by either row misalignment or an
  oracle action ordering.
- Public data cannot establish Kaleido value.

## Next falsifiable step

Do not spend additional compute promoting LaDe remaining-time JEPA. Use it only
for unit/integration coverage or, if revived, pre-register an observable action
policy and a new untouched temporal interval. The active JEPA experiment is the
AIS Port Call Deviation Twin in Decision 0005.

