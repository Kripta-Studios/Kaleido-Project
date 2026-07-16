# Decision 0002: Event-JEPA results and world-model gate

Date: 2026-07-16

Status: accepted for `smoke_only`

## Hypothesis

A future-latent objective can learn useful event-prefix representations, and an
explicit action channel can recover transition effects when actions are observable
and supported.

## Decision

Keep Event-JEPA as a first-class research track and frozen shadow component. Do not
promote it as the product predictor or as an operational world model. Serve quantile
boosting until JEPA beats that floor or provides a pre-agreed, separately measured
calibration, robustness or data-efficiency benefit.

## Evidence

Dataset/export: Warehouse Outbound Event Log 2025, export 1; SHA-256
`2e6767ddd724304fe3716b0c768233e4a8db520f25780ce5ccd3795811348416`.
Split: frozen chronological future holdout, grouped by operation, 14,000/3,000/3,000.
Seeds: 11, 42 and 73 for all neural comparisons. Models and epochs were selected on
validation only; test did not influence a choice. Claim state: `smoke_only`.

| Experiment | Test MAE min | Interpretation |
|---|---:|---|
| Quantile boosting | 734.51 | product floor; one seed, operation-bootstrap IC95 % 697.99–771.49 |
| ProcessTransformer | 809.54 ± 17.89 | rejected vs floor |
| Event-JEPA frozen | 763.51 ± 2.84 | improves Transformer, not boosting |
| Completion-only JEPA | 763.03 ± 1.77 | multi-horizon advantage not supported |
| Shuffled temporal pairs | 763.91 ± 2.15 | correct pairing advantage is only 0.40 min and not seed-robust |
| Random encoder/no JEPA | 772.78 ± 2.67 | JEPA objective has downstream signal |
| No SIGReg | 811.95 ± 13.90 | collapse; regularizer is necessary |

The first generated-action boosting benchmark rejected action value because correct
actions did not beat shuffled actions. Action-Event-JEPA with SIGReg also failed the
per-seed and collapse gates. A predeclared VISReg fallback then produced:

- correct action: 725.75 ± 3.20;
- current prefix only: 731.45 ± 20.50;
- shuffled action: 740.43 ± 14.85;
- correct beats shuffled in 3/3 seeds; mean improvement 14.67 min;
- effective rank 15.41–16.16 of 32, but mean dimension std 0.020–0.023 remains
  below the 0.05 collapse threshold.

This is recovery of an injected effect, not causal evidence and not comparable with
real remaining-time MAE.

## Consequences

- dashboard: boosting is the reference prediction; Event-JEPA is labelled shadow;
- R&D: retain SIGReg, VISReg and collapse diagnostics; test target semantics, not
  only representation rank;
- naming: action-free component is `Event-JEPA`; `world model` remains a research
  hypothesis until real action-conditioned held-out evidence exists;
- product: no action recommendation, counterfactual, savings or early-warning claim;
- pilot: correct actions must beat shuffled actions across seeds without collapse.

## Limitations

The remaining-time dataset is non-port, has no immutable plan revisions and no
verified actions. The generated overlay encodes known effects by construction. No
result transfers to Kaleido without a frozen Kaleido protocol.

## Next falsifiable step

On a Kaleido export, define operator-controllable action fields before looking at
outcomes, freeze plan-at-cutoff semantics and split by future operations. Compare
correct with within-support shuffled actions and prefix-only controls over at least
three seeds. Reject action value if the per-seed gate, calibration gate or collapse
gate fails.
