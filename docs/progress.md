# FlowTwin implementation progress

Last updated: 2026-07-17. Public results are `smoke_only` or `diagnostic`.

| Milestone | State | Evidence | Remaining gate |
|---|---|---|---|
| M0 bootstrap | complete for smoke | Python 3.12, uv, CPU fixture | fresh-machine replay |
| M1 contract/audit | complete for smoke | roles, cutoffs, plans, censoring, leakage | Kaleido semantics |
| M2 OCEL/process | complete for smoke | object graph, variants, conformance, dashboard | operator review |
| M3 baselines | complete | naive/survival/linear/quantile boosting | real plan target |
| M4 sequential | complete; rejected vs floor | GRU/Transformer, grouped future test | richer sequence data |
| M5 JEPA | active AIS development | Phys-JEPA port-call dynamics | clean future AIS improvement |
| M6 scenarios/actions | synthetic only | correct-vs-shuffled VISReg experiment | real controllable actions |
| M7 serving | complete for smoke | read-only API/dashboard/export/audit | Kaleido review |
| Aligned ETA | **6/6 public gates passed** | AIS ETA 1.88 h MAE on 85 future trips | Vigo/port subgroup test |
| Aligned OCEL | diagnostic | correct graph 86.64 h vs flat 88.17 h on test | graph selected on validation |
| M8 pilot | planned | data request and frozen protocol | agreement/data/operators |

## Current evidence decision

The primary demonstration is NOAA AIS ETA v3. The tabular model was selected on
validation and tested once on 1-7 February 2025: MAE 1.875 h, trip-bootstrap
IC95 % 1.70-2.08 h, 60.62 % within +/-2 h and 87.3 % within +/-4 h. It improves
the kinematic ETA by 75.92 % and the historical port-distance median by 31.23 %.
All six predeclared gates passed.

Limits remain material: P90 width is 9.04 h; 87.6 % of test prefixes are New
Orleans; the data are public US AIS and the circular geofences are inferred. No
Kaleido accuracy or business claim follows.

OCEL Container Logistics preserves object identity and visible plan revisions.
The correct object graph improved test MAE, but validation selected the flat
trace; it is process/diagnostic evidence only.

The historical warehouse result (~734 min MAE) is retained as a rejected
demonstrator. JEPA learned nontrivial representation but its embeddings did not
add held-out value to boosting. It stays in research shadow.

See [Decision 0004](decisions/0004-aligned-public-benchmark-pivot.md).

## Active research: Port Call Deviation Twin

LaDe remaining-route time is no longer the flagship JEPA task. The v1
full-label downstream head was invalidated by shuffled embedding extraction;
v2 used eventual delivery order as an oracle action; v3 corrected this to a
cutoff-visible FIFO proxy but was stopped after case selection. See
[Decision 0006](decisions/0006-lade-jepa-invalidation-ledger.md).

The active case is an AIS Port Call Deviation Twin for Shipping Board and
Freight Intelligence. It predicts state evolution at 0.5/1/2 hours and compares
physical progress with observed evolution. The development comparison keeps
persistence, kinematics, trajectory GBT, GRU, Transformer, plain JEPA and
Phys-JEPA. February 1--7 is already opened and may only select a candidate; a
new calendar interval must remain untouched until code/config are committed.
See [Decision 0005](decisions/0005-port-call-deviation-twin-phys-jepa.md).

The three-seed development selection chose Phys-JEPA + VICReg. Its frozen
representation improved hybrid trajectory validation MAE from 3.336 to 3.124
km (6.37%) without collapse. A validation-only compressed sparse probe improved
ETA from 1.318 to 1.300 h and delay AUPRC from 0.754 to 0.761 using the same 10%
of labelled train trips. These are diagnostic selection results; the new NOAA
future week was downloaded as seven opaque compressed files and hash-frozen; it
has not yet been parsed or opened for target construction.

The first target-build attempt stopped before parsing because the manifest
mistook the ETA v2 input-cache hash for the ETA v3 combined-prefix hash. The
fail-closed audit corrected it to `52ab4006...f5a954` in a separate commit;
holdout model choices and gates did not change.

## Verification

```powershell
uv run ruff check .
uv run mypy src/flowtwin
uv run pytest -q
uv run flowtwin build-final-package
```

The generated HTML, TeX and three PDFs are rebuilt only from hash-verified runs.

## Next falsifiable step

Freeze one Vigo/Kaleido export, agree per-decision ETA tolerances and pre-register
lead-time, calibration, interval-width and worst-port gates. Reject transfer if
the future replay fails any required gate.
