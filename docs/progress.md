# FlowTwin implementation progress

Last updated: 2026-07-17. Public results are `smoke_only`, `diagnostic` or a
strictly scoped `claim_eligible` clean core result.

| Milestone | State | Evidence | Remaining gate |
|---|---|---|---|
| M0 bootstrap | complete for smoke | Python 3.12, uv, CPU fixture | fresh-machine replay |
| M1 contract/audit | complete for smoke | roles, cutoffs, plans, censoring, leakage | Kaleido semantics |
| M2 OCEL/process | complete for smoke | object graph, variants, conformance, dashboard | operator review |
| M3 baselines | complete | naive/survival/linear/quantile boosting | real plan target |
| M4 sequential | complete; rejected vs floor | GRU/Transformer, grouped future test | richer sequence data |
| M5 JEPA | core clean gate passed; full gate closed | GBT + Phys-JEPA trajectory/deviation | Kaleido future replay |
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
future week was downloaded as seven opaque compressed files and hash-frozen.

The first target-build attempt stopped before parsing because the manifest
mistook the ETA v2 input-cache hash for the ETA v3 combined-prefix hash. The
fail-closed audit corrected it to `52ab4006...f5a954` in a separate commit;
holdout model choices and gates did not change.

The clean run then executed from commit
`cdae9b7ac9b46b4dfda4186f0ebb135f41335ab8` in a detached worktree with
`dirty=false`. The first clean attempt stopped before training because the
environment lacked the optional PyTorch extra; it produced no metrics. After
installing the declared `sequence` extra in that ignored worktree, the frozen
run completed without changing code or config.

Clean dataset/export:
`noaa_marinecadastre_ais_2025_phys_jepa_holdout_02_08_02_14`, v1; processed
prefix SHA-256 `4f1c7bce...9ef382d`; fixed future grouped split 341/83/57 trips
and 3,778/976/750 samples. Metrics SHA-256 `99b4162f...f8270ca`. Three seeds;
physical 10 km two-hour threshold fixed before test; test did not influence a
choice; claim state `claim_eligible` for the public core only.

Trajectory GBT obtains 2.635 km MAE. The GBT + Phys-JEPA seed mean is
2.587 +/- 0.053 km (1.84% gain; wins 3/3 seeds). The served three-model ensemble
is 2.326 km, a paired-trip bootstrap gain of 11.72% with IC95% 5.90%-17.13% and
`P(improvement)=0.9995`. Deviation AUPRC changes 0.880 to 0.904. Conformal 90%
coverage is 89.79% with 12.00 km mean width. Effective rank is 11.42-12.46 and
no selected representation collapses.

The combined gate remains closed: sparse ETA improves only 0.59% and sparse
delay AUPRC regresses 0.619 to 0.606. Product decision: shadow the physical
trajectory/deviation core, keep GBT for ETA and reject the delay head. See
[Decision 0007](decisions/0007-phys-jepa-clean-holdout-result.md).

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
