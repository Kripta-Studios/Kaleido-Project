# FlowTwin implementation progress

Last updated: 2026-07-16. All results are `smoke_only`.

| Milestone | State | Evidence | Remaining gate |
|---|---|---|---|
| M0 bootstrap | complete for smoke | Python 3.12, uv, CPU fixture | fresh-machine replay |
| M1 contract/audit | complete for smoke | roles, cutoffs, plans, censoring, leakage | Kaleido semantics |
| M2 OCEL/process | complete for smoke | object graph, variants, conformance, dashboard | operator review |
| M3 baselines | complete | naive/survival/linear/quantile boosting | real plan target |
| M4 sequential | complete; rejected vs floor | GRU/Transformer, grouped future test | richer sequence data |
| M5 JEPA | research shadow | Event/T/Var-JEPA and ablations | incremental held-out value |
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
