# FlowTwin continuity summary

Last updated: 2026-07-16 (Europe/Madrid)

Read `AGENTS.md` and its complete "Read first" list before changing code. This is
a hand-off, not a replacement for the contract.

## Claim boundary

All results are `smoke_only`. Public/generated data prove pipeline capability
only. They do not prove Kaleido accuracy, plan-deviation risk, causal action value,
ROI, savings, production readiness or deployment.

## Primary evidence: NOAA AIS ETA v3

- Official public NOAA MarineCadastre AIS, 38 days from 2025-01-01 to 2025-02-07,
  6.92 GiB compressed. Raw files are ignored; hashes live in the run manifest.
- Cutoff-safe grouped arrival trips; MMSI, arrival time and future positions are
  forbidden features.
- Split: 303 train / 73 validation / 85 untouched future test trips; 5,893 /
  1,381 / 1,780 prefixes. Test is 1-7 February 2025.
- Validation selected tabular ETA boosting.
- Test MAE 1.875 h; median AE 1.37 h; trip-bootstrap IC95 % 1.70-2.08 h.
- 41.97 % within +/-1 h, 60.62 % within +/-2 h, 87.3 % within +/-4 h.
- Baselines: kinematic 7.786 h, port-distance median 2.726 h,
  physics-residual 1.881 h.
- 75.92 % gain versus kinematic and 31.23 % versus historical; all 6/6 frozen
  gates passed.
- P90 coverage 94.49 %, width 9.04 h. New Orleans represents 87.6 % of test
  prefixes. This is the main limitation and next subgroup gate.

## Secondary evidence: OCEL logistics

- OCEL 2.0 Container Logistics, simulated public dataset, source SHA-256
  `3C91FF97F736615196902039EEC8F00BCD2F83F653D282D8A5AAEF3BAA7D3D7E`.
- 1,966 finished containers; 1,376 / 295 / 295 grouped chronological split.
- Test MAE: flat 88.17 h, correct graph 86.64 h, shuffled graph 88.28 h.
- Validation selected flat, so graph promotion is rejected; use this run for
  object-centric process intelligence and diagnostics only.

## Implemented system

- M0-M2: uv/Python bootstrap, contracts, timestamps, immutable plans, object graph,
  grouped splits, leakage audit and process mining.
- M3-M5: naive/survival/boosting, GRU, ProcessTransformer, Event-JEPA, Temporal
  T-JEPA, Var-Event-JEPA, uncertainty and ablations.
- M6: separate generated-action overlay and action-conditioned JEPA gates.
- M7: read-only FastAPI, dashboard, export/audit surfaces and regenerated
  HTML/TeX/PDF package.
- New aligned benchmarks: `flowtwin benchmark-ais-eta` and
  `flowtwin benchmark-ocel-logistics`.

## Historical negative evidence

The Warehouse Outbound remaining-time line remains auditable but is no longer the
demonstrator. Raw boosting obtained about 734 minutes MAE (~12.2 h), only modestly
better than weak medians, with very wide intervals. Temporal/Var-JEPA and hybrids
did not beat raw boosting. The Action-JEPA recovered injected signal but failed the
scale gate. These results justify keeping JEPA in research shadow, not serving that
predictor.

Invalidated runs remain preserved with `INVALIDATED.md`: warehouse M3 v1, sequence
v2/v3, dependent Event-JEPA v1 and AIS exploratory v1. AIS v2 is a valid diagnostic
that failed one predeclared gate; v3 is the untouched final test.

## Decision and next falsifiable step

Present AIS ETA as public capability evidence, OCEL as process/object diagnostic
and JEPA as gated I+D. Run a semantic-contract session on 3-5 pseudonymized
Kaleido cases, freeze a Vigo/operation export and pre-register tolerance,
lead-time, interval-width and worst-group gates before a read-only shadow replay.
