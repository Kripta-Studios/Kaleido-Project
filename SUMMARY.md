# Kaleido FlowTwin continuity summary

Last updated: 2026-07-15 (Europe/Madrid)

This document is the hand-off record for a future agent. Read `AGENTS.md` and all
documents listed in its "Read first" section before changing code. The engineering
contract is normative and overrides shortcuts suggested by this summary.

## Mission and claim boundary

FlowTwin is a read-only predictive operations layer intended to complement Kaleido
Trace Port, Shipping Board and Freight Intelligence. The primary target is remaining
time and material plan-deviation risk for one operation/shift, with calibrated
uncertainty, useful lead time and an auditable explanation.

No public dataset in the repository proves Kaleido accuracy, early warning, action
value, ROI, savings, production readiness or deployment success. Public and synthetic
runs are `smoke_only`. Event-JEPA may be implemented and compared as a representation
learner without actions, but it is not a world model and cannot support action-value or
counterfactual claims until timestamped, operator-controllable Kaleido actions exist.

## Evidence files that must remain untouched

- `correspondencia/CorreoEnviado1.txt`
- `correspondencia/CorreoRecibido1.txt`
- `correspondencia/adjuntos/Industrial_World_Model_MVP_es.pdf`

They have been read and not modified.

## Public data downloaded

All public downloads requested by `DATASETS.md` completed without manual steps. Raw
files are intentionally ignored by Git. Total local download size is 1,440,723,790
bytes (about 1.441 GB decimal). Versioned metadata and checksums are in
`data/manifests/`.

- Container Logistics OCEL v3: JSON, SQLite and XML.
- Warehouse Outbound 2025: three CSV exports.
- Order Management: two ZIP archives.
- Inventory Management, Zenodo record 15535073: ten files.
- BPI Challenge 2019: optional competence dataset.

Published hashes were checked where supplied. Run `uv run flowtwin verify-data` (or
the manifest-specific equivalent shown by CLI help) before reusing local downloads.

## Implemented repository surface

The repository now has a Python 3.12/`uv` project with Ruff, strict mypy and pytest.
Major implemented capabilities are:

- canonical Pydantic contracts for events, immutable plan revisions and censored
  outcomes;
- conservative field-role classification and fail-closed leakage audits;
- timestamp, DST, ordering and plan-at-cutoff validation;
- reversible object graph and chronological grouped splits;
- read-only Trace Port, Shipping Board, Freight Intelligence and OCEL adapters;
- process discovery, variants, conformance and bottleneck analytics;
- causal prefix construction, temporal/categorical features and as-of context joins;
- median, Kaplan-Meier, ridge/logistic and quantile boosting baselines;
- GRU and ProcessTransformer baseline architectures;
- remaining-time, risk, calibration, lead-time and grouped metrics;
- discrete-event simulation kept separate from learned action ranking;
- reproducible run manifests, artifacts, model cards and reports;
- read-only FastAPI service and a modern local dashboard with synthetic watermarking;
- CLI and script wrappers for audit, process discovery, training, evaluation, report
  generation, shadow replay and serving.

## Verified smoke runs

### Synthetic Trace Port-like fixture

- 240 operations, 2,185 events, 11 censored outcomes.
- Chronological grouped split: 168 train / 36 validation / 36 test operations.
- Timestamp and leakage audits passed.
- Object graph: 766 objects.
- Claim state: `smoke_only` because the data are synthetic.

### Public Container Logistics OCEL v3

- Parsed 35,372 events, 13,882 objects and 74,272 event-object relationships from
  the downloaded SQLite export.
- 13,745 object traces, 47 variants, top-20 variant coverage about 0.9964.
- Timestamp and relationship-integrity checks passed.
- The publisher page and parsed SQLite count differ (35,761 versus 35,372); retain
  this as a provenance finding rather than silently rewriting it.
- Claim state: `smoke_only`.

## Invalidated warehouse M3 result

`outputs/warehouse_smoke_v1` is invalidated and contains `INVALIDATED.md`. It used
`progress_ratio = observed_event_count / final_event_count`, which leaks knowledge of
the completed trace at the prediction cutoff, and it included terminal prefixes. Its
previous MAE and risk values must not be cited or used to open M4/M5.

The code has been corrected to:

1. remove `progress_ratio` from training and serving features;
2. exclude terminal prefixes (`event_index < case_events`);
3. use only observed sequence length in sequential numeric features;
4. add an integration assertion that remaining time is strictly positive and that
   `progress_ratio` is absent.

The corrected M3 output target is `outputs/warehouse_smoke_v2`. Regenerate it before
training M4.

## Event-JEPA research status

The 29 user-supplied primary papers are being reviewed. The working design translation
is:

- I-JEPA: predict target embeddings from informative context; use asymmetric temporal
  views rather than reconstructing raw events.
- LeJEPA / LeWorldModel: start with an end-to-end predictive loss plus SIGReg; monitor
  complete and dimensional collapse.
- VISReg: reserve as an anticolapse ablation for long-tailed/low-rank event data if
  SIGReg gradients or rank diagnostics fail.
- Var-JEPA: reserve a variational latent target as the uncertainty ablation; deterministic
  JEPA plus calibrated quantile heads remains the simpler first experiment.
- V-JEPA 2: action-free pretraining is legitimate representation learning; a separate
  action-conditioned post-training stage is required before calling it a world model.
- Fast-LeWM / HWM / FF-JEPA: predict several direct horizons in parallel and consider
  hierarchy instead of long autoregressive rollouts that accumulate error.
- WorldDP: keep project, operation, shift, cargo, resource and incident identities as
  object-centric tokens; do not flatten away relationships.
- JEPA-DNA / MJEPA: test auxiliary next-event and cross-view objectives, but keep the
  pure JEPA objective as an ablation.
- AdaJEPA: any adaptation must be shadow-only with a frozen reference, replay, trust
  region, signed version and rollback; do not update the reference model online.
- PatchCore / intuitive-physics work: latent surprise can be a diagnostic anomaly score,
  not automatically an incident probability or causal early-warning claim.
- “When Does LeJEPA Learn a World Model?” gives guarantees under stationary,
  additive-noise assumptions; sparse logistics logs may violate them, so no theoretical
  world-model guarantee transfers automatically.
- DINOv2/v3 and V-JEPA 2.1 matter only when valid pre-cutoff photos/video are added;
  vision is not on the critical path for the current event-log MVP.

A full per-paper decision table and citations still need to be committed under
`docs/decisions/`.

## Two Event-JEPA protocols to implement

### A. Action-free representation benchmark

Use the same corrected public warehouse prefixes and frozen grouped chronological split
as M3/M4. Train an event encoder to predict future-window embeddings at several direct
horizons. Compare frozen-probe and fine-tuned remaining-time heads against boosting,
GRU and ProcessTransformer over at least three seeds. This tests representation value
only. Name it Event-JEPA, not Event World Model.

Required diagnostics: embedding standard deviation, effective rank, covariance/isotropy,
prediction loss, validation loss, checkpoint reload, P50/P90 metrics and seed variance.
Test must not select architecture, regularizer, horizon or threshold.

### B. Synthetic-action recovery benchmark

Do not relabel warehouse activities such as picking or packing as controllable actions.
Instead, construct a separate simulator/structural generator in which actions are explicit,
timestamped and have known effects. Candidate synthetic actions grounded in warehouse
semantics are:

- `expedite_release`: reduces the next queue wait at an explicit capacity/overtime cost;
- `add_temporary_capacity`: increases service capacity for a bounded interval;
- `priority_dispatch`: changes queue order for eligible work, with congestion side effects;
- `reroute_to_parallel_station`: changes the assigned resource when one is available;
- `planned_hold`: delays service intentionally under a documented constraint.

The generator must log the policy/propensity, action time, pre-action state, support,
random seed and structural effect. Run mandatory ablations: current prefix only, context
only, correct actions, shuffled actions, action only, no JEPA objective, one versus multiple
horizons, no anticolapse regularizer, frozen versus fine-tuned encoder, and object graph
versus flattened trace. If correct actions do not beat shuffled actions, reject action value.
Even a successful synthetic result proves only recovery of injected signal.

## Immediate reproducible next steps

From the repository root:

```powershell
uv sync --extra sequence
uv run flowtwin train-baselines `
  data/raw/public/warehouse_outbound_2025/outbound_2024_06_10_obfuscated.csv `
  --output outputs/warehouse_smoke_v2
uv run flowtwin train-sequence `
  data/raw/public/warehouse_outbound_2025/outbound_2024_06_10_obfuscated.csv `
  --baseline-run outputs/warehouse_smoke_v2 `
  --output outputs/warehouse_sequence_smoke_v2
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

Then inspect `outputs/warehouse_sequence_smoke_v2/m4_gate.json`. M5 public smoke may
start only after corrected M3 and M4 artifacts exist. Kaleido claim promotion remains
blocked regardless of public performance because there are no Kaleido exports, immutable
plan revisions, verified action fields or material deviation outcomes.

## Current verification state

After the leakage correction:

- `uv run ruff check .`: passed.
- `uv run mypy src tests`: passed (67 files).
- `uv run pytest -q`: 36 passed; one FastAPI/Starlette TestClient deprecation warning.

No raw data, credentials, customer identifiers or generated outputs should be committed.
