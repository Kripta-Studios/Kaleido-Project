# Decision 0007: Phys-JEPA clean holdout result and product boundary

Date: 2026-07-17
Status: accepted as `claim_eligible` public core evidence; full product gate closed

## Hypothesis

A physics-informed JEPA state should add useful multi-horizon dynamics to a
strong trajectory GBT when trained from abundant AIS observations, even when
arrival and delay labels are scarce. It should not replace a winning ETA
baseline unless the ETA and delay probes also pass their frozen gates.

## Changes

- Added a read-only Port Call Deviation Twin at 0.5/1/2-hour horizons.
- Conditioned the JEPA predictor on a constant-course physical future and
  learned the residual future state.
- Compared persistence, kinematics, trajectory GBT, GRU, Transformer and direct
  Phys-JEPA without removing winning baselines.
- Froze VICReg after development comparisons with VISReg, SIGReg and no
  regularizer; all variants were checked for collapse.
- Added a GBT + Phys-JEPA downstream hybrid, paired trip bootstrap and split
  conformal intervals.

## Frozen evidence contract

- Dataset/export: `noaa_marinecadastre_ais_2025_phys_jepa_holdout_02_08_02_14`,
  version 1.
- Official source: NOAA MarineCadastre AIS 2025.
- Holdout: seven files, 2025-02-08 through 2025-02-14, 1,431,169,298 bytes;
  hashes were committed before parsing. Download receipt SHA-256:
  `95948657f9be6242dc617b5268264490476a908efa41090df21ce70a82e7f130`.
- Processed prefix SHA-256:
  `4f1c7bce92f6567599b3e1ebb66678f631c1a884adc2f4540a6bca0489ef382d`.
- Split: chronological future and grouped by arrival trip; train 3,778
  samples/341 trips, validation 976/83, test 750/57; trip sets are disjoint.
- Frozen code commit: `cdae9b7ac9b46b4dfda4186f0ebb135f41335ab8`.
- Run: clean detached worktree, `dirty=false`, three seeds 11/42/73.
- Metrics SHA-256:
  `99b4162f7b40a71506491d433b74306748e6c887c8b947534662a8b48f8270ca`.
- Selection: architecture, regularizer, heads and thresholds used development
  validation only. Test did not influence a choice.
- Deviation threshold: fixed physical two-hour shortfall over 10 km; it was not
  selected on the test.

## Results

| Model/result | Clean future test |
|---|---:|
| Persistence | 6.849 km MAE |
| Constant-course kinematics | 4.018 km MAE |
| Trajectory GBT | 2.635 km MAE; deviation AUPRC 0.880 |
| GRU, mean of 3 seeds | 2.798 km MAE |
| Transformer, mean of 3 seeds | 2.830 km MAE |
| Direct Phys-JEPA, mean of 3 seeds | 3.036 km MAE |
| GBT + Phys-JEPA, mean of 3 seeds | 2.587 +/- 0.053 km MAE |
| GBT + Phys-JEPA, three-model ensemble | **2.326 km MAE** |

The individual hybrid beats raw GBT in 3/3 seeds. Its seed mean improves the
raw floor by 1.84%. The served three-model ensemble improves it by 11.72%; a
paired bootstrap over 57 trips and 2,000 resamples gives an improvement IC95%
of 5.90%-17.13% and `P(improvement)=0.9995`. The two statistics answer different
questions: seed mean measures training stability; the ensemble measures the
variance-reduced product candidate.

Mean deviation AUPRC changes from 0.880 to 0.904. Split conformal intervals at
90% nominal coverage achieve 89.79% mean test coverage with 12.00 km mean
width. Effective rank is 11.42-12.46 on test and none of the three selected
representations collapses.

The sparse outputs do not pass:

- ETA MAE: 1.1334 to 1.1268 hours, 0.59% improvement, below the frozen 1% gate;
- delay AUPRC: 0.6189 to 0.6064, a clean-test regression.

Therefore `product_candidate.gate.passed=false`,
`promotion.clean_public_test=false` and `promotion.kaleido=false`.

## Decision

Expose `trajectory GBT + Phys-JEPA` only as a versioned, read-only shadow core
for trajectory and material-deviation evidence. Keep GBT as the ETA/remaining
time floor. Reject the sparse delay head. Do not describe the full system as
promoted, production-ready, causal or validated for Kaleido.

## Tests

Before presentation integration: 69 tests passed, Ruff passed and mypy passed
for 86 source modules. Final counts are regenerated after integration.

## Limitations

Public US AIS proves pipeline capability, not Vigo transfer, Kaleido value,
operator usefulness, savings or deployment success. AIS positions are
observations/context, not controllable actions. The deviation label is a
physical proxy rather than a Kaleido material incident. Test contains 57 trips,
and interval width remains operationally material.

## Next falsifiable step

Freeze a Kaleido Shipping Board/Freight Intelligence export, agree the material
deviation and decision tolerance with operators, then replay the same GBT-only
and GBT + Phys-JEPA comparison on future disjoint trips. Reject transfer if the
hybrid fails the agreed lead-time, calibration, false-alert or worst-group gate.
