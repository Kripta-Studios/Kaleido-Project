# Decision 0005: move JEPA from tabular ETA to port-call dynamics

Date: 2026-07-17  
Status: validation selection complete; public clean-test evidence pending

## Hypothesis

JEPA is not expected to add value when the target is nearly determined by a
small set of engineered coordinates, clock and progress features. Its useful
Kaleido role is instead to learn the multi-horizon dynamics of a port call and
the residual between a physical approach model and the observed vessel state.

The proposed product is a read-only **Port Call Deviation Twin** embedded in
Shipping Board and Freight Intelligence. At a cutoff it predicts vessel state
at 30, 60 and 120 minutes; as observations arrive it exposes latent surprise
and a physical-progress shortfall score. Trace Port may consume the resulting
exception, but the component does not write to any source system.

## Changes

- Keep NOAA AIS ETA boosting as the ETA product floor.
- Retire LaDe remaining-route time as the primary JEPA demonstration.
- Build cutoff-safe AIS sequences grouped by arrival trip.
- Compare persistence, constant-course kinematics, trajectory GBT, GRU,
  supervised Transformer, plain JEPA and physics-informed JEPA.
- Adapt Phys-JEPA by conditioning the latent predictor on a known kinematic
  future and learning its residual.
- Evaluate future distance, speed, two-hour deviation AUPRC, transition error
  and representation collapse.

## Tests and evidence available at decision time

Dataset/export: `noaa_marinecadastre_ais_2025_01_01_02_07`, export version 1.
The committed manifest is `data/manifests/noaa_ais_2025_jan_feb.yaml`. The local
source contains 38 compressed daily files (7,425,690,235 bytes). The existing
prefix cache SHA-256 is recorded by every run.
Its value for this selection is
`1d72175cf85194629b252d07b7b267ac6a356c7cbabf888443d4fc08f2d7f4b4`.

Split protocol: fixed chronological partitions grouped by arrival trip. The
already processed development cache contains 303/73/85 train/validation/test
trips. February 1--7 was previously opened for ETA work, so it is development
data for this decision and cannot support a new clean-test claim.

Initial non-neural development floors, before selecting JEPA hyperparameters:

| Model | Validation multi-horizon distance MAE |
|---|---:|
| Constant-course kinematics | 5.876 km |
| Trajectory GBT | 3.336 km |

The numbers above were produced from the existing prefix cache with horizons
near 0.5/1/2 hours. They are diagnostic development measurements, not promotion
evidence.

The frozen-capacity three-seed regularizer comparison selected VICReg using
only the full-trajectory validation MAE of the hybrid GBT:

| Variant | Hybrid validation MAE, mean +/- seed SD | Gain vs raw GBT | Minimum effective rank | Collapsed |
|---|---:|---:|---:|---:|
| Phys-JEPA + VICReg | **3.124 +/- 0.056 km** | **6.37%** | 9.57 | no |
| Phys-JEPA + no regularizer | 3.129 +/- 0.037 km | 6.22% | 8.57 | no |
| Phys-JEPA + VISReg | 3.143 +/- 0.069 km | 5.81% | 8.92 | no |
| Phys-JEPA + SIGReg | 3.204 +/- 0.047 km | 3.97% | 9.42 | no |
| Plain JEPA + VISReg | 3.313 +/- 0.066 km | 0.70% | 5.37 | no |

This establishes two development facts: none of the tested representations
collapsed, and the known-physics residual is more important than the choice
among anti-collapse losses. It does not establish a clean future result.

The first sparse head concatenated all 64 latent dimensions and overfit 27
labelled train trips. A validation-only feature/head sweep therefore froze a
smaller policy: JEPA distance/speed forecasts for ETA and eight train-only PCA
state components plus JEPA distance forecasts for delay. With identical labels
for raw and hybrid heads, ETA validation MAE changed from 1.318 +/- 0.064 to
1.300 +/- 0.064 hours (1.36%) and delay AUPRC from 0.754 +/- 0.077 to
0.761 +/- 0.052. Multi-horizon features were
marginally better than the one-two-hour-horizon ablation on both probes. No
February 1--7 test metric was evaluated by that sweep.

The regenerated diagnostic artifact is
`outputs/noaa_ais_phys_jepa_development_v3/metrics.json`, SHA-256
`9baae4e2598ec7c2eed0f4b4aff7664a5729ddb49455ec5739e032cf25055d0a`.
Its already-opened development test improved trajectory and delay, while sparse
ETA improved 0.94%, just below the 1% gate. That test result did not change the
VICReg, feature or head selections and is not clean evidence.

## Acceptance rule

Architecture, regularizer, sparse features and head families use
development/validation only. The clean candidate is `Phys-JEPA + VICReg` and
must be non-collapsed and improve all predeclared product probes over their raw
trajectory floors:

- mean multi-horizon distance MAE;
- two-hour physical-progress deviation AUPRC;
- ETA MAE and delay AUPRC with 10% grouped labelled trips.

All strong baselines remain in the report even when they win. A positive result
cannot be manufactured by removing GBT or changing a metric after opening the
future test.

## Claim boundary

Public US AIS proves pipeline competence only. It does not prove Kaleido/Vigo
accuracy, a material port incident, causal action value, savings, ROI, operator
acceptance or production readiness. No action-conditioned claim is made because
AIS contains observations and context, not Kaleido operator decisions.

## Next falsifiable step

1. Freeze config, thresholds, code and commit.
2. Download a new NOAA calendar interval without inspecting its outcomes.
3. Commit its file hashes before target construction.
4. Run the frozen three-seed protocol once from a clean worktree.
5. Reject promotion if the clean future interval does not pass the gates.
