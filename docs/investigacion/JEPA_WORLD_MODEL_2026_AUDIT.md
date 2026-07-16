# JEPA and world-model paper audit for Kaleido FlowTwin

Last updated: 2026-07-17. This document routes the papers supplied by the user
to implemented experiments. It is not a claim that FlowTwin reproduces every
paper or reaches state of the art.

## Product decision

The selected research product is a read-only **Port Call Deviation Twin** for
Shipping Board and Freight Intelligence. It consumes a vessel approach prefix,
predicts latent and physical state at 0.5/1/2 hours and exposes:

- distance/speed forecasts and a comparison with constant-course physics;
- a two-hour physical-progress shortfall score;
- a JEPA state used by a GBT head when arrival/deviation labels are scarce;
- latent surprise once a future observation arrives;
- cutoff, model version, uncertainty and provenance.

JEPA is not used as a replacement for the strong ETA GBT. The product candidate
is `trajectory GBT + Phys-JEPA state/forecast`. This choice follows the central
JEPA use case: learn reusable dynamics from abundant observations, then use a
smaller labelled sample for an operational head.

## Directly implemented ideas

| Source | Idea used | FlowTwin implementation/gate |
|---|---|---|
| [I-JEPA, 2301.08243](https://arxiv.org/abs/2301.08243) | predict target embeddings instead of reconstructing raw input | online context encoder, stop-gradient target encoder and latent predictor |
| [LeJEPA, 2511.08544](https://arxiv.org/abs/2511.08544) | Gaussian latent regularization through characteristic-function projections | SIGReg implementation and no-collapse ablation |
| [VISReg, 2606.02572](https://arxiv.org/abs/2606.02572) | variance/invariance/sketching regularization | VISReg implementation, collapse diagnostics and comparison with SIGReg/VICReg/none |
| [LeWorldModel, 2603.19312](https://arxiv.org/abs/2603.19312) | latent transition model and surprise for physically implausible evolution | multi-horizon latent transition error and future-observation surprise |
| [Fast LeWorldModel, 2606.26217](https://arxiv.org/abs/2606.26217) | predict several horizons in parallel rather than autoregressive rollout | parallel 0.5/1/2-hour horizon tokens |
| [Phys-JEPA, 2606.16076](https://arxiv.org/abs/2606.16076) | separate known physical evolution from unresolved residual dynamics | constant-course future is passed to the latent predictor; decoder learns a residual |
| [V-JEPA 2, 2506.09985](https://arxiv.org/abs/2506.09985) | observational pretraining before task-specific adaptation | AIS future-state pretraining excludes arrival time, remaining time and delay label; downstream heads are separate |

The implementation is a domain adaptation, not an exact reproduction. AIS is a
multivariate event sequence rather than pixels/video; the physical component is
a constant-course port-approach prior rather than a simulator.

## Papers that shape the next experiments

| Source | Relevance | Decision |
|---|---|---|
| [Hierarchical Planning with Latent World Models, 2604.03208](https://arxiv.org/abs/2604.03208) | multi-scale state/action reasoning | later only if Shipping Board supplies immutable plans/actions; no action claim from AIS |
| [When Does LeJEPA Learn a World Model?, 2605.26379](https://arxiv.org/abs/2605.26379) | conditions under which a JEPA representation contains dynamics | require external probes/forecast gains, not latent loss alone |
| [FF-JEPA, 2606.09311](https://arxiv.org/abs/2606.09311) | latent planning over longer horizons | out of scope until the transition model passes prediction gates and actions are real |
| [AdaJEPA, 2606.32026](https://arxiv.org/abs/2606.32026) | adaptive latent world model | possible shadow adapter only, with frozen reference, trust region, replay and rollback |
| [On Training in Imagination, 2605.06732](https://arxiv.org/abs/2605.06732) | rollout error and representation smoothness affect imagined policy value | no imagined action ranking before model-error gates; simulator remains separate |
| [Var-JEPA, 2603.20111](https://arxiv.org/abs/2603.20111) | variational predictive/generative bridge | existing warehouse research only; not selected for AIS before the deterministic floor passes |
| [V-JEPA 2.1, 2603.14482](https://arxiv.org/abs/2603.14482) | dense self-supervised features | relevant if Kaleido supplies CCTV/visual occupancy, not required for AIS MVP |
| [Intuitive physics from video, 2502.11831](https://arxiv.org/abs/2502.11831) | latent prediction can encode physical regularities | supports physical surprise concept; no claim of intuitive physics from AIS |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | conceptual JEPA/world-model architecture | product remains advisory and read-only; no autonomous controller |
| [MIRA world model](https://mira-wm.com/paper/) | interactive multi-agent world modelling | relevant to future vessel/resource interaction, not the single-vessel public benchmark |

## Useful representation work, but not the selected temporal model

| Source | Why it is not the main implementation |
|---|---|
| [DINOv2, 2304.07193](https://arxiv.org/abs/2304.07193) and [DINOv3, 2508.10104](https://arxiv.org/abs/2508.10104) | strong visual encoders; applicable to images, not AIS dynamics |
| [Spectral view of SSL, 2205.11508](https://arxiv.org/abs/2205.11508) | theoretical interpretation of contrastive/non-contrastive embeddings; informs diagnostics only |
| [PatchCore industrial anomaly detection, 2106.08265](https://arxiv.org/abs/2106.08265) | appropriate image anomaly baseline if port inspection photos become available |
| [Lightweight EBM-JEPA library, 2602.03604](https://arxiv.org/abs/2602.03604) | implementation reference; FlowTwin keeps a small PyTorch CPU model and its own contracts |
| [VL-JEPA, 2512.10942](https://arxiv.org/abs/2512.10942) | vision-language prediction requires image/text data absent from the AIS benchmark |
| [JEPA-DNA, 2602.17162](https://arxiv.org/abs/2602.17162) | domain-specific genomic grounding does not transfer directly |
| [MJEPA audio-visual, 2606.25225](https://arxiv.org/abs/2606.25225) | future multimodal CCTV/radio option, not current scope |
| [Sparse multimodal neuroimaging JEPA, 2606.14957](https://arxiv.org/abs/2606.14957) | sparse multimodal fusion is conceptually useful but the domain/objective differs |
| [SkyJEPA, 2606.23444](https://arxiv.org/abs/2606.23444) | long-horizon control is relevant to robotics, not an advisory port-call predictor |
| [Object-centric world model + diffusion policy, 2606.08775](https://arxiv.org/abs/2606.08775) | object hierarchy is useful for Trace Port, but diffusion policy/action control is out of scope |
| [Temporal-difference visual representations, 2606.15956](https://arxiv.org/abs/2606.15956) | alternative temporal SSL objective reserved for an ablation after the Phys-JEPA gate |

## Evidence policy

Every run must report dataset/export/hash, grouped chronological split, all
baselines, seeds, validation-only selection, prior test exposure and claim
state. A non-collapsed embedding is necessary but insufficient. Promotion needs
an operational improvement over raw trajectory GBT on a new calendar holdout.

Public NOAA data proves only pipeline competence. Kaleido value requires an
export or shadow feed with declared destination/plan revisions, AIS or zone
events, outcome definitions and operator review.

## Next falsifiable step

VICReg was selected over VISReg, SIGReg and no regularizer using only hybrid
trajectory MAE on the already opened development validation split. Freeze that
`Phys-JEPA + GBT` protocol and commit before downloading NOAA 2025-02-08 through
2025-02-14. Open that interval once and reject promotion if any required
clean-test gate fails.
