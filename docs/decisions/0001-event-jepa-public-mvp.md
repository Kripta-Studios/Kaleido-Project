# Decision 0001: Event-JEPA in the public-data MVP

Date: 2026-07-15

Status: accepted for `smoke_only` implementation

## Decision

Implement and train an action-free Event-JEPA now on the public Warehouse Outbound
event log. Evaluate whether its temporal representations improve remaining-time
prediction over the frozen M3 quantile-boosting baseline and the M4 GRU and
ProcessTransformer baselines. Do not call this component an operational world model.

In parallel, implement a separate action-conditioned synthetic benchmark whose action
effects are known because FlowTwin generates them. That benchmark demonstrates whether
the architecture can recover an injected action signal; it does not establish causal
action value in Warehouse Outbound or Kaleido.

Promotion to a Kaleido world-model or action-value claim remains blocked until Kaleido
provides timestamped controllable actions, immutable plan revisions, outcomes and a
held-out protocol. This claim boundary does not block the public MVP.

## Why Event-JEPA was described as blocked

Three different gates had been compressed into one word:

1. **Implementation gate.** M1-M4 had to exist before M5 under `AGENTS.md`. M1-M3
   existed, while the corrected M4 run was still pending. This is procedural and can be
   resolved locally.
2. **Public MVP gate.** Public data can test whether JEPA representations add predictive
   value over other architectures. No real action field is required for this test. This
   gate is now open.
3. **Operational world-model gate.** Action-conditioned counterfactual or planning claims
   require verified operator-controllable actions and outcomes. Public Warehouse Outbound
   has obfuscated observations and no action contract. This gate remains closed.

The innovation is therefore not removed. It is decomposed into a falsifiable
representation experiment now and a world-model validation later.

## Translation of the 29 supplied papers

All links below are primary paper pages or the authors' paper endpoint. The 2026 papers
are recent preprints and are design input, not consolidated evidence.

| Paper | Useful idea | FlowTwin decision | What it does not solve |
|---|---|---|---|
| [I-JEPA (2301.08243)](https://arxiv.org/abs/2301.08243) | Predict embeddings of informative, relatively large target blocks from distributed context; asymmetric context/target views prevent trivial copying. | Encode the causal event prefix as context and future event-state windows as targets. Never reconstruct raw rows. | Images are not irregular object-centric logistics traces; masking success does not prove temporal action dynamics. |
| [LeWorldModel (2603.19312)](https://arxiv.org/abs/2603.19312) | Stable end-to-end next-latent prediction with a shared encoder, predictor and SIGReg, without EMA/stop-gradient; actions enter the predictor. | Use its compact end-to-end recipe for the first Event-JEPA, omit the action input in the public representation run, and add it only in the synthetic-action run. | Its planning results require observation-action trajectories and do not transfer to obfuscated warehouse events. |
| [HWM (2604.03208)](https://arxiv.org/abs/2604.03208) | Shared latent space across temporal scales; long-horizon predictions become subgoals for short-horizon prediction; macro-actions compress action chunks. | Predict direct event/operation horizons and later add event-operation-shift hierarchy. | No observed macro-actions exist in the public log; robotic MPC is not a port intervention policy. |
| [Intuitive Physics (2502.11831)](https://arxiv.org/abs/2502.11831) | Prediction error can support violation-of-expectation tests better than pixel/text baselines. | Expose latent surprise as a diagnostic process-deviation signal and test lead time. | Surprise is not automatically incident probability, root cause or causal early warning. |
| [LeJEPA (2511.08544)](https://arxiv.org/abs/2511.08544) | An alignment objective plus Sketched Isotropic Gaussian Regularization (SIGReg) prevents collapse and gives a usable selection loss. | Default anticolapse method: Epps-Pulley characteristic-function matching over resampled random projections. Record variance, rank and isotropy as independent diagnostics. | Its optimality assumptions do not guarantee that logistics semantics or actions are identifiable. |
| [DINOv3 (2508.10104)](https://arxiv.org/abs/2508.10104) | Careful data preparation and Gram anchoring protect dense features during long training. | Deduplicate and rebalance process variants; consider Gram/layer anchoring only if pre-cutoff images enter scope. | It is not needed for the current event-only critical path. |
| [Spectral SSL (2205.11508)](https://arxiv.org/abs/2205.11508) | The pairwise relation graph induced by views determines the recovered global/local spectral structure; bad relations produce bad representations. | Treat target-view construction as a semantic contract. Compare true temporal pairing with shuffled timestamps/operations. | Anticolapse alone cannot repair semantically wrong positive pairs. |
| [V-JEPA 2.1 (2603.14482)](https://arxiv.org/abs/2603.14482) | Dense predictive loss and deep self-supervision improve local and global representations. | Reserve intermediate-layer probes and deep loss as ablations if the small model underuses event-level structure. | Video-token density and scale do not exist in the sparse public event log. |
| [DINOv2 (2304.07193)](https://arxiv.org/abs/2304.07193) | Curated diversity and frozen-feature evaluation are as important as scale. | Evaluate frozen Event-JEPA probes and report variant balance; use DINO only for valid pre-cutoff photos. | Generic vision transfer cannot replace operational temporal data. |
| [PatchCore (2106.08265)](https://arxiv.org/abs/2106.08265) | A representative nominal-feature memory bank works in cold-start/few-anomaly regimes. | Add a nearest-nominal latent-distance baseline for diagnostic anomaly detection. | Image anomaly AUROC does not establish operational deviation risk or lead time. |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) | Hierarchical JEPA, latent-variable prediction, configurable world models and multiple plausible futures. | Use it as the architectural roadmap: event-operation-project hierarchy and explicit uncertainty. | It is a position paper, not evidence that a particular Event-JEPA will beat simple baselines. |
| [When Does LeJEPA Learn a World Model? (2605.26379)](https://arxiv.org/abs/2605.26379) | Linear identifiability is shown for stationary additive-noise transitions, with Gaussian latents uniquely supporting the stated guarantee. | Log stationarity, autocorrelation and Gaussian diagnostics; present the guarantee as inapplicable unless its assumptions are tested. | Sparse logistics processes with redesigns, policies and censoring need not satisfy those assumptions. |
| [V-JEPA 2 (2506.09985)](https://arxiv.org/abs/2506.09985) | Action-free representation pretraining can be followed by a much smaller action-conditioned post-training stage. | This directly motivates the two-stage MVP: public action-free Event-JEPA now, action-conditioned synthetic/Kaleido post-training later. | Web-video scale and robot actions do not create logistics actions in our data. |
| [VL-JEPA (2512.10942)](https://arxiv.org/abs/2512.10942) | Predict continuous language-target embeddings and decode selectively. | Later align timestamp-valid incident notes to event states and decode explanations only when needed. | Notes can leak outcomes; no text path is enabled until cutoff and redaction audits pass. |
| [EB-JEPA (2602.03604)](https://arxiv.org/abs/2602.03604) | Modular energy-based components, single-GPU examples and ablations. | Keep encoder, target encoder, predictor, loss and probes separate and checkpoint-testable in CPU smoke mode. | A library design is not performance evidence. |
| [JEPA-DNA (2602.17162)](https://arxiv.org/abs/2602.17162) | Hybrid token/generative and global latent-predictive objectives can be complementary. | Add an auxiliary next-event head as an ablation, never silently fold it into the JEPA result. | Genomic sequence regularities do not transfer directly to process logs. |
| [Var-JEPA (2603.20111)](https://arxiv.org/abs/2603.20111) | An explicit ELBO turns deterministic JEPA into a probabilistic latent model and supports latent uncertainty; tabular experiments are relevant. | Compare deterministic SIGReg JEPA with a variational target only after the simpler model is stable; keep conformal quantile heads as the floor. | A latent posterior is not automatically calibrated remaining-time or risk uncertainty. |
| [FF-JEPA (2606.09311)](https://arxiv.org/abs/2606.09311) | Action-free high-level prediction can generate subgoals while a short-horizon action-conditioned component handles control. | Keep representation learning and action conditioning as separate stages; consider shift/operation targets above event transitions. | Its long-horizon planning evidence is preliminary and robotic. |
| [MIRA](https://mira-wm.com/paper/) | Separate action streams from multiple agents to attribute scene changes and evaluate physical behavior, not appearance alone. | In future Kaleido data, represent operator, resource/controller and external actor action streams separately. | A 5B-parameter multiplayer video model is outside MVP compute and domain. |
| [VISReg (2606.02572)](https://arxiv.org/abs/2606.02572) | Separating variance scale from sliced-Wasserstein distribution shape gives gradients under collapse and helps long-tailed/low-rank data. | Use as the predefined fallback/ablation if SIGReg shows vanishing gradients, poor effective rank or long-tail instability. | It still requires correctly paired views and does not create missing dynamics. |
| [Fast LeWorldModel (2606.26217)](https://arxiv.org/abs/2606.26217) | Predict action-prefix outcomes at several horizons in parallel, avoiding autoregressive error accumulation. | Predict all selected future event-state horizons in one forward pass; avoid recursively feeding predicted latents during MVP scoring. | Action-prefix prediction is action-free only after removing its planning interpretation. |
| [On Training in Imagination (2605.06732)](https://arxiv.org/abs/2605.06732) | Dynamics and reward errors affect imagined return; smoother/lower-Lipschitz latent maps tighten error bounds. | No learned policy training until transition and outcome errors are separately measured; monitor sensitivity/Lipschitz proxies. | It cannot justify simulated savings when dynamics or cost labels are wrong. |
| [MJEPA (2606.25225)](https://arxiv.org/abs/2606.25225) | Cross-modal prediction is essential; a shared encoder without cross-modal alignment can underperform unimodal baselines. | Add modality-specific and cross-modal ablations only when modalities are genuinely time-aligned. | Concatenating notes/photos to events is not multimodal learning. |
| [AdaJEPA (2606.32026)](https://arxiv.org/abs/2606.32026) | Self-supervised test-time transition updates can recalibrate a model under distribution shift. | Future adaptation is shadow-only: frozen reference, replay, trust region, acceptance gate, signed version and rollback. | Online updates are unsafe without observed post-action transitions and anti-forgetting evidence. |
| [Phys-JEPA (2606.16076)](https://arxiv.org/abs/2606.16076) | Split latent state into physically constrained and residual components; enforce consistency in the latent transition. | If Kaleido supplies quantities/resources, constrain conservation, nonnegative work and plan milestones while retaining a residual latent. | Obfuscated warehouse columns do not support invented physical laws. |
| [SkyJEPA (2606.23444)](https://arxiv.org/abs/2606.23444) | Structured simulation data generation plus interpretable probes can test long-horizon latent dynamics. | Generate synthetic actions through a documented simulator and train operational probes for queue, work-in-progress and service state. | Sim-to-real success in quadrotors does not establish warehouse-to-port transfer. |
| [Neuro-JEPA (2606.14957)](https://arxiv.org/abs/2606.14957) | Heterogeneous evaluation and simple baselines reveal when large foundation approaches are inconsistent. | Retain simple medians/linear/boosting and report per-project/process groups; JEPA is rejected if it does not add value. | Scale and multimodal medical results do not imply logistics gains. |
| [WorldDP (2606.08775)](https://arxiv.org/abs/2606.08775) | Object-centric high-level world models separate entities and stages before low-level execution. | Preserve OCEL objects and compare object-graph tokens with flattened traces; keep scenario ranking separate from representation learning. | Diffusion-policy execution is out of scope for a read-only product. |
| [TDV (2606.15956)](https://arxiv.org/abs/2606.15956) | Model the next representation as current representation plus a learned temporal-difference/motion code. | Add a temporal-difference predictor as a lightweight action-free ablation. | “Past causes future” is not enough for intervention-level causal attribution. |

## Technical blockers and adopted remedies

| Blocker | Paper-derived remedy | Gate |
|---|---|---|
| Constant or low-rank collapse | SIGReg default; VISReg ablation; std/effective-rank/isotropy diagnostics | Training fails if diagnostics cross predefined collapse limits. |
| Stochastic future | Direct P50/P90 heads now; Var-JEPA latent distribution later | Improvement must include calibration/coverage, not MAE alone. |
| Long/irregular horizons | Parallel direct horizon targets; no deep autoregressive rollout | Compare one versus multiple horizons on validation. |
| Sparse event labels | Self-supervised future-latent pretraining plus frozen probe | Compare frozen and fine-tuned encoder against supervised-only baselines. |
| Object relationships | Object-centric tokens/graph baseline | Graph versus flattened trace is mandatory when relationships exist. |
| Missing public actions | Action-free representation run plus separate generated-action benchmark | No action claim on Warehouse Outbound. |
| Drift/process redesign | Frozen reference and shadow adapters only | No reference update without replay, trust gate and rollback. |
| Surprise confused with risk | Separate latent-distance diagnostic from supervised risk head | Report lead time, false alerts and calibration independently. |

## Action-free Event-JEPA experiment contract

- Dataset: `warehouse_outbound_event_log_2025`, export 1.
- Source SHA-256:
  `2e6767ddd724304fe3716b0c768233e4a8db520f25780ce5ccd3795811348416`.
- Split: the exact chronological operation-grouped M3 v2 manifest.
- Context view: only activity/timing tokens at or before `prediction_cutoff`.
- Target views: future event-state embeddings at direct horizons selected without test.
- Default loss: future-latent squared error plus SIGReg.
- Downstream: frozen and fine-tuned P50/P90 remaining-time heads.
- Baselines: median, Kaplan-Meier, ridge, quantile boosting, GRU and
  ProcessTransformer.
- Seeds: at least 11, 42 and 73.
- Selection: self-supervised checkpoint and head choices on validation only.
- Test: used once after choices; `test_influenced_choice=false`.
- Claim state: `smoke_only`.

The public Event-JEPA is retained only if it is stable and improves an operational metric
over a baseline under seed uncertainty, or provides a separately measured calibration,
robustness or data-efficiency advantage. It is honestly rejected if it does not.

## Synthetic action generator contract

The public activities are obfuscated observations, not verified actions. Synthetic actions
therefore live in a separate calibrated queue/process simulator and are never inserted into
the raw public event log as if observed.

Planned actions:

| Action | Eligibility and timestamp | Structural effect | Logged cost/side effect |
|---|---|---|---|
| `expedite_release` | Waiting job, before service starts | Multiplies the next queue wait by a seeded factor below one | Expediting/overtime cost |
| `add_temporary_capacity` | Congestion above a train-derived threshold | Adds one bounded service-capacity unit for a fixed interval | Staffing/equipment cost and expiry |
| `priority_dispatch` | Multiple eligible jobs in queue | Changes queue order, not service duration | Delay transferred to non-priority work |
| `reroute_parallel_station` | Compatible parallel resource available | Changes resource and its queue | Transfer/setup time |
| `planned_hold` | Explicit constraint active | Adds a deliberate delay before release | May reduce a separately modeled incident exposure; never assumed beneficial by default |

The behavior policy uses train-derived state thresholds plus randomized exploration with
logged propensities to ensure overlap. Every row records pre-action state, action time,
eligibility, propensity, seed, intended structural equation, realized stochastic effect and
support status.

Required ablations are current prefix only, context only, correct actions, shuffled
actions, action only, no JEPA objective, one versus multiple horizons, no anticolapse
regularizer, frozen versus fine-tuned encoder, and object graph versus flattened trace.
Correct actions must beat shuffled actions across seeds before even a synthetic action-signal
claim is allowed.

## Consequences

- M5 public implementation is unblocked after the corrected M4 artifacts exist.
- “Event-JEPA” is acceptable for the public representation component.
- “Event World Model” is reserved for a verified action-conditioned transition model.
- The dashboard may demonstrate synthetic scenarios, but must label them simulation and
  must not imply realized savings.
- Public results can support a technical MVP demo and data-conversation with Kaleido;
  they cannot substitute the pilot acceptance protocol.

## Next falsifiable step

Train the action-free Event-JEPA on the frozen public split for three seeds, compare it to
M3/M4, and publish collapse diagnostics and P50/P90 metrics. If it fails to improve or to
offer a measured robustness/calibration benefit, keep boosting/sequence baselines as the
MVP predictor and reject the JEPA promotion.
