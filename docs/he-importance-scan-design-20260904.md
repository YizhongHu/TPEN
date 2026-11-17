# TPEN helium importance scan — consolidated design

Date: 2026-09-04 (America/New_York). Status: revised design, not permission to execute.

This document incorporates the operator's accepted A8, B-grid, optimizer, seed, and checkpoint recommendations. It replaces the active proposal in `2026-09-04-tpen-helium-importance-scan-operator-revision.md`. The old A–L fraction, its control-point discussion, and the old 256-config acceptance criterion are historical, not this scan. Their original source is preserved unchanged; this document does not redefine those old columns.

The companion [implementation step plan](2026-09-04-tpen-helium-importance-implementation-step-plan.md) references this document for scientific coordinates. Do not maintain another copy of the factor tables there or in a Task Orchestrator note.

Evidence labels apply throughout:

- **RELAYED:** operator instructions, another agent's receipt, or published literature/facility documentation. A receipt is not an independent measurement by this author.
- **MEASURED:** directly inspected code, records, or design arithmetic; not necessarily a runtime experiment. The type of observation is stated.
- **INFERRED:** design choices, deductions, predictions, and proposed gates. All proposed numeric settings and experiment counts below are INFERRED unless explicitly identified as operator-accepted. They are not helium results.

## 1. What is settled, and what is not

### 1.1 Operator decisions incorporated — RELAYED

| Decision | Current disposition |
|---|---|
| Scientific purpose | Find which design choices matter, as well as whether a frozen TPEN procedure reaches helium chemical accuracy. Do not collapse the programme to one small contrast. |
| He-v1 | Pilot with no independent seed scan. Mechanics, instruments, and hypotheses only; no accuracy conclusion or scientific elimination of an axis. |
| 25k | Mechanics-only checkpoint, not an accuracy experiment. |
| Fixed architecture | One TPEN layer, body order 2, canonical paths, virtual support fixed at 2, completion `sum`. |
| Required trainability | Readout weights, e–e ranges, and e–n curvature/range remain trainable. |
| A8 update | Exactly `x+u`, `u`, and `x+Gaussian(r)u`; Gaussian width fixed at one bohr with the existing all-electron-radius semantics. |
| B | Raw versus distance-augmented inputs crossed with absent versus the minimal two-parameter symmetric log-Jastrow. |
| B deferrals | B6 path subsets deferred; B7 and the broader B9–B13 programme deferred. The accepted two-parameter Jastrow is the narrow exception, not approval of general Jastrow networks. B8 uses the current aggregation pipeline only. |
| O(3) | No hard invariant replacement/projection in this scan. Keep Cartesian routes. Treat the operator's ansatz-compatibility warning seriously. |
| Optimizers | VMC-first: KFAC, SR, SPRING, linear method; Adam is the generic control. No RAdam/Lion/SGD tournament. |
| Other optimization | Constant schedules; parameter-group scan deferred. Final weights only; EMA/SWA deferred. |
| Sampler | MALA is stale and untested; exclude it from current science. Global-proposal families deferred. |
| Replication | Eight fresh training seed bundles per scientific screening point; twelve in integrative selection; forty-eight untouched confirmation seeds per protocol. |
| Horizon/storage | 50k is a first budget waypoint. Target checkpoint cadence 1k, subject to storage qualification; longer matched continuations are possible. |
| Reference firewall | Training configurations and training runners must never contain reference energy. Evaluation may contain it. |
| Resources | Plan a substantial scan; do not impose Cannon's queue bottleneck on Polaris or Frontier without evidence. |

### 1.2 Outstanding items

**INFERRED.** No further choice between the accepted A8/B options is needed to write this plan. Remaining details are explicit below, not inferred from unspecified operator preferences:

1. **Science release:** freeze the literal evaluation/inference implementation, adequacy criteria, practical-effect margins, and confirmation contract. Proposed values are supplied here. An estimator with an unjustified tail/dependence assumption is not launch-ready merely because the grid is fixed.
2. **Code development:** complete or qualify the reference firewall, replacement updater, minimal B additions, tail-safe coordinates, VMC update methods, and exact resume. Existing linear/hybrid mixing and MLP primitives must be reused, not reimplemented.
3. **Execution:** obtain current allocation/admission limits and clean He-cutover backend facts; measure representative runtime, packing, method-state memory, and checkpoint storage. Select the admitted horizon and storage policy from those facts.

None of these authorizes jobs in this design-only task. Any change to an accepted scientific choice returns to the operator; a release engineer may not substitute a convenient method or shrink seed counts silently.

## 2. Science surface

### 2.1 Target and permitted claims

**INFERRED contract.** Use the nonrelativistic, infinite-nuclear-mass, point-nucleus helium Hamiltonian in atomic units:

\[
H=-\tfrac12(\nabla_1^2+\nabla_2^2)-2/r_1-2/r_2+1/r_{12}.
\]

One nucleus at the origin, one up/one down electron, real fermionic wavefunction. Check the declared spin/exchange semantics; do not assume that a fixed spin-label input alone proves a pure singlet. No Gaussian confinement, finite-mass, relativistic, or QED terms. Do not claim physical-helium or general-system accuracy from this experiment.

**RELAYED literature, evaluation only.** Use `E_ref = -2.903724377034119598` hartree, rounded from high-precision nonrelativistic helium calculations; its rounding is negligible at the thresholds here. See [Sims and Hagstrom, high-precision Hy-CI calculation, Table 3](https://www.nist.gov/document/hehycipdf). Report both the conventional one-kcal/mol band (approximately 1.59360 mHa) and the stronger 1.00000 mHa band. Freeze the unit-conversion source/version in the evaluation manifest.

**INFERRED.** A fixed-model energy interval must lie wholly within `[E_ref-delta, E_ref+delta]` to certify that band. A suspiciously low energy is not a better result. A confidence interval is statistical evidence, not a theorem-grade variational upper bound. Keep separate:

- **Existence:** a frozen, independently evaluated model met the criterion.
- **Repeatability:** a frozen training procedure met it across the prespecified fresh seeds.
- **Importance:** a particular change affected a stated endpoint in the tested contexts and at the tested budget.
- **Theorem:** domain membership and rigorous energy enclosures; a separate future route, not implied by successful Monte Carlo.

### 2.2 Reference-energy firewall

**RELAYED requirement; INFERRED implementation contract.** The reference number, reference gaps, target-band thresholds, and reference-driven callbacks must not enter training configs, training runner inputs, training checkpoint metadata, training launch environments, or training decision logic. This includes inherited Hydra defaults and apparently unused fields. A training-only runner must reject the wrong schema before constructing the model, callbacks, evaluator, or output artifacts.

Training still uses its own sampled local energies and centered raw VMC energy gradient. That is essential to VMC and is not a reference target. The firewall concerns externally supplied answer values, not the Hamiltonian or sampled energy.

Use separate train and evaluation manifests/processes. Only the evaluation/analysis side resolves the literature value. Stage-level selection may use evaluation results and then issue a new training manifest containing the chosen method and fixed budget, but never the reference or a target-triggered stop rule. Import tests, nested-config fixtures, callback allowlists, and checkpoint/launch-manifest audits must enforce this separation. A substring scan alone is insufficient.

**MEASURED, source inspection.** The historical He-v1 training config contains a reference-energy field. It is not a compliant template. The generic runner's construction of a shared context/callbacks means filtering after training starts is too late. These are implementation tasks, not reasons to relax the requirement.

### 2.3 Literal control protocol

**INFERRED completed control, incorporating RELAYED accepted choices.** This is a reference protocol for the new scan, not another A–L control-point resolution. Every nonvaried coordinate below stays fixed.

| Owner | Literal value |
|---|---|
| System/numerics | Hamiltonian above; nucleus at origin; spatial dimension 3; float64; Coulomb distance floor 0.0 |
| Input | Raw Cartesian coordinates and spins; B additions off |
| Embedding | Order-specific MLPs; persistent channels `C1=C2=32`; each hidden width 128, one hidden layer, SiLU, bias enabled |
| Stack/path metadata | One TPEN layer; body order 2; canonical; virtual support 2; completion `sum`; immutable producer/aggregation path layout |
| Producer | Tensor-product mixing; vectorized implementation; slow implementation is an oracle, not a scientific arm |
| Aggregation | Producer activation per path, learned weighted path contraction/sum, then channel activation; SiLU at both activation sites |
| Update | `x+u`; no normalization or coordinate gate |
| Mixing initialization | Every producer weight entry initialized to 0.50; this is a constant initial value, not the standard deviation of a random distribution |
| Readout | Trainable weighted Pfaffian; `C2` pair channels; initial weights `1/C2`; current signed-log conversion epsilon `1e-12` on squared amplitude, with active-floor admission gate |
| e–e factor | Current cusp law, same/opposite radial-law slopes 1/4 and 1/2; both positive ranges trainable and initially 1.0; explicit current `eps=1e-12`, serving both distance regularization and positivity offset |
| e–n factor | Trainable `v(r)=-2r+c*r^2/(1+d*r)`; initial `c=.01,d=1`; tail-safe coordinates defined below |
| Additional log-Jastrow | Absent |
| Adam | `lr=.005`, betas `(.9,.999)`, epsilon `1e-8`, no decay, one group, constant schedule |
| Objective/protection | Raw centered energy-score update; no clipped-energy surrogate, auxiliary penalty, global gradient clip, or Adam displacement cap |
| Training sampler | Current persistent all-electron Gaussian random-walk Metropolis; proposal standard deviation .50 bohr; initial coordinate standard deviation 1 bohr; initial burn-in 100 proposals; 4096 walkers; 10 joint proposals per optimizer update |
| Weights/checkpoints | Final weights at the declared horizon; checkpoint target every 1000 completed updates; full resume at extension boundaries |
| Local energy | Analytic e–n cusp evaluator after qualification, with the identical Hamiltonian/factors; paired naive/slow and sparse independent derivative checks |

**INFERRED initializer convention.** For embedding and channel-MLP affine layers, use explicitly seeded PyTorch affine initialization: independent uniform weights/biases on `[-1/sqrt(fan_in),+1/sqrt(fan_in)]`, generated in float64 by named module streams. Path-aggregation matrices use named-stream Xavier uniform, gain 1: bounds `+/-sqrt(6/(Cm+Pm))` for `Cm` channels and `Pm` paths. Pin producer/path ordering. No construction-order global-RNG accident is a seed definition. B-distance paired starts use the special zero-added-column rule in §2.6.

**MEASURED numerical caveat.** The e–e factor's current distance is `sqrt(r12^2+eps^2)`, whereas e–n uses raw distances. Its exact zero-distance derivative is therefore regularized, even though the underlying radial law has the Kato coefficient. This is separate from the zero Coulomb-potential floor. Keep the current small regularization explicit; do not claim an exact e–e cusp theorem for it. **INFERRED:** regularization changes the trial wavefunction, not the target Hamiltonian; it does not automatically bias an exact evaluation of that trial wavefunction. Qualification must reconcile its actual kinetic derivatives with the unfloored potential and expose numerical error rather than silently switching ansatz. Removing/splitting that regularizer is a separate correctness amendment if needed, not a new speculative cusp family or an automatic edit.

**INFERRED tail-safety choice, not a scanned cusp family.** Use dimensionless atomic-unit coordinates

\[
d=10^{-12}+\operatorname{softplus}(q_d),\quad
\kappa=0.1+\operatorname{softplus}(q_\kappa),\quad
c=d(2-\kappa).
\]

The 0.1 inverse-bohr margin is a designer-selected conservative positive margin, not a learned physical decay constant. Initialize by inversion to each requested `(c,d)`. Both coordinates train; signed curvature remains possible; the outer e–n slope is `-kappa`. This changes optimization coordinates relative to unconstrained `c,d`; apply it in every arm and document that fact. Do not silently project or clip parameters after a step.

For finite parameters and the present smooth network families, polynomial growth bounds on the neural prefactor combined with this exponential envelope and a bounded Jastrow supply a sufficient decay route. This is not, by itself, a certified full operator-domain proof or a proof that the Pfaffian is nonzero. An active log-amplitude floor, zeros, and derivative consistency need their own gates.

### 2.4 A: exact active architecture coordinates

**RELAYED:** balanced channel choices and the A8 trio were accepted. **INFERRED:** the remaining detailed levels complete the proposed scan.

| ID | What changes | Exact levels |
|---|---|---|
| A1/A2 | Producer and nested linear path policy | Tensor; linear/coordinate-neighbor; linear/orbit-complete; hybrid/coordinate-neighbor; hybrid/orbit-complete — five packages, not six duplicated tensor entries |
| A3 | Virtual support | Fixed at 2; no V1 arm |
| A4 | Balanced persistent-state channels | `(C1,C2)=(16,16),(24,24),(32,32),(48,48)` |
| A5 | Unequal allocation at 64 total channels | `(40,24),(32,32),(24,40)`; the balanced point is shared with A4 |
| A6 | Mixing/aggregation activations | `(SiLU,SiLU)`, `(tanh,tanh)`, `(exact GELU,exact GELU)`, `(Softplus beta2,Softplus beta2)`, `(channel-MLP,SiLU)`, `(SiLU,channel-MLP)`, `(channel-MLP,channel-MLP)` |
| A7 | Each embedding MLP's hidden width/depth | `(64,1),(128,1),(128,2),(256,2)`; embedding activation SiLU |
| A8 | Feature update | Residual; replacement; Gaussian-gated residual, defined below |
| A9 | Constant producer-weight initialization | `.25,.50,1.0` |
| A10 | Opposite-spin e–e range initialization | `.25,1,4`; same-spin range initially 1 in every arm; both remain trainable |
| A11 | e–n `(c,d)` initialization | `(.0025,.25),(.01,1),(.04,4)`; common `c/d=.01`, hence identical initial outer slope |
| A12 | Readout initialization | Equal `w=1/C2`; signed L1 `w=z/sum(abs(z))`; random-direction L2 `w=z/(sqrt(C2)*norm(z))`, with seeded independent standard-normal `z` |

**INFERRED clarification.** A4/A5 change persistent channels throughout embedding output, producer input/output, aggregation, update, and readout (`pair_channels=C2`). They do not change A7 hidden widths. A6 channel-MLP means `Cm -> 4*Cm -> Cm`, one SiLU hidden layer, linear output, biases, no internal residual; it acts on the channel axis independently at each tuple/path. Its hidden width is a declared companion to `Cm`, not another accidental embedding-width change. Implement a channel-axis adapter; a last-axis MLP must not act on a particle axis. Softplus beta2 means stable `logaddexp(0,2*z)/2`, without a hard switch to a linear branch at a finite threshold; qualify its coordinate derivatives explicitly.

For hybrids, concatenate the linear paths before tensor paths and apply the shared producer activation once; then aggregate. For unary producers, use the declared canonical support metadata with no support exceeding 2; the policy names mean their helium-admissible subsets if a generic generator contains higher-support paths. Supply validated explicit metadata in that case. Do not invent a nonexistent unary `max_virtual_order` constructor argument. Both families use `sum` and vectorized kernels explicitly.

A12's two random normalizations share the same direction under paired seeds. They initially differ only by an amplitude scale, which cancels from exact local energy. This is a parameterization/learning-dynamics question, not extra physical expressivity. A global sign flip is likewise not a new arm.

**INFERRED A8 definition, using accepted values.** For each feature block, let `u` be the already aggregated update, with no normalization added:

| Name | Formula | What it asks |
|---|---|---|
| residual | `x_next=x+u` | Current skip connection |
| replacement | `x_next=u` | Must the initial embedding survive directly? |
| Gaussian residual | `x_next=x+g(R)*u`, `g=exp[-(r1^2+r2^2)/(2*sigma^2)]`, `sigma=1 bohr` | Should the learned correction fade when electrons are far from the nucleus? |

The gate is one configuration-level scalar, shared across orders, channels and tuples. It is not `exp(-u^2)`, not a channel norm, not a per-electron gate, not RMS normalization, and not a Gaussian multiplier on the whole wavefunction. The ungated skip path and Coulomb envelope remain. At this one-layer readout, order-1 output updates need not influence the energy; do not count a structurally inactive output branch as a failed trainability contract.

### 2.5 Why the minimal B study, rather than an unlimited cusp menu

**INFERRED.** “Pairwise” does not mean that the final log wavefunction is restricted to a sum of independent pair functions. Nonlinear pair kernels, channel mixing, and the logarithm of a readout can carry nonseparable correlations. Increasing channel count above two does not create more electrons or a new coalescence order. Neither two channels nor pairwise feature labels are a proof of inability to approximate a smooth three-distance helium correlation on a compact region.

There is a narrower, important distinction:

| Question | What can and cannot be concluded | Current action |
|---|---|---|
| Exact leading two-body cusps | A smooth Cartesian prefactor does not generically supply the nonanalytic radial cusp. Explicit e–n/e–e radial laws supply the intended slopes; the implemented e–e regularizer needs the separate caveat in §2.3. More width does not enforce exact identities. | Keep them, always trainable where required; qualify spherical-average slopes and numerical realization. |
| Smooth environment-dependent correlation | Not proven unrepresentable by this TPEN class. It can nevertheless be awkward to learn through the current parameterization. | Test only the minimal B grid below. |
| Triple-coalescence logarithms | A smooth, nonzero residual with only pairwise radial cusp factors does not generically contain the exact mixed logarithmic coalescence expansion. Width alone does not insert that singular term exactly. This does not prove a chemical-accuracy barrier. | Defer explicit Fock-log design unless fresh residual/accuracy evidence warrants it. |
| Asymptotic decay guarantees | A finite-window fit is not a global normalizability guarantee. Explicit tail control is useful even if a neural subnetwork could learn additional decay. | Common tail-safe coordinates, not a family scan. |
| Nodes/spin/symmetry sector | A positive Jastrow cannot repair a wrong nodal surface or spin contract. | Independent validity diagnostics; redesign if a genuine obstruction appears. |

These distinctions follow the role of coalescence conditions in [Kato's theorem](https://doi.org/10.1002/cpa.3160100201) and Coulomb-wavefunction regularity in [Fournais et al.](https://arxiv.org/abs/math-ph/0312060); the priority judgement is INFERRED, not a He-v1 accuracy finding.

The previous B4/B5 comment meant “make potentially useful correlation easier to learn,” not “TPEN has no access to angles” or “helium requires every proposed cusp feature.” The accepted B study asks whether better input information, a short direct log-amplitude route, or their interaction matters. No separate angular/radial-family proliferation is needed now.

Hard O(3) projection stays excluded. The operator's warning is a current-ansatz compatibility constraint, not a general theorem that Monte Carlo invalidates the helium Hamiltonian's rotational symmetry. Appending scalar distances while retaining Cartesian inputs does not hard-enforce O(3).

### 2.6 B: exact 2×2

**INFERRED formulas completing the accepted grid.** Define `r_i=|R_i|`, `r_ij=|R_i-R_j|`, and compute directly from squared distances

\[
q(r)=\frac{r^2}{\ell^2+r^2},\qquad \ell=1\ \mathrm{bohr}.
\]

| Package | Input features | Additional log-Jastrow |
|---|---|---|
| B00 | Raw Cartesian/spin | Absent |
| B10 | Raw plus bounded distances | Absent |
| B01 | Raw Cartesian/spin | Two terms below |
| B11 | Raw plus bounded distances | Two terms below |

Distance augmentation appends `q_i` to electron `i`'s one-body input and `(q_i,q_j,q_ij)` to the ordered `(i,j)` pair input. The pair channel contract owns their ordering and exchange law. Preserve the existing Cartesian/spin features. Do not broadcast pair information into one-body channels as an undocumented substitute for typed pair ingress.

For the two-electron system, the added log-amplitude is

\[
J=\theta_s q_{12}\frac{q_1+q_2}{2}
  +\theta_t q_{12}(q_1-q_2)^2,
\quad \theta_s=\theta_t=0\text{ initially}.
\]

Both coefficients train. No learned length, extra MLP, angle term, or amplitude clipping. The first term couples separation to the overall radial environment; the second distinguishes radially similar versus asymmetric configurations. The two new coefficients, rather than a large network, are the deliberately cheap direct route.

For matched B seed starts, copy shared raw-input weights exactly and set added input columns to zero. They remain trainable and must have a reachable gradient. With zero J coefficients this gives the same initial function for all four cells, removing initial-function changes as the first explanation. This is a specific learned-feature experiment, not a comparison of random feature-initialization recipes.

**INFERRED algebra.** Both terms are bounded for finite coefficients, symmetric under electron exchange, smooth in Cartesian coordinates, and multiply the wavefunction by positive `exp(J)`; they preserve its zeros and sign. Their e–e approach contribution starts at second order. At an e–n approach, the `q12` dependence can have a nonzero directional derivative; its spherical average has zero first-order contribution. Thus test the actual Kato spherical-average condition, not the false stronger claim that every directional derivative vanishes. The bounded addition supplies no new exponential tail slope.

### 2.7 VMC-first optimizer grid

**RELAYED roster; INFERRED numerical tuning.** The following 120 cells replace the old 128-cell menu by removing RAdam. No schedules or parameter-group policies are crossed.

| Method | Exact levels | Cells |
|---|---|---:|
| Adam | `lr={.0005,.0015,.005,.015}` × `beta2={.99,.999}`; `beta1=.9`, epsilon `1e-8`, no decay | 8 |
| SR | `eta={.003,.01,.03,.1}` × relative damping `{1e-4,1e-3,1e-2,.1}` × squared Euclidean update-norm bound `{1e-4,1e-3}` | 32 |
| KFAC | `eta={.01,.03,.05,.1}` × relative damping `{1e-4,1e-3,1e-2,.1}` × factor EMA `{.95,.99}` | 32 |
| SPRING | `eta={.002,.006,.02,.06}` × relative damping `{1e-4,1e-3}` × history `mu={.98,.99}` × squared Euclidean update-norm bound `{1e-4,1e-3}` | 32 |
| Linear method | Derivative-block shift `{1e-4,1e-3,1e-2,.1}` hartree × overlap-metric trust radius `{.01,.03,.1,.3}` | 16 |
| Total | 112 VMC-specialized cells, eight Adam cells | 120 |

The old phrase “step cap” was underspecified. Here a bound `C` means `||Delta theta||_2^2 <= C`, not `||Delta theta|| <= C`; persist the unscaled SPRING history separately. These are parameter-coordinate-dependent protections, so their activity must be reported across widths/tail coordinates. They are not global gradient clipping.

Use real centered log-amplitude scores `O`, divided by `sqrt(B)`, and `e=(mean(E_L)-E_L)/sqrt(B)`. Then `S=O^T O`, and the raw energy gradient is `-2 O^T e`. SR's unscaled direction solves `(S+lambda I) phi=O^T e`; apply `eta` and the stated norm bound. Set `lambda=lambda_rel*max(trace(S)/P,1e-12)` with `P` the complete declared parameter count, and report both damping values and inactive coordinates. This fixes the factor-of-two/LR convention rather than hiding it in a solver.

SPRING uses the published stabilized projected-history recurrence, centered/scaled as above, including the rank-one constant-mode stabilization and zero initial history. Its first direction with zero history must match the corresponding damped MinSR direction. Do not replace SPRING by SR plus ordinary momentum. See [Goldshlager et al., Algorithm 1](https://arxiv.org/html/2401.10190v2).

KFAC uses the same log-score geometry and half-gradient direction convention; curvature factors every update, inverse refresh every 10, zero separate momentum, predicted-KL cap `1e-3`. Relative damping is normalized to the trace per dimension of each declared approximate block. The implementation must specify exact factor damping and scalar/readout/cusp blocks; no unregistered tensor can silently receive identity preconditioning. Its TPEN block partition and damping implementation remain a code-qualification artifact, frozen before O1. KFAC is relevant to VMC, not just generic ML; see its use in [Pfau et al.](https://doi.org/10.1103/PhysRevResearch.2.033429).

The linear method targets the full wavefunction-plus-derivative tangent space and shifts the derivative sector, not the constant state. Freeze its Hamiltonian estimator, overlap-rank cutoff, eigenvector selection, and trust-step rule before O1, with an independent small oracle. Sample-space/matrix-free realizations must preserve that declared target; an arbitrary trainable-parameter subset is a different method, not a memory optimization. See [Toulouse and Umrigar](https://arxiv.org/abs/physics/0701039). If full coverage is infeasible, retain explicit unavailable cells rather than silently redefining LM.

Dense SR, sample-space MinSR, and converged matrix-free SR are execution realizations of the same scientific update only when centering, damping, direction, cap, and resulting step agree at frozen residual/vector tolerances. One nominal solver name does not establish parity.

### 2.8 Exact staged inventory

**INFERRED design.** Use direct, mixed-level factorial blocks with literal settings. No unexplained `-1/+1` notation, missing D-optimal rows, or inherited 256-point acceptance clause. A scientific “point” below is a complete model, training method and fixed budget, not one checkpoint.

Two optimizer contexts are fixed before tuning:

- **O-raw:** the literal control model, B00, residual.
- **O-composite:** hybrid/coordinate-neighbor, channel-MLP at both sites, B11, residual; otherwise control.

| Stage | Construction | Seeds per point | Maximum fresh training lineages |
|---|---|---:|---:|
| Q | 120 optimizer cells × 2 contexts, 2000-update mechanics runlets | 2 | 480 runlets |
| O1 | Same 120 × 2 contexts, prospective 50k scientific tuning | 8 | 1920 |
| O2 | Best two eligible cells per family, at most 10, × B00/B10/B01/B11 on the control model | 8 | 320 |
| A | 254 distinct architecture packages below × `{fixed Adam control, O*}` | 8 | 4064 |
| B | Five producer packages × three A8 updates × four B cells × the same two optimizers | 8 | 960 |
| R | Eight selected architecture packages × four B cells × the same two optimizers | 12 | 768 |
| F | One frozen selected procedure and the literal control, same declared final horizon | 48 | 96 |

`O*` is one qualified specialized VMC cell selected after O2, not the best Adam cell. O2's tuned Adam candidates are informative comparisons; they do not silently change the fixed `.005` control. If a family or O* is unavailable, mark the associated slots unavailable. Do not relabel/duplicate Adam to keep counts constant, and do not reassign scientific seed slots after seeing outcomes.

#### The 254 A packages, without hidden row generation

Hold B00 throughout A. Take the union of these three complete blocks:

1. **Core interactions:** five producer packages × seven A6 activation packages × three A8 updates = **105**. All width/embedding/initialization coordinates at control.
2. **Capacity:** six unique `(C1,C2)` pairs from A4/A5 × four A7 embedding packages × three A8 updates = **72**. Tensor/SiLU and control initialization. Three rows already occur in block 1.
3. **Initialization:** three A9 × three A10 × three A11 × three A12 levels = **81**. Tensor/SiLU, control widths/embedding, residual. One row already occurs in the union above.

Therefore `105+72+81-3-1 = 254`, deduplicated by the literal scientific configuration, not display name. The formula and tables fully determine the rows. A release manifest must instantiate every admitted row and report exclusions; a declaration of a count is not an instantiation receipt.

Core estimates include producer × activation × updater interactions. Capacity estimates are conditional on tensor/SiLU; initialization estimates are conditional on the control architecture. This is not an exhaustive Cartesian product of all A factors, and a quiet factor in one block is not globally unimportant. B estimates include input × Jastrow interactions across producers, updaters and optimizer context.

#### Selection rules

Q checks mechanics only. It cannot rank scientific quality or eliminate a finite, valid method for poor early energy. Mechanically failed cells receive a defect disposition before scientific launch. Any repair changing a method or level increments the manifest version and reruns its relevant qualification; no mutation of an arm during O1.

For O1/O2, rank eligible candidates lexicographically: complete valid seed/context blocks; most conventional-band certificates; smallest worst seed/context absolute-energy upper error bound; smallest median such bound; smallest context spread; smallest billed cost; stable lexical ID. A certificate uses the evaluation interval, never training energy. Select up to two per family in O1, then O* among specialized families in O2. Missing/statistically unresolved blocks are not evidence of inferiority.

For R, choose eight distinct A architectures: the literal control; the best two noncontrol architectures within each A8 update class (six); then the best remaining architecture overall. Rank architectures by the worse of their two optimizer summaries using the same fixed lexicographic rule. Deduplicate literal configurations. If fewer than eight are eligible, leave holes and report why; do not inspect confirmation to fill them. Cross these with the full B grid and both optimizers. This deliberately preserves all three updater families into integration instead of pruning an interaction solely on a weak marginal effect.

R selects one procedure once, using the same complete-block ranking on twelve seeds. **INFERRED promotion threshold:** at least 11/12 valid conventional-band certificates before spending the untouched confirmation packet. If none qualifies, revise or continue discovery; do not describe confirmation as already warranted.

### 2.9 Seeds, pairing, failures, and no-peeking

**RELAYED seed counts; INFERRED exact identities.** Reserve training seed labels:

| Stage | Labels |
|---|---|
| Q | 810001–810002 |
| O1 | 820001–820008 |
| O2 | 821001–821008 |
| A | 830001–830008 |
| B | 840001–840008 |
| R | 850001–850012 |
| F | 860001–860048 |

These are new stage namespaces, not old He-v1 seeds. The manifest must expand each into separate `model_initialization`, `training_sampler`, `method_randomness`, `diagnostic`, `evaluation_calibration`, and `evaluation_inference` streams. Use a versioned semantic hash into the supported RNG range and emit a collision-checked table. Do not use Python's process-dependent `hash()`.

Key paired shared model streams by stage, seed label, component identity and compatible geometry, not by the entire treatment ID; otherwise purported paired starts would actually be independent. Different architectures may share compatible tensors, but never claim identical functions without a test. Independent stream identities may intentionally have identical paired initial states; record these equivalence groups explicitly. Evaluation streams are checkpoint/chain-specific and never training streams. A code-version change creates a new cohort; retain the cohort's original namespace on resume.

Within a package, seed bundles are independent training attempts. Across treatments, seed labels are matched blocks. Checkpoints along one trajectory and multiple evaluation chains of one trained model are not additional training seeds. Persist all RNG states on resume; extending a lineage neither redraws its seed nor resets walkers/method history.

Scientifically bad or unstable seeds stay in the denominator and are not replaced. Infrastructure retries use the same scientific identity, preserve attempt records, and resume exact state when possible. If exact continuation is impossible, an explicitly declared restart from the same initial state is not a new independent seed. Unavailable cells, inference failures, and valid-but-inaccurate results must have distinct statuses.

### 2.10 Horizon and evaluation

All O1/O2/A/B/R discovery points initially receive 50k updates. Intermediate 1k checkpoints support dynamics, recovery, and future questions; 25k remains mechanics-only. Saving every 1k does not authorize evaluating every checkpoint for accuracy and selecting a lucky minimum.

**INFERRED continuation policy.** Budget optional R extensions to 100k and then 200k, retaining the same complete twelve-seed package blocks. Default at each chosen extension is to extend the whole eligible R comparison set, not just a currently losing seed. If only a subset can be afforded, freeze that subset before continuation and restrict matched-budget claims to it. The extension ladder is a proposed budget reserve, not an instruction to run it now.

Before F, freeze one terminal horizon `H*` from the discovery ladder, the final-weights rule, and both protocols. F's 48 seeds per protocol are fresh and run to that same horizon. No extension of only failing confirmation seeds, no best-checkpoint substitution, no EMA/SWA. An inconclusive confirmation requires a new registered experiment, not continued peeking.

If the winner is literally the control, F has one 48-seed protocol, not two copies under different names. The 96-lineage cost/error allocation is a conservative maximum; any such deduplication is recorded before F initialization.

**INFERRED provisional evaluation packet, to qualify before scientific release:**

| Class | Independent calibration chains/model | Independent inference chains/model |
|---|---|---|
| O1/O2/A/B/R terminal model | 2 × 16,384 retained records | 4 × 32,768 retained records |
| F terminal model | 4 × 16,384 retained records | 8 × 65,536 retained records |

Here a chain is one separately initialized trajectory, not an ensemble row renamed a chain. Starts use standard-normal coordinates, Gaussian RWM scale .5, initial burn-in 10,000 proposals, and 10 proposals between retained draws. These are designer defaults, not claims of adequate mixing. Vectorized independent chains are allowed with explicit chain/draw axes; do not multiply effective sample size by treating temporally correlated draws as independent. Qualification may set a larger fixed packet before O1/F; it may not stop a favorable confirmation interval early. Chain seeds are disjoint by phase.

Calibration data diagnose dependence/tails and validate the predeclared inference route; they are not reused as inference records. Preserve raw signed local energies, nonfinite counts, term reconstruction, node/floor flags, and provenance. Do not silently drop nonfinite rows. The same exact-model distribution and untruncated energy mean remain the target whichever estimator is used.

Conventional MCSE requires justified finite variance and a Markov-chain limit/long-run-variance argument. Compare independent long-run-variance implementations where that regime is admitted. A fitted tail index or apparently stable ESS does not prove those assumptions. If using truncation, add an independently justified tail-remainder bound; a clipped mean without that bound does not certify the raw energy. Without a defensible interval, label `inference unresolved` and block an accuracy certificate. No energy estimator is declared qualified by this document.

MAD/IQR are robust local-energy dispersion diagnostics, not energies. The He-v1 tail findings motivate these checks but select no winner, training horizon, or accuracy estimator by themselves.

### 2.11 Importance and confirmation decisions

**INFERRED proposed practical margins:** energy effect 0.25 mHa, local-energy MAD ratio `[0.8,1.25]`, training failure probability difference 0.05, and billed cost ratio `[0.9,1.1]`. Analyze energy with validated model-level estimates and training-seed blocks; analyze MAD separately. Define the primary typical-training effect as the median paired seed difference in true model energy, not the median of local energies. Do not quietly change to an across-seed mean.

Predeclare contrasts within each factorial block, with familywise multiplicity control and interactions. Eight seeds is an initial scientific replication allocation, not a power guarantee for a 0.25 mHa equivalence margin. A finite-seed interval construction and its dependence assumptions must be frozen in the analysis release. Widen/augment discovery on fresh seed blocks if effects remain unresolved.

Classify each axis as:

- **Important in tested contexts:** a validated effect exceeds the practical margin, or a reproducible interaction does.
- **Negligible within the tested levels/budget:** equivalence bounds lie inside the practical margins for the stated endpoints, with failures accounted for.
- **Conditional:** direction/size depends on another tested factor or optimizer.
- **Unresolved:** insufficient precision, incomplete block, invalid inference, or inconsistent evidence.
- **Not evaluated:** deferred, absent, or unqualified code. Never translate this into “unimportant.”

Primary comparisons use equal updates and the same sampler workload. Report parameter counts, derivative cost, cap activity, and node-hours as part of each package's effect. This alone does not prove an equal-cost advantage. A subsequent matched-cost contrast must receive fixed, independently calibrated update budgets and new seed blocks; do not cherry-pick saved checkpoints to approximate equal cost.

**INFERRED F contract:** allocate global error 0.05 as 0.025 to simultaneous model-energy intervals for the 96 winner/control models and 0.025 to the winner's seed-success statement. Using a conservative Bonferroni allocation, each model interval receives error at most `0.025/96`. The same intervals support both nested accuracy bands. No conventional interval is admissible merely because this error allocation was written down.

Target at least 45/48 certified passes for the winning procedure. Under independent seed draws, `Pr[Binomial(48,.8)>=45] = .00788889 < .025`; together with valid simultaneous energy certificates, this supports success probability greater than .8 at the stated combined error level. Inference-unresolved seeds are not certified passes; distinguish that operational failure from proof of an inaccurate true energy. The control is descriptive unless a separate winner-versus-control inferential contrast was preregistered. Report all 48 outcomes, not only successes.

## 3. Code-development surface

### 3.1 What was actually inspected

**MEASURED, read-only local refs on 2026-09-04:** worktree HEAD `f3aa10054d3fb64d28bdfe02694f872bfddd3389`; locally available `origin/dev` `7ee49320443a5ddd8315587e706081805186d1e0`. No fetch, test, or runtime evaluation. These refs differ: a file's absence in the worktree is not evidence of absence from `dev`.

| Area | Source-inspection status | Remaining work |
|---|---|---|
| Tensor/linear/hybrid | Linear and composite producers exist on inspected `origin/dev`, including two unary policies and shared path layout | Qualify exact HI combinations; no duplicate implementation |
| Activations/width | Smooth activations, reusable MLP, per-order channel maps exist | Channel-axis-safe A6 wiring, explicit widths and seed initialization |
| A8 | Residual and Gaussian coordinate envelope exist; replacement is experimental/unexported in inspected tree | Promote/qualify replacement and verify gate placement, derivatives, metadata and restore |
| B inputs | Typed basis has a reserved pair field; embedding does not consume it | Owned pair ingress plus bounded distance package |
| B log-Jastrow | Accepted exact two-term formula is new study work | Owned factor, two trainable scalars, analytic/AD derivative and symmetry checks |
| e–n safety | Curved trainable law exists; unconstrained `c` can violate tail bound | Tail-safe coordinates and evaluator-compatible derivatives/state migration |
| VMC methods | Existing autograd/optimizer adapter and typed score seams; no qualified specialized method established by this inspection | Finish existing SR lane; KFAC/SPRING/LM with complete coverage and state |
| Checkpoints | Arbitrary cadence, atomic full model/optimizer/trainer/sampler/RNG saves and byte/write receipts exist; inspected `keep_last` is ignored | Persist new method state; exact resume; storage policy qualification |
| Local energy | Naive and analytic e–n cusp evaluators and slow oracle exist | Paired checks on all new factors; active-floor/finite-row audit |
| Planner | Typed operator planner/registry exists, not live evaluation integration | Do not make dormant lowering machinery a prerequisite if direct qualified evaluator suffices |
| Distribution | Existing status receipts do not establish qualified end-to-end DDP for this scan | Pack independent rows first; admit DDP only after exact method/reduction/resume checks |

**RELAYED workflow.** SR already has an owning Task Orchestrator lane; reuse it. Existing linear/hybrid and activation work must be reconciled by exact SHA/receipt. Runtime readiness cannot be inferred from source presence, a merged primitive, or the historical v0.4.0 test count.

### 3.2 Cheapest falsifiers and what disk already answers

All proposed probes below are future, authorized-cluster qualification work, not actions performed for this report.

| Option | Cheapest falsifier of its prerequisite/mechanism | What disk answers now | Failure action / expensive question left |
|---|---|---|---|
| Producer family | Tiny exact unary/bilinear contraction, path layout and permutation oracle; hybrid zero-family reduction | APIs, paths, parameter shapes, activation placement | Block wrong wiring. Fresh replicated training answers usefulness, not the algebra check. |
| A4/A5/A7 capacity | Static tensor/parameter manifest plus tiny reachability check | Which network owns each width; declared parameter count | Reject malformed geometry. Equal-budget training tests learning benefit. |
| A6 activation | Small value/gradient/Laplacian fixtures, including zero/unwritten tuples | Smoothness, channel-axis contract, duplicate activation risks | Block derivative/axis failure. Zero-preserving versus nonzero-at-zero behavior must stay explicit. |
| A8 | Set `u=0`: residual/gated return `x`, replacement returns zero; then test finite nonzero update and coordinate derivatives | Existing residual/gate semantics | Block incorrect replacement/gate, not declare the idea unimportant. Three-way fresh-seed contrast tests benefit. |
| A9–A12 initialization | Materialized state and parameter-reachability audit; exact scale-gauge comparison | Constant versus random meaning; inactive same-spin range; tail slopes | Reject malformed seed/parameter coordinates. Dynamics/conditioning need training. |
| B distance | Paired-start equality, nonconstant features, nonzero added-column gradient, ordered-pair exchange | Current missing ingress seam; feature formula | Block feature path if unused. Four-cell training tests easier learning. |
| B Jastrow | Zero-coefficient exact recovery, analytic derivatives, spherical-average cusp check, boundedness | Algebra and two-parameter cost | Block wrong cusp/sign/tail behavior. A zero learned coefficient in one seed does not establish irrelevance. |
| SR | Dense/sample-space/matrix-free same small score problem, direction and cap parity | Existing owned solver task and seams | Block incorrect solver; no extra science label for a numerical mismatch. |
| KFAC | Exactly factorable small affine oracle and exhaustive parameter-coverage report | Registered module inventory and expected factor sizes | Block uncovered parameters or incorrect damping; never silent fallback. |
| SPRING | First-step SR equality, explicit two-step history recurrence, exact restart | Published recurrence and current missing state | Block wrong recurrence/history. Full training asks whether history helps TPEN. |
| LM | Explicit small tangent-space Hamiltonian/overlap oracle and accepted-step reconstruction | Estimator/solver requirements and full-space memory scaling | Block wrong/partial target or infeasible state; preserve unavailable cells. |
| Evaluator | Paired frozen-coordinate value/gradient/Laplacian/term reconstruction | Existing analytic/naive agreement receipts are useful, not proof for new factors | Block evaluator disagreement; a lower reported energy is not a win. |
| Training firewall | Nested defaults/context/callback adversarial construction must reject a reference-bearing train packet before construction | Current historical config leak and runner construction order | Block all training release until absent by construction. |
| Resume/storage | Small uninterrupted versus checkpoint/resume identity plus representative serial/concurrent publication receipts | Current saved-state types and recorded byte receipts | Block continuation on missing state; adjust only the future measured storage policy. |

These cheap checks falsify implementation claims or necessary conditions. They do not replace the eight-seed scientific comparisons and cannot prove that a valid design is scientifically unimportant.

## 4. Execution surface

### 4.1 Facility boundary and facts still needed

**RELAYED.** The target dispatch system is the He-cutover Parsl backend. The user reports that the multi-node attach width restriction was fixed in `887988c`. Verify its inclusion in the selected release/backend receipt; do not assume the historical shape cost applies to this grid.

**MEASURED read-only documentation review.** Read `cluster-access`, TPEN root `cluster-access-read-first`, Clusters `00-orientation` and `facility-claims-have-a-shelf-life-2026-08-31`, ALCF/OLCF provider orientations and access/account notes, Polaris/Frontier read-first pointers and usage conventions. No host connection, queue query, storage probe, or allocation check was made. The skill's shelf-life rule prevents old queue/FairShare receipts from becoming current capacity assumptions.

**RELAYED facility documentation, checked 2026-09-04:** Polaris exposes four GPU devices per node; Frontier four MI250X packages/eight separately exposed GCD devices per node. Node-hours, packed occupancy, admissions and walltime all matter; a GCD-hour is not interchangeable with an A100-hour. Polaris documents routing production queues and a small capacity queue; Frontier documents an eligible-job limit and distinct batch/extended policies. These are not promises of this account's admissions. Consult [Polaris job policy](https://docs.alcf.anl.gov/polaris/running-jobs/) and [Frontier guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) and the current host notes before an actual resource decision.

**INFERRED primary execution approach:** fill nodes with independent single-device lineages where memory permits. Whole-node billing does not imply that each model should use DDP. Wider rows are admitted only after measured benefit/necessity and correct CPU/GPU binding; read the current topology notes before choosing binding. Frontier is a separate backend/environment qualification, not a drop-in NVIDIA overflow pool.

Before release obtain a clean, dated operational packet for: project balance/eligibility; production queues/partitions and walltime/eligible-job limits; backend release and supported row widths; device visibility and CPU affinity; Python 3.12+/library environment; method-specific memory; row runtime including derivative and checkpoint cost; target storage quota/inodes/write bandwidth; allocated concurrent nodes. Ask the operator for any fact available only inside a mixed baselines note. Do not open that note to extract it.

OLCF uses the exact existing operator-authenticated Paseo terminal, with no SSH/authentication recovery by this agent. ALCF uses its separate access route. Accounts, environments, filesystems and approval rules do not transfer between facilities. This document grants neither access nor submission authority.

### 4.2 Cost ledger, in the quantities that can bind

**MEASURED arithmetic of the INFERRED design:** the initial inventory is **8128 full scientific lineages × 50k = 406.4 million requested updates**, plus **480 × 2k = 0.96 million mechanics updates**. Thus **407.36 million requested updates**, excluding deterministic qualification, retries and evaluation. These are design counts, not measured runtime or an allocation request.

For execution class `j`, record lineage count `n_j`, device width `w_j`, measured elapsed hours per lineage `t_j` including its own overhead, devices/node `g`, and measured packing utilization `u_j`. A useful planning estimate is

\[
\mathrm{NH}_{train}\approx\sum_j n_j w_j t_j/(g u_j).
\]

The actual billed ledger is the sum over scheduler allocations of `allocated_nodes × charged_elapsed_hours`; use that for final cost. Add evaluation, qualification, partially filled waves, failures/retries and allocation startup/idle time explicitly. Estimate wall-clock makespan from measured queue delays and admitted concurrency, not node-hours alone. Record admissions and useful updates per admission when eligibility is binding.

With all R protocols extended from 50k to 200k, the added discovery work is **115.2 million updates**. F at 200k rather than 50k adds **14.4 million**. The full reserved upper ladder is therefore **536.96 million updates including Q**, before the additional costs above. A 100k or smaller admitted ladder reduces this deterministically. More saved checkpoints alone adds no optimizer updates.

At the initial full inventory, the provisional evaluation packet adds **1,372,585,984 retained local-energy records**, across **49,344 independent chain tasks** (48,192 discovery and 1,152 confirmation). This is not an ESS estimate. It excludes Q and burn-in; each added R horizon adds another 125,829,120 retained records if all 768 models are evaluated. Streaming/storage and proposal/model-evaluation cost must be charged separately from optimizer updates.

| Work class | Potential benefit | Binding cost to measure |
|---|---|---|
| Linear versus tensor/hybrid | Representation/optimization comparison; possible reduced mixer burden | Seconds/update and HBM per width, derivative overhead, packed rows/node |
| Width/MLP/update blocks | Capacity and conditioning, including interactions | Parameter/activation memory, Hessian/backward time, optimizer-state checkpoint bytes |
| Minimal B distance/J | Better learning route with small feature/two-scalar addition | Incremental derivative/feature cost; all four cells share qualification overhead |
| SR/SPRING | Empirical score geometry; projected history | Score materialization/streaming, sample-space solve, memory/bandwidth, history checkpoint bytes |
| KFAC | Approximate curvature without a full parameter-space metric | Tensor-product factor sizes, factor/inverse refresh time, coverage state memory |
| LM | Hamiltonian-aware tangent update | Directional Hamiltonian calculations and full-space solver memory/time; potentially admission-limiting |
| Longer horizon | Does a selected procedure need more optimization? | Same complete seed blocks at matched budgets; checkpoint growth and more admissions/resumes |
| Evaluation | Defensible energy/importance certificates | Independent retained chains, derivative time, raw-record storage, valid precision per node-hour |

No absolute node-hour figure is fabricated. The broad inventory is a proposed science budget; “ample compute” does not establish its exact affordability. If measured cost exceeds the authorized allocation, return a staged funding/admission decision. Do not shrink the science silently or claim an unrun factor negligible.

### 4.3 Checkpoint and storage policy

**RELAYED target:** every 1k completed updates. **INFERRED default:** full train-resume state, plus the final step if off-cadence. At 50k this is 50 distinct directories per scientific lineage; Q has two. Initial inventory: **407,360 checkpoint directories**, before initialization artifacts, retries and evaluation output. The full 200k R/F reserve raises this to **536,960**. A cadence/final write at the same step is one directory, not two.

Probe the actual largest state in each class: control/wide tensor Adam; hybrid channel-MLP/B11; SR; KFAC; SPRING; LM. Reuse qualified Q runlets for 1k/2k publications where possible. First one writer, then intended concurrency. Include model, method/optimizer, trainer, sampler, RNG, manifest, hash and publication receipts. With two or three writes, use maximum observed bytes/duration, not an invented p95.

Reserve `1.25 × (sum of expected checkpoint bytes + retained evaluation records + logs/receipts)` and a separately measured inode budget; 25% is a proposed margin, not storage permission. Measure walltime cost of concurrent writes. For illustration only, 100 MB per checkpoint would make the initial checkpoint set about 40.7 TB before headroom; 1 GB would make it about 407 TB. KFAC/LM state makes an Adam-only extrapolation unsafe.

If full state every 1k is too expensive, the declared fallback to qualify is model-only snapshots every 1k plus full resume every 5k and at every extension/terminal boundary. This requires explicit dual-payload support and revised byte/recovery accounting; it is not automatic. If still unaffordable, return to the operator with measured options. Do not delete old checkpoints or rely on the ignored `keep_last` setting. Scientific outputs and failure evidence remain protected until a specific lifecycle disposition permits cleanup.

Keep observation, evaluation and recovery cadence distinct: frequent saving permits later questions, but it does not create independent seeds or authorize checkpoint-shopping for accuracy.

## 5. Release gates and failure actions

**INFERRED.** These gates are required work, not assertions that they passed:

| Gate | Evidence needed | Failure action |
|---|---|---|
| Design/manifest | One coordinate authority; full literal configs; correct union/counts; accepted invariants; explicit unavailable cells | Stop materialization; do not revive the old A–L grid or fill defaults by guesswork |
| Reference isolation | Pre-construction train rejection; closed schemas/import/callback tests; clean training payloads | Stop all training dispatch |
| Physics/derivatives | Nonzero wavefunction checks, cusp/tail constraints, exchange/sector contract, no active-floor mismatch, independent evaluator agreement | Invalid arm; repair/requalify, no energy ranking |
| Update methods | Oracle direction/solve checks, exhaustive coverage, finite raw rows, telemetry and reproducible state | Method unavailable until repaired; no silent optimizer fallback |
| Seeds/resume | Disjoint phase streams; valid paired starts; exact next-state replay on qualified execution topology | Stop cross-seed or continuation inference; preserve attempts |
| Scientific inference | Qualified untruncated-energy intervals with dependence/tail assumptions; frozen margins/multiplicity | Report unresolved; no accuracy/equivalence claim |
| Stable code | Exact independently verified release promoted by the operator to stable `main`; all dependent receipts match SHA | No durable science on an unqualified `dev` snapshot |
| Facility | Clean current operational packet, canaries, binding/packing/memory/runtime/storage receipts and approved budget | No launch; resolve through operator/owning lane |
| Confirmation | Immutable finalist/control/horizon/48-seed packet and untouched inference plan | Remain in discovery; no relabelled confirmation |

Deferred does not mean rejected science: atomic radial bases, independent angular features, path subsets, low-rank/higher interaction parameterizations, backflow, general correlated factors, explicit triple-coalescence logarithms, hard symmetry redesign, learning-rate schedules, parameter groups, robust objectives, alternative sampler families, EMA/SWA and theorem-grade integration stay in the future register. In this revision, global clipping/transactional protection contrasts (old C4), walker/move packages (C13), initial spread (C14), and proposal-scale scanning (C18) are also held at control rather than secretly dropped as unimportant; they remain specific successor candidates if optimizer/sampler diagnostics warrant them. Reopen one when a fresh, specific failure or importance result justifies its implementation cost.

## 6. Delivery and provenance

**MEASURED.** This turn used local source/document inspection, clean scoped Task Orchestrator records, published literature/facility pages, and arithmetic only. No cluster connection, storage probe, scheduler operation, test, training, evaluation, source/config edit, branch operation, or new agent dispatch was performed. Existing unrelated `AGENTS.md`/`CLAUDE.md` changes were preserved.

**RELAYED decision receipts:** Task Orchestrator design item `bfb2247e-e750-49ad-8ed3-96a69a6b7951`, notes `operator-a8-b-grid-decisions-2026-09-04` and `operator-recommendations-accepted-and-horizon-2026-09-04`. Workflow topology: HI programme `67987933-9eb3-4d4c-b90c-7dc232d83ffc`, `topology-decision-5-lanes-2026-09-04`. Code-mechanics audit: design item's `revision-code-semantics-2026-09-04`, supplemented by direct inspected refs above.

No baselines-lane scientific output or mixed baselines note was opened or used for this revision. Operational facts needed from that lane must arrive as a clean operator relay.
