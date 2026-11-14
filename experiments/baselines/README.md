# SOTA NN-QMC baselines — survey, claims registry, and comparison plan

> Tracked research document for the *SOTA NN-QMC baseline comparison* program.
> Task Orchestrator root: `19fcf084`, under project `58348558` (TPEN).
> Survey performed 2026-08-12 (`America/New_York`); this file is the tracked
> form of that survey and is updated in place as claims are verified.

## 0. Evidence discipline

Every number below is tagged:

- **[V]** verified this session from the cited source page.
- **[T]** *to transcribe* — the claim exists in a table in the cited paper, but
  the digits have not been read from the source yet. **No [T] value may be
  copied from model memory into the registry, a plot, or a paper.** Item
  `725453a8` (system + reference-energy registry) exists specifically to close
  these by transcription with page/table provenance.

## 1. Why this is harder than "run the baselines"

TPEN's only working physical system today is the **2-electron Hooke pair**
(3D harmonic trap, `omega = 0.5`, singlet), per
`experiments/hooke/tpen-pair-v1/configs/train.yaml`. Every SOTA NN-QMC result
is on **molecular Coulomb** systems. There is currently *no system on which
both TPEN and a SOTA baseline can run*, so "compare to SOTA" is not yet a
runnable operation — it is a bridging problem.

Two bridges, and the plan builds both because each answers a different question:

| Bridge | What it does | Cost | Answers |
|---|---|---|---|
| **Down** — port baselines to the harmonic trap | FermiNet/LapNet get a custom local energy (kinetic + trap + e-e) | small | *Is TPEN's ansatz competitive where TPEN already runs?* |
| **Up** — grow TPEN to atoms | nucleus-anchored envelope + e-n cusp; He first | small-to-medium | *Does TPEN transfer to the systems the field actually benchmarks?* |

The **down** bridge is cheap and confirmed feasible: `ferminet/base_config.py`
exposes `cfg.system.make_local_energy_fn` ("String set to
`module.make_local_energy`, ... a callable which creates a function which
evaluates the local energy") plus `make_local_energy_kwargs` **[V]**. No fork
needed — supply a module with a harmonic trap term. LapNet is a FermiNet
derivative and is expected to carry the same hook (verify, do not assume).

The **up** bridge is smaller than it looks. `tpen/nn/embedding.py` consumes raw
per-particle coordinates (plus spin/aux), *not* trap-relative or
distance-derived features, so an atom placed at the origin needs no new
embedding features. `tpen/physics/potential.py` already implements
`ElectronNucleusInteraction`. What is missing is only on the ansatz side:

1. a nucleus-anchored **exponential** envelope (`log|psi| -= zeta * sum_i |r_i - R_A|`)
   to replace `GaussianConfinement`, whose Gaussian tail is wrong for Coulomb;
2. an **electron-nucleus cusp** envelope (`ElectronElectronCusp` has the e-e half);
3. config wiring + a reference-energy regression test.

He (2 electrons, singlet) then runs on today's `n=2` machinery. That is the
single highest-leverage slice in this program — item `ba761202`.

## 2. The landscape

Grouped by what each contributes, because "SOTA" is not one axis. A comparison
that only reports energies will silently compare an ansatz against an
optimizer against a Laplacian implementation.

### 2.1 Ansatz architecture

| Method | Ref | Core idea | Headline claim | Code |
|---|---|---|---|---|
| **FermiNet** | Pfau et al. 2020, arXiv:1909.02487 | Permutation-equivariant streams + generalized Slater determinants; no fixed basis set | 99.8% of correlation energy on boron; >99% vs CCSD(T)/CBS up to ethene (16 e-) **[V, blog-level; per-system table T]** | `google-deepmind/ferminet` (JAX) |
| **PauliNet** | Hermann et al. 2020, Nat. Chem. | HF/CASSCF baseline + deep Jastrow/backflow correction | physically-motivated baseline, fewer parameters | `deepqmc/deepqmc` |
| **DeepErwin** | Scherbela et al. | weight sharing across geometries | transfer across PES points | `deepqmc` config |
| **Psiformer** | von Glehn, Spencer, Pfau 2023, arXiv:2211.13672 | self-attention replaces FermiNet's streams; drop-in | ground-state energy improved "by dozens of kcal/mol" on larger molecules **[V]**; per-system table **[T]** | in `ferminet` |
| **Neural Pfaffian** | Gao & Günnemann, NeurIPS 2024 (oral), arXiv:2405.14762 | **Pfaffian** instead of Slater det; non-square skew matrix; no hand-crafted orbital selection; generalizes across molecules | chemical accuracy across systems from one model; beats CCSD(T)/CBS on TinyMol by **1.9 mHa**; up to 1 order of magnitude lower error than prior generalized wave functions **[V]** | `n-gao/neural-pfaffian` |
| **Excited Pfaffians** | arXiv:2603.14515 (2026) | Neural Pfaffian extended to excited states | generalization across structure *and* state **[T]** | — |
| **Orbformer** | arXiv:2506.19960 (2025) | wavefunction "foundation model" | accurate bond breaking from a pretrained model **[T]** | — |

**Directly relevant to TPEN's design choices:** Neural Pfaffian is the closest
architectural relative — it also antisymmetrizes with a Pfaffian rather than a
determinant. TPEN's `PfaffianReadout` should be compared against it explicitly,
including the non-square/overparametrized construction.

### 2.2 Cost / differentiation

| Method | Ref | Claim |
|---|---|---|
| **Forward Laplacian / LapNet** | Li et al., Nat. Mach. Intell. 2024, arXiv:2307.08214 | Laplacian is the NN-VMC bottleneck; forward-propagation Laplacian gives **">1 order of magnitude" speedup** and roughly halves total cost vs standard methods **[V]**; per-system timings **[T]** | `bytedance/LapNet`, `lapjax` |
| **Pseudopotentials** | arXiv:2505.19909 (2025) | local pseudopotentials remove core electrons from the NN-VMC cost **[T]** |

The forward-Laplacian claim is *the* cheapest high-value reproduction (fixed
ansatz, fixed system, one flag, one GPU, ~100 iterations) and it directly
calibrates TPEN's own Laplacian cost — item `2dcab64b`.

### 2.3 Optimization

| Method | Claim |
|---|---|
| KFAC (FermiNet/LapNet default) | the standard; all headline energies use it, not Adam |
| SPRING / MinSR | cheaper stochastic-reconfiguration-like updates **[T]** |
| **LAVA + neural scaling laws** | arXiv:2508.02570 (2025): absolute energy error shows **power-law decay in model capacity and compute**, reaching **1 kJ/mol sub-chemical accuracy** on benzene, N2 dissociation, cyclobutadiene automerization, cyclic ozone **[V]** |

**TPEN currently trains with Adam.** Any TPEN-vs-SOTA energy gap is confounded
by optimizer until item `3428933a` separates them. This is the single largest
methodological risk to the whole comparison.

### 2.4 Beyond variational

| Method | Claim |
|---|---|
| **FermiNet + DMC** | Ren, Fu, Chen, Nat. Commun. 2023: projecting a neural ansatz with DMC gives "substantial improvement in both accuracy and efficiency"; tested on atoms, N2, cyclobutadiene, water dimer, benzene, benzene dimer **[V]**; numbers **[T]** |

Relevant because a variational-only comparison understates every ansatz by a
different amount. Report VMC-vs-VMC; note DMC as a separate tier.

### 2.5 Non-molecular / trapped systems — the natural meeting point

| Work | Why it matters here |
|---|---|
| **Kim, Pescia, Fore, Nys, Carleo, Gandolfi, Hjorth-Jensen, Lovato**, *Neural-network quantum states for ultra-cold Fermi gases*, Commun. Phys. 2024 (arXiv:2305.08831) | **Pfaffian-Jastrow message-passing NQS** for **harmonically trapped** fermions. Same antisymmetrization family as TPEN, same confinement family as Hooke. Closest published comparison surface to TPEN as it exists today. **[V on scope; energies T]** |
| Pescia/Carleo et al., NQS for periodic systems in continuous space; message-passing NQS for the homogeneous electron gas | continuous-space non-molecular NQS |
| Keeble et al., PRA 108, 063320 (2023) | 1D spinless trapped fermions with NQS |
| Kolmogorov-Arnold wavefunctions for trapped fermions, arXiv:2512.07800 | recent trapped-fermion ansatz |
| Quantum-dot DMC literature (Pederiva, Umrigar, et al.) | decades of parabolic-dot benchmark energies, N = 2..20 **[T]** |

### 2.6 Reference implementation

**DeepQMC** (arXiv:2307.14123, JCP 2023) ships configs reproducing *Psiformer,
PauliNet, FermiNet, DeepErwin, LapNet and TransPsiformer* in one codebase, and
reports that its re-implementations match the reference energies, with residual
discrepancies attributed to setup differences (optimization steps, batch size,
older TF implementation) **[V]**. This makes DeepQMC the cheapest single
dependency for a multi-ansatz sweep — and its own "we match the references"
statement is itself a reproducible claim.

## 3. Points of comparison (the scorecard)

Per **(system, code, ansatz)** row. Item `0b0a456f` turns this into the JSONL
record schema every adapter writes.

**Accuracy**
1. `E_VMC` with MC standard error.
2. `dE = E_VMC - E_ref` in mHa, and as % of correlation energy where an HF
   reference exists. For Hooke at Taut-solvable `omega`, `E_ref` is exact.
3. Chemical accuracy (1.6 mHa) and the newer 1 kJ/mol (0.38 mHa) thresholds as
   pass/fail flags.

**Statistical quality**
4. Local-energy variance per electron `sigma^2/N` — the ansatz-quality signal
   that is nearly free and rarely reported.
5. Variance-extrapolated energy (`E` vs `sigma^2` -> 0).
6. **Seed spread**: same config, >=3 seeds, report mean and range. Most papers
   report one run. TPEN's existing run-artifact discipline makes this cheap and
   it is a genuine contribution axis.

**Efficiency — three denominators, all reported**
7. `E` vs optimization step (optimizer quality).
8. `E` vs cumulative local-energy evaluations (sample efficiency; hardware-free).
9. `E` vs GPU-seconds on identical hardware (what a user pays).

Reporting only (9) hides ansatz quality behind kernel engineering; reporting
only (7) hides that FermiNet's step is a KFAC step. Report all three.

**Cost**
10. ms/step at fixed batch size and fixed MCMC schedule; peak GPU memory.
11. Measured cost scaling with `N` (fit exponent) — not the claimed one.
12. Parameter count; determinant/Pfaffian count; layer/width settings.

**Structural (no run needed — table, not measurement)**
13. Antisymmetry mechanism: generalized Slater (FermiNet/Psiformer) vs Pfaffian
    (TPEN, Neural Pfaffian, Kim et al.) vs explicit antisymmetrization.
14. Cusp handling: analytic envelope (TPEN) vs learned vs baseline-orbital.
15. Laplacian method: naive vs forward Laplacian.
16. Equivariance guarantee and **runtime certification** — TPEN checks
    permutation equivariance during training (`RuntimeEquivariance`); baselines
    have equivariance by construction but ship no runtime check. State this as
    a verification-methodology difference, not an accuracy claim.
17. Transferability: single-system vs amortized/generalized across geometries.

## 4. Confounder protocol (item `c15c84c6`)

Must be matched or explicitly reported as a delta:

| Knob | TPEN today | Typical baseline | Action |
|---|---|---|---|
| dtype | **float64** | FermiNet **float32** | match float32 both ways for cost rows; report float64 accuracy rows separately |
| optimizer | Adam | KFAC / SPRING / LAVA | separate ablation (`3428933a`); never compare Adam-TPEN to KFAC-FermiNet as an ansatz result |
| energy clipping | check current behavior | median-based clipping standard in FermiNet | match or report |
| MCMC | 10 steps/update, `proposal_scale 0.5`, 128 walkers | typically larger batch, tuned acceptance | match walker count and steps/update; target the same acceptance |
| pretraining | none | FermiNet pretrains to HF | *large* effect on iterations-to-accuracy; must be reported |
| iteration budget | 25 (smoke) | 1e5-2e5 | fix a common budget per tier |
| hardware | A100 / H200 on Cannon | varies per paper | always report GPU model; never compare raw wall-clock across papers |

## 5. Reproduction tiers on Cannon

Cannon scouting (2026-08-12, read-only, login `holylogin08`) **[V]**:

- `gpu_test`: 12 nodes, `nvidia_a100_3g.20gb` MIG x8, 12 h, max 2 concurrent.
- `kozinsky_gpu`: 2 nodes, `nvidia_a100-sxm4-80gb` x4, 7 d.
- `seas_gpu`: A100-80GB x4 nodes **plus 30 nodes of `nvidia_h200` x4**, 2 d.
- Modules: `cuda/12.9.1-fasrc01`, `cudnn/9.10.2.21_cuda12-fasrc01`; `uv 0.11.16`
  on PATH; system `python3` is 3.6.8 (unusable — use uv-managed).
- Storage contract: checkout, venvs, `UV_CACHE_DIR`, logs go on Netscratch, never
  `$HOME` (`$HOME` 95 GiB, currently 61%). Facility absolute paths are never
  written into this repository; resolve them from the environment at run time.

JAX `cuda12` pip wheels vendor their own CUDA/cuDNN, so a baseline venv likely
needs no module load at all — but the module path exists if wheel-vendored CUDA
misbehaves.

| Tier | Content | Est. cost | Value |
|---|---|---|---|
| **R0** | env stand-up + 1 published smoke config per code | ~1 GPU-h | proves the environment |
| **R1** | Forward-Laplacian speedup, flag on/off, same ansatz/system/GPU | ~2 GPU-h | tests a headline claim cheaply; calibrates TPEN |
| **R2** | FermiNet He/Li/Be at published settings vs transcribed table | ~1-3 GPU-days per atom | tests the accuracy claim at a scale we can afford |
| **R3** | Baselines ported to Hooke pair + a quantum dot | ~10 GPU-h | **creates the first shared TPEN/SOTA surface** |
| **R4** | Psiformer vs FermiNet on one mid-size molecule | 100s of GPU-h | expensive; only if the earlier tiers justify it |
| **R5** | Benzene / bicyclobutane / scaling-law-scale claims | not affordable here | cite, do not reproduce |

R2's cost follows from FermiNet's own published protocol (below): 2e5 updates at
batch 4096. Cutting iterations is the obvious lever, but iteration count is then
a documented delta, not a reproduction.

### 5.1 Transcribed FermiNet protocol and reference table

Source: arXiv:1909.02487 via ar5iv HTML, read 2026-08-12. **Digits still need a
spot-check against the published PDF before any of these enter a plot or paper**
— they were extracted by an HTML-reading pass, not by hand off the page.

Protocol: 4 hidden layers, 256 units (one-electron stream) / 32 units
(two-electron stream), **16 determinants**, **2e5 parameter updates**, **batch
4096**, **KFAC**, 10 MCMC steps between updates; 8x V100 for systems under 20
electrons, 16 GPUs above. Ethene ~2 days on 8 GPUs; bicyclobutane (30 e-)
~1 month on 16 GPUs.

**Corrected 2026-08-13.** The first extraction of this table was wrong in a way
worth recording, because the failure mode generalizes. FermiNet's tables are
*wide multi-method* tables, not `method | reference` pairs:

```text
Table 1 (atoms):     Atom | FermiNet | VMC | DMC | CCSD(T)/CBS | HF/CBS | Exact | %corr
Table 2 (molecules): System | R | FermiNet | VMC | DMC | CCSD(T)/CBS | HF/CBS | Exact | %corr
```

The HTML-reading pass collapsed them and, for the molecules, read the
**CCSD(T)/CBS** column and stored it as the exact reference. An independent
re-read confirmed the column ordering two ways that do not depend on parsing:
the HF/CBS column holds the known Hartree-Fock limits (LiH `-7.98737`, Li2
`-14.87155`, N2 `-108.9940`), and the VMC/DMC columns reproduce
[arXiv:1012.0709](https://arxiv.org/abs/1012.0709) character-for-character.

| System | FermiNet (Ha) | Exact (Ha) | % corr. |
|---|---|---|---|
| Li | -7.47798(1) | -7.47806032 | 99.82(3) |
| Be | -14.66733(3) | -14.66736 | 99.97(3) |
| B | -24.65370(3) | -24.65391 | 99.83(3) |
| C | -37.84471(5) | -37.8450 | 99.81(3) |
| N | -54.58882(6) | -54.5892 | 99.80(3) |
| O | -75.06655(7) | -75.0673 | 99.70(3) |
| F | -99.7329(1) | -99.7339 | 99.69(3) |
| Ne | -128.9366(1) | -128.9376 | 99.74(3) |
| LiH | -8.07050(1) | **-8.070548** | 99.94(1) |
| Li2 | -14.99475(1) | **-14.9954** | 99.47(1) |
| N2 | -109.5388(1) | **-109.5423** | 99.36(2) |
| Ethene | -78.5844(1) | -78.5888 **(CCSD(T)/CBS)** | 99.16(2) |
| Bicyclobutane | -155.9263(6) | -155.9575 **(CCSD(T)/CBS)** | 96.94(5) |

All thirteen FermiNet energies and all eight atomic reference values survived
the re-read digit-for-digit. Four things changed:

1. LiH, Li2 and N2 now carry the exact column, not CCSD(T)/CBS. Sources are
   Cencek & Rychlewski (2000) for LiH and Filippi & Umrigar (1996) for Li2/N2,
   at R = 3.015, 5.051 and 2.068 bohr respectively.
2. **Ethene and bicyclobutane have no exact value in the source at all.** Their
   entries are CCSD(T)/CBS, so the 99.16% and 96.94% figures are correlation
   energy relative to a *computed* reference, not an exact one.
3. Li's reference is right but misattributed in FermiNet's own footnote, which
   cites Chakravorty et al. for the whole column. Li is Puchalski & Pachucki,
   *Phys. Rev. A* **73**, 022503 (2006); Be through Ne genuinely are Chakravorty.
4. FermiNet **never states its dtype** (only "TensorFlow 1 built with CUDA 9"),
   and these table values are a *post-training evaluation* — O(1e5) MCMC steps
   sampling the mean local energy every 10 steps, with Flyvbjerg-Petersen
   blocking error bars — not a training-tail average. Anything this program
   compares against them must match that estimator or declare the difference.

He and H2 are not in this table — the atom sequence starts at Li, and H2 is not
among the molecules. Their references come from the atomic and molecular
literature instead and are recorded in `systems.yaml`.

Two consequences for planning. First, the % correlation figures are *not*
uniformly 99.8%: bicyclobutane is 96.9%, so "SOTA" degrades with system size and
TPEN should be compared at matched size. Second, the Li row shows FermiNet
~0.08 mHa above the reference while Be is within 0.03 mHa — the error is not
monotone in electron count, so a single-system comparison proves little.

**Definition used for "reproduce"**: reduced-scale reproduction with *every*
delta from the published setting documented (iterations, batch, dtype,
hardware, pretraining). Full-scale reproduction of R4/R5 is out of budget and
saying so is part of the result.

## 6. Harness (item `aaefaa5a`)

Target layout. Only the parts marked *landed* exist today; the rest is the
plan, and nothing here should be read as an existing interface.

```
experiments/baselines/
  README.md                 # this document (landed)
  systems.yaml              # system id -> Hamiltonian spec, spin sector, E_ref + citation (landed)
  test_systems.py           # registry schema + evidence-discipline test (landed)
  adapters/ferminet.py      # + lapnet.py, deepqmc.py: run, then emit the common record
  adapters/tpen.py          # wraps tpen.run.run_from_config (the one sanctioned import)
  records.py                # the common (system, code, run) results record (landed)
  collect.py                # scan run roots -> results.jsonl (landed)
  spread.py                 # across-seed mean, range, sample std and SEM (landed)
  test_records.py           # record schema + collector tests (landed)
  compare.py                # per-system table + E-vs-cost plots
  local_energy/harmonic.py  # make_local_energy for FermiNet-family trap runs
```

Collection contract: **the emitter owns code-specific knowledge, not the
collector.** Any run — TPEN, FermiNet, LapNet — drops one
`baseline_record.json` in its run directory; `collect.py` walks a run root,
validates each file against `records.BaselineRecord`, and writes
`results.jsonl`. That is why there is no adapter framework or plugin registry
here. Unknown quantities stay `null`; a malformed record is reported with its
path and makes the run exit non-zero rather than shrinking the output file.

```
uv run python -m pytest experiments/baselines
uv run python -m experiments.baselines.collect --run-root <dir> --output results.jsonl
python -m experiments.baselines.spread --results results.jsonl \
  --output seed-spread.json --markdown-output seed-spread.md
```

### Across-seed spread

`spread.py` groups records by `(system_id, code, ansatz, estimator)`. It counts
seedless records but excludes them from the statistics, and reports the mean,
minimum, maximum, range, sample standard deviation (`ddof=1`), and standard
error of the mean across distinct seeds. Fewer than three seeds are retained
with an `insufficient_seeds` flag: the program rule for a seed-spread claim is
at least three independent seeds. Duplicate seeds are rejected loudly rather
than averaged, and estimator values are never pooled.

The input may instead be a run root (`--run-root`); it uses the existing
collector validation and record reader. The JSON output is suitable for later
comparison tooling, while `--markdown-output` produces a short review table.

Use `python -m pytest` rather than bare `pytest` under `experiments/`: the test
modules import `experiments.<package>`, which needs the repository root on
`sys.path`.

Constraints already binding: `experiments/` must not import `tpen` except
`tpen.run.run_from_config`; experiment tests live under `experiments/`;
baseline venvs live on Netscratch and stay out of `uv.lock`; no facility
absolute path is ever written into the repo.

## 7. Open decisions (for the operator)

1. **Install approval** for third-party baselines (`ferminet`, `lapnet`+`lapjax`,
   `deepqmc`, possibly `neural-pfaffian`) in isolated Netscratch venvs.
   CLAUDE.md requires asking before adding packages. These do **not** touch
   `pyproject.toml`/`uv.lock`.
2. **Bridge priority** — down (baselines to the trap), up (TPEN to He), or both.
3. **Reproduction depth** — cheap-claims-only (R0-R1), accuracy tier (R2-R3), or
   push to R4.
4. **Tracking** — *resolved 2026-08-12*: `experiments/baselines/` is tracked, and
   lands as a linear stack of reviewable PRs based on `main`. Third-party
   baseline checkouts stay on the cluster and are never committed here.

## 8. TPEN vs Neural Pfaffian — how close are they really?

Answer: **same antisymmetrizer, different everything else.** Both are
"weighted sum of Pfaffians of learned skew matrices", and both deliberately
avoid a determinant. That is where the overlap stops.

Neural Pfaffian (arXiv:2405.14762, read via ar5iv 2026-08-12):

```
Psi = exp(J(r)) * sum_k c_k Pf( Phi_k(r) A_k Phi_k(r)^T )
```

TPEN (`tpen/nn/readout/pfaffian.py`):

```
Psi = sum_c w_c Pf( K_c ),   K_c = 0.5 * (x_c - x_c^T)
```

where `x_c` is the order-2 (pair) feature channel straight out of the
equivariant stack.

| Axis | Neural Pfaffian | TPEN | Same? |
|---|---|---|---|
| antisymmetrizer | Pfaffian | Pfaffian | **yes** |
| combination | `sum_k c_k Pf(...)`, not `Pf(sum)` | `sum_c w_c Pf(K_c)`, explicitly not `Pf` of the channel-mixed kernel | **yes**, same argument (Pf is degree-n/2 and nonlinear) |
| skew matrix origin | **factorized** `Phi A Phi^T`; `Phi` is `N_e x N_o` orbitals, `A` a learnable `N_o x N_o` skew matrix | **direct**: skew part of the pair feature, full `N x N`, no factorization | **no — the central difference** |
| overparametrization | `N_o >= max(N_up, N_dn)` orbitals; rectangular `Phi` removes discrete orbital selection | channel count `pair_channels` plays a loosely analogous role | partial |
| where orbitals live | a fixed number per **nucleus**, which is what makes transfer across molecules possible | no nuclei at all yet | no |
| backbone | Moon message-passing equivariant net | tuple/path equivariant stack with an explicit path axis, body order `m` | no — this is TPEN's actual novelty |
| spin | two orbital sets, order swapped for spin-down, plus a learnable `eta` keyed on `N_up - N_dn` | spin appended to the embedding input vector | no |
| Jastrow | explicit `exp(J)` | none; additive log-amplitude envelopes only (`GaussianConfinement`, `ElectronElectronCusp`) | no |
| envelope | per-nucleus exponentials linearly recombined with learnable weights | Gaussian trap + e-e cusp | no |
| optimizer / pretraining | SPRING + HF pretraining, ~2e5 VMC steps | Adam, no pretraining | no |
| goal | one model amortized across molecules | exact permutation equivariance at arbitrary body order | no |

Practical readings:

1. The **factorized vs direct** skew matrix is the interesting scientific
   question, not a detail. `Phi A Phi^T` buys per-nucleus orbitals (hence
   transferability and a memory-efficient envelope), at the cost of forcing the
   pair kernel through an orbital bottleneck. TPEN's direct pair kernel is the
   more general function class and needs no orbital-selection heuristic, but has
   no transfer story and no obvious molecular envelope. Item `e40b815e`.
2. Neural Pfaffian is the correct citation for "Pfaffians remove hand-crafted
   orbital selection". TPEN should not re-claim that as novel; TPEN's claim is
   the equivariant tuple/path construction feeding the Pfaffian.
3. Their reported "7x fewer parameters than Globe on N2" is a *parameter
   efficiency* claim — one of the non-energy axes worth mirroring.

### Blocker discovered while reading TPEN's readout

`_pfaffian_single` is a **recursive minor expansion**, and `pfaffian()` loops
over the batch in Python with `torch.stack`. Cost per sample is `(n-1)!!`:

| n | terms |
|---|---|
| 2 | 1 |
| 4 | 3 |
| 6 | 15 |
| 10 | 945 |
| 20 | 654,729,075 |

The docstring already flags it ("A production implementation should replace it
with a stable batched routine"). It is fine for the Hooke pair (`n=2`) and
survivable for Be (`n=4`), but it **blocks molecular Coulomb**, which is the
stated direction. Needs a batched skew factorization (Parlett-Reid or
Householder tridiagonalization, or signed `sqrt(det)` with tracked sign) with
signed-log output, validated against the recursive version at small `n`.
Item `8e1a56dd`. Also note `channel_weights` defaults to `trainable=False`.

## 9. DeepQMC on Polaris (item `9f762980`)

Only FermiNet and PsiFormer were installed on Polaris, so the four
DeepQMC-driven ansatzes — `default`/PauliNet, `deeperwin`, `lapnet`,
`transpsiformer` — could not use any of that allocation. This section records
the environment that changes it, and the two things about it that are easy to
get wrong.

### The environment is a NEW venv, not an addition to the FermiNet one

DeepQMC's `pyproject.toml` declares `jax<0.9.0`. The existing Polaris
environment `~/.venvs/ferminet-jax092-60c9fab3` carries jax 0.9.2 and is
therefore excluded by that constraint. It must not be modified or upgraded to
suit DeepQMC: it is the only working JAX-on-Polaris environment this program
has, it survived the 2026-08-19 facility CUDA 13 upgrade, and breaking it would
strand the FermiNet and PsiFormer work as well.

```text
interpreter  /soft/applications/conda/2025-09-25/mconda3/bin/python   (3.12.11)
venv         /home/rhu/.venvs/deepqmc-jax083-edf373e7
source       /home/rhu/src/deepqmc-edf373e7, detached at edf373e7
pins         experiments/baselines/polaris_deepqmc_requirements.txt
```

The recipe that works is an **absolute** facility interpreter, a venv on top of
it, and `jax[cuda12]` pip wheels that vendor their own CUDA runtime. That path
never consults the facility CUDA stack, which is exactly why it survived a
facility CUDA change. `module load conda` currently fails on missing
`gcc-native/14.2` and `cray-hdf5-parallel/1.14.3.5`, and is not needed.

Two build details are deliberate. The pins are **generated** by enumerating a
working reference environment, never hand-written, so a transcription slip
cannot enter a version. And DeepQMC is installed editable with `--no-deps`, so
the resolver gets no opportunity to move a pin underneath the install. Verify
the result by comparing the whole name-to-version map against the reference —
a clean install log is not the check.

Note that `pip freeze` is useless for that comparison: these venvs are built by
`uv` and contain no `pip`, so `pip freeze` returns **empty and exits 0**, which
reads exactly like "nothing is installed". Enumerate with
`importlib.metadata.distributions()` instead.

### The seed override fails silently, so it is checked in a pair

`task.seed=N` is what the whole seed-spread contribution rests on, and if it
does not take, every row runs the default seed, produces near-identical
energies, and looks like a legitimate result.

`check_polaris_deepqmc_env.py seed` reads the value back out of the run's own
`training/.hydra/config.yaml` — the file the run actually resolved — rather than
trusting the command line that requested it. One trap: `task:` is line 1 of that
file and `seed:` is around line 50, so `grep -A2 task: | grep seed` prints
nothing and reads as "no seed key". It is there.

The check must be run as a **pair**, and neither half is sufficient alone:

| probe | override | config must read |
|-------|----------|------------------|
| X     | `task.seed=7` | 7 |
| Y     | *(none)* | 0 |

The reason is that DeepQMC's default seed is 0, which is also the seed of the
Cannon comparator row. A run launched with `task.seed=0` whose override is
silently dropped still writes `seed: 0`, so that check passes identically
whether the mechanism works or is entirely broken. Probe X uses a non-default
value so it can only appear if the override took; probe Y establishes what the
file says without an override, so X cannot be explained away as some unrelated
default. Hydra writes the config at startup, before training, so both probes
cost seconds rather than a training row.

### Running the verification

`check_polaris_deepqmc_env.py` has two subcommands and no submission machinery of
its own, deliberately. Polaris submission belongs to the baselines submission
path, not to this module.

```bash
# inside a PBS allocation, after XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m experiments.baselines.check_polaris_deepqmc_env env \
    --expect-prefix /home/rhu/.venvs/deepqmc-jax083-edf373e7 --expect-jax 0.8.3

python -m experiments.baselines.check_polaris_deepqmc_env seed \
    --run-dir <run> --expect-seed 7
```

Two environment facts the caller owns, because this module cannot enforce them
from inside a job it did not launch. `XLA_PYTHON_CLIENT_PREALLOCATE=false` must
be set before any JAX import, or JAX takes ~75% of the card and every memory
reading is identical regardless of workload. And GPU visibility is the job's
responsibility: PBS does not constrain it on Polaris.

The verification runs for this port are recorded on Task Orchestrator item
`9f762980` with their PBS job ids and durable paths under
`/eagle/HetRxnEnergy/rhu/runs`; the job script that produced them was scaffolding
and is not carried here.

## Sources

- [FermiNet (Pfau et al. 2020), arXiv:1909.02487](https://arxiv.org/pdf/1909.02487) · [code](https://github.com/google-deepmind/ferminet) · [DeepMind blog](https://deepmind.google/blog/ferminet-quantum-physics-and-chemistry-from-first-principles/)
- [Psiformer, arXiv:2211.13672](https://arxiv.org/abs/2211.13672)
- [Forward Laplacian, arXiv:2307.08214](https://arxiv.org/abs/2307.08214) · [Nat. Mach. Intell.](https://www.nature.com/articles/s42256-024-00794-x) · [LapNet code](https://github.com/bytedance/LapNet)
- [Neural Pfaffians, arXiv:2405.14762](https://arxiv.org/abs/2405.14762) · [Excited Pfaffians, arXiv:2603.14515](https://arxiv.org/pdf/2603.14515)
- [DeepQMC, arXiv:2307.14123](https://arxiv.org/pdf/2307.14123) · [code](https://github.com/deepqmc/deepqmc) · [JCP](https://pubs.aip.org/aip/jcp/article/159/9/094108/2909731/DeepQMC-An-open-source-software-suite-for)
- [Neural scaling laws / LAVA, arXiv:2508.02570](https://arxiv.org/html/2508.02570v2)
- [Local pseudopotentials for NN-QMC, arXiv:2505.19909](https://arxiv.org/html/2505.19909)
- [DMC on neural networks, Nat. Commun. 2023](https://www.nature.com/articles/s41467-023-37609-3)
- [Positronic chemistry NN-VMC, Nat. Commun. 2024](https://www.nature.com/articles/s41467-024-49290-1)
- [NQS for ultra-cold Fermi gases, arXiv:2305.08831](https://arxiv.org/pdf/2305.08831) · [Commun. Phys.](https://www.nature.com/articles/s42005-024-01613-w)
- [Trapped fermions via Kolmogorov-Arnold wavefunctions, arXiv:2512.07800](https://arxiv.org/pdf/2512.07800)
- [Taut, Phys. Rev. A 48, 3561 (1993)](https://link.aps.org/doi/10.1103/PhysRevA.48.3561) — exact Hooke solutions at a denumerable set of omega
- [Orbformer, arXiv:2506.19960](https://arxiv.org/pdf/2506.19960)
