# TPEN H2-v1

This development study is the second all-electron TPEN surface: the
fixed-nuclei (Born-Oppenheimer), nonrelativistic hydrogen-molecule singlet
ground state at R = 1.4 bohr. It is a smoke/contract study, not a production
comparison, and reuses the same generic atom API as `experiments/atomistic/he-v1`
-- there is no molecule-specific wavefunction, Hamiltonian, or envelope class.
The two studies differ only in `system.nuclei`/`system.spin` data: two
unit-charge nuclei here instead of He's one Z=2 nucleus.

The model composes two generic post-readout `LogAmplitudeFactor`s directly:
`ElectronElectronCusp` and `ElectronNucleusCusp`. `ElectronNucleusCusp` is
constructed from the same declarative `atoms:` `AtomicConfiguration` used by
the Hamiltonian's `ElectronNucleusPotential`/`NucleusNucleusPotential` terms,
and its default linear cusp law sums independently over both H nuclei. It
intentionally has no Gaussian confinement.

The reference is -1.1744757142204434 Ha at R = 1.4 bohr, mirrored from the
canonical `h2_molecule` record in `experiments/baselines/systems.yaml`. It is
the total Born-Oppenheimer energy and includes the +1/R nuclear-repulsion
term; the primary citation is Pachucki, arXiv:1007.0322, Table IV, row 1.40.
Train and eval must agree exactly on model wiring; eval restores with `strict: true`.

Evaluation records the variational MCMC energy, finite-sample standard error,
and scalar reference comparison, together with sampled fermionic
antisymmetry/equivariance contracts (`full_model_antisymmetry`,
`spatial_exchange_symmetry`, `trace_equivariance`) -- the same four
system-agnostic tasks He also wires. This study does not wire He's
`he_radial_profiles` task: `HeliumRadialGridGenerator` requires exactly one
Z=2 nucleus and has no generic multi-center equivalent, so no substitute
radial diagnostic is invented here.

Correlation-aware external tau, ESS, and MCSE statistics remain explicitly
unavailable/absent, as in he-v1: no fixed-model trajectory JSONL producer
exists yet, so this study does not estimate them or wire a callback that
could imply they do.
