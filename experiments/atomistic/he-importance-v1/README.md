# He-importance v1: S1a factor dictionary

This directory is the S1a definition surface for the next helium importance
scan. [`factor_dictionary.yaml`](factor_dictionary.yaml) is the single,
versioned source of truth for the closed scientific vocabularies, Stage 0
optimizer candidates, core packages, breadth factors A–L, and bridge anchors
Q0–Q3. It is a dictionary, not a runnable helium composition or a production
planner.

`factor_dictionary.py` validates that file and resolves one complete factor
assignment plus an absolute learning rate and runtime training seed into a
single serializable Hydra configuration. It keeps `science_assignment_id`
separate from the normalized `structural_signature`, so intentionally distinct
scientific assignments are retained when their runnable structures collide.
Stage S2 owns row/attempt IDs, seed namespaces, manifests, absolute paths, and
content hashes.

## Closed choices

- Mixing: `tensor`, `k-GNN-like-linear`, `hybrid-additive`, mapped explicitly
  to the landed `EquivariantMixing`, `LinearEquivariantMixing`, and
  `CompositeMixing` producer slots.
- Activation: `pointwise` (SiLU), `MLP-mixing`, `MLP-aggregation`, and
  `MLP-both`. CPMLP slots are per-order, one hidden layer of width `2C`, with
  Tanh hidden activation and unchanged output channels.
- Optimizers and learning-rate candidates are exact in the YAML dictionary.
  Adam remains centred at `0.005`; non-Adam centre × multiplier resolution
  fails closed until a signed Stage 0 centre is supplied.

## Breadth amendment

Breadth columns A–H are independent signs, and I–L are derived in lexicographic
resolution-V order. Column C is intentionally not the parent-design body-order
factor: feature/body `max_order` is fixed at `2` in every cell, while C selects
only interaction/path `max_virtual_order` `1` or `2`. This departure is cited in
the dictionary as **user decision 2026-09-04 (Q1)**.

The I–L generator equations define membership in the 256-row design fraction;
they are not validity constraints on complete A–L assignments. The named
control is deliberately outside that fraction and resolves through the same
surface. Use `is_fraction_row` only when a caller explicitly needs to classify
design membership.

## Hooke choice surface and seeds

`build_shared_hooke_choice_surface` emits a self-contained Hydra fragment for
all three producer modes and both activation consumers. It parameterizes width,
body order, virtual order, path family, aggregation, initial weight, activation
placement, and initializer streams while leaving existing TPEN core modules
untouched. Mixing activations receive tensors with tuple axes starting at `3`
(`[B,C,P,N^m]`); aggregation activations receive tuple axes starting at `2`
(`[B,C,N^m]`).

When a host supplies one runtime training seed, the CPMLP mixing and aggregation
slots use that seed through distinct named streams containing layer and slot
identity. With no host seed, standalone Hooke opt-in defaults remain `701` and
`702` for those slots. A seed is an initialization input, not a second science
factor.

The control reference is deliberately symbolic:
`CONTROL_COORDINATES=PENDING-USER-DETAILED-REVIEW`. S1a does not freeze control
coordinates. No reference energy, reference error, chemical-accuracy decision,
clip probe, or production design generator belongs here. Parameter count is an
endpoint; core package effects are not parameter-count-matched causal effects.
