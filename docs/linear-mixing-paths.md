# Linear Equivariant Mixing Paths

## Status

This document is the mathematical and typed architecture contract for the
linear path family that composes with TPEN tensor-product paths. It does not
claim that the implementation already exists.

## Static model-construction invariant

The path layout and every trainable tensor are static parts of the model.
They are constructed eagerly during `model.__init__`, before optimizer
construction and before the first forward pass.

In particular:

- No path is discovered from a runtime feature tensor.
- No parameter is created lazily during `forward`.
- Runtime particle count may change an orbit's cardinality or make an orbit
  empty, but it never adds or removes paths.
- An empty orbit still has a registered path and weight. Its reduction must
  have explicitly defined zero-safe behavior.
- "Static" refers to parameter allocation, shapes, keys, and path layout.
  Parameter values remain trainable unless explicitly frozen.
- The union interaction layout and path-aggregation parameters must also be
  constructed eagerly in `__init__`.

## What is a linear mixing path?

A linear mixing path is one possible particle-overlap relationship between an
input tuple and an output tuple. It is a basis element of a linear
permutation-equivariant map, not an execution path through the neural network.

Let

- `I = (i_1, ..., i_l)` be an ordered-distinct output tuple of order `l`;
- `J = (j_1, ..., j_m)` be an ordered-distinct input tuple of order `m`;
- `q` be an exact partial matching between the output slots `[l]` and input
  slots `[m]`.

A matched pair `(a, b)` in `q` means

```text
i_a = j_b.
```

Exactness means that all matched equalities hold and that no additional
cross-tuple equalities hold.

For a fixed output tuple `I`, path `q` computes

```text
h[I, q]^d = Reduce over J satisfying Eq(I, J) = q of
            W[q]^(d <- c) x[J]^c.
```

Here:

- `Reduce` is `sum` or `completion_mean`;
- `W[q]` is a static parameter with shape `[D_l, C_m]`;
- `C_m` is the number of input channels at order `m`;
- `D_l` is the number of output channels at order `l`;
- `Eq(I, J)` is the exact equality relation between the two tuples.

Particle relabeling preserves equality relations. Therefore, every path is
independently permutation equivariant.

If path `q` matches `r` particles, then for a fixed `I` its orbit contains

```text
(N - l) falling-factorial (m - r)
```

input tuples, provided the orbit is nonempty. This cardinality depends on the
runtime particle count `N`, while the path and its weight do not.

## Examples

### Order 1 to order 1

For output particle `i`, there are two complete linear paths.

Self path:

```text
h[i, self] = W[self] x[i]
```

Other-particle path:

```text
h[i, other] = mean over j != i of W[other] x[j]
```

These represent the two possible equality relations:

```text
j = i
j != i
```

### Order 1 to order 2

For output pair `I = (i_1, i_2)` and input particle `j`, there are three
complete paths:

```text
j = i_1
j = i_2
j not in {i_1, i_2}
```

### Order 2 to order 1

For output particle `i` and input pair `J = (j_1, j_2)`, there are three
complete paths:

```text
j_1 = i and j_2 != i
j_2 = i and j_1 != i
i not in {j_1, j_2}
```

### Order 2 to order 2

For output pair `I = (i_1, i_2)`, the complete basis contains seven paths.

Two shared particles:

```text
J = (i_1, i_2)    # identity matching
J = (i_2, i_1)    # swapped matching
```

One shared particle:

```text
J = (i_1, k)
J = (k, i_1)
J = (i_2, k)
J = (k, i_2)
```

where `k` is outside `{i_1, i_2}`.

No shared particles:

```text
J = (k_1, k_2)
```

where `k_1` and `k_2` are distinct and both lie outside `{i_1, i_2}`.

The path count is therefore

```text
2 + 4 + 1 = 7.
```

## Complete path count

For one input-order/output-order pair `m -> l`, the number of complete
partial-matching paths is

```text
 P(l, m) = sum over r = 0..min(l, m) of
          C(l, r) C(m, r) r!.
```

Here `r` is the number of matched particles. Some useful counts are:

| Mapping | Complete paths |
| --- | ---: |
| `1 -> 1` | 2 |
| `1 -> 2` | 3 |
| `2 -> 1` | 3 |
| `2 -> 2` | 7 |
| `3 -> 3` | 34 |

The configured path set remains static even if a small runtime `N` makes some
of these orbits empty or linearly dependent.

## Basis policies

### `coordinate_neighbor`

This is the proposed inexpensive, slot-aware k-GNN-style basis for same-order
mixing. It retains:

1. the aligned identity path; and
2. one path for replacing each aligned tuple coordinate.

For an order-two tuple `I = (i_1, i_2)`, the paths are

```text
identity:       J = (i_1, i_2)
replace first:  J = (k, i_2)
replace second: J = (i_1, k)
```

where replacement particle `k` is outside the output tuple. Thus the basis
contains `m + 1` paths at order `m`.

This deliberately excludes swapped matching, crossed slot matching, and
fully disjoint aggregation.

### `orbit_complete`

This policy includes every exact partial matching between configured input and
output orders. It may be configured as either:

- same-order only; or
- cross-order, allowing every selected input order to feed each output order.

For `max_order = 2`, the same-order counts are

```text
L_1 = 2
L_2 = 7
```

If both input orders 1 and 2 feed both output orders, the totals are

```text
L_1 = P(1, 1) + P(1, 2) = 2 + 3 = 5
L_2 = P(2, 1) + P(2, 2) = 3 + 7 = 10
```

### `explicit`

This policy loads a deterministic, metadata-selected subset of paths. It is
useful for scientific ablations and hand-designed intermediate bases.

## Interaction with TPEN path aggregation

Each selected relation remains a separate entry on the `Interaction` path
axis. The linear producer returns

```text
LinearInteraction[l]: [B, D_l, L_l, N, ..., N]
```

with `l` particle-index axes.

The tensor-product producer returns

```text
TPInteraction[l]: [B, D_l, T_l, N, ..., N].
```

Hybrid composition concatenates them:

```text
Interaction[l]: [B, D_l, L_l + T_l, N, ..., N]

path order: [linear paths | tensor-product paths]
```

The shared static aggregation parameter has shape

```text
U[l]: [D_l, L_l + T_l]
```

and contracts only the path axis:

```text
Update[l]: [B, D_l, N, ..., N].
```

For a residual update, `D_l` must equal the persistent feature channel count
`C_l`.

## Dense storage and semantic tuples

An order-`m` dense block stores `N^m` tuple positions, but only

```text
(N)_m
```

ordered-distinct positions are semantically active. Linear path enumeration
must operate through the explicit ordered-distinct tuple contract and must not
infer particle semantics by recursively inspecting arbitrary containers.

Repeated-index behavior must remain explicit. In particular, a pointwise
activation satisfying `Gamma(0) != 0` can write an invariant constant into
positions that were zero before activation, matching the existing TPEN
activation contract.

## Recommended first implementation scope

1. Define metadata capable of expressing complete partial matchings.
2. Implement the slow literal reference contraction.
3. Initially expose `coordinate_neighbor` as the inexpensive default.
4. Test a vectorized implementation against the slow reference.
5. Add `orbit_complete` without changing the producer or aggregation API.
6. Preserve every path and weight as an eager, static part of model
   construction.

## Resolved architecture contract

A unary SupportPath is specified by injective maps tau_out: [l] -> [s] and
tau_in: [m] -> [s], with image union [s]. For an ordered-distinct output I,
the completion set is

    C(I,q) = { K in [N]^s_distinct : K o tau_out = I }.

The unary producer computes

    y[I,q]^d = a(I,q) sum over K in C(I,q)
                W[q]^(d <- c) x[K o tau_in]^c,

where a is one for sum and 1 / |C(I,q)| for a nonempty orbit mean. An empty
orbit has zero-safe mean equal to zero before activation. Under relabeling
sigma, K -> sigma o K is a bijection from C(I,q) to C(sigma o I,q), proving
equivariance by change of variables. Tensor-product paths use the same
bijection on their two projections.

For output order l, the complete interaction path axis is the disjoint union

    P_l = P_l^linear union P_l^TP.

Linear paths precede TP paths in a deterministic canonical order. The shared
union aggregation learns one U over this full axis; family-wise and
hierarchical aggregation are deferred.

One immutable PathLayout is passed directly to both
CompositeMixing(layout=layout, producers=(...)) and
PathAggregation(layout=layout, ...). There is no factory and no
InteractionStage owner. The layout owns typed path metadata, family, orders,
channel contracts, normalization, version, counts, and a canonical fingerprint.
Value-equivalent layouts must have equal fingerprints.

The semantic core uses frozen dataclasses and tuples, not arbitrary nested
dictionaries. Concrete producer types are called directly: no getattr,
setattr, hasattr, import-by-name, string-based class or method lookup, or
string-keyed type registry is permitted. Hydra and JSON strings are boundary
adapters only.

The three Hydra producer sequences are:

    linear-only: [LinearEquivariantMixing]
    hybrid:      [LinearEquivariantMixing, EquivariantMixing]
    TP-only:     [EquivariantMixing]

These are ordered producer values, not three model classes or a Python mode
switch. All producer and aggregation parameters are registered eagerly before
the optimizer or DDP. Runtime N may create typed index tensors and empty
orbits, but cannot change path membership, parameter shapes, names, or keys.

## Hydra activation and interaction choice library

The opt-in choice fragment
`experiments/hooke/choices/tpen_presets.yaml` follows the shared Hooke choice
library convention. Merge it with a host config before calling `tpen.run`; it
is not a new production config root. Its `choices.activation.channel_mlp`
entry contains two explicit Hydra targets: `mixing` uses
`ChannelActivationAxes(tuple_axes_start=3)` for `[B,C,P,N^m]`, and
`aggregation` uses `tuple_axes_start=2` for `[B,C,N^m]`. Each interpolation is
instantiated at its consumer, so the eager MLP parameters are independent.

The three `choices.interaction` entries spell out concrete producer lists and
their immutable `PathLayout`: `linear`, `hybrid`, and `tensor_product`. The
`mode` value is boundary metadata only; no Python registry, factory, or string
dispatch selects an operator. Existing atomistic and Hooke production configs
remain unchanged, and the fragment does not alter the layout fingerprint.

## Amendment: canonical representation and compatibility details

The SupportPath representation is canonical and normative. Set tau_out(a)=a
for every output slot. A matched input slot receives its matched output slot's
label; unmatched input slots receive l, l+1, ... in increasing input-slot
order. Thus s=l+m-r is inferred, not independently chosen. Mathematical
notation is one-based; the serialized boundary representation is zero-based.
Validation, including explicit metadata, rejects noncanonical records and
duplicate canonical records. This prevents gauge-equivalent raw injections from
becoming unequal frozen records, distinct fingerprints, and duplicate weights.

Every producer returns pre-Gamma values. CompositeMixing concatenates the
linear and TP family outputs, applies the common Gamma once, and passes that
result to shared PathAggregation. The post-aggregation Gamma_c remains a
separate activation. Legacy direct EquivariantMixing remains compatible by
retaining its existing internal activation behavior at its direct public
boundary; a composite adapter treats its checked-in TP output contract as the
legacy boundary and must not apply Gamma a second time. New composite producers
must use the pre-Gamma contract.

Define the particle action by

    (rho_m(sigma) x)_J = x_(sigma^(-1) J).

The typed equivariance identity is

    y_q(rho_m(sigma) x)_(sigma I) = y_q(x)_I,

equivalently y_q(rho_m(sigma) x) = rho_l(sigma) y_q(x).

The canonical comparator for generated linear paths is input order m, then
overlap size r, then the lexicographically sorted matched output/input slot
pairs, then unmatched input-slot order. This defines a stable linear path id.
Path position in OutputPathLayout is contractual because U indexes that
sequence. The explicit policy preserves declared metadata order and does not
re-sort it; generated policies use the comparator.

Checked-in TP JSON is loaded verbatim and never regenerated. TP global_ids and
their relative path-axis order are preserved, while union-axis positions are
distinct from TP-local ids. TP-only composition keeps the existing
mixing.weights.g<global_id> state-dict namespace and o<order> aggregation keys;
it does not insert a producers.0 level. In hybrid composition, existing TP
columns follow the linear prefix. Hybrid U changes shape, so old TP-only
checkpoints load only in TP-only mode; hybrid requires explicit migration or
new aggregation initialization.

The fingerprint version is path-layout-v1. It is SHA-256 over deterministic
JSON serialization of recursive as_tuple() values, with stable ordering and no
whitespace variation. It includes version, ordered paths, family, orders,
channel contracts, normalization, and counts; it excludes Python class names
and field names. Value-equivalent means equal recursive tuple values and hence
equal fingerprints. Producer output carries this fingerprint as typed metadata;
composition binds it to the same PathLayout before concatenation, so equal path
counts cannot conceal different column meanings.

The sole normalization vocabulary is completion_mean, the v1 default, scoped
per path completion set; sum is an explicit alternative. Sum is zero on an
empty set. completion_mean is zero on an empty set and otherwise divides by
the positive cardinality, so zero times one-over-zero is never evaluated.
Cross-order orbit_complete receives its constructor-owned immutable tuple of
input orders and never discovers orders from runtime tensors. Explicit metadata
contains canonical SupportPath records and declared order; duplicates and
noncanonical records are rejected. Producers receive immutable family slices
and validate common D_l. A zero-path family contributes an empty typed slice
and no columns, while shared aggregation remains defined.

The dense active-position count is the falling factorial (N)_m, with
(N)_m=0 for m>N. At small N the static complete path family need not be a
literal basis; the nonempty-orbit indicators are the basis. For m>=3,
coordinate_neighbor also excludes swaps, crossings, and disjoint paths.
