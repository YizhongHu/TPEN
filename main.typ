// TPEN Design Document (Typst). Source of truth for the TPEN architecture;
// replaces the retired SpENN (Specht-module / irrep / Fourier) design doc
// per decision O2 / D15a.
//#import "@preview/noteworthy:0.2.0": * // Need to make TOC breakable
#import "lib.typ": *
#import "@preview/showybox:2.0.4": showybox
#import "@preview/equate:0.3.2": equate, share-align
#import "@preview/physica:0.9.6": braket, pdv, grad, curl, dd, Order
#import "@preview/mannot:0.3.0": markrect
// Theoretic imported with noteworthy
#show link: underline

#show: noteworthy.with(
  paper-size: "a4",
  font: "New Computer Modern",
  language: "EN",
  title: "TPEN Design Document",
  author: "Richard Hu",
  contact-details: "TPEN-QMC",
  toc-title: none,
  watermark: "DRAFT", // Optional: Watermark for the document
)

#show ref: theoretic.show-ref

#set cite(style: "chicago-notes")
#show ref: footnote

#set heading(numbering: "1.")

#let appendix(body) = {
  set heading(numbering: "A.a.i", supplement: [Appendix])
  counter(heading).update(0)
  body
}

#show heading.where(level:1): it => {
  counter(math.equation).update(0)
  it
}
#show: equate.with(breakable: true)

#set math.equation(numbering: it => {
  // Get the chapter number (first element of the heading counter)
  let chapter_num = counter(heading).get().first()
  // Format the number as (chapter.equation_number)
  numbering("(1.1)", chapter_num, it)
})
// #set math.mat(delim: "[")
#set enum(numbering: "1.a.i.")


// Write here

#let proposition = theorem.with(kind: "proposition", supplement: "Proposition", fmt-suffix: none)
#let question = proof.with(kind: "question", supplement: "Question", number: none, fmt-suffix: none)
#let answer = proof.with(kind: "answer", supplement: "Answer", number: none, fmt-suffix: none)
#let test = proof.with(kind: "test", supplement: "Test", number: none, fmt-suffix: none)
#let results = proof.with(kind: "results", supplement: "Results", number: none, fmt-suffix: none)

#let note = note.with(number: none)

// Physics symbols
#let varphi = $phi$ // Old habits die hard
#let phi = $phi.alt$
#let ang = $angstrom$
#let eV = $e V$
#let Hhat = $hat(H)$
#let That = $hat(T)$
#let Vhat = $hat(V)$
#let Vhee = $Vhat_(e e)$
#let ip(bra, ket) = {
  $chevron.l bra mid(|) ket chevron.r$
}
#let qf(bra, op, ket) = {
  $chevron.l bra mid(|) op mid(|) ket chevron.r$
}
#let EXC = $E_(X C)$

// Math Symbols

#let neq = $eq.not$
#let sgn = "sgn"
#let cA = $cal(A)$
#let mapsto = $arrow.r.bar$
#let mixprod = math.op(sym.triangle.stroked.small.t)

// Vector symbols

#let bh = $bold("h")$
#let bk = $bold("k")$
#let bl = $bold("l")$
#let bm = $bold("m")$
#let bq = $bold("q")$
#let br = $bold("r")$
#let bR = $bold("R")$
#let bu = $bold("u")$
#let bv = $bold("v")$
#let bw = $bold("w")$
#let bx = $bold("x")$
#let by = $bold("y")$
#let bz = $bold("z")$
#let bEta = $bold(eta)$
#let ij = $i j$
#let jk = $j k$
#let ki = $k i$
#let ijk = $i j k$

// Words
#let GSD = "generalized slater determinant"
#let ansatze = "ansätze"
#let Schrodinger = "Schrödinger"

#pagebreak()

= Introduction

Tensor-product Permutation Equivariant Network is a neural network
structure based on permutation-equivariant many-body features. The design
philosophies of this architecture can extend to other forms of permutation in/equi-variance,
and possibly become another generalization of GNN,
but we will focus on using it as an antisymmetric Quantum Monte Carlo ansatz.

= Background

== Definitions

=== Tuples
For a positive integer $n$, define $[n] = (1, dots, n)$, the tuple of elements from $1$ to $n$.
Depending on the situation, this can also be interpreted as a set ${1, dots, n}$.

Permutations are injective (and thus bijective) maps $sigma: [n] mapsto [n]$. We denote
injective maps with a hooked arrow: $sigma: [n] arrow.hook [m]$. The set of permutations
of order $n$ is denoted $S_n$. This forms a group under composition.

For a generic tuple of positive integers $I = (i_1, i_2, dots, i_ell: i_k in [n])$, the order
or cardinality of the tuple $|I| = n$.

=== Tuple Maps
The tuple $I = (i_1, i_2, dots, i_ell: i_k in [n])$ contains the same information as a map
$
tau_I: [ell] & mapsto [n]\
        k & mapsto i_k
$
Thus a tuple can also be used inter-changeably with a map.
For a tuple $I:[ell] mapsto [n]$ and $J:[m] mapsto [ell]$,
we can define $I_J = (i_j_k: k in [m])$ as the composition of the two tuples.

For a map $tau: [m]mapsto[n]$, its image is defined as
$ "im"(tau):= {tau(k) :k in [m]} $
For a set $S subset.eq [n]$, it can be filtered and removed from a tuple:
$ I\\S := {i: i in.not S} $

== Quantum Monte Carlo and Antisymmetric functions

Quantum Monte Carlo attempts to obtain the ground state of the #Schrodinger
equation:
$ Hhat psi = E psi $
where $psi: (RR^3 times {plus.minus 1})^n mapsto CC$ is the wave function
of the electrons, and
$ Hhat = sum_i -1/2 nabla^2_i + sum_(A, i) - (Z_A)/(||br_i - bold("R")_A||)
+ sum_(i < j) 1/(||br_i - br_j||) $
Importantly, since electrons are fermions, the wave function is antisymmetric:
$ psi(br_(sigma^(-1) (1)), br_(sigma^(-1) (2)), dots, br_(sigma^(-1) (n)))
 = sgn(sigma)psi(br_1, br_2, dots, br_n) $
for all $sigma in S_n$, a permutation of $n$ elements, and $sgn(sigma)$
the sign (or parity) of the permutation.

To calculate the ground state, we minimize the following functional
$ E_0 = "min"_(psi) integral psi^* (br) Hhat psi (br) d br $
To approximate the integral, we convert it into an expectation value:
$ integral psi^* (br) H  psi (br) d br = EE_(br ~ |psi(br)|^2) E_"loc" (br) $
where $E_"loc"$ the *local energy* is defined as
$ E_"loc" (br) = (H psi(br))/(psi(br)) $
Hence we can use a MCMC sampler on $|psi(br)|^2$ to estimate $E$
$ E = 1/N sum_(a=1)^N E_"loc" (br^a) $
Using TPEN as an ansatz for $psi$, we estimate the energy $E$ and train the parameters
of the neural network to minimize it.

= Design

== Inputs and Embedding

The inputs are all fed into an embedding layer that gives a tensor $bx_I$, representing the
$m$-th body order interaction.

$ bx_I^(m) = phi^m (br_i_1, dots.c, br_i_m) $

Here $I$ is a tuple of non-repeating indices $(i_1, i_2, dots, i_m : i_k in [n])$
#footnote[We use $[n]$ to represent the set ${1, dots, n}$].

Since the body order $m = |I|$, we will suppress $m$ unless otherwise stated, since it
can be directly derived from $I$.

This embedding scheme means that we are not restricted to sending $RR^3$
coordinates into the encoder. In fact, this can be an arbitrary vector that describes
as much about the particle as possible. We gain a more extensible model for free.

In the QMC case, we can easily encode spin in this way:
$ bv_i = (br_i, s_i) = (x_i, y_i, z_i, s_i) $
where $s_i in {1, -1}$.

To generalize even more, one can even build a full QMC model with atomic coordinates this way:
$ bv_i = (br_i, s_i, ["one-hot encoding of particle type"]) $
which variables are able to freely move is determined by the MC walker. The model can be
agnostic about what type of particle it is dealing with.

To preserve more information from the input, we can stack multiple channels:
$ bx^(0, c, m)_I = phi^(c m) (br_i_1, dots.c, br_i_m) $
where $c$ is the channel index.

== TPEN Layers Overview <tpen-layers>
#block(width: 100%)[
There are two steps in a TPEN layer: path production and aggregation.
  #set math.equation(numbering: none)
  $
  bx^m_I stretch(->)^"path producers"_(W) by^m_(I, p)
  stretch(->)^"concatenate paths + activation"_Gamma bh^m_(I,p)
  stretch(->)^"aggregation"_U bw^m_I
  stretch(->)^"activation"_(Gamma_c) bu_I
  stretch(->)^("update") bx_I^m
  $
  All operations happen in real space. The path axis $p$ is opened by mixing and
  contracted by aggregation, so the post-aggregation update $bu_I$ carries no path
  axis.
]

== Equivariant Mixing

The mixing process takes features $x^(t c)_I$ and produces equivariant paths
$bh^((t+1)c)_(I, p)$. A path is indexed by its family and its typed support
metadata. This information may be passed on to features of equal or lower
body-order than the interaction.

Mixing has the general form (before the common activation):
$ bh^c_(I, p) = sum_(J_([s]\\"im"(tau))) W_p^(c<-c_1c_2) bx_(J circle.small tau_1)^c_1 bx_(J circle.small tau_2)^c_2 $
where the path index $p = (s, m, m_1, m_2, tau, tau_1, tau_2)$ describes the different equivariant paths indices
through which $bx$ can interact and produce features on index $I$.
- $s$: interaction order--total number of particles involved in this interaction.
- $m$: output order--the order of the output feature, the feature that accepts the information from the interaction.
  $m lt.eq s$
- $m_1, m_2$: input orders--the orders of the input features that interact with each other and form a virtual
  feature of order $s$. $m_1 + m_2 gt.eq s$.
and activation has the general form
- $tau$: $[m] arrow.hook [s]$,  output injection--an injective function the describes where the indices of $I$
  are positioned in the virtual interaction.
- $tau_1$: $[m_1] arrow.hook[s]$, left input injection--an injective function that describes where the indices of $I_1$
  are positioned in the virtual interaction.
- $tau_2$: $[m_2] arrow.hook[s]$, right input injection--an injective function that describes where the indices of $I_2$
  are positioned in the virtual interaction.
- $sum_(J_([s]\\"im"(tau)))$ means summing over all the indices
  in the virtual interactions that is not already in the image of $tau$, i.e. the indices in $I$ that are corresponding to.
- Every producer returns these pre-$Gamma$ path values. The common
  shape-preserving $Gamma$ is owned by the composite path expansion and is
  applied exactly once after all family outputs have been concatenated. It may
  be pointwise or the opt-in channel-preserving MLP, which mixes channels only
  at each fixed inert position.

=== Unary support paths

The tensor-product formula above is one path producer. A linear, unary producer
uses the same ordered-distinct tuple semantics. Let $I$ have output order $l$
and let $J$ have input order $m$. A *SupportPath* is the typed record
$q = (l, m, tau_o, tau_i)$, where
$tau_o: [l] arrow.hook [s]$ and $tau_i: [m] arrow.hook [s]$ are injective and
$"im"(tau_o) union "im"(tau_i) = [s]$. It records exactly which output and
input slots denote the same support particle; the overlap has size
$r = |"im"(tau_o) intersection "im"(tau_i)|$.

For an ordered-distinct output tuple $I$, define the completion set
$
  c(I,q) = { K in [n]^s_"distinct" : K circle.small tau_o = I }
$
and the unary path
$
  y^d_(I,q) = "Reduce"_(K in c(I,q))
    W_q^(d <- c) x^c_(K circle.small tau_i).
$
Here Reduce is either a sum or the configured completion_mean. Equivalently, this
is a completion sum with normalization $a_(I,q)$:
$ y^d_(I,q) = a_(I,q) sum_(K in c(I,q)) W_q^(d <- c) x^c_(K circle.small tau_i) $,
where $a_(I,q)=1$ for sum and $a_(I,q)=1/|c(I,q)|$ for a nonempty completion_mean.
An empty completion has a zero-safe mean of zero before activation. The weight
$W_q$ has static shape $[D_l, C_m]$.

The support labeling is canonical and normative: $tau_o(a)=a$ for output
slots; a matched input slot takes the label of its matched output slot; and
unmatched input slots take labels $l, l+1, dots$ in increasing input-slot
order. Thus $s=l+m-r$ is inferred from the canonical matching, rather than
being an independent choice. Mathematical slots are one-based, while the
serialized boundary representation is zero-based. Validation, including the
explicit policy, rejects noncanonical records. Gauge copies are not new paths:
if admitted, frozen-record equality and the layout fingerprint would allocate
duplicate parameters for the same map.

=== Equivariance of completion paths

For a relabeling $sigma in S_n$, define the feature action by
$ (rho_m(sigma)x)_J = x_(sigma^(-1) J) $. Apply $sigma$ entrywise to tuples. The map
$K mapsto sigma circle.small K$ is a bijection
$c(I,q) mapsto c(sigma circle.small I,q)$, because it preserves distinctness
and all equalities recorded by the same SupportPath $q$. Therefore, changing
variables in the completion sum gives
$ y_q(rho_m(sigma)x)_(sigma circle.small I) = y_q(x)_I $,
equivalently $y_q(rho_m(sigma)x) = rho_l(sigma)y_q(x)$.
The same argument applies to a tensor-product path: its two input projections
are evaluated on the same completion set. Thus unary and tensor-product paths
are separately equivariant, and concatenating their path axes preserves
equivariance.

=== Path families and policies

For output order $l$, the path axis is the disjoint union
$ P_l = P_l^"linear" union.sq P_l^"TP" $.
The linear family uses unary SupportPaths; the TP family uses the existing
bilinear support maps. A linear basis policy may be coordinate_neighbor,
orbit_complete, or explicit:

- coordinate_neighbor keeps the aligned identity and one replacement path
  for each aligned coordinate, so an order-$m$ same-order feature has $m+1$
  paths. For $m >= 3$ it also excludes swaps, crossings, and disjoint paths;
  it is intentionally incomplete.
- orbit_complete keeps every exact partial matching, optionally across
  input orders.
- explicit keeps a deterministic metadata-selected subset.

The complete number of paths from input order $m$ to output order $l$ is
$
  P(l,m) = sum_(r=0)^min(l,m) binom(l,r) binom(m,r) r! .
$
The examples are $1 -> 1: 2$, $1 -> 2: 3$, $2 -> 1: 3$, and
$2 -> 2: 7$ (and $3 -> 3: 34$). The $r!$ factor orders the matched slot
pairs. For finite $n$, an orbit may be empty, but its path and parameter stay
registered. The static list is a complete path family, while at small $N$ the
nonempty-orbit indicators are the literal basis.

=== Shared interaction layout and aggregation

The path producers return tensors with a common layout:
$
  y^"linear"_l: [B, D_l, |P_l^"linear"|, N^l], quad
  y^"TP"_l: [B, D_l, |P_l^"TP"|, N^l].
$
Every producer returns pre-$Gamma$ values. They concatenate on the path axis in
deterministic order, with linear paths followed by TP paths; the composite
applies $Gamma$ exactly once, then shared union aggregation contracts the full
axis:
$
  u^d_I = sum_(p in P_l) U_l^(d,p) Gamma(y^d_(I,p)),
$
or, in the implementation order, applies the common shape-preserving $Gamma$
once after concatenation and then $U_l$. This is the v1 shared-union policy;
family-wise and hierarchical aggregation are deferred.

One immutable typed PathLayout value is passed directly to both
CompositeMixing(layout=layout, producers=(...)) and
PathAggregation(layout=layout, ...). There is no factory and no separate
InteractionStage owner. The layout owns ordered path metadata, family,
orders, channel contracts, normalization, version, counts, and a canonical
layout fingerprint. Equal fingerprints are required for concatenation and
aggregation; object identity is not required.

The architecture rule is strict: semantic records are frozen dataclasses and
tuples, and all producer and aggregation parameters are allocated eagerly in
construction. Forward may materialize typed index tensors for the current
$N$, but may not add paths or parameters. Named parameters and state keys are
unchanged when $N$ changes. Concrete producer modules are called directly;
there is no reflective access (getattr, setattr, or string-based method or
class lookup) and no string-keyed type registry. Hydra and JSON strings are
boundary adapters only.

Path ordering is contractual because aggregation weights index positions. For
each output order, the stable linear path id is the lexicographic tuple
$(m, r, M)$, where $m$ is input order, $r$ is overlap size, and $M$ is the
lexicographically sorted list of matched output/input slot pairs, followed by
the unmatched input-slot order. The comparator is input order, then increasing
overlap, then matching pairs, then unmatched slots. The explicit policy
preserves its declared metadata order and does not re-sort it; all other
generated policies use the comparator. The TP JSON is loaded verbatim, never
regenerated: its global ids and relative path-axis order are preserved.
Union-axis positions are distinct from TP-local ids. TP-only composition keeps
the existing mixing.weights.g<global_id> namespace and o<order> aggregation
keys; it does not insert a producers.0 level. In hybrid composition, existing
TP columns map after the linear prefix. Because hybrid aggregation changes $U$
shape, old TP-only checkpoints load only in TP-only mode; hybrid requires an
explicit migration or a newly initialized aggregation parameter.

The fingerprint version is path-layout-v1. It is SHA-256 over deterministic
JSON serialization of recursive as_tuple() values, with stable list ordering
and no whitespace variation. It includes version, ordered paths, family,
orders, channel contracts, normalization, and counts; it excludes Python class
names and field names. Value-equivalent layouts have equal recursive tuple
values and therefore equal fingerprints. Producer outputs carry the layout
fingerprint as typed metadata; composition binds that metadata to the same
PathLayout before concatenation, so equal path counts cannot hide different
column meanings.

Normalization is named completion_mean and is the v1 default, scoped per path
completion set (sum remains an explicit alternative). Its reduction is
piecewise: sum is zero for an empty set; completion_mean is zero for an empty
set and otherwise is the sum divided by its positive cardinality, so no
implementation evaluates zero times one over zero.

The constructor owns the finite input-order tuple for cross-order
orbit_complete; it is not discovered from runtime tensors. Explicit metadata
contains canonical SupportPath records and declared order; duplicate canonical
records are rejected, and noncanonical/gauge duplicates are rejected rather
than deduplicated silently. Producers receive an immutable family slice of the
PathLayout and validate the common output channel width $D_l$ before binding.
A family may contribute zero paths at an output order; its empty slice remains
typed, contributes no columns, and shared aggregation remains defined.

=== Hydra producer sequences

Hydra selects an ordered producer sequence, not a Python mode switch or one of
three model classes:

- TP-only: producers: [tensor_product];
- linear-only: producers: [linear];
- hybrid: producers: [linear, tensor_product].

The sequence determines the static path layout and is recorded in its
fingerprint. All weights exist before optimizer or distributed-data-parallel
construction, including weights for empty runtime orbits.

=== Decoherent Paths

When $"im"(tau) subset.eq "im"(tau_1) union "im"(tau_2)$, the set of indices in the output
feature is a subset of the set of indices in the input. This is known as *coherent*. But
not a lot is stopping us from creating *decoherent* paths, for example:
$ i <- j, j\
  ij <- i k, k i
$
Although they are not physically sound, I don't see them breaking equivariance, so
they can very-well be included.


== Aggregation

Aggregation contracts the path axis per input channel and then applies an owned
activation $Gamma_c$ across the channel axis:

$ bu^(c)_(I) = Gamma_c (sum_p U^(1)_(p) bh^1_(I, p), dots, sum_p U^(C_"in")_(p) bh^(C_"in")_(I, p)), quad c in [C_"out"] $

where each per-input-channel weight $U^(c)_(p)$ contracts only the (inert) path
axis $p$, and $Gamma_c: RR^(C_"in") mapsto RR^(C_"out")$ acts identically on
every tuple $I$. Since it touches only the inert path and channel axes, this
operation is permutation-equivariant.

The current implementation includes the elementwise $Gamma_c$ with
$C_"out" = C_"in"$, so that form leaves channel mixing in the mixing weights
$W$. It also includes the opt-in channel-preserving
`ChannelPreservingMLPActivation`, which retains $C_"out" = C_"in"$ while
mixing channels inside one eagerly constructed MLP per tensor order. The
activation instance owns those MLP parameters, order selection, channel-axis
movement, validation, initialization, and its immutable
`ChannelActivationAxes`/`OrderMLPLayout`; every non-channel axis is inert to
the MLP and is folded into its leading batch positions.

== Activation and Updates

Arbitrary point-wise activation on tensors preserves equivariance:
$ bh_I = Gamma(by_I) $

The channel-preserving MLP refinement is not pointwise in channels, but it is
shared over every inert position. Therefore it preserves the same tuple-index
permutation action while making channel ownership explicit at the activation
boundary.

For an update $bu^(t+1)$, we can directly apply the update:
$ bx^(t+1) = bold("u")^(t+1) $
For $u^(t+1) ~ bx$, we can use a residual update:
$ bx^(t+1) = bx^t + bold("u")^(t+1) $

== Normalization and Envelopes

Normalization is a function on the feature/update itself:
$ "Normalize"(bx) $
An envelope is a *multiplicative* re-scaling of the feature by a function of the
input coordinates:
$ "Envelope"(br)bx $
They can also happen at the same time (though it is usually not a good idea)
$ "Envelope"(br)"Normalize"(bx) $
They are ways we can keep the scale of the outputs in check
in case polynomial growth causes problems.

These multiplicative coordinate envelopes are `tpen.nn.CoordinateEnvelope` and
its descendant `tpen.nn.GaussianCoordinateEnvelope`. Each is a single
`EquivariantMap` that owns its own multiplication --- there is no producer/applier
split --- multiplying intermediate `Feature` or `Update` blocks inside
`tpen.nn.Embedding` or `tpen.nn.TPENLayer`. They are distinct from the additive
log-amplitude factors (cusps and confinements) of the Cusps and Confinement
section, which add to $log abs(psi)$ rather than multiplying features.

== Antisymmetric Readout

=== Readout vs. Encoder Anti-symmetrization tradeoff <as-irreps>

TPEN passes real-space tuple features between its layers; there is no irrep or
Fourier round-trip carrying data from one layer to the next. Nonetheless, the
*choice of where to antisymmetrize* is most naturally analyzed through the
representation theory of $S_n$. Consider an order-$m$ feature. It is acted on by
the $S_m times S_(n-m)$ subgroup of $S_n$: an element of $S_m$ describes an
*orbit* in $S_n$, the subgroup of elements under which the indices of some tuple
$I$ stay invariant.

The irreps that we are using is
$ "Ind"^(S_n)_(S_m times S_(n-m)) (S^lambda times.square bold("1")) $
where we assume that all the variation happens within $S_m$ and the irrep is *permutation-invariant*
to $S_(n-m)$. The advantage of this is that it is very easy to encode the irreps at the start,
but we need to anti-symmetrize at the end.

The alternative is to instead work with antisymmetrized irreps
$ "Ind"^(S_n)_(S_m times S_(n-m)) (S^lambda times.square S^((1^(n-m)))) $
This assumes that the irreps are *anti-symmetric* to the indices not recorded, but encoding
these irreps faithfully is challenging. The correct way of doing this is
$ hat(bx)^lambda_I = sum_(sigma in S_(n-m)) sgn(sigma)rho^lambda (sigma)phi^({|I|})_"AS" (br_I, br_(sigma^(-1) ([n] backslash I))) $
This has no trivial simplification and is factorial time with respect to the number of particles.
We will need to craft the features meticulously to loose as little important information as possible.
But this benefit of this construction is that the final irreps are readily anti-symmetric and we
can just take a linear combination of them.

Anti-symmetrization is unavoidable, and choosing to do it in the "correct" place is very important.
For simplicity, we again choose to do anti-symmetrization in the end. This sounds *very similar*
to many of the existing methods, but I think that there is a lot of room to explore based on this framework.

=== Pfaffian Readout

The most obvious readout method is using pfaffians on the (1,1) irrep.
$ psi(br) = sum_(c=1)^C b_c "Pf"[(hat(bx)^(c, (1,1)))_ij] $
For the cases where $n$ is odd, we can instead do
$ psi(br) = sum_(c=1)^C b_c "Pf" mat(
  hat(bx)^(c"," (1","1)), -hat(bx)^(c"," (1));
  (hat(bx)^(c"," (1)))^T, 0
) $
This is because the tensor product of $n\/2$ $(1,1)$ irreps contain a copy of the $(1^n)$ irrep.

Assuming that channels are sufficiently mixed in the TPEN layers, this is the only
irrep with order $n$ that we can read-out from a network of maximum interaction order $M=2$.

=== Generalized Pfaffian readout

For order $m=3$, the readout is (for $n=3r$):
$ Psi^(c,(1,1,1)) = sum_(sigma in S_n) epsilon.alt_(i_1 j_1 k_1 dots i_r j_r k_r) hat(bx)^(c, (1,1,1))_(i_1 j_1 k_1) dots hat(bx)^(c, (1,1,1))_(i_r j_r k_r) $
where $epsilon.alt$ is the generalized Levi-Civita tensor:
$ epsilon.alt_(i_1 j_1 k_1 dots i_r j_r k_r) =
cases(
  sgn(i_1 j_1 k_1 dots i_r j_r k_r) & ", no repeated entries",
  0 & ", otherwise"
)
$
A similar form exists for the Pfaffian, just with three-body interactions, but the catch is that while
the Pfaffian can be calculated in $O(n^3)$ time, the generalized Pfaffian of order 3 cannot. In fact,
it is exponential in terms of $n$. Because of this, we have to abandon calculating the exact generalized
pfaffian for order-3 and above.

=== Higher-order readouts

Given that the maximum order $M=3$, the only channel-wise order-n read-outs are the pfaffian and the generalized
pfaffian, but we can increase the order to include other irreps, i.e., the polynomial order of the irreps in the
read-out phase $r$ has been correlated with $n$ in the two cases we presented ($r=n/2$ for pfaffians and $r = n/3$
for order-3 pfaffians), but taking higher order tensor products of irreps can result in more copies of $(1^n)$.

The generalized pfaffian for polynomial order $r$ is
$
  Psi_(m,r)(X)
  =
  sum_c
  sum_(phi in Phi_(m,r))
  b_(phi,c)
  sum_(alpha_1, dots.c, alpha_n)
  epsilon_(alpha_1 dots.c alpha_n)
  product_(k=1)^r
  a^c_(
    alpha_(phi(k,1))
    dots.c
    alpha_(phi(k,m))
  )
$
for $Phi_(m, r)$ the set of surjective functions $phi:{1 dots r}times{1 dots m}mapsto {1 dots n}$ and $a^c_I := hat(bx)^(c, T, lambda)_I$

A more familiar form uses the generalized Levi-Civita tensor:
$
E_phi(i_(1,1), dots.c, i_(r,m))
=
sum_(alpha_1, dots.c, alpha_n)
epsilon_(alpha_1 dots.c alpha_n)
product_(p=1)^r
product_(q=1)^m
delta_(i_(p,q), alpha_(phi(p,q)))
$
so we can define
$
Psi_(m,r)(X)
=
sum_(c=1)^C
sum_(phi in Phi_(m,r))
b_(phi,c)
sum_(i_(p,q))
E_phi(i_(1,1), dots.c, i_(r,m))
product_(p=1)^r
a^c_(i_(p,1) dots.c i_(p,m))
$
Note that the dimensions of $b_(phi,c)$ grows exponentially with $r$ and $n$.

=== Channel-mixing

Since channels already mixes in aggregation, we consider it rather redundant to mix channels
again in the readout phase, but we must highlight a very common method of channel-mixing readout:
determinants.
$ Psi = det [hat(bx)^(c, T, (1))_i] $
We form a matrix with axis 0 being the channels and axis 1 being the particle index.
This type of readout has been used extensively in mainstream NN-QMC methods.

== Cusps and Confinement

The wavefunction carries a *required* additive log-amplitude factor applied
outside the antisymmetric TPEN/readout stack:

$
psi(br) = exp(J (br)) psi_theta (br),
$

or equivalently,

$
log abs(psi(br)) = J (br) + log abs(psi_theta (br)).
$

This keeps short-range coalescence and long-range confinement independent from
the determinant/Pfaffian readout and preserves the antisymmetry of $psi_theta$.

The factor $J$ is *additive* in $log abs(psi)$. We reserve the word *envelope*
for the *multiplicative* coordinate factors of the Normalization and Envelopes
section. Within the additive-to-$log abs(psi)$ factors, terminology is fixed
by *behavior*, not body order: a *cusp* enforces the short-range Kato condition
at a coalescence point --- the required derivative discontinuity of $psi$ as
two particles, or a particle and a fixed nucleus, approach each other. *Decay*
(equivalently *confinement*) supplies the long-range guarantee that
$|psi|^2$ stays normalizable as particles move away from the system. A cusp
can be two-body (electron-electron) or summed one-particle-at-a-time over
fixed external centers (electron-nucleus); both are cusps because both enforce
a short-range Kato condition. The Gaussian confinement below, and its
generalizations, are decay/confinement terms because they govern long-range
normalizability, not short-range coalescence. In code these components
currently compose through `tpen.nn.AdditiveEnvelope`, which the wavefunction
takes as a required argument. `AdditiveEnvelope` is a *compatibility name*: the
word "envelope" is reserved above for the multiplicative coordinate factors,
so the shipped class name does not match the terminology it implements. The
target generic post-readout log-amplitude interface is `LogAmplitudeFactor`,
composing any mix of cusp and decay/confinement terms; `AdditiveCusp` is the
narrower composition specifically over cusp factors (e.g. electron-electron
and, later, electron-nucleus). Renaming `AdditiveEnvelope` to `LogAmplitudeFactor`
is tracked separately and is not part of this terminology-only change.
`Envelope` and `AdditiveEnvelope` (module `tpen.nn.envelope`) are legacy
output-factor/composition APIs for the current minor version, not a
feature-normalization step; neither carries a runtime deprecation warning
yet. The canonical `LogAmplitudeFactor`/`AdditiveCusp`
interfaces live in `tpen.nn.factor`, and the concrete cusp factors
(`ElectronElectronCusp`, `ElectronNucleusCusp`, and their laws) live in
`tpen.nn.cusp`; `AdditiveCusp` is retained only as a legacy compatibility
compositor, not for use in new configs or docs. Every name that previously
resolved through `tpen.nn.envelope` still does, via re-export, so this module
split changes no import path. `tpen.nn.TPENWaveFunction.factors` is the
canonical composition seam for `LogAmplitudeFactor` terms (see Model Workflow
below). The name `FeatureEnvelope` is reserved for a future typed
feature-space transform --- a distinct concept from the multiplicative
coordinate `Envelope` of the Normalization and Envelopes section --- and must
never be introduced as an alias or rename of that existing `Envelope`.

$
J (br)
=
J_"ee" (br) + J_"conf" (br).
$

Here $J_"ee"$ is the two-body electron-electron cusp and $J_"conf"$ is a one-body
confinement. The shipped Hooke stack is
`AdditiveEnvelope(ElectronElectronCusp, GaussianConfinement)`, i.e. a
`LogAmplitudeFactor` composing one `AdditiveCusp` term (`ElectronElectronCusp`)
and one decay/confinement term (`GaussianConfinement`).

=== Electron-electron cusp

The two-body electron-electron cusp (`tpen.nn.ElectronElectronCusp`) is the
shipped cusp module. It is an explicit pairwise term

$
J_"ee"(R)
=
sum_(i < j) u_(sigma_i sigma_j)(r_ij),
$

with

$
u_(sigma_i sigma_j)(r)
=
frac(a_(sigma_i sigma_j) r, 1 + b_(sigma_i sigma_j) r).
$

The cusp slope is fixed by

$
u'_(sigma_i sigma_j)(0) = a_(sigma_i sigma_j).
$

Use separate slopes for same-spin and opposite-spin pairs:

$
a_"same" = 1 / 4,
quad
a_"opp" = 1 / 2.
$

The range parameters $b_"same"$ and $b_"opp"$ may be trainable, constrained positive by e.g.

$
b = "softplus"(tilde(b)) + epsilon.
$

=== Gaussian confinement

Because the Pfaffian readout is polynomial in the features, a trapped system
needs a guaranteed-by-construction decay factor for $|psi|^2$ normalizability
--- a *decay/confinement* term, not a cusp. The one-body Gaussian confinement
(`tpen.nn.GaussianConfinement`) supplies it:

$
J_"conf"(br)
=
- alpha sum_i |r_i|^2,
quad alpha >= 0.
$

For a Hooke or oscillator frequency $omega$, the fixed Gaussian ground-state tail
uses

$
alpha = omega / 2,
$

exposed by the $omega$-parametrized convenience subclass
`tpen.nn.HookeGaussianConfinement`. This factor is smooth rather than cusp-like,
but it is still additive in $log abs(psi)$ and is applied through the additive
log-amplitude interface, not inside the antisymmetric readout.

=== Electron-nucleus cusp

For all-electron Hamiltonians, electron-nucleus coalescence also needs explicit
handling. Although it sums one particle at a time against fixed external
centers, it is named by short-range Kato behavior, not body order: it is a
*cusp*, `tpen.nn.ElectronNucleusCusp`, not a confinement. It is shipped: it
composes a typed `ElectronNucleusCuspLaw` (default
`LinearElectronNucleusCuspLaw`, reproducing the legacy He linear cusp $-Z r$)
against a constructor-owned `AtomicConfiguration`. The Hooke systems studied
elsewhere in this document contain no nuclei, so the term is simply absent
from their factor list rather than deferred.

Fixed nuclei are described generically: a constructor-owned
`AtomicConfiguration` holds the nuclear positions $R_A$ and charges $Z_A$ for
one system, and is passed once, at construction, to whichever `HamiltonianTerm`,
cusp, or decay/confinement module needs it. Helium and molecular hydrogen are
two `AtomicConfiguration` instances --- *data*, not distinct wavefunction
subclasses or branches. There is no He- or H2-specific wavefunction path;
`tpen.nn.TPENWaveFunction` and its consumers stay generic over any
`AtomicConfiguration`. Particle counts, permutations, and validation come only
from typed `.permute(...)`, `.compare(...)`, and `.validate(...)` contracts and
explicit `n_electrons`/nuclear metadata --- never from recursively probing an
arbitrary container.

The same canonical/legacy split applies on the Hamiltonian side. A `*Potential`
class (`tpen.physics.potential.ElectronNucleusPotential`,
`tpen.physics.potential.NucleusNucleusPotential`) is the canonical fixed-
`AtomicConfiguration` Hamiltonian API: it is constructed once from one
`AtomicConfiguration` and applies that single geometry to every sample. The
legacy `*Interaction` classes
(`ElectronNucleusInteraction`, `NucleusNucleusInteraction`) are a supported
minor-release compatibility surface: they read nuclear geometry from
batch-transported metadata and additionally support *per-configuration*
geometry that varies within one batch --- a broader batch geometry capability
the constructor-owned `*Potential` API does not need. Neither generation is
deprecated in this minor version.

`tpen.physics.hamiltonian.NaiveLocalEnergyEvaluator` remains the reference
local-energy evaluator for every Hamiltonian, cusp, and decay/confinement term,
including any future all-electron system built on `AtomicConfiguration`. A
future analytic electron-nucleus cusp capability must be introduced as its own
typed `LocalEnergyEvaluator`/context pair that fails loudly when misapplied ---
it must not silently fall back to container traversal or branch on which
molecule is present.

Use

$
J_"en"(R)
=
sum_i sum_A v_A (r_(i A)),
quad
r_(i A) = norm(r_i - R_A).
$

The required short-range slope is

$
v_A'(0) = -Z_A,
$

and the simplest analytic form that enforces it exactly is

$
v_A (r)
=
frac(-Z_A r, 1 + b_A r).
$

The electron-nucleus term is spin-independent: $v_A (r_(i A))$ does not depend on
$sigma_i$. The range parameter $b_A$ may be fixed globally, shared by nuclear
charge, or learned per nucleus:

$
b_A = b
quad "global",
quad
b_A = b_(Z_A)
quad "shared by nuclear charge",
quad
b_A
quad "one per nucleus",
$

constrained positive by $b_A = "softplus"(tilde(b)_A) + epsilon$. A global $b$ or
one $b_(Z)$ per nuclear charge is the recommended starting point.

Every electron-nucleus cusp law must keep fixing the first-order Kato slope
$v_A'(0) = -Z_A$ by nuclear charge alone. A law may separately expose an
*optional* trainable regular radial component $w_A (r)$ that contributes only
to second-order (and higher) curvature, i.e. satisfying $w_A (0) = 0$ and
$w_A'(0) = 0$ so it cannot perturb the enforced cusp slope:

$
v_A (r) = -Z_A r + w_A (r), quad w_A (0) = 0, quad w_A'(0) = 0.
$

This lets curvature near the nucleus be learned without touching the
mandatory cusp condition. `tpen.nn.cusp.CurvatureElectronNucleusCuspLaw`
implements this contract with $w_A (r) = c r^2 slash (1 + d r)$ for a trainable
(unconstrained-sign) $c$ and positive $d$; `LinearElectronNucleusCuspLaw`
remains the exact compatibility default ($w_A = 0$).

== Loss function

The loss function that we will be using is the following:
$ cal(L)(theta) = 2 EE_(br ~ |psi|^2)[(E_"loc" (br) - E)_"detach" log psi_theta (br)] $
It will be approximated by the LLN estimator
$ cal(L)(theta) = 2/N sum_(a=1)^N [(E_"loc" (br^a) - E)_"detach" log psi_theta (br^a)] $
where $br^a$ are iid samples of $|psi|^2$.

The gradient of this loss function $nabla_theta cal(L)$ is equivalent to $nabla_theta E$.

= Model Workflow

Implemented in `tpen.nn.TPENWaveFunction`.

+ Input: particle positions $bv_i = (br_i, s_i)$ (Optional particle-wise basis)
+ Embedding (`tpen.nn.Embedding`), learnable: $phi^(m): bv_I mapsto bx_I^(0, c, m)$ (Optional embedding normalization/envelope)
+ TPEN Stack (`tpen.nn.TPENStack`)
  + TPEN layer 1 (`tpen.nn.TPENLayer`)
    + mixing in real space (`tpen.nn.EquivariantMixing`)
      $ bh^c_(I, p) = sum_(J_([s]\\"im"(tau))) W_p^(c<-c_1c_2)
      bx_(J circle.small tau_1)^c_1 bx_(J circle.small tau_2)^c_2 $
      followed by the common path activation $Gamma$ after all producer
      families are concatenated.
    + Obtain update with `tpen.nn.PathAggregation`
      $ bu^(c)_(I) = Gamma_c (sum_p U^(1)_(p) bh^1_(I, p), dots, sum_p U^(C_"in")_(p) bh^(C_"in")_(I, p)) $
      followed by optional update normalization/envelope
    + Feature update (`tpen.nn.update`)
      $ bx^1 = "Update"(bx^0, bu^1) $
      most commonly with `tpen.nn.update.ResidualUpdater`.
      Followed by optional feature normalization/envelope
  + TPEN layer 2
    $
    bx^2 = "TPENLayer"(bx^1).
    $

  + $dots$

  + TPEN layer $T$
    $
    bx^T = "TPENLayer"(bx^(T-1)).
    $
+ Readout with `tpen.nn.PfaffianReadout`
  $ Psi = sum_(c) w^(c) "Pf"[bx^(T c)_(i j) - bx^(T c)_(j i)] $
+ Applied additive post-readout log-amplitude factor with
  `tpen.nn.AdditiveEnvelope` (compatibility name for the target
  `LogAmplitudeFactor` interface), required as
  `AdditiveEnvelope(ElectronElectronCusp, GaussianConfinement)` --- an
  `AdditiveCusp`-composed cusp term plus a decay/confinement term
  $ psi(bv) = exp(J (br))Psi(bv), quad J (br) = J_"ee" (br) + J_"conf" (br) $
+ Output: $psi(bv)$

#pagebreak()

#show: appendix
= Derivation of QMC Loss

Define

$
  Z_theta
  =
  integral
  psi_theta (br)^2 dif br
$

and the normalized Born distribution

$
  p_theta (br)
  =
  (|psi_theta (br)|^2) / Z_theta.
$

Then the variational energy can also be written as

$
  E (theta)
  =
  integral
  p_theta (br)
  E_"loc" ^ theta (br) dif br.
$

At first sight, differentiating this expectation appears to require
differentiating both $p_theta$ and $E_"loc" ^ theta$. However, Hermiticity of
$H$ allows the derivative to be written entirely in score-function form.

Define

$
  N_theta
  =
  integral
  psi_theta (br)
  H psi_theta (br)dif br.
$

Then

$
  E (theta)
  =
  N_theta / Z_theta,
$

and hence

$
  nabla_theta E (theta)
  =
  (
    nabla_theta N_theta
  ) / Z_theta
  -
  E (theta)
  (
    nabla_theta Z_theta
  ) / Z_theta.
$

For a real wavefunction and a Hermitian Hamiltonian,

$
  nabla_theta N_theta
  =
  2
  integral
  (
    nabla_theta psi_theta (br)
  )
  H psi_theta (br) dif br,
$

while

$
  nabla_theta Z_theta
  =
  2
  integral
  (
    nabla_theta psi_theta (br)
  )
  psi_theta (br) dif br.
$

Therefore,

$
  nabla_theta E (theta)
  =
  2 / Z_theta
  integral
  (
    nabla_theta psi_theta (br)
  )
  (
    H psi_theta (br)
    -
    E (theta) psi_theta (br)
  )br.
$

Using

$
  nabla_theta psi_theta (br)
  =
  psi_theta (br)
  nabla_theta log |psi_theta (br)|
$

and

$
  H psi_theta (R)
  =
  E_"loc" ^ theta (R)
  psi_theta (R),
$

we obtain

$
  nabla_theta E (theta)
  =
  2
  integral
  p_theta (br)
  (
    E_"loc" ^ theta (br)
    -
    E (theta)
  )
  nabla_theta log |psi_theta (br)|dif br.
$

Equivalently,

$
  nabla_theta E (theta)
  =
  2
  E_(br tilde p_theta)
  [
    (
      E_"loc" ^ theta (br)
      -
      E (theta)
    )
    nabla_theta log |psi_theta (br)|
  ].
$

This is the VMC score-gradient formula.

The implementation uses a scalar surrogate loss whose gradient is the
expression above. Let $"sg"(x)$ denote a stop-gradient operation. For samples

$
  br^a tilde p_theta,
  a = 1, dots, N,
$

define the batch energy

$
  E_"batch"
  =
  1 / N
  sum_(a = 1)^N
  E_"loc" ^ theta (br^a).
$

The QMC surrogate loss is

$
  L_"QMC" (theta)
  =
  2 / N
  sum_(a = 1)^N
  "sg"(
    E_"loc" ^ theta (br^a)
    -
    E_"batch"
  )
  log |psi_theta (br^a)|.
$

Because the centered local energies are detached,

$
  nabla_theta L_"QMC" (theta)
  =
  2 / N
  sum_(a = 1)^N
  (
    E_"loc" ^ theta (br^a)
    -
    E_"batch"
  )
  nabla_theta log |psi_theta (br^a)|.
$

Thus the value of $L_"QMC"$ is not itself the physical energy. Its gradient is
a Monte Carlo estimator of the variational-energy gradient.

A parameter-decoupled formulation makes the stop-gradient operation explicit.
Introduce a frozen parameter copy $alpha$ and a live parameter copy $theta$.
The frozen copy defines

$
  p_alpha (R)
  =
  (psi_alpha (R)^2) / Z_alpha,
$

$
  E_"loc" ^ alpha (R)
  =
  (H psi_alpha (R)) / (psi_alpha (R)),
$

and the centered local-energy signal

$
  A_alpha (R)
  =
  E_"loc" ^ alpha (R)
  -
  E (alpha).
$

Define

$
  L (theta; alpha)
  =
  2
  E_(br tilde p_alpha)
  [
    A_alpha (br)
    log |psi_theta (br)|
  ].
$

Differentiating only with respect to the live parameters gives

$
  nabla_theta L (theta; alpha)
  =
  2
  E_(br tilde p_alpha)
  [
    A_alpha (br)
    nabla_theta log |psi_theta (br)|
  ].
$

After differentiation, setting $theta = alpha$ gives

$
  nabla_theta L (theta; alpha)
  =
  nabla_alpha E (alpha)
$

at $theta = alpha$.

Thus VMC training can be viewed as follows:

- Freeze the current wavefunction.
- Sample configurations from its Born distribution.
- Compute detached centered local energies.
- Increase the log-amplitude of below-average-local-energy configurations.
- Decrease the log-amplitude of above-average-local-energy configurations.


= Connecting QMC Sampling and RL

== Bellman Expectation
The standard discounted Bellman equation for a fixed policy $pi$ is

$
  V^pi (s)
  =
  EE_(
    a tilde pi (dot | s),
    s' tilde P (dot | s, a)
  )
  [
    r (s, a)
    +
    gamma V^pi (s')
  ].
$

It decomposes the value of a state into an immediate reward and the discounted
value of the next state.

For VMC, the closer analogy is the average-cost Bellman equation. Consider a
Markov process with state $s$, transition kernel $P (s' | s)$, per-state cost
$c (s)$, and stationary average cost $rho$. Its relative value function $h$
satisfies the Poisson equation

$
  h (s)
  =
  c (s)
  -
  rho
  +
  EE_(s' tilde P (. | s))
  [
    h (s')
  ].
$

The corresponding one-step Bellman residual is

$
  delta (s, s')
  =
  c (s)
  -
  rho
  +
  h (s')
  -
  h (s).
$

In VMC, the MCMC sampler may be viewed as an artificial pseudo-dynamical
system. Its state is an electron configuration $R_t$, and its transition
kernel is

$
  br^(t + 1)
  tilde
  T_theta ( dot |br^t).
$

The transition kernel is constructed so that its stationary distribution is

$
  p_theta (br)
  prop
  |psi_theta (br)|^2.
$

The analogue of the instantaneous cost is the local energy,

$
  c_theta (br)
  =
  E_"loc" ^ theta (br),
$

and the stationary average cost is the variational energy,

$
  rho_theta
  =
  EE_(R tilde p_theta)
  [
    E_"loc" ^ theta (br)
  ]
  =
  E (theta).
$

The average-cost Bellman equation for the sampler pseudo-dynamics is therefore

$
  h_theta (br)
  =
  E_"loc" ^ theta (br)
  -
  E (theta)
  +
  E_(R' tilde T_theta (dot | br))
  [
    h_theta (br')
  ].
$

Its one-step Bellman residual is

$
  delta_theta (br, br')
  =
  E_"loc" ^ theta (br)
  -
  E (theta)
  +
  h_theta (br')
  -
  h_theta (br).
$

The centered local energy used by VMC is

$
  A_"QMC" (br)
  =
  E_"loc" ^ theta (br)
  -
  E (theta).
$

It can therefore be interpreted as the average-cost Bellman residual under
the approximation

$
  h_theta (br)
  approx
  0.
$

Under this approximation,

$
  delta_theta (br, br')
  approx
  E_"loc" ^ theta (br)
  -
  E (theta).
$

The QMC surrogate loss consequently has the same score-weighted form as a
policy-gradient loss:

$
  L_"QMC" (theta)
  =
  2
  E_(br tilde p_theta)
  [
    "sg" (
      A_"QMC" (br)
    )
    log |psi_theta (br)|
  ].
$

For comparison, a policy-gradient estimator has the form

$
  nabla_theta J (theta)
  =
  E
  [
    A^pi (s, a)
    nabla_theta log pi_theta (a | s)
  ],
$

whereas the VMC energy gradient is

$
  nabla_theta E (theta)
  =
  2
  E_(R tilde p_theta)
  [
    (
      E_"loc" ^ theta (R)
      -
      E (theta)
    )
    nabla_theta log |psi_theta (R)|
  ].
$

The correspondence is:

- The policy distribution $pi_theta (a | s)$ corresponds to the Born
  distribution $p_theta (br)$.
- An action corresponds to a full electron configuration $br$.
- The reward advantage corresponds to the negative centered local energy.
- The policy log-probability corresponds to
  $2 log |psi_theta (br)|$.
- Reward maximization corresponds to energy minimization.

The factor of two follows from

$
  log p_theta (br)
  =
  2 log |psi_theta (br)|
  -
  log Z_theta.
$

The normalization term does not contribute after centering because

$
  EE_(br tilde p_theta)
  [
    E_"loc" ^ theta (br)
    -
    E (theta)
  ]
  =
  0.
$

There is nevertheless an important distinction between ordinary RL and VMC.
In ordinary RL, the environment transition kernel is usually external and
independent of the policy parameters. In VMC, the sampler transition kernel
$T_theta$ is artificial and generally depends on $theta$, because its
acceptance probabilities depend on $|psi_theta|^2$.

Standard VMC does not differentiate through these sampler transitions.
Instead, the sampler is treated as an on-policy data generator for the
stationary Born distribution. The optimization target remains the stationary
Rayleigh quotient, not the energy obtained after a finite number of MCMC
steps.

The most precise interpretation is therefore:

$
  "VMC is stationary policy gradient on the Born distribution."
$

The MCMC sampler supplies correlated on-policy samples, while

$
  E_"loc" ^ theta (br)
  -
  E (theta)
$

acts as the cost advantage. The omitted Bellman term

$
  h_theta (br')
  -
  h_theta (br)
$

belongs to the pseudo-dynamics of the sampler. It describes finite-chain
transients and autocorrelation rather than the underlying quantum dynamics.
Such a term may be useful as a sampler-dependent control variate, but it is
not part of the standard VMC loss.
