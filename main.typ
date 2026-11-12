// TPEN Design Document (Typst). Source of truth for the TPEN architecture;
// replaces the retired SpENN (Specht-module / irrep / Fourier) design doc
// per decision O2 / D15a.
#import "@preview/noteworthy:0.4.0": * 
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
  title: "TPEN Proposal",
  author: "Richard Hu",
  contact-details: "TPEN-QMC",
  toc-title: none,
  toc-depth: 2,
  watermark: "DRAFT", // Optional: Watermark for the document
)

#show ref: theoretic.show-ref

// #set cite(style: "chicago-notes")
//#show ref: footnote

#set heading(numbering: "1.")

#let appendix(body) = {
  set heading(numbering: "A1.1a", supplement: [Appendix], outlined: true)
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

#show figure.where(kind: table): set block(breakable: true)


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

#let bg = $bold("g")$
#let bN = $bold("N")$
#let bL = $bold("L")$
#let bG = $bold("G")$
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


= Background

Representing antisymmetric functions is a central problem
in neural-network-based variational Monte Carlo (VMC).
A fermionic wavefunction must satisfy
$
Psi(bv_(sigma(1)), dots, bv_(sigma(N)))
= "sgn"(sigma) Psi(bv_1, dots, bv_N)
$
for every permutation $sigma in S_N$.

The challenge is not merely to enforce this symmetry,
but to construct a sufficiently expressive and computationally
tractable approximation to the ground-state wavefunction.
In VMC, the quality of the ansatz is determined by the
variational energy, which depends on the wavefunction and
its spatial derivatives. Consequently, universal
approximation of continuous antisymmetric functions
does not by itself guarantee accurate local energies.

== Determinant-based neural wavefunctions

The traditional approach to fermionic antisymmetry
uses Slater determinants, often combined with Jastrow
factors to account for electron correlation. A single
Slater determinant describes an antisymmetrized product
of one-particle orbitals, while linear combinations of
determinants provide increasingly flexible many-electron
wavefunctions.

Such expansions form the basis of configuration
interaction methods, but the number of determinants
required for a general correlated state can grow
combinatorially with system size
@PhysRevResearch.2.033429.

Neural wavefunctions such as FermiNet
@PhysRevResearch.2.033429 and PsiFormer
@vonglehn2023selfattentionansatzabinitioquantum
extend this approach by allowing each orbital
to depend equivariantly on the entire electronic
configuration. A representative ansatz is
$
Psi_theta (bv)
= e^(J(bv))
  sum_(d=1)^D c_d det(F_theta^d (bv)),
$
where the orbital matrices $F_theta^d$ are generated
by a permutation-equivariant neural network.

The expressivity of generalized Slater determinants
has been investigated theoretically. Hutter
@hutter2020representing showed that a single
generalized determinant suffices to represent
arbitrary antisymmetric functions. However, for
spatial dimensions $d > 1$, the construction relies
on discontinuous equivariant orbital functions @ye2024continuousantisymmetric,
establishing $L^p$ approximation rather than
uniform approximation of continuous functions.

This limitation was subsequently addressed by
Ye et al. @ye2024continuousantisymmetric, who
proved that every continuous antisymmetric
function on $(RR^d)^N$ admits an exact representation
as a sum of at most $d N + 1$ generalized Slater
determinants with continuous equivariant orbitals.

Thus, the continuity obstruction does not imply
a fundamental incompleteness of determinant-based
ansätze. Nevertheless, universality alone does not
establish efficient approximation, favorable
optimization properties, or preservation of higher
regularity. These questions remain relevant
when designing neural wavefunctions for VMC.

This substantially increases the expressive power
of determinant-based constructions. Indeed, generalized
Slater determinants admit universality arguments,
even with a single determinant
@PhysRevResearch.2.033429. However, such theoretical
constructions need not yield continuous orbital maps
or efficient finite-dimensional representations.

Thus, the relevant question is not simply whether
Slater-based ansätze can represent an arbitrary
antisymmetric function, but whether they provide
an efficient and physically appropriate inductive
bias for correlated many-electron systems.

In particular, Slater determinants primarily encode
antisymmetry through one-particle orbital assignments.
Although neural backflow allows these orbitals to
depend on all electrons, the final antisymmetrization
still proceeds through a determinant.

This structure need not be optimal for strongly
paired or multireference states. Pfaffian pairing
wavefunctions have long demonstrated improvements
over Hartree--Fock nodal structures
@bajdich2006pfaffian, while more recent work
shows that replacing determinant antisymmetrization
with a pairing-based construction can substantially
improve neural wavefunctions for superfluid systems
@lou2024neuralwavefunctions.

== Alternative antisymmetric constructions

These considerations motivate alternative ways of
representing fermionic antisymmetry.

Neural Pfaffians
@gao2024neuralpfaffianssolvingmanyelectron
use learned pairing matrices and overparameterized
orbitals to construct antisymmetric kernels of the form
$
K = F A F^T, quad A^T = -A,
$
whose Pfaffians define the wavefunction. This
generalizes determinant-based constructions and
allows orbital dimensions to vary with the molecular
system.

Fermi Sets
@fu2026fermisetsuniversalinterpretable
take a different approach by decomposing the
wavefunction into symmetric and antisymmetric factors:
$
Psi(bv) = sum_(k=1)^K phi_k (bv) eta_k (bv),
$
where $phi_k$ is permutation-invariant and
$eta_k$ is an antisymmetric signature function.

Building on an exact continuous representation
@fu2025minimaluniversalrepresentationfermionic,
Fermi Sets establish universal approximation for
continuous antisymmetric functions using suitable
signature encoders and symmetric networks.
Corresponding differentiability guarantees require
additional consideration.

This construction transfers much of the approximation
problem to permutation-invariant functions. However,
the existence of such a representation does not
automatically establish the computational efficiency
of evaluating the chosen signatures or learning
their symmetric coefficients.

More broadly, these architectures demonstrate that
antisymmetry can be implemented through several
algebraically distinct constructions. Their relative
expressivity, computational complexity, and inductive
bias therefore deserve separate consideration.

== Tensor-product Permutation Equivariant Networks

Tensor-product Permutation Equivariant Networks (TPEN)
approach this problem from the representation of
many-body correlations rather than from a prescribed
antisymmetrization mechanism.

Instead of storing only one-electron embeddings, TPEN
maintains equivariant features indexed by ordered
tuples of distinct particles:
$
x_I^(c m),
quad I = (i_1,dots,i_m),
quad |I| = m,
$
where $m$ denotes particle-feature order and $c$
denotes the channel index.

Interactions between these features are expressed
through equivariant tensor-product contractions.
Their outputs may be retained at the same particle
order, projected to lower orders, or used to construct
higher-order features.

The central idea is to model many-body correlations
explicitly while preserving permutation equivariance
at every layer. Antisymmetry is imposed only at the
readout stage, allowing the intermediate representations
to retain general permutation-equivariant information.

This design separates three choices:
- the maximum particle-feature order;
- the tensor-product interactions between features;
- the final antisymmetric readout.

We show that several major neural-wavefunction
architectures arise as specializations of TPEN,
including determinant-, Pfaffian-, and Fermi
Sets-based constructions.

The same tensor-product formalism also encompasses
higher-order graph message passing, motivating a
closer comparison with graph neural networks.

= Limitations of Graph Neural Networks

== Message Passing

Graph neural networks (GNNs) provide a natural framework
for functions that must be invariant or equivariant under
relabeling of their inputs. This is closely related to
the permutation symmetry encountered in fermionic
wavefunction modeling.

A typical message-passing GNN updates node features by
aggregating information from their neighborhoods:
$
bm_(i j)^ell
  &= f^ell (bx_i^ell, bx_j^ell, e_(i j)^ell), \
bx_i^(ell+1)
  &= u^ell (
    bx_i^ell,
    sum_(j in cal(N)(i)) bm_(i j)^ell
  ).
$

The same message and update functions are applied
at every node. Since neighborhood aggregation is
permutation-invariant, the resulting network is
equivariant to graph relabeling.

This mechanism is powerful, but the symmetry guarantee
alone says little about expressive completeness.

In particular, conventional message-passing GNNs
are bounded in graph-distinguishing power by the
one-dimensional Weisfeiler--Leman (1-WL) test
@morris2021weisfeilerlemanneuralhigherorder
@xu2019howpowerful.

For example, consider the cycle $C_6$ and the
disjoint union $C_3 union C_3$. Both have six
vertices, each of degree two. Starting from
identical node features, standard message passing
produces identical node representations in both
graphs at every depth, despite their different
triangle structures.

This illustrates a structural limitation that
cannot be resolved merely by stacking more layers
of the same message-passing architecture.

These graph-distinguishability results do not
directly imply a corresponding impossibility
theorem for neural wavefunctions with continuous
particle coordinates. Nevertheless, they motivate
the use of richer equivariant representations.

== Higher-order graph features

One approach is to associate features with collections
of multiple vertices rather than individual nodes.

The $k$-GNN architecture of
@morris2021weisfeilerlemanneuralhigherorder
uses features $bh_S$ indexed by unordered
$k$-element vertex subsets $S$.

Its message-passing neighborhoods consist of subsets
that differ by the replacement of one vertex:
$
cal(N)_k(S)
= {T : |T| = k, |S inter T| = k-1}.
$

A representative update is
$
bh_S^(ell+1)
= sigma(
  W_0 bh_S^ell
  + W_1 sum_(T in cal(N)_k(S)) bh_T^ell
).
$

Compared with ordinary node message passing,
this allows the network to maintain and propagate
information about higher-order graph structures.

The resulting hierarchy has stronger
graph-distinguishing capabilities than conventional
message-passing GNNs. However, this result should
not be confused with universal approximation of
arbitrary equivariant functions.

More generally, universality results for invariant
networks show that higher-order tensors can be
necessary for certain symmetry groups and classes
of equivariant architectures
@maron2019universality.

The cost of higher-order representations is
substantial. A set-based $k$-GNN stores
$
binom(N,k) = O(N^k)
$
features, and direct replacement-neighborhood
aggregation can require
$
O(N^(k+1))
$
operations per layer.

Thus, the maximum feature order provides a natural
tradeoff between expressive capacity and
computational cost.

== Direct interactions between higher-order features

Higher-order features alone do not determine how
information is exchanged between them.

In the standard $k$-GNN update, neighboring subset
features are linearly aggregated before applying
the update nonlinearity. This does not explicitly
construct products between the features of
different neighboring subsets.

For example, consider an order-two interaction
$
y_(i j) = sum_(k != i,j) x_(i k) x_(k j).
$

This expression couples two pair features through
their shared particle $k$. It represents a
three-particle interaction whose result is stored
as an order-two feature.

Such an operation is not a primitive of standard
linear neighborhood aggregation. Its effect must
instead be represented through additional
transformations or a modified message-passing rule.

TPEN directly implements this operation through
a tensor-product path
$
(2,2) arrow.r 2
$
with virtual interaction order $s=3$.

More generally, TPEN permits bilinear contractions
between features of different body orders, with
arbitrary shared indices and subsequent reductions.

The distinction is therefore not simply that TPEN
supports higher-order features. It also provides
a systematic family of interactions between them,
including operations that do not arise explicitly
in conventional $k$-GNN message passing.

For maximum stored feature order $M$, TPEN requires
$O(N^M)$ feature storage at fixed channel width.
A direct contraction with virtual interaction order
$s$ requires at most $O(N^s)$ operations, excluding
channel-dependent costs.

The separation between stored order $M$ and virtual
interaction order $s$ allows TPEN to construct
higher-order interactions while retaining their
outputs in lower-order representations.

In the following sections, we investigate the
expressive consequences of this design by
constructing explicit specializations of existing
neural-wavefunction and higher-order GNN
architectures.


= Network Design

TPEN provides a general framework to think about interactions between nodes on a graph.

#block(width: 100%)[
  #set math.equation(numbering: none)
  $
  bv_i stretch(->)^"embedding"_phi^(c m)
  underbracket(
  bx^m_I stretch(->)^"mixing"_(W) by^m_(I, p)
  stretch(->)^"activation"_Gamma bh^m_(I,p)
  stretch(->)^"aggregation"_U bw^m_I
  stretch(->)^"activation"_(Gamma_c) bu_I
  stretch(->)^("update") bx_I^m,
  arrow.cw
  )
  $ <tpen-layers>
  The path axis $p$ is opened by mixing and
  contracted by aggregation, so the post-aggregation update $bu_I$ carries no path
  axis.
]

Implemented in `tpen.nn.TPENWaveFunction`. More detailed design discussion in
@sec.model-design.

+ Input: particle positions $bv_i = (br_i, s_i)$ (Optional particle-wise basis or orbital)
+ Embedding (`tpen.nn.Embedding`), learnable: $phi^(m): bv_I mapsto bx_I^(0, c, m)$ (Optional embedding normalization/envelope)
+ TPEN Stack (`tpen.nn.TPENStack`)
  + TPEN layer 1 (`tpen.nn.TPENLayer`)
    + mixing in real space (`tpen.nn.EquivariantMixing`)
      $ by^(1, c)_(I, p) = sum_(J_([s]\\"im"(tau))) W_p^(1, c<-c_1c_2)
      bx_(J circle.small tau_1)^(0, c_1) bx_(J circle.small tau_2)^(0, c_2) $
      where path index $p = (s, m, m_1, m_2, tau, tau_1, tau_2)$. This is
      followed by the point-wise path activation: $bh^(1, c)_(I, p) = Gamma^(1, c)_p (by_I) $
    + Obtain update with `tpen.nn.PathAggregation`
      $ bw^(1, c_"out")_(I) = sum_p U^(1, c_"out" <- c_"in")_(p) bh^(1, c_"in")_(I, p) $
      followed by point-wise path activation: $bu^(1, c)_(I) = Gamma^(1, c) (bw_I) $
    + Feature update (`tpen.nn.update`): $bx^1 = "Update"(bx^0, bu^1)$
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
  $ Psi(bv) = sum_(c) w^(c) "Pf"[bx^(T c)_(i j) - bx^(T c)_(j i)] $
+ Apply Jastrow factor
  $ psi(bv) = exp(J (br))Psi(bv), quad J (br) = J_"ee" (br) + J_"en" (br; bR) $
+ Output: $psi(bv)$


== Inputs and Embedding

The first layer of features is created by embedding real-space coordinates:
$ bx^(0, c, m)_(i_1, dots.c, i_m) = phi^(c m) (bv_i_1, dots.c, bv_i_m) $
where $c$ is the channel index.
We will suppress $m$ unless otherwise stated, since it
can be directly derived from $I = (i_1, dots.c, i_m)$. $bv$ is a real-space
configuration of the electrons. We can choose it to be $bv_i = (br_i, s_i) = (x_i, y_i, z_i, s_i)$
where $s_i in {1, -1}$.




== Equivariant Mixing
The core component of TPEN is a tensor-product that takes features $bx^(t c)_I$ and
produces equivariant paths
$bh^((t+1)c)_(I, p)$. A path is indexed by its family and its typed support
metadata. This information may be passed on to features of equal or lower
body-order than the interaction.

Mixing has the general form (before the common activation):
$ by^c_(I, p) = sum_(J_([s]\\"im"(tau))) W_p^(c<-c_1c_2) bx_(J circle.small tau_1)^c_1 bx_(J circle.small tau_2)^c_2 $
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


== Aggregation

Aggregation contracts the path axis per input channel and then applies an owned
activation $Gamma_c$ across the channel axis:

$ bw^(1, c_"out")_(I) = sum_p U^(1, c_"out" <- c_"in")_(p) bh^(1, c_"in")_(I, p), $

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


== Pfaffian Readout

The most obvious readout method is using pfaffians on the (1,1) irrep.
$ psi(br) = sum_(c=1)^C b_c "Pf"[(hat(bx)^(c))_ij] $
For the cases where $n$ is odd, we can instead do
$ psi(br) = sum_(c=1)^C b_c "Pf" mat(
  hat(bx)^(c, 2) - (hat(bx)^(c, 2))^T, -hat(bx)^(c, 1);
  (hat(bx)^(c, 2))^T, 0
) $
This is because the tensor product of $n\/2$ $(1,1)$ irreps contain a copy of the $(1^n)$ irrep.

Assuming that channels are sufficiently mixed in the TPEN layers, this is the only
irrep with order $n$ that we can read-out from a network of maximum interaction order $M=2$.

= Relationship to existing neural-QMC architectures and GNNs

== FermiNet

Fermi Net@PhysRevResearch.2.033429 is equivalent to TPEN with order-1 and order-2 features.

#proposition(label: <prop.ferminet>)[
For fixed electron number $n$ and spin sector, the standard real-valued FermiNet architecture is a specialization of TPEN with maximum body order $M = 2$, provided the embedding admits the required analytic features and orbital envelopes.
]


Alternatively, the initial particle features may be supplied directly in an orbital basis:
$
bv_i = (varphi_1(br_i), dots, varphi_ell (br_i), s_i),
$
where $ell$ is the highest retained orbital index/order.

The same TPEN construction applies to this embedding. Exact containment of a coordinate-input FermiNet additionally requires that the orbital descriptors retain sufficient information to reproduce its geometric features.


== PsiFormer

PsiFormer @vonglehn2023selfattentionansatzabinitioquantum
uses self-attention to update its order-one features. This can be
reproduced by TPEN using tensor-product paths without an explicit
attention component.

#proposition(label: <prop.psiformer>)[
PsiFormer's self-attention mechanism is a specialization of TPEN
with maximum body order $M = 2$, assuming arbitrary pointwise activations.
]


The diagonal attention terms are constructed separately using
order-one paths. The remaining MLP, residual, and optional
LayerNorm operations follow from local channel operations.
PsiFormer's electron-electron Jastrow factor agrees with
TPEN's rational cusp up to a constant for positive range
parameters. Its orbital envelopes and determinant readout
follow the FermiNet specialization.



== DeepErwin

DeepErwin@NEURIPS2023_4b5721f7 is a transferrable VMC ansatz that 
can adapt to changes in the nuclear environment. 
It has SchNet-style interactions, local coordinate frames,
and geometry-dependent orbital features. 

TPEN is currently
not designed to be transferrable. However, 
TPEN permits arbitrary particle descriptors,
including nuclear coordinates and species labels. Alternatively,
electrons and nuclei may be represented as typed particles, in
which case electron-nucleus interactions correspond directly to
type-selective $(2,1) arrow.r 1$ paths.

#proposition(label: <prop.deeperwin>)[
For a family of molecular systems with bounded particle counts,
the standard real-valued DeepErwin architecture is a specialization
of TPEN with maximum body order $M = 2$, assuming the required
geometric and reference-orbital information is available through
the embedding. The specialization preserves shared network weights
across molecular systems, with sufficient fixed channel capacity.
]


== LapNet

LapNet@li2023forwardlaplaciannewcomputational extends PsiFormer
with separate local and dense one-body feature streams,
enabling sparse Laplacian computation.

#proposition(label: <prop.lapnet>)[
The standard real-valued LapNet wavefunction is a specialization
of TPEN with maximum body order $M = 2$, under the same embedding
and activation assumptions as PsiFormer.
]


*Computational limitation.* The specialization establishes
equality of wavefunctions and their mathematical derivatives,
but not equality of computational complexity. LapNet exploits
sparse derivative structure through specialized forward-Laplacian
propagation, whereas a generic TPEN implementation may compute
dense derivatives and incur substantially greater time and
memory costs. Recovering LapNet's computational advantage
requires corresponding sparse differentiation algorithms,
which are not implied by architectural containment.


== OneQMC/Orbformer

OrbFormer@Foster_2026 is a chemically
transferable neural wavefunction architecture that generates
molecular orbitals dynamically from nuclear configurations.

#proposition(label: <prop.orbformer>)[
Suppose TPEN admits typed auxiliary orbital indices,
electron-restricted antisymmetric readout, and the invariant
normalization operations used by OrbFormer. Then the standard
real-valued OrbFormer ansatz is a specialization of TPEN with
maximum feature order $M = 2$.
]


*Locality and size consistency.* OrbFormer incorporates
distance-dependent interactions and localized orbital
constructions to promote composability across separated
molecular fragments. The specialization reproduces these
operations and consequently inherits their locality and
composability properties. TPEN itself imposes no such
restriction: its general tensor-product paths permit
interactions between arbitrarily distant particles.

In the noninteracting limit, composability implies that
the orbital matrix becomes block diagonal and the Jastrow
factor becomes additive. For a single determinant,
$
Psi_(A union B) = Psi_A Psi_B,
$
up to fermionic ordering, yielding size-consistent energies.

For multiple determinants, however,
$
sum_d det(F_A^d) det(F_B^d)
neq
(sum_d det(F_A^d)) (sum_e det(F_B^e)).
$
Hence composability alone does not guarantee size consistency
for the multideterminant ansatz. This limitation applies
equally to the corresponding TPEN specialization.

== Neural Pfaffian

Neural Pfaffians@gao2024neuralpfaffianssolvingmanyelectron introduce an
overparameterized Pfaffian ansatz with geometry-dependent
orbital projections and pairing matrices.

#proposition(label: <prop.neural-pfaffian>)[
Assume nuclear node and pair features are admissible as
typed auxiliary TPEN features, with electron-restricted
antisymmetric readout. For a fixed set of supported nuclear
species and orbital slots per species, the real-valued
Neural Pfaffian ansatz is a specialization of TPEN with
maximum body order $M = 2$.
]


*Computational limitation.* Neural Pfaffians use factorized
envelopes and specialized Pfaffian differentiation.
The specialization establishes functional equivalence,
not equivalent runtime or memory complexity. Generic
TPEN contractions may be less efficient without
corresponding computational optimizations.

*Pretraining.* Neural Pfaffians additionally use
Hartree-Fock-based orbital and geminal matching to
initialize their parameters. These are optimization
procedures rather than restrictions on the forward
wavefunction, and are not required for specialization.

== Fermi Sets


Fermi Sets@fu2026fermisetsuniversalinterpretable
represent fermionic wavefunctions as
$
Psi(bv) = sum_(k=1)^K phi_k (bv) eta_k (bv),
$
where $phi_k$ is permutation-invariant and $eta_k$
is antisymmetric.

The underlying parity-graded formulation admits an
exact continuous representation@fu2025minimaluniversalrepresentationfermionic,
although corresponding regularity guarantees for
differentiability remain to be established.

We establish exact containment for the principal
signature constructions, followed by a constructive
result for arbitrary finitely computable readouts.


#figure(
  table(
    columns: (2.3fr, 0.6fr, 1.8fr, 1.5fr, 0.8fr),
    inset: 4pt,
    align: left,
    table.header(
      [*Fermi Sets construction*],
      [*$M$*],
      [*Readout*],
      [*Time complexity*],
      [*Status*],
    ),
    [Deep Sets],
    [$2$],
    [Invariant pooling],
    [$O(N)$],
    [Exact],

    [Slater signatures],
    [$2$],
    [Pfaffian],
    [$O(N^3)$],
    [Exact],

    [Jordan--Wigner signatures],
    [$2$],
    [Pfaffian or sign product],
    [$O(N^3)$ or $O(N^2)$],
    [Exact],

    [General pair-product signatures],
    [$2$],
    [Complete-graph generalized readout],
    [$O(N^2)$],
    [Exact],

    [Arbitrary finitely computable antisymmetric signatures],
    [$N$],
    [Top-order antisymmetrization],
    [$O(N! T_C)$],
    [Exact],

    [Arbitrary continuous antisymmetric functions],
    [$N$],
    [Top-order antisymmetrization],
    [Approximation-dependent],
    [Uniform approximation],
  ),
  caption: [Summary of Fermi Sets specialization results.
    Complexities exclude feature generation unless explicitly
    included. $T_C$ denotes the cost of evaluating the
    underlying computational circuit. Exact containment
    assumes the required operations are TPEN-admissible.],
)


=== Specializations

#proposition(label: <prop.fermi-sets-slater>, title: [TPEN Specialization into Slater-based Fermi Sets])[
For representable generalized orbitals and
Deep Sets symmetric factors, Slater-based
Fermi Sets are exact specializations of TPEN
with maximum order $M=2$.
]


#proposition(label: <prop.fermi-sets-jordan-wigner>, title: [TPEN Specialization into Jordan--Wigner signatures])[
On distinct, totally ordered particle labels,
the Jordan--Wigner signature is exactly
representable by TPEN with $M=2$.
]


#proposition(label: <prop.fermi-sets-pair-product>, title: [TPEN Specialization into General pair-product signatures])[
Let $f(bv_i,bv_j)=-f(bv_j,bv_i)$ be any
representable antisymmetric pair function.

The signature
$
eta(bv) = product_(i<j) f(bv_i,bv_j)
$
is exactly representable using TPEN order-two
features and a generalized higher-order readout.

The readout can be evaluated in
$O(N^2)$ time after pair features have been
computed, independently of the separation
rank of $f$.
]


#proposition(label: <prop.fermi-sets-general>, title: [Arbitrary finitely computable Fermi Sets])[
Fix $N$. Suppose each symmetric factor $phi_k$
and antisymmetric signature $eta_k$ is specified
by a finite computational circuit using
TPEN-admissible operations.

Then
$
Psi(bv)
= sum_(k=1)^K phi_k (bv) eta_k (bv)
$
admits an exact, finite-depth TPEN realization
with maximum feature order $M=N$.

If the complete Fermi Sets evaluation circuit
has cost $T_Psi$, the direct realization has
an $O(N! T_Psi)$ time upper bound.
]


=== Approximation and Complexity

The arbitrary-readout result concerns finite
circuits. It does not assert exact finite-network
representation of every continuous function.

Nevertheless, on a compact permutation-invariant
domain, let $Psi$ be any continuous antisymmetric
function. Choose a polynomial $P$ satisfying
$
norm(P-Psi)_infinity < epsilon.
$

Because antisymmetrization is a contraction
in the uniform norm,
$
norm(cal(A)_N P-Psi)_infinity

< epsilon.
$

Polynomial circuits are admissible in TPEN, so
the circuit-compilation lemma yields a constructive
uniform-approximation result with $M=N$.

This does not establish approximation of
derivatives or local energies.

The complexity bounds distinguish general
and structured readouts:

- Arbitrary finite antisymmetric circuits:
  factorial-time upper bound $O(N! T_C)$.
- Slater readouts: cubic-time determinant or
  Pfaffian evaluation, excluding orbital generation.
- Jordan--Wigner signatures: quadratic-time
  pair construction, with cubic-time generic
  Pfaffian evaluation.
- Complete-graph pair products: quadratic-time
  evaluation after pair features are available.

The polynomial-time results rely on specific
algebraic contraction patterns. They do not
imply efficient evaluation of generalized
higher-order readouts in general.


== $k$-GNNs


The $k$-GNN architecture@morris2021weisfeilerlemanneuralhigherorder of
performs message passing between $k$-element
subsets of graph vertices. We consider the
set-based architecture without its graph-level
readout.

=== Specialization
#proposition(label: <prop.k-gnn-set>, title: [Set-based $k$-GNN containment])[
Assume graph adjacency and the required
initial subset features are available
through TPEN's embedding.

Every finite-depth, sum-aggregating
set-based $k$-GNN is exactly reproducible
by TPEN, without graph-level readout.

The construction uses feature order
at most $k$ and interaction order
at most $k+1$.
]


#proposition(label: <prop.k-gnn-hierarchical>, title: [Hierarchical $k$-GNN containment])[
The hierarchical $1$-$k$-GNN
architecture is also reproducible
by TPEN using feature orders at most
$k$ and interaction orders at most
$k+1$.
]



=== Computational complexity

A set-based $k$-GNN stores
$binom(N,k) = O(N^k)$
subset features. Its complete replacement neighborhood
contains
$
binom(N,k) k(N-k) = O(N^(k+1))
$
directed messages, giving $O(N^(k+1))$ time
per layer under direct message evaluation.

For local message passing on an undirected graph,
the number of directed messages is
$
2 |E| binom(N-2,k-1),
$
giving $O(|E| N^(k-1))$ time for fixed $k$.

A dense TPEN realization stores
$
frac(N!, (N-k)!) = k! binom(N,k) = O(N^k)
$
ordered-tuple features.

The replacement aggregation uses virtual interaction
order $s=k+1$, giving $O(N^(k+1))$ time under
direct TPEN contraction evaluation. Thus both
architectures have the same asymptotic feature
and computational complexity for fixed $k$.

The factor $k!$ arises from representing unordered
subsets as ordered tuples and can be eliminated by
exploiting the internal $S_k$ symmetry.

Specialized factorizations may reduce the cost of
particular contractions, but are not assumed in
this comparison.


=== Scope
This specialization concerns
the set-based $k$-GNN of Morris et al.,
not the unrestricted $k$-WL algorithm
on all ordered tuples, including tuples
with repeated vertex indices.


= Research Plans

== Loss function

The loss function that we will be using is the following:
$ cal(L)(theta) = 2 EE_(br ~ |psi|^2)[(E_"loc" (br) - E)_"detach" log psi_theta (br)] $
It will be approximated by the LLN estimator
$ cal(L)(theta) = 2/N sum_(a=1)^N [(E_"loc" (br^a) - E)_"detach" log psi_theta (br^a)] $
where $br^a$ are iid samples of $|psi|^2$.

The gradient of this loss function $nabla_theta cal(L)$ is equivalent to $nabla_theta E$.

== Research questions and testable hypotheses

#text(fill: red)[IN PROGRESS]

Hypotheses:
+ H1: Bilinear mixing is better than linear mixing ($k$-GNN type)\
  test: replace with $k$-GNN-type mixing
+ H2: Higher order features improve accuracy\
  test: using higher interaction order of $M=2, 3, "or", 4$\
  Note: this is a good way to gain confidence that TPEN is better than other models.
  We focus on high-correlation systems.
+ H3: Virtual interaction orders provide sufficient compromise
  test: $M = 2, s_"max" = 3$ compared to $M=3$ and $M = 2, s_"max" = 2$
+ H4: TPEN outperforms other neural-QMC models on high-correlation systems
+ H5: TPEN is more efficient than other neural-QMC models
  

== Experimental design and baseline comparisons

#text(fill: red)[IN PROGRESS]


We organize the benchmark systems into three stages:
implementation controls, architectural characterization,
and competitive evaluation against established neural-QMC
ansätze. The progression increases the number of electrons
and the complexity of the physical system separately,
allowing architectural effects to be distinguished from
implementation and optimization difficulties.

#figure(
  table(
    columns: (1.5fr, 0.4fr, 1.6fr, 2.5fr),
    inset: 5pt,
    align: left,
    table.header(
      [*System*],
      [*$N$*],
      [*Reference*],
      [*Purpose*],
    ),

    table.cell(colspan: 4)[*I. Implementation controls*],
    [Hydrogen atom], [$1$],
    [Exact: $-0.5$ Ha],
    [Smooth out problems with cusps, regularization, and normalization.],
    
    [Hooke atom, $omega=0.5$], [$2$],
    [Exact: $2$ Ha],
    [Verify that electron-electron interactions behave properly.],

    [Helium atom], [$2$],
    [High precision:
    $-2.903724377$ Ha],
    [Validate all-electron Coulomb interactions,
    electron-nucleus cusps, and sampling.],

    [$H_2$, $R=1.4$ bohr], [$2$],
    [Established:
    $-1.17447$ Ha],
    [Validate multi-center nuclear geometry and
    molecular wavefunctions.],

    table.cell(colspan: 4)[*II. Architectural characterization*],

    [Three-electron harmonium], [$3$],
    [Sub-microhartree published benchmarks],
    [Test odd-electron Pfaffian readout,
    virtual interaction order $s=3$,
    and explicit three-body features.],

    [Four-electron harmonium], [$4$],
    [Few-microhartree published benchmarks],
    [Compare feature orders $M=2,3,4$,
    investigate stronger correlations
    and different spin states.],

    [Lithium atom], [$3$],
    [High precision:
    $-7.47806032$ Ha],
    [Transfer the conclusions from harmonium
    to a three-electron Coulomb atom.],

    [Beryllium atom], [$4$],
    [Estimated exact:
    $-14.66736$ Ha],
    [Evaluate higher-order correlations
    in a four-electron atomic system.],

    table.cell(colspan: 4)[*III. Competitive and strongly correlated benchmarks*],

    [LiH], [$4$],
    [$-8.070548$ Ha at
    $R=3.015$ bohr],
    [Test heteronuclear molecular systems
    against established QMC baselines.],

    [$H_2$ dissociation], [$2$],
    [High-accuracy references
    at selected geometries],
    [Test the description of changing
    multireference character during bond stretching.],

    [$H_4$ chain], [$4$],
    [Geometry-matched continuum
    references required],
    [Test higher-order interactions under
    strong correlation and bond stretching.],

    [$N_2$ dissociation], [$14$],
    [Published benchmarks],
    [Long-term comparison against mature
    neural-QMC methods on a difficult molecule.],
  ),
  caption: [Proposed TPEN benchmark ladder.
    Reference energies are in Hartree. Values for the
    existing atomic and molecular systems are taken
    from the project's baseline registry; references
    for additional systems require registration and
    geometry matching.],
) <tab.benchmark-ladder>

=== Reference data

The Hooke atom, helium, hydrogen molecule, lithium,
beryllium, lithium hydride, and nitrogen molecule
already have reference entries in
`experiments/baselines/systems.yaml`.

High-accuracy calculations for three- and four-electron
harmonium are available in the literature:

- #link("https://doi.org/10.1063/1.4717461")[
  Three-electron harmonium benchmark calculations
  (Cioslowski et al., 2012)].
- #link("https://doi.org/10.1063/1.4891301")[
  Four-electron harmonium benchmark calculations
  (Cioslowski et al., 2014)].

The harmonium references cover multiple confinement
strengths and spin states. Their numerical values must
be registered together with the corresponding Hamiltonian
parameters and spin sectors before comparison.

For molecular benchmarks, the nuclear geometry must
agree with the reference calculation. Estimated-exact
reference energies must not be treated as rigorous
lower bounds at arbitrarily high precision.

=== Timeline estimates

- Stage 1: Proving TPEN stability in single run
  - Hydrogen: by Nov 15
  - Hooke atom: by Nov 15
  - Helium: by Nov 22
  - H2: by Nov 29
  Prove that during training and evaluation:
  - Training loss reduces
  - Values remain finite
  - Variance reduces
- Stage 2: Building batch experiment machinery (finish by Nov 30)\
  Enables experiments on DoE computers. Already in the works in the background.
- Stage 3: Hyper-parameter scan with $n > 2$ electron systems (finish by Dec 13)
- Stage 4: Hyper-parameter scan with high-correlation systems (finish by Dec 27)\
  no I am not going to take winter break.

== Evaluation and decision criteria

#text(fill: red)[MISSING — metrics and criteria for interpreting results.]

= Limitations and Risks

#text(fill: red)[MISSING — limitations and risks section.]

#pagebreak()

#bibliography("2026_11_10.bib")

#pagebreak()

#show: appendix

= Specialization Proofs

== FermiNet

#proof(title: [@prop.ferminet])[
FermiNet maintains one-body features $bh_i$ and two-body features $bg_(i j)$. Its equivariant operations are built from local features and spin-resolved averages:
$
bar(bh)_sigma &= 1/n_sigma sum_(j: s_j = sigma) bh_j, \
bar(bg)_(i sigma) &= 1/n_sigma sum_(j: s_j = sigma) bg_(j i).
$

Let $chi_sigma(j) = 1_(s_j = sigma)$ be the spin indicator. TPEN reproduces the off-diagonal pooling terms through the mixing paths
$
sum_(j eq.not i) chi_sigma(j) bh_j
&: (s,m,m_1,m_2) = (2,1,1,2), \
sum_(j eq.not i) chi_sigma(j) bg_(j i)
&: (s,m,m_1,m_2) = (2,1,2,1).
$
The first path uses an order-two spin-indicator channel. Diagonal contributions, including $bg_(i i)$, are stored as additional one-body channels and handled locally.

FermiNet's remaining affine transformations, $tanh$ activations, and residual updates are reproduced by TPEN's mixing, aggregation, activation, and update operations, possibly using additional layers and channels.

Finally, let $F^(k)$ denote the equivariant orbital matrix of determinant $k$. For even $n$, choose a constant skew-symmetric matrix $J$ with $"Pf"(J) = 1$. The TPEN order-two feature
$
A^(k) = F^(k) J (F^(k))^T
$
is constructed by bilinear mixing of one-body orbital features. By the Pfaffian identity,
$
"Pf"(A^(k)) = det(F^(k)).
$
Odd $n$ is handled by a bordered Pfaffian. Hence,
$
Psi_"FermiNet"
= sum_k det(F^(k))
= sum_k "Pf"(A^(k))
= Psi_"TPEN".
$
]


== PsiFormer

#proof(title: [@prop.psiformer])[
For one attention head, let $bq_i, bk_i, bv_i$ be the query, key,
and value projections. The attention update is
$
bh'_i = frac(
  sum_j exp(bq_i^T bk_j / sqrt(d)) bv_j,
  sum_j exp(bq_i^T bk_j / sqrt(d))
).
$

TPEN reproduces this through the following operations:

+ *Scores:* The path $(1,1) arrow.r 2$ computes
  $z_(i j) = bq_i^T bk_j / sqrt(d)$.
+ *Exponentiation:* Pointwise activation $Gamma(z) = exp(z)$
  produces $w_(i j) = exp(z_(i j))$.
+ *Aggregation:* Paths $(2,1) arrow.r 1$ compute
  $bN_i = sum_j w_(i j) bv_j$ and $D_i = sum_j w_(i j)$.
+ *Normalization:* Pointwise reciprocal activation and
  order-one multiplication yield $bh'_i = bN_i / D_i$.

Diagonal terms $i = j$ are handled by order-one paths.
Multiple attention heads correspond to additional channels.

The remaining MLP, residual, orbital projection, and
antisymmetric readout operations follow the FermiNet
specialization. Hence PsiFormer is contained in TPEN
under the same embedding and readout assumptions.
]


== DeepErwin

#proof(title: [@prop.deeperwin])[
DeepErwin's electron-electron interactions take the form
$
u_i^c = sum_j B^c (bg_(i j)) C^c (bh_j),
$
which TPEN reproduces through $(2,1) arrow.r 1$ paths,
with diagonal terms handled as in the FermiNet specialization.

Electron-nucleus interactions have the analogous form
$
u_i^c = sum_A B^c (bg_(i A)) C^c (bh_A).
$
These can be represented through geometry-dependent one-body
features supplied by the embedding. Alternatively, nuclei may
be included as typed particles, making these interactions
ordinary type-selective tensor-product contractions.

The remaining pooling, MLP, residual, and backflow operations
follow from the FermiNet construction and repeated bilinear
mixing. Local coordinate frames and reference orbitals obtained from
external electronic-structure calculations (CASSCF or
Hartree--Fock) are treated as preprocessing. Their basis
coefficients and orbital evaluation, including at
backflow-shifted coordinates, are admitted by the embedding.
TPEN therefore need not reproduce the underlying
self-consistent-field or orbital-optimization procedures.
 The learned orbital factors,
Jastrow correction, and multi-determinant readout are then
reproduced as before.

Crucially, TPEN's mixing weights depend on channels and paths,
not particle counts or nuclear configurations. The same
parameter choices therefore apply across molecular systems,
while system-dependent information enters through the embedding.
The Pfaffian readout accommodates varying electron counts,
with sufficient orbital channels for the determinant construction.
]


== LapNet

#proof(title: [@prop.lapnet])[
LapNet maintains local features $bh_i^s$ and dense features $bh_i^d$.
Its cross-attention mechanism computes queries and keys from the
local stream and values from the dense stream:
$
bq_i = W_Q bh_i^s, quad
bk_i = W_K bh_i^s, quad
bv_i = W_V bh_i^d.
$
The attention update is
$
bu_i = frac(
  sum_j exp(bq_i^T bk_j / sqrt(d)) bv_j,
  sum_j exp(bq_i^T bk_j / sqrt(d))
).
$

This is reproduced by the same tensor-product paths and pointwise
activations as the PsiFormer specialization, using separate
order-one channels for the two streams. The remaining MLP,
residual, normalization, orbital, and readout operations follow
the preceding constructions.

Restricting the local stream to one-body updates preserves
$
pdv(bh_i^s, br_j) = 0 quad "for" i != j,
$
and hence the sparsity of attention-logit derivatives.
]


== OneQMC/Orbformer

#proof(title: [@prop.orbformer])[
Represent electrons, nuclei, and auxiliary orbital slots as
typed indices. The nuclear graph network and orbital generator
are reproduced by tensor-product contractions over nuclear
indices, while electron self-attention follows the PsiFormer
construction.

Orbital features indexed by an orbital slot $a$ and nucleus
$A$ are combined with electron-nucleus features through
contractions over $A$, reproducing the dynamically generated
orbital envelopes. Local channel operations then combine
these envelopes with electron embeddings to construct the
orbital matrix $F_(i a)$.

For even electron counts, choose an antisymmetric pairing
matrix $J$ with $"Pf"(J) = 1$ and construct
$
K_(i j) = sum_(a,b) F_(i a) J_(a b) F_(j b).
$
The determinant-to-Pfaffian identity gives
$
"Pf"(K) = det(F).
$
Odd electron counts are handled by the bordered Pfaffian
construction. Applying this to each determinant channel and
incorporating the Jastrow factor recovers the OrbFormer
wavefunction.

All learned operations use shared channel weights and
tensor-product paths. Varying the numbers of electrons,
nuclei, and orbital slots therefore does not require
changing these weights.
]


== Neural Pfaffian

#proof(title: [@prop.neural-pfaffian])[
Neural Pfaffians construct the electron pairing kernel
$
K = F A F^T,
$
where $F$ is a rectangular matrix of equivariant electron
orbitals and $A$ is a geometry-dependent skew-symmetric
matrix over orbital indices.

Write the orbital index as $(A, alpha)$, where $A$ labels
a nucleus and $alpha$ a local orbital slot. The local
orbital-slot indices occupy channels, while the nuclear
indices occupy TPEN tensor axes.

The orbital projections $F_(i A alpha)$ are represented
by electron-nucleus features, and the pairing parameters
$A_(A alpha, B beta)$ by nuclear features of order at most
two. Their contractions are
$
T_(i B beta)
  = sum_(A, alpha)
      F_(i A alpha) A_(A alpha, B beta),
$
$
K_(i j)
  = sum_(B, beta)
      T_(i B beta) F_(j B beta).
$
Both are $(2,2) arrow.r 2$ tensor-product paths, with
same-nucleus terms handled through one-body features.

The nuclear MetaGNN and Moon embedding consist of local
transformations, pairwise message passing, and normalized
aggregation, reproduced by the corresponding TPEN paths
and pointwise nonlinearities. The factorized exponential
envelopes follow from electron-nucleus contractions.

The resulting skew-symmetric kernel is passed directly
to TPEN's Pfaffian readout. Odd electron counts use the
one-body bordered construction, and multiple Pfaffians
are combined across channels.

All learned transformations are shared across systems.
The total orbital dimension grows with the number of
nuclei, while the number of channels and learned weights
remains fixed.
]


== Fermi Sets

=== Shared Lemmas

#let lemma = theorem.with(
  kind: "lemma",
  supplement: "Lemma",
  fmt-suffix: none,
)

#lemma(label: <prop.fermi-sets-invariant-pooling>, title: [Invariant pooling])[
Let
$
phi(bv) = rho(sum_(i=1)^N h(bv_i)),
$
where $h$ and $rho$ are representable by the admitted
pointwise operations. Then $phi$ can be reproduced
by TPEN with maximum feature order $M=2$.
]

#proof(title: [@prop.fermi-sets-invariant-pooling])[
Initialize one-body channels with $bx_i=h(bv_i)$.
A $(1,1) arrow.r 1$ mixing path gives
$
bg_i = sum_(j != i) bx_j + bx_i
    = sum_j bx_j,
$
where the diagonal term is supplied by a local path.

Since $bg_i$ is independent of $i$, applying $rho$
produces an invariant scalar broadcast to every
particle. All operations preserve permutation
equivariance.
]

#lemma(label: <prop.fermi-sets-circuit-compilation>, title: [Full-tuple circuit compilation])[
Let $C(bv_1,dots,bv_N)$ be a scalar function specified
by a finite computational circuit using TPEN-admissible
arithmetic and pointwise operations.

Then TPEN with $M=N$ can construct
$
X_(i_1 dots i_N)
= C(bv_(i_1),dots,bv_(i_N)).
$

Its top-order antisymmetric readout (@sec.higher-order-readouts) computes
$
cal(A)_N C(bv)
= frac(1,N!)
  sum_(sigma in S_N)
  "sgn"(sigma)
  C(bv_(sigma(1)),dots,bv_(sigma(N))).
$

If $C$ is antisymmetric, this readout equals $C$
exactly.

For a circuit with evaluation time $T_C$ and
intermediate storage $S_C$, the materialized
construction has upper bounds
$O(N! T_C)$ time and $O(N! S_C)$ memory.
]

#proof(title: [@prop.fermi-sets-circuit-compilation])[
Introduce a constant order-$N$ feature on every
ordered tuple of distinct particle indices.

Using mixing paths without index contraction,
copy the one-body inputs $bv_(i_a)$ into channels
of the order-$N$ feature, for every $a in [N]$.

Every gate of the circuit can now be simulated
independently on each ordered tuple. Addition and
multiplication follow from channel mixing and
bilinear tensor-product paths; admitted pointwise
operations are applied identically to each tuple.

This constructs $X$ by a finite sequence of
equivariant TPEN layers.

Choose the $m=N$, $r=1$ generalized readout:
$
cal(A)_N(X)
= frac(1,N!)
  sum_(sigma in S_N)
  "sgn"(sigma) X_(sigma(1) dots sigma(N)).
$

For antisymmetric $C$,
$
C(bv_(sigma(1)),dots,bv_(sigma(N)))
= "sgn"(sigma) C(bv).
$
Thus every term contributes $C(bv)/N!$,
and the result equals $C(bv)$.

There are $N!$ ordered tuples, each requiring
one evaluation of the circuit. This gives the
stated upper bounds.
]

#lemma(label: <prop.fermi-sets-complete-graph>, title: [Complete-graph contraction])[
Let $a_(i j)=-a_(j i)$ and let $phi$ enumerate
all unordered edges of the complete graph on
$N$ vertices.

The generalized higher-order readout with
$m=2$ and $r=binom(N,2)$ satisfies
$
cal(R)_phi(a)
= N! product_(i<j) a_(i j).
$

Although its formal definition contains $N!$
summands, it admits an $O(N^2)$ evaluation
once the pair features are available.
]

#proof(title: [@prop.fermi-sets-complete-graph])[
By the generalized Levi-Civita definition,
$
cal(R)_phi(a)
= sum_(sigma in S_N)
  "sgn"(sigma)
  product_(i<j) a_(sigma(i) sigma(j)).
$

Antisymmetry of $a$ implies
$
product_(i<j) a_(sigma(i) sigma(j))
= "sgn"(sigma) product_(i<j) a_(i j).
$

Consequently,
$
cal(R)_phi(a)
= sum_(sigma in S_N)
  "sgn"(sigma)^2 product_(i<j) a_(i j)
= N! product_(i<j) a_(i j).
$

The final product contains $binom(N,2)$
factors and can be evaluated directly.
]


=== Specializations

==== TPEN Specialization into Slater-based Fermi Sets

#proof(title: [@prop.fermi-sets-slater])[
By the invariant-pooling lemma, each symmetric
coefficient $phi_k$ is available as an invariant
feature.

For a Slater matrix $F_k$, multiply its first
orbital column by $phi_k$. Multilinearity gives
$
det[phi_k F_(k,1), F_(k,2),dots]
= phi_k det(F_k).
$

The determinant-to-Pfaffian construction from
the FermiNet specialization reproduces this
term. Summing over readout channels recovers
the full Slater-based Fermi Sets ansatz.
]


==== TPEN Specialization into Jordan--Wigner signatures

#proof(title: [@prop.fermi-sets-jordan-wigner])[
Initialize an antisymmetric pair feature
$
S_(i j) = "sgn"(x_j-x_i).
$

For even $N$, after sorting the particle
labels, every entry above the diagonal of
$S$ equals $1$. The Pfaffian of this matrix
is $1$, as follows by induction using its
Pfaffian expansion.

Permuting the particles multiplies the
Pfaffian by the permutation sign. Therefore
$
"Pf"(S)
= product_(i<j) "sgn"(x_j-x_i).
$

For odd $N$, append a final row and column
with $S_(i,N+1)=1$ and $S_(N+1,i)=-1$.
The same argument applies.

An invariant symmetric coefficient may be
multiplied into the final readout.
]


==== TPEN Specialization into General pair-product signatures

#proof(title: [@prop.fermi-sets-pair-product])[
Initialize
$
a_(i j) = f(bv_i,bv_j).
$

Choose the complete-graph contraction pattern
from the preceding lemma. Then
$
eta(bv)
= frac(1,N!) cal(R)_phi(a).
$

This is one of TPEN's generalized higher-order
readouts with $m=2$ and $r=binom(N,2)$.

No decomposition of $f$ into one-body
functions is required.

Multiplication by a representable invariant
coefficient $phi_k$ is admissible as an
invariant-scalar readout operation, giving
$phi_k (bv) eta_k (bv)$ exactly.
]


==== Arbitrary finitely computable Fermi Sets

#proof(title: [@prop.fermi-sets-general])[
The finite sum of products defining $Psi$ is
itself a finite computational circuit.

Since every $phi_k$ is symmetric and every
$eta_k$ is antisymmetric, $Psi$ is antisymmetric.

Apply the full-tuple circuit-compilation lemma
with $C=Psi$. Its top-order antisymmetric
readout returns $Psi$ exactly, with the stated
complexity bound.
]



== $k$-GNNs

=== Shared Lemmas
#lemma(label: <prop.k-gnn-subset-embedding>, title:[Subset-feature embedding])[
Let $bh_S$ be a feature indexed by unordered
$k$-element subsets $S subset.eq [N]$.
Then
$
bx_(i_1 dots i_k) = bh_{i_1,dots,i_k}
$
defines a permutation-equivariant TPEN feature
of order $k$, invariant under permutations
of its tuple positions.
]

#proof(title: [@prop.k-gnn-subset-embedding])[
Every ordered tuple of distinct indices
determines a unique $k$-element subset.

For any $sigma in S_N$, the tuple indexed
by $(sigma(i_1),dots,sigma(i_k))$ receives
the feature of the correspondingly relabeled
subset. Thus the lifting is equivariant.

Since the feature depends only on the
underlying subset, it is invariant under
permutations of tuple positions.
]

#lemma(label: <prop.k-gnn-vertex-replacement>, title: [Vertex-replacement aggregation])[
Let $bx_I$ be an order-$k$ TPEN feature,
and let $a_(i j)$ be an order-two feature.

For each position $b in [k]$, the operation
$
bL_I^b = sum_(j in.not I)
  a_(i_b j) bx_(I^(b -> j))
$
is a TPEN mixing path with
$
(m_1,m_2,m,s) = (k,2,k,k+1).
$

Here $I^(b -> j)$ denotes the tuple
obtained by replacing $i_b$ with $j$.
]

#proof(title: [@prop.k-gnn-vertex-replacement])[
Choose the output injection
$
tau(a) = a, quad a in [k].
$

The first input injection agrees with
$tau$ except at position $b$, where
$
tau_1(b) = k+1.
$

The second input injection is
$
tau_2(1) = b,
quad tau_2(2) = k+1.
$

The virtual interaction contains the
$k$ output indices and one additional
index $j$. Summing over this index
gives exactly
$
sum_(j in.not I)
  bx_(I^(b -> j)) a_(i_b j).
$

Summing over $b$ combines the
$k$ replacement paths.
]


=== Specialization

==== Set-based $k$-GNN containment

#proof(title: [@prop.k-gnn-set])[
By the subset-feature lemma, represent
each $k$-set feature $bh_S^t$ as an
order-$k$ TPEN feature $bx_I^t$.

Let $A_(i j)$ denote graph adjacency.
The local and global neighborhood
aggregations are
$
bL_I^t
= sum_(b=1)^k sum_(j in.not I)
  A_(i_b j) bx_(I^(b -> j))^t,
$
$
bG_I^t
= sum_(b=1)^k sum_(j in.not I)
  (1-A_(i_b j)) bx_(I^(b -> j))^t.
$

By the vertex-replacement lemma,
both are TPEN contractions.

The $k$-GNN update
$
bh_S^(t+1) = sigma(
  W_0 bh_S^t
  + W_L bL_S^t
  + W_G bG_S^t
)
$
is therefore reproduced through TPEN
aggregation and channel-wise activation.

Induction over network depth proves
exact equivalence of all subset features.

The local-only variant follows by
setting $W_G=0$.
]


==== Hierarchical $k$-GNN containment

#proof(title: [@prop.k-gnn-hierarchical])[
The hierarchical architecture initializes
order-$m$ features using features from
all $(m-1)$-element subsets:
$
bh_S^(m,0) = sigma(
  W_m [
    bh_S^"iso",
    sum_(T subset S, |T|=m-1)
      bh_T^(m-1)
  ]
).
$

The subset sum is reproduced by
$(m-1,1) arrow.r m$ paths, using
constant one-body features and
summing over the $m$ possible omitted
indices.

The remaining order-$m$ message
passing follows from the preceding
proposition, independently for
$m=1,dots,k$.
]



= Model Design <sec.model-design>

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


== Notes on Equivariant Mixing

=== Unary support paths

The tensor-product formula above is one path producer. A linear, unary producer
uses the same ordered-distinct tuple semantics. Let $I$ have output order $l$
and let $J$ have input order $m$. A *SupportPath* is the typed record
$q = (l, m, tau_o, tau_i)$, where
$tau_o: [l] arrow.hook [s]$ and $tau_i: [m] arrow.hook [s]$ are injective and
$"im"(tau_o) union "im"(tau_i) = [s]$. It records exactly which output and
input slots denote the same support particle; the overlap has size
$r = |"im"(tau_o) inter "im"(tau_i)|$.

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
$ y^d_(I,q) = a_(I,q) sum_(K in c(I,q)) W_q^(d <- c) x^c_(K circle.small tau_i), $
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

=== Higher-order readouts <sec.higher-order-readouts>

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
J_"ee"(br)
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
- alpha sum_i |br_i|^2,
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
`AtomicConfiguration` holds the nuclear positions $bR_A$ and charges $Z_A$ for
one system, and is passed once, at construction, to whichever `HamiltonianTerm`,
cusp, or decay/confinement module needs it. Helium and molecular hydrogen are
two `AtomicConfiguration` instances --- *data*, not distinct wavefunction
subclasses or branches. There is no He- or H2-specific wavefunction path;
`tpen.nn.TPENWaveFunction` and its consumers stay generic over any
`AtomicConfiguration`. Particle counts, permutations, and validation come only
from typed `.permute(...)`, `.compare(...)`, and `.validate(...)` contracts and
explicit `n_electrons`/nuclear metadata --- never from recursively probing an
arbitrary container.

The same canonical/legacy split applies on the Hamiltonian side. A `Potential`
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
J_"en"(br)
=
sum_i sum_A v_A (r_(i A)),
quad
r_(i A) = norm(br_i - bR_A).
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
  H psi_theta (br)
  =
  E_"loc" ^ theta (br)
  psi_theta (br),
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
  p_alpha (br)
  =
  (psi_alpha (br)^2) / Z_alpha,
$

$
  E_"loc" ^ alpha (br)
  =
  (H psi_alpha (br)) / (psi_alpha (br)),
$

and the centered local-energy signal

$
  A_alpha (br)
  =
  E_"loc" ^ alpha (br)
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
