# H-v2 atomic-hydrogen scientific contract

H-v2 is a study of **atomic hydrogen**: one electron, nuclear charge `Z = 1`,
and the exact nonrelativistic infinite-nuclear-mass ground-state energy
`E_0 = -0.5 Ha`. It is **not molecular hydrogen (H2)** and **not Hooke's
atom**. The hydrogen reference is an instrument control, not a replacement for
the energy of a trained state.

This directory contains the contract and its machine-readable admission inputs.
It contains no launcher, training code, or science result. The next slice owns
the executable controls named here.

## Comparison with He-v1

The comparison source records are all at `origin/dev` commit
`67beab2095cc987eb044abceeecfdb549a5793fb`:

| record | purpose |
| --- | --- |
| `experiments/atomistic/he-v1/README.md` | study scope, existing record and evaluation conventions |
| `experiments/atomistic/he-v1/configs/production_grid.yaml` | declared checkpoint/evaluation design and complete-record requirements |
| `experiments/atomistic/he-v1/collect.py` | source, checkpoint, trajectory-record, and identity reconciliation |
| `experiments/atomistic/he-v1/report.py` | report/receipt rendering and aggregate interpretation |

The preserved evidence records were also read, but are not repository source
at that commit: `/Users/yizhonghu/TPEN/HANDOFF-he-v1-2026-08-31.md` and
`/Users/yizhonghu/TPEN/HE-V1-SESSION-SYNTHESIS-2026-08-28.md` are untracked,
read-only handoffs. They identify the corrected collection
`f4-collect-20260828T0100--0400`, the `train-seed0000` lineage, and the
`tpen-f4-eval42-78152c40-20260826T194602--0400` run. Those paths are evidence
provenance, not live inputs for H-v2.

What He-v1 establishes is narrower than the motivating shorthand. The handoff
reports 25,000-update Hill tail estimates of `2.469--6.045` and 50,000-update
estimates of `0.465--1.063`; later tail-curvature work correctly demotes both
to finite-range diagnostics rather than population moment theorems. The
corrected synthesis reports a 25k-to-50k endpoint dispersion contrast and an
unresolved mean shift. A separate retained training-history analysis shows a
broad variance minimum around 9,000--15,000 updates and gradual onset in
training history (`s10=26570`, `s90=44982`), but those checkpoints were not
retained and freshly evaluated. Thus He-v1 does **not** establish a full
decrease-then-increase trajectory on one common fresh-evaluation estimand.
H-v2 must test recurrence; it must not treat the helium endpoint contrast or
training-history variance as a threshold or as proof of a fresh-evaluation
trajectory.

## Estimand and records

For each checkpoint and each fresh evaluation stream, the primary estimand is
a tail-sensitive dispersion profile of the local-energy random variable
`E_L = H Psi / Psi` under the checkpoint's learned `|Psi|^2` distribution. It
contains predeclared high quantiles, exceedance probabilities above fixed
thresholds, and the local tail-curvature diagnostic. This profile is the
scientific estimand because a robust center statistic can be blind to a rare
event-weight defect. The retained finite-sample MAD,

`D = median(|E_L - median(E_L)|)`,

computed over **all** complete draw-by-walker records is secondary, useful for
ranking but not sufficient to accept a bounded null or reject a tail defect.
Report the median and IQR alongside MAD for auditability. The ordinary sample
variance and conventional MCSE are descriptive only and must not be promoted
to a population claim when the tail diagnostic is unresolved.

Each checkpoint window is a fresh evaluation of a frozen checkpoint, with a
new evaluation seed identity and a declared burn-in, stride, walker count,
draw count, and record schema. The retained checkpoint schedule must be dense
enough to resolve both a possible early minimum and a later rise; the proposal
is 2,500-update spacing from 5,000 through 20,000 followed by 5,000-update
spacing through the horizon, subject to the pending operator decision. A
window is complete only when every planned
draw-by-walker record is present exactly once, has finite coordinates and a
finite local energy, and passes source/checkpoint/config identity checks. No
clipping, filtering, replacement draw, final-draw shortcut, or pooled partial
window is allowed. An incomplete window is `unresolved`, not zero or a pass.

Training and evaluation seeds are separate namespaces and must be distinct:
`training_seed = (lineage, initialization_seed)` and
`evaluation_seed = (lineage, checkpoint, evaluation_stream_seed)`. Reusing a
training seed as an evaluation stream seed is rejected. Lineages are distinct
training identities, not repeated evaluations of one training run.

### Tail-existence rule

The proposed diagnostic is a held-out threshold trajectory: local log-survival
slope/curvature (`beta`) plus a generalized-Pareto shape estimate (`xi`) on
the upper tail of `|E_L - median(E_L)|`, with nested prefixes, threshold
stability, and minimum exceedance counts chosen by the operator in the pending
inputs. A tail class is declared only when the selected diagnostic is
admissible and its stability rule is met. A Hill estimate may be reported as
context, but it is not by itself a population tail theorem. If the diagnostic
is not admissible or its class interval is inconclusive, tail status is
`unresolved`, not absent. MAD/IQR may rank windows, but tail claims and the
science result must be carried by the tail-sensitive profile; variance and
conventional MCSE remain descriptive.

## Predeclared outcome rules

The result category is one of exactly `recurrence`, `bounded_null`, or
`unresolved`.

* `recurrence`: complete records exist for the declared horizon and checkpoints;
  the accepted tail diagnostic is either absent or explicitly handled; and at
  least the declared replication count of independent lineages shows the same
  early-to-middle decrease and middle-to-late increase in `D`, each exceeding
  the operator-approved margins and persisting at the required late checkpoints.
* `bounded_null`: complete records exist for the declared horizon and
  checkpoints, no lineage satisfies the recurrence rule, and every lineage's
  `D` remains within the operator-approved bound relative to the operator's
  helium reference behavior. A null is bounded to this protocol and horizon;
  it does not explain helium or prove absence at other horizons.
* `unresolved`: any required record, identity, tail decision, checkpoint, or
  operator threshold is absent; the tail diagnostic is inconclusive; or the
  evidence falls between the recurrence and bounded-null rules.

These rules are deliberately closed: no fixed threshold is inherited from
the superseded one-electron draft, and no helium endpoint ratio is reused.

## Controls specified for the next slice

The implementation must provide: the exact hydrogen energy `-1/2 Ha`; an
independently coded scalar local-energy oracle; rejection of a known-imperfect
state; sampler-target and mutation rejection; a strict no-regularization rule
at radius zero; and source plus resolved-input identity checks. These controls
qualify the instrument and admission; they do not themselves constitute a
science result.

## Admission gate

Training admission is prohibited while any required operator decision in
`contract_inputs.json` is `absent` or `pending`. The six required decisions
are: helium reference behavior to match; update horizon and checkpoints;
lineage count; walker and fresh-evaluation budget; acceptable tail diagnostic;
and science facility/resource ceiling. The file contains clearly labelled
proposals to make the choices concrete, but proposals are not decisions.

