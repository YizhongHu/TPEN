# H-v2 atomic-hydrogen scientific contract

H-v2 studies **atomic hydrogen**, with one electron, nuclear charge `Z = 1`,
and exact nonrelativistic infinite-nuclear-mass ground-state energy `-0.5 Ha`.
It is not molecular hydrogen (`H2`) or Hooke's atom. The exact state is an
instrument control, not a substitute for a trained-state result.

This directory is a documentation and admission slice. It contains no
launcher, training code, science output, or qualification result.

## Evidence boundary

The comparison source is pinned to `origin/dev` commit
`67beab2095cc987eb044abceeecfdb549a5793fb`: the He-v1 README, production
grid, collector, and report. Historical run identity and corrections are
listed in `contract_inputs.json`. The current `production_grid.yaml` at that
commit names a 100k/200k/300k plan; it is not the executed historical
25k/50k plan.

The historical helium basin is a **5k-block median of within-batch variance
on a changing training model**, approximately `0.00173 Ha^2`. It is training
history, not a frozen-checkpoint population estimate. The reported `s10=26570`
and `s90=44982` values are definition-dependent summaries; the definition is
only in the unopened Netscratch report, so neither value is the point where a
basin was first left. The retained fresh evaluations show a finite-range
25k-to-50k tail contrast, but do not establish a population moment class or a
common-estimand decrease-then-increase trajectory.

## Lane proposals pending operator review

The seven entries in `contract_inputs.json` are deliberately lane proposals,
not operator decisions. Each has a rationale and alternatives. The live
contention is visible: percentile `W` versus event probability as the primary
axis; fixed versus trainable nuclear curvature; three versus five lineages;
the factor-two/`0.01 Ha` and `0.002` effect margins; and Cannon single-card
science versus efficiently packed Polaris science.

The proposed science design is 50,000 completed updates, retaining update zero
and every 1,000 updates (51 retained checkpoints) while fresh-evaluating a
separate 27-checkpoint set. It proposes three independent lineages with
initialization seeds `73001..73003` and sampler seeds `83001..83003`, four
fresh streams per evaluated checkpoint, 4,096 walkers, 256 draws, burn-in
1,000, and stride 20. Evaluation seeds use the formula recorded in the JSON;
pilot seeds are `999001` and `999002` and are excluded from science.

The proposed primary statistic is

`W = P99.9(abs(E_L - median(E_L)))`.

The contract also reports high quantiles, separate positive/negative survival
curves, and fixed-threshold exceedance probabilities at `0.01`, `0.05`,
`0.1`, `0.5`, and `1 Ha`. MAD, IQR, ordinary variance, and conventional
MCSE are secondary/descriptive only. The `W` recurrence rule requires a factor
of two and `0.01 Ha` change with simultaneous uncertainty support; the bounded
null additionally requires every fixed-threshold exceedance probability to
exclude a late absolute increase of `0.002`. A flat MAD is not equivalence.

Held-out beta survival-slope curvature and calibrated generalized-Pareto `xi`
are interpretation diagnostics. Hill is context only. An absent or
inconclusive admissible tail diagnostic is `unresolved`, never a pass. No
finite-prefix diagnostic proves an ultimate finite or infinite variance.

## Closed outcomes

The only result categories are `recurrence`, `bounded_null`, and
`unresolved`. Recurrence requires complete records, qualified uncertainty,
the predeclared basin search, late persistence, and at least two of three
independent lineages. Bounded null requires all three complete lineages,
simultaneous inference over the full predeclared search, and both the `W` and
exceedance safeguards. Missing identities, records, controls, dependence
qualification, operator decisions, or supported margins produce `unresolved`.

## Two admission scopes

The non-science one-electron **PIPELINE CHECK** is admitted under the
operator's 2026-11-05 direction. Parts (a)-(f), including the independent
analytic oracle, imperfect-state rejection, sampler controls, short learned
route, fresh restore, and complete-record refusal controls, belong to slice
`5cfcca3e` on Cannon CPU. Part (g), the excluded-seed GPU route smoke, belongs
to slice `406194c5` on Polaris. Slice `2f1b2921` reruns the integrated check
on the exact candidate. This check is not a science result and does not
authorize the executor or a lineage.

The **SCIENCE LINEAGES** remain prohibited until the operator has reviewed the
science plan and explicitly approves it. No pipeline pass, scheduler state,
pilot, or capacity estimate can waive that hold.

The executable controls must separately enforce the raw `-1/r` potential and
`-1/2` Laplacian, absence of an electron-electron term, the intended nuclear
cusp and decay, the odd-electron readout as a readout device rather than a
second electron, rejection at undefined radius zero without regularization,
an independently coded scalar hydrogen oracle, sampler-target/mutation
rejection, and source/resolved-input identity checks.
