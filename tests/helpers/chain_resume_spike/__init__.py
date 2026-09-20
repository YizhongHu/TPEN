"""Shared chain-resume continuation fixture for the R0 comparative spike.

This package is the ARM T (torch-free) half of TPEN chain-resume spike R0. It
exists so that R1/R2/R3 -- the Balsam, IRI/PBS and Globus Compute candidate
lanes -- can exercise backend continuation mechanics against one small,
executable, stochastic fixture without importing ``torch`` and without needing
a GPU. Backend mechanics testing is thereby decoupled from facility runtime
qualification.

Two properties of this package are load-bearing and must not be relaxed:

1. Every resume attempt is a **fresh OS process**, spawned with an absolute
   ``sys.executable``. In-process resume is forbidden as vacuous -- a live RNG
   object surviving in memory makes a continuation gate pass for free.
2. Continuation parity is asserted on **subsequent draws** taken after the
   restore, never on serialized state bytes and never on deterministic outputs
   alone. Prior spike DS-A0 saved RNG state, never restored it, and its
   deterministic parity assertion passed anyway.

See ``experiments/chain-resume-spikes/README.md`` for the gate coverage map and
for what this package does and does not establish.
"""
