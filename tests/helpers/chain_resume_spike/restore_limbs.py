"""The three restorable entropy limbs, and the apply seam that can skip one.

This module is the anti-DS-A0 core of the spike. Prior spike DS-A0 saved RNG
state, never restored it, and its deterministic parity assertion passed anyway
because nothing it compared depended on the restored stream. Its non-vacuity
control inherited the same blindness. Three design rules follow, and this
module implements all three:

**Three genuinely independent consumers.** Each limb feeds a disjoint set of
observable channels in the compared trace (:data:`LIMB_CHANNELS`). A limb whose
output never reaches the trace makes its own mutation arm vacuous, so the
channel map is asserted disjoint and total by a lane test.

**Disable is a flag at the apply seam.** :func:`apply_limb_states` skips exactly
the apply call for a disabled limb, which is what a code omission does. It is
not an edit-and-revert: that is not re-runnable and can be left half-restored.

**Verify the disable.** :func:`apply_limb_states` returns a
:class:`LimbApplication` whose ``consumed`` map is derived by *observing the
live object* after the seam -- fingerprinting the limb's actual runtime state
and comparing it against the checkpointed fingerprint -- never by echoing the
flag that was passed in. An unverified disable can invert into a strengthening
and go green; a flag echo cannot detect that, and a fingerprint can.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class RestoreLimb(Enum):
    """Closed set of independently restorable entropy sources.

    ``SAMPLER_GENERATOR``
        The fixture's private ``numpy.random.Generator``. Mirrors
        ``tpen.sampling.metropolis.MetropolisSampler``'s private generator,
        whose state travels in ``sampler.pt``.
    ``GLOBAL_RNG``
        The *process* globals: :mod:`random` and NumPy's legacy global state.
        Mirrors what ``tpen.checkpoint.rng.apply_rng_state`` restores.
    ``CADENCE_RNG``
        A callback-cadence-like private ``random.Random`` together with its
        ``num_calls`` counter. Mirrors
        ``tpen.callback.cadence.StepCadenceGate``, whose state is absent from
        TPEN's checkpoint inventory entirely (see
        ``tpen/checkpoint/payload.py`` lines 4-5, which declare checkpoint
        payloads deliberately callback-free).
    """

    SAMPLER_GENERATOR = "sampler_generator"
    GLOBAL_RNG = "global_rng"
    CADENCE_RNG = "cadence_rng"


#: Which observable trace channels each limb feeds.
#:
#: Disjointness is load-bearing and is asserted by a lane test rather than
#: merely intended. If two limbs shared a channel, a mutation arm that observed
#: divergence there could not attribute it to the limb it disabled, and the two
#: clauses would mask each other.
LIMB_CHANNELS: Mapping[RestoreLimb, frozenset[str]] = {
    RestoreLimb.SAMPLER_GENERATOR: frozenset({"proposal", "acceptance_uniform", "walkers"}),
    RestoreLimb.GLOBAL_RNG: frozenset({"noise_term", "parameters"}),
    RestoreLimb.CADENCE_RNG: frozenset({"measurement_gate", "measurement_value"}),
}


@dataclass(frozen=True)
class RestorePolicy:
    """Which limbs the apply seam is allowed to restore.

    Parameters
    ----------
    disabled : frozenset of RestoreLimb
        Limbs whose apply call is skipped. Everything else is applied.
    """

    disabled: frozenset[RestoreLimb]

    def __post_init__(self) -> None:
        # Explicit because typeguard is scoped to ``tpen`` only, so the
        # annotation above guards nothing at runtime here. A bare ``set``
        # would work by accident; a string would silently disable nothing.
        if not isinstance(self.disabled, frozenset):
            raise TypeError(
                f"disabled must be a frozenset of RestoreLimb, got {type(self.disabled).__name__}"
            )
        for limb in self.disabled:
            if not isinstance(limb, RestoreLimb):
                raise TypeError(f"disabled contains a non-RestoreLimb: {limb!r}")

    @classmethod
    def all_enabled(cls) -> "RestorePolicy":
        """Return the policy that restores every limb. The green arm."""

        return cls(disabled=frozenset())

    @classmethod
    def only_disabled(cls, limb: RestoreLimb) -> "RestorePolicy":
        """Return the policy disabling exactly ``limb``, others enabled.

        One limb per arm. Two clauses sharing one falsifier mask each other,
        so each mutation arm isolates a single limb.
        """

        if not isinstance(limb, RestoreLimb):
            raise TypeError(f"limb must be RestoreLimb, got {type(limb).__name__}")
        return cls(disabled=frozenset({limb}))

    @classmethod
    def all_disabled(cls) -> "RestorePolicy":
        """Return the policy disabling every limb.

        The all-disabled arm exists alongside the single-limb arms so that a
        limb whose individual arm passes for the wrong reason still has to
        account for itself here.
        """

        return cls(disabled=frozenset(RestoreLimb))

    def is_enabled(self, limb: RestoreLimb) -> bool:
        """Return whether ``limb`` will be applied under this policy."""

        return limb not in self.disabled

    def to_tokens(self) -> tuple[str, ...]:
        """Return the disabled limbs as sorted stable tokens for argv/receipts."""

        return tuple(sorted(limb.value for limb in self.disabled))

    @classmethod
    def from_tokens(cls, tokens: tuple[str, ...]) -> "RestorePolicy":
        """Rebuild a policy from :meth:`to_tokens` output.

        Members are looked up against the closed enum by value, never through
        ``getattr`` or arbitrary dict dispatch.
        """

        return cls(disabled=frozenset(RestoreLimb(token) for token in tokens if token))


@dataclass(frozen=True)
class LimbApplication:
    """What the apply seam actually did, as observed rather than as requested.

    Parameters
    ----------
    requested_enabled : frozenset of RestoreLimb
        Limbs the policy asked to apply.
    consumed : Mapping of RestoreLimb to bool
        Whether each limb's checkpointed state is, in fact, now live in the
        process. Derived by fingerprinting the live object after the seam and
        comparing against the checkpointed fingerprint -- never by echoing
        ``requested_enabled``. This is the mechanism test's evidence.
    live_fingerprints : Mapping of RestoreLimb to str
        Post-seam fingerprint of each live limb.
    checkpoint_fingerprints : Mapping of RestoreLimb to str
        Fingerprint of each limb as stored in the checkpoint.
    """

    requested_enabled: frozenset[RestoreLimb]
    consumed: Mapping[RestoreLimb, bool]
    live_fingerprints: Mapping[RestoreLimb, str]
    checkpoint_fingerprints: Mapping[RestoreLimb, str]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable mapping form."""

        return {
            "requested_enabled": sorted(limb.value for limb in self.requested_enabled),
            "consumed": {limb.value: flag for limb, flag in self.consumed.items()},
            "live_fingerprints": {
                limb.value: value for limb, value in self.live_fingerprints.items()
            },
            "checkpoint_fingerprints": {
                limb.value: value for limb, value in self.checkpoint_fingerprints.items()
            },
        }


def _digest(payload: Any) -> str:
    """Return a stable sha256 over a JSON-canonicalizable payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def capture_sampler_generator(generator: np.random.Generator) -> dict[str, Any]:
    """Return the JSON-serializable state of a NumPy ``Generator``."""

    state = generator.bit_generator.state
    # ``state`` nests NumPy integers inside a dict; round-trip through a JSON
    # coercion so the checkpoint payload holds plain ints and the digest below
    # is stable across processes.
    return json.loads(json.dumps(state, default=int))


def apply_sampler_generator(
    generator: np.random.Generator, state: Mapping[str, Any]
) -> None:
    """Restore a NumPy ``Generator`` from :func:`capture_sampler_generator`."""

    generator.bit_generator.state = dict(state)


def capture_global_rng() -> dict[str, Any]:
    """Return the JSON-serializable state of both process-global RNGs.

    Both are captured, because TPEN's own ``rng.pt`` restores both the
    :mod:`random` module state and NumPy's legacy global state, and a limb that
    covered only one would leave a live stream that the parity trace could
    still see.
    """

    python_state = random.getstate()
    legacy = np.random.get_state()
    return {
        "python": [python_state[0], list(python_state[1]), python_state[2]],
        "numpy_legacy": [
            legacy[0],
            [int(value) for value in legacy[1]],
            int(legacy[2]),
            int(legacy[3]),
            float(legacy[4]),
        ],
    }


def apply_global_rng(state: Mapping[str, Any]) -> None:
    """Restore both process-global RNGs from :func:`capture_global_rng`."""

    python_state = state["python"]
    random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    legacy = state["numpy_legacy"]
    np.random.set_state(
        (
            legacy[0],
            np.array(legacy[1], dtype=np.uint32),
            int(legacy[2]),
            int(legacy[3]),
            float(legacy[4]),
        )
    )


def capture_cadence(cadence_rng: random.Random, num_calls: int) -> dict[str, Any]:
    """Return the JSON-serializable state of a cadence gate.

    Both halves are captured. ``num_calls`` matters independently of the RNG:
    ``tpen.callback.cadence.StepCadenceGate`` advances ``num_calls`` on every
    admission (line 207) but touches its RNG only when ``probability < 1``
    (lines 216-217). A gate with ``probability = 1`` and a ``max_calls`` cap
    therefore has *no* live RNG dependence and a fully live counter dependence.
    """

    state = cadence_rng.getstate()
    return {
        "rng": [state[0], list(state[1]), state[2]],
        "num_calls": int(num_calls),
    }


def apply_cadence(cadence_rng: random.Random, state: Mapping[str, Any]) -> int:
    """Restore a cadence gate, returning the restored ``num_calls``."""

    rng_state = state["rng"]
    cadence_rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
    return int(state["num_calls"])


def limb_fingerprints_from_state(state: Mapping[str, Any]) -> dict[RestoreLimb, str]:
    """Return the fingerprint of each limb as stored in a checkpoint payload."""

    return {
        RestoreLimb.SAMPLER_GENERATOR: _digest(state["sampler_generator"]),
        RestoreLimb.GLOBAL_RNG: _digest(state["global_rng"]),
        RestoreLimb.CADENCE_RNG: _digest(state["cadence"]),
    }


def live_limb_fingerprints(system: Any) -> dict[RestoreLimb, str]:
    """Return the fingerprint of each limb as it currently lives in the process.

    Reads the actual runtime objects. This is what makes the disable
    verifiable: a limb that was genuinely skipped still carries the fresh
    process's own state here, and its fingerprint therefore differs from the
    checkpointed one.

    Parameters
    ----------
    system : ContinuationSystem
        The live fixture system. Typed loosely to avoid a circular import with
        :mod:`tests.helpers.chain_resume_spike.fixture`, which owns the system.
    """

    return {
        RestoreLimb.SAMPLER_GENERATOR: _digest(
            capture_sampler_generator(system.sampler_generator)
        ),
        RestoreLimb.GLOBAL_RNG: _digest(capture_global_rng()),
        RestoreLimb.CADENCE_RNG: _digest(
            capture_cadence(system.cadence_rng, system.cadence_num_calls)
        ),
    }


def apply_limb_states(
    system: Any, state: Mapping[str, Any], policy: RestorePolicy
) -> LimbApplication:
    """Apply every enabled limb to ``system``; skip each disabled one entirely.

    This is *the* apply seam. A disabled limb's apply call does not run, which
    is exactly what a code omission does -- the DS-A0 failure being reproduced
    on demand. The returned :class:`LimbApplication` reports what actually
    happened, observed from the live objects after the seam.

    Parameters
    ----------
    system : ContinuationSystem
        Live fixture system, mutated in place.
    state : Mapping
        Checkpointed limb states, as written by
        ``ContinuationSystem.limb_state_dict``.
    policy : RestorePolicy
        Which limbs to apply.

    Returns
    -------
    LimbApplication
        Requested-versus-observed record of the seam.
    """

    if not isinstance(policy, RestorePolicy):
        raise TypeError(f"policy must be RestorePolicy, got {type(policy).__name__}")

    checkpoint_fingerprints = limb_fingerprints_from_state(state)

    if policy.is_enabled(RestoreLimb.SAMPLER_GENERATOR):
        apply_sampler_generator(system.sampler_generator, state["sampler_generator"])
    if policy.is_enabled(RestoreLimb.GLOBAL_RNG):
        apply_global_rng(state["global_rng"])
    if policy.is_enabled(RestoreLimb.CADENCE_RNG):
        system.cadence_num_calls = apply_cadence(system.cadence_rng, state["cadence"])

    live = live_limb_fingerprints(system)
    # Observed, not echoed. ``consumed`` is a comparison of real bytes now in
    # the process against real bytes in the checkpoint; it is not a copy of
    # ``policy``. That is what lets a lane test prove the disable was a genuine
    # skip rather than an inverted no-op that quietly still restored.
    consumed = {
        limb: live[limb] == checkpoint_fingerprints[limb] for limb in RestoreLimb
    }
    return LimbApplication(
        requested_enabled=frozenset(
            limb for limb in RestoreLimb if policy.is_enabled(limb)
        ),
        consumed=consumed,
        live_fingerprints=live,
        checkpoint_fingerprints=checkpoint_fingerprints,
    )


__all__ = [
    "LIMB_CHANNELS",
    "LimbApplication",
    "RestoreLimb",
    "RestorePolicy",
    "apply_cadence",
    "apply_global_rng",
    "apply_limb_states",
    "apply_sampler_generator",
    "capture_cadence",
    "capture_global_rng",
    "capture_sampler_generator",
    "limb_fingerprints_from_state",
    "live_limb_fingerprints",
]
