"""The torch-free stochastic continuation fixture, and its commit replica.

This is ARM T of spike R0: a small executable stochastic system with a fixed
total logical target, three independent entropy consumers, and a checkpoint
commit sequence that reproduces ``tpen/checkpoint/save.py`` order exactly.

WHAT IS PRODUCTION CODE HERE, and it is most of it. The generation *selection*
and *validity* semantics are not reimplemented: this module calls
``tpen.checkpoint.artifact.checkpoint_step_dir_name`` /
``list_complete_checkpoints`` / ``write_latest``, the real
``tpen.checkpoint.manifest.CheckpointManifest``, the real
``tpen.checkpoint.reference.CheckpointRef``, the real
``tpen.checkpoint.catalog.CheckpointCatalog``, the real
``tpen.checkpoint.hashing.file_sha256`` and the real
``tpen.checkpoint.receipt.record_publication_receipt``. All of those were
measured torch-free at runtime, so the G6 clauses are asserted against
production code on a host with no torch installed at all.

WHAT IS A DISCLOSED REPLICA, and it is only this. ``save_checkpoint`` writes its
payload files with ``torch.save`` (save.py lines 161, 167, 179, 186). Those four
calls, and only those, are replaced by a torch-free byte writer.
:data:`BOUNDARY_ORDER` declares the sequence this module performs, and
``tests/unit/chain_resume_spike/test_generation_preservation.py`` asserts that
declaration still matches the order of the corresponding anchors in the real
``save.py`` source -- so a production reorder breaks this fixture's test rather
than silently invalidating its claim.

FIXTURE SIMPLIFICATION, disclosed because it is load-bearing for attribution.
The parameter update here depends on the global-RNG limb only; it does not read
the walkers. Real VMC couples them, and under coupling a disabled sampler limb
would perturb every downstream channel at once. The decoupling is what makes
:data:`~tests.helpers.chain_resume_spike.restore_limbs.LIMB_CHANNELS` disjoint
and therefore what makes per-limb attribution meaningful. The attribution claim
is a claim about this fixture, not about production TPEN.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tpen.checkpoint.artifact import (
    checkpoint_step_dir_name,
    list_complete_checkpoints,
    read_latest,
    resolve_checkpoint_dir,
    write_latest,
)
from tpen.checkpoint.catalog import CheckpointCatalog, publication_catalog_path
from tpen.checkpoint.hashing import file_sha256
from tpen.checkpoint.manifest import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManifest,
)
from tpen.checkpoint.receipt import publication_receipt_path, record_publication_receipt
from tpen.checkpoint.reference import CheckpointRef

from .faults import FaultAction, FaultPlan, FaultPoint, InjectedFault, record_fire
from .identity import ChainIdentity
from .restore_limbs import (
    RestorePolicy,
    apply_limb_states,
    capture_cadence,
    capture_global_rng,
    capture_sampler_generator,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Observable trace channels, in the order one step produces them.
#:
#: The order is load-bearing: parity reports the FIRST diverging channel, and
#: "first" is meaningful only against the order in which a step actually emits
#: them. A mutation arm asserts that the first divergence falls in the channel
#: set its own disabled limb feeds.
CHANNELS: tuple[str, ...] = (
    "proposal",
    "acceptance_uniform",
    "walkers",
    "noise_term",
    "parameters",
    "measurement_gate",
    "measurement_value",
)

#: Number of walkers in the toy sampler.
N_WALKERS = 4
#: Dimension of the toy parameter vector.
N_PARAMETERS = 3
#: Cadence admission probability. Deliberately below 1 so the cadence limb's RNG
#: is genuinely consumed and therefore genuinely observable in the trace. A
#: ``probability = 1`` gate never draws (mirroring
#: ``tpen.callback.cadence.StepCadenceGate._draw`` lines 216-217), which would
#: make the cadence mutation arm vacuous on its RNG half.
CADENCE_PROBABILITY = 0.6


@dataclass
class ContinuationSystem:
    """One process-local instance of the toy stochastic system.

    Three entropy consumers, deliberately independent:

    * ``sampler_generator`` -- private ``numpy.random.Generator`` feeding
      proposals and acceptance uniforms.
    * the *process globals* :mod:`random` and NumPy legacy, feeding the
      parameter noise term. Not held as a field, because they are global.
    * ``cadence_rng`` plus ``cadence_num_calls`` -- a callback-cadence-like
      gate deciding which steps emit a measurement, and what it emits.

    Parameters
    ----------
    sampler_generator : numpy.random.Generator
        Private sampler stream.
    cadence_rng : random.Random
        Private cadence stream.
    walkers : numpy.ndarray
        Current walker positions.
    parameters : numpy.ndarray
        Current parameter vector.
    step : int
        Next logical step index this system will execute.
    cadence_num_calls : int
        How many measurements the cadence gate has admitted so far.
    cadence_max_calls : int or None, optional
        Optional admission cap, mirroring ``StepCadence.max_calls``.
    cadence_probability : float, optional
        Admission probability, mirroring ``StepCadence.probability``.
    """

    sampler_generator: np.random.Generator
    cadence_rng: random.Random
    walkers: np.ndarray
    parameters: np.ndarray
    step: int = 0
    cadence_num_calls: int = 0
    cadence_max_calls: int | None = None
    cadence_probability: float = CADENCE_PROBABILITY
    process_seed: int = 0

    @classmethod
    def fresh(
        cls,
        seed: int,
        *,
        cadence_max_calls: int | None = None,
        cadence_probability: float = CADENCE_PROBABILITY,
    ) -> "ContinuationSystem":
        """Build a system seeded from ``seed``, seeding the process globals too.

        Every entropy source the fixture uses is (re)seeded here, including the
        two process globals. A resumed attempt calls this with its OWN fresh
        seed before restoring, precisely so that an unrestored limb carries
        visibly different state rather than accidentally matching the parent.

        Parameters
        ----------
        seed : int
            Seed for this process. In a real attempt this comes from OS
            entropy and is recorded in the receipt, so distinctness from the
            parent is auditable rather than assumed.
        cadence_max_calls : int or None, optional
            Admission cap for the cadence gate.
        cadence_probability : float, optional
            Admission probability for the cadence gate.
        """

        random.seed(seed)
        np.random.seed(seed % (2**32))
        generator = np.random.default_rng(seed)
        return cls(
            sampler_generator=generator,
            cadence_rng=random.Random(seed ^ 0x5DEECE66D),
            walkers=generator.normal(size=N_WALKERS),
            parameters=generator.normal(size=N_PARAMETERS),
            step=0,
            cadence_num_calls=0,
            cadence_max_calls=cadence_max_calls,
            cadence_probability=cadence_probability,
            process_seed=int(seed),
        )

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step_once(self) -> dict[str, Any]:
        """Advance one logical step, returning that step's observable channels.

        The returned mapping has exactly the keys of :data:`CHANNELS`, plus
        ``step``. Every value is JSON-round-trippable and exactly comparable;
        Python's float repr is shortest-round-trip, so a JSON trace compares
        bit-for-bit against an in-memory one.
        """

        # --- Limb: SAMPLER_GENERATOR ---------------------------------
        proposal = self.sampler_generator.normal(size=N_WALKERS) * 0.35
        acceptance_uniform = self.sampler_generator.random(size=N_WALKERS)
        # A Metropolis-shaped accept test. Deterministic given the two draws
        # above, so it introduces no fourth entropy source.
        candidate = self.walkers + proposal
        log_ratio = -(candidate**2 - self.walkers**2)
        accepted = acceptance_uniform < np.exp(np.minimum(log_ratio, 0.0))
        self.walkers = np.where(accepted, candidate, self.walkers)

        # --- Limb: GLOBAL_RNG ----------------------------------------
        # Both process globals are drawn from, because TPEN's own ``rng.pt``
        # restores both and a limb covering only one would leave a live stream.
        python_draw = random.gauss(0.0, 1.0)
        numpy_legacy_draw = float(np.random.standard_normal())
        noise_term = [python_draw, numpy_legacy_draw]
        self.parameters = self.parameters * 0.97 + 0.1 * (python_draw + numpy_legacy_draw)

        # --- Limb: CADENCE_RNG ---------------------------------------
        gate = self._cadence_admits()
        measurement_value = self.cadence_rng.gauss(0.0, 1.0) if gate else None

        trace = {
            "step": self.step,
            "proposal": [float(value) for value in proposal],
            "acceptance_uniform": [float(value) for value in acceptance_uniform],
            "walkers": [float(value) for value in self.walkers],
            "noise_term": noise_term,
            "parameters": [float(value) for value in self.parameters],
            "measurement_gate": bool(gate),
            "measurement_value": measurement_value,
        }
        self.step += 1
        return trace

    def _cadence_admits(self) -> bool:
        """Decide admission, mirroring ``StepCadenceGate.should_run`` ordering.

        The cap is consulted before the probability draw and the counter
        advances only on admission -- the same order as
        ``tpen.callback.cadence.StepCadenceGate`` lines 199-207. That ordering
        is what makes ``num_calls`` an independently observable piece of state:
        once the cap is reached no draw happens at all.
        """

        if self.cadence_max_calls is not None and self.cadence_num_calls >= self.cadence_max_calls:
            return False
        if self.cadence_probability >= 1.0:
            admitted = True
        elif self.cadence_probability <= 0.0:
            admitted = False
        else:
            admitted = self.cadence_rng.random() < self.cadence_probability
        if not admitted:
            return False
        self.cadence_num_calls += 1
        return True

    def run(self, steps: int) -> list[dict[str, Any]]:
        """Advance ``steps`` logical steps, returning the trace of each."""

        return [self.step_once() for _ in range(steps)]

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def limb_state_dict(self) -> dict[str, Any]:
        """Return the three limbs' states, JSON-serializable."""

        return {
            "sampler_generator": capture_sampler_generator(self.sampler_generator),
            "global_rng": capture_global_rng(),
            "cadence": capture_cadence(self.cadence_rng, self.cadence_num_calls),
        }

    def carried_state_dict(self) -> dict[str, Any]:
        """Return the non-entropy state, which every restore applies.

        Walkers and parameters are carried unconditionally, exactly as
        ``MetropolisSampler.load_mcmc_state_dict`` restores walkers alongside
        -- but separably from -- its private generator. Keeping them out of the
        limb set is what makes a disabled sampler limb diverge on the very
        first *draw* rather than on the initial positions.
        """

        return {
            "walkers": [float(value) for value in self.walkers],
            "parameters": [float(value) for value in self.parameters],
            "step": int(self.step),
            "cadence_max_calls": self.cadence_max_calls,
            "cadence_probability": float(self.cadence_probability),
        }

    def apply_carried_state(self, state: dict[str, Any]) -> None:
        """Apply :meth:`carried_state_dict` output in place."""

        self.walkers = np.array(state["walkers"], dtype=float)
        self.parameters = np.array(state["parameters"], dtype=float)
        self.step = int(state["step"])
        self.cadence_max_calls = state["cadence_max_calls"]
        self.cadence_probability = float(state["cadence_probability"])


def restore_system(
    system: ContinuationSystem, payload: dict[str, Any], policy: RestorePolicy
) -> Any:
    """Restore ``payload`` into ``system`` under ``policy``.

    Carried state is applied first and unconditionally; the three entropy limbs
    then go through the single apply seam in
    :func:`~tests.helpers.chain_resume_spike.restore_limbs.apply_limb_states`,
    which is the only place a limb can be skipped.

    Returns
    -------
    LimbApplication
        Observed record of which limbs actually took effect.
    """

    system.apply_carried_state(payload["carried"])
    return apply_limb_states(system, payload["limbs"], policy)


# ----------------------------------------------------------------------
# Commit boundary replica
# ----------------------------------------------------------------------

#: The boundary sequence this module performs, in order.
#:
#: Cross-checked against the real ``tpen/checkpoint/save.py`` by
#: ``tests/unit/chain_resume_spike/test_generation_preservation.py``. The names
#: are this fixture's; the ORDER is production's, and the test is what keeps the
#: two bound together.
BOUNDARY_ORDER: tuple[str, ...] = (
    "tmp_dir_named",
    "stale_tmp_removed",
    "tmp_dir_created",
    "resolved_config_written",
    "payload_written",
    "component_hashes_computed",
    "manifest_written",
    "complete_marker_written",
    "renamed",
    "ref_built",
    "catalog_published",
    "latest_written",
    "receipt_recorded",
)


def publish_generation(
    root: Path,
    system: ContinuationSystem,
    identity: ChainIdentity,
    *,
    credited_steps: tuple[int, ...],
    fault: FaultPlan | None = None,
    boundary_log: list[str] | None = None,
) -> Path:
    """Commit one checkpoint generation, reproducing ``save_checkpoint`` order.

    Parameters
    ----------
    root : pathlib.Path
        Checkpoint root, the analogue of ``save_checkpoint``'s ``output_dir``.
    system : ContinuationSystem
        System whose state is committed.
    identity : ChainIdentity
        Chain identity recorded in the manifest provenance.
    credited_steps : tuple of int
        Logical steps this generation credits, recorded in the manifest payload
        so the next attempt can rebuild the ledger without double counting.
    fault : FaultPlan or None, optional
        Interruption to inject. ``None`` runs the whole sequence.
    boundary_log : list of str or None, optional
        When supplied, each boundary name is appended AS IT EXECUTES. This is
        what lets a test pin the order this function actually performs, rather
        than the order :data:`BOUNDARY_ORDER` merely declares. A static
        comparison of the declaration against production catches a drift in
        either of those two, and misses a drift in the third thing -- the
        executable code here that the declaration stands for.

    Returns
    -------
    pathlib.Path
        The committed generation directory.
    """

    def _boundary(name: str) -> None:
        """Record that a boundary executed, in execution order."""

        if boundary_log is not None:
            boundary_log.append(name)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    created_at = time.time()
    # Real production naming helper, not a local f-string.
    final_dir = root / checkpoint_step_dir_name(system.step)
    tmp_dir = root / f"{final_dir.name}.tmp"          # save.py:141
    _boundary("tmp_dir_named")
    if final_dir.exists():                            # save.py:142
        raise FileExistsError(f"checkpoint already exists: {final_dir}")
    # Recorded when the SWEEP RUNS, not when it removes something: the
    # ``if`` itself is the boundary, and production executes it every time.
    if tmp_dir.exists():                              # save.py:144
        shutil.rmtree(tmp_dir)
    _boundary("stale_tmp_removed")
    files: dict[str, str] = {}
    try:
        tmp_dir.mkdir(parents=True)                   # save.py:157
        _boundary("tmp_dir_created")
        (tmp_dir / "resolved_config.yaml").write_text(
            "fixture: chain_resume_spike\n", encoding="utf-8"
        )                                             # save.py:158
        files["resolved_config"] = "resolved_config.yaml"
        _boundary("resolved_config_written")

        # save.py:161/167/179/186 write these with ``torch.save``. THIS is the
        # replica: a torch-free byte writer, same files, same order.
        (tmp_dir / "model.pt").write_bytes(
            json.dumps(system.carried_state_dict(), sort_keys=True).encode("utf-8")
        )
        files["model"] = "model.pt"
        _fire_if(fault, FaultPoint.DURING_PAYLOAD_WRITE)
        (tmp_dir / "sampler.pt").write_bytes(
            json.dumps(system.limb_state_dict(), sort_keys=True).encode("utf-8")
        )
        files["sampler"] = "sampler.pt"
        (tmp_dir / "rng.pt").write_bytes(
            json.dumps({"note": "limbs travel in sampler.pt"}, sort_keys=True).encode("utf-8")
        )
        files["rng"] = "rng.pt"
        _boundary("payload_written")
        _fire_if(fault, FaultPoint.AFTER_PAYLOAD_BEFORE_MANIFEST)

        # Real production hashing, over the real written bytes.
        hashes: dict[str, str | None] = {
            f"{key}_sha256": file_sha256(tmp_dir / name) for key, name in files.items()
        }                                             # save.py:194
        _boundary("component_hashes_computed")

        manifest = CheckpointManifest(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            kind=CHECKPOINT_KIND,
            next_iteration=int(system.step),
            completed_updates=int(system.step),
            created_at_unix=created_at,
            files=files,
            hashes=hashes,
            runtime={"device": "cpu", "dtype": "float64"},
            provenance={
                "run_id": identity.run_id,
                "attempt_id": identity.attempt_id,
                "attempt_index": identity.attempt_index,
                "parent_generation": identity.parent_generation,
                "credited_steps": list(credited_steps),
            },
        )
        manifest.write(tmp_dir / "manifest.json")     # save.py:209
        _boundary("manifest_written")
        _fire_if(fault, FaultPoint.AFTER_MANIFEST_BEFORE_COMPLETE)

        (tmp_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")  # save.py:210
        _boundary("complete_marker_written")
        _fire_if(fault, FaultPoint.AFTER_COMPLETE_BEFORE_RENAME)

        tmp_dir.rename(final_dir)                     # save.py:211  THE COMMIT
        _boundary("renamed")
        _fire_if(fault, FaultPoint.AFTER_RENAME_BEFORE_CATALOG)

        ref = CheckpointRef.from_directory(final_dir)  # save.py:216
        _boundary("ref_built")
        catalog = CheckpointCatalog(publication_catalog_path(root))
        catalog.publish(ref)                          # save.py:226
        _boundary("catalog_published")
        _fire_if(fault, FaultPoint.AFTER_CATALOG_BEFORE_LATEST)

        write_latest(
            root, final_dir, step=int(system.step), created_at_unix=created_at
        )                                             # save.py:229
        _boundary("latest_written")
        _fire_if(fault, FaultPoint.AFTER_LATEST_BEFORE_RECEIPT)

        record_publication_receipt(
            ref,
            final_dir,
            files,
            publication_receipt_path(root),
            write_duration_sec=0.0,
            publish_duration_sec=0.0,
        )                                             # save.py:237
        _boundary("receipt_recorded")

        # TORN_CATALOG_ROW is damage to the INDEX after a clean commit, not an
        # interruption of the commit, so it is applied here rather than by
        # aborting mid-sequence. Making it a real registered injection -- rather
        # than something a test arranges by hand -- is what lets the coverage
        # check observe it FIRING like every other point.
        _fire_torn_catalog_row(fault, root)
    finally:
        # ``finally``, matching save.py:245. Note what it does NOT cover, in
        # production or here: ``os._exit`` and an unhandled ``SIGTERM``/
        # ``SIGKILL`` terminate without unwinding, so this never runs for
        # those fault actions. That asymmetry is the point of having both.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return final_dir


def _fire_if(fault: FaultPlan | None, point: FaultPoint) -> None:
    """Fire ``fault`` when it targets ``point``; otherwise do nothing."""

    if fault is not None and fault.point is point:
        _apply_fault(fault)


def _fire_torn_catalog_row(fault: FaultPlan | None, root: Path) -> None:
    """Tear the catalog's final row, if ``fault`` targets that point.

    Truncates the last line and drops its terminating newline -- exactly what an
    append interrupted mid-write leaves behind, and the shape
    ``CheckpointCatalog.iter_publications`` diagnoses as recoverable
    (catalog.py:127-150). The committed directory is untouched.
    """

    if fault is None or fault.point is not FaultPoint.TORN_CATALOG_ROW:
        return
    record_fire(FaultPoint.TORN_CATALOG_ROW)
    catalog_path = publication_catalog_path(root)
    rows = catalog_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not rows:
        raise InjectedFault("cannot tear an empty catalog")
    torn = rows[-1][: max(1, len(rows[-1]) // 2)].rstrip("\n")
    catalog_path.write_text("".join(rows[:-1]) + torn, encoding="utf-8")


def _apply_fault(fault: FaultPlan) -> None:
    """Terminate the attempt the way ``fault`` asks.

    ``SIGKILL_FROM_PARENT`` is not self-inflicted: the child cannot deliver a
    signal no handler can run for and still be the thing being tested. It
    writes a ready marker and blocks; the parent kills the process group.
    """

    # Recorded FIRST, and durably. ``os._exit`` and ``SIGKILL`` below leave no
    # chance to record anything afterwards, which is exactly their purpose.
    record_fire(fault.point)

    if fault.action is FaultAction.RAISE:
        raise InjectedFault(f"injected fault at {fault.point.value}")
    if fault.action is FaultAction.OS_EXIT:
        # No unwinding: no ``finally``, no atexit, no buffered flush.
        os._exit(fault.exit_code)
    if fault.action is FaultAction.SIGTERM_SELF:
        os.kill(os.getpid(), signal.SIGTERM)
        # Under Python's default handler this does not return.
        time.sleep(30)
    if fault.action is FaultAction.SIGKILL_FROM_PARENT:
        ready = os.environ.get("CHAIN_RESUME_SPIKE_READY_MARKER")
        if ready:
            Path(ready).write_text("ready\n", encoding="utf-8")
        # Block until the parent's SIGKILL lands. Bounded so a parent that
        # never kills produces a timeout rather than a hang forever.
        time.sleep(60)


# ----------------------------------------------------------------------
# Generation selection, over production primitives
# ----------------------------------------------------------------------


def resume_generation(root: Path) -> Path | None:
    """Return the generation PRODUCTION RESUME would continue from, or ``None``.

    THIS IS THE SELECTION SURFACE THAT GOVERNS RESUME, and it is the one the
    fixture uses. ``restore_checkpoint`` reaches its checkpoint through exactly
    one path: ``restore.py:118`` calls
    ``tpen.checkpoint.artifact.resolve_checkpoint_dir``, which reads
    ``latest.json`` (artifact.py:68 and :71). Measured at this revision:
    ``list_complete_checkpoints`` has NO caller anywhere in ``tpen/checkpoint``
    outside its own definition and the package ``__init__`` re-export. **The
    production resume path never lists directories.**

    Returns ``None`` when no pointer exists yet, which is a cold start.
    """

    root = Path(root)
    if not (root / "latest.json").is_file():
        return None
    return resolve_checkpoint_dir(root)


def newest_listed_generation(root: Path) -> Path | None:
    """Return the newest generation a DIRECTORY LISTING would select.

    Delegates to ``tpen.checkpoint.artifact.list_complete_checkpoints``, which
    rejects ``.tmp`` names and directories lacking ``manifest.json`` or
    ``COMPLETE``.

    KEPT DELIBERATELY, AND DELIBERATELY NOT USED FOR RESUME. This function
    exists so the DISAGREEMENT between the two surfaces can be pinned as an
    explicit property. In the ``after_rename_before_catalog`` and
    ``after_catalog_before_latest`` windows a newer generation is complete on
    disk while ``latest.json`` still names the older one, so listing and the
    pointer return DIFFERENT directories. R1/R2/R3 will build controllers, and
    a controller that lists instead of reading the pointer selects a generation
    the production resume path would not.
    """

    complete = list_complete_checkpoints(root)
    return complete[-1] if complete else None


def pointer_target_name(root: Path) -> str | None:
    """Return the bare directory name ``latest.json`` records, or ``None``.

    Reads through production ``read_latest``. Exposed separately from
    :func:`resume_generation` so a test can observe the pointer's CONTENT
    without also exercising the validity checks ``resolve_checkpoint_dir``
    applies on the way.
    """

    root = Path(root)
    if not (root / "latest.json").is_file():
        return None
    return str(read_latest(root)["checkpoint_dir"])


def read_generation_payload(checkpoint_dir: Path) -> dict[str, Any]:
    """Read a committed generation's fixture payload back."""

    checkpoint_dir = Path(checkpoint_dir)
    return {
        "carried": json.loads((checkpoint_dir / "model.pt").read_bytes()),
        "limbs": json.loads((checkpoint_dir / "sampler.pt").read_bytes()),
    }


def read_generation_provenance(checkpoint_dir: Path) -> dict[str, Any]:
    """Read the chain provenance recorded in a generation's manifest."""

    manifest = json.loads((Path(checkpoint_dir) / "manifest.json").read_text(encoding="utf-8"))
    return dict(manifest["provenance"])


# ----------------------------------------------------------------------
# Fresh-process attempt launcher
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptLaunch:
    """Result of launching one attempt as a fresh OS process.

    Parameters
    ----------
    exit_code : int
        The child's exit status. Negative values are ``-signum``.
    receipt_path : pathlib.Path
        Where the child was told to write its receipt. May not exist if the
        child died first -- absence is data, not an error.
    log_path : pathlib.Path
        Captured child stdout/stderr. Always written, so a passing negative
        test still leaves an attributable diagnostic on disk.
    killed_by_parent : bool
        Whether the parent had to deliver the ``SIGKILL``.
    """

    exit_code: int
    receipt_path: Path
    log_path: Path
    killed_by_parent: bool = False


def spawn_attempt(
    workspace: Path,
    *,
    root: Path,
    run_id: str,
    attempt_index: int,
    steps: int,
    total_target: int,
    checkpoint_every: int,
    policy: RestorePolicy | None = None,
    fault: FaultPlan | None = None,
    restore_from: Path | None = None,
    seed: int | None = None,
    timeout: float = 120.0,
) -> AttemptLaunch:
    """Run one continuation attempt in a FRESH OS PROCESS and collect its receipt.

    In-process resume is forbidden throughout this package: a live RNG object
    surviving in the parent's memory makes a continuation gate pass for free,
    which is precisely the DS-A0 failure. Every attempt is therefore a genuine
    ``subprocess.Popen`` in its own session.

    The interpreter is ``sys.executable``, which is absolute. It is never a
    bare name: ``experiments/baselines/test_scaling_probe.py`` line 168 spawns
    a bare ``"python"``, inherits the job ``PATH``, and is the one pre-existing
    red on trunk. ``test_spawn_uses_absolute_interpreter.py`` enforces the rule
    for this package structurally rather than by writer discipline.

    Parameters
    ----------
    workspace : pathlib.Path
        Directory for this attempt's receipt, log and markers.
    root : pathlib.Path
        Shared checkpoint root for the whole chain.
    run_id : str
        Stable logical run id.
    attempt_index : int
        Zero-based attempt ordinal.
    steps : int
        Maximum logical steps this attempt may execute before yielding.
    total_target : int
        Fixed total for the whole chain.
    checkpoint_every : int
        Commit a generation every this many steps.
    policy : RestorePolicy or None, optional
        Restore policy. ``None`` means all limbs enabled.
    fault : FaultPlan or None, optional
        Fault to inject.
    seed : int or None, optional
        Pin this attempt's seed instead of drawing OS entropy. TEST-ONLY, and
        legitimate for exactly one purpose: giving a COLD START a reproducible
        initial condition, so an uninterrupted reference run and a chain can be
        compared at all. Two independent cold starts share no trajectory, and
        comparing them measures nothing.

        Never pin it on a RESUMED attempt. Seed distinctness between a parent
        and its resume is what stops parity from being satisfiable by
        reinitialization, and it is asserted directly in
        ``test_fixture_stream_sensitivity.py``.
    restore_from : pathlib.Path or None, optional
        Explicit committed generation to continue from. ``None`` selects the
        newest valid generation under ``root``. Naming it explicitly lets the
        G0 arm restore generation K of a run that already committed K+L, which
        is what makes the uninterrupted and resumed arms share one trajectory.
    timeout : float, optional
        Outer bound on the child, in seconds.

    Returns
    -------
    AttemptLaunch
        The child's exit status and artifact locations.
    """

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    receipt_path = workspace / "receipt.json"
    log_path = workspace / "attempt.log"
    ready_marker = workspace / "READY"

    policy = RestorePolicy.all_enabled() if policy is None else policy

    argv = [
        # Absolute by construction. `sys.executable` is the running
        # interpreter's full path; a bare name would resolve against PATH.
        sys.executable,
        "-m",
        "tests.helpers.chain_resume_spike.entrypoint",
        "--root",
        str(root),
        "--run-id",
        run_id,
        "--attempt-index",
        str(attempt_index),
        "--steps",
        str(steps),
        "--total-target",
        str(total_target),
        "--checkpoint-every",
        str(checkpoint_every),
        "--receipt-path",
        str(receipt_path),
    ]
    if restore_from is not None:
        argv += ["--restore-from", str(restore_from)]
    if seed is not None:
        if restore_from is not None:
            raise ValueError(
                "refusing to pin the seed of a resumed attempt: seed distinctness "
                "from the parent is what makes the parity gate non-vacuous"
            )
        argv += ["--seed", str(seed)]
    for token in policy.to_tokens():
        argv += ["--disable-limb", token]
    if fault is not None:
        fault_path = workspace / "fault_plan.json"
        fault_path.write_text(json.dumps(fault.to_dict(), sort_keys=True), encoding="utf-8")
        argv += ["--fault-plan-path", str(fault_path)]

    environment = dict(os.environ)
    environment["CHAIN_RESUME_SPIKE_READY_MARKER"] = str(ready_marker)

    killed_by_parent = False
    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(
            argv,
            cwd=str(_REPO_ROOT),
            env=environment,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            if fault is not None and fault.action is FaultAction.SIGKILL_FROM_PARENT:
                killed_by_parent = _kill_when_ready(process, ready_marker, timeout)
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            exit_code = process.wait(timeout=10.0)
        finally:
            # Unconditional: a child that exited on its own can still have
            # left something alive in its group.
            _kill_group(process)

    return AttemptLaunch(
        exit_code=exit_code,
        receipt_path=receipt_path,
        log_path=log_path,
        killed_by_parent=killed_by_parent,
    )


def spawn_publish_only(
    workspace: Path,
    *,
    root: Path,
    generation: Path,
    cwd: Path,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Publish an already-committed generation from a FRESH PROCESS at ``cwd``.

    The path-spelling arm's launcher. ``generation`` and ``root`` are passed
    exactly as written -- relative, absolute, or through a symlink -- and the
    child runs with ``cwd`` as its working directory, so a relative spelling is
    genuinely resolved by the child against a different directory rather than
    being normalised by the parent on the way.

    A separate process is not decoration here. Spelling divergence is precisely
    a per-process property: cwd, relative-versus-absolute argv, and a symlinked
    or remounted scratch root all differ between allocations, which is the real
    situation a chain meets on its second link.

    Returns
    -------
    dict
        The child's JSON report: the spelling it used, the ``content_id``, the
        serialized ``checkpoint_dir``, whether publish succeeded, the exact
        exception type and message if not, and what the resume pointer resolved
        to under the same spelling.
    """

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    output = workspace / "publish_only.json"
    log_path = workspace / "publish_only.log"

    argv = [
        sys.executable,
        "-m",
        "tests.helpers.chain_resume_spike.entrypoint",
        "--root",
        str(root),
        "--run-id",
        "spelling-probe",
        "--attempt-index",
        "0",
        "--steps",
        "0",
        "--total-target",
        "1",
        "--checkpoint-every",
        "1",
        "--receipt-path",
        str(output),
        "--publish-only",
        str(generation),
    ]
    # The repo root goes on PYTHONPATH rather than being the cwd. The whole
    # point of this launcher is that the child's WORKING DIRECTORY differs, and
    # cwd is normally what puts the repo on sys.path -- so the import path has
    # to be supplied explicitly or the child cannot import its own entrypoint.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        str(_REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    )

    with open(log_path, "wb") as log_file:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=environment,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            exit_code = process.wait(timeout=10.0)
        finally:
            _kill_group(process)

    if exit_code != 0 or not output.is_file():
        raise AssertionError(
            f"publish-only probe failed (rc={exit_code}):\n"
            + log_path.read_text(encoding="utf-8", errors="replace")
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _kill_when_ready(
    process: "subprocess.Popen[bytes]", ready_marker: Path, timeout: float
) -> bool:
    """Wait for the child's ready marker, then SIGKILL its group.

    Returns whether the kill was actually delivered. A child that exits before
    posting the marker is reported honestly as not-killed rather than having
    the parent claim a kill it never sent.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_marker.exists():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                return False
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.02)
    return False


def _kill_group(process: "subprocess.Popen[bytes]") -> None:
    """Kill the child's process GROUP, addressed by the child's own PID.

    ``os.killpg(process.pid, ...)`` rather than resolving the PGID first: the
    child was started with ``start_new_session=True``, so it IS its own group
    leader. Resolving the PGID after the child was reaped could name a
    stranger's group once the PID is recycled -- the same reasoning as
    ``tests/helpers/run_id_agreement_harness.py``.
    """

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


__all__ = [
    "BOUNDARY_ORDER",
    "CADENCE_PROBABILITY",
    "CHANNELS",
    "N_PARAMETERS",
    "N_WALKERS",
    "AttemptLaunch",
    "ContinuationSystem",
    "newest_listed_generation",
    "pointer_target_name",
    "resume_generation",
    "publish_generation",
    "read_generation_payload",
    "read_generation_provenance",
    "restore_system",
    "spawn_attempt",
    "spawn_publish_only",
]
