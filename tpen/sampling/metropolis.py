"""Batched Metropolis sampler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from tpen.accelerator import canonical_device
from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import Walkers, WavefunctionOutput
from tpen.dependencies import require_torch, require_torch_nn
from tpen.sampling.diagnostics import summarize_walker_geometry
from tpen.sampling.moves import GaussianMove
from tpen.sampling.stats import SamplerStats
from tpen.events import EventEmitter
from tpen.physics.system import SystemDescriptor
from tpen.sampling.events import SamplerBurnIn, SamplerProduction

torch = require_torch(feature="Metropolis sampling")
nn = require_torch_nn(feature="Metropolis sampling")


def _canonical_device(device) -> "torch.device":
    """Return a fully indexed accelerator device so ``cuda`` and ``cuda:0`` compare equal.

    Tensors always report an indexed accelerator device (``cuda:0``, ``xpu:0``),
    while configs and callers usually pass the index-less form; ``torch.device``
    treats those as unequal. CPU devices are reported index-less by tensors, so
    they pass through unchanged.

    Backend resolution is owned by :mod:`tpen.accelerator`. This comparison only
    detects device mismatch; per ADR-013 a generator's state is device-bound and
    is never reinterpreted across device types.
    """

    return canonical_device(device, feature="Metropolis sampling")


class MetropolisSampler(nn.Module):
    """Batched, stateful Metropolis-Hastings sampler.

    The sampler owns a persistent Markov chain: it holds the current walkers,
    burns in once, and advances the existing chain on each `collect_samples`
    call unless a reset is requested. It also owns all Markov-chain randomness
    through a sampler-local `torch.Generator` (initial walker positions,
    proposal noise, one-electron index selection, and accept/reject uniforms).
    Sampler code never mutates global Torch RNG state, and the runner/trainer
    must not seed on the sampler's behalf. Walker state and generator state are
    checkpointed together by `mcmc_state_dict`/`load_mcmc_state_dict`.

    Parameters
    ----------
    name : str, optional
        Human-readable sampler name.
    move : torch.nn.Module or None, optional
        Proposal kernel exposing ``propose(walkers, *, generator)`` and
        returning proposed positions plus a proposal log-ratio. The move
        consumes the sampler's generator; it does not own an RNG.
    n_walkers : int, optional
        Default number of walkers to initialize.
    burn_in : int, optional
        Number of equilibration steps run once per chain by `collect_samples`.
    n_steps : int, optional
        Default number of MCMC steps per sampling call.
    proposal_scale : float, optional
        Gaussian proposal scale used when `move` is ``None``.
    seed : int or None, optional
        Seed for the sampler-local generator. Controls only Markov-chain
        randomness, not model parameter initialization.
    n_electrons : int, optional
        Number of electrons per walker.
    spatial_dim : int, optional
        Spatial dimension of each electron coordinate.
    n_up, n_down : int or None, optional
        Spin partition. When both are given, walkers are initialized with the
        corresponding ``+1``/``-1`` spin labels.
    system : object or None, optional
        Configured physical-system descriptor carried into sampler-produced
        batches as immutable construction metadata.
    nuclear_positions : torch.Tensor or None, optional
        Fixed nuclear coordinates with shape ``[n_nuclei, spatial_dim]``.
    nuclear_charges : torch.Tensor or None, optional
        Fixed nuclear charges with shape ``[n_nuclei]``.
    initial_scale : float, optional
        Standard deviation of normally initialized walker positions.
    dtype : torch.dtype or str, optional
        Floating-point dtype for initialized walkers.
    """

    def __init__(
        self,
        name: str = "metropolis",
        move: nn.Module | None = None,
        n_walkers: int = 1024,
        burn_in: int = 100,
        n_steps: int = 10,
        proposal_scale: float = 0.05,
        seed: int | None = None,
        n_electrons: int = 2,
        spatial_dim: int = 3,
        n_up: int | None = None,
        n_down: int | None = None,
        system: object | None = None,
        nuclear_positions: torch.Tensor | None = None,
        nuclear_charges: torch.Tensor | None = None,
        initial_scale: float = 1.0,
        dtype: torch.dtype | str = torch.float64,
    ) -> None:
        super().__init__()
        self.name = name
        self.move = move or GaussianMove(step_size=proposal_scale)
        self.n_walkers = n_walkers
        self.burn_in = burn_in
        self.n_steps = n_steps
        self.proposal_scale = proposal_scale
        self.seed = seed
        self.n_electrons = n_electrons
        self.spatial_dim = spatial_dim
        self.n_up = n_up
        self.n_down = n_down
        self.system = system
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        fixed_positions, fixed_charges = _fixed_nuclear_context(
            nuclear_positions,
            nuclear_charges,
            spatial_dim=spatial_dim,
            dtype=self.dtype,
        )
        self.atomic_configuration: AtomicConfiguration | None = (
            None if fixed_positions is None else AtomicConfiguration(positions=fixed_positions, charges=fixed_charges)
        )
        self.initial_scale = initial_scale
        self.acceptance_rate = 0.0
        self.last_metrics: dict[str, float] = {}

        # Sampler-owned RNG and persistent Markov-chain state.
        self._generator_device = torch.device("cpu")
        self._generator = torch.Generator(device=self._generator_device)
        if self.seed is not None:
            self._generator.manual_seed(int(self.seed))
        self._walkers: Walkers | None = None
        self._has_burned_in = False

    @property
    def walkers(self) -> Walkers | None:
        """Return the current persistent walker state (``None`` before reset)."""

        return self._walkers

    @property
    def has_burned_in(self) -> bool:
        """Return whether the current chain has completed burn-in."""

        return self._has_burned_in

    @property
    def nuclear_positions(self) -> torch.Tensor | None:
        """Return the configured nuclear positions, derived from `atomic_configuration`."""

        return None if self.atomic_configuration is None else self.atomic_configuration.positions

    @property
    def nuclear_charges(self) -> torch.Tensor | None:
        """Return the configured nuclear charges, derived from `atomic_configuration`."""

        return None if self.atomic_configuration is None else self.atomic_configuration.charges

    def initialize(self, n_walkers: int | None = None, device=None) -> Walkers:
        """Initialize normally distributed walkers using the sampler generator.

        Parameters
        ----------
        n_walkers : int or None, optional
            Number of walkers to initialize. If ``None``, `self.n_walkers` is
            used.
        device : torch.device, str, or None, optional
            Optional device assertion. Must match the sampler generator device;
            use `reset` to move the chain to a new device.

        Returns
        -------
        Walkers
            Walker state with positions shaped ``[n_walkers, n_electrons,
            spatial_dim]``.
        """

        self._require_device(device)
        n_walkers = n_walkers or self.n_walkers
        positions = self.initial_scale * torch.randn(
            n_walkers,
            self.n_electrons,
            self.spatial_dim,
            device=self._generator_device,
            dtype=self.dtype,
            generator=self._generator,
        )
        spins = _default_spins(
            n_up=self.n_up,
            n_down=self.n_down,
            n_electrons=self.n_electrons,
            n_walkers=n_walkers,
            device=self._generator_device,
            dtype=self.dtype,
        )
        return Walkers(
            positions=positions,
            spins=spins,
            atomic_configuration=_configuration_on(self.atomic_configuration, self._generator_device, self.dtype),
            aux={"system": self.system} if self.system is not None else {},
        )

    def reset(self, n_walkers: int | None = None, device=None) -> Walkers:
        """Re-seed the generator and start a fresh, un-burned-in chain.

        Parameters
        ----------
        n_walkers : int or None, optional
            Number of walkers to initialize.
        device : torch.device, str, or None, optional
            Device for the new chain and generator. Defaults to the current
            generator device.

        Returns
        -------
        Walkers
            The freshly initialized walker state.
        """

        target_device = _canonical_device(device) if device is not None else self._generator_device
        self._generator_device = target_device
        self._generator = torch.Generator(device=target_device)
        if self.seed is not None:
            self._generator.manual_seed(int(self.seed))
        self._walkers = self.initialize(n_walkers=n_walkers)
        self._has_burned_in = False
        return self._walkers

    def _require_device(self, device) -> None:
        if device is not None and _canonical_device(device) != self._generator_device:
            raise ValueError(
                f"sampler generator is on {self._generator_device}; cannot operate on "
                f"{_canonical_device(device)}. Call reset(device=...) to move the chain."
            )

    def _evaluate(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        batch = walkers.make_batch()
        with torch.no_grad():
            output = model(batch)
        if not isinstance(output, WavefunctionOutput):
            raise TypeError(f"Wavefunction model must return WavefunctionOutput, got {type(output)!r}")
        logabs = output.logabs
        sign = output.sign
        if logabs.shape != (walkers.batch_size,):
            raise ValueError(f"Model logabs must have shape [{walkers.batch_size}], got {tuple(logabs.shape)}")
        if sign.shape != (walkers.batch_size,):
            raise ValueError(f"Model sign must have shape [{walkers.batch_size}], got {tuple(sign.shape)}")
        return logabs, sign

    def _propose(self, model, walkers: Walkers) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        if not hasattr(self.move, "propose"):
            raise TypeError("MetropolisSampler move must expose propose(walkers)")
        proposals, log_q_ratio = self.move.propose(walkers, generator=self._generator)
        if proposals.shape != walkers.positions.shape:
            raise ValueError(f"Proposal positions must have shape {tuple(walkers.positions.shape)}, got {tuple(proposals.shape)}")
        if log_q_ratio.shape != (walkers.batch_size,):
            raise ValueError(f"Proposal log-ratio must have shape [{walkers.batch_size}], got {tuple(log_q_ratio.shape)}")
        return proposals, log_q_ratio

    def step(self, model, walkers: Walkers) -> Walkers:
        """Run one Metropolis-Hastings step.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        walkers : Walkers
            Current walker state.

        Returns
        -------
        Walkers
            Updated walker state with cached wavefunction values and sampler
            diagnostics in ``aux``.

        Notes
        -----
        This method trusts any populated ``walkers.logabs``/``sign`` cache and
        does not re-evaluate it. ``collect_samples`` is the entry point that
        refreshes retained walkers against the supplied model once per call.
        Callers that reuse walkers after a parameter or model change must call
        ``Walkers.update_cache(model)`` or invalidate the cache first, for
        example with ``with_positions(..., invalidate_cache=True)``.
        """

        self._require_device(walkers.device)
        current_logabs = walkers.logabs
        current_sign = walkers.sign
        if current_logabs is None or current_sign is None:
            current_logabs, current_sign = self._evaluate(model, walkers)
        proposals, log_q_ratio = self._propose(model, walkers)
        proposal_walkers = Walkers(
            positions=proposals,
            spins=walkers.spins,
            atomic_configuration=walkers.atomic_configuration,
            aux=dict(walkers.aux),
        )
        proposed_logabs, proposed_sign = self._evaluate(model, proposal_walkers)
        log_accept_ratio = torch.nan_to_num(2.0 * (proposed_logabs - current_logabs) + log_q_ratio, nan=-torch.inf)
        log_accept = torch.clamp(log_accept_ratio, max=0.0)
        uniforms = torch.rand(
            log_accept.shape,
            device=log_accept.device,
            dtype=log_accept.dtype,
            generator=self._generator,
        )
        accepted = torch.log(uniforms.clamp_min(1e-12)) < log_accept
        accepted_mask = accepted.view(-1, 1, 1)
        positions = torch.where(accepted_mask, proposals, walkers.positions)
        logabs = torch.where(accepted, proposed_logabs, current_logabs)
        sign = torch.where(accepted, proposed_sign, current_sign)
        self.acceptance_rate = accepted.to(dtype=torch.float32).mean().item()
        self.last_metrics = {
            "acceptance_rate": self.acceptance_rate,
            "mean_logabs": float(logabs.detach().mean().item()),
        }
        if hasattr(self.move, "step_size"):
            self.last_metrics["proposal_scale"] = float(self.move.step_size)
        return Walkers(
            positions=positions.detach(),
            logabs=logabs.detach(),
            sign=sign.detach(),
            spins=None if walkers.spins is None else walkers.spins.detach(),
            atomic_configuration=walkers.atomic_configuration,
            aux={
                **walkers.aux,
                "accepted": accepted.detach(),
                "log_accept_ratio": log_accept_ratio.detach(),
                "acceptance_rate": self.acceptance_rate,
            },
        )

    def sample(self, model, walkers: Walkers, n_steps: int | None = None) -> Walkers:
        """Run multiple Metropolis-Hastings steps.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        walkers : Walkers
            Current walker state.
        n_steps : int or None, optional
            Number of MCMC steps. If ``None``, `self.n_steps` is used.

        Returns
        -------
        Walkers
            Walker state after sampling. ``self.acceptance_rate`` is the mean
            acceptance rate over all steps in this call.

        Notes
        -----
        This method trusts any populated ``walkers.logabs``/``sign`` cache and
        does not re-evaluate it. ``collect_samples`` is the entry point that
        refreshes retained walkers against the supplied model once per call.
        Callers that reuse walkers after a parameter or model change must call
        ``Walkers.update_cache(model)`` or invalidate the cache first, for
        example with ``with_positions(..., invalidate_cache=True)``.
        """

        total_steps = self.n_steps if n_steps is None else n_steps
        if total_steps < 0:
            raise ValueError("n_steps must be non-negative")
        acceptance_sum = 0.0
        for _ in range(total_steps):
            walkers = self.step(model, walkers)
            acceptance_sum += float(walkers.aux["acceptance_rate"])
        if total_steps:
            self.acceptance_rate = acceptance_sum / total_steps
            self.last_metrics["acceptance_rate"] = self.acceptance_rate
        walkers.aux["sample_acceptance_rate"] = self.acceptance_rate
        return walkers

    def collect_samples(
        self,
        model,
        *,
        reset: bool = False,
        device=None,
        emitter: EventEmitter | None,
    ) -> tuple[Walkers, SamplerStats]:
        """Advance the persistent chain and draw production samples.

        On the first call (or when ``reset=True``) the chain is initialized and
        burned in once; subsequent calls advance the existing walkers without
        re-burning. The sampler owns its walkers and RNG across calls.

        Parameters
        ----------
        model : callable
            Wavefunction model returning `WavefunctionOutput`.
        reset : bool, optional
            Force a fresh, re-seeded, un-burned-in chain.
        device : torch.device, str, or None, optional
            Target device for the chain. On a fresh chain this selects the
            device; on an existing chain a mismatching device raises.

        Returns
        -------
        tuple
            Pair ``(walkers, stats)`` where ``walkers`` holds the final samples
            and ``stats`` is the typed `SamplerStats` record for logging.
        """

        if reset or self._walkers is None:
            self.reset(device=device)
        else:
            self._require_device(device)
            # Persistent walkers may outlive a model/parameter update.  Refresh
            # their current values once at this public boundary so acceptance
            # compares values from one model version without consuming RNG.
            with torch.no_grad():
                self._walkers = self._walkers.update_cache(model)
        if not self._has_burned_in and self.burn_in:
            if emitter is None:
                self._walkers = self.sample(model, self._walkers, self.burn_in)
            else:
                with emitter.scope(SamplerBurnIn(n_steps=self.burn_in)):
                    self._walkers = self.sample(model, self._walkers, self.burn_in)
            self._has_burned_in = True
        if emitter is None:
            self._walkers = self.sample(model, self._walkers, self.n_steps)
        else:
            with emitter.scope(SamplerProduction(n_steps=self.n_steps)):
                self._walkers = self.sample(model, self._walkers, self.n_steps)
        stats = SamplerStats(
            acceptance_rate=self.acceptance_rate,
            n_walkers=self._walkers.batch_size,
            burn_in=self.burn_in,
            n_steps=self.n_steps,
            proposal_scale=getattr(self.move, "step_size", self.proposal_scale),
            # Geometry diagnostics describe the production samples actually
            # returned to the caller; phase namespacing (train/validation/eval)
            # is owned by whoever logs these stats.
            geometry=summarize_walker_geometry(self._walkers),
            seed=self.seed,
        )
        return self._walkers, stats

    def mcmc_state_dict(self) -> dict[str, Any]:
        """Return checkpointable Markov-chain and RNG state.

        This is intentionally separate from `torch.nn.Module.state_dict`, which
        keeps its normal module-parameter semantics. MCMC state (walkers,
        burn-in flag, running acceptance, and generator state) is persisted here
        instead so checkpointing does not abuse the standard module API.
        `atomic_configuration` is serialized explicitly and unconditionally
        (not only via `walkers`), so the configured system round-trips even
        before the chain has ever been reset/initialized.
        """

        state = {
            "walkers": self._walkers,
            "atomic_configuration": self.atomic_configuration,
            "has_burned_in": self._has_burned_in,
            "acceptance_rate": float(self.acceptance_rate),
            "generator_state": self._generator.get_state(),
            "generator_device": str(self._generator_device),
        }
        # Keep the historical system=None state byte-compatible: no new key
        # is emitted for the default path.  A configured descriptor is also
        # stored at the sampler boundary so a checkpoint taken before reset()
        # retains the sector binding, not just the copy in walkers.aux.
        if isinstance(self.system, SystemDescriptor):
            state["system"] = self.system
        return state

    def load_mcmc_state_dict(self, state: Mapping[str, Any], *, device=None) -> None:
        """Restore Markov-chain and RNG state from `mcmc_state_dict`.

        Recreates the generator on the requested runtime device when provided,
        otherwise on the checkpointed device. Exact generator state is restored
        only when the checkpoint and target generator devices match; CPU/CUDA
        generators do not share a portable state representation.

        The checkpoint's canonical `atomic_configuration` entry is the source
        of truth; a restored `walkers.atomic_configuration` (if present) must
        agree with it exactly, guarding against a hand-built or malformed
        checkpoint carrying divergent context (this cannot arise from
        `mcmc_state_dict`, which always serializes the same reference for
        both). The canonical entry is adopted only when this sampler is not
        already configured (legacy Hooke neither/neither stays `None` when
        the checkpoint also carries none). When this sampler is already
        configured, a present canonical entry must agree exactly
        (`AtomicConfiguration.__eq__`); a mismatch raises `ValueError` rather
        than silently overriding the constructor-owned system. A checkpoint
        carrying no context never clears an already-configured sampler.
        """

        restored_configuration = state.get("atomic_configuration")
        restored_walkers = state["walkers"]
        if restored_walkers is not None and restored_walkers.atomic_configuration is not None:
            if restored_configuration is None:
                restored_configuration = restored_walkers.atomic_configuration
            elif restored_walkers.atomic_configuration != restored_configuration:
                raise ValueError(
                    "MetropolisSampler checkpoint's walkers.atomic_configuration does not match "
                    "its canonical atomic_configuration entry"
                )

        restored_system = state.get("system")
        if restored_system is not None and not isinstance(restored_system, SystemDescriptor):
            raise ValueError("MetropolisSampler checkpoint carries an invalid SystemDescriptor")
        if restored_walkers is not None and "system" in restored_walkers.aux:
            walker_system = restored_walkers.aux["system"]
            if isinstance(walker_system, SystemDescriptor):
                if restored_system is None:
                    restored_system = walker_system
                elif walker_system.system_id() != restored_system.system_id():
                    raise ValueError(
                        "MetropolisSampler checkpoint's walkers.system does not match its canonical system"
                    )
            elif restored_system is not None:
                raise ValueError(
                    "MetropolisSampler checkpoint's walkers.system does not match its canonical system"
                )
        if isinstance(self.system, SystemDescriptor) and restored_system is not None:
            if self.system.system_id() != restored_system.system_id():
                raise ValueError(
                    "MetropolisSampler is configured with a SystemDescriptor that does not "
                    "match the restored checkpoint's system"
                )
        elif self.system is None and restored_system is not None:
            self.system = restored_system

        if self.atomic_configuration is None:
            self.atomic_configuration = restored_configuration
        elif restored_configuration is not None and restored_configuration != self.atomic_configuration:
            raise ValueError(
                "MetropolisSampler is configured with an atomic_configuration that does not "
                "match the restored checkpoint's atomic_configuration"
            )
        checkpoint_device = _canonical_device(state["generator_device"])
        self._generator_device = _canonical_device(device) if device is not None else checkpoint_device
        self._generator = torch.Generator(device=self._generator_device)
        if self._generator_device == checkpoint_device:
            self._generator.set_state(state["generator_state"])
        elif self.seed is not None:
            self._generator.manual_seed(int(self.seed))
        self.atomic_configuration = _configuration_on(self.atomic_configuration, self._generator_device, self.dtype)
        walkers = state["walkers"]
        self._walkers = None if walkers is None else walkers.to(device=self._generator_device)
        if self._walkers is not None and isinstance(self.system, SystemDescriptor):
            # Rebind the resumed chain to the validated sampler-owned
            # descriptor.  This makes reset() and continuation retain the
            # sector guard even when a legacy checkpoint omitted aux metadata.
            self._walkers = replace(
                self._walkers,
                aux={**self._walkers.aux, "system": self.system},
            )
        if self._walkers is not None and self.atomic_configuration is not None:
            # Normalize to the single resolved reference so reset()/inference
            # never see two distinct-but-equal AtomicConfiguration instances.
            self._walkers = replace(self._walkers, atomic_configuration=self.atomic_configuration)
        self._has_burned_in = bool(state["has_burned_in"])
        self.acceptance_rate = float(state.get("acceptance_rate", 0.0))


def _configuration_on(
    configuration: AtomicConfiguration | None,
    device: "torch.device",
    dtype: "torch.dtype",
) -> AtomicConfiguration | None:
    """Return `configuration` materialized on `device`/`dtype`, by reference when already there.

    `AtomicConfiguration.to()` always constructs a new instance, even when
    the device and dtype already match. The sampler owns one persistent
    `atomic_configuration`; callers (`initialize`, `load_mcmc_state_dict`)
    must carry that exact object by reference whenever no conversion is
    actually needed, so it is not silently replaced by an equal-but-distinct
    instance on every chain reset or restore.
    """

    if configuration is None:
        return None
    if configuration.device == _canonical_device(device) and configuration.dtype == dtype:
        return configuration
    return configuration.to(device=device, dtype=dtype)


def _fixed_nuclear_context(
    nuclear_positions: torch.Tensor | None,
    nuclear_charges: torch.Tensor | None,
    *,
    spatial_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate immutable nuclear metadata owned by a sampler.

    Nuclear positions and charges form one atomic context: callers either
    provide both tensors or neither. The sampler keeps this canonical CPU
    representation and materializes it on the persistent-chain device when it
    creates walkers.
    """

    if (nuclear_positions is None) != (nuclear_charges is None):
        raise ValueError("MetropolisSampler nuclear_positions and nuclear_charges must be provided together")
    if nuclear_positions is None:
        return None, None
    positions = torch.as_tensor(nuclear_positions, dtype=dtype, device="cpu").detach().clone()
    charges = torch.as_tensor(nuclear_charges, dtype=dtype, device="cpu").detach().clone()
    if positions.ndim != 2 or positions.shape[1] != spatial_dim:
        raise ValueError("MetropolisSampler nuclear_positions must have shape [n_nuclei, spatial_dim]")
    if charges.ndim != 1:
        raise ValueError("MetropolisSampler nuclear_charges must have shape [n_nuclei]")
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("MetropolisSampler nuclear_positions and nuclear_charges must agree on n_nuclei")
    if not torch.isfinite(positions).all() or not torch.isfinite(charges).all():
        raise ValueError("MetropolisSampler nuclear context must be finite")
    return positions, charges


def _default_spins(
    *,
    n_up: int | None,
    n_down: int | None,
    n_electrons: int,
    n_walkers: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Return repeated spin labels from a spin partition.

    Parameters
    ----------
    n_up, n_down : int or None
        Spin partition. If either is ``None``, no spin labels are produced.
    n_electrons : int
        Number of electrons; must equal ``n_up + n_down`` when both are given.
    n_walkers : int
        Number of walkers.
    device : torch.device, str, or None
        Target device for the spin tensor.
    dtype : torch.dtype
        Target dtype for the spin tensor.

    Returns
    -------
    torch.Tensor or None
        Spin labels with shape ``[n_walkers, n_electrons]`` when a partition is
        available, otherwise ``None``.
    """

    if n_up is None or n_down is None:
        return None
    spins = torch.tensor([1.0] * n_up + [-1.0] * n_down, device=device, dtype=dtype)
    if spins.numel() != n_electrons:
        raise ValueError("Spin partition must match n_electrons")
    return spins.unsqueeze(0).expand(n_walkers, -1).clone()
