"""Pure-PyTorch multi-mode tensor SOAP optimizer for TPEN.

Implements SOAP (Shampoo with Adam in the Preconditioner's Eigenbasis) as a
standard :class:`torch.optim.Optimizer`, following Vyas et al. (2024,
arXiv:2409.11321v2) for eigenbasis rotation, subspace-iteration QR refreshes,
and rotated Adam updates, combined with Gupta et al. (2018, arXiv:1802.09568)
multi-mode tensor factor formulation so arbitrary rank ``>= 2`` tensors keep
every axis unmerged.

Design invariants
-----------------
- **Unmerged multi-mode factors**: Every axis ``k`` of a parameter tensor with
  ``ndim >= 2`` and ``1 < shape[k] <= max_precond_dim`` gets its own symmetric
  ``(shape[k], shape[k])`` Gram EMA ``GG[k]`` and orthonormal eigenbasis
  ``Q[k]``. TPEN's rank-3 ``(32, 32, 32)`` tensor-product mixing weights
  therefore maintain three ``(32, 32)`` factors rather than merging axes or
  falling back to 2D-only preconditioning.
- **Float64 preservation**: For ``torch.float64`` parameters, all factor EMAs,
  mode contractions, eigendecompositions, QR refreshes, and moment updates run
  in ``torch.float64``. Precision is never downgraded to ``float32``.
- **Honest first-step update accounting**: Following Vyas et al. (2024)
  Algorithm 1's post-step preconditioner update order, step 1 executes an
  unrotated AdamW update in the initial identity basis ``Q = I`` (avoiding
  exact-SVD ``0 / eps`` roundoff-sign amplification on 2D and rank-deficient
  tensors) and then updates ``GG`` and initializes ``Q_1`` via exact symmetric
  eigendecomposition when ``GG`` is nonzero so step 2+ run in the rotated
  eigenbasis. Step 1 is never silently skipped, so
  :class:`~tpen.training.trainer.VMCTrainer` ``completed_updates`` agrees with
  every active parameter's optimizer step count.
- **Atomic pre-flight gradient validation**: Before mutating any optimizer
  state or parameter, ``step()`` checks every managed parameter with a present
  gradient for complex/non-floating/sparse/non-strided/non-finite entries and
  raises ``RuntimeError`` without touching any parameter or state if any check
  fails. Parameters with ``grad is None`` (such as disconnected even-electron
  TPEN parameters) are skipped without allocating state.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from tpen.dependencies import require_torch

torch = require_torch(feature="SOAP optimizer")

_REQUIRED_STATE_KEYS = ("step", "exp_avg", "exp_avg_sq", "GG", "Q", "mode_eigh_done")
_VALID_STEP_INT_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _validate_real_strided_float_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Validate that a restored state entry is a dense real floating tensor."""

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"SOAP state[{name}] must be a torch.Tensor")
    if value.is_sparse or value.layout != torch.strided:
        raise ValueError(
            f"SOAP state[{name}] must be a dense strided tensor, got layout {value.layout}"
        )
    if value.is_complex() or not value.is_floating_point():
        raise ValueError(
            f"SOAP state[{name}] must be a real floating-point tensor, got dtype {value.dtype}"
        )
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"SOAP state[{name}] shape {tuple(value.shape)} does not match "
            f"expected shape {expected_shape}"
        )
    return value


def _validate_float(
    name: str,
    value: Any,
    *,
    min_value: float | None = None,
    min_inclusive: bool = True,
    max_value: float | None = None,
    max_inclusive: bool = True,
) -> float:
    """Validate and normalize one finite real scalar hyperparameter."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SOAP hyperparameter {name!r} must be a real number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"SOAP hyperparameter {name!r} must be finite, got {value!r}")
    if min_value is not None:
        if min_inclusive and numeric < min_value:
            raise ValueError(f"SOAP hyperparameter {name!r} must be >= {min_value}, got {value!r}")
        if not min_inclusive and numeric <= min_value:
            raise ValueError(f"SOAP hyperparameter {name!r} must be > {min_value}, got {value!r}")
    if max_value is not None:
        if max_inclusive and numeric > max_value:
            raise ValueError(f"SOAP hyperparameter {name!r} must be <= {max_value}, got {value!r}")
        if not max_inclusive and numeric >= max_value:
            raise ValueError(f"SOAP hyperparameter {name!r} must be < {max_value}, got {value!r}")
    return numeric


def _validate_betas(betas: Any) -> tuple[float, float]:
    """Validate and normalize ``betas`` as a 2-tuple of floats in ``[0, 1)``."""

    if isinstance(betas, (str, bytes)) or not isinstance(betas, Sequence) or len(betas) != 2:
        raise ValueError(f"SOAP hyperparameter 'betas' must be a sequence of length 2, got {betas!r}")
    beta1 = _validate_float("betas[0]", betas[0], min_value=0.0, min_inclusive=True, max_value=1.0, max_inclusive=False)
    beta2 = _validate_float("betas[1]", betas[1], min_value=0.0, min_inclusive=True, max_value=1.0, max_inclusive=False)
    return (beta1, beta2)


def _validate_shampoo_beta(value: Any) -> float:
    """Validate ``shampoo_beta`` as ``-1.0`` (alias for ``betas[1]``) or in ``[0, 1)``."""

    numeric = _validate_float("shampoo_beta", value)
    if numeric == -1.0:
        return -1.0
    if 0.0 <= numeric < 1.0:
        return numeric
    raise ValueError(
        "SOAP hyperparameter 'shampoo_beta' must be -1.0 or in [0, 1), "
        f"got {value!r}"
    )


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate a positive integer hyperparameter while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"SOAP hyperparameter {name!r} must be an integer >= 1, got {value!r}")
    return int(value)


def _validate_bool(name: str, value: Any) -> bool:
    """Validate a strict boolean hyperparameter."""

    if type(value) is not bool:
        raise ValueError(f"SOAP hyperparameter {name!r} must be a bool, got {value!r}")
    return value


def _validate_param_group(group: dict[str, Any]) -> None:
    """Validate and normalize one parameter group's hyperparameters in place."""

    group["lr"] = _validate_float("lr", group["lr"], min_value=0.0, min_inclusive=False)
    group["betas"] = _validate_betas(group["betas"])
    group["shampoo_beta"] = _validate_shampoo_beta(group["shampoo_beta"])
    group["eps"] = _validate_float("eps", group["eps"], min_value=0.0, min_inclusive=False)
    group["weight_decay"] = _validate_float("weight_decay", group["weight_decay"], min_value=0.0, min_inclusive=True)
    group["precondition_frequency"] = _validate_positive_int(
        "precondition_frequency", group["precondition_frequency"]
    )
    group["max_precond_dim"] = _validate_positive_int("max_precond_dim", group["max_precond_dim"])
    group["precondition_1d"] = _validate_bool("precondition_1d", group["precondition_1d"])
    group["correct_bias"] = _validate_bool("correct_bias", group["correct_bias"])


def _factor_dtype(param_dtype: torch.dtype) -> torch.dtype:
    """Return the linear-algebra dtype for factors and eigenbases.

    Preserves ``float64`` and ``float32`` exactly. Only lower-precision dtypes
    (such as ``float16`` or ``bfloat16``) are promoted to ``float32`` because
    ``torch.linalg.eigh`` and ``torch.linalg.qr`` require at least single
    precision.
    """

    if param_dtype in (torch.float64, torch.float32):
        return param_dtype
    return torch.float32


def _preconditioned_axes(
    shape: tuple[int, ...],
    *,
    max_precond_dim: int,
    precondition_1d: bool,
) -> list[bool]:
    """Return per-axis boolean flags indicating which modes are preconditioned.

    Parameters
    ----------
    shape : tuple of int
        Parameter tensor shape.
    max_precond_dim : int
        Maximum axis extent eligible for preconditioning.
    precondition_1d : bool
        Whether 1D parameter vectors with ``1 < shape[0] <= max_precond_dim``
        receive a preconditioner factor.

    Returns
    -------
    list of bool
        One flag per axis. Empty tensors (containing a zero-sized axis) and 0D
        scalars have no preconditioned axes.
    """

    ndim = len(shape)
    if ndim == 0:
        return []
    if 0 in shape:
        return [False] * ndim
    if ndim == 1:
        return [bool(precondition_1d and 1 < shape[0] <= max_precond_dim)]
    return [1 < dim <= max_precond_dim for dim in shape]


def _mode_gram(grad: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute the symmetric mode-``axis`` Gram matrix of ``grad`` without merging axes.

    For an order-``r`` tensor ``G`` of shape ``(d_0, ..., d_{r-1})``, returns
    the ``(d_axis, d_axis)`` matrix obtained by contracting ``G`` with itself
    over all axes other than ``axis``.
    """

    if grad.ndim == 1:
        gram = torch.outer(grad, grad)
    else:
        other_axes = [index for index in range(grad.ndim) if index != axis]
        gram = torch.tensordot(grad, grad, dims=(other_axes, other_axes))
    # Symmetrize explicitly so minor floating-point asymmetry from contraction
    # summation order does not accumulate in the factor EMA.
    return 0.5 * (gram + gram.T)


def _eigh_basis(sym: torch.Tensor) -> torch.Tensor:
    """Compute a deterministic descending-eigenvalue orthonormal basis for ``sym``.

    Each eigenvector column is sign-canonicalized so its largest-magnitude
    component is positive (ties broken by the earliest row index via
    ``torch.argmax``), eliminating arbitrary LAPACK sign flips.
    """

    _, eigvecs = torch.linalg.eigh(sym)
    # ``eigh`` returns ascending eigenvalues; SOAP orders eigenvectors descending.
    eigvecs = eigvecs.flip(1)
    pivot_rows = torch.argmax(eigvecs.abs(), dim=0)
    pivot_vals = eigvecs.gather(0, pivot_rows.unsqueeze(0)).squeeze(0)
    signs = torch.where(pivot_vals < 0, -1.0, 1.0).to(dtype=eigvecs.dtype)
    eigvecs = eigvecs * signs.unsqueeze(0)
    if not bool(torch.isfinite(eigvecs).all()):
        raise RuntimeError("SOAP eigendecomposition produced non-finite eigenvectors")
    return eigvecs


def _qr_refresh_basis(sym: torch.Tensor, q_old: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform one subspace-iteration QR refresh on ``q_old`` against ``sym``.

    Uses a shift-invariant diagonal regularization ``S_shifted = sym + c * I``
    with ``c = max(max_eig * sqrt(eps_mach), sqrt(tiny))`` when ``max_eig > 0``.
    Because ``sym`` and ``sym + c * I`` share identical eigenvectors and
    eigenvalue ordering for every ``c > 0``, active subspace iteration is
    preserved to ``O(sqrt(eps_mach))`` while nullspace columns of rank-deficient
    Gram factors keep well-conditioned norm ``c >> eps_mach * max_eig`` instead
    of orthonormalizing floating-point roundoff noise into arbitrary directions.

    Parameters
    ----------
    sym : torch.Tensor
        Symmetrized ``(d, d)`` factor EMA matrix.
    q_old : torch.Tensor
        Previous orthonormal eigenbasis of shape ``(d, d)``.

    Returns
    -------
    q_new : torch.Tensor
        Refreshed orthonormal eigenbasis of shape ``(d, d)`` with deterministic
        column signs (non-negative ``R`` diagonal).
    sort_idx : torch.Tensor
        Column permutation applied before QR so callers can permute the
        corresponding axis of ``exp_avg_sq``.
    """

    # Estimate eigenvalues in the current basis via Rayleigh quotients.
    est_eig = (q_old * (sym @ q_old)).sum(dim=0)
    sort_idx = torch.argsort(est_eig, descending=True, stable=True)
    q_sorted = q_old.index_select(1, sort_idx)
    max_eig = torch.clamp(est_eig.max(), min=0.0)
    if float(max_eig.item()) == 0.0:
        return q_sorted, sort_idx
    finfo = torch.finfo(sym.dtype)
    # Shift by c = max(max_eig * sqrt(eps), sqrt(tiny)) so S + c*I stays
    # scale-invariant across small float32/float64 gradients while keeping
    # nullspace columns well above ULP noise (and avoiding subnormal squaring).
    shift = torch.clamp(max_eig * math.sqrt(finfo.eps), min=math.sqrt(finfo.tiny))
    q_new, r_mat = torch.linalg.qr(sym @ q_sorted + shift * q_sorted)
    signs = torch.where(torch.diagonal(r_mat) < 0, -1.0, 1.0).to(dtype=q_new.dtype)
    q_new = q_new * signs.unsqueeze(0)
    if not bool(torch.isfinite(q_new).all()):
        raise RuntimeError("SOAP QR refresh produced non-finite eigenvectors")
    return q_new, sort_idx


def _project(tensor: torch.Tensor, q_mats: Sequence[torch.Tensor | None]) -> torch.Tensor:
    """Rotate ``tensor`` from parameter coordinates into the multi-mode eigenbasis.

    Iterates across all axes in order, contracting axis ``0`` with column basis
    ``Q_k`` (appending the eigenbasis axis to the end) or rotating unpreconditioned
    axes to the end via ``movedim(0, -1)``. After ``ndim`` steps, axis order is
    restored in eigenbasis coordinates.
    """

    if tensor.ndim == 0 or all(mat is None for mat in q_mats):
        return tensor
    out = tensor
    for mat in q_mats:
        if mat is None:
            out = out.movedim(0, -1)
        else:
            out = torch.tensordot(out, mat, dims=([0], [0]))
    return out


def _project_back(tensor: torch.Tensor, q_mats: Sequence[torch.Tensor | None]) -> torch.Tensor:
    """Rotate ``tensor`` from the multi-mode eigenbasis back to parameter coordinates.

    Exact orthogonal inverse of :func:`_project`: contracting axis ``0`` with
    the second axis of ``Q_k`` restores the original parameter coordinates.
    """

    if tensor.ndim == 0 or all(mat is None for mat in q_mats):
        return tensor
    out = tensor
    for mat in q_mats:
        if mat is None:
            out = out.movedim(0, -1)
        else:
            out = torch.tensordot(out, mat, dims=([0], [1]))
    return out


class SOAP(torch.optim.Optimizer):
    """Multi-mode tensor SOAP optimizer (Vyas et al., 2024; Gupta et al., 2018).

    Maintains per-axis symmetric gradient Gram EMAs ``GG`` and orthonormal
    eigenbases ``Q`` for every eligible axis of rank ``>= 2`` parameter tensors
    (and optionally 1D tensors), applying AdamW moment normalization inside the
    rotated eigenbasis and projecting the normalized update back to parameter
    space.

    Parameters
    ----------
    params : iterable of torch.nn.Parameter or dict
        Model parameters or parameter-group dictionaries.
    lr : float, optional
        Learning rate (must be finite and ``> 0``). Default is ``3e-3``.
    betas : sequence of two float, optional
        Exponential decay rates ``(beta1, beta2)`` for the first moment (stored
        in parameter coordinates) and rotated second moment (stored in
        eigenbasis coordinates). Both must be finite in ``[0, 1)``. Default is
        ``(0.95, 0.95)``.
    shampoo_beta : float, optional
        Exponential decay rate for the per-mode Gram factor EMAs ``GG``. When
        set to ``-1.0`` (default), resolves to ``betas[1]``. Must be ``-1.0`` or
        a finite float in ``[0, 1)``.
    eps : float, optional
        Denominator stability term added after the square root in rotated Adam
        normalization (finite, ``> 0``). Default is ``1e-8``.
    weight_decay : float, optional
        Decoupled AdamW-style weight decay coefficient (finite, ``>= 0``).
        Default is ``0.0``.
    precondition_frequency : int, optional
        Refresh cadence in applied parameter steps (integer ``>= 1``). Per-mode
        Gram EMAs ``GG`` and eigenbases ``Q`` update after each parameter step
        (Vyas et al., 2024, Algorithm 1): exact eigendecomposition initializes
        ``Q`` immediately after the first step where a mode's ``GG`` becomes
        nonzero, and subspace-iteration QR refreshes run post-step on
        subsequent steps divisible by ``precondition_frequency``. Default is
        ``10``.
    max_precond_dim : int, optional
        Maximum axis size eligible for preconditioning (integer ``>= 1``). Axes
        with ``dim <= 1`` or ``dim > max_precond_dim`` are left unrotated.
        Default is ``128``.
    precondition_1d : bool, optional
        Whether 1D parameter vectors with ``1 < len <= max_precond_dim`` are
        preconditioned. When ``False`` (default), 0D and 1D parameters reduce
        to decoupled AdamW.
    correct_bias : bool, optional
        Whether to apply Adam bias correction ``1 - beta1**t`` and
        ``1 - beta2**t`` to the rotated first and second moments. Default is
        ``True``.

    Notes
    -----
    All hyperparameters are validated and stored in ``param_groups`` as
    JSON/CSV-safe Python scalars (with ``betas`` normalized to a 2-tuple of
    floats) for full compatibility with
    :func:`tpen.training.update.carrier_settings` and
    :class:`tpen.training.update.UpdateMethodDescription`.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 3e-3,
        betas: Sequence[float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_precond_dim: int = 128,
        precondition_1d: bool = False,
        correct_bias: bool = True,
    ) -> None:
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "precondition_1d": precondition_1d,
            "correct_bias": correct_bias,
        }
        _validate_param_group(defaults)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add and validate one parameter group."""

        super().add_param_group(param_group)
        try:
            _validate_param_group(self.param_groups[-1])
        except Exception:
            # Roll back partial insertion if per-group overrides fail validation.
            self.param_groups.pop()
            raise

    def state_dict(self) -> dict[str, Any]:
        """Return a deep-copied checkpointable optimizer state dictionary."""

        # Deep-copying prevents external mutation of returned tensors/lists from
        # silently corrupting live factor EMAs or eigenbases.
        return copy.deepcopy(super().state_dict())

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore optimizer state with strict shape, dtype, and device checks.

        Validates and normalizes incoming ``GG`` and ``Q`` factors directly into
        ``_factor_dtype(param.dtype)`` without routing them through
        ``torch.optim.Optimizer.load_state_dict``'s parameter-dtype downcast,
        preserving ``float32`` factors bitwise on ``float16`` / ``bfloat16``
        parameters.

        Parameters
        ----------
        state_dict : Mapping of str to Any
            Serialized optimizer state produced by :meth:`state_dict`.

        Raises
        ------
        TypeError
            If ``state_dict`` is not a mapping.
        ValueError
            If any restored parameter group, state key, factor shape, tensor
            dtype/layout, or ``mode_eigh_done`` metadata is malformed or
            disagrees with the live parameter layout. On failure, the
            optimizer's prior state and parameter groups remain untouched.
        """

        if not isinstance(state_dict, Mapping):
            raise TypeError(f"SOAP.load_state_dict expects a mapping, got {type(state_dict)!r}")

        if "state" not in state_dict or "param_groups" not in state_dict:
            raise ValueError("SOAP state_dict must contain 'state' and 'param_groups' keys")

        raw_state_map = state_dict["state"]
        raw_param_groups = state_dict["param_groups"]
        if not isinstance(raw_state_map, Mapping):
            raise ValueError(
                f"SOAP state_dict['state'] must be a mapping, got {type(raw_state_map)!r}"
            )
        if isinstance(raw_param_groups, (str, bytes)) or not isinstance(
            raw_param_groups, Sequence
        ):
            raise ValueError(
                "SOAP state_dict['param_groups'] must be a sequence, "
                f"got {type(raw_param_groups)!r}"
            )

        saved_groups = list(raw_param_groups)
        live_groups = self.param_groups
        if len(live_groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")

        id_map: dict[Any, torch.Tensor] = {}
        updated_groups: list[dict[str, Any]] = []
        param_config: dict[torch.Tensor, tuple[int, bool]] = {}

        for live_group, saved_group in zip(live_groups, saved_groups, strict=True):
            if not isinstance(saved_group, Mapping):
                raise ValueError(
                    f"SOAP state_dict['param_groups'] entries must be mappings, got {type(saved_group)!r}"
                )
            if "params" not in saved_group:
                raise ValueError("SOAP state_dict param_group is missing required 'params' key")
            saved_params = saved_group["params"]
            if isinstance(saved_params, (str, bytes)) or not isinstance(saved_params, Sequence):
                raise ValueError(
                    "SOAP state_dict param_group['params'] must be a sequence, "
                    f"got {type(saved_params)!r}"
                )
            live_params = live_group["params"]
            if len(live_params) != len(saved_params):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match "
                    "the size of optimizer's group"
                )

            for live_param, saved_id in zip(live_params, saved_params, strict=True):
                try:
                    existing = id_map.get(saved_id)
                except TypeError as exc:
                    raise ValueError(
                        f"SOAP state_dict parameter id must be hashable, got {saved_id!r}"
                    ) from exc
                if existing is not None and existing is not live_param:
                    raise ValueError(
                        f"SOAP state_dict contains duplicate parameter id {saved_id!r}"
                    )
                id_map[saved_id] = live_param

            new_group: dict[str, Any] = {
                key: copy.deepcopy(value)
                for key, value in saved_group.items()
                if key != "params"
            }
            missing_hparams = [key for key in self.defaults if key not in new_group]
            if missing_hparams:
                raise ValueError(
                    f"SOAP state_dict param_group is missing required hyperparameter(s): "
                    f"{missing_hparams}"
                )
            new_group["params"] = list(live_params)
            _validate_param_group(new_group)
            updated_groups.append(new_group)

            cfg = (new_group["max_precond_dim"], new_group["precondition_1d"])
            for live_param in live_params:
                param_config[live_param] = cfg

        new_state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)
        for saved_id, raw_param_state in raw_state_map.items():
            if saved_id not in id_map:
                raise ValueError(
                    f"SOAP state_dict['state'] contains unmapped parameter key {saved_id!r}"
                )
            if not isinstance(raw_param_state, Mapping):
                raise ValueError(
                    f"SOAP state_dict['state'][{saved_id!r}] must be a mapping, "
                    f"got {type(raw_param_state)!r}"
                )
            if len(raw_param_state) == 0:
                raise ValueError(
                    f"SOAP state_dict['state'][{saved_id!r}] must not be empty"
                )
            live_param = id_map[saved_id]
            max_precond_dim, precondition_1d = param_config[live_param]
            new_state[live_param] = self._validate_and_normalize_param_state(
                live_param,
                raw_param_state,
                max_precond_dim=max_precond_dim,
                precondition_1d=precondition_1d,
            )

        # Atomic commit: live optimizer state and groups are updated only after
        # every group and parameter state entry passes full validation.
        self.param_groups = updated_groups
        self.state = new_state

    @staticmethod
    def _validate_and_normalize_param_state(
        param: torch.Tensor,
        raw_state: Mapping[str, Any],
        *,
        max_precond_dim: int,
        precondition_1d: bool,
    ) -> dict[str, Any]:
        """Validate one parameter's restored state and normalize devices/dtypes."""

        missing = [key for key in _REQUIRED_STATE_KEYS if key not in raw_state]
        if missing:
            raise ValueError(f"SOAP restored state is missing required key(s): {missing}")

        step_raw = raw_state["step"]
        if isinstance(step_raw, torch.Tensor):
            if step_raw.is_sparse or step_raw.layout != torch.strided:
                raise ValueError(
                    f"SOAP state['step'] must be a dense strided tensor, got layout {step_raw.layout}"
                )
            if step_raw.ndim != 0:
                raise ValueError(
                    f"SOAP state['step'] must be a 0-D scalar tensor, got shape {tuple(step_raw.shape)}"
                )
            if step_raw.dtype not in _VALID_STEP_INT_DTYPES:
                raise ValueError(
                    f"SOAP state['step'] tensor must have an integer dtype, got {step_raw.dtype}"
                )
            step_int = int(step_raw.item())
        elif not isinstance(step_raw, bool) and isinstance(step_raw, int):
            step_int = int(step_raw)
        else:
            raise ValueError(f"SOAP state['step'] must be an integer scalar, got {step_raw!r}")
        if step_int < 0:
            raise ValueError(f"SOAP state['step'] must be >= 0, got {step_int}")

        normalized: dict[str, Any] = {
            # Match PyTorch Adam/AdamW convention: 0-D CPU int64 tensor.
            "step": torch.tensor(step_int, dtype=torch.int64, device="cpu"),
        }
        for moment_key in ("exp_avg", "exp_avg_sq"):
            tensor = _validate_real_strided_float_tensor(
                repr(moment_key), raw_state[moment_key], tuple(param.shape)
            )
            normalized[moment_key] = tensor.detach().to(device=param.device, dtype=param.dtype).clone()

        precond_flags = _preconditioned_axes(
            tuple(param.shape),
            max_precond_dim=max_precond_dim,
            precondition_1d=precondition_1d,
        )
        factor_dtype = _factor_dtype(param.dtype)

        gg_raw = raw_state["GG"]
        q_raw = raw_state["Q"]
        eigh_done_raw = raw_state["mode_eigh_done"]
        for list_key, container in (("GG", gg_raw), ("Q", q_raw), ("mode_eigh_done", eigh_done_raw)):
            if not isinstance(container, list) or len(container) != param.ndim:
                raise ValueError(
                    f"SOAP state[{list_key!r}] must be a list of length {param.ndim}"
                )

        gg_norm: list[torch.Tensor | None] = []
        q_norm: list[torch.Tensor | None] = []
        eigh_done_norm: list[bool] = []
        for axis, (dim, is_precond) in enumerate(zip(param.shape, precond_flags, strict=True)):
            done_flag = eigh_done_raw[axis]
            if type(done_flag) is not bool:
                raise ValueError(
                    f"SOAP state['mode_eigh_done'][{axis}] must be bool, got {done_flag!r}"
                )

            gg_item = gg_raw[axis]
            q_item = q_raw[axis]
            if is_precond:
                gg_tensor = _validate_real_strided_float_tensor(
                    f"'GG'][{axis}", gg_item, (dim, dim)
                )
                q_tensor = _validate_real_strided_float_tensor(
                    f"'Q'][{axis}", q_item, (dim, dim)
                )
                gg_norm.append(gg_tensor.detach().to(device=param.device, dtype=factor_dtype).clone())
                q_norm.append(q_tensor.detach().to(device=param.device, dtype=factor_dtype).clone())
                eigh_done_norm.append(done_flag)
            else:
                if done_flag:
                    raise ValueError(
                        f"SOAP axis {axis} (dim={dim}) is not preconditioned; "
                        "mode_eigh_done entry must be False"
                    )
                if gg_item is not None or q_item is not None:
                    raise ValueError(
                        f"SOAP axis {axis} (dim={dim}) is not preconditioned; "
                        "GG and Q entries must be None"
                    )
                gg_norm.append(None)
                q_norm.append(None)
                eigh_done_norm.append(False)

        normalized["GG"] = gg_norm
        normalized["Q"] = q_norm
        normalized["mode_eigh_done"] = eigh_done_norm
        return normalized

    def _validate_gradients_before_step(self) -> None:
        """Atomically verify all present gradients before any state mutation."""

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if p.is_complex() or grad.is_complex():
                    raise RuntimeError("SOAP does not support complex parameters or gradients")
                if not p.is_floating_point() or not grad.is_floating_point():
                    raise RuntimeError("SOAP requires real floating-point parameters and gradients")
                if grad.is_sparse or grad.layout != torch.strided:
                    raise RuntimeError("SOAP does not support sparse gradients")
                if not bool(torch.isfinite(grad).all()):
                    raise RuntimeError("SOAP encountered non-finite gradients")

    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        """Perform one SOAP optimization step.

        Parameters
        ----------
        closure : callable or None, optional
            Optional zero-argument closure that re-evaluates the model and
            returns the loss.

        Returns
        -------
        Any
            The loss returned by ``closure`` when supplied, otherwise ``None``.

        Raises
        ------
        RuntimeError
            If any parameter with a gradient is sparse, complex, non-floating,
            or contains non-finite values, or if an eigendecomposition/QR
            refresh produces non-finite basis vectors. Pre-flight gradient
            failures leave all parameters and optimizer state bitwise untouched.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Pass 1: atomic validation across every managed parameter before
        # mutating any state entry or parameter tensor.
        self._validate_gradients_before_step()

        # Pass 2: update factors, refresh/initialize eigenbases, update moments,
        # and step parameters in place.
        with torch.no_grad():
            for group in self.param_groups:
                lr: float = group["lr"]
                beta1, beta2 = group["betas"]
                shampoo_beta: float = group["shampoo_beta"]
                effective_shampoo_beta = beta2 if shampoo_beta < 0.0 else shampoo_beta
                eps: float = group["eps"]
                weight_decay: float = group["weight_decay"]
                precondition_frequency: int = group["precondition_frequency"]
                max_precond_dim: int = group["max_precond_dim"]
                precondition_1d: bool = group["precondition_1d"]
                correct_bias: bool = group["correct_bias"]

                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue

                    state = self.state[p]
                    factor_dtype = _factor_dtype(p.dtype)
                    if len(state) == 0:
                        precond_flags = _preconditioned_axes(
                            tuple(p.shape),
                            max_precond_dim=max_precond_dim,
                            precondition_1d=precondition_1d,
                        )
                        state["step"] = torch.tensor(0, dtype=torch.int64, device="cpu")
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["GG"] = [
                            torch.zeros((dim, dim), device=p.device, dtype=factor_dtype)
                            if flag
                            else None
                            for dim, flag in zip(p.shape, precond_flags, strict=True)
                        ]
                        state["Q"] = [
                            torch.eye(dim, device=p.device, dtype=factor_dtype)
                            if flag
                            else None
                            for dim, flag in zip(p.shape, precond_flags, strict=True)
                        ]
                        state["mode_eigh_done"] = [False] * p.ndim

                    state["step"].add_(1)
                    step_int = int(state["step"].item())

                    gg_list: list[torch.Tensor | None] = state["GG"]
                    q_list: list[torch.Tensor | None] = state["Q"]
                    eigh_done: list[bool] = state["mode_eigh_done"]

                    # Step 1 of Algorithm 1 (Vyas et al., 2024): apply the
                    # rotated AdamW parameter update in the current eigenbasis
                    # Q (initially I on step 1, avoiding exact-SVD 0 / eps
                    # roundoff amplification). Avoid any cast when parameter and
                    # factor dtypes match (always true for float64 and float32).
                    if p.dtype == factor_dtype:
                        q_proj: Sequence[torch.Tensor | None] = q_list
                    else:
                        q_proj = [
                            None if mat is None else mat.to(dtype=p.dtype) for mat in q_list
                        ]

                    grad_proj = _project(grad, q_proj)

                    # First moment lives in original parameter coordinates so it
                    # transports cleanly across basis refreshes; second moment
                    # lives in the active rotated eigenbasis.
                    exp_avg: torch.Tensor = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                    exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad_proj, grad_proj, value=1.0 - beta2)

                    exp_avg_proj = _project(exp_avg, q_proj)
                    if correct_bias:
                        bc1 = 1.0 - (beta1 ** step_int)
                        bc2 = 1.0 - (beta2 ** step_int)
                        denom = (exp_avg_sq / bc2).sqrt().add_(eps)
                        step_dir_proj = (exp_avg_proj / bc1) / denom
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)
                        step_dir_proj = exp_avg_proj / denom

                    step_dir = _project_back(step_dir_proj, q_proj)

                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(step_dir, alpha=-lr)

                    # Step 2 of Algorithm 1: post-step preconditioner update.
                    # Update per-mode Gram factor EMAs in factor_dtype (float64
                    # whenever the parameter is float64), initialize Q via exact
                    # eigh immediately after the first nonzero-gradient step for
                    # each mode, and run QR refreshes on multiples of
                    # precondition_frequency.
                    if any(gg is not None for gg in gg_list):
                        grad_f = grad if grad.dtype == factor_dtype else grad.to(dtype=factor_dtype)
                        for axis, gg in enumerate(gg_list):
                            if gg is None:
                                continue
                            gram = _mode_gram(grad_f, axis)
                            gg.mul_(effective_shampoo_beta).add_(
                                gram, alpha=1.0 - effective_shampoo_beta
                            )

                        for axis, gg in enumerate(gg_list):
                            if gg is None:
                                continue
                            sym = 0.5 * (gg + gg.T)
                            if float(sym.abs().max().item()) == 0.0:
                                continue
                            if not eigh_done[axis]:
                                q_list[axis] = _eigh_basis(sym)
                                eigh_done[axis] = True
                            elif step_int % precondition_frequency == 0:
                                q_old = q_list[axis]
                                assert q_old is not None
                                q_new, sort_idx = _qr_refresh_basis(sym, q_old)
                                q_list[axis] = q_new
                                # Permute exp_avg_sq along this axis to stay
                                # aligned with the reordered eigenvectors.
                                state["exp_avg_sq"] = state["exp_avg_sq"].index_select(axis, sort_idx)

        return loss


__all__ = ["SOAP"]
