"""Bounded two-coefficient log-Jastrow factor built on the B coordinate.

This is the CD4 half of the minimal B wavefunction: a correlation factor
expressed entirely in the bounded distance ``q`` that
:func:`tpen.nn.basis.bounded_distance` already provides, so it inherits that
feature's two load-bearing properties rather than re-deriving them. ``q`` is
computed from SQUARED distances and is therefore smooth at coincidence, and it
is bounded in ``[0, 1)`` and so adds no exponential tail slope of its own.

`BoundedTwoCoefficientJastrow` is a `tpen.nn.factor.LogAmplitudeFactor`: it
contributes additively to ``log |psi|``, which means it multiplies the
wavefunction by the strictly positive ``exp(J)``. A strictly positive factor
cannot move a node or flip a sign, so this factor changes correlation without
touching the sign structure the readout is responsible for.
"""

from __future__ import annotations

from tpen.data.batch import ElectronBatch
from tpen.dependencies import require_torch, require_torch_nn
from tpen.nn.basis import bounded_distance
from tpen.nn.factor import LogAmplitudeFactor

torch = require_torch(feature="TPEN Jastrow modules")
nn = require_torch_nn(feature="TPEN Jastrow modules")


class BoundedTwoCoefficientJastrow(LogAmplitudeFactor):
    r"""Two-coefficient log-Jastrow in the bounded distance ``q``.

    For the two-electron system the consolidated authority specifies

    .. math::

        J = \theta_s \, q_{12} \frac{q_1 + q_2}{2}
          + \theta_t \, q_{12} (q_1 - q_2)^2 ,

    with both coefficients starting at zero and both trained.

    **The authority states the two-electron case; the pair sum below is this
    module's generalization of it, not the authority's.** Recorded explicitly so
    a later reader does not attribute the choice to the design document. This
    class evaluates

    .. math::

        J = \sum_{i<j} \left[
              \theta_s \, q_{ij} \frac{q_i + q_j}{2}
            + \theta_t \, q_{ij} (q_i - q_j)^2
            \right],

    which reduces EXACTLY to the authority's expression at ``n = 2`` -- the
    single ordered pair ``(1, 2)`` -- and remains invariant under electron
    permutation above it, because each summand is symmetric in ``i`` and ``j``
    and the sum runs over unordered pairs.

    Here ``q_i = q(|R_i|)`` is the bounded distance of electron ``i`` from the
    origin and ``q_{ij} = q(|R_i - R_j|)`` the bounded separation of the pair,
    matching `tpen.nn.basis.BoundedDistanceBasis` exactly: for a single atom
    placed at the origin ``q_i`` is the bounded electron-nucleus distance. This
    was read off the basis implementation rather than assumed, since the whole
    factor would be silently mis-centred otherwise.

    Notes
    -----
    **What this factor deliberately does NOT contain.** No learned length
    scale, no MLP, no angular term, and no amplitude clipping. The length is a
    fixed constructor value rather than a trained one.

    It is NOT shared with the basis, and an earlier version of this sentence
    said it was. Each stores its own; see the ``length`` parameter for the
    unenforced matching obligation that creates.

    **Why it cannot spoil the Kato cusp.** Both summands carry a factor of
    ``q_{ij}``, and ``q(r) = r^2 / (l^2 + r^2)`` has ``q(0) = 0`` and
    ``q'(0) = 0``. So ``J`` and its radial derivative both vanish at
    coalescence and the analytic cusp supplied by
    `tpen.nn.cusp.ElectronElectronCusp` is untouched at any coefficient value,
    before or after an optimizer step.

    The correct statement of that property is about the SPHERICAL AVERAGE of
    the derivative at coalescence, which is what the Kato condition constrains.
    It is NOT the stronger claim that every directional derivative vanishes;
    that stronger claim is false in general and must not be tested for.

    **Both coefficients are unconstrained in sign.** ``theta_s`` and
    ``theta_t`` may take either sign, and both start at exactly zero, so the
    factor begins as the identity (``exp(0) = 1``) and any deviation from it is
    learned rather than assumed.

    Parameters
    ----------
    mean_coefficient : float, optional
        ``theta_s``, multiplying ``q_ij (q_i + q_j) / 2``. Defaults to ``0.0``.
    difference_coefficient : float, optional
        ``theta_t``, multiplying ``q_ij (q_i - q_j)^2``. Defaults to ``0.0``.
    length : float, optional
        The bounded-distance length ``l`` in bohr. The study's basis uses
        ``1.0``, and this factor defaults to the same value -- but that is a
        matching DEFAULT, not a shared setting: the two are independent
        parameters that happen to agree until someone changes one.

        **If this factor is composed alongside the basis the two must match,
        and NOTHING ENFORCES THAT.** This factor and
        `tpen.nn.basis.BoundedDistanceBasis` each store their own ``length``
        and compute the same ``q`` formula from it; there is no shared owner
        and no schema rule comparing them. A model composed with ``1.0`` here
        and ``2.0`` there is admitted, and ``q`` would silently mean two
        different things in one wavefunction.

        Not a live defect: no shipped configuration composes this factor at
        all. It is an obligation on whoever composes it, tracked as item
        ``7bae63d8-f3a1-4d63-b240-1bd12d823e04``, and stated here because that
        is where someone setting the value will be looking.
    trainable : bool, optional
        Whether the two coefficients are optimized. When ``False`` they are
        fixed non-persistent buffers at the constructor values, so the module
        contributes no checkpoint state; when ``True`` it contributes
        ``theta_s`` and ``theta_t`` to the state dict.
    """

    def __init__(
        self,
        mean_coefficient: float = 0.0,
        difference_coefficient: float = 0.0,
        length: float = 1.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if not length > 0.0:
            raise ValueError(f"length must be positive, got {length}")
        self.length = float(length)
        self.trainable = bool(trainable)
        # float64 storage matches the electron-nucleus curvature laws in
        # `tpen.nn.cusp`; both are cast to the batch dtype at use.
        if self.trainable:
            self.theta_s = nn.Parameter(torch.tensor(float(mean_coefficient), dtype=torch.float64))
            self.theta_t = nn.Parameter(
                torch.tensor(float(difference_coefficient), dtype=torch.float64)
            )
        else:
            self.register_buffer(
                "_theta_s",
                torch.tensor(float(mean_coefficient), dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                "_theta_t",
                torch.tensor(float(difference_coefficient), dtype=torch.float64),
                persistent=False,
            )

    @property
    def mean_coefficient(self) -> torch.Tensor:
        """Return ``theta_s``, the coefficient of the pair-mean term."""

        return self.theta_s if self.trainable else self._theta_s

    @property
    def difference_coefficient(self) -> torch.Tensor:
        """Return ``theta_t``, the coefficient of the squared-difference term."""

        return self.theta_t if self.trainable else self._theta_t

    def factor_value(self, batch: ElectronBatch) -> torch.Tensor:
        """Return the log-Jastrow contribution for a flattened batch.

        Parameters
        ----------
        batch : ElectronBatch
            Flattened electron batch.

        Returns
        -------
        torch.Tensor
            Contribution to ``log |psi|`` with shape ``[batch]``.
        """

        positions = batch.positions

        # Kept squared throughout, exactly as `BoundedDistanceBasis` does: `r`
        # has an infinite gradient at the origin, so a chain rule through
        # `sqrt` would produce a NaN precisely at coincidence and on the
        # nucleus -- the configurations this factor exists to describe.
        radius_squared = positions.square().sum(dim=-1)
        q_single = bounded_distance(radius_squared, self.length)

        # Explicit difference rather than the expanded
        # |R_i|^2 + |R_j|^2 - 2 R_i.R_j, which loses precision by cancellation
        # exactly where the electrons are close.
        separation = positions.unsqueeze(-2) - positions.unsqueeze(-3)
        q_pair = bounded_distance(separation.square().sum(dim=-1), self.length)

        q_i = q_single.unsqueeze(-1)
        q_j = q_single.unsqueeze(-2)
        theta_s = self.mean_coefficient.to(device=positions.device, dtype=positions.dtype)
        theta_t = self.difference_coefficient.to(device=positions.device, dtype=positions.dtype)

        contribution = q_pair * (
            theta_s * 0.5 * (q_i + q_j) + theta_t * (q_i - q_j).square()
        )

        # Unordered pairs: the strict upper triangle counts each pair once and
        # drops the diagonal, whose q_ii is 0 anyway. Masking rather than
        # indexing keeps the shape static, which matters for a zero-electron
        # batch -- `n = 0` yields an empty [batch, 0, 0] and sums to zero
        # rather than failing.
        upper = torch.triu(torch.ones_like(contribution, dtype=torch.bool), diagonal=1)
        output = contribution.masked_fill(~upper, 0.0).sum(dim=(-2, -1))
        assert output.shape == (batch.batch_size,)
        return output

    def scalar_diagnostics(self) -> dict[str, float]:
        """Return both coefficients.

        Unlike the softplus-constrained range parameters elsewhere in the
        model, these are stored directly and have no separate raw axis, so
        there is nothing further to report beside them.
        """

        return {
            "jastrow_mean_coefficient": float(self.mean_coefficient.item()),
            "jastrow_difference_coefficient": float(self.difference_coefficient.item()),
        }


__all__ = ["BoundedTwoCoefficientJastrow"]
