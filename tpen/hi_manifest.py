"""The helium-importance evaluation manifest -- the only reference holder.

This module is the deliberate counterpart to :mod:`tpen.hi_schema`. That module
refuses a reference anywhere on the training path; this one is the single place
a reference is allowed to live.

The separation is structural, not conventional
----------------------------------------------
``tpen.hi_schema`` does not import this module, and neither does ``tpen.run``.
A reference value is therefore not merely *unused* by a training process -- it
is not reachable from one. ``tests/unit/test_hi_reference_separation.py``
asserts that import boundary, because a boundary nobody checks is a comment.

Why it matters that the value is absent rather than ignored: a reference
reachable during training is an arm-selection hazard. Any quantity that can be
compared against it mid-run can be used to choose between arms on accuracy, and
whoever does so need not intend to. Physical absence removes the possibility
instead of relying on discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "HI_EVALUATION_SCHEMA",
    "ReferenceEnergy",
    "load_evaluation_manifest",
    "reference_energy",
]


# Durable external identifier, written into manifest files.
HI_EVALUATION_SCHEMA = "tpen.hi.evaluation.v1"


@dataclass(frozen=True)
class ReferenceEnergy:
    """A qualified reference energy for one system.

    Parameters
    ----------
    energy : float
        The reference value.
    qualification : str
        What the number actually is, e.g. ``"infinite_mass_nonrelativistic"``.
        Required rather than optional: an unqualified reference energy invites
        comparison against a value computed under different physics, and the
        resulting discrepancy would be read as model error.
    units : str
        Energy units, e.g. ``"hartree"``.
    system_id : str
        The system the reference describes.
    """

    energy: float
    qualification: str
    units: str
    system_id: str


def load_evaluation_manifest(path: str | Path) -> DictConfig:
    """Load an HI evaluation manifest.

    Parameters
    ----------
    path : str or Path
        Path to the manifest YAML.

    Returns
    -------
    DictConfig
        The loaded manifest.

    Raises
    ------
    ValueError
        When the file does not declare :data:`HI_EVALUATION_SCHEMA`. A training
        config loaded here by mistake would otherwise be read as a manifest
        with no reference in it, which looks like success.
    """

    cfg = OmegaConf.load(Path(path))
    declared = OmegaConf.select(cfg, "schema", default=None)
    if declared != HI_EVALUATION_SCHEMA:
        raise ValueError(
            f"{path} declares schema {declared!r}, expected {HI_EVALUATION_SCHEMA!r}. "
            "An evaluation manifest is the only permitted reference holder; refusing "
            "to read an unlabelled file as one"
        )
    return cfg


def reference_energy(manifest: DictConfig) -> ReferenceEnergy:
    """Extract the qualified reference energy from a loaded manifest.

    Parameters
    ----------
    manifest : DictConfig
        A manifest from :func:`load_evaluation_manifest`.

    Returns
    -------
    ReferenceEnergy
        The reference and its qualification.

    Raises
    ------
    ValueError
        When any field is missing. Every field is required, including the
        qualification: a reference energy without one cannot be compared
        honestly, because nothing records which physics produced it.
    """

    missing = [
        field
        for field in ("energy", "qualification", "units", "system_id")
        if OmegaConf.select(manifest, f"reference.{field}", default=None) is None
    ]
    if missing:
        raise ValueError(f"evaluation manifest is missing reference field(s): {sorted(missing)}")

    return ReferenceEnergy(
        energy=float(OmegaConf.select(manifest, "reference.energy")),
        qualification=str(OmegaConf.select(manifest, "reference.qualification")),
        units=str(OmegaConf.select(manifest, "reference.units")),
        system_id=str(OmegaConf.select(manifest, "reference.system_id")),
    )
