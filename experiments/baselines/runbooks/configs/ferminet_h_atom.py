"""FermiNet-codebase configuration for the BL-R1 hydrogen atom.

The upstream atom configuration owns molecule construction.  This thin
configuration pins its physical inputs instead of relying on a launcher's
element-name default: one hydrogen nucleus (Z=1) at its atomic-config origin,
one electron, and the (1, 0) spin sector.  Network and optimisation choices
remain explicit command-line flags in the runbook.
"""

from ferminet.configs import atom


def get_config():
    """Return the upstream atomic configuration pinned to neutral hydrogen."""

    config = atom.get_config()
    config.system.atom = "H"
    config.system.electrons = (1, 0)
    return config
