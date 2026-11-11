"""Local pytest config root for facility-specific Cannon tests.

Deliberately empty of fixtures: this directory's own ``pytest.ini`` makes
pytest treat it as its own rootdir, so collection here never reaches up to
the core ``tests/conftest.py`` (typeguard skip logic, logger isolation) --
none of which applies to these subprocess-driven shell-script tests.
"""

from __future__ import annotations
