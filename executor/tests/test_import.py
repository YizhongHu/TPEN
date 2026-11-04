"""Import-level checks for the standalone executor project."""

import tpen_executor


def test_public_metadata_is_available() -> None:
    """The project exposes version metadata without constructing a runtime."""
    assert tpen_executor.__version__ == "0.1.0"
    assert "__version__" in tpen_executor.__all__
