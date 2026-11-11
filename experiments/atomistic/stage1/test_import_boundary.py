"""The stage layer must not import private TPEN modules at import time."""

from pathlib import Path


def test_stage_sources_use_only_gateway_imports() -> None:
    root = Path(__file__).parent
    for path in root.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        assert "from tpen." not in text or "from tpen.run import run_from_config" in text
