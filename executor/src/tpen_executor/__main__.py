"""Module entrypoint for ``python -m tpen_executor``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
