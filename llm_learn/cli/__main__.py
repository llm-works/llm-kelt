"""Allow running as python -m llm_learn.cli."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
