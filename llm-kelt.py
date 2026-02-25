#!/usr/bin/env python3
"""Learning framework cli entry point."""

import sys
from pathlib import Path

# Ensure local source takes precedence over installed package
sys.path.insert(0, str(Path(__file__).parent))

from llm_kelt.cli import main

if __name__ == "__main__":
    main()
