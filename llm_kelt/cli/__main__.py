# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Allow running as python -m llm_kelt.cli."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
