# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""The ML stack is imported only inside llm_kelt.training.runtime (and the CLI).

Type checkers follow every import statement, including those inside functions. A module of
kelt's API that imported torch or a runtime module by name would make every project
type-checking against kelt analyse the ML stack.
"""

import ast
from pathlib import Path

import llm_kelt

ML_PACKAGES = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "trl",
}
RUNTIME = "llm_kelt.training.runtime"
PACKAGE_ROOT = Path(llm_kelt.__file__).parent
EXEMPT = (PACKAGE_ROOT / "training" / "runtime", PACKAGE_ROOT / "cli")


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported(path: Path) -> list[tuple[int, str]]:
    """(line, absolute module or module.name) for every import in the file."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            found += [(node.lineno, f"{base}.{alias.name}") for alias in node.names]
    return found


def _violation(name: str) -> bool:
    if name.split(".")[0] in ML_PACKAGES:
        return True
    # The runtime package itself may be imported for load(); its modules may not.
    return name.startswith(f"{RUNTIME}.") and name != f"{RUNTIME}.load"


def test_api_modules_do_not_import_the_ml_stack():
    files = [p for p in PACKAGE_ROOT.rglob("*.py") if not any(p.is_relative_to(e) for e in EXEMPT)]
    violations = [
        f"{path.relative_to(PACKAGE_ROOT.parent)}:{line} imports {name}"
        for path in files
        for line, name in _imported(path)
        if _violation(name)
    ]
    assert not violations, "\n".join(violations)
