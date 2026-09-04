#!/usr/bin/env python3
"""Validate repository structure without importing optional runtime packages.

The check deliberately uses only the Python standard library. It catches
syntax errors, missing module docstrings, obsolete pre-refactor imports, an
oversized landing README, and tracked runtime cache files before slower tests
or container builds start.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "tests",
)
REQUIRED_PATHS = (
    PROJECT_ROOT / "src" / "hkpl_agent" / "app.py",
    PROJECT_ROOT / "src" / "hkpl_agent" / "api" / "schemas.py",
    PROJECT_ROOT / "src" / "hkpl_agent" / "agent" / "graph.py",
    PROJECT_ROOT / "src" / "hkpl_agent" / "rag" / "retrieval.py",
    PROJECT_ROOT / "src" / "hkpl_agent" / "ingestion" / "service.py",
    PROJECT_ROOT / "infra" / "docker" / "Dockerfile.agent",
    PROJECT_ROOT / "infra" / "postgres" / "init.sql",
    PROJECT_ROOT / "docs" / "README.md",
    PROJECT_ROOT / "Makefile",
    PROJECT_ROOT / ".github" / "workflows" / "ci.yml",
    PROJECT_ROOT / ".github" / "workflows" / "container.yml",
)


def python_files() -> list[Path]:
    """Return maintained Python sources in stable display order."""

    files = [PROJECT_ROOT / "main.py"]
    for root in PYTHON_ROOTS:
        files.extend(root.rglob("*.py"))
    return sorted(files)


def tracked_files() -> list[str]:
    """Return Git-tracked file names, or an empty list outside a checkout."""

    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def main() -> int:
    """Run repository invariants and return a process exit code."""

    errors: list[str] = []

    for path in REQUIRED_PATHS:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(PROJECT_ROOT)}")

    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        line_count = len(readme.read_text(encoding="utf-8").splitlines())
        if line_count > 250:
            errors.append(
                f"README.md has {line_count} lines; move detailed guidance to docs/"
            )

    for path in python_files():
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            errors.append(f"{path.relative_to(PROJECT_ROOT)}: {error}")
            continue

        if ast.get_docstring(tree) is None:
            errors.append(f"{path.relative_to(PROJECT_ROOT)}: missing module docstring")

        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src" or alias.name.startswith("src."):
                        errors.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                            f"legacy import {alias.name!r}"
                        )
            if module == "src" or module.startswith("src."):
                errors.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: "
                    f"legacy import {module!r}"
                )

    forbidden_parts = {"__pycache__", ".pytest_cache", ".mypy_cache"}
    for tracked in tracked_files():
        path = Path(tracked)
        if forbidden_parts.intersection(path.parts) or path.suffix in {".pyc", ".pyo"}:
            errors.append(f"tracked runtime artifact: {tracked}")

    if errors:
        print("Repository checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository checks passed ({len(python_files())} Python files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
