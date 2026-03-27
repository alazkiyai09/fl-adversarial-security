"""Smoke coverage for migrated planned test surface."""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULES = [
    'src.core.config'
]

REQUIRED_PATHS = [
    'src/core/config.py'
]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


def _module_source_exists(module_name: str) -> bool:
    root = _repo_root()
    rel = Path(*module_name.split('.'))
    file_candidate = root / f"{rel}.py"
    init_candidate = root / rel / "__init__.py"
    return file_candidate.exists() or init_candidate.exists()


def test_planned_modules_are_present() -> None:
    missing = [m for m in MODULES if not _module_source_exists(m)]
    assert not missing, f"Missing module source files: {missing}"



def test_planned_paths_exist() -> None:
    root = _repo_root()
    missing = [p for p in REQUIRED_PATHS if not (root / p).exists()]
    assert not missing, f"Missing expected paths: {missing}"
