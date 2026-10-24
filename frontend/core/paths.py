"""Resolve project paths regardless of the process working directory."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root (parent of ``frontend/``)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


def models_dir() -> Path:
    return project_root() / "models"


def assets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"
