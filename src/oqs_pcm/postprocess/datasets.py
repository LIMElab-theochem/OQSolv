from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


HARTREE_TO_EV = 27.2114


@dataclass(slots=True)
class LevelDataset:
    """Energy-level data used by the level-diagram figures."""

    ensspol: np.ndarray
    encorr1: np.ndarray
    encorr2: np.ndarray
    en: np.ndarray
    enbo: np.ndarray


@dataclass(slots=True)
class SpectraDatasetGroup:
    """One family of spectra curves, each stored as (energy, oscillator) pairs.

    This structure intentionally covers both PCM-based and Onsager-based figure
    tables. At the plotting stage they look the same: each method produces a
    stick spectrum as a two-column text file. The physical difference enters
    earlier, when the tables are computed:

    - PCM-based OQS uses electrostatic potentials on a molecular cavity.
    - Onsager-based OQS uses state and transition dipole moments in a spherical
      cavity model.
    """

    tables: dict[str, np.ndarray]


@dataclass(slots=True)
class CutoffDataset:
    """Data used by the cutoff comparison figure."""

    spectra_tables: dict[str, np.ndarray]
    dielectric_tables: dict[str, np.ndarray]


@dataclass(slots=True)
class ConvergenceDataset:
    """Ground-state convergence tables used in the convergence figures."""

    tables: dict[str, np.ndarray]


def load_vector(path: str | Path) -> np.ndarray:
    """Read a one-column text file as a flat vector."""

    return np.loadtxt(path, ndmin=1)


def load_table(path: str | Path) -> np.ndarray:
    """Read a two-column or multi-column plain-text table."""

    return np.loadtxt(path, ndmin=2)


def load_level_dataset(base_dir: str | Path) -> LevelDataset:
    """Load the exact text files consumed by the former `levels*.py` scripts."""

    base = Path(base_dir)
    return LevelDataset(
        ensspol=load_vector(base / "ensspol.txt"),
        encorr1=load_vector(base / "encorr1.txt"),
        encorr2=load_vector(base / "encorr2.txt"),
        en=load_vector(base / "en.txt"),
        enbo=load_vector(base / "enbo.txt"),
    )


def load_named_tables(base_dir: str | Path, filenames: dict[str, str]) -> dict[str, np.ndarray]:
    """Load a named set of plain-text tables from one directory."""

    base = Path(base_dir)
    return {name: load_table(base / filename) for name, filename in filenames.items()}


def load_spectra_dataset(base_dir: str | Path, filenames: dict[str, str]) -> SpectraDatasetGroup:
    return SpectraDatasetGroup(tables=load_named_tables(base_dir, filenames))


def load_cutoff_dataset(base_dir: str | Path) -> CutoffDataset:
    base = Path(base_dir)
    spectra_files = {
        "sspol": "sspol.txt",
        "full": "full.txt",
        "cut": "cut.txt",
        "fit": "fit.txt",
    }
    dielectric_files = {
        "reeps": "reeps.txt",
        "reepscut": "reepscut.txt",
        "reepsfit": "reepsfit.txt",
    }
    return CutoffDataset(
        spectra_tables=load_named_tables(base, spectra_files),
        dielectric_tables=load_named_tables(base, dielectric_files),
    )


def load_convergence_dataset(base_dir: str | Path, filenames: dict[str, str]) -> ConvergenceDataset:
    return ConvergenceDataset(tables=load_named_tables(base_dir, filenames))
