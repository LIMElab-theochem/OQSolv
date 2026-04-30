from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oqs_pcm.postprocess.datasets import HARTREE_TO_EV


DEFAULT_FIGURE_STYLE = {
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}


def apply_default_style(fontsize: int = 12) -> None:
    """Apply a consistent matplotlib style for all generated figures."""

    style = dict(DEFAULT_FIGURE_STYLE)
    for key in list(style):
        style[key] = fontsize
    plt.rcParams.update(style)


def broaden_spectrum(energies_ev: np.ndarray, oscillators: np.ndarray, x_grid_ev: np.ndarray, sigma_ev: float) -> np.ndarray:
    """Apply simple Gaussian broadening to stick spectra.

    This mirrors the broadening used in the existing Python plotting scripts.
    It is kept separate so postprocessing and plotting can share one implementation.
    """

    y_values = np.zeros_like(x_grid_ev)
    for energy_ev, oscillator in zip(energies_ev, oscillators, strict=True):
        y_values += oscillator * np.exp(-((x_grid_ev - energy_ev) ** 2) / (2 * sigma_ev**2))
    return y_values


def energy_table_hartree_to_ev(table: np.ndarray) -> np.ndarray:
    """Convert the first column of a spectrum table from Hartree to eV."""

    converted = np.array(table, copy=True, dtype=float)
    converted[:, 0] *= HARTREE_TO_EV
    return converted


def save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """Write a figure to disk, creating parent directories when needed."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
