from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from oqs_pcm.figures.common import apply_default_style, broaden_spectrum, energy_table_hartree_to_ev, save_figure
from oqs_pcm.postprocess.datasets import SpectraDatasetGroup, load_spectra_dataset


SPECTRA_FILENAMES = {
    "sspol": "sspol.txt",
    "sspolc6": "sspolc6.txt",
    "sspolons": "sspolons.txt",
    "full": "full.txt",
    "fullc6": "fullc6.txt",
    "fullons": "fullons.txt",
    "bo": "bo.txt",
    "boc6": "boc6.txt",
    "boons": "boons.txt",
}


def _plot_panel(ax: plt.Axes, tables: dict[str, np.ndarray], datasets: tuple[str, ...], panel_label: str, panel_letter: str, x_grid: np.ndarray, sigma_ev: float, fontsize: int) -> None:
    """Plot one spectra panel.

    The same plotting function is used for PCM-based and Onsager-based data.
    That is deliberate: once the postprocessing layer has turned the results
    into `(energy, oscillator strength)` tables, the rendering logic no longer
    depends on whether those tables originated from cavity potentials or from
    dipole moments.
    """

    color_map = {
        "sspol": "orange",
        "sspolc6": "orange",
        "sspolons": "orange",
        "full": "blue",
        "fullc6": "blue",
        "fullons": "blue",
        "bo": "red",
        "boc6": "red",
        "boons": "red",
    }
    legend_map = {
        "sspol": r"SS-pol",
        "sspolc6": r"SS-pol",
        "sspolons": r"SS-pol",
        "full": r"OQS-PCM($\omega$)",
        "fullc6": r"OQS-PCM($\omega$)",
        "fullons": r"OQS-Onsager($\omega$)",
        "bo": r"BO-sol",
        "boc6": r"BO-sol",
        "boons": r"BO-sol",
    }

    for name in datasets:
        table = energy_table_hartree_to_ev(tables[name])
        y_values = broaden_spectrum(table[:, 0], table[:, 1], x_grid, sigma_ev)
        ax.plot(x_grid, y_values, lw=2, color=color_map[name], label=legend_map[name])

    ax.set_xlim(5, 12)
    ax.set_ylim(0, 0.32)
    ax.grid(False)
    ax.text(0.03, 0.95, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=fontsize)
    ax.text(0.98, 0.95, panel_letter, transform=ax.transAxes, ha="right", va="top", fontsize=fontsize, fontweight="bold")
    ax.tick_params(top=True, right=True, direction="in", labeltop=False, labelright=False, length=6)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", length=3)


def build_pcm_spectra_figure(dataset: SpectraDatasetGroup, fontsize: int = 14) -> plt.Figure:
    apply_default_style(fontsize)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    x_grid = np.linspace(2, 12, 5000)
    sigma_ev = 0.1

    panels = [
        (("sspol", "full", "bo"), "in water", ""),
        (("sspolc6", "fullc6", "boc6"), "in benzene", ""),
    ]
    for ax, (datasets, label, letter) in zip(axes, panels, strict=True):
        _plot_panel(ax, dataset.tables, datasets, label, letter, x_grid, sigma_ev, fontsize)

    legend_handles = [
        Line2D([0], [0], color="orange", lw=2, label="SS-pol"),
        Line2D([0], [0], color="blue", lw=2, label=r"OQS-PCM($\omega$)"),
        Line2D([0], [0], color="red", lw=2, label="BO-sol"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, fontsize=fontsize, bbox_to_anchor=(0.5, 1.05))
    axes[0].set_xlabel("Energy (eV)", fontsize=fontsize)
    axes[1].set_xlabel("Energy (eV)", fontsize=fontsize)
    axes[0].set_ylabel("Oscillator strength (au)", fontsize=fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def build_onsager_spectra_figure(dataset: SpectraDatasetGroup, fontsize: int = 12) -> plt.Figure:
    """Build the figure that uses the Onsager-based datasets.

    In the original Mathematica/Python workflow these curves were already
    exported as `sspolons`, `fullons`, and `boons`. The refactor keeps those
    names for operational continuity while making the PCM/Onsager distinction
    explicit in the surrounding documentation.
    """

    apply_default_style(fontsize)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
    x_grid = np.linspace(2, 12, 5000)
    sigma_ev = 0.1
    _plot_panel(ax, dataset.tables, ("sspolons", "fullons", "boons"), "", "", x_grid, sigma_ev, fontsize)
    ax.set_ylabel("Oscillator strength (au)", fontsize=fontsize)
    ax.set_xlabel("Energy (eV)", fontsize=fontsize)
    ax.legend(loc="upper left", frameon=False, fontsize=fontsize)
    fig.tight_layout()
    return fig


def render_spectra_figures(data_dir: str | Path, output_dir: str | Path, fontsize: int = 12) -> None:
    dataset = load_spectra_dataset(data_dir, SPECTRA_FILENAMES)
    save_figure(build_pcm_spectra_figure(dataset, fontsize=max(fontsize, 14)), Path(output_dir) / "spectra_pcm.pdf")
    save_figure(build_onsager_spectra_figure(dataset, fontsize=fontsize), Path(output_dir) / "spectra_onsager.pdf")
