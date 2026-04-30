from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from oqs_pcm.figures.common import apply_default_style, broaden_spectrum, energy_table_hartree_to_ev, save_figure
from oqs_pcm.postprocess.datasets import CutoffDataset, load_cutoff_dataset


def build_cutoff_figure(dataset: CutoffDataset, fontsize: int = 10) -> plt.Figure:
    apply_default_style(fontsize)

    color_map = {
        "sspol": "orange",
        "full": "blue",
        "cut": "darkgreen",
        "fit": "magenta",
        "reeps": "blue",
        "reepscut": "darkgreen",
        "reepsfit": "magenta",
    }
    legend_map = {
        "sspol": "SS-pol",
        "full": r"OQS-PCM($\omega$)",
        "cut": "cutoff",
        "fit": "fit cutoff",
        "reeps": "empirical data",
        "reepscut": "cutoff",
        "reepsfit": "fit cutoff",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    x_grid = np.linspace(2, 12, 5000)
    sigma_ev = 0.1

    ax = axes[0]
    for name in ["sspol", "full", "cut", "fit"]:
        table = energy_table_hartree_to_ev(dataset.spectra_tables[name])
        y_values = broaden_spectrum(table[:, 0], table[:, 1], x_grid, sigma_ev)
        ax.plot(x_grid, y_values, color=color_map[name], lw=2, label=legend_map[name])
    ax.set_xlim(5, 12)
    ax.set_ylim(0, 0.32)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Oscillator strength (au)")
    ax.legend(loc="upper left", frameon=False, fontsize=fontsize)
    ax.grid(False)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", which="major", top=True, direction="in", length=6)
    ax.tick_params(axis="x", which="minor", top=True, direction="in", length=3)
    ax.tick_params(axis="y", right=True, direction="in", length=6)

    ax = axes[1]
    for name in ["reeps", "reepscut", "reepsfit"]:
        table = np.array(dataset.dielectric_tables[name], copy=True, dtype=float)
        table[:, 0] *= 27.2114
        ax.plot(table[:, 0], table[:, 1], color=color_map[name], lw=2, label=legend_map[name])
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.58, 3.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([1.0])
    ax.set_yticklabels(["1"])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 2.0, 3.0], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.tick_params(axis="y", which="major", right=True, direction="in", length=6)
    ax.tick_params(axis="y", which="minor", right=True, direction="in", length=3)
    ax.tick_params(axis="x", which="major", top=True, direction="in", length=6)
    ax.tick_params(axis="x", which="minor", top=True, direction="in", length=3)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Real part of dielectric function (au)")
    ax.legend(loc="upper left", frameon=False, fontsize=fontsize)
    ax.grid(False)

    fig.tight_layout()
    return fig


def render_cutoff_figure(data_dir: str | Path, output_path: str | Path, fontsize: int = 10) -> None:
    dataset = load_cutoff_dataset(data_dir)
    fig = build_cutoff_figure(dataset, fontsize=fontsize)
    save_figure(fig, output_path)
