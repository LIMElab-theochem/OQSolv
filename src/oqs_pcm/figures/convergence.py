from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from oqs_pcm.figures.common import HARTREE_TO_EV, apply_default_style, save_figure
from oqs_pcm.postprocess.datasets import ConvergenceDataset, load_convergence_dataset


CONVERGENCE_FILES = {
    "bo": "GSsolbo.txt",
    "full": "GSsolfull.txt",
    "cut": "GSsolcut.txt",
    "fit": "GSsolfit.txt",
    "bo_osc": "GSsolbo_oscillations.txt",
}

CONVERGENCE_WATER_FILES = {
    "bo": "GSsolbo_new.txt",
    "full": "GSsolfull.txt",
    "cut": "GSsolcut.txt",
    "fit": "GSsolfit.txt",
}

CONVERGENCE_BENZENE_FILES = {
    "bo": "GSsolbo_benzene.txt",
    "full": "GSsolfull_benzene.txt",
    "cut": "GSsolcut_benzene.txt",
    "fit": "GSsolfit_benzene.txt",
}


def build_convergence_figure(dataset: ConvergenceDataset, fontsize: int = 10) -> plt.Figure:
    apply_default_style(fontsize)
    color_map = {"bo": "red", "full": "blue", "cut": "darkgreen", "fit": "magenta", "bo_osc": "red"}
    legend_map = {"bo": "BO-sol", "full": r"OQS-PCM($\omega$)", "cut": "cutoff", "fit": "fit cutoff", "bo_osc": "BO-sol"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)

    ax = axes[0]
    for name in ["bo", "full", "cut", "fit"]:
        table = dataset.tables[name]
        ax.plot(table[:, 0], table[:, 1] * HARTREE_TO_EV, color=color_map[name], lw=1.2, marker="o", markersize=3, label=legend_map[name])
    ax.set_xlim(0, 625)
    ax.set_ylim(-2.1, 0)
    ax.set_xlabel("Total Number States")
    ax.set_ylabel("GS solvation energy (eV)")
    ax.legend(loc="lower left", frameon=False, fontsize=fontsize)
    ax.grid(False)
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="x", which="major", top=True, direction="in", length=6)
    ax.tick_params(axis="x", which="minor", top=True, direction="in", length=3)
    ax.tick_params(axis="y", which="major", right=True, direction="in", length=6)
    ax.tick_params(axis="y", which="minor", right=True, direction="in", length=3)
    ax.text(0.92, 0.3, "in water", transform=ax.transAxes, horizontalalignment="right", verticalalignment="top", fontsize=fontsize + 2)

    ax = axes[1]
    table = dataset.tables["bo_osc"]
    ax.plot(table[:, 0], table[:, 1] * HARTREE_TO_EV, color=color_map["bo_osc"], lw=1.2, marker="o", markersize=3, label=legend_map["bo_osc"])
    ax.set_xlim(0, 625)
    ax.set_ylim(-2.1, 0)
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="y", which="major", right=True, direction="in", length=6)
    ax.tick_params(axis="y", which="minor", right=True, direction="in", length=3)
    ax.tick_params(axis="x", which="major", top=True, direction="in", length=6)
    ax.tick_params(axis="x", which="minor", top=True, direction="in", length=3)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Total Number States")
    ax.legend(loc="lower left", frameon=False, fontsize=fontsize)
    ax.grid(False)

    fig.tight_layout()
    return fig


def build_single_panel_convergence_figure(dataset: ConvergenceDataset, region_label: str, fontsize: int = 10) -> plt.Figure:
    apply_default_style(fontsize)
    color_map = {"bo": "red", "full": "blue", "cut": "darkgreen", "fit": "magenta"}
    legend_map = {"bo": "BO-sol", "full": r"OQS-PCM($\omega$)", "cut": "cutoff", "fit": "fit cutoff"}
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for name in ["bo", "full", "cut", "fit"]:
        table = dataset.tables[name]
        ax.plot(table[:, 0], table[:, 1] * HARTREE_TO_EV, color=color_map[name], lw=1.5, marker="o", markersize=3, label=legend_map[name])
    ax.set_xlim(0, 625)
    ax.set_xlabel("Total Number States")
    ax.set_ylabel("GS solvation energy (eV)")
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="x", which="major", top=True, direction="in", length=6)
    ax.tick_params(axis="x", which="minor", top=True, direction="in", length=3)
    ax.tick_params(axis="y", which="major", right=True, direction="in", length=6)
    ax.tick_params(axis="y", which="minor", right=True, direction="in", length=3)
    ax.legend(loc="lower left", frameon=False, fontsize=fontsize)
    ax.grid(False)
    ax.text(0.92, 0.3, region_label, transform=ax.transAxes, horizontalalignment="right", verticalalignment="top", fontsize=fontsize + 2)
    fig.tight_layout()
    return fig


def render_convergence_figures(data_dir: str | Path, output_dir: str | Path, fontsize: int = 10) -> None:
    output = Path(output_dir)
    save_figure(build_convergence_figure(load_convergence_dataset(data_dir, CONVERGENCE_FILES), fontsize=fontsize), output / "convergence.pdf")
    save_figure(build_single_panel_convergence_figure(load_convergence_dataset(data_dir, CONVERGENCE_WATER_FILES), "in water", fontsize=fontsize), output / "convergence_water.pdf")
    save_figure(build_single_panel_convergence_figure(load_convergence_dataset(data_dir, CONVERGENCE_BENZENE_FILES), "in benzene", fontsize=fontsize), output / "convergence_benzene.pdf")
