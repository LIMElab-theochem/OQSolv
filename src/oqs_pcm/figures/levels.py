from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from oqs_pcm.figures.common import HARTREE_TO_EV, apply_default_style, save_figure
from oqs_pcm.postprocess.datasets import LevelDataset, load_level_dataset


def build_levels_figure(dataset: LevelDataset, fontsize: int = 12) -> plt.Figure:
    """Build the two-panel level figure previously produced by `levels.py`.

    The plotting logic is intentionally close to the original script so it is
    easy to verify against the existing figure-generation workflow.
    """

    apply_default_style(fontsize)

    ensspol = np.array(dataset.ensspol, copy=True) * HARTREE_TO_EV
    encorr1 = np.array(dataset.encorr1, copy=True) * HARTREE_TO_EV
    encorr2 = np.array(dataset.encorr2, copy=True) * HARTREE_TO_EV
    en = np.array(dataset.en, copy=True) * HARTREE_TO_EV
    enbo = np.array(dataset.enbo, copy=True) * HARTREE_TO_EV

    color_map = {
        "SS-pol": "orange",
        "SS-pol_eps": "saddlebrown",
        "SS-pol+disp": "cornflowerblue",
        "OQS-PCM": "blue",
        "BO-sol": "red",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for index in range(len(ensspol)):
        ax.hlines(ensspol[index], 0.2, 1, color=color_map["SS-pol"], lw=2)
        ax.hlines(encorr1[index], 1.2, 2, color=color_map["SS-pol_eps"], lw=2)
        ax.hlines(encorr2[index], 2.2, 3, color=color_map["SS-pol+disp"], lw=2)
        ax.hlines(en[index], 3.2, 4, color=color_map["OQS-PCM"], lw=2)
        ax.plot([1.0, 1.2], [ensspol[index], encorr1[index]], ls="--", color="gray")
        ax.plot([2.0, 2.2], [encorr1[index], encorr2[index]], ls="--", color="gray")
        ax.plot([3.0, 3.2], [encorr2[index], en[index]], ls="--", color="gray")

    ax.set_ylim(-3.5, 10)
    ax.set_yticks(np.arange(-2, 11, 1))
    ax.set_yticks(np.arange(-1.5, 10.5, 0.5), minor=True)
    ax.tick_params(axis="y", which="both", direction="in", length=6)
    ax.get_xaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.annotate("", xy=(3.9, -3), xytext=(0.3, -3), arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.text(2.1, -3.7, "complexity", ha="center", fontsize=fontsize)
    ax.text(0.6, -2, r"SS-pol [$\epsilon_d$]", ha="center", fontsize=fontsize)
    ax.text(1.6, -2, r"SS-pol [$\epsilon(\omega)$]", ha="center", fontsize=fontsize)
    ax.text(2.6, -2, r"SS-pol+disp [$\epsilon(\omega)$]", ha="center", fontsize=fontsize)
    ax.text(3.6, -2, r"OQS-PCM($\omega$)", ha="center", fontsize=fontsize)
    ax.set_ylabel("Free Energy (eV)", fontsize=fontsize)

    ax = axes[1]
    for index in range(min(len(ensspol), len(en), len(enbo))):
        ax.hlines(ensspol[index], 0.2, 1, color=color_map["SS-pol"], lw=2)
        ax.hlines(en[index], 1.2, 2, color=color_map["OQS-PCM"], lw=2)
        ax.hlines(enbo[index], 2.2, 3, color=color_map["BO-sol"], lw=2)
        ax.plot([1.0, 1.2], [ensspol[index], en[index]], ls="--", color="gray")
        ax.plot([2.0, 2.2], [en[index], enbo[index]], ls="--", color="gray")

    ax.set_ylim(-4, 10)
    ax.set_yticks(np.arange(-3, 11, 1))
    ax.set_yticks(np.arange(-2.5, 10.5, 0.5), minor=True)
    ax.tick_params(axis="y", which="both", direction="in", length=6)
    ax.get_xaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.annotate("", xy=(2.9, -3.5), xytext=(0.3, -3.5), arrowprops=dict(arrowstyle="->", linewidth=2))
    ax.text(1.6, -4.4, "solute/solvent timescale", ha="center", fontsize=fontsize)
    ax.text(0.6, -4.4, "fast/slow", ha="center", fontsize=fontsize)
    ax.text(2.6, -4.4, "slow/fast", ha="center", fontsize=fontsize)
    ax.text(0.6, -2.5, "SS-pol", ha="center", fontsize=fontsize)
    ax.text(1.6, -2.5, r"OQS-PCM($\omega$)", ha="center", fontsize=fontsize)
    ax.text(2.6, -2.5, "BO-sol", ha="center", fontsize=fontsize)
    ax.set_ylabel("Free Energy (eV)", fontsize=fontsize)

    fig.tight_layout()
    return fig


def render_levels_figure(data_dir: str | Path, output_path: str | Path, fontsize: int = 12) -> None:
    dataset = load_level_dataset(data_dir)
    fig = build_levels_figure(dataset, fontsize=fontsize)
    save_figure(fig, output_path)
