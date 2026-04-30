from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oqs_pcm.dielectric import DielectricModel
from oqs_pcm.solver import KernelBundle, build_pcm_kernel_bundle


@dataclass(slots=True)
class PCMParameters:
    """Parameters specific to the PCM-based OQS model.

    PCM uses state/transition electrostatic potentials defined on a discretized
    cavity that follows the molecular shape. The generic solver does not need
    to know those details; it only receives the assembled coupling kernels and
    the coupling tensor.
    """

    optical_window_ev: tuple[float, float]
    cutoff: float


def build_pcm_kernels(
    lambda_path: Path,
    dielectric_model: DielectricModel,
    energies: np.ndarray,
    optical_window_ev: tuple[float, float],
) -> KernelBundle:
    """Build the PCM kernel bundle in the generic solver format."""

    return build_pcm_kernel_bundle(
        lambda_path=lambda_path,
        dielectric_model=dielectric_model,
        energies=energies,
        optical_window=optical_window_ev,
    )
