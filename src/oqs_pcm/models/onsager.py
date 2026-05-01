from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oqs_pcm.dielectric import DielectricModel
from oqs_pcm.io import io_load_state_transition_dipole_tensor


@dataclass(slots=True)
class OnsagerParameters:
    """Parameters specific to the Onsager-based OQS model.

    Unlike PCM, which works with electrostatic potentials on a discretized
    cavity surface, the Onsager formulation uses state and transition electric
    dipole moments coupled through a spherical cavity of radius `cavity_radius`.
    """

    cavity_radius: float
    cutoff: float
    fit_epsilon: float = 2.0302901591898928
    fit_cutoff: float = 0.5172311136437814


@dataclass(slots=True)
class OnsagerKernelBundle:
    """Onsager coupling factors expressed in the generic solver format.

    The shared solver expects a per-state row of coupling coefficients. For
    Onsager the same scalar response factor is repeated for the three Cartesian
    dipole components, so each row has length three.
    """

    static_factor: float
    optical_factor: float
    dynamic_real: np.ndarray
    dynamic_imag: np.ndarray
    fitted_factor: float


def load_onsager_dipoles(path: str | Path, nsts: int, state_offset: int = 1) -> np.ndarray:
    """Load the state/transition dipole tensor used by the Onsager workflow."""

    return io_load_state_transition_dipole_tensor(path, nsts, state_offset=state_offset)


def onsager_reaction_factor(epsilon_value: complex | float, cavity_radius: float) -> complex:
    """Return the Onsager reaction factor for one dielectric value.

    This is the analytic analogue of the solvent coupling used in the Onsager
    notebook. The factor multiplies dipole moments instead of cavity
    electrostatic potentials.
    """

    return 2.0 * ((epsilon_value - 1.0) / (2.0 * epsilon_value + 1.0)) / (cavity_radius**3)


def build_onsager_kernels(
    dielectric_model: DielectricModel,
    energies: np.ndarray,
    cavity_radius: float,
    optical_window_ev: tuple[float, float],
    fit_epsilon: float,
) -> OnsagerKernelBundle:
    """Build static, optical, dynamic, and fitted Onsager response factors.

    The dynamic factors are expanded into `(n_states, 3)` arrays so they can be
    consumed by the same generic solver that already handles PCM couplings.
    """

    static_factor = float(np.real(onsager_reaction_factor(dielectric_model.epsilon_static(), cavity_radius)))
    optical_eps = dielectric_model.epsilon_optical_average(*optical_window_ev)
    optical_factor = float(np.real(onsager_reaction_factor(optical_eps, cavity_radius)))

    dynamic_eps = dielectric_model.epsilon(-energies, include_ir=True)
    dynamic_factors = np.array([onsager_reaction_factor(value, cavity_radius) for value in dynamic_eps], dtype=complex)
    dynamic_real = np.repeat(np.real(dynamic_factors)[:, np.newaxis], 3, axis=1)
    dynamic_imag = np.repeat(np.imag(dynamic_factors)[:, np.newaxis], 3, axis=1)

    fitted_factor = float(np.real(onsager_reaction_factor(fit_epsilon, cavity_radius)))

    return OnsagerKernelBundle(
        static_factor=static_factor,
        optical_factor=optical_factor,
        dynamic_real=dynamic_real,
        dynamic_imag=dynamic_imag,
        fitted_factor=fitted_factor,
    )


def build_onsager_diagonal_kernel(nstates: int, state_index: int, factor: float) -> np.ndarray:
    """Build the state-selective optical kernel used by SS-pol-like Onsager runs."""

    kernel = np.zeros((nstates, 3), dtype=float)
    kernel[state_index, :] = factor
    return kernel


def build_onsager_cutoff_mask(energies: np.ndarray, cutoff: float) -> np.ndarray:
    """Build the cutoff mask used by the Onsager cutoff approximation."""

    return (np.abs(energies[:, None] - energies[None, :]) <= cutoff).astype(float)
