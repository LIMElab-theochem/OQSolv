from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(slots=True)
class DielectricModel:
    debye_c: np.ndarray
    debye_g: np.ndarray
    drude_w: np.ndarray
    drude_c: np.ndarray
    drude_g: np.ndarray
    optical_drude_start_index: int
    hartree_to_ev: float

    def epsilon(self, omega_hartree: np.ndarray | float, include_ir: bool = True) -> np.ndarray:
        """Evaluate the complex dielectric function from the Mathematica model."""

        omega_hartree = np.asarray(omega_hartree, dtype=float)
        x = omega_hartree * self.hartree_to_ev
        eps = np.ones_like(x, dtype=complex)

        if include_ir:
            for c_value, g_value in zip(self.debye_c, self.debye_g, strict=True):
                eps += c_value / (1 - 1j * x * g_value)
            drude_start = 0
        else:
            drude_start = self.optical_drude_start_index - 1

        for idx in range(drude_start, len(self.drude_w)):
            numerator = self.drude_c[idx] * self.drude_w[idx] ** 2
            denominator = self.drude_w[idx] ** 2 - x ** 2 - 1j * x * self.drude_g[idx]
            eps += numerator / denominator

        return eps

    def epsilon_static(self) -> float:
        return float(np.real(self.epsilon(0.0, include_ir=True)))

    def epsilon_optical_average(self, window_min_ev: float, window_max_ev: float, samples: int = 4000) -> float:
        """Approximate Mathematica's `NIntegrate` with a dense trapezoidal grid."""

        grid_ev = np.linspace(window_min_ev, window_max_ev, samples)
        grid_hartree = grid_ev / self.hartree_to_ev
        values = np.real(self.epsilon(grid_hartree, include_ir=False))
        return float(np.trapezoid(values, grid_hartree) / abs(grid_hartree[-1] - grid_hartree[0]))


def build_dielectric_model(config) -> DielectricModel:
    return DielectricModel(
        debye_c=np.asarray(config.debye_c, dtype=float),
        debye_g=np.asarray(config.debye_g, dtype=float),
        drude_w=np.asarray(config.drude_w, dtype=float),
        drude_c=np.asarray(config.drude_c, dtype=float),
        drude_g=np.asarray(config.drude_g, dtype=float),
        optical_drude_start_index=int(config.optical_drude_start_index),
        hartree_to_ev=27.2114,
    )


def build_kernel_from_lambda(lambda_values: np.ndarray, epsilon_value: float | complex) -> np.ndarray:
    feps = (epsilon_value + 1) / (epsilon_value - 1)
    return (2 * np.pi - lambda_values) / (2 * np.pi * feps - lambda_values)


def build_state_resolved_kernel_from_lambda(
    lambda_values: np.ndarray,
    epsilon_values: np.ndarray,
) -> np.ndarray:
    """Build the state-resolved PCM kernel directly from `Lambda`."""

    epsilon_values = np.asarray(epsilon_values, dtype=complex)
    feps_values = (epsilon_values + 1) / (epsilon_values - 1)
    lambda_values = np.asarray(lambda_values, dtype=float)
    return (2 * np.pi - lambda_values[np.newaxis, :]) / (
        2 * np.pi * feps_values[:, np.newaxis] - lambda_values[np.newaxis, :]
    )
