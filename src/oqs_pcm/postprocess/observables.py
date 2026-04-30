from __future__ import annotations

import numpy as np


def oscillator_strengths(energies: np.ndarray, transition_vectors: np.ndarray) -> np.ndarray:
    """Compute oscillator strengths from energies and transition vectors.

    This follows the standard formula used throughout the Mathematica notebooks:

    `(2/3) * DeltaE * ||mu||^2`

    The function is intentionally generic:

    - for PCM-related postprocessing, `transition_vectors` can represent any
      already transformed transition observable reduced to Cartesian form
    - for Onsager, the vectors are directly electric dipole moments
    """

    reference_energy = energies[0]
    strengths = []
    for state_index in range(1, len(energies)):
        delta_e = energies[state_index] - reference_energy
        strengths.append((2.0 / 3.0) * delta_e * (np.linalg.norm(transition_vectors[state_index]) ** 2))
    return np.asarray(strengths, dtype=float)
