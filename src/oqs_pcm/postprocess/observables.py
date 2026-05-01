from __future__ import annotations

import numpy as np


def transform_state_transition_observable_to_basis(observable: np.ndarray, basis_states: np.ndarray) -> np.ndarray:
    """Transform a state/transition vector observable to a new electronic basis.

    `observable[a, b, :]` is a Cartesian vector connecting electronic states
    `a` and `b` in the input basis. `basis_states[i, a]` is the expansion
    coefficient of new state `i` in old state `a`. The contraction mirrors the
    Mathematica expression used for dipoles in the inertial basis:

    `new_mu = states . mu . Transpose[states]`

    The final symmetrization removes tiny numerical asymmetries introduced by
    floating-point matrix products.
    """

    transformed_components = []
    for component in range(observable.shape[2]):
        transformed_components.append(basis_states @ observable[:, :, component] @ basis_states.T)
    transformed = np.stack(transformed_components, axis=2)
    return 0.5 * (transformed + np.swapaxes(transformed, 0, 1))


def ground_to_excited_transition_vectors_from_states(observable: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Compute ground-to-state transition vectors from solved-state expansions.

    The rows of `states` are the solved electronic states exported in
    `newstates.dat`. The first row is treated as the solved ground state. For
    each solved state `k`, this evaluates

    `sum_ab observable[a, b, :] * states[0, a] * states[k, b]`.

    This is the Python equivalent of the `gs2esDipolesFromStates` routine in
    `figures-new.nb`, generalized to any Cartesian vector observable.
    """

    n_basis = states.shape[1]
    reduced_observable = observable[:n_basis, :n_basis, :]
    vectors = []
    ground_state = states[0, :]
    for state in states:
        vectors.append(np.einsum("a,abx,b->x", ground_state, reduced_observable, state, optimize=True))
    return np.asarray(vectors)


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
