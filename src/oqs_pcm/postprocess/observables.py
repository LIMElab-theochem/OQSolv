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
    """Compute oscillator strengths with the Hermitian-state approximation.

    This follows the standard real-state expression historically used in the
    postprocessing layer:

    `(2/3) * DeltaE * ||mu||^2`

    It remains the appropriate default for Hermitian branches such as BO-sol,
    SS-pol, cutoff, fit-cutoff, and OQS-Onsager variants whose solved states are
    treated with the usual Euclidean norm.
    """

    real_energies = np.real(energies)
    reference_energy = real_energies[0]
    strengths = []
    for state_index in range(1, len(real_energies)):
        delta_e = real_energies[state_index] - reference_energy
        strengths.append((2.0 / 3.0) * delta_e * (np.linalg.norm(transition_vectors[state_index]) ** 2))
    return np.asarray(strengths, dtype=float)


def oscillator_strengths_complex_symmetric(
    energies: np.ndarray,
    transition_vectors: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Compute oscillator strengths for complex-symmetric OQS Hamiltonians.

    The dissipative full OQS-PCM Hamiltonian is complex symmetric, not
    Hermitian.  Its left eigenvector is therefore represented by the transpose
    of the right eigenvector rather than by the conjugate transpose.  The
    corresponding scalar product is the c-product, so neither the state
    normalization nor the dipole square uses complex conjugation:

    `f_i = 2/3 Re[(E_i - E_0) sum_alpha mu_0i,alpha^2 / ((c_0^T c_0)(c_i^T c_i))]`

    This function assumes `transition_vectors` were built with the same
    transpose convention used by `ground_to_excited_transition_vectors_from_states`.
    """

    reference_energy = energies[0]
    ground_c_norm = np.dot(states[0], states[0])
    strengths = []
    for state_index in range(1, len(energies)):
        state_c_norm = np.dot(states[state_index], states[state_index])
        norm_product = ground_c_norm * state_c_norm
        if abs(norm_product) == 0:
            raise ValueError("Cannot compute complex-symmetric oscillator strength for a zero c-product norm.")
        delta_e = energies[state_index] - reference_energy
        dipole_square = np.sum(transition_vectors[state_index] ** 2)
        strengths.append((2.0 / 3.0) * np.real(delta_e * dipole_square / norm_product))
    return np.asarray(strengths, dtype=float)
