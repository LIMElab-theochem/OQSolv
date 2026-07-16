from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from oqs_pcm.dielectric import build_kernel_from_lambda, build_state_resolved_kernel_from_lambda
from oqs_pcm.io import io_load_pcm_lambda_eigenvalues, io_write_matrix, io_write_named_results, io_write_vector


@dataclass(slots=True)
class SolverConfig:
    nsts: int
    diagonalization_states: int
    corrected_states: int
    tolerance: float
    mixing_window: int
    cutoff: float
    write_text_exports: bool
    output_dir: Path


@dataclass(slots=True)
class KernelBundle:
    static_kernel: np.ndarray
    optical_kernel: np.ndarray
    dispersive_kernel_real: np.ndarray
    dispersive_kernel_imag: np.ndarray


def build_reference_diagonal_hamiltonian(energies: np.ndarray, n: int) -> np.ndarray:
    return np.diag(energies[:n])


def transform_coupling_tensor_to_inertial_basis(states: np.ndarray, potential: np.ndarray) -> np.ndarray:
    """Rotate a generic state-coupling tensor to a new electronic basis.

    The same algebra applies to:

    - PCM state/transition electrostatic potentials
    - Onsager state/transition dipole tensors

    The only requirement is a tensor with shape `(n_state, n_state, n_channel)`.
    """

    transformed_slices = []
    for tessera_index in range(potential.shape[2]):
        transformed_slices.append(states @ potential[:, :, tessera_index] @ states.T)
    stacked = np.stack(transformed_slices, axis=2)
    return 0.5 * (stacked + np.transpose(stacked, (1, 0, 2)))


def solve_inertial_basis_reference_problem(
    energies: np.ndarray,
    potential: np.ndarray,
    optical_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the inertial reference problem from the commented Mathematica block."""

    hamiltonian = np.diag(energies)
    # Mathematica builds the inertial correction in two steps:
    # 1. assemble the inertial cavity-charge-like vector from the ground-state
    #    diagonal block by a Hadamard product with the optical kernel
    # 2. contract that vector with every state/transition potential slice
    #
    # In symbols:
    #   qd0[t] = Kerneld[t] * V[0, 0, t]
    #   deltaH[i, j] = Sum_t qd0[t] * V[i, j, t]
    qd0 = optical_kernel * potential[0, 0, :]
    delta_h = np.einsum("t,ijt->ij", qd0, potential)
    evals, evecs = np.linalg.eigh(hamiltonian + delta_h)
    order = np.argsort(evals)
    return evals[order], evecs[:, order].T


def build_low_temperature_dissipation_factor(energies: np.ndarray, energy: float) -> np.ndarray:
    """Replicate `1 + Theta[w] - Theta[-w]` for a given trial energy."""

    delta = energy - energies
    return 1.0 + np.heaviside(delta, 1.0) - np.heaviside(-delta, 0.0)


def select_state_by_reference_overlap(evecs: np.ndarray, state_index: int) -> int:
    """Choose the eigenvector with the largest weight on the target state."""

    weights = np.abs(evecs[state_index, :]) ** 2
    return int(np.argmax(weights))


def build_pcm_state_resolved_kernel(
    lambda_values: np.ndarray,
    dielectric_model,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the real and imaginary dispersive kernels for every state."""

    # The dynamic OQS correction uses only the fast/non-inertial dielectric
    # response. The slow inertial part is already accounted for in the frozen
    # reference calculation and in the explicit inertial-basis step.
    eps_values = dielectric_model.epsilon(-energies, include_ir=False)
    kernel = build_state_resolved_kernel_from_lambda(lambda_values, eps_values)
    return np.real(kernel), np.imag(kernel)


def build_pcm_kernel_bundle(
    lambda_path: Path,
    dielectric_model,
    energies: np.ndarray,
    optical_window: tuple[float, float],
) -> KernelBundle:
    lambda_values = io_load_pcm_lambda_eigenvalues(lambda_path)
    static_eps = dielectric_model.epsilon_static()
    optical_eps = dielectric_model.epsilon_optical_average(*optical_window)
    dispersive_real, dispersive_imag = build_pcm_state_resolved_kernel(lambda_values, dielectric_model, energies)

    return KernelBundle(
        static_kernel=build_kernel_from_lambda(lambda_values, static_eps),
        optical_kernel=build_kernel_from_lambda(lambda_values, optical_eps),
        dispersive_kernel_real=dispersive_real,
        dispersive_kernel_imag=dispersive_imag,
    )


def build_coupling_induced_hamiltonian_shift(potential: np.ndarray, kernel_row: np.ndarray, n: int) -> np.ndarray:
    """Build the dense Hamiltonian correction used by several approximations.

    `potential` is intentionally named conservatively for backward continuity
    with the PCM codepath, but mathematically it is a generic coupling tensor.
    For Onsager this same routine acts on dipole-moment couplings.
    """

    scr = np.array([potential[row, :, :] * kernel_row[np.newaxis, :] for row in range(n)])
    return -0.5 * np.array(
        [[np.sum(potential[i, :, :] * scr[j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )


def build_coupling_induced_hamiltonian_shift_from_kernel_matrix(
    potential: np.ndarray,
    kernel_matrix: np.ndarray,
    n: int,
) -> np.ndarray:
    """Build the dense Hamiltonian correction from a state-resolved kernel.

    `kernel_matrix[k, t]` stores the solvent kernel for RI state `k` and
    cavity channel `t`. This matches the full OQS-PCM Mathematica algebra,
    where each intermediate state contributes its own channel-resolved kernel.
    """

    scr = np.array([potential[row, :, :] * kernel_matrix for row in range(n)])
    return -0.5 * np.array(
        [[np.sum(potential[i, :, :] * scr[j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )


def build_reference_state_only_hamiltonian_shift(potential: np.ndarray, kernel_row: np.ndarray, n: int, state_index: int) -> np.ndarray:
    """Build the simplified correction that only uses one reference state."""

    scr = np.array([potential[row, state_index, :] * kernel_row for row in range(n)])
    return -0.5 * np.array(
        [[np.sum(potential[i, state_index, :] * scr[j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )


def build_reference_state_only_hamiltonian_shift_from_kernel_matrix(
    potential: np.ndarray,
    kernel_matrix: np.ndarray,
    n: int,
    state_index: int,
) -> np.ndarray:
    """Build the simplified correction from one target-state kernel row."""

    kernel_row = kernel_matrix[state_index, :]
    scr = np.array([potential[row, state_index, :] * kernel_row for row in range(n)])
    return -0.5 * np.array(
        [[np.sum(potential[i, state_index, :] * scr[j]) for j in range(n)] for i in range(n)],
        dtype=complex,
    )


def resolve_kernel(
    kernel: np.ndarray | callable,
    state_index: int,
    trial_energy: float,
) -> np.ndarray:
    """Resolve either a precomputed kernel array or a trial-energy callback."""

    if callable(kernel):
        return np.asarray(kernel(state_index, trial_energy))
    return np.asarray(kernel)


def solve_state_specific_energy_iteratively(
    base_h: np.ndarray,
    potential: np.ndarray,
    energies: np.ndarray,
    state_index: int,
    kernel_real: np.ndarray | callable,
    kernel_imag: np.ndarray | callable | None,
    tolerance: float,
    mixing_window: int,
    diagonal_only: bool = False,
    first_iteration_only: bool = False,
) -> tuple[complex, np.ndarray]:
    """Run the repeated self-consistent state loop used in the .m scripts.

    This is the shared OQS engine. PCM and Onsager differ in how they build the
    coupling tensor and the state-dependent kernel rows, but once those objects
    exist, the iterative diagonalization procedure is the same.
    """

    history = [float(np.real(energies[state_index]))]

    while True:
        if len(history) <= mixing_window:
            trial_energy = history[-1]
        else:
            trial_energy = float(np.mean(history[-mixing_window:]))

        resolved_real_kernel = resolve_kernel(kernel_real, state_index, trial_energy)
        if diagonal_only:
            if resolved_real_kernel.ndim == 1:
                real_shift = build_reference_state_only_hamiltonian_shift(
                    potential, resolved_real_kernel, base_h.shape[0], state_index
                )
            else:
                real_shift = build_reference_state_only_hamiltonian_shift_from_kernel_matrix(
                    potential, resolved_real_kernel, base_h.shape[0], state_index
                )
        else:
            if resolved_real_kernel.ndim == 1:
                real_shift = build_coupling_induced_hamiltonian_shift(
                    potential, resolved_real_kernel, base_h.shape[0]
                )
            else:
                real_shift = build_coupling_induced_hamiltonian_shift_from_kernel_matrix(
                    potential, resolved_real_kernel, base_h.shape[0]
                )

        delta_h = real_shift

        if kernel_imag is not None:
            tfactor = build_low_temperature_dissipation_factor(energies, trial_energy)
            resolved_imag_kernel = resolve_kernel(kernel_imag, state_index, trial_energy)
            if resolved_imag_kernel.ndim == 1:
                imag_kernel = tfactor * resolved_imag_kernel
            else:
                imag_kernel = tfactor[:, np.newaxis] * resolved_imag_kernel
            if diagonal_only:
                if imag_kernel.ndim == 1:
                    imag_shift = build_reference_state_only_hamiltonian_shift(
                        potential, imag_kernel, base_h.shape[0], state_index
                    )
                else:
                    imag_shift = build_reference_state_only_hamiltonian_shift_from_kernel_matrix(
                        potential, imag_kernel, base_h.shape[0], state_index
                    )
            else:
                if imag_kernel.ndim == 1:
                    imag_shift = build_coupling_induced_hamiltonian_shift(
                        potential, imag_kernel, base_h.shape[0]
                    )
                else:
                    imag_shift = build_coupling_induced_hamiltonian_shift_from_kernel_matrix(
                        potential, imag_kernel, base_h.shape[0]
                    )
            # `imag_shift` already includes the `-0.5` prefactor from the
            # shared shift builders. The Mathematica code applies `-0.5 I`
            # directly to the unfactored matrix, so at this stage the correct
            # mapping is `+ I * imag_shift`.
            delta_h = delta_h + 1j * imag_shift

        evals, evecs = np.linalg.eig(base_h + delta_h)
        order = np.argsort(np.real(evals))
        evals = evals[order]
        evecs = evecs[:, order]
        chosen = select_state_by_reference_overlap(evecs, state_index)
        energy = evals[chosen]
        state = evecs[:, chosen]

        if first_iteration_only:
            return energy, state

        converged = abs(np.real(energy - history[-1])) < tolerance
        history.append(float(np.real(energy)))
        if converged:
            return energy, state


def solve_state_specific_family(
    label: str,
    base_h: np.ndarray,
    potential: np.ndarray,
    energies: np.ndarray,
    kernel_real: np.ndarray | callable,
    kernel_imag: np.ndarray | callable | None,
    cfg: SolverConfig,
    diagonal_only: bool = False,
    first_iteration_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one of the repeated state-by-state approximations."""

    count = min(cfg.corrected_states, base_h.shape[0])
    new_energies = np.empty(count, dtype=complex)
    new_states = np.empty((count, base_h.shape[0]), dtype=complex)
    result_dir = cfg.output_dir / label

    for state_index in range(count):
        energy, state = solve_state_specific_energy_iteratively(
            base_h=base_h,
            potential=potential,
            energies=energies,
            state_index=state_index,
            kernel_real=kernel_real,
            kernel_imag=kernel_imag,
            tolerance=cfg.tolerance,
            mixing_window=cfg.mixing_window,
            diagonal_only=diagonal_only,
            first_iteration_only=first_iteration_only,
        )
        new_energies[state_index] = energy
        new_states[state_index, :] = state
        if cfg.write_text_exports:
            write_state_specific_progress(
                result_dir,
                new_energies[: state_index + 1],
                new_states[: state_index + 1, :],
                completed_states=state_index + 1,
                total_states=count,
            )

    if cfg.write_text_exports:
        # `np.linalg.eig` returns right eigenvectors normalized with the usual
        # Euclidean norm, as if the problem were Hermitian.  For dissipative
        # full OQS-PCM branches the Hamiltonian is complex symmetric instead;
        # property calculations that use these exported states must therefore
        # apply c-product normalization at the point where the property is
        # evaluated.
        io_write_named_results(result_dir, new_energies, new_states)

    return new_energies, new_states


def write_state_specific_progress(
    result_dir: Path,
    energies: np.ndarray,
    states: np.ndarray,
    *,
    completed_states: int,
    total_states: int,
) -> None:
    """Write incremental state-specific results for long OQS runs.

    The final `newenergies.dat` and `newstates.dat` files are still written only
    after the full variant completes.  These partial files are monitoring and
    recovery aids for expensive full-space calculations; OQSolv does not yet
    resume from them automatically.
    """

    result_dir.mkdir(parents=True, exist_ok=True)
    io_write_vector(result_dir / "newenergies_partial.dat", energies)
    io_write_matrix(result_dir / "newstates_partial.dat", states)
    progress = {
        "completed_states": completed_states,
        "total_states": total_states,
        "complete": completed_states >= total_states,
    }
    (result_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


def solve_static_state_specific_family(
    label: str,
    base_h: np.ndarray,
    potential: np.ndarray,
    optical_kernel: np.ndarray,
    cfg: SolverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the non-iterative static state-specific approximation.

    This matches the Mathematica `ss-pol` branch: for each target state, build
    one state-specific Hamiltonian correction from the static/inertial kernel,
    diagonalize once, and select the eigenstate by maximum reference overlap.
    No self-consistent energy iteration is performed.
    """

    count = min(cfg.corrected_states, base_h.shape[0])
    new_energies = np.empty(count, dtype=float)
    new_states = np.empty((count, base_h.shape[0]), dtype=float)

    for state_index in range(count):
        delta_h = build_reference_state_only_hamiltonian_shift(
            potential=potential,
            kernel_row=optical_kernel,
            n=base_h.shape[0],
            state_index=state_index,
        ).real
        evals, evecs = np.linalg.eigh(base_h + delta_h)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        chosen = select_state_by_reference_overlap(evecs, state_index)
        new_energies[state_index] = evals[chosen]
        new_states[state_index, :] = evecs[:, chosen]

    if cfg.write_text_exports:
        io_write_named_results(cfg.output_dir / label, new_energies, new_states)

    return new_energies, new_states


def solve_born_oppenheimer_limit(
    base_h: np.ndarray,
    potential: np.ndarray,
    optical_kernel: np.ndarray,
    cfg: SolverConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the BO-sol limiting case."""

    n = base_h.shape[0]
    scr = np.array([potential[row, :, :] * optical_kernel for row in range(n)])
    delta_h = -0.5 * np.array(
        [[np.sum(potential[i, :, :] * scr[j]) for j in range(n)] for i in range(n)],
        dtype=float,
    )
    evals, evecs = np.linalg.eigh(base_h + delta_h)
    order = np.argsort(evals)
    count = min(cfg.corrected_states, n)
    selected_energies = evals[order][:count]
    selected_states = evecs[:, order].T[:count]

    if cfg.write_text_exports:
        io_write_named_results(cfg.output_dir / "bo", selected_energies, selected_states)

    return selected_energies, selected_states


def solve_cutoff_approximation(
    base_h: np.ndarray,
    potential: np.ndarray,
    energies: np.ndarray,
    optical_kernel: np.ndarray,
    cfg: SolverConfig,
    *,
    label: str = "cutoff",
    cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the cutoff approximation.

    `fit_cutoff` uses this same algebra with a different dielectric kernel and
    frequency threshold. Keeping the implementation shared avoids giving the
    fitted variant any hidden numerical differences beyond those parameters.
    """

    count = min(cfg.corrected_states, base_h.shape[0])
    n = base_h.shape[0]
    nsts = potential.shape[1]
    cutoff_value = cfg.cutoff if cutoff is None else cutoff
    theta = (np.abs(energies[:nsts, None] - energies[None, :nsts]) <= cutoff_value).astype(float)
    new_energies = np.empty(count, dtype=float)
    new_states = np.empty((count, n), dtype=float)

    for state_index in range(count):
        masked_kernel = theta[state_index, :, None] * optical_kernel[np.newaxis, :]
        delta_h = build_coupling_induced_hamiltonian_shift_from_kernel_matrix(
            potential=potential,
            kernel_matrix=masked_kernel,
            n=n,
        )
        evals, evecs = np.linalg.eigh(base_h + delta_h.real)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        chosen = select_state_by_reference_overlap(evecs, state_index)
        new_energies[state_index] = evals[chosen]
        new_states[state_index, :] = evecs[:, chosen]

    if cfg.write_text_exports:
        io_write_named_results(cfg.output_dir / label, new_energies, new_states)

    return new_energies, new_states


# Backward-compatible aliases while the scientific naming refactor propagates.
diagonal_hamiltonian = build_reference_diagonal_hamiltonian
basis_transform_potential = transform_coupling_tensor_to_inertial_basis
solve_inertial_basis = solve_inertial_basis_reference_problem
build_tfactor = build_low_temperature_dissipation_factor
pick_state_closest_to_reference = select_state_by_reference_overlap
build_state_resolved_kernel = build_pcm_state_resolved_kernel
build_kernel_bundle = build_pcm_kernel_bundle
matrix_shift = build_coupling_induced_hamiltonian_shift
diagonal_only_shift = build_reference_state_only_hamiltonian_shift
iterate_state_energy = solve_state_specific_energy_iteratively
run_statewise_solver = solve_state_specific_family
run_static_statewise_solver = solve_static_state_specific_family
run_bo_solvent = solve_born_oppenheimer_limit
run_cutoff = solve_cutoff_approximation
