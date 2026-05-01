from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oqs_pcm.dipole_algebra import (
    dipoles_assemble_completed_state_transition_tensor,
    dipoles_build_occupied_virtual_indices,
    dipoles_transform_ao_to_mo,
)
from oqs_pcm.io import io_load_state_transition_dipole_tensor, io_write_state_transition_dipole_tensor


@dataclass(slots=True)
class DipoleCompletionInputs:
    """Numerical ingredients needed to reconstruct a completed `ci_mut.inp`.

    This utility is needed for LR-TDDFT workflows where the electronic-structure
    program provides only ground-to-excited transition dipoles. OQSolv also
    needs ground-state, excited-state, and excited-excited state/transition
    dipoles for OQS-Onsager and oscillator-strength postprocessing.
    """

    ao_dipoles: np.ndarray
    mo_coefficients: np.ndarray
    x_amplitudes: np.ndarray
    y_amplitudes: np.ndarray
    n_occupied: int
    nuclear_dipole: np.ndarray
    n_frozen_occupied: int = 0


@dataclass(slots=True)
class DipoleCompletionDiagnostics:
    """Optional comparison against a limited reference `ci_mut.inp` file."""

    max_abs_ground_transition_difference: float
    compared_transitions: int


def reconstruct_completed_state_transition_dipoles(inputs: DipoleCompletionInputs) -> np.ndarray:
    """Reconstruct all state/transition electric dipoles from LR-TDDFT data.

    The calculation follows the validated `dipoles.nb` algebra:

    - transform AO dipole integrals to the MO basis
    - build occupied/virtual single-excitation labels
    - use TDDFT `X` and `Y` amplitudes to form excited-state and transition
      dipoles
    - compute the ground-state dipole including the nuclear contribution
    - assemble the completed tensor with the ground state at index 0
    """

    mo_dipoles = dipoles_transform_ao_to_mo(
        ao_dipoles=inputs.ao_dipoles,
        mo_coefficients=inputs.mo_coefficients,
    )
    occupied_virtual_indices = dipoles_build_occupied_virtual_indices(
        n_mos=mo_dipoles.shape[0],
        n_occupied=inputs.n_occupied,
        n_frozen_occupied=inputs.n_frozen_occupied,
    )
    return dipoles_assemble_completed_state_transition_tensor(
        mo_dipoles=mo_dipoles,
        x_coefficients=inputs.x_amplitudes,
        y_coefficients=inputs.y_amplitudes,
        occupied_virtual_indices=occupied_virtual_indices,
        n_occupied=inputs.n_occupied,
        nuclear_dipole=inputs.nuclear_dipole,
    )


def complete_ci_mut_file(
    *,
    output_path: str | Path,
    inputs: DipoleCompletionInputs,
) -> np.ndarray:
    """Reconstruct and write a completed `ci_mut.inp`-style file."""

    tensor = reconstruct_completed_state_transition_dipoles(inputs)
    io_write_state_transition_dipole_tensor(output_path, tensor)
    return tensor


def compare_ground_transition_dipoles_to_reference(
    completed_tensor: np.ndarray,
    reference_ci_mut: str | Path,
) -> DipoleCompletionDiagnostics:
    """Compare reconstructed GS-ES dipoles with a limited reference file.

    Locally modified GAMESS-US outputs may already contain ground-to-excited
    transition dipoles. Those rows are not enough for OQSolv, but they are a
    useful diagnostic for validating this reconstruction layer.
    """

    n_excited = completed_tensor.shape[0] - 1
    reference = io_load_state_transition_dipole_tensor(reference_ci_mut, nsts=n_excited)
    ref_ground_transitions = reference[0, 1:, :]
    mask = np.any(ref_ground_transitions != 0.0, axis=1)
    if not np.any(mask):
        return DipoleCompletionDiagnostics(
            max_abs_ground_transition_difference=0.0,
            compared_transitions=0,
        )
    diff = completed_tensor[0, 1:, :][mask] - ref_ground_transitions[mask]
    return DipoleCompletionDiagnostics(
        max_abs_ground_transition_difference=float(np.max(np.abs(diff))),
        compared_transitions=int(np.count_nonzero(mask)),
    )


def load_dipole_completion_inputs_from_npy(
    *,
    ao_dipoles_path: str | Path,
    mo_coefficients_path: str | Path,
    x_amplitudes_path: str | Path,
    y_amplitudes_path: str | Path,
    n_occupied: int,
    nuclear_dipole: np.ndarray,
    n_frozen_occupied: int = 0,
) -> DipoleCompletionInputs:
    """Load standalone dipole-completion inputs from NumPy array files.

    This is a deliberately simple first command-line interface. GAMESS-specific
    extraction/parsing should live in a separate layer so it can be validated
    independently from the scientific dipole algebra.
    """

    return DipoleCompletionInputs(
        ao_dipoles=np.load(ao_dipoles_path),
        mo_coefficients=np.load(mo_coefficients_path),
        x_amplitudes=np.load(x_amplitudes_path),
        y_amplitudes=np.load(y_amplitudes_path),
        n_occupied=n_occupied,
        nuclear_dipole=np.asarray(nuclear_dipole, dtype=float),
        n_frozen_occupied=n_frozen_occupied,
    )
