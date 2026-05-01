from __future__ import annotations

import numpy as np


def dipoles_transform_ao_to_mo(ao_dipoles: np.ndarray, mo_coefficients: np.ndarray) -> np.ndarray:
    """Transform Cartesian dipole-integral matrices from AO to MO basis.

    This is the Python equivalent of the Mathematica operation
    `mos . (xint . Transpose[mos])`, repeated for x, y, and z.  The expected
    shapes are:

    - `ao_dipoles[ao_i, ao_j, cartesian]`
    - `mo_coefficients[mo_i, ao_i]`
    - result `dip[mo_i, mo_j, cartesian]`
    """

    return np.einsum("ma,abc,nb->mnc", mo_coefficients, ao_dipoles, mo_coefficients)


def dipoles_build_occupied_virtual_indices(
    n_mos: int,
    n_occupied: int,
    n_frozen_occupied: int = 0,
) -> np.ndarray:
    """Build zero-based occupied/virtual single-excitation indices.

    The notebook stores each Slater determinant by the occupied orbital `i` and
    virtual orbital `a` involved in the excitation.  Frozen occupied orbitals are
    excluded from the excitation manifold but remain part of the closed-shell
    ground-state density used in the ground-state dipole.
    """

    pairs = [
        (occupied, virtual)
        for occupied in range(n_frozen_occupied, n_occupied)
        for virtual in range(n_occupied, n_mos)
    ]
    return np.asarray(pairs, dtype=int)


def dipoles_build_slater_dipole_matrices(
    mo_dipoles: np.ndarray,
    occupied_virtual_indices: np.ndarray,
    n_occupied: int,
) -> np.ndarray:
    """Build dipole matrices in the single-excitation Slater basis.

    This implements the Slater-Condon matrix construction repeated in
    `dipoles.nb` for the x, y, and z components.  For excitations `i -> a` and
    `j -> b`, the one-particle dipole matrix element is:

    - diagonal: `2 sum_occ mu_kk - mu_ii + mu_aa`
    - same occupied orbital: `mu_ab`
    - same virtual orbital: `-mu_ji`
    - otherwise: zero
    """

    ov = np.asarray(occupied_virtual_indices, dtype=int)
    n_slater = len(ov)
    n_components = mo_dipoles.shape[2]
    matrices = np.zeros((n_slater, n_slater, n_components), dtype=float)
    closed_shell_sum = 2.0 * np.sum(mo_dipoles[:n_occupied, :n_occupied, :].diagonal(axis1=0, axis2=1), axis=1)

    for left, (i_occ, a_virt) in enumerate(ov):
        for right, (j_occ, b_virt) in enumerate(ov):
            if left == right:
                matrices[left, right, :] = (
                    closed_shell_sum
                    - mo_dipoles[i_occ, i_occ, :]
                    + mo_dipoles[a_virt, a_virt, :]
                )
            elif i_occ == j_occ:
                matrices[left, right, :] = mo_dipoles[a_virt, b_virt, :]
            elif a_virt == b_virt:
                matrices[left, right, :] = -mo_dipoles[j_occ, i_occ, :]
    return matrices


def dipoles_compute_excited_state_dipoles(
    x_coefficients: np.ndarray,
    y_coefficients: np.ndarray,
    slater_dipole_matrices: np.ndarray,
) -> np.ndarray:
    """Compute excited-state and excited-transition dipoles in CIS/TDDFT space.

    The notebook evaluates `X . M . Transpose[X] - Y . M . Transpose[Y]` for
    each Cartesian component.  The returned tensor has shape
    `(n_excited, n_excited, 3)`.
    """

    x_part = np.einsum("is,stc,jt->ijc", x_coefficients, slater_dipole_matrices, x_coefficients)
    y_part = np.einsum("is,stc,jt->ijc", y_coefficients, slater_dipole_matrices, y_coefficients)
    return x_part - y_part


def dipoles_compute_ground_state_dipole(
    mo_dipoles: np.ndarray,
    n_occupied: int,
    nuclear_dipole: np.ndarray,
) -> np.ndarray:
    """Compute the closed-shell RHF ground-state molecular dipole."""

    electronic = -2.0 * np.sum(mo_dipoles[:n_occupied, :n_occupied, :].diagonal(axis1=0, axis2=1), axis=1)
    return electronic + np.asarray(nuclear_dipole, dtype=float)


def dipoles_compute_ground_to_excited_transition_dipoles(
    transition_coefficients: np.ndarray,
    mo_dipoles: np.ndarray,
    occupied_virtual_indices: np.ndarray,
) -> np.ndarray:
    """Compute ground-to-excited transition dipoles.

    In the notebook `transition_coefficients` is `Xcoeff + Ycoeff`, and the
    transition dipole is `-sqrt(2) Sum_ia C_state,ia mu_i,a`.
    """

    ov = np.asarray(occupied_virtual_indices, dtype=int)
    orbital_transition_dipoles = mo_dipoles[ov[:, 0], ov[:, 1], :]
    return -np.sqrt(2.0) * np.einsum("is,sc->ic", transition_coefficients, orbital_transition_dipoles)


def dipoles_assemble_completed_state_transition_tensor(
    mo_dipoles: np.ndarray,
    x_coefficients: np.ndarray,
    y_coefficients: np.ndarray,
    occupied_virtual_indices: np.ndarray,
    n_occupied: int,
    nuclear_dipole: np.ndarray,
) -> np.ndarray:
    """Assemble the completed `ci_mut`-style state/transition dipole tensor.

    The output includes the ground state as index 0, followed by the excited
    states.  This is the semantic replacement for the final `mu` tensor built in
    `dipoles.nb`.
    """

    n_excited = x_coefficients.shape[0]
    tensor = np.zeros((n_excited + 1, n_excited + 1, 3), dtype=float)

    slater_matrices = dipoles_build_slater_dipole_matrices(
        mo_dipoles=mo_dipoles,
        occupied_virtual_indices=occupied_virtual_indices,
        n_occupied=n_occupied,
    )
    cis_dipoles = dipoles_compute_excited_state_dipoles(
        x_coefficients=x_coefficients,
        y_coefficients=y_coefficients,
        slater_dipole_matrices=slater_matrices,
    )
    nuclear = np.asarray(nuclear_dipole, dtype=float)

    tensor[0, 0, :] = dipoles_compute_ground_state_dipole(
        mo_dipoles=mo_dipoles,
        n_occupied=n_occupied,
        nuclear_dipole=nuclear,
    )
    ground_transitions = dipoles_compute_ground_to_excited_transition_dipoles(
        transition_coefficients=x_coefficients + y_coefficients,
        mo_dipoles=mo_dipoles,
        occupied_virtual_indices=occupied_virtual_indices,
    )
    tensor[0, 1:, :] = ground_transitions
    tensor[1:, 0, :] = ground_transitions

    # The CIS electronic contribution uses the opposite sign in the exported
    # molecular dipoles; the nuclear contribution is added only to diagonal
    # excited-state blocks.
    tensor[1:, 1:, :] = -cis_dipoles
    for state_index in range(n_excited):
        tensor[state_index + 1, state_index + 1, :] += nuclear
    return tensor
