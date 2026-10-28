from __future__ import annotations

from pathlib import Path

import numpy as np


def io_load_numeric_table(path: str | Path) -> np.ndarray:
    """Load whitespace-separated numeric data from a plain-text file.

    Prefix rationale:
    `io_` marks this as a data-access helper rather than as a scientific
    routine. This naming convention is used throughout the refactor so that
    scientific code paths stand out from plumbing code when a domain user reads
    the sources.
    """

    return np.loadtxt(path, ndmin=2)


def io_load_excited_state_energy_vector(path: str | Path, hartree_to_ev: float) -> np.ndarray:
    """Read a CI energy table and return the excitation-energy vector in Hartree.

    The first entry is the ground-state reference energy, set to zero to match
    the conventions used in the Mathematica implementation and the OQS-Continuum
    derivations in the manuscript.
    """

    energies_raw = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        last_token = stripped.split()[-1].replace("D", "E").replace("d", "e")
        energies_raw.append(float(last_token))
    energies = np.asarray(energies_raw, dtype=float) / hartree_to_ev
    return np.insert(energies, 0, 0.0)


def io_load_pcm_lambda_eigenvalues(path: str | Path) -> np.ndarray:
    """Load the PCM cavity-eigenvalue vector from `Lambda.dat`."""

    return io_load_numeric_table(path)[:, 0]


def io_load_normal_mode_transformation_matrix(path: str | Path) -> np.ndarray:
    """Load the normal-mode transformation matrix used for coupling tensors."""

    return io_load_numeric_table(path).T


def io_write_vector(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values).reshape(-1, 1), fmt="%.18e")


def io_write_matrix(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values), fmt="%.18e")


def io_load_vector(path: str | Path) -> np.ndarray:
    """Load a one-dimensional numeric text export."""

    return np.loadtxt(path, ndmin=1)


def io_load_inertial_energy_vector(path: str | Path) -> np.ndarray:
    """Load an inertial energy vector stored in atomic units.

    Unlike ``ci_energy.inp``, an inertial energy artifact is a complete vector
    in atomic units.  The direct-reuse artifact is shifted so that its first
    entry is zero; the historical ``newenergies.dat`` cache may retain the
    absolute inertial reference and is validated separately by the workflow.
    The loader accepts the real or complex text representation emitted by
    NumPy and rejects genuinely complex reference energies.
    """

    values = np.loadtxt(path, dtype=complex, ndmin=1)
    if np.any(np.abs(np.imag(values)) > 1.0e-12):
        raise ValueError(f"Inertial reference energies must be real: {path}")
    return np.real(values)


def io_load_matrix(path: str | Path) -> np.ndarray:
    """Load a two-dimensional numeric text export."""

    return np.loadtxt(path, ndmin=2)


def io_write_named_results(base_dir: str | Path, energies: np.ndarray, states: np.ndarray) -> None:
    """Mirror the Mathematica result layout using two plain-text files."""

    base = Path(base_dir)
    io_write_vector(base / "newenergies.dat", energies)
    io_write_matrix(base / "newstates.dat", states)


def io_write_pcm_normal_mode_potential_tensor(path: str | Path, potential: np.ndarray) -> None:
    """Write the PCM normal-mode potential tensor in the legacy text format.

    The tensor contains state-specific and transition electrostatic potentials
    over PCM cavity tesserae, already transformed to the chosen normal-mode
    basis. The text layout is preserved so existing datasets remain reusable.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nstates, _, nts = potential.shape

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{nts}\n")
        for i_state in range(nstates):
            for j_state in range(i_state, nstates):
                handle.write(f"{i_state + 1} {j_state + 1}\n")
                for tessera in range(nts):
                    handle.write(f"{potential[i_state, j_state, tessera]:.18e}\n")


def io_load_state_transition_dipole_tensor(path: str | Path, nsts: int, state_offset: int = 1) -> np.ndarray:
    """Load state/transition electric dipole moments into a symmetric tensor.

    These dipoles are general molecular observables.  They are used by the
    Onsager coupling model, but also by PCM/OQS-PCM postprocessing such as
    oscillator strengths.  Two text layouts are accepted:

    - legacy completed `ci_mut` rows:
      `States i and j mux muy muz`, with one-based state labels
    - lightweight numeric rows whose relevant columns are:
      column 2: initial state index; column 4: final state index; last three
      columns: x, y, z dipole components

    `state_offset` applies only to the numeric layout.  The legacy `States`
    layout is already one-based and includes the ground state.
    """

    tensor = np.zeros((nsts + 1, nsts + 1, 3), dtype=float)
    rows = [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    for tokens in rows:
        if tokens[0].lower() == "states":
            i_state = int(tokens[1]) - 1
            j_state = int(tokens[3]) - 1
        else:
            i_state = int(float(tokens[1])) + state_offset
            j_state = int(float(tokens[3])) + state_offset

        dipole = np.asarray([float(token.replace("D", "E").replace("d", "e")) for token in tokens[-3:]])
        tensor[i_state, j_state, :] = dipole
        tensor[j_state, i_state, :] = dipole

    return tensor


def io_write_state_transition_dipole_tensor(path: str | Path, dipole_tensor: np.ndarray) -> None:
    """Write a completed state/transition dipole tensor in legacy `ci_mut` form."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nstates = dipole_tensor.shape[0]

    with path.open("w", encoding="utf-8") as handle:
        for i_state in range(nstates):
            for j_state in range(i_state, nstates):
                mux, muy, muz = dipole_tensor[i_state, j_state, :]
                handle.write(
                    f"States {i_state + 1} and {j_state + 1} "
                    f"{mux:.18e} {muy:.18e} {muz:.18e}\n"
                )


def io_load_onsager_transition_dipole_tensor(path: str | Path, nsts: int, state_offset: int = 1) -> np.ndarray:
    """Backward-compatible alias for the general state/transition dipole loader."""

    return io_load_state_transition_dipole_tensor(path, nsts=nsts, state_offset=state_offset)


# Backward-compatible aliases used while the naming refactor propagates.
load_numeric_table = io_load_numeric_table
load_energy_vector = io_load_excited_state_energy_vector
load_lambda_vector = io_load_pcm_lambda_eigenvalues
load_sm12t = io_load_normal_mode_transformation_matrix
write_vector = io_write_vector
write_matrix = io_write_matrix
write_named_results = io_write_named_results
write_normal_mode_potential = io_write_pcm_normal_mode_potential_tensor
load_state_transition_dipoles = io_load_state_transition_dipole_tensor
write_state_transition_dipoles = io_write_state_transition_dipole_tensor
load_transition_dipoles = io_load_state_transition_dipole_tensor
