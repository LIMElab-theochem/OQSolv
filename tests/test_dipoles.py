from pathlib import Path

import numpy as np

from oqs_pcm.dipoles import (
    dipoles_assemble_completed_state_transition_tensor,
    dipoles_build_occupied_virtual_indices,
    dipoles_build_slater_dipole_matrices,
)
from oqs_pcm.gamess import (
    gamess_extract_ci_energies,
    gamess_extract_potential_block,
    gamess_extract_reference_ground_excited_dipoles,
)
from oqs_pcm.io import io_load_state_transition_dipole_tensor, io_write_state_transition_dipole_tensor


def test_completed_ci_mut_round_trip_uses_one_based_total_state_labels(tmp_path: Path) -> None:
    tensor = np.zeros((3, 3, 3), dtype=float)
    tensor[0, 0] = [1.0, 0.0, 0.0]
    tensor[0, 1] = tensor[1, 0] = [0.1, 0.2, 0.3]
    tensor[1, 2] = tensor[2, 1] = [0.4, 0.5, 0.6]

    path = tmp_path / "ci_mut.inp"
    io_write_state_transition_dipole_tensor(path, tensor)
    loaded = io_load_state_transition_dipole_tensor(path, nsts=2)

    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("States 1 and 1")
    np.testing.assert_allclose(loaded, tensor)


def test_gamess_reference_dipoles_are_zero_based_diagnostics(tmp_path: Path) -> None:
    log_path = tmp_path / "gamess.log"
    log_path.write_text(
        "\n".join(
            [
                " 0 A label 0.000 1.0 0.0 0.0",
                " 1 A label 3.307 0.1 0.2 0.3",
                " 2 A label 6.491 0.4 0.5 0.6",
            ]
        ),
        encoding="utf-8",
    )

    assert gamess_extract_ci_energies(log_path) == [(1, 3.307), (2, 6.491)]
    assert gamess_extract_reference_ground_excited_dipoles(log_path) == [
        (0, 0, (1.0, 0.0, 0.0)),
        (0, 1, (0.1, 0.2, 0.3)),
        (0, 2, (0.4, 0.5, 0.6)),
    ]


def test_gamess_potential_block_removes_marker_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "gamess.log"
    log_path.write_text(
        "before\n"
        " INIT POT FILE FOR WAVET\n"
        "335\n"
        "1 1\n"
        "0.123\n"
        " END POT FILE FOR WAVET\n"
        "after\n",
        encoding="utf-8",
    )

    assert gamess_extract_potential_block(log_path) == "335\n1 1\n0.123\n"


def test_dipole_tensor_assembly_matches_small_hand_checked_case() -> None:
    mo_dipoles = np.zeros((2, 2, 3), dtype=float)
    mo_dipoles[0, 0] = [1.0, 0.0, 0.0]
    mo_dipoles[1, 1] = [3.0, 0.0, 0.0]
    mo_dipoles[0, 1] = mo_dipoles[1, 0] = [2.0, 0.0, 0.0]
    ov = dipoles_build_occupied_virtual_indices(n_mos=2, n_occupied=1)
    x_coefficients = np.array([[1.0]])
    y_coefficients = np.array([[0.0]])
    nuclear = np.array([10.0, 0.0, 0.0])

    slater = dipoles_build_slater_dipole_matrices(mo_dipoles, ov, n_occupied=1)
    tensor = dipoles_assemble_completed_state_transition_tensor(
        mo_dipoles=mo_dipoles,
        x_coefficients=x_coefficients,
        y_coefficients=y_coefficients,
        occupied_virtual_indices=ov,
        n_occupied=1,
        nuclear_dipole=nuclear,
    )

    np.testing.assert_allclose(slater[0, 0], [4.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[0, 0], [8.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[0, 1], [-np.sqrt(2.0) * 2.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[1, 1], [6.0, 0.0, 0.0])
