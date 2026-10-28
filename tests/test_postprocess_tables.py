from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oqs_pcm.io import io_write_named_results
from oqs_pcm.postprocess.observables import (
    ground_to_excited_transition_vectors_from_states,
    transform_state_transition_observable_to_basis,
)
from oqs_pcm.postprocess.tables import (
    compute_pcm_reference_polarization_shift,
    resolve_oscillator_strength_mode,
    write_absolute_energy_table,
    write_ground_state_convergence_table,
    write_spectrum_table,
    write_notebook_equivalent_tables,
)
from oqs_pcm.workflow import load_inertial_energy_offset


def test_pcm_reference_shift_adds_dynamic_work_for_inertial_reference(monkeypatch) -> None:
    ground_potential = np.array([2.0, -1.0])
    optical_kernel = np.array([0.5, 2.0])
    fake_config = SimpleNamespace(
        paths=SimpleNamespace(normal_mode_potential="v0.inp", lambda_file="Lambda.dat"),
        problem=SimpleNamespace(
            nsts=2,
            optical_window_min_ev=1.63,
            optical_window_max_ev=3.26,
        ),
        dielectric=object(),
    )

    monkeypatch.setattr(
        "oqs_pcm.postprocess.tables.pcm_load_ground_normal_mode_potential",
        lambda _: (2, ground_potential),
    )
    monkeypatch.setattr(
        "oqs_pcm.postprocess.tables.build_dielectric_model",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "oqs_pcm.postprocess.tables.build_pcm_kernels",
        lambda **_: SimpleNamespace(optical_kernel=optical_kernel),
    )

    # With K_plus = -K_paper, ww = -W_d = +1/2 V.T K_plus V.
    assert compute_pcm_reference_polarization_shift(fake_config) == 2.0


def test_absolute_energy_table_applies_reference_shift_and_delta_scf(tmp_path: Path) -> None:
    result_dir = tmp_path / "full"
    io_write_named_results(result_dir, np.array([0.1, 0.2]), np.eye(2))

    output = write_absolute_energy_table(
        result_dir,
        tmp_path / "en.txt",
        reference_shift=0.03,
        delta_scf=-0.01,
    )

    assert output == tmp_path / "en.txt"
    assert np.allclose(np.loadtxt(output), [0.12, 0.22])


def test_direct_inertial_offset_is_separate_from_excitation_energies(tmp_path: Path) -> None:
    energy_path = tmp_path / "ci_energy_inertial.inp"
    offset_path = tmp_path / "ci_energy_inertial_offset.inp"
    energy_path.write_text("0.0\n0.2\n", encoding="utf-8")
    offset_path.write_text("-0.35\n", encoding="utf-8")
    config = SimpleNamespace(
        execution=SimpleNamespace(
            potential_basis="normal_mode_inertial",
            reuse_inertial_cache=True,
        ),
        paths=SimpleNamespace(
            inertial_energy=energy_path,
            inertial_energy_offset=offset_path,
        ),
    )

    assert load_inertial_energy_offset(config) == -0.35


def test_direct_inertial_offset_is_added_to_absolute_level_tables(
    tmp_path: Path, monkeypatch
) -> None:
    solver_root = tmp_path / "solver"
    io_write_named_results(solver_root / "ss_pol", np.array([0.2, 0.4]), np.eye(2))
    energy_path = tmp_path / "ci_energy_inertial.inp"
    offset_path = tmp_path / "ci_energy_inertial_offset.inp"
    energy_path.write_text("0.0\n0.2\n", encoding="utf-8")
    offset_path.write_text("-0.35\n", encoding="utf-8")
    config = SimpleNamespace(
        paths=SimpleNamespace(
            output_dir=solver_root,
            normal_mode_potential="v0.inp",
            lambda_file="Lambda.dat",
            inertial_energy=energy_path,
            inertial_energy_offset=offset_path,
            inertial_dipoles=None,
        ),
        problem=SimpleNamespace(
            nsts=2,
            selected_states=None,
            optical_window_min_ev=1.63,
            optical_window_max_ev=3.26,
        ),
        execution=SimpleNamespace(
            potential_basis="normal_mode_inertial",
            reuse_inertial_cache=True,
            solve_inertial_basis=True,
        ),
        postprocess=SimpleNamespace(
            dipoles=None,
            dipole_basis="frozen",
            oscillator_strength_mode=None,
        ),
        dielectric=object(),
    )
    monkeypatch.setattr(
        "oqs_pcm.postprocess.tables.compute_pcm_reference_polarization_shift",
        lambda _: 0.1,
    )

    write_notebook_equivalent_tables(
        config,
        output_dir=tmp_path / "tables",
        include_levels=True,
        include_spectra=False,
    )

    assert np.allclose(np.loadtxt(tmp_path / "tables" / "ensspol.txt"), [-0.05, 0.15])


def test_spectrum_table_uses_energy_differences_so_constant_shift_cancels(tmp_path: Path) -> None:
    result_dir = tmp_path / "full"
    io_write_named_results(result_dir, np.array([1.0, 1.5]), np.eye(2))
    dipoles = np.zeros((2, 2, 3), dtype=float)
    dipoles[0, 1, 0] = 2.0
    dipoles[1, 0, 0] = 2.0

    output = write_spectrum_table(
        result_dir,
        tmp_path / "full.txt",
        state_transition_dipoles=dipoles,
        reference_shift=100.0,
    )

    table = np.loadtxt(output, ndmin=2)
    assert np.allclose(table[:, 0], [0.5])
    assert np.allclose(table[:, 1], [(2.0 / 3.0) * 0.5 * 4.0])


def test_spectrum_table_can_use_complex_symmetric_c_product_formula(tmp_path: Path) -> None:
    result_dir = tmp_path / "full"
    result_dir.mkdir()
    (result_dir / "newenergies.dat").write_text("0.0+0.0j\n1.0+0.5j\n", encoding="utf-8")
    (result_dir / "newstates.dat").write_text(
        "1.0+1.0j 0.0+0.0j\n0.0+0.0j 2.0+0.0j\n",
        encoding="utf-8",
    )
    dipoles = np.zeros((2, 2, 3), dtype=complex)
    dipoles[0, 1, 0] = 1.0
    dipoles[1, 0, 0] = 1.0

    output = write_spectrum_table(
        result_dir,
        tmp_path / "full.txt",
        state_transition_dipoles=dipoles,
        oscillator_strength_mode="complex_symmetric",
    )

    table = np.loadtxt(output, ndmin=2)
    assert np.allclose(table[:, 0], [1.0])
    assert np.allclose(table[:, 1], [2.0 / 3.0])


def test_oscillator_strength_mode_defaults_to_complex_symmetric_only_for_full_branches() -> None:
    assert resolve_oscillator_strength_mode(None, "full") == "complex_symmetric"
    assert resolve_oscillator_strength_mode(None, "full_nodiss") == "complex_symmetric"
    assert resolve_oscillator_strength_mode(None, "bo") == "hermitian"
    assert resolve_oscillator_strength_mode(None, "ss_pol") == "hermitian"
    assert resolve_oscillator_strength_mode("hermitian", "full") == "hermitian"


def test_ground_state_convergence_table_keeps_nstates_order(tmp_path: Path) -> None:
    root_100 = tmp_path / "n100" / "bo"
    root_50 = tmp_path / "n50" / "bo"
    io_write_named_results(root_100, np.array([0.2]), np.eye(1))
    io_write_named_results(root_50, np.array([0.1]), np.eye(1))

    output = write_ground_state_convergence_table(
        {100: root_100, 50: root_50},
        tmp_path / "GSsolbo.txt",
        reference_shift=0.5,
        delta_scf=-0.2,
    )

    table = np.loadtxt(output, ndmin=2)
    assert np.allclose(table, [[50, 0.4], [100, 0.5]])


def test_transform_observable_to_basis_matches_matrix_similarity_per_component() -> None:
    observable = np.zeros((2, 2, 3), dtype=float)
    observable[:, :, 0] = np.array([[1.0, 2.0], [2.0, 4.0]])
    basis = np.array([[1.0, 0.0], [0.0, -1.0]])

    transformed = transform_state_transition_observable_to_basis(observable, basis)

    assert np.allclose(transformed[:, :, 0], [[1.0, -2.0], [-2.0, 4.0]])


def test_ground_to_excited_transition_vectors_follow_solved_state_coefficients() -> None:
    observable = np.zeros((2, 2, 3), dtype=float)
    observable[0, 1, :] = [1.0, 2.0, 3.0]
    observable[1, 0, :] = [1.0, 2.0, 3.0]
    states = np.eye(2)

    vectors = ground_to_excited_transition_vectors_from_states(observable, states)

    assert np.allclose(vectors[0], [0.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [1.0, 2.0, 3.0])
