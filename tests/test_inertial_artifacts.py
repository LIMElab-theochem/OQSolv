from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oqs_pcm.config import (
    AppConfig,
    DielectricConfig,
    ExecutionConfig,
    PathsConfig,
    PostprocessConfig,
    ProblemConfig,
    SafetyConfig,
)
from oqs_pcm.io import io_load_state_transition_dipole_tensor
from oqs_pcm.potential import pcm_load_normal_mode_potential_tensor
from oqs_pcm.postprocess.tables import load_postprocess_dipoles
from oqs_pcm.postprocess.tables import write_notebook_equivalent_tables
from oqs_pcm.workflow import (
    ensure_normal_mode_potential,
    load_inertial_energy_offset,
    load_inertial_normal_mode_potential,
    load_inertial_energies,
    write_inertial_basis_artifacts,
)


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            mat_sd=tmp_path / "mat_SD.inp",
            cavity=tmp_path / "cavity.inp",
            ci_potential=tmp_path / "ci_pot.inp",
            normal_mode_potential=tmp_path / "ci_pot_normal_mode.inp",
            ci_energy=tmp_path / "ci_energy.inp",
            sm12t=tmp_path / "Sm12T.dat",
            lambda_file=tmp_path / "Lambda.dat",
            output_dir=tmp_path / "output",
        ),
        problem=ProblemConfig(
            nsts=2,
            diagonalization_states=2,
            corrected_states=2,
            convergence_tolerance=1e-6,
            mixing_window=2,
            cutoff=0.1,
            fit_cutoff=0.2,
            fit_epsilon=2.0,
            optical_window_min_ev=1.0,
            optical_window_max_ev=2.0,
            hartree_to_ev=27.2114,
            resolution_of_identity_states=2,
        ),
        execution=ExecutionConfig(
            build_cavity_mode_basis=False,
            build_potential_modes=False,
            solve_oqs_pcm=False,
            solve_inertial_basis=True,
            reuse_inertial_cache=True,
            write_text_exports=True,
            write_inertial_potentials=True,
            write_inertial_dipoles=True,
        ),
        dielectric=DielectricConfig(
            debye_c=[0.47],
            debye_g=[1.0],
            drude_w=[0.1],
            drude_c=[0.3],
            drude_g=[0.5],
            optical_drude_start_index=0,
        ),
        postprocess=PostprocessConfig(
            output_dir=None,
            dipoles=tmp_path / "ci_mut.inp",
            delta_scf=0.0,
            oscillator_strength_mode=None,
        ),
        safety=SafetyConfig(
            max_memory_gb=24.0,
            min_available_memory_gb=4.0,
            memory_safety_factor=2.5,
            min_free_disk_gb=4.0,
            disk_safety_factor=1.25,
        ),
    )


def test_write_inertial_artifacts_transforms_all_requested_observables(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    frozen = np.zeros((2, 2, 2))
    frozen[0, 0] = [0.0, 1.0]
    frozen[0, 1] = frozen[1, 0] = [2.0, 3.0]
    frozen[1, 1] = [4.0, 5.0]
    nuclear = np.array([10.0, 20.0])

    def fake_load_frozen(path: str, nsts: int):
        return 2, frozen.copy(), nuclear.copy()

    monkeypatch.setattr("oqs_pcm.workflow.pcm_load_frozen_state_transition_potential_tensor", fake_load_frozen)
    config.postprocess.dipoles.write_text(
        "States 1 and 1 1.0 2.0 3.0\n"
        "States 1 and 2 4.0 5.0 6.0\n"
        "States 2 and 2 7.0 8.0 9.0\n",
        encoding="utf-8",
    )
    normal_mode = frozen + 100.0
    states = np.eye(2)

    written = write_inertial_basis_artifacts(config, normal_mode, states, np.array([0.2, 0.5]))

    assert set(written) == {
        "energy",
        "energy_offset",
        "frozen_potential",
        "normal_mode_potential",
        "dipoles",
    }
    assert np.allclose(np.loadtxt(written["energy"]), [0.0, 0.3])
    assert np.allclose(np.loadtxt(written["energy_offset"]), [0.2])
    _, frozen_result = pcm_load_normal_mode_potential_tensor(written["frozen_potential"], 2)
    _, normal_result = pcm_load_normal_mode_potential_tensor(written["normal_mode_potential"], 2)
    expected_frozen = frozen.copy()
    expected_frozen[0, 0] += nuclear
    expected_frozen[1, 1] += nuclear
    np.testing.assert_allclose(frozen_result, expected_frozen)
    np.testing.assert_allclose(normal_result, normal_mode)
    dipoles_result = io_load_state_transition_dipole_tensor(written["dipoles"], 1)
    np.testing.assert_allclose(dipoles_result[0, 1], [4.0, 5.0, 6.0])


def test_generated_normal_modes_are_not_reloaded(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    config.execution.build_potential_modes = True
    generated = np.arange(8.0).reshape(2, 2, 2)

    monkeypatch.setattr("oqs_pcm.workflow.build_normal_mode_file", lambda _: generated)

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("generated normal modes must be passed in memory")

    monkeypatch.setattr("oqs_pcm.workflow.pcm_load_normal_mode_potential_tensor", fail_if_reloaded)
    nts, result = ensure_normal_mode_potential(config)

    assert nts == 2
    np.testing.assert_array_equal(result, generated)


def test_pretransformed_normal_mode_potential_is_loaded_without_rotation(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path = tmp_path / "ci_pot_normal_mode_inertial.inp"
    expected = np.zeros((2, 2, 2))
    expected[0, 0] = [1.0, 2.0]
    expected[0, 1] = expected[1, 0] = [3.0, 4.0]
    expected[1, 1] = [5.0, 6.0]
    from oqs_pcm.io import io_write_pcm_normal_mode_potential_tensor

    io_write_pcm_normal_mode_potential_tensor(path, expected)
    config.paths.inertial_normal_mode_potential = path

    nts, result = load_inertial_normal_mode_potential(config)

    assert nts == 2
    np.testing.assert_allclose(result, expected)


def test_inertial_dipoles_are_not_rotated_again(tmp_path: Path) -> None:
    dipole_path = tmp_path / "ci_mut_inertial.inp"
    dipole_path.write_text(
        "States 1 and 1 1.0 2.0 3.0\n"
        "States 1 and 2 4.0 5.0 6.0\n"
        "States 2 and 2 7.0 8.0 9.0\n",
        encoding="utf-8",
    )

    result = load_postprocess_dipoles(
        dipole_path,
        total_states=2,
        inertial_state_file=tmp_path / "must_not_be_read.dat",
        dipole_basis="inertial",
    )

    np.testing.assert_allclose(result[0, 1], [4.0, 5.0, 6.0])


def test_direct_inertial_energy_input_precedes_legacy_cache(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.paths.inertial_energy = tmp_path / "ci_energy_inertial.inp"
    config.paths.inertial_energy.write_text("0.0\n0.125\n", encoding="utf-8")
    legacy_energy, legacy_states = config.paths.output_dir / "inertial_diag2" / "newenergies.dat", config.paths.output_dir / "inertial_diag2" / "newstates.dat"
    legacy_energy.parent.mkdir(parents=True)
    legacy_energy.write_text("0.0\n0.25\n", encoding="utf-8")
    legacy_states.write_text("1 0\n0 1\n", encoding="utf-8")

    result = load_inertial_energies(config)

    np.testing.assert_allclose(result, [0.0, 0.125])


def test_direct_inertial_offset_is_loaded_as_one_value(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.execution.potential_basis = "normal_mode_inertial"
    config.paths.inertial_energy = tmp_path / "ci_energy_inertial.inp"
    config.paths.inertial_energy_offset = tmp_path / "ci_energy_inertial_offset.inp"
    config.paths.inertial_energy.write_text("0.0\n0.125\n", encoding="utf-8")
    config.paths.inertial_energy_offset.write_text("-0.375\n", encoding="utf-8")

    assert load_inertial_energy_offset(config) == -0.375


def test_reuse_selects_complete_inertial_dipoles_for_postprocessing(
    tmp_path: Path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    config.paths.output_dir = tmp_path / "solver"
    config.paths.inertial_dipoles = tmp_path / "ci_mut_inertial.inp"
    config.postprocess.dipoles = tmp_path / "ci_mut_frozen.inp"
    result_dir = config.paths.output_dir / "ss_pol"
    result_dir.mkdir(parents=True)
    (result_dir / "newenergies.dat").write_text("0.0\n0.2\n", encoding="utf-8")
    (result_dir / "newstates.dat").write_text("1 0\n0 1\n", encoding="utf-8")

    captured = {}
    monkeypatch.setattr(
        "oqs_pcm.postprocess.tables.compute_pcm_reference_polarization_shift",
        lambda _: 0.0,
    )

    def fake_load(path, **kwargs):
        captured["path"] = path
        captured["basis"] = kwargs["dipole_basis"]
        return np.zeros((2, 2, 3))

    monkeypatch.setattr("oqs_pcm.postprocess.tables.load_postprocess_dipoles", fake_load)

    write_notebook_equivalent_tables(
        config,
        output_dir=tmp_path / "tables",
        dipole_file=None,
        include_levels=False,
        include_spectra=True,
    )

    assert captured == {"path": config.paths.inertial_dipoles, "basis": "inertial"}


def test_direct_inertial_solver_route_does_not_read_frozen_energy(
    tmp_path: Path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    config.paths.ci_energy = None
    config.paths.inertial_normal_mode_potential = tmp_path / "ci_pot_normal_mode_inertial.inp"
    config.paths.inertial_energy = tmp_path / "ci_energy_inertial.inp"
    config.execution.potential_basis = "normal_mode_inertial"
    config.execution.write_inertial_potentials = False
    config.execution.write_inertial_dipoles = False
    config.paths.inertial_energy.write_text("0.0\n0.125\n", encoding="utf-8")
    from oqs_pcm.io import io_write_pcm_normal_mode_potential_tensor

    io_write_pcm_normal_mode_potential_tensor(
        config.paths.inertial_normal_mode_potential,
        np.zeros((2, 2, 1)),
    )
    config.paths.lambda_file.write_text("1.0\n", encoding="utf-8")

    monkeypatch.setattr(
        "oqs_pcm.workflow.build_dielectric_model",
        lambda _: object(),
    )
    monkeypatch.setattr(
        "oqs_pcm.workflow.build_pcm_kernels",
        lambda **_: SimpleNamespace(optical_kernel=np.ones(1)),
    )
    monkeypatch.setattr(
        "oqs_pcm.workflow.solve_born_oppenheimer_limit",
        lambda *args: (np.array([0.0, 0.125]), np.eye(2)),
    )

    from oqs_pcm.workflow import solve_oqs_pcm

    result = solve_oqs_pcm(config, requested_variants={"bo"})

    assert "bo" in result
