from __future__ import annotations

from pathlib import Path

from oqs_pcm.config import load_config


def test_load_config_reads_expected_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(
        """
[paths]
mat_sd = "/tmp/mat_SD.inp"
cavity = "/tmp/cavity.inp"
ci_potential = "/tmp/ci_pot.inp"
normal_mode_potential = "/tmp/ci_pot_normal_mode.inp"
ci_energy = "/tmp/ci_energy.inp"
sm12t = "/tmp/Sm12T.dat"
lambda_file = "/tmp/Lambda.dat"
output_dir = "/tmp/output"

[problem]
nsts = 5
diagonalization_states = 3
resolution_of_identity_states = 4
corrected_states = 2
convergence_tolerance = 1e-6
mixing_window = 4
cutoff = 0.1
fit_cutoff = 0.2
fit_epsilon = 2.5
optical_window_min_ev = 1.0
optical_window_max_ev = 2.0
hartree_to_ev = 27.2114

[execution]
build_cavity_mode_basis = true
build_potential_modes = true
solve_oqs_pcm = false
solve_inertial_basis = false
reuse_inertial_cache = false
write_text_exports = true

[postprocess]
output_dir = "/tmp/figure_tables"
dipoles = "/tmp/ci_mut.inp"
delta_scf = -0.01
oscillator_strength_mode = "complex_symmetric"

[dielectric]
debye_c = [0.47, 72.62]
debye_g = [1.0, 2.0]
drude_w = [0.1, 0.2]
drude_c = [0.3, 0.4]
drude_g = [0.5, 0.6]
optical_drude_start_index = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.problem.nsts == 5
    assert config.problem.diagonalization_states == 3
    assert config.problem.resolution_of_identity_states == 4
    assert config.problem.fit_cutoff == 0.2
    assert config.problem.fit_epsilon == 2.5
    assert config.execution.build_cavity_mode_basis is True
    assert config.execution.build_potential_modes is True
    assert config.execution.reuse_inertial_cache is False
    assert config.paths.output_dir == Path("/tmp/output")
    assert config.paths.inertial_energy is None
    assert config.paths.inertial_dipoles is None
    assert config.dielectric.drude_w == [0.1, 0.2]
    assert config.postprocess.output_dir == Path("/tmp/figure_tables")
    assert config.postprocess.dipoles == Path("/tmp/ci_mut.inp")
    assert config.postprocess.delta_scf == -0.01
    assert config.postprocess.oscillator_strength_mode == "complex_symmetric"


def test_load_config_defaults_to_reusing_inertial_cache(tmp_path: Path) -> None:
    config_path = tmp_path / "test_config_default_cache.toml"
    config_path.write_text(
        """
[paths]
mat_sd = "/tmp/mat_SD.inp"
cavity = "/tmp/cavity.inp"
ci_potential = "/tmp/ci_pot.inp"
normal_mode_potential = "/tmp/ci_pot_normal_mode.inp"
ci_energy = "/tmp/ci_energy.inp"
sm12t = "/tmp/Sm12T.dat"
lambda_file = "/tmp/Lambda.dat"
output_dir = "/tmp/output"

[problem]
nsts = 5
diagonalization_states = 3
resolution_of_identity_states = 4
corrected_states = 2
convergence_tolerance = 1e-6
mixing_window = 4
cutoff = 0.1
optical_window_min_ev = 1.0
optical_window_max_ev = 2.0
hartree_to_ev = 27.2114

[execution]
build_potential_modes = false
solve_oqs_pcm = true
solve_inertial_basis = true
write_text_exports = true

[dielectric]
debye_c = [0.47, 72.62]
debye_g = [1.0, 2.0]
drude_w = [0.1, 0.2]
drude_c = [0.3, 0.4]
drude_g = [0.5, 0.6]
optical_drude_start_index = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.execution.build_cavity_mode_basis is True
    assert config.execution.reuse_inertial_cache is True
    assert config.postprocess.output_dir is None
    assert config.postprocess.dipoles is None
    assert config.postprocess.delta_scf == 0.0
    assert config.postprocess.oscillator_strength_mode is None
    assert config.problem.fit_cutoff == 0.5172311136437814
    assert config.problem.fit_epsilon == 2.0302901591898928


def test_load_config_accepts_complete_inertial_input_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "test_config_direct_inertial.toml"
    config_path.write_text(
        """
[paths]
mat_sd = "/tmp/mat_SD.inp"
cavity = "/tmp/cavity.inp"
ci_potential = "/tmp/ci_pot.inp"
normal_mode_potential = "/tmp/ci_pot_normal_mode.inp"
sm12t = "/tmp/Sm12T.dat"
lambda_file = "/tmp/Lambda.dat"
output_dir = "/tmp/output"
inertial_normal_mode_potential = "/tmp/ci_pot_normal_mode_inertial.inp"
inertial_energy = "/tmp/ci_energy_inertial.inp"
inertial_dipoles = "/tmp/ci_mut_inertial.inp"

[problem]
nsts = 2
diagonalization_states = 2
resolution_of_identity_states = 2
corrected_states = 2
convergence_tolerance = 1e-6
mixing_window = 2
cutoff = 0.1
optical_window_min_ev = 1.0
optical_window_max_ev = 2.0
hartree_to_ev = 27.2114

[execution]
build_potential_modes = false
solve_oqs_pcm = true
solve_inertial_basis = true
reuse_inertial_cache = true
write_text_exports = true
potential_basis = "normal_mode_inertial"

[dielectric]
debye_c = [0.47]
debye_g = [1.0]
drude_w = [0.1]
drude_c = [0.3]
drude_g = [0.5]
optical_drude_start_index = 0
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.paths.ci_energy is None
    assert config.paths.inertial_normal_mode_potential == Path("/tmp/ci_pot_normal_mode_inertial.inp")
    assert config.paths.inertial_energy == Path("/tmp/ci_energy_inertial.inp")
    assert config.paths.inertial_dipoles == Path("/tmp/ci_mut_inertial.inp")


def test_load_config_accepts_legacy_build_normal_modes_key(tmp_path: Path) -> None:
    config_path = tmp_path / "test_config_legacy_normal_modes.toml"
    config_path.write_text(
        """
[paths]
mat_sd = "/tmp/mat_SD.inp"
cavity = "/tmp/cavity.inp"
ci_potential = "/tmp/ci_pot.inp"
normal_mode_potential = "/tmp/ci_pot_normal_mode.inp"
ci_energy = "/tmp/ci_energy.inp"
sm12t = "/tmp/Sm12T.dat"
lambda_file = "/tmp/Lambda.dat"
output_dir = "/tmp/output"

[problem]
nsts = 5
diagonalization_states = 3
resolution_of_identity_states = 4
corrected_states = 2
convergence_tolerance = 1e-6
mixing_window = 4
cutoff = 0.1
optical_window_min_ev = 1.0
optical_window_max_ev = 2.0
hartree_to_ev = 27.2114

[execution]
build_normal_modes = true
solve_oqs_pcm = true
solve_inertial_basis = false
write_text_exports = true

[dielectric]
debye_c = [0.47, 72.62]
debye_g = [1.0, 2.0]
drude_w = [0.1, 0.2]
drude_c = [0.3, 0.4]
drude_g = [0.5, 0.6]
optical_drude_start_index = 1
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.execution.build_potential_modes is True


def test_load_config_rejects_unknown_oscillator_strength_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "test_config_bad_oscillator_mode.toml"
    config_path.write_text(
        """
[paths]
mat_sd = "/tmp/mat_SD.inp"
cavity = "/tmp/cavity.inp"
ci_potential = "/tmp/ci_pot.inp"
normal_mode_potential = "/tmp/ci_pot_normal_mode.inp"
ci_energy = "/tmp/ci_energy.inp"
sm12t = "/tmp/Sm12T.dat"
lambda_file = "/tmp/Lambda.dat"
output_dir = "/tmp/output"

[problem]
nsts = 5
diagonalization_states = 3
resolution_of_identity_states = 4
corrected_states = 2
convergence_tolerance = 1e-6
mixing_window = 4
cutoff = 0.1
optical_window_min_ev = 1.0
optical_window_max_ev = 2.0
hartree_to_ev = 27.2114

[execution]
build_potential_modes = false
solve_oqs_pcm = true
solve_inertial_basis = true
write_text_exports = true

[postprocess]
oscillator_strength_mode = "auto"

[dielectric]
debye_c = [0.47, 72.62]
debye_g = [1.0, 2.0]
drude_w = [0.1, 0.2]
drude_c = [0.3, 0.4]
drude_g = [0.5, 0.6]
optical_drude_start_index = 1
""".strip(),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="oscillator_strength_mode"):
        load_config(config_path)
