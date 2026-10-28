from __future__ import annotations

from pathlib import Path

from oqs_pcm.config import (
    AppConfig,
    DielectricConfig,
    ExecutionConfig,
    PathsConfig,
    PostprocessConfig,
    ProblemConfig,
    SafetyConfig,
)
from oqs_pcm.workflow import get_inertial_cache_paths, should_recompute_inertial_basis, uses_complete_inertial_inputs


def _config(
    *,
    build_cavity_mode_basis: bool = False,
    build_potential_modes: bool,
    solve_inertial_basis: bool,
    reuse_inertial_cache: bool,
) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            mat_sd=Path("/tmp/mat_SD.inp"),
            cavity=Path("/tmp/cavity.inp"),
            ci_potential=Path("/tmp/ci_pot.inp"),
            normal_mode_potential=Path("/tmp/ci_pot_normal_mode.inp"),
            ci_energy=Path("/tmp/ci_energy.inp"),
            sm12t=Path("/tmp/Sm12T.dat"),
            lambda_file=Path("/tmp/Lambda.dat"),
            output_dir=Path("/tmp/output"),
        ),
        problem=ProblemConfig(
            nsts=172,
            diagonalization_states=50,
            resolution_of_identity_states=172,
            corrected_states=5,
            convergence_tolerance=1e-6,
            mixing_window=10,
            cutoff=0.299561,
            fit_cutoff=0.5172311136437814,
            fit_epsilon=2.0302901591898928,
            optical_window_min_ev=1.63,
            optical_window_max_ev=3.26,
            hartree_to_ev=27.2114,
        ),
        execution=ExecutionConfig(
            build_cavity_mode_basis=build_cavity_mode_basis,
            build_potential_modes=build_potential_modes,
            solve_oqs_pcm=True,
            solve_inertial_basis=solve_inertial_basis,
            reuse_inertial_cache=reuse_inertial_cache,
            write_text_exports=True,
        ),
        dielectric=DielectricConfig(
            debye_c=[0.47, 72.62],
            debye_g=[1.0, 2.0],
            drude_w=[0.1, 0.2],
            drude_c=[0.3, 0.4],
            drude_g=[0.5, 0.6],
            optical_drude_start_index=1,
        ),
        postprocess=PostprocessConfig(
            output_dir=None,
            dipoles=None,
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


def test_inertial_cache_paths_follow_mathematica_style_output_layout() -> None:
    config = _config(
        build_potential_modes=False,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )

    energy_path, state_path = get_inertial_cache_paths(config)

    assert energy_path == Path("/tmp/output/inertial_diag172/newenergies.dat")
    assert state_path == Path("/tmp/output/inertial_diag172/newstates.dat")


def test_recomputes_inertial_basis_after_rebuilding_normal_modes() -> None:
    config = _config(
        build_potential_modes=True,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )

    assert should_recompute_inertial_basis(config) is True


def test_recomputes_inertial_basis_after_rebuilding_cavity_mode_basis() -> None:
    config = _config(
        build_cavity_mode_basis=True,
        build_potential_modes=False,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )

    assert should_recompute_inertial_basis(config) is True


def test_recomputes_inertial_basis_when_cache_reuse_is_disabled() -> None:
    config = _config(
        build_potential_modes=False,
        solve_inertial_basis=True,
        reuse_inertial_cache=False,
    )

    assert should_recompute_inertial_basis(config) is True


def test_can_reuse_inertial_basis_when_normal_modes_are_reused() -> None:
    config = _config(
        build_potential_modes=False,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )

    assert should_recompute_inertial_basis(config) is False


def test_basis_choice_flag_remains_independent_from_cache_policy() -> None:
    config = _config(
        build_potential_modes=False,
        solve_inertial_basis=False,
        reuse_inertial_cache=True,
    )

    # The downstream solver family may still be run in the original basis.
    assert config.execution.solve_inertial_basis is False
    # Cache policy remains a separate execution control.
    assert should_recompute_inertial_basis(config) is False


def test_complete_inertial_input_route_is_detected() -> None:
    config = _config(
        build_potential_modes=False,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )
    config.execution.potential_basis = "normal_mode_inertial"
    config.paths.inertial_normal_mode_potential = Path("/tmp/ci_pot_normal_mode_inertial.inp")
    config.paths.inertial_energy = Path("/tmp/ci_energy_inertial.inp")

    assert uses_complete_inertial_inputs(config) is True
