from __future__ import annotations

from pathlib import Path

import numpy as np

from oqs_pcm.cavity import build_cavity_mode_basis, write_cavity_mode_basis
from oqs_pcm.config import AppConfig
from oqs_pcm.dielectric import build_dielectric_model, build_kernel_from_lambda, build_state_resolved_kernel_from_lambda
from oqs_pcm.io import (
    io_load_excited_state_energy_vector,
    io_load_matrix,
    io_load_normal_mode_transformation_matrix,
    io_load_pcm_lambda_eigenvalues,
    io_load_state_transition_dipole_tensor,
    io_load_vector,
    io_write_pcm_normal_mode_potential_tensor,
    io_write_state_transition_dipole_tensor,
    io_write_matrix,
    io_write_vector,
)
from oqs_pcm.models.pcm import build_pcm_kernels
from oqs_pcm.memory import (
    enforce_disk_estimate,
    enforce_memory_estimate,
    estimate_pcm_normal_mode_text_disk,
    estimate_pcm_potential_mode_memory,
)
from oqs_pcm.potential import pcm_add_nuclear_potential_to_state_diagonals, pcm_load_frozen_state_transition_potential_tensor, pcm_load_normal_mode_potential_tensor, transform_coupling_tensor_to_normal_modes
from oqs_pcm.solver import (
    SolverConfig,
    build_reference_diagonal_hamiltonian,
    solve_born_oppenheimer_limit,
    solve_cutoff_approximation,
    solve_inertial_basis_reference_problem,
    solve_static_state_specific_family,
    solve_state_specific_family,
    transform_coupling_tensor_to_inertial_basis,
)


def build_normal_mode_file(config: AppConfig) -> np.ndarray:
    """Run the workflow originally implemented in `potential_normal_modes.m`."""

    estimate = estimate_pcm_potential_mode_memory(config)
    enforce_memory_estimate(estimate, minimum_available_gb=config.safety.min_available_memory_gb)
    disk_estimate = estimate_pcm_normal_mode_text_disk(config)
    enforce_disk_estimate(disk_estimate)
    nts, ve, nuclear = pcm_load_frozen_state_transition_potential_tensor(str(config.paths.ci_potential), config.problem.nsts - 1)
    ve = pcm_add_nuclear_potential_to_state_diagonals(ve, nuclear)
    sm12t = io_load_normal_mode_transformation_matrix(config.paths.sm12t)
    transformed = transform_coupling_tensor_to_normal_modes(ve, sm12t)
    io_write_pcm_normal_mode_potential_tensor(config.paths.normal_mode_potential, transformed)
    return transformed


def build_cavity_mode_basis_files(config: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate `Lambda.dat` and `Sm12T.dat` from PCM cavity matrices."""

    lambda_values, sm12t = build_cavity_mode_basis(config.paths.mat_sd, config.paths.cavity)
    write_cavity_mode_basis(config.paths.lambda_file, config.paths.sm12t, lambda_values, sm12t)
    return lambda_values, sm12t


def ensure_normal_mode_potential(config: AppConfig) -> tuple[int, np.ndarray]:
    """Load an existing transformed potential or generate it first."""

    if config.execution.build_cavity_mode_basis:
        build_cavity_mode_basis_files(config)

    potential_path = Path(config.paths.normal_mode_potential)
    if config.execution.build_cavity_mode_basis or config.execution.build_potential_modes or not potential_path.exists():
        transformed = build_normal_mode_file(config)
        return transformed.shape[2], transformed[: config.problem.nsts, : config.problem.nsts, :]
    return pcm_load_normal_mode_potential_tensor(str(potential_path), config.problem.nsts)


def load_inertial_normal_mode_potential(config: AppConfig) -> tuple[int, np.ndarray]:
    """Load a complete normal-mode potential already in the inertial basis."""

    potential_path = config.paths.inertial_normal_mode_potential
    if potential_path is None:
        raise ValueError(
            "`paths.inertial_normal_mode_potential` is required when "
            "`execution.potential_basis = normal_mode_inertial`."
        )
    return pcm_load_normal_mode_potential_tensor(str(potential_path), config.problem.nsts)


def load_inertial_energies(config: AppConfig) -> np.ndarray:
    """Load inertial reference energies without loading the basis vectors."""

    energy_path, _ = get_inertial_cache_paths(config)
    if not energy_path.exists():
        raise FileNotFoundError(
            "The pre-transformed potential requires inertial energies at "
            f"{energy_path}."
        )
    return io_load_vector(energy_path)


def get_inertial_cache_paths(config: AppConfig) -> tuple[Path, Path]:
    """Return the standard cache locations for inertial-basis intermediates."""

    cache_root = config.paths.inertial_cache_dir or config.paths.output_dir
    inertial_dir = cache_root / f"inertial_diag{config.problem.nsts}"
    return inertial_dir / "newenergies.dat", inertial_dir / "newstates.dat"


def get_inertial_artifact_paths(config: AppConfig) -> dict[str, Path]:
    """Return paths for optional full inertial-basis artifacts."""

    artifact_dir = config.paths.output_dir / "inertial_artifacts"
    return {
        "frozen_potential": artifact_dir / "ci_pot_inertial.inp",
        "normal_mode_potential": artifact_dir / "ci_pot_normal_mode_inertial.inp",
        "dipoles": artifact_dir / "ci_mut_inertial.inp",
    }


def write_inertial_basis_artifacts(
    config: AppConfig,
    normal_mode_potential: np.ndarray,
    inertial_states: np.ndarray,
) -> dict[str, Path]:
    """Write user-requested potentials and dipoles in the inertial basis.

    The normal-mode potential is the tensor generated by OQSolv in the
    current workflow; this routine never reads ``ci_pot_normal_mode.inp``.
    The frozen potential is loaded from the original GAMESS/PCM input and
    receives the same nuclear diagonal contribution used to generate the
    normal-mode tensor.  Outputs are opt-in because each full potential file
    can be large.
    """

    execution = config.execution
    if not execution.write_inertial_potentials and not execution.write_inertial_dipoles:
        return {}
    if not config.execution.solve_inertial_basis:
        raise ValueError(
            "Writing inertial-basis artifacts requires "
            "`execution.solve_inertial_basis = true`."
        )

    paths = get_inertial_artifact_paths(config)
    written: dict[str, Path] = {}

    if execution.write_inertial_potentials:
        _, frozen_potential, nuclear = pcm_load_frozen_state_transition_potential_tensor(
            str(config.paths.ci_potential), config.problem.nsts - 1
        )
        frozen_potential = pcm_add_nuclear_potential_to_state_diagonals(frozen_potential, nuclear)
        frozen_inertial = transform_coupling_tensor_to_inertial_basis(inertial_states, frozen_potential)
        io_write_pcm_normal_mode_potential_tensor(paths["frozen_potential"], frozen_inertial)
        del frozen_potential, frozen_inertial, nuclear
        normal_inertial = transform_coupling_tensor_to_inertial_basis(inertial_states, normal_mode_potential)
        io_write_pcm_normal_mode_potential_tensor(paths["normal_mode_potential"], normal_inertial)
        del normal_inertial
        written.update(
            {
                "frozen_potential": paths["frozen_potential"],
                "normal_mode_potential": paths["normal_mode_potential"],
            }
        )

    if execution.write_inertial_dipoles:
        if config.postprocess.dipoles is None:
            raise ValueError(
                "Writing inertial-basis dipoles requires `postprocess.dipoles` "
                "to point to the complete state/transition dipole file."
            )
        dipoles = io_load_state_transition_dipole_tensor(
            config.postprocess.dipoles, config.problem.nsts - 1
        )
        inertial_dipoles = transform_coupling_tensor_to_inertial_basis(inertial_states, dipoles)
        io_write_state_transition_dipole_tensor(paths["dipoles"], inertial_dipoles)
        written["dipoles"] = paths["dipoles"]

    return written


def should_recompute_inertial_basis(config: AppConfig) -> bool:
    """Return whether the inertial reference problem must be recomputed.

    A fresh cavity-basis build changes `Sm12T.dat` and therefore invalidates
    both the potential-mode tensor and the inertial cache. A fresh potential
    transformation also invalidates the inertial cache.
    """

    return (
        config.execution.build_cavity_mode_basis
        or config.execution.build_potential_modes
        or not config.execution.reuse_inertial_cache
    )


def ensure_inertial_basis_reference(
    config: AppConfig,
    energies: np.ndarray,
    potential: np.ndarray,
    optical_kernel: np.ndarray,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached inertial intermediates or compute and save them.

    The original Mathematica workflow exports the inertial energies and states
    as explicit intermediate artifacts. Keeping the same pattern in Python
    makes the subsequent full-frequency runs reproducible and lets users rerun
    only the downstream OQS variants without recomputing the inertial step.
    """

    energy_path, state_path = get_inertial_cache_paths(config)
    if not force_recompute and energy_path.exists() and state_path.exists():
        return io_load_vector(energy_path), io_load_matrix(state_path)

    inertial_energies, inertial_states = solve_inertial_basis_reference_problem(
        energies, potential, optical_kernel
    )
    io_write_vector(energy_path, inertial_energies)
    io_write_matrix(state_path, inertial_states)
    return inertial_energies, inertial_states


def solve_inertial_basis_only(config: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build and solve only the full inertial reference problem.

    This is the public workflow entry point for ``oqs-pcm run --mode
    inertial``.  It deliberately does not run SS-pol, BO-sol, or any other
    downstream OQS variant.  The complete source state space is retained in
    the exported cache so that later selected-state calculations can choose
    nonconsecutive states without recomputing the inertial basis.
    """

    if config.execution.potential_basis != "normal_mode":
        raise ValueError(
            "The standalone inertial reference solve requires "
            "`execution.potential_basis = normal_mode`."
        )
    _, potential = ensure_normal_mode_potential(config)
    energies = io_load_excited_state_energy_vector(
        config.paths.ci_energy,
        config.problem.hartree_to_ev,
    )[: config.problem.nsts]
    dielectric = build_dielectric_model(config.dielectric)
    kernels = build_pcm_kernels(
        lambda_path=config.paths.lambda_file,
        dielectric_model=dielectric,
        energies=energies,
        optical_window_ev=(config.problem.optical_window_min_ev, config.problem.optical_window_max_ev),
    )
    inertial_energies, inertial_states = ensure_inertial_basis_reference(
        config,
        energies,
        potential,
        kernels.optical_kernel,
        force_recompute=should_recompute_inertial_basis(config),
    )
    write_inertial_basis_artifacts(config, potential, inertial_states)
    return inertial_energies, inertial_states


def solve_oqs_pcm(
    config: AppConfig,
    requested_variants: set[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run the PCM-based OQS workflow and return all named result sets.

    This module currently covers the branch derived from the traditional
    polarizable continuum model, where solvent coupling is represented through
    electrostatic potentials on a molecular cavity.

    The Onsager-based branch uses the same iterative diagonalization logic but
    builds its non-diagonal corrections from state/transition dipole moments
    rather than state/transition electrostatic potentials. In other words, the
    distinction belongs in the coupling builder, not in the eigensolver engine.
    """

    potential_is_already_inertial = config.execution.potential_basis == "normal_mode_inertial"
    if potential_is_already_inertial:
        _, potential = load_inertial_normal_mode_potential(config)
        if not config.execution.solve_inertial_basis:
            raise ValueError(
                "A pre-transformed normal-mode potential requires "
                "`execution.solve_inertial_basis = true`."
            )
        if config.execution.write_inertial_potentials or config.execution.write_inertial_dipoles:
            raise ValueError(
                "Pre-transformed normal-mode input cannot regenerate inertial artifacts; "
                "use the original normal-mode input for that stage."
            )
    elif config.execution.potential_basis == "frozen":
        _, potential, nuclear = pcm_load_frozen_state_transition_potential_tensor(
            str(config.paths.ci_potential), config.problem.nsts - 1
        )
        potential = pcm_add_nuclear_potential_to_state_diagonals(potential, nuclear)
    else:
        _, potential = ensure_normal_mode_potential(config)
    dielectric = build_dielectric_model(config.dielectric)
    energies = io_load_excited_state_energy_vector(
        config.paths.ci_energy,
        config.problem.hartree_to_ev,
    )[: config.problem.nsts]
    selected_indices = np.asarray(config.problem.selected_state_indices(), dtype=int)
    all_states_selected = config.problem.selected_states is None
    effective_nsts = len(selected_indices)
    ri_states = config.problem.resolution_of_identity_states or config.problem.nsts

    kernels = build_pcm_kernels(
        lambda_path=config.paths.lambda_file,
        dielectric_model=dielectric,
        energies=energies[selected_indices][:ri_states],
        optical_window_ev=(config.problem.optical_window_min_ev, config.problem.optical_window_max_ev),
    )
    lambda_values = io_load_pcm_lambda_eigenvalues(config.paths.lambda_file)

    def build_dynamic_pcm_kernel(_: int, trial_energy: float) -> np.ndarray:
        omega = trial_energy - ri_energies
        # Dynamic OQS branches use only the non-inertial dielectric response.
        eps_values = dielectric.epsilon(omega, include_ir=False)
        return build_state_resolved_kernel_from_lambda(lambda_values, eps_values)

    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_cfg = SolverConfig(
        nsts=effective_nsts,
        diagonalization_states=config.problem.diagonalization_states,
        corrected_states=min(config.problem.corrected_states, config.problem.diagonalization_states),
        tolerance=config.problem.convergence_tolerance,
        mixing_window=config.problem.mixing_window,
        cutoff=config.problem.cutoff,
        write_text_exports=config.execution.write_text_exports,
        output_dir=output_dir,
    )

    inertial_energies = energies
    inertial_states = np.eye(config.problem.nsts)
    if config.execution.solve_inertial_basis:
        # Execution semantics:
        # - `build_cavity_mode_basis` controls whether `Sm12T.dat` and
        #   `Lambda.dat` are rebuilt from `mat_SD.inp` and `cavity.inp`.
        # - `build_potential_modes` controls whether the molecular
        #   state/transition potentials are transformed with `Sm12T.dat`.
        # - `solve_inertial_basis` controls whether downstream OQS variants are
        #   run in the inertial basis or remain in the original frozen/gas-like
        #   reference basis.
        # - inertial-cache reuse is independent from potential-mode reuse,
        #   except that a fresh potential-mode build invalidates any previous
        #   inertial cache because the basis-defining tensor has changed.
        if potential_is_already_inertial:
            if should_recompute_inertial_basis(config):
                raise ValueError(
                    "A pre-transformed normal-mode potential requires reuse of "
                    "an existing inertial cache."
                )
            inertial_energies = load_inertial_energies(config)
        else:
            inertial_energies, inertial_states = ensure_inertial_basis_reference(
                config,
                energies,
                potential,
                kernels.optical_kernel,
                force_recompute=should_recompute_inertial_basis(config),
            )
            write_inertial_basis_artifacts(config, potential, inertial_states)
            potential = transform_coupling_tensor_to_inertial_basis(inertial_states, potential)

    if not all_states_selected:
        potential = potential[np.ix_(selected_indices, selected_indices, np.arange(potential.shape[2]))]
        inertial_energies = inertial_energies[selected_indices]

    base_h = build_reference_diagonal_hamiltonian(inertial_energies, config.problem.diagonalization_states)
    # `diagonalization_states` is the explicit Hamiltonian dimension, while
    # `ri_states` is the size of the resolution-of-identity space. The latter
    # must not be smaller than the former; that consistency check is enforced at
    # config-load time.
    reduced_potential = potential[: config.problem.diagonalization_states, :ri_states, :]
    ri_energies = inertial_energies[:ri_states]

    variant_set = requested_variants or {
        "ss_pol",
        "full_nodisp_nodiss",
        "full_nodiss",
        "full",
        "first_full",
        "first_full_nodisp_nodiss",
        "first_full_nodiss",
        "bo",
        "cutoff",
        "fit_cutoff",
    }

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if "ss_pol" in variant_set:
        results["ss_pol"] = solve_static_state_specific_family(
            label="ss_pol",
            base_h=base_h,
            potential=reduced_potential,
            optical_kernel=kernels.optical_kernel,
            cfg=solver_cfg,
        )
    if "full_nodisp_nodiss" in variant_set:
        results["full_nodisp_nodiss"] = solve_state_specific_family(
            label="full_nodisp_nodiss",
            base_h=base_h,
            potential=reduced_potential,
            energies=ri_energies,
            kernel_real=lambda state_index, trial_energy: np.real(build_dynamic_pcm_kernel(state_index, trial_energy)),
            kernel_imag=None,
            cfg=solver_cfg,
            diagonal_only=True,
        )
    if "full_nodiss" in variant_set:
        results["full_nodiss"] = solve_state_specific_family(
            label="full_nodiss",
            base_h=base_h,
            potential=reduced_potential,
            energies=ri_energies,
            kernel_real=lambda state_index, trial_energy: np.real(build_dynamic_pcm_kernel(state_index, trial_energy)),
            kernel_imag=None,
            cfg=solver_cfg,
        )
    if "full" in variant_set:
        results["full"] = solve_state_specific_family(
            label="full",
            base_h=base_h,
            potential=reduced_potential,
            energies=ri_energies,
            kernel_real=lambda state_index, trial_energy: np.real(build_dynamic_pcm_kernel(state_index, trial_energy)),
            kernel_imag=lambda state_index, trial_energy: np.imag(build_dynamic_pcm_kernel(state_index, trial_energy)),
            cfg=solver_cfg,
        )
    if "first_full" in variant_set:
        results["first_full"] = solve_state_specific_family(
            label="first_full",
            base_h=base_h,
            potential=reduced_potential,
            energies=ri_energies,
            kernel_real=lambda state_index, trial_energy: np.real(build_dynamic_pcm_kernel(state_index, trial_energy)),
            kernel_imag=lambda state_index, trial_energy: np.imag(build_dynamic_pcm_kernel(state_index, trial_energy)),
            cfg=solver_cfg,
            first_iteration_only=True,
        )
    if "first_full_nodisp_nodiss" in variant_set:
        results["first_full_nodisp_nodiss"] = solve_state_specific_family(
            label="first_full_nodisp_nodiss",
            base_h=base_h,
            potential=reduced_potential,
            energies=ri_energies,
            kernel_real=lambda state_index, trial_energy: np.real(build_dynamic_pcm_kernel(state_index, trial_energy)),
            kernel_imag=None,
            cfg=solver_cfg,
            diagonal_only=True,
            first_iteration_only=True,
        )
    if "first_full_nodiss" in variant_set:
        results["first_full_nodiss"] = solve_state_specific_family(
            label="first_full_nodiss",
            base_h=base_h,
            potential=reduced_potential,
            energies=ri_energies,
            kernel_real=lambda state_index, trial_energy: np.real(build_dynamic_pcm_kernel(state_index, trial_energy)),
            kernel_imag=None,
            cfg=solver_cfg,
            first_iteration_only=True,
        )
    if "bo" in variant_set:
        results["bo"] = solve_born_oppenheimer_limit(base_h, reduced_potential, kernels.optical_kernel, solver_cfg)
    if "cutoff" in variant_set:
        results["cutoff"] = solve_cutoff_approximation(base_h, reduced_potential, ri_energies, kernels.optical_kernel, solver_cfg)
    if "fit_cutoff" in variant_set:
        fit_kernel = build_kernel_from_lambda(lambda_values, config.problem.fit_epsilon)
        results["fit_cutoff"] = solve_cutoff_approximation(
            base_h,
            reduced_potential,
            ri_energies,
            fit_kernel,
            solver_cfg,
            label="fit_cutoff",
            cutoff=config.problem.fit_cutoff,
        )
    return results
