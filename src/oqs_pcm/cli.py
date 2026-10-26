from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from oqs_pcm.config import load_config, validate_runtime_dimensions
from oqs_pcm.dipole_completion import (
    compare_ground_transition_dipoles_to_reference,
    complete_ci_mut_file,
    load_dipole_completion_inputs_from_npy,
)
from oqs_pcm.gamess import (
    gamess_extract_ci_energies,
    gamess_extract_potential_block,
    gamess_extract_reference_ground_excited_dipoles,
    gamess_write_ci_energy,
    gamess_write_reference_ground_excited_dipoles,
    gamess_write_text,
)
from oqs_pcm.postprocess.tables import write_notebook_equivalent_tables
from oqs_pcm.memory import (
    apply_process_memory_limit,
    estimate_pcm_normal_mode_text_disk,
    estimate_pcm_potential_mode_memory,
    format_disk_estimate,
    format_memory_estimate,
)
from oqs_pcm.workflow import (
    build_cavity_mode_basis_files,
    build_normal_mode_file,
    get_inertial_cache_paths,
    get_inertial_artifact_paths,
    solve_inertial_basis_only,
    solve_oqs_pcm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oqs-pcm",
        description="Integrated Python workflow for solver, postprocessing, and figures formerly spread across Mathematica scripts and plotting utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the OQSolv preprocessing and/or solver workflow from a TOML config file.")
    run_parser.add_argument("config", type=Path, help="Path to the TOML configuration file.")
    run_parser.add_argument(
        "--mode",
        choices=("cavity-basis", "potential-modes", "normal-modes", "inertial", "solve", "all"),
        default="all",
        help="Which part of the workflow to run.",
    )
    run_parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
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
        ),
        help="Optional subset of solver variants to run. By default all supported PCM variants are computed.",
    )
    run_parser.add_argument("--output-dir", type=Path, help="Override [paths].output_dir for this run.")
    run_parser.add_argument("--normal-mode-potential", type=Path, help="Override [paths].normal_mode_potential.")
    run_parser.add_argument(
        "--inertial-normal-mode-potential",
        type=Path,
        help="Override [paths].inertial_normal_mode_potential.",
    )
    run_parser.add_argument("--sm12t", type=Path, help="Override [paths].sm12t.")
    run_parser.add_argument("--lambda-file", type=Path, help="Override [paths].lambda_file.")
    run_parser.add_argument("--dipoles", type=Path, help="Override [postprocess].dipoles.")
    run_parser.add_argument("--selected-states", nargs="+", type=int, help="One-based state labels, starting with state 1.")
    run_parser.add_argument("--diagonalization-states", type=int)
    run_parser.add_argument("--resolution-of-identity-states", type=int)
    run_parser.add_argument("--corrected-states", type=int)
    run_parser.add_argument(
        "--potential-basis",
        choices=("normal_mode", "normal_mode_inertial", "frozen"),
    )
    for option in (
        "build_cavity_mode_basis",
        "build_potential_modes",
        "solve_inertial_basis",
        "reuse_inertial_cache",
        "write_inertial_potentials",
        "write_inertial_dipoles",
    ):
        run_parser.add_argument(
            f"--{option.replace('_', '-')}",
            f"--no-{option.replace('_', '-')}",
            dest=option,
            action=argparse.BooleanOptionalAction,
            default=None,
        )

    estimate_parser = subparsers.add_parser(
        "estimate-memory",
        help="Estimate memory and disk requirements for heavy OQSolv preprocessing stages without running them.",
    )
    estimate_parser.add_argument("config", type=Path, help="Path to the TOML configuration file.")

    figures_parser = subparsers.add_parser("figures", help="Render figure files from postprocessed text tables.")
    figures_parser.add_argument("kind", choices=("levels", "spectra", "cutoff", "convergence", "all"))
    figures_parser.add_argument("data_dir", type=Path, help="Directory containing the text tables consumed by the figure routines.")
    figures_parser.add_argument("output_dir", type=Path, help="Directory where the generated figure files should be written.")
    figures_parser.add_argument("--fontsize", type=int, default=12, help="Base font size for the generated figures.")

    postprocess_parser = subparsers.add_parser(
        "postprocess",
        help="Generate notebook-equivalent reduced text tables from OQSolv solver outputs.",
    )
    postprocess_parser.add_argument("config", type=Path, help="Path to the TOML configuration file used for the OQSolv run.")
    postprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override [postprocess].output_dir from the TOML file.",
    )
    postprocess_parser.add_argument(
        "--dipoles",
        type=Path,
        help="Override [postprocess].dipoles from the TOML file.",
    )
    postprocess_parser.add_argument(
        "--dipole-basis",
        choices=("frozen", "inertial"),
        help="Override [postprocess].dipole_basis from the TOML file.",
    )
    postprocess_parser.add_argument(
        "--delta-scf",
        type=float,
        help="Override [postprocess].delta_scf from the TOML file.",
    )
    postprocess_parser.add_argument(
        "--skip-levels",
        action="store_true",
        help="Do not write absolute level-energy tables such as en.txt and enbo.txt.",
    )
    postprocess_parser.add_argument(
        "--skip-spectra",
        action="store_true",
        help="Do not write spectra tables such as full.txt, bo.txt, cut.txt, and fit.txt.",
    )
    postprocess_parser.add_argument(
        "--convergence-root",
        nargs=2,
        action="append",
        metavar=("NSTS", "OUTPUT_ROOT"),
        help="Optional convergence source: total-state count and OQSolv output root containing variant directories.",
    )
    postprocess_parser.add_argument(
        "--solver-output-dir",
        type=Path,
        help="Override [paths].output_dir containing solver result directories.",
    )
    postprocess_parser.add_argument(
        "--selected-states",
        nargs="+",
        type=int,
        help="One-based state labels used only to reduce postprocessed dipoles.",
    )

    prep_parser = subparsers.add_parser("prep-gamess", help="Extract OQSolv ingredient files from a GAMESS-US log.")
    prep_parser.add_argument("log_file", type=Path, help="GAMESS-US log file.")
    prep_parser.add_argument("--ci-energy", type=Path, help="Output path for ci_energy.inp.")
    prep_parser.add_argument("--ci-pot", type=Path, help="Output path for ci_pot.inp.")
    prep_parser.add_argument(
        "--reference-ci-mut",
        type=Path,
        help="Output path for the diagnostic GS-ES dipole reference extracted from the log.",
    )

    dipoles_parser = subparsers.add_parser(
        "xdip",
        aliases=["complete-dipoles"],
        help="Run Xdip: reconstruct a completed ci_mut.inp from LR-TDDFT dipole ingredients.",
    )
    dipoles_parser.add_argument("--ao-dipoles", type=Path, required=True, help="NumPy .npy file with AO dipole integrals, shape (nao, nao, 3).")
    dipoles_parser.add_argument("--mo-coefficients", type=Path, required=True, help="NumPy .npy file with MO coefficients, shape (nmo, nao).")
    dipoles_parser.add_argument("--x-amplitudes", type=Path, required=True, help="NumPy .npy file with TDDFT X amplitudes, shape (nstates, n_ov).")
    dipoles_parser.add_argument("--y-amplitudes", type=Path, required=True, help="NumPy .npy file with TDDFT Y amplitudes, shape (nstates, n_ov).")
    dipoles_parser.add_argument("--n-occupied", type=int, required=True, help="Number of occupied MOs in the closed-shell reference.")
    dipoles_parser.add_argument("--n-frozen-occupied", type=int, default=0, help="Number of frozen occupied MOs excluded from the LR excitation space.")
    dipoles_parser.add_argument(
        "--nuclear-dipole",
        type=float,
        nargs=3,
        required=True,
        metavar=("MUX", "MUY", "MUZ"),
        help="Nuclear dipole contribution in atomic units.",
    )
    dipoles_parser.add_argument("--output", type=Path, required=True, help="Output completed ci_mut.inp path.")
    dipoles_parser.add_argument(
        "--reference-ci-mut",
        type=Path,
        help="Optional limited ci_mut.inp containing native GAMESS GS-ES dipoles for diagnostics.",
    )

    return parser


def apply_run_overrides(config, args):
    """Apply optional command-line overrides to one loaded TOML config."""

    path_updates = {}
    for argument, field in (
        ("output_dir", "output_dir"),
        ("normal_mode_potential", "normal_mode_potential"),
        ("inertial_normal_mode_potential", "inertial_normal_mode_potential"),
        ("sm12t", "sm12t"),
        ("lambda_file", "lambda_file"),
    ):
        value = getattr(args, argument)
        if value is not None:
            path_updates[field] = value
    if path_updates:
        config = replace(config, paths=replace(config.paths, **path_updates))

    postprocess_updates = {}
    if args.dipoles is not None:
        postprocess_updates["dipoles"] = args.dipoles
    if postprocess_updates:
        config = replace(config, postprocess=replace(config.postprocess, **postprocess_updates))

    problem_updates = {}
    for argument, field in (
        ("selected_states", "selected_states"),
        ("diagonalization_states", "diagonalization_states"),
        ("resolution_of_identity_states", "resolution_of_identity_states"),
        ("corrected_states", "corrected_states"),
    ):
        value = getattr(args, argument)
        if value is not None:
            problem_updates[field] = value
    if problem_updates:
        config = replace(config, problem=replace(config.problem, **problem_updates))

    execution_updates = {}
    if args.potential_basis is not None:
        execution_updates["potential_basis"] = args.potential_basis
    for field in (
        "build_cavity_mode_basis",
        "build_potential_modes",
        "solve_inertial_basis",
        "reuse_inertial_cache",
        "write_inertial_potentials",
        "write_inertial_dipoles",
    ):
        value = getattr(args, field)
        if value is not None:
            execution_updates[field] = value
    if execution_updates:
        config = replace(config, execution=replace(config.execution, **execution_updates))

    validate_runtime_dimensions(config.problem)
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command in {None, "run"}:
        config = load_config(args.config)
        config = apply_run_overrides(config, args)
        apply_process_memory_limit(config.safety.max_memory_gb)

        if args.mode in {"cavity-basis", "all"}:
            lambda_values, sm12t = build_cavity_mode_basis_files(config)
            print(f"Cavity eigenvalues written to {config.paths.lambda_file}")
            print(f"Cavity mode transformation written to {config.paths.sm12t}")
            print(f"Lambda shape: {lambda_values.shape}; Sm12T shape: {sm12t.shape}")

        if args.mode in {"potential-modes", "normal-modes", "all"}:
            transformed = build_normal_mode_file(config)
            print(f"Normal-mode potential written to {config.paths.normal_mode_potential}")
            print(f"Tensor shape: {transformed.shape}")

        if args.mode == "inertial":
            inertial_energies, inertial_states = solve_inertial_basis_only(config)
            energy_path, state_path = get_inertial_cache_paths(config)
            print(f"Inertial energies written to {energy_path}")
            print(f"Inertial states written to {state_path}")
            print(f"Energy shape: {inertial_energies.shape}; state shape: {inertial_states.shape}")
            if config.execution.write_inertial_potentials or config.execution.write_inertial_dipoles:
                for name, path in get_inertial_artifact_paths(config).items():
                    if path.exists() and (
                        (name != "dipoles" and config.execution.write_inertial_potentials)
                        or (name == "dipoles" and config.execution.write_inertial_dipoles)
                    ):
                        print(f"Inertial {name} written to {path}")
            return

        if args.mode in {"solve", "all"}:
            results = solve_oqs_pcm(config, requested_variants=set(args.variants) if args.variants else None)
            print(f"Wrote {len(results)} result sets to {config.paths.output_dir}")
            print(f"Available result labels: {', '.join(results)}")
        return

    if args.command == "estimate-memory":
        config = load_config(args.config)
        memory_estimate = estimate_pcm_potential_mode_memory(config)
        disk_estimate = estimate_pcm_normal_mode_text_disk(config)
        print(format_memory_estimate(memory_estimate))
        print()
        print(format_disk_estimate(disk_estimate))
        return

    if args.command == "figures":
        # Import matplotlib-dependent modules only when figure rendering is
        # actually requested. This keeps CLI parsing and non-figure tests usable
        # in environments where matplotlib is not yet installed.
        from oqs_pcm.figures.convergence import render_convergence_figures
        from oqs_pcm.figures.cutoff import render_cutoff_figure
        from oqs_pcm.figures.levels import render_levels_figure
        from oqs_pcm.figures.spectra import render_spectra_figures

        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.kind in {"levels", "all"}:
            render_levels_figure(args.data_dir, args.output_dir / "levels.pdf", fontsize=args.fontsize)
        if args.kind in {"spectra", "all"}:
            render_spectra_figures(args.data_dir, args.output_dir, fontsize=args.fontsize)
        if args.kind in {"cutoff", "all"}:
            render_cutoff_figure(args.data_dir, args.output_dir / "figure_cutoff.pdf", fontsize=args.fontsize)
        if args.kind in {"convergence", "all"}:
            render_convergence_figures(args.data_dir, args.output_dir, fontsize=args.fontsize)
        print(f"Figures written to {args.output_dir}")
        return

    if args.command == "postprocess":
        config = load_config(args.config)
        path_updates = {}
        if args.solver_output_dir is not None:
            path_updates["output_dir"] = args.solver_output_dir
        if path_updates:
            config = replace(config, paths=replace(config.paths, **path_updates))
        output_dir = args.output_dir or config.postprocess.output_dir
        dipoles = args.dipoles or config.postprocess.dipoles
        delta_scf = args.delta_scf if args.delta_scf is not None else config.postprocess.delta_scf
        if args.dipole_basis is not None:
            config = replace(config, postprocess=replace(config.postprocess, dipole_basis=args.dipole_basis))
        if output_dir is None:
            raise SystemExit("Missing postprocess output directory. Set [postprocess].output_dir or pass --output-dir.")
        selected_state_indices = None
        if args.selected_states is not None:
            if not args.selected_states or args.selected_states[0] != 1:
                raise SystemExit("--selected-states must start with one-based state 1.")
            if len(set(args.selected_states)) != len(args.selected_states) or any(
                state < 1 or state > config.problem.nsts for state in args.selected_states
            ):
                raise SystemExit("--selected-states contains duplicate or out-of-range labels.")
            selected_state_indices = tuple(state - 1 for state in args.selected_states)
        convergence_roots = None
        if args.convergence_root:
            convergence_roots = {int(nstates): Path(root) for nstates, root in args.convergence_root}
        summary = write_notebook_equivalent_tables(
            config,
            output_dir=output_dir,
            dipole_file=dipoles,
            delta_scf=delta_scf,
            include_levels=not args.skip_levels,
            include_spectra=not args.skip_spectra,
            include_convergence=bool(convergence_roots),
            convergence_roots_by_nstates=convergence_roots,
            selected_state_indices=selected_state_indices,
        )
        print(f"Postprocessed tables written to {output_dir}")
        print(f"Reference polarization shift ww: {summary.reference_shift:.16e} Hartree")
        print(f"Level tables: {len(summary.level_files)}")
        print(f"Spectra tables: {len(summary.spectra_files)}")
        print(f"Convergence tables: {len(summary.convergence_files)}")
        return

    if args.command == "prep-gamess":
        if args.ci_energy is not None:
            roots = gamess_extract_ci_energies(args.log_file)
            gamess_write_ci_energy(args.ci_energy, roots)
            print(f"Wrote {len(roots)} excited-state energies to {args.ci_energy}")
        if args.ci_pot is not None:
            text = gamess_extract_potential_block(args.log_file)
            gamess_write_text(args.ci_pot, text)
            print(f"Wrote PCM potential block to {args.ci_pot}")
        if args.reference_ci_mut is not None:
            rows = gamess_extract_reference_ground_excited_dipoles(args.log_file)
            gamess_write_reference_ground_excited_dipoles(args.reference_ci_mut, rows)
            print(f"Wrote {len(rows)} diagnostic GS-ES dipoles to {args.reference_ci_mut}")
        return

    if args.command in {"xdip", "complete-dipoles"}:
        inputs = load_dipole_completion_inputs_from_npy(
            ao_dipoles_path=args.ao_dipoles,
            mo_coefficients_path=args.mo_coefficients,
            x_amplitudes_path=args.x_amplitudes,
            y_amplitudes_path=args.y_amplitudes,
            n_occupied=args.n_occupied,
            n_frozen_occupied=args.n_frozen_occupied,
            nuclear_dipole=args.nuclear_dipole,
        )
        tensor = complete_ci_mut_file(output_path=args.output, inputs=inputs)
        print(f"Completed state/transition dipoles written to {args.output}")
        print(f"Tensor shape: {tensor.shape}")
        if args.reference_ci_mut is not None:
            diagnostics = compare_ground_transition_dipoles_to_reference(tensor, args.reference_ci_mut)
            print(f"Compared GS-ES transitions: {diagnostics.compared_transitions}")
            print(
                "Max abs GS-ES difference: "
                f"{diagnostics.max_abs_ground_transition_difference:.16e}"
            )
        return


if __name__ == "__main__":
    main()
