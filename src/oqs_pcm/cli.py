from __future__ import annotations

import argparse
from pathlib import Path

from oqs_pcm.config import load_config
from oqs_pcm.workflow import build_cavity_mode_basis_files, build_normal_mode_file, solve_oqs_pcm


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
        choices=("cavity-basis", "potential-modes", "normal-modes", "solve", "all"),
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
        ),
        help="Optional subset of solver variants to run. By default all supported PCM variants are computed.",
    )

    figures_parser = subparsers.add_parser("figures", help="Render figure files from postprocessed text tables.")
    figures_parser.add_argument("kind", choices=("levels", "spectra", "cutoff", "convergence", "all"))
    figures_parser.add_argument("data_dir", type=Path, help="Directory containing the text tables consumed by the figure routines.")
    figures_parser.add_argument("output_dir", type=Path, help="Directory where the generated figure files should be written.")
    figures_parser.add_argument("--fontsize", type=int, default=12, help="Base font size for the generated figures.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command in {None, "run"}:
        config = load_config(args.config)

        if args.mode in {"cavity-basis", "all"}:
            lambda_values, sm12t = build_cavity_mode_basis_files(config)
            print(f"Cavity eigenvalues written to {config.paths.lambda_file}")
            print(f"Cavity mode transformation written to {config.paths.sm12t}")
            print(f"Lambda shape: {lambda_values.shape}; Sm12T shape: {sm12t.shape}")

        if args.mode in {"potential-modes", "normal-modes", "all"}:
            transformed = build_normal_mode_file(config)
            print(f"Normal-mode potential written to {config.paths.normal_mode_potential}")
            print(f"Tensor shape: {transformed.shape}")

        if args.mode in {"solve", "all"}:
            results = solve_oqs_pcm(config, requested_variants=set(args.variants) if args.variants else None)
            print(f"Wrote {len(results)} result sets to {config.paths.output_dir}")
            print(f"Available result labels: {', '.join(results)}")
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


if __name__ == "__main__":
    main()
