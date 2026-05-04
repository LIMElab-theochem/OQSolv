from __future__ import annotations

from oqs_pcm.cli import build_parser


def test_cli_parser_accepts_all_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "example_config.toml", "--mode", "all"])

    assert args.command == "run"
    assert str(args.config) == "example_config.toml"
    assert args.mode == "all"


def test_cli_parser_defaults_to_all_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "example_config.toml"])

    assert args.mode == "all"


def test_cli_parser_accepts_fit_cutoff_variant() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "example_config.toml", "--variants", "fit_cutoff"])

    assert args.variants == ["fit_cutoff"]


def test_cli_parser_accepts_figures_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["figures", "all", "data", "out", "--fontsize", "14"])

    assert args.command == "figures"
    assert args.kind == "all"
    assert str(args.data_dir) == "data"
    assert str(args.output_dir) == "out"
    assert args.fontsize == 14


def test_cli_parser_accepts_postprocess_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "postprocess",
            "example_config.toml",
            "--output-dir",
            "tables",
            "--dipoles",
            "ci_mut.inp",
            "--delta-scf",
            "-0.01",
            "--convergence-root",
            "50",
            "out50",
        ]
    )

    assert args.command == "postprocess"
    assert str(args.config) == "example_config.toml"
    assert str(args.output_dir) == "tables"
    assert str(args.dipoles) == "ci_mut.inp"
    assert args.delta_scf == -0.01
    assert args.convergence_root == [["50", "out50"]]


def test_cli_parser_accepts_postprocess_command_with_config_only() -> None:
    parser = build_parser()
    args = parser.parse_args(["postprocess", "example_config.toml"])

    assert args.command == "postprocess"
    assert str(args.config) == "example_config.toml"
    assert args.output_dir is None
    assert args.dipoles is None
    assert args.delta_scf is None


def test_cli_parser_accepts_memory_estimate_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["estimate-memory", "example_config.toml"])

    assert args.command == "estimate-memory"
    assert str(args.config) == "example_config.toml"


def test_cli_parser_accepts_xdip_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "xdip",
            "--ao-dipoles",
            "ao.npy",
            "--mo-coefficients",
            "mo.npy",
            "--x-amplitudes",
            "x.npy",
            "--y-amplitudes",
            "y.npy",
            "--n-occupied",
            "5",
            "--nuclear-dipole",
            "1.0",
            "2.0",
            "3.0",
            "--output",
            "ci_mut_gral.inp",
        ]
    )

    assert args.command == "xdip"
    assert str(args.ao_dipoles) == "ao.npy"
    assert args.n_occupied == 5
    assert args.nuclear_dipole == [1.0, 2.0, 3.0]
