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


def test_cli_parser_accepts_figures_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["figures", "all", "data", "out", "--fontsize", "14"])

    assert args.command == "figures"
    assert args.kind == "all"
    assert str(args.data_dir) == "data"
    assert str(args.output_dir) == "out"
    assert args.fontsize == 14
