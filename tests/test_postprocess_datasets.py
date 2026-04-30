from __future__ import annotations

from pathlib import Path

from oqs_pcm.postprocess.datasets import load_convergence_dataset, load_cutoff_dataset, load_level_dataset, load_spectra_dataset


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_level_dataset(tmp_path: Path) -> None:
    for name in ["ensspol.txt", "encorr1.txt", "encorr2.txt", "en.txt", "enbo.txt"]:
        _write_text(tmp_path / name, "0.1\n0.2\n")

    dataset = load_level_dataset(tmp_path)

    assert dataset.ensspol.shape == (2,)
    assert dataset.enbo.shape == (2,)


def test_load_spectra_dataset(tmp_path: Path) -> None:
    files = {"full": "full.txt", "bo": "bo.txt"}
    for name in files.values():
        _write_text(tmp_path / name, "0.1 1.0\n0.2 0.5\n")

    dataset = load_spectra_dataset(tmp_path, files)

    assert set(dataset.tables) == {"full", "bo"}
    assert dataset.tables["full"].shape == (2, 2)


def test_load_cutoff_dataset(tmp_path: Path) -> None:
    for name in ["sspol.txt", "full.txt", "cut.txt", "fit.txt", "reeps.txt", "reepscut.txt", "reepsfit.txt"]:
        _write_text(tmp_path / name, "0.1 1.0\n0.2 0.5\n")

    dataset = load_cutoff_dataset(tmp_path)

    assert set(dataset.spectra_tables) == {"sspol", "full", "cut", "fit"}
    assert set(dataset.dielectric_tables) == {"reeps", "reepscut", "reepsfit"}


def test_load_convergence_dataset(tmp_path: Path) -> None:
    mapping = {"full": "GSsolfull.txt", "bo": "GSsolbo.txt"}
    for name in mapping.values():
        _write_text(tmp_path / name, "50 0.1\n100 0.2\n")

    dataset = load_convergence_dataset(tmp_path, mapping)

    assert dataset.tables["full"].shape == (2, 2)
