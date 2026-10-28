from __future__ import annotations

from pathlib import Path

import numpy as np

from oqs_pcm.io import (
    io_load_normal_mode_transformation_matrix,
    io_write_named_results,
    io_write_pcm_normal_mode_potential_tensor,
)


def test_load_sm12t_preserves_native_tessera_by_mode_orientation(tmp_path: Path) -> None:
    sm12t = tmp_path / "Sm12T.dat"
    sm12t.write_text("1 2\n3 4\n", encoding="utf-8")

    loaded = io_load_normal_mode_transformation_matrix(sm12t)

    np.testing.assert_allclose(loaded, [[1.0, 2.0], [3.0, 4.0]])


def test_write_named_results_creates_expected_files(tmp_path: Path) -> None:
    base_dir = tmp_path / "results"
    energies = np.array([0.1, 0.2], dtype=float)
    states = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    io_write_named_results(base_dir, energies, states)

    assert (base_dir / "newenergies.dat").exists()
    assert (base_dir / "newstates.dat").exists()


def test_write_normal_mode_potential_creates_text_output(tmp_path: Path) -> None:
    output_path = tmp_path / "ci_pot_normal_mode.inp"
    potential = np.zeros((2, 2, 3), dtype=float)
    potential[0, 0, :] = [1.0, 2.0, 3.0]
    potential[0, 1, :] = [4.0, 5.0, 6.0]
    potential[1, 1, :] = [7.0, 8.0, 9.0]

    io_write_pcm_normal_mode_potential_tensor(output_path, potential)

    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("3\n")
    assert "1 1" in text
    assert "1 2" in text
    assert "2 2" in text
