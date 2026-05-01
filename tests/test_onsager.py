from __future__ import annotations

from pathlib import Path

import numpy as np

from oqs_pcm.io import io_load_state_transition_dipole_tensor
from oqs_pcm.models.onsager import build_onsager_cutoff_mask, build_onsager_diagonal_kernel, onsager_reaction_factor


def test_load_transition_dipoles_builds_symmetric_tensor(tmp_path: Path) -> None:
    data_path = tmp_path / "dipoles.txt"
    data_path.write_text(
        "\n".join(
            [
                "0 0 0 1 0 0.1 0.2 0.3",
                "0 1 0 2 0 0.4 0.5 0.6",
            ]
        ),
        encoding="utf-8",
    )

    tensor = io_load_state_transition_dipole_tensor(data_path, nsts=3, state_offset=1)

    assert tensor.shape == (4, 4, 3)
    assert np.allclose(tensor[1, 2], [0.1, 0.2, 0.3])
    assert np.allclose(tensor[2, 1], [0.1, 0.2, 0.3])
    assert np.allclose(tensor[2, 3], [0.4, 0.5, 0.6])


def test_onsager_reaction_factor_returns_complex_value() -> None:
    value = onsager_reaction_factor(2.0 + 0.5j, 3.0)
    assert isinstance(value, complex)


def test_build_onsager_diagonal_kernel_sets_one_state_only() -> None:
    kernel = build_onsager_diagonal_kernel(nstates=5, state_index=2, factor=1.25)

    assert kernel.shape == (5, 3)
    assert np.allclose(kernel[2], [1.25, 1.25, 1.25])
    assert np.allclose(kernel[1], [0.0, 0.0, 0.0])


def test_build_onsager_cutoff_mask_is_symmetric() -> None:
    energies = np.array([0.0, 0.1, 0.35], dtype=float)
    mask = build_onsager_cutoff_mask(energies, cutoff=0.2)

    assert np.array_equal(mask, mask.T)
    assert mask[0, 1] == 1.0
    assert mask[0, 2] == 0.0
