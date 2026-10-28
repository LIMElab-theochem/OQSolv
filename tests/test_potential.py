from __future__ import annotations

from pathlib import Path

import numpy as np

from oqs_pcm.potential import (
    pcm_load_frozen_state_transition_potential_tensor,
    pcm_load_ground_normal_mode_potential,
    pcm_load_normal_mode_potential_tensor,
    transform_coupling_tensor_to_normal_modes,
)


def test_normal_mode_transform_right_multiplication_matches_column_form() -> None:
    potential = np.arange(12.0).reshape(2, 2, 3)
    sm12t = np.array(
        [
            [1.0, 0.2],
            [-0.5, 2.0],
            [0.3, -0.7],
        ]
    )

    transformed = transform_coupling_tensor_to_normal_modes(potential, sm12t)
    expected = np.stack(
        [sm12t.T @ potential[i, j] for i in range(2) for j in range(2)]
    ).reshape(2, 2, 2)

    np.testing.assert_allclose(transformed, expected)


def test_normal_mode_transform_rejects_incompatible_tessera_axis() -> None:
    potential = np.zeros((2, 2, 3))
    sm12t = np.zeros((4, 2))

    with np.testing.assert_raises(ValueError):
        transform_coupling_tensor_to_normal_modes(potential, sm12t)


def test_load_frozen_state_transition_potential_tensor_streams_legacy_format(tmp_path: Path) -> None:
    ci_pot = tmp_path / "ci_pot.inp"
    ci_pot.write_text(
        "\n".join(
            [
                "2",
                "0 0",
                "1.0 0.0 10.0",
                "2.0 0.0 20.0",
                "0 1",
                "3.0",
                "4.0",
                "1 1",
                "5.0",
                "6.0",
            ]
        ),
        encoding="utf-8",
    )

    nts, potential, nuclear = pcm_load_frozen_state_transition_potential_tensor(str(ci_pot), nsts=1)

    assert nts == 2
    np.testing.assert_allclose(nuclear, [10.0, 20.0])
    np.testing.assert_allclose(potential[0, 0], [1.0, 2.0])
    np.testing.assert_allclose(potential[0, 1], [3.0, 4.0])
    np.testing.assert_allclose(potential[1, 0], [3.0, 4.0])
    np.testing.assert_allclose(potential[1, 1], [5.0, 6.0])


def test_load_normal_mode_potential_tensor_streams_legacy_format(tmp_path: Path) -> None:
    normal_modes = tmp_path / "ci_pot_normal_mode.inp"
    normal_modes.write_text(
        "\n".join(
            [
                "2",
                "1 1",
                "1.0",
                "2.0",
                "1 2",
                "3.0",
                "4.0",
                "2 2",
                "5.0",
                "6.0",
            ]
        ),
        encoding="utf-8",
    )

    nts, potential = pcm_load_normal_mode_potential_tensor(str(normal_modes), nsts=2)

    assert nts == 2
    np.testing.assert_allclose(potential[0, 0], [1.0, 2.0])
    np.testing.assert_allclose(potential[0, 1], [3.0, 4.0])
    np.testing.assert_allclose(potential[1, 0], [3.0, 4.0])
    np.testing.assert_allclose(potential[1, 1], [5.0, 6.0])


def test_load_ground_normal_mode_potential_reads_only_first_block(tmp_path: Path) -> None:
    normal_modes = tmp_path / "ci_pot_normal_mode.inp"
    normal_modes.write_text(
        "\n".join(
            [
                "2",
                "1 1",
                "1.0",
                "2.0",
                "1 2",
                "3.0",
                "4.0",
            ]
        ),
        encoding="utf-8",
    )

    nts, ground_potential = pcm_load_ground_normal_mode_potential(str(normal_modes))

    assert nts == 2
    np.testing.assert_allclose(ground_potential, [1.0, 2.0])
