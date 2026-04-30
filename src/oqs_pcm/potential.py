from __future__ import annotations

import numpy as np

from oqs_pcm.io import io_load_numeric_table


def pcm_load_frozen_state_transition_potential_tensor(path: str, nsts: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Load frozen-solvent PCM state and transition electrostatic potentials.

    Returns:
    - number of tesserae
    - state/transition electrostatic-potential tensor with shape
      `(nsts + 1, nsts + 1, nts)`
    - nuclear contribution vector with shape (nts,)
    """

    def parse_numbers(line: str) -> list[float]:
        return [float(token.replace("D", "E").replace("d", "e")) for token in line.split()]

    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    nts = int(parse_numbers(lines[0])[0])
    ve = np.zeros((nsts + 1, nsts + 1, nts), dtype=float)

    ground_state_rows = [parse_numbers(line) for line in lines[2: 2 + nts]]
    ve[0, 0, :] = np.array([row[0] for row in ground_state_rows], dtype=float)
    nuclear = np.array([row[2] for row in ground_state_rows], dtype=float)

    cursor = 2 + nts
    rows_per_group = nts + 1
    num_groups = (len(lines) - cursor) // rows_per_group

    for group_index in range(num_groups):
        start = cursor + group_index * rows_per_group
        i_state, j_state = (int(value) for value in parse_numbers(lines[start])[:2])
        values = np.array(
            [parse_numbers(lines[start + 1 + tessera])[0] for tessera in range(nts)],
            dtype=float,
        )
        ve[i_state, j_state, :] = values
        ve[j_state, i_state, :] = values

    return nts, ve, nuclear


def pcm_add_nuclear_potential_to_state_diagonals(ve: np.ndarray, nuclear: np.ndarray) -> np.ndarray:
    """Add the nuclear electrostatic contribution to every diagonal state block."""

    result = ve.copy()
    for state_index in range(result.shape[0]):
        result[state_index, state_index, :] += nuclear
    return result


def transform_coupling_tensor_to_normal_modes(coupling_tensor: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """Transform a state-coupling tensor to the normal-mode representation.

    In the current PCM workflow this acts on the tensor of state/transition
    electrostatic potentials. The same algebra also applies more generally to
    any coupling tensor resolved over cavity channels.
    """

    channel_first = np.transpose(coupling_tensor, (2, 0, 1))
    transformed = np.tensordot(transformation_matrix, channel_first, axes=(1, 0))
    return np.transpose(transformed, (1, 2, 0))


def pcm_load_normal_mode_potential_tensor(path: str, nsts: int) -> tuple[int, np.ndarray]:
    """Read the PCM normal-mode potential tensor from the legacy text format."""

    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    nts = int(float(lines[0]))
    tensor = np.zeros((nsts, nsts, nts), dtype=float)
    cursor = 1
    rows_per_group = nts + 1
    num_groups = len(lines[cursor:]) // rows_per_group

    for group_index in range(num_groups):
        start = cursor + group_index * rows_per_group
        i_state, j_state = (int(part) for part in lines[start].split())
        values = np.array([float(item) for item in lines[start + 1: start + 1 + nts]], dtype=float)
        tensor[i_state - 1, j_state - 1, :] = values
        tensor[j_state - 1, i_state - 1, :] = values

    return nts, tensor


# Backward-compatible aliases used while the naming refactor propagates.
load_ci_potential = pcm_load_frozen_state_transition_potential_tensor
add_nuclear_diagonal = pcm_add_nuclear_potential_to_state_diagonals
transform_to_normal_modes = transform_coupling_tensor_to_normal_modes
load_normal_mode_potential = pcm_load_normal_mode_potential_tensor
