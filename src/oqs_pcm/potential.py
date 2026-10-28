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

    def next_nonempty_line(handle) -> str:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                return stripped
        raise EOFError("Unexpected end of PCM potential file.")

    with open(path, encoding="utf-8") as handle:
        nts = int(parse_numbers(next_nonempty_line(handle))[0])
        # The legacy file stores a ground-state label line before the
        # ground-state potential/nuclear rows. Stream the file instead of
        # materializing all text lines; HONO-size LR-TDDFT files can be >1 GB.
        next_nonempty_line(handle)
        ve = np.zeros((nsts + 1, nsts + 1, nts), dtype=float)

        ground_values = np.empty(nts, dtype=float)
        nuclear = np.empty(nts, dtype=float)
        for tessera in range(nts):
            row = parse_numbers(next_nonempty_line(handle))
            ground_values[tessera] = row[0]
            nuclear[tessera] = row[2]
        ve[0, 0, :] = ground_values

        while True:
            try:
                label_line = next_nonempty_line(handle)
            except EOFError:
                break
            i_state, j_state = (int(value) for value in parse_numbers(label_line)[:2])
            values = np.empty(nts, dtype=float)
            for tessera in range(nts):
                values[tessera] = parse_numbers(next_nonempty_line(handle))[0]
            ve[i_state, j_state, :] = values
            ve[j_state, i_state, :] = values

    return nts, ve, nuclear


def pcm_add_nuclear_potential_to_state_diagonals(ve: np.ndarray, nuclear: np.ndarray) -> np.ndarray:
    """Add the nuclear electrostatic contribution to every diagonal state block."""

    result = ve.copy()
    for state_index in range(result.shape[0]):
        result[state_index, state_index, :] += nuclear
    return result


def transform_coupling_tensor_to_normal_modes(
    coupling_tensor: np.ndarray,
    sm12t: np.ndarray,
) -> np.ndarray:
    """Transform a state-coupling tensor to the normal-mode representation.

    The input stores each state/transition potential as a row along its last
    axis, with shape ``(..., tessera)``.  ``Sm12T`` is stored as
    ``(tessera, mode)`` and is therefore applied on the right:

    ``V_normal[..., mode] = V_frozen[..., tessera] @ Sm12T[tessera, mode]``.

    This is the row-vector form of ``Sm12T.T @ V`` and avoids an unnecessary
    transpose/reordering of the coupling tensor.  The same algebra applies
    more generally to any coupling tensor resolved over cavity channels.
    """

    coupling_tensor = np.asarray(coupling_tensor)
    sm12t = np.asarray(sm12t)
    if coupling_tensor.ndim < 1:
        raise ValueError("The coupling tensor must have a cavity-channel axis.")
    if sm12t.ndim != 2:
        raise ValueError("Sm12T must be a two-dimensional (tessera, mode) matrix.")
    if coupling_tensor.shape[-1] != sm12t.shape[0]:
        raise ValueError(
            "The coupling tensor's last axis must match the Sm12T tessera axis: "
            f"{coupling_tensor.shape[-1]} != {sm12t.shape[0]}."
        )
    return np.matmul(coupling_tensor, sm12t)


def pcm_load_normal_mode_potential_tensor(path: str, nsts: int) -> tuple[int, np.ndarray]:
    """Read the PCM normal-mode potential tensor from the legacy text format."""

    def next_nonempty_line(handle) -> str:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                return stripped
        raise EOFError("Unexpected end of PCM normal-mode potential file.")

    with open(path, encoding="utf-8") as handle:
        nts = int(float(next_nonempty_line(handle)))
        tensor = np.zeros((nsts, nsts, nts), dtype=float)
        while True:
            try:
                label_line = next_nonempty_line(handle)
            except EOFError:
                break
            i_state, j_state = (int(part) for part in label_line.split())
            values = np.empty(nts, dtype=float)
            for tessera in range(nts):
                values[tessera] = float(next_nonempty_line(handle))
            tensor[i_state - 1, j_state - 1, :] = values
            tensor[j_state - 1, i_state - 1, :] = values

    return nts, tensor


def pcm_load_ground_normal_mode_potential(path: str) -> tuple[int, np.ndarray]:
    """Read only the ground-state/ground-state normal-mode potential block.

    The legacy `ci_pot_normal_mode.inp` format starts with `nts`, then the
    first state-pair label, then `nts` values.  OQSolv writes the upper
    triangular state pairs in order, so the first block is the ground-state
    `1 1` block.  Postprocessing uses this block to compute the scalar
    reference polarization shift `ww`; reading the full multi-GB tensor would
    be unnecessary for that operation.
    """

    def next_nonempty_line(handle) -> str:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                return stripped
        raise EOFError("Unexpected end of PCM normal-mode potential file.")

    with open(path, encoding="utf-8") as handle:
        nts = int(float(next_nonempty_line(handle)))
        label_line = next_nonempty_line(handle)
        i_state, j_state = (int(part) for part in label_line.split())
        if (i_state, j_state) != (1, 1):
            raise ValueError(
                "Expected the first normal-mode potential block to be the "
                f"ground-state pair `1 1`, got `{i_state} {j_state}`."
            )
        values = np.empty(nts, dtype=float)
        for tessera in range(nts):
            values[tessera] = float(next_nonempty_line(handle))
    return nts, values


# Backward-compatible aliases used while the naming refactor propagates.
load_ci_potential = pcm_load_frozen_state_transition_potential_tensor
add_nuclear_diagonal = pcm_add_nuclear_potential_to_state_diagonals
transform_to_normal_modes = transform_coupling_tensor_to_normal_modes
load_normal_mode_potential = pcm_load_normal_mode_potential_tensor
