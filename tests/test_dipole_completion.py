from __future__ import annotations

from pathlib import Path

import numpy as np

from oqs_pcm.dipole_completion import (
    DipoleCompletionInputs,
    compare_ground_transition_dipoles_to_reference,
    complete_ci_mut_file,
    reconstruct_completed_state_transition_dipoles,
)
from oqs_pcm.io import io_write_state_transition_dipole_tensor


def _small_inputs() -> DipoleCompletionInputs:
    ao_dipoles = np.zeros((2, 2, 3), dtype=float)
    ao_dipoles[0, 0] = [1.0, 0.0, 0.0]
    ao_dipoles[1, 1] = [3.0, 0.0, 0.0]
    ao_dipoles[0, 1] = ao_dipoles[1, 0] = [2.0, 0.0, 0.0]
    return DipoleCompletionInputs(
        ao_dipoles=ao_dipoles,
        mo_coefficients=np.eye(2),
        x_amplitudes=np.array([[1.0]]),
        y_amplitudes=np.array([[0.0]]),
        n_occupied=1,
        nuclear_dipole=np.array([10.0, 0.0, 0.0]),
    )


def test_reconstruct_completed_state_transition_dipoles_wraps_validated_algebra() -> None:
    tensor = reconstruct_completed_state_transition_dipoles(_small_inputs())

    np.testing.assert_allclose(tensor[0, 0], [8.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[0, 1], [-np.sqrt(2.0) * 2.0, 0.0, 0.0])
    np.testing.assert_allclose(tensor[1, 1], [6.0, 0.0, 0.0])


def test_complete_ci_mut_file_writes_completed_tensor(tmp_path: Path) -> None:
    output = tmp_path / "ci_mut_gral.inp"

    tensor = complete_ci_mut_file(output_path=output, inputs=_small_inputs())

    assert output.exists()
    assert output.read_text(encoding="utf-8").splitlines()[0].startswith("States 1 and 1")
    assert tensor.shape == (2, 2, 3)


def test_compare_ground_transition_dipoles_to_reference_uses_native_gs_es_rows(tmp_path: Path) -> None:
    tensor = reconstruct_completed_state_transition_dipoles(_small_inputs())
    reference = np.zeros_like(tensor)
    reference[0, 1] = reference[1, 0] = tensor[0, 1]
    reference_path = tmp_path / "ci_mut_reference.inp"
    io_write_state_transition_dipole_tensor(reference_path, reference)

    diagnostics = compare_ground_transition_dipoles_to_reference(tensor, reference_path)

    assert diagnostics.compared_transitions == 1
    assert diagnostics.max_abs_ground_transition_difference == 0.0
