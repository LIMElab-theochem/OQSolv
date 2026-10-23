import numpy as np
import pytest

from oqs_pcm.config import ProblemConfig, validate_runtime_dimensions
from oqs_pcm.solver import transform_coupling_tensor_to_inertial_basis


def make_problem(selected_states=None) -> ProblemConfig:
    return ProblemConfig(
        nsts=5,
        diagonalization_states=2,
        corrected_states=2,
        convergence_tolerance=1e-6,
        mixing_window=2,
        cutoff=0.1,
        fit_cutoff=0.2,
        fit_epsilon=2.0,
        optical_window_min_ev=1.0,
        optical_window_max_ev=2.0,
        hartree_to_ev=27.2114,
        resolution_of_identity_states=2,
        selected_states=selected_states,
    )


def test_selected_states_use_one_based_nonconsecutive_labels() -> None:
    problem = make_problem([1, 5])
    validate_runtime_dimensions(problem)
    assert problem.selected_state_indices() == (0, 4)


def test_selected_states_require_ground_state_first() -> None:
    with pytest.raises(ValueError, match="start with ground state"):
        validate_runtime_dimensions(make_problem([2, 5]))


def test_inertial_tensor_transform_preserves_channel_axis() -> None:
    states = np.array([[1.0, 0.0], [0.0, 1.0]])
    potential = np.arange(12.0).reshape(2, 2, 3)
    transformed = transform_coupling_tensor_to_inertial_basis(states, potential)
    expected = np.stack([states @ potential[:, :, channel] @ states.T for channel in range(3)], axis=2)
    np.testing.assert_allclose(transformed, 0.5 * (expected + np.swapaxes(expected, 0, 1)))
