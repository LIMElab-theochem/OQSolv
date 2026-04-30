from __future__ import annotations

import pytest

from oqs_pcm.config import ProblemConfig, validate_runtime_dimensions


def _problem(**overrides) -> ProblemConfig:
    data = dict(
        nsts=6,
        diagonalization_states=4,
        resolution_of_identity_states=5,
        corrected_states=3,
        convergence_tolerance=1e-6,
        mixing_window=5,
        cutoff=0.1,
        optical_window_min_ev=1.0,
        optical_window_max_ev=2.0,
        hartree_to_ev=27.2114,
    )
    data.update(overrides)
    return ProblemConfig(**data)


def test_validate_runtime_dimensions_accepts_consistent_values() -> None:
    validate_runtime_dimensions(_problem())


@pytest.mark.parametrize(
    "overrides",
    [
        {"diagonalization_states": 7},
        {"resolution_of_identity_states": 7},
        {"resolution_of_identity_states": 3},
        {"corrected_states": 5},
    ],
)
def test_validate_runtime_dimensions_rejects_inconsistent_values(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        validate_runtime_dimensions(_problem(**overrides))
