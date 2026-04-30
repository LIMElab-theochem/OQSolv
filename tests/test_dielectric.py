from __future__ import annotations

import numpy as np

from oqs_pcm.dielectric import DielectricModel


def test_dielectric_model_returns_complex_values() -> None:
    model = DielectricModel(
        debye_c=np.array([0.47, 72.62], dtype=float),
        debye_g=np.array([1.0, 2.0], dtype=float),
        drude_w=np.array([0.1, 0.2], dtype=float),
        drude_c=np.array([0.3, 0.4], dtype=float),
        drude_g=np.array([0.5, 0.6], dtype=float),
        optical_drude_start_index=1,
        hartree_to_ev=27.2114,
    )

    values = model.epsilon(np.array([0.0, 0.01], dtype=float))

    assert values.shape == (2,)
    assert np.iscomplexobj(values)


def test_dielectric_static_epsilon_is_real_scalar() -> None:
    model = DielectricModel(
        debye_c=np.array([0.47, 72.62], dtype=float),
        debye_g=np.array([1.0, 2.0], dtype=float),
        drude_w=np.array([0.1, 0.2], dtype=float),
        drude_c=np.array([0.3, 0.4], dtype=float),
        drude_g=np.array([0.5, 0.6], dtype=float),
        optical_drude_start_index=1,
        hartree_to_ev=27.2114,
    )

    value = model.epsilon_static()

    assert isinstance(value, float)
