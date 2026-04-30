import numpy as np

from oqs_pcm.cavity import choose_stable_eigenvector_signs


def test_stable_eigenvector_signs_are_applied_per_eigenvector_column():
    eigenvectors = np.array(
        [
            [-0.1, 0.2, -0.3],
            [-0.9, -0.8, 0.1],
            [0.4, 0.1, -0.7],
        ]
    )

    signed = choose_stable_eigenvector_signs(eigenvectors)

    largest_component_indices = np.argmax(np.abs(signed), axis=0)
    for vector_index, component_index in enumerate(largest_component_indices):
        assert signed[component_index, vector_index] > 0

    np.testing.assert_allclose(np.abs(signed), np.abs(eigenvectors))
