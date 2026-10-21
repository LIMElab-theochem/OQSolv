import numpy as np

import oqs_pcm.cavity as cavity
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


def test_build_cavity_mode_basis_stores_article_column_orientation(monkeypatch):
    s_matrix = np.diag([1.0, 2.0, 4.0])
    s_half = np.diag(np.sqrt(np.diag(s_matrix)))
    s_minus_half = np.diag(1.0 / np.sqrt(np.diag(s_matrix)))
    transformed = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 2.0, 0.3],
            [0.0, 0.3, -0.5],
        ]
    )
    d_matrix = s_half @ transformed @ s_minus_half
    monkeypatch.setattr(cavity, "load_calderon_matrices", lambda _: (s_matrix, d_matrix))
    monkeypatch.setattr(cavity, "load_tessera_areas", lambda _: np.ones(3))

    lambda_values, stored_sm12t = cavity.build_cavity_mode_basis("mat_SD.inp", "cavity.inp")

    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    expected_vectors = cavity.choose_stable_eigenvector_signs(eigenvectors[:, order])
    np.testing.assert_allclose(lambda_values, eigenvalues[order])
    np.testing.assert_allclose(stored_sm12t, s_minus_half @ expected_vectors)
    np.testing.assert_allclose(stored_sm12t.T, expected_vectors.T @ s_minus_half)
    assert not np.allclose(stored_sm12t, s_minus_half @ expected_vectors.T)
