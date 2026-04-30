from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_ragged_numeric_rows(path: str | Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append([float(token.replace("D", "E").replace("d", "e")) for token in stripped.split()])
    return rows


def load_calderon_matrices(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Calderon `S` and `D` matrices from the legacy `mat_SD.inp` file."""

    rows = _read_ragged_numeric_rows(path)
    n_tesserae = int(rows[0][0])
    body = np.asarray(rows[1:], dtype=float)
    if body.shape != (n_tesserae * n_tesserae, 2):
        raise ValueError(
            "`mat_SD.inp` does not contain the expected two-column S/D block "
            f"for {n_tesserae} tesserae."
        )

    s_matrix = body[:, 0].reshape((n_tesserae, n_tesserae), order="F")
    d_matrix = body[:, 1].reshape((n_tesserae, n_tesserae), order="F")
    return s_matrix, d_matrix


def load_tessera_areas(path: str | Path) -> np.ndarray:
    """Load tessera areas from the legacy `cavity.inp` file."""

    rows = _read_ragged_numeric_rows(path)
    n_spheres = int(rows[0][1])
    tessera_rows = rows[n_spheres + 1 :]
    return np.asarray([row[-2] for row in tessera_rows], dtype=float)


def symmetric_matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    """Compute a real symmetric matrix power using an eigen-decomposition."""

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= 0):
        raise ValueError("Matrix power requires a positive-definite S matrix.")
    powered = (eigenvectors * (eigenvalues**exponent)) @ eigenvectors.T
    return 0.5 * (powered + powered.T)


def choose_stable_eigenvector_signs(eigenvectors_columns: np.ndarray) -> np.ndarray:
    """Apply a deterministic sign convention to eigenvectors.

    Eigenvectors of a real symmetric matrix are defined only up to an overall
    sign: both `v` and `-v` solve the same eigenvalue equation.  For reproducible
    OQSolv output, each complete eigenvector is multiplied by one sign chosen so
    that its largest-magnitude component is positive.

    The operation is applied column-by-column because NumPy stores eigenvectors
    as columns.  This never changes individual components independently.
    """

    signed = np.array(eigenvectors_columns, copy=True)
    largest_component_indices = np.argmax(np.abs(signed), axis=0)
    for vector_index, component_index in enumerate(largest_component_indices):
        if signed[component_index, vector_index] < 0:
            signed[:, vector_index] *= -1.0
    return signed


def build_cavity_mode_basis(
    mat_sd_path: str | Path,
    cavity_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `Lambda` and `Sm12T` from PCM cavity matrices.

    This ports the operational part of `lambda_sm12t.nb`:

    1. reshape Calderon `S` and `D`
    2. multiply `D` by tessera areas
    3. diagonalize `S^{-1/2} D A S^{1/2}`
    4. return cavity eigenvalues and the transformation matrix `S^{-1/2} T`
    """

    s_matrix, d_matrix = load_calderon_matrices(mat_sd_path)
    areas = load_tessera_areas(cavity_path)
    if len(areas) != s_matrix.shape[0]:
        raise ValueError("`cavity.inp` tessera count does not match `mat_SD.inp`.")

    d_area = d_matrix * areas[np.newaxis, :]
    s_half = symmetric_matrix_power(s_matrix, 0.5)
    s_minus_half = symmetric_matrix_power(s_matrix, -0.5)
    transformed = s_minus_half @ (d_area @ s_half)
    transformed = 0.5 * (transformed + transformed.T)

    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    lambda_values = eigenvalues[order]
    eigenvectors = choose_stable_eigenvector_signs(eigenvectors[:, order])
    # NumPy returns eigenvectors as columns. Mathematica's `Eigensystem`
    # returns the eigenvector list as a matrix whose rows are the eigenvectors,
    # so the row-oriented representation is `eigenvectors.T` after ordering.
    # This follows the notebook expression `Sm12 . eigen[[2]]`.
    sm12t = s_minus_half @ eigenvectors.T
    return lambda_values, sm12t


def write_cavity_mode_basis(
    lambda_path: str | Path,
    sm12t_path: str | Path,
    lambda_values: np.ndarray,
    sm12t: np.ndarray,
) -> None:
    """Write cavity-mode basis files consumed by downstream PCM steps."""

    lambda_path = Path(lambda_path)
    sm12t_path = Path(sm12t_path)
    lambda_path.parent.mkdir(parents=True, exist_ok=True)
    sm12t_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(lambda_path, lambda_values.reshape(-1, 1), fmt="%.18e")
    np.savetxt(sm12t_path, sm12t, fmt="%.18e")
