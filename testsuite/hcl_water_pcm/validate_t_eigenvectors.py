"""Validate the Python cavity eigenvectors against Mathematica's `T.dat`.

This is a diagnostic/porting check, not a normal OQSolv runtime step.  Eigenvector
signs are arbitrary, so this script compares selected entries of `|T|` rather
than `T` itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from oqs_pcm.cavity import (
    load_calderon_matrices,
    load_tessera_areas,
    symmetric_matrix_power,
)


ROOT = Path(__file__).resolve().parents[2]
BLOCK_SIZE = 5
TOLERANCE = 1.0e-10


def compute_python_t_matrix() -> np.ndarray:
    """Compute the Mathematica-style eigenvector matrix T.

    Mathematica's `Eigensystem` stores eigenvectors as a list of row vectors.
    NumPy stores eigenvectors as columns, so the transpose gives the comparable
    row-oriented matrix.
    """

    s_matrix, d_matrix = load_calderon_matrices(ROOT / "mat_SD.inp")
    areas = load_tessera_areas(ROOT / "cavity.inp")

    d_area = d_matrix * areas[np.newaxis, :]
    s_half = symmetric_matrix_power(s_matrix, 0.5)
    s_minus_half = symmetric_matrix_power(s_matrix, -0.5)

    transformed = s_minus_half @ (d_area @ s_half)
    transformed = 0.5 * (transformed + transformed.T)

    eigenvalues, eigenvectors_columns = np.linalg.eigh(transformed)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvectors_columns[:, order].T


def main() -> None:
    python_t = compute_python_t_matrix()
    mathematica_t = np.loadtxt(ROOT / "T.dat")

    print("row col |Python T| |Mathematica T| difference")
    max_difference = 0.0
    for row in range(BLOCK_SIZE):
        for column in range(BLOCK_SIZE):
            python_value = abs(python_t[row, column])
            mathematica_value = abs(mathematica_t[row, column])
            difference = python_value - mathematica_value
            max_difference = max(max_difference, abs(difference))
            print(
                f"{row + 1:3d} {column + 1:3d} "
                f"{python_value:.16e} {mathematica_value:.16e} {difference: .3e}"
            )

    print(f"maximum checked modulus difference: {max_difference:.3e}")
    if max_difference > TOLERANCE:
        raise SystemExit(
            "not successful: Python T differs from Mathematica T in the checked block"
        )
    print(
        f"successful: checked {BLOCK_SIZE * BLOCK_SIZE} out of "
        f"{python_t.size} T entries modulus-wise"
    )


if __name__ == "__main__":
    main()
