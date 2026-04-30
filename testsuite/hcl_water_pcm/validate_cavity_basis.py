from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from oqs_pcm.cavity import (
    load_calderon_matrices,
    load_tessera_areas,
    symmetric_matrix_power,
    write_cavity_mode_basis,
)
from oqs_pcm.config import load_config
from oqs_pcm.workflow import build_normal_mode_file, solve_oqs_pcm


WORKSPACE = Path("/workspace")
SUITE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SUITE_DIR / "reference"
CURRENT_DIR = SUITE_DIR / "current_cavity_basis_validation"

VARIANTS = (
    "bo",
    "ss_pol",
    "cutoff",
    "first_full_nodisp_nodiss",
    "first_full_nodiss",
    "first_full",
    "full_nodisp_nodiss",
    "full_nodiss",
    "full",
)


def build_hcl_cavity_basis_with_mathematica_t_signs(config) -> None:
    """Build HCl/water cavity files with Mathematica eigenvector signs.

    This is deliberately restricted to this regression suite.  Normal OQSolv
    runs use the production sign convention in `oqs_pcm.cavity`; here we use the
    temporary Mathematica export `T.dat` only to keep the HCl/water numerical
    regression continuous with the already validated Mathematica reference.
    """

    s_matrix, d_matrix = load_calderon_matrices(config.paths.mat_sd)
    areas = load_tessera_areas(config.paths.cavity)
    d_area = d_matrix * areas[np.newaxis, :]
    s_half = symmetric_matrix_power(s_matrix, 0.5)
    s_minus_half = symmetric_matrix_power(s_matrix, -0.5)

    transformed = s_minus_half @ (d_area @ s_half)
    transformed = 0.5 * (transformed + transformed.T)

    eigenvalues, eigenvectors_columns = np.linalg.eigh(transformed)
    order = np.argsort(np.abs(eigenvalues))[::-1]
    lambda_values = eigenvalues[order]
    python_t = eigenvectors_columns[:, order].T

    mathematica_t = np.loadtxt(WORKSPACE / "T.dat")
    if mathematica_t.shape != python_t.shape:
        raise AssertionError(
            f"T.dat shape {mathematica_t.shape} does not match Python T shape {python_t.shape}"
        )

    signs = np.sign(np.sum(python_t * mathematica_t, axis=1))
    signs[signs == 0] = 1.0
    signed_t = signs[:, np.newaxis] * python_t
    if np.max(np.abs(np.abs(signed_t) - np.abs(mathematica_t))) > 1e-10:
        raise AssertionError("T.dat is not sign-equivalent to the Python cavity eigenvectors")

    sm12t = s_minus_half @ signed_t
    write_cavity_mode_basis(config.paths.lambda_file, config.paths.sm12t, lambda_values, sm12t)


def max_abs(path: Path, reference_path: Path, dtype=float) -> float:
    actual = np.loadtxt(path, dtype=dtype)
    reference = np.loadtxt(reference_path, dtype=dtype)
    return float(np.max(np.abs(actual - reference)))


def main() -> int:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(WORKSPACE / "hcl_water_diag50.toml")
    config = replace(
        config,
        paths=replace(
            config.paths,
            lambda_file=CURRENT_DIR / "Lambda.dat",
            sm12t=CURRENT_DIR / "Sm12T.dat",
            normal_mode_potential=CURRENT_DIR / "ci_pot_normal_mode.inp",
            output_dir=CURRENT_DIR,
        ),
        execution=replace(
            config.execution,
            build_cavity_mode_basis=False,
            build_potential_modes=True,
            solve_inertial_basis=True,
            reuse_inertial_cache=False,
        ),
    )

    build_hcl_cavity_basis_with_mathematica_t_signs(config)
    build_normal_mode_file(config)
    solve_oqs_pcm(config, requested_variants=set(VARIANTS))

    lambda_diff = max_abs(CURRENT_DIR / "Lambda.dat", WORKSPACE / "Lambda.dat")
    sm12t = np.loadtxt(CURRENT_DIR / "Sm12T.dat")
    sm12t_reference = np.loadtxt(WORKSPACE / "Sm12T.dat")
    sm12t_row_gram_diff = float(np.max(np.abs(sm12t @ sm12t.T - sm12t_reference @ sm12t_reference.T)))

    print(f"Lambda max abs difference vs Mathematica export: {lambda_diff:.6e}")
    print(f"Sm12T row-Gram max abs difference vs Mathematica export: {sm12t_row_gram_diff:.6e}")

    max_energy_diff = 0.0
    for variant in VARIANTS:
        diff = max_abs(
            CURRENT_DIR / variant / "newenergies.dat",
            REFERENCE_DIR / variant / "newenergies.dat",
            dtype=complex,
        )
        max_energy_diff = max(max_energy_diff, diff)
        print(f"{variant} energy max abs difference vs Mathematica-basis reference: {diff:.6e}")

    if lambda_diff > 1e-10:
        raise AssertionError(f"Lambda spectrum differs too much: {lambda_diff:.6e}")
    if sm12t_row_gram_diff > 1e-10:
        raise AssertionError(f"Sm12T row metric differs too much: {sm12t_row_gram_diff:.6e}")
    if max_energy_diff > 5e-5:
        raise AssertionError(f"Downstream energy drift is too large: {max_energy_diff:.6e}")

    print("successful: Python-native cavity basis is validated against Mathematica exports")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"not successful: {exc}", file=sys.stderr)
        raise SystemExit(1)
