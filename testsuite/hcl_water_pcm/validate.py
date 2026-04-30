from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from oqs_pcm.config import (
    AppConfig,
    DielectricConfig,
    ExecutionConfig,
    PathsConfig,
    ProblemConfig,
)
from oqs_pcm.workflow import build_normal_mode_file, solve_oqs_pcm


WORKSPACE = Path("/workspace")
SUITE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SUITE_DIR / "reference"
CURRENT_DIR = SUITE_DIR / "current_output"

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


def build_config(build_potential_modes: bool) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            mat_sd=WORKSPACE / "mat_SD.inp",
            cavity=WORKSPACE / "cavity.inp",
            ci_potential=WORKSPACE / "ci_pot.inp",
            normal_mode_potential=CURRENT_DIR / "ci_pot_normal_mode.inp",
            ci_energy=WORKSPACE / "ci_energy.inp",
            sm12t=WORKSPACE / "Sm12T.dat",
            lambda_file=WORKSPACE / "Lambda.dat",
            output_dir=CURRENT_DIR,
        ),
        problem=ProblemConfig(
            nsts=172,
            diagonalization_states=50,
            resolution_of_identity_states=172,
            corrected_states=5,
            convergence_tolerance=1e-6,
            mixing_window=10,
            cutoff=0.299561,
            optical_window_min_ev=1.63,
            optical_window_max_ev=3.26,
            hartree_to_ev=27.2114,
        ),
        execution=ExecutionConfig(
            build_cavity_mode_basis=False,
            build_potential_modes=build_potential_modes,
            solve_oqs_pcm=True,
            solve_inertial_basis=True,
            reuse_inertial_cache=not build_potential_modes,
            write_text_exports=True,
        ),
        dielectric=DielectricConfig(
            debye_c=[0.47, 72.62],
            debye_g=[146198.83040935672, 12531.328320802005],
            drude_w=[
                0.000846,
                0.00419,
                0.0212,
                0.0625,
                0.0849,
                0.204,
                0.418,
                8.34,
                9.5,
                10.41,
                11.67,
                12.95,
                14.13,
                15.5,
                17.17,
                18.89,
                21.45,
                30.06,
                49.45,
            ],
            drude_c=[
                0.259,
                1.04,
                1.62,
                0.555,
                0.238,
                0.0134,
                0.0717,
                0.0447,
                0.0327,
                0.0466,
                0.0667,
                0.0742,
                0.093,
                0.0779,
                0.079,
                0.0418,
                0.107,
                0.133,
                0.0566,
            ],
            drude_g=[
                0.000392,
                0.00743,
                0.026,
                0.0398,
                0.0299,
                0.00843,
                0.0341,
                0.75,
                1.12,
                1.26,
                1.58,
                1.65,
                1.86,
                2.22,
                2.7,
                2.82,
                6.87,
                18.28,
                36.28,
            ],
            optical_drude_start_index=8,
        ),
    )


def load_complex_array(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=complex, ndmin=1)


def assert_close(path: Path, rtol: float = 1e-8, atol: float = 1e-8) -> None:
    actual = load_complex_array(CURRENT_DIR / path)
    expected = load_complex_array(REFERENCE_DIR / path)
    if actual.shape != expected.shape:
        raise AssertionError(f"{path}: shape mismatch {actual.shape} != {expected.shape}")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.max(np.abs(actual - expected))
        raise AssertionError(f"{path}: max abs diff {diff:.6e}")


def assert_state_vectors_close(path: Path, rtol: float = 1e-8, atol: float = 1e-8) -> None:
    actual = load_complex_array(CURRENT_DIR / path)
    expected = load_complex_array(REFERENCE_DIR / path)
    if actual.shape != expected.shape:
        raise AssertionError(f"{path}: shape mismatch {actual.shape} != {expected.shape}")

    # Eigenvectors are sign/phase ambiguous. Compare each row up to the phase
    # implied by its largest reference component.
    actual_rows = np.atleast_2d(actual)
    expected_rows = np.atleast_2d(expected)
    aligned = actual_rows.copy()
    for row_index, expected_row in enumerate(expected_rows):
        pivot = int(np.argmax(np.abs(expected_row)))
        if abs(aligned[row_index, pivot]) == 0:
            continue
        phase = expected_row[pivot] / aligned[row_index, pivot]
        aligned[row_index, :] *= phase / abs(phase)

    if not np.allclose(aligned, expected_rows, rtol=rtol, atol=atol):
        diff = np.max(np.abs(aligned - expected_rows))
        raise AssertionError(f"{path}: max phase-aligned abs diff {diff:.6e}")


def assert_normal_mode_file_matches() -> None:
    actual = (CURRENT_DIR / "ci_pot_normal_mode.inp").read_text(encoding="utf-8")
    expected = (REFERENCE_DIR / "ci_pot_normal_mode.inp").read_text(encoding="utf-8")
    if actual != expected:
        raise AssertionError("ci_pot_normal_mode.inp differs from reference")


def main() -> int:
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config(build_potential_modes=True)
    build_normal_mode_file(config)
    assert_normal_mode_file_matches()

    solve_config = replace(
        config,
        execution=ExecutionConfig(
            build_cavity_mode_basis=False,
            build_potential_modes=False,
            solve_oqs_pcm=True,
            solve_inertial_basis=True,
            reuse_inertial_cache=True,
            write_text_exports=True,
        ),
    )
    solve_oqs_pcm(solve_config, requested_variants=set(VARIANTS))

    assert_close(Path("inertial_diag172/newenergies.dat"))
    assert_state_vectors_close(Path("inertial_diag172/newstates.dat"))

    for variant in VARIANTS:
        assert_close(Path(variant) / "newenergies.dat")
        assert_state_vectors_close(Path(variant) / "newstates.dat")

    print("successful: HCl-water PCM regression fixture matches all references")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"not successful: {exc}", file=sys.stderr)
        raise SystemExit(1)
