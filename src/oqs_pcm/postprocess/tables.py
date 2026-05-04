from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oqs_pcm.config import AppConfig
from oqs_pcm.dielectric import build_dielectric_model
from oqs_pcm.io import (
    io_load_matrix,
    io_load_state_transition_dipole_tensor,
    io_load_vector,
    io_write_matrix,
    io_write_vector,
)
from oqs_pcm.models.pcm import build_pcm_kernels
from oqs_pcm.postprocess.observables import (
    ground_to_excited_transition_vectors_from_states,
    oscillator_strengths,
    transform_state_transition_observable_to_basis,
)
from oqs_pcm.potential import pcm_load_ground_normal_mode_potential
from oqs_pcm.workflow import get_inertial_cache_paths


DEFAULT_LEVEL_EXPORTS = {
    "ss_pol": "ensspol.txt",
    "full_nodisp_nodiss": "encorr1.txt",
    "full_nodiss": "encorr2.txt",
    "full": "en.txt",
    "bo": "enbo.txt",
}

DEFAULT_SPECTRA_EXPORTS = {
    "ss_pol": "sspol.txt",
    "full": "full.txt",
    "bo": "bo.txt",
    "cutoff": "cut.txt",
    "fit_cutoff": "fit.txt",
}

DEFAULT_CONVERGENCE_EXPORTS = {
    "full": "GSsolfull.txt",
    "cutoff": "GSsolcut.txt",
    "fit_cutoff": "GSsolfit.txt",
    "bo": "GSsolbo.txt",
}


@dataclass(slots=True)
class PostprocessSummary:
    """Files written by the notebook-equivalent postprocessing stage."""

    reference_shift: float
    level_files: list[Path]
    spectra_files: list[Path]
    convergence_files: list[Path]


def compute_pcm_reference_polarization_shift(config: AppConfig) -> float:
    """Compute the corrected internal `ww` shift for absolute postprocessing.

    The original notebook used `.5 (Kerneld V).V` with a positive sign.  The
    physically consistent postprocessing shift has the opposite sign, so OQSolv
    evaluates `-.5 (Kerneld V).V` internally.  This quantity is deliberately not
    exposed in the user configuration because it must remain coherent with the
    PCM kernel and normal-mode potential used by the current calculation.
    """

    _, ground_potential = pcm_load_ground_normal_mode_potential(str(config.paths.normal_mode_potential))
    dielectric = build_dielectric_model(config.dielectric)
    kernels = build_pcm_kernels(
        lambda_path=config.paths.lambda_file,
        dielectric_model=dielectric,
        energies=np.zeros(config.problem.nsts, dtype=float),
        optical_window_ev=(config.problem.optical_window_min_ev, config.problem.optical_window_max_ev),
    )
    return float(-0.5 * np.sum(kernels.optical_kernel * ground_potential * ground_potential))


def load_result_set(result_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one solver result directory containing `newenergies/newstates`."""

    base = Path(result_dir)
    # Some OQS variants write complex-valued text exports. Mathematica's
    # postprocessing used `Re[...]` for energies but kept state coefficients as
    # exported when constructing transition dipoles.
    energies = np.loadtxt(base / "newenergies.dat", dtype=complex, ndmin=1)
    states = np.loadtxt(base / "newstates.dat", dtype=complex, ndmin=2)
    return energies, states


def write_absolute_energy_table(
    result_dir: str | Path,
    output_path: str | Path,
    *,
    reference_shift: float,
    delta_scf: float,
) -> Path:
    """Write an absolute-energy vector equivalent to notebook `en + ww + deltascf` exports."""

    energies, _ = load_result_set(result_dir)
    output = Path(output_path)
    io_write_vector(output, np.real(energies) + reference_shift + delta_scf)
    return output


def write_ground_state_convergence_table(
    result_dirs_by_nstates: dict[int, str | Path],
    output_path: str | Path,
    *,
    reference_shift: float,
    delta_scf: float,
) -> Path:
    """Write `{nsts, ground_energy + ww + deltascf}` convergence data."""

    rows = []
    for nstates, result_dir in sorted(result_dirs_by_nstates.items()):
        energies, _ = load_result_set(result_dir)
        rows.append((nstates, float(np.real(energies[0]) + reference_shift + delta_scf)))
    output = Path(output_path)
    io_write_matrix(output, np.asarray(rows, dtype=float))
    return output


def write_spectrum_table(
    result_dir: str | Path,
    output_path: str | Path,
    *,
    state_transition_dipoles: np.ndarray,
    reference_shift: float = 0.0,
) -> Path:
    """Write a two-column `(transition energy, oscillator strength)` table.

    The optional `reference_shift` is added before computing oscillator
    strengths to mirror the notebook, although it cancels from transition
    energies because both ground and excited states receive the same constant.
    """

    energies, states = load_result_set(result_dir)
    shifted_energies = np.real(energies) + reference_shift
    transition_vectors = ground_to_excited_transition_vectors_from_states(state_transition_dipoles, states)
    strengths = oscillator_strengths(shifted_energies, transition_vectors)
    transition_energies = shifted_energies[1:] - shifted_energies[0]
    output = Path(output_path)
    io_write_matrix(output, np.column_stack((transition_energies, strengths)))
    return output


def load_postprocess_dipoles(
    dipole_file: str | Path,
    *,
    total_states: int,
    inertial_state_file: str | Path | None = None,
) -> np.ndarray:
    """Load dipoles and optionally transform them to the inertial basis."""

    dipoles = io_load_state_transition_dipole_tensor(dipole_file, nsts=total_states - 1)
    if inertial_state_file is None:
        return dipoles
    inertial_states = io_load_matrix(inertial_state_file)
    return transform_state_transition_observable_to_basis(dipoles, inertial_states)


def write_notebook_equivalent_tables(
    config: AppConfig,
    *,
    output_dir: str | Path,
    dipole_file: str | Path | None = None,
    delta_scf: float = 0.0,
    include_levels: bool = True,
    include_spectra: bool = True,
    include_convergence: bool = False,
    spectra_exports: dict[str, str] | None = None,
    level_exports: dict[str, str] | None = None,
    convergence_exports: dict[str, str] | None = None,
    convergence_roots_by_nstates: dict[int, Path] | None = None,
) -> PostprocessSummary:
    """Generate the clean table layer formerly assembled in `figures-new.nb`.

    This routine intentionally writes reduced text tables, not figures. Figure
    rendering remains in `oqs_pcm.figures`, so numerical postprocessing can be
    tested independently from matplotlib.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shift = compute_pcm_reference_polarization_shift(config)
    level_files: list[Path] = []
    spectra_files: list[Path] = []
    convergence_files: list[Path] = []

    if include_levels:
        for variant, filename in (level_exports or DEFAULT_LEVEL_EXPORTS).items():
            result_dir = config.paths.output_dir / variant
            if result_dir.exists():
                level_files.append(
                    write_absolute_energy_table(
                        result_dir,
                        destination / filename,
                        reference_shift=shift,
                        delta_scf=delta_scf,
                    )
                )

    if include_spectra:
        if dipole_file is None:
            raise ValueError("`dipole_file` is required when spectra table generation is enabled.")
        inertial_state_file = None
        if config.execution.solve_inertial_basis:
            _, inertial_state_file = get_inertial_cache_paths(config)
        dipoles = load_postprocess_dipoles(
            dipole_file,
            total_states=config.problem.nsts,
            inertial_state_file=inertial_state_file,
        )
        for variant, filename in (spectra_exports or DEFAULT_SPECTRA_EXPORTS).items():
            result_dir = config.paths.output_dir / variant
            if result_dir.exists():
                spectra_files.append(
                    write_spectrum_table(
                        result_dir,
                        destination / filename,
                        state_transition_dipoles=dipoles,
                        reference_shift=shift,
                    )
                )

    if include_convergence:
        if convergence_roots_by_nstates is None:
            raise ValueError("`convergence_roots_by_nstates` is required when convergence table generation is enabled.")
        for variant, filename in (convergence_exports or DEFAULT_CONVERGENCE_EXPORTS).items():
            result_dirs = {
                nstates: root / variant
                for nstates, root in convergence_roots_by_nstates.items()
                if (root / variant).exists()
            }
            if result_dirs:
                convergence_files.append(
                    write_ground_state_convergence_table(
                        result_dirs,
                        destination / filename,
                        reference_shift=shift,
                        delta_scf=delta_scf,
                    )
                )

    return PostprocessSummary(
        reference_shift=shift,
        level_files=level_files,
        spectra_files=spectra_files,
        convergence_files=convergence_files,
    )
