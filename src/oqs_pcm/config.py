from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class PathsConfig:
    mat_sd: Path
    cavity: Path
    ci_potential: Path
    normal_mode_potential: Path
    sm12t: Path
    lambda_file: Path
    output_dir: Path
    ci_energy: Path | None = None
    inertial_cache_dir: Path | None = None
    inertial_normal_mode_potential: Path | None = None
    inertial_energy: Path | None = None
    inertial_energy_offset: Path | None = None
    inertial_dipoles: Path | None = None


@dataclass(slots=True)
class ProblemConfig:
    nsts: int
    diagonalization_states: int
    corrected_states: int
    convergence_tolerance: float
    mixing_window: int
    cutoff: float
    fit_cutoff: float
    fit_epsilon: float
    optical_window_min_ev: float
    optical_window_max_ev: float
    hartree_to_ev: float
    resolution_of_identity_states: int | None = None
    selected_states: list[int] | None = None

    def selected_state_indices(self) -> tuple[int, ...]:
        """Return selected electronic states as zero-based indices.

        TOML state labels are one-based, matching the legacy input files and
        Mathematica workflow.  ``None`` means retain the complete state space.
        """

        if self.selected_states is None:
            return tuple(range(self.nsts))
        return tuple(state - 1 for state in self.selected_states)


@dataclass(slots=True)
class ExecutionConfig:
    build_cavity_mode_basis: bool
    build_potential_modes: bool
    solve_oqs_pcm: bool
    solve_inertial_basis: bool
    reuse_inertial_cache: bool
    write_text_exports: bool
    write_inertial_potentials: bool = False
    write_inertial_dipoles: bool = False
    potential_basis: str = "normal_mode"


@dataclass(slots=True)
class DielectricConfig:
    debye_c: list[float]
    debye_g: list[float]
    drude_w: list[float]
    drude_c: list[float]
    drude_g: list[float]
    optical_drude_start_index: int


@dataclass(slots=True)
class PostprocessConfig:
    output_dir: Path | None
    dipoles: Path | None
    delta_scf: float
    oscillator_strength_mode: str | None
    dipole_basis: str = "frozen"


@dataclass(slots=True)
class SafetyConfig:
    max_memory_gb: float
    min_available_memory_gb: float
    memory_safety_factor: float
    min_free_disk_gb: float
    disk_safety_factor: float


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    problem: ProblemConfig
    execution: ExecutionConfig
    dielectric: DielectricConfig
    postprocess: PostprocessConfig
    safety: SafetyConfig


def validate_runtime_dimensions(problem: ProblemConfig) -> None:
    """Validate the coupled dimensions of one OQS-Continuum run."""

    n_total_states = problem.nsts
    selected_states = problem.selected_states
    effective_states = n_total_states if selected_states is None else len(selected_states)
    n_hamiltonian_states = problem.diagonalization_states
    n_ri_states = problem.resolution_of_identity_states or n_total_states
    n_corrected_states = problem.corrected_states

    if n_total_states < 1:
        raise ValueError("`problem.nsts` must be at least 1 because it includes the ground state.")
    if n_hamiltonian_states < 1:
        raise ValueError("`problem.diagonalization_states` must be at least 1.")
    if n_ri_states < 1:
        raise ValueError("`problem.resolution_of_identity_states` must be at least 1.")
    if n_corrected_states < 1:
        raise ValueError("`problem.corrected_states` must be at least 1.")
    if n_hamiltonian_states > effective_states:
        raise ValueError("`problem.diagonalization_states` cannot exceed the selected state count.")
    if n_ri_states > effective_states:
        raise ValueError("`problem.resolution_of_identity_states` cannot exceed the selected state count.")
    if n_ri_states < n_hamiltonian_states:
        raise ValueError(
            "`problem.resolution_of_identity_states` cannot be smaller than "
            "`problem.diagonalization_states` because the RI space must span at "
            "least the Hamiltonian dimension."
        )
    if n_corrected_states > n_hamiltonian_states:
        raise ValueError(
            "`problem.corrected_states` cannot exceed `problem.diagonalization_states`."
        )
    if selected_states is not None:
        if len(selected_states) < 1:
            raise ValueError("`problem.selected_states` must contain at least one state.")
        if len(set(selected_states)) != len(selected_states):
            raise ValueError("`problem.selected_states` must not contain duplicates.")
        if any(state < 1 or state > n_total_states for state in selected_states):
            raise ValueError("`problem.selected_states` uses one-based labels within `problem.nsts`.")
        if selected_states[0] != 1:
            raise ValueError("`problem.selected_states` must start with ground state 1.")


def load_config(path: str | Path) -> AppConfig:
    """Load the user configuration from TOML.

    A TOML file keeps the tool readable for users who are comfortable editing
    structured text but do not want to modify Python source code.
    """

    file_path = Path(path)
    data = tomllib.loads(file_path.read_text())

    problem_data = dict(data["problem"])
    problem_data.setdefault("resolution_of_identity_states", None)
    problem_data.setdefault("fit_cutoff", 0.5172311136437814)
    problem_data.setdefault("fit_epsilon", 2.0302901591898928)
    problem_data.setdefault("selected_states", None)
    execution_data = dict(data["execution"])
    # Rebuild the cavity basis by default so Lambda.dat and Sm12T.dat are
    # generated with the convention used by the frequency-dependent model.
    execution_data.setdefault("build_cavity_mode_basis", True)
    if "build_potential_modes" not in execution_data and "build_normal_modes" in execution_data:
        execution_data["build_potential_modes"] = execution_data.pop("build_normal_modes")
    execution_data.setdefault("reuse_inertial_cache", True)
    execution_data.setdefault("write_inertial_potentials", False)
    execution_data.setdefault("write_inertial_dipoles", False)
    execution_data.setdefault("potential_basis", "normal_mode")
    if execution_data["potential_basis"] not in {"normal_mode", "normal_mode_inertial", "frozen"}:
        raise ValueError(
            "`execution.potential_basis` must be `normal_mode`, "
            "`normal_mode_inertial`, or `frozen`."
        )

    paths_data = dict(data["paths"])
    paths_data.setdefault("inertial_cache_dir", None)
    paths_data.setdefault("inertial_normal_mode_potential", None)
    paths_data.setdefault("inertial_energy", None)
    paths_data.setdefault("inertial_energy_offset", None)
    paths_data.setdefault("inertial_dipoles", None)
    for key, value in paths_data.items():
        if value is not None:
            paths_data[key] = Path(value)
    postprocess_data = dict(data.get("postprocess", {}))
    for key in ("output_dir", "dipoles"):
        if postprocess_data.get(key) is not None:
            postprocess_data[key] = Path(postprocess_data[key])
    postprocess_data.setdefault("output_dir", None)
    postprocess_data.setdefault("dipoles", None)
    postprocess_data.setdefault("delta_scf", 0.0)
    postprocess_data.setdefault("oscillator_strength_mode", None)
    postprocess_data.setdefault("dipole_basis", "frozen")
    if postprocess_data["dipole_basis"] not in {"frozen", "inertial"}:
        raise ValueError("`postprocess.dipole_basis` must be `frozen` or `inertial`.")
    if postprocess_data["oscillator_strength_mode"] not in {None, "hermitian", "complex_symmetric"}:
        raise ValueError(
            "`postprocess.oscillator_strength_mode` must be either "
            "'hermitian' or 'complex_symmetric' when set."
        )
    safety_data = dict(data.get("safety", {}))
    safety_data.setdefault("max_memory_gb", 24.0)
    safety_data.setdefault("min_available_memory_gb", 4.0)
    safety_data.setdefault("memory_safety_factor", 2.5)
    safety_data.setdefault("min_free_disk_gb", 4.0)
    safety_data.setdefault("disk_safety_factor", 1.25)

    paths = PathsConfig(**paths_data)
    direct_inertial_route = (
        execution_data["potential_basis"] == "normal_mode_inertial"
        and paths.inertial_normal_mode_potential is not None
        and paths.inertial_energy is not None
    )
    if paths.ci_energy is None and not direct_inertial_route:
        raise ValueError(
            "`paths.ci_energy` is required unless the direct inertial-input "
            "route supplies both inertial inputs with "
            "`execution.potential_basis = normal_mode_inertial`."
        )

    config = AppConfig(
        paths=paths,
        problem=ProblemConfig(**problem_data),
        execution=ExecutionConfig(**execution_data),
        dielectric=DielectricConfig(**data["dielectric"]),
        postprocess=PostprocessConfig(**postprocess_data),
        safety=SafetyConfig(**safety_data),
    )
    validate_runtime_dimensions(config.problem)
    return config
