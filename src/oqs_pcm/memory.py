from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource

from oqs_pcm.config import AppConfig


BYTES_PER_FLOAT64 = 8
BYTES_PER_GB = 1024**3


@dataclass(slots=True)
class MemoryEstimate:
    """Conservative memory estimate for an OQSolv execution stage."""

    stage: str
    nstates: int
    nsites: int
    required_gb: float
    limit_gb: float
    available_gb: float | None
    details: dict[str, float]


def bytes_to_gb(value: float) -> float:
    return value / BYTES_PER_GB


def gb_to_bytes(value: float) -> int:
    return int(value * BYTES_PER_GB)


def memory_available_gb() -> float | None:
    """Return currently available memory from `/proc/meminfo` when available."""

    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            kb = float(line.split()[1])
            return kb / (1024**2)
    return None


def read_pcm_site_count(path: str | Path) -> int:
    """Read `nts`, the number of PCM tesserae/sites, from a PCM potential file."""

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return int(float(stripped.split()[0].replace("D", "E").replace("d", "e")))
    raise ValueError(f"Could not infer the PCM site count from empty file: {path}")


def estimate_pcm_potential_mode_memory(config: AppConfig) -> MemoryEstimate:
    """Estimate the heavy normal-mode potential transformation memory footprint.

    The transformation path is intentionally estimated conservatively because
    NumPy may hold both input and output arrays, plus BLAS/tensordot temporaries.
    The estimate therefore starts from the dense `(nsts, nsts, nts)` tensor
    size and multiplies by the configured safety factor.
    """

    nstates = config.problem.nsts
    nsites = read_pcm_site_count(config.paths.ci_potential)
    tensor_gb = bytes_to_gb(nstates * nstates * nsites * BYTES_PER_FLOAT64)
    cavity_matrix_gb = bytes_to_gb(nsites * nsites * BYTES_PER_FLOAT64)
    vector_gb = bytes_to_gb(nsites * BYTES_PER_FLOAT64)
    raw_required_gb = (2.0 * tensor_gb) + cavity_matrix_gb + (4.0 * vector_gb)
    required_gb = raw_required_gb * config.safety.memory_safety_factor
    return MemoryEstimate(
        stage="pcm_potential_mode_transformation",
        nstates=nstates,
        nsites=nsites,
        required_gb=required_gb,
        limit_gb=config.safety.max_memory_gb,
        available_gb=memory_available_gb(),
        details={
            "single_state_site_tensor_gb": tensor_gb,
            "cavity_matrix_gb": cavity_matrix_gb,
            "raw_required_gb": raw_required_gb,
            "safety_factor": config.safety.memory_safety_factor,
        },
    )


def enforce_memory_estimate(estimate: MemoryEstimate, *, minimum_available_gb: float) -> None:
    """Raise before heavy allocation if a stage exceeds configured safety limits."""

    if estimate.required_gb > estimate.limit_gb:
        raise MemoryError(
            f"{estimate.stage} estimated peak memory is {estimate.required_gb:.2f} GB, "
            f"above configured safety.max_memory_gb={estimate.limit_gb:.2f} GB. "
            "Increase the limit only on a machine with sufficient RAM, or reduce the problem size."
        )
    if estimate.available_gb is not None and estimate.available_gb < minimum_available_gb:
        raise MemoryError(
            f"Only {estimate.available_gb:.2f} GB appears available, below "
            f"safety.min_available_memory_gb={minimum_available_gb:.2f} GB."
        )


def apply_process_memory_limit(max_memory_gb: float) -> None:
    """Install an address-space limit so the OS kills allocation before freezing."""

    if max_memory_gb <= 0:
        return
    limit_bytes = gb_to_bytes(max_memory_gb)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = limit_bytes if soft < 0 else min(soft, limit_bytes)
    new_hard = limit_bytes if hard < 0 else min(hard, limit_bytes)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))


def format_memory_estimate(estimate: MemoryEstimate) -> str:
    available = "unknown" if estimate.available_gb is None else f"{estimate.available_gb:.2f} GB"
    lines = [
        f"stage: {estimate.stage}",
        f"states: {estimate.nstates}",
        f"PCM sites/tesserae: {estimate.nsites}",
        f"estimated peak: {estimate.required_gb:.2f} GB",
        f"configured limit: {estimate.limit_gb:.2f} GB",
        f"currently available: {available}",
    ]
    for key, value in estimate.details.items():
        lines.append(f"{key}: {value:.6g}")
    return "\n".join(lines)
