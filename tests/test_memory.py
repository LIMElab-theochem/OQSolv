from __future__ import annotations

from pathlib import Path

import pytest

from oqs_pcm.memory import enforce_memory_estimate, estimate_pcm_potential_mode_memory, read_pcm_site_count
from tests.test_execution_semantics import _config


def test_read_pcm_site_count_reads_first_nonempty_line(tmp_path: Path) -> None:
    path = tmp_path / "ci_pot.inp"
    path.write_text("\n335\n", encoding="utf-8")

    assert read_pcm_site_count(path) == 335


def test_estimate_pcm_potential_mode_memory_uses_state_and_site_counts(tmp_path: Path) -> None:
    ci_pot = tmp_path / "ci_pot.inp"
    ci_pot.write_text("10\n", encoding="utf-8")
    config = _config(
        build_potential_modes=True,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )
    config.paths.ci_potential = ci_pot

    estimate = estimate_pcm_potential_mode_memory(config)

    assert estimate.nstates == 172
    assert estimate.nsites == 10
    assert estimate.required_gb > 0


def test_enforce_memory_estimate_blocks_above_limit(tmp_path: Path) -> None:
    ci_pot = tmp_path / "ci_pot.inp"
    ci_pot.write_text("1000\n", encoding="utf-8")
    config = _config(
        build_potential_modes=True,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )
    config.paths.ci_potential = ci_pot
    config.safety.max_memory_gb = 1e-9
    estimate = estimate_pcm_potential_mode_memory(config)

    with pytest.raises(MemoryError):
        enforce_memory_estimate(estimate, minimum_available_gb=0.0)
