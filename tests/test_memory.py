from __future__ import annotations

from pathlib import Path

import pytest

from oqs_pcm.memory import (
    enforce_disk_estimate,
    enforce_memory_estimate,
    estimate_pcm_normal_mode_text_disk,
    estimate_pcm_potential_mode_memory,
    read_pcm_site_count,
)
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


def test_estimate_pcm_normal_mode_text_disk_uses_state_pairs_and_sites(tmp_path: Path) -> None:
    ci_pot = tmp_path / "ci_pot.inp"
    ci_pot.write_text("10\n", encoding="utf-8")
    config = _config(
        build_potential_modes=True,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )
    config.paths.ci_potential = ci_pot
    config.paths.normal_mode_potential = tmp_path / "ci_pot_normal_mode.inp"
    config.problem.nsts = 4
    config.safety.disk_safety_factor = 1.0

    estimate = estimate_pcm_normal_mode_text_disk(config)

    assert estimate.details["state_pairs"] == 10
    assert estimate.details["value_lines"] == 100
    assert estimate.required_gb > 0


def test_enforce_disk_estimate_blocks_when_projected_free_is_too_low(tmp_path: Path) -> None:
    ci_pot = tmp_path / "ci_pot.inp"
    ci_pot.write_text("10\n", encoding="utf-8")
    config = _config(
        build_potential_modes=True,
        solve_inertial_basis=True,
        reuse_inertial_cache=True,
    )
    config.paths.ci_potential = ci_pot
    config.paths.normal_mode_potential = tmp_path / "ci_pot_normal_mode.inp"
    config.safety.min_free_disk_gb = 10**9
    estimate = estimate_pcm_normal_mode_text_disk(config)

    with pytest.raises(OSError):
        enforce_disk_estimate(estimate)
