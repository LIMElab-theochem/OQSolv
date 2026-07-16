from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oqs_pcm.solver import SolverConfig, solve_state_specific_family


def test_state_specific_family_writes_incremental_checkpoints(tmp_path: Path) -> None:
    cfg = SolverConfig(
        nsts=2,
        diagonalization_states=2,
        corrected_states=2,
        tolerance=1e-8,
        mixing_window=2,
        cutoff=0.0,
        write_text_exports=True,
        output_dir=tmp_path,
    )

    energies, states = solve_state_specific_family(
        label="full",
        base_h=np.diag([0.0, 1.0]),
        potential=np.zeros((2, 2, 1)),
        energies=np.array([0.0, 1.0]),
        kernel_real=np.zeros(1),
        kernel_imag=None,
        cfg=cfg,
        first_iteration_only=True,
    )

    result_dir = tmp_path / "full"
    progress = json.loads((result_dir / "progress.json").read_text(encoding="utf-8"))

    assert progress == {"completed_states": 2, "total_states": 2, "complete": True}
    assert np.allclose(np.loadtxt(result_dir / "newenergies_partial.dat", dtype=complex), energies)
    assert np.allclose(np.loadtxt(result_dir / "newstates_partial.dat", dtype=complex), states)
    assert np.allclose(np.loadtxt(result_dir / "newenergies.dat", dtype=complex), energies)
    assert np.allclose(np.loadtxt(result_dir / "newstates.dat", dtype=complex), states)
