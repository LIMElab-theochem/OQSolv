# HCl Water PCM Regression Fixture

This fixture stores the validated HCl-in-water OQS-PCM outputs used during the
Mathematica-to-Python validation.

It checks:

- PCM normal-mode potential generation
- inertial-basis energies and states
- `bo`
- `ss_pol`
- `cutoff`
- `first_full_nodisp_nodiss`
- `first_full_nodiss`
- `first_full`
- `full_nodisp_nodiss`
- `full_nodiss`
- `full`

The reference outputs are in `reference/`. They were generated with:

- `nsts = 172`
- `diagonalization_states = 50`
- `resolution_of_identity_states = 172`
- `corrected_states = 5`
- `solve_inertial_basis = true`

The validator expects the HCl input files to remain at the workspace root:

- `ci_pot.inp`
- `ci_energy.inp`
- `Sm12T.dat`
- `Lambda.dat`

Run from the workspace root:

```bash
PYTHONPATH=/workspace/src /workspace/.venv/bin/python testsuite/hcl_water_pcm/validate.py
```

The script writes regenerated outputs to `current_output/` and compares them
against `reference/`.
