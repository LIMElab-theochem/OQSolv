# User Manual

## What This Program Does

This tool combines two previous Mathematica steps into one Python command-line program:

- build the normal-mode potential file
- run the OQS-PCM calculations from that transformed input
- generate figures from the postprocessed text tables previously used by `python_graphs`

The broader original workflow also contains two important preprocessing stages:

- `lambda_sm12t.nb`, which prepares `Sm12T.dat` and `Lambda.dat`
- `dipoles.nb`, which completes the dipole data later used by the Onsager
  branch and by oscillator-strength postprocessing

There are two related model families in the broader workflow:

- PCM-based OQS, which uses state/transition electrostatic potentials on a molecular-shaped cavity
- Onsager-based OQS, which uses electric dipole moments with a spherical cavity

The current Python package contains:

- a shared iterative solver engine
- a working PCM-based workflow
- integrated Onsager-related figure handling
- initial Onsager coupling utilities for the ongoing full refactor

Useful terminology:

- `frozen-solvent basis`: basis produced by the external frozen-solvent
  calculation.
- `inertial-solvent basis`: basis after incorporating the inertial part of the
  solvent polarization.
- `dynamic response`: fast, frequency-dependent solvent electronic response
  entering the OQS correction.

## Before You Start

You need:

- Python 3.11 or newer
- the original input data files referenced in `example_config.toml`
- permission to read those files and write into an output directory

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Copy `example_config.toml` and edit the file paths.

Important fields:

- `paths.ci_potential`: input used to build the normal-mode potential
- `paths.normal_mode_potential`: output file for the transformed potential, and later input for the solver
- `paths.ci_energy`: CI energies file
- `paths.sm12t`: normal-mode transformation matrix
- `paths.lambda_file`: `Lambda.dat`
- `paths.output_dir`: directory for solver results

About `problem.nsts`:

- for now, you still set this explicitly in the config
- it should match the number of states implied by your input files
- in practice, it can be inferred from:
  - `ci_energy.inp`: number of rows, plus the ground state
  - `ci_pot.inp` for PCM or `ci_mut.inp` for Onsager: largest state index in
    the file

Other run parameters:

- `problem.diagonalization_states`: size of the Hamiltonian you explicitly
  diagonalize
- `problem.resolution_of_identity_states`: size of the RI space used in the sum
  over intermediate states
- `problem.corrected_states`: number of solvated states you actually correct

These must satisfy:

- `resolution_of_identity_states` cannot be smaller than
  `diagonalization_states`
- `corrected_states` cannot be larger than `diagonalization_states`
- neither can exceed `nsts`

Execution switches:

- `build_potential_modes`:
  rebuild `ci_pot_normal_mode.inp`
- `solve_inertial_basis`:
  solve and use the inertial basis before running downstream variants such as
  `ss_pol`, `full`, `bo`, and `cutoff`
- `reuse_inertial_cache`:
  if the inertial basis is requested and cached inertial files already exist,
  reuse them instead of recomputing

Practical rule:

- you may skip the normal-mode rebuild and still recompute the inertial basis
- you may reuse the inertial basis only when the normal-mode tensor is being
  reused too
- the same basis-choice rule should also apply to `fit_cutoff` when that
  variant is enabled

## Input Files

Minimal format summary:

- `ci_energy.inp`:
  whitespace-separated numeric table whose last column contains excitation
  energies in eV
- PCM `ci_pot.inp`:
  text file containing state/transition electrostatic potentials on PCM cavity
  tesserae
- Onsager `ci_mut.inp`:
  text file containing state/transition electric dipole moments, with the last
  three columns giving the Cartesian components
- `Sm12T.dat`:
  numeric transformation matrix for the PCM normal-mode step
- `Lambda.dat`:
  PCM cavity-eigenvalue table

Practical notes:

- `ci_energy.inp` contains excitation energies only, so the program treats the
  ground-state reference energy separately.
- In the original workflow, the completed `ci_mut.inp` used by Onsager-style
  calculations was prepared upstream from a more limited dipole dataset.
- In the original workflow, `Sm12T.dat` and `Lambda.dat` were prepared
  upstream from cavity geometry and cavity matrices.

The developer manual contains a more detailed description of these file formats.

## Commands

Build only the normal-mode potential:

```bash
oqs-pcm my_config.toml --mode normal-modes
```

Run only the OQS-PCM solver:

```bash
oqs-pcm my_config.toml --mode solve
```

Run both steps:

```bash
oqs-pcm run my_config.toml --mode all
```

Render all integrated figures from a directory of figure-ready text files:

```bash
oqs-pcm figures all /path/to/figure_tables /path/to/output_figures
```

Render one figure family only:

```bash
oqs-pcm figures levels /path/to/figure_tables /path/to/output_figures
```

## Output

The solver writes one directory per model variant inside `output_dir`.

Each result directory contains:

- `newenergies.dat`
- `newstates.dat`

## Notes

- The current version is designed to stay close to the original Mathematica workflow.
- The code contains extra comments so that users with limited Python experience can still inspect the logic.
- Scientific interpretation of the output is not yet documented here. This manual currently covers only operation and usage.
- The integrated figure commands currently assume the same plain-text table names that were previously expected by the plotting scripts in `python_graphs`.
- The Onsager notebook is partly reflected in the integrated figure workflow, and the Python code now includes Onsager-specific coupling utilities, but the full Onsager computational branch is not yet exposed as a standalone Python solver command.
- The preprocessing notebooks `lambda_sm12t.nb` and `dipoles.nb` are now part
  of the documented workflow, but their full Python command-line replacements
  are still pending.
