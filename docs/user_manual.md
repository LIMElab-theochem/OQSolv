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
- Completed `ci_mut.inp`:
  text file containing general state/transition electric dipole moments.  It is
  used by OQS-Onsager coupling and by oscillator-strength postprocessing for
  OQS-PCM and OQS-Onsager.
- `Sm12T.dat`:
  numeric transformation matrix for the PCM normal-mode step
- `Lambda.dat`:
  PCM cavity-eigenvalue table

Practical notes:

- `ci_energy.inp` contains excitation energies only, so the program treats the
  ground-state reference energy separately.
- In the original workflow, `dipoles.nb` prepared the completed `ci_mut.inp`
  from MO coefficients, dipole integrals, TDDFT amplitudes, and nuclear
  geometry.  The limited GAMESS log extraction of GS-ES dipoles is only a
  reference check, not a required input to that reconstruction.
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

Estimate memory before a large run:

```bash
oqs-pcm estimate-memory my_config.toml
```

For large PCM systems, run this before `--mode potential-modes` or `--mode all`.
OQSolv will also perform the same safety check automatically before allocating
the dense normal-mode potential arrays.

Render all integrated figures from a directory of figure-ready text files:

```bash
oqs-pcm figures all /path/to/figure_tables /path/to/output_figures
```

Generate the figure-ready text files from OQSolv solver outputs:

```bash
oqs-pcm postprocess my_config.toml
```

The `postprocess` command is the Python replacement for the relevant export
logic in `figures-new.nb`. It reads `newenergies.dat` and `newstates.dat` from
the solver output directories, applies the internally computed reference
polarization shift `ww`, applies the external `delta_scf` shift from the TOML
file, and writes the reduced text tables consumed by the figure commands.

The corresponding TOML section is:

```toml
[postprocess]
output_dir = "./figure_tables"
dipoles = "ci_mut.inp"
delta_scf = -0.0063860537
```

`ww` is not a user option because it must remain coherent with the PCM kernel
and normal-mode potential used by the current OQSolv run.

`delta_scf` follows the Mathematica notebook convention:

```text
delta_scf = <Psi|H(0) + V/2|Psi>_solvent - <Psi|H(0)|Psi>_gas
```

It must be taken from the upstream in-solvent and in-vacuo electronic-structure
calculations. OQSolv does not infer it from `newenergies.dat`.

## Memory Safety

Large PCM jobs can require several dense arrays with shape roughly
`nsts x nsts x nts`, where `nsts` is the total number of electronic states and
`nts` is the number of PCM tesserae. Configure safety limits in TOML:

```toml
[safety]
max_memory_gb = 24.0
min_available_memory_gb = 4.0
memory_safety_factor = 2.5
```

`max_memory_gb` is also installed as a process address-space limit on supported
systems. This is meant to fail the OQSolv process before the operating system
becomes unstable.

Render one figure family only:

```bash
oqs-pcm figures levels /path/to/figure_tables /path/to/output_figures
```

## Output

The solver writes one directory per model variant inside `output_dir`.

Each result directory contains:

- `newenergies.dat`
- `newstates.dat`

The postprocessing command writes reduced table files such as:

- absolute level-energy tables: `ensspol.txt`, `encorr1.txt`, `encorr2.txt`, `en.txt`, `enbo.txt`
- spectra tables: `sspol.txt`, `full.txt`, `bo.txt`, `cut.txt`, `fit.txt`
- optional convergence tables: `GSsolfull.txt`, `GSsolcut.txt`, `GSsolfit.txt`, `GSsolbo.txt`

## Notes

- The current version is designed to stay close to the original Mathematica workflow.
- The code contains extra comments so that users with limited Python experience can still inspect the logic.
- Scientific interpretation of the output is not yet documented here. This manual currently covers only operation and usage.
- The integrated figure commands currently assume the same plain-text table names that were previously expected by the plotting scripts in `python_graphs`.
- The Onsager notebook is partly reflected in the integrated figure workflow, and the Python code now includes Onsager-specific coupling utilities, but the full Onsager computational branch is not yet exposed as a standalone Python solver command.
- The preprocessing notebooks `lambda_sm12t.nb` and `dipoles.nb` are now part
  of the documented workflow, but their full Python command-line replacements
  are still pending.
