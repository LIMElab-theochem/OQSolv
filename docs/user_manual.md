# User Manual

## What This Program Does

OQSolv is a command-line tool for running OQS-PCM(omega) calculations and the
related approximation schemes supported by the package.

The program can:

- build the PCM normal-mode potential used by the solver
- run the configured OQS-PCM and approximation variants
- postprocess raw solver outputs into figure-ready tables
- render figures from those tables

Useful terminology:

- `frozen-solvent basis`: basis obtained from the external frozen-solvent
  calculation
- `inertial-solvent basis`: basis after the inertial part of the solvent
  polarization has been incorporated
- `dynamic response`: fast, frequency-dependent solvent electronic response
  used in the OQS correction

## Before You Start

You need:

- Python 3.11 or newer
- the input files referenced in `example_config.toml`
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

- you set this explicitly in the config
- it should match the number of states implied by your input files
- it includes the ground state, so the number of excited states is
  `problem.nsts - 1`

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
  `ss_pol`, `full`, `bo`, `cutoff`, and `fit_cutoff`
- `reuse_inertial_cache`:
  if the inertial basis is requested and cached inertial files already exist,
  reuse them instead of recomputing

Practical rule:

- you may skip the normal-mode rebuild and still recompute the inertial basis
- you may reuse the inertial basis only when the normal-mode tensor is being
  reused too
- `fit_cutoff` follows the same basis-choice rule as `cutoff`; it differs only
  by using `[problem].fit_epsilon` and `[problem].fit_cutoff`

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
- `ci_pot.inp` contains state and transition electrostatic potentials on the
  PCM cavity tesserae.
- `ci_mut.inp` contains completed state and transition electric dipole
  moments.
- `Sm12T.dat` and `Lambda.dat` are the PCM cavity-mode transformation and
  eigenvalue inputs used to build the normal-mode potential.

## Commands

Build only the normal-mode potential:

```bash
oqs-pcm run my_config.toml --mode potential-modes
```

Run only the OQS-PCM solver:

```bash
oqs-pcm run my_config.toml --mode solve
```

Run both steps:

```bash
oqs-pcm run my_config.toml --mode all
```

Estimate memory and disk use before a large run:

```bash
oqs-pcm estimate-memory my_config.toml
```

For large PCM systems, run this before `--mode potential-modes` or `--mode all`.
OQSolv will also perform the same safety checks automatically before allocating
the dense normal-mode potential arrays or writing the large normal-mode
potential text file.

Render all integrated figures from a directory of figure-ready text files:

```bash
oqs-pcm figures all /path/to/figure_tables /path/to/output_figures
```

Generate the figure-ready text files from OQSolv solver outputs:

```bash
oqs-pcm postprocess my_config.toml
```

The `postprocess` command reads `newenergies.dat` and `newstates.dat` from the
solver output directories, applies the internally computed reference
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
and normal-mode potential used by the current run.

`delta_scf` follows the calculation convention:

```text
delta_scf = <Psi|H(0) + V/2|Psi>_solvent - <Psi|H(0)|Psi>_gas
```

It must be taken from the in-solvent and in-vacuo electronic-structure
calculations. OQSolv does not infer it from `newenergies.dat`.

## Resource Safety

Large PCM jobs can require several dense arrays with shape roughly
`nsts x nsts x nts`, where `nsts` is the total number of electronic states and
`nts` is the number of PCM tesserae. They can also write multi-GB
`ci_pot_normal_mode.inp` files. Configure safety limits in TOML:

```toml
[safety]
max_memory_gb = 24.0
min_available_memory_gb = 4.0
memory_safety_factor = 2.5
min_free_disk_gb = 4.0
disk_safety_factor = 1.25
```

`max_memory_gb` is also installed as a process address-space limit on supported
systems. This is meant to fail the OQSolv process before the operating system
becomes unstable. `min_free_disk_gb` prevents large text-output stages from
starting if they are expected to leave too little free disk space.

Render one figure family only:

```bash
oqs-pcm figures levels /path/to/figure_tables /path/to/output_figures
```

## Output

All text outputs are written in atomic units unless the file format naturally
stores a dimensionless quantity.

The solver writes one directory per model variant inside `output_dir`.

Each result directory contains:

- `newenergies.dat`: solvated energies for the corrected states, in atomic
  units
- `newstates.dat`: expansion coefficients of the solvated states in the chosen
  reference basis

The postprocessing command writes reduced table files such as:

- level/free-energy tables: `ensspol.txt`, `encorr1.txt`, `encorr2.txt`,
  `en.txt`, `enbo.txt`
- spectra tables: `sspol.txt`, `full.txt`, `bo.txt`, `cut.txt`, `fit.txt`
- GS solvation free energy tables: `GSsolfull.txt`, `GSsolcut.txt`,
  `GSsolfit.txt`, `GSsolbo.txt`

Interpretation of the table families:

- `en*.txt`: free energies in solution, one value per state, in atomic units
- `bo.txt`, `full.txt`, `sspol.txt`, `cut.txt`, `fit.txt`: ordered pairs
  containing excitation energy and oscillator strength for each state, in
  atomic units
- `GSsol*.txt`: two-column `GS solvation free energy` tables whose first column
  is the total number of states and whose second column is the ground-state
  solvation free energy, in atomic units

## Notes

- The current version is designed to be used in stages: build inputs, solve,
  postprocess, then render figures.
- The solver output tables are intentionally plain text so they can be inspected
  and archived easily.
- The developer manual gives the scientific and architectural rationale behind
  the implementation choices.
