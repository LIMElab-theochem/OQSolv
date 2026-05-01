# Developer Manual

## Purpose

This project is the first Python refactor of two Mathematica workflows:

- `potential_normal_modes.m`
- `oqs-pcm.m`
- relevant postprocessing and export routines from `figures-new.nb`
- figure generation previously stored in `python_graphs/`
- upstream preprocessing logic from `lambda_sm12t.nb`
- upstream preprocessing logic from `dipoles.nb`

The current goal is operational parity at the workflow level, not yet a domain-level rewrite.

## Model Structure

The codebase should be read as one framework with two related solvation-model
families.

- PCM-based OQS:
  this branch uses state/transition electrostatic potentials defined on a
  discretized molecular cavity that follows the solute shape.
- Onsager-based OQS:
  this branch uses state/transition electric dipole moments in a spherical
  cavity model.

This distinction matters for architecture:

- the solver core should not force PCM-specific ingredients into Onsager code
- postprocessing and figure generation can often be shared once the data have
  been reduced to common table formats such as energies, oscillator strengths,
  dielectric curves, and convergence tables

The correct abstraction boundary is the coupling construction:

- shared:
  iterative diagonalization, convergence control, state tracking, frozen to
  inertial basis changes
- model-specific:
  how the non-diagonal Hamiltonian correction is assembled from either
  electrostatic potentials or dipole moments

## Project Layout

- `src/oqs_pcm/config.py`: reads the TOML configuration file.
- `src/oqs_pcm/io.py`: low-level text input and output helpers.
- `src/oqs_pcm/potential.py`: PCM-specific loaders and transformations for state/transition electrostatic-potential tensors.
- `src/oqs_pcm/dielectric.py`: dielectric model and direct PCM-kernel construction from `Lambda.dat`.
- `src/oqs_pcm/solver.py`: OQS-PCM solver variants.
- `src/oqs_pcm/models/`: model-specific coupling builders over the shared solver engine.
  `pcm.py` and `onsager.py` are the intended symmetry point.
- `src/oqs_pcm/postprocess/`: loaders and reusable figure-ready dataset structures.
- `src/oqs_pcm/figures/`: reusable matplotlib figure builders that replace the ad hoc scripts in `python_graphs/`.
- `src/oqs_pcm/workflow.py`: orchestrates the integrated workflow.
- `src/oqs_pcm/cli.py`: command-line entry point.
- `example_config.toml`: template configuration file.

## Upstream Preprocessing Stages

Two Mathematica notebooks in the workspace are not alternative solver branches.
They are upstream data-preparation stages that feed the same OQS-Continuum
workflow and therefore belong in the long-term Python integration plan.

### `lambda_sm12t.nb`

This notebook is part of the PCM preprocessing path.

- It loads cavity-operator data from `mat_SD.inp`.
- It infers `nts` from that file.
- It reshapes the Calderon `S` and `D` matrices.
- It combines them with cavity geometry and tessera data.
- It computes the normal-mode transformation matrix `Sm12T`.
- It computes the cavity eigenvalue vector written to `Lambda.dat`.

Scientifically, this notebook prepares PCM cavity-mode ingredients used before
the actual OQS state-specific diagonalizations begin.

The Python implementation applies one deterministic eigenvector sign convention
after diagonalizing the symmetric cavity-mode matrix.  Each full eigenvector is
multiplied by one overall sign so that its largest-magnitude component is
positive.  This is only a reproducibility convention: it must be applied to the
entire eigenvector, never component-by-component across different eigenvectors.
NumPy stores eigenvectors as columns, so the sign convention is applied to
columns before transposing to the row-oriented matrix corresponding to
Mathematica's `eigen[[2]]`.

### `dipoles.nb`

This notebook is part of the general dipole-observable preprocessing path.

- It starts from a limited dipole dataset, historically obtained from a
  GAMESS-US workflow in which only ground-state to excited-state transition
  dipoles were directly available.
- It loads additional one-electron dipole integrals and orbital information.
- It combines them with TDDFT amplitudes and Slater-Condon-style contractions.
- It adds the nuclear dipole contribution.
- It reconstructs a fuller state/transition dipole tensor and exports the
  completed `ci_mut.inp`-style dataset.

Scientifically, this notebook is relevant to the Onsager coupling model and to
OQS-PCM/OQS-Onsager postprocessing routines such as oscillator strengths,
because those quantities depend on state/transition electric dipole moments in
the appropriate basis.  The limited `ci_mut.awk` extraction from a GAMESS-US log
is not the source of the completed tensor; it is a diagnostic reference for the
ground-state to excited-state transition dipoles.

## Naming Convention

The code now follows a naming convention meant to stay readable for developers
coming from scientific computing rather than from Python alone.

- Scientific routines use human-readable names tied to the manuscript
  vocabulary, for example:
  `solve_state_specific_energy_iteratively`,
  `transform_coupling_tensor_to_inertial_basis`,
  `pcm_load_frozen_state_transition_potential_tensor`.
- Input/output helpers use an explicit `io_` prefix, for example:
  `io_load_excited_state_energy_vector`,
  `io_write_pcm_normal_mode_potential_tensor`,
  `io_load_onsager_transition_dipole_tensor`.
- Model-specific code uses explicit prefixes or module placement:
  `pcm_*` for PCM-specific routines,
  `Onsager*`/`onsager_*` for Onsager-specific routines.
- Backward-compatible aliases are temporarily kept in a few modules while the
  refactor propagates. New code should prefer the explicit names.

## State-Count Semantics

To avoid the ambiguity that existed in the Mathematica notebooks, the Python
refactor uses one meaning for each state-count variable.

- `problem.nsts`: total number of electronic states in the full space,
  including the ground state.
- `n_excited_states_full`: derived quantity equal to `problem.nsts - 1`.
- `problem.diagonalization_states`: Hamiltonian dimension used in the
  state-specific nonlinear diagonalizations. This is a running parameter.
- `problem.resolution_of_identity_states`: number of states retained in the RI
  / sum-over-states space. This is a running parameter.
- `problem.corrected_states`: number of target solvated states actually
  corrected and returned. This is a running parameter.

Mandatory relations enforced by the code:

- `1 <= diagonalization_states <= nsts`
- `1 <= resolution_of_identity_states <= nsts`
- `resolution_of_identity_states >= diagonalization_states`
- `1 <= corrected_states <= diagonalization_states`

Interpretation:

- the RI space cannot be smaller than the Hamiltonian dimension because the
  effective OQS Hamiltonian uses the RI states as intermediate states
- the number of corrected solvated states cannot exceed the Hamiltonian
  dimension because one cannot correct more target states than those explicitly
  represented in the diagonalization

## Scientific Semantics

The manuscript and SI define the scientific vocabulary that should be mirrored
in the code, comments, and manuals.

- `frozen-solvent basis`:
  eigenstates of the frozen-solvent Hamiltonian used as the initial electronic
  basis obtained from external electronic-structure calculations.
- `inertial-solvent basis`:
  eigenstates of the Hamiltonian that includes the inertial part of solvent
  polarization. OQS corrections are expressed in this basis.
- `dynamic solvent response`:
  fast solvent electronic response described through the filtered,
  frequency-dependent dielectric function.
- `state-specific Hamiltonian`:
  the effective OQS Hamiltonian depends on the target solvated state through
  the state energy being corrected, so each state requires its own nonlinear
  diagonalization.
- `OQS-Continuum`:
  umbrella term for the common algorithmic structure shared by OQS-PCM and
  OQS-Onsager.

Current status of the layering:

- shared iterative solver engine: implemented
- PCM coupling backend: implemented in the existing workflow
- Onsager coupling backend: partially integrated through loader and kernel utilities
- Onsager-relevant figure pathway: integrated at the dataset/figure level
- upstream preprocessing semantics for PCM cavity modes and Onsager dipoles:
  documented and mapped, but not yet fully ported as runnable Python commands

## Environment Setup

1. Create a virtual environment.
2. Install the package in editable mode.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the Tool

Normal-mode transformation only:

```bash
oqs-pcm example_config.toml --mode normal-modes
```

OQS-PCM solver only:

```bash
oqs-pcm example_config.toml --mode solve
```

Complete workflow:

```bash
oqs-pcm run example_config.toml --mode all
```

Render integrated figures from figure-ready text tables:

```bash
oqs-pcm figures all /path/to/figure_tables /path/to/output_figures
```

Generate the figure-ready tables from raw OQSolv solver outputs:

```bash
oqs-pcm postprocess example_config.toml
```

This command is the refactored replacement for the relevant export cells in
`figures-new.nb`. It intentionally stops at plain-text tables so that numerical
postprocessing and matplotlib rendering remain separately testable.

Estimate the heavy dense-array memory footprint before running a large PCM
potential transformation:

```bash
oqs-pcm estimate-memory example_config.toml
```

The estimator reads `nts` from `ci_pot.inp` and combines it with `problem.nsts`
to estimate the dense `nsts x nsts x nts` potential tensors plus the cavity
matrix. The estimate is multiplied by `[safety].memory_safety_factor` to account
for NumPy/BLAS temporaries.

Postprocessing semantics:

- `ww` is computed by `compute_pcm_reference_polarization_shift()` as the
  corrected negative half-contraction `-.5 (Kerneld V).V` in the PCM
  normal-mode basis. The original Mathematica source used the opposite sign in
  the postprocessing notebook; OQSolv intentionally fixes that convention.
- `ww` is not exposed in TOML or the CLI because it must remain internally
  coherent with the PCM kernel and normal-mode potential of the run being
  postprocessed.
- `deltascf` is not inferred by OQSolv because it comes from external SCF
  total-energy bookkeeping; it is supplied as `[postprocess].delta_scf`.
- The notebook convention for `deltascf` is
  `<Psi|H(0) + V/2|Psi>_solvent - <Psi|H(0)|Psi>_gas`, in Hartree. This value
  belongs to the upstream frozen-solvent/in-vacuo electronic-structure data,
  not to the OQS diagonalization layer.
- absolute level-energy exports receive `energy + ww + deltascf`.
- spectra exports use transition energies `energy[k] - energy[0]`, so a
  constant `ww` shift cancels from the x-axis, matching the Mathematica logic.
- oscillator strengths are computed from state/transition dipoles transformed
  into the inertial basis when `solve_inertial_basis = true`.

## Execution Semantics

The execution controls are intentionally separated because they govern
different intermediate stages of the OQS-Continuum workflow.

- `build_potential_modes`:
  rebuild the PCM normal-mode potential tensor from `ci_pot.inp` and
  `Sm12T.dat`.
- `solve_inertial_basis`:
  use the inertial reference problem before running downstream variants such as
  `ss_pol`, `full`, `bo`, and `cutoff`.
- `reuse_inertial_cache`:
  when `solve_inertial_basis = true`, reuse cached inertial energies and
  inertial-basis states if they already exist.

Important interaction rule:

- normal-mode reuse and inertial-cache reuse are independent in general
- however, if `build_potential_modes = true` and `solve_inertial_basis = true`,
  the inertial reference problem must be recomputed because the transformed
  coupling tensor has changed

This separation also keeps open the scientifically relevant possibility of
running the downstream solvers in a basis other than the inertial one, for
example the original frozen or gas-phase-like basis in weakly polar solvents.
The same basis-choice rule should also apply to `fit_cutoff` when that variant
is added to the Python implementation.

## Memory Safety

The highest-risk step is PCM potential processing:

- loading the frozen state/transition electrostatic potential tensor
- adding nuclear diagonal contributions
- transforming the tensor to the PCM normal-mode basis
- writing `ci_pot_normal_mode.inp`

The code now guards this path in two ways:

- `estimate_pcm_potential_mode_memory()` computes a conservative peak estimate
  before allocating the large arrays.
- `apply_process_memory_limit()` installs a process address-space limit from
  `[safety].max_memory_gb` before `oqs-pcm run` starts.

The default TOML safety values are conservative for a 32 GB workstation:

```toml
[safety]
max_memory_gb = 24.0
min_available_memory_gb = 4.0
memory_safety_factor = 2.5
```

For cluster jobs, increase `max_memory_gb` only when the requested node memory
comfortably exceeds the estimator output.

## Input File Formats

The current implementation relies on plain-text tables exported by the
electronic-structure workflow. Their layout should be treated as part of the
public interface of the first refactor.

### `ci_energy.inp`

- Whitespace-separated numeric table.
- The last column contains excitation energies in eV.
- The Python loader converts that last column to Hartree and prepends a
  ground-state reference energy equal to zero.
- The number of electronic states can therefore be inferred as:
  `nsts = number_of_rows_in_ci_energy_file + 1`.

### PCM `ci_pot.inp`

This file stores state-specific and transition electrostatic potentials on the
PCM cavity tesserae.

- First row:
  first column is the number of cavity tesserae, `nts`.
- The next relevant block contains the ground-state electrostatic potential and
  the nuclear contribution sampled on those tesserae.
- The remaining rows are grouped in blocks of length `nts + 1`.
- For each block:
  - the first row stores the pair of electronic-state indices `(i, j)`
  - the next `nts` rows store the tessera-resolved electrostatic potential for
    that state pair
- In the current loader, the number of electronic states may be inferred from
  the largest state index appearing in these pair headers.

### PCM normal-mode potential file

The transformed file written by the Python code keeps the Mathematica layout.

- First line:
  number of tesserae, `nts`.
- Then, for each upper-triangular state pair `(i, j)`:
  - one header line containing `i j`
  - `nts` lines with one potential value per tessera

### Completed `ci_mut.inp`

This file stores general state-specific and transition electric dipole moments.
It is consumed by the OQS-Onsager coupling backend and by oscillator-strength
postprocessing for OQS-PCM and OQS-Onsager.

- Canonical OQSolv rows use `States i and j mux muy muz`.
- State labels are one-based total-state labels:
  label 1 is the ground state, label 2 is the first excited state.
- The current loader constructs a symmetric tensor
  `mu[state_i, state_j, cartesian_component]`.
- Internally, Python uses zero-based total-state indices:
  internal index 0 is the ground state, internal index 1 is the first excited state.
- Historically, the raw electronic-structure output may have contained only a
  limited GS-ES subset of these dipoles. In the original workflow,
  `dipoles.nb` completed the missing state/transition dipole information before
  the data were used by Onsager routines or oscillator-strength postprocessing.

### `Sm12T.dat`

- Numeric matrix used to transform PCM electrostatic-potential tensors from the
  original cavity representation to the normal-mode representation.
- In the original workflow, this file was produced upstream by
  `lambda_sm12t.nb` from cavity geometry and Calderon matrices.

### `Lambda.dat`

- Numeric table.
- The first column contains the PCM cavity eigenvalues used to assemble the PCM
  response kernel.
- In the original workflow, this file was produced upstream by
  `lambda_sm12t.nb`.

## Configuration Notes

- `problem.nsts` is still present in the configuration for explicitness and for
  compatibility with the current code paths.
- Scientifically, it should match the state count implied by `ci_energy.inp`
  and by the largest state index present in the coupling file
  (`ci_pot.inp` for PCM, `ci_mut*.inp` for Onsager).
- A future cleanup can infer `nsts` automatically and treat the configuration
  value as optional or as a consistency check only.

## Running Tests

The project currently includes a minimal smoke test suite. These tests are meant
to check package wiring and basic file and parser behavior before any scientific
validation is attempted.

Run the suite with:

```bash
python -m pytest tests
```

Current smoke-test coverage:

- TOML configuration loading
- CLI argument parsing
- creation of plain-text output files
- basic dielectric-model evaluation
- figure-dataset loading

## Regression Testsuite

The directory `testsuite/hcl_water_pcm/` contains a validated HCl-in-water
OQS-PCM reference fixture. It is intentionally larger and slower than the smoke
tests because it protects the scientific workflow against numerical
regressions.

Run it from the workspace root with:

```bash
PYTHONPATH=/workspace/src /workspace/.venv/bin/python testsuite/hcl_water_pcm/validate.py
```

The validator regenerates:

- PCM normal-mode potential
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

It compares the regenerated files against `testsuite/hcl_water_pcm/reference/`.
Generated comparison outputs are written to
`testsuite/hcl_water_pcm/current_output/`, which is ignored by Git.

## Refactoring Decisions

- The original Mathematica scripts were highly procedural and repeated nearly identical eigensolver loops. The Python version centralizes that repeated logic in `run_statewise_solver()` and `iterate_state_energy()`.
- Hard-coded filesystem paths were moved into `example_config.toml`.
- The solver, postprocessing, and figure layers are now one package with one CLI.
- The relevant table-generation parts of `figures-new.nb` are now represented
  by `src/oqs_pcm/postprocess/tables.py`; the plotting layer consumes those
  tables from `src/oqs_pcm/figures/`.
- The comments are intentionally more explicit than usual Python production code because the immediate target reader is not necessarily a Python specialist.
- The old `python_graphs` scripts were used as behavior references and are being replaced by library modules under `src/oqs_pcm/figures/`.
- The Onsager notebook confirmed that repeated execution blocks differ mainly by parameter choices and data sources. That repetition should eventually collapse into parameterized Python workflows rather than notebook cell duplication.
- `lambda_sm12t.nb` and `dipoles.nb` clarified that part of the original
  Mathematica project is best viewed as upstream scientific preprocessing.
  Those stages should eventually become explicit Python preprocessing commands
  rather than remain hidden manual prerequisites.
- Terminology used consistently in the refactor:
  PCM uses state/transition electrostatic potentials; Onsager uses state/transition dipole moments.

## Known Gaps

- This first version assumes the original text input formats remain unchanged.
- Numerical validation against the Mathematica outputs has not yet been done in this workspace because the required data files are not yet wired into a runnable local configuration.
- The current code preserves the operational model variants from the Mathematica script, but it does not yet explain the scientific rationale for each approximation.
- The notebook `figures-new.nb` is larger than the subset currently integrated.
  The refactor now covers the relevant table-generation and figure-consumption
  path used by the former `python_graphs` routines; unrelated exploratory
  notebook cells remain intentionally excluded.
- `oqs-onsager-hono-water-eps-el.nb` has been inspected for architecture and figure-workflow relevance. The solver engine shared with PCM is now explicit, and Onsager-specific coupling utilities have been added, but a full end-to-end Onsager command workflow still needs to be wired and validated.

## Recommended Next Step

The next development step should be a validation pass against one trusted dataset:

1. run the Mathematica version and keep reference outputs
2. run the Python version with the same data
3. compare energies and state vectors
4. resolve indexing or parser mismatches before extending functionality
