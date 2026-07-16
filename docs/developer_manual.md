# Developer Manual

## Purpose

This document describes the internal scientific layering of OQSolv. It is not a
usage guide. The goal is to explain how the package is assembled, which module
owns each physical transformation, and which invariants must hold for the code
to remain scientifically consistent.

The code is organized around the OQS-Continuum workflow:

- build the solvent coupling object from cavity or dipole data
- transform the coupling into the basis used by the solver
- solve state-specific nonlinear eigenproblems in the inertial reference basis
- postprocess the resulting energies and observables
- turn the reduced data into figures and regression tables

The implementation should remain readable to scientists working in continuum
solvation, quantum chemistry, and physical chemistry. Internal structure is a
tool for preserving the physics, not an end in itself.

## Scientific Layering

OQSolv has three scientific layers.

### 1. Model layer

This layer builds the coupling object that represents the solvent response.
The model layer is where PCM and Onsager diverge.

- PCM uses cavity-based electrostatic potentials and a cavity-mode basis.
- Onsager uses state and transition electric dipoles and a spherical reaction
  field.

The model layer belongs in `src/oqs_pcm/models/` and the cavity-specific
preparation code in `src/oqs_pcm/cavity.py`, `src/oqs_pcm/potential.py`, and
`src/oqs_pcm/dielectric.py`.

### 2. Solver layer

This layer is shared by the different solvation models once the coupling
object has been built.

- the solver operates in the frozen-solvent basis and then in the inertial
  basis
- each corrected solvated state is obtained from a state-specific nonlinear
  diagonalization
- the selected root is the one that best overlaps the corresponding inertial
  reference state

This logic is implemented in `src/oqs_pcm/solver.py` and orchestrated through
`src/oqs_pcm/workflow.py`.

### 3. Output layer

This layer turns computed arrays into plain-text tables and figures.

- `src/oqs_pcm/postprocess/` prepares reduced datasets and scientific tables
- `src/oqs_pcm/figures/` renders those tables into publication figures
- `src/oqs_pcm/io.py` provides the low-level text readers and writers used by
  the whole package

The solver and postprocessing layers are separated on purpose so that the
physics can be validated independently from the final plotting step.

## Dependency Map

The package is intentionally layered so that each scientific stage depends only
on the previous one.

### Configuration and guards

`config.py`, `memory.py`, and `gamess.py` sit at the boundary with the outside
world.

- `config.py` turns TOML data into typed runtime objects and checks the state
  counting rules.
- `memory.py` estimates resource use and enforces safety fences.
- `gamess.py` converts GAMESS-US log output into the scientific input files used
  by the rest of the package.

These modules should remain simple and explicit. They are the first line of
defense against inconsistent input data.

### Scientific model construction

`cavity.py`, `potential.py`, `dielectric.py`, and `models/` build the physical
coupling object.

- `cavity.py` and `potential.py` handle the PCM cavity-mode representation and
  the normal-mode transformation.
- `dielectric.py` maps a frequency-dependent dielectric model to kernel
  factors.
- `models/pcm.py` combines cavity information, dielectric factors, and the
  normal-mode representation into the PCM coupling.
- `models/onsager.py` builds the dipole-based coupling used by the Onsager
  backend.

This layer is where model-specific mathematics belongs. The solver should not
need to know how the coupling object was assembled.

### Solver and workflow

`solver.py` and `workflow.py` are the core of the package.

- `solver.py` contains the state-specific nonlinear diagonalization routines
  and the approximation branches built on top of them.
- `workflow.py` decides which caches can be reused, when the inertial basis
  must be rebuilt, and how the model layer is handed to the solver layer.

This split is deliberate. The solver focuses on the mathematical problem; the
workflow decides when each stage must be executed and which cached objects are
still valid.

State-specific OQS branches can require many independent nonlinear solves. For
long full-space runs, `solve_state_specific_family` writes incremental
monitoring files after each completed corrected state:
`newenergies_partial.dat`, `newstates_partial.dat`, and `progress.json`. These
files deliberately live beside the final `newenergies.dat` and `newstates.dat`
but are not canonical results. They are also not restart inputs yet; adding
resume semantics should be a separate change so the solver invariants remain
easy to audit.

### Postprocessing and figures

`postprocess/` and `figures/` consume solver output and never reach back into
the solver.

- `postprocess/observables.py` transforms state/transition observables between
  bases and derives oscillator strengths.
- `postprocess/tables.py` applies the scalar shifts needed for the final
  reduced tables.
- `postprocess/datasets.py` packages those tables into figure-ready datasets.
- `figures/*.py` renders the final visual summaries.

The key rule is that figures must stay downstream of the scientific
calculation. They should not contain hidden physics.

### Shared IO

`io.py` is intentionally low level and widely reusable.

- it reads and writes the plain-text scientific files used throughout the
  package
- it should not encode model assumptions beyond the file formats themselves

## Module Responsibilities

The most important modules are:

- `config.py`: typed configuration and runtime consistency checks.
- `workflow.py`: execution order, cache reuse, and stage selection.
- `solver.py`: nonlinear OQS solvers and shared approximation logic.
- `dielectric.py`: dielectric models and kernel factors.
- `cavity.py`: PCM cavity-mode basis construction.
- `potential.py`: PCM potential loading and basis transformation.
- `dipole_algebra.py`: dipole algebra for LR-TDDFT observables.
- `dipole_completion.py`: completion of partial dipole tensors.
- `models/pcm.py`: PCM coupling construction.
- `models/onsager.py`: Onsager coupling construction.
- `postprocess/tables.py`: reduced tables and scalar shifts.
- `postprocess/observables.py`: observable basis transformations and
  oscillator strengths.
- `postprocess/datasets.py`: figure-ready dataset containers.
- `figures/*.py`: publication figures and diagnostics.
- `memory.py`: memory and disk estimates plus safety fences.
- `gamess.py`: extraction helpers for GAMESS-US outputs.
- `io.py`: shared plain-text IO.

## Physical Data Flow

The code follows a fixed scientific path.

1. External electronic-structure data provide the frozen-solvent states and the
   state/transition observable tensors.
2. PCM runs additionally provide cavity electrostatic potentials and the cavity
   transformation data needed to build normal-mode potentials.
3. The model layer builds the coupling object, either PCM-based or Onsager-
   based.
4. The workflow layer transforms the coupling into the inertial reference
   basis and stores the cached inertial representation when appropriate.
5. The solver layer iterates the state-specific nonlinear eigenproblem for each
   requested approximation branch.
6. The postprocessing layer combines the solved energies with the scalar
   reference shifts and observable transformations needed for spectra and
   convergence tables.
7. The figure layer consumes only the reduced tables.

This separation keeps the expensive dense-array work in the solver stage and
keeps the plotting layer independent of the scientific kernel construction.

### Core call chain

The practical call chain is:

`cli.py` -> `workflow.py` -> `models/` + `cavity.py` + `potential.py` +
`dielectric.py` -> `solver.py` -> `postprocess/` -> `figures/`

The chain should not be short-circuited unless a new command is intentionally
introduced for a distinct scientific stage.

## PCM and Onsager Division of Labor

PCM and Onsager share the same OQS-Continuum solver logic but differ in how the
coupling object is built.

- PCM couples through electrostatic potentials on the discretized cavity.
- Onsager couples through dipole moments in a spherical-cavity reaction-field
  picture.

After the coupling has been built, both models use the same solver concepts:

- inertial basis construction
- root tracking by overlap
- state-specific diagonalization
- approximation branches such as Born-Oppenheimer, self-consistent limit, and
  cutoff variants

The developer rule is simple: keep the coupling builders model-specific, and
keep the solver and postprocessing logic as shared as possible.

## Solver Branches

The following branches are intentionally implemented as variations of the same
scientific core:

- `ss_pol`: static self-polarization branch
- `full`: full frequency-dependent OQS-PCM
- `full_nodiss`: full model without dissipation
- `full_nodisp_nodiss`: full model without dispersion or dissipation
- `first_full`: first-iteration variant of the full model
- `first_full_nodiss`: first-iteration variant without dissipation
- `first_full_nodisp_nodiss`: first-iteration variant without dispersion or
  dissipation
- `bo`: Born-Oppenheimer solvation approximation
- `cutoff`: cutoff approximation
- `fit_cutoff`: cutoff-like approximation with fitted parameters

These branches should reuse the same infrastructure whenever the mathematical
form is the same. New branch-specific code is justified only when the coupling
construction or the iteration rule genuinely differs.

## State Semantics

The package distinguishes three related counts:

- `problem.nsts`: total number of electronic states, including the ground
  state
- `n_excited_states_full`: derived quantity equal to `problem.nsts - 1`
- `problem.diagonalization_states`, `problem.resolution_of_identity_states`,
  and `problem.corrected_states`: running parameters for a particular solve

The running parameters have different roles:

- the diagonalization dimension is the size of the explicit Hamiltonian
- the RI dimension is the size of the intermediate-state space used in the
  OQS correction
- the corrected-state count is the number of target solvated states returned by
  the solver

The code enforces the following relations:

- `1 <= problem.diagonalization_states <= problem.nsts`
- `1 <= problem.resolution_of_identity_states <= problem.nsts`
- `problem.resolution_of_identity_states >= problem.diagonalization_states`
- `1 <= problem.corrected_states <= problem.diagonalization_states`

These checks are not bookkeeping noise. They protect the interpretation of the
OQS Hamiltonian and prevent inconsistent RI/Hamiltonian dimensions from being
silently accepted.

## Naming and Scientific Vocabulary

The package uses names that reflect the physical role of each object.

- input/output helpers use the `io_` prefix
- cavity and potential routines keep the PCM terminology visible
- dipole utilities use the `dipole_` prefix for algebra and completion
- model-specific code lives under `models/`

The preferred scientific vocabulary is:

- `frozen-solvent basis`: starting electronic basis from the external
  electronic-structure calculation
- `inertial-solvent basis`: basis after the inertial part of the solvent
  response has been incorporated
- `dynamic solvent response`: fast frequency-dependent solvent contribution
- `state-specific Hamiltonian`: the effective Hamiltonian for one target
  solvated state
- `free energy in solution`: scalar level output written by the energy tables
- `oscillator strength`: the transition-intensity output written by the
  spectra tables

This terminology should appear in comments and docstrings unless a lower-level
implementation detail forces a more technical phrasing.

## Postprocessing Semantics

The postprocessing layer transforms solver outputs into reduced tables.

- `ww` is computed internally from the PCM normal-mode data. It is not a user
  parameter because it must remain coherent with the current kernel and basis.
- `delta_scf` is external bookkeeping from the electronic-structure side and is
  supplied by configuration.
- level-energy tables receive the absolute solution energy plus the scalar
  shifts relevant to that run.
- spectra tables use transition energies and oscillator strengths, so any
  constant scalar shift cancels from the horizontal axis.
- oscillator strengths are branch-aware.  Hermitian branches use the usual
  Euclidean transition-dipole norm.  The dissipative full OQS-PCM branches use
  the complex-symmetric expression based on the c-product:
  `f_i = 2/3 Re[(E_i - E_0) sum_alpha mu_0i,alpha^2 / ((c_0^T c_0)(c_i^T c_i))]`.
  No complex conjugation appears in this normalization because the left
  eigenvector of a complex-symmetric Hamiltonian is represented by the
  transpose of the right eigenvector.
- ground-state solvation free energy tables are convergence tables for the
  scalar ground-state correction.

The practical separation is important: solver outputs remain raw scientific
results, while the postprocessing layer applies the presentation-level scalar
shifts needed for the final tables.

The user-facing `postprocess.oscillator_strength_mode` key accepts only
`hermitian` or `complex_symmetric`.  If it is omitted, `postprocess/tables.py`
chooses the branch default internally: complex-symmetric for dissipative full
OQS-PCM spectra and Hermitian for BO-sol, SS-pol, cutoff, and fit-cutoff.  Do
not expose an `auto` value in TOML; the absence of the key is the automatic
mode.

## Resource Safety

Large PCM jobs can allocate dense arrays with shape roughly
`nsts x nsts x nts` and can also write large normal-mode potential files. The
`memory.py` module handles:

- memory estimates before the heavy potential transformation
- disk estimates before large text output
- process memory limits where the platform supports them

The workflow layer uses these checks so that failures happen before the
machine or container becomes unstable.

## Extending the Code

When adding a new scientific feature, start by asking which layer it belongs to.

- If it changes the coupling object, it belongs in the model layer.
- If it changes root tracking or the nonlinear solve, it belongs in the solver
  layer.
- If it changes the final tables or plots, it belongs in the output layer.

Prefer reusing shared solver and postprocessing code over creating new branches
unless the mathematics truly differs. That keeps the PCM and Onsager
implementations aligned and reduces the risk of divergence between equivalent
operations.

When a change affects both PCM and Onsager, prefer factoring the common logic
upward and keeping only the coupling-builder differences in the model layer.
When a change affects only figures or tables, keep it out of the solver unless
it genuinely changes the scientific result.

## Testing

The repository includes unit tests and regression checks. Developer work should
be validated with the test suite before merge. The project also keeps
regression data for the validated scientific cases so that changes in the
solver, postprocessing, or figure generation can be checked against known
outputs.

## Environment Setup

The developer environment is the same editable Python environment used for the
project itself.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Execution Sketch

The stage ordering is the main operational idea to keep in mind when changing
the code:

1. configuration and safety checks
2. coupling construction
3. optional PCM normal-mode reconstruction
4. inertial-basis solve
5. branch-specific state-specific solves
6. postprocessing of energies and observables
7. figure-table generation and plotting

The only stage that should control the rest of the workflow is the model and
cache state. A change in the coupling object must invalidate any cached inertial
representation derived from it.

## Validation and Regression

The project keeps two validation tiers.

- Smoke tests check package wiring, parsing, and basic numerical helpers.
- Regression fixtures check that the scientific pipeline still reproduces the
  validated reference datasets for representative solvation cases.

When extending the code, prefer keeping the regression fixtures narrow but
scientifically meaningful. They should exercise the coupling layer, the solver
layer, and the postprocessing layer without requiring every run to be
recomputed during ordinary development.

## Maintenance Notes

The most important maintenance rule is to preserve the separation between
scientific layers.

- Do not push model-specific assumptions into the solver unless the
  mathematics demands it.
- Do not let plotting code become a hidden continuation of the solver.
- Do not encode file-format details in multiple places unless the format really
  differs by layer.

When a change affects both PCM and Onsager, consolidate the shared behavior at
the solver or postprocessing layer and keep only the coupling construction
separate.

## Environment Setup

The developer environment is the same editable Python environment used for the
project itself.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```
