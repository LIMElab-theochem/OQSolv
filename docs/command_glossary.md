# OQSolv Command Glossary

All commands below use the installed entry point. From a source checkout,
replace `oqs-pcm` with `python -m oqs_pcm.cli`.

## General help

```bash
oqs-pcm --help
oqs-pcm run --help
oqs-pcm estimate-memory --help
oqs-pcm figures --help
oqs-pcm postprocess --help
oqs-pcm prep-gamess --help
oqs-pcm xdip --help
```

## `run`

```bash
oqs-pcm run CONFIG.toml
```

Run the configured workflow; the default mode is `all`.

```bash
oqs-pcm run CONFIG.toml --mode cavity-basis
```

Build `Lambda.dat` and `Sm12T.dat` from `mat_SD.inp` and `cavity.inp`.

```bash
oqs-pcm run CONFIG.toml --mode potential-modes
oqs-pcm run CONFIG.toml --mode normal-modes
```

Build the normal-mode potential; both mode names are accepted.

```bash
oqs-pcm run CONFIG.toml --mode inertial
```

Build/load the normal-mode potential and solve only the complete inertial reference problem.

```bash
oqs-pcm run CONFIG.toml --mode solve
```

Run the configured OQS-PCM solver variants without preprocessing.

```bash
oqs-pcm run CONFIG.toml --mode all
```

Build requested inputs and run the configured solver in one invocation.

### Solver variants

Append `--variants` to `run` to select one or more variants:

```bash
oqs-pcm run CONFIG.toml --mode solve --variants ss_pol full_nodisp_nodiss full_nodiss full first_full first_full_nodisp_nodiss first_full_nodiss bo cutoff fit_cutoff
```

Variants are: `ss_pol` (SS-pol), `full_nodisp_nodiss`, `full_nodiss`, `full`,
`first_full`, `first_full_nodisp_nodiss`, `first_full_nodiss`, `bo` (BO-sol),
`cutoff`, and `fit_cutoff`.

### `run` overrides

```bash
oqs-pcm run CONFIG.toml --output-dir DIR --normal-mode-potential FILE --inertial-normal-mode-potential FILE --sm12t FILE --lambda-file FILE --dipoles FILE
```

Override paths without editing the TOML.

```bash
oqs-pcm run CONFIG.toml --selected-states 1 5 --diagonalization-states 5 --resolution-of-identity-states 5 --corrected-states 5
```

Override selected one-based states and the three solver-space sizes.

```bash
oqs-pcm run CONFIG.toml --potential-basis frozen
oqs-pcm run CONFIG.toml --potential-basis normal_mode
oqs-pcm run CONFIG.toml --potential-basis normal_mode_inertial
```

Choose frozen, generated normal-mode, or already transformed inertial normal-mode potentials.

Each execution switch accepts both forms, `--OPTION` and `--no-OPTION`:
`build-cavity-mode-basis`, `build-potential-modes`, `solve-inertial-basis`,
`reuse-inertial-cache`, `write-inertial-potentials`, and
`write-inertial-dipoles`.

## `estimate-memory`

```bash
oqs-pcm estimate-memory CONFIG.toml
```

Estimate dense-array memory and normal-mode text-output disk requirements without running.

## `postprocess`

```bash
oqs-pcm postprocess CONFIG.toml
```

Generate level, spectrum, and configured convergence tables from solver outputs.

```bash
oqs-pcm postprocess CONFIG.toml --output-dir DIR --solver-output-dir DIR --dipoles FILE --dipole-basis frozen --delta-scf VALUE
```

Override output, solver, dipole, basis, and external SCF-shift inputs.

```bash
oqs-pcm postprocess CONFIG.toml --dipole-basis inertial --selected-states 1 5
```

Use a complete inertial dipole tensor and restrict postprocessed dipoles to selected states.

```bash
oqs-pcm postprocess CONFIG.toml --skip-levels --skip-spectra
oqs-pcm postprocess CONFIG.toml --convergence-root 50 DIR50 --convergence-root 321 DIR321
```

Skip either table family and/or add total-state/output-root pairs for convergence tables.

## `figures`

```bash
oqs-pcm figures levels DATA_DIR FIGURE_DIR
oqs-pcm figures spectra DATA_DIR FIGURE_DIR
oqs-pcm figures cutoff DATA_DIR FIGURE_DIR
oqs-pcm figures convergence DATA_DIR FIGURE_DIR
oqs-pcm figures all DATA_DIR FIGURE_DIR --fontsize 12
```

Render one figure family or all families from postprocessed tables.

## `prep-gamess`

```bash
oqs-pcm prep-gamess GAMESS.log --ci-energy ci_energy.inp --ci-pot ci_pot.inp --reference-ci-mut ci_mut_reference.inp
```

Extract any requested CI energies, PCM potential block, and diagnostic GS--ES dipoles.

## `xdip` / `complete-dipoles`

```bash
oqs-pcm xdip --ao-dipoles AO.npy --mo-coefficients MO.npy --x-amplitudes X.npy --y-amplitudes Y.npy --n-occupied N --n-frozen-occupied N --nuclear-dipole MUX MUY MUZ --output ci_mut.inp
```

Construct a completed state/transition dipole file from NumPy ingredients.

```bash
oqs-pcm complete-dipoles --ao-dipoles AO.npy --mo-coefficients MO.npy --x-amplitudes X.npy --y-amplitudes Y.npy --n-occupied N --nuclear-dipole MUX MUY MUZ --output ci_mut.inp --reference-ci-mut ci_mut_reference.inp
```

Use the alias and optionally compare GS--ES transitions with a reference file.
