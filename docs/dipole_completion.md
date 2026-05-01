# Xdip Dipole Completion Utility

Xdip is the Python replacement for the scientifically relevant part of
`dipoles.nb`.

## When It Is Needed

Use this tool when the molecular data come from an LR-TDDFT calculation whose
output contains only ground-state to excited-state transition dipoles.

The locally modified GAMESS-US workflow used during the OQSolv refactor can
produce files that feed OQSolv, but it does not provide the full
state/transition electric dipole tensor:

- it may contain GS-ES transition dipoles
- it does not contain the ground-state dipole in the completed OQSolv format
- it does not contain ES-ES transition dipoles
- it does not contain excited-state dipoles, corresponding to ES-ES diagonal
  blocks

OQSolv needs the completed tensor for:

- OQS-Onsager coupling construction
- oscillator-strength postprocessing for OQS-PCM and OQS-Onsager

## Scientific Inputs

The reconstruction computes all state/transition dipoles from:

- AO electric dipole integrals for `x`, `y`, and `z`
- MO coefficients expressed in the AO basis
- LR-TDDFT `X` transition amplitudes
- LR-TDDFT `Y` transition amplitudes
- number of occupied molecular orbitals
- nuclear dipole contribution

The implemented algebra follows the validated `dipoles.nb` workflow:

- transform AO dipole integrals to the MO basis
- build occupied/virtual single-excitation labels
- form Slater-Condon dipole matrices
- compute excited-state and excited-transition dipoles as `X M X^T - Y M Y^T`
- compute the closed-shell ground-state molecular dipole
- compute GS-ES transition dipoles from `X + Y`
- write a completed `ci_mut.inp`-style tensor with one-based total-state labels

The original notebook was validated against GAMESS-US native ground-state and
GS-ES transition dipoles. OQSolv preserves the GS-ES comparison as an optional
diagnostic.

## Current Command-Line Interface

The first standalone Xdip command consumes NumPy `.npy` arrays:

```bash
oqs-pcm xdip \
  --ao-dipoles ao_dipoles.npy \
  --mo-coefficients mo_coefficients.npy \
  --x-amplitudes x_amplitudes.npy \
  --y-amplitudes y_amplitudes.npy \
  --n-occupied 5 \
  --nuclear-dipole 0.0 0.0 0.0 \
  --output ci_mut_gral.inp
```

Optional diagnostic against a limited native GS-ES reference:

```bash
oqs-pcm xdip \
  --ao-dipoles ao_dipoles.npy \
  --mo-coefficients mo_coefficients.npy \
  --x-amplitudes x_amplitudes.npy \
  --y-amplitudes y_amplitudes.npy \
  --n-occupied 5 \
  --nuclear-dipole 0.0 0.0 0.0 \
  --output ci_mut_gral.inp \
  --reference-ci-mut ci_mut.inp
```

## Format Boundary

The scientific reconstruction is independent from GAMESS-US text parsing.
GAMESS-specific extractors should remain a separate layer so they can be tested
and validated independently. This prevents file-format parsing from obscuring
the dipole algebra.
