# OQSolv

OQSolv is a scientific Python package that implements the **Open Quantum System
Polarizable Continuum Model** in its full frequency-dependent formulation,
OQS-PCM(omega).

Starting from a frozen-solvent electronic basis, OQSolv solves the
time-independent electronic problem of a molecule in solution. By using the
empirical dielectric function of the solvent as a function of frequency,
OQS-PCM(omega) includes state-specific polarization, dispersion, and
dissipation effects within a continuum-solvation framework.

In addition to the full OQS-PCM(omega) framework, the code includes several
related and derived approaches:

- Born-Oppenheimer, or antiadiabatic, solvation approximation
- self-consistent limit of OQS-PCM
- cutoff and fitted-cutoff approximations
- OQS-Onsager approach

These methods enable the study of solvation effects on quantum systems across
different levels of approximation and computational cost.

## Features

- Full frequency-dependent OQS-PCM(omega) implementation.
- Multiple approximation schemes for comparing solute-solvent timescale
  regimes.
- Modular structure for extending continuum-solvation models.
- Integrated postprocessing for energy levels, spectra, and convergence data.
- Resource checks for large calculations.

## Installation

For a complete Linux installation, including operating-system prerequisites,
virtual-environment setup, verification, and troubleshooting, see
[Linux Installation Guide](docs/installation_linux.md).

The short installation path is:

```bash
git clone https://github.com/LIMElab-theochem/OQSolv.git
cd OQSolv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

## Quick Start

Copy and edit the example configuration:

```bash
cp example_config.toml my_calculation.toml
```

Run a configured calculation:

```bash
oqs-pcm run my_calculation.toml --mode all
```

Generate postprocessed tables and figures:

```bash
oqs-pcm postprocess my_calculation.toml
oqs-pcm figures all figure_tables figures
```

For large systems, inspect resource requirements before running:

```bash
oqs-pcm estimate-memory my_calculation.toml
```

## Theory

OQS-PCM extends traditional continuum solvation by treating the solvent
electronic degrees of freedom as an environment coupled to the molecular
electronic system. The full OQS-PCM(omega) formulation uses the
frequency-dependent dielectric response of the solvent to describe how
different solute electronic states and transitions interact with the
environment.

The approximation hierarchy implemented in OQSolv makes it possible to compare
the full frequency-dependent treatment against simpler physical regimes, from
self-consistent polarization to the Born-Oppenheimer solvation limit.

## Documentation

- `docs/user_manual.md`: how to prepare inputs, configure calculations, run
  OQSolv, postprocess outputs, and generate figures.
- `docs/developer_manual.md`: internal structure, scientific layering, solver
  variants, data flow, and extension guidelines.
- `example_config.toml`: configuration template.

## Testing

Run the test suite after code changes:

```bash
python -m pytest tests
```

## References

```bibtex
@Article{guido-etal2020,
  author  = "Guido, C. A. and Rosa, M. and Cammi, R. and Corni, S.",
  title   = "An open quantum system theory for polarizable continuum models",
  journal = "J. Chem. Phys.",
  year    = "2020",
  volume  = "152",
  pages   = "174114",
  doi     = "10.1063/5.0003523"
}
```

```bibtex
@Article{gil-corni-guido2026,
  author  = "Gil, G. and Corni, S. and Guido, C. A.",
  title   = "A fully frequency-dependent implementation of the Open Quantum System Polarizable Continuum Model",
  journal = "submitted",
  year    = "2026",
  doi     = "to be assigned"
}
```

## Authors And Contact

- Gabriel Gil, Light interacting with Molecules in Embeddings (LIME) Lab, Dipartimento di Scienze e Innovazione Tecnologica, Università del Piemonte Orientale,
  [gabrieljose.gilperez@uniupo.it](mailto:gabrieljose.gilperez@uniupo.it)
- Ciro A. Guido, Light interacting with Molecules in Embeddings (LIME) Lab, Dipartimento di Scienze e Innovazione Tecnologica, Università del Piemonte Orientale,
  [ciro.guido@uniupo.it](mailto:ciro.guido@uniupo.it)
- Stefano Corni, Nanostructures and biomolecules modeling group, Dipartimento di Scienze Chimiche, Università degli Studi di Padova; S3, Istituto di Nanoscienze, Modena
