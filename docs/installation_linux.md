# Linux Installation Guide

This guide installs OQSolv in an isolated Python environment. The commands
below are for Debian/Ubuntu-like systems; equivalent package names may differ
on Fedora, Arch, or other distributions.

## 1. Install operating-system prerequisites

OQSolv needs Python 3.11 or newer, Git, and the Python virtual-environment
module. A compiler toolchain is useful if `pip` must build a dependency.

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential
```

If `sudo` is unavailable, run the package-manager commands as `root`, or ask
the system administrator to install these packages. Without administrator
rights, use an already-installed Python 3.11+ interpreter and skip this step.

Check the versions before continuing:

```bash
python3 --version
git --version
```

The Python version must be at least 3.11. If `python3` is older but a
version-specific interpreter is available, use it consistently, for example
`python3.12 -m venv .venv` and then `python3.12 --version`. If `python3 -m
venv` reports that `ensurepip` is unavailable, install the version-specific
package as well, for example `sudo apt install python3.12-venv`.

## 2. Obtain the repository

Use SSH when a GitHub key is configured:

```bash
mkdir -p "$HOME/src"
git clone git@github.com:LIMElab-theochem/OQSolv.git "$HOME/src/OQSolv"
cd "$HOME/src/OQSolv"
```

Without SSH configuration, use HTTPS instead:

```bash
git clone https://github.com/LIMElab-theochem/OQSolv.git "$HOME/src/OQSolv"
cd "$HOME/src/OQSolv"
```

## 3. Create and populate the virtual environment

Keep the environment inside the checkout and activate it for each new shell:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Do not install into the system Python. Debian/Ubuntu may enforce PEP 668 and
reject system-wide `pip`; the virtual environment is the intended solution.

If `.venv` gives `Permission denied`, the checkout is not writable by the
current user. Clone it into a user-owned directory such as `$HOME/src/OQSolv`
or correct the directory ownership; do not use `sudo pip`.

## 4. Verify the installation

```bash
python -c "import matplotlib, numpy, sympy, oqs_pcm; print('Python imports: OK')"
python -m pip check
oqs-pcm --help
oqs-pcm estimate-memory --help
```

If the command is not found, activate the environment again or call it by its
absolute path: `"$PWD/.venv/bin/oqs-pcm" --help`. The equivalent module form
is `python -m oqs_pcm.cli --help`.

For machines without a graphical display, figure generation normally works
with the non-interactive backend:

```bash
export MPLBACKEND=Agg
```

## 5. Install development and test tools (optional)

`pytest` is not a runtime dependency. Install it only for development or
validation, then run the focused test suite:

```bash
python -m pip install pytest
cd /path/to/OQSolv
python -m pytest tests
```

Run the tests from the repository root; some tests import shared helpers as the
`tests` package. In this repository revision, a successful run reports 69
passed tests.

## 6. Scientific inputs and disk space

Installation does not provide molecular data. A calculation must point to
readable `ci_pot.inp`, `ci_energy.inp`, `mat_SD.inp`, `cavity.inp`, and (when
postprocessing oscillator strengths) `ci_mut.inp` files. Copy
`example_config.toml`, replace its placeholders with absolute paths, and check
large-job requirements before running:

```bash
cp example_config.toml my_calculation.toml
oqs-pcm estimate-memory my_calculation.toml
```

The estimate covers both RAM and the potentially large normal-mode text file;
keep the output filesystem writable and retain several gigabytes of free space.

## Troubleshooting checklist

- `python3: command not found`: install Python with the distribution package manager.
- `No module named venv` or `ensurepip`: install `python3-venv` (and the version-specific package if needed).
- `externally-managed-environment`: activate `.venv` and use its `python -m pip`.
- `oqs-pcm: command not found`: run `source .venv/bin/activate` from the repository.
- Import or figure errors: verify the active interpreter with `which python` and reinstall using `python -m pip install -e .`.
- Path or permission errors: use absolute input/output paths and test read/write access before a large run.
