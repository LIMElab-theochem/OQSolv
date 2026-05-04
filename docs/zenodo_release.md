# Zenodo Release Notes

This file records the release packaging policy for OQSolv. It stays in the Git
repository so maintainers can keep the archival layout synchronized with the
codebase. It is not part of the Zenodo payload itself.

## Archive Scope

The Zenodo archive should be user-facing and minimal:

- `README.md`
- `CITATION.cff`
- `docs/user_manual.md`
- `example_config.toml`
- validated source code under `src/`
- the minimal regression fixtures required to reproduce the published
  scientific results

The Zenodo archive should not include maintenance material such as:

- `docs/developer_manual.md`
- `run/`
- `.pytest_cache/`
- virtual environments
- historical notebook sources or transient diagnostic outputs

## Existing Metadata

The repository already contains the machine-readable citation and archive
metadata:

- `CITATION.cff`
- `.zenodo.json`

Those files should remain aligned with the release scope above. The DOI is not
hard-coded in the repository; Zenodo will mint it at archive time.

## Pre-Release Checks

Before creating a release intended for Zenodo, check:

```bash
git remote -v
git status --short --ignored
python -m pytest tests
PYTHONPATH=/workspace/src /workspace/.venv/bin/python testsuite/hcl_water_pcm/validate.py
```

The release should only contain files that are intended for public reuse and
scientific reproduction. Temporary inputs, scratch outputs, and workstation
artifacts should remain outside the archive.
