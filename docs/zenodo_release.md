# Zenodo Preparation Checklist

This file records the steps needed to make OQSolv ready for a future Zenodo
archive. It does not imply that a release, tag, push, or Zenodo upload has
already been performed.

## Prepared Metadata

- `CITATION.cff` gives GitHub and Zenodo machine-readable citation metadata.
- `.zenodo.json` gives Zenodo machine-readable archive metadata.
- The DOI is intentionally absent for now. Zenodo will mint it only after a
  GitHub release is archived.
- The software license is intentionally not declared here yet. Choose the
  license explicitly before the public archival release.

## Pre-Release Checks

Before creating a GitHub release connected to Zenodo, check:

```bash
git remote -v
git status --short --ignored
python -m pytest tests
PYTHONPATH=/workspace/src /workspace/.venv/bin/python testsuite/hcl_water_pcm/validate.py
```

The repository should not track temporary or historical material such as
`old_sources/`, `references/`, `Mathematica_reference/`, virtual environments,
Python cache folders, or scratch validation outputs unless they are deliberately
part of the scientific archive.

## Future Release Procedure

When you are ready to archive, create a version tag and GitHub release:

```bash
git tag -a v0.1.0 -m "OQSolv v0.1.0"
git push
git push origin v0.1.0
```

Then create a GitHub release from that tag. If the GitHub repository is linked
to Zenodo, Zenodo will archive the release and mint a DOI.

After the DOI exists, add it to:

- `CITATION.cff`
- `.zenodo.json`
- `README.md`
- `docs/user_manual.md`
- `docs/developer_manual.md`
