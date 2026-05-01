# OQS-PCM Tool

Integrated Python refactor of the Mathematica workflows stored in
`potential_normal_modes.m` and `oqs-pcm.m`.

The package now also includes the figure-generation workflow that was previously
split between relevant parts of `figures-new.nb` and the standalone scripts in
`python_graphs/`.

The relevant table-generation cells from `figures-new.nb` are also represented
in Python: OQSolv can now convert raw solver outputs into the reduced text
tables used by the integrated figure renderers.

The original Mathematica project also contains two upstream preprocessing
notebooks that belong to the same overall workflow:

- `lambda_sm12t.nb`, which prepares `Sm12T.dat` and `Lambda.dat` from cavity
  geometry and Calderon matrices and implicitly defines the PCM kernel formula
- `dipoles.nb`, which completes the state/transition dipole dataset used by
  the Onsager branch and by oscillator-strength postprocessing

The broader project contains two related model families:

- PCM-based OQS, which uses electrostatic potentials on a molecular cavity
- Onsager-based OQS, which uses electric dipole moments in a spherical cavity

The current solver implementation covers the PCM-based branch. The integrated
figure workflow already includes the relevant Onsager-related plotting pathway
when the needed text tables are available.

See:

- `docs/user_manual.md`
- `docs/developer_manual.md`
- `example_config.toml`

Typical command sequence:

```bash
oqs-pcm run example_config.toml --mode all
oqs-pcm postprocess example_config.toml
oqs-pcm figures all figure_tables figures
```
