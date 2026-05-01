from __future__ import annotations

from pathlib import Path


def _read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def gamess_extract_ci_energies(log_path: str | Path) -> list[tuple[int, float]]:
    """Extract `ci_energy.inp` entries from a GAMESS-US log.

    The AWK script `ci_energy.awk` scans lines like `1 A ... energy ...` and
    writes `Root k : energy`.  The root label counts excited states only; OQSolv
    prepends the ground state internally.
    """

    roots: list[tuple[int, float]] = []
    for line in _read_lines(log_path):
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and parts[1] == "A" and parts[0] != "0":
            roots.append((len(roots) + 1, float(parts[3].replace("D", "E").replace("d", "e"))))
    return roots


def gamess_write_ci_energy(path: str | Path, roots: list[tuple[int, float]]) -> None:
    """Write extracted excited-state energies in canonical `ci_energy.inp` form."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for root_index, energy in roots:
            handle.write(f"Root {root_index:<2d} : {energy:g}\n")


def gamess_extract_reference_ground_excited_dipoles(log_path: str | Path) -> list[tuple[int, int, tuple[float, float, float]]]:
    """Extract raw GAMESS ground/excited dipoles for diagnostics.

    This ports `ci_mut.awk`.  The result is intentionally a reference dataset:
    it is not the completed state/transition dipole tensor produced by the
    `dipoles.nb` algebra.  State labels here are zero-based because the AWK
    diagnostic labels the ground state as 0.
    """

    rows: list[tuple[int, int, tuple[float, float, float]]] = []
    excited_state = 0
    for line in _read_lines(log_path):
        parts = line.split()
        if len(parts) >= 7 and parts[0].isdigit() and parts[1] == "A":
            dipole = tuple(float(parts[index].replace("D", "E").replace("d", "e")) for index in (4, 5, 6))
            if parts[0] == "0":
                rows.append((0, 0, dipole))
            else:
                excited_state += 1
                rows.append((0, excited_state, dipole))
    return rows


def gamess_write_reference_ground_excited_dipoles(
    path: str | Path,
    rows: list[tuple[int, int, tuple[float, float, float]]],
) -> None:
    """Write the diagnostic `ci_mut.awk`-style reference dipole file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for left, right, dipole in rows:
            handle.write(
                f"States {left} and {right:2d} "
                f"{dipole[0]:.18e} {dipole[1]:.18e} {dipole[2]:.18e}\n"
            )


def gamess_extract_block(log_path: str | Path, start_marker: str, end_marker: str) -> str:
    """Extract an inclusive text block delimited by two marker lines."""

    lines = _read_lines(log_path)
    collecting = False
    block: list[str] = []
    for line in lines:
        if start_marker in line:
            collecting = True
        if collecting:
            block.append(line)
        if collecting and end_marker in line:
            break
    return "\n".join(block) + ("\n" if block else "")


def gamess_extract_potential_block(log_path: str | Path) -> str:
    """Extract the `ci_pot.inp` block from a GAMESS-US log.

    This ports the `sed` portion of `prep.sh`: marker lines are removed and the
    payload between them is returned.
    """

    block = gamess_extract_block(
        log_path,
        "INIT POT FILE FOR WAVET",
        "END POT FILE FOR WAVET",
    ).splitlines()
    if len(block) <= 2:
        return ""
    return "\n".join(block[1:-1]) + "\n"


def gamess_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
