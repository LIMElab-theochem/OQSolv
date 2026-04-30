from __future__ import annotations

import pytest


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter, exitstatus: int) -> None:
    """Print a plain-language test summary after the standard pytest output.

    The default pytest summary is concise but assumes the reader already knows
    how to interpret it. This extra line makes the result explicit by stating
    how many tests passed out of the total and whether the run was successful.
    """

    stats = terminalreporter.stats
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    errors = len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    xfailed = len(stats.get("xfailed", []))
    xpassed = len(stats.get("xpassed", []))
    total = passed + failed + errors + skipped + xfailed + xpassed

    if exitstatus == 0:
        terminalreporter.write_sep("=", f"successful: {passed} out of {total} tests passed")
    else:
        terminalreporter.write_sep(
            "=",
            f"not successful: {passed} out of {total} tests passed, {failed} failed, {errors} errors",
        )
